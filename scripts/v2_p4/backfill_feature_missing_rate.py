# -*- coding: utf-8 -*-
"""旧模型 feature_missing_rate 补录工具（P4 后维护项 B 方案，T4）。

复杂度豁免：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。

定位（设计文档 v3 §4.1）
========================
把「可审计重建基线」的逐列训练期缺失率 μ 补录进现役模型 metadata——
**不声称复现原训练帧**：冻结 cs_train 只有窗口/列集可核，无法证明与当年训练
逐位同帧；产物按登记的 24022 注册声明重放，附完整证据链与独立重跑一致性。

重放配方（24022 = 旧链 deploy 产物，`ml/walk_forward/training_core.py:154-155`
`_train_model_on_window` 的行域顺序）
======================================
1. 交易日历：`clean/trade_cal`，`is_open==1`，窗口 = 注册外层
   `train_start_date`~`train_end_date`（日历驱动，与 `load_features_data` 同口径）；
2. 逐日读 `features/cs_train/{YYYYMMDD}.parquet`（**分块池化**：总缺失率 =
   1 − Σnotna/Σ行数，与整帧 `df[col].isna().mean()` 代数等价，避免 ~9 GiB 整帧内存）；
3. 主板域过滤（`clean/stock_basic` market ∈ ("主板",)，复用旧链
   `_build_main_board_codes`/`_filter_to_main_board` 同源函数）；
4. 训练标记过滤（`is_st`/`is_suspended`/`is_limit_up` 逐列 `~astype(bool)`，
   与 `prepare._filter_and_decay` 同式）；
5. 标签 dropna（注册 `label_column`）——分母 = 标签可用样本；
6. 事件 freshness 衰减：**不重放值运算**（登记等价性：`apply_event_freshness_decay`
   为 `raw * exp(-ln2*days/half_life)`，NaN×w 仍 NaN、freshness 缺失权重 fillna(1.0)
   ⇒ 只改值不改 NaN 形态，μ 不受影响）；
7. μ = 1 − Σnotna/Σ行数，键集精确 = 注册 `features.json` ∪ `SELECTION_FILTER_COLUMNS`。

校验（fail-closed）
===================
- 157 列全部在分区内（缺席 ⇒ 配置漂移，拒绝）；
- 全部 μ ∈ [0,1)、无 μ≥1.0（训练期全空列本不应进入注册列集）；
- 154 模型列 μ ≤ 0.6（24022 注册时无 `removed_high_missing_features`，事实要求；
  不一致必须停下归因，不得放宽）——**硬门**；
- `cons_target_upside` 与登记参考值 57.2%~57.7% 的偏差仅登记告警（设计 §1.3 登记值
  未复算，非硬门）；
- 分母行数与 metadata 顶层 `n_samples` 不要求相等（顶层 = X_train_len+X_val_len，
  差 = calibration + embargo + train/val 隔离带），仅登记差值。

写入（--write；默认 dry-run）
============================
- sidecar `v{N}_metadata.json`：备份 → `train_params` additive 三键 → 原子写 → 回读；
- `model_registry.json` 对应版本条目**定向** additive 更新（`format_host_model_info`
  registry 优先、sidecar 兜底 ⇒ 只写 sidecar 会让展示读旧值）→ 备份 → 原子写 → 回读；
- 双份回读一致校验；证据 JSON 落 `data/frozen_reference/v2_p4/<run_id>/`。

用法
====
``python scripts/v2_p4/backfill_feature_missing_rate.py --version 24022``（dry-run）
``python scripts/v2_p4/backfill_feature_missing_rate.py --version 24022 --write``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from src.lazybull.data import DataLoader, Storage  # noqa: E402
from src.lazybull.ml.walk_forward.training_core import (  # noqa: E402
    _build_main_board_codes,
    _filter_to_main_board,
)
from src.lazybull.v2.common.model_schema import (  # noqa: E402
    FEATURE_MISSING_RATE_SCHEMA_VERSION,
    SELECTION_FILTER_COLUMNS,
    build_feature_missing_rate_entry,
    read_feature_missing_rate,
    summarize_daily_missing_rates,
)

#: 24022 事件衰减登记参数（仅证据登记；μ 对衰减不变，见模块 docstring 第 6 条）
_FRESHNESS_REPLAY_NOTE = "skipped_nan_invariant"

#: 24022 注册事实：无 removed_high_missing_features ⇒ 模型列 μ 硬上限 0.6
_MAX_FEATURE_MISSING_RATIO_24022 = 0.6

#: 登记参考值区间（daily_assembly 原白名单注释：池化 57.2%(中位)~57.7%(均值)）
_REFERENCE_MU = {"cons_target_upside": (0.572, 0.577), "tolerance_pp": 2.0}


@dataclass(frozen=True)
class BackfillSpec:
    """补录重放声明（来自模型注册 metadata）。"""

    version: int
    feature_columns: Tuple[str, ...]
    label_column: str
    train_start_date: str
    train_end_date: str
    n_samples_registered: int

    @property
    def coverage_columns(self) -> Tuple[str, ...]:
        """注册侧键集精确 = 模型列 ∪ 选股过滤列（B 方案 §2.3）。"""
        return tuple(dict.fromkeys([*self.feature_columns, *SELECTION_FILTER_COLUMNS]))


@dataclass
class RebuildCounts:
    """分块累加器（池化分母/非空计数 + 逐日截面缺失率序列）。"""

    n_rows: int = 0  # 标签可用样本（池化分母）
    n_samples_after_filter: int = 0  # 标记过滤后（标签 dropna 前）
    n_dropped_label_nan: int = 0
    notna: Dict[str, int] = field(default_factory=dict)
    daily_rates: Dict[str, List[float]] = field(default_factory=dict)  # B′：逐日率序列
    day_stats: List[dict] = field(default_factory=list)
    missing_days: List[str] = field(default_factory=list)

    def add_day(self, day: str, raw_rows: int, after_filter: int, after_label: int) -> None:
        self.day_stats.append(
            {
                "trade_date": day,
                "raw_rows": int(raw_rows),
                "after_filter": int(after_filter),
                "after_label_dropna": int(after_label),
            }
        )

    def add_daily_rates(self, day: str, rates: Mapping[str, float]) -> None:
        """记录当日截面缺失率（滚动分位/分布摘要的数据源）。"""
        for col, rate in rates.items():
            self.daily_rates.setdefault(col, []).append(float(rate))
        self.day_stats[-1]["n_rate_columns"] = len(rates)


def load_spec(models_dir: Path, version: int) -> BackfillSpec:
    """读模型注册 metadata 与特征列清单，构造重放声明（缺字段 ⇒ fail-closed）。"""
    metadata_file = models_dir / f"v{version}_metadata.json"
    features_file = models_dir / f"v{version}_features.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"模型元数据旁路缺失: {metadata_file}")
    if not features_file.exists():
        raise FileNotFoundError(f"模型特征列清单缺失: {features_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    feature_columns = json.loads(features_file.read_text(encoding="utf-8"))
    return BackfillSpec(
        version=int(version),
        feature_columns=tuple(str(c) for c in feature_columns),
        label_column=str(metadata["label_column"]),
        train_start_date=str(metadata["train_start_date"]),
        train_end_date=str(metadata["train_end_date"]),
        n_samples_registered=int(metadata.get("n_samples") or 0),
    )


def resolve_open_days(loader: DataLoader, start: str, end: str) -> List[str]:
    """窗口内开市日（与 `load_features_data` 同口径：cal_date 区间 + is_open==1）。"""
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    if trade_cal is None:
        raise RuntimeError("交易日历缺失（clean/trade_cal 与 raw/trade_cal 均不可用）")
    days = trade_cal[
        (trade_cal["cal_date"] >= start)
        & (trade_cal["cal_date"] <= end)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()
    return [str(day) for day in days]


def _filter_frame(frame, spec: BackfillSpec, main_board_codes: set):
    """单日帧行域：主板过滤 + 训练标记过滤 + 标签 dropna（旧链 prepare 同式）。"""
    filtered = _filter_to_main_board(frame, main_board_codes, f"补录重放 {spec.label_column}")
    after_filter = len(filtered)
    mask = None
    for col in ("is_st", "is_suspended", "is_limit_up"):
        if col not in filtered.columns:
            continue
        col_mask = ~filtered[col].astype(bool)
        mask = col_mask if mask is None else (mask & col_mask)
    if mask is not None:
        filtered = filtered[mask]
    filtered = filtered.dropna(subset=[spec.label_column])
    return after_filter, filtered


def scan(spec: BackfillSpec, storage: Storage, loader: DataLoader) -> Tuple[RebuildCounts, dict]:
    """逐日分块重放并累加池化计数（返回计数与数据态摘要）。"""
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        raise RuntimeError("clean/stock_basic 缺失，无法构建主板域")
    main_board_codes = _build_main_board_codes(stock_basic)
    days = resolve_open_days(loader, spec.train_start_date, spec.train_end_date)
    if not days:
        raise ValueError(f"窗口 [{spec.train_start_date}, {spec.train_end_date}] 内无开市日")

    counts = RebuildCounts()
    expected = list(spec.coverage_columns)
    file_digest_items: List[Tuple[str, int]] = []
    for index, day in enumerate(days, start=1):
        frame = storage.load_cs_train_day(day)
        if frame is None or len(frame) == 0:
            counts.missing_days.append(day)
            continue
        missing_cols = [col for col in expected if col not in frame.columns]
        if missing_cols:
            raise ValueError(
                f"cs_train 分区 {day} 缺补录列 {len(missing_cols)} 个（配置漂移，拒绝补录）: "
                f"{missing_cols[:10]}"
            )
        raw_rows = len(frame)
        after_filter, kept = _filter_frame(frame, spec, main_board_codes)
        after_label = len(kept)
        counts.add_day(day, raw_rows, after_filter, after_label)
        counts.n_samples_after_filter += after_filter
        counts.n_dropped_label_nan += after_filter - after_label
        counts.n_rows += after_label
        if after_label:
            notna = kept[expected].notna().sum()
            for col in expected:
                counts.notna[col] = counts.notna.get(col, 0) + int(notna[col])
            counts.add_daily_rates(day, kept[expected].isna().mean().to_dict())
        if index % 200 == 0:
            logger.info(f"补录重放进度 {index}/{len(days)} 日（累计样本 {counts.n_rows}）")
        partition = Path(storage.root_path) / "features" / "cs_train" / f"{day}.parquet"
        if partition.exists():
            file_digest_items.append((partition.name, partition.stat().st_size))

    data_state = {
        "cs_train_partitions": len(file_digest_items),
        "cs_train_bytes": int(sum(size for _, size in file_digest_items)),
        "cs_train_digest": hashlib.sha256(
            json.dumps(sorted(file_digest_items), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "calendar_days": len(days),
        "missing_days": counts.missing_days,
        # 标签可用末日（reviewer3 P2-1）：尾部零标签日不计入声明与日数
        "actual_data_end": next(
            (
                row["trade_date"]
                for row in reversed(counts.day_stats)
                if row["after_label_dropna"] > 0
            ),
            None,
        ),
        "n_trade_days": sum(1 for row in counts.day_stats if row["after_label_dropna"] > 0),
    }
    return counts, data_state


def compute_distribution(counts: RebuildCounts, spec: BackfillSpec) -> Dict[str, dict]:
    """逐列**逐日缺失率分布摘要**（B′ schema v2，单一实现 = common.model_schema）。"""
    if counts.n_rows <= 0:
        raise ValueError("补录分母为零（标签可用样本为空），拒绝产出基线")
    required = list(spec.coverage_columns)
    missing = [col for col in required if col not in counts.daily_rates]
    if missing:
        raise ValueError(f"逐日缺失率序列缺列（重放异常）: {missing[:10]}")
    return summarize_daily_missing_rates({col: counts.daily_rates[col] for col in required})


def compute_pooled(counts: RebuildCounts, spec: BackfillSpec) -> Dict[str, float]:
    """池化缺失率（1 − Σnotna/Σ行数；仅作 24022 事实健全性检查与证据，不入 metadata）。"""
    return {
        col: float(1.0 - counts.notna.get(col, 0) / counts.n_rows) for col in spec.coverage_columns
    }


def validate(
    distribution: Mapping[str, Mapping], pooled: Mapping[str, float], spec: BackfillSpec
) -> List[dict]:
    """补录校验（结构硬门失败即抛；参考偏差登记告警）。返回校验记录清单。"""
    records: List[dict] = []
    expected = set(spec.coverage_columns)
    keys = set(distribution)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"分布键集与注册列集不匹配: 缺 {missing[:10]} / 多 {extra[:10]}")
    coverage, error = read_feature_missing_rate(
        {
            "feature_missing_rate_schema_version": FEATURE_MISSING_RATE_SCHEMA_VERSION,
            "feature_missing_rate": dict(distribution),
        }
    )
    if error is not None:
        raise ValueError(f"分布结构非法（读侧校验 {error}）——重建异常，拒绝补录")
    records.append({"check": "keyset_and_distribution", "status": "pass", "n_keys": len(keys)})

    high = sorted(
        col for col in spec.feature_columns if pooled[col] > _MAX_FEATURE_MISSING_RATIO_24022
    )
    if high:
        raise ValueError(
            f"模型列池化缺失率 > 0.6（24022 注册时无高缺失移除，事实不一致，须停下归因）: "
            f"{[(col, round(pooled[col], 4)) for col in high]}"
        )
    records.append(
        {
            "check": "model_columns_pooled_le_0.6",
            "status": "pass",
            "n_columns": len(spec.feature_columns),
            "max_pooled": max(pooled[col] for col in spec.feature_columns),
        }
    )

    ref_col = "cons_target_upside"
    if ref_col in pooled:
        low, high_ref = _REFERENCE_MU[ref_col]
        tol = _REFERENCE_MU["tolerance_pp"] / 100.0
        delta = pooled[ref_col] - (low + high_ref) / 2.0
        status = "pass" if abs(delta) <= tol else "warn"
        records.append(
            {
                "check": "cons_target_upside_pooled_reference",
                "status": status,
                "pooled": pooled[ref_col],
                "p95": distribution[ref_col]["p95"],
                "p99": distribution[ref_col]["p99"],
                "reference": [low, high_ref],
                "delta": delta,
                "note": "登记参考区间（池化口径），非硬门；判据基线 = 分布分位",
            }
        )
        if status == "warn":
            logger.warning(
                f"{ref_col} 池化={pooled[ref_col]:.4f} 偏离登记参考区间 {low:.3f}~{high_ref:.3f} "
                f"（Δ={delta:+.4f} > {tol:.2f}），仅登记，不阻断"
            )
    return records


def build_addendum(
    distribution: Mapping[str, Mapping],
    spec: BackfillSpec,
    counts: RebuildCounts,
    data_state: dict,
    evidence_rel: str,
) -> dict:
    """构造 `train_params` additive 三键（键集/分布经 common 单一来源校验）。

    注意：``build_feature_missing_rate_entry`` 返回**纯投影 dict**（列名 → 分布摘要），
    不是三键包装——必须显式补 ``feature_missing_rate_schema_version`` 与
    ``feature_missing_rate`` 两个键（首版误展开导致 157 个值写成 train_params
    顶层键，已回滚重写并加测试锁定）。
    """
    projected = build_feature_missing_rate_entry(dict(distribution), list(spec.coverage_columns))
    stat = {
        "window": [spec.train_start_date, spec.train_end_date],
        "actual_data_end": data_state["actual_data_end"],
        "n_trade_days": int(data_state["n_trade_days"]),
        "frame": "train_daily_rate_distribution_rebuilt",
        "denominator": "label_available_rows",
        "row_domain": "train_filtered_main_board",
        "rate_scope": "per_trade_date_cross_section",
        "coverage_n_samples": int(counts.n_rows),
        "n_samples_after_filter": int(counts.n_samples_after_filter),
        "n_dropped_label_nan": int(counts.n_dropped_label_nan),
        "n_samples_registered_top_level": int(spec.n_samples_registered),
        "n_samples_vs_top_level_delta": int(counts.n_rows - spec.n_samples_registered),
        "window_source": "registration",
        "source": "backfilled",
        "backfilled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backfill_evidence": evidence_rel,
        "freshness_replay": _FRESHNESS_REPLAY_NOTE,
    }
    if stat["coverage_n_samples"] != (stat["n_samples_after_filter"] - stat["n_dropped_label_nan"]):
        raise ValueError("互证等式不成立（coverage_n_samples != after_filter − label_nan）")
    return {
        "feature_missing_rate_schema_version": FEATURE_MISSING_RATE_SCHEMA_VERSION,
        "feature_missing_rate": projected,
        "feature_missing_rate_stat": stat,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _backup(path: Path, stamp: str) -> Path:
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


_BLOCK_KEYS = (
    "feature_missing_rate_schema_version",
    "feature_missing_rate",
    "feature_missing_rate_stat",
)


def verify_backfill(models_dir: Path, versions: Sequence[int]) -> Dict[int, dict]:
    """全量双份校验（可多版本，覆盖 ensemble 多成员）。

    逐版本深比较 sidecar 与 registry 的三键块（schema/分布/统计基），并过读侧
    结构校验；任一不一致或非法即抛（不返回部分成功）。
    """
    results: Dict[int, dict] = {}
    registry_file = models_dir / "model_registry.json"
    registry_models = json.loads(registry_file.read_text(encoding="utf-8")).get("models", [])
    for version in versions:
        sidecar = models_dir / f"v{version}_metadata.json"
        sidecar_params = dict(
            json.loads(sidecar.read_text(encoding="utf-8")).get("train_params") or {}
        )
        entry = next(
            (item for item in registry_models if int(item.get("version", -1)) == version),
            None,
        )
        if entry is None:
            raise RuntimeError(f"校验失败：registry 缺 v{version} 条目")
        registry_params = dict(entry.get("train_params") or {})
        # 三键必须**显式存在**（reviewer3 R2-P1-1：`.get()` 对双缺失返回 None==None）
        missing_side = [key for key in _BLOCK_KEYS if key not in sidecar_params]
        missing_reg = [key for key in _BLOCK_KEYS if key not in registry_params]
        if missing_side or missing_reg:
            raise RuntimeError(
                f"双份一致性无法证明（v{version}）：sidecar 缺 {missing_side} / registry 缺 {missing_reg}"
            )
        for key in _BLOCK_KEYS:
            if sidecar_params[key] != registry_params[key]:
                raise RuntimeError(f"双份不一致（v{version} / {key}）：sidecar 与 registry 不同值")
        coverage, error = read_feature_missing_rate(sidecar_params)
        if error is not None or not coverage:
            raise RuntimeError(f"双份一致性无法证明（v{version} 读侧校验 {error}）")
        results[version] = {"n_columns": len(coverage or {}), "readback_equal": True}
    return results


def _journal_path(models_dir: Path, version: int) -> Path:
    return models_dir / f"backfill_journal_v{version}.json"


def _block_of(train_params: Mapping) -> dict:
    """从 train_params 提取三键块（用于双份修复）。"""
    return {
        key: train_params[key]
        for key in (
            "feature_missing_rate_schema_version",
            "feature_missing_rate",
            "feature_missing_rate_stat",
        )
        if key in train_params
    }


def _has_valid_block(train_params: Mapping) -> bool:
    coverage, error = read_feature_missing_rate(train_params)
    return error is None and coverage is not None


def recover_incomplete_backfill(models_dir: Path, version: int) -> Optional[str]:
    """启动恢复（reviewer3 P1-5）：journal 残留时按单边权威块修复双份。

    两个 JSON 不具备单原子写能力，故以 journal 标记 + 单边修复替代"假装原子"：
    任一单写完成态都可由另一份的合法块补齐；双份一致则仅清理 journal。
    """
    journal = _journal_path(models_dir, version)
    if not journal.exists():
        return None
    sidecar = models_dir / f"v{version}_metadata.json"
    registry_file = models_dir / "model_registry.json"
    sidecar_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_params = dict(sidecar_metadata.get("train_params") or {})
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    entry = next(
        (item for item in registry.get("models", []) if int(item.get("version", -1)) == version),
        None,
    )
    if entry is None:
        raise RuntimeError(f"恢复失败：registry 缺 v{version} 条目")
    registry_params = dict(entry.get("train_params") or {})
    sidecar_ok = _has_valid_block(sidecar_params)
    registry_ok = _has_valid_block(registry_params)
    if sidecar_ok and registry_ok:
        # 双份均"结构合法"≠一致：必须做三键深比较（reviewer3 R2-P1-1）
        verify_backfill(models_dir, [version])
        journal.unlink()
        return "journal 残留但双份一致：已清理"
    if sidecar_ok:
        block = _block_of(sidecar_params)
        registry_params.update(block)
        entry["train_params"] = registry_params
        _backup(registry_file, datetime.now().strftime("%Y%m%d%H%M%S"))
        _atomic_write_json(registry_file, registry)
        journal.unlink()
        return "registry 缺失：已从 sidecar 修复"
    if registry_ok:
        block = _block_of(registry_params)
        sidecar_params.update(block)
        sidecar_metadata["train_params"] = sidecar_params
        _backup(sidecar, datetime.now().strftime("%Y%m%d%H%M%S"))
        _atomic_write_json(sidecar, sidecar_metadata)
        journal.unlink()
        return "sidecar 缺失：已从 registry 修复"
    raise RuntimeError("恢复失败：双份均无合法块（journal 残留但无可修复来源）")


def write_backfill(
    models_dir: Path,
    spec: BackfillSpec,
    addendum: dict,
    evidence: dict,
    extra_versions: Sequence[int] = (),
) -> dict:
    """写入 sidecar + registry 定向条目（备份 → 原子写 → 双份回读一致）。"""
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    sidecar = models_dir / f"v{spec.version}_metadata.json"
    registry_file = models_dir / "model_registry.json"
    journal = _journal_path(models_dir, spec.version)

    recovered = recover_incomplete_backfill(models_dir, spec.version)
    if recovered:
        logger.warning(f"检测到上次未完成补录并已恢复：{recovered}")
    _atomic_write_json(
        journal,
        {"version": spec.version, "state": "started", "started_at": stamp},
    )

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    train_params = dict(metadata.get("train_params") or {})
    if _has_valid_block(train_params):
        # 幂等 no-op 也必须先证明双份一致（reviewer3 R2-P1-1：否则 split-brain 假成功）
        verify_backfill(models_dir, [spec.version, *extra_versions])
        journal.unlink(missing_ok=True)
        return {
            "noop": True,
            "reason": f"v{spec.version} 已存在合法分布且双份一致（幂等 no-op，未改写任何文件）",
            "sidecar": str(sidecar),
            "registry": str(registry_file),
            "readback_equal": True,
            "verified_versions": [spec.version, *extra_versions],
            "written_at": None,
            "coverage_sha256": evidence.get("feature_missing_rate_sha256"),
        }
    train_params.update(addendum)
    metadata["train_params"] = train_params
    sidecar_backup = _backup(sidecar, stamp)
    _atomic_write_json(sidecar, metadata)
    _atomic_write_json(journal, {"version": spec.version, "state": "sidecar_written"})

    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    entry = next(
        (
            item
            for item in registry.get("models", [])
            if int(item.get("version", -1)) == spec.version
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"model_registry.json 未找到 v{spec.version} 条目")
    entry_params = dict(entry.get("train_params") or {})
    entry_params.update(addendum)
    entry["train_params"] = entry_params
    registry_backup = _backup(registry_file, stamp)
    _atomic_write_json(registry_file, registry)
    _atomic_write_json(journal, {"version": spec.version, "state": "registry_written"})

    # 全量双份回读（reviewer3 P2-3/R2-P1-1：三键显式存在 + 深比较 + 多成员逐版本）
    verify_backfill(models_dir, [spec.version, *extra_versions])
    journal.unlink(missing_ok=True)

    return {
        "sidecar": str(sidecar),
        "sidecar_backup": str(sidecar_backup),
        "registry": str(registry_file),
        "registry_backup": str(registry_backup),
        "readback_equal": True,
        "written_at": stamp,
        "coverage_sha256": evidence["feature_missing_rate_sha256"],
    }


def _repo_git_state() -> dict:
    import subprocess

    def _run(args: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(_REPO_ROOT), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"git_commit": _run(["rev-parse", "HEAD"]), "git_dirty": _run(["status", "--porcelain"])}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="旧模型 feature_missing_rate 补录（B 方案 T4）")
    parser.add_argument("--version", type=int, default=24022)
    parser.add_argument("--models-dir", default=None, help="缺省 = 生产 stock_selection 模型根")
    parser.add_argument(
        "--evidence-dir", default=None, help="缺省 = data/frozen_reference/v2_p4/<run_id>"
    )
    parser.add_argument("--write", action="store_true", help="落盘（缺省 dry-run，只打印与校验）")
    args = parser.parse_args(argv)

    from src.lazybull.common.config import get_stock_selection_models_root

    models_dir = Path(args.models_dir or get_stock_selection_models_root())
    run_id = f"backfill_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evidence_dir = Path(
        args.evidence_dir or (_REPO_ROOT / "data" / "frozen_reference" / "v2_p4" / run_id)
    )

    spec = load_spec(models_dir, args.version)
    logger.info(
        f"补录重放声明: v{spec.version} 窗口 {spec.train_start_date}~{spec.train_end_date} / "
        f"{len(spec.feature_columns)} 模型列 + {len(SELECTION_FILTER_COLUMNS)} 过滤列 / "
        f"标签 {spec.label_column}"
    )
    storage = Storage()
    loader = DataLoader(storage)
    started = time.time()
    counts, data_state = scan(spec, storage, loader)
    distribution = compute_distribution(counts, spec)
    pooled = compute_pooled(counts, spec)
    validations = validate(distribution, pooled, spec)
    logger.info(
        f"重放完成: {len(counts.day_stats)} 日 / 分母 {counts.n_rows} 行 / "
        f"耗时 {time.time() - started:.1f}s"
    )
    cons_stats = distribution["cons_target_upside"]
    logger.info(
        f"cons_target_upside: 池化={pooled['cons_target_upside']:.4f}（登记参考 0.572~0.577）"
        f" / p95={cons_stats['p95']:.4f} / p99={cons_stats['p99']:.4f}"
        f" / 逐日 max={cons_stats['max']:.4f} / n_days={cons_stats['n_days']}"
    )
    logger.info(
        f"模型列 p95 最大 = {max(distribution[c]['p95'] for c in spec.feature_columns):.4f}；"
        f"p99 不稳定列数 = {sum(1 for c in spec.feature_columns if distribution[c]['p99_unstable'])}"
    )
    for record in validations:
        logger.info(f"校验[{record['status']}] {record['check']}: {record}")

    evidence = {
        "tool": "backfill_feature_missing_rate",
        "run_id": run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **_repo_git_state(),
        "model": {
            "version": spec.version,
            "label_column": spec.label_column,
            "n_feature_columns": len(spec.feature_columns),
            "n_coverage_columns": len(spec.coverage_columns),
        },
        "window": {
            "declared": [spec.train_start_date, spec.train_end_date],
            "actual_data_end": data_state["actual_data_end"],
            "n_trade_days": data_state["n_trade_days"],
            "calendar_days": data_state["calendar_days"],
            "n_missing_days": len(data_state["missing_days"]),
            "missing_days": data_state["missing_days"][:20],
        },
        "denominator": {
            "coverage_n_samples": counts.n_rows,
            "samples_after_filter": counts.n_samples_after_filter,
            "n_dropped_label_nan": counts.n_dropped_label_nan,
            "n_samples_registered_top_level": spec.n_samples_registered,
            "delta_vs_top_level": counts.n_rows - spec.n_samples_registered,
        },
        "data_state": data_state,
        "freshness_replay": {
            "strategy": "state_keep_event_decay",
            "half_life_days": 120.0,
            "replay": _FRESHNESS_REPLAY_NOTE,
            "rationale": "raw*exp(-ln2*days/half_life) 保 NaN；freshness 缺失权重 fillna(1.0) ⇒ μ 不变",
        },
        "feature_missing_rate": {col: distribution[col] for col in sorted(distribution)},
        "pooled_missing_rate": {col: pooled[col] for col in sorted(pooled)},
        "feature_missing_rate_sha256": hashlib.sha256(
            json.dumps({col: distribution[col] for col in sorted(distribution)}).encode("utf-8")
        ).hexdigest(),
        "validations": validations,
        "write": None,
    }
    evidence_rel = (
        str((evidence_dir / "backfill_evidence.json").relative_to(_REPO_ROOT))
        if str(evidence_dir).startswith(str(_REPO_ROOT))
        else str(evidence_dir / "backfill_evidence.json")
    )
    addendum = build_addendum(distribution, spec, counts, data_state, evidence_rel)

    if not args.write:
        logger.warning("dry-run：未写入任何 metadata（加 --write 落盘）")
        logger.info(
            f"三键摘要: {json.dumps({k: (v if k != 'feature_missing_rate' else '<157 键>') for k, v in addendum.items()}, ensure_ascii=False)}"
        )
        return 0

    evidence["write"] = write_backfill(models_dir, spec, addendum, evidence)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(evidence_dir / "backfill_evidence.json", evidence)
    _atomic_write_json(
        evidence_dir / "feature_missing_rate.json",
        {col: distribution[col] for col in sorted(distribution)},
    )
    _atomic_write_json(
        evidence_dir / "pooled_missing_rate.json", {col: pooled[col] for col in sorted(pooled)}
    )
    logger.info(f"补录完成：sidecar + registry 双份一致；证据 = {evidence_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
