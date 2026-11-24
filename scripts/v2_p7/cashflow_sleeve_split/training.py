# -*- coding: utf-8 -*-
# 复杂度豁免：本文件属 P7 CF-SPLIT 阶段专用验收工具（scripts/v2_p7/**，§7.5 临时开发
# 文件口径），overlay_pit_frame/build_pit_day_index 为 PIT 叠加核心（checkpoint-2 冻结
# 实现，行数/嵌套由分块向量化与双缓存根逻辑决定）；拆分计划 = 审点二收口后若转生产
# 再按单元边界拆出 pit_overlay.py，训练链重跑期间禁改核心逻辑。
"""run-a 两臂训练驱动（协议 §5.2；审点二 A 批）。

口径（冻结）
============
1. **同配方**：超参 = ``VALUE_SLEEVE_SPEC``（生产配置代码化单一来源）+
   ``n_estimators=1500``（readiness ``model_config_evidence`` 裁决：部署/折元数据
   优先于 500 棵复跑示例；协议 §5.2-3）。
2. **同折缝**：逐折显式四日期 = ``resolve_explicit_split_dates(外层窗, 0.2,
   窗内日历日, label_delta=21)``——与生产 ``execute_split_training`` 逐字同源；
   外层窗 = 锚批次折表（B0 replay summary，checkpoint-1 时钟已对账）。
   部署段 = 比例路径（生产 deploy 无显式缝；日志 es 20241126..20250828 为其产物）。
3. **质量门后投影**：三臂共同生产特征准备 → 质量门 → 有序列投影
   （``feature_columns_override`` = 臂列 F_k / R_k；协议 §5.2-4）。F 臂同样传
   原注册列以锁 schema。
4. **隔离注册**：每臂独立 ``ModelRegistry``（campaign ``models/<arm>/``）；
   注册时按**本臂最终列**生成现金流 live-column 元数据（协议 §5.2-6）。
5. **段净化登记**：折段锁定日志值；早停/校准的跨段成熟事实（last usable +
   purged）按 checkpoint-1 时钟逐折随 manifest 登记，不新增训练侧净化
   （r0_003 复核第 2 项 Codex 通过口径）。

数据源：v2 panel store（``SOURCE_PANEL``；``cs_train_verify`` 的调用方白名单
不含 ``scripts/v2_p7``，历史段内容已由 P3 验收证明与 cs_train 等价）。
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.cashflow_sleeve_split import spec

#: 两臂注册根（campaign 相对路径）
ARM_REGISTRY_RELATIVE = {"F": "models/f", "R": "models/r"}

#: 裁决后的树数（readiness.model_config_evidence；metadata 口径）
RESOLVED_N_ESTIMATORS = 1500

SAMPLE_KEY_COLUMNS = ["trade_date", "ts_code"]

#: PIT 叠加的双缓存根（账户窗 + 历史窗；checkpoint-2 Codex 第 2 项）
PIT_CACHE_ROOTS = ("structural_full", "structural_history")


class TrainingError(RuntimeError):
    """训练输入缺件、缝换算不符或注册校验失败（fail-fast）。"""


# ── checkpoint-2 修复 1：早停尾部按冻结可用日截断（跨段成熟泄漏） ────────────


@contextmanager
def es_maturity_purge(usable_day: str) -> Any:
    """训练侧早停净化（checkpoint-2 Codex 第 1 项裁决；取代 r0_003「只登记不净化」口径）。

    冻结时钟逐折给出 ``es_last_usable_sample``：其后的 es 尾部样本标签成熟日
    跨入校准段（成熟日 ≥ calib_start），构成跨段泄漏。生产 ``prepare`` 的
    es/校准/embargo 三段按日历日等比拆分、无成熟日清理 ⇒ 本包装在原始拆分
    后按冻结可用日截断 es 子集（被剔行整体丢弃、绝不并入校准），并回写统计。
    """
    prepare = sys.modules.get("src.lazybull.v2.train.core.prepare")
    if prepare is None:
        import importlib

        prepare = importlib.import_module("src.lazybull.v2.train.core.prepare")
    original = prepare.split_val_for_selection_protocol_by_date
    state: Dict[str, Any] = {"stats": None}

    def wrapper(df_val: Any, embargo_days: Any) -> Any:
        from src.lazybull.v2.store.store_identity import frame_content_digest

        es, calib, embargo, stats = original(df_val, embargo_days)
        purged = es.iloc[:0]
        if usable_day and len(es) > 0:
            dates = es["trade_date"].astype(str)
            keep = dates <= usable_day
            dropped_rows = int((~keep).sum())
            if dropped_rows:
                purged = es.loc[~keep, SAMPLE_KEY_COLUMNS]
                es = es.loc[keep].copy()
                stats = dict(stats)
                stats["val_es_end_date"] = usable_day
                stats["val_es_samples"] = len(es)
                stats["val_es_purged_rows"] = dropped_rows
                stats["val_es_purged_dates"] = int(dates[~keep].nunique())
                stats["val_es_purged_usable_day"] = usable_day
        state["stats"] = stats
        state["keys"] = {
            "purged_es": frame_content_digest(purged[SAMPLE_KEY_COLUMNS]),
            "embargo": frame_content_digest(embargo[SAMPLE_KEY_COLUMNS]),
        }
        return es, calib, embargo, stats

    prepare.split_val_for_selection_protocol_by_date = wrapper
    try:
        yield state
    finally:
        prepare.split_val_for_selection_protocol_by_date = original


def assert_es_purged(
    state: Mapping[str, Any], usable_day: str, emit: Callable[[str], None]
) -> None:
    """断言本轮 es 拆分确实按冻结可用日截断（fail-closed）。"""
    stats = state.get("stats")
    if not stats:
        raise TrainingError("早停净化未生效（拆分统计缺失）")
    end = str(stats.get("val_es_end_date", "")).replace("-", "")[:8]
    if end != usable_day:
        raise TrainingError(f"早停净化断言失败：es 末端 {end} != 冻结可用日 {usable_day}")
    emit(
        f"      es 净化生效：末端 {end}，剔除 {stats.get('val_es_purged_rows', 0)} 行"
        f"/{stats.get('val_es_purged_dates', 0)} 日"
    )


def _split_key_summary(base_result: Any) -> Dict[str, Any]:
    """实际训练/早停/校准/embargo 键摘要（checkpoint-2 第 1 项留痕要求）。"""
    stats = (base_result or {}).get("data_stats") or {}
    keys = (
        "train_start_date",
        "train_end_date",
        "val_raw_start_date",
        "val_raw_end_date",
        "val_es_start_date",
        "val_es_end_date",
        "val_es_samples",
        "val_es_purged_rows",
        "val_es_purged_dates",
        "val_calib_start_date",
        "val_calib_end_date",
        "val_calib_samples",
        "val_embargo_start_date",
        "val_embargo_end_date",
        "val_embargo_samples",
    )
    return {key: stats.get(key) for key in keys if stats.get(key) is not None}


def _training_input_digests(result: Mapping[str, Any], params: Any) -> Dict[str, Any]:
    """记录模型实用的有序主键、标签和按共同实现复算的训练权重。"""
    from src.lazybull.v2.store.store_identity import frame_content_digest
    from src.lazybull.v2.train.walk_forward.training_core import _build_sample_weights

    label = result["label_column"]
    train = result["df_train_split"]
    partitions = {
        "train": train,
        "early_stopping": result["df_val_split"],
        "calibration": result["df_val_split_original"],
    }
    digests: Dict[str, Any] = {}
    for name, rows in partitions.items():
        labels = rows[[label]]
        if name != "calibration":
            labels = labels.astype(np.float32)
        digests[name] = {
            "rows": len(rows),
            "keys_sha256": frame_content_digest(rows[SAMPLE_KEY_COLUMNS]),
            "labels_sha256": frame_content_digest(labels),
        }
    weights = _build_sample_weights(train, label, params)
    digests["train"]["weights_sha256"] = (
        frame_content_digest(pd.DataFrame({"sample_weight": weights}))
        if weights is not None else None
    )
    return digests


def _assert_peer_inputs(record: Mapping[str, Any], other: Any, arm: str) -> None:
    if other and record["input_digests"] != other.get("input_digests"):
        raise TrainingError(f"{arm} v{record['version']} 与另一臂的样本、标签或权重摘要不一致")


# ── checkpoint-2 修复 2：训练帧 PIT 行业口径叠加 ─────────────────────────────


def build_pit_day_index(camp: Path) -> Dict[str, Path]:
    """结构缓存逐日文件索引（账户窗 structural_full + 历史窗 structural_history）。

    只读取 ``camp/cache/<spec16>/`` 下有完成清单的分段。两根重叠日优先使用
    评分域 ``structural_full``，历史根补早期训练日。
    """
    base = Path(camp) / "cache" / spec.spec_sha256()[:16]
    day_index: Dict[str, Path] = {}
    for name in PIT_CACHE_ROOTS:
        root = base / name
        if not root.exists():
            continue
        for seg in sorted(root.iterdir()):
            if not (seg / "MANIFEST.json").is_file():
                continue
            for path in seg.glob("*.parquet"):
                day = path.stem
                if day.isdigit() and len(day) == 8:
                    day_index.setdefault(day, path)
    return day_index


def overlay_pit_frame(
    frame: Any,
    day_index: Mapping[str, Path],
    start: str,
    end: str,
    *,
    label_columns: Sequence[str],
    emit: Callable[[str], None],
) -> Dict[str, Any]:
    """训练帧逐键叠加重算 PIT 结构帧（行业口径统一；checkpoint-2 第 2 项）。

    训练适配器的 panel/标签来自生产分区（行业中性化用构建时行业），评分结构
    帧按 PIT 行业 ⇒ 两者在行业派生列（``sw_industry_*``/``alpha_industry_*``/
    ``ind_*``/``neu_ret_*``/``neu_y_ret_*``）与中性化标签上系统性不一致。
    本函数对帧与结构帧**共同列**逐键覆写为 PIT 值：行存在于结构帧 ⇒ 全列采用
    （含 NaN——与评分域同口径；标签 NaN 由下游 dropna 落样本）；行整体缺失 ⇒
    保留生产值并计数（领域差应≈0）。原始收益类列应逐值相同（如不符会在变更
    计数中显现）。覆盖日域必须完整覆盖窗口，否则 fail-closed。
    """
    df = frame.df
    days_in_frame = sorted(df["trade_date"].astype(str).unique())
    window_days = [day for day in days_in_frame if start <= day <= end]
    missing_days = [day for day in window_days if day not in day_index]
    if missing_days:
        raise TrainingError(
            f"结构缓存未覆盖训练窗 {start}~{end}：缺 {missing_days[:3]}…（{len(missing_days)} 日）"
            "（先跑 rebuild-history）"
        )
    # 去碎片化（适配器逐列 insert 造成的碎片帧会让逐日 .loc setitem 触发整帧块拷贝）
    object.__setattr__(frame, "df", df.copy())
    df = frame.df
    key_cols = ("trade_date", "ts_code")
    first_day = pd.read_parquet(day_index[window_days[0]])
    shared_cols = [c for c in df.columns if c in first_day.columns and c not in key_cols]
    # 目标 dtype = 帧既有 dtype；整型列升为同宽浮点（PIT NaN 需要浮点宿主，
    # 训练/质量管线对 0/1 浮点语义不变）。浮点列走块级向量化，字符串列逐列处理。
    probe_block = df.loc[df.index[: min(len(df), 1000)], shared_cols]
    float_cols: List[str] = []
    other_cols: List[str] = []
    for col in shared_cols:
        dtype = probe_block[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype("float32")
            float_cols.append(col)
        elif pd.api.types.is_float_dtype(dtype):
            float_cols.append(col)
        else:
            other_cols.append(col)
    changed: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    missing_rows_total = 0
    total_rows = 0
    # 复合键（日期|代码）单列索引：块内一次 reindex 对齐，避免逐日循环
    frame_keys = df["trade_date"].astype(str) + "|" + df["ts_code"].astype(str)
    chunk_size = 100
    for chunk_start in range(0, len(window_days), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(window_days))
        chunk = window_days[chunk_start:chunk_end]
        chunk_set = set(chunk)
        frames = []
        for day in chunk:
            part = pd.read_parquet(day_index[day])
            part["__k"] = part["trade_date"].astype(str) + "|" + part["ts_code"].astype(str)
            frames.append(part)
        pit_chunk = pd.concat(frames, ignore_index=True).set_index("__k")
        # 窗口全部行都要参与（缺失 PIT 的行保留生产值并计数）
        in_window = df["trade_date"].astype(str).isin(chunk_set)
        part_index = df.index[in_window]
        if len(part_index) == 0:
            continue
        aligned = pit_chunk.reindex(frame_keys.loc[part_index])
        # 布尔逻辑全部走 numpy（Series & DataFrame 会按列对齐，爆出 N×N 方阵）
        present_np = frame_keys.loc[part_index].isin(pit_chunk.index).to_numpy()
        missing_rows_total += int((~present_np).sum())
        # 浮点块：一次 where + 一次赋值（present ⇒ 取 PIT 值，含 NaN）
        if float_cols:
            old_float = df.loc[part_index, float_cols].to_numpy(dtype=np.float64)
            new_float = aligned[float_cols].to_numpy(dtype=np.float64)
            merged_float = np.where(present_np[:, None], new_float, old_float)
            df.loc[part_index, float_cols] = merged_float
            old_na = np.isnan(old_float)
            new_na = np.isnan(new_float)
            diff_counts = ((~old_na) & (~new_na) & (new_float != old_float)).sum(axis=0)
            drop_counts = (present_np[:, None] & new_na & (~old_na)).sum(axis=0)
            for position, col in enumerate(float_cols):
                n_diff = int(diff_counts[position])
                n_drop = int(drop_counts[position])
                if n_diff:
                    changed[col] = changed.get(col, 0) + n_diff
                if n_drop:
                    dropped[col] = dropped.get(col, 0) + n_drop
        # 字符串列（少量）：逐列 object where
        for col in other_cols:
            new_vals = aligned[col].to_numpy(dtype=object)
            old_vals = df.loc[part_index, col].to_numpy(dtype=object)
            df.loc[part_index, col] = np.where(present_np, new_vals, old_vals)
        total_rows += len(part_index)
    summary = {
        "window": [start, end],
        "days": len(window_days),
        "rows": int(total_rows),
        "changed_by_column": dict(sorted(changed.items(), key=lambda kv: -kv[1])),
        "pit_nan_dropped_by_column": dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
        "rows_missing_in_pit": int(missing_rows_total),
        "changed_total": int(sum(changed.values())),
    }
    emit(
        f"      PIT 叠加：{summary['days']} 日/{summary['rows']} 行；"
        f"变更 {summary['changed_total']} 格（{len(changed)} 列）；"
        f"行缺失 {summary['rows_missing_in_pit']}"
    )
    return summary


def arm_registry_dir(camp: Path, arm: str) -> Path:
    """臂隔离注册根（不预先创建，ModelRegistry 自建）。"""
    if arm not in ARM_REGISTRY_RELATIVE:
        raise TrainingError(f"未知臂：{arm}")
    return Path(camp) / ARM_REGISTRY_RELATIVE[arm]


def build_model_config() -> Any:
    """从 value 袖规格组装 ModelConfig（n_estimators 按裁决覆盖为 1500）。"""
    from src.lazybull.v2.common.types import ModelConfig
    from src.lazybull.v2.sleeves._registry import get_sleeve_spec

    sleeve = get_sleeve_spec("value")
    hyper = dict(sleeve.hyperparams)
    hyper["n_estimators"] = RESOLVED_N_ESTIMATORS
    return ModelConfig(
        model_family=str(sleeve.model_family),
        feature_columns=list(sleeve.feature_columns),
        label_name=str(sleeve.label_name),
        hyperparams=hyper,
        train_window_months=int(sleeve.train_window_months),
        val_window_months=None,
        ensemble_seeds=tuple(int(seed) for seed in sleeve.ensemble_seeds),
        label_spec=dict(sleeve.label_spec),
        feature_recipe=dict(sleeve.feature_recipe),
        split_spec=dict(sleeve.split_spec),
        domain=str(sleeve.domain),
    )


def build_main_board_codes(root: Path) -> set:
    """域股票池（``universe/domains.py`` 单一来源；与生产编排器同口径）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage
    from src.lazybull.universe.domains import domain_market_whitelist
    from src.lazybull.v2.sleeves._registry import get_sleeve_spec
    from src.lazybull.v2.train.walk_forward.training_core import _build_main_board_codes

    loader = DataLoader(Storage(root_path=str(Path(root) / "data")))
    stock_basic = loader.load_stock_basic()
    if stock_basic is None or len(stock_basic) == 0:
        raise TrainingError("无法加载 stock_basic（域股票池构建失败）")
    markets = tuple(domain_market_whitelist(get_sleeve_spec("value").domain))
    return _build_main_board_codes(stock_basic, markets=markets)


def load_trade_calendar(root: Path) -> List[str]:
    """SSE 开市日升序全表（缝换算与账户日历共用）。"""
    import pandas as pd

    path = Path(root) / "data/clean/trade_cal.parquet"
    if not path.exists():
        raise TrainingError(f"交易日历缺件：{path}")
    table = pd.read_parquet(path, columns=["exchange", "is_open", "cal_date"])
    return sorted(
        table.loc[table.exchange.eq("SSE") & table.is_open.astype(int).eq(1), "cal_date"]
        .astype(str)
        .unique()
        .tolist()
    )


def resolve_fold_seams(fold: Mapping[str, Any], calendar: Sequence[str]) -> Dict[str, str]:
    """单折显式四日期（与生产 ``execute_split_training`` 逐字同源）。

    生产链路：``trade_dates = 窗内日历日``；``label_delta = max(horizon+1, 5)``；
    ``resolve_explicit_split_dates(train_start, train_end, val_ratio, ...)``。
    换算结果与原训练日志逐折对账（val_start 必须 == 日志 es_start）。
    """
    from src.lazybull.v2.train.walk_forward.split_training import _label_delta
    from src.lazybull.v2.train.walk_forward.training_core import resolve_explicit_split_dates

    outer_start = str(fold["outer_train_start"])
    outer_end = str(fold["outer_train_end"])
    trade_dates = [day for day in calendar if outer_start <= day <= outer_end]
    if not trade_dates:
        raise TrainingError(f"折窗无交易日：{outer_start}~{outer_end}")
    params_delta = _label_delta(spec.SPLIT_CLOCK["label_name"])
    if params_delta != int(spec.SPLIT_CLOCK["label_maturity_offset"]):
        raise TrainingError(
            f"label_delta 与冻结成熟偏移不一致：{params_delta} != "
            f"{spec.SPLIT_CLOCK['label_maturity_offset']}"
        )
    t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
        outer_start, outer_end, float(spec.SPLIT_CLOCK["val_ratio"]), trade_dates, params_delta
    )
    log_es_start = str(fold.get("es_start") or "")
    if log_es_start and v_start != log_es_start:
        raise TrainingError(
            f"折 {fold.get('model_version')} 显式缝与日志不符：val_start={v_start} "
            f"!= log es_start={log_es_start}（口径漂移，fail-fast）"
        )
    return {
        "train_start": t_start,
        "train_end": t_end,
        "val_start": v_start,
        "val_end": v_end,
    }


def make_frame_provider(root: Path, config: Any, params: Any) -> Tuple[Any, Callable]:
    """构造 V2Trainer（panel 模式）并返回 ``(trainer, frame_provider)``。

    trainer 只用于 ``make_frame_provider`` / ``_register_fold``（生产同源组装）；
    逐种子训练由本模块驱动（需要 ``feature_columns_override``）。
    """
    from src.lazybull.v2.store.data_store import PanelDataStore
    from src.lazybull.v2.train.frame_cache import TrainingFrameCache
    from src.lazybull.v2.train.trainer import V2Trainer

    base = Path(root)
    store = PanelDataStore(base / "data")
    trainer = V2Trainer(
        registry=None,  # type: ignore[arg-type]
        main_board_codes=set(),
        source_mode="panel",
        frame_cache=TrainingFrameCache(),
    )
    provider = trainer.make_frame_provider(config, params, store)
    return trainer, provider


def train_seed_models(
    frame: Any,
    params: Any,
    main_board_codes: set,
    seeds: Sequence[int],
    arm_columns: Sequence[str],
    *,
    explicit_train_end: str | None,
    explicit_val_start: str | None,
    is_deploy: bool,
    emit: Callable[[str], None],
) -> Tuple[List[Any], Dict[str, Any], List[Dict[str, Any]]]:
    """逐种子训练 + 全保留集成组装（协议 §5.2-3：42/61/82 三种子、保留三者）。

    与 ``_build_ensemble_sub_models`` 的差异只有一处：``feature_columns_override``
    恒为臂列（生产代码在多种子路径下首种子不投影、后续对齐首种子列；
    本协议要求三种子全部投影到同一臂列，§5.2-4）。
    """
    from src.lazybull.v2.train.walk_forward.training_core import (
        _score_sub_models,
        _select_ensemble_validation_result,
        _train_model_on_window,
    )

    sub_models: List[Any] = []
    records: List[Dict[str, Any]] = []
    input_digests: Optional[Dict[str, Any]] = None
    started = time.perf_counter()
    for index, seed in enumerate(seeds, 1):
        emit(f"      种子 {seed}（{index}/{len(seeds)}）训练中…")
        result = _train_model_on_window(
            frame,
            params,
            main_board_codes,
            runtime_derivers=None,
            random_state_override=int(seed),
            feature_columns_override=list(arm_columns),
            explicit_train_end=explicit_train_end,
            explicit_val_start=explicit_val_start,
        )
        columns = list(result["feature_columns"])
        if columns != list(arm_columns):
            raise TrainingError(
                f"种子 {seed} 最终列与臂列不符：{len(columns)} vs {len(arm_columns)}"
                f"（首差异 {next((c for c, a in zip(columns, arm_columns) if c != a), '?')}）"
            )
        current_digests = _training_input_digests(result, params)
        if input_digests is not None and current_digests != input_digests:
            raise TrainingError(f"种子 {seed} 与同臂其他种子的样本、标签或权重摘要不一致")
        input_digests = current_digests
        sub_models.append(result["model"])
        records.append({"train_result": result})
    emit(f"      三种子完成 {time.perf_counter() - started:.0f}s；集成筛选中…")
    kept_models, base_result, retained, meta_part = _score_sub_models(
        records, params, [20, 30, 100, 300], 1.0, len(seeds), "部署" if is_deploy else ""
    )
    if len(retained) != len(records):
        raise TrainingError("keep_top_ratio=1.0 下种子被筛除（协议要求保留三者）")
    validation_result = _select_ensemble_validation_result(retained)
    meta_part["_ensemble_validation_result"] = validation_result
    meta_part["input_digests"] = input_digests
    return kept_models, base_result, meta_part


def _assemble_full_train_params(
    config: Any,
    params: Any,
    train_params: Mapping[str, Any],
    feature_columns: Sequence[str],
    model: Any,
    *,
    ensemble_seeds: Sequence[int],
    data_stats: Mapping[str, Any] | None,
    train_start: str,
    val_end: str,
    is_deploy: bool,
) -> Dict[str, Any]:
    """注册参数组装（``V2Trainer._register_fold`` 逐键同源；部署补 ``is_deploy``）。

    生产两条路径的公共段 = ``_register_fold``（折）与 ``_assemble_deploy_full_params``
    （部署）；差异仅部署多 ``is_deploy=True`` 一键。本函数逐键复刻公共段。
    """
    from src.lazybull.v2.common.types import to_plain_data
    from src.lazybull.v2.store.store_identity import canonical_json_sha256
    from src.lazybull.v2.train.core import (
        attach_cashflow_quality_train_params,
        attach_cons_revision_schema_version,
        attach_dividend_train_params,
        attach_feature_missing_rate_train_params,
    )
    from src.lazybull.v2.train.ensemble import EnsembleModel
    from src.lazybull.v2.train.walk_forward.training_core import (
        _build_feature_flag_train_params,
    )

    full: Dict[str, Any] = dict(train_params)
    attach_cons_revision_schema_version(full, params.enable_consensus_revision_features)
    attach_cashflow_quality_train_params(
        full, params.enable_cashflow_quality_features, feature_columns=list(feature_columns)
    )
    attach_dividend_train_params(full, params.enable_dividend_policy_features)
    full.update(
        {
            "algorithm": params.algorithm,
            "task": params.task,
            "label_transform": (params.label_transform if params.task == "regression" else None),
            "winsorize_p": (params.winsorize_p if params.label_transform == "cs_zscore" else None),
            "pos_quantile": params.pos_quantile if params.task == "classification" else None,
            "pos_topk": params.pos_topk if params.task == "classification" else None,
            "scale_pos_weight_manual": params.scale_pos_weight is not None,
            **_build_feature_flag_train_params(params),
            "freshness_strategy": params.freshness_strategy,
            "event_freshness_half_life_days": params.event_freshness_half_life_days,
        }
    )
    if is_deploy:
        full["is_deploy"] = True
    spec_sources = {
        "label_spec": config.label_spec,
        "feature_recipe": config.feature_recipe,
        "split_spec": config.split_spec,
    }
    for name, value in spec_sources.items():
        if value and "version" not in value:
            raise TrainingError(f"ModelConfig.{name} 缺版本号（F10：规格须带版本号）")
        spec_dict = to_plain_data(value) if value is not None else None
        full[name] = spec_dict
        full[f"{name}_digest"] = canonical_json_sha256(spec_dict) if spec_dict is not None else None
    attach_feature_missing_rate_train_params(
        full,
        dict(data_stats) if data_stats else None,
        list(feature_columns),
        train_start=str(train_start),
        train_end=str(val_end),
    )
    full["domain"] = params.stock_domain
    full.update(
        {
            "factor_prune": params.factor_prune,
            "factor_exclude_file": params.factor_exclude_file,
            "feature_stability_filter": params.feature_stability_filter,
            "max_feature_missing_ratio": params.max_feature_missing_ratio,
            "rank_weight_enabled": params.rank_weight_enabled,
            "rank_weight_topk": params.rank_weight_topk,
            "rank_weight": params.rank_weight,
            "rank_weight_topk_weight_mode": params.rank_weight_topk_weight_mode,
            "time_decay_half_life": params.time_decay_half_life,
        }
    )
    if isinstance(model, EnsembleModel):
        full["ensemble_offsets"] = 0
        full["ensemble_seeds"] = [int(seed) for seed in ensemble_seeds]
        full["ensemble_seed_keep_top_ratio"] = params.ensemble_seed_keep_top_ratio
        full["ensemble_seed_keep_min_models"] = params.ensemble_seed_keep_min_models
        full["ensemble_n_models"] = model.n_models
    return full


def register_fold_model(
    config: Any,
    params: Any,
    registry: Any,
    model: Any,
    base_result: Mapping[str, Any],
    meta_part: Mapping[str, Any],
    *,
    train_start: str,
    val_end: str,
    is_deploy: bool,
    topk_values: Sequence[int],
    emit: Callable[[str], None],
) -> Tuple[int, Dict[str, Any]]:
    """注册单折/部署模型（直接 ``ModelRegistry.register_model``，生产同源组装）。"""
    from src.lazybull.v2.train.core import evaluate_validation_daily
    from src.lazybull.v2.train.ensemble import EnsembleModel

    ensemble_model = EnsembleModel(list(model))
    validation_result = meta_part["_ensemble_validation_result"]
    val_daily: Dict[str, Any] = {}
    df_val = validation_result.get("df_val_split_original")
    if df_val is not None and len(df_val) > 0:
        val_daily = evaluate_validation_daily(
            model=ensemble_model,
            df_val=df_val,
            feature_columns=validation_result["feature_columns"],
            original_return_col=validation_result["label_column"],
            task=params.task,
            topk_values=list(topk_values),
        )
    performance_metrics = {
        "train": base_result["train_metrics"],
        "validation": base_result["val_metrics"],
        "validation_daily": val_daily,
        "test": {},
        "test_daily": {},
    }
    feature_columns = list(base_result["feature_columns"])
    full_params = _assemble_full_train_params(
        config,
        params,
        base_result["train_params"],
        feature_columns,
        ensemble_model,
        ensemble_seeds=list(params.ensemble_seeds or ()),
        data_stats=base_result.get("data_stats"),
        train_start=str(train_start),
        val_end=str(val_end),
        is_deploy=bool(is_deploy),
    )
    version = registry.register_model(
        model=ensemble_model,
        model_type=f"{params.algorithm}_{params.task}_wf",
        train_start_date=str(train_start),
        train_end_date=str(val_end),
        feature_columns=feature_columns,
        label_column=params.label_column,
        n_samples=int(base_result["X_train_len"]) + int(base_result["X_val_len"]),
        train_params=full_params,
        performance_metrics=performance_metrics,
    )
    emit(
        f"      注册完成：registered_v{version}"
        f"（列 {len(feature_columns)}，样本 {base_result['X_train_len']}+{base_result['X_val_len']}）"
    )
    return int(version), dict(full_params)


def registry_model_sha(registry: Any, version: int) -> str:
    """注册模型文件 SHA256（分片复用登记用）。"""
    from scripts.v2_p7.common import sha256_file

    metadata = registry._load_metadata(int(version))
    return sha256_file(registry.models_dir / str(metadata["model_file"]))


def _verify_registry_columns(registry: Any, version: int, arm_columns: Sequence[str]) -> None:
    """注册后重载核对：metadata 特征列与臂列逐位一致（协议 §5.2-6「隔离注册→重载」）。"""
    _, metadata = registry.load_model(version=int(version), strict_version_check=True)
    columns = [str(item) for item in metadata.get("feature_columns") or []]
    if columns != list(arm_columns):
        raise TrainingError(f"v{version} 注册列与臂列不符：{len(columns)} vs {len(arm_columns)}")


def train_arm_folds(
    root: Path,
    camp: Path,
    arm: str,
    routes: Mapping[int, Mapping[str, Any]],
    clock: Mapping[str, Any],
    state: Dict[str, Any],
    emit: Callable[[str], None],
    *,
    resume: bool,
    persist: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """单臂 14 折 + 部署训练（断点续训：已注册且校验通过的折直接跳过）。"""
    from src.lazybull.v2.common.model_registry import ModelRegistry
    from src.lazybull.v2.train.trainer import params_from_config

    base = Path(root)
    config = build_model_config()
    params = params_from_config(config)
    main_board_codes = build_main_board_codes(base)
    registry = ModelRegistry(models_dir=str(arm_registry_dir(Path(camp), arm)))
    trainer, provider = make_frame_provider(base, config, params)
    calendar = load_trade_calendar(base)
    topk_values = [20, 30, 100, 300]
    pit_day_index = build_pit_day_index(Path(camp))
    if not pit_day_index:
        raise TrainingError("结构缓存缺件（rebuild/rebuild-history 未跑）")

    fold_rows = sorted(
        (row for row in clock["rows"] if row.get("kind") == "fold"),
        key=lambda row: int(row["split_index"]),
    )
    if len(fold_rows) != len(spec.FOLD_VERSIONS):
        raise TrainingError(f"折数不符：{len(fold_rows)} != {len(spec.FOLD_VERSIONS)}")

    records: List[Dict[str, Any]] = []
    peer = "R" if arm == "F" else "F"
    for row in fold_rows:
        version_frozen = int(row["model_version"])
        route = routes.get(version_frozen)
        if route is None:
            raise TrainingError(f"缺 v{version_frozen} 路由")
        done = state.get("folds", {}).get(str(version_frozen), {}).get(arm)
        if resume and done:
            registered = int(done["registered_version"])
            # 分片复用内容核验：注册模型可加载且列与臂路由逐位一致（协议 §2.3）
            _verify_registry_columns(registry, registered, list(route[arm]))
            sha = registry_model_sha(registry, registered)
            emit(f"    折 {version_frozen} 已注册 v{registered}（SHA {sha[:12]}…，复用）")
            done = {**done, "model_sha256": sha}
            state.setdefault("folds", {}).setdefault(str(version_frozen), {})[arm] = done
            records.append({**done, "version": version_frozen})
            continue
        record = _train_one_fold(
            row,
            arm,
            list(route[arm]),
            config,
            params,
            main_board_codes,
            registry,
            trainer,
            provider,
            calendar,
            state,
            topk_values,
            emit,
            pit_day_index=pit_day_index,
        )
        other = state.get("folds", {}).get(str(version_frozen), {}).get(peer)
        _assert_peer_inputs(record, other, arm)
        records.append(record)
        if persist is not None:
            persist()

    deploy_record = _train_deploy_arm(
        base,
        arm,
        routes,
        clock,
        config,
        params,
        main_board_codes,
        registry,
        trainer,
        provider,
        state,
        topk_values,
        emit,
        resume=resume,
        pit_day_index=pit_day_index,
    )
    _assert_peer_inputs(deploy_record, state.get("deploy", {}).get(peer), arm)
    records.append(deploy_record)

    return {
        "arm": arm,
        "registry_relative": ARM_REGISTRY_RELATIVE[arm],
        "config_digest": _config_digest(config),
        "n_estimators": RESOLVED_N_ESTIMATORS,
        "records": records,
    }


def _train_one_fold(
    row: Mapping[str, Any],
    arm: str,
    arm_columns: List[str],
    config: Any,
    params: Any,
    main_board_codes: set,
    registry: Any,
    trainer: Any,
    provider: Callable,
    calendar: Sequence[str],
    state: Dict[str, Any],
    topk_values: Sequence[int],
    emit: Callable[[str], None],
    pit_day_index: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    """单折训练 + 注册 + 状态登记（含段净化事实随记录落 manifest）。"""
    version_frozen = int(row["model_version"])
    seams = resolve_fold_seams(row, calendar)
    emit(
        f"    折 {version_frozen}：窗 {seams['train_start']}~{seams['val_end']}，"
        f"缝 {seams['train_end']}|{seams['val_start']}，列 {len(arm_columns)}"
    )
    frame = provider(seams["train_start"], seams["val_end"])
    if pit_day_index is not None:
        # TrainingFrame 为 frozen dataclass；overlay 摘要以旁路属性随帧携带
        object.__setattr__(
            frame,
            "pit_overlay",
            overlay_pit_frame(
                frame,
                pit_day_index,
                seams["train_start"],
                seams["val_end"],
                label_columns=spec.LABEL_VALUE_COLUMNS,
                emit=emit,
            ),
        )
    usable_day = str(row.get("es_last_usable_sample") or "")
    with es_maturity_purge(usable_day) as purge_state:
        models, base_result, meta_part = train_seed_models(
            frame,
            params,
            main_board_codes,
            list(params.ensemble_seeds or ()),
            arm_columns,
            explicit_train_end=seams["train_end"],
            explicit_val_start=seams["val_start"],
            is_deploy=False,
            emit=emit,
        )
    if usable_day:
        assert_es_purged(purge_state, usable_day, emit)
    registered, full_params = register_fold_model(
        config,
        params,
        registry,
        models,
        base_result,
        meta_part,
        train_start=seams["train_start"],
        val_end=seams["val_end"],
        is_deploy=False,
        topk_values=topk_values,
        emit=emit,
    )
    _verify_registry_columns(registry, registered, arm_columns)
    # 帧缓存按折清空（与生产编排器同口径；跨折保留会累积数 GB 触发 OOM）
    trainer.frame_cache.clear()
    record = {
        "version": version_frozen,
        "registered_version": registered,
        "model_sha256": registry_model_sha(registry, registered),
        "seams": seams,
        "columns_n": len(arm_columns),
        "es_purged_samples": int(row.get("es_purged_samples") or 0),
        "es_last_usable_sample": str(row.get("es_last_usable_sample") or ""),
        "calib_purged_samples": int(row.get("calib_purged_samples") or 0),
        "calib_last_usable_sample": str(row.get("calib_last_usable_sample") or ""),
        "split_key_summary": _split_key_summary(base_result),
        "input_digests": {**meta_part["input_digests"], **purge_state["keys"]},
        "pit_overlay": getattr(frame, "pit_overlay", None),
        "full_train_params_digest": _params_digest(full_params),
    }
    state.setdefault("folds", {}).setdefault(str(version_frozen), {})[arm] = {
        key: value for key, value in record.items() if key != "version"
    }
    return record


def _train_deploy_arm(
    base: Path,
    arm: str,
    routes: Mapping[int, Mapping[str, Any]],
    clock: Mapping[str, Any],
    config: Any,
    params: Any,
    main_board_codes: set,
    registry: Any,
    trainer: Any,
    provider: Callable,
    state: Dict[str, Any],
    topk_values: Sequence[int],
    emit: Callable[[str], None],
    *,
    resume: bool,
    pit_day_index: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    """部署模型训练 + 注册（比例路径；日志外层窗为唯一权威）。"""
    deploy = clock["deploy"]
    deploy_route = routes.get(spec.DEPLOY_VERSION)
    if deploy_route is None:
        raise TrainingError(f"缺 v{spec.DEPLOY_VERSION} 路由")
    deploy_columns = list(deploy_route[arm])
    done = state.get("deploy", {}).get(arm)
    if resume and done:
        registered = int(done["registered_version"])
        _verify_registry_columns(registry, registered, deploy_columns)
        emit(
            f"    部署已注册 v{registered}（SHA {registry_model_sha(registry, registered)[:12]}…，复用）"
        )
        return {**done, "deploy_sha256": registry_model_sha(registry, registered)}
    log = deploy_log_row(base)
    emit(
        f"    部署 {spec.DEPLOY_VERSION}：比例路径窗 {log['train_start']}~{log['train_end']}，"
        f"列 {len(deploy_columns)}"
    )
    frame = provider(log["train_start"], log["train_end"])
    if pit_day_index is not None:
        object.__setattr__(
            frame,
            "pit_overlay",
            overlay_pit_frame(
                frame,
                pit_day_index,
                log["train_start"],
                log["train_end"],
                label_columns=spec.LABEL_VALUE_COLUMNS,
                emit=emit,
            ),
        )
    usable_day = str(deploy.get("es_last_usable_sample") or "")
    with es_maturity_purge(usable_day) as purge_state:
        models, base_result, meta_part = train_seed_models(
            frame,
            params,
            main_board_codes,
            list(params.ensemble_seeds or ()),
            deploy_columns,
            explicit_train_end=None,
            explicit_val_start=None,
            is_deploy=True,
            emit=emit,
        )
    if usable_day:
        assert_es_purged(purge_state, usable_day, emit)
    registered, full_params = register_fold_model(
        config,
        params,
        registry,
        models,
        base_result,
        meta_part,
        train_start=log["train_start"],
        val_end=log["train_end"],
        is_deploy=True,
        topk_values=topk_values,
        emit=emit,
    )
    _verify_registry_columns(registry, registered, deploy_columns)
    trainer.frame_cache.clear()
    deploy_record = {
        "arm": arm,
        "version": spec.DEPLOY_VERSION,
        "registered_version": registered,
        "model_sha256": registry_model_sha(registry, registered),
        "train_start": log["train_start"],
        "train_end": log["train_end"],
        "columns_n": len(deploy_columns),
        "es_start": str(deploy.get("es_start") or ""),
        "es_end": str(deploy.get("es_end") or ""),
        "es_purged_samples": int(deploy.get("es_purged_samples") or 0),
        "es_last_usable_sample": str(deploy.get("es_last_usable_sample") or ""),
        "split_key_summary": _split_key_summary(base_result),
        "input_digests": {**meta_part["input_digests"], **purge_state["keys"]},
        "pit_overlay": getattr(frame, "pit_overlay", None),
        "full_train_params_digest": _params_digest(full_params),
    }
    state.setdefault("deploy", {})[arm] = deploy_record
    return deploy_record


def resolve_day_route_version(root: Path, day: str) -> Optional[int]:
    """日期 → 冻结路由版本（折测试域；接缝沿用末折；部署段用部署版本）。

    与 run.py ``_version_map`` 同源（``fold_clock.build_timeline``），供打分侧
    逐日绑定模型；账户窗外的日期返回 None。
    """
    from scripts.v2_p7.cashflow_sleeve_split import fold_clock

    base = Path(root)
    clock = fold_clock.build_fold_clock(base)
    timeline = fold_clock.build_timeline(
        [row for row in clock["rows"] if row.get("kind") == "fold"], base
    )
    day = str(day)
    for entry in timeline:
        if str(entry["first"]) <= day <= str(entry["last"]):
            return int(entry["model_version"])
    return None


def deploy_log_row(root: Path) -> Dict[str, str]:
    """部署段外层训练窗（``ml_train_runs.csv`` deploy 行，唯一权威）。"""
    import pandas as pd

    path = Path(root) / spec.TRAINING_LOG["relative"]
    table = pd.read_csv(path)
    row = table[table[spec.TRAINING_LOG["version_column"]].astype(int) == spec.DEPLOY_VERSION]
    if len(row) != 1:
        raise TrainingError(f"部署日志行数不符：{len(row)}")
    record = row.iloc[0]
    return {
        "train_start": str(_maybe_day(record[spec.TRAINING_LOG["columns"]["outer_train_start"]])),
        "train_end": str(_maybe_day(record[spec.TRAINING_LOG["columns"]["outer_train_end"]])),
    }


def _maybe_day(value: Any) -> str:
    text = "" if value is None or str(value) == "nan" else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _params_digest(full_params: Mapping[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        full_params, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_digest(config: Any) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {
            "feature_columns": list(config.feature_columns),
            "label_name": config.label_name,
            "hyperparams": dict(config.hyperparams),
            "train_window_months": config.train_window_months,
            "ensemble_seeds": list(config.ensemble_seeds),
            "label_spec": dict(config.label_spec or {}),
            "feature_recipe": dict(config.feature_recipe or {}),
            "split_spec": dict(config.split_spec or {}),
            "domain": config.domain,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ARM_REGISTRY_RELATIVE",
    "RESOLVED_N_ESTIMATORS",
    "TrainingError",
    "arm_registry_dir",
    "build_main_board_codes",
    "build_model_config",
    "deploy_log_row",
    "load_trade_calendar",
    "make_frame_provider",
    "resolve_day_route_version",
    "resolve_fold_seams",
    "train_arm_folds",
    "train_seed_models",
]
