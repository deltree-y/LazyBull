# -*- coding: utf-8 -*-
"""P3 T5 F11 数据影响归因——a~e 节计算（P3 规划 §7.5 / §13.3.7）。

- **a 节**（逐折 × 消费列输入差异）：``attribute_fold_inputs`` 按窗口逐日比较
  panel vs cs_train 两源在该折模型实际消费列上的值。数值口径复用
  ``panel_reconcile._compare_day``（atol=1e-6、NaN==NaN 一致）；词表分类复用
  ``panel_reconcile._classify_attrib_column``（内部即 ``reconcile_rules.attribute_column``
  词表唯一来源，禁止复制）。**流式聚合**：panel 按月读消费列、cs_train 逐日读，
  逐日结果喂入 ``ColumnImpactAggregator`` 后即释放，全窗口大帧不常驻内存；
  不做 readiness 重跑（不经 ``load_training_frame``）。**完整覆盖前置**
  （R2-T5A-04）：预期日域 = trade_cal ``is_open=1`` ∩ boundary 窗口（独立于分区
  清单），``days_compared == 预期日数`` 且 ``days_compared > 0`` 才可闭合；任一源
  缺日 ⇒ missing 登记并破坏闭合；窗口内非开市日显式登记 excluded 计数。
  **跨折单日结果缓存**（P2-7）：``DayCompareCache`` 键 = ``(day, 消费列集指纹)``，
  纯加速，命中与否不改变结果。
- **b 节**（预测影响，R2-T5A-06）：逐折读两臂 ``predictions.parquet``——键集
  对称差计数（同日换股候选日分列）+ 交集键 ``%.17g`` 逐值 diff（同键改分）+
  指纹/n_rows/score_days 元数据对比；产物或指纹增键缺失 ⇒ 必需证据缺失 FAIL。
- **c/d 节**（成交/持仓影响）：键集对称差 + 交集数值差；同键多行先按键聚合
  （P3-3：trades 同方向 shares/amount 求和、price=Σamount/Σshares 推导；holdings
  数值列求和），重复键分组计数随节登记，禁止 merge 笛卡尔展开失真。
- **e 节**（净值影响）：chain_nav 逐折按日对齐差（split_index 显式 ``astype(int)``
  统一 dtype，P3-8）+ 逐折 daily 汇总差。
- **输出级一致性**（R2-T5A-05）：``evaluate_output_consistency``——归因闭合 =
  a 节闭合 ∧ 输出级一致性（规则① 零输入差异 ⇒ 零输出差异；规则② 逐折对应：
  输出劣化折须该折 a 节有已登记消费列差异）。

「禁止把列级归因直接当交易路径等价」：a 节差异计数是输入面证据，b~e 节是输出面
影响证据，两者分列登记、不互相推导。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3 import t5_attribution_io as io  # noqa: E402

# 数值口径与 F11 词表分类的唯一来源（禁止复制语义，一律 import）
from src.lazybull.v2.store.panel_reconcile import (  # noqa: E402
    _classify_attrib_column,
    _classify_strict_column,
    _compare_day,
    _slice_day,
)
from src.lazybull.v2.store.reconcile_rules import (  # noqa: E402
    ATTRIB_VIOLATION_TEXT,
    CYQ5_COLUMNS,
    CYQ5_LAST_DAY,
    SCHEMA_EVOLUTION_END,
    is_d11_sparse_companion_class,
)

_SAMPLE_CAP = 5


def _is_registered_cols_only_panel(col: str, days: Sequence[str]) -> bool:
    """panel 单侧列结构差异的登记白名单（复用 P1 冻结口径，禁止复制语义）。

    - CYQ5 物化列且末日 ≤ ``CYQ5_LAST_DAY``（参照侧 cyq_perf 数据态差异）；
    - 任意列且末日 ≤ ``SCHEMA_EVOLUTION_END``（2012 冷启动 schema 演进）。
    """
    if not days:
        return False
    last = max(days)
    return (col in CYQ5_COLUMNS and last <= CYQ5_LAST_DAY) or last <= SCHEMA_EVOLUTION_END


#: b 节证据粒度口径注记（预测行已按跨路契约逐折持久化，逐值归因可兑现；
#: 两源行集语义不同，禁止跨源当等价判据）
PREDICTION_CALIBER_NOTE = (
    "证据粒度 = 逐折 predictions.parquet 键集对称差 + 交集键 %.17g 逐值 diff + "
    "fingerprint/n_rows/score_days 元数据（预测行持久化跨路契约）；"
    "臂②=cs_train_verify 源行集（隐含标签非空）、臂③=panel 源行集（无标签条件），"
    "两源口径不同（t05b_fingerprint reviewer1-S7 注记），指纹不一致不直接判实现漂移，"
    "仅登记为数据面切换影响证据"
)

#: 预测行键列（跨路契约：trade_date str YYYYMMDD / ts_code str / score float64）
_PREDICTION_KEY_COLUMNS = ("trade_date", "ts_code")

#: 成交/持仓键与数值列（runs 契约列集，见 v2/evidence/runs_schema.py）
_TRADE_KEY_COLUMNS = ("trade_date", "ts_code", "action")
_TRADE_VALUE_COLUMNS = ("price", "shares", "amount")
_HOLDINGS_KEY_COLUMNS = ("trade_date", "ts_code")
_HOLDINGS_VALUE_COLUMNS = ("shares", "market_value", "weight")


# ---------------------------------------------------------------------------
# a 节：逐折 × 消费列输入差异（流式聚合 + F11 词表分类）
# ---------------------------------------------------------------------------


class _ColumnAccumulator:
    """单列流式累计态（逐日喂入、全窗口帧不常驻）。"""

    def __init__(self) -> None:
        self.over_rows = 0
        self.total_rows = 0
        self.max_abs_diff = 0.0
        self.diff_days: list[tuple[str, int, int, int]] = []
        self.samples: list[dict[str, Any]] = []


class ColumnImpactAggregator:
    """逐折输入差异流式聚合器：逐日消费 ``_compare_day`` 结果，累计列级口径。

    闭合（closed）定义 = 差异列 100% 落入 F11 词表登记类且无拒绝项：
    任一列 ``boundary_violation``（越界/第六类未归因）或行集差异日或消费列结构
    单侧缺失或单侧缺日 ⇒ 不闭合；``days_compared=0`` 绝对不得闭合（R2-T5A-04：
    零比较日 = 完整覆盖证据缺失，禁止当闭合）。
    """

    def __init__(self) -> None:
        self._columns: dict[str, _ColumnAccumulator] = {}
        self.row_diff_days: list[dict[str, Any]] = []
        self.cols_only_cs_train: dict[str, list[str]] = {}
        self.cols_only_panel: dict[str, list[str]] = {}
        self.missing_cs_train_days: list[str] = []
        self.missing_panel_days: list[str] = []
        self.days_compared = 0

    def add_day(self, day_result: Mapping[str, Any]) -> None:
        """喂入单日 ``_compare_day`` 结果（仅差异列与行集/列结构差入累计）。"""
        self.days_compared += 1
        date = str(day_result["date"])
        if day_result["n_only_ref"] or day_result["n_only_panel"]:
            self.row_diff_days.append(
                {
                    "date": date,
                    "n_only_cs_train": day_result["n_only_ref"],
                    "n_only_panel": day_result["n_only_panel"],
                    "sample_only_cs_train": day_result["sample_only_ref"],
                    "sample_only_panel": day_result["sample_only_panel"],
                }
            )
        for col in day_result["cols_only_ref"]:
            self.cols_only_cs_train.setdefault(col, []).append(date)
        for col in day_result["cols_only_panel"]:
            self.cols_only_panel.setdefault(col, []).append(date)
        for col, n_rows in day_result.get("col_rows", {}).items():
            self._acc(col).total_rows += int(n_rows)
        for col, stat in day_result["columns"].items():
            self._add_column_day(col, date, stat)

    def _acc(self, col: str) -> _ColumnAccumulator:
        return self._columns.setdefault(col, _ColumnAccumulator())

    def _add_column_day(self, col: str, date: str, stat: Mapping[str, Any]) -> None:
        acc = self._acc(col)
        acc.over_rows += int(stat["over_rows"])
        acc.max_abs_diff = max(acc.max_abs_diff, float(stat["max_abs_diff"]))
        acc.diff_days.append(
            (date, int(stat["over_rows"]), int(stat["rows"]), int(stat["over_nan_rows"]))
        )
        if len(acc.samples) < _SAMPLE_CAP:
            acc.samples.extend(_rename_samples(stat["samples"])[: _SAMPLE_CAP - len(acc.samples)])

    def finalize(self) -> dict[str, Any]:
        """收尾：列级报告 + F11 词表分类 + 闭合判定。"""
        columns: dict[str, Any] = {}
        rejected: dict[str, Any] = {}
        for col, acc in sorted(self._columns.items()):
            if acc.over_rows == 0:
                continue
            klass, violated = _classify_attrib_column(col, acc.diff_days, acc.max_abs_diff)
            attribution_mode = "attrib"
            if violated and klass == ATTRIB_VIOLATION_TEXT:
                # F11 归因门词表 = 归因分类器 ∪ strict 分类器已登记的 D-11 伴随稀疏单点。
                # 兜底只允许「归因门第六类（词表无类）」的伴随缺口：归因门已给出明确类
                # 而越界者（D-14 专项三上限 / D-04 登记窗等）保留**否决效力**，禁止被
                # strict 分类器覆盖（R2-T5AE-R1-01——稀疏伴随分支无超幅检查，
                # 不区分拒绝原因会重新打开「稀疏但超幅」绕过专项上限的敞口）。
                strict_klass, strict_violated = _classify_strict_column(
                    col, acc.diff_days, acc.max_abs_diff
                )
                if not strict_violated and is_d11_sparse_companion_class(strict_klass):
                    klass, violated = strict_klass, False
                    attribution_mode = "strict_d11_fallback"
            entry = {
                "over_rows": acc.over_rows,
                "total_rows": acc.total_rows,
                "over_share": acc.over_rows / acc.total_rows if acc.total_rows else 0.0,
                "max_abs_diff": acc.max_abs_diff,
                "attribution": klass,
                "attribution_mode": attribution_mode,
                "boundary_violation": bool(violated),
                "diff_day_summary": _diff_day_summary(acc.diff_days),
                "samples": acc.samples,
            }
            columns[col] = entry
            if violated:
                rejected[col] = entry
        legal_cols_only_panel: dict[str, Any] = {}
        illegal_cols_only_panel: dict[str, Any] = {}
        for col, days in sorted(self.cols_only_panel.items()):
            info = {"days": len(days), "first": min(days), "last": max(days)}
            if _is_registered_cols_only_panel(col, days):
                legal_cols_only_panel[col] = info
            else:
                illegal_cols_only_panel[col] = info
        closed = (
            self.days_compared > 0
            and not rejected
            and not self.row_diff_days
            and not self.cols_only_cs_train
            and not illegal_cols_only_panel
            and not self.missing_cs_train_days
            and not self.missing_panel_days
        )
        return {
            "days_compared": self.days_compared,
            "columns": columns,
            "rejected_columns": rejected,
            "row_diff_days": self.row_diff_days,
            "cols_only_cs_train": {
                col: {"days": len(days), "first": min(days), "last": max(days)}
                for col, days in sorted(self.cols_only_cs_train.items())
            },
            "cols_only_panel": illegal_cols_only_panel,
            "registered_cols_only_panel": legal_cols_only_panel,
            "missing_cs_train_days": self.missing_cs_train_days,
            "missing_panel_days": self.missing_panel_days,
            "closed": closed,
        }


def _rename_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """样例键重命名（replay_compare 冻结键 replay/cs_train → 本报告口径两侧实名）。

    调用 ``_compare_day(date, ref=cs_train 日帧, panel_day)`` 时 left=ref=cs_train、
    right=panel，故 ``replay`` 键承载 cs_train 侧值、``cs_train`` 键承载 panel 侧值。
    """
    renamed = []
    for sample in samples:
        renamed.append(
            {
                "ts_code": sample["ts_code"],
                "cs_train_value": sample.get("replay"),
                "panel_value": sample.get("cs_train"),
                "abs_diff": sample.get("abs_diff"),
            }
        )
    return renamed


def _diff_day_summary(diff_days: Sequence[tuple[str, int, int, int]]) -> dict[str, Any]:
    return {
        "days": len(diff_days),
        "first": diff_days[0][0],
        "last": diff_days[-1][0],
        "top": sorted(diff_days, key=lambda item: -item[1])[:_SAMPLE_CAP],
    }


class DayCompareCache:
    """a 节跨折单日比较结果缓存（P2-7 + R2-T5A-R1-04 命名空间/条目身份加固）。

    - 缓存键 = ``(day, 消费列集指纹)``（列集指纹 = 有序列清单 sha256 前 16 位）；
    - **命名空间隔离**（R1-04①）：磁盘目录 ``<cache_dir>/<FORMAT_VERSION>/<namespace>/``，
      namespace = 两臂 ``input_snapshot_digest`` + FORMAT_VERSION + 比较口径摘要（atol
      等）的合成哈希（runner 合成注入）——另一数据态/另一根的结果不可能被命中；
    - **条目内容身份**（R1-04②）：每条登记 ``identity = {cs_train: 日文件 sha256_16,
      panel: 月分区 receipt 合成 sha256_16}``（读数时现算），命中前校验条目身份，
      不符按未命中重算并覆盖；
    - **版本递增义务**（R1-04③，写死）：**比较口径/条目结构**任何变更 ⇒
      ``FORMAT_VERSION`` 必须递增（同时入口径摘要 ⇒ 旧缓存自然失效）；条目只存
      比较结果（逐日差异计数/数值），F11 分类与闭合规则在下游重算 ⇒ 闭合规则
      变更不污染比较结果、无需递增（P3-R3-01 澄清）；
    - 纯加速——命中与否不改变结果（数据冻结另有消费快照复核门 + 实读绑定核验）。
    """

    FORMAT_VERSION = "v3"  # v2→v3：日历/读后及运行后身份硬门（R2-T5A-R2-02）

    def __init__(self, cache_dir: Path | None = None, *, namespace: str = "") -> None:
        self._dir = Path(cache_dir) if cache_dir is not None else None
        self._namespace = namespace
        self._memory: dict[tuple[str, str], dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def columns_fingerprint(columns: Sequence[str]) -> str:
        """消费列集指纹 = 有序列清单（换行拼接）sha256 前 16 位。"""
        text = "\n".join(str(c) for c in columns)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _path(self, day: str, columns_fp: str) -> Path:
        assert self._dir is not None
        return self._dir / self.FORMAT_VERSION / self._namespace / f"{day}_{columns_fp}.json"

    def get(self, day: str, columns_fp: str, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        """读取缓存（内存 → 磁盘两级；条目身份不符/损坏按未命中处理）。"""
        key = (day, columns_fp)
        if key in self._memory:
            entry = self._memory[key]
            if entry["identity"] == dict(identity):
                self.hits += 1
                return entry["result"]
        if self._dir is not None:
            entry = self._read_disk(day, columns_fp)
            if entry is not None and entry["identity"] == dict(identity):
                self._memory[key] = entry
                self.hits += 1
                return entry["result"]
        self.misses += 1
        return None

    def _read_disk(self, day: str, columns_fp: str) -> dict[str, Any] | None:
        path = self._path(day, columns_fp)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # ValueError 含 JSONDecodeError
            logger.warning(f"a 节缓存条目损坏，按未命中重算: {path}（{exc}）")
            return None
        if not isinstance(entry, dict) or entry.get("date") != day:
            return None
        if entry.get("columns_fp") != columns_fp or not isinstance(entry.get("identity"), dict):
            return None
        return entry if isinstance(entry.get("result"), dict) else None

    def put(
        self,
        day: str,
        columns_fp: str,
        identity: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        """写入缓存（内存 + 磁盘写-through；JSON 序列化，NaN 沿 Python json 口径往返）。"""
        entry = {
            "date": day,
            "columns_fp": columns_fp,
            "identity": dict(identity),
            "result": dict(result),
        }
        self._memory[(day, columns_fp)] = entry
        if self._dir is not None:
            path = self._path(day, columns_fp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict[str, Any]:
        """缓存命中率登记（CLI/报告口径：hits / (hits + misses)）。"""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "cache_dir": str(self._dir) if self._dir is not None else None,
            "namespace": self._namespace,
            "format_version": self.FORMAT_VERSION,
        }


class PanelMonthCache:
    """panel 月帧跨折复用（P3-N4）：``(month, 消费列集指纹)`` 键的内存 LRU（容量 3）。

    条目 = 月帧 + 当月实读文件身份（store receipt 合成 sha256_16，读取时点采集）。
    纯加速——命中与否不改变判据结果；命中率经 ``stats()`` 登记。

    P3-R3-02 观测：键含折级列集指纹，逐折列集互异场景命中率实测 0%
    （1,108 misses / 0 hits）；day 级缓存承担主加速，本层保留作同列集复用。
    """

    def __init__(self, capacity: int = 3) -> None:
        self._capacity = max(1, int(capacity))
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, month: str, columns_fp: str) -> dict[str, Any] | None:
        """LRU 读取（命中即刷新为最近使用）。"""
        key = (month, columns_fp)
        entry = self._entries.pop(key, None)
        if entry is None:
            self.misses += 1
            return None
        self._entries[key] = entry  # 重插到末尾 = 最近使用
        self.hits += 1
        return entry

    def put(self, month: str, columns_fp: str, frame: pd.DataFrame, identity: str) -> None:
        """LRU 写入（超出容量逐出最久未用条目）。"""
        key = (month, columns_fp)
        self._entries.pop(key, None)
        self._entries[key] = {"frame": frame, "identity": identity}
        while len(self._entries) > self._capacity:
            self._entries.pop(next(iter(self._entries)))

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "capacity": self._capacity,
        }


def _panel_records_identity16(records: Sequence[Mapping[str, Any]]) -> str:
    """panel 月分区实读文件身份 = receipt 记录（路径+sha256 排序拼接）sha256 前 16 位。"""
    if not records:
        return "<empty>"
    text = "\n".join(
        f"{record['path']}:{record['sha256']}"
        for record in sorted(records, key=lambda item: str(item["path"]))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def attribute_fold_inputs(
    store: Any,
    storage: Any,
    fold: io.FoldInputs,
    *,
    expected_days: Sequence[str] | None = None,
    excluded_non_open_count: int = 0,
    cache: DayCompareCache | None = None,
    month_cache: PanelMonthCache | None = None,
    ledger: io.ReadLedger | None = None,
) -> dict[str, Any]:
    """a 节单折：窗口 ∪ 段内逐日 panel vs cs_train 消费列比较 + 词表分类。

    消费列 = 两臂一致的模型登记列（列集一致性由调用方先行校验，不一致不得进入
    本函数）；窗口 = boundary.json 外层训练窗 ∪ test 窗。逐日粒度：panel 按月读
    消费列切片到日，cs_train 逐日读分区并投影到消费列；任一源缺日登记为
    missing_*_days（行集级差异，破坏闭合）；任一侧同日重复 ts_code 键登记为
    duplicate_key_days（``_compare_day`` 的 set 键域对重键会错位，沿 t1 先例跳过
    比较并破坏闭合）。

    完整覆盖前置（R2-T5A-04）：``expected_days`` = trade_cal ``is_open=1`` ∩ 窗口的
    独立预期日域；传入时按预期日域驱动逐日迭代（双源同缺也显式记 missing），收尾
    校验 ``days_compared == 预期日数``，不等 ⇒ 不闭合；``excluded_non_open_count``
    登记窗口内合法剔除的非开市日计数；两源存在但不在预期日域的分区日记入
    ``unexpected_partition_days``（信息性登记）。
    ``cache`` = 跨折单日结果缓存（P2-7 + R1-04 命名空间/条目身份）；``month_cache``
    = panel 月帧 LRU（P3-N4）；``ledger`` = 实读文件台账（R1-03 绑定核验输入）。
    三者均为纯加速/证据设施，不改变判据结果。
    """
    columns = list(fold.columns_arm3)
    columns_fp = DayCompareCache.columns_fingerprint(columns)
    expected = sorted(set(expected_days)) if expected_days is not None else None
    cs_days = set(io.list_cs_train_days(_storage_root(storage), fold.window))
    panel_days_all: set[str] = set()
    aggregator = ColumnImpactAggregator()
    duplicate_key_days: list[dict[str, str]] = []
    for month in io.window_months(fold.window):
        panel_frame, panel_identity = _load_month_cached(
            store, columns, month, columns_fp, month_cache
        )
        panel_days = _panel_days_in_window(panel_frame, fold.window)
        panel_days_all |= panel_days
        for day in _days_to_compare(cs_days, panel_days, expected, month):
            _compare_one_day(
                storage,
                fold,
                day,
                panel_frame,
                panel_identity,
                aggregator,
                duplicate_key_days,
                cache=cache,
                columns_fp=columns_fp,
                ledger=ledger,
            )
    result = aggregator.finalize()
    if duplicate_key_days:
        result["closed"] = False
    if expected is not None:
        _apply_coverage_gate(
            result, expected, excluded_non_open_count, cs_days, panel_days_all, duplicate_key_days
        )
    return {
        "fold": fold.fold,
        "model_version_arm2": fold.model_version_arm2,
        "model_version_arm3": fold.model_version_arm3,
        "consumed_columns": columns,
        "consumed_column_count": len(columns),
        "window": {
            "outer_train": [fold.window.outer_train_start, fold.window.outer_train_end],
            "test": [fold.window.test_start, fold.window.test_end],
        },
        "identical_columns": [c for c in columns if c not in result["columns"]],
        "duplicate_key_days": duplicate_key_days,
        **result,
    }


def _load_month_cached(
    store: Any,
    columns: Sequence[str],
    month: str,
    columns_fp: str,
    month_cache: PanelMonthCache | None,
) -> tuple[pd.DataFrame, str]:
    """panel 月帧 + 当月实读文件身份（receipt 合成 sha256_16）；LRU 命中即复用。"""
    if month_cache is not None:
        hit = month_cache.get(month, columns_fp)
        if hit is not None:
            return hit["frame"], hit["identity"]
    before = len(store.consumed_files)
    frame = io.load_panel_month(store, columns, month)
    identity = _panel_records_identity16(list(store.consumed_files[before:]))
    if month_cache is not None:
        month_cache.put(month, columns_fp, frame, identity)
    return frame, identity


def _days_to_compare(
    cs_days: set[str],
    panel_days: set[str],
    expected: list[str] | None,
    month: str,
) -> list[str]:
    """当月待比较日清单：预期日域驱动（R2-T5A-04）；无预期域时退回两源并集（旧口径）。"""
    prefix = month.replace("-", "")
    if expected is not None:
        return [day for day in expected if day.startswith(prefix)]
    return sorted(_cs_days_in_month(cs_days, month) | panel_days)


def _apply_coverage_gate(
    result: dict[str, Any],
    expected: list[str],
    excluded_non_open_count: int,
    cs_days: set[str],
    panel_days: set[str],
    duplicate_key_days: Sequence[Mapping[str, str]],
) -> None:
    """完整覆盖门收尾（R2-T5A-04）：days_compared == 预期日数，否则不闭合。

    覆盖缺口 = 未比较的预期日 = missing 日 ∪ 重键跳过日；窗口内非开市日显式登记
    excluded 计数（合法剔除）；两源存在但不在预期日域的分区日记 unexpected（信息性）。
    """
    gap = sorted(
        set(result["missing_cs_train_days"])
        | set(result["missing_panel_days"])
        | {entry["date"] for entry in duplicate_key_days}
    )
    unexpected = sorted((cs_days | panel_days) - set(expected))
    result["expected_days_count"] = len(expected)
    result["excluded_non_open_days"] = int(excluded_non_open_count)
    result["coverage_gap_count"] = len(gap)
    result["coverage_gap_days"] = gap[:_SAMPLE_CAP]
    result["unexpected_partition_days_count"] = len(unexpected)
    result["unexpected_partition_days"] = unexpected[:_SAMPLE_CAP]
    if result["days_compared"] != len(expected):
        result["closed"] = False


def _storage_root(storage: Any) -> Path:
    return Path(storage.root_path)


def _cs_days_in_month(cs_days: set[str], month: str) -> set[str]:
    prefix = month.replace("-", "")
    return {day for day in cs_days if day.startswith(prefix)}


def _panel_days_in_window(panel_frame: pd.DataFrame, window: io.FoldWindow) -> set[str]:
    if panel_frame.empty:
        return set()
    days = {str(day) for day in panel_frame.index.get_level_values("trade_date")}
    return {day for day in days if window.contains(day)}


def _compare_one_day(
    storage: Any,
    fold: io.FoldInputs,
    day: str,
    panel_frame: pd.DataFrame,
    panel_identity: str,
    aggregator: ColumnImpactAggregator,
    duplicate_key_days: list[dict[str, str]],
    *,
    cache: DayCompareCache | None = None,
    columns_fp: str = "",
    ledger: io.ReadLedger | None = None,
) -> None:
    """单日两源比较并喂入聚合器（缺日/重键登记后跳过值比较；缓存命中直接复用）。

    双源同缺也登记（missing_cs_train_days + missing_panel_days 双侧入册）——
    预期日域驱动迭代下「双方都空」不再被静默跳过（R2-T5A-04）。
    缓存条目身份（R2-T5A-R1-04②）：cs_train 日文件内容 sha256 读取时点现算
    （命中前校验，不符按未命中重算），panel 月分区身份随月帧（receipt 合成）；
    实读文件入 ``ledger`` 台账（R2-T5A-R1-03 绑定核验输入）。
    """
    root = _storage_root(storage)
    cs_path = io.cs_train_day_path(root, day)
    cs_sha = io.file_sha256(cs_path) if cs_path.is_file() else None
    read_ledger = ledger if ledger is not None else io.ReadLedger()
    if cs_sha is not None:
        read_ledger.record(cs_path, digest=cs_sha)
    identity = {"cs_train": cs_sha[:16] if cs_sha is not None else None, "panel": panel_identity}
    if cache is not None:
        cached = cache.get(day, columns_fp, identity)
        if cached is not None:
            if cs_sha is not None:
                read_ledger.assert_unchanged(cs_path, cs_sha)
            aggregator.add_day(cached)
            return
    cs_frame = io.load_cs_train_day(storage, day) if cs_sha is not None else None
    if cs_sha is not None:
        read_ledger.assert_unchanged(cs_path, cs_sha)
    panel_day = _slice_day(panel_frame, day).reset_index()
    if cs_frame is None:
        aggregator.missing_cs_train_days.append(day)
        if panel_day.empty:
            aggregator.missing_panel_days.append(day)
        return
    if panel_day.empty:
        aggregator.missing_panel_days.append(day)
        return
    cs_projected = _project_consumed_columns(cs_frame, fold.columns_arm3)
    if cs_projected.duplicated(["ts_code"]).any() or panel_day.duplicated(["ts_code"]).any():
        duplicate_key_days.append({"date": day, "fold": fold.fold})
        return
    day_result = _compare_day(day, cs_projected, panel_day)
    if cache is not None:
        cache.put(day, columns_fp, identity, day_result)
    aggregator.add_day(day_result)


def _project_consumed_columns(cs_frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """cs_train 日帧投影到 键列 + 消费列（消费列缺失留在 cols_only_panel 结构登记）。"""
    keep = ["trade_date", "ts_code"] + [c for c in columns if c in cs_frame.columns]
    return cs_frame[keep]


# ---------------------------------------------------------------------------
# b 节：预测影响（预测行键集/逐值对比 + 指纹/n_rows/score_days 元数据）
# ---------------------------------------------------------------------------


def compare_fingerprint_docs(
    fold: str, doc_arm2: Mapping[str, Any], doc_arm3: Mapping[str, Any]
) -> dict[str, Any]:
    """b 节单折元数据：score_days 对称差 + n_rows 差 + 指纹一致标记。

    ``fingerprint_match`` 要求两侧指纹均非 None 且相等（R2-T5A-04：修掉
    None==None 误判为一致；必备字段缺失由调用方先行拒绝，此处防御性收口）。
    """
    days2 = [str(d) for d in doc_arm2.get("score_days") or []]
    days3 = [str(d) for d in doc_arm3.get("score_days") or []]
    sym_diff = sorted(set(days2) ^ set(days3))
    n_rows2 = doc_arm2.get("n_rows")
    n_rows3 = doc_arm3.get("n_rows")
    fp2 = doc_arm2.get("fingerprint")
    fp3 = doc_arm3.get("fingerprint")
    return {
        "fold": fold,
        "fingerprint_match": fp2 is not None and fp3 is not None and fp2 == fp3,
        "fingerprint_arm2": fp2,
        "fingerprint_arm3": fp3,
        "n_rows_arm2": n_rows2,
        "n_rows_arm3": n_rows3,
        "n_rows_delta": (int(n_rows3) - int(n_rows2)) if _is_int_like(n_rows2, n_rows3) else None,
        "score_days_arm2": len(days2),
        "score_days_arm3": len(days3),
        "score_days_sym_diff_count": len(sym_diff),
        "score_days_sym_diff": sym_diff,
        "source_mode_arm2": doc_arm2.get("source_mode"),
        "source_mode_arm3": doc_arm3.get("source_mode"),
    }


def _is_int_like(*values: Any) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def compare_prediction_frames(frame2: pd.DataFrame, frame3: pd.DataFrame) -> dict[str, Any]:
    """b 节单折预测行逐值归因（R2-T5A-06 / §19.3 冻结口径）。

    键集 (trade_date, ts_code) 对称差计数 + 样例；**同日换股**（同一 trade_date 上
    两臂各出现对方没有的键 ⇒ 换股候选日）与**同键改分**（交集键 ``%.17g`` 文本化
    逐值 diff）分列，可区分；score 读取时 -0.0 归一 0.0（契约口径），NaN/Inf
    防御性拒绝（生产侧已拒，此处复核证据完整性）。
    """
    left = _normalize_prediction_frame(frame2, "臂②")
    right = _normalize_prediction_frame(frame3, "臂③")
    keys2 = _key_set(left, _PREDICTION_KEY_COLUMNS)
    keys3 = _key_set(right, _PREDICTION_KEY_COLUMNS)
    only2 = sorted(keys2 - keys3)
    only3 = sorted(keys3 - keys2)
    swap_dates = sorted({k[0] for k in only2} & {k[0] for k in only3})
    merged = left.merge(
        right, on=list(_PREDICTION_KEY_COLUMNS), how="inner", suffixes=("_arm2", "_arm3")
    )
    return {
        "rows_arm2": int(len(frame2)),
        "rows_arm3": int(len(frame3)),
        "keys_arm2": len(keys2),
        "keys_arm3": len(keys3),
        "key_sym_diff_count": len(only2) + len(only3),
        "keys_only_arm2": len(only2),
        "keys_only_arm3": len(only3),
        "sample_only_arm2": [list(k) for k in only2[:_SAMPLE_CAP]],
        "sample_only_arm3": [list(k) for k in only3[:_SAMPLE_CAP]],
        "same_day_swap_dates_count": len(swap_dates),
        "same_day_swap_dates": swap_dates[:_SAMPLE_CAP],
        "intersection_keys": int(len(merged)),
        **_prediction_value_diffs(merged),
        "value_caliber": "交集键 score 按 %.17g 文本化逐值比较；-0.0 归一 0.0（跨路契约口径）",
    }


def _normalize_prediction_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """预测行归一化：键列 str、score 数值化 + -0.0 归一 + 有限性复核。"""
    missing = [c for c in (*_PREDICTION_KEY_COLUMNS, "score") if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} 预测行缺必需列 {missing}")
    work = frame[list(_PREDICTION_KEY_COLUMNS) + ["score"]].copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["ts_code"] = work["ts_code"].astype(str)
    work["score"] = pd.to_numeric(work["score"], errors="coerce") + 0.0  # -0.0 归一 0.0
    if not bool(np.isfinite(work["score"].to_numpy(dtype=float)).all()):
        raise ValueError(f"{label} 预测行 score 含 NaN/Inf（跨路契约已于生产侧拒绝）")
    return work


def _prediction_value_diffs(merged: pd.DataFrame) -> dict[str, Any]:
    """交集键 ``%.17g`` 文本化逐值 diff：差异行数 / max|Δ| / 样例（同键改分证据）。"""
    if merged.empty:
        return {"value_diff_rows": 0, "value_max_abs_diff": None, "value_diff_samples": []}
    fmt2 = merged["score_arm2"].map(lambda v: format(float(v), ".17g"))
    fmt3 = merged["score_arm3"].map(lambda v: format(float(v), ".17g"))
    mask = (fmt2 != fmt3).to_numpy()
    diff_rows = merged.loc[mask]
    abs_delta = (merged["score_arm3"] - merged["score_arm2"]).abs()
    samples = [
        {
            "trade_date": str(row["trade_date"]),
            "ts_code": str(row["ts_code"]),
            "score_arm2": format(float(row["score_arm2"]), ".17g"),
            "score_arm3": format(float(row["score_arm3"]), ".17g"),
            "abs_diff": float(abs(row["score_arm3"] - row["score_arm2"])),
        }
        for _, row in diff_rows.head(_SAMPLE_CAP).iterrows()
    ]
    return {
        "value_diff_rows": int(mask.sum()),
        "value_max_abs_diff": float(abs_delta.max()),
        "value_diff_samples": samples,
    }


# ---------------------------------------------------------------------------
# c/d 节：成交 / 持仓影响（键集对称差 + 交集数值最大绝对差；P3-3 键聚合口径）
# ---------------------------------------------------------------------------


def compare_trade_frames(frame2: pd.DataFrame, frame3: pd.DataFrame) -> dict[str, Any]:
    """c 节单折：trades 键集 (trade_date, ts_code, action) 对称差 + 交集价格/数量差。

    P3-3 口径：同日同股同方向多笔成交（分批/补仓）先按键聚合——shares/amount
    求和、price = Σamount/Σshares 推导（Σshares=0 ⇒ NaN），重复键分组计数随
    结果登记；禁止 merge 多×多笛卡尔展开失真。
    """
    return _compare_keyed_frames(
        frame2, frame3, _TRADE_KEY_COLUMNS, _TRADE_VALUE_COLUMNS, derive_price=True
    )


def compare_holdings_frames(frame2: pd.DataFrame, frame3: pd.DataFrame) -> dict[str, Any]:
    """d 节单折：holdings_snapshot 键集 (trade_date, ts_code) 对称差 + 交集数量/市值差。

    P3-3 口径：同键多行先按键聚合（shares/market_value/weight 求和），重复键
    分组计数随结果登记。
    """
    return _compare_keyed_frames(
        frame2, frame3, _HOLDINGS_KEY_COLUMNS, _HOLDINGS_VALUE_COLUMNS, derive_price=False
    )


def _compare_keyed_frames(
    frame2: pd.DataFrame,
    frame3: pd.DataFrame,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    derive_price: bool,
) -> dict[str, Any]:
    """键集对称差计数 + 交集行数值列最大绝对差（NaN 不一致单列计数）。

    比较前两臂各自按键聚合（P3-3）：数值列求和；``derive_price=True`` 时 price 由
    Σamount/Σshares 重新推导。重复键分组数与样例随结果登记。
    """
    missing2 = [c for c in key_columns if c not in frame2.columns]
    missing3 = [c for c in key_columns if c not in frame3.columns]
    if missing2 or missing3:
        raise ValueError(f"键列缺失（臂②缺 {missing2} / 臂③缺 {missing3}），无法按键对比")
    agg2, dup2 = _aggregate_by_keys(frame2, key_columns, value_columns, derive_price=derive_price)
    agg3, dup3 = _aggregate_by_keys(frame3, key_columns, value_columns, derive_price=derive_price)
    keys2 = _key_set(agg2, key_columns)
    keys3 = _key_set(agg3, key_columns)
    only2 = sorted(keys2 - keys3)
    only3 = sorted(keys3 - keys2)
    merged = agg2.merge(agg3, on=list(key_columns), how="inner", suffixes=("_arm2", "_arm3"))
    value_diffs = {
        col: _max_abs_value_diff(merged, col)
        for col in value_columns
        if f"{col}_arm2" in merged.columns and f"{col}_arm3" in merged.columns
    }
    return {
        "rows_arm2": int(len(frame2)),
        "rows_arm3": int(len(frame3)),
        "duplicate_key_groups_arm2": dup2["groups"],
        "duplicate_key_groups_arm3": dup3["groups"],
        "sample_duplicate_keys_arm2": dup2["samples"],
        "sample_duplicate_keys_arm3": dup3["samples"],
        "keys_arm2": len(keys2),
        "keys_arm3": len(keys3),
        "key_sym_diff_count": len(only2) + len(only3),
        "keys_only_arm2": len(only2),
        "keys_only_arm3": len(only3),
        "sample_only_arm2": [list(k) for k in only2[:_SAMPLE_CAP]],
        "sample_only_arm3": [list(k) for k in only3[:_SAMPLE_CAP]],
        "intersection_rows": int(len(merged)),
        "value_max_abs_diff": value_diffs,
    }


def _aggregate_by_keys(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    derive_price: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按键聚合数值列并统计重复键（P3-3）；返回 (聚合帧, 重复键统计)。"""
    numeric = [c for c in value_columns if c in frame.columns]
    work = frame[list(key_columns) + numeric].copy()
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    sizes = work.groupby(list(key_columns), sort=True).size()
    duplicated = sizes[sizes > 1]
    aggregated = work.groupby(list(key_columns), as_index=False, sort=True)[numeric].sum(
        min_count=1
    )
    if derive_price and {"price", "amount", "shares"} <= set(aggregated.columns):
        shares = aggregated["shares"]
        aggregated["price"] = aggregated["amount"] / shares.where(shares != 0)
    stats = {
        "groups": int(len(duplicated)),
        "samples": [[str(part) for part in key] for key in duplicated.index[:_SAMPLE_CAP]],
    }
    return aggregated, stats


def _key_set(frame: pd.DataFrame, key_columns: Sequence[str]) -> set[tuple[str, ...]]:
    normalized = frame[list(key_columns)].astype(str)
    return set(map(tuple, normalized.itertuples(index=False, name=None)))


def _max_abs_value_diff(merged: pd.DataFrame, col: str) -> dict[str, Any]:
    left = pd.to_numeric(merged[f"{col}_arm2"], errors="coerce")
    right = pd.to_numeric(merged[f"{col}_arm3"], errors="coerce")
    diff = (left - right).abs()
    peak = diff.max()
    return {
        "max_abs_diff": 0.0 if pd.isna(peak) else float(peak),
        "nan_mismatch_rows": int((left.isna() != right.isna()).sum()),
    }


# ---------------------------------------------------------------------------
# e 节：净值影响（chain_nav 逐日差 + 逐折 daily 汇总差）
# ---------------------------------------------------------------------------


def _require_finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    """净值类列有限性前置（R2-T5A-R1-07；口径同 t5_compare_gates R2-T5A-07：
    ``np.isfinite`` 全值，非有限即拒绝——不得登记为零差异）。"""
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        bad = int((~np.isfinite(values)).sum())
        if bad:
            raise ValueError(f"{label} 列 {col} 含 NaN/Inf 共 {bad} 行（有限性前置拒绝）")


def compare_chain_navs(nav2: pd.DataFrame, nav3: pd.DataFrame) -> dict[str, Any]:
    """e 节链式净值：逐折按 date 对齐的 max|Δnav| 与末日 Δ，及全局汇总。

    P3-8：``split_index`` 先显式 ``astype(int)`` 统一 dtype（副本，不改调用方帧），
    再按折过滤——runs 契约虽定为 int，落盘往返（parquet/CSV）可能漂移为字符串。
    R2-T5A-R1-07：比较前对 ``nav`` 做有限性前置（NaN/Inf ⇒ 抛错，由调用方记 FAIL）。
    """
    for name, frame in (("臂②", nav2), ("臂③", nav3)):
        missing = [c for c in ("date", "nav", "split_index") if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} chain_nav 缺必需列 {missing}")
        _require_finite(frame, ("nav",), f"{name} chain_nav")
    nav2 = nav2.assign(split_index=nav2["split_index"].astype(int))
    nav3 = nav3.assign(split_index=nav3["split_index"].astype(int))
    splits = sorted(set(nav2["split_index"]) | set(nav3["split_index"]))
    fold_entries: list[dict[str, Any]] = []
    global_max = 0.0
    for split in splits:
        seg2 = nav2[nav2["split_index"] == split]
        seg3 = nav3[nav3["split_index"] == split]
        entry = _chain_nav_split_diff(split, seg2, seg3)
        fold_entries.append(entry)
        if entry["max_abs_diff"] is not None:
            global_max = max(global_max, entry["max_abs_diff"])
    last = fold_entries[-1] if fold_entries else {}
    return {
        "folds": fold_entries,
        "max_abs_diff": global_max,
        "final_fold_last_date_diff": last.get("last_date_diff"),
    }


def _chain_nav_split_diff(split: int, seg2: pd.DataFrame, seg3: pd.DataFrame) -> dict[str, Any]:
    dates2 = {str(d) for d in seg2["date"]}
    dates3 = {str(d) for d in seg3["date"]}
    aligned = seg2.merge(seg3, on="date", how="inner", suffixes=("_arm2", "_arm3"))
    entry: dict[str, Any] = {
        "split_index": split,
        "days_arm2": int(len(seg2)),
        "days_arm3": int(len(seg3)),
        "date_sym_diff_count": len(dates2 ^ dates3),
        "common_days": int(len(aligned)),
        "max_abs_diff": None,
        "last_date_diff": None,
    }
    if aligned.empty:
        return entry
    diff = (pd.to_numeric(aligned["nav_arm3"]) - pd.to_numeric(aligned["nav_arm2"])).abs()
    entry["max_abs_diff"] = float(diff.max())
    last_row = aligned.sort_values("date").iloc[-1]
    entry["last_date"] = str(last_row["date"])
    entry["last_date_diff"] = float(last_row["nav_arm3"] - last_row["nav_arm2"])
    return entry


def compare_daily_frames(frame2: pd.DataFrame, frame3: pd.DataFrame) -> dict[str, Any]:
    """e 节逐折 daily：按 trade_date 对齐的 max|Δnav| / max|Δtotal_value| + 日域对称差。

    R2-T5A-R1-07：比较前对 ``nav``/``total_value`` 做有限性前置（NaN/Inf ⇒ 抛错）。
    """
    for name, frame in (("臂②", frame2), ("臂③", frame3)):
        if "trade_date" not in frame.columns:
            raise ValueError(f"{name} daily 缺 trade_date 列")
        _require_finite(frame, ("nav", "total_value"), f"{name} daily")
    dates2 = {str(d) for d in frame2["trade_date"]}
    dates3 = {str(d) for d in frame3["trade_date"]}
    merged = frame2.merge(frame3, on="trade_date", how="inner", suffixes=("_arm2", "_arm3"))
    value_diffs = {
        col: _max_abs_value_diff(merged, col)
        for col in ("nav", "total_value")
        if f"{col}_arm2" in merged.columns and f"{col}_arm3" in merged.columns
    }
    return {
        "days_arm2": int(len(frame2)),
        "days_arm3": int(len(frame3)),
        "date_sym_diff_count": len(dates2 ^ dates3),
        "common_days": int(len(merged)),
        "value_max_abs_diff": value_diffs,
    }


# ---------------------------------------------------------------------------
# 输出级一致性（R2-T5A-05）：归因闭合 = a 节闭合 ∧ 输出级一致性
# ---------------------------------------------------------------------------

#: 输出级一致性规则文本（随报告 rule 字段落盘；§13.3.6 ΔMaxDD 逐折归因闭合按规则②落地）
OUTPUT_CONSISTENCY_RULE = (
    "归因闭合 = a 节闭合 ∧ 输出级一致性：① 零输入差异 ⇒ 零输出差异（a 节全折全列零差异时，"
    "b~e 节必须全部无差异，否则 ⇒ 不闭合 FAIL）；② 逐折对应（任一折存在输出劣化——e 节逐折 "
    "nav/daily 差异超容差 1e-6，或 b/c/d 节有差异——该折 a 节必须有已登记的消费列差异"
    "（词表分类在册），否则 ⇒ 不闭合 FAIL）；ΔMaxDD 劣化的逐折归因闭合（§13.3.6）按 ② 落地"
)


def evaluate_output_consistency(
    a_folds: Sequence[Mapping[str, Any]],
    b_folds: Sequence[Mapping[str, Any]],
    c_folds: Sequence[Mapping[str, Any]],
    d_folds: Sequence[Mapping[str, Any]],
    e_chain: Mapping[str, Any] | None,
    e_daily_folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """输出级一致性核验（R2-T5A-05 新口径，规则文本见 ``OUTPUT_CONSISTENCY_RULE``）。

    「输入闭合不得直接当输出归因闭合」：a 节闭合只证明输入面差异 100% 词表在册；
    本核验把 b~e 节输出面证据接入闭合——规则① 拒绝「没有输入差异证据却有未解释
    输出变化」；规则② 要求每一处输出劣化都能对应到该折已登记的输入差异。
    净值类容差 = runs 契约 §9.1 的 1e-6（``t5_compare_gates.NAV_TOLERANCE``，import
    复用不复制）；成交/持仓数值列差按精确零口径（同代码态 + 同输入 ⇒ 逐位一致）。
    """
    from scripts.v2_p3.t5_compare_gates import NAV_TOLERANCE

    degraded = _degraded_output_folds(
        b_folds, c_folds, d_folds, e_chain, e_daily_folds, NAV_TOLERANCE
    )
    zero_input = bool(a_folds) and all(_a_fold_zero_diff(fold) for fold in a_folds)
    violations_zero: list[str] = []
    if zero_input and degraded:
        violations_zero = [
            f"折 {fold} {section} 节存在输出差异，但 a 节全折全列零差异"
            "（规则① 零输入差异 ⇒ 零输出差异被破坏）"
            for fold, section in degraded
        ]
    a_registered = {
        str(fold["fold"])
        for fold in a_folds
        if fold.get("columns") or fold.get("registered_cols_only_panel")
    }
    violations_unmatched = [
        f"折 {fold} {section} 节输出劣化，但该折 a 节无已登记消费列差异（规则② 逐折对应失败）"
        for fold, section in degraded
        if fold not in a_registered
    ]
    return {
        "consistent": not violations_zero and not violations_unmatched,
        "rule": OUTPUT_CONSISTENCY_RULE,
        "zero_input": zero_input,
        "degraded_output_folds": sorted({fold for fold, _ in degraded}),
        "zero_input_zero_output_violations": violations_zero,
        "unattributed_output_violations": violations_unmatched,
    }


def _degraded_output_folds(
    b_folds: Sequence[Mapping[str, Any]],
    c_folds: Sequence[Mapping[str, Any]],
    d_folds: Sequence[Mapping[str, Any]],
    e_chain: Mapping[str, Any] | None,
    e_daily_folds: Sequence[Mapping[str, Any]],
    nav_tol: float,
) -> list[tuple[str, str]]:
    """检出输出差异的 (折, 节) 清单（b/c/d/e 逐折谓词）。"""
    degraded: list[tuple[str, str]] = []
    for fold in b_folds:
        if _b_fold_has_diff(fold):
            degraded.append((str(fold["fold"]), "b"))
    for fold in c_folds:
        if _keyed_fold_has_diff(fold):
            degraded.append((str(fold["fold"]), "c"))
    for fold in d_folds:
        if _keyed_fold_has_diff(fold):
            degraded.append((str(fold["fold"]), "d"))
    for entry in (e_chain or {}).get("folds") or []:
        if _chain_fold_has_diff(entry, nav_tol):
            degraded.append((f"split{int(entry['split_index']):02d}", "e.chain_nav"))
    for fold in e_daily_folds:
        if _daily_fold_has_diff(fold, nav_tol):
            degraded.append((str(fold["fold"]), "e.daily"))
    return degraded


def _a_fold_zero_diff(fold: Mapping[str, Any]) -> bool:
    """a 节单折零差异 = 无差异列且无行集/结构/缺日/重键破坏（覆盖缺口单列另由闭合门拦截）。"""
    return not (
        fold.get("columns")
        or fold.get("row_diff_days")
        or fold.get("cols_only_cs_train")
        or fold.get("cols_only_panel")
        or fold.get("registered_cols_only_panel")
        or fold.get("missing_cs_train_days")
        or fold.get("missing_panel_days")
        or fold.get("duplicate_key_days")
    )


def _b_fold_has_diff(fold: Mapping[str, Any]) -> bool:
    """b 节单折输出差异 = 预测行键集/逐值差异或指纹/n_rows/score_days 元数据差异。"""
    pred = fold.get("predictions") or {}
    return bool(
        not fold.get("fingerprint_match", True)
        or fold.get("n_rows_delta") not in (0, None)
        or (fold.get("score_days_sym_diff_count") or 0) > 0
        or (pred.get("key_sym_diff_count") or 0) > 0
        or (pred.get("value_diff_rows") or 0) > 0
    )


def _keyed_fold_has_diff(fold: Mapping[str, Any]) -> bool:
    """c/d 节单折输出差异 = 键集对称差非零或交集数值列精确非零（含 NaN 不一致）。"""
    if (fold.get("key_sym_diff_count") or 0) > 0:
        return True
    return any(
        (entry.get("max_abs_diff") or 0.0) > 0.0 or (entry.get("nan_mismatch_rows") or 0) > 0
        for entry in (fold.get("value_max_abs_diff") or {}).values()
    )


def _chain_fold_has_diff(entry: Mapping[str, Any], tol: float) -> bool:
    """e 节 chain_nav 单折输出劣化 = 日域对称差非零或 max|Δnav| 超容差。"""
    if (entry.get("date_sym_diff_count") or 0) > 0:
        return True
    peak = entry.get("max_abs_diff")
    return peak is not None and float(peak) > tol


def _daily_fold_has_diff(fold: Mapping[str, Any], tol: float) -> bool:
    """e 节逐折 daily 输出劣化 = 日域对称差非零或 max|Δnav|/max|Δtotal_value| 超容差。"""
    if (fold.get("date_sym_diff_count") or 0) > 0:
        return True
    return any(
        (entry.get("max_abs_diff") or 0.0) > tol or (entry.get("nan_mismatch_rows") or 0) > 0
        for entry in (fold.get("value_max_abs_diff") or {}).values()
    )
