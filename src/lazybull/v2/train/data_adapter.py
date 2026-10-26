# -*- coding: utf-8 -*-
"""v2 训练数据只读适配层（P3 T1 唯一适配点；P3 前规划 §6.2 全 9 条 + P3 规划 §4/§4.1）。

两种数据源模式：

- ``panel``（默认）：经 ``PanelDataStore`` 公共读 API 消费 panel 特征 + 三表标签——
  panel core 为左表，labels 按 ``(trade_date, ts_code)`` 一对一 left join（重复键 /
  缺表 / 意外增删键 fail-fast）；恢复六标签原名 ``y_ret_5/10/20``、
  ``neu_y_ret_5/10/20``（``label_column`` 语义保留，供 prepare 的 N+1 推导继续成立）；
  训练前逐日共享 readiness 校验（``label_refresh.validate_label_day_readiness``，
  ``E_20 ≤ W_in`` 仍 forming / 缺分区 / 三表不齐 / 行域未知 / 日级半成品 ⇒ 硬失败）；
  ``E_20 > W_in`` 的合法 forming 尾部不清空不拒绝，交下游按实际标签列 dropna；
  ``maturity_status`` 只用于成熟度校验，不进特征矩阵。
  **P3-T2 加载成本门（§16.5-②）**：readiness 改经收集变体
  （``validate_label_day_readiness_and_collect``），其校验已全量读取的三表标签帧
  直接复用于六标签装配（``_assemble_label_columns``），消除 T1 版 readiness 与
  ``store.load_labels`` 双通道重复全量读；输出语义与 T1 版逐位一致。
- ``cs_train_verify``（验收专用过渡例外，P3 规划 §3.1/§3.3，去除节点 = P4 评审）：
  显式开关默认关；须显式声明验收用途（G-code 配对参照 / 固定旧模型回归边）；
  纸面 / 实盘 / 纸面宿主代码路径不可达（调用栈校验）；输出必须与现役
  ``ml/train_core/split.py::load_features_data`` 逐值一致（含六标签与键集，
  §4.2 G-code 数据前提）；运行时派生四族默认关，启用则必须传与旧路径同源的
  lookup（启用未传 ⇒ 报错），派生动作仍由下游训练侧执行，本层不重复派生。

水位三对象分列（§4.1，禁止 min 合并掩盖含义差异）：实际输入行情水位 ``W_in``
（clean daily 最大分区日）、三表封存水位 ``sealed_through``、各表最新分区日。
首次迁移回放限制在 ≤20260702 冻结段：窗口上界越界拒绝（§6.2-6 / §4 表第 6 行）。
"""

from __future__ import annotations

import calendar as calendar_module
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.types import FeatureQuery, TradeDate, TSCode
from src.lazybull.v2.store.column_groups import LABEL_TABLES, LABEL_VALUE_COLUMNS
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.label_refresh import validate_label_day_readiness_and_collect
from src.lazybull.v2.store.row_domain import LABEL_FILTERED_THROUGH

__all__ = [
    "ACCEPTANCE_PURPOSES",
    "RUNTIME_DERIVED_FAMILIES",
    "SOURCE_CS_TRAIN_VERIFY",
    "SOURCE_PANEL",
    "AdapterConfig",
    "AdapterWatermarks",
    "TrainingFrame",
    "load_training_frame",
    "resolve_input_watermark",
    "resolve_trading_dates",
    "validate_frame_request",
]

#: 数据源模式：panel = v2 store 消费（默认）；cs_train_verify = cs_train 验证源（验收专用）
SOURCE_PANEL = "panel"
SOURCE_CS_TRAIN_VERIFY = "cs_train_verify"

#: cs_train 验证源模式允许范围（§3.1-③）：G-code 配对参照 + 固定旧模型回归边
ACCEPTANCE_PURPOSES = frozenset({"g_code_parity", "regression_edge"})

#: cs_train 验证源模式允许的验收入口文件前缀（仓库相对路径；验收工具与测试专用）
_ALLOWED_CALLER_PREFIXES = ("scripts/v2_p3/", "tests/")
#: 引擎内部帧前缀（v2/train 包内）：G-code 臂②引擎经适配层加载属合法内部调用
# （reviewer1-R1），但调用栈必须在引擎帧之后出现验收入口帧，生产入口依旧拒绝
_ENGINE_PACKAGE_PREFIX = "src/lazybull/v2/train/"

#: 运行时派生四族（§13.3.1；panel 已物化为 announcement 族直读列，禁止 derive 双写）
#: P3-T3 评审处置（reviewer1-P2-2）口径注记：本集合 = **物化进 panel 的四族**;
#: top_inst(已终结族)不物化、不在本集合——其派生兼容面(cs_train_verify 同源
#: lookup)由训练/评估侧的 RuntimeDerivers.top_inst 与 _EVAL_DERIVE_FAMILIES 承载,
#: 两侧集合差异属有意口径(物化族 ≠ 派生兼容族),不是一个 bug。
RUNTIME_DERIVED_FAMILIES = frozenset(
    {"holdertrade", "repurchase", "top10fh", "availability_markers"}
)

#: 族 → cs_train 验证源模式下必需的同源 lookup 键（availability_markers 自身派生无外部表）
_FAMILY_LOOKUP_KEYS = {
    "holdertrade": "holdertrade_lookup",
    "repurchase": "repurchase_lookup",
    "top10fh": "top10fh_panel",
}

_KEY_COLUMNS = ("trade_date", "ts_code")
_DATE_RE = re.compile(r"\d{8}")


@dataclass(frozen=True)
class AdapterConfig:
    """适配请求配置。

    - ``feature_columns``：候选超集列清单（panel 模式必填非空；cs_train 验证源模式
      忽略——该模式返回 cs_train 分区原始全 schema 以保证逐值一致）；
    - ``frozen_through``：回放窗口上界（默认 20260702）。只允许收紧不允许放大——
      超过权威边界 ``LABEL_FILTERED_THROUGH`` 直接拒绝（R2-P3-T1-02），跨缝须以
      已批准的新契约接入，普通日期参数不构成授权；
    - ``runtime_families`` / ``runtime_lookups``：运行时派生四族开关与同源 lookup，
      仅 ``cs_train_verify`` 模式可启用（默认空 = 关）；panel 模式两字段必须为空
      （物化列直读，防 derive 双写/覆盖，§3.3-③）。
    """

    start_date: str
    end_date: str
    feature_columns: tuple[str, ...] = ()
    label_column: str = "neu_y_ret_20"
    universe: tuple[str, ...] | None = None
    source_mode: str = SOURCE_PANEL
    frozen_through: str = LABEL_FILTERED_THROUGH
    runtime_families: frozenset[str] = frozenset()
    runtime_lookups: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AdapterWatermarks:
    """水位三对象分列记录（§4.1；各自独立口径，禁止 min 合并混用）。

    cs_train 验证源模式无 store 三表：``label_sealed_through`` / ``label_latest_partition``
    / ``panel_latest_partition`` 置 None，改填 ``cs_train_latest_partition``。
    """

    input_watermark: str  # W_in = 本次冻结输入的 clean daily 数据端点
    label_sealed_through: Mapping[str, str | None]
    label_latest_partition: Mapping[str, str | None]
    panel_latest_partition: str | None
    cs_train_latest_partition: str | None = None


@dataclass(frozen=True)
class TrainingFrame:
    """适配输出：平铺 DataFrame（trade_date/ts_code 为列，镜像 cs_train 形态）+ 证据。

    ``df`` 列序 = 键列 + 请求特征列序 + 六标签原名（LABEL_VALUE_COLUMNS 序）；
    ``maturity_status`` 不入 df（§6.2-3），逐日成熟度见 ``day_readiness``。
    ``df`` 实际日期保证 ⊆ 已校验日域（day_readiness 覆盖集）：区间读取若带入日历外
    数据（如日历修订），按日历过滤剔除并显式记入 ``excluded_days``（reviewer2-N1 余项；
    计数一致不等于日域一致）。
    """

    df: pd.DataFrame
    trade_days: int
    source_mode: str
    label_column: str
    feature_columns: tuple[str, ...]  # panel = 请求候选超集；cs_train_verify = 实际加载列
    watermarks: AdapterWatermarks
    day_readiness: tuple[dict[str, Any], ...]
    manifest_fingerprint: str | None  # panel = manifest 内容指纹；cs_train_verify 无 store 置 None
    runtime_lookups: Mapping[str, Any] | None
    excluded_days: tuple[str, ...] = ()  # 区间读取带入但被日历域剔除的日期（升序；空=无）


# ---------------------------------------------------------------------------
# 基础输入解析（clean/trade_cal 与 clean/daily 为非 store 基础输入，§5 数据态口径）
# ---------------------------------------------------------------------------


def resolve_trading_dates(data_root: Path | str) -> list[str]:
    """交易日历（clean/trade_cal 优先、raw 兜底，与现役读取同源）：is_open==1 升序。"""
    root = Path(data_root)
    frame = _read_calendar(root / "clean" / "trade_cal.parquet")
    if frame is None:
        frame = _read_calendar(root / "raw" / "trade_cal.parquet")
    if frame is None:
        raise ValueError(f"无法加载交易日历（clean/raw 均缺失）: {root}")
    open_days = frame[frame["is_open"] == 1]["cal_date"].tolist()
    return sorted({str(day).replace("-", "") for day in open_days})


def resolve_input_watermark(data_root: Path | str) -> str:
    """``W_in`` = clean/daily 最大分区日（``YYYY-MM-DD.parquet`` 归一为 YYYYMMDD）。"""
    daily_dir = Path(data_root) / "clean" / "daily"
    if not daily_dir.is_dir():
        raise ValueError(f"clean/daily 目录缺失，无法解析输入水位: {daily_dir}")
    days = [_normalize_partition_day(path.stem) for path in daily_dir.glob("*.parquet")]
    days = [day for day in days if day is not None]
    if not days:
        raise ValueError(f"clean/daily 无分区，无法解析输入水位: {daily_dir}")
    return max(days)


def _normalize_partition_day(stem: str) -> str | None:
    """分区名归一：``YYYY-MM-DD`` 或 ``YYYYMMDD`` → ``YYYYMMDD``；非日期名返回 None。"""
    digits = stem.replace("-", "")
    return digits if _DATE_RE.fullmatch(digits) else None


def _read_calendar(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    if "cal_date" not in frame.columns or "is_open" not in frame.columns:
        raise ValueError(f"交易日历缺 cal_date/is_open 列: {path}")
    work = frame.copy()
    work["cal_date"] = work["cal_date"].astype(str).str.replace("-", "", regex=False)
    return work


# ---------------------------------------------------------------------------
# 配置校验与模式守卫
# ---------------------------------------------------------------------------


def _validate_yyyymmdd(value: str, *, context: str) -> str:
    text = str(value)
    if not _DATE_RE.fullmatch(text):
        raise ValueError(f"{context} 须为 YYYYMMDD 八位数字日，实得 {value!r}")
    TradeDate.from_str(text)  # 非法日历日（如 20261340）在此抛 ValueError
    return text


def _validate_config(config: AdapterConfig) -> None:
    """配置 fail-fast：日期格式与窗口序、冻结段权威边界、标签名、列集纯净、模式合法。"""
    start = _validate_yyyymmdd(config.start_date, context="start_date")
    end = _validate_yyyymmdd(config.end_date, context="end_date")
    frozen = _validate_yyyymmdd(config.frozen_through, context="frozen_through")
    if frozen > LABEL_FILTERED_THROUGH:
        raise ValueError(
            f"frozen_through {frozen} 超出权威冻结边界 {LABEL_FILTERED_THROUGH}：本期固定限制 "
            f"≤ {LABEL_FILTERED_THROUGH}（R2-P3-T1-02），跨缝须以已批准的新契约接入，"
            "普通日期参数不构成授权"
        )
    if start > end:
        raise ValueError(f"窗口起止倒置: {start} > {end}")
    if end > frozen:
        raise ValueError(
            f"首次迁移回放限 ≤ 冻结段 {frozen}（§6.2-6）：窗口上界 {end} 越界，"
            "跨入前必须另行预登记"
        )
    if config.label_column not in LABEL_VALUE_COLUMNS:
        raise ValueError(
            f"label_column 须为六标签原名之一 {list(LABEL_VALUE_COLUMNS)}，"
            f"实得 {config.label_column!r}"
        )
    if config.source_mode == SOURCE_PANEL and not config.feature_columns:
        raise ValueError("panel 模式 feature_columns 为空（候选超集至少一列）")
    if config.source_mode == SOURCE_CS_TRAIN_VERIFY and config.feature_columns:
        raise ValueError(
            "cs_train 验证源模式返回分区原始全 schema，feature_columns 不得传入"
            "（传入会被静默忽略，禁止歧义）"
        )
    bad_keys = [c for c in config.feature_columns if c in _KEY_COLUMNS]
    bad_labels = [c for c in config.feature_columns if c in LABEL_VALUE_COLUMNS]
    if bad_keys or bad_labels:
        raise ValueError(
            f"feature_columns 不得含键列 {bad_keys} 或标签列 {bad_labels}"
            "（标签由三表恢复原名，不经 panel 特征列请求）"
        )
    if config.source_mode not in (SOURCE_PANEL, SOURCE_CS_TRAIN_VERIFY):
        raise ValueError(
            f"未知数据源模式: {config.source_mode!r}（合法值 {SOURCE_PANEL}/{SOURCE_CS_TRAIN_VERIFY}）"
        )
    if config.universe is not None:
        if isinstance(config.universe, str) or not isinstance(config.universe, tuple):
            raise ValueError(
                f"universe 须为股票代码 tuple，实得 {type(config.universe).__name__}"
                "（str 会被按字符迭代）"
            )
        if len(config.universe) == 0:
            raise ValueError(
                "universe 为空 tuple：None 才表示全市场，空池不得 fail-open 静默退化为全市场"
            )


def _assert_cs_train_verify_reachable(acceptance_purpose: str | None) -> None:
    """cs_train 验证源模式可达性守卫（§3.1-②）：显式用途声明 + 调用方文件白名单。

    白名单（reviewer1-P2-1/R1）：自上而下扫描调用栈，跳过适配层与 ``v2/train`` 引擎
    内部帧后，**首个非引擎帧必须落在 ``scripts/v2_p3/`` 或 ``tests/``**——G-code 可由
    验收入口驱动引擎内部调用适配层；以 ``__main__`` 运行的生产入口
    （scripts/train_ml_model.py 等）、纸面/实盘、REPL/``python -c`` 一律拒绝。
    """
    if acceptance_purpose not in ACCEPTANCE_PURPOSES:
        raise ValueError(
            "cs_train 验证源模式须显式声明验收用途 "
            f"{sorted(ACCEPTANCE_PURPOSES)}，实得 {acceptance_purpose!r}（默认关）"
        )
    _assert_acceptance_entry()


def _assert_acceptance_entry() -> None:
    """首个非引擎调用帧必须是验收入口（scripts/v2_p3 或 tests）；否则拒绝。"""
    repo_root = Path(__file__).resolve().parents[4]
    for frame_info in inspect.stack():
        filename = frame_info.filename
        if not filename or filename.startswith("<"):
            raise RuntimeError("cs_train 验证源模式无法判定调用方（非文件调用栈），拒绝放行")
        try:
            rel = Path(filename).resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel = ""
        if rel.startswith(_ENGINE_PACKAGE_PREFIX):
            continue  # 适配层自身与 v2/train 引擎内部帧（合法内部调用）
        if rel.startswith(_ALLOWED_CALLER_PREFIXES):
            return  # 验收入口确认；更深帧属入口基础设施（pytest 等），不再检查
        raise RuntimeError(
            "cs_train 验证源模式仅允许 scripts/v2_p3 或 tests 验收入口调用"
            f"（实得 {rel or filename}）"
        )
    raise RuntimeError("cs_train 验证源模式调用栈缺少验收入口帧，拒绝放行")


def _validate_runtime_families(config: AdapterConfig) -> None:
    """四族契约：panel 模式禁 derive（物化直读）；cs_train 模式默认关、启用必传同源 lookup。"""
    families = set(config.runtime_families)
    unknown = families - RUNTIME_DERIVED_FAMILIES
    if unknown:
        raise ValueError(
            f"未知运行时派生族: {sorted(unknown)}（合法值 {sorted(RUNTIME_DERIVED_FAMILIES)}）"
        )
    if config.source_mode == SOURCE_PANEL:
        if families or config.runtime_lookups:
            raise ValueError(
                "panel 模式下四族已物化为 announcement 族直读列，禁止再走 derive "
                "（runtime_families / runtime_lookups 必须为空，§3.3-③ 防双写/覆盖）"
            )
        return
    for family in sorted(families):
        key = _FAMILY_LOOKUP_KEYS.get(family)
        if key is None:
            continue  # availability_markers 自身派生，无外部 lookup
        lookups = config.runtime_lookups or {}
        if lookups.get(key) is None:
            raise ValueError(
                f"启用 {family} 必须传入 {key}（与旧路径同源；"
                "启用而未传派生表必须报错，禁止静默补 NaN）"
            )


# ---------------------------------------------------------------------------
# panel 模式：store 公共读 API + 逐日 readiness + 一对一 join
# ---------------------------------------------------------------------------


def _window_days(calendar: Sequence[str], start_date: str, end_date: str) -> list[str]:
    days = [day for day in calendar if start_date <= day <= end_date]
    if not days:
        raise ValueError(f"窗口 [{start_date}, {end_date}] 内无交易日")
    return days


def _run_day_readiness(
    store: PanelDataStore, days: Sequence[str], calendar: Sequence[str], input_watermark: str
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, pd.DataFrame]]]:
    """逐日共享 readiness（§4 表第 4 行）：任一硬失败即抛（报错自带日期上下文）。

    P3-T2 加载成本门（§16.5-②）：改经收集变体——readiness 校验已全量读取的
    三表标签帧随返回（``frames_by_day[day][label_name]``），供六标签装配复用，
    消除 readiness 与标签装配的双通道重复全量读。
    """
    results = []
    frames_by_day: dict[str, dict[str, pd.DataFrame]] = {}
    for day in days:
        readiness, frames = validate_label_day_readiness_and_collect(
            store,
            day,
            consumer=True,
            trading_dates=calendar,
            input_watermark=input_watermark,
        )
        results.append(
            {
                "trade_date": day,
                "status": str(readiness["status"]),
                "rows": {str(k): int(v) for k, v in dict(readiness["rows"]).items()},
            }
        )
        frames_by_day[day] = frames
    return tuple(results), frames_by_day


def _assemble_label_columns(
    frames_by_day: dict[str, dict[str, pd.DataFrame]],
    days: Sequence[str],
    config: AdapterConfig,
) -> pd.DataFrame:
    """由 readiness 已读帧装配六标签列原名（P3-T2 收敛路径，§16.5-②）。

    与 T1 版 ``_load_label_columns``（经 ``store.load_labels`` 双变体读取）输出语义
    逐位一致：同一来源帧 ⇒ raw/neu 键集一致由构造保证（原双变体键集校验改为空转）；
    缺表 / 重复键 / 三表状态不一 ⇒ 失败（重复键与三表状态亦经 readiness 日级校验）。
    """
    columns: dict[str, pd.Series] = {}
    status_frames: list[pd.Series] = []
    for name, (value_col, neu_col) in LABEL_TABLES.items():
        table = pd.concat([frames_by_day[day][name] for day in days], ignore_index=True)
        # 与 load_labels 同形：(trade_date, ts_code) 索引 + 排序 + universe 过滤
        table = table.set_index(list(_KEY_COLUMNS)).sort_index()
        if config.universe is not None:
            codes = {str(getattr(c, "value", c)) for c in config.universe}
            table = table[table.index.get_level_values("ts_code").isin(codes)]
        if table.empty:
            raise RuntimeError(
                f"标签表 {name} 在窗口 [{config.start_date}, {config.end_date}] 缺表"
                "（无分区或未接入 labels 目录）"
            )
        if table.index.has_duplicates:
            raise RuntimeError(
                f"标签表 {name} 键 (trade_date, ts_code) 重复（一对一 join 前提破坏）"
            )
        columns[value_col] = table["label_value"]
        columns[neu_col] = table["neu_label_value"]
        status_frames.append(table["maturity_status"].rename(name))
    labels = pd.concat(columns, axis=1)
    status_df = pd.concat(status_frames, axis=1)
    if (status_df.nunique(axis=1) > 1).any():
        raise RuntimeError("三表 maturity_status 不一致（共享 max(h)=20 状态，禁止混用）")
    return labels


def _universe_codes(universe: tuple[str, ...] | None) -> list[TSCode] | None:
    return None if universe is None else [TSCode(str(code)) for code in universe]


def _join_features_labels(
    features: pd.DataFrame, labels: pd.DataFrame, config: AdapterConfig
) -> pd.DataFrame:
    """panel core 左表一对一 join 六标签列；意外增删键 fail-fast（§6.2-1）。"""
    if features.empty:
        raise RuntimeError(
            f"panel 特征窗口 [{config.start_date}, {config.end_date}] 为空（core 分区缺失）"
        )
    if not _index_keys_equal(features.index, labels.index):
        missing = features.index.difference(labels.index)
        extra = labels.index.difference(features.index)
        raise RuntimeError(
            "标签与 panel core 键集不一致（意外增删键 fail-fast）："
            f"core 有而标签缺 {len(missing)} 键（如 {list(missing[:3])}），"
            f"标签多出 {len(extra)} 键（如 {list(extra[:3])}）"
        )
    merged = features.join(labels)
    out = merged.reset_index()
    ordered = [*_KEY_COLUMNS, *config.feature_columns, *LABEL_VALUE_COLUMNS]
    return out[ordered]


def _index_keys_equal(left: pd.Index, right: pd.Index) -> bool:
    """按键值判等（str 归一）：规避 pandas 3.0 字符串 dtype 演进造成的假不等（reviewer1-P3-5）。"""
    if len(left) != len(right):
        return False
    left_frame = left.to_frame(index=False).astype(str)
    right_frame = right.to_frame(index=False).astype(str)
    return bool(left_frame.equals(right_frame))


def _collect_store_watermarks(store: PanelDataStore, input_watermark: str) -> AdapterWatermarks:
    """水位三对象分列采集（§4.1）：W_in + 三表 sealed_through + 各表最新分区日。

    分区信息一律取 **manifest 登记**（reviewer1-P3-4 单一来源），不扫目录名，
    避免孤儿 / 未登记文件污染证据。
    """
    snapshot = store.manifest.snapshot()
    sealed = {name: store.manifest.label_sealed_through(name) for name in LABEL_TABLES}
    label_latest = {}
    for name in LABEL_TABLES:
        partitions = snapshot["labels"].get(name, {}).get("partitions", {})
        days = [fn.removesuffix(".parquet") for fn in partitions]
        days = [d for d in days if _DATE_RE.fullmatch(d)]
        label_latest[name] = max(days) if days else None
    panel_latest = _latest_panel_day(snapshot)
    return AdapterWatermarks(
        input_watermark=input_watermark,
        label_sealed_through=sealed,
        label_latest_partition=label_latest,
        panel_latest_partition=panel_latest,
    )


def _latest_panel_day(snapshot: Mapping[str, Any]) -> str | None:
    """panel 最新分区日（manifest 登记）：热区日分区与冷区月分区（真实月末日）取最大。"""
    candidates: list[str] = []
    for rel_path in snapshot["partitions"]:
        parts = str(rel_path).split("/")
        if len(parts) != 3:
            continue
        if parts[0] == "panel" and _DATE_RE.fullmatch(parts[1]):
            candidates.append(parts[1])
        elif parts[0] == "panel_archive" and re.fullmatch(r"\d{4}-\d{2}", parts[1]):
            year, month = (int(part) for part in parts[1].split("-"))
            last_day = calendar_module.monthrange(year, month)[1]
            candidates.append(f"{year:04d}{month:02d}{last_day:02d}")
    return max(candidates) if candidates else None


def _load_panel_frame(
    store: PanelDataStore, config: AdapterConfig
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...], str, AdapterWatermarks, tuple[str, ...]]:
    root = store.root
    calendar = resolve_trading_dates(root)
    input_watermark = resolve_input_watermark(root)
    days = _window_days(calendar, config.start_date, config.end_date)
    readiness, frames_by_day = _run_day_readiness(store, days, calendar, input_watermark)
    features = store.load_features(
        FeatureQuery(
            columns=list(config.feature_columns),
            start_date=TradeDate.from_str(config.start_date),
            end_date=TradeDate.from_str(config.end_date),
            universe=_universe_codes(config.universe),
        )
    )
    # P3-T2 收敛路径（§16.5-②）：标签来自已校验日域（readiness 逐日收集），
    # 特征按区间读取可能带入日历外日期 ⇒ join 前先按已校验日域限定（与 T1 版
    # 「join 后限定」输出逐位一致；剔除日显式登记，不静默交给训练）。
    features_df, excluded_days = _restrict_features_to_validated_days(features.df, days)
    labels = _assemble_label_columns(frames_by_day, days, config)
    df = _join_features_labels(features_df, labels, config)
    df, excluded_post = _restrict_to_validated_days(df, days)
    excluded_days = tuple(sorted(set(excluded_days) | set(excluded_post)))
    watermarks = _collect_store_watermarks(store, input_watermark)
    return df, readiness, str(features.manifest_fingerprint), watermarks, excluded_days


def _restrict_features_to_validated_days(
    features: pd.DataFrame, days: Sequence[str]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """特征帧（index=(trade_date, ts_code)）限定已校验日域；保留索引形态（P3-T2）。

    与 ``_restrict_to_validated_days``（平铺帧版）同一语义：区间读取带入的日历外
    数据按日历过滤剔除并显式登记；剔除日期升序返回（正常路径应为空集）。
    """
    day_values = features.index.get_level_values("trade_date").astype(str)
    excluded = tuple(day for day in sorted(set(day_values)) if day not in set(days))
    if excluded:
        logger.warning(
            f"适配特征含已校验日域外日期（日历修订或分区越域），按日历过滤剔除: {list(excluded)}"
        )
        features = features[~day_values.isin(excluded)]
    return features, excluded


def _restrict_to_validated_days(
    df: pd.DataFrame, days: Sequence[str]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """输出帧限定已校验日域（reviewer2-N1 余项）：区间读取带入的日历外数据按日历过滤。

    与现役 ``load_features_data`` 的日历驱动枚举行集语义一致（G-code 行集前提）；
    剔除日期显式记录返回（不静默保留交给训练），正常路径应为空集。
    """
    present = sorted(str(day) for day in df["trade_date"].unique())
    validated = set(days)
    excluded = tuple(day for day in present if day not in validated)
    if excluded:
        logger.warning(
            f"适配输出含已校验日域外日期（日历修订或分区越域），按日历过滤剔除: {list(excluded)}"
        )
        df = df[df["trade_date"].isin(validated)].reset_index(drop=True)
    return df, excluded


# ---------------------------------------------------------------------------
# cs_train 验证源模式（验收专用过渡例外；镜像 load_features_data 读取语义）
# ---------------------------------------------------------------------------


def _cs_train_calendar_days(loader: Any, config: AdapterConfig) -> list[str]:
    """镜像 ``load_features_data`` 的日历口径：clean 优先 raw 兜底、is_open==1、窗口过滤。"""
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    if trade_cal is None:
        raise ValueError("无法加载交易日历（clean/raw 均缺失）")
    days = trade_cal[
        (trade_cal["cal_date"] >= config.start_date)
        & (trade_cal["cal_date"] <= config.end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()
    if not days:
        raise ValueError(f"窗口 [{config.start_date}, {config.end_date}] 内无交易日")
    return [str(day) for day in days]


def _assert_cs_train_coverage(storage: Any, days: Sequence[str]) -> None:
    """cs_train 分区覆盖核验（§4.1）：窗口内任一交易日缺分区 ⇒ 硬失败（带上下文）。"""
    cs_train_dir = Path(storage.features_path) / "cs_train"
    missing = [day for day in days if not (cs_train_dir / f"{day}.parquet").is_file()]
    if missing:
        raise RuntimeError(
            f"cs_train 验证源分区覆盖不全：窗口内 {len(missing)} 个交易日缺分区 "
            f"（首日 {missing[0]}，末日 {missing[-1]}）——现役读取静默跳过缺日，"
            "适配层按 §4.1 对查询窗口缺分区硬失败"
        )


def _load_cs_train_frame(
    storage: Any, loader: Any, config: AdapterConfig
) -> tuple[pd.DataFrame, int, AdapterWatermarks, tuple[dict[str, Any], ...]]:
    if config.universe is not None:
        raise ValueError("cs_train 验证源模式不支持 universe（现役 load_features_data 无此语义）")
    days = _cs_train_calendar_days(loader, config)
    _assert_cs_train_coverage(storage, days)
    frames = []
    for day in days:
        frame = storage.load_cs_train_day(day)
        if frame is not None and len(frame) > 0:
            frames.append(frame)
    if not frames:
        raise ValueError("指定日期区间内没有特征数据")
    df = pd.concat(frames, ignore_index=True)
    missing_labels = [c for c in LABEL_VALUE_COLUMNS if c not in df.columns]
    if missing_labels:
        raise RuntimeError(f"cs_train 分区缺标签列 {missing_labels}（六标签原名恢复前提破坏）")
    input_watermark = resolve_input_watermark(Path(storage.root_path))
    cs_train_latest = _latest_cs_train_day(Path(storage.features_path) / "cs_train")
    watermarks = AdapterWatermarks(
        input_watermark=input_watermark,
        label_sealed_through={name: None for name in LABEL_TABLES},
        label_latest_partition={name: None for name in LABEL_TABLES},
        panel_latest_partition=None,
        cs_train_latest_partition=cs_train_latest,
    )
    day_marks = tuple({"trade_date": day, "status": "cs_train_partition"} for day in days)
    return df, len(days), watermarks, day_marks


def _latest_cs_train_day(cs_train_dir: Path) -> str | None:
    if not cs_train_dir.is_dir():
        return None
    days = [p.stem for p in cs_train_dir.glob("*.parquet") if _DATE_RE.fullmatch(p.stem)]
    return max(days) if days else None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def validate_frame_request(
    config: AdapterConfig,
    *,
    store: PanelDataStore | None = None,
    storage: Any | None = None,
    loader: Any | None = None,
    acceptance_purpose: str | None = None,
) -> None:
    """训练帧请求轻量守卫（P3-T2 评审处置 R-CODEX-02）：配置校验 + 四族契约 +
    模式必需对象 + cs_train 验收可达性（含调用方白名单）。

    ``load_training_frame`` 与 ``TrainingFrameCache.get_or_load``（**含缓存命中路径**）
    每次调用都执行——缓存不得绕过适配器的强制拒绝规则；守卫不含 IO
    （合法命中零额外读取）。
    """
    _validate_config(config)
    if config.source_mode == SOURCE_PANEL:
        _validate_runtime_families(config)
        if store is None:
            raise ValueError("panel 模式必须提供 PanelDataStore（store=）")
        return
    _assert_cs_train_verify_reachable(acceptance_purpose)
    _validate_runtime_families(config)
    if storage is None or loader is None:
        raise ValueError("cs_train 验证源模式必须提供旧 Storage / DataLoader（storage=/loader=）")


def load_training_frame(
    config: AdapterConfig,
    *,
    store: PanelDataStore | None = None,
    storage: Any | None = None,
    loader: Any | None = None,
    acceptance_purpose: str | None = None,
) -> TrainingFrame:
    """加载训练帧（唯一适配点；``panel`` 默认 / ``cs_train_verify`` 验收专用）。

    Args:
        config: 适配请求配置（候选超集列集 + 窗口 + 标签列 + 模式）；
        store: panel 模式必填（``PanelDataStore``，只读消费）；
        storage/loader: cs_train 验证源模式必填（旧 ``Storage`` / ``DataLoader``，
            验收专用过渡依赖，去除节点 = P4 评审）；
        acceptance_purpose: cs_train 验证源模式必填（``ACCEPTANCE_PURPOSES`` 之一）。

    Returns:
        TrainingFrame（平铺 df 镜像 cs_train 形态 + 水位/逐日 readiness 证据）。
    """
    validate_frame_request(
        config,
        store=store,
        storage=storage,
        loader=loader,
        acceptance_purpose=acceptance_purpose,
    )
    if config.source_mode == SOURCE_PANEL:
        df, readiness, fingerprint, watermarks, excluded_days = _load_panel_frame(store, config)
        logger.info(
            f"panel 训练帧就绪: [{config.start_date}, {config.end_date}] "
            f"{readiness[-1]['trade_date'] if readiness else '-'} 止 {len(readiness)} 日 / "
            f"{len(df)} 行 / {len(config.feature_columns)} 特征列"
        )
        return TrainingFrame(
            df=df,
            trade_days=len(readiness),
            source_mode=config.source_mode,
            label_column=config.label_column,
            feature_columns=config.feature_columns,
            watermarks=watermarks,
            day_readiness=readiness,
            manifest_fingerprint=fingerprint,
            runtime_lookups=None,
            excluded_days=excluded_days,
        )
    df, n_days, watermarks, day_marks = _load_cs_train_frame(storage, loader, config)
    logger.info(
        f"cs_train 验证源训练帧就绪（用途 {acceptance_purpose}）: "
        f"[{config.start_date}, {config.end_date}] {n_days} 日 / {len(df)} 行"
    )
    actual_feature_columns = tuple(
        c for c in df.columns if c not in _KEY_COLUMNS and c not in LABEL_VALUE_COLUMNS
    )
    return TrainingFrame(
        df=df,
        trade_days=n_days,
        source_mode=config.source_mode,
        label_column=config.label_column,
        feature_columns=actual_feature_columns,
        watermarks=watermarks,
        day_readiness=day_marks,
        manifest_fingerprint=None,
        runtime_lookups=config.runtime_lookups,
    )
