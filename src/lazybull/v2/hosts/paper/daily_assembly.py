# -*- coding: utf-8 -*-
"""v2 纸面宿主当日数据组装 + 量化就绪门（P4 T2，D4 定稿）。

职责边界
========
为纸面宿主组装「当日驱动引擎所需的数据」：特征按日平铺（panel 整窗读取 →
逐日截面字典）、行情窗口（clean/daily 近 250 个交易日，覆盖波动率 / Kelly
窗口）、以及组装前的量化就绪门报告。本模块**只读消费** v2/store panel
（构建语义禁动）与旧数据面 ``DataLoader``（E4 登记在册的只读过渡依赖），
不写任何数据根；交易日历由调用方（宿主）注入，本模块不自行决定日历。

特征列集 = 模型 feature_columns 并上选股过滤列 并上事件 freshness 闭包
----------------------------------------------------------------------
``amount_ma20`` / ``total_mv`` / ``sw_l1_code`` 是 MLSignal 选股过滤
（20 日均成交额下限 / 总市值区间 / 金融股剔除）的消费列，**不在**模型
feature_columns 内（现役 v24022 的 154 列已核实不含此三列）。不显式并上
查询列集时三列缺席，MLSignal 的过滤守卫（``if "amount_ma20" in
features_df.columns`` 形态）会**静默跳过**过滤——选股域失真且不可发现。
这是已核实的陷阱，故查询列集强制并上 ``SELECTION_FILTER_COLUMNS``，就绪门
第 4 条对过滤列同样做在场 + 缺失率校验，未登记 manifest 时直接判未就绪。

同类陷阱（F4 评审处置，2026-10 评审轮）：推理侧事件衰减以 freshness 辅助列在
**最终模型输入帧在场**为前提；策略缺省值与 MLSignal 一致，均为
``state_keep_event_decay``。查询列集并上 freshness 闭包；panel 缺列或行级 NaN 时，
cs_infer 回退值按 ts_code 真正合入当日帧（panel 非空值优先），而非仅作存在性放行。
集成模型经 ``freshness_members`` 按各子模型实际列集/策略分别计算依赖后取并集。

D4 量化就绪门口径（T2 定义并锁定）
==================================
逐条判定、全部写入 ``ReadinessReport``；**任一不过 ⇒ ok=False**。
「未就绪 ⇒ 推迟生成并告警、禁止静默用陈旧数据」的处置由宿主负责，本模块
只产出报告（不就绪本身不抛异常；契约 / 完整性错误如读取期间文件漂移仍穿透）。

1. **当日 panel 分区存在**：当日截面非空（store core 左表锚定，缺 core 即缺日）。
2. **行数带**：当日行数 ∈ 近 ``ROW_BAND_WINDOW``\\ =20 个交易日 panel 日行数
   中位数 × [``ROW_BAND_LOWER``\\ =0.9, ``ROW_BAND_UPPER``\\ =1.1]（T2 授权
   定稿的写死数值口径；中位数窗口含当日，仅统计窗口内有 panel 行的交易日；
   窗口内无任何 panel 行 ⇒ 基准缺失、不过）。
3. **逐列来源清单（显式来源制，F6 评审处置修订）**：模型 feature_columns
   逐列归类 = ``panel``（store manifest 已登记）/ ``sentinel``（形如
   ``*_schema_v<数字>`` 的哨兵列）/ ``derived``（availability ``has_*``、
   四族前缀 ``ht_`` / ``rp_`` / ``tfh_`` / ``ti_``、以及
   ``zscore_holdertrade`` / ``zscore_repurchase`` / ``zscore_top10fh`` /
   ``zscore_top_inst``）/ **未覆盖 ⇒ 直接失败**。分类规则 = manifest 登记
   优先 + 形态归类；现役 v24022 的 154 列全部落在 panel（事实已核）。
   「已覆盖」判定为显式来源制：(a) panel 类列纳入 store 查询、必须实际读出；
   (b) derived 列必须同时有族声明与 ``derived_producers`` 中**完整列名对应的生产者**，
   族名/前缀声明不再构成来源；生产者在就绪判定前执行，最终输入缺列或全 NaN 仍失败。
   sentinel 形态列按普通 store 列处理：未登记 ⇒ 未覆盖、直接失败。
4. **每列当日缺失率阈值**：panel 类列（模型 panel 列 + 选股过滤列）当日截面
   NaN 占比 ≤ ``COLUMN_MISSING_RATE_MAX``\\ =0.6（模块常量，构造可覆盖）。
   期望来自 store 的列未出现在组装结果 ⇒ 计入 ``uncovered_columns`` 且失败
   （``uncovered_columns`` = 第 3 条未覆盖列 ∪ 本条未出现列 ∪ 未登记过滤列
   的排序并集）。当日截面为空时逐列检查跳过（缺口已由第 1 条登记，不重复
   刷屏）。
   **例外（M-P4-1 裁决 C，2026-10-29）**：``SPARSE_TOLERATED_COLUMNS``（现役 =
   ``cons_target_upside``）为已知稀疏列——其当日缺失率超阈**不构成缺口**，但仍须
   实际在场且不得全空（100% 仍失败）；超阈事实按 WARNING 落日志、缺失率仍写入报告
   供留证。依据与后续 B 方案见该常量注释。
   **追加（F4）**：事件 freshness 闭包列的就绪判定归属本段，不参与上述缺失率阈值。
   策略缺省 = state_keep_event_decay；最终合入后的当日帧至少一行非 NaN 才算在场，
   panel 与 cs_infer 均无有效值 ⇒ 未就绪。当日 panel 截面为空时跳过（门 1 已登记）。
5. **派生族开关与模型 metadata 一致**：模型 ``train_params`` 的
   ``enable_holdertrade_features`` / ``enable_repurchase_features`` /
   ``enable_top10fh_features`` / ``enable_top_inst_features`` /
   ``enable_availability_markers`` 键（缺键 = 未启用）与宿主声明提供的派生族
   集合必须精确一致——模型启用而宿主未提供 ⇒ 失败；宿主提供而模型未启用 ⇒
   失败（禁止无谓派生）。本模块只核对 + 登记 ``derived_family_note``。
   现役 v24022 train_params 无上述键（已核实）⇒ 宿主不得启用任何派生。
6. **clean/daily 当日分区存在**：行情窗口帧含当日行（且 close 列在场，
   close 为 replay 口径必需列）。``sell_price_type == "open"`` 时追加校验
   （F9 评审处置）：当日行 ``open`` 列在场且至少一行非 NaN——卖出口径为
   open 时 open 缺失不在此拦截会拖到引擎执行才暴露；``None`` / ``"close"``
   不追加（维持原口径）。

只读冒烟辅助（非判据）
======================
``compare_cs_infer_vs_panel``：同一历史日 cs_infer 分区（旧推理特征面）vs
panel 同日截面的列覆盖 + 非空率对照（**不要求逐值一致**；任一侧缺分区按
空侧输出，不抛错），供 scripts/v2_p4 验收工具复用（P4 计划 D4-T2 只读
冒烟行）。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import AbstractSet, Any, Callable, Mapping, Sequence

import pandas as pd
from loguru import logger

# 只读过渡依赖（E4 登记在册）：旧数据面 DataLoader 仅用于 clean/daily 行情
# 窗口与 cs_infer 只读冒烟对照；v2/store 只读消费（构建语义禁动）。
from src.lazybull.data import DataLoader
from src.lazybull.v2.common.freshness import (
    EVENT_FRESHNESS_TO_VALUE_COLUMNS,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
)
from src.lazybull.v2.common.types import FeatureQuery, TradeDate
from src.lazybull.v2.store.data_store import PanelDataStore

#: 精确列级生产者：输入当日日期与当前特征帧，返回与该帧索引完全对齐的 Series；
#: None 表示未产出，由就绪门 fail-closed。生产者所需原始表/查询对象由闭包显式捕获。
DerivedProducer = Callable[[str, pd.DataFrame], pd.Series | None]

__all__ = [
    "DailyAssembly",
    "ReadinessReport",
    "DerivedProducer",
    "assemble_daily_data",
    "compare_cs_infer_vs_panel",
    "COLUMN_MISSING_RATE_MAX",
    "COLUMN_SOURCE_DERIVED",
    "COLUMN_SOURCE_PANEL",
    "COLUMN_SOURCE_SENTINEL",
    "COLUMN_SOURCE_UNCOVERED",
    "DERIVED_COLUMN_EXACT",
    "DERIVED_COLUMN_PREFIXES",
    "DERIVED_FAMILY_SWITCHES",
    "FEATURE_LOOKBACK_DAYS",
    "PRICE_DESIRED_COLUMNS",
    "PRICE_LOOKBACK_DAYS",
    "ROW_BAND_LOWER",
    "ROW_BAND_UPPER",
    "ROW_BAND_WINDOW",
    "SELECTION_FILTER_COLUMNS",
    "SPARSE_TOLERATED_COLUMNS",
]

# ---------------------------------------------------------------------------
# D4 定稿数值口径（T2 授权写死；改动 = 就绪门口径变更，须回契约登记）
# ---------------------------------------------------------------------------

#: 特征窗口交易日数（含当日）：取 N≈10 以覆盖补齐路径的 D-1 需求
FEATURE_LOOKBACK_DAYS = 10
#: 行情窗口交易日数（含当日）：覆盖波动率 / Kelly 窗口
PRICE_LOOKBACK_DAYS = 250
#: 行数带基准窗口（近 N 个交易日 panel 日行数中位数，含当日）
ROW_BAND_WINDOW = 20
#: 行数带下界系数（当日行数 >= 中位数 × 0.9）
ROW_BAND_LOWER = 0.9
#: 行数带上界系数（当日行数 <= 中位数 × 1.1）
ROW_BAND_UPPER = 1.1
#: 单列当日截面 NaN 占比阈值（缺失率 <= 0.6 放行；构造可覆盖）
COLUMN_MISSING_RATE_MAX = 0.6

#: 已知稀疏列白名单（M-P4-1 用户裁决 2026-10-29 = 方案 C「登记放行」）：
#: 命中列的当日缺失率超 ``COLUMN_MISSING_RATE_MAX`` **不构成缺口**，但仍要求
#: **实际在场**（未出现在组装结果 ⇒ 失败）且 **不得全空**（缺失率 100% ⇒ 仍失败，
#: 对齐旧链 ``signals/ml_signal.py`` 的「全空硬拒」）。
#: 登记依据（实测，可复算）：
#: - 训练入口 ``ml/train_core/prepare.py::max_feature_missing_ratio=0.6`` 的口径是
#:   **整训练帧池化**缺失率——``cons_target_upside`` 池化 57.2%(中位)~57.7%(均值)
#:   < 0.6 ⇒ 保留进 v24022 的 154 列，说明该列「高缺失」是训练期既有常态；
#: - 训练窗 20200106~20260105 内 **19.7% 的交易日该列即已 >60%**，模型照常训练；
#: - 旧链推理侧只对「全空」硬拒，``>50%`` 仅 WARNING（``_MISSING_RATE_WARN_THRESHOLD``），
#:   从不因此放弃该列 ⇒ 此前逐日截面 0.6 判据属「就绪门误判」（§15.6 系统原因词表），
#:   直接导致 v2 宿主每个交易日都不出信号、门 5/门 6 不可执行。
#: 后续义务：P4 实施规划 §23.8 方案 B（按列「训练期覆盖率 + 容差」替换本白名单，
#: 需训练侧把逐列 coverage 写入模型 metadata）；**本集合每次扩列均须回契约登记**。
SPARSE_TOLERATED_COLUMNS: frozenset[str] = frozenset({"cons_target_upside"})

#: 选股过滤列（MLSignal 过滤消费、不在模型 feature_columns 内；必须显式并上
#: 查询列集，否则过滤静默跳过——已核实陷阱，见模块 docstring）
SELECTION_FILTER_COLUMNS = ("amount_ma20", "total_mv", "sw_l1_code")

#: 行情列裁剪口径（与 hosts/backtest/replay.py::_load_replay_price_data 同款；
#: 按存在列交集裁剪，缺列不报错——close 缺席由就绪门第 6 条兜住）
PRICE_DESIRED_COLUMNS = (
    "ts_code",
    "trade_date",
    "close",
    "close_adj",
    "open",
    "open_adj",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "vol",
    "pct_chg",
    "is_st",
    "list_days",
    "tradable",
)

# ---------------------------------------------------------------------------
# 逐列来源分类常量（就绪门第 3 条）
# ---------------------------------------------------------------------------

COLUMN_SOURCE_PANEL = "panel"
COLUMN_SOURCE_SENTINEL = "sentinel"
COLUMN_SOURCE_DERIVED = "derived"
COLUMN_SOURCE_UNCOVERED = "uncovered"

#: 哨兵列形态（schema 版本标记常量列，如 cashflow_quality_schema_v2）
_SENTINEL_RE = re.compile(r"_schema_v\d+$")

#: 派生族列形态白名单：四族前缀 + availability ``has_*`` 前缀
DERIVED_COLUMN_PREFIXES = ("ht_", "rp_", "tfh_", "ti_", "has_")
#: 派生族精确名列（训练侧四族 zscore 列名，无前缀可匹配）
DERIVED_COLUMN_EXACT = frozenset(
    {"zscore_holdertrade", "zscore_repurchase", "zscore_top10fh", "zscore_top_inst"}
)

#: 派生族 → 模型 train_params 开关键（缺键 = 未启用；键名与训练侧
#: params_from_config 读取面一致。现役 v24022 五键全缺 ⇒ 未启用任何派生）
DERIVED_FAMILY_SWITCHES: dict[str, str] = {
    "availability": "enable_availability_markers",
    "holdertrade": "enable_holdertrade_features",
    "repurchase": "enable_repurchase_features",
    "top10fh": "enable_top10fh_features",
    "top_inst": "enable_top_inst_features",
}

#: 派生族列前缀 → 族名（F6 显式来源制的族归属单一来源；与
#: ``DERIVED_COLUMN_PREFIXES`` 同形，后者保留为导出口径常量）
_DERIVED_PREFIX_FAMILY: dict[str, str] = {
    "ht_": "holdertrade",
    "rp_": "repurchase",
    "tfh_": "top10fh",
    "ti_": "top_inst",
    "has_": "availability",
}

#: 派生族精确列 → 族名（与 ``DERIVED_COLUMN_EXACT`` 同形）
_DERIVED_EXACT_FAMILY: dict[str, str] = {
    "zscore_holdertrade": "holdertrade",
    "zscore_repurchase": "repurchase",
    "zscore_top10fh": "top10fh",
    "zscore_top_inst": "top_inst",
}


def _derived_family_of(column: str) -> str | None:
    """derived 形态列的族归属（键与 ``DERIVED_FAMILY_SWITCHES`` 对齐；非派生形态 ⇒ None）。"""
    exact = _DERIVED_EXACT_FAMILY.get(column)
    if exact is not None:
        return exact
    for prefix, family in _DERIVED_PREFIX_FAMILY.items():
        if column.startswith(prefix):
            return family
    return None


# ---------------------------------------------------------------------------
# 数据包与就绪报告
# ---------------------------------------------------------------------------


@dataclass
class ReadinessReport:
    """量化就绪门报告（D4；逐日登记，宿主据此决定推迟 / 告警）。"""

    date: str  # YYYYMMDD
    ok: bool  # 六条判定全过
    gaps: list[str]  # 未就绪原因清单（按判定条序）
    row_count: int  # 当日 panel 行数（缺日 = 0）
    row_count_band: tuple[float, float]  # 行数允许区间（基准缺失 = (nan, nan)）
    column_missing_rates: dict[str, float]  # panel 类列当日 NaN 占比
    uncovered_columns: list[str]  # 未覆盖列并集（排序；口径见模块 docstring 第 3/4 条）
    derived_family_note: str  # 派生族一致性核对登记


@dataclass
class DailyAssembly:
    """纸面宿主当日驱动数据包（就绪门未过时处置归宿主，本对象只承载事实）。"""

    date: str  # YYYYMMDD
    # 键 = YYYYMMDD 字符串；值 = 当日特征平铺 DataFrame（含 ts_code 列）
    features_by_date: dict[str, pd.DataFrame]
    # 行情窗口（含当日），trade_date 为 YYYYMMDD 字符串
    price_data: pd.DataFrame
    readiness: ReadinessReport


# ---------------------------------------------------------------------------
# 输入规范化与窗口
# ---------------------------------------------------------------------------


def _normalize_inputs(date: str, trading_dates: Sequence[str]) -> tuple[str, list[str]]:
    """入参规范化：当日格式校验 + 注入日历升序无重复校验 + 当日在历校验。"""
    date_str = str(date)
    TradeDate.from_str(date_str)  # 格式 / 真实日期校验（非法 ⇒ ValueError）
    calendar = [str(day) for day in trading_dates]
    if not calendar or len(set(calendar)) != len(calendar) or calendar != sorted(calendar):
        raise ValueError("注入的交易日历必须非空、升序且无重复（日历由宿主持有）")
    if date_str not in set(calendar):
        raise ValueError(f"当日 {date_str} 不在注入的交易日历中（本模块不自定日历）")
    return date_str, calendar


def _window_dates(calendar: Sequence[str], date: str, size: int) -> list[str]:
    """注入日历中 <= date 的最近 size 个交易日（含当日；调用方保证 date 在历）。"""
    index = calendar.index(date)
    return list(calendar[max(0, index - size + 1) : index + 1])


# ---------------------------------------------------------------------------
# 逐列来源分类（就绪门第 3 条）
# ---------------------------------------------------------------------------


def _classify_column(column: str, registered: AbstractSet[str]) -> str:
    """单列来源归类：manifest 登记优先（panel），形态归类兜底（sentinel/derived）。

    分类只承载形态信息，**不代表已覆盖**——覆盖判定为显式来源制（F6），
    由就绪门第 3 条按「panel 实际读出 / derived 族经宿主声明提供」判定。
    """
    if column in registered:
        return COLUMN_SOURCE_PANEL
    if _SENTINEL_RE.search(column):
        return COLUMN_SOURCE_SENTINEL
    if _derived_family_of(column) is not None:
        return COLUMN_SOURCE_DERIVED
    return COLUMN_SOURCE_UNCOVERED


def _classify_columns(
    feature_columns: Sequence[str], registered: AbstractSet[str]
) -> dict[str, str]:
    """模型列逐列来源分类（去重保序）。"""
    return {col: _classify_column(col, registered) for col in dict.fromkeys(feature_columns)}


def _store_query_columns(
    classification: Mapping[str, str],
    registered: AbstractSet[str],
    freshness_required: Sequence[str] = (),
) -> list[str]:
    """store 查询列集：panel 类模型列 + 已登记的选股过滤列 + 已登记的事件
    freshness 闭包列（去重保序）。

    sentinel / derived 类列不经 store 读取（未登记列查询会被 store 硬报错）：
    已登记哨兵归 panel 类自动纳入（F6：哨兵按普通 store 列处理）；derived
    由宿主声明的派生族在运行时提供；未登记的过滤列 / freshness 闭包列不进
    查询（分别由过滤列缺口与 freshness 在场判定登记未覆盖）。
    """
    columns = [col for col, source in classification.items() if source == COLUMN_SOURCE_PANEL]
    columns.extend(col for col in SELECTION_FILTER_COLUMNS if col in registered)
    columns.extend(col for col in freshness_required if col in registered)
    return list(dict.fromkeys(columns))


# ---------------------------------------------------------------------------
# 事件 freshness 推理闭包（F4 评审处置：state_keep_event_decay 辅助列在场判定）
# ---------------------------------------------------------------------------


def _required_freshness_columns(
    feature_columns: Sequence[str],
    train_params: Mapping[str, Any] | None,
    freshness_members: Sequence[tuple[Sequence[str], Mapping[str, Any]]] | None = None,
) -> list[str]:
    """事件衰减推理必需的 freshness 辅助列（单模型/集成子模型依赖并集）。

    缺省策略与生产 ``MLSignal`` 一致，均为 ``state_keep_event_decay``。集成模型
    逐子模型使用其**实际列集与实际策略**计算后取并集，禁止只采用 A 臂参数或
    用合并列集放大依赖。映射唯一来源 = ``EVENT_FRESHNESS_TO_VALUE_COLUMNS``。
    """
    members = (
        ((feature_columns, dict(train_params or {})),)
        if freshness_members is None
        else freshness_members
    )
    required: set[str] = set()
    for member_columns, member_params in members:
        params = dict(member_params or {})
        strategy = params.get("freshness_strategy", FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY)
        if strategy != FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY:
            continue
        member_feature_set = set(str(col) for col in member_columns)
        required.update(
            freshness_col
            for freshness_col, value_cols in EVENT_FRESHNESS_TO_VALUE_COLUMNS.items()
            if any(str(value) in member_feature_set for value in value_cols)
        )
    return sorted(required)


def _load_cs_infer_freshness_frame(
    loader: DataLoader, date: str, freshness_required: Sequence[str]
) -> pd.DataFrame:
    """读取 cs_infer 当日 freshness 回退原料（缺分区 ⇒ 空帧；完整性错误穿透）。"""
    frame = loader.storage.load_cs_train_day(date, subdir="cs_infer")
    if frame is None or frame.empty:
        return pd.DataFrame()
    columns = [
        col for col in ("trade_date", "ts_code", *freshness_required) if col in frame.columns
    ]
    return frame[columns].copy()


def _merge_cs_infer_freshness(
    day_frame: pd.DataFrame | None,
    cs_infer: pd.DataFrame,
    freshness_required: Sequence[str],
) -> tuple[pd.DataFrame | None, frozenset[str]]:
    """把 cs_infer 的 freshness **实际值**按 ts_code 合入当日模型输入。

    panel 非空值优先，cs_infer 仅补列缺失或行级 NaN；返回最终帧与回退后实际在场列。
    cs_infer 键重复属于输入契约错误，fail-fast，禁止任意取一行造成不确定衰减。
    """
    if day_frame is None or day_frame.empty or cs_infer.empty:
        return day_frame, frozenset()
    if "ts_code" not in day_frame.columns or "ts_code" not in cs_infer.columns:
        return day_frame, frozenset()
    merged = day_frame.copy()
    available: set[str] = set()
    for col in freshness_required:
        if col not in cs_infer.columns:
            continue
        source = cs_infer[["ts_code", col]].dropna(subset=["ts_code"])
        if source["ts_code"].duplicated().any():
            raise ValueError(f"cs_infer freshness 回退键 ts_code 重复（列 {col}），拒绝合入")
        values = source.set_index("ts_code")[col]
        fallback = merged["ts_code"].map(values)
        if col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), fallback)
        else:
            merged[col] = fallback
        if bool(merged[col].notna().any()):
            available.add(col)
    return merged, frozenset(available)


def _check_freshness_columns(
    freshness_required: Sequence[str],
    day_frame: pd.DataFrame | None,
    cs_infer_columns: AbstractSet[str],
) -> tuple[list[str], list[str]]:
    """第 4 条追加（F4）：事件 freshness 辅助列在场判定（fail-closed）。

    panel 侧在场 = 当日截面该列至少一行非 NaN；cs_infer 侧在场由
    ``cs_infer_columns``（同口径预筛结果）给出。必需 freshness 列两侧均缺席
    ⇒ 未就绪（返回缺失列清单与缺口清单）。当日 panel 截面为空 ⇒ 跳过
    （缺口已由第 1 条登记，不重复刷屏）。
    """
    if not freshness_required or day_frame is None or day_frame.empty:
        return [], []
    missing = [
        col
        for col in freshness_required
        if col not in cs_infer_columns
        and not (col in day_frame.columns and bool(day_frame[col].notna().any()))
    ]
    if not missing:
        return [], []
    gaps = [
        f"事件 freshness 辅助列缺失（state_keep_event_decay 推理必需，panel 与 "
        f"cs_infer 两侧均缺席 ⇒ 推理将静默跳过事件衰减，train/serve skew）: {missing}"
    ]
    return missing, gaps


# ---------------------------------------------------------------------------
# panel / 行情读取
# ---------------------------------------------------------------------------


def _load_panel_guarded(
    store: PanelDataStore, columns: Sequence[str], start: str, end: str
) -> tuple[pd.DataFrame, str | None]:
    """panel 整窗读取（守卫形态）：契约类失败（列登记 / available_from PIT /
    类型守卫）转为就绪缺口并返回空帧；完整性错误（读取期间文件漂移等
    RuntimeError）不在此吞掉，直接穿透。
    """
    try:
        frame = store.load_features(
            FeatureQuery(
                columns=list(columns),
                start_date=TradeDate.from_str(start),
                end_date=TradeDate.from_str(end),
            )
        )
    except (ValueError, TypeError) as exc:
        gap = f"panel 读取契约失败（列登记/PIT/类型守卫）: {exc}"
        logger.warning(f"纸面宿主数据组装 {end}: {gap}")
        return pd.DataFrame(), gap
    return frame.df, None


def _split_panel_frame_by_date(panel_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """panel 整窗帧（index=(trade_date, ts_code)）按日切分为逐日平铺帧。

    与 ``hosts/backtest/replay.py::_split_panel_frame_by_date`` 同款拆分语义
    （hosts 层不跨宿主 import 兄弟模块，同款逻辑本地承载）：输出形态镜像
    cs_infer 日分区（trade_date/ts_code 为列），行序 = panel 排序索引序。
    """
    features_by_date: dict[str, pd.DataFrame] = {}
    if panel_df is None or panel_df.empty:
        return features_by_date
    flat = panel_df.reset_index()
    flat["trade_date"] = flat["trade_date"].astype(str)
    for trade_date, day_df in flat.groupby("trade_date", sort=True):
        features_by_date[str(trade_date)] = day_df.reset_index(drop=True)
    return features_by_date


def _trim_price_columns(daily_data: pd.DataFrame) -> pd.DataFrame:
    """行情列裁剪（口径同 replay._load_replay_price_data：期望列按存在交集）。"""
    existing = [col for col in PRICE_DESIRED_COLUMNS if col in daily_data.columns]
    return daily_data[existing].copy()


def _load_price_window(
    loader: DataLoader, calendar: Sequence[str], date: str, lookback_days: int
) -> pd.DataFrame:
    """clean/daily 行情窗口读取（近 lookback_days 个交易日，含当日）。"""
    window = _window_dates(calendar, date, lookback_days)
    daily = loader.load_clean_daily(window[0], date)
    if daily is None or len(daily) == 0:
        logger.warning(f"纸面宿主数据组装 {date}: clean/daily {window[0]}~{date} 为空")
        return pd.DataFrame()
    return _trim_price_columns(daily)


# ---------------------------------------------------------------------------
# 就绪门六条判定（D4；每条返回缺口清单，全部记入 ReadinessReport.gaps）
# ---------------------------------------------------------------------------


def _check_day_partition(day_frame: pd.DataFrame | None, date: str) -> list[str]:
    """第 1 条：当日 panel 分区存在（当日截面非空）。"""
    if day_frame is None or len(day_frame) == 0:
        return [f"当日 panel 分区缺失或截面为空: {date}"]
    return []


def _check_row_band(
    row_count: int, day_counts: Mapping[str, int]
) -> tuple[tuple[float, float], list[str]]:
    """第 2 条：行数带（当日行数 ∈ 窗口日行数中位数 × [0.9, 1.1]，含当日）。"""
    if not day_counts:
        nan = float("nan")
        return (nan, nan), ["行数带基准缺失：窗口内无任何 panel 日行数"]
    median = float(statistics.median(day_counts.values()))
    band = (median * ROW_BAND_LOWER, median * ROW_BAND_UPPER)
    if not band[0] <= row_count <= band[1]:
        gap = (
            f"当日 panel 行数 {row_count} 掉出行数带 [{band[0]:.1f}, {band[1]:.1f}]"
            f"（近 {len(day_counts)} 交易日中位数 {median:.1f}"
            f" × [{ROW_BAND_LOWER}, {ROW_BAND_UPPER}]）"
        )
        return band, [gap]
    return band, []


def _check_column_sources(
    classification: Mapping[str, str],
    derived_families_provided: Sequence[str],
    derived_producer_columns: AbstractSet[str],
) -> tuple[list[str], list[str]]:
    """第 3 条：逐列来源清单（族开关 + 精确列生产者双重来源制）。

    derived 列必须同时满足：其族已声明提供、且存在以**完整列名**为键的生产者；
    族名或前缀本身不再构成来源。最终实际产出另由门 4++ 校验。
    """
    provided = set(derived_families_provided)
    gaps: list[str] = []
    uncovered: list[str] = []
    sentinel = sorted(
        col for col, source in classification.items() if source == COLUMN_SOURCE_SENTINEL
    )
    if sentinel:
        gaps.append(
            f"哨兵形态列未登记 store manifest（按普通 store 列处理，必须实际存在于 "
            f"panel；未登记 ⇒ 无产出者）: {sentinel}"
        )
        uncovered.extend(sentinel)
    derived_unprovided = sorted(
        col
        for col, source in classification.items()
        if source == COLUMN_SOURCE_DERIVED and _derived_family_of(col) not in provided
    )
    if derived_unprovided:
        gaps.append(
            f"派生形态列无声明来源（宿主 derived_families_provided 未提供其族，"
            f"运行时无产出者）: {derived_unprovided}"
        )
        uncovered.extend(derived_unprovided)
    derived_without_producer = sorted(
        col
        for col, source in classification.items()
        if source == COLUMN_SOURCE_DERIVED and col not in derived_producer_columns
    )
    if derived_without_producer:
        gaps.append(
            f"派生形态列无明确生产者（derived_producers 必须按完整列名注册，"
            f"族名/前缀声明不构成来源）: {derived_without_producer}"
        )
        uncovered.extend(derived_without_producer)
    plain = sorted(
        col for col, source in classification.items() if source == COLUMN_SOURCE_UNCOVERED
    )
    if plain:
        gaps.append(
            f"模型 feature_columns 存在未覆盖列（panel/sentinel/derived 均不命中）: {plain}"
        )
        uncovered.extend(plain)
    return gaps, sorted(uncovered)


def _check_missing_rates(
    day_frame: pd.DataFrame | None,
    expected_store_columns: Sequence[str],
    max_rate: float,
    tolerated: AbstractSet[str] = SPARSE_TOLERATED_COLUMNS,
) -> tuple[dict[str, float], list[str], list[str]]:
    """第 4 条：panel 类列当日缺失率阈值 + 期望列在场校验。

    返回 (各列缺失率, 未出现列清单, 缺口清单)；当日截面为空 ⇒ 逐列检查跳过
    （缺口已由第 1 条登记，不重复刷屏）。

    ``tolerated``（已知稀疏列，见 ``SPARSE_TOLERATED_COLUMNS``）：缺失率超阈
    不构成缺口，但**全空（100%）仍算缺口**——对齐旧链「全空硬拒、高缺失仅告警」
    的两级语义；缺失率一律写入返回值供证据留证，超阈事实另落 WARNING。
    """
    rates: dict[str, float] = {}
    missing: list[str] = []
    gaps: list[str] = []
    if day_frame is None or day_frame.empty:
        gaps.append("当日 panel 截面为空，逐列缺失率检查跳过（见第 1 条分区存在性缺口）")
        return rates, missing, gaps
    over: list[str] = []
    tolerated_over: list[str] = []
    for col in expected_store_columns:
        if col not in day_frame.columns:
            missing.append(col)
            continue
        rate = float(day_frame[col].isna().mean())
        rates[col] = rate
        if rate > max_rate:
            if col in tolerated and rate < 1.0:
                tolerated_over.append(f"{col}={rate:.1%}")
            else:
                over.append(f"{col}={rate:.1%}")
    if tolerated_over:
        logger.warning(
            f"已知稀疏列缺失率超阈但不构成缺口（{len(tolerated_over)} 列，"
            f"M-P4-1 裁决 C 登记放行；>={max_rate:.0%} 且 <100%）: {tolerated_over}"
        )
    if missing:
        gaps.append(f"期望来自 store 的列未出现在当日组装结果: {sorted(missing)}")
    if over:
        gaps.append(f"当日列缺失率超阈（>{max_rate:.0%}）: {over}")
    return rates, missing, gaps


def _apply_derived_producers(
    date: str,
    day_frame: pd.DataFrame | None,
    derived_columns: Sequence[str],
    derived_producers: Mapping[str, DerivedProducer],
) -> pd.DataFrame | None:
    """执行精确列级生产者并合入当日最终输入；未注册/未产出由就绪门拒绝。"""
    extra = sorted(set(derived_producers) - set(derived_columns))
    if extra:
        raise ValueError(f"derived_producers 注册了非模型所需 derived 列: {extra}")
    if day_frame is None or day_frame.empty:
        return day_frame
    merged = day_frame.copy()
    for col in derived_columns:
        producer = derived_producers.get(col)
        if producer is None:
            continue
        if not callable(producer):
            raise TypeError(f"derived 列生产者不可调用: {col}（{type(producer).__name__}）")
        values = producer(date, merged)
        if values is None:
            continue
        if not isinstance(values, pd.Series):
            raise TypeError(
                f"derived 列生产者必须返回索引对齐的 pd.Series 或 None: {col}（"
                f"实际 {type(values).__name__}）"
            )
        if not values.index.equals(merged.index):
            raise ValueError(f"derived 列生产者返回索引未与当日特征帧完全对齐: {col}")
        merged[col] = values
    return merged


def _check_derived_outputs(
    day_frame: pd.DataFrame | None,
    derived_columns: Sequence[str],
    derived_producer_columns: AbstractSet[str],
) -> tuple[list[str], list[str]]:
    """门 4++：有生产者的 derived 列必须在最终输入中实际存在且至少一个非 NaN。"""
    if day_frame is None or day_frame.empty:
        return [], []
    missing = [
        col
        for col in derived_columns
        if col in derived_producer_columns
        and (col not in day_frame.columns or not bool(day_frame[col].notna().any()))
    ]
    if not missing:
        return [], []
    return missing, [
        f"derived 生产者未在最终输入实际产出所需列/有效值（禁止静默补 NaN）: {missing}"
    ]


def _check_derived_consistency(
    train_params: Mapping[str, Any] | None,
    derived_families_provided: Sequence[str],
) -> tuple[str, list[str]]:
    """第 5 条：派生族开关与模型 metadata 一致（返回登记 note 与缺口清单）。

    宿主声明未知族名 ⇒ ValueError（调用方配置错误，fail-fast 不进缺口）。
    """
    provided = set(derived_families_provided)
    unknown = sorted(provided - set(DERIVED_FAMILY_SWITCHES))
    if unknown:
        raise ValueError(
            f"宿主声明的派生族未知: {unknown}（合法值 {sorted(DERIVED_FAMILY_SWITCHES)}）"
        )
    params = dict(train_params or {})
    enabled = sorted(
        family for family, key in DERIVED_FAMILY_SWITCHES.items() if bool(params.get(key))
    )
    missing = sorted(set(enabled) - provided)
    extra = sorted(provided - set(enabled))
    consistent = not missing and not extra
    note = (
        f"模型启用派生族={enabled}；宿主提供派生族={sorted(provided)}；"
        f"核对结论={'一致' if consistent else '不一致'}"
    )
    gaps: list[str] = []
    if missing:
        keys = [DERIVED_FAMILY_SWITCHES[family] for family in missing]
        gaps.append(f"派生族开关不一致：模型 metadata 启用而宿主未提供: {missing}（开关 {keys}）")
    if extra:
        gaps.append(f"派生族开关不一致：宿主提供而模型 metadata 未启用（禁止无谓派生）: {extra}")
    return note, gaps


def _check_price_today(
    price_data: pd.DataFrame, date: str, sell_price_type: str | None = None
) -> list[str]:
    """第 6 条：clean/daily 当日分区存在（行情帧含当日行）+ close 列在场。

    ``sell_price_type == "open"`` 时追加校验当日行 ``open`` 列在场且至少一行
    非 NaN（F9 评审处置：卖出口径为 open 时 open 缺失不在此拦截会拖到引擎
    执行才暴露）；``None`` / ``"close"`` 不追加，维持原口径。
    """
    if price_data.empty or "trade_date" not in price_data.columns:
        return [f"clean/daily 行情为空或缺 trade_date 列，当日 {date} 行情不可用"]
    if "close" not in price_data.columns:
        return ["clean/daily 行情帧缺 close 列（replay 口径必需列）"]
    today_mask = price_data["trade_date"].astype(str) == date
    if not bool(today_mask.any()):
        return [f"clean/daily 当日分区缺失：行情窗口内无 {date} 行"]
    if sell_price_type == "open":
        if "open" not in price_data.columns:
            return ["clean/daily 行情帧缺 open 列（sell_price=open 卖出口径必需列）"]
        if not bool(price_data.loc[today_mask, "open"].notna().any()):
            return [
                f"clean/daily 当日 {date} 行 open 列全为 NaN（sell_price=open 卖出口径无法定价）"
            ]
    return []


def _filter_uncovered_gaps(filter_uncovered: Sequence[str]) -> list[str]:
    """选股过滤列未登记 manifest 的缺口（过滤列缺席 ⇒ MLSignal 过滤静默跳过）。"""
    if not filter_uncovered:
        return []
    return [
        f"选股过滤列未登记 store manifest: {sorted(filter_uncovered)}"
        "（过滤列缺席 ⇒ MLSignal 选股过滤静默跳过，见模块 docstring）"
    ]


def _evaluate_readiness(
    date: str,
    day_frame: pd.DataFrame | None,
    day_counts: Mapping[str, int],
    classification: Mapping[str, str],
    expected_store_columns: Sequence[str],
    price_data: pd.DataFrame,
    train_params: Mapping[str, Any] | None,
    derived_families_provided: Sequence[str],
    column_missing_rate_max: float,
    pre_gaps: Sequence[str],
    pre_uncovered: Sequence[str],
    freshness_required: Sequence[str] = (),
    cs_infer_columns: AbstractSet[str] = frozenset(),
    derived_producer_columns: AbstractSet[str] = frozenset(),
    sell_price_type: str | None = None,
) -> ReadinessReport:
    """按条序组合六条判定（含第 4 条 F4 freshness 追加段），汇总为 ReadinessReport。"""
    gaps = list(pre_gaps)
    row_count = 0 if day_frame is None else len(day_frame)
    gaps.extend(_check_day_partition(day_frame, date))  # 1
    band, band_gaps = _check_row_band(row_count, day_counts)  # 2
    gaps.extend(band_gaps)
    source_gaps, source_uncovered = _check_column_sources(  # 3
        classification, derived_families_provided, derived_producer_columns
    )
    gaps.extend(source_gaps)
    freshness_set = set(freshness_required)
    # freshness 闭包列的就绪判定唯一归属 4+ 追加段（在场语义），不参与门 4
    # 缺失率阈值判定（panel 无值 ⇒ reindex 全 NaN，缺失率判定会遮蔽回退侧口径）
    rate_columns = [col for col in expected_store_columns if col not in freshness_set]
    rates, missing_columns, rate_gaps = _check_missing_rates(  # 4
        day_frame, rate_columns, column_missing_rate_max
    )
    gaps.extend(rate_gaps)
    freshness_missing, freshness_gaps = _check_freshness_columns(  # 4+（F4）
        freshness_required, day_frame, cs_infer_columns
    )
    gaps.extend(freshness_gaps)
    derived_output_missing, derived_output_gaps = _check_derived_outputs(  # 4++
        day_frame,
        [col for col, source in classification.items() if source == COLUMN_SOURCE_DERIVED],
        derived_producer_columns,
    )
    gaps.extend(derived_output_gaps)
    note, derived_gaps = _check_derived_consistency(train_params, derived_families_provided)  # 5
    gaps.extend(derived_gaps)
    gaps.extend(_check_price_today(price_data, date, sell_price_type))  # 6
    uncovered = sorted(
        set(pre_uncovered)
        | set(source_uncovered)
        | set(missing_columns)
        | set(freshness_missing)
        | set(derived_output_missing)
    )
    return ReadinessReport(
        date=date,
        ok=not gaps,
        gaps=gaps,
        row_count=row_count,
        row_count_band=band,
        column_missing_rates=rates,
        uncovered_columns=uncovered,
        derived_family_note=note,
    )


def _log_readiness(report: ReadinessReport) -> None:
    """就绪结论日志留痕（推迟生成与告警的处置归宿主）。"""
    if report.ok:
        logger.info(
            f"纸面宿主就绪门通过: {report.date}（当日行数 {report.row_count}"
            f" ∈ [{report.row_count_band[0]:.1f}, {report.row_count_band[1]:.1f}]）"
        )
        return
    logger.warning(
        f"纸面宿主就绪门未通过: {report.date}，缺口 {len(report.gaps)} 项: {report.gaps}"
    )


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def assemble_daily_data(
    date: str,
    store: PanelDataStore,
    loader: DataLoader,
    trading_dates: Sequence[str],
    feature_columns: Sequence[str],
    train_params: Mapping[str, Any] | None = None,
    freshness_members: Sequence[tuple[Sequence[str], Mapping[str, Any]]] | None = None,
    derived_families_provided: Sequence[str] = (),
    derived_producers: Mapping[str, DerivedProducer] | None = None,
    sell_price_type: str | None = None,
    feature_lookback_days: int = FEATURE_LOOKBACK_DAYS,
    price_lookback_days: int = PRICE_LOOKBACK_DAYS,
    row_band_window: int = ROW_BAND_WINDOW,
    column_missing_rate_max: float = COLUMN_MISSING_RATE_MAX,
) -> DailyAssembly:
    """组装当日驱动数据并执行量化就绪门（D4 六条；口径见模块 docstring）。

    panel 读取窗口 = max(row_band_window, feature_lookback_days) 个交易日
    （一次读取同时供行数带基准与特征按日拆分）；``features_by_date`` 只保留
    近 feature_lookback_days 个交易日（含当日）。

    Args:
        date: 当日交易日（YYYYMMDD；必须在注入日历内）。
        store: v2 PanelDataStore（只读消费；读取回执随 store 产出）。
        loader: 旧数据面 DataLoader（E4 只读过渡依赖；仅用于 clean/daily）。
        trading_dates: 宿主注入的完整交易日历（升序 YYYYMMDD，无重复）。
        feature_columns: 模型 metadata 的实际特征列（就绪门第 3/4 条判定基准；
            查询时并上选股过滤列 ``SELECTION_FILTER_COLUMNS``）。
        train_params: 模型 metadata 的 train_params（派生族核对与单模型 freshness 判定；
            freshness_strategy 缺省 = state_keep_event_decay）。
        freshness_members: 集成各子模型 (实际列集, train_params)，逐臂计算后取并集。
        derived_families_provided: 宿主声明提供的运行时派生族（合法值见
            ``DERIVED_FAMILY_SWITCHES``；未知值 ⇒ ValueError）。
        derived_producers: 完整 derived 列名 → 实际生产者；返回索引对齐 Series 或 None。
        sell_price_type: 卖出口径价格列（宿主接线 ``TradingConfig.sell_price``）；
            ``"open"`` 时就绪门第 6 条追加当日 ``open`` 列在场与非空校验（F9），
            ``None`` / ``"close"`` 维持原口径。
        feature_lookback_days: 特征窗口交易日数（默认 ``FEATURE_LOOKBACK_DAYS``）。
        price_lookback_days: 行情窗口交易日数（默认 ``PRICE_LOOKBACK_DAYS``）。
        row_band_window: 行数带基准窗口（默认 ``ROW_BAND_WINDOW``）。
        column_missing_rate_max: 单列当日缺失率阈值（默认
            ``COLUMN_MISSING_RATE_MAX``，构造可覆盖）。

    Returns:
        DailyAssembly；未就绪时 ``readiness.ok=False`` 且 gaps 全列——
        「推迟生成并告警、禁止静默用陈旧数据」的处置由宿主负责。

    Raises:
        ValueError: 日历非法 / date 不在历 / feature_columns 为空 /
            派生族声明未知。
    """
    date_str, calendar = _normalize_inputs(date, trading_dates)
    if not feature_columns:
        raise ValueError("feature_columns 不能为空（模型列集是就绪门第 3/4 条的判定基准）")
    registered = set(store.manifest.snapshot()["columns"])
    classification = _classify_columns(feature_columns, registered)
    freshness_required = _required_freshness_columns(
        feature_columns, train_params, freshness_members
    )
    query_columns = _store_query_columns(classification, registered, freshness_required)
    filter_uncovered = [c for c in SELECTION_FILTER_COLUMNS if c not in registered]
    panel_window = _window_dates(calendar, date_str, max(feature_lookback_days, row_band_window))
    panel_df, load_gap = _load_panel_guarded(store, query_columns, panel_window[0], date_str)
    all_days = _split_panel_frame_by_date(panel_df)
    day_counts = {day: len(frame) for day, frame in all_days.items()}
    keep = set(_window_dates(calendar, date_str, feature_lookback_days))
    features_by_date = {day: frame for day, frame in all_days.items() if day in keep}
    price_data = _load_price_window(loader, calendar, date_str, price_lookback_days)
    # freshness 回退值必须进入最终模型输入，而非仅用于“别处存在”的就绪放行。
    cs_infer_columns: frozenset[str] = frozenset()
    if freshness_required:
        cs_infer = _load_cs_infer_freshness_frame(loader, date_str, freshness_required)
        merged_day, cs_infer_columns = _merge_cs_infer_freshness(
            all_days.get(date_str), cs_infer, freshness_required
        )
        if merged_day is not None:
            all_days[date_str] = merged_day
            if date_str in features_by_date:
                features_by_date[date_str] = merged_day
    derived_columns = [
        col for col, source in classification.items() if source == COLUMN_SOURCE_DERIVED
    ]
    producers = dict(derived_producers or {})
    derived_day = _apply_derived_producers(
        date_str, all_days.get(date_str), derived_columns, producers
    )
    if derived_day is not None:
        all_days[date_str] = derived_day
        if date_str in features_by_date:
            features_by_date[date_str] = derived_day
    pre_gaps = ([load_gap] if load_gap else []) + _filter_uncovered_gaps(filter_uncovered)
    readiness = _evaluate_readiness(
        date_str,
        all_days.get(date_str),
        day_counts,
        classification,
        query_columns,
        price_data,
        train_params,
        derived_families_provided,
        column_missing_rate_max,
        pre_gaps,
        filter_uncovered,
        freshness_required,
        cs_infer_columns,
        set(producers),
        sell_price_type,
    )
    _log_readiness(readiness)
    return DailyAssembly(
        date=date_str,
        features_by_date=features_by_date,
        price_data=price_data,
        readiness=readiness,
    )


# ---------------------------------------------------------------------------
# 只读冒烟辅助（T2，非判据；供 scripts/v2_p4 验收工具复用）
# ---------------------------------------------------------------------------


def _nonnull_rates(df: pd.DataFrame) -> dict[str, float]:
    """逐列非空率（空帧 ⇒ 空表）。"""
    if df is None or df.empty:
        return {}
    return {str(col): float(df[col].notna().mean()) for col in df.columns}


def _load_panel_day_for_compare(store: PanelDataStore, date: str) -> pd.DataFrame:
    """panel 单日全登记列截面（平铺形态；契约失败按空侧输出并告警）。"""
    columns = sorted(store.manifest.snapshot()["columns"])
    try:
        frame = store.load_features(
            FeatureQuery(
                columns=columns,
                start_date=TradeDate.from_str(date),
                end_date=TradeDate.from_str(date),
            )
        )
    except (ValueError, TypeError) as exc:
        logger.warning(f"panel 当日截面读取契约失败 {date}（对照按空侧输出）: {exc}")
        return pd.DataFrame()
    if frame.df.empty:
        logger.warning(f"panel 当日截面为空 {date}（对照按空侧输出）")
        return pd.DataFrame()
    return frame.df.reset_index()


def compare_cs_infer_vs_panel(
    date: str, store: PanelDataStore, loader: DataLoader
) -> dict[str, Any]:
    """只读冒烟辅助（T2，非判据）：同一历史日 cs_infer 分区 vs panel 同日截面。

    输出列覆盖 + 非空率对照（**不要求逐值一致**）；任一侧缺分区按空侧输出
    （空列清单 + 空非空率表）并告警，不抛错。供 scripts/v2_p4 就绪门冒烟
    工具复用（P4 计划 D4-T2 只读冒烟行）。

    Returns:
        {date, cs_infer_cols, panel_cols, common_cols, only_cs_infer, only_panel,
        nonnull_rate_cs_infer, nonnull_rate_panel}；列清单均为排序后的字符串
        列表，非空率为 {列名: 0~1 浮点}（各侧仅覆盖本侧实到列）。
    """
    date_str = str(date)
    TradeDate.from_str(date_str)  # 格式校验（非法 ⇒ ValueError）
    cs_infer = loader.storage.load_cs_train_day(date_str, subdir="cs_infer")
    if cs_infer is None:
        logger.warning(f"cs_infer 分区不存在 {date_str}（对照按空侧输出）")
        cs_infer = pd.DataFrame()
    panel_day = _load_panel_day_for_compare(store, date_str)
    cs_cols = sorted(str(col) for col in cs_infer.columns)
    panel_cols = sorted(str(col) for col in panel_day.columns)
    return {
        "date": date_str,
        "cs_infer_cols": cs_cols,
        "panel_cols": panel_cols,
        "common_cols": sorted(set(cs_cols) & set(panel_cols)),
        "only_cs_infer": sorted(set(cs_cols) - set(panel_cols)),
        "only_panel": sorted(set(panel_cols) - set(cs_cols)),
        "nonnull_rate_cs_infer": _nonnull_rates(cs_infer),
        "nonnull_rate_panel": _nonnull_rates(panel_day),
    }
