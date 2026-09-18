"""股东增减持因子模块（stk_holdertrade）。

数据来源：TuShare `stk_holdertrade`（Phase 0 审计见 `docs/stk_holdertrade_pit_audit.md`），
raw 按 `ann_date` 年分区落盘（`data/raw/stk_holdertrade/YYYY-12-31.parquet`）。

**PIT 契约**：唯一可用时间是 `ann_date`（无 `begin_date/close_date`），因此 T 日特征只允许
使用 `ann_date <= T` 的公告；窗口一律按自然日切（30/90 日），与分红因子的"自然日"口径一致。

**量纲选择**：幅值一律用 `change_ratio`（变动占流通股比例，缺失率 0.02%），不用
`change_vol × avg_price`——`avg_price` 缺失 28.5%，且金额还要额外除流通市值；
`change_ratio` 本身已按流通股归一，跨股票可比、无价格依赖。

**稀疏性处理（与训练入口 0.6 缺失率门禁的关键配合）**：
- 查询表只输出**窗口内有事件的股票**（90 日活跃集中位约 970 只，远小于全市场），控制内存；
- 消费侧（`features/factor_handlers.py::HoldertradeFactorHandler`）对合并后的缺失**显式填 0**，
  语义 = "窗口内无增减持事件 ⇒ 净变动为 0"，从而使 9 个因子列在全市场口径下**全覆盖**，
  不会被 0.6 缺失率门禁整体删除（这是 Phase 0 审计标记的风险点）。

**为什么不做"事件新鲜度指数衰减"**：本模块输出的是**滚动窗口聚合**，窗口滑动本身就带时间衰减
（事件移出窗口即归零），若再套公共事件衰减会双重衰减；`ht_freshness_days` 因此作为普通信息列输出。
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd

#: 因子值列（滚动窗口聚合；由 handler 缺失填 0 后全覆盖）
HOLDERTRADE_COLS = [
    "ht_net_ratio_30d",  # 30 日净增持占流通比例（%，增持 − 减持）
    "ht_net_ratio_90d",  # 90 日净增持占流通比例（%）
    "ht_net_ratio_30d_exec",  # 30 日高管（holder_type=G）净增持占流通比例（%）
    "ht_net_ratio_30d_other",  # 30 日非高管净增持占流通比例（%）
    "ht_buy_count_30d",  # 30 日内出现增持披露的公告日数
    "ht_sell_count_30d",  # 30 日内出现减持披露的公告日数
    "ht_net_count_90d",  # 90 日净披露日数（增持日数 − 减持日数）
    "ht_net_ratio_accel",  # 近期强度加速度 = 30 日净比例 − 90 日净比例/3
]

#: 公告新鲜度（自然日；仅窗口内有事件的股票非空）
HOLDERTRADE_FRESHNESS_COL = "ht_freshness_days"

#: **精简列集（core）**：只保留 Phase 3 体检/诊断证据支持的 4 列
#: （`docs/holdertrade_factor_health.md`）——最强且逐年同号的两列（90 日净额 / 90 日净披露日数，
#: IC t≈15、7/7 年正、偏 IC t=6.9）+ 30 日减持披露计数（t=−10.9）+ 30 日净额（t=8.8）。
#: 剔除：`ht_buy_count_30d`（t=−0.32）、`ht_net_ratio_accel`（净额线性组合镜像）、
#: `ht_freshness_days`（符号翻转）、`ht_net_ratio_30d_other`（与 `ht_net_ratio_30d` 同簇）。
HOLDERTRADE_CORE_COLS = [
    "ht_net_ratio_90d",
    "ht_net_count_90d",
    "ht_sell_count_30d",
    "ht_net_ratio_30d",
]

#: 列集开关取值（签名维度，禁止跨取值并组比较）
HOLDERTRADE_FEATURE_SET_FULL = "full"
HOLDERTRADE_FEATURE_SET_CORE = "core"
HOLDERTRADE_FEATURE_SETS = (HOLDERTRADE_FEATURE_SET_FULL, HOLDERTRADE_FEATURE_SET_CORE)

#: schema 哨兵列与当前版本（语义重做时递增；训练入口校验）
HOLDERTRADE_VERSION_COL = "holdertrade_schema_v1"
HOLDERTRADE_SCHEMA_VERSION = 1

#: 窗口（自然日）
SHORT_WINDOW_DAYS = 30
LONG_WINDOW_DAYS = 90

#: 事件聚合内部列
_EVENT_COLS = [
    "ratio_net",
    "ratio_net_exec",
    "ratio_net_other",
    "has_in",
    "has_de",
]


def _to_ordinal(dates: pd.Series) -> np.ndarray:
    """YYYYMMDD 字符串 → 自然日序号（int32；非法日期为 -1）。"""
    parsed = pd.to_datetime(dates.astype(str), format="%Y%m%d", errors="coerce")
    ordinals = parsed.to_numpy(dtype="datetime64[D]").astype("int64")
    ordinals[pd.isna(parsed).to_numpy()] = -1
    return ordinals.astype("int64")


def aggregate_holdertrade_events(df: pd.DataFrame) -> pd.DataFrame:
    """把逐股东明细聚合为 (ts_code, ann_date) 事件行（PIT 事件单位）。

    逐行明细（同一股东分笔、多个股东）在因子层聚合：比例为**求和**，计数为**是否在该公告日出现**
    （`has_in`/`has_de`）——计数值按**公告日**统计而不是按行，是因为源数据含大量整行重复
    （审计实测 24.7%），按行计数会把重复披露放大成"多笔"。

    **前置条件**：输入应已由 raw 层（`holdertrade_raw.deduplicate_holdertrade`）按全字段去重；
    本函数检测到整行重复时告警（不静默）。`change_ratio` 缺失的行不计入比例（缺失率 0.02%）。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts_code", "ann_date", "ann_ord", *_EVENT_COLS])
    work = df.copy()
    for col in ("ts_code", "ann_date", "in_de", "holder_type", "change_ratio"):
        if col not in work.columns:
            raise ValueError(f"stk_holdertrade 缺少必要列: {col}")
    dup_rows = int(work.duplicated().sum())
    if dup_rows:
        logger.warning(
            f"[stk_holdertrade] 输入含 {dup_rows} 行整行重复（未去重？）——"
            "比例与计数可能被放大，请先经 raw 层去重"
        )
    work["ann_date"] = normalize_series_to_yyyymmdd(work["ann_date"])
    work["ann_ord"] = _to_ordinal(work["ann_date"])
    invalid = int((work["ann_ord"] < 0).sum())
    if invalid:
        logger.warning(f"[stk_holdertrade] 剔除 {invalid} 条非法 ann_date 记录（禁止伪日期）")
        work = work.loc[work["ann_ord"] >= 0].copy()
    if len(work) == 0:
        return pd.DataFrame(columns=["ts_code", "ann_date", "ann_ord", *_EVENT_COLS])

    direction = work["in_de"].astype(str).str.upper()
    unknown = sorted(set(direction.unique()) - {"IN", "DE"})
    if unknown:
        logger.warning(f"[stk_holdertrade] 忽略 in_de 非 IN/DE 的记录: {unknown}")
    work["_in"] = (direction == "IN").astype(float)
    work["_de"] = (direction == "DE").astype(float)
    ratio = pd.to_numeric(work["change_ratio"], errors="coerce").fillna(0.0)
    is_exec = work["holder_type"].astype(str).str.upper() == "G"

    work["ratio_net"] = ratio * (work["_in"] - work["_de"])
    work["ratio_net_exec"] = np.where(is_exec, work["ratio_net"], 0.0)
    work["ratio_net_other"] = np.where(is_exec, 0.0, work["ratio_net"])
    work["has_in"] = work["_in"]
    work["has_de"] = work["_de"]

    keys = ["ts_code", "ann_date", "ann_ord"]
    grouped = work.groupby(keys, as_index=False)
    # 比例列求和（同一公告日多股东/多笔累加），flag 列取 max（该日是否出现增/减持披露）
    events = grouped[["ratio_net", "ratio_net_exec", "ratio_net_other"]].sum()
    flags = grouped[["has_in", "has_de"]].max()
    events = events.merge(flags, on=keys, how="inner")
    events = events.sort_values(["ts_code", "ann_ord"], kind="mergesort").reset_index(drop=True)
    return events


def _window_sums(
    event_ord: np.ndarray,
    cum: np.ndarray,
    trade_ord: np.ndarray,
    window_days: int,
) -> np.ndarray:
    """滚动窗口和：Σ(事件序号 ∈ (T−window, T])，事件序号需升序。"""
    hi = np.searchsorted(event_ord, trade_ord, side="right")
    lo = np.searchsorted(event_ord, trade_ord - window_days, side="right")
    return cum[hi] - cum[lo]


def build_holdertrade_lookup_by_date(
    df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """按 `ann_date` PIT 对齐到日频的滚动窗口查询表。

    Args:
        df: raw stk_holdertrade（逐股东明细行）
        trading_dates: 交易日列表（YYYYMMDD，升序）

    Returns:
        ``{trade_date: DataFrame(ts_code + 因子列 + 哨兵列)}``，**只包含最近 90 自然日内
        有公告的股票**（消费侧需对其余股票显式填 0，见模块 docstring）。
    """
    events = aggregate_holdertrade_events(df)
    if len(events) == 0 or not trading_dates:
        return {}

    trade_dates = [normalize_series_to_yyyymmdd(pd.Series([d])).iloc[0] for d in trading_dates]
    trade_ord = _to_ordinal(pd.Series(trade_dates))
    if (trade_ord < 0).any():
        bad = [d for d, o in zip(trade_dates, trade_ord) if o < 0]
        raise ValueError(f"trading_dates 含非法日期: {bad[:5]}")
    order = np.argsort(trade_ord, kind="stable")
    trade_ord_sorted = trade_ord[order]

    out_ord: List[np.ndarray] = []
    out_code: List[np.ndarray] = []
    out_values: List[np.ndarray] = []

    for ts_code, grp in events.groupby("ts_code", sort=False):
        event_ord = grp["ann_ord"].to_numpy(dtype="int64")
        # 事件已按 ann_ord 升序（aggregate 内排序）；同日内多事件已聚合成一行
        cums = {
            col: np.concatenate([[0.0], np.cumsum(grp[col].to_numpy(dtype="float64"))])
            for col in _EVENT_COLS
        }
        net30 = _window_sums(event_ord, cums["ratio_net"], trade_ord_sorted, SHORT_WINDOW_DAYS)
        net90 = _window_sums(event_ord, cums["ratio_net"], trade_ord_sorted, LONG_WINDOW_DAYS)
        exec30 = _window_sums(
            event_ord, cums["ratio_net_exec"], trade_ord_sorted, SHORT_WINDOW_DAYS
        )
        other30 = _window_sums(
            event_ord, cums["ratio_net_other"], trade_ord_sorted, SHORT_WINDOW_DAYS
        )
        cnt_in30 = _window_sums(event_ord, cums["has_in"], trade_ord_sorted, SHORT_WINDOW_DAYS)
        cnt_de30 = _window_sums(event_ord, cums["has_de"], trade_ord_sorted, SHORT_WINDOW_DAYS)
        cnt_net90 = _window_sums(
            event_ord, cums["has_in"] - cums["has_de"], trade_ord_sorted, LONG_WINDOW_DAYS
        )
        # 事件数（90 日内）> 0 才算活跃；由 cumsum 差值给出
        ones = np.concatenate([[0.0], np.cumsum(np.ones(len(event_ord), dtype="float64"))])
        events_90 = _window_sums(event_ord, ones, trade_ord_sorted, LONG_WINDOW_DAYS)
        active = events_90 > 0
        if not active.any():
            continue
        hi = np.searchsorted(event_ord, trade_ord_sorted, side="right")
        last_ord = np.where(hi > 0, event_ord[np.clip(hi - 1, 0, None)], -1)
        freshness = np.where(hi > 0, trade_ord_sorted - last_ord, np.nan)

        rows = np.stack(
            [
                net30,
                net90,
                exec30,
                other30,
                cnt_in30,
                cnt_de30,
                cnt_net90,
                net30 - net90 / (LONG_WINDOW_DAYS / SHORT_WINDOW_DAYS),  # accel
                freshness,
            ],
            axis=1,
        )[active]
        out_ord.append(trade_ord_sorted[active])
        out_code.append(np.repeat(str(ts_code), int(active.sum())))
        out_values.append(rows)

    if not out_ord:
        return {}

    all_ord = np.concatenate(out_ord)
    all_code = np.concatenate(out_code)
    all_values = np.concatenate(out_values, axis=0).astype("float32", copy=False)
    columns = HOLDERTRADE_COLS + [HOLDERTRADE_FRESHNESS_COL]

    sort_idx = np.argsort(all_ord, kind="stable")
    all_ord, all_code, all_values = all_ord[sort_idx], all_code[sort_idx], all_values[sort_idx]
    unique_days, starts = np.unique(all_ord, return_index=True)
    bounds = np.append(starts, len(all_ord))

    lookup: Dict[str, pd.DataFrame] = {}
    for i, day_ord in enumerate(unique_days):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        frame = pd.DataFrame(all_values[lo:hi], columns=columns)
        frame.insert(0, "ts_code", all_code[lo:hi])
        frame[HOLDERTRADE_VERSION_COL] = np.int8(HOLDERTRADE_SCHEMA_VERSION)
        # 交易日自身日期（而非事件日）作为键：按交易日序号反查
        position = int(np.searchsorted(trade_ord_sorted, day_ord))
        lookup[trade_dates[position]] = frame
    logger.info(
        f"[stk_holdertrade] 因子查询表构建完成: {len(lookup)} 个交易日, "
        f"活跃事件行 {len(all_code)}（已按 90 日窗口收敛）"
    )
    return lookup


def available_holdertrade_columns() -> List[str]:
    """因子模块输出的全部列（含哨兵列），用于 schema 与 handler 默认列。"""
    return list(HOLDERTRADE_COLS) + [HOLDERTRADE_FRESHNESS_COL, HOLDERTRADE_VERSION_COL]


def holdertrade_feature_columns(feature_set: str = HOLDERTRADE_FEATURE_SET_FULL) -> List[str]:
    """按列集取值返回训练/派生使用的列清单（含哨兵列，单一取值判定，无回退）。"""
    if feature_set == HOLDERTRADE_FEATURE_SET_FULL:
        return available_holdertrade_columns()
    if feature_set == HOLDERTRADE_FEATURE_SET_CORE:
        return list(HOLDERTRADE_CORE_COLS) + [HOLDERTRADE_VERSION_COL]
    raise ValueError(
        f"未知 holdertrade 列集: {feature_set!r}（可选 {list(HOLDERTRADE_FEATURE_SETS)}）"
    )


def load_holdertrade_lookup(
    loader: Any,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """从 raw 加载 stk_holdertrade 并构建按交易日的查询表（运行时派生入口）。

    Args:
        loader: `DataLoader`（或具备 `load_stk_holdertrade()` 的等价对象）
        trading_dates: 需要覆盖的交易日（YYYYMMDD，升序）

    Returns:
        ``{trade_date: DataFrame}``；raw 为空时返回空字典（调用方按"无事件"处理）。
    """
    raw = loader.load_stk_holdertrade()
    if raw is None or len(raw) == 0:
        logger.warning("[stk_holdertrade] raw 为空，运行时派生将全部按 0 处理")
        return {}
    return build_holdertrade_lookup_by_date(raw, trading_dates)


def build_holdertrade_runtime_lookup(
    loader: Any,
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """按日期区间构建运行时查询表（YYYYMMDD；交易日由 loader 解析）。

    供训练 / OOS 评估入口一次性构建（全折共用同一张表）：查询表按**自然日**窗口聚合，
    与请求的区间长度无关，因此区间只决定覆盖哪些交易日。
    """
    if start_date > end_date:
        raise ValueError(f"日期区间非法: start={start_date} > end={end_date}")
    start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    trading_dates = [
        d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
        for d in loader.get_trading_dates(start_fmt, end_fmt)
    ]
    if not trading_dates:
        raise ValueError(f"无法解析交易日: {start_date} ~ {end_date}")
    return load_holdertrade_lookup(loader, trading_dates)


def derive_holdertrade_columns(
    frame: pd.DataFrame,
    lookup: Optional[Dict[str, pd.DataFrame]],
    wanted: Optional[Iterable[str]] = None,
    log_prefix: str = "",
) -> List[str]:
    """训练/OOS 评估侧就地派生股东增减持列（**不写回 cs_train / cs_infer 分区**）。

    与 `available_holdertrade_columns()` 同语义来源，复用 `HoldertradeFactorHandler`
    的"窗口外显式填 0 + 哨兵恒写"实现，保证训练侧、OOS 评估侧与推理侧逐值一致。

    规则（与可用性标记运行时派生一致）：
    - 需要 `trade_date` 与 `ts_code` 列，缺任一列直接报错（不得静默跳过）；
    - **已存在的列不覆盖**（特征分区若已含本族列，以分区为准）；
    - `wanted` 用于只派生模型实际使用的列（None = 全部）；
    - 查询表缺失的日期（或 lookup 为 None/空）⇒ 8 个因子列填 0、freshness 为 NaN、
      哨兵列写当前版本。

    Returns:
        实际新增的列名列表（按 `available_holdertrade_columns()` 顺序）。
    """
    # 延迟导入：features.factor_handlers 依赖本模块，避免顶层循环导入
    from ..features.factor_handlers import HoldertradeFactorHandler

    for col in ("trade_date", "ts_code"):
        if col not in frame.columns:
            raise ValueError(f"股东增减持运行时派生缺少必要列: {col}")
    if not frame.index.is_unique:
        raise ValueError("股东增减持运行时派生要求输入帧索引唯一（按位置回填）")

    candidates = available_holdertrade_columns()
    if wanted is not None:
        wanted_set = set(wanted)
        candidates = [col for col in candidates if col in wanted_set]
    targets = [col for col in candidates if col not in frame.columns]
    if not targets:
        return []

    values = {col: np.full(len(frame), np.nan, dtype="float64") for col in targets}
    handler = HoldertradeFactorHandler()
    lookup = lookup or {}
    for trade_date, day_df in frame.groupby("trade_date", sort=False):
        day_key = normalize_series_to_yyyymmdd(pd.Series([trade_date])).iloc[0]
        day_data = lookup.get(day_key)
        # lookup 为 None/缺该日 ⇒ 传空表（语义 = 窗口内无事件 ⇒ 0 填充），
        # 不能传 None（handler 视 None 为"整族未启用"而返回空字典）
        produced = handler.apply(
            day_df, day_data if day_data is not None else pd.DataFrame(), day_key, pd.DataFrame()
        )
        for col in targets:
            series = produced.get(col)
            values[col][frame.index.get_indexer(day_df.index)] = (
                series.to_numpy(dtype="float64")
                if series is not None
                else np.full(len(day_df), np.nan)
            )

    if targets:
        # 批量赋值（逐列赋值会让 383 列的特征帧高度碎片化并触发 PerformanceWarning）
        derived = pd.DataFrame({col: values[col] for col in targets}, index=frame.index)
        frame[targets] = derived
    if log_prefix:
        logger.info(f"{log_prefix}股东增减持运行时派生列: {targets}")
    return targets


def holdertrade_coverage_report(lookup: Dict[str, pd.DataFrame]) -> Optional[Tuple[int, float]]:
    """粗略覆盖率自检：返回 (活跃行总数, 单日活跃股票中位数)。"""
    if not lookup:
        return None
    counts = np.array([len(frame) for frame in lookup.values()], dtype="int64")
    return int(counts.sum()), float(np.median(counts))
