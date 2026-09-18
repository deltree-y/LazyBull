"""股票回购因子模块（repurchase）。

数据来源：TuShare `repurchase`（Phase 0 审计与 **Phase 2 口径定稿**见
`docs/repurchase_pit_audit.md` §5.1 / §6.1），raw 按 `ann_date` 年分区落盘
（`data/raw/repurchase/YYYY-12-31.parquet`）。

**PIT 契约**：raw 无 `begin_date`，`end_date` 是进度报告期而非"发生日"，因此唯一可用时间是
`ann_date`；T 日特征只允许使用 `ann_date <= T` 的公告，窗口一律按**自然日**开区间切（90/180 日）。

**列集（最小集，一次到位）**：`rp_amount_to_mv_90d` / `rp_amount_to_mv_180d`（公告回购金额 ÷
流通市值）、`rp_exec_flag_90d`（是否已进入实施/完成）、`rp_price_headroom`（回购价上限相对现价的
"托底空间"）、`rp_freshness_days`（新鲜度）与哨兵列 `repurchase_schema_v1`。

**金额缺失不兜底**（口径定稿 §6.1 第 2 条）：`amount` 缺失 5.7%，**不用** `vol × high_limit` 近似
（两种口径混进同一列会让强度跨股票不可比）；缺失行增量记 0 并计数告警；
`amount` 整列缺失或全空**直接报错**（禁止静默零因子）。

**执行类金额为累计口径 ⇒ 必须取增量**（口径定稿 §6.1 第 4.1 条，2026-09-18 实测发现）：
`proc ∈ {实施, 完成}` 的 `amount` 是**累计已回购金额**（同一股票连续公告中位末/首比 ≈12.5），
窗口内直接求和会把同一笔回购重复计 10 倍以上；非执行类行（预案/股东大会通过）的 `amount` 是
**计划金额**且同一计划会跨期重复披露（末首比中位 1.04）⇒ 不进入强度列（只贡献价格上限与新鲜度）。

**价格口径**：`rp_price_headroom` 用 VWAP（`amount(千元) × 10 ÷ vol(手)`，元/股）而不是 `close`——
cs_train **不含**未复权 `close`，用 VWAP 可让四侧接线（训练 / OOS 评估 / OOS 回测 / 纸面）零额外管道。

**不做计划级版本化**：`proc` 是状态字段且同一计划跨多期公告，"进度/完成率"类因子必须另立方案
（以 `(ts_code, exp_date/end_date)` 近似计划身份）；本轮只做公告级窗口聚合。

**稀疏性处理**：查询表只输出「180 自然日内有公告」的股票；消费侧
（`features/factor_handlers.py::RepurchaseFactorHandler`）对窗口外股票**显式填 0**（语义 = 窗口内
无回购计划），保证 4 个因子列全市场覆盖，不会被训练入口 0.6 缺失率门禁整体删除。

**不做公共事件衰减**：窗口滑动本身即时间衰减（同 stk_holdertrade 契约）。
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd

#: 因子值列（滚动窗口聚合；由 handler 缺失填 0 后全覆盖）
REPURCHASE_COLS = [
    "rp_amount_to_mv_90d",  # 近 90 自然日公告回购金额 ÷ 流通市值
    "rp_amount_to_mv_180d",  # 近 180 自然日公告回购金额 ÷ 流通市值
    "rp_exec_flag_90d",  # 90 日内是否出现「实施/完成」阶段公告（0/1）
    "rp_price_headroom",  # 窗口内最新 high_limit ÷ 当日 VWAP − 1
]

#: 公告新鲜度（自然日；仅窗口内有事件的股票非空）
REPURCHASE_FRESHNESS_COL = "rp_freshness_days"

#: schema 哨兵列与当前版本（语义重做时递增；训练入口校验）
REPURCHASE_VERSION_COL = "repurchase_schema_v1"
REPURCHASE_SCHEMA_VERSION = 1

#: 窗口（自然日）
SHORT_WINDOW_DAYS = 90
LONG_WINDOW_DAYS = 180

#: `proc` 登记取值（其余取值忽略阶段语义、仅计金额并告警）
PROC_VALUES = ("预案", "股东大会通过", "实施", "完成", "停止")

#: 进入「已执行」状态的 proc 取值（其 `amount` 为**累计已回购金额**）
EXECUTED_PROCS = ("实施", "完成")

#: 查询表内部列（不进模型；`rp_limit_price` 由 handler 结合当日 VWAP 折算成 headroom）
LOOKUP_VALUE_COLS = [
    "rp_amt_90d",
    "rp_amt_180d",
    "rp_exec_90d",
    "rp_limit_price",
]

#: 事件聚合后的列
_EVENT_COLS = ["amt", "exec_flag", "limit_price"]


def _to_ordinal(dates: pd.Series) -> np.ndarray:
    """YYYYMMDD 字符串 → 自然日序号（int64；非法日期为 -1）。"""
    parsed = pd.to_datetime(dates.astype(str), format="%Y%m%d", errors="coerce")
    ordinals = parsed.to_numpy(dtype="datetime64[D]").astype("int64")
    ordinals[pd.isna(parsed).to_numpy()] = -1
    return ordinals.astype("int64")


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


def _executed_increments(events: pd.DataFrame) -> pd.Series:
    """执行类公告的**增量**金额（源数据为累计口径，禁止窗口内直接求和）。

    规则（口径定稿 §6.1 第 4.1 条）：同一股票的执行类行按 `ann_date` 升序，
    `增量 = 本行金额 − 上一有效执行类行金额`；若本行金额更小（新计划开始/数据修正）或为
    首行 ⇒ 视为新计划，`增量 = 本行金额`；`amount` 为 NaN ⇒ 增量 0 且**不更新基线**。
    非执行类行增量恒为 0。

    Args:
        events: 已按 `(ts_code, ann_ord)` 升序的明细/聚合行（需含 `amt_raw` 与 `exec_flag`）

    Returns:
        与 `events` 同索引的增量金额 Series。
    """
    raw = pd.to_numeric(events["amt_raw"], errors="coerce").to_numpy(dtype="float64")
    is_exec = events["exec_flag"].to_numpy(dtype="float64") > 0
    codes = events["ts_code"].to_numpy()
    increments = np.zeros(len(events), dtype="float64")
    base = np.nan
    current = None
    for i in range(len(events)):
        code = codes[i]
        if code != current:
            current = code
            base = np.nan
        if not is_exec[i]:
            continue
        value = raw[i]
        if np.isnan(value):
            continue  # 缺失不兜底：增量 0，基线不变
        increments[i] = value if (np.isnan(base) or value < base) else value - base
        base = value
    return pd.Series(increments, index=events.index)


def aggregate_repurchase_events(df: pd.DataFrame) -> pd.DataFrame:
    """把逐公告明细聚合为 (ts_code, ann_date) 事件行（PIT 事件单位）。

    同一公告日的多行（多阶段/多明细，源数据 `ts_code + ann_date` 重复 4.78%）按以下口径聚合：
    - `amt` = 该日各行**增量金额**之和（执行类行先取累计增量，见 `_executed_increments`；
      非执行类行（预案/股东大会通过）的 `amount` 是**计划金额**且跨期重复披露 ⇒ 增量恒 0）；
    - `exec_flag` **取 max**（该日是否出现「实施/完成」阶段）；
    - `limit_price` 取该日**非空最大值**（`high_limit`，元/股）。

    禁止按行计数：源数据无自然键，按行统计会把同一公告的重复明细放大成"多笔"。

    **前置条件**：输入应已由 raw 层（`repurchase_raw.deduplicate_repurchase`）按全字段去重；
    本函数检测到整行重复时告警（不静默）。
    """
    empty = pd.DataFrame(
        columns=["ts_code", "ann_date", "ann_ord", *_EVENT_COLS],
    )
    if df is None or len(df) == 0:
        return empty
    work = df.copy()
    for col in ("ts_code", "ann_date", "proc"):
        if col not in work.columns:
            raise ValueError(f"repurchase 缺少必要列: {col}")
    if "amount" not in work.columns:
        raise ValueError("repurchase 缺少 amount 列（禁止静默零因子，请重下数据）")
    if pd.to_numeric(work["amount"], errors="coerce").notna().sum() == 0:
        raise ValueError(
            "repurchase amount 整列为空（禁止静默零因子）：请重下数据"
            "（python scripts/download_raw.py --download repurchase）"
        )

    dup_rows = int(work.duplicated().sum())
    if dup_rows:
        logger.warning(
            f"[repurchase] 输入含 {dup_rows} 行整行重复（未去重？）——"
            "金额与计数可能被放大，请先经 raw 层去重"
        )

    work["ann_date"] = normalize_series_to_yyyymmdd(work["ann_date"])
    work["ann_ord"] = _to_ordinal(work["ann_date"])
    invalid = int((work["ann_ord"] < 0).sum())
    if invalid:
        logger.warning(f"[repurchase] 剔除 {invalid} 条非法 ann_date 记录（禁止伪日期）")
        work = work.loc[work["ann_ord"] >= 0].copy()
    if len(work) == 0:
        return empty

    amount = pd.to_numeric(work["amount"], errors="coerce")
    missing_amount = int(amount.isna().sum())
    if missing_amount:
        # 口径定稿 §6.1 第 2 条：不做 vol × high_limit 兜底，缺失增量记 0 并计数告警
        logger.warning(
            f"[repurchase] {missing_amount} 条公告缺 amount"
            f"（{missing_amount / len(work):.2%}），按登记口径**不兜底**：该行增量记 0"
            "（执行类行的累计基线不更新）"
        )
    work["amt_raw"] = amount

    proc = work["proc"].astype(str).str.strip()
    unknown = sorted(set(proc.unique()) - set(PROC_VALUES))
    if unknown:
        logger.warning(f"[repurchase] 未登记 proc 取值（忽略阶段语义，不计金额）: {unknown}")
    work["exec_flag"] = proc.isin(EXECUTED_PROCS).astype("float64")

    if "high_limit" in work.columns:
        limit = pd.to_numeric(work["high_limit"], errors="coerce")
        work["limit_price"] = limit.where(limit > 0)
    else:
        work["limit_price"] = np.nan

    # 执行类金额为**累计口径** ⇒ 先按公告日升序取增量，再按 (ts_code, ann_date) 聚合
    # （口径定稿 §6.1 第 4.1 条：窗口内直接求和会把同一笔回购重复计 10 倍以上）。
    # 次级键 `end_date`（进度报告期）：同日多行若按文件行序处理，遇到 end_date 与行序
    # 不一致的两行（实测 300138.SZ 20240604 同时报 20240531/20240603 两期）会把"回落"
    # 误判为新计划、虚增一笔增量；NaN 的 end_date 排在最后（不打断已知时间序）。
    if "end_date" in work.columns:
        end_ord = _to_ordinal(normalize_series_to_yyyymmdd(work["end_date"]))
        end_ord = np.where(end_ord < 0, np.iinfo("int64").max, end_ord)
    else:
        end_ord = np.full(len(work), np.iinfo("int64").max, dtype="int64")
    work["end_ord"] = end_ord
    work = work.sort_values(["ts_code", "ann_ord", "end_ord"], kind="mergesort")
    work["amt"] = _executed_increments(work)

    keys = ["ts_code", "ann_date", "ann_ord"]
    grouped = work.groupby(keys, as_index=False)
    events = grouped[["amt", "exec_flag"]].sum()
    limits = grouped["limit_price"].max()
    events = events.merge(limits, on=keys, how="left")
    events = events.sort_values(["ts_code", "ann_ord"], kind="mergesort").reset_index(drop=True)
    return events


def build_repurchase_lookup_by_date(
    df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """按 `ann_date` PIT 对齐到日频的滚动窗口查询表。

    Args:
        df: raw repurchase（逐公告明细行）
        trading_dates: 交易日列表（YYYYMMDD，升序）

    Returns:
        ``{trade_date: DataFrame(ts_code + 内部列 + 哨兵列)}``，**只包含最近 180 自然日内
        有公告的股票**（消费侧需对其余股票显式填 0，见模块 docstring）。
    """
    events = aggregate_repurchase_events(df)
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
        amt_cum = np.concatenate([[0.0], np.cumsum(grp["amt"].to_numpy(dtype="float64"))])
        exec_cum = np.concatenate([[0.0], np.cumsum(grp["exec_flag"].to_numpy(dtype="float64"))])
        ones = np.concatenate([[0.0], np.cumsum(np.ones(len(event_ord), dtype="float64"))])
        amt90 = _window_sums(event_ord, amt_cum, trade_ord_sorted, SHORT_WINDOW_DAYS)
        amt180 = _window_sums(event_ord, amt_cum, trade_ord_sorted, LONG_WINDOW_DAYS)
        exec90 = _window_sums(event_ord, exec_cum, trade_ord_sorted, SHORT_WINDOW_DAYS)
        events_180 = _window_sums(event_ord, ones, trade_ord_sorted, LONG_WINDOW_DAYS)
        active = events_180 > 0
        if not active.any():
            continue

        hi = np.searchsorted(event_ord, trade_ord_sorted, side="right")
        lo = np.searchsorted(event_ord, trade_ord_sorted - LONG_WINDOW_DAYS, side="right")
        # 窗口内**最后一条带价格上限的公告**：逐事件维护"最后一个非空索引"（O(n)，无循环）
        limit = grp["limit_price"].to_numpy(dtype="float64")
        idx = np.arange(len(limit))
        last_valid = np.maximum.accumulate(np.where(~np.isnan(limit), idx, -1))
        cand = np.where(hi > 0, last_valid[np.clip(hi - 1, 0, None)], -1)
        limit_price = np.where(cand >= lo, limit[np.clip(cand, 0, None)], np.nan)

        last_ord = np.where(hi > 0, event_ord[np.clip(hi - 1, 0, None)], -1)
        freshness = np.where(hi > 0, trade_ord_sorted - last_ord, np.nan)

        rows = np.stack(
            [
                amt90,
                amt180,
                (exec90 > 0).astype("float64"),
                limit_price,
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
    all_values = np.concatenate(out_values, axis=0)
    columns = LOOKUP_VALUE_COLS + [REPURCHASE_FRESHNESS_COL]

    sort_idx = np.argsort(all_ord, kind="stable")
    all_ord, all_code, all_values = all_ord[sort_idx], all_code[sort_idx], all_values[sort_idx]
    unique_days, starts = np.unique(all_ord, return_index=True)
    bounds = np.append(starts, len(all_ord))

    lookup: Dict[str, pd.DataFrame] = {}
    for i, day_ord in enumerate(unique_days):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        frame = pd.DataFrame(all_values[lo:hi], columns=columns)
        frame.insert(0, "ts_code", all_code[lo:hi])
        frame[REPURCHASE_VERSION_COL] = np.int8(REPURCHASE_SCHEMA_VERSION)
        position = int(np.searchsorted(trade_ord_sorted, day_ord))
        lookup[trade_dates[position]] = frame
    logger.info(
        f"[repurchase] 因子查询表构建完成: {len(lookup)} 个交易日, "
        f"活跃事件行 {len(all_code)}（已按 180 日窗口收敛）"
    )
    return lookup


def available_repurchase_columns() -> List[str]:
    """因子模块输出的全部列（含哨兵列），用于 schema 与 handler 默认列。"""
    return list(REPURCHASE_COLS) + [REPURCHASE_FRESHNESS_COL, REPURCHASE_VERSION_COL]


def load_repurchase_lookup(loader: Any, trading_dates: List[str]) -> Dict[str, pd.DataFrame]:
    """从 raw 加载 repurchase 并构建按交易日的查询表（运行时派生入口）。

    Args:
        loader: `DataLoader`（或具备 `load_repurchase()` 的等价对象）
        trading_dates: 需要覆盖的交易日（YYYYMMDD，升序）

    Returns:
        ``{trade_date: DataFrame}``；raw 为空时返回空字典（调用方按"无事件"处理）。
    """
    raw = loader.load_repurchase()
    if raw is None or len(raw) == 0:
        logger.warning("[repurchase] raw 为空，运行时派生将全部按 0 处理")
        return {}
    return build_repurchase_lookup_by_date(raw, trading_dates)


def build_repurchase_runtime_lookup(
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
    return load_repurchase_lookup(loader, trading_dates)


def _market_value_yuan(features: pd.DataFrame) -> pd.Series:
    """流通市值（元）：cs_train / cs_infer 的 `circ_mv` 单位为万元，缺列硬报错。"""
    if "circ_mv" not in features.columns:
        raise ValueError(
            "repurchase 因子需要 circ_mv（流通市值，万元）列；缺列禁止静默零因子，"
            "请检查特征分区 schema"
        )
    return pd.to_numeric(features["circ_mv"], errors="coerce") * 10000.0


def _vwap_yuan(features: pd.DataFrame) -> pd.Series:
    """当日 VWAP（元/股）= amount(千元) × 10 ÷ vol(手)；停牌日（vol=0）为 NaN。"""
    for col in ("amount", "vol"):
        if col not in features.columns:
            raise ValueError(
                f"repurchase 因子需要 {col} 列（VWAP = amount × 10 ÷ vol）；"
                "缺列禁止静默零因子，请检查特征分区 schema"
            )
    amount = pd.to_numeric(features["amount"], errors="coerce")  # 千元
    vol = pd.to_numeric(features["vol"], errors="coerce")  # 手
    return (amount * 10.0) / vol.where(vol > 0)


def build_repurchase_feature_frame(
    features: pd.DataFrame,
    merged: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """由"当日特征帧 + 查询表合并结果"算出因子列（**缺失一律填 0**）。

    单一实现来源：handler（构建/推理侧）与 `derive_repurchase_columns`（训练/OOS 侧）共用，
    保证四侧逐值一致。`merged is None` 表示当日无活跃公告 ⇒ 4 个值列全 0。

    Returns:
        DataFrame（index 与 `features` 一致），列 = `REPURCHASE_COLS` + `rp_freshness_days`
        （新鲜度保留 NaN，不参与 0 填充）。
    """
    n = len(features)
    result = pd.DataFrame(index=features.index)
    # 缺列校验无条件执行（即使当日无活跃公告也要显式失败，避免无事件日静默通过）
    denom = _market_value_yuan(features).where(lambda s: s > 0)
    vwap = _vwap_yuan(features)
    if merged is None:
        for col in REPURCHASE_COLS:
            result[col] = np.float32(0.0)
        result[REPURCHASE_FRESHNESS_COL] = np.nan
        return result

    def _source(col: str) -> pd.Series:
        if col in merged.columns:
            return pd.to_numeric(merged[col], errors="coerce")
        return pd.Series(np.nan, index=features.index)

    result["rp_amount_to_mv_90d"] = (_source("rp_amt_90d") / denom).fillna(0.0)
    result["rp_amount_to_mv_180d"] = (_source("rp_amt_180d") / denom).fillna(0.0)
    result["rp_exec_flag_90d"] = _source("rp_exec_90d").fillna(0.0)
    # 回购价上限相对现价的托底空间（无有效上限或价格不可得 ⇒ 0，口径定稿 §6.1 第 7 条）
    result["rp_price_headroom"] = (_source("rp_limit_price") / vwap - 1.0).fillna(0.0)
    result[REPURCHASE_FRESHNESS_COL] = _source(REPURCHASE_FRESHNESS_COL)
    for col in REPURCHASE_COLS:
        result[col] = result[col].astype("float32")
    if n == 0:
        logger.debug("[repurchase] 空特征帧，返回空列")
    return result


def derive_repurchase_columns(
    frame: pd.DataFrame,
    lookup: Optional[Dict[str, pd.DataFrame]],
    wanted: Optional[Iterable[str]] = None,
    log_prefix: str = "",
) -> List[str]:
    """训练/OOS 评估侧就地派生股票回购列（**不写回 cs_train / cs_infer 分区**）。

    与 `available_repurchase_columns()` 同语义来源，复用 `RepurchaseFactorHandler`
    的"窗口外显式填 0 + 哨兵恒写"实现，保证训练侧、OOS 评估侧与推理侧逐值一致。

    规则（与可用性标记 / 股东增减持运行时派生一致）：
    - 需要 `trade_date` 与 `ts_code` 列，缺任一列直接报错（不得静默跳过）；
    - **已存在的列不覆盖**（特征分区若已含本族列，以分区为准）；
    - `wanted` 用于只派生模型实际使用的列（None = 全部）；
    - 查询表缺失的日期（或 lookup 为 None/空）⇒ 4 个值列填 0、freshness 为 NaN、
      哨兵列写当前版本。

    Returns:
        实际新增的列名列表（按 `available_repurchase_columns()` 顺序）。
    """
    from ..features.factor_handlers import RepurchaseFactorHandler

    for col in ("trade_date", "ts_code"):
        if col not in frame.columns:
            raise ValueError(f"股票回购运行时派生缺少必要列: {col}")
    if not frame.index.is_unique:
        raise ValueError("股票回购运行时派生要求输入帧索引唯一（按位置回填）")

    candidates = available_repurchase_columns()
    if wanted is not None:
        wanted_set = set(wanted)
        candidates = [col for col in candidates if col in wanted_set]
    targets = [col for col in candidates if col not in frame.columns]
    if not targets:
        return []

    values = {col: np.full(len(frame), np.nan, dtype="float64") for col in targets}
    handler = RepurchaseFactorHandler()
    lookup = lookup or {}
    for trade_date, day_df in frame.groupby("trade_date", sort=False):
        day_key = normalize_series_to_yyyymmdd(pd.Series([trade_date])).iloc[0]
        day_data = lookup.get(day_key)
        # lookup 为 None/缺该日 ⇒ 传空表（语义 = 窗口内无公告 ⇒ 0 填充），
        # 不能传 None（handler 视 None 为"整族未启用"而返回空字典）
        produced = handler.apply(
            day_df, day_data if day_data is not None else pd.DataFrame(), day_key, pd.DataFrame()
        )
        positions = frame.index.get_indexer(day_df.index)
        for col in targets:
            series = produced.get(col)
            values[col][positions] = (
                series.to_numpy(dtype="float64")
                if series is not None
                else np.full(len(day_df), np.nan)
            )

    derived = pd.DataFrame({col: values[col] for col in targets}, index=frame.index)
    frame[targets] = derived
    if log_prefix:
        logger.info(f"{log_prefix}股票回购运行时派生列: {targets}")
    return targets


def repurchase_coverage_report(lookup: Dict[str, pd.DataFrame]) -> Optional[Tuple[int, float]]:
    """粗略覆盖率自检：返回 (活跃行总数, 单日活跃股票中位数)。"""
    if not lookup:
        return None
    counts = np.array([len(frame) for frame in lookup.values()], dtype="int64")
    return int(counts.sum()), float(np.median(counts))
