# -*- coding: utf-8 -*-
"""龙虎榜机构席位因子模块（top_inst）。

数据来源：TuShare `top_inst`，raw 按 `trade_date` 年分区落盘
（`data/raw/top_inst/YYYY-12-31.parquet`）。数据契约与全库清洗验证见
`docs/top_inst_factor_health.md`（§2 存储布局 / §3 清洗口径 / §4 体检结果）。

**数据源结构缺陷与清洗口径（`clean_top_inst`，体检报告 §3 全库实测）**：

- S1 列互换修正：数据源部分年份 `buy`/`sell` 列互换（2013–2018 年 30%~75% 行、
  2019 年 30% 行）——判定 `|sell−buy−net_buy| < 1 且 |buy−sell−net_buy| ≥ 1` 时交换；
  **`net_buy` 是唯一可靠锚**（新旧文本格式行净额一致）；
- S2 去重：同一事实在多榜单 / 新旧文本格式下重复披露——按
  `(ts_code, trade_date, exalter, side, 修正后 buy, sell, net_buy)` 去重（**禁止按行求和**）；
- S3 单日榜优先：`reason` 含“连续/累计”的行为 N 日窗口口径，禁止与单日榜混用——
  剔除连续榜行，仅当该股票日无单日榜行时保留（对齐 `factors/lhb.py` 先例）。

**机构聚合**：`exalter == "机构专用"` 精确匹配 → 按 (trade_date, ts_code) 求 sum
（同榜单下多行 = 多家机构分列披露，sum 为机构合计；文本重复行已在 S2 去重）。

**因子（最小列集，探索期实验用）**：5 值列 + 哨兵 `ti_schema_v1`——
`ti_inst_net_ratio`（当日机构净买 ÷ 流通市值）、`ti_inst_buy_ratio`、
`ti_inst_sell_ratio`、`ti_inst_net_sum_20`（近 20 交易日净买累计 ÷ 流通市值）、
`ti_inst_days_20`（近 20 交易日机构出现天数）。窗口一律 **20 交易日**（滚动）。

**填充语义（沿 holdertrade / repurchase 事件族先例）**：查询表只输出「近 20 交易日有
机构记录」的股票；消费侧其余股票显式填 0（语义 = 窗口内无机构事件）。
流通市值归一化在特征帧侧完成（`build_top_inst_feature_frame`），`circ_mv` 缺列硬报错。

**接线 = 运行时派生**（训练 / OOS 评估 / OOS 回测侧；纸面侧待实验放行后另行接线），
**不写回 cs_train / cs_infer**。
"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd

#: 滚动窗口（交易日）——与体检探索口径一致（2026-09-23）
TOP_INST_WINDOW_DAYS = 20

#: 机构席位名称（精确匹配，禁止模糊子串）
TOP_INST_SEAT_NAME = "机构专用"

#: 最终因子值列（特征帧输出；由 handler / derive 消费）
TOP_INST_NET_RATIO_COL = "ti_inst_net_ratio"
TOP_INST_BUY_RATIO_COL = "ti_inst_buy_ratio"
TOP_INST_SELL_RATIO_COL = "ti_inst_sell_ratio"
TOP_INST_NET_SUM_20_COL = "ti_inst_net_sum_20"
TOP_INST_DAYS_20_COL = "ti_inst_days_20"

TOP_INST_COLS = [
    TOP_INST_NET_RATIO_COL,
    TOP_INST_BUY_RATIO_COL,
    TOP_INST_SELL_RATIO_COL,
    TOP_INST_NET_SUM_20_COL,
    TOP_INST_DAYS_20_COL,
]

#: schema 哨兵列与当前版本（语义重做时递增；训练入口校验）
TOP_INST_VERSION_COL = "ti_schema_v1"
TOP_INST_SCHEMA_VERSION = 1

#: 查询表内部列（金额原值，未归一化；归一化需特征帧 circ_mv）
_LOOKUP_COLS = [
    "ti_inst_buy_amt",
    "ti_inst_sell_amt",
    "ti_inst_net_amt",
    "ti_inst_net_sum_20_amt",
    "ti_inst_days_20",
]

#: 清洗必需列（raw 契约列；缺列直接报错，不得静默降级）
_REQUIRED_RAW_COLS = ["ts_code", "trade_date", "exalter", "side", "buy", "sell", "net_buy", "reason"]


def clean_top_inst(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 raw top_inst（S1 列互换修正 → S2 三元组去重 → S3 单日榜优先）。

    口径为体检全库验证版（`docs/top_inst_factor_health.md` §3.4）：
    清洗后 `sum(net_buy)`（股票日）与 `top_list` 股票级 `net_amount` 中位相对误差 0.0%。

    Args:
        df: raw top_inst（至少含 `_REQUIRED_RAW_COLS`）

    Returns:
        清洗后的明细行（保留原始行列；`buy`/`sell` 已按 `net_buy` 锚修正）。

    Raises:
        ValueError: 缺少必需列（禁止静默降级）。
    """
    missing = [c for c in _REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"top_inst raw 缺少必需列: {missing}（不得静默降级清洗）")
    if len(df) == 0:
        return df.copy()

    d = df.copy()
    for c in ("buy", "sell", "net_buy"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # S1：以 net_buy 为锚修正列互换
    resid = (d["buy"] - d["sell"] - d["net_buy"]).abs()
    swap = ((d["sell"] - d["buy"] - d["net_buy"]).abs() < 1.0) & (resid >= 1.0)
    d["buy"], d["sell"] = (
        d["buy"].where(~swap, d["sell"]),
        d["sell"].where(~swap, d["buy"]),
    )

    # S2：同一事实的重复披露（多榜单 / 新旧文本格式）——按修正后金额三元组去重
    d = d.drop_duplicates(
        subset=["ts_code", "trade_date", "exalter", "side", "buy", "sell", "net_buy"]
    )

    # S3：单日榜优先（连续/累计类为 N 日窗口口径，禁止与单日榜混用）
    cont = (
        d["reason"].astype(str).str.replace(" ", "", regex=False).str.contains(
            "连续|累计", regex=True, na=False
        )
    )
    any_daily = (~cont).groupby([d["ts_code"], d["trade_date"]]).transform("any")
    d = d[(~cont) | (~any_daily)]
    return d.reset_index(drop=True)


def aggregate_inst_daily(clean_df: pd.DataFrame) -> pd.DataFrame:
    """机构专用行 → 日表 `(trade_date, ts_code)`（sum 多机构明细）。

    Returns:
        ``DataFrame(trade_date, ts_code, inst_buy, inst_sell, inst_net, inst_rows)``。
        无机构记录时返回空表（列齐）。
    """
    cols = ["trade_date", "ts_code", "inst_buy", "inst_sell", "inst_net", "inst_rows"]
    if clean_df is None or len(clean_df) == 0:
        return pd.DataFrame(columns=cols)
    inst = clean_df[
        clean_df["exalter"].astype(str).str.strip() == TOP_INST_SEAT_NAME
    ]
    if len(inst) == 0:
        return pd.DataFrame(columns=cols)
    g = (
        inst.groupby(["trade_date", "ts_code"], sort=False)
        .agg(
            inst_buy=("buy", "sum"),
            inst_sell=("sell", "sum"),
            inst_net=("net_buy", "sum"),
            inst_rows=("buy", "size"),
        )
        .reset_index()
    )
    g["trade_date"] = normalize_series_to_yyyymmdd(g["trade_date"])
    return g


def build_top_inst_lookup_by_date(
    inst_daily: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """按交易日构建 20 交易日滚动窗口查询表（逐值口径见模块 docstring）。

    对每个交易日 T，输出「窗口 `[T−19, T]`（含）内有机构记录」的股票：
    - 当日值：窗口末日（T）有记录则取值，否则 0；
    - `ti_inst_net_sum_20_amt`：窗口内净额累计；`ti_inst_days_20`：窗口内出现天数
      （日表每股每日至多一行，故天数 = 记录条数）。

    Args:
        inst_daily: `aggregate_inst_daily` 输出（稀疏日表）
        trading_dates: 交易日列表（YYYYMMDD，升序）

    Returns:
        ``{trade_date: DataFrame(ts_code + 内部金额列)}``；为空时返回 {}。
    """
    if inst_daily is None or len(inst_daily) == 0 or not trading_dates:
        return {}
    trade_dates = [normalize_series_to_yyyymmdd(pd.Series([d])).iloc[0] for d in trading_dates]
    if any(not d or len(d) != 8 for d in trade_dates):
        raise ValueError("trading_dates 含非法日期（应为 YYYYMMDD）")
    day_index = {d: i for i, d in enumerate(trade_dates)}
    n_days = len(trade_dates)

    work = inst_daily.copy()
    work["trade_date"] = normalize_series_to_yyyymmdd(work["trade_date"])
    # 先按请求区间预过滤（全库日表传入时区间外记录属正常，debug 级）
    in_range = (work["trade_date"] >= trade_dates[0]) & (work["trade_date"] <= trade_dates[-1])
    out_of_range = int((~in_range).sum())
    if out_of_range:
        logger.debug(
            f"[top_inst] 剔除 {out_of_range} 行请求区间外记录"
            f"（{trade_dates[0]}~{trade_dates[-1]} 之外，正常）"
        )
    work = work[in_range].copy()
    work["_di"] = work["trade_date"].map(day_index)
    miss = work["_di"].isna()
    if miss.any():
        logger.warning(
            f"[top_inst] {int(miss.sum())} 行机构日表记录的日期在请求区间内但不在交易日序列（剔除）"
        )
        work = work[~miss]
    if len(work) == 0:
        return {}

    out_days: List[int] = []
    out_codes: List[str] = []
    out_rows: List[tuple] = []
    win = TOP_INST_WINDOW_DAYS

    for code, grp in work.groupby("ts_code", sort=False):
        grp = grp.sort_values("_di")
        di = grp["_di"].to_numpy(dtype="int64")
        buy = grp["inst_buy"].to_numpy(dtype="float64")
        sell = grp["inst_sell"].to_numpy(dtype="float64")
        net = grp["inst_net"].to_numpy(dtype="float64")
        cum = np.concatenate([[0.0], np.cumsum(net)])
        first, last = int(di[0]), int(di[-1])
        stop = min(last + win, n_days)
        for t in range(first, stop):
            j = int(np.searchsorted(di, t, side="right"))
            i = int(np.searchsorted(di, t - win + 1, side="left"))
            days = j - i
            if days == 0:
                continue
            net_sum = float(cum[j] - cum[i])
            if j > 0 and int(di[j - 1]) == t:
                b_t, s_t, n_t = float(buy[j - 1]), float(sell[j - 1]), float(net[j - 1])
            else:
                b_t = s_t = n_t = 0.0
            out_days.append(t)
            out_codes.append(str(code))
            out_rows.append((b_t, s_t, n_t, net_sum, float(days)))

    if not out_days:
        return {}
    all_vals = np.asarray(out_rows, dtype="float64")
    frame = pd.DataFrame(all_vals, columns=_LOOKUP_COLS)
    frame.insert(0, "ts_code", out_codes)
    frame.insert(0, "_di", out_days)

    lookup: Dict[str, pd.DataFrame] = {}
    for t, g in frame.groupby("_di", sort=False):
        lookup[trade_dates[int(t)]] = g.drop(columns=["_di"]).reset_index(drop=True)
    logger.info(
        f"[top_inst] 因子查询表构建完成: {len(lookup)} 个交易日, "
        f"活跃股票日 {len(frame)}（20 交易日窗口收敛）"
    )
    return lookup


def _mv_denominator(features: pd.DataFrame) -> np.ndarray:
    """流通市值（元）：cs_train / cs_infer 的 `circ_mv` 单位为万元，缺列硬报错。"""
    if "circ_mv" not in features.columns:
        raise ValueError(
            "top_inst 因子需要 circ_mv（流通市值，万元）列；缺列禁止静默零因子，"
            "请检查特征帧列集"
        )
    return pd.to_numeric(features["circ_mv"], errors="coerce").to_numpy(dtype="float64") * 10000.0


def build_top_inst_feature_frame(
    features: pd.DataFrame,
    merged: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """由（特征帧 × 查询表）merge 结果构建最终 5 列（市值归一 + 0 填充）。

    `merged` 为 None 或缺少某列 ⇒ 该列按 0 填充（语义 = 窗口内无机构事件）。
    市值无效（NaN/≤0）时比率列置 0（不引入新缺失，保持全市场覆盖）。
    """
    n = len(features)
    mv = _mv_denominator(features)
    valid = np.isfinite(mv) & (mv > 0)

    def source(col: str) -> np.ndarray:
        if merged is not None and col in merged.columns:
            return pd.to_numeric(merged[col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        return np.zeros(n, dtype="float64")

    def ratio(amt: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype="float64")
        np.divide(amt, mv, out=out, where=valid)
        return out

    result = pd.DataFrame(index=features.index)
    result[TOP_INST_NET_RATIO_COL] = ratio(source("ti_inst_net_amt"))
    result[TOP_INST_BUY_RATIO_COL] = ratio(source("ti_inst_buy_amt"))
    result[TOP_INST_SELL_RATIO_COL] = ratio(source("ti_inst_sell_amt"))
    result[TOP_INST_NET_SUM_20_COL] = ratio(source("ti_inst_net_sum_20_amt"))
    result[TOP_INST_DAYS_20_COL] = source("ti_inst_days_20")
    return result


def available_top_inst_columns() -> List[str]:
    """因子模块输出的全部列（含哨兵列），用于 schema 与 handler 默认列。"""
    return list(TOP_INST_COLS) + [TOP_INST_VERSION_COL]


def top_inst_feature_columns() -> List[str]:
    """训练/派生使用的列清单（含哨兵列；与 `ml/train_core/constants.py` 双处单一来源）。"""
    return available_top_inst_columns()


def load_top_inst_lookup(
    loader: Any,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """从 raw 加载 top_inst 并构建按交易日的查询表（运行时派生入口）。

    Args:
        loader: `DataLoader`（或具备 `load_top_inst()` 的等价对象）
        trading_dates: 需要覆盖的交易日（YYYYMMDD，升序）

    Returns:
        ``{trade_date: DataFrame}``；raw 为空时返回空字典（调用方按“无事件”处理）。
    """
    raw = loader.load_top_inst()
    if raw is None or len(raw) == 0:
        logger.warning("[top_inst] raw 为空，运行时派生将全部按 0 处理")
        return {}
    daily = aggregate_inst_daily(clean_top_inst(raw))
    return build_top_inst_lookup_by_date(daily, trading_dates)


def build_top_inst_runtime_lookup(
    loader: Any,
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """按日期区间构建运行时查询表（YYYYMMDD；交易日由 loader 解析）。

    供训练 / OOS 评估 / OOS 回测入口一次性构建（全折共用同一张表）。
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
    return load_top_inst_lookup(loader, trading_dates)


def derive_top_inst_columns(
    frame: pd.DataFrame,
    lookup: Optional[Dict[str, pd.DataFrame]],
    wanted: Optional[Iterable[str]] = None,
    log_prefix: str = "",
) -> List[str]:
    """训练 / OOS 评估 / OOS 回测侧就地派生长虎榜机构席位列（**不写回特征分区**）。

    与 `available_top_inst_columns()` 同语义来源，复用 `TopInstFactorHandler`
    的“窗口外显式填 0 + 哨兵恒写 + 市值归一”实现，保证各消费侧逐值一致。

    规则（与 holdertrade / repurchase 运行时派生一致）：
    - 需要 `trade_date` 与 `ts_code` 列，缺任一列直接报错（不得静默跳过）；
    - **已存在的列不覆盖**（特征分区若已含本族列，以分区为准）；
    - `wanted` 用于只派生模型实际使用的列（None = 全部）；
    - 查询表缺失的日期（或 lookup 为 None/空）⇒ 5 个值列填 0、哨兵列写当前版本。

    Returns:
        实际新增的列名列表（按 `available_top_inst_columns()` 顺序）。
    """
    # 延迟导入：features.factor_handlers 依赖本模块，避免顶层循环导入
    from ..features.factor_handlers import TopInstFactorHandler

    for col in ("trade_date", "ts_code"):
        if col not in frame.columns:
            raise ValueError(f"top_inst 运行时派生缺少必要列: {col}")
    if not frame.index.is_unique:
        raise ValueError("top_inst 运行时派生要求输入帧索引唯一（按位置回填）")

    candidates = available_top_inst_columns()
    if wanted is not None:
        wanted_set = set(wanted)
        candidates = [col for col in candidates if col in wanted_set]
    targets = [col for col in candidates if col not in frame.columns]
    if not targets:
        return []

    values = {col: np.full(len(frame), np.nan, dtype="float64") for col in targets}
    handler = TopInstFactorHandler()
    lookup = lookup or {}
    for trade_date, day_df in frame.groupby("trade_date", sort=False):
        day_key = normalize_series_to_yyyymmdd(pd.Series([trade_date])).iloc[0]
        day_data = lookup.get(day_key)
        # lookup 缺该日 ⇒ 传空表（语义 = 窗口内无事件 ⇒ 0 填充），
        # 不能传 None（handler 视 None 为“整族未启用”而返回空字典）
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
        # 批量赋值（逐列赋值会让数百列的特征帧高度碎片化）
        derived = pd.DataFrame({col: values[col] for col in targets}, index=frame.index)
        frame[targets] = derived
    if log_prefix:
        logger.info(f"{log_prefix}top_inst 运行时派生列: {targets}")
    return targets
