"""龙虎榜因子模块

基于 TuShare top_list 接口的个股上榜数据构造截面特征。

数据来源：Tushare top_list（2000 积分）
- trade_date, ts_code, l_buy, l_sell, l_amount, net_amount, net_rate
- amount_rate: 龙虎榜成交额占总成交额比
- turnover_rate, float_values, reason

因子说明：
- lhb_on_list: 当日是否上榜 (0/1)
- lhb_net_amount: 当日龙虎榜净买入额（元）
- lhb_net_rate: 当日净买入占流通市值比
- lhb_amount_rate: 龙虎榜成交占比
- lhb_up_days_20: 近 20 日累计上榜次数
- lhb_net_sum_5 / lhb_net_sum_20: 近 5/20 日净买入累计
- lhb_reason_count: 当日上榜理由数（同一股票同日可有多条）
- lhb_cont_on_list: 当日是否因"连续异动"类理由上榜 (0/1)
  （reason 含"连续", 如连续三个交易日涨幅偏离累计达 20%; 事件级信号,
    与选中主记录无关, 当日任一条 reason 含"连续"即记为 1）

注: top_list 同一 (trade_date, ts_code) 可能出现多条记录
(不同上榜理由), 需先做 groupby 聚合。
"""

import bisect
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd


LHB_COLS = [
    "lhb_on_list",
    "lhb_net_amount",
    "lhb_net_rate",
    "lhb_amount_rate",
    "lhb_up_days_20",
    "lhb_net_sum_5",
    "lhb_net_sum_20",
    "lhb_reason_count",
    "lhb_cont_on_list",
]


def build_lhb_lookup_by_date(
    top_list_df: pd.DataFrame,
    trading_dates: List[str],
    calendar_dates: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """构建龙虎榜日频查询表（按交易日重采样，保时序连续性）

    修复要点（2026-08 审计）：
    1. 同日同股票多条记录优先"单日榜"（排除"连续 N 个交易日"类重叠周期）；
       但若当日只有连续类理由则保留（不能误标为未上榜）；组内取净买入绝对值
       最大的一条（同周期重复行，禁止 sum；净额全 NaN 组取第一条防崩溃）；
    2. rolling 按完整交易日历重采样（未上榜日补 0），使 lhb_up_days_20 /
       lhb_net_sum_5 / lhb_net_sum_20 真正表示"近 5/20 个交易日"的累计；
    3. 每个交易日输出所有"近 20 个交易日内上过榜"的股票（含当日未上榜者），
       历史累计不会在次日凭空消失，保证时序连续性。

    Args:
        top_list_df: top_list 原始 DataFrame, 至少含 trade_date, ts_code,
                    net_amount (部分日期可能缺失)
        trading_dates: 需要输出的交易日列表 (YYYYMMDD 字符串)
        calendar_dates: 用于滚动重采样的完整交易日历 (YYYYMMDD, 已排序)。
            默认取 trading_dates。单日推断（纸面交易）应传入包含历史窗口的
            完整日历，否则滚动窗口无法覆盖历史。

    Returns:
        Dict[trade_date -> DataFrame(ts_code, lhb_*)]

    说明: 未上榜且近 20 日也无记录的个股不会在 result 中出现, 由
    FeatureBuilder 合并时以 left-join 方式保留 NaN, 再在基础特征里用 0 填充。
    """
    if top_list_df is None or len(top_list_df) == 0:
        logger.warning("龙虎榜因子: 输入数据为空")
        return {}

    df = top_list_df.copy()
    if "ts_code" not in df.columns:
        logger.warning("龙虎榜因子: 数据缺少 ts_code 列, 跳过")
        return {}
    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

    # 1. 同日同股票多条记录的选择逻辑（修复 2026-08 二轮审计）:
    #    - 优先"单日榜"记录（排除"连续 N 个交易日"类重叠周期）;
    #    - 若当日只有连续类理由（如仅触发连续异动）, 必须保留, 不能误标为未上榜;
    #    - 组内取净买入绝对值最大的一条（同周期重复行, 禁止 sum）;
    #    - 净买入全为 NaN 的组取第一条, 避免 idxmax 返回 NaN 索引崩溃。
    if "reason" in df.columns:
        df["_lhb_is_cont"] = df["reason"].astype(str).str.contains("连续", na=False)
    else:
        df["_lhb_is_cont"] = False

    for col in ["net_amount", "net_rate", "amount_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "net_amount" in df.columns:
        df["_lhb_abs"] = df["net_amount"].abs()
        sort_cols = ["trade_date", "ts_code", "_lhb_is_cont", "_lhb_abs"]
        sort_asc = [True, True, True, False]
    else:
        sort_cols = ["trade_date", "ts_code", "_lhb_is_cont"]
        sort_asc = [True, True, True]
    # 排序: 按 (trade_date, ts_code) 保证同组记录连续, 非连续优先 + 组内净买入
    # 绝对值最大在前。排序后 drop_duplicates 保留第一整行（不能用 groupby.first(),
    # 它会逐列取首个非空值, 把不同记录的字段拼接成不存在的行）。
    df = df.sort_values(sort_cols, ascending=sort_asc, na_position="last")
    grouped = df.drop_duplicates(subset=["trade_date", "ts_code"], keep="first").copy()

    # 当日去重后的上榜理由数（与选中类别一致: 选中单日类则计单日类去重数,
    # 仅连续类时计全组去重数）
    grouped["lhb_reason_count"] = 1.0
    if "reason" in df.columns:
        daily_mask = ~df["_lhb_is_cont"]
        rc_daily = df[daily_mask].groupby(["trade_date", "ts_code"])["reason"].nunique().to_dict()
        rc_all = df.groupby(["trade_date", "ts_code"])["reason"].nunique().to_dict()
        grouped["_rc"] = [
            rc_daily.get((td, tc)) if not cont else rc_all.get((td, tc))
            for td, tc, cont in zip(
                grouped["trade_date"], grouped["ts_code"], grouped["_lhb_is_cont"]
            )
        ]
        grouped["lhb_reason_count"] = grouped["_rc"].fillna(1.0)
        grouped = grouped.drop(columns=["_rc"])

    # lhb_cont_on_list: 当日是否存在"连续异动"类上榜理由（事件级信号, 与选中
    # 主记录无关; 只要当日任一条 reason 含"连续"即记为 1）
    if "reason" in df.columns:
        cont_any = (
            df.groupby(["trade_date", "ts_code"])["_lhb_is_cont"]
            .max()
            .rename("_lhb_cont_any")
            .reset_index()
        )
        grouped = grouped.merge(cont_any, on=["trade_date", "ts_code"], how="left")
        grouped["lhb_cont_on_list"] = grouped["_lhb_cont_any"].fillna(False).astype(float)
        grouped = grouped.drop(columns=["_lhb_cont_any"])
    else:
        grouped["lhb_cont_on_list"] = 0.0

    grouped = grouped.drop(columns=["_lhb_is_cont", "_lhb_abs"], errors="ignore")
    grouped = grouped.rename(
        columns={
            "net_amount": "lhb_net_amount",
            "net_rate": "lhb_net_rate",
            "amount_rate": "lhb_amount_rate",
        }
    )
    grouped["lhb_on_list"] = 1.0

    # 3. 按完整交易日历重采样后滚动（未上榜日补 0）
    calendar = sorted(set(calendar_dates)) if calendar_dates else sorted(set(trading_dates))
    output_set = set(trading_dates)
    has_net = "lhb_net_amount" in grouped.columns

    grouped = grouped.sort_values(["ts_code", "trade_date"])
    result_rows: Dict[str, List[dict]] = {}
    for code, sub in grouped.groupby("ts_code", sort=False):
        sub = sub.sort_values("trade_date")
        i0 = bisect.bisect_left(calendar, sub["trade_date"].iloc[0])
        i1 = bisect.bisect_right(calendar, sub["trade_date"].iloc[-1])
        end = min(len(calendar), i1 + 20)  # 最后上榜日后 20 个交易日窗口
        if i0 >= end:
            continue
        daily = sub.set_index("trade_date").reindex(calendar[i0:end])
        for col in [
            "lhb_on_list",
            "lhb_net_amount",
            "lhb_net_rate",
            "lhb_amount_rate",
            "lhb_reason_count",
            "lhb_cont_on_list",
        ]:
            if col in daily.columns:
                daily[col] = daily[col].fillna(0.0)
        daily["lhb_up_days_20"] = daily["lhb_on_list"].rolling(20, min_periods=1).sum()
        if has_net:
            daily["lhb_net_sum_5"] = daily["lhb_net_amount"].rolling(5, min_periods=1).sum()
            daily["lhb_net_sum_20"] = daily["lhb_net_amount"].rolling(20, min_periods=1).sum()

        # 4. 只保留"近 20 个交易日内上过榜"的日期（时序连续性）
        keep = daily[daily["lhb_up_days_20"] > 0]
        for td, row in keep.iterrows():
            if td not in output_set:
                continue
            rec = {"ts_code": code}
            for col in LHB_COLS:
                if col != "ts_code":
                    rec[col] = float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
            result_rows.setdefault(td, []).append(rec)

    result: Dict[str, pd.DataFrame] = {}
    for td, rows in result_rows.items():
        if rows:
            result[td] = pd.DataFrame(rows)[["ts_code"] + LHB_COLS].reset_index(drop=True)

    logger.info(f"龙虎榜因子查询表: 覆盖 {len(result)}/{len(output_set)} 个输出交易日")
    return result
