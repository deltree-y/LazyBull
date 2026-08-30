"""北向资金因子模块

基于 TuShare moneyflow_hsgt 接口的沪深股通日频资金流数据构造市场级另类因子,
以广播形式写入每只股票 (同一交易日所有 ts_code 共享相同的北向值)。

数据来源：Tushare moneyflow_hsgt（2000 积分）
- hgt: 沪股通当日净买入（百万元）
- sgt: 深股通当日净买入（百万元）
- north_money: 北向资金净流入（百万元）= hgt + sgt
（单位验证: 2014-11-17 沪港通首日 hgt=13000, 即 130 亿元额度当日用尽）

口径切换（2024-08-19 起交易所调整北向披露口径，不再披露实时净买入）：
- 净流入口径（切换前）: hgt/sgt/north_money = 当日净买入（百万元，可为负）
- 成交额口径（切换后）: hgt/sgt/north_money = 当日成交额（百万元，恒正）
适配方式：全程统一 ÷100 换算为亿元；滚动窗口按口径段独立计算，不跨口径切换日；
净买入与成交额拆为两套互斥特征，非所属口径统一置 0，并保留 north_turnover_flag。
这样首次跨制度 OOS 中，旧模型不会把成交额误读为净买入；有新口径训练样本后，模型可独立
学习成交额特征。方向 streak 窗口化为近 20 日，消除加载范围裁剪导致的训练/推理偏差；
z20 段内预热不足时置 0 中性，避免全 NaN 列触发推理侧特征质量门禁拒绝整日预测。

因子说明：
- north_net_buy*: 切换前净买入及其滚动特征；切换后统一为 0
- north_turnover*: 切换后成交额及其滚动特征；切换前统一为 0
- north_net_buy_sign_streak: 近 20 日连续净流入/流出方向，切换后为 0
- north_turnover_change_streak: 近 20 日连续放量/缩量方向，切换前为 0
- north_turnover_flag: 口径指示（0=净流入口径, 1=成交额口径），供模型区分跨口径样本
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd

NORTH_NET_BUY_COLS = [
    "north_net_buy",
    "north_net_buy_ma5",
    "north_net_buy_ma20",
    "north_net_buy_z20",
    "north_net_buy_sum5",
    "north_net_buy_sign_streak",
]

NORTH_TURNOVER_COLS = [
    "north_turnover",
    "north_turnover_ma5",
    "north_turnover_ma20",
    "north_turnover_z20",
    "north_turnover_sum5",
    "north_turnover_change_streak",
]

NORTH_COLS = NORTH_NET_BUY_COLS + NORTH_TURNOVER_COLS + ["north_turnover_flag"]

# 北向披露口径切换日期：2024-08-19 起交易所停止披露实时净买入，改为披露成交额
NORTH_TURNOVER_SWITCH_DATE = "20240819"

# 单位换算系数：moneyflow_hsgt 全程单位为百万元（含切换前净买入），统一 -> 亿元
_NORTH_TURNOVER_SCALE = 100.0


def _compute_sign_streak(series: pd.Series, window: int = 20) -> pd.Series:
    """计算近 window 日窗口内连续同方向天数 (正=连续流入, 负=连续流出, 0=本日净零)。

    窗口化约束: streak 为无限累积量时, 加载范围（如推理侧近 40 日裁剪）会改变
    累积起点导致训练/推理偏差; 限制为固定窗口后仅依赖最近 window 日, 消除偏差。
    """
    sign = np.sign(series.fillna(0.0))
    n = len(sign)
    streak = np.zeros(n, dtype=float)
    values = sign.values
    for i in range(n):
        s = values[i]
        if s == 0:
            continue
        total = 0.0
        for j in range(i, max(-1, i - window), -1):
            if values[j] == s:
                total += s
            else:
                break
        streak[i] = total
    return pd.Series(streak, index=series.index)


def build_north_flow_lookup_by_date(
    hsgt_df: pd.DataFrame,
    trading_dates: List[str],
    calendar_dates: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """构建北向资金市场级因子查询表

    与其他因子不同, 北向是市场级数据 (一天一条), 因此 lookup 的 value
    是 `Dict[str, float]` (列名 -> 当日值), FeatureBuilder 合并时对
    全部 ts_code 广播同一份值。

    Args:
        hsgt_df: moneyflow_hsgt 原始 DataFrame, 需含 trade_date, north_money
        trading_dates: 需要输出的交易日列表 (YYYYMMDD 字符串)
        calendar_dates: 滚动计算使用的完整 A 股交易日历，默认与输出日期相同

    Returns:
        Dict[trade_date -> Dict[col_name -> float]]
    """
    if hsgt_df is None or len(hsgt_df) == 0:
        logger.warning("北向资金因子: 输入数据为空")
        return {}

    df = hsgt_df.copy()
    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

    # 统一原始金额列；接口在切换日前后分别表示净买入与成交额
    if "north_money" in df.columns:
        df["_north_amount"] = pd.to_numeric(df["north_money"], errors="coerce")
    elif "hgt" in df.columns and "sgt" in df.columns:
        hgt_amount = pd.to_numeric(df["hgt"], errors="coerce").fillna(0.0)
        sgt_amount = pd.to_numeric(df["sgt"], errors="coerce").fillna(0.0)
        df["_north_amount"] = hgt_amount + sgt_amount
    else:
        logger.warning("北向资金因子: 缺少 north_money/hgt/sgt 列")
        return {}

    df = df.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)

    # 港股休市但 A 股开市时接口无记录。内部休市金额按 0 进入后续滚动窗口；
    # 源数据末尾之后不补值，避免掩盖下载滞后或接口故障。
    source_start = str(df["trade_date"].min())
    source_end = str(df["trade_date"].max())
    source_dates = set(df["trade_date"])
    calculation_dates = calendar_dates if calendar_dates is not None else trading_dates
    neutralized_dates = sorted(
        {
            trade_date
            for trade_date in calculation_dates
            if source_start <= trade_date <= source_end and trade_date not in source_dates
        }
    )
    df["_is_internal_holiday"] = False
    if neutralized_dates:
        holiday_rows = pd.DataFrame(
            {
                "trade_date": neutralized_dates,
                "_north_amount": 0.0,
                "_is_internal_holiday": True,
            }
        )
        df = pd.concat([df, holiday_rows], ignore_index=True, sort=False)
        df = df.sort_values("trade_date").reset_index(drop=True)

    # 口径分段：0=净流入口径（切换日前），1=成交额口径（切换日起）
    df["_era"] = (df["trade_date"] >= NORTH_TURNOVER_SWITCH_DATE).astype(int)

    # 单位换算：moneyflow_hsgt 全程单位为百万元（切换前后一致），统一 ÷100 换算亿元
    df["_north_amount"] = df["_north_amount"] / _NORTH_TURNOVER_SCALE

    # 滚动特征：按口径段独立计算，窗口不跨口径切换日
    df["_amount_ma5"] = df.groupby("_era")["_north_amount"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    df["_amount_ma20"] = df.groupby("_era")["_north_amount"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )
    roll_std = df.groupby("_era")["_north_amount"].transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    # 段内样本不足（如口径切换后前 4 日）无 z 信息, 置 0 中性: 全零列不触发
    # 推理侧特征质量门禁硬拒绝（north_ 前缀豁免警告）
    df["_amount_z20"] = (
        (df["_north_amount"] - df["_amount_ma20"]) / roll_std.replace(0, np.nan)
    ).fillna(0.0)
    df["_amount_sum5"] = df.groupby("_era")["_north_amount"].transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )

    # 口径指示列：0=净流入口径, 1=成交额口径。同列跨口径语义由该列显式区分,
    # 供模型学习口径条件分支, 避免语义断裂被模型误读为数值变化。
    df["north_turnover_flag"] = df["_era"].astype(float)

    # 方向序列：净流入口径=净流入符号；成交额口径=成交额环比方向（放量/缩量）。
    # 窗口化为近 20 日（见 _compute_sign_streak），不受加载范围裁剪影响。
    flow_dir = df["_north_amount"].fillna(0.0)
    turnover_dir = flow_dir.groupby(df["_era"]).diff().fillna(0.0)
    df["_dir"] = np.where(df["_era"].values == 1, np.sign(turnover_dir), np.sign(flow_dir))
    df["_amount_streak"] = df.groupby("_era")["_dir"].transform(
        lambda s: _compute_sign_streak(s, window=20)
    )

    # 口径交互编码：每套因子只在自身口径激活，另一口径置 0。
    # 训练仅含旧口径时，成交额列会作为常数自然移除；切换后旧列输入中性 0，
    # 从而避免把成交额数值送入净买入语义的树分支。
    amount_sources = [
        "_north_amount",
        "_amount_ma5",
        "_amount_ma20",
        "_amount_z20",
        "_amount_sum5",
        "_amount_streak",
    ]
    is_turnover = df["_era"].eq(1)
    for net_col, turnover_col, source_col in zip(
        NORTH_NET_BUY_COLS, NORTH_TURNOVER_COLS, amount_sources
    ):
        df[net_col] = df[source_col].where(~is_turnover, 0.0)
        df[turnover_col] = df[source_col].where(is_turnover, 0.0)

    date_set = set(trading_dates)
    result: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        td = row["trade_date"]
        if td not in date_set:
            continue
        if row["_is_internal_holiday"]:
            rec = {column: 0.0 for column in NORTH_COLS}
            rec["north_turnover_flag"] = float(td >= NORTH_TURNOVER_SWITCH_DATE)
        else:
            rec = {}
            for col in NORTH_COLS:
                val = row.get(col)
                rec[col] = float(val) if pd.notna(val) else np.nan
        result[td] = rec

    source_mask = ~df["_is_internal_holiday"]
    pre_n = int((source_mask & df["_era"].eq(0)).sum())
    post_n = int((source_mask & df["_era"].eq(1)).sum())
    logger.info(
        f"北向资金因子查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日; "
        f"口径分段: 净流入 {pre_n} 日 / 成交额 {post_n} 日"
        f"，内部休市中性化 {len(neutralized_dates)} 日"
        f"（全程百万元 ÷{_NORTH_TURNOVER_SCALE:.0f} 统一换算亿元, {NORTH_TURNOVER_SWITCH_DATE} 起切换口径）"
    )
    return result
