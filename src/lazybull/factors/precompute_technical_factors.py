"""技术指标与波动率批量预计算模块

批量计算全量日线数据中的技术指标与波动率，供 FeatureBuilder 实例级缓存复用，
避免按日切片重复计算。

典型用法（FeatureBuilder 内部）：
    # 首次调用时触发预计算
    tech_all = precompute_technical_factors(daily_adj, vol_windows=[5, 10, 20])
    # 后续每日仅做 O(1) 查表
    tech_today = tech_all[tech_all['trade_date'] == trade_date]
"""

import time
from typing import List, Optional

import pandas as pd
from loguru import logger

from .returns import compute_ret_1
from .technical_indicators import (
    calculate_bollinger_bands,
    calculate_kdj,
    calculate_macd,
    calculate_rsi,
)
from .volatility import calculate_atr, calculate_volatility

# 技术指标所需最小历史天数（与 builder._add_advanced_factors 保持一致）
_MIN_HIST_DAYS_FOR_TECH = 30


def precompute_technical_factors(
    daily_adj: pd.DataFrame,
    vol_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """批量预计算技术指标与波动率因子

    对全量 daily_adj 一次性计算 RSI(14)、KDJ(9,3,3)、MACD(12,26,9)、
    布林带(20,2) 以及多窗口滚动波动率，输出一张以 (ts_code, trade_date)
    为键的宽表，供 FeatureBuilder 按日查表。

    Args:
        daily_adj: 全量后复权日线数据，需包含 ts_code、trade_date、
                   close_adj，以及 high_adj/low_adj（KDJ 所需），
                   pct_chg 或 ret_1（波动率所需）
        vol_windows: 波动率滚动窗口列表，默认 [5, 10, 20]

    Returns:
        宽表 DataFrame，包含列：
        ts_code、trade_date、rsi_14、kdj_k、kdj_d、kdj_j、
        macd_dif、macd_dea、macd_hist、bb_middle、bb_upper、bb_lower、
        bb_width、bb_pct、volatility_5（及其他 vol_windows）、atr_14
        无法计算的列保留 NaN。
    """
    if vol_windows is None:
        vol_windows = [5, 10, 20]

    if daily_adj is None or len(daily_adj) == 0:
        logger.warning("precompute_technical_factors: daily_adj 为空，返回空 DataFrame")
        return pd.DataFrame(columns=['ts_code', 'trade_date'])

    t0 = time.time()
    logger.info("开始批量预计算技术指标与波动率因子...")

    # ---- 步骤 1：仅排序一次，后续各函数内部不再重复全局排序 ----
    # 各计算函数内部会自行排序（sort_values 调用幂等，无额外副本风险）
    # 这里只确保传入数据有正确的列，不做额外 copy 以节省内存

    result = daily_adj[['ts_code', 'trade_date']].copy()

    # ---- 步骤 2：RSI(14) ----
    if 'close_adj' in daily_adj.columns:
        logger.debug("批量计算 RSI(14)...")
        try:
            rsi_df = calculate_rsi(daily_adj, window=14)
            result = result.merge(
                rsi_df[['ts_code', 'trade_date', 'rsi_14']],
                on=['ts_code', 'trade_date'],
                how='left',
            )
        except Exception as e:
            logger.error(f"批量计算 RSI 失败：{e}")
    else:
        logger.warning("precompute_technical_factors: 缺少 close_adj 列，跳过 RSI 计算")

    # ---- 步骤 3：KDJ(9,3,3) ----
    if all(col in daily_adj.columns for col in ['high_adj', 'low_adj', 'close_adj']):
        logger.debug("批量计算 KDJ(9,3,3)...")
        try:
            kdj_df = calculate_kdj(daily_adj, n=9, m1=3, m2=3)
            result = result.merge(
                kdj_df[['ts_code', 'trade_date', 'kdj_k', 'kdj_d', 'kdj_j']],
                on=['ts_code', 'trade_date'],
                how='left',
            )
        except Exception as e:
            logger.error(f"批量计算 KDJ 失败：{e}")
    else:
        logger.warning("precompute_technical_factors: 缺少 high_adj/low_adj/close_adj，跳过 KDJ 计算")

    # ---- 步骤 3b：ATR(14) ----
    if all(col in daily_adj.columns for col in ['high_adj', 'low_adj', 'close_adj']):
        logger.debug("批量计算 ATR(14)...")
        try:
            atr_df = calculate_atr(daily_adj, window=14)
            result = result.merge(
                atr_df[['ts_code', 'trade_date', 'atr_14', 'atr_pct_14']],
                on=['ts_code', 'trade_date'],
                how='left',
            )
        except Exception as e:
            logger.error(f"批量计算 ATR 失败：{e}")

    # ---- 步骤 4：MACD(12,26,9) ----
    if 'close_adj' in daily_adj.columns:
        logger.debug("批量计算 MACD(12,26,9)...")
        try:
            macd_df = calculate_macd(daily_adj, fast=12, slow=26, signal=9)
            result = result.merge(
                macd_df[['ts_code', 'trade_date', 'macd_dif', 'macd_dea', 'macd_hist']],
                on=['ts_code', 'trade_date'],
                how='left',
            )
        except Exception as e:
            logger.error(f"批量计算 MACD 失败：{e}")

    # ---- 步骤 5：布林带(20,2) ----
    if 'close_adj' in daily_adj.columns:
        logger.debug("批量计算布林带(20,2)...")
        try:
            bb_df = calculate_bollinger_bands(daily_adj, window=20, num_std=2.0)
            result = result.merge(
                bb_df[['ts_code', 'trade_date', 'bb_middle', 'bb_upper',
                        'bb_lower', 'bb_width', 'bb_pct']],
                on=['ts_code', 'trade_date'],
                how='left',
            )
        except Exception as e:
            logger.error(f"批量计算布林带失败：{e}")

    # ---- 步骤 6：波动率（多窗口滚动标准差） ----
    # 使用共用 compute_ret_1 构造收益率：优先 close_adj pct_change，其次 pct_chg/100
    ret_1_series = compute_ret_1(daily_adj)
    ret_col_available = not ret_1_series.isna().all()

    if ret_col_available:
        logger.debug(f"批量计算滚动波动率（窗口={vol_windows}）...")
        try:
            vol_input = daily_adj[['ts_code', 'trade_date']].copy()
            vol_input['ret_1'] = ret_1_series.values

            vol_df = calculate_volatility(vol_input, ret_col='ret_1', windows=vol_windows)
            vol_cols = [f'volatility_{w}' for w in vol_windows]
            existing_vol_cols = [c for c in vol_cols if c in vol_df.columns]
            result = result.merge(
                vol_df[['ts_code', 'trade_date'] + existing_vol_cols],
                on=['ts_code', 'trade_date'],
                how='left',
            )
        except Exception as e:
            logger.error(f"批量计算波动率失败：{e}")
    else:
        logger.warning("precompute_technical_factors: 无法构造 ret_1（缺少 ret_1/close_adj/pct_chg 列），跳过波动率计算")

    elapsed = time.time() - t0
    output_cols = [c for c in result.columns if c not in ('ts_code', 'trade_date')]
    logger.info(
        f"批量预计算技术指标与波动率因子完成："
        f"耗时 {elapsed:.2f} 秒，输出 {len(output_cols)} 个因子列，"
        f"共 {len(result)} 条记录"
    )

    return result
