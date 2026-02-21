"""市场状态特征模块

计算全市场截面统计量（每日一个值，回填到当日所有股票）：
- mkt_vol_cnt: 全市场收益率截面标准差（仅 tradable==1）
- mkt_vol_20: mkt_vol_cnt 过去 20 日滚动均值（无前瞻）
- mkt_turnover_ratio: 市场拥挤度因子 sum(amount)/sum(circ_mv)（仅 tradable==1）
- mkt_ret_avg_20: 过去 20 日全市场平均收益率之和（仅 tradable==1）
- mkt_turnover_std: 全市场换手率截面标准差（仅 tradable==1）
- mkt_adv_dec_ratio: 过去 60 日涨跌家数比值滚动均值（仅 tradable==1）
"""

import numpy as np
import pandas as pd
from loguru import logger


def _get_ret1_for_date(daily_data: pd.DataFrame, date: str) -> pd.Series:
    """获取指定日期的 ret_1，以 vol>0 作为可交易代理"""
    day_data = daily_data[daily_data['trade_date'] == date]
    if len(day_data) == 0:
        return pd.Series(dtype=float)

    # 计算 ret_1
    if 'ret_1' in day_data.columns:
        ret_col = day_data['ret_1']
    elif 'pct_chg' in day_data.columns:
        ret_col = day_data['pct_chg'] / 100.0
    else:
        return pd.Series(dtype=float)

    # 以 vol>0 作为可交易代理
    if 'vol' in day_data.columns:
        mask = day_data['vol'] > 0
        return ret_col[mask].dropna()
    return ret_col.dropna()


def _compute_daily_market_stats(daily_data: pd.DataFrame, date: str) -> dict:
    """计算单日市场截面统计量

    Returns:
        dict: 包含 vol_cnt、mean_ret、adv_dec_ratio 三个键
    """
    ret = _get_ret1_for_date(daily_data, date)

    result = {
        'vol_cnt': np.nan,
        'mean_ret': np.nan,
        'adv_dec_ratio': np.nan,
    }

    if len(ret) < 2:
        return result

    result['vol_cnt'] = float(ret.std())
    result['mean_ret'] = float(ret.mean())

    adv = int((ret > 0).sum())
    dec = int((ret < 0).sum())
    result['adv_dec_ratio'] = (adv + 1) / (dec + 1)

    return result


def compute_market_state_features(
    daily_data: pd.DataFrame,
    trade_date: str,
    trading_dates: list,
    current_idx: int,
    daily_basic_data: pd.DataFrame = None,
) -> dict:
    """计算当日市场状态特征（标量值字典）

    Args:
        daily_data: 全部历史日线数据（含 ts_code, trade_date, pct_chg/ret_1, vol, amount）
        trade_date: 目标交易日（YYYYMMDD）
        trading_dates: 交易日列表（已排序）
        current_idx: trade_date 在 trading_dates 中的索引
        daily_basic_data: 全部历史每日指标数据（可选，含 circ_mv, turnover_rate_f 等）

    Returns:
        dict: {特征名: 标量值}
    """
    VOL_WINDOW = 20
    RET_AVG_WINDOW = 20
    ADV_DEC_WINDOW = 60

    # --- 当日截面特征 ---
    today_stats = _compute_daily_market_stats(daily_data, trade_date)
    mkt_vol_cnt = today_stats['vol_cnt']

    # mkt_turnover_ratio 和 mkt_turnover_std 仅用当日 daily_basic 数据，无需滚动
    mkt_turnover_ratio = np.nan
    mkt_turnover_std = np.nan

    if daily_basic_data is not None and len(daily_basic_data) > 0:
        db_today = daily_basic_data[daily_basic_data['trade_date'] == trade_date].copy()
        today_data = daily_data[daily_data['trade_date'] == trade_date]

        # 以 vol>0 确定可交易股票集合
        if 'vol' in today_data.columns:
            tradable_codes = set(today_data.loc[today_data['vol'] > 0, 'ts_code'])
            db_tradable = db_today[db_today['ts_code'].isin(tradable_codes)]
        else:
            db_tradable = db_today

        # mkt_turnover_ratio = sum(amount) / sum(circ_mv)
        if 'circ_mv' in db_tradable.columns and len(db_tradable) > 0:
            tradable_today = today_data[today_data['ts_code'].isin(db_tradable['ts_code'])]
            if 'amount' in tradable_today.columns:
                total_amount = tradable_today['amount'].sum()
                total_circ_mv = db_tradable['circ_mv'].sum()
                if total_circ_mv > 0:
                    mkt_turnover_ratio = float(total_amount / total_circ_mv)

        # mkt_turnover_std = std(turnover_rate_f)，若不存在则回退至 turnover_rate
        tf_col = 'turnover_rate_f' if 'turnover_rate_f' in db_tradable.columns else 'turnover_rate'
        if tf_col in db_tradable.columns:
            tf_vals = db_tradable[tf_col].dropna()
            if len(tf_vals) >= 2:
                mkt_turnover_std = float(tf_vals.std())

    # --- 需要历史数据的滚动特征 ---
    max_window = max(VOL_WINDOW, RET_AVG_WINDOW, ADV_DEC_WINDOW)
    hist_start_idx = max(0, current_idx - max_window + 1)
    # 包含当日在内的历史窗口日期
    hist_dates = trading_dates[hist_start_idx: current_idx + 1]

    vol_cnt_list = []
    mean_ret_list = []
    adv_dec_ratio_list = []

    for d in hist_dates:
        stats = _compute_daily_market_stats(daily_data, d)
        vol_cnt_list.append(stats['vol_cnt'])
        mean_ret_list.append(stats['mean_ret'])
        adv_dec_ratio_list.append(stats['adv_dec_ratio'])

    # mkt_vol_20: 最近 20 日 vol_cnt 均值（含当日，min_periods=1）
    if vol_cnt_list:
        window_vals = [v for v in vol_cnt_list[-VOL_WINDOW:] if not np.isnan(v)]
        mkt_vol_20 = float(np.mean(window_vals)) if window_vals else np.nan
    else:
        mkt_vol_20 = np.nan

    # mkt_ret_avg_20: 最近 20 日截面平均收益率之和
    if mean_ret_list:
        window_vals = [v for v in mean_ret_list[-RET_AVG_WINDOW:] if not np.isnan(v)]
        mkt_ret_avg_20 = float(sum(window_vals)) if window_vals else np.nan
    else:
        mkt_ret_avg_20 = np.nan

    # mkt_adv_dec_ratio: 最近 60 日涨跌比均值
    if adv_dec_ratio_list:
        window_vals = [v for v in adv_dec_ratio_list[-ADV_DEC_WINDOW:] if not np.isnan(v)]
        mkt_adv_dec_ratio = float(np.mean(window_vals)) if window_vals else np.nan
    else:
        mkt_adv_dec_ratio = np.nan

    logger.debug(
        f"市场状态特征 [{trade_date}]: "
        f"mkt_vol_cnt={mkt_vol_cnt} mkt_vol_20={mkt_vol_20} "
        f"mkt_turnover_ratio={mkt_turnover_ratio} "
        f"mkt_ret_avg_20={mkt_ret_avg_20} "
        f"mkt_adv_dec_ratio={mkt_adv_dec_ratio}"
    )

    return {
        'mkt_vol_cnt': mkt_vol_cnt,
        'mkt_vol_20': mkt_vol_20,
        'mkt_turnover_ratio': mkt_turnover_ratio,
        'mkt_ret_avg_20': mkt_ret_avg_20,
        'mkt_turnover_std': mkt_turnover_std,
        'mkt_adv_dec_ratio': mkt_adv_dec_ratio,
    }
