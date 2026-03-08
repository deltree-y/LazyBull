"""市场状态特征模块

计算全市场截面统计量（每日一个值，回填到当日所有股票）：
- mkt_vol_cnt: 全市场收益率截面标准差（仅 tradable==1）
- mkt_vol_20: mkt_vol_cnt 过去 20 日滚动均值（无前瞻）
- mkt_turnover_ratio: 市场拥挤度因子 sum(amount)/sum(circ_mv)（仅 tradable==1）
- mkt_ret_avg_20: 过去 20 日全市场平均收益率之和（仅 tradable==1）
- mkt_turnover_std: 全市场换手率截面标准差（仅 tradable==1）
- mkt_adv_dec_ratio: 过去 60 日涨跌家数比值滚动均值（仅 tradable==1）

性能优化：批量构建时应使用 `precompute_market_state_features()` 一次性计算所有交易日，
再通过 `FeatureBuilder` 实例缓存按日 O(1) 取值，避免逐日重复计算。
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
    # +1 为 Laplace 平滑，防止 dec==0 时除以零，同时缓解极端单边行情的影响
    result['adv_dec_ratio'] = (adv + 1) / (dec + 1)

    return result


def _rolling_mean(values: list, window: int) -> float:
    """对列表最后 window 个有效（非 NaN）值求均值（min_periods=1）"""
    window_vals = [v for v in values[-window:] if not np.isnan(v)]
    return float(np.mean(window_vals)) if window_vals else np.nan


def _rolling_sum(values: list, window: int) -> float:
    """对列表最后 window 个有效（非 NaN）值求和（min_periods=1）"""
    window_vals = [v for v in values[-window:] if not np.isnan(v)]
    return float(sum(window_vals)) if window_vals else np.nan


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
    mkt_vol_20 = _rolling_mean(vol_cnt_list, VOL_WINDOW)

    # mkt_ret_avg_20: 最近 20 日截面平均收益率之和
    mkt_ret_avg_20 = _rolling_sum(mean_ret_list, RET_AVG_WINDOW)

    # mkt_adv_dec_ratio: 最近 60 日涨跌比均值
    mkt_adv_dec_ratio = _rolling_mean(adv_dec_ratio_list, ADV_DEC_WINDOW)

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
        # 以下新增特征在逐日计算模式下不可用（需要完整历史），填 NaN
        'mkt_ma_trend': np.nan,
        'mkt_drawdown_20': np.nan,
        'mkt_ret_avg_60': np.nan,
    }


def precompute_market_state_features(
    daily_data: pd.DataFrame,
    trading_dates: list,
    daily_basic_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """批量预计算所有交易日的市场状态特征（高性能版本）

    通过向量化 groupby + pandas rolling 一次性计算全部交易日，
    避免逐日循环调用 compute_market_state_features() 带来的重复计算。
    输出口径与 compute_market_state_features() 完全一致。

    Args:
        daily_data: 全量历史日线数据（含 ts_code, trade_date, pct_chg/ret_1, vol, amount）
        trading_dates: 已排序的交易日列表（YYYYMMDD 格式）
        daily_basic_data: 全量历史每日指标数据（可选，含 circ_mv, turnover_rate_f 等）

    Returns:
        以 trade_date 为索引的 DataFrame，包含 6 个市场状态列：
        mkt_vol_cnt, mkt_vol_20, mkt_turnover_ratio, mkt_ret_avg_20,
        mkt_turnover_std, mkt_adv_dec_ratio
    """
    VOL_WINDOW = 20
    RET_AVG_WINDOW = 20
    ADV_DEC_WINDOW = 60

    nan_result = pd.DataFrame(
        {
            'mkt_vol_cnt': np.nan,
            'mkt_vol_20': np.nan,
            'mkt_turnover_ratio': np.nan,
            'mkt_ret_avg_20': np.nan,
            'mkt_turnover_std': np.nan,
            'mkt_adv_dec_ratio': np.nan,
        },
        index=pd.Index(trading_dates, name='trade_date'),
    )

    if daily_data is None or len(daily_data) == 0 or not trading_dates:
        return nan_result

    # --- 步骤 1：生成 _ret 列 ---
    if 'ret_1' in daily_data.columns:
        work = daily_data[['trade_date', 'ts_code', 'ret_1']].copy()
        work = work.rename(columns={'ret_1': '_ret'})
    elif 'pct_chg' in daily_data.columns:
        work = daily_data[['trade_date', 'ts_code', 'pct_chg']].copy()
        work['_ret'] = work['pct_chg'] / 100.0
        work = work.drop(columns=['pct_chg'])
    else:
        logger.warning("precompute_market_state_features: 缺少 ret_1 / pct_chg 列，返回全 NaN")
        return nan_result

    # 附加 vol 和 amount 列（后续使用）
    for col in ('vol', 'amount'):
        if col in daily_data.columns:
            work[col] = daily_data[col].values

    # --- 步骤 2：过滤可交易（vol > 0 代理）---
    if 'vol' in work.columns:
        tradable = work[work['vol'] > 0].copy()
    else:
        tradable = work.copy()

    #--- 步骤 3：批量计算每日市场统计量（vol_cnt, mean_ret, adv_dec_ratio）---
    # 3.1. 预计算基础指标 (批量 C 级运算)
    # count 会自动忽略 NaN
    stats = tradable.groupby('trade_date')['_ret'].agg(
        vol_cnt='std',     # 默认 ddof=1
        mean_ret='mean',
        valid_count='count',
        adv=lambda x: (x > 0).sum(),
        dec=lambda x: (x < 0).sum()
    )

    # 3.2. 向量化处理逻辑判断 (代替 if len < 2)
    # 如果样本数 < 2，将统计量设为 NaN
    mask_too_small = stats['valid_count'] < 2
    stats.loc[mask_too_small, ['vol_cnt', 'mean_ret']] = np.nan

    # 3.3. 向量化计算 adv_dec_ratio (代替字典里的计算)
    stats['adv_dec_ratio'] = (stats['adv'] + 1) / (stats['dec'] + 1)

    # 3.4. 只保留需要的列
    daily_stats = stats[['vol_cnt', 'mean_ret', 'adv_dec_ratio']]

    # --- 步骤 4：对齐到 trading_dates（缺失日期补 NaN）---
    daily_stats = daily_stats.reindex(trading_dates)

    # --- 步骤 5：pandas rolling 计算窗口特征 ---
    # min_periods=1：有效值不足窗口时仍计算（与原 _rolling_mean/_rolling_sum 行为一致）
    mkt_vol_20 = daily_stats['vol_cnt'].rolling(window=VOL_WINDOW, min_periods=1).mean()
    mkt_ret_avg_20 = daily_stats['mean_ret'].rolling(window=RET_AVG_WINDOW, min_periods=1).sum()
    mkt_adv_dec_ratio = daily_stats['adv_dec_ratio'].rolling(window=ADV_DEC_WINDOW, min_periods=1).mean()

    # --- 步骤 6：计算 turnover_ratio 和 turnover_std（当日截面特征）---
    mkt_turnover_ratio = pd.Series(np.nan, index=pd.Index(trading_dates, name='trade_date'))
    mkt_turnover_std = pd.Series(np.nan, index=pd.Index(trading_dates, name='trade_date'))

    if daily_basic_data is not None and len(daily_basic_data) > 0:
        # 确定换手率列名（优先 turnover_rate_f）
        tf_col = (
            'turnover_rate_f'
            if 'turnover_rate_f' in daily_basic_data.columns
            else 'turnover_rate'
        )

        basic_cols = ['trade_date', 'ts_code']
        if 'circ_mv' in daily_basic_data.columns:
            basic_cols.append('circ_mv')
        if tf_col in daily_basic_data.columns:
            basic_cols.append(tf_col)

        db = daily_basic_data[basic_cols].copy()

        # 与 daily_data 的 (trade_date, ts_code, vol, amount) 向量化 merge
        merge_cols = ['trade_date', 'ts_code']
        if 'vol' in daily_data.columns:
            merge_cols.append('vol')
        if 'amount' in daily_data.columns:
            merge_cols.append('amount')

        daily_sub = daily_data[merge_cols].copy()
        merged = db.merge(daily_sub, on=['trade_date', 'ts_code'], how='left')

        # 只保留可交易（vol > 0）
        if 'vol' in merged.columns:
            tradable_merged = merged[merged['vol'] > 0]
        else:
            tradable_merged = merged

        # mkt_turnover_ratio = sum(amount) / sum(circ_mv)
        if 'circ_mv' in tradable_merged.columns and 'amount' in tradable_merged.columns:
            tr_grp = tradable_merged.groupby('trade_date').agg(
                total_amount=('amount', 'sum'),
                total_circ_mv=('circ_mv', 'sum'),
            )
            tr_grp['ratio'] = tr_grp['total_amount'] / tr_grp['total_circ_mv'].replace(0, np.nan)
            mkt_turnover_ratio = tr_grp['ratio'].reindex(
                trading_dates
            ).set_axis(pd.Index(trading_dates, name='trade_date'))

        # mkt_turnover_std = std(turnover_rate_f)，ddof=1（pandas 默认）
        if tf_col in tradable_merged.columns:
            tf_grp = tradable_merged.groupby('trade_date')[tf_col].std()
            mkt_turnover_std = tf_grp.reindex(
                trading_dates
            ).set_axis(pd.Index(trading_dates, name='trade_date'))

    # --- 步骤 7：市场择时特征 ---
    MA_SHORT = 20
    MA_LONG = 60
    DRAWDOWN_WINDOW = 20

    # mkt_cumret: 全市场日均收益的累积曲线（用于计算 MA 趋势和回撤）
    mkt_cumret = (1 + daily_stats['mean_ret'].fillna(0)).cumprod()

    # mkt_ma_trend: 短期 MA / 长期 MA，>1 为牛市趋势，<1 为熊市趋势
    ma_short = mkt_cumret.rolling(window=MA_SHORT, min_periods=1).mean()
    ma_long = mkt_cumret.rolling(window=MA_LONG, min_periods=1).mean()
    mkt_ma_trend = (ma_short / ma_long.replace(0, np.nan)).fillna(1.0)

    # mkt_drawdown_20: 近 20 日全市场累积收益的最大回撤（值 ≤ 0）
    rolling_max = mkt_cumret.rolling(window=DRAWDOWN_WINDOW, min_periods=1).max()
    mkt_drawdown_20 = ((mkt_cumret - rolling_max) / rolling_max.replace(0, np.nan)).fillna(0.0)

    # mkt_ret_avg_60: 近 60 日全市场平均收益之和（中长期动量信号）
    mkt_ret_avg_60 = daily_stats['mean_ret'].rolling(window=60, min_periods=1).sum()

    # --- 步骤 8：组装结果 DataFrame ---
    result = pd.DataFrame(
        {
            'mkt_vol_cnt': daily_stats['vol_cnt'].values,
            'mkt_vol_20': mkt_vol_20.values,
            'mkt_turnover_ratio': mkt_turnover_ratio.values,
            'mkt_ret_avg_20': mkt_ret_avg_20.values,
            'mkt_turnover_std': mkt_turnover_std.values,
            'mkt_adv_dec_ratio': mkt_adv_dec_ratio.values,
            'mkt_ma_trend': mkt_ma_trend.values,
            'mkt_drawdown_20': mkt_drawdown_20.values,
            'mkt_ret_avg_60': mkt_ret_avg_60.values,
        },
        index=pd.Index(trading_dates, name='trade_date'),
    )

    logger.debug(
        f"precompute_market_state_features: 已批量预计算 {len(trading_dates)} 个交易日的市场状态特征"
    )
    return result
