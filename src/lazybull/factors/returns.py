"""收益率构造工具模块

提供统一的 ret_1（单日收益率）构造函数，供各因子模块复用，
避免因口径不一致导致波动率等衍生指标产生差异。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_ret_1(daily_adj: pd.DataFrame) -> pd.Series:
    """构造单日收益率 ret_1，与 daily_adj 行索引对齐

    优先级（从高到低）：
    1. 若 daily_adj 已含 ``ret_1`` 列：直接返回（不做拷贝）
    2. elif 含 ``close_adj`` 列：按 ``ts_code`` 分组、``trade_date`` 升序排序后
       调用 ``pct_change()``，不产生跨股票边界差分（无前瞻）
    3. elif 含 ``pct_chg`` 列：使用 ``pct_chg / 100``
    4. 否则：返回全 NaN Series 并记录 warning

    Args:
        daily_adj: 日线数据 DataFrame，应包含 ``ts_code``、``trade_date`` 列

    Returns:
        与 ``daily_adj`` 行索引对齐的 ret_1 Series（float64）
    """
    if 'ret_1' in daily_adj.columns:
        return daily_adj['ret_1']

    if 'close_adj' in daily_adj.columns:
        # 按 ts_code 分组、trade_date 升序排序后计算 pct_change
        # fill_method=None：不做 NaN 前向填充，停牌/缺行段保持 NaN，
        # 避免跨停牌期计算收益（配合按交易日对齐的滚动窗口，审计问题6）
        sorted_df = daily_adj[['ts_code', 'trade_date', 'close_adj']].sort_values(
            ['ts_code', 'trade_date']
        )
        ret = sorted_df.groupby('ts_code', sort=False)['close_adj'].pct_change(fill_method=None)
        # 恢复到原始行索引顺序
        return ret.reindex(daily_adj.index)

    if 'pct_chg' in daily_adj.columns:
        logger.warning(
            "compute_ret_1: 缺少 ret_1 与 close_adj 列，"
            "fallback 使用 pct_chg/100（口径可能与复权收益率不一致）"
        )
        return daily_adj['pct_chg'] / 100.0

    logger.warning(
        "compute_ret_1: daily_adj 缺少 ret_1、close_adj、pct_chg 列，"
        "返回全 NaN Series"
    )
    return pd.Series(float('nan'), index=daily_adj.index, dtype='float64')
