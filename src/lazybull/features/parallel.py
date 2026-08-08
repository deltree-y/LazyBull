"""并行特征构建 —— 无状态入口函数。

将 FeatureBuilder 的缓存与 build_features_for_day() 逻辑解耦为纯函数，
支持通过 joblib.Parallel 多进程并行构建多日特征。

设计要点：
- daily_adj_dict、tech_factor_cache_dict 等大数据结构通过 loky 的
  copy-on-write 语义在 worker 间共享引用，不深拷贝
- 每个 worker 仅读取这些共享缓存，不做修改
- 各日特征构建完全独立，结果收集后统一写入
"""

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


def build_features_for_day_static(
    trade_date: str,
    ctx: "FeatureContext",  # noqa: F821
    daily_adj_dict: Dict[str, pd.DataFrame],
    tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]],
    market_state_cache: Optional[pd.DataFrame],
    trading_dates_list: List[str],
    trading_date_index: Dict[str, int],
    daily_adj_precomputed: Optional[pd.DataFrame],
    factor_registry,
    risk_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None,
    risk_factor_names: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """纯函数版单日特征构建：所有状态外部传入，可 pickle 用于多进程。

    Returns:
        特征 DataFrame，若当日无数据则返回 None。
    """
    # ── 检查交易日 ──
    current_idx = trading_date_index.get(trade_date, -1)
    if current_idx == -1:
        logger.warning(f"{trade_date} 不是交易日，跳过")
        return None

    # ── 获取当日截面 ──
    if daily_adj_dict is not None:
        current_data = daily_adj_dict.get(trade_date, pd.DataFrame()).copy()
    elif daily_adj_precomputed is not None:
        current_data = daily_adj_precomputed[
            daily_adj_precomputed["trade_date"] == trade_date
        ].copy()
    else:
        current_data = pd.DataFrame()

    if len(current_data) == 0:
        logger.warning(f"{trade_date} 没有行情数据")
        return None

    # ── 标签 ──
    from .labels import compute_forward_returns

    labels = compute_forward_returns(
        current_data=current_data,
        trade_date=trade_date,
        trading_dates=trading_dates_list,
        current_idx=current_idx,
        horizons=ctx.horizons,
        daily_adj_dict=daily_adj_dict,
    )

    # ── 基础特征 ──
    from .builder import _calculate_base_features

    features = _calculate_base_features(
        current_data=current_data,
        daily_adj_dict=daily_adj_dict,
        trade_date=trade_date,
        trading_dates=trading_dates_list,
        current_idx=current_idx,
        lookback_windows=ctx.lookback_windows,
        trading_date_index=trading_date_index,
        daily_basic_data=ctx.daily_basic_data,
        moneyflow_data=ctx.moneyflow_data,
    )

    # ── 价值红利 + 资金流 ──
    from .builder import (
        _add_moneyflow_features_static,
        _add_value_dividend_features_static,
        _backfill_fundamental_proxy_features_static,
    )

    if ctx.daily_basic_data is not None and len(ctx.daily_basic_data) > 0:
        features = _add_value_dividend_features_static(
            features=features,
            daily_basic_data=ctx.daily_basic_data,
            trade_date=trade_date,
        )
    if ctx.moneyflow_data is not None and len(ctx.moneyflow_data) > 0:
        features = _add_moneyflow_features_static(
            features=features,
            moneyflow_data=ctx.moneyflow_data,
            trade_date=trade_date,
            trading_dates=trading_dates_list,
            current_idx=current_idx,
            trading_date_index=trading_date_index,
        )

    # ── 因子处理器 ──
    features = factor_registry.apply_all(features, ctx, current_data)

    # ── 基本面代理回填（须在因子处理器之后，与串行路径保持一致）──
    features = _backfill_fundamental_proxy_features_static(features)

    # ── 行业合并 ──
    if ctx.shenwan_industry is not None:
        from .industry_merge import merge_shenwan_industry

        features = merge_shenwan_industry(
            features, ctx.shenwan_industry, ctx.shenwan_level, ctx.verbose
        )

    # ── 高级因子 ──
    from .builder import _add_advanced_factors_static

    features = _add_advanced_factors_static(
        features=features,
        current_data=current_data,
        trade_date=trade_date,
        trading_dates=trading_dates_list,
        current_idx=current_idx,
        lookback_windows=ctx.lookback_windows,
        tech_factor_cache_dict=tech_factor_cache_dict,
    )

    # ── 风控因子（批量预计算缓存查表 + 公告类逐日计算）──
    from .builder import _attach_risk_factors_static

    features = _attach_risk_factors_static(
        features, trade_date, risk_factor_cache_dict, risk_factor_names
    )

    # ── 合并特征和标签 ──
    result = features.merge(labels, on=["trade_date", "ts_code"], how="inner")

    # ── 过滤标记 ──
    from .builder import _add_filter_flags_static, _add_limit_flags_static, _apply_filters_static

    result = _add_filter_flags_static(result, ctx.stock_basic, ctx.suspend_info, trade_date)
    result = _add_limit_flags_static(result, ctx.daily_data, ctx.limit_info, trade_date)
    result = _apply_filters_static(
        result,
        require_label=ctx.require_label,
        label_filter_mode=ctx.label_filter_mode,
        horizon=ctx.horizon,
        horizons=ctx.horizons,
        min_list_days=ctx.min_list_days,
    )

    # ── 中性化 ──
    if ctx.apply_industry_neutralization and ctx.shenwan_industry is not None:
        from .neutralization import apply_industry_neutralization

        result = apply_industry_neutralization(
            result, ctx.horizons, ctx.lookback_windows, ctx.shenwan_level
        )
    if ctx.apply_size_neutralization:
        from .neutralization import apply_size_neutralization

        result = apply_size_neutralization(result)

    # ── 个股特征 ──
    from .builder import _add_new_individual_features_static

    result = _add_new_individual_features_static(result)

    # ── 市场状态 ──
    from .market_state import add_market_state_features

    result = add_market_state_features(
        result=result,
        daily_adj=None,  # 使用缓存
        trade_date=trade_date,
        trading_dates=trading_dates_list,
        current_idx=current_idx,
        market_state_cache=market_state_cache,
    )

    return result


def build_features_parallel(
    trading_dates: List[str],
    ctx_prototype: "FeatureContext",  # noqa: F821
    daily_adj_dict: Dict[str, pd.DataFrame],
    tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]],
    market_state_cache: Optional[pd.DataFrame],
    trading_dates_list: List[str],
    trading_date_index: Dict[str, int],
    daily_adj_precomputed: Optional[pd.DataFrame],
    factor_registry,
    save_fn: Callable[[str, pd.DataFrame], None],
    n_jobs: int = -1,
    risk_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None,
    risk_factor_names: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """并行构建多日特征并写入存储。

    Args:
        trading_dates: 待处理的交易日列表
        ctx_prototype: 共享的 FeatureContext 原型（每日期数据由 data_getter 提供）
        daily_adj_dict: 预计算的复权日线字典
        tech_factor_cache_dict: 预计算的技术因子字典
        market_state_cache: 预计算的市场状态缓存
        trading_dates_list: 全量交易日列表
        trading_date_index: O(1) 日期索引
        daily_adj_precomputed: 预计算的复权日线全量 DataFrame
        factor_registry: FactorRegistry 实例
        save_fn: (trade_date, features_df) -> None 写入回调
        n_jobs: 并行 worker 数，-1 表示全部核心
        risk_factor_cache_dict: 预计算的风控因子字典（trade_date -> 截面）
        risk_factor_names: 预计算风控因子名列表

    Returns:
        (success_count, error_count)
    """
    from joblib import Parallel, delayed

    # 为每日期创建上下文（共享大数据结构）
    def _build_day_ctx(td: str) -> "FeatureContext":
        """为指定交易日创建当日上下文。"""
        # 复用原型，仅替换 trade_date
        import copy

        day_ctx = copy.copy(ctx_prototype)
        day_ctx.trade_date = td
        return day_ctx

    def _process_one_day(td: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        """处理单日并返回 (trade_date, df_or_None, error_or_None)。"""
        try:
            day_ctx = _build_day_ctx(td)
            df = build_features_for_day_static(
                trade_date=td,
                ctx=day_ctx,
                daily_adj_dict=daily_adj_dict,
                tech_factor_cache_dict=tech_factor_cache_dict,
                market_state_cache=market_state_cache,
                trading_dates_list=trading_dates_list,
                trading_date_index=trading_date_index,
                daily_adj_precomputed=daily_adj_precomputed,
                factor_registry=factor_registry,
                risk_factor_cache_dict=risk_factor_cache_dict,
                risk_factor_names=risk_factor_names,
            )
            return td, df, None
        except Exception as e:
            logger.error(f"并行构建 {td} 失败: {e}")
            return td, None, str(e)

    logger.info(f"启动并行特征构建: {len(trading_dates)} 天, workers={n_jobs}")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_process_one_day)(td) for td in trading_dates
    )

    success = 0
    errors = 0
    for td, df, err in results:
        if err is not None:
            errors += 1
            continue
        if df is not None and len(df) > 0:
            try:
                save_fn(td, df)
                success += 1
            except Exception as e:
                logger.error(f"保存 {td} 特征失败: {e}")
                errors += 1

    logger.info(f"并行构建完成: 成功={success}, 失败={errors}")
    return success, errors
