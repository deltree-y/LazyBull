"""Walk-forward 单 split OOS 回测执行。"""

from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from ...common.backtest_runtime import (
    create_backtest_engine_from_config,
    create_or_reuse_signal,
    infer_rebalance_freq_from_label,
)
from ...common.config import get_data_root
from ...common.trading_config import TradingConfig
from ...data import DataLoader, Storage
from ...factors.holdertrade import derive_holdertrade_columns
from ...factors.repurchase import derive_repurchase_columns
from ...factors.top10_floatholders import derive_top10fh_columns
from ...factors.top_inst import derive_top_inst_columns
from ...universe import BasicUniverse
from ...universe.domains import domain_market_whitelist


def run_oos_backtest(
    model_version: int,
    bt_start: str,
    bt_end: str,
    storage: Storage,
    loader: DataLoader,
    trade_cal: pd.DataFrame,
    stock_basic: pd.DataFrame,
    label_column: str,
    bt_top_n: int = 30,
    bt_rebalance_freq: Optional[int] = None,
    data_root: Optional[str] = None,
    persistent_signal=None,
    bt_exclude_st: bool = True,
    bt_min_list_days: int = 365,
    bt_sell_timing: str = "open",
    bt_max_weight_per_stock: Optional[float] = None,
    bt_max_per_industry: Optional[int] = None,
    bt_stop_loss_enabled: bool = False,
    bt_stop_loss_drawdown_pct: float = 30.0,
    bt_stop_loss_consecutive_limit_down: int = 2,
    position_sizing: str = "equal",
    kelly_vol_window: int = 60,
    kelly_max_leverage: float = 0.25,
    stagger_tranches: int = 1,
    enable_early_rebalance_on_empty: bool = True,
    initial_capital: float = 1000000.0,
    split_num: Optional[int] = None,
    exposure_table: Optional[Dict[str, float]] = None,
    exposure_policy: Any = None,
    exposure_replenish: bool = False,
    exposure_trim_tolerance: Optional[float] = None,
    exposure_budget_discount_replenish: bool = False,
    stock_domain: str = "main",
    holdertrade_lookup: Optional[Dict[str, pd.DataFrame]] = None,
    repurchase_lookup: Optional[Dict[str, pd.DataFrame]] = None,
    top10fh_panel: Optional[pd.DataFrame] = None,
    top_inst_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """对单个 split 模型运行 OOS 回测并返回组合级绩效指标。

    Args:
        holdertrade_lookup: 股东增减持运行时查询表（与训练侧同一张）；提供时逐日
            补齐本族列，避免 MLSignal 因 cs_train 无本族列而静默补 NaN。
        repurchase_lookup: 股票回购运行时查询表（与训练侧同一张）；语义同上。
        top10fh_panel: 十大流通股东**报告期面板**（与训练侧同一份）；提供时逐日派生补齐本族列
            （本族逐日全市场稠密 ⇒ 用面板而非逐日字典）。
    """
    data_root = data_root or get_data_root()
    logger.info(f"OOS 回测: {bt_start} ~ {bt_end}（模型 v{model_version}, Top{bt_top_n}）")

    daily_data = loader.load_clean_daily(bt_start, bt_end)
    if daily_data is None or len(daily_data) == 0:
        logger.warning(f"OOS回测: 无法加载 {bt_start}~{bt_end} 日线数据，跳过")
        return {}

    desired_cols = [
        "ts_code",
        "trade_date",
        "close",
        "close_adj",
        "open",
        "open_adj",
        "is_suspended",
        "is_limit_up",
        "is_limit_down",
        "vol",
        "pct_chg",
        "is_st",
        "list_days",
        "tradable",
    ]
    existing_cols = [column for column in desired_cols if column in daily_data.columns]
    price_data = daily_data[existing_cols].copy()
    if "close" not in price_data.columns:
        logger.warning("OOS回测: 缺少 close 列，跳过")
        return {}

    trade_dates = trade_cal[
        (trade_cal["cal_date"] >= bt_start)
        & (trade_cal["cal_date"] <= bt_end)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    features_by_date = {}
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            features_by_date[trade_date] = features

    # 股东增减持因子为**运行时派生**（cs_train 不含本族列）：回测复用同一张查询表
    # 逐日补齐，否则 MLSignal 预测将因缺列而静默补 NaN（train/serve 偏差）。
    if holdertrade_lookup:
        derived_days = 0
        for trade_date, features in features_by_date.items():
            if derive_holdertrade_columns(features, holdertrade_lookup):
                derived_days += 1
        logger.info(f"OOS 回测特征派生股东增减持列: {derived_days}/{len(features_by_date)} 日")

    # 股票回购因子同为运行时派生（cs_train 不含本族列）：复用同一张查询表逐日补齐
    if repurchase_lookup:
        rp_days = 0
        for trade_date, features in features_by_date.items():
            if derive_repurchase_columns(features, repurchase_lookup):
                rp_days += 1
        logger.info(f"OOS 回测特征派生股票回购列: {rp_days}/{len(features_by_date)} 日")

    # 十大流通股东因子同为运行时派生（cs_train 不含本族列）：复用同一份报告期面板
    if top10fh_panel is not None and len(top10fh_panel) > 0:
        tfh_days = 0
        for trade_date, features in features_by_date.items():
            if derive_top10fh_columns(features, top10fh_panel):
                tfh_days += 1
        logger.info(f"OOS 回测特征派生十大流通股东列: {tfh_days}/{len(features_by_date)} 日")

    # 龙虎榜机构席位因子同为运行时派生（cs_train 不含本族列）：复用同一张查询表逐日补齐
    if top_inst_lookup:
        ti_days = 0
        for trade_date, features in features_by_date.items():
            if derive_top_inst_columns(features, top_inst_lookup):
                ti_days += 1
        logger.info(f"OOS 回测特征派生龙虎榜机构席位列: {ti_days}/{len(features_by_date)} 日")

    if not features_by_date:
        logger.warning(f"OOS回测: 无特征数据 {bt_start}~{bt_end}，跳过")
        return {}

    logger.info(f"OOS回测数据: 日线={len(daily_data)}条, 特征={len(features_by_date)}日")
    effective_config = TradingConfig(
        model_version=model_version,
        top_n=bt_top_n,
        rebalance_freq=(
            bt_rebalance_freq
            if bt_rebalance_freq is not None
            else infer_rebalance_freq_from_label(label_column)
        ),
        stagger_tranches=stagger_tranches,
        max_per_industry=bt_max_per_industry,
        max_weight_per_stock=bt_max_weight_per_stock,
        enable_early_rebalance_on_empty=enable_early_rebalance_on_empty,
        exclude_st=bt_exclude_st,
        min_list_days=bt_min_list_days,
        stop_loss_enabled=bt_stop_loss_enabled,
        stop_loss_drawdown_pct=bt_stop_loss_drawdown_pct,
        stop_loss_consecutive_limit_down=bt_stop_loss_consecutive_limit_down,
        position_sizing=position_sizing,
        kelly_vol_window=kelly_vol_window,
        kelly_max_leverage=kelly_max_leverage,
        initial_capital=initial_capital,
        sell_price=bt_sell_timing,
    )

    universe = BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=bt_exclude_st,
        min_list_days=bt_min_list_days,
        # 域市场白名单（universe/domains.py 单一来源；默认主板与历史行为逐位一致；
        # main_gem 放开创业板；市值上下限在 MLSignal 侧过滤）
        markets=domain_market_whitelist(stock_domain),
        verbose=False,
    )
    signal = create_or_reuse_signal(
        effective_config,
        data_root=data_root,
        persistent_signal=persistent_signal,
        verbose=False,
    )
    engine = create_backtest_engine_from_config(
        trading_config=effective_config,
        universe=universe,
        signal=signal,
        features_by_date=features_by_date,
        stock_basic=stock_basic,
        data_storage=storage,
        initial_capital=initial_capital,
        sell_timing=bt_sell_timing,
        verbose=False,
        completion_window_days=5,
        enable_pending_order=True,
    )
    # 政策旁路（terminal_loss P2-1）：开启逐日持仓快照（只读，不影响任何决策）
    engine.record_holdings_snapshot = True
    # 政策层 E2（P2-3 shadow）：装载暴露系数表（None = 不启用，成交与净值逐位一致）
    engine.set_exposure_table(
        exposure_table,
        verbose=exposure_table is not None,
        replenish=exposure_replenish,
        trim_tolerance=exposure_trim_tolerance,
        budget_discount_replenish=exposure_budget_discount_replenish,
    )
    # 政策层 P2-5：在线现算 provider（与系数表互斥；None = 不启用）
    if exposure_policy is not None:
        engine.set_exposure_policy(
            exposure_policy,
            verbose=True,
            replenish=exposure_replenish,
            trim_tolerance=exposure_trim_tolerance,
            budget_discount_replenish=exposure_budget_discount_replenish,
        )

    nav_curve = engine.run(
        start_date=pd.Timestamp(bt_start),
        end_date=pd.Timestamp(bt_end),
        trading_dates=[pd.Timestamp(date) for date in trade_dates],
        price_data=price_data,
    )
    if nav_curve is None or nav_curve.empty or "nav" not in nav_curve.columns:
        logger.warning("OOS回测: 净值曲线为空，跳过")
        return {}

    total_return = nav_curve["return"].iloc[-1]
    nav_values = nav_curve["nav"].values
    cumulative_max = pd.Series(nav_values).cummax()
    max_drawdown = ((pd.Series(nav_values) - cumulative_max) / cumulative_max).min()
    trading_days = len(nav_curve)
    years = trading_days / 252
    annual_return = total_return / years if years > 0 else 0
    daily_returns = nav_curve["nav"].pct_change(fill_method=None).dropna()
    volatility = daily_returns.std() * (252**0.5)
    sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        "bt_total_return": round(total_return, 6),
        "bt_annual_return": round(annual_return, 6),
        "bt_max_drawdown": round(max_drawdown, 6),
        "bt_volatility": round(volatility, 6),
        "bt_sharpe": round(sharpe, 4),
        "bt_calmar": round(calmar, 4),
        "bt_trading_days": trading_days,
        "bt_start": bt_start,
        "bt_end": bt_end,
        "bt_top_n": bt_top_n,
    }
    split_tag = f"Split {split_num} | " if split_num is not None else ""
    # 政策层 E2（P2-3 shadow）：减仓/回补统计（未启用暴露表时不输出）
    if (
        getattr(engine, "exposure_table", None) is not None
        or getattr(engine, "exposure_policy_provider", None) is not None
    ):
        logger.info(f"{split_tag}暴露门控减仓统计: {engine.get_exposure_trim_report()}")
        logger.info(f"{split_tag}暴露门控回补统计: {engine.get_exposure_replenish_report()}")
    logger.info(
        f"\n{'#' * 80}\n"
        f"{split_tag}{bt_start}-{bt_end}\n"
        f"OOS回测结果: 总收益={total_return*100:.2f}%, "
        f"年化={annual_return*100:.2f}%, "
        f"最大回撤={max_drawdown*100:.2f}%, "
        f"夏普={sharpe:.2f} \n"
        f"{'#' * 80}\n\n"
    )
    metrics["_nav_curve"] = nav_curve
    metrics["_trades"] = engine.get_trades()
    metrics["_execution_attribution"] = engine.get_execution_attribution()
    metrics["_holdings_snapshot"] = engine.get_holdings_snapshot()
    return metrics
