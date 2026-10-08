"""Walk-forward 单 split OOS 回测重放驱动（P2a-T7 迁移复制件，行为冻结）。

来源蓝本（B2）：旧 ``ml/walk_forward/backtest.py::run_oos_backtest`` 复制改指。
WF 编排（runner.py）本身不迁；本模块是 B0 重放 14 折的单折执行体。

改指清单：
- runtime 三函数 → ``src.lazybull.v2.hosts.backtest.runtime``（A1）；
- ``TradingConfig`` → ``src.lazybull.v2.common.trading_config``（T1 复制件）；
- ``get_data_root`` → 旧 ``src.lazybull.common.config``（只读过渡依赖，
  全局单例不复制，去除节点 P4 展平/切换评审）；
- ``DataLoader / Storage`` → 旧 ``src.lazybull.data``（只读过渡依赖，
  去除节点 P2b/P3 数据面切换，§3.5 登记表）；
- factors 四族运行时派生函数（holdertrade / repurchase / top10_floatholders /
  top_inst）→ 旧包**永久沿用**（§3.5 登记表）；
- ``BasicUniverse`` / ``domain_market_whitelist`` → 旧 ``universe``（永久沿用）。

D6 摘除（契约授权例外）：exposure 政策族已裁决退役——删除
``engine.set_exposure_table(...)`` 与 ``engine.set_exposure_policy(...)``
两次调用及文末 exposure 统计日志块；**签名保留 5 个 exposure 参数**，非默认值
显式 fail-fast（B0 重放只允许全默认值）。

D8 拆分（T7 实测触发）：旧 ``run_oos_backtest`` 在全仓复杂度基线在册
（func_lines / mccabe 21 两条），逐字复制即新增未豁免硬超限 ⇒ 按 D8 同例
提取式拆分为 4 个模块级 helper（fail-fast / 日线加载 / 单族派生 / 指标段），
语句、守卫条件、日志文本与顺序逐字保留，行为零变化。
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd
from loguru import logger

# 只读过渡依赖（去除节点 P4）：config 为全局单例，不复制（§3.5 / V2R-02）。
from src.lazybull.common.config import get_data_root

# 只读过渡依赖（去除节点 P2b/P3，D2）：数据面在 P2a 沿用旧 cs_train 单通道。
from src.lazybull.data import DataLoader, Storage

# 运行时派生四族因子（永久沿用，§3.5 登记表）：factors 不迁移。
from src.lazybull.factors.holdertrade import derive_holdertrade_columns
from src.lazybull.factors.repurchase import derive_repurchase_columns
from src.lazybull.factors.top10_floatholders import derive_top10fh_columns
from src.lazybull.factors.top_inst import derive_top_inst_columns

# universe 永久沿用（§3.5 登记表）。
from src.lazybull.universe import BasicUniverse
from src.lazybull.universe.domains import domain_market_whitelist
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.hosts.backtest.runtime import (
    create_backtest_engine_from_config,
    create_or_reuse_signal,
    infer_rebalance_freq_from_label,
)


def _fail_fast_retired_exposure(
    exposure_table: Optional[Dict[str, float]],
    exposure_policy: Any,
    exposure_replenish: bool,
    exposure_trim_tolerance: Optional[float],
    exposure_budget_discount_replenish: bool,
) -> None:
    """D6 摘除 fail-fast：exposure 政策族已退役摘除（P2a D6），B0 重放只允许全默认值；
    非默认值立即报错，禁止静默丢弃（行为冻结的例外边界显式化）。"""
    if (
        exposure_table is not None
        or exposure_policy is not None
        or exposure_replenish
        or exposure_trim_tolerance is not None
        or exposure_budget_discount_replenish
    ):
        raise ValueError(
            "exposure 政策族已退役摘除（P2a D6），B0 重放只允许全默认值："
            f"exposure_table="
            f"{type(exposure_table).__name__ if exposure_table is not None else None}"
            f", exposure_policy="
            f"{type(exposure_policy).__name__ if exposure_policy is not None else None}"
            f", exposure_replenish={exposure_replenish}"
            f", exposure_trim_tolerance={exposure_trim_tolerance}"
            f", exposure_budget_discount_replenish={exposure_budget_discount_replenish}"
        )


def _load_replay_price_data(
    loader: DataLoader, bt_start: str, bt_end: str
) -> Optional[pd.DataFrame]:
    """日线加载与列裁剪（D8 提取自 run_oos_backtest；语句/日志逐字保留）。

    Returns:
        price_data；None = 旧侧早退（``return {}``）触发条件。
        注：列子集不改行数，``len(price_data)`` 即旧侧 ``len(daily_data)``。
    """
    daily_data = loader.load_clean_daily(bt_start, bt_end)
    if daily_data is None or len(daily_data) == 0:
        logger.warning(f"OOS回测: 无法加载 {bt_start}~{bt_end} 日线数据，跳过")
        return None

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
        return None
    return price_data


def _derive_feature_family(
    features_by_date: Dict[str, pd.DataFrame],
    source: Any,
    derive_fn: Callable[[pd.DataFrame, Any], bool],
    log_label: str,
) -> None:
    """单族运行时派生逐日补齐（D8 参数化提取：旧四段的公共形态，日志文本逐字）。"""
    derived_days = 0
    for trade_date, features in features_by_date.items():
        if derive_fn(features, source):
            derived_days += 1
    logger.info(f"OOS 回测特征派生{log_label}列: {derived_days}/{len(features_by_date)} 日")


def _load_replay_features(
    storage: Storage,
    trade_dates: list,
    bt_start: str,
    bt_end: str,
    holdertrade_lookup: Optional[Dict[str, pd.DataFrame]],
    repurchase_lookup: Optional[Dict[str, pd.DataFrame]],
    top10fh_panel: Optional[pd.DataFrame],
    top_inst_lookup: Optional[Dict[str, pd.DataFrame]],
) -> Optional[Dict[str, pd.DataFrame]]:
    """特征分区加载 + 四族运行时派生补齐（D8 提取；守卫与日志文本逐字保留）。

    Returns:
        features_by_date；None = 旧侧早退（``return {}``）触发条件（无特征数据）。
    """
    features_by_date = {}
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            features_by_date[trade_date] = features

    # 股东增减持因子为**运行时派生**（cs_train 不含本族列）：回测复用同一张查询表
    # 逐日补齐，否则 MLSignal 预测将因缺列而静默补 NaN（train/serve 偏差）。
    if holdertrade_lookup:
        _derive_feature_family(
            features_by_date, holdertrade_lookup, derive_holdertrade_columns, "股东增减持"
        )

    # 股票回购因子同为运行时派生（cs_train 不含本族列）：复用同一张查询表逐日补齐
    if repurchase_lookup:
        _derive_feature_family(
            features_by_date, repurchase_lookup, derive_repurchase_columns, "股票回购"
        )

    # 十大流通股东因子同为运行时派生（cs_train 不含本族列）：复用同一份报告期面板
    if top10fh_panel is not None and len(top10fh_panel) > 0:
        _derive_feature_family(
            features_by_date, top10fh_panel, derive_top10fh_columns, "十大流通股东"
        )

    # 龙虎榜机构席位因子同为运行时派生（cs_train 不含本族列）：复用同一张查询表逐日补齐
    if top_inst_lookup:
        _derive_feature_family(
            features_by_date, top_inst_lookup, derive_top_inst_columns, "龙虎榜机构席位"
        )

    if not features_by_date:
        logger.warning(f"OOS回测: 无特征数据 {bt_start}~{bt_end}，跳过")
        return None
    return features_by_date


def _build_replay_trading_config(
    model_version: int,
    label_column: str,
    bt_top_n: int,
    bt_rebalance_freq: Optional[int],
    bt_exclude_st: bool,
    bt_min_list_days: int,
    bt_sell_timing: str,
    bt_max_weight_per_stock: Optional[float],
    bt_max_per_industry: Optional[int],
    bt_stop_loss_enabled: bool,
    bt_stop_loss_drawdown_pct: float,
    bt_stop_loss_consecutive_limit_down: int,
    position_sizing: str,
    kelly_vol_window: int,
    kelly_max_leverage: float,
    stagger_tranches: int,
    enable_early_rebalance_on_empty: bool,
    initial_capital: float,
) -> TradingConfig:
    """回测侧 TradingConfig 装配（D8 提取自 run_oos_backtest；参数映射逐字保留）。"""
    return TradingConfig(
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


def _assemble_replay_engine(
    effective_config: TradingConfig,
    stock_basic: pd.DataFrame,
    stock_domain: str,
    data_root: str,
    persistent_signal,
    features_by_date: Dict[str, pd.DataFrame],
    storage: Storage,
    initial_capital: float,
    bt_sell_timing: str,
    bt_exclude_st: bool,
    bt_min_list_days: int,
):
    """universe / signal / 引擎装配（D8 提取自 run_oos_backtest；调用顺序逐字保留）。

    侧效应：``engine.record_holdings_snapshot = True``（快照是 runs 产物数据源）。
    """
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
    # D6 摘除：旧侧此处的 engine.set_exposure_table(...)（None = 不启用，B0 下 no-op
    # 守卫）与 engine.set_exposure_policy(...) 条件调用已删除；exposure 统计日志块
    # 同步删除（exposure 引擎属性不再存在）。
    return engine


def _compute_replay_metrics(
    nav_curve: pd.DataFrame,
    bt_start: str,
    bt_end: str,
    bt_top_n: int,
    split_num: Optional[int],
) -> Dict:
    """组合级绩效指标计算与结果日志（D8 提取自 run_oos_backtest 尾段，逐字保留）。"""
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
    logger.info(
        f"\n{'#' * 80}\n"
        f"{split_tag}{bt_start}-{bt_end}\n"
        f"OOS回测结果: 总收益={total_return*100:.2f}%, "
        f"年化={annual_return*100:.2f}%, "
        f"最大回撤={max_drawdown*100:.2f}%, "
        f"夏普={sharpe:.2f} \n"
        f"{'#' * 80}\n\n"
    )
    return metrics


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
    _fail_fast_retired_exposure(
        exposure_table,
        exposure_policy,
        exposure_replenish,
        exposure_trim_tolerance,
        exposure_budget_discount_replenish,
    )
    data_root = data_root or get_data_root()
    logger.info(f"OOS 回测: {bt_start} ~ {bt_end}（模型 v{model_version}, Top{bt_top_n}）")

    price_data = _load_replay_price_data(loader, bt_start, bt_end)
    if price_data is None:
        return {}

    trade_dates = trade_cal[
        (trade_cal["cal_date"] >= bt_start)
        & (trade_cal["cal_date"] <= bt_end)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    features_by_date = _load_replay_features(
        storage,
        trade_dates,
        bt_start,
        bt_end,
        holdertrade_lookup,
        repurchase_lookup,
        top10fh_panel,
        top_inst_lookup,
    )
    if features_by_date is None:
        return {}

    logger.info(f"OOS回测数据: 日线={len(price_data)}条, 特征={len(features_by_date)}日")
    effective_config = _build_replay_trading_config(
        model_version,
        label_column,
        bt_top_n,
        bt_rebalance_freq,
        bt_exclude_st,
        bt_min_list_days,
        bt_sell_timing,
        bt_max_weight_per_stock,
        bt_max_per_industry,
        bt_stop_loss_enabled,
        bt_stop_loss_drawdown_pct,
        bt_stop_loss_consecutive_limit_down,
        position_sizing,
        kelly_vol_window,
        kelly_max_leverage,
        stagger_tranches,
        enable_early_rebalance_on_empty,
        initial_capital,
    )

    engine = _assemble_replay_engine(
        effective_config,
        stock_basic,
        stock_domain,
        data_root,
        persistent_signal,
        features_by_date,
        storage,
        initial_capital,
        bt_sell_timing,
        bt_exclude_st,
        bt_min_list_days,
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

    metrics = _compute_replay_metrics(nav_curve, bt_start, bt_end, bt_top_n, split_num)
    metrics["_nav_curve"] = nav_curve
    metrics["_trades"] = engine.get_trades()
    metrics["_execution_attribution"] = engine.get_execution_attribution()
    metrics["_holdings_snapshot"] = engine.get_holdings_snapshot()
    return metrics
