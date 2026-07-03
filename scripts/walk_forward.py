#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-forward 滚动训练脚本

功能：
- 按指定频率（季度/月度/半年度）滚动训练模型
- 每次训练使用固定窗口的历史数据（默认5年）
- 在紧随其后的样本外窗口进行评估（默认6个月）
- 生成多段 OOS 表现序列
- 复用 train_ml_model.py 的所有能力（训练、评估、模型注册、日志记录等）

使用示例：
    # 使用默认参数（季度滚动，5年训练窗口，6个月测试窗口）
    python scripts/walk_forward.py --split-count 12 --final-date 20231231
    
    # 指定按月度滚动
    python scripts/walk_forward.py --split-count 24 --final-date 20231231 --step monthly
    
    # 自定义窗口大小
    python scripts/walk_forward.py --split-count 12 --final-date 20231231 \
        --train-window-years 3 --test-window-months 3
    
    # 透传训练参数
    python scripts/walk_forward.py --split-count 12 --final-date 20231231 \
        --task classification --pos-topk 300 --label y_ret_20
"""

import argparse
import copy
import gc
import re
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.lazybull.common.config import get_data_root, get_models_root
from src.lazybull.common.backtest_runtime import (
    create_backtest_engine_from_config,
    create_or_reuse_signal,
    infer_rebalance_freq_from_label,
    persist_signal_quality_state,
    restore_signal_quality_state,
)
from src.lazybull.common.logger import setup_logger
from src.lazybull.common.trading_config import TradingConfig
from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.train_core import (
    load_features_data,
    prepare_training_data,
    transform_labels_cs_zscore,
    generate_classification_labels,
    train_xgboost_model,
    train_lightgbm_model,
    evaluate_validation_daily,
    build_rank_sample_weights,
    build_time_decay_weights,
)
from src.lazybull.ml.walk_forward_utils import (
    generate_walk_forward_splits_by_count,
    print_splits_summary,
    resolve_deploy_train_window,
    WalkForwardSplit
)
from src.lazybull.ml.ensemble import EnsembleModel, TreeLimitedModel
from src.lazybull.ml.run_logger import (
    TrainingRunRecord,
    write_training_run_to_csv
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")
# test 期延伸到数据末尾时，标签列（如 y_ret_20）在最近 N 个交易日全为 NaN，concat 时触发此警告
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame concatenation with empty or all-NA entries.*")


ADAPTIVE_LOW_BEST_ITER_THRESHOLD = 50
ADAPTIVE_LOW_BEST_ITER_MAX_RETRIES = 10
ADAPTIVE_HIT_CAP_RATIO = 0.90
ADAPTIVE_REPLACEMENT_TOP30_WEIGHT = 0.70
ADAPTIVE_REPLACEMENT_RANKIC_IR_WEIGHT = 0.30
ADAPTIVE_REPLACEMENT_MIN_SCORE = 0.00
SEED_ENSEMBLE_KEEP_TOP_RATIO = 0.30
SEED_ENSEMBLE_KEEP_MIN_MODELS = 3
POSTERIOR_TREE_AUTO_GRID = [
    8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096,
]


def _build_main_board_codes(stock_basic: pd.DataFrame) -> set:
    """从 stock_basic 构建主板股票代码集合。"""
    if stock_basic is None or len(stock_basic) == 0:
        raise ValueError("stock_basic 为空，无法构建主板股票池")
    if "ts_code" not in stock_basic.columns or "market" not in stock_basic.columns:
        raise ValueError("stock_basic 缺少 ts_code/market 列，无法做主板过滤")

    board_df = stock_basic[stock_basic["market"] == "主板"]
    board_codes = set(board_df["ts_code"].astype(str).tolist())
    if not board_codes:
        raise ValueError("stock_basic 中 market=主板 的股票为空，无法做主板过滤")
    return board_codes


def _filter_to_main_board(df: pd.DataFrame, main_board_codes: set, stage: str) -> pd.DataFrame:
    """按主板股票池过滤样本，确保训练/评估与交易口径一致。"""
    if df is None or len(df) == 0:
        return df
    if "ts_code" not in df.columns:
        raise ValueError(f"{stage} 数据缺少 ts_code 列，无法做主板过滤")

    before = len(df)
    filtered = df[df["ts_code"].astype(str).isin(main_board_codes)].copy()
    after = len(filtered)
    logger.info(f"{stage} 主板过滤: {before} -> {after}（移除 {before - after}）")
    if after == 0:
        raise ValueError(f"{stage} 主板过滤后样本为空，请检查数据与股票池配置")
    return filtered


def _normalize_selected_split_indices(raw_indices: Optional[List[int]]) -> List[int]:
    """规范化 split 下标列表：去重保序，且要求非负整数。"""
    if not raw_indices:
        return []

    normalized: List[int] = []
    seen = set()
    for raw_index in raw_indices:
        split_index = int(raw_index)
        if split_index < 0:
            raise ValueError(f"selected_split_indices 仅支持非负整数，收到: {split_index}")
        if split_index in seen:
            continue
        seen.add(split_index)
        normalized.append(split_index)
    return normalized


def _filter_splits_by_selected_indices(
    splits: List[WalkForwardSplit],
    selected_split_indices: List[int],
) -> List[WalkForwardSplit]:
    """按指定 split 下标过滤切分；为空时返回原列表。"""
    if not selected_split_indices:
        return splits

    existing_indices = {split.split_index for split in splits}
    missing_indices = [index for index in selected_split_indices if index not in existing_indices]
    if missing_indices:
        raise ValueError(
            f"selected_split_indices 包含不存在的 split 下标: {missing_indices}"
        )

    selected_index_set = set(selected_split_indices)
    return [split for split in splits if split.split_index in selected_index_set]


def summarize_ma250_signal_coverage(
    trade_dates: List[str],
    features_by_date: Dict[str, pd.DataFrame],
    threshold: float,
    rebalance_freq: int,
    stagger_tranches: int = 1,
) -> Dict[str, object]:
    """统计 MA250 阈值在交易日与调仓信号日上的命中情况。"""
    if rebalance_freq <= 0:
        raise ValueError(f"rebalance_freq 必须为正整数，当前值: {rebalance_freq}")

    sorted_trade_dates = sorted(trade_dates)

    def get_ratio(trade_date: str) -> Optional[float]:
        features_df = features_by_date.get(trade_date)
        if features_df is None or len(features_df) == 0:
            return None
        if "mkt_ma250_ratio" not in features_df.columns:
            return None
        ratio = features_df["mkt_ma250_ratio"].iloc[0]
        if pd.isna(ratio):
            return None
        return float(ratio)

    hit_trade_dates = []
    for trade_date in sorted_trade_dates:
        ratio = get_ratio(trade_date)
        if ratio is not None and ratio < threshold:
            hit_trade_dates.append((trade_date, ratio))

    if stagger_tranches <= 1:
        signal_dates = sorted_trade_dates[::rebalance_freq]
    else:
        offset = max(1, rebalance_freq // stagger_tranches)
        signal_date_set = set()
        for tranche_idx in range(stagger_tranches):
            start_idx = tranche_idx * offset
            signal_date_set.update(sorted_trade_dates[start_idx::rebalance_freq])
        signal_dates = sorted(signal_date_set)

    hit_signal_dates = []
    for trade_date in signal_dates:
        ratio = get_ratio(trade_date)
        if ratio is not None and ratio < threshold:
            hit_signal_dates.append((trade_date, ratio))

    first_hit_trade_date, first_hit_trade_ratio = (
        hit_trade_dates[0] if hit_trade_dates else (None, None)
    )
    first_hit_signal_date, first_hit_signal_ratio = (
        hit_signal_dates[0] if hit_signal_dates else (None, None)
    )

    return {
        "trade_days": len(sorted_trade_dates),
        "signal_days": len(signal_dates),
        "hit_trade_days": len(hit_trade_dates),
        "hit_signal_days": len(hit_signal_dates),
        "first_hit_trade_date": first_hit_trade_date,
        "first_hit_trade_ratio": first_hit_trade_ratio,
        "first_hit_signal_date": first_hit_signal_date,
        "first_hit_signal_ratio": first_hit_signal_ratio,
    }


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
    signal_confidence_gate_enabled: bool = False,
    signal_confidence_gate_top_k: int = 10,
    signal_confidence_gate_thresholds: Optional[List[float]] = None,
    signal_confidence_gate_exposure_levels: Optional[List[float]] = None,
    signal_gate_mode: str = "legacy",
    signal_gate_cost_multiplier: float = 2.0,
    signal_gate_round_trip_cost: float = 0.003,
    signal_gate_quality_enabled: bool = False,
    signal_gate_quality_window: int = 5,
    signal_gate_quality_threshold: float = 0.4,
    signal_gate_quality_halflife: int = 3,
    signal_gate_percentile_warmup: int = 20,
    signal_gate_dynamic_topn: bool = False,
    signal_gate_topn_high_multiplier: float = 0.6,
    signal_gate_topn_low_multiplier: float = 1.5,
    holding_bonus_enabled: bool = False,
    holding_bonus_sigma: float = 0.5,
    bt_exclude_st: bool = True,
    bt_min_list_days: int = 365,
    bt_sell_timing: str = "open",
    bt_max_weight_per_stock: Optional[float] = None,
    bt_max_per_industry: Optional[int] = None,
    bt_stop_loss_enabled: bool = False,
    bt_stop_loss_drawdown_pct: float = 30.0,
    bt_stop_loss_trailing_enabled: bool = False,
    bt_stop_loss_trailing_pct: float = 15.0,
    bt_stop_loss_consecutive_limit_down: int = 2,
    bt_equity_curve_enabled: bool = False,
    bt_equity_curve_drawdown_thresholds: Optional[List[float]] = None,
    bt_equity_curve_exposure_levels: Optional[List[float]] = None,
    bt_equity_curve_ma_short: int = 5,
    bt_equity_curve_ma_long: int = 20,
    bt_equity_curve_recovery_mode: str = "gradual",
    bt_equity_curve_recovery_step: float = 0.25,
    bt_equity_curve_recovery_delay_periods: int = 0,
    market_regime_enabled: bool = False,
    market_regime_mode: str = "binary",
    market_regime_bear_threshold: float = -0.02,
    market_regime_bear_exposure: float = 0.3,
    market_regime_vol_target: float = 0.15,
    market_regime_trend_threshold: float = 1.0,
    market_regime_min_exposure: float = 0.2,
    market_regime_combine_method: str = "min",
    market_regime_trend_guard: bool = True,
    market_regime_drawdown_guard: bool = True,
    market_regime_drawdown_threshold: float = -0.08,
    market_regime_ma250_hard_stop: bool = False,
    market_regime_ma250_threshold: float = 1.0,
    market_regime_ma250_exposure: float = 0.0,
    market_regime_ma250_atr_scaling: bool = False,
    industry_momentum_filter: bool = False,
    industry_momentum_bottom_pct: float = 0.2,
    industry_rotation_enhanced: bool = False,
    industry_rotation_alpha: float = 0.3,
    position_sizing: str = "equal",
    kelly_vol_window: int = 60,
    kelly_max_leverage: float = 0.25,
    stagger_tranches: int = 1,
    enable_profit_based_holding: bool = False,
    early_exit_loss_threshold: float = -0.05,
    early_exit_holding_ratio: float = 0.6,
    profit_extension_threshold: float = 0.05,
    profit_extension_days: int = 5,
    profit_extension_mode: str = "pnl",
    profit_extension_strength_threshold: float = 0.6,
    profit_extension_strength_weights: Optional[Dict[str, float]] = None,
    use_atr_for_early_exit: bool = False,
    atr_multiplier: float = 2.0,
    early_exit_mode: str = "disabled",
    early_exit_strength_protect_threshold: float = 0.55,
    early_exit_max_reprieves: int = 2,
    time_stop_loss_enabled: bool = True,
    time_stop_loss_days: int = 15,
    time_stop_loss_profit_ratio: float = -0.02,
    weakness_exit_enabled: bool = False,
    weakness_exit_threshold: float = 0.6,
    weakness_exit_consecutive_days: int = 3,
    weakness_exit_min_holding_days: int = 5,
    weakness_exit_weights: str = "30,25,25,20",
    weakness_exit_industry_filter: bool = False,
    weakness_exit_industry_bottom_pct: float = 0.3,
    take_profit_threshold: Optional[float] = None,
    take_profit_refill: bool = True,
    enable_early_rebalance_on_empty: bool = True,
    initial_capital: float = 1000000.0,
    split_num: Optional[int] = None,
) -> Dict:
    """对单个 split 模型运行 OOS 回测，返回组合级绩效指标

    Args:
        model_version: 刚注册的模型版本号
        bt_start: 回测起始日期 YYYYMMDD
        bt_end: 回测结束日期 YYYYMMDD
        storage: Storage 实例
        loader: DataLoader 实例
        trade_cal: 交易日历 DataFrame
        stock_basic: 股票基本信息 DataFrame
        label_column: 标签列名（用于自动推断调仓频率）
        bt_top_n: 回测 Top N 持仓数
        bt_rebalance_freq: 调仓频率（None 则从 label 自动推断）
        data_root: 数据根目录
        stagger_tranches: 分批调仓批次数（1=不分批）

    Returns:
        回测指标字典，键以 bt_ 前缀开头；无数据时返回空字典
    """
    from src.lazybull.universe import BasicUniverse

    data_root = data_root or get_data_root()

    logger.info(f"OOS 回测: {bt_start} ~ {bt_end}（模型 v{model_version}, Top{bt_top_n}）")

    # 1. 加载日线数据
    daily_data = loader.load_clean_daily(bt_start, bt_end)
    if daily_data is None or len(daily_data) == 0:
        logger.warning(f"OOS回测: 无法加载 {bt_start}~{bt_end} 日线数据，跳过")
        return {}

    # 2. 准备价格数据
    desired_cols = [
        'ts_code', 'trade_date', 'close', 'close_adj', 'open', 'open_adj',
        'is_suspended', 'is_limit_up', 'is_limit_down', 'vol', 'pct_chg',
        'is_st', 'list_days', 'tradable'
    ]
    existing_cols = [c for c in desired_cols if c in daily_data.columns]
    price_data = daily_data[existing_cols].copy()

    if 'close' not in price_data.columns:
        logger.warning("OOS回测: 缺少 close 列，跳过")
        return {}

    # 3. 加载特征数据
    trade_dates = trade_cal[
        (trade_cal['cal_date'] >= bt_start) &
        (trade_cal['cal_date'] <= bt_end) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()

    features_by_date = {}
    for td in trade_dates:
        features = storage.load_cs_train_day(td)
        if features is not None and len(features) > 0:
            features_by_date[td] = features

    if len(features_by_date) == 0:
        logger.warning(f"OOS回测: 无特征数据 {bt_start}~{bt_end}，跳过")
        return {}

    logger.info(f"OOS回测数据: 日线={len(daily_data)}条, 特征={len(features_by_date)}日")

    effective_config = TradingConfig(
        model_version=model_version,
        top_n=bt_top_n,
        signal_confidence_gate_enabled=signal_confidence_gate_enabled,
        signal_confidence_gate_top_k=signal_confidence_gate_top_k,
        signal_confidence_gate_thresholds=(
            signal_confidence_gate_thresholds or [0.8, 1.2, 1.6]
        ),
        signal_confidence_gate_exposure_levels=(
            signal_confidence_gate_exposure_levels or [0.3, 0.6, 1.0]
        ),
        signal_gate_mode=signal_gate_mode,
        signal_gate_cost_multiplier=signal_gate_cost_multiplier,
        signal_gate_round_trip_cost=signal_gate_round_trip_cost,
        signal_gate_quality_enabled=signal_gate_quality_enabled,
        signal_gate_quality_window=signal_gate_quality_window,
        signal_gate_quality_threshold=signal_gate_quality_threshold,
        signal_gate_quality_halflife=signal_gate_quality_halflife,
        signal_gate_percentile_warmup=signal_gate_percentile_warmup,
        signal_gate_dynamic_topn=signal_gate_dynamic_topn,
        signal_gate_topn_high_multiplier=signal_gate_topn_high_multiplier,
        signal_gate_topn_low_multiplier=signal_gate_topn_low_multiplier,
        holding_bonus_enabled=holding_bonus_enabled,
        holding_bonus_sigma=holding_bonus_sigma,
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
        stop_loss_trailing_enabled=bt_stop_loss_trailing_enabled,
        stop_loss_trailing_pct=bt_stop_loss_trailing_pct,
        stop_loss_consecutive_limit_down=bt_stop_loss_consecutive_limit_down,
        equity_curve_enabled=bt_equity_curve_enabled,
        equity_curve_drawdown_thresholds=(
            bt_equity_curve_drawdown_thresholds or [5.0, 10.0, 15.0, 20.0]
        ),
        equity_curve_exposure_levels=(
            bt_equity_curve_exposure_levels or [0.8, 0.6, 0.4, 0.2]
        ),
        equity_curve_ma_short=bt_equity_curve_ma_short,
        equity_curve_ma_long=bt_equity_curve_ma_long,
        equity_curve_recovery_mode=bt_equity_curve_recovery_mode,
        equity_curve_recovery_step=bt_equity_curve_recovery_step,
        equity_curve_recovery_delay_periods=bt_equity_curve_recovery_delay_periods,
        market_regime_enabled=market_regime_enabled,
        market_regime_mode=market_regime_mode,
        market_regime_bear_threshold=market_regime_bear_threshold,
        market_regime_bear_exposure=market_regime_bear_exposure,
        market_regime_vol_target=market_regime_vol_target,
        market_regime_trend_threshold=market_regime_trend_threshold,
        market_regime_min_exposure=market_regime_min_exposure,
        market_regime_combine_method=market_regime_combine_method,
        market_regime_trend_guard=market_regime_trend_guard,
        market_regime_drawdown_guard=market_regime_drawdown_guard,
        market_regime_drawdown_threshold=market_regime_drawdown_threshold,
        market_regime_ma250_hard_stop=market_regime_ma250_hard_stop,
        market_regime_ma250_threshold=market_regime_ma250_threshold,
        market_regime_ma250_exposure=market_regime_ma250_exposure,
        market_regime_ma250_atr_scaling=market_regime_ma250_atr_scaling,
        industry_momentum_filter=industry_momentum_filter,
        industry_momentum_bottom_pct=industry_momentum_bottom_pct,
        industry_rotation_enhanced=industry_rotation_enhanced,
        industry_rotation_alpha=industry_rotation_alpha,
        position_sizing=position_sizing,
        kelly_vol_window=kelly_vol_window,
        kelly_max_leverage=kelly_max_leverage,
        enable_profit_based_holding=enable_profit_based_holding,
        early_exit_loss_threshold=early_exit_loss_threshold,
        early_exit_holding_ratio=early_exit_holding_ratio,
        profit_extension_threshold=profit_extension_threshold,
        profit_extension_days=profit_extension_days,
        profit_extension_mode=profit_extension_mode,
        profit_extension_strength_threshold=profit_extension_strength_threshold,
        profit_extension_strength_weights=profit_extension_strength_weights,
        use_atr_for_early_exit=use_atr_for_early_exit,
        atr_multiplier=atr_multiplier,
        early_exit_mode=early_exit_mode,
        early_exit_strength_protect_threshold=early_exit_strength_protect_threshold,
        early_exit_max_reprieves=early_exit_max_reprieves,
        take_profit_threshold=take_profit_threshold,
        take_profit_refill=take_profit_refill,
        time_stop_loss_enabled=time_stop_loss_enabled,
        time_stop_loss_days=time_stop_loss_days,
        time_stop_loss_profit_ratio=time_stop_loss_profit_ratio,
        weakness_exit_enabled=weakness_exit_enabled,
        weakness_exit_threshold=weakness_exit_threshold,
        weakness_exit_consecutive_days=weakness_exit_consecutive_days,
        weakness_exit_min_holding_days=weakness_exit_min_holding_days,
        weakness_exit_weights=weakness_exit_weights,
        weakness_exit_industry_filter=weakness_exit_industry_filter,
        weakness_exit_industry_bottom_pct=weakness_exit_industry_bottom_pct,
        initial_capital=initial_capital,
        sell_price=bt_sell_timing,
    )

    # 4. 创建回测组件
    universe = BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=bt_exclude_st,
        min_list_days=bt_min_list_days,
        markets=['主板'],
        verbose=False,
    )

    signal = create_or_reuse_signal(
        effective_config,
        data_root=data_root,
        persistent_signal=persistent_signal,
        verbose=False,
    )

    # 自动推断调仓频率
    bt_rebalance_freq = effective_config.rebalance_freq

    if market_regime_ma250_hard_stop:
        ma250_stats = summarize_ma250_signal_coverage(
            trade_dates=trade_dates,
            features_by_date=features_by_date,
            threshold=market_regime_ma250_threshold,
            rebalance_freq=bt_rebalance_freq,
            stagger_tranches=stagger_tranches,
        )
        logger.info(
            "MA250硬条件统计: "
            f"threshold={market_regime_ma250_threshold}, "
            f"交易日命中 {ma250_stats['hit_trade_days']}/{ma250_stats['trade_days']}, "
            f"调仓信号日命中 {ma250_stats['hit_signal_days']}/{ma250_stats['signal_days']}"
        )
        if ma250_stats["first_hit_signal_date"] is not None:
            logger.info(
                "MA250硬条件首次命中调仓信号日: "
                f"{ma250_stats['first_hit_signal_date']} "
                f"(mkt_ma250_ratio={ma250_stats['first_hit_signal_ratio']:.3f})"
            )
        elif ma250_stats["first_hit_trade_date"] is not None:
            logger.warning(
                "MA250硬条件在当前 OOS 窗口仅命中过普通交易日，"
                f"未命中调仓信号日；首次交易日命中为 {ma250_stats['first_hit_trade_date']} "
                f"(mkt_ma250_ratio={ma250_stats['first_hit_trade_ratio']:.3f})"
            )
        elif ma250_stats["signal_days"] > 0:
            logger.warning(
                "MA250硬条件已开启，但当前 OOS 窗口没有任何调仓信号日命中；"
                "回测结果可能与关闭时接近"
            )

    # 5. 运行回测
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

    # 从持久化 signal 恢复质量监控状态（跨 split 积累，避免每次重置预热期）
    if persistent_signal is not None:
        restore_signal_quality_state(engine, persistent_signal)

    trading_dates_ts = [pd.Timestamp(d) for d in trade_dates]

    nav_curve = engine.run(
        start_date=pd.Timestamp(bt_start),
        end_date=pd.Timestamp(bt_end),
        trading_dates=trading_dates_ts,
        price_data=price_data
    )

    # 回测结束后将质量监控状态保存到持久化 signal，供下一个 split 继续
    if persistent_signal is not None:
        persist_signal_quality_state(engine, persistent_signal)

    confidence_gate_stats = engine.get_confidence_gate_stats()

    # 6. 提取绩效指标
    if nav_curve is None or nav_curve.empty or 'nav' not in nav_curve.columns:
        logger.warning("OOS回测: 净值曲线为空，跳过")
        return {}

    total_return = nav_curve['return'].iloc[-1]
    nav_values = nav_curve['nav'].values

    cummax = pd.Series(nav_values).cummax()
    drawdown = (pd.Series(nav_values) - cummax) / cummax
    max_drawdown = drawdown.min()

    trading_days = len(nav_curve)
    years = trading_days / 252
    # 简单年化收益率（不假设收益再投入）
    annual_return = (total_return / years) if years > 0 else 0

    daily_returns = nav_curve['nav'].pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5)

    risk_free_rate = 0.03
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
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
        "bt_signal_confidence_signal_days": confidence_gate_stats["signal_days"],
        "bt_signal_confidence_blocked_days": confidence_gate_stats["blocked_days"],
        "bt_signal_confidence_block_rate": round(confidence_gate_stats["block_rate"], 6),
        "bt_signal_confidence_avg_exposure": round(
            confidence_gate_stats["avg_exposure"], 6
        ),
        "bt_signal_confidence_avg_score": round(confidence_gate_stats["avg_score"], 6),
    }

    if market_regime_enabled:
        metrics["bt_market_regime"] = True
        metrics["bt_market_regime_mode"] = market_regime_mode

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

    # 附带 nav_curve 用于串联全周期净值
    metrics["_nav_curve"] = nav_curve

    return metrics


# ── 多偏移集成辅助函数 ────────────────────────────────────────────────


def _align_to_trade_date(
    date_str: str, trade_dates: List[str], forward: bool = True
) -> str:
    """将日期对齐到最近的交易日

    Args:
        date_str: 日期字符串 YYYYMMDD
        trade_dates: 已排序的交易日列表
        forward: True=向后找（最近的>=date），False=向前找（最近的<=date）

    Returns:
        对齐后的交易日字符串
    """
    if forward:
        for td in trade_dates:
            if td >= date_str:
                return td
        return trade_dates[-1]
    else:
        for td in reversed(trade_dates):
            if td <= date_str:
                return td
        return trade_dates[0]


def compute_offset_windows(
    train_start: str,
    train_end: str,
    offset_months: int,
    trade_cal: pd.DataFrame,
) -> List[Tuple[str, str]]:
    """计算多偏移训练窗口

    生成原始窗口 + 前移 + 后移共3个训练窗口，用于多偏移集成训练。

    Args:
        train_start: 原始训练起始日期 YYYYMMDD
        train_end: 原始训练结束日期 YYYYMMDD
        offset_months: 偏移月数（如1表示±1个月）
        trade_cal: 交易日历 DataFrame

    Returns:
        [(start, end), ...] — 原始窗口 + 前移 + 后移，共3个
    """
    all_dates = (
        trade_cal[trade_cal["is_open"] == 1]["cal_date"].sort_values().tolist()
    )
    windows = [(train_start, train_end)]

    for sign in [-1, 1]:
        start_dt = datetime.strptime(train_start, "%Y%m%d") + relativedelta(
            months=sign * offset_months
        )
        end_dt = datetime.strptime(train_end, "%Y%m%d") + relativedelta(
            months=sign * offset_months
        )
        start_aligned = _align_to_trade_date(
            start_dt.strftime("%Y%m%d"), all_dates, forward=True
        )
        end_aligned = _align_to_trade_date(
            end_dt.strftime("%Y%m%d"), all_dates, forward=False
        )
        windows.append((start_aligned, end_aligned))

    return windows


def _train_model_on_window(
    train_start: str,
    train_end: str,
    storage: Storage,
    loader: DataLoader,
    args,
    main_board_codes: set,
    random_state_override: Optional[int] = None,
) -> Dict:
    """在指定训练窗口上训练单个模型

    封装 load_features_data → 标签变换 → prepare_training_data →
    样本权重 → 训练 的完整流程。

    Args:
        train_start: 训练起始日期
        train_end: 训练结束日期
        storage: Storage 实例
        loader: DataLoader 实例
        args: 命令行参数

    Returns:
        包含 model, feature_columns, train_params 等的字典
    """
    # 1. 加载训练数据
    df_train, train_days_count = load_features_data(
        storage, loader, train_start, train_end
    )
    df_train = _filter_to_main_board(df_train, main_board_codes, "训练窗口")
    total_train_samples = len(df_train)

    # 2. 应用标签变换
    if args.task == "classification":
        if args.pos_quantile is None and args.pos_topk is None:
            raise ValueError("分类任务必须指定 --pos-quantile 或 --pos-topk")
        df_train = generate_classification_labels(
            df_train,
            label_column=args.label_column,
            pos_quantile=args.pos_quantile,
            pos_topk=args.pos_topk,
        )
        binary_label_col = f"{args.label_column}_binary"
        actual_label_column = binary_label_col
    else:
        actual_label_column = args.label_column

    # 3. 准备训练数据（按 trade_date 粒度切分训练集/验证集）
    label_transform_fn = None
    if args.task == "regression" and args.label_transform == "cs_zscore":
        label_transform_fn = lambda d: transform_labels_cs_zscore(
            d, label_column=actual_label_column, winsorize_p=args.winsorize_p
        )
    (
        X_train, y_train, X_val, y_val,
        feature_columns, df_train_split, df_val_split,
        data_stats, df_val_split_original,
    ) = prepare_training_data(
        df_train,
        actual_label_column,
        val_ratio=args.val_ratio,
        label_transform_fn=label_transform_fn,
        enable_fundamental_features=args.enable_fundamental_features,
        enable_alt_features=args.enable_alt_features,
        enable_margin_features=args.enable_margin_features,
        enable_cyq_features=args.enable_cyq_features,
        enable_fund_features=args.enable_fund_features,
        enable_express_features=args.enable_express_features,
        enable_enhanced_features=getattr(args, "enable_enhanced_features", False),
        enable_north_features=getattr(args, "enable_north_features", False),
        enable_lhb_features=getattr(args, "enable_lhb_features", False),
        enable_consensus_features=getattr(args, "enable_consensus_features", False),
        enable_cashflow_quality_features=getattr(
            args, "enable_cashflow_quality_features", False
        ),
        enable_consensus_revision_features=getattr(
            args, "enable_consensus_revision_features", False
        ),
        feature_stability_filter=args.feature_stability_filter,
        factor_prune=args.factor_prune,
    )

    # 原始 df_train 已不再需要，释放 ~3 GiB 内存
    del df_train
    gc.collect()

    # 4. 构造样本权重（rank-weight + 时间衰减，可叠加）
    rank_sample_weight = None
    if args.rank_weight_enabled:
        rank_sample_weight = build_rank_sample_weights(
            df_train=df_train_split,
            label_column=actual_label_column,
            topk=args.rank_weight_topk,
            top_weight=args.rank_weight,
            topk_weight_mode=getattr(args, "rank_weight_topk_weight_mode", "linear_decay"),
        )
    if args.time_decay_half_life > 0:
        td_weights = build_time_decay_weights(
            df_train=df_train_split,
            half_life_years=args.time_decay_half_life,
        )
        if rank_sample_weight is not None:
            rank_sample_weight = rank_sample_weight * td_weights
        else:
            rank_sample_weight = td_weights

    # 5. 训练模型
    skip_label_winsorize = (
        args.task == "regression" and args.label_transform == "cs_zscore"
    )
    algorithm = getattr(args, "algorithm", "xgboost")
    train_fn = train_lightgbm_model if algorithm == "lightgbm" else train_xgboost_model

    extra_kwargs = {}
    if algorithm == "lightgbm":
        num_leaves_val = getattr(args, "num_leaves", None)
        if num_leaves_val is not None:
            extra_kwargs["num_leaves"] = num_leaves_val
    if algorithm == "xgboost":
        objective_type = getattr(args, "objective", "mse")
        extra_kwargs["objective_type"] = objective_type
        if objective_type == "lambdarank":
            extra_kwargs["df_train_for_group"] = df_train_split
            extra_kwargs["df_val_for_group"] = df_val_split

    es_rounds = args.early_stopping_rounds if args.early_stopping_rounds else None

    model, train_params, train_metrics, val_metrics = train_fn(
        X_train, y_train, X_val, y_val,
        task=args.task,
        skip_label_winsorize=skip_label_winsorize,
        scale_pos_weight=args.scale_pos_weight,
        sample_weight=rank_sample_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=(
            args.random_state if random_state_override is None else random_state_override
        ),
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        early_stopping_rounds=es_rounds,
        early_stopping_metric=args.early_stopping_metric,
        **extra_kwargs,
    )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "train_params": train_params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "df_train_split": df_train_split,
        "df_val_split": df_val_split,
        "df_val_split_original": df_val_split_original,
        "data_stats": data_stats,
        "train_days_count": train_days_count,
        "total_train_samples": total_train_samples,
        "X_train_len": len(X_train),
        "X_val_len": len(X_val),
    }


def _resolve_ensemble_seeds(args) -> List[int]:
    """解析 --ensemble-seeds，返回去重保序的种子列表。

    为空或未提供时回退到 [args.random_state]，保证向后兼容（单种子=原行为）。
    """
    raw = getattr(args, "ensemble_seeds", None)
    if not raw:
        return [args.random_state]
    seeds: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        seed = int(token)
        if seed not in seeds:
            seeds.append(seed)
    return seeds if seeds else [args.random_state]


def _build_ensemble_sub_models(
    windows: List[tuple],
    storage: Storage,
    loader: DataLoader,
    args,
    main_board_codes: set,
    seeds: List[int],
    is_deploy: bool = False,
    topk_values: Optional[List[int]] = None,
    enable_live_adaptive: bool = False,
) -> tuple:
    """对（窗口 × 种子）笛卡尔训练子模型。

    Args:
        windows: [(win_start, win_end), ...] 训练窗口列表
        seeds: 随机种子列表（多种子 bagging）
        is_deploy: 是否为部署模型训练（仅影响日志文案）

    Returns:
        (sub_models, base_result, adaptive_meta)
    """
    sub_models: List = []
    sub_model_records: List[Dict] = []
    base_result: Optional[Dict] = None
    adaptive_meta = {
        "live_adaptive_triggered": False,
        "live_adaptive_trigger_count": 0,
        "live_adaptive_used_count": 0,
        "live_adaptive_last_action": None,
        "live_adaptive_last_best_iteration": None,
        "live_adaptive_retry_count": 0,
        "live_adaptive_last_retry_seed": None,
        "live_adaptive_last_candidate_best_iteration": None,
        "live_adaptive_final_random_state": getattr(args, "random_state", None),
        "live_adaptive_final_learning_rate": getattr(args, "learning_rate", None),
        "live_adaptive_final_n_estimators": getattr(args, "n_estimators", None),
    }
    rolling_args = args
    keep_top_ratio = float(
        getattr(args, "ensemble_seed_keep_top_ratio", SEED_ENSEMBLE_KEEP_TOP_RATIO)
    )
    keep_top_ratio = min(1.0, max(0.01, keep_top_ratio))
    keep_min_models = int(
        getattr(args, "ensemble_seed_keep_min_models", SEED_ENSEMBLE_KEEP_MIN_MODELS)
    )
    keep_min_models = max(1, keep_min_models)
    max_low_iter_retries = max(
        1,
        int(getattr(args, "adaptive_low_iter_max_retries", ADAPTIVE_LOW_BEST_ITER_MAX_RETRIES)),
    )
    total = len(windows) * len(seeds)
    idx = 0
    prefix = "部署" if is_deploy else ""
    for win_idx, (win_start, win_end) in enumerate(windows):
        win_label = ["基础", "前移", "后移"][win_idx] if win_idx < 3 else f"偏移{win_idx}"
        for seed in seeds:
            idx += 1
            seed_note = f" seed={seed}" if len(seeds) > 1 else ""
            logger.info(
                f"{'='*60}\n"
                f"  {prefix}子模型 {idx}/{total}（{win_label}{seed_note}）: "
                f"{win_start} ~ {win_end}\n"
                f"{'='*60}"
            )
            selected_tr = _train_model_on_window(
                win_start, win_end, storage, loader, rolling_args, main_board_codes,
                random_state_override=seed
            )

            if enable_live_adaptive and topk_values is not None:
                best_iter = selected_tr["train_params"].get("best_iteration")
                action = _resolve_adaptive_best_iter_action(
                    best_iter, getattr(rolling_args, "n_estimators", None)
                )
                if action is not None:
                    adaptive_meta["live_adaptive_triggered"] = True
                    adaptive_meta["live_adaptive_trigger_count"] += 1
                    adaptive_meta["live_adaptive_last_action"] = action
                    adaptive_meta["live_adaptive_last_best_iteration"] = best_iter
                    base_val_daily = _evaluate_train_result_val_daily(
                        selected_tr, rolling_args.label_column, rolling_args.task, topk_values,
                        emit_logs=False,
                    )

                    if action == "low_iter":
                        base_seed = int(seed)
                        best_retry_tr = None
                        best_retry_metrics = None
                        best_retry_seed = None
                        best_retry_best_iteration = None
                        last_candidate_best_iteration = best_iter
                        last_retry_seed = None
                        for retry_index in range(1, max_low_iter_retries + 1):
                            retry_seed = _resolve_adaptive_retry_seed(base_seed, retry_index)
                            last_retry_seed = retry_seed
                            adaptive_meta["live_adaptive_retry_count"] = retry_index
                            adaptive_meta["live_adaptive_last_retry_seed"] = retry_seed
                            candidate_args = _build_adaptive_candidate_args(args, action)
                            candidate_args.random_state = retry_seed
                            logger.warning(
                                f"  子模型 {idx}/{total} 触发 split 内低迭代重试: "
                                f"attempt={retry_index}/{max_low_iter_retries}, "
                                f"best_iter={best_iter}, base_seed={base_seed}, retry_seed={retry_seed}, "
                                f"orig_lr={args.learning_rate:.6f}, "
                                f"orig_n_estimators={args.n_estimators}"
                            )

                            challenger_tr = _train_model_on_window(
                                win_start,
                                win_end,
                                storage,
                                loader,
                                candidate_args,
                                main_board_codes,
                                random_state_override=retry_seed,
                            )
                            challenger_best_iteration = challenger_tr["train_params"].get("best_iteration")
                            last_candidate_best_iteration = challenger_best_iteration
                            adaptive_meta[
                                "live_adaptive_last_candidate_best_iteration"
                            ] = challenger_best_iteration
                            challenger_val_daily = _evaluate_train_result_val_daily(
                                challenger_tr,
                                candidate_args.label_column,
                                candidate_args.task,
                                topk_values,
                                emit_logs=False,
                            )

                            logger.warning(
                                f"  子模型 {idx}/{total} 低迭代重试指标对比: "
                                f"attempt={retry_index}, retry_seed={retry_seed}, "
                                f"{_format_adaptive_metric_compare(challenger_val_daily, best_retry_metrics)}"
                            )

                            if _is_better_adaptive_candidate(challenger_val_daily, best_retry_metrics):
                                best_retry_tr = challenger_tr
                                best_retry_metrics = challenger_val_daily
                                best_retry_seed = retry_seed
                                best_retry_best_iteration = challenger_best_iteration

                            base_top30m = _safe_float(base_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            cand_top30m = _safe_float(challenger_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            base_ir = _safe_float(base_val_daily.get('daily_rankic_ir'))
                            cand_ir = _safe_float(challenger_val_daily.get('daily_rankic_ir'))
                            logger.warning(
                                f"  子模型 {idx}/{total} 继续低迭代重试: "
                                f"attempt={retry_index}, retry_seed={retry_seed}, "
                                f"candidate_best_iter={challenger_best_iteration}, "
                                f"base/candidate_top30_med={'nan' if base_top30m is None else f'{base_top30m:.6f}'}/"
                                f"{'nan' if cand_top30m is None else f'{cand_top30m:.6f}'}, "
                                f"base/candidate_val_ir={'nan' if base_ir is None else f'{base_ir:.4f}'}/"
                                f"{'nan' if cand_ir is None else f'{cand_ir:.4f}'}"
                            )

                        adaptive_meta["live_adaptive_final_random_state"] = best_retry_seed
                        adaptive_meta["live_adaptive_final_learning_rate"] = rolling_args.learning_rate
                        adaptive_meta["live_adaptive_final_n_estimators"] = rolling_args.n_estimators
                        adaptive_meta[
                            "live_adaptive_last_candidate_best_iteration"
                        ] = best_retry_best_iteration

                        if best_retry_metrics is not None:
                            best_top30m = _safe_float(best_retry_metrics.get('diagnostic_Top30_逐日均值_50分位'))
                            best_ir = _safe_float(best_retry_metrics.get('daily_rankic_ir'))
                            logger.warning(
                                f"  子模型 {idx}/{total} 低迭代重试最优候选: "
                                f"best_retry_seed={best_retry_seed}, best_retry_best_iter={best_retry_best_iteration}, "
                                f"best_retry_top30_median={'nan' if best_top30m is None else f'{best_top30m:.6f}'}, "
                                f"best_retry_val_ir={'nan' if best_ir is None else f'{best_ir:.4f}'}"
                            )

                        if best_retry_tr is not None and _candidate_passes_adaptive_replacement(
                            base_val_daily, best_retry_metrics
                        ):
                            selected_tr = best_retry_tr
                            adaptive_meta["live_adaptive_used_count"] += 1
                            base_top30m = _safe_float(base_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            cand_top30m = _safe_float(best_retry_metrics.get('diagnostic_Top30_逐日均值_50分位'))
                            base_ir = _safe_float(base_val_daily.get('daily_rankic_ir'))
                            cand_ir = _safe_float(best_retry_metrics.get('daily_rankic_ir'))
                            logger.warning(
                                f"  子模型 {idx}/{total} 采用低迭代重试最优候选模型: "
                                f"retry_seed={best_retry_seed}, "
                                f"base/candidate_top30_med={'nan' if base_top30m is None else f'{base_top30m:.6f}'}/"
                                f"{'nan' if cand_top30m is None else f'{cand_top30m:.6f}'}, "
                                f"base/candidate_val_ir={'nan' if base_ir is None else f'{base_ir:.4f}'}/"
                                f"{'nan' if cand_ir is None else f'{cand_ir:.4f}'}"
                            )
                        else:
                            logger.warning(
                                f"  子模型 {idx}/{total} 低迭代重试完成但未满足替换条件: "
                                f"max_retries={max_low_iter_retries}, "
                                f"last_retry_seed={last_retry_seed}, "
                                f"last_best_iter={last_candidate_best_iteration}"
                            )
                    else:
                        candidate_args = _build_adaptive_candidate_args(args, action)
                        logger.warning(
                            f"  子模型 {idx}/{total} 触发 split 内实时自适应: action={action}, "
                            f"best_iter={best_iter}, orig_lr={args.learning_rate:.6f}, "
                            f"orig_n_estimators={args.n_estimators}, "
                            f"candidate_lr={candidate_args.learning_rate:.6f}, "
                            f"candidate_n_estimators={candidate_args.n_estimators}"
                        )

                        challenger_tr = _train_model_on_window(
                            win_start,
                            win_end,
                            storage,
                            loader,
                            candidate_args,
                            main_board_codes,
                            random_state_override=seed,
                        )

                        challenger_val_daily = _evaluate_train_result_val_daily(
                            challenger_tr,
                            candidate_args.label_column,
                            candidate_args.task,
                            topk_values,
                            emit_logs=False,
                        )

                        if _candidate_passes_adaptive_replacement(base_val_daily, challenger_val_daily):
                            selected_tr = challenger_tr
                            adaptive_meta["live_adaptive_used_count"] += 1
                            base_top30m = _safe_float(base_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            cand_top30m = _safe_float(challenger_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            base_ir = _safe_float(base_val_daily.get('daily_rankic_ir'))
                            cand_ir = _safe_float(challenger_val_daily.get('daily_rankic_ir'))
                            logger.warning(
                                f"  子模型 {idx}/{total} 采用自适应候选模型: action={action}, "
                                f"base/candidate_top30_med={'nan' if base_top30m is None else f'{base_top30m:.6f}'}/"
                                f"{'nan' if cand_top30m is None else f'{cand_top30m:.6f}'}, "
                                f"base/candidate_val_ir={'nan' if base_ir is None else f'{base_ir:.4f}'}/"
                                f"{'nan' if cand_ir is None else f'{cand_ir:.4f}'}"
                            )
                        else:
                            base_top30m = _safe_float(base_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            cand_top30m = _safe_float(challenger_val_daily.get('diagnostic_Top30_逐日均值_50分位'))
                            base_ir = _safe_float(base_val_daily.get('daily_rankic_ir'))
                            cand_ir = _safe_float(challenger_val_daily.get('daily_rankic_ir'))
                            logger.warning(
                                f"  子模型 {idx}/{total} 保留原模型: action={action}, "
                                f"base/candidate_top30_med={'nan' if base_top30m is None else f'{base_top30m:.6f}'}/"
                                f"{'nan' if cand_top30m is None else f'{cand_top30m:.6f}'}, "
                                f"base/candidate_val_ir={'nan' if base_ir is None else f'{base_ir:.4f}'}/"
                                f"{'nan' if cand_ir is None else f'{cand_ir:.4f}'}"
                            )

                        # 仅影响当前 split 尚未启动的后续子模型，不影响其他 split
                        rolling_args = candidate_args
                        adaptive_meta["live_adaptive_final_random_state"] = getattr(
                            rolling_args, "random_state", None
                        )
                        adaptive_meta["live_adaptive_final_learning_rate"] = rolling_args.learning_rate
                        adaptive_meta["live_adaptive_final_n_estimators"] = rolling_args.n_estimators
                        remaining_models = total - idx
                        logger.warning(
                            f"!!! [LIVE-ADAPTIVE] 滚动参数已更新并将用于当前 split 后续 {remaining_models} 个子模型: "
                            f"action={action}, lr={rolling_args.learning_rate:.5f}, "
                            f"n_estimators={rolling_args.n_estimators}"
                        )

            sub_models.append(selected_tr["model"])
            sub_model_records.append({
                "train_result": selected_tr,
            })
            if base_result is None:
                base_result = selected_tr
            elif set(selected_tr["feature_columns"]) != set(base_result["feature_columns"]):
                logger.warning(
                    f"  子模型 {idx} 特征列数量({len(selected_tr['feature_columns'])})"
                    f"与基础模型({len(base_result['feature_columns'])})不一致"
                )

    # 多种子场景下，仅保留排序指标最好的前30%子模型，且至少保留3个
    if len(seeds) > 1 and len(sub_model_records) > 0:
        eval_topk_values = topk_values if topk_values else [30]
        scored_records: List[Dict] = []
        for record in sub_model_records:
            train_result = record["train_result"]
            seed_metrics = _evaluate_train_result_val_daily(
                train_result,
                args.label_column,
                args.task,
                eval_topk_values,
                emit_logs=False,
            )
            score = _adaptive_sort_score(seed_metrics)
            scored_records.append(
                {
                    "train_result": train_result,
                    "seed_metrics": seed_metrics,
                    "score": score,
                }
            )

        scored_records.sort(key=lambda item: item["score"], reverse=True)
        raw_keep_count = int(np.ceil(len(scored_records) * keep_top_ratio))
        keep_count = min(
            len(scored_records),
            max(keep_min_models, raw_keep_count),
        )
        kept_records = scored_records[:keep_count]
        sub_models = [item["train_result"]["model"] for item in kept_records]
        base_result = kept_records[0]["train_result"]

        logger.warning(
            f"{prefix}多种子筛选: total={len(scored_records)}, keep={keep_count}, "
            f"ratio={keep_top_ratio:.0%}, min_keep={keep_min_models}"
        )
        for rank_idx, item in enumerate(kept_records, start=1):
            train_result = item["train_result"]
            metrics = item["seed_metrics"]
            seed_used = train_result["train_params"].get("random_state")
            top30_med = _safe_float(metrics.get("diagnostic_Top30_逐日均值_50分位"))
            val_ir = _safe_float(metrics.get("daily_rankic_ir"))
            logger.warning(
                f"  保留子模型#{rank_idx}: seed={seed_used}, "
                f"top30_median={'nan' if top30_med is None else f'{top30_med:.6f}'}, "
                f"val_ir={'nan' if val_ir is None else f'{val_ir:.4f}'}"
            )

        # 收集保留子模型的 best_iteration（供回测前打印）
        sub_model_best_iters: List[Tuple] = []
        for item in kept_records:
            tr = item["train_result"]
            seed_val = tr["train_params"].get("random_state")
            best_iter_val = tr["train_params"].get("best_iteration")
            sub_model_best_iters.append((seed_val, best_iter_val))
        adaptive_meta["sub_model_best_iterations"] = sub_model_best_iters
    else:
        # 单种子场景也收集 best_iteration
        sub_model_best_iters: List[Tuple] = []
        for record in sub_model_records:
            tr = record["train_result"]
            seed_val = tr["train_params"].get("random_state")
            best_iter_val = tr["train_params"].get("best_iteration")
            sub_model_best_iters.append((seed_val, best_iter_val))
        adaptive_meta["sub_model_best_iterations"] = sub_model_best_iters
    return sub_models, base_result, adaptive_meta


def _evaluate_train_result_val_daily(
    train_result: Dict,
    original_return_col: str,
    task: str,
    topk_values: List[int],
    emit_logs: bool = True,
) -> Dict:
    df_val = train_result.get("df_val_split_original")
    if df_val is None or len(df_val) == 0:
        return {}
    return evaluate_validation_daily(
        model=train_result["model"],
        df_val=df_val,
        feature_columns=train_result["feature_columns"],
        original_return_col=original_return_col,
        task=task,
        topk_values=topk_values,
        emit_logs=emit_logs,
    )


def _copy_args_with_training_overrides(args, **overrides):
    candidate_args = copy.copy(args)
    for key, value in overrides.items():
        setattr(candidate_args, key, value)
    return candidate_args


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def _fmt_metric(value: Optional[float], fmt: str = ".4f") -> str:
    return "nan" if value is None else f"{value:{fmt}}"


def _fmt_pct(value: Optional[float]) -> str:
    return "nan" if value is None else f"{value * 100:.1f}%"


def _metric_value(metrics: Dict, key: str) -> Optional[float]:
    return _safe_float(metrics.get(key))


def _topk_key_metrics(metrics: Dict, topk: int) -> Dict[str, Optional[float]]:
    return {
        "median": _metric_value(metrics, f"diagnostic_Top{topk}_逐日均值_50分位"),
        "mean": _metric_value(metrics, f"top{topk}_return_mean"),
        "lift": _metric_value(metrics, f"diagnostic_Top{topk}_相对全市场提升_均值"),
        "hit_rate": _metric_value(metrics, f"diagnostic_Top{topk}_命中率_日均收益为正"),
        "excess_hit_rate": _metric_value(metrics, f"diagnostic_Top{topk}_超额命中率_跑赢全市场"),
    }


def _print_oos_focus_panel(split_index: int, test_daily_metrics: Dict) -> None:
    """打印 OOS 重点指标面板，避免关键 TopK 信息被普通日志淹没。"""
    top20 = _topk_key_metrics(test_daily_metrics, 20)
    top30 = _topk_key_metrics(test_daily_metrics, 30)
    top20_list = str(test_daily_metrics.get("diagnostic_Top20_最新股票列表") or "")
    top30_list = str(test_daily_metrics.get("diagnostic_Top30_最新股票列表") or "")
    latest_date = str(
        test_daily_metrics.get("diagnostic_Top20_最新日期")
        or test_daily_metrics.get("diagnostic_Top30_最新日期")
        or ""
    )

    logger.opt(colors=True).warning("<cyan><bold>" + "=" * 92 + "</bold></cyan>")
    logger.opt(colors=True).warning(
        f"<cyan><bold>Split {split_index} OOS 重点 TopK 指标</bold></cyan> "
        f"<cyan>(hit rate=TopK逐日平均收益>0占比, list=最新OOS日期 {latest_date})</cyan>"
    )
    logger.opt(colors=True).warning(
        "<yellow><bold>Top20</bold></yellow> | "
        f"hit={_fmt_pct(top20['hit_rate'])} | "
        f"均值中位数={_fmt_metric(top20['median'], '.6f')} | "
        f"超额均值={_fmt_metric(top20['lift'], '.6f')} | "
        f"跑赢全市场={_fmt_pct(top20['excess_hit_rate'])}"
    )
    logger.opt(colors=True).warning(
        "<yellow><bold>Top30</bold></yellow> | "
        f"hit={_fmt_pct(top30['hit_rate'])} | "
        f"均值中位数={_fmt_metric(top30['median'], '.6f')} | "
        f"超额均值={_fmt_metric(top30['lift'], '.6f')} | "
        f"跑赢全市场={_fmt_pct(top30['excess_hit_rate'])}"
    )
    logger.opt(colors=True).warning(f"<yellow>Top20 list:</yellow> {top20_list}")
    logger.opt(colors=True).warning(f"<yellow>Top30 list:</yellow> {top30_list}")
    logger.opt(colors=True).warning("<cyan><bold>" + "=" * 92 + "</bold></cyan>")


def _posterior_metric_score(metrics: Dict, metric_name: str, topk: int) -> Tuple[float, float, float]:
    topk_metrics = _topk_key_metrics(metrics, topk)
    rankic_ir = _metric_value(metrics, "daily_rankic_ir")
    rankic_mean = _metric_value(metrics, "daily_rankic_mean")
    metric_map = {
        "topk_median": topk_metrics["median"],
        "topk_mean": topk_metrics["mean"],
        "topk_lift": topk_metrics["lift"],
        "topk_hit_rate": topk_metrics["hit_rate"],
        "topk_excess_hit_rate": topk_metrics["excess_hit_rate"],
        "rankic_ir": rankic_ir,
        "rankic_mean": rankic_mean,
    }
    primary = metric_map.get(metric_name, topk_metrics["median"])
    return (
        -np.inf if primary is None else primary,
        -np.inf if rankic_ir is None else rankic_ir,
        -np.inf if rankic_mean is None else rankic_mean,
    )


def _build_summary_key_fields(test_daily_metrics: Dict) -> Dict[str, object]:
    top20 = _topk_key_metrics(test_daily_metrics, 20)
    top30 = _topk_key_metrics(test_daily_metrics, 30)
    return {
        "KEY_说明": "重点: hit rate=TopK逐日平均收益>0占比; list=最新OOS日期预测名单",
        "KEY_Top20_list": test_daily_metrics.get("diagnostic_Top20_最新股票列表"),
        "KEY_Top30_list": test_daily_metrics.get("diagnostic_Top30_最新股票列表"),
        "KEY_Top20_hit_rate": top20["hit_rate"],
        "KEY_Top20_avg_return_median": top20["median"],
        "KEY_Top20_lift_mean": top20["lift"],
        "KEY_Top30_hit_rate": top30["hit_rate"],
        "KEY_Top30_avg_return_median": top30["median"],
        "KEY_Top30_lift_mean": top30["lift"],
    }


def _resolve_model_max_trees(model) -> Optional[int]:
    if isinstance(model, TreeLimitedModel):
        return model.max_trees
    if isinstance(model, EnsembleModel):
        child_limits = [_resolve_model_max_trees(child) for child in model.models]
        child_limits = [limit for limit in child_limits if limit is not None]
        return min(child_limits) if child_limits else None

    # XGBoost: 以 booster 的实际训练轮数为准，防止 n_estimators 虚高导致越界
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if hasattr(booster, "num_boosted_rounds"):
                rounds = int(booster.num_boosted_rounds())
                if rounds > 0:
                    return rounds
        except Exception:
            pass

    # LightGBM: 优先读取 booster 当前可用迭代轮数
    if hasattr(model, "booster_"):
        try:
            rounds = int(model.booster_.current_iteration())
            if rounds > 0:
                return rounds
        except Exception:
            pass

    for attr in ("n_estimators", "n_estimators_"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            continue
        if resolved > 0:
            return resolved
    return None


def _wrap_model_with_tree_limit(model, tree_limit: int):
    if isinstance(model, EnsembleModel):
        return EnsembleModel([
            TreeLimitedModel(child, tree_limit, max_trees=_resolve_model_max_trees(child))
            for child in model.models
        ])
    return TreeLimitedModel(model, tree_limit, max_trees=_resolve_model_max_trees(model))


def _resolve_posterior_tree_candidate_limits(
    args,
    max_trees: Optional[int],
    base_best_iteration: Optional[int] = None,
) -> List[int]:
    if max_trees is None or max_trees <= 0:
        return []

    raw_candidates = getattr(args, "posterior_tree_candidates", "") or ""
    candidates: List[int] = []

    if str(raw_candidates).strip():
        for token in str(raw_candidates).split(","):
            token = token.strip()
            if not token:
                continue
            candidate = int(token)
            if candidate <= 0:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    else:
        for candidate in POSTERIOR_TREE_AUTO_GRID:
            if candidate <= max_trees and candidate not in candidates:
                candidates.append(candidate)

    if base_best_iteration is not None and base_best_iteration > 0:
        candidates.append(int(base_best_iteration))
    candidates.append(int(max_trees))

    normalized = sorted({candidate for candidate in candidates if 0 < candidate <= max_trees})
    return normalized


def _select_posterior_tree_model(
    model,
    feature_columns: List[str],
    df_val: pd.DataFrame,
    args,
    topk_values: List[int],
    train_params: Dict,
    model_label: str,
) -> Tuple[object, Dict, Dict]:
    posterior_mode = getattr(args, "posterior_tree_selection_mode", "disabled")
    posterior_metric = getattr(args, "posterior_tree_selection_metric", "topk_median")
    posterior_topk = int(getattr(args, "posterior_tree_selection_topk", 20) or 20)
    base_best_iteration = train_params.get("best_iteration")
    base_best_iteration = int(base_best_iteration) if base_best_iteration is not None else None

    meta = {
        "posterior_tree_selection_mode": posterior_mode,
        "posterior_tree_selection_metric": posterior_metric,
        "posterior_tree_selection_topk": posterior_topk,
        "posterior_tree_selection_enabled": posterior_mode != "disabled",
        "posterior_tree_base_best_iteration": base_best_iteration,
        "posterior_tree_model_max_trees": None,
        "posterior_tree_candidate_limits": [],
        "posterior_tree_selected_limit": base_best_iteration,
        "posterior_tree_selected_top30_median": None,
        "posterior_tree_selected_topk_median": None,
        "posterior_tree_selected_topk_lift": None,
        "posterior_tree_selected_topk_hit_rate": None,
        "posterior_tree_selected_rankic_ir": None,
        "posterior_tree_selected_rankic_mean": None,
        "posterior_tree_selected_topk_mean": None,
    }

    if posterior_mode == "disabled" or len(df_val) == 0:
        return model, {}, meta

    max_trees = _resolve_model_max_trees(model)
    meta["posterior_tree_model_max_trees"] = max_trees
    candidate_limits = _resolve_posterior_tree_candidate_limits(args, max_trees, base_best_iteration)
    meta["posterior_tree_candidate_limits"] = candidate_limits
    if len(candidate_limits) == 0:
        logger.warning(
            f"!!! [{model_label}] 候选树数后验选优未生效："
            f"base_best_iter={base_best_iteration}, model_max_trees={max_trees}, candidates=[]"
        )
        return model, {}, meta

    candidate_count = len(candidate_limits)
    search_space_note = "有效搜索空间很小" if candidate_count <= 2 else "存在有效搜索空间"
    logger.warning(
        "!" * 92
    )
    logger.warning(
        f"!!! [{model_label}] 候选树数后验选优启动 | mode={posterior_mode} | "
        f"metric={posterior_metric} | topk={posterior_topk} | "
        f"base_best_iter={base_best_iteration} | model_max_trees={max_trees} | "
        f"candidate_count={candidate_count} | {search_space_note}"
    )
    logger.warning(
        f"!!! [{model_label}] 候选树数列表: {candidate_limits}"
    )
    logger.warning(
        "!" * 92
    )

    scored_candidates: List[Tuple[float, float, float, int, Dict, object]] = []
    eval_topk_values = sorted({*topk_values, posterior_topk}) if topk_values else [posterior_topk]
    primary_topk_key = f"top{posterior_topk}_return_mean"
    primary_diag_key = f"diagnostic_Top{posterior_topk}_逐日均值_50分位"
    primary_lift_key = f"diagnostic_Top{posterior_topk}_相对全市场提升_均值"
    primary_hit_key = f"diagnostic_Top{posterior_topk}_命中率_日均收益为正"

    for tree_limit in candidate_limits:
        limited_model = _wrap_model_with_tree_limit(model, tree_limit)
        metrics = evaluate_validation_daily(
            model=limited_model,
            df_val=df_val,
            feature_columns=feature_columns,
            original_return_col=args.label_column,
            task=args.task,
            topk_values=eval_topk_values,
            emit_logs=False,
        )
        primary_score, rankic_ir_score, rankic_mean_score = _posterior_metric_score(
            metrics, posterior_metric, posterior_topk
        )
        scored_candidates.append((
            primary_score,
            rankic_ir_score,
            rankic_mean_score,
            tree_limit,
            metrics,
            limited_model,
        ))

    best_score = max(scored_candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    selected_limit = best_score[3]
    selected_metrics = best_score[4]
    selected_model = best_score[5]

    meta["posterior_tree_selected_limit"] = selected_limit
    meta["posterior_tree_selected_top30_median"] = _safe_float(selected_metrics.get(primary_diag_key))
    meta["posterior_tree_selected_topk_median"] = _safe_float(selected_metrics.get(primary_diag_key))
    meta["posterior_tree_selected_topk_lift"] = _safe_float(selected_metrics.get(primary_lift_key))
    meta["posterior_tree_selected_topk_hit_rate"] = _safe_float(selected_metrics.get(primary_hit_key))
    meta["posterior_tree_selected_rankic_ir"] = _safe_float(selected_metrics.get("daily_rankic_ir"))
    meta["posterior_tree_selected_rankic_mean"] = _safe_float(selected_metrics.get("daily_rankic_mean"))
    meta["posterior_tree_selected_topk_mean"] = _safe_float(selected_metrics.get(primary_topk_key))

    logger.info(
        f"{model_label} 候选树数后验选优完成: selected_limit={selected_limit}, "
        f"metric={posterior_metric}, topk={posterior_topk}, "
        f"TopK_Median={meta['posterior_tree_selected_topk_median']}, "
        f"TopK_Hit={meta['posterior_tree_selected_topk_hit_rate']}, "
        f"RankIC_IR={meta['posterior_tree_selected_rankic_ir']}, "
        f"RankIC={meta['posterior_tree_selected_rankic_mean']}"
    )

    return selected_model, selected_metrics, meta


def _print_pre_backtest_model_summary(
    split_index: int,
    adaptive_meta: Dict,
    val_daily_metrics: Dict,
    test_daily_metrics: Dict,
) -> None:
    """在回测前打印模型摘要：各子模型迭代轮数 + 验证集/测试集关键指标。"""
    # ── 1. 各子模型迭代轮数 ──
    sub_iters = adaptive_meta.get("sub_model_best_iterations", [])
    if sub_iters:
        seed_parts = []
        for seed_val, best_iter_val in sub_iters:
            seed_str = str(seed_val) if seed_val is not None else "?"
            iter_str = str(best_iter_val) if best_iter_val is not None else "?"
            seed_parts.append(f"seed={seed_str}:best_iter={iter_str}")
        logger.opt(colors=True).warning(
            f"<yellow><bold>Split {split_index} 子模型迭代轮数:</bold></yellow> "
            f"{', '.join(seed_parts)}"
        )

    # ── 2. 指标提取 ──
    def _extract(m: Dict) -> Dict[str, str]:
        def _f(key: str, fmt: str = ".6f") -> str:
            v = _safe_float(m.get(key))
            return "nan" if v is None else f"{v:{fmt}}"

        top20_med = _safe_float(m.get("diagnostic_Top20_逐日均值_50分位"))
        top30_med = _safe_float(m.get("diagnostic_Top30_逐日均值_50分位"))
        univ_mean = _safe_float(m.get("diagnostic_全市场收益_逐日均值的均值"))
        lift20_med = (
            top20_med - univ_mean
            if top20_med is not None and univ_mean is not None
            else None
        )
        lift30_med = (
            top30_med - univ_mean
            if top30_med is not None and univ_mean is not None
            else None
        )
        return {
            "RankIC均值": _f("daily_rankic_mean", ".4f"),
            "Top20中位数": _f("diagnostic_Top20_逐日均值_50分位"),
            "Top20命中率": _fmt_pct(_safe_float(m.get("diagnostic_Top20_命中率_日均收益为正"))),
            "Top20提升中位数": "nan" if lift20_med is None else f"{lift20_med:.6f}",
            "Top30中位数": _f("diagnostic_Top30_逐日均值_50分位"),
            "Top30命中率": _fmt_pct(_safe_float(m.get("diagnostic_Top30_命中率_日均收益为正"))),
            "Top30提升中位数": "nan" if lift30_med is None else f"{lift30_med:.6f}",
            "Top30提升均值": _f("diagnostic_Top30_相对全市场提升_均值"),
        }

    val_info = _extract(val_daily_metrics) if val_daily_metrics else {}
    test_info = _extract(test_daily_metrics) if test_daily_metrics else {}

    # ── 3. 打印 ──
    metric_labels = [
        "RankIC均值", "Top20中位数", "Top20命中率", "Top20提升中位数",
        "Top30中位数", "Top30命中率", "Top30提升中位数", "Top30提升均值",
    ]
    logger.opt(colors=True).warning(
        f"<yellow><bold>Split {split_index} 模型指标对比（验证集 vs 测试集）:</bold></yellow>"
    )
    for label in metric_labels:
        v_val = val_info.get(label, "-")
        v_test = test_info.get(label, "-")
        logger.opt(colors=True).warning(
            f"  <yellow>{label:16s}</yellow>  "
            f"验证={v_val:>12s}  |  测试={v_test:>12s}"
        )


def _resolve_adaptive_best_iter_action(best_iteration, n_estimators) -> Optional[str]:
    best_iter = _safe_float(best_iteration)
    tree_limit = _safe_float(n_estimators)
    if best_iter is None or tree_limit is None or tree_limit <= 0:
        return None
    if best_iter <= ADAPTIVE_LOW_BEST_ITER_THRESHOLD:
        return "low_iter"
    if best_iter >= ADAPTIVE_HIT_CAP_RATIO * tree_limit:
        return "hit_cap"
    return None


def _adaptive_sort_score(metrics: Dict) -> Tuple[float, float]:
    """自适应候选排序评分：Top30 逐日收益中位数优先，其次逐日 RankIC IR。"""
    top30_median = _safe_float(metrics.get("diagnostic_Top30_逐日均值_50分位"))
    rankic_ir = _safe_float(metrics.get("daily_rankic_ir"))
    return (
        -np.inf if top30_median is None else top30_median,
        -np.inf if rankic_ir is None else rankic_ir,
    )


def _is_better_adaptive_candidate(candidate_metrics: Dict, best_metrics: Optional[Dict]) -> bool:
    if best_metrics is None:
        return True
    return _adaptive_sort_score(candidate_metrics) > _adaptive_sort_score(best_metrics)


def _format_adaptive_metric_compare(candidate_metrics: Dict, reference_metrics: Optional[Dict]) -> str:
    def _fmt(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:.4f}"

    candidate_top30_med = _safe_float(candidate_metrics.get("diagnostic_Top30_逐日均值_50分位"))
    candidate_ir = _safe_float(candidate_metrics.get("daily_rankic_ir"))
    if reference_metrics is None:
        return (
            f"candidate_top30_median={_fmt(candidate_top30_med)}, "
            f"candidate_ir={_fmt(candidate_ir)}, "
            "reference=NONE, decision=SET_AS_BEST"
        )

    ref_top30_med = _safe_float(reference_metrics.get("diagnostic_Top30_逐日均值_50分位"))
    ref_ir = _safe_float(reference_metrics.get("daily_rankic_ir"))
    better = _is_better_adaptive_candidate(candidate_metrics, reference_metrics)
    decision = "UPDATE_BEST" if better else "KEEP_BEST"
    return (
        f"candidate/best_top30_median={_fmt(candidate_top30_med)}/{_fmt(ref_top30_med)}, "
        f"candidate/best_ir={_fmt(candidate_ir)}/{_fmt(ref_ir)}, "
        f"decision={decision}"
    )


def _candidate_passes_adaptive_replacement(base_metrics: Dict, candidate_metrics: Dict) -> bool:
    score = _adaptive_replacement_score(base_metrics, candidate_metrics)
    return score is not None and score > ADAPTIVE_REPLACEMENT_MIN_SCORE


def _adaptive_replacement_score(base_metrics: Dict, candidate_metrics: Dict) -> Optional[float]:
    base_ir = _safe_float(base_metrics.get("daily_rankic_ir"))
    candidate_ir = _safe_float(candidate_metrics.get("daily_rankic_ir"))
    base_top30_med = _safe_float(base_metrics.get("diagnostic_Top30_逐日均值_50分位"))
    candidate_top30_med = _safe_float(candidate_metrics.get("diagnostic_Top30_逐日均值_50分位"))

    if base_ir is None or candidate_ir is None:
        return None
    if base_top30_med is None or candidate_top30_med is None:
        return None

    ir_denominator = max(abs(base_ir), 1e-6)
    top30_denominator = max(abs(base_top30_med), 1e-6)
    ir_gain_ratio = (candidate_ir - base_ir) / ir_denominator
    top30_gain_ratio = (candidate_top30_med - base_top30_med) / top30_denominator

    return (
        ADAPTIVE_REPLACEMENT_TOP30_WEIGHT * top30_gain_ratio
        + ADAPTIVE_REPLACEMENT_RANKIC_IR_WEIGHT * ir_gain_ratio
    )


def _build_split_training_candidate(
    split: WalkForwardSplit,
    storage: Storage,
    loader: DataLoader,
    args,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: Optional[pd.DataFrame] = None,
    candidate_name: str = "base",
) -> Dict:
    ensemble_offsets = getattr(args, "ensemble_offsets", 0)
    ensemble_seeds = _resolve_ensemble_seeds(args)
    use_ensemble = (
        ensemble_offsets > 0 and trade_cal is not None
    ) or len(ensemble_seeds) > 1

    if use_ensemble:
        if ensemble_offsets > 0 and trade_cal is not None:
            windows = compute_offset_windows(
                split.train_start, split.train_end, ensemble_offsets, trade_cal
            )
        else:
            windows = [(split.train_start, split.train_end)]
        logger.info(
            f"{candidate_name} 集成训练: {len(windows)}个窗口 × {len(ensemble_seeds)}个种子 "
            f"= {len(windows) * len(ensemble_seeds)}个子模型"
            f"（偏移±{ensemble_offsets}个月, seeds={ensemble_seeds}）"
        )

        sub_models, base_result, adaptive_meta = _build_ensemble_sub_models(
            windows,
            storage,
            loader,
            args,
            main_board_codes,
            ensemble_seeds,
            topk_values=topk_values,
            enable_live_adaptive=getattr(args, "adaptive_best_iter_retrain", False),
        )

        model = EnsembleModel(sub_models)
        feature_columns = base_result["feature_columns"]
        train_params = base_result["train_params"]
        train_metrics = base_result["train_metrics"]
        val_metrics = base_result["val_metrics"]
        df_val_split_original = base_result["df_val_split_original"]
        data_stats = base_result["data_stats"]
        train_days_count = base_result["train_days_count"]
        total_train_samples = base_result["total_train_samples"]
        X_train_len = base_result["X_train_len"]
        X_val_len = base_result["X_val_len"]

        logger.info(f"{candidate_name} 集成模型创建完成: {model}")
    else:
        # 单模型路径：当只有一个 seed 时（ensemble_seeds=[seed] 或回退到 [args.random_state]），
        # 也需要应用该 seed，而不是默认使用 args.random_state
        tr = _train_model_on_window(
            split.train_start, split.train_end, storage, loader, args, main_board_codes,
            random_state_override=ensemble_seeds[0]
        )
        model = tr["model"]
        feature_columns = tr["feature_columns"]
        train_params = tr["train_params"]
        train_metrics = tr["train_metrics"]
        val_metrics = tr["val_metrics"]
        df_val_split_original = tr["df_val_split_original"]
        data_stats = tr["data_stats"]
        train_days_count = tr["train_days_count"]
        total_train_samples = tr["total_train_samples"]
        X_train_len = tr["X_train_len"]
        X_val_len = tr["X_val_len"]
        adaptive_meta = {
            "live_adaptive_triggered": False,
            "live_adaptive_trigger_count": 0,
            "live_adaptive_used_count": 0,
            "live_adaptive_last_action": None,
            "live_adaptive_last_best_iteration": None,
            "live_adaptive_final_learning_rate": args.learning_rate,
            "live_adaptive_final_n_estimators": args.n_estimators,
        }

    val_daily_metrics = {}
    if len(df_val_split_original) > 0:
        val_daily_metrics = evaluate_validation_daily(
            model=model,
            df_val=df_val_split_original,
            feature_columns=feature_columns,
            original_return_col=args.label_column,
            task=args.task,
            topk_values=topk_values,
        )

    return {
        "candidate_name": candidate_name,
        "model": model,
        "feature_columns": feature_columns,
        "train_params": train_params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_daily_metrics": val_daily_metrics,
        "df_val_split_original": df_val_split_original,
        "data_stats": data_stats,
        "train_days_count": train_days_count,
        "total_train_samples": total_train_samples,
        "X_train_len": X_train_len,
        "X_val_len": X_val_len,
        "adaptive_meta": adaptive_meta,
    }


def _build_adaptive_candidate_args(args, action: Optional[str]):
    if action == "low_iter":
        return _copy_args_with_training_overrides(
            args,
            learning_rate=args.learning_rate,# * 0.1,
            n_estimators=args.n_estimators,
        )
    if action == "hit_cap":
        return _copy_args_with_training_overrides(
            args,
            learning_rate=args.learning_rate * 2,# * 1.5,
            n_estimators=int(args.n_estimators * 2),
        )
    return None


def _resolve_adaptive_retry_seed(base_random_state: int, retry_index: int) -> int:
    return int(base_random_state) + int(retry_index)


def execute_split_training(
    split: WalkForwardSplit,
    wf_run_id: str,
    storage: Storage,
    loader: DataLoader,
    registry: ModelRegistry,
    args,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: Optional[pd.DataFrame] = None,
) -> Dict:
    """执行单个 split 的训练

    支持多偏移集成：当 args.ensemble_offsets > 0 时，训练 3 个偏移模型并包装为
    EnsembleModel，对外接口不变。

    Args:
        split: WalkForwardSplit 对象
        wf_run_id: walk-forward 运行ID
        storage: Storage 实例
        loader: DataLoader 实例
        registry: ModelRegistry 实例
        args: 命令行参数
        topk_values: TopK 评估值列表
        trade_cal: 交易日历（多偏移集成需要）

    Returns:
        包含训练结果的字典
    """
    logger.info("=" * 80)
    logger.info(f"开始训练 Split {split.split_index}")
    logger.info(f"  训练区间: {split.train_start} 至 {split.train_end}")
    logger.info(f"  测试区间: {split.test_start} 至 {split.test_end}")
    logger.info("=" * 80)

    # ── Phase 1: 训练模型，必要时基于 best_iteration 生成候选模型 ─────────────
    selected_candidate = _build_split_training_candidate(
        split, storage, loader, args, main_board_codes, topk_values, trade_cal,
        candidate_name="base"
    )
    adaptive_action = _resolve_adaptive_best_iter_action(
        selected_candidate["train_params"].get("best_iteration"), args.n_estimators
    )
    adaptive_meta = selected_candidate.get("adaptive_meta", {})
    adaptive_retrain_enabled = getattr(args, "adaptive_best_iter_retrain", False)
    max_low_iter_retries = max(
        1,
        int(getattr(args, "adaptive_low_iter_max_retries", ADAPTIVE_LOW_BEST_ITER_MAX_RETRIES)),
    )
    adaptive_candidate_used = False
    adaptive_candidate_evaluated = False
    adaptive_candidate_retry_count = 0
    adaptive_base_best_iteration = selected_candidate["train_params"].get("best_iteration")
    adaptive_candidate_best_iteration = None
    adaptive_candidate_last_retry_seed = None

    live_adaptive_already_handled = bool(adaptive_meta.get("live_adaptive_triggered", False))
    if adaptive_retrain_enabled and adaptive_action is not None and not live_adaptive_already_handled:
        base_val_daily = selected_candidate["val_daily_metrics"]
        base_seed = int(getattr(args, "random_state", 42))
        if adaptive_action == "low_iter":
            logger.warning(
                f"Split {split.split_index} 触发 best_iteration 低迭代重试: "
                f"base_best_iter={adaptive_base_best_iteration}, base_seed={base_seed}, "
                f"max_retries={max_low_iter_retries}, "
                f"base_lr={args.learning_rate:.6f}, base_n_estimators={args.n_estimators}"
            )
            best_retry_candidate = None
            best_retry_metrics = None
            best_retry_seed = None
            best_retry_best_iteration = None
            for retry_index in range(1, max_low_iter_retries + 1):
                adaptive_candidate_retry_count = retry_index
                retry_seed = _resolve_adaptive_retry_seed(base_seed, retry_index)
                adaptive_candidate_last_retry_seed = retry_seed
                candidate_args = _build_adaptive_candidate_args(
                    args,
                    adaptive_action,
                )
                candidate_args.random_state = retry_seed
                adaptive_candidate_evaluated = True
                logger.warning(
                    f"Split {split.split_index} 低迭代重试第 {retry_index}/{max_low_iter_retries} 次: "
                    f"retry_seed={retry_seed}, candidate_lr={candidate_args.learning_rate:.6f}, "
                    f"candidate_n_estimators={candidate_args.n_estimators}"
                )
                challenger = _build_split_training_candidate(
                    split,
                    storage,
                    loader,
                    candidate_args,
                    main_board_codes,
                    topk_values,
                    trade_cal,
                    candidate_name=f"adaptive_{adaptive_action}_retry{retry_index}",
                )
                adaptive_candidate_best_iteration = challenger["train_params"].get("best_iteration")
                challenger_val_daily = challenger["val_daily_metrics"]
                challenger_val_ir = _safe_float(challenger_val_daily.get("daily_rankic_ir"))
                challenger_top30_med = _safe_float(challenger_val_daily.get("diagnostic_Top30_逐日均值_50分位"))
                base_val_ir = _safe_float(base_val_daily.get("daily_rankic_ir"))
                base_top30_med = _safe_float(base_val_daily.get("diagnostic_Top30_逐日均值_50分位"))

                logger.warning(
                    f"Split {split.split_index} 低迭代重试指标对比: "
                    f"attempt={retry_index}, retry_seed={retry_seed}, "
                    f"{_format_adaptive_metric_compare(challenger_val_daily, best_retry_metrics)}"
                )

                if _is_better_adaptive_candidate(challenger_val_daily, best_retry_metrics):
                    best_retry_candidate = challenger
                    best_retry_metrics = challenger_val_daily
                    best_retry_seed = retry_seed
                    best_retry_best_iteration = adaptive_candidate_best_iteration

                logger.warning(
                    f"Split {split.split_index} 继续低迭代重试: attempt={retry_index}, retry_seed={retry_seed}, "
                    f"candidate_best_iter={adaptive_candidate_best_iteration}, "
                    f"base/cadidate_top30_med={'nan' if base_top30_med is None else f'{base_top30_med:.6f}'}/"
                    f"{'nan' if challenger_top30_med is None else f'{challenger_top30_med:.6f}'}, "
                    f"base/candidate_val_ir={'nan' if base_val_ir is None else f'{base_val_ir:.4f}'}/"
                    f"{'nan' if challenger_val_ir is None else f'{challenger_val_ir:.4f}'}"
                )

            if best_retry_metrics is not None:
                logger.warning(
                    f"Split {split.split_index} 低迭代重试最优候选: "
                    f"best_retry_seed={best_retry_seed}, best_retry_best_iter={best_retry_best_iteration}, "
                    f"best_retry_top30_median={_safe_float(best_retry_metrics.get('diagnostic_Top30_逐日均值_50分位')):.6f}, "
                    f"best_retry_val_ir={_safe_float(best_retry_metrics.get('daily_rankic_ir')):.4f}"
                )

            if best_retry_candidate is not None:
                adaptive_candidate_best_iteration = best_retry_best_iteration
                adaptive_candidate_last_retry_seed = best_retry_seed

            if best_retry_candidate is not None and _candidate_passes_adaptive_replacement(
                base_val_daily, best_retry_metrics
            ):
                selected_candidate = best_retry_candidate
                adaptive_candidate_used = True
                cand_best_top30m = _safe_float(best_retry_metrics.get('diagnostic_Top30_逐日均值_50分位'))
                cand_best_ir = _safe_float(best_retry_metrics.get('daily_rankic_ir'))
                logger.warning(
                    f"Split {split.split_index} 采用低迭代重试最优候选模型: "
                    f"retry_seed={best_retry_seed}, "
                    f"base/candidate_top30_med={'nan' if base_top30_med is None else f'{base_top30_med:.6f}'}/"
                    f"{'nan' if cand_best_top30m is None else f'{cand_best_top30m:.6f}'}, "
                    f"base/candidate_val_ir={'nan' if base_val_ir is None else f'{base_val_ir:.4f}'}/"
                    f"{'nan' if cand_best_ir is None else f'{cand_best_ir:.4f}'}"
                )

            if not adaptive_candidate_used:
                logger.warning(
                    f"Split {split.split_index} 低迭代重试完成但未满足替换条件: "
                    f"max_retries={max_low_iter_retries}, "
                    f"last_retry_seed={adaptive_candidate_last_retry_seed}, "
                    f"last_best_iter={adaptive_candidate_best_iteration}"
                )
        else:
            candidate_args = _build_adaptive_candidate_args(args, adaptive_action)
            logger.warning(
                f"Split {split.split_index} 触发 best_iteration 自适应重训: "
                f"action={adaptive_action}, base_best_iter={adaptive_base_best_iteration}, "
                f"base_lr={args.learning_rate:.6f}, base_n_estimators={args.n_estimators}, "
                f"candidate_lr={candidate_args.learning_rate:.6f}, "
                f"candidate_n_estimators={candidate_args.n_estimators}"
            )
            adaptive_candidate_evaluated = True
            challenger = _build_split_training_candidate(
                split,
                storage,
                loader,
                candidate_args,
                main_board_codes,
                topk_values,
                trade_cal,
                candidate_name=f"adaptive_{adaptive_action}",
            )
            adaptive_candidate_best_iteration = challenger["train_params"].get("best_iteration")
            base_val_ir = _safe_float(selected_candidate["val_daily_metrics"].get("daily_rankic_ir"))
            challenger_val_ir = _safe_float(challenger["val_daily_metrics"].get("daily_rankic_ir"))
            base_top30_med = _safe_float(
                selected_candidate["val_daily_metrics"].get("diagnostic_Top30_逐日均值_50分位")
            )
            challenger_top30_med = _safe_float(
                challenger["val_daily_metrics"].get("diagnostic_Top30_逐日均值_50分位")
            )
            if _candidate_passes_adaptive_replacement(
                selected_candidate["val_daily_metrics"], challenger["val_daily_metrics"]
            ):
                selected_candidate = challenger
                adaptive_candidate_used = True
                logger.warning(
                    f"Split {split.split_index} 采用自适应候选模型: action={adaptive_action}, "
                    f"base/candidate_top30_med={'nan' if base_top30_med is None else f'{base_top30_med:.6f}'}/"
                    f"{'nan' if challenger_top30_med is None else f'{challenger_top30_med:.6f}'}, "
                    f"base/candidate_val_ir={'nan' if base_val_ir is None else f'{base_val_ir:.4f}'}/"
                    f"{'nan' if challenger_val_ir is None else f'{challenger_val_ir:.4f}'}"
                )
            else:
                logger.warning(
                    f"Split {split.split_index} 保留基础模型: action={adaptive_action}, "
                    f"base/candidate_top30_med={'nan' if base_top30_med is None else f'{base_top30_med:.6f}'}/"
                    f"{'nan' if challenger_top30_med is None else f'{challenger_top30_med:.6f}'}, "
                    f"base/candidate_val_ir={'nan' if base_val_ir is None else f'{base_val_ir:.4f}'}/"
                    f"{'nan' if challenger_val_ir is None else f'{challenger_val_ir:.4f}'}"
                )

    posterior_model, posterior_val_daily_metrics, posterior_meta = _select_posterior_tree_model(
        model=selected_candidate["model"],
        feature_columns=selected_candidate["feature_columns"],
        df_val=selected_candidate["df_val_split_original"],
        args=args,
        topk_values=topk_values,
        train_params=selected_candidate["train_params"],
        model_label=f"Split {split.split_index}",
    )
    if posterior_meta.get("posterior_tree_selection_enabled"):
        selected_candidate["model"] = posterior_model
        if posterior_val_daily_metrics:
            selected_candidate["val_daily_metrics"] = posterior_val_daily_metrics
        selected_candidate["train_params"]["posterior_tree_base_best_iteration"] = (
            posterior_meta.get("posterior_tree_base_best_iteration")
        )
        selected_candidate["train_params"].update(posterior_meta)
        selected_candidate["train_params"]["best_iteration"] = posterior_meta.get(
            "posterior_tree_selected_limit"
        )

    model = selected_candidate["model"]
    feature_columns = selected_candidate["feature_columns"]
    train_params = selected_candidate["train_params"]
    train_metrics = selected_candidate["train_metrics"]
    val_metrics = selected_candidate["val_metrics"]
    val_daily_metrics = selected_candidate["val_daily_metrics"]
    data_stats = selected_candidate["data_stats"]
    train_days_count = selected_candidate["train_days_count"]
    total_train_samples = selected_candidate["total_train_samples"]
    X_train_len = selected_candidate["X_train_len"]
    X_val_len = selected_candidate["X_val_len"]

    # ── Phase 2: 加载测试数据 ──────────────────────────────────────────
    df_test, test_days_count = load_features_data(
        storage, loader, split.test_start, split.test_end
    )
    df_test = _filter_to_main_board(df_test, main_board_codes, "测试窗口")
    total_test_samples = len(df_test)

    # ── Phase 3: 样本外测试集评估（walk-forward 的核心）──────────────
    logger.info("=" * 60)
    logger.info("样本外测试集评估（OOS Evaluation）")
    logger.info("=" * 60)

    # 准备测试数据
    df_test_eval = df_test.copy()

    # 过滤测试集样本（与训练时一致：过滤 ST、停牌、涨停；跌停可买入，保留）
    filter_columns = ["is_st", "is_suspended", "is_limit_up"]
    mask = pd.Series(True, index=df_test_eval.index)
    for col in filter_columns:
        if col in df_test_eval.columns:
            mask = mask & (~df_test_eval[col].astype(bool))
    df_test_eval = df_test_eval[mask].copy()

    # 移除标签为 NaN 的样本
    if args.label_column in df_test_eval.columns:
        df_test_eval = df_test_eval.dropna(subset=[args.label_column])

    logger.info(f"测试集样本数（过滤后）: {len(df_test_eval)}")

    # 测试集预测（EnsembleModel.predict 自动平均多模型）
    X_test_features = df_test_eval[feature_columns]

    if args.task == "classification":
        y_test_pred_proba = model.predict_proba(X_test_features)[:, 1]
        df_test_eval["pred_score"] = y_test_pred_proba
    else:
        y_test_pred = model.predict(X_test_features)
        df_test_eval["pred_score"] = y_test_pred

    # 测试集逐日评估
    test_daily_metrics = evaluate_validation_daily(
        model=model,
        df_val=df_test_eval,
        feature_columns=feature_columns,
        original_return_col=args.label_column,
        task=args.task,
        topk_values=topk_values,
        emit_logs=False,
    )

    _print_oos_focus_panel(split.split_index, test_daily_metrics)

    # ── 回测前打印模型摘要（子模型迭代轮数 + 验证集/测试集关键指标）──
    if getattr(args, "oos_detail_metrics", False):
        _print_pre_backtest_model_summary(
            split_index=split.split_index,
            adaptive_meta=selected_candidate.get("adaptive_meta", {}),
            val_daily_metrics=val_daily_metrics,
            test_daily_metrics=test_daily_metrics,
        )
    else:
        logger.info(
            f"Split {split.split_index} OOS简报: "
            f"RankIC={_fmt_metric(_safe_float(test_daily_metrics.get('daily_rankic_mean')), '.4f')} | "
            f"Top20_hit={_fmt_pct(_safe_float(test_daily_metrics.get('diagnostic_Top20_命中率_日均收益为正')))} | "
            f"Top20_median={_fmt_metric(_safe_float(test_daily_metrics.get('diagnostic_Top20_逐日均值_50分位')), '.6f')} | "
            f"Top30_hit={_fmt_pct(_safe_float(test_daily_metrics.get('diagnostic_Top30_命中率_日均收益为正')))} | "
            f"Top30_median={_fmt_metric(_safe_float(test_daily_metrics.get('diagnostic_Top30_逐日均值_50分位')), '.6f')}"
        )

    # ── Phase 5: 注册模型 ─────────────────────────────────────────────
    # 准备性能指标（包含训练集、验证集、测试集）
    performance_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},
        "test_daily": test_daily_metrics,
    }

    # 准备完整的训练参数
    algorithm = getattr(args, "algorithm", "xgboost")
    full_train_params = train_params.copy()
    full_train_params.update({
        "algorithm": algorithm,
        "task": args.task,
        "label_transform": args.label_transform if args.task == "regression" else None,
        "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
        "pos_quantile": args.pos_quantile if args.task == "classification" else None,
        "pos_topk": args.pos_topk if args.task == "classification" else None,
        "scale_pos_weight_manual": args.scale_pos_weight is not None,
        "enable_cashflow_quality_features": getattr(
            args, "enable_cashflow_quality_features", False
        ),
        "enable_consensus_revision_features": getattr(
            args, "enable_consensus_revision_features", False
        ),
    })
    if isinstance(model, EnsembleModel):
        full_train_params["ensemble_offsets"] = getattr(args, "ensemble_offsets", 0)
        full_train_params["ensemble_seeds"] = _resolve_ensemble_seeds(args)
        full_train_params["ensemble_seed_keep_top_ratio"] = getattr(
            args, "ensemble_seed_keep_top_ratio", SEED_ENSEMBLE_KEEP_TOP_RATIO
        )
        full_train_params["ensemble_seed_keep_min_models"] = getattr(
            args, "ensemble_seed_keep_min_models", SEED_ENSEMBLE_KEEP_MIN_MODELS
        )
        full_train_params["ensemble_n_models"] = model.n_models
    full_train_params["adaptive_best_iter_retrain"] = adaptive_retrain_enabled
    full_train_params["adaptive_low_iter_max_retries"] = max_low_iter_retries
    full_train_params["adaptive_best_iter_action"] = adaptive_action
    full_train_params["adaptive_candidate_evaluated"] = adaptive_candidate_evaluated
    full_train_params["adaptive_candidate_used"] = adaptive_candidate_used
    full_train_params["adaptive_candidate_retry_count"] = adaptive_candidate_retry_count
    full_train_params["adaptive_base_best_iteration"] = adaptive_base_best_iteration
    full_train_params["adaptive_candidate_best_iteration"] = adaptive_candidate_best_iteration
    full_train_params["adaptive_candidate_last_retry_seed"] = adaptive_candidate_last_retry_seed
    full_train_params["adaptive_live_triggered"] = adaptive_meta.get("live_adaptive_triggered", False)
    full_train_params["adaptive_live_trigger_count"] = adaptive_meta.get("live_adaptive_trigger_count", 0)
    full_train_params["adaptive_live_used_count"] = adaptive_meta.get("live_adaptive_used_count", 0)
    full_train_params["adaptive_live_last_action"] = adaptive_meta.get("live_adaptive_last_action")
    full_train_params["adaptive_live_last_best_iteration"] = adaptive_meta.get(
        "live_adaptive_last_best_iteration"
    )
    full_train_params["adaptive_live_retry_count"] = adaptive_meta.get("live_adaptive_retry_count", 0)
    full_train_params["adaptive_live_last_retry_seed"] = adaptive_meta.get("live_adaptive_last_retry_seed")
    full_train_params["adaptive_live_last_candidate_best_iteration"] = adaptive_meta.get(
        "live_adaptive_last_candidate_best_iteration"
    )
    full_train_params["adaptive_live_final_random_state"] = adaptive_meta.get(
        "live_adaptive_final_random_state"
    )
    full_train_params["adaptive_live_final_learning_rate"] = adaptive_meta.get(
        "live_adaptive_final_learning_rate"
    )
    full_train_params["adaptive_live_final_n_estimators"] = adaptive_meta.get(
        "live_adaptive_final_n_estimators"
    )
    full_train_params["posterior_tree_selection_mode"] = train_params.get(
        "posterior_tree_selection_mode", "disabled"
    )
    full_train_params["posterior_tree_selection_metric"] = train_params.get(
        "posterior_tree_selection_metric", getattr(args, "posterior_tree_selection_metric", "topk_median")
    )
    full_train_params["posterior_tree_selection_topk"] = train_params.get(
        "posterior_tree_selection_topk", getattr(args, "posterior_tree_selection_topk", 20)
    )
    full_train_params["posterior_tree_selection_enabled"] = train_params.get(
        "posterior_tree_selection_enabled", False
    )
    full_train_params["posterior_tree_base_best_iteration"] = train_params.get(
        "posterior_tree_base_best_iteration"
    )
    full_train_params["posterior_tree_model_max_trees"] = train_params.get(
        "posterior_tree_model_max_trees"
    )
    full_train_params["posterior_tree_candidate_limits"] = train_params.get(
        "posterior_tree_candidate_limits"
    )
    full_train_params["posterior_tree_selected_limit"] = train_params.get(
        "posterior_tree_selected_limit"
    )
    full_train_params["posterior_tree_candidate_count"] = len(
        train_params.get("posterior_tree_candidate_limits") or []
    )
    full_train_params["posterior_tree_selected_top30_median"] = train_params.get(
        "posterior_tree_selected_top30_median"
    )
    full_train_params["posterior_tree_selected_topk_median"] = train_params.get(
        "posterior_tree_selected_topk_median"
    )
    full_train_params["posterior_tree_selected_topk_lift"] = train_params.get(
        "posterior_tree_selected_topk_lift"
    )
    full_train_params["posterior_tree_selected_topk_hit_rate"] = train_params.get(
        "posterior_tree_selected_topk_hit_rate"
    )
    full_train_params["posterior_tree_selected_rankic_ir"] = train_params.get(
        "posterior_tree_selected_rankic_ir"
    )
    full_train_params["posterior_tree_selected_rankic_mean"] = train_params.get(
        "posterior_tree_selected_rankic_mean"
    )
    full_train_params["posterior_tree_selected_topk_mean"] = train_params.get(
        "posterior_tree_selected_topk_mean"
    )

    # 注册模型（EnsembleModel 通过 joblib 序列化，包含所有子模型）
    version = registry.register_model(
        model=model,
        model_type=f"{algorithm}_{args.task}_wf",
        train_start_date=split.train_start,
        train_end_date=split.train_end,
        feature_columns=feature_columns,
        label_column=args.label_column,
        n_samples=X_train_len + X_val_len,
        train_params=full_train_params,
        performance_metrics=performance_metrics,
    )

    logger.info(f"模型已注册: v{version}")

    # ── Phase 6: 记录训练运行日志到CSV ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 特征筛选信息
    ffi = data_stats.get("feature_filter_info")
    feature_filter_summary = {}
    if ffi and not ffi.get("skipped", True):
        feature_filter_summary = {
            "feature_total": ffi["total_features"],
            "feature_stable": ffi["stable_count"],
            "feature_removed": ffi["removed_count"],
        }

    complete_data_stats = {
        "trade_days_count": train_days_count,
        "total_samples": total_train_samples,
        "samples_after_filter": data_stats["samples_after_filter"],
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_start_date": data_stats["val_start_date"],
        "val_end_date": data_stats["val_end_date"],
        "val_ratio": args.val_ratio,
        "val_raw_start_date": data_stats.get("val_raw_start_date", data_stats["val_start_date"]),
        "val_raw_end_date": data_stats.get("val_raw_end_date", data_stats["val_end_date"]),
        "val_raw_n_dates": data_stats.get("val_raw_n_dates", 0),
        "val_raw_samples": data_stats.get("val_raw_samples", 0),
        "val_es_start_date": data_stats.get("val_es_start_date", data_stats["val_start_date"]),
        "val_es_end_date": data_stats.get("val_es_end_date", data_stats["val_end_date"]),
        "val_es_n_dates": data_stats.get("val_es_n_dates", 0),
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "val_embargo_days_applied": data_stats.get("val_embargo_days_applied", 0),
        "val_embargo_n_dates": data_stats.get("val_embargo_n_dates", 0),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_start_date": data_stats.get("val_embargo_start_date", "N/A"),
        "val_embargo_end_date": data_stats.get("val_embargo_end_date", "N/A"),
        **feature_filter_summary,
    }

    # 创建训练运行记录
    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=split.train_start,
        end_date=split.train_end,
        label_column=args.label_column,
        task=args.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index=split.split_index,
        step_frequency=args.step,
        test_start_date=split.test_start,
        test_end_date=split.test_end,
    )

    # 写入CSV
    csv_path = (
        args.run_log_csv
        if args.run_log_csv
        else f"{args.data_root}/models/ml_train_runs.csv"
    )
    write_training_run_to_csv(run_record, csv_path)

    logger.info(f"训练运行日志已记录到: {csv_path}")

    # 返回结果
    result = {
        "split_index": split.split_index,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "model_version": version,
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "test_samples": len(df_test_eval),
        "best_iteration": train_params.get("best_iteration"),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "adaptive_best_iter_action": adaptive_action,
        "adaptive_candidate_evaluated": adaptive_candidate_evaluated,
        "adaptive_candidate_used": adaptive_candidate_used,
        "adaptive_base_best_iteration": adaptive_base_best_iteration,
        "adaptive_candidate_best_iteration": adaptive_candidate_best_iteration,
        "adaptive_selected_learning_rate": train_params.get("learning_rate", None),
        "adaptive_selected_n_estimators": train_params.get("n_estimators", None),
        "adaptive_live_triggered": adaptive_meta.get("live_adaptive_triggered", False),
        "adaptive_live_trigger_count": adaptive_meta.get("live_adaptive_trigger_count", 0),
        "adaptive_live_used_count": adaptive_meta.get("live_adaptive_used_count", 0),
        "adaptive_live_last_action": adaptive_meta.get("live_adaptive_last_action"),
        "adaptive_live_last_best_iteration": adaptive_meta.get("live_adaptive_last_best_iteration"),
        "adaptive_live_final_learning_rate": adaptive_meta.get("live_adaptive_final_learning_rate"),
        "adaptive_live_final_n_estimators": adaptive_meta.get("live_adaptive_final_n_estimators"),
        "posterior_tree_selection_mode": train_params.get("posterior_tree_selection_mode"),
        "posterior_tree_selection_metric": train_params.get("posterior_tree_selection_metric"),
        "posterior_tree_selection_topk": train_params.get("posterior_tree_selection_topk"),
        "posterior_tree_selection_enabled": train_params.get("posterior_tree_selection_enabled"),
        "posterior_tree_base_best_iteration": train_params.get("posterior_tree_base_best_iteration"),
        "posterior_tree_model_max_trees": train_params.get("posterior_tree_model_max_trees"),
        "posterior_tree_candidate_limits": train_params.get("posterior_tree_candidate_limits"),
        "posterior_tree_candidate_count": len(
            train_params.get("posterior_tree_candidate_limits") or []
        ),
        "posterior_tree_selected_limit": train_params.get("posterior_tree_selected_limit"),
        "posterior_tree_selected_top30_median": train_params.get("posterior_tree_selected_top30_median"),
        "posterior_tree_selected_topk_median": train_params.get("posterior_tree_selected_topk_median"),
        "posterior_tree_selected_topk_lift": train_params.get("posterior_tree_selected_topk_lift"),
        "posterior_tree_selected_topk_hit_rate": train_params.get("posterior_tree_selected_topk_hit_rate"),
        "posterior_tree_selected_rankic_ir": train_params.get("posterior_tree_selected_rankic_ir"),
        "posterior_tree_selected_rankic_mean": train_params.get("posterior_tree_selected_rankic_mean"),
        "test_daily_metrics": test_daily_metrics,
    }

    return result


def execute_deploy_training(
    deploy_train_end: str,
    wf_run_id: str,
    storage: Storage,
    loader: DataLoader,
    registry: ModelRegistry,
    args,
    main_board_codes: set,
    topk_values: List[int],
    trade_cal: pd.DataFrame,
) -> Optional[Dict]:
    """在 walk-forward 评估完成后，用最新数据训练部署模型

    部署模型使用最后一个 split 的 test_end 作为 train_end，
    使得模型能覆盖最新的可用数据，消除 walk-forward 评估模型的时间滞后。

    与 execute_split_training 的区别：
    - 不加载测试数据，不做 OOS 测试集评估
    - 模型 metadata 中标记 is_deploy=True

    Args:
        deploy_train_end: 部署模型的训练结束日期（通常为最后split的test_end）
        wf_run_id: walk-forward 运行ID
        storage: Storage 实例
        loader: DataLoader 实例
        registry: ModelRegistry 实例
        args: 命令行参数
        topk_values: TopK 评估值列表
        trade_cal: 交易日历 DataFrame

    Returns:
        包含训练结果的字典，失败返回 None
    """
    # 计算 train_start/train_end 并对齐到交易日
    train_start_dt = datetime.strptime(deploy_train_end, "%Y%m%d") - relativedelta(
        years=args.train_window_years
    )
    train_start_str = train_start_dt.strftime("%Y%m%d")

    train_start, train_end = resolve_deploy_train_window(
        trade_cal=trade_cal,
        deploy_train_end=deploy_train_end,
        train_window_years=args.train_window_years,
    )

    if train_start is None:
        logger.error(f"无法找到有效的部署模型 train_start（目标: {train_start_str}）")
        return None

    if train_end is None:
        logger.error(f"无法找到有效的部署模型 train_end（目标: {deploy_train_end}）")
        return None

    logger.info("=" * 80)
    logger.info("部署模型训练（Deploy Training）")
    logger.info(f"  训练区间: {train_start} 至 {train_end}")
    logger.info(f"  （无测试区间，用于部署）")
    logger.info("=" * 80)

    # ── 训练模型（支持多偏移集成）──────────────────────────────────────
    ensemble_offsets = getattr(args, "ensemble_offsets", 0)
    ensemble_seeds = _resolve_ensemble_seeds(args)
    use_ensemble = ensemble_offsets > 0 or len(ensemble_seeds) > 1

    if use_ensemble:
        if ensemble_offsets > 0:
            windows = compute_offset_windows(
                train_start, train_end, ensemble_offsets, trade_cal
            )
        else:
            windows = [(train_start, train_end)]
        logger.info(
            f"部署模型集成训练: {len(windows)}个窗口 × {len(ensemble_seeds)}个种子 "
            f"= {len(windows) * len(ensemble_seeds)}个子模型"
            f"（偏移±{ensemble_offsets}个月, seeds={ensemble_seeds}）"
        )

        sub_models, base_result, _ = _build_ensemble_sub_models(
            windows,
            storage,
            loader,
            args,
            main_board_codes,
            ensemble_seeds,
            is_deploy=True,
        )

        model = EnsembleModel(sub_models)
        feature_columns = base_result["feature_columns"]
        train_params = base_result["train_params"]
        train_metrics = base_result["train_metrics"]
        val_metrics = base_result["val_metrics"]
        df_val_split_original = base_result["df_val_split_original"]
        data_stats = base_result["data_stats"]
        train_days_count = base_result["train_days_count"]
        total_train_samples = base_result["total_train_samples"]
        X_train_len = base_result["X_train_len"]
        X_val_len = base_result["X_val_len"]

        logger.info(f"部署集成模型创建完成: {model}")
    else:
        # 单模型路径：当只有一个 seed 时，也需要应用该 seed
        tr = _train_model_on_window(
            train_start, train_end, storage, loader, args, main_board_codes,
            random_state_override=ensemble_seeds[0]
        )
        model = tr["model"]
        feature_columns = tr["feature_columns"]
        train_params = tr["train_params"]
        train_metrics = tr["train_metrics"]
        val_metrics = tr["val_metrics"]
        df_val_split_original = tr["df_val_split_original"]
        data_stats = tr["data_stats"]
        train_days_count = tr["train_days_count"]
        total_train_samples = tr["total_train_samples"]
        X_train_len = tr["X_train_len"]
        X_val_len = tr["X_val_len"]

    # 验证集逐日评估
    val_daily_metrics = {}
    if len(df_val_split_original) > 0:
        original_return_col = args.label_column
        val_daily_metrics = evaluate_validation_daily(
            model=model,
            df_val=df_val_split_original,
            feature_columns=feature_columns,
            original_return_col=original_return_col,
            task=args.task,
            topk_values=topk_values,
        )

    posterior_model, posterior_val_daily_metrics, posterior_meta = _select_posterior_tree_model(
        model=model,
        feature_columns=feature_columns,
        df_val=df_val_split_original,
        args=args,
        topk_values=topk_values,
        train_params=train_params,
        model_label="部署模型",
    )
    if posterior_meta.get("posterior_tree_selection_enabled"):
        model = posterior_model
        if posterior_val_daily_metrics:
            val_daily_metrics = posterior_val_daily_metrics
        train_params["posterior_tree_base_best_iteration"] = posterior_meta.get(
            "posterior_tree_base_best_iteration"
        )
        train_params.update(posterior_meta)
        train_params["best_iteration"] = posterior_meta.get("posterior_tree_selected_limit")

    # 6. 注册模型（与 wf 模型同一版本序列）
    performance_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_daily": val_daily_metrics,
        "test": {},
        "test_daily": {},
    }

    algorithm = getattr(args, "algorithm", "xgboost")
    full_train_params = train_params.copy()
    full_train_params.update({
        "algorithm": algorithm,
        "task": args.task,
        "label_transform": args.label_transform if args.task == "regression" else None,
        "winsorize_p": args.winsorize_p if args.label_transform == "cs_zscore" else None,
        "pos_quantile": args.pos_quantile if args.task == "classification" else None,
        "pos_topk": args.pos_topk if args.task == "classification" else None,
        "scale_pos_weight_manual": args.scale_pos_weight is not None,
        "is_deploy": True,
        "enable_cashflow_quality_features": getattr(
            args, "enable_cashflow_quality_features", False
        ),
        "enable_consensus_revision_features": getattr(
            args, "enable_consensus_revision_features", False
        ),
    })
    if isinstance(model, EnsembleModel):
        full_train_params["ensemble_offsets"] = ensemble_offsets
        full_train_params["ensemble_seeds"] = ensemble_seeds
        full_train_params["ensemble_seed_keep_top_ratio"] = getattr(
            args, "ensemble_seed_keep_top_ratio", SEED_ENSEMBLE_KEEP_TOP_RATIO
        )
        full_train_params["ensemble_seed_keep_min_models"] = getattr(
            args, "ensemble_seed_keep_min_models", SEED_ENSEMBLE_KEEP_MIN_MODELS
        )
        full_train_params["ensemble_n_models"] = model.n_models
    full_train_params["posterior_tree_selection_mode"] = train_params.get(
        "posterior_tree_selection_mode", "disabled"
    )
    full_train_params["posterior_tree_selection_metric"] = train_params.get(
        "posterior_tree_selection_metric", getattr(args, "posterior_tree_selection_metric", "topk_median")
    )
    full_train_params["posterior_tree_selection_topk"] = train_params.get(
        "posterior_tree_selection_topk", getattr(args, "posterior_tree_selection_topk", 20)
    )
    full_train_params["posterior_tree_selection_enabled"] = train_params.get(
        "posterior_tree_selection_enabled", False
    )
    full_train_params["posterior_tree_base_best_iteration"] = train_params.get(
        "posterior_tree_base_best_iteration"
    )
    full_train_params["posterior_tree_model_max_trees"] = train_params.get(
        "posterior_tree_model_max_trees"
    )
    full_train_params["posterior_tree_candidate_limits"] = train_params.get(
        "posterior_tree_candidate_limits"
    )
    full_train_params["posterior_tree_selected_limit"] = train_params.get(
        "posterior_tree_selected_limit"
    )
    full_train_params["posterior_tree_candidate_count"] = len(
        train_params.get("posterior_tree_candidate_limits") or []
    )
    full_train_params["posterior_tree_selected_top30_median"] = train_params.get(
        "posterior_tree_selected_top30_median"
    )
    full_train_params["posterior_tree_selected_topk_median"] = train_params.get(
        "posterior_tree_selected_topk_median"
    )
    full_train_params["posterior_tree_selected_topk_lift"] = train_params.get(
        "posterior_tree_selected_topk_lift"
    )
    full_train_params["posterior_tree_selected_topk_hit_rate"] = train_params.get(
        "posterior_tree_selected_topk_hit_rate"
    )
    full_train_params["posterior_tree_selected_rankic_ir"] = train_params.get(
        "posterior_tree_selected_rankic_ir"
    )
    full_train_params["posterior_tree_selected_rankic_mean"] = train_params.get(
        "posterior_tree_selected_rankic_mean"
    )
    full_train_params["posterior_tree_selected_topk_mean"] = train_params.get(
        "posterior_tree_selected_topk_mean"
    )

    version = registry.register_model(
        model=model,
        model_type=f"{algorithm}_{args.task}_wf",
        train_start_date=train_start,
        train_end_date=train_end,
        feature_columns=feature_columns,
        label_column=args.label_column,
        n_samples=X_train_len + X_val_len,
        train_params=full_train_params,
        performance_metrics=performance_metrics,
    )

    logger.info(f"部署模型已注册: v{version}")

    # 7. 记录训练运行日志到CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 特征筛选信息（部署模型路径）
    ffi_deploy = data_stats.get("feature_filter_info")
    ff_summary_deploy = {}
    if ffi_deploy and not ffi_deploy.get("skipped", True):
        ff_summary_deploy = {
            "feature_total": ffi_deploy["total_features"],
            "feature_stable": ffi_deploy["stable_count"],
            "feature_removed": ffi_deploy["removed_count"],
        }

    complete_data_stats = {
        "trade_days_count": train_days_count,
        "total_samples": total_train_samples,
        "samples_after_filter": data_stats["samples_after_filter"],
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_start_date": data_stats["val_start_date"],
        "val_end_date": data_stats["val_end_date"],
        "val_ratio": args.val_ratio,
        "val_raw_start_date": data_stats.get("val_raw_start_date", data_stats["val_start_date"]),
        "val_raw_end_date": data_stats.get("val_raw_end_date", data_stats["val_end_date"]),
        "val_raw_n_dates": data_stats.get("val_raw_n_dates", 0),
        "val_raw_samples": data_stats.get("val_raw_samples", 0),
        "val_es_start_date": data_stats.get("val_es_start_date", data_stats["val_start_date"]),
        "val_es_end_date": data_stats.get("val_es_end_date", data_stats["val_end_date"]),
        "val_es_n_dates": data_stats.get("val_es_n_dates", 0),
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "val_embargo_days_applied": data_stats.get("val_embargo_days_applied", 0),
        "val_embargo_n_dates": data_stats.get("val_embargo_n_dates", 0),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_start_date": data_stats.get("val_embargo_start_date", "N/A"),
        "val_embargo_end_date": data_stats.get("val_embargo_end_date", "N/A"),
        **ff_summary_deploy,
    }

    run_record = create_training_run_record_from_training_session(
        timestamp=timestamp,
        start_date=train_start,
        end_date=train_end,
        label_column=args.label_column,
        task=args.task,
        model_version=version,
        train_params=full_train_params,
        data_stats=complete_data_stats,
        performance_metrics=performance_metrics,
        wf_run_id=wf_run_id,
        split_index="deploy",
        step_frequency=args.step,
        test_start_date=None,
        test_end_date=None,
    )

    csv_path = (
        args.run_log_csv
        if args.run_log_csv
        else f"{args.data_root}/models/ml_train_runs.csv"
    )
    write_training_run_to_csv(run_record, csv_path)

    logger.info(f"部署模型训练运行日志已记录到: {csv_path}")

    return {
        "split_index": "deploy",
        "train_start": train_start,
        "train_end": train_end,
        "test_start": None,
        "test_end": None,
        "model_version": version,
        "train_samples": X_train_len,
        "val_samples": X_val_len,
        "val_es_samples": data_stats.get("val_es_samples", X_val_len),
        "val_embargo_samples": data_stats.get("val_embargo_samples", 0),
        "val_embargo_days": data_stats.get("val_embargo_days", 0),
        "test_samples": 0,
        "best_iteration": train_params.get("best_iteration"),
        "val_rankic_ir": val_daily_metrics.get("daily_rankic_ir"),
        "posterior_tree_selection_mode": train_params.get("posterior_tree_selection_mode"),
        "posterior_tree_selection_metric": train_params.get("posterior_tree_selection_metric"),
        "posterior_tree_selection_topk": train_params.get("posterior_tree_selection_topk"),
        "posterior_tree_selection_enabled": train_params.get("posterior_tree_selection_enabled"),
        "posterior_tree_base_best_iteration": train_params.get("posterior_tree_base_best_iteration"),
        "posterior_tree_model_max_trees": train_params.get("posterior_tree_model_max_trees"),
        "posterior_tree_candidate_limits": train_params.get("posterior_tree_candidate_limits"),
        "posterior_tree_candidate_count": len(
            train_params.get("posterior_tree_candidate_limits") or []
        ),
        "posterior_tree_selected_limit": train_params.get("posterior_tree_selected_limit"),
        "posterior_tree_selected_top30_median": train_params.get("posterior_tree_selected_top30_median"),
        "posterior_tree_selected_topk_median": train_params.get("posterior_tree_selected_topk_median"),
        "posterior_tree_selected_topk_lift": train_params.get("posterior_tree_selected_topk_lift"),
        "posterior_tree_selected_topk_hit_rate": train_params.get("posterior_tree_selected_topk_hit_rate"),
        "posterior_tree_selected_rankic_ir": train_params.get("posterior_tree_selected_rankic_ir"),
        "posterior_tree_selected_rankic_mean": train_params.get("posterior_tree_selected_rankic_mean"),
        "test_daily_metrics": {},
    }


def create_training_run_record_from_training_session(
    timestamp: str,
    start_date: str,
    end_date: str,
    label_column: str,
    task: str,
    model_version: int,
    train_params: Dict,
    data_stats: Dict,
    performance_metrics: Dict,
    wf_run_id: Optional[str] = None,
    split_index: Optional[int] = None,
    step_frequency: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None
) -> TrainingRunRecord:
    """从训练会话创建训练运行记录
    
    支持 walk-forward 场景的扩展参数。
    """
    # 基本信息
    record = TrainingRunRecord(
        timestamp=timestamp,
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
        label_column=label_column,
        task=task
    )
    
    # Walk-forward 字段
    record.wf_run_id = wf_run_id
    record.split_index = split_index
    record.step_frequency = step_frequency
    record.test_start_date = test_start_date
    record.test_end_date = test_end_date
    
    # 训练配置
    record.label_transform = train_params.get("label_transform")
    record.winsorize_p = train_params.get("winsorize_p")
    record.pos_quantile = train_params.get("pos_quantile")
    record.pos_topk = train_params.get("pos_topk")
    record.scale_pos_weight = train_params.get("scale_pos_weight")
    record.scale_pos_weight_mode = "manual" if train_params.get("scale_pos_weight_manual") else "auto"
    
    # XGBoost超参数
    record.n_estimators = train_params.get("n_estimators", 0)
    record.max_depth = train_params.get("max_depth", 0)
    record.num_leaves = train_params.get("num_leaves", 0)
    record.learning_rate = train_params.get("learning_rate", 0.0)
    record.subsample = train_params.get("subsample", 0.0)
    record.colsample_bytree = train_params.get("colsample_bytree", 0.0)
    record.gamma = train_params.get("gamma", 0.0)
    record.reg_alpha = train_params.get("reg_alpha", 0.0)
    record.reg_lambda = train_params.get("reg_lambda", 0.0)
    record.early_stopping_rounds = train_params.get("early_stopping_rounds", 0)
    record.tree_method = train_params.get("tree_method", "")
    record.random_state = train_params.get("random_state", 0)
    record.n_jobs = train_params.get("n_jobs", 0)
    
    # 数据统计
    record.trade_days_count = data_stats.get("trade_days_count", 0)
    record.total_samples = data_stats.get("total_samples", 0)
    record.samples_after_filter = data_stats.get("samples_after_filter", 0)
    record.train_samples = data_stats.get("train_samples", 0)
    record.val_samples = data_stats.get("val_samples", 0)
    record.val_start_date = data_stats.get("val_start_date", "")
    record.val_end_date = data_stats.get("val_end_date", "")
    record.val_ratio = data_stats.get("val_ratio", 0.2)
    
    # 训练结果
    record.best_iteration = train_params.get("best_iteration")
    
    # 训练集评估指标
    train_metrics = performance_metrics.get("train", {})
    if task == "regression":
        record.train_mse = train_metrics.get("mse")
        record.train_rmse = train_metrics.get("rmse")
        record.train_r2 = train_metrics.get("r2")
        record.train_ic = train_metrics.get("ic")
    else:
        record.train_accuracy = train_metrics.get("accuracy")
        record.train_auc = train_metrics.get("auc")
        record.train_precision = train_metrics.get("precision")
        record.train_recall = train_metrics.get("recall")
    
    # 验证集评估指标
    val_metrics = performance_metrics.get("validation", {})
    if task == "regression":
        record.val_mse = val_metrics.get("mse")
        record.val_rmse = val_metrics.get("rmse")
        record.val_r2 = val_metrics.get("r2")
        record.val_ic = val_metrics.get("ic")
        record.val_rank_ic = val_metrics.get("rank_ic")
    else:
        record.val_accuracy = val_metrics.get("accuracy")
        record.val_auc = val_metrics.get("auc")
        record.val_precision = val_metrics.get("precision")
        record.val_recall = val_metrics.get("recall")
    
    # 验证集逐日评估
    val_daily_metrics = performance_metrics.get("validation_daily", {})
    record.val_daily_rankic_mean = val_daily_metrics.get("daily_rankic_mean")
    record.val_daily_rankic_std = val_daily_metrics.get("daily_rankic_std")
    record.val_daily_rankic_ir = val_daily_metrics.get("daily_rankic_ir")
    
    # 额外指标（包含测试集指标）
    additional_metrics = {}

    # 验证集隔离统计（若存在）
    for key in [
        "val_raw_start_date",
        "val_raw_end_date",
        "val_raw_n_dates",
        "val_raw_samples",
        "val_es_start_date",
        "val_es_end_date",
        "val_es_n_dates",
        "val_es_samples",
        "val_embargo_days",
        "val_embargo_days_applied",
        "val_embargo_n_dates",
        "val_embargo_samples",
        "val_embargo_start_date",
        "val_embargo_end_date",
    ]:
        if key in data_stats:
            additional_metrics[key] = data_stats.get(key)
    
    # 验证集TopK收益和诊断统计
    for key, value in val_daily_metrics.items():
        if key not in ["daily_rankic_mean", "daily_rankic_std", "daily_rankic_ir"]:
            additional_metrics[f"val_{key}"] = value
    
    # 测试集逐日评估指标（walk-forward 的核心）
    test_daily_metrics = performance_metrics.get("test_daily", {})
    for key, value in test_daily_metrics.items():
        additional_metrics[f"test_{key}"] = value
    
    record.additional_metrics = additional_metrics
    
    return record


def write_walk_forward_summary(
    results: List[Dict],
    output_path: str,
    args,
    wf_run_id: str
) -> None:
    """生成 walk-forward 汇总 CSV

    Args:
        results: 所有 split 的结果列表
        output_path: 输出CSV路径
        args: 命令行参数（用于写入训练参数列）
        wf_run_id: walk-forward 运行ID
    """
    if len(results) == 0:
        logger.warning("没有结果可以写入汇总文件")
        return

    logger.info(f"生成 walk-forward 汇总文件: {output_path}")

    def _sanitize_train_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """将未启用功能对应的子参数清空，避免 compare 表出现误导性默认值。"""
        params = dict(raw_params)

        def clear(*keys: str) -> None:
            for key in keys:
                if key in params:
                    params[key] = None

        if not params.get("oos_backtest"):
            clear(
                "oos_backtest_months",
                "bt_top_n",
                "signal_confidence_gate_enabled",
                "signal_confidence_gate_top_k",
                "signal_confidence_gate_thresholds",
                "signal_confidence_gate_exposure_levels",
                "signal_gate_mode",
                "signal_gate_cost_multiplier",
                "signal_gate_round_trip_cost",
                "signal_gate_quality_enabled",
                "signal_gate_quality_window",
                "signal_gate_quality_threshold",
                "signal_gate_quality_halflife",
                "signal_gate_percentile_warmup",
                "signal_gate_dynamic_topn",
                "signal_gate_topn_high_multiplier",
                "signal_gate_topn_low_multiplier",
                "holding_bonus_enabled",
                "holding_bonus_sigma",
                "bt_sell_timing",
                "bt_exclude_st",
                "bt_min_list_days",
                "bt_max_weight_per_stock",
                "bt_max_per_industry",
                "bt_stop_loss_enabled",
                "bt_stop_loss_drawdown_pct",
                "bt_stop_loss_trailing_enabled",
                "bt_stop_loss_trailing_pct",
                "bt_stop_loss_consecutive_limit_down",
                "bt_weakness_exit_enabled",
                "bt_weakness_exit_threshold",
                "bt_weakness_exit_consecutive_days",
                "bt_weakness_exit_min_holding_days",
                "bt_weakness_exit_weights",
                "bt_weakness_exit_industry_filter",
                "bt_weakness_exit_industry_bottom_pct",
                "bt_equity_curve_enabled",
                "bt_equity_curve_drawdown_thresholds",
                "bt_equity_curve_exposure_levels",
                "bt_equity_curve_ma_short",
                "bt_equity_curve_ma_long",
                "bt_equity_curve_recovery_mode",
                "bt_equity_curve_recovery_step",
                "bt_equity_curve_recovery_delay_periods",
                "industry_momentum_filter",
                "industry_momentum_bottom_pct",
                "industry_rotation_enhanced",
                "industry_rotation_alpha",
                "position_sizing",
                "kelly_vol_window",
                "kelly_max_leverage",
                "market_regime",
                "market_regime_bear_threshold",
                "market_regime_bear_exposure",
                "market_regime_mode",
                "market_regime_vol_target",
                "market_regime_trend_threshold",
                "market_regime_min_exposure",
                "market_regime_combine_method",
                "market_regime_trend_guard",
                "market_regime_drawdown_guard",
                "market_regime_drawdown_threshold",
                "market_regime_ma250_hard_stop",
                "market_regime_ma250_threshold",
                "market_regime_ma250_exposure",
                "market_regime_ma250_atr_scaling",
                "stagger_tranches",
                "enable_profit_based_holding",
                "early_exit_loss_threshold",
                "early_exit_holding_ratio",
                "profit_extension_threshold",
                "profit_extension_days",
                "profit_extension_mode",
                "profit_extension_strength_threshold",
                "use_atr_for_early_exit",
                "atr_multiplier",
                "early_exit_mode",
                "early_exit_strength_protect_threshold",
                "early_exit_max_reprieves",
                "take_profit_threshold",
                "take_profit_refill",
                "enable_early_rebalance_on_empty",
            )
            return params

        signal_gate_mode = params.get("signal_gate_mode")
        signal_gate_active = signal_gate_mode == "composite" or (
            signal_gate_mode == "legacy" and params.get("signal_confidence_gate_enabled")
        )

        if signal_gate_mode not in ("legacy", "composite"):
            clear("signal_confidence_gate_top_k")

        if signal_gate_mode != "legacy":
            clear(
                "signal_confidence_gate_enabled",
                "signal_confidence_gate_thresholds",
                "signal_confidence_gate_exposure_levels",
            )
        elif not params.get("signal_confidence_gate_enabled"):
            clear(
                "signal_confidence_gate_top_k",
                "signal_confidence_gate_thresholds",
                "signal_confidence_gate_exposure_levels",
            )

        if signal_gate_mode != "composite":
            clear(
                "signal_gate_cost_multiplier",
                "signal_gate_round_trip_cost",
                "signal_gate_percentile_warmup",
            )

        if not params.get("signal_gate_quality_enabled"):
            clear(
                "signal_gate_quality_window",
                "signal_gate_quality_threshold",
                "signal_gate_quality_halflife",
            )

        if not signal_gate_active:
            clear(
                "signal_gate_dynamic_topn",
                "signal_gate_topn_high_multiplier",
                "signal_gate_topn_low_multiplier",
            )
        elif not params.get("signal_gate_dynamic_topn"):
            clear(
                "signal_gate_topn_high_multiplier",
                "signal_gate_topn_low_multiplier",
            )

        if not params.get("holding_bonus_enabled"):
            clear("holding_bonus_sigma")

        if not params.get("bt_stop_loss_enabled"):
            clear(
                "bt_stop_loss_drawdown_pct",
                "bt_stop_loss_trailing_enabled",
                "bt_stop_loss_trailing_pct",
                "bt_stop_loss_consecutive_limit_down",
            )
        elif not params.get("bt_stop_loss_trailing_enabled"):
            clear("bt_stop_loss_trailing_pct")

        if not params.get("bt_weakness_exit_enabled"):
            clear(
                "bt_weakness_exit_threshold",
                "bt_weakness_exit_consecutive_days",
                "bt_weakness_exit_min_holding_days",
                "bt_weakness_exit_weights",
                "bt_weakness_exit_industry_filter",
                "bt_weakness_exit_industry_bottom_pct",
            )
        elif not params.get("bt_weakness_exit_industry_filter"):
            clear("bt_weakness_exit_industry_bottom_pct")

        if not params.get("bt_equity_curve_enabled"):
            clear(
                "bt_equity_curve_drawdown_thresholds",
                "bt_equity_curve_exposure_levels",
                "bt_equity_curve_ma_short",
                "bt_equity_curve_ma_long",
                "bt_equity_curve_recovery_mode",
                "bt_equity_curve_recovery_step",
                "bt_equity_curve_recovery_delay_periods",
            )

        if not params.get("industry_momentum_filter"):
            clear("industry_momentum_bottom_pct")

        if not params.get("industry_rotation_enhanced"):
            clear("industry_rotation_alpha")

        if params.get("position_sizing") not in ("kelly", "half_kelly"):
            clear("kelly_vol_window", "kelly_max_leverage")

        if not params.get("market_regime"):
            clear(
                "market_regime_bear_threshold",
                "market_regime_bear_exposure",
                "market_regime_mode",
                "market_regime_vol_target",
                "market_regime_trend_threshold",
                "market_regime_min_exposure",
                "market_regime_combine_method",
                "market_regime_trend_guard",
                "market_regime_drawdown_guard",
                "market_regime_drawdown_threshold",
            )
        else:
            market_regime_mode = params.get("market_regime_mode")
            if market_regime_mode == "binary":
                clear(
                    "market_regime_vol_target",
                    "market_regime_trend_threshold",
                    "market_regime_min_exposure",
                    "market_regime_combine_method",
                    "market_regime_trend_guard",
                )
            elif market_regime_mode == "vol_target":
                clear(
                    "market_regime_bear_threshold",
                    "market_regime_bear_exposure",
                    "market_regime_trend_threshold",
                    "market_regime_combine_method",
                    "market_regime_trend_guard",
                )
            elif market_regime_mode == "trend":
                clear(
                    "market_regime_bear_threshold",
                    "market_regime_bear_exposure",
                    "market_regime_vol_target",
                    "market_regime_combine_method",
                    "market_regime_trend_guard",
                )
            elif market_regime_mode == "combined":
                clear(
                    "market_regime_bear_threshold",
                    "market_regime_bear_exposure",
                )

        if not params.get("market_regime_ma250_hard_stop"):
            clear(
                "market_regime_ma250_threshold",
                "market_regime_ma250_exposure",
                "market_regime_ma250_atr_scaling",
            )

        if not params.get("market_regime_drawdown_guard"):
            clear("market_regime_drawdown_threshold")

        if not params.get("enable_profit_based_holding"):
            clear(
                "early_exit_loss_threshold",
                "early_exit_holding_ratio",
                "profit_extension_threshold",
                "profit_extension_days",
                "profit_extension_mode",
                "profit_extension_strength_threshold",
                "use_atr_for_early_exit",
                "atr_multiplier",
                "early_exit_mode",
                "early_exit_strength_protect_threshold",
                "early_exit_max_reprieves",
                "time_stop_loss_enabled",
                "time_stop_loss_days",
                "time_stop_loss_profit_ratio",
            )
        else:
            profit_extension_mode = params.get("profit_extension_mode")
            if profit_extension_mode != "pnl":
                clear("profit_extension_threshold")
            if profit_extension_mode != "strength":
                clear("profit_extension_strength_threshold")
            if profit_extension_mode == "disabled":
                clear("profit_extension_days")

            if not params.get("use_atr_for_early_exit"):
                clear("atr_multiplier")

            if params.get("early_exit_mode") != "strength_veto":
                clear(
                    "early_exit_strength_protect_threshold",
                    "early_exit_max_reprieves",
                )

            if not params.get("time_stop_loss_enabled"):
                clear("time_stop_loss_days", "time_stop_loss_profit_ratio")

        if params.get("take_profit_threshold") is None:
            clear("take_profit_refill")

        return params

    derived_wf_start_date = getattr(args, "wf_start_date", results[0]["train_start"])
    derived_wf_end_date = getattr(args, "wf_end_date", results[-1]["test_end"])

    # 训练参数（所有 split 共享，写入每行方便后续对比脚本独立使用）
    train_params_cols = {
        "wf_run_id": wf_run_id,
        "batch_run_id": getattr(args, 'batch_run_id', None),
        "batch_period_label": getattr(args, 'batch_period_label', None),
        "split_count": getattr(args, 'split_count', len(results)),
        "final_date": getattr(args, 'final_date', derived_wf_end_date),
        "wf_start_date": derived_wf_start_date,
        "wf_end_date": derived_wf_end_date,
        "algorithm": args.algorithm,
        "step": args.step,
        "train_window_years": args.train_window_years,
        "test_window_months": args.test_window_months,
        "val_ratio": args.val_ratio,
        "label_column": args.label_column,
        "task": args.task,
        "label_transform": args.label_transform if args.task == "regression" else None,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "num_leaves": getattr(args, 'num_leaves', None),
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_metric": args.early_stopping_metric,
        "adaptive_best_iter_retrain": getattr(args, "adaptive_best_iter_retrain", False),
        "posterior_tree_selection_mode": getattr(
            args, "posterior_tree_selection_mode", "disabled"
        ),
        "posterior_tree_selection_metric": getattr(
            args, "posterior_tree_selection_metric", "topk_median"
        ),
        "posterior_tree_selection_topk": getattr(args, "posterior_tree_selection_topk", 20),
        "posterior_tree_candidates": getattr(args, "posterior_tree_candidates", ""),
        "rank_weight_enabled": args.rank_weight_enabled,
        "rank_weight_topk": args.rank_weight_topk,
        "rank_weight": args.rank_weight,
        "rank_weight_topk_weight_mode": getattr(args, "rank_weight_topk_weight_mode", "linear_decay"),
        "time_decay_half_life": args.time_decay_half_life,
        "objective": getattr(args, 'objective', 'mse'),
        "enable_fundamental": args.enable_fundamental_features,
        "enable_alt": args.enable_alt_features,
        "enable_margin": args.enable_margin_features,
        "enable_cyq": args.enable_cyq_features,
        "enable_fund": args.enable_fund_features,
        "enable_express": args.enable_express_features,
        "feature_stability_filter": args.feature_stability_filter,
        "factor_prune": getattr(args, 'factor_prune', False),
        "ensemble_offsets": getattr(args, 'ensemble_offsets', 0),
        "ensemble_seeds": getattr(args, 'ensemble_seeds', None),
        "enable_enhanced_features": getattr(args, 'enable_enhanced_features', False),
        "enable_north_features": getattr(args, 'enable_north_features', False),
        "enable_lhb_features": getattr(args, 'enable_lhb_features', False),
        "enable_consensus_features": getattr(args, 'enable_consensus_features', False),
        "enable_cashflow_quality_features": getattr(
            args, 'enable_cashflow_quality_features', False
        ),
        "enable_consensus_revision_features": getattr(
            args, 'enable_consensus_revision_features', False
        ),
        "oos_backtest": getattr(args, 'oos_backtest', False),
        "oos_backtest_months": getattr(args, 'oos_backtest_months', None),
        "bt_top_n": getattr(args, 'bt_top_n', None),
        "bt_rebalance_freq": getattr(args, 'bt_rebalance_freq', None),
        "bt_initial_capital": getattr(args, 'bt_initial_capital', None),
        "signal_confidence_gate_enabled": getattr(args, 'signal_confidence_gate_enabled', False),
        "signal_confidence_gate_top_k": getattr(args, 'signal_confidence_gate_top_k', 10),
        "signal_confidence_gate_thresholds": getattr(
            args, 'signal_confidence_gate_thresholds', [0.8, 1.2, 1.6]
        ),
        "signal_confidence_gate_exposure_levels": getattr(
            args, 'signal_confidence_gate_exposure_levels', [0.3, 0.6, 1.0]
        ),
        "signal_gate_mode": getattr(args, 'signal_gate_mode', 'legacy'),
        "signal_gate_cost_multiplier": getattr(args, 'signal_gate_cost_multiplier', 2.0),
        "signal_gate_round_trip_cost": getattr(args, 'signal_gate_round_trip_cost', 0.003),
        "signal_gate_quality_enabled": getattr(args, 'signal_gate_quality_enabled', False),
        "signal_gate_quality_window": getattr(args, 'signal_gate_quality_window', 5),
        "signal_gate_quality_threshold": getattr(args, 'signal_gate_quality_threshold', 0.4),
        "signal_gate_quality_halflife": getattr(args, 'signal_gate_quality_halflife', 3),
        "signal_gate_percentile_warmup": getattr(args, 'signal_gate_percentile_warmup', 20),
        "signal_gate_dynamic_topn": getattr(args, 'signal_gate_dynamic_topn', False),
        "signal_gate_topn_high_multiplier": getattr(args, 'signal_gate_topn_high_multiplier', 0.6),
        "signal_gate_topn_low_multiplier": getattr(args, 'signal_gate_topn_low_multiplier', 1.5),
        "holding_bonus_enabled": getattr(args, 'holding_bonus_enabled', False),
        "holding_bonus_sigma": getattr(args, 'holding_bonus_sigma', 0.5),
        "bt_sell_timing": getattr(args, 'bt_sell_timing', 'open'),
        "bt_exclude_st": getattr(args, 'bt_exclude_st', True),
        "bt_min_list_days": getattr(args, 'bt_min_list_days', 365),
        "bt_max_weight_per_stock": getattr(args, 'bt_max_weight_per_stock', None),
        "bt_max_per_industry": getattr(args, 'bt_max_per_industry', None),
        "bt_stop_loss_enabled": getattr(args, 'bt_stop_loss_enabled', False),
        "bt_stop_loss_drawdown_pct": getattr(args, 'bt_stop_loss_drawdown_pct', 30.0),
        "bt_stop_loss_trailing_enabled": getattr(args, 'bt_stop_loss_trailing_enabled', False),
        "bt_stop_loss_trailing_pct": getattr(args, 'bt_stop_loss_trailing_pct', 15.0),
        "bt_stop_loss_consecutive_limit_down": getattr(
            args, 'bt_stop_loss_consecutive_limit_down', 2
        ),
        "bt_weakness_exit_enabled": getattr(args, 'bt_weakness_exit_enabled', False),
        "bt_weakness_exit_threshold": getattr(args, 'bt_weakness_exit_threshold', 0.6),
        "bt_weakness_exit_consecutive_days": getattr(
            args, 'bt_weakness_exit_consecutive_days', 3
        ),
        "bt_weakness_exit_min_holding_days": getattr(
            args, 'bt_weakness_exit_min_holding_days', 5
        ),
        "bt_weakness_exit_weights": getattr(args, 'bt_weakness_exit_weights', "30,25,25,20"),
        "bt_weakness_exit_industry_filter": getattr(
            args, 'bt_weakness_exit_industry_filter', False
        ),
        "bt_weakness_exit_industry_bottom_pct": getattr(
            args, 'bt_weakness_exit_industry_bottom_pct', 0.3
        ),
        "bt_equity_curve_enabled": getattr(args, 'bt_equity_curve_enabled', False),
        "bt_equity_curve_drawdown_thresholds": getattr(
            args, 'bt_equity_curve_drawdown_thresholds', [5.0, 10.0, 15.0, 20.0]
        ),
        "bt_equity_curve_exposure_levels": getattr(
            args, 'bt_equity_curve_exposure_levels', [0.8, 0.6, 0.4, 0.2]
        ),
        "bt_equity_curve_ma_short": getattr(args, 'bt_equity_curve_ma_short', 5),
        "bt_equity_curve_ma_long": getattr(args, 'bt_equity_curve_ma_long', 20),
        "bt_equity_curve_recovery_mode": getattr(
            args, 'bt_equity_curve_recovery_mode', 'gradual'
        ),
        "bt_equity_curve_recovery_step": getattr(args, 'bt_equity_curve_recovery_step', 0.25),
        "bt_equity_curve_recovery_delay_periods": getattr(
            args, 'bt_equity_curve_recovery_delay_periods', 0
        ),
        "industry_momentum_filter": getattr(args, 'industry_momentum_filter', False),
        "industry_momentum_bottom_pct": getattr(args, 'industry_momentum_bottom_pct', 0.2),
        "industry_rotation_enhanced": getattr(args, 'industry_rotation_enhanced', False),
        "industry_rotation_alpha": getattr(args, 'industry_rotation_alpha', 0.3),
        "position_sizing": getattr(args, 'position_sizing', 'equal'),
        "kelly_vol_window": getattr(args, 'kelly_vol_window', 60),
        "kelly_max_leverage": getattr(args, 'kelly_max_leverage', 0.25),
        "market_regime": getattr(args, 'market_regime', False),
        "market_regime_bear_threshold": getattr(args, 'market_regime_bear_threshold', None),
        "market_regime_bear_exposure": getattr(args, 'market_regime_bear_exposure', None),
        "market_regime_mode": getattr(args, 'market_regime_mode', 'binary'),
        "market_regime_vol_target": getattr(args, 'market_regime_vol_target', None),
        "market_regime_trend_threshold": getattr(args, 'market_regime_trend_threshold', None),
        "market_regime_min_exposure": getattr(args, 'market_regime_min_exposure', None),
        "market_regime_combine_method": getattr(args, 'market_regime_combine_method', None),
        "market_regime_trend_guard": getattr(args, 'market_regime_trend_guard', True),
        "market_regime_drawdown_guard": getattr(args, 'market_regime_drawdown_guard', True),
        "market_regime_drawdown_threshold": getattr(args, 'market_regime_drawdown_threshold', None),
        "market_regime_ma250_hard_stop": getattr(args, 'market_regime_ma250_hard_stop', False),
        "market_regime_ma250_threshold": getattr(args, 'market_regime_ma250_threshold', 1.0),
        "market_regime_ma250_exposure": getattr(args, 'market_regime_ma250_exposure', 0.0),
        "market_regime_ma250_atr_scaling": getattr(args, 'market_regime_ma250_atr_scaling', False),
        "stagger_tranches": getattr(args, 'stagger_tranches', 1),
        "enable_profit_based_holding": getattr(args, 'enable_profit_based_holding', False),
        "early_exit_loss_threshold": getattr(args, 'early_exit_loss_threshold', -0.05),
        "early_exit_holding_ratio": getattr(args, 'early_exit_holding_ratio', 0.6),
        "profit_extension_threshold": getattr(args, 'profit_extension_threshold', 0.05),
        "profit_extension_days": getattr(args, 'profit_extension_days', 5),
        "profit_extension_mode": getattr(args, 'profit_extension_mode', 'pnl'),
        "profit_extension_strength_threshold": getattr(args, 'profit_extension_strength_threshold', 0.6),
        "use_atr_for_early_exit": getattr(args, 'use_atr_for_early_exit', False),
        "atr_multiplier": getattr(args, 'atr_multiplier', 2.0),
        "time_stop_loss_enabled": getattr(args, 'time_stop_loss_enabled', True),
        "time_stop_loss_days": getattr(args, 'time_stop_loss_days', 15),
        "time_stop_loss_profit_ratio": getattr(args, 'time_stop_loss_profit_ratio', -0.02),
        "early_exit_mode": getattr(args, 'early_exit_mode', 'disabled'),
        "early_exit_strength_protect_threshold": getattr(args, 'early_exit_strength_protect_threshold', 0.55),
        "early_exit_max_reprieves": getattr(args, 'early_exit_max_reprieves', 2),
        "take_profit_threshold": getattr(args, 'take_profit_threshold', None),
        "take_profit_refill": getattr(args, 'take_profit_refill', True),
        "enable_early_rebalance_on_empty": getattr(args, 'enable_early_rebalance_on_empty', True),
        "no_deploy_train": getattr(args, 'no_deploy_train', False),
        "skip_training": getattr(args, 'skip_training', False),
        "start_model_version": getattr(args, 'start_model_version', None),
        "selected_split_indices": getattr(args, 'selected_split_indices', None),
    }
    train_params_cols = _sanitize_train_params(train_params_cols)

    # 提取每个 split 的关键指标
    summary_rows = []

    for result in results:
        row = {
            **_build_summary_key_fields(result.get("test_daily_metrics", {})),
            "split_index": result["split_index"],
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
            "model_version": result["model_version"],
            "train_samples": result.get("train_samples"),
            "val_samples": result.get("val_samples"),
            "test_samples": result.get("test_samples"),
            "best_iteration": result.get("best_iteration"),
            "val_rankic_ir": result.get("val_rankic_ir"),
            "adaptive_best_iter_action": result.get("adaptive_best_iter_action"),
            "adaptive_candidate_evaluated": result.get("adaptive_candidate_evaluated"),
            "adaptive_candidate_used": result.get("adaptive_candidate_used"),
            "adaptive_base_best_iteration": result.get("adaptive_base_best_iteration"),
            "adaptive_candidate_best_iteration": result.get("adaptive_candidate_best_iteration"),
            "adaptive_selected_learning_rate": result.get("adaptive_selected_learning_rate", None),
            "adaptive_selected_n_estimators": result.get("adaptive_selected_n_estimators", None),
            "adaptive_live_triggered": result.get("adaptive_live_triggered"),
            "adaptive_live_trigger_count": result.get("adaptive_live_trigger_count"),
            "adaptive_live_used_count": result.get("adaptive_live_used_count"),
            "adaptive_live_last_action": result.get("adaptive_live_last_action"),
            "adaptive_live_last_best_iteration": result.get("adaptive_live_last_best_iteration"),
            "adaptive_live_final_learning_rate": result.get("adaptive_live_final_learning_rate"),
            "adaptive_live_final_n_estimators": result.get("adaptive_live_final_n_estimators"),
            "posterior_tree_selection_mode": result.get("posterior_tree_selection_mode"),
            "posterior_tree_selection_metric": result.get("posterior_tree_selection_metric"),
            "posterior_tree_selection_topk": result.get("posterior_tree_selection_topk"),
            "posterior_tree_selection_enabled": result.get("posterior_tree_selection_enabled"),
            "posterior_tree_base_best_iteration": result.get("posterior_tree_base_best_iteration"),
            "posterior_tree_selected_limit": result.get("posterior_tree_selected_limit"),
            "posterior_tree_selected_topk_median": result.get("posterior_tree_selected_topk_median"),
            "posterior_tree_selected_topk_lift": result.get("posterior_tree_selected_topk_lift"),
            "posterior_tree_selected_topk_hit_rate": result.get("posterior_tree_selected_topk_hit_rate"),
            "posterior_tree_selected_rankic_ir": result.get("posterior_tree_selected_rankic_ir"),
            "posterior_tree_selected_rankic_mean": result.get("posterior_tree_selected_rankic_mean"),
        }

        # 添加测试集逐日评估指标
        test_daily = result.get("test_daily_metrics", {})
        row.update(test_daily)

        # 添加 OOS 回测指标
        bt = result.get("bt_metrics", {})
        row.update(bt)

        # 追加训练参数列（放在最后）
        row.update(train_params_cols)

        summary_rows.append(row)

    # 转为 DataFrame 并写入
    df_summary = pd.DataFrame(summary_rows)

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"汇总文件已保存: {output_path}")
    logger.info(f"  共 {len(summary_rows)} 个切分")


def chain_nav_splits(
    results: List[Dict],
    summary_csv_path: str,
    wf_run_id: str,
) -> None:
    """将各 split 的 OOS 回测净值首尾串联成全周期净值曲线

    每个 split 的净值曲线被视为一个独立阶段，以上一阶段终值作为
    下一阶段起点，依次拼接。输出 CSV 与 summary 同目录。

    Args:
        results: 包含 _nav_curve 的结果列表
        summary_csv_path: summary CSV 路径（用于同目录输出）
        wf_run_id: walk-forward 运行 ID
    """
    nav_parts = []
    for r in results:
        nav = r.get("_nav_curve")
        if nav is not None and not nav.empty and 'nav' in nav.columns:
            part = nav[['nav']].copy()
            part['split_index'] = r['split_index']
            nav_parts.append(part)

    if not nav_parts:
        logger.info("无 OOS 回测净值可串联，跳过")
        return

    # 串联：每段净值归一化为上一段终值
    chained_records = []
    cumulative_nav = 1.0

    for part in nav_parts:
        raw = part['nav'].values
        if len(raw) == 0:
            continue
        # 归一化：该段起始 = cumulative_nav，按该段涨跌幅缩放
        scale = cumulative_nav / raw[0] if raw[0] != 0 else 1.0
        scaled = raw * scale
        for i, val in enumerate(scaled):
            chained_records.append({
                'date': part.index[i] if not isinstance(part.index[i], int) else i,
                'nav': val,
                'split_index': part['split_index'].iloc[i],
            })
        cumulative_nav = scaled[-1]

    df_chain = pd.DataFrame(chained_records)

    # 计算全周期指标
    total_return = cumulative_nav - 1.0
    trading_days = len(df_chain)
    years = trading_days / 252 if trading_days > 0 else 1
    # 简单年化收益率（不假设收益再投入）
    total_return_chain = cumulative_nav - 1.0
    cagr = (total_return_chain / years) if years > 0 else 0
    cummax = df_chain['nav'].cummax()
    drawdown = (df_chain['nav'] - cummax) / cummax
    max_dd = drawdown.min()
    daily_ret = df_chain['nav'].pct_change().dropna()
    vol = daily_ret.std() * (252 ** 0.5)
    sharpe = (cagr - 0.03) / vol if vol > 0 else 0

    logger.info("=" * 60)
    logger.info("全周期串联净值（Walk-forward Chain）")
    logger.info(f"  总收益:   {total_return*100:.1f}%")
    logger.info(f"  CAGR:     {cagr*100:.1f}%")
    logger.info(f"  最大回撤: {max_dd*100:.1f}%")
    logger.info(f"  夏普:     {sharpe:.2f}")
    logger.info(f"  交易日数: {trading_days}")
    logger.info("=" * 60)

    # 保存
    out_dir = Path(summary_csv_path).parent
    chain_path = out_dir / f"chain_nav_{wf_run_id}.csv"
    df_chain.to_csv(chain_path, index=False, encoding='utf-8-sig')
    logger.info(f"串联净值已保存: {chain_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Walk-forward 滚动训练")
    
    # Walk-forward 参数
    parser.add_argument(
        "--split-count",
        type=int,
        required=True,
        help="切分数量（正整数）"
    )
    parser.add_argument(
        "--final-date",
        type=str,
        required=True,
        help=(
            "最终日期，格式 YYYYMMDD。"
            "若启用部署训练，表示部署训练数据最后一天；"
            "若禁用部署训练，表示最后一个 split 测试结束日期"
        )
    )
    parser.add_argument(
        "--step",
        type=str,
        default="quarterly",
        choices=["monthly", "quarterly", "semiannual"],
        help="滚动频率（monthly|quarterly|semiannual），默认 quarterly"
    )
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=5,
        help="训练窗口年数，默认 5"
    )
    parser.add_argument(
        "--test-window-months",
        type=int,
        default=11,
        help="测试窗口月数，默认 11"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="训练数据内部验证集比例，默认 0.1"
    )
    parser.add_argument(
        "--selected-split-indices",
        type=int,
        nargs="*",
        default=[],
        help="仅训练指定 split 下标（如 0 4 5 7 9）；留空表示训练全部 split"
    )
    
    # 数据参数
    parser.add_argument(
        "--label-column",
        type=str,
        default="y_ret_5",
        help="标签列名，默认 y_ret_5"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"],
        help="标签选择（y_ret_5|y_ret_10|y_ret_20|neu_y_ret_5|neu_y_ret_10|neu_y_ret_20），默认 y_ret_5。优先级高于 --label-column"
    )
    
    # 任务类型和标签变换参数
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="任务类型（regression|classification），默认 regression"
    )
    parser.add_argument(
        "--label-transform",
        type=str,
        default="raw",
        choices=["raw", "cs_zscore"],
        help="标签变换方式（raw|cs_zscore），默认 raw。仅对 regression 任务生效"
    )
    parser.add_argument(
        "--winsorize-p",
        type=float,
        default=0.01,
        help="winsorize 参数（截断比例），默认 0.01（截断上下1%%）。仅当 label-transform=cs_zscore 时生效"
    )
    parser.add_argument(
        "--pos-quantile",
        type=float,
        default=None,
        help="分类任务正类百分比阈值（例如 0.2 表示 Top20%%），与 pos-topk 二选一"
    )
    parser.add_argument(
        "--pos-topk",
        type=int,
        default=None,
        help="分类任务正类数量阈值（例如 300 表示每日 Top300），与 pos-quantile 二选一，优先级更高"
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="分类任务正类权重，None 表示自动计算为 neg/pos（默认）"
    )
    
    # 算法选择
    parser.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm"],
        help="训练算法（xgboost|lightgbm），默认 xgboost"
    )

    # 模型参数
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="树的数量，默认 200"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="树的最大深度，默认 5（金融数据噪声大不宜过深）"
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=None,
        help="LightGBM 叶子数，默认 31。仅 LightGBM 有效，XGBoost 忽略此参数"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="学习率，默认 0.05"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="样本采样比例，默认 0.8"
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="特征采样比例，默认 0.8"
    )
    parser.add_argument(
        "--min-child-weight",
        type=int,
        default=100,
        help="叶节点最少样本权重和，防止过拟合，默认 100（金融数据建议 100-500）"
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.05,
        help="L1 正则化系数，默认 0.05（建议范围 0.05-0.5）"
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 正则化系数，默认 1.0（建议范围 1.0-5.0）"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="节点分裂最小损失下降，默认 0.1（建议范围 0.0-1.0）"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，默认 42"
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="早停轮数（验证集指标连续N轮不改善则停止），默认 200。设为 0 则禁用早停，使用固定 n_estimators"
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="rank_ic",
        choices=["auto", "rank_ic"],
        help="早停监控指标：auto（mae/auc，默认指标）或 rank_ic（Spearman Rank IC，尺度无关，跨 split 更稳定）。默认 rank_ic"
    )
    parser.add_argument(
        "--adaptive-best-iter-retrain",
        action="store_true",
        default=False,
        help=(
            "启用 walk-forward best_iteration 自适应候选重训："
            "best_iter<=50 时按随机种子重试并按 Top30中位数/RankIC IR 择优；"
            "best_iter>=90%%*n_estimators 时 lr*2 且 n_estimators*2；"
            "候选替换改为加权打分：Top30逐日均值中位数提升(70%%)+RankIC IR提升(30%%) > 0"
        )
    )
    parser.add_argument(
        "--adaptive-low-iter-max-retries",
        type=int,
        default=ADAPTIVE_LOW_BEST_ITER_MAX_RETRIES,
        help="low_iter（best_iter<=100）随机种子重试上限，默认 10"
    )
    parser.add_argument(
        "--posterior-tree-selection-mode",
        type=str,
        default="disabled",
        choices=["disabled", "grid"],
        help=(
            "候选树数后验选优模式：disabled=关闭；grid=训练完成后对候选树数网格做逐日验证，"
            "按 RankIC IR / RankIC 均值择优并替换最终模型复杂度"
        )
    )
    parser.add_argument(
        "--posterior-tree-candidates",
        type=str,
        default="",
        help=(
            "候选树数列表，逗号分隔，如 16,32,64,128。留空时使用内置自适应网格，"
            "并自动补入 early-stopping best_iteration 与 n_estimators"
        )
    )
    parser.add_argument(
        "--posterior-tree-selection-metric",
        type=str,
        default="topk_median",
        choices=[
            "topk_median", "topk_mean", "topk_lift", "topk_hit_rate",
            "topk_excess_hit_rate", "rankic_ir", "rankic_mean",
        ],
        help="候选树数后验选优主指标，默认 topk_median；disabled 模式下不生效"
    )
    parser.add_argument(
        "--posterior-tree-selection-topk",
        type=int,
        default=20,
        help="候选树数后验选优使用的 TopK，默认 20；可改为 30 快速切回旧 Top30 口径"
    )
    parser.add_argument(
        "--oos-detail-metrics",
        action="store_true",
        default=False,
        help=(
            "启用每个 split 的 OOS 详细指标对比表（验证集 vs 测试集）。"
            "默认关闭，仅输出重点 TopK 面板与一行简报，减少日志噪音"
        ),
    )

    # rank-weight 参数：Top/Bottom K 样本增强权重
    parser.add_argument(
        "--rank-weight-enabled",
        action="store_true",
        default=True,
        help="启用 Top/Bottom K 样本权重增强（默认开启）"
    )
    parser.add_argument(
        "--no-rank-weight",
        action="store_false",
        dest="rank_weight_enabled",
        help="禁用 rank-weight（覆盖 --rank-weight-enabled）"
    )
    parser.add_argument(
        "--rank-weight-topk",
        type=int,
        default=30,
        help="每日 Top/Bottom K 样本数，默认 30"
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=5.0,
        help="Top/Bottom K 样本权重，默认 5.0"
    )
    parser.add_argument(
        "--rank-weight-topk-weight-mode",
        type=str,
        default="linear_decay",
        choices=["linear_decay", "flat"],
        help="TopK 权重分配模式：linear_decay（默认）| flat（TopK 同权）"
    )

    # 时间衰减权重
    parser.add_argument(
        "--time-decay-half-life",
        type=float,
        default=0,
        help="时间衰减半衰期（年）。0 表示禁用。例如 1.0 → 1年前样本权重=0.5，2年前=0.25"
    )

    # 目标函数
    parser.add_argument(
        "--objective",
        type=str,
        default="mse",
        choices=["mse", "lambdarank"],
        help="目标函数类型：mse（回归，默认）或 lambdarank（排序学习，直接优化股票排序）"
    )

    # 基本面因子
    parser.add_argument(
        "--enable-fundamental-features",
        action="store_true",
        help="启用基本面因子（ROE、营收增速等）作为训练特征"
    )

    # 另类数据因子
    parser.add_argument(
        "--enable-alt-features",
        action="store_true",
        help="启用另类数据因子（股东人数、业绩预告等）"
    )

    # 融资融券因子
    parser.add_argument(
        "--enable-margin-features",
        action="store_true",
        help="启用融资融券因子（融资余额变动、融券/融资比、净买入比等）"
    )

    # 筹码胜率因子（5000 积分）
    parser.add_argument(
        "--enable-cyq-features",
        action="store_true",
        help="启用筹码胜率因子（winner_rate、成本偏离、筹码集中度等）"
    )

    # 基金持仓因子（5000 积分）
    parser.add_argument(
        "--enable-fund-features",
        action="store_true",
        help="启用基金持仓因子（持股比例、基金数量及其变化）"
    )

    # 业绩快报因子（5000 积分）
    parser.add_argument(
        "--enable-express-features",
        action="store_true",
        help="启用业绩快报因子（实际营收/净利润增速、业绩惊喜等）"
    )
    parser.add_argument(
        "--feature-stability-filter",
        action="store_true",
        help="启用特征稳定性筛选（移除跨时期IC方向不一致的特征）"
    )

    parser.add_argument(
        "--factor-prune",
        action="store_true",
        help="启用因子精简（从 data/models/factor_exclude_list.json 加载排除列表）"
    )

    # 多偏移集成
    parser.add_argument(
        "--ensemble-offsets",
        type=int,
        default=0,
        help="多偏移集成：偏移月数（0=禁用, 1=±1个月→3模型, 2=±2个月→3模型）"
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default=None,
        help="多种子 bagging：逗号分隔的随机种子列表（如 42,1,2,3,4）。"
             "默认 None=单种子（用 --random-state），与多偏移正交可叠加"
    )
    parser.add_argument(
        "--ensemble-seed-keep-top-ratio",
        type=float,
        default=SEED_ENSEMBLE_KEEP_TOP_RATIO,
        help="多种子筛选保留比例（0~1），默认 0.30"
    )
    parser.add_argument(
        "--ensemble-seed-keep-min-models",
        type=int,
        default=SEED_ENSEMBLE_KEEP_MIN_MODELS,
        help="多种子筛选最少保留模型数，默认 3"
    )

    # 因子增强（2.2）
    parser.add_argument(
        "--enable-enhanced-features",
        action="store_true",
        default=False,
        help="启用增强因子（开盘强度、日内波动结构、委托不平衡）"
    )

    # 北向资金因子
    parser.add_argument(
        "--enable-north-features",
        action="store_true",
        default=False,
        help="启用北向资金因子（moneyflow_hsgt, 市场级广播）"
    )

    # 龙虎榜因子
    parser.add_argument(
        "--enable-lhb-features",
        action="store_true",
        default=False,
        help="启用龙虎榜因子（top_list, 稀疏数据未上榜填 0）"
    )

    # 一致预期因子
    parser.add_argument(
        "--enable-consensus-features",
        action="store_true",
        default=False,
        help="启用卖方一致预期因子（report_rc, 滚动 30/60/90 日聚合）"
    )

    # 现金流质量因子（需 cashflow 接口，2000 积分）
    parser.add_argument(
        "--enable-cashflow-quality-features",
        action="store_true",
        default=False,
        help="启用现金流质量因子（需先下载 cashflow 数据）"
    )

    # 一致预期修正因子（基于已有 report_rc 构建时序修正信号）
    parser.add_argument(
        "--enable-consensus-revision-features",
        action="store_true",
        default=False,
        help="启用一致预期修正因子（EPS修正加速度/分歧度等时序信号）"
    )

    # 其他参数
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.* 配置"
    )
    parser.add_argument(
        "--run-log-csv",
        type=str,
        default=None,
        help="训练运行日志CSV路径，默认为 {data_root}/models/ml_train_runs.csv"
    )
    parser.add_argument(
        "--wf-summary-csv",
        type=str,
        default=None,
        help="walk-forward 汇总CSV路径，默认为 {data_root}/walk_forward/walk_forward_summary.csv"
    )
    parser.add_argument(
        "--batch-run-id",
        type=str,
        default=None,
        help="批量脚本生成的批次ID，仅用于汇总追踪"
    )
    parser.add_argument(
        "--batch-period-label",
        type=str,
        default=None,
        help="批量脚本传入的时间段标签，仅用于汇总追踪"
    )
    
    # OOS 回测参数
    parser.add_argument(
        "--oos-backtest",
        action="store_true",
        default=True,
        help="每个 split 训练后运行 OOS 回测（默认开启）"
    )
    parser.add_argument(
        "--no-oos-backtest",
        action="store_false",
        dest="oos_backtest",
        help="禁用 OOS 回测（仅保留统计指标评估）"
    )
    parser.add_argument(
        "--oos-backtest-months",
        type=int,
        default=0,
        help="OOS 回测时长（月），默认 0 表示自动对齐 test_window_months"
    )
    parser.add_argument(
        "--bt-top-n",
        type=int,
        default=30,
        help="OOS 回测持仓 Top N，默认 30"
    )
    parser.add_argument(
        "--bt-rebalance-freq",
        type=int,
        default=None,
        help="OOS 回测调仓频率（交易日），默认从标签自动推断"
    )

    parser.add_argument(
        "--signal-confidence-gate-enabled",
        action="store_true",
        default=False,
        help="启用信号置信度门控：低置信度时降仓或持币"
    )
    parser.add_argument(
        "--signal-confidence-gate-top-k",
        type=int,
        default=10,
        help="信号置信度评估使用的头部候选数量，默认 10"
    )
    parser.add_argument(
        "--signal-confidence-gate-thresholds",
        type=float,
        nargs="+",
        default=[0.8, 1.2, 1.6],
        help="信号置信度阈值列表；低于首档时持币，默认 0.8 1.2 1.6"
    )
    parser.add_argument(
        "--signal-confidence-gate-exposure-levels",
        type=float,
        nargs="+",
        default=[0.3, 0.6, 1.0],
        help="各信号置信度阈值对应的仓位系数，默认 0.3 0.6 1.0"
    )

    # 信号入口门控 v2
    parser.add_argument(
        "--signal-gate-mode", type=str, default="legacy",
        choices=["legacy", "composite", "disabled"],
        help="信号入口门控模式：legacy=旧公式, composite=新公式, disabled=关闭"
    )
    parser.add_argument(
        "--signal-gate-cost-multiplier", type=float, default=2.0,
        help="composite模式：预测收益至少覆盖交易成本的倍数（默认：2.0）"
    )
    parser.add_argument(
        "--signal-gate-round-trip-cost", type=float, default=0.003,
        help="composite模式：往返交易成本估算（默认：0.003）"
    )
    parser.add_argument(
        "--signal-gate-quality-enabled", action="store_true", default=False,
        help="启用滚动模型质量监控"
    )
    parser.add_argument(
        "--signal-gate-quality-window", type=int, default=5,
        help="滚动质量监控回看的调仓周期数（默认：5）"
    )
    parser.add_argument(
        "--signal-gate-quality-threshold", type=float, default=0.4,
        help="滚动质量监控最低hit rate（默认：0.4）"
    )
    parser.add_argument(
        "--signal-gate-quality-halflife", type=int, default=3,
        help="滚动质量EWM半衰期（默认：3）"
    )
    parser.add_argument(
        "--signal-gate-percentile-warmup", type=int, default=20,
        help="百分位归一化预热期（调仓次数，默认：20）"
    )
    parser.add_argument(
        "--signal-gate-dynamic-topn", action="store_true", default=False,
        help="启用动态Top-N：高置信度时集中选股，低置信度时分散选股"
    )
    parser.add_argument(
        "--signal-gate-topn-high-multiplier", type=float, default=0.6,
        help="动态Top-N高置信度缩减系数（默认：0.6）"
    )
    parser.add_argument(
        "--signal-gate-topn-low-multiplier", type=float, default=1.5,
        help="动态Top-N低置信度扩大系数（默认：1.5）"
    )
    parser.add_argument(
        "--holding-bonus-enabled", action="store_true", default=False,
        help="启用持仓保留奖励：调仓时对已持仓股票加分，降低换手率（默认：关闭）"
    )
    parser.add_argument(
        "--holding-bonus-sigma", type=float, default=0.5,
        help="持仓保留奖励幅度（截面分数标准差的倍数，默认：0.5）"
    )

    # 回测初始资金
    parser.add_argument(
        "--bt-initial-capital",
        type=float,
        default=1000000.0,
        help="OOS 回测初始资金（默认：1000000）"
    )
    parser.add_argument(
        "--bt-sell-timing",
        type=str,
        default="open",
        choices=["open", "close"],
        help="OOS 回测卖出时机：open 或 close，默认 open"
    )
    parser.add_argument(
        "--bt-exclude-st",
        action="store_true",
        default=True,
        dest="bt_exclude_st",
        help="OOS 回测排除 ST 股票（默认开启）"
    )
    parser.add_argument(
        "--bt-no-exclude-st",
        action="store_false",
        dest="bt_exclude_st",
        help="OOS 回测不排除 ST 股票"
    )
    parser.add_argument(
        "--bt-min-list-days",
        type=int,
        default=365,
        help="OOS 回测最少上市天数，默认 365"
    )
    parser.add_argument(
        "--bt-max-weight-per-stock",
        type=float,
        default=None,
        help="OOS 回测单股最大权重（0~1），默认不限制"
    )
    parser.add_argument(
        "--bt-max-per-industry",
        type=int,
        default=None,
        help="OOS 回测单行业最大持仓数量，默认不限制"
    )

    # OOS 回测止损参数
    parser.add_argument(
        "--bt-stop-loss-enabled",
        action="store_true",
        default=False,
        help="启用 OOS 回测止损功能"
    )
    parser.add_argument(
        "--bt-stop-loss-drawdown-pct",
        type=float,
        default=30.0,
        help="OOS 回测回撤止损阈值（%%），默认 30.0"
    )
    parser.add_argument(
        "--bt-stop-loss-trailing-enabled",
        action="store_true",
        default=False,
        help="启用 OOS 回测移动止损"
    )
    parser.add_argument(
        "--bt-stop-loss-trailing-pct",
        type=float,
        default=15.0,
        help="OOS 回测移动止损阈值（%%），默认 15.0"
    )
    parser.add_argument(
        "--bt-stop-loss-consecutive-limit-down",
        type=int,
        default=2,
        help="OOS 回测连续跌停止损天数，默认 2"
    )

    # OOS 回测 表现弱势退出 参数
    parser.add_argument(
        "--bt-weakness-exit-enabled",
        action="store_true",
        default=False,
        help="启用 OOS 回测表现弱势退出",
    )
    parser.add_argument(
        "--bt-weakness-exit-threshold",
        type=float,
        default=0.6,
        help="OOS 弱势评分触发阈值（默认：0.6）",
    )
    parser.add_argument(
        "--bt-weakness-exit-consecutive-days",
        type=int,
        default=3,
        help="OOS 需连续弱势天数（默认：3）",
    )
    parser.add_argument(
        "--bt-weakness-exit-min-holding-days",
        type=int,
        default=5,
        help="OOS 最低持有天数（默认：5）",
    )
    parser.add_argument(
        "--bt-weakness-exit-weights",
        type=str,
        default="30,25,25,20",
        help="OOS 4 维度权重（默认：30,25,25,20）",
    )
    parser.add_argument(
        "--bt-weakness-exit-industry-filter",
        action="store_true",
        default=False,
        help="OOS 叠加弱势行业过滤",
    )
    parser.add_argument(
        "--bt-weakness-exit-industry-bottom-pct",
        type=float,
        default=0.3,
        help="OOS 行业底部阈值（默认：0.3）",
    )

    # OOS 回测 ECT 参数
    parser.add_argument(
        "--bt-equity-curve-enabled",
        action="store_true",
        default=False,
        help="启用 OOS 回测权益曲线交易（ECT）"
    )
    parser.add_argument(
        "--bt-equity-curve-drawdown-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0],
        help="OOS 回测 ECT 回撤阈值列表（%%），默认 5 10 15 20"
    )
    parser.add_argument(
        "--bt-equity-curve-exposure-levels",
        type=float,
        nargs="+",
        default=[0.8, 0.6, 0.4, 0.2],
        help="OOS 回测 ECT 对应仓位系数列表，默认 0.8 0.6 0.4 0.2"
    )
    parser.add_argument(
        "--bt-equity-curve-ma-short",
        type=int,
        default=5,
        help="OOS 回测 ECT 短期均线窗口，默认 5"
    )
    parser.add_argument(
        "--bt-equity-curve-ma-long",
        type=int,
        default=20,
        help="OOS 回测 ECT 长期均线窗口，默认 20"
    )
    parser.add_argument(
        "--bt-equity-curve-recovery-mode",
        type=str,
        default="gradual",
        choices=["gradual", "immediate"],
        help="OOS 回测 ECT 恢复模式，默认 gradual"
    )
    parser.add_argument(
        "--bt-equity-curve-recovery-step",
        type=float,
        default=0.25,
        help="OOS 回测 ECT 逐步恢复步长，默认 0.25"
    )
    parser.add_argument(
        "--bt-equity-curve-recovery-delay-periods",
        type=int,
        default=0,
        help="OOS 回测 ECT 恢复等待周期，默认 0"
    )

    # 分批调仓
    parser.add_argument(
        "--stagger-tranches",
        type=int,
        default=1,
        help="分批调仓批次数（默认1=不分批）。设为K时将资金分成K份错开调仓，降低时点风险"
    )

    # 行业动量过滤
    parser.add_argument(
        "--industry-momentum-filter",
        action="store_true",
        default=False,
        help="启用行业动量过滤（剔除弱势行业股票，自动补位），默认关闭"
    )
    parser.add_argument(
        "--industry-momentum-bottom-pct",
        type=float,
        default=0.2,
        help="行业动量过滤阈值：剔除排名后 X%% 的行业（0~1），默认 0.2（后20%%）"
    )
    parser.add_argument(
        "--industry-rotation-enhanced",
        action="store_true",
        default=False,
        help="启用行业轮动加权：按行业动量排名对候选分数做乘性调整（强势加分、弱势扣分）"
    )
    parser.add_argument(
        "--industry-rotation-alpha",
        type=float,
        default=0.3,
        help="行业轮动加权强度（0=不调整, 1=强调整），默认 0.3"
    )

    # 仓位管理模式
    parser.add_argument(
        "--position-sizing",
        type=str,
        default="equal",
        choices=["equal", "score", "kelly", "half_kelly"],
        help="仓位管理模式: equal=等权, score=按分数, kelly=Kelly最优, half_kelly=半Kelly"
    )
    parser.add_argument(
        "--kelly-vol-window",
        type=int,
        default=60,
        help="Kelly 波动率估计窗口（交易日），默认 60"
    )
    parser.add_argument(
        "--kelly-max-leverage",
        type=float,
        default=0.25,
        help="Kelly 单只股票仓位上限（占总资产），默认 0.25"
    )

    # 市场择时仓位管理参数
    parser.add_argument(
        "--market-regime",
        action="store_true",
        default=False,
        help="启用市场择时仓位管理（熊市自动降仓），默认关闭"
    )
    parser.add_argument(
        "--market-regime-bear-threshold",
        type=float,
        default=-0.02,
        help="mkt_ret_avg_20 低于此值判定为熊市，默认 -0.02"
    )
    parser.add_argument(
        "--market-regime-bear-exposure",
        type=float,
        default=0.3,
        help="熊市仓位系数（0~1），默认 0.3（仅 binary 模式）"
    )
    parser.add_argument(
        "--market-regime-mode",
        type=str,
        default="binary",
        choices=["binary", "vol_target", "trend", "combined"],
        help="市场择时模式: binary(二值) | vol_target(波动率目标) | trend(趋势叠加) | combined(组合)，默认 binary"
    )
    parser.add_argument(
        "--market-regime-vol-target",
        type=float,
        default=0.15,
        help="波动率目标（年化），默认 0.15（15%%）。仅 vol_target/combined 模式有效"
    )
    parser.add_argument(
        "--market-regime-trend-threshold",
        type=float,
        default=1.0,
        help="趋势阈值（mkt_ma_trend 低于此值开始降仓），默认 1.0。仅 trend/combined 模式有效"
    )
    parser.add_argument(
        "--market-regime-min-exposure",
        type=float,
        default=0.2,
        help="最低仓位系数（非 binary 模式的下限），默认 0.2"
    )
    parser.add_argument(
        "--market-regime-combine-method",
        type=str,
        default="min",
        choices=["min", "multiply"],
        help="combined 模式组合方式: min(取最小) | multiply(相乘)，默认 min"
    )
    parser.add_argument(
        "--market-regime-trend-guard",
        action="store_true",
        default=True,
        help="combined 模式下趋势保护：上行趋势时跳过 vol_target 强制满仓，避免高波动上涨误杀，默认开启"
    )
    parser.add_argument(
        "--no-market-regime-trend-guard",
        action="store_false",
        dest="market_regime_trend_guard",
        help="关闭趋势保护"
    )
    parser.add_argument(
        "--market-regime-drawdown-guard",
        action="store_true",
        default=True,
        help="回撤保护：市场已大幅下跌时停止降仓，避免底部减仓踏空反弹，默认开启"
    )
    parser.add_argument(
        "--no-market-regime-drawdown-guard",
        action="store_false",
        dest="market_regime_drawdown_guard",
        help="关闭回撤保护"
    )
    parser.add_argument(
        "--market-regime-drawdown-threshold",
        type=float,
        default=-0.08,
        help="回撤保护阈值：mkt_drawdown_20 低于此值时停止降仓，默认 -0.08（-8%%）"
    )
    parser.add_argument(
        "--market-regime-ma250-hard-stop",
        action="store_true",
        default=False,
        help="启用 MA250 长周期硬条件：大盘跌破 250 日均线时强制降至 ma250_exposure 仓位"
    )
    parser.add_argument(
        "--market-regime-ma250-threshold",
        type=float,
        default=1.0,
        help="MA250 硬条件触发阈值（mkt_ma250_ratio < 此值触发），默认 1.0"
    )
    parser.add_argument(
        "--market-regime-ma250-exposure",
        type=float,
        default=0.0,
        help="MA250 硬条件触发后的仓位系数，默认 0.0（完全空仓）"
    )
    parser.add_argument(
        "--ma250-atr-scaling",
        action="store_true",
        default=False,
        dest="market_regime_ma250_atr_scaling",
        help="MA250 模块启用 ATR 动态仓位缩放：仓位 = base × MA(ATR,250)/CurrentATR"
    )
    # 盈亏动态持仓参数
    parser.add_argument(
        "--enable-profit-based-holding",
        action="store_true",
        default=False,
        help="启用盈亏动态持仓：亏损提前换出 + 盈利延续持有"
    )
    parser.add_argument(
        "--early-exit-loss-threshold",
        type=float,
        default=-0.05,
        help="亏损提前换出阈值（盈亏率），默认 -0.05（亏损5%%）"
    )
    parser.add_argument(
        "--early-exit-holding-ratio",
        type=float,
        default=0.6,
        help="亏损提前换出最早触发时点（占持有期比例），默认 0.6"
    )
    parser.add_argument(
        "--profit-extension-threshold",
        type=float,
        default=0.05,
        help="盈利延续持有阈值（盈亏率），默认 0.05（盈利5%%）"
    )
    parser.add_argument(
        "--profit-extension-days",
        type=int,
        default=5,
        help="盈利延续持有的额外天数（交易日），默认 5"
    )
    parser.add_argument(
        "--profit-extension-mode",
        type=str,
        default="pnl",
        choices=["pnl", "strength", "disabled"],
        help="盈利延续持有判据模式: pnl=单一浮盈率(默认,兼容原行为) | strength=5维度强势度评分 | disabled=关闭延续"
    )
    parser.add_argument(
        "--profit-extension-strength-threshold",
        type=float,
        default=0.6,
        help="strength 模式下的延续阈值 [0,1]，默认 0.6"
    )
    parser.add_argument(
        "--use-atr-for-early-exit",
        action="store_true",
        default=False,
        help="用个股 ATR 动态阈值替代固定 early_exit_loss_threshold（需同时开启 --enable-profit-based-holding）"
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=2.0,
        help="ATR 倍数，亏损超过 N×ATR%% 时提前换出，默认 2.0"
    )
    parser.add_argument(
        "--time-stop-loss-enabled",
        action="store_true",
        default=True,
        help="启用时间止损：持仓超限未达盈利要求时提前换出（需同时开启 --enable-profit-based-holding）"
    )
    parser.add_argument(
        "--no-time-stop-loss",
        dest="time_stop_loss_enabled",
        action="store_false",
        help="关闭时间止损"
    )
    parser.add_argument(
        "--time-stop-loss-days",
        type=int,
        default=15,
        help="时间止损最低持有天数（交易日），默认 15"
    )
    parser.add_argument(
        "--time-stop-loss-profit-ratio",
        type=float,
        default=-0.02,
        help="时间止损利润阈值，当前盈亏低于此值时触发，默认 -0.02（-2%%）"
    )
    parser.add_argument(
        "--early-exit-mode",
        type=str,
        default="disabled",
        choices=["disabled", "strength_veto"],
        help="亏损提前换出模式: disabled=原硬卖(默认), strength_veto=二次确认门控"
    )
    parser.add_argument(
        "--early-exit-strength-protect-threshold",
        type=float,
        default=0.55,
        help="strength_veto 模式下的保护阈值，评分>=此值时否决卖出，默认 0.55"
    )
    parser.add_argument(
        "--early-exit-max-reprieves",
        type=int,
        default=2,
        help="strength_veto 模式下单只股票最多缓刑次数，默认 2"
    )
    parser.add_argument(
        "--take-profit-threshold",
        type=float,
        default=None,
        help="整体持仓止盈阈值（如 0.15 表示整体浮盈15%%时清仓，默认禁用）"
    )
    parser.add_argument(
        "--no-take-profit-refill",
        dest="take_profit_refill",
        action="store_false",
        default=True,
        help="整体止盈后不触发补位买入（默认开启补位）"
    )
    parser.add_argument(
        "--no-early-rebalance-on-empty",
        dest="enable_early_rebalance_on_empty",
        action="store_false",
        default=True,
        help="禁用空仓/持有期拖尾时的提前调仓（默认启用：仓位清空或持有期满后残留盈利延续持仓时提前触发新一轮T0）"
    )

    # 部署训练参数
    parser.add_argument(
        "--no-deploy-train",
        action="store_true",
        default=False,
        help="禁用部署模型训练（默认开启：walk-forward完成后自动训练部署模型）"
    )

    # 跳过训练、复用已有模型（仅调参回测）
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help="跳过模型训练，直接使用已有模型做 OOS 回测（需配合 --start-model-version）"
    )
    parser.add_argument(
        "--start-model-version",
        type=int,
        default=None,
        help="skip-training 模式下第一个 split 对应的模型版本号，后续 split 依次 +1"
    )

    args = parser.parse_args()

    args.selected_split_indices = _normalize_selected_split_indices(
        getattr(args, "selected_split_indices", [])
    )
    if args.adaptive_low_iter_max_retries <= 0:
        logger.warning(
            f"adaptive_low_iter_max_retries={args.adaptive_low_iter_max_retries} 非法，自动修正为 1"
        )
        args.adaptive_low_iter_max_retries = 1
    if args.ensemble_seed_keep_top_ratio <= 0 or args.ensemble_seed_keep_top_ratio > 1:
        logger.warning(
            f"ensemble_seed_keep_top_ratio={args.ensemble_seed_keep_top_ratio} 非法，自动修正为 {SEED_ENSEMBLE_KEEP_TOP_RATIO}"
        )
        args.ensemble_seed_keep_top_ratio = SEED_ENSEMBLE_KEEP_TOP_RATIO
    if args.ensemble_seed_keep_min_models <= 0:
        logger.warning(
            f"ensemble_seed_keep_min_models={args.ensemble_seed_keep_min_models} 非法，自动修正为 {SEED_ENSEMBLE_KEEP_MIN_MODELS}"
        )
        args.ensemble_seed_keep_min_models = SEED_ENSEMBLE_KEEP_MIN_MODELS

    # 如果指定了 --label，则覆盖 --label-column
    if args.label is not None:
        args.label_column = args.label
    
    # 设置日志
    setup_logger()
    
    logger.info("=" * 80)
    logger.info("Walk-forward 滚动训练")
    logger.info("=" * 80)
    logger.info(f"切分数量: {args.split_count}")
    logger.info(f"最终日期: {args.final_date}")
    logger.info(f"滚动频率: {args.step}")
    logger.info(f"训练窗口: {args.train_window_years} 年")
    logger.info(f"测试窗口: {args.test_window_months} 个月")
    logger.info(
        "指定 split: %s"
        % (args.selected_split_indices if args.selected_split_indices else "全部")
    )
    logger.info(f"标签列: {args.label_column}")
    logger.info(f"任务类型: {args.task}")
    logger.info(f"早停: rounds={args.early_stopping_rounds if args.early_stopping_rounds else '禁用'}, metric={args.early_stopping_metric}")
    logger.info(
        f"best_iteration 自适应重训: {'启用' if args.adaptive_best_iter_retrain else '关闭'}"
    )
    if args.adaptive_best_iter_retrain:
        logger.info(f"  low_iter 最大重试次数: {args.adaptive_low_iter_max_retries}")
    logger.info(
        f"多种子筛选: top_ratio={args.ensemble_seed_keep_top_ratio:.0%}, "
        f"min_models={args.ensemble_seed_keep_min_models}"
    )
    if args.enable_enhanced_features:
        logger.info("因子增强: 启用（开盘强度、日内波动结构、委托不平衡）")
    # oos_backtest_months=0 表示自动对齐 test_window_months
    if args.oos_backtest_months <= 0:
        args.oos_backtest_months = args.test_window_months

    logger.info(f"OOS 回测: {'启用' if args.oos_backtest else '禁用'}")
    if args.oos_backtest:
        logger.info(f"  回测时长: {args.oos_backtest_months} 个月")
        logger.info(f"  持仓 Top N: {args.bt_top_n}")
        logger.info(f"  卖出时机: {args.bt_sell_timing}")
        logger.info(f"  调仓频率: {args.bt_rebalance_freq or '自动推断'}")
        logger.info(f"  信号入口门控: mode={args.signal_gate_mode}")
        if args.signal_gate_mode == "composite":
            logger.info(
                f"    cost_multiplier={args.signal_gate_cost_multiplier}, "
                f"cost={args.signal_gate_round_trip_cost}, "
                f"warmup={args.signal_gate_percentile_warmup}"
            )
        elif args.signal_gate_mode == "legacy" and args.signal_confidence_gate_enabled:
            logger.info(
                f"    legacy门控: top_k={args.signal_confidence_gate_top_k}, "
                f"thresholds={args.signal_confidence_gate_thresholds}, "
                f"exposures={args.signal_confidence_gate_exposure_levels}"
            )
        if args.signal_gate_quality_enabled:
            logger.info(
                f"  滚动质量监控: window={args.signal_gate_quality_window}, "
                f"threshold={args.signal_gate_quality_threshold}, "
                f"halflife={args.signal_gate_quality_halflife}"
            )
        logger.info(f"  排除 ST: {'是' if args.bt_exclude_st else '否'}")
        logger.info(f"  最少上市天数: {args.bt_min_list_days}")
        if args.bt_max_weight_per_stock is not None:
            logger.info(f"  单股最大权重: {args.bt_max_weight_per_stock:.2%}")
        if args.bt_max_per_industry is not None:
            logger.info(f"  单行业最大持仓数: {args.bt_max_per_industry}")
        logger.info(f"  止损: {'启用' if args.bt_stop_loss_enabled else '关闭'}")
        if args.bt_stop_loss_enabled:
            logger.info(
                f"    drawdown={args.bt_stop_loss_drawdown_pct}%, "
                f"trailing={'开' if args.bt_stop_loss_trailing_enabled else '关'}, "
                f"trailing_pct={args.bt_stop_loss_trailing_pct}%, "
                f"consecutive_limit_down={args.bt_stop_loss_consecutive_limit_down}"
            )
        logger.info(f"  ECT: {'启用' if args.bt_equity_curve_enabled else '关闭'}")
        if args.bt_equity_curve_enabled:
            logger.info(
                f"    drawdown_thresholds={args.bt_equity_curve_drawdown_thresholds}, "
                f"exposures={args.bt_equity_curve_exposure_levels}, "
                f"ma=({args.bt_equity_curve_ma_short},{args.bt_equity_curve_ma_long}), "
                f"recovery={args.bt_equity_curve_recovery_mode}/"
                f"{args.bt_equity_curve_recovery_step}/delay={args.bt_equity_curve_recovery_delay_periods}"
            )
        if args.stagger_tranches > 1:
            logger.info(f"  分批调仓: {args.stagger_tranches} 批")
        if args.market_regime:
            regime_detail = f"mode={args.market_regime_mode}"
            if args.market_regime_mode == "binary":
                regime_detail += f", bear_threshold={args.market_regime_bear_threshold}, bear_exposure={args.market_regime_bear_exposure}"
            else:
                regime_detail += f", vol_target={args.market_regime_vol_target}, trend_threshold={args.market_regime_trend_threshold}, min_exposure={args.market_regime_min_exposure}"
                if args.market_regime_mode == "combined":
                    regime_detail += f", combine={args.market_regime_combine_method}, trend_guard={args.market_regime_trend_guard}"
                regime_detail += f", dd_guard={args.market_regime_drawdown_guard}, dd_threshold={args.market_regime_drawdown_threshold}"
            logger.info(f"  市场择时: 开启 ({regime_detail})")
        else:
            logger.info(f"  市场择时: 关闭")
    effective_data_root = args.data_root or get_data_root()
    logger.info(f"数据目录: {effective_data_root}")
    
    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        registry = ModelRegistry(
            models_dir=get_models_root(str(Path(args.data_root) / "models") if args.data_root else None)
        )

        # 加载股票基本信息（OOS 回测需要）
        stock_basic = None
        if args.oos_backtest:
            stock_basic = loader.load_clean_stock_basic()
            if stock_basic is None:
                stock_basic = loader.load_stock_basic()
            if stock_basic is None:
                logger.warning("无法加载股票基本信息，OOS 回测将被禁用")
                args.oos_backtest = False

        # 训练/评估统一使用主板股票池，保证与交易口径一致
        if stock_basic is None:
            stock_basic = loader.load_clean_stock_basic()
        if stock_basic is None:
            stock_basic = loader.load_stock_basic()
        if stock_basic is None:
            raise ValueError("无法加载股票基本信息，无法执行主板过滤训练")
        main_board_codes = _build_main_board_codes(stock_basic)
        logger.info(f"主板股票池加载完成: {len(main_board_codes)} 只")

        # 生成 walk-forward ID
        wf_run_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Walk-forward 运行ID: {wf_run_id}")
        
        # 1. 生成 walk-forward 切分
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            trade_cal = loader.load_trade_cal()
        
        # 推断调仓频率（与 run_oos_backtest 内逻辑保持一致）
        if args.bt_rebalance_freq is not None:
            _rebalance_freq = args.bt_rebalance_freq
        else:
            _match = re.search(r'(\d+)', args.label_column)
            _rebalance_freq = int(_match.group(1)) if _match else 20

        splits = generate_walk_forward_splits_by_count(
            trade_cal=trade_cal,
            split_count=args.split_count,
            final_date=args.final_date,
            step_frequency=args.step,
            train_window_years=args.train_window_years,
            test_window_months=args.test_window_months,
            rebalance_freq=_rebalance_freq,
        )
        generated_split_count = len(splits)

        splits = _filter_splits_by_selected_indices(
            splits=splits,
            selected_split_indices=args.selected_split_indices,
        )
        
        if len(splits) == 0:
            logger.error("未生成任何切分，请检查参数设置")
            sys.exit(1)

        if args.selected_split_indices:
            logger.info(
                f"按下标筛选 split: {args.selected_split_indices}，"
                f"保留 {len(splits)} / {generated_split_count} 个"
            )
        else:
            logger.info(f"未指定 split 下标，默认训练全部 {len(splits)} 个 split")

        # 兼容汇总与对比脚本：写入推导出的 WF 覆盖区间
        args.wf_start_date = splits[0].train_start
        args.wf_end_date = splits[-1].test_end
        logger.info(
            f"推导区间: {args.wf_start_date} 至 {args.wf_end_date} "
            f"（由 split_count={args.split_count}, final_date={args.final_date} 反推）"
        )

        # skip-training 模式参数校验
        skip_training = getattr(args, "skip_training", False)
        start_model_version = getattr(args, "start_model_version", None)
        if skip_training and start_model_version is None:
            logger.error("--skip-training 模式必须指定 --start-model-version")
            sys.exit(1)

        deploy_train_start = None
        deploy_train_end_for_run = None
        if not args.no_deploy_train and not skip_training:
            deploy_train_start, deploy_train_end_for_run = resolve_deploy_train_window(
                trade_cal=trade_cal,
                deploy_train_end=args.final_date,
                train_window_years=args.train_window_years,
            )

            if deploy_train_start is None or deploy_train_end_for_run is None:
                logger.warning(
                    f"部署训练区间解析失败，无法在切分汇总中展示（目标train_end={args.final_date}）"
                )
                deploy_train_start = None
                deploy_train_end_for_run = None
            else:
                last_split = splits[-1]
                if (
                    deploy_train_start == last_split.train_start
                    and deploy_train_end_for_run == last_split.train_end
                ):
                    logger.error(
                        "部署训练区间与最后一个 split 的训练区间完全重叠，"
                        "请调整 split_count 或 final_date 后重试"
                    )
                    sys.exit(1)

                if deploy_train_end_for_run <= last_split.train_end:
                    logger.error(
                        f"部署训练结束日({deploy_train_end_for_run}) 不晚于"
                        f"最后一个 split 训练结束日({last_split.train_end})，"
                        "会造成训练区间冲突，请调整 final_date"
                    )
                    sys.exit(1)

        print_splits_summary(
            splits,
            deploy_train_start=deploy_train_start,
            deploy_train_end=deploy_train_end_for_run,
        )
        
        # 2. 执行每个 split 的训练
        results = []
        topk_values = sorted({
            20, 30, 100, 300, int(getattr(args, "posterior_tree_selection_topk", 20) or 20)
        })

        # 创建跨 split 持久化 signal（仅 OOS 回测时使用）
        # 作用：门控历史缓冲区在 split 间累积，百分位归一化/自校准阈值能够完成预热
        persistent_signal = None
        if args.oos_backtest:
            from src.lazybull.signals import MLSignal
            persistent_signal = MLSignal(
                top_n=args.bt_top_n,
                model_version=None,  # 首次 split 时通过 update_model_version 设置
                models_dir=get_models_root(
                    str(Path(args.data_root) / "models") if args.data_root else None
                ),
                signal_confidence_gate_enabled=args.signal_confidence_gate_enabled,
                signal_confidence_gate_top_k=args.signal_confidence_gate_top_k,
                signal_confidence_gate_thresholds=args.signal_confidence_gate_thresholds,
                signal_confidence_gate_exposure_levels=args.signal_confidence_gate_exposure_levels,
                signal_gate_mode=args.signal_gate_mode,
                signal_gate_cost_multiplier=args.signal_gate_cost_multiplier,
                signal_gate_round_trip_cost=args.signal_gate_round_trip_cost,
                signal_gate_percentile_warmup=args.signal_gate_percentile_warmup,
                verbose=False,
            )
            logger.info(
                f"持久化 MLSignal 已创建: mode={args.signal_gate_mode}, "
                f"将跨 {len(splits)} 个 split 积累门控历史"
            )

        for split in splits:
            try:
                if skip_training:
                    # 跳过训练，直接用预设版本号构造 result
                    model_version = start_model_version + split.split_index
                    logger.info(
                        f"[跳过训练] Split {split.split_index}: "
                        f"使用已有模型 v{model_version}，"
                        f"测试区间 {split.test_start} ~ {split.test_end}"
                    )
                    result = {
                        "split_index": split.split_index,
                        "train_start": split.train_start,
                        "train_end": split.train_end,
                        "test_start": split.test_start,
                        "test_end": split.test_end,
                        "model_version": model_version,
                        "bt_metrics": {},
                    }
                else:
                    result = execute_split_training(
                        split=split,
                        wf_run_id=wf_run_id,
                        storage=storage,
                        loader=loader,
                        registry=registry,
                        args=args,
                        main_board_codes=main_board_codes,
                        topk_values=topk_values,
                        trade_cal=trade_cal,
                    )

                # OOS 回测（每个 split 训练后运行真实回测）
                if args.oos_backtest and result.get("model_version"):
                    try:
                        bt_start = split.test_start
                        bt_end_dt = datetime.strptime(bt_start, '%Y%m%d') + relativedelta(months=args.oos_backtest_months)
                        bt_end = bt_end_dt.strftime('%Y%m%d')
                        bt_metrics = run_oos_backtest(
                            model_version=result["model_version"],
                            bt_start=bt_start,
                            bt_end=bt_end,
                            storage=storage,
                            loader=loader,
                            trade_cal=trade_cal,
                            stock_basic=stock_basic,
                            label_column=args.label_column,
                            bt_top_n=args.bt_top_n,
                            bt_rebalance_freq=args.bt_rebalance_freq,
                            data_root=args.data_root,
                            persistent_signal=persistent_signal,
                            signal_confidence_gate_enabled=args.signal_confidence_gate_enabled,
                            signal_confidence_gate_top_k=args.signal_confidence_gate_top_k,
                            signal_confidence_gate_thresholds=args.signal_confidence_gate_thresholds,
                            signal_confidence_gate_exposure_levels=args.signal_confidence_gate_exposure_levels,
                            signal_gate_mode=args.signal_gate_mode,
                            signal_gate_cost_multiplier=args.signal_gate_cost_multiplier,
                            signal_gate_round_trip_cost=args.signal_gate_round_trip_cost,
                            signal_gate_quality_enabled=args.signal_gate_quality_enabled,
                            signal_gate_quality_window=args.signal_gate_quality_window,
                            signal_gate_quality_threshold=args.signal_gate_quality_threshold,
                            signal_gate_quality_halflife=args.signal_gate_quality_halflife,
                            signal_gate_percentile_warmup=args.signal_gate_percentile_warmup,
                            signal_gate_dynamic_topn=args.signal_gate_dynamic_topn,
                            signal_gate_topn_high_multiplier=args.signal_gate_topn_high_multiplier,
                            signal_gate_topn_low_multiplier=args.signal_gate_topn_low_multiplier,
                            holding_bonus_enabled=args.holding_bonus_enabled,
                            holding_bonus_sigma=args.holding_bonus_sigma,
                            bt_exclude_st=args.bt_exclude_st,
                            bt_min_list_days=args.bt_min_list_days,
                            bt_sell_timing=args.bt_sell_timing,
                            bt_max_weight_per_stock=args.bt_max_weight_per_stock,
                            bt_max_per_industry=args.bt_max_per_industry,
                            bt_stop_loss_enabled=args.bt_stop_loss_enabled,
                            bt_stop_loss_drawdown_pct=args.bt_stop_loss_drawdown_pct,
                            bt_stop_loss_trailing_enabled=args.bt_stop_loss_trailing_enabled,
                            bt_stop_loss_trailing_pct=args.bt_stop_loss_trailing_pct,
                            bt_stop_loss_consecutive_limit_down=args.bt_stop_loss_consecutive_limit_down,
                            bt_equity_curve_enabled=args.bt_equity_curve_enabled,
                            bt_equity_curve_drawdown_thresholds=args.bt_equity_curve_drawdown_thresholds,
                            bt_equity_curve_exposure_levels=args.bt_equity_curve_exposure_levels,
                            bt_equity_curve_ma_short=args.bt_equity_curve_ma_short,
                            bt_equity_curve_ma_long=args.bt_equity_curve_ma_long,
                            bt_equity_curve_recovery_mode=args.bt_equity_curve_recovery_mode,
                            bt_equity_curve_recovery_step=args.bt_equity_curve_recovery_step,
                            bt_equity_curve_recovery_delay_periods=args.bt_equity_curve_recovery_delay_periods,
                            market_regime_enabled=args.market_regime,
                            market_regime_mode=args.market_regime_mode,
                            market_regime_bear_threshold=args.market_regime_bear_threshold,
                            market_regime_bear_exposure=args.market_regime_bear_exposure,
                            market_regime_vol_target=args.market_regime_vol_target,
                            market_regime_trend_threshold=args.market_regime_trend_threshold,
                            market_regime_min_exposure=args.market_regime_min_exposure,
                            market_regime_combine_method=args.market_regime_combine_method,
                            market_regime_trend_guard=args.market_regime_trend_guard,
                            market_regime_drawdown_guard=args.market_regime_drawdown_guard,
                            market_regime_drawdown_threshold=args.market_regime_drawdown_threshold,
                            market_regime_ma250_hard_stop=args.market_regime_ma250_hard_stop,
                            market_regime_ma250_threshold=args.market_regime_ma250_threshold,
                            market_regime_ma250_exposure=args.market_regime_ma250_exposure,
                            market_regime_ma250_atr_scaling=args.market_regime_ma250_atr_scaling,
                            industry_momentum_filter=args.industry_momentum_filter,
                            industry_momentum_bottom_pct=args.industry_momentum_bottom_pct,
                            industry_rotation_enhanced=getattr(args, 'industry_rotation_enhanced', False),
                            industry_rotation_alpha=getattr(args, 'industry_rotation_alpha', 0.3),
                            position_sizing=getattr(args, 'position_sizing', 'equal'),
                            kelly_vol_window=getattr(args, 'kelly_vol_window', 60),
                            kelly_max_leverage=getattr(args, 'kelly_max_leverage', 0.25),
                            stagger_tranches=args.stagger_tranches,
                            enable_profit_based_holding=args.enable_profit_based_holding,
                            early_exit_loss_threshold=args.early_exit_loss_threshold,
                            early_exit_holding_ratio=args.early_exit_holding_ratio,
                            profit_extension_threshold=args.profit_extension_threshold,
                            profit_extension_days=args.profit_extension_days,
                            profit_extension_mode=getattr(args, 'profit_extension_mode', 'pnl'),
                            profit_extension_strength_threshold=getattr(args, 'profit_extension_strength_threshold', 0.6),
                            profit_extension_strength_weights=getattr(args, 'profit_extension_strength_weights', None),
                            use_atr_for_early_exit=args.use_atr_for_early_exit,
                            atr_multiplier=args.atr_multiplier,
                            early_exit_mode=getattr(args, 'early_exit_mode', 'disabled'),
                            early_exit_strength_protect_threshold=getattr(args, 'early_exit_strength_protect_threshold', 0.55),
                            early_exit_max_reprieves=getattr(args, 'early_exit_max_reprieves', 2),
                            take_profit_threshold=args.take_profit_threshold,
                            take_profit_refill=args.take_profit_refill,
                            time_stop_loss_enabled=args.time_stop_loss_enabled,
                            time_stop_loss_days=args.time_stop_loss_days,
                            time_stop_loss_profit_ratio=args.time_stop_loss_profit_ratio,
                            weakness_exit_enabled=args.bt_weakness_exit_enabled,
                            weakness_exit_threshold=args.bt_weakness_exit_threshold,
                            weakness_exit_consecutive_days=args.bt_weakness_exit_consecutive_days,
                            weakness_exit_min_holding_days=args.bt_weakness_exit_min_holding_days,
                            weakness_exit_weights=args.bt_weakness_exit_weights,
                            weakness_exit_industry_filter=args.bt_weakness_exit_industry_filter,
                            weakness_exit_industry_bottom_pct=args.bt_weakness_exit_industry_bottom_pct,
                            enable_early_rebalance_on_empty=args.enable_early_rebalance_on_empty,
                            initial_capital=args.bt_initial_capital,
                            split_num=split.split_index,
                        )
                        # 提取 nav_curve 用于串联，不写入 CSV
                        nav_curve = bt_metrics.pop("_nav_curve", None)
                        if nav_curve is not None:
                            result["_nav_curve"] = nav_curve
                        result["bt_metrics"] = bt_metrics
                    except Exception as e:
                        logger.error(f"Split {split.split_index} OOS回测失败: {e}")
                        logger.error(traceback.format_exc())
                        result["bt_metrics"] = {}

                results.append(result)
            except Exception as e:
                logger.error(f"Split {split.split_index} 训练失败: {e}")
                logger.error(traceback.format_exc())
                logger.warning("继续执行下一个 split...")
                continue

        # 3. 部署模型训练（使用最新可用数据）
        if not args.no_deploy_train and not skip_training and len(results) > 0:
            if deploy_train_end_for_run is None:
                logger.error("部署训练区间未成功解析，跳过部署模型训练")
                deploy_train_end = None
            else:
                deploy_train_end = deploy_train_end_for_run
            logger.info("=" * 80)
            logger.info("开始部署模型训练（使用最新可用数据）")
            logger.info(f"  部署模型 train_end: {deploy_train_end}（由 final_date 对齐）")
            logger.info("=" * 80)
            if deploy_train_end is not None:
                try:
                    deploy_result = execute_deploy_training(
                        deploy_train_end=deploy_train_end,
                        wf_run_id=wf_run_id,
                        storage=storage,
                        loader=loader,
                        registry=registry,
                        args=args,
                        main_board_codes=main_board_codes,
                        topk_values=topk_values,
                        trade_cal=trade_cal,
                    )
                    if deploy_result:
                        logger.info(f"部署模型已注册: v{deploy_result['model_version']}")
                except Exception as e:
                    logger.error(f"部署模型训练失败: {e}")
                    logger.error(traceback.format_exc())

        # 4. 生成 walk-forward 汇总文件（统一输出到 raw/ 子目录）
        if len(results) > 0:
            if args.wf_summary_csv:
                summary_csv_path = args.wf_summary_csv
            else:
                summary_csv_path = str(
                    Path(args.data_root or get_data_root())
                    / "walk_forward"
                    / "raw"
                    / f"walk_forward_summary_{wf_run_id}.csv"
                )

            write_walk_forward_summary(results, summary_csv_path, args, wf_run_id)

            # ── 串联各 split 的 OOS 回测净值曲线 ──────────────────
            chain_nav_splits(results, summary_csv_path, wf_run_id)
        else:
            logger.warning("没有成功完成的训练，跳过生成汇总文件")

        logger.info("=" * 80)
        logger.info("Walk-forward 滚动训练完成！")
        logger.info(f"  成功完成: {len(results)} / {len(splits)} 个切分")
        logger.info(f"  运行ID: {wf_run_id}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Walk-forward 训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
