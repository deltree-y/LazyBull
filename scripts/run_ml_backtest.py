#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML 信号回测脚本

功能：
- 使用训练好的 ML 模型生成信号
- 运行回测并生成报告
- 支持指定模型版本、Top N、回测日期区间等参数

使用示例：
    # 使用最新模型回测
    python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231
    
    # 指定模型版本和 Top N
    python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
        --model-version 1 --top-n 50
    
    # 指定调仓频率（每N个交易日）
    python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
        --rebalance-freq 5  # 每5个交易日调仓一次
"""

import argparse
from datetime import datetime
import sys
import traceback
from pathlib import Path
import hashlib
from collections import defaultdict, deque

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import csv
from loguru import logger

from src.lazybull.backtest import BacktestEngine, BacktestEngineML, Reporter
from src.lazybull.common.backtest_runtime import (
    create_backtest_engine_from_config,
    infer_rebalance_freq_from_label,
)
from src.lazybull.common.config import get_models_root, get_reports_root
from src.lazybull.common.cost import CostModel
from src.lazybull.common.logger import setup_logger
from src.lazybull.common.trading_config import TradingConfig, add_trading_args
from src.lazybull.common.signal_factory import create_signal
from src.lazybull.data import DataLoader, Storage
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse
from src.lazybull.risk.stop_loss import StopLossConfig, create_stop_loss_config_from_dict
from src.lazybull.risk.equity_curve import EquityCurveConfig, create_equity_curve_config_from_dict
import warnings
# 匹配告警信息中的关键字符串，设置为 ignore
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")

def load_backtest_data(
    loader: DataLoader,
    storage: Storage,
    start_date: str,
    end_date: str
) -> tuple:
    """加载回测所需数据
    
    Args:
        loader: DataLoader 实例
        storage: Storage 实例
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        
    Returns:
        (trade_cal, stock_basic, daily_data, features_by_date) 元组
    """
    logger.info(f"加载回测数据: {start_date} 至 {end_date}")
    
    # 加载交易日历
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    
    # 加载股票基本信息
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        stock_basic = loader.load_stock_basic()
    
    # 加载日线数据
    daily_data = loader.load_clean_daily(start_date, end_date)
    if daily_data is None:
        logger.warning("没有 clean 层日线数据，尝试加载 raw 数据")
        daily_data = storage.load_raw("daily")
        if daily_data is not None:
            daily_data = daily_data[
                (daily_data['trade_date'] >= start_date) & 
                (daily_data['trade_date'] <= end_date)
            ]
    
    # 加载特征数据（按日期组织）
    trade_dates = trade_cal[
        (trade_cal['cal_date'] >= start_date) & 
        (trade_cal['cal_date'] <= end_date) & 
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    features_by_date = {}
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            features_by_date[trade_date] = features
    
    logger.info(
        f"数据加载完成: 交易日={len(trade_dates)}, "
        f"日线数据={len(daily_data) if daily_data is not None else 0}, "
        f"特征数据={len(features_by_date)} 日"
    )
    
    return trade_cal, stock_basic, daily_data, features_by_date


def prepare_price_data(daily_data: pd.DataFrame) -> pd.DataFrame:
    """准备价格数据
    
    Args:
        daily_data: 日线数据
        
    Returns:
        价格数据 DataFrame（包含 ts_code, trade_date, close）
    """
    if daily_data is None or len(daily_data) == 0:
        raise ValueError("没有价格数据")

    # 回测中既要成交价格（close），也要绩效价格（close_adj）
    desired_cols = [
        'ts_code', 'trade_date',

        # 价格口径
        'close', 'close_adj', 'open', 'open_adj',

        # 交易状态相关（用于 is_tradeable / is_limit_up / is_suspended 等）
        'is_suspended', 'is_limit_up', 'is_limit_down',
        'vol', 'pct_chg',

        # 股票池基础过滤可能用到的字段（按存在保留）
        'is_st', 'list_days', 'tradable'
    ]

    # 实际存在的列才保留，避免 raw 数据缺列时报错
    existing_cols = [c for c in desired_cols if c in daily_data.columns]
    price_data = daily_data[existing_cols].copy()

    # 关键列检查：close 必须有
    if 'close' not in price_data.columns:
        raise ValueError("价格数据缺少 'close' 列，无法进行回测")

    # close_adj 可选：没有就退化（engine 里也会退化）
    if 'close_adj' not in price_data.columns:
        logger.warning("prepare_price_data: 未找到 close_adj，绩效价格将退化为 close（不复权）")

    # 交易状态列缺失要明确提示（否则你以为过滤生效但其实没生效）
    missing_status_cols = [c for c in ['is_suspended', 'is_limit_up', 'is_limit_down'] if c not in price_data.columns]
    if missing_status_cols:
        logger.warning(f"prepare_price_data: 缺少交易状态列 {missing_status_cols}，涨跌停/停牌过滤将退化")

    return price_data


def run_ml_backtest(
    signal: MLSignal,
    universe: BasicUniverse,
    start_date: str,
    end_date: str,
    trading_dates: list,
    price_data: pd.DataFrame,
    features_by_date: dict,
    initial_capital: float = 1000000.0,
    rebalance_freq: int = 5,
    stagger_tranches: int = 1,
    cost_model: CostModel = None,
    stop_loss_config: StopLossConfig = None,
    equity_curve_config: EquityCurveConfig = None,
    sell_timing: str = 'open',
    max_weight_per_stock: float = None,
    max_per_industry: int = None,
    stock_basic: pd.DataFrame = None,
    holding_bonus_enabled: bool = False,
    holding_bonus_sigma: float = 0.5,
    profit_extension_mode: str = "pnl",
    profit_extension_strength_threshold: float = 0.6,
    profit_extension_strength_weights: dict = None,
    industry_rotation_enhanced: bool = False,
    industry_rotation_alpha: float = 0.3,
    position_sizing: str = "equal",
    kelly_vol_window: int = 60,
    kelly_max_leverage: float = 0.25,
    enable_profit_based_holding: bool = True,
    early_exit_loss_threshold: float = -0.07,
    early_exit_holding_ratio: float = 0.5,
    profit_extension_threshold: float = 0.1,
    profit_extension_days: int = 2,
    use_atr_for_early_exit: bool = False,
    atr_multiplier: float = 2.0,
    early_exit_mode: str = "strength_veto",
    early_exit_strength_protect_threshold: float = 0.1,
    early_exit_max_reprieves: int = 1,
    take_profit_threshold: float = None,
    take_profit_refill: bool = True,
    enable_early_rebalance_on_empty: bool = True,
    signal_gate_quality_enabled: bool = False,
    signal_gate_quality_window: int = 5,
    signal_gate_quality_threshold: float = 0.4,
    signal_gate_quality_halflife: int = 3,
    signal_gate_dynamic_topn: bool = False,
    signal_gate_topn_high_multiplier: float = 0.6,
    signal_gate_topn_low_multiplier: float = 1.5,
    trading_config: TradingConfig = None,
    data_storage: Storage = None,
) -> tuple:
    """运行 ML 信号回测
    
    Args:
        signal: ML 信号生成器
        universe: 股票池
        start_date: 开始日期
        end_date: 结束日期
        trading_dates: 交易日列表
        price_data: 价格数据
        features_by_date: 按日期组织的特征数据字典
        initial_capital: 初始资金
        rebalance_freq: 调仓频率（交易日数），必须为正整数
        cost_model: 成本模型
        stop_loss_config: 止损配置（可选）
        equity_curve_config: ECT 配置（可选）
        sell_timing: 卖出时机
        max_weight_per_stock: 单股最大权重（可选）
        max_per_industry: 单行业最大持仓数量（可选）
        stock_basic: 股票基本信息 DataFrame（用于行业约束，可选）
        holding_bonus_enabled: 是否启用持仓奖励（降低换手）
        holding_bonus_sigma: 持仓奖励强度（标准差倍数）
        
    Returns:
        (nav_curve, trades) 元组
    """
    logger.info("开始运行 ML 信号回测...")
    
    effective_config = trading_config or TradingConfig(
        top_n=getattr(signal, "top_n", 30),
        initial_capital=initial_capital,
        rebalance_freq=rebalance_freq,
        stagger_tranches=stagger_tranches,
        max_weight_per_stock=max_weight_per_stock,
        max_per_industry=max_per_industry,
        stop_loss_enabled=bool(stop_loss_config and stop_loss_config.enabled),
        stop_loss_drawdown_pct=(
            stop_loss_config.drawdown_pct if stop_loss_config and stop_loss_config.enabled else 30.0
        ),
        stop_loss_trailing_enabled=(
            stop_loss_config.trailing_stop_enabled
            if stop_loss_config and stop_loss_config.enabled
            else False
        ),
        stop_loss_trailing_pct=(
            stop_loss_config.trailing_stop_pct if stop_loss_config and stop_loss_config.enabled else 15.0
        ),
        stop_loss_consecutive_limit_down=(
            stop_loss_config.consecutive_limit_down_days
            if stop_loss_config and stop_loss_config.enabled
            else 2
        ),
        equity_curve_enabled=bool(equity_curve_config and equity_curve_config.enabled),
        equity_curve_drawdown_thresholds=(
            equity_curve_config.drawdown_thresholds
            if equity_curve_config and equity_curve_config.enabled
            else [5.0, 10.0, 15.0, 20.0]
        ),
        equity_curve_exposure_levels=(
            equity_curve_config.exposure_levels
            if equity_curve_config and equity_curve_config.enabled
            else [0.8, 0.6, 0.4, 0.2]
        ),
        equity_curve_ma_short=(
            equity_curve_config.ma_short_window if equity_curve_config and equity_curve_config.enabled else 5
        ),
        equity_curve_ma_long=(
            equity_curve_config.ma_long_window if equity_curve_config and equity_curve_config.enabled else 20
        ),
        equity_curve_recovery_mode=(
            equity_curve_config.recovery_mode if equity_curve_config and equity_curve_config.enabled else "gradual"
        ),
        equity_curve_recovery_step=(
            equity_curve_config.recovery_step if equity_curve_config and equity_curve_config.enabled else 0.25
        ),
        equity_curve_recovery_delay_periods=(
            equity_curve_config.recovery_delay_periods
            if equity_curve_config and equity_curve_config.enabled
            else 0
        ),
        holding_bonus_enabled=holding_bonus_enabled,
        holding_bonus_sigma=holding_bonus_sigma,
        profit_extension_mode=profit_extension_mode,
        profit_extension_strength_threshold=profit_extension_strength_threshold,
        profit_extension_strength_weights=profit_extension_strength_weights,
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
        use_atr_for_early_exit=use_atr_for_early_exit,
        atr_multiplier=atr_multiplier,
        early_exit_mode=early_exit_mode,
        early_exit_strength_protect_threshold=early_exit_strength_protect_threshold,
        early_exit_max_reprieves=early_exit_max_reprieves,
        take_profit_threshold=take_profit_threshold,
        take_profit_refill=take_profit_refill,
        enable_early_rebalance_on_empty=enable_early_rebalance_on_empty,
        signal_gate_quality_enabled=signal_gate_quality_enabled,
        signal_gate_quality_window=signal_gate_quality_window,
        signal_gate_quality_threshold=signal_gate_quality_threshold,
        signal_gate_quality_halflife=signal_gate_quality_halflife,
        signal_gate_dynamic_topn=signal_gate_dynamic_topn,
        signal_gate_topn_high_multiplier=signal_gate_topn_high_multiplier,
        signal_gate_topn_low_multiplier=signal_gate_topn_low_multiplier,
        sell_price=sell_timing,
    )

    # 共享引擎工厂：确保与 walk_forward 使用同一套策略参数透传逻辑
    engine = create_backtest_engine_from_config(
        trading_config=effective_config,
        universe=universe,
        signal=signal,
        features_by_date=features_by_date,
        stock_basic=stock_basic,
        data_storage=data_storage,
        initial_capital=initial_capital,
        sell_timing=sell_timing,
        verbose=False,
        completion_window_days=5,
        enable_pending_order=True,
        cost_model=cost_model or CostModel(),
    )

    # 运行回测
    nav_curve = engine.run(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        trading_dates=trading_dates,
        price_data=price_data
    )
    
    # 获取交易记录
    trades = engine.get_trades()
    
    return nav_curve, trades

def _append_dict_to_csv(file_path: Path, row: dict, fieldnames: list = None):
    """把一个 dict 追加到 CSV（如果不存在则写 header）
    
    Args:
        file_path: 目标文件 Path
        row: 要写入的一行 dict
        fieldnames: 列顺序列表（如果 None 则使用 row.keys() 的顺序）
    """
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.exists()

    # 使用 utf-8-sig 以便 Excel 直接识别中文
    with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
        if fieldnames is None:
            fieldnames_local = list(row.keys())
        else:
            fieldnames_local = fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames_local)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _generate_run_id(args) -> str:
    """生成唯一的回测ID
    
    Args:
        args: 命令行参数
        
    Returns:
        回测ID字符串（时间戳_参数hash）
    """
    # 使用时间戳和关键参数生成唯一ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 将关键参数拼接成字符串并计算hash
    params_str = f"{args.start_date}_{args.end_date}_{args.model_version}_{args.top_n}_{args.position_sizing}_{args.rebalance_freq}_{args.initial_capital}_{args.sell_timing}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    return f"{timestamp}_{params_hash}"


def _append_trades_to_cumulative_file(
    trades: pd.DataFrame,
    args,
    reporter: 'Reporter',
    run_id: str,
    run_time: str
):
    """将交易记录追加到累加文件中，并计算交易盈亏
    
    Args:
        trades: 本次回测的交易记录DataFrame
        args: 命令行参数
        reporter: Reporter实例
        run_id: 回测ID
        run_time: 回测执行时间
    """
    if trades is None or len(trades) == 0:
        logger.info("本次回测无交易记录，跳过累加文件写入")
        return
    
    try:
        # 累加文件路径
        cumulative_file = Path(reporter.output_dir) / "ml_backtest_trades_runs.csv"
        
        # 准备要添加的参数列（中文列名）
        model_version_str = "最新版本" if args.model_version is None else str(args.model_version)
        
        # 定义所有字段及其顺序（先是核心参数，再是原有交易字段）
        fieldnames = [
            "回测ID",
            "回测时间",
            "开始日期",
            "结束日期",
            "模型版本",
            "TopN",
            "仓位管理",
            "调仓频率",
            "初始资金",
            "卖出时机",
            # 原有交易记录字段
            "交易日期",
            "股票代码",
            "操作",
            "成交价格",
            "成交股数",
            "成交金额",
            "交易成本",
            "买入价格",
            "收益金额",
            "收益率"
        ]
        
        # 构建买入价格字典（FIFO：先进先出）
        # 存储格式：buy_prices[股票代码] = [{'price': 买入价格, 'amount': 买入金额, 'cost': 买入成本, 'shares': 股数}, ...]
        buy_prices = defaultdict(deque)
        
        # 第一遍遍历：记录所有买入交易
        for _, trade in trades.iterrows():
            action = trade.get('action', '')
            if action == 'buy':
                stock = trade.get('stock', '')
                buy_prices[stock].append({
                    'price': trade.get('price', 0),
                    'amount': trade.get('amount', 0),
                    'cost': trade.get('cost', 0),
                    'shares': trade.get('shares', 0)
                })
        
        # 第二遍遍历：计算卖出交易的盈亏，并写入累加文件
        for _, trade in trades.iterrows():
            action = trade.get('action', '')
            stock = trade.get('stock', '')
            
            # 初始化收益字段
            buy_price_value = ''
            profit_amount = ''
            profit_pct = ''
            
            # 如果是卖出交易，计算收益
            if action == 'sell':
                # 从 FIFO 队列中获取对应的买入信息
                if stock in buy_prices and len(buy_prices[stock]) > 0:
                    buy_info = buy_prices[stock].popleft()  # 先进先出
                    
                    # 计算收益
                    # 买入成本 = 买入金额 + 买入交易成本
                    buy_cost = buy_info['amount'] + buy_info['cost']
                    
                    # 卖出金额和成本
                    sell_amount = trade.get('amount', 0)
                    sell_cost = trade.get('cost', 0)
                    
                    # 收益金额 = 卖出金额 - 买入成本 - 卖出成本
                    profit_amount_value = sell_amount - buy_cost - sell_cost
                    
                    # 收益率 = (收益金额 / 买入成本) × 100%
                    if buy_cost > 0:
                        profit_pct_value = (profit_amount_value / buy_cost) * 100
                        profit_pct = f"{profit_pct_value:.2f}%"
                    else:
                        profit_pct = "0.00%"
                    
                    # 格式化收益金额（保留2位小数）
                    profit_amount = f"{profit_amount_value:.2f}"
                    buy_price_value = buy_info['price']
                else:
                    # 没有找到对应的买入记录（理论上不应发生）
                    logger.warning(f"卖出交易未找到对应买入记录: {stock} @ {trade.get('date', '')}")
            
            # 构建完整的交易记录（参数 + 交易明细）
            trade_with_params = {
                "回测ID": run_id,
                "回测时间": run_time,
                "开始日期": args.start_date,
                "结束日期": args.end_date,
                "模型版本": model_version_str,
                "TopN": args.top_n,
                "仓位管理": args.position_sizing,
                "调仓频率": args.rebalance_freq,
                "初始资金": args.initial_capital,
                "卖出时机": args.sell_timing,
                # 原有交易字段
                "交易日期": trade.get('date', ''),
                "股票代码": stock,
                "操作": "买入" if action == 'buy' else "卖出" if action == 'sell' else action,
                "成交价格": trade.get('price', ''),
                "成交股数": trade.get('shares', ''),
                "成交金额": trade.get('amount', ''),
                "交易成本": trade.get('cost', ''),
                "买入价格": buy_price_value,
                "收益金额": profit_amount,
                "收益率": profit_pct
            }
            
            # 追加到累加文件
            _append_dict_to_csv(cumulative_file, trade_with_params, fieldnames=fieldnames)
        
        logger.info(f"本次回测 {len(trades)} 笔交易已追加到累加文件: {cumulative_file}")
        
    except Exception as ex:
        # 记录追加失败不影响回测结果输出，但记录错误信息
        logger.exception(f"写交易记录到累加文件失败: {ex}")


def equal_count_grouping(scores: pd.Series, n_groups: int = 10) -> pd.Series:
    """按预测分数等数量分组
    
    将样本按分数降序排序后，切分成近似等大小的 n_groups 组。
    前面的组可能多1个样本（如果总数不能被 n_groups 整除）。
    
    Args:
        scores: 预测分数 Series（index 为 ts_code 或其他标识）
        n_groups: 分组数量
        
    Returns:
        pd.Series: 分组标签（1 表示最高分组，n_groups 表示最低分组）
    """
    if len(scores) == 0:
        return pd.Series(dtype=int)
    
    # 按分数降序排序
    sorted_scores = scores.sort_values(ascending=False)
    n_samples = len(sorted_scores)
    
    # 计算每组的基础大小和需要额外加1的组数
    base_size = n_samples // n_groups
    extra = n_samples % n_groups
    
    # 分配组标签
    group_labels = []
    for i in range(n_groups):
        # 前 extra 组每组多1个
        group_size = base_size + (1 if i < extra else 0)
        group_labels.extend([i + 1] * group_size)
    
    # 创建分组 Series（保持原始索引顺序）
    groups = pd.Series(group_labels, index=sorted_scores.index)
    return groups


def evaluate_daily(
    date: str,
    signal: MLSignal,
    universe: list,
    features_df: pd.DataFrame,
    label_column: str,
    n_groups: int = 10,
    topk: int = None
) -> tuple:
    """评估单日的 ML 信号质量
    
    Args:
        date: 交易日期（YYYYMMDD 格式字符串）
        signal: ML 信号生成器
        universe: 股票池
        features_df: 当日特征数据（包含 label 列）
        label_column: 真实收益标签列名（如 y_ret_5）
        n_groups: 分组数量
        topk: TopK 指标的 K（若为 None 则不计算）
        
    Returns:
        (daily_metrics, group_details) 元组
        - daily_metrics: dict，日度指标
        - group_details: list of dict，每组的详细信息
    """
    # 使用 MLSignal.generate_ranked 得到排序候选
    ranked = signal.generate_ranked(
        date=pd.Timestamp(date),
        universe=universe,
        data={'features': features_df}
    )
    
    if not ranked or len(ranked) == 0:
        logger.warning(f"{date}: 无排序候选，跳过评估")
        return None, None
    
    # 转换为 DataFrame
    ranked_df = pd.DataFrame(ranked, columns=['ts_code', 'score'])
    
    # 与 label 列对齐
    if label_column not in features_df.columns:
        logger.warning(f"{date}: 特征数据中缺少标签列 {label_column}，跳过评估")
        return None, None
    
    # 合并预测分数和真实标签
    eval_df = ranked_df.merge(
        features_df[['ts_code', label_column]],
        on='ts_code',
        how='left'
    )
    
    # 过滤缺失标签的样本
    eval_df = eval_df.dropna(subset=[label_column])
    
    if len(eval_df) == 0:
        logger.warning(f"{date}: 所有样本的标签都缺失，跳过评估")
        return None, None
    
    n_samples = len(eval_df)
    
    # 计算 RankIC (Spearman)
    rank_ic = eval_df['score'].corr(eval_df[label_column], method='spearman')
    
    # 等数量分组
    eval_df['group'] = equal_count_grouping(eval_df.set_index('ts_code')['score'], n_groups).values
    
    # 计算各组平均真实收益
    group_stats = eval_df.groupby('group').agg({
        'ts_code': 'count',
        label_column: 'mean',
        'score': 'mean'
    }).rename(columns={
        'ts_code': 'count',
        label_column: 'avg_real_return',
        'score': 'avg_score'
    })
    
    # Top 组和 Bottom 组
    top_group_return = group_stats.loc[1, 'avg_real_return'] if 1 in group_stats.index else None
    bottom_group_return = group_stats.loc[n_groups, 'avg_real_return'] if n_groups in group_stats.index else None
    long_short_return = top_group_return - bottom_group_return if (top_group_return is not None and bottom_group_return is not None) else None
    
    # TopK 平均真实收益
    topk_return = None
    if topk is not None and topk > 0:
        topk_samples = eval_df.nlargest(min(topk, len(eval_df)), 'score')
        if len(topk_samples) > 0:
            topk_return = topk_samples[label_column].mean()
    
    # 日度指标
    daily_metrics = {
        '交易日期': date,
        '样本数': n_samples,
        'RankIC': rank_ic,
        'TopK平均收益': topk_return,
        'Top组平均收益': top_group_return,
        'Bottom组平均收益': bottom_group_return,
        '多空收益': long_short_return
    }
    
    # 分组明细
    group_details = []
    for group_id, row in group_stats.iterrows():
        group_details.append({
            '交易日期': date,
            '组号': group_id,
            '组内股票数': int(row['count']),
            '组内平均真实收益': row['avg_real_return'],
            '组内平均预测分数': row['avg_score']
        })
    
    return daily_metrics, group_details


def export_evaluation_panel(
    signal: MLSignal,
    universe_obj: BasicUniverse,
    features_by_date: dict,
    trading_dates: list,
    label_column: str,
    output_dir: str,
    output_name: str,
    n_groups: int = 10,
    topk: int = None,
    args = None
):
    """导出评估面板 CSV
    
    Args:
        signal: ML 信号生成器
        universe_obj: 股票池对象
        features_by_date: 按日期组织的特征数据字典 {date_str: features_df}
        trading_dates: 交易日列表（pd.Timestamp）
        label_column: 真实收益标签列名
        output_dir: 输出目录
        output_name: 输出文件名前缀
        n_groups: 分组数量
        topk: TopK 指标的 K
        args: 命令行参数（用于汇总 CSV）
    """
    logger.info("=" * 60)
    logger.info("开始导出评估面板...")
    logger.info(f"标签列: {label_column}")
    logger.info(f"分组数: {n_groups}")
    logger.info(f"TopK: {topk}")
    logger.info("=" * 60)
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 准备输出文件
    daily_file = output_dir_path / f"{output_name}_eval_daily.csv"
    groups_file = output_dir_path / f"{output_name}_eval_groups.csv"
    summary_file = output_dir_path / f"{output_name}_eval_summary.csv"
    
    # 删除旧文件（如果存在）
    for f in [daily_file, groups_file]:
        if f.exists():
            f.unlink()
    
    # 遍历每个交易日进行评估
    all_daily_metrics = []
    all_group_details = []
    
    for date_ts in trading_dates:
        date_str = date_ts.strftime('%Y%m%d')
        
        # 检查是否有特征数据
        if date_str not in features_by_date:
            logger.debug(f"{date_str}: 无特征数据，跳过评估")
            continue
        
        features_df = features_by_date[date_str]
        
        # 获取当日股票池
        universe = universe_obj.get_stocks(date_ts)
        
        # 评估当日
        daily_metrics, group_details = evaluate_daily(
            date=date_str,
            signal=signal,
            universe=universe,
            features_df=features_df,
            label_column=label_column,
            n_groups=n_groups,
            topk=topk
        )
        
        if daily_metrics is not None:
            all_daily_metrics.append(daily_metrics)
        
        if group_details is not None:
            all_group_details.extend(group_details)
    
    # 写入日度评估 CSV
    if all_daily_metrics:
        daily_df = pd.DataFrame(all_daily_metrics)
        daily_df.to_csv(daily_file, index=False, encoding='utf-8-sig')
        logger.info(f"日度评估 CSV 已保存: {daily_file} ({len(daily_df)} 行)")
    else:
        logger.warning("没有日度评估数据")
    
    # 写入分组明细 CSV
    if all_group_details:
        groups_df = pd.DataFrame(all_group_details)
        groups_df.to_csv(groups_file, index=False, encoding='utf-8-sig')
        logger.info(f"分组明细 CSV 已保存: {groups_file} ({len(groups_df)} 行)")
    else:
        logger.warning("没有分组明细数据")
    
    # 计算汇总指标
    if all_daily_metrics:
        daily_df = pd.DataFrame(all_daily_metrics)
        
        # 聚合指标
        rank_ic_mean = daily_df['RankIC'].mean()
        rank_ic_std = daily_df['RankIC'].std()
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-10 else None
        
        topk_mean = daily_df['TopK平均收益'].mean() if 'TopK平均收益' in daily_df.columns else None
        long_short_mean = daily_df['多空收益'].mean() if '多空收益' in daily_df.columns else None
        
        # 构建汇总记录
        summary_record = {
            '回测时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            '开始日期': args.start_date if args else None,
            '结束日期': args.end_date if args else None,
            '标签列': label_column,
            '模型版本': args.model_version if args and args.model_version is not None else '最新版本',
            'TopN': args.top_n if args else None,
            '仓位管理': args.position_sizing if args else None,
            '调仓频率': args.rebalance_freq if args else None,
            '初始资金': args.initial_capital if args else None,
            '卖出时机': args.sell_timing if args else None,
            '分组数': n_groups,
            'TopK': topk,
            '评估天数': len(daily_df),
            'RankIC均值': rank_ic_mean,
            'RankIC标准差': rank_ic_std,
            'RankIC_IR': rank_ic_ir,
            'TopK平均收益': topk_mean,
            '多空平均收益': long_short_mean,
        }
        
        # 追加到汇总 CSV
        summary_fieldnames = list(summary_record.keys())
        _append_dict_to_csv(summary_file, summary_record, fieldnames=summary_fieldnames)
        logger.info(f"汇总指标 CSV 已保存: {summary_file}")
        
        # 打印汇总信息
        logger.info("=" * 60)
        logger.info("评估面板汇总指标:")
        logger.info(f"  评估天数: {len(daily_df)}")
        logger.info(f"  RankIC 均值: {rank_ic_mean:.4f}")
        logger.info(f"  RankIC 标准差: {rank_ic_std:.4f}")
        logger.info(f"  RankIC IR: {rank_ic_ir:.4f}" if rank_ic_ir is not None else "  RankIC IR: N/A")
        logger.info(f"  TopK 平均收益: {topk_mean:.4f}" if topk_mean is not None else "  TopK 平均收益: N/A")
        logger.info(f"  多空平均收益: {long_short_mean:.4f}" if long_short_mean is not None else "  多空平均收益: N/A")
        logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行 ML 信号回测")

    # 回测专用参数
    parser.add_argument("--start-date", type=str, required=True,
                        help="回测开始日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", type=str, required=True,
                        help="回测结束日期，格式 YYYYMMDD")
    parser.add_argument("--initial-capital", type=float, default=500000.0,
                        help="初始资金，默认 500000")
    parser.add_argument("--label", type=str, default=None,
                        choices=["y_ret_5", "y_ret_10", "y_ret_20"],
                        help="标签选择（y_ret_5|y_ret_10|y_ret_20）。若未指定则使用模型训练时的标签")
    parser.add_argument("--sell-timing", type=str, default="open",
                        choices=["open", "close"],
                        help="卖出时机，默认 open")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.* 配置",
    )
    parser.add_argument("--output-name", type=str, default="ml_backtest",
                        help="报告输出名称，默认 ml_backtest")

    # 评估面板参数
    parser.add_argument("--export-eval", action="store_true", help="导出评估面板 CSV（默认开启）")
    parser.add_argument("--no-export-eval", action="store_false", dest="export_eval",
                        help="禁用评估面板导出")
    parser.add_argument("--eval-groups", type=int, default=10,
                        help="评估面板分组数量，默认 10")
    parser.add_argument("--eval-topk", type=int, default=None,
                        help="评估面板 TopK 指标的 K，默认使用 --top-n")
    parser.set_defaults(export_eval=True)

    # 公共策略参数（模型、组合、股票池、止损、ECT）
    add_trading_args(parser)
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger()
    
    # 初始化组件以加载模型元数据
    from src.lazybull.ml import ModelRegistry
    storage = Storage(root_path=args.data_root)
    models_dir = get_models_root(str(Path(args.data_root) / "models") if args.data_root else None)
    reports_dir = get_reports_root(str(Path(args.data_root) / "reports") if args.data_root else None)
    registry = ModelRegistry(models_dir=models_dir)
    
    # 加载模型元数据以获取 label_column
    try:
        _, model_metadata = registry.load_model(version=args.model_version)
        model_label = model_metadata.get('label_column', 'y_ret_5')
    except ValueError as e:
        # 没有找到模型或模型版本不存在
        logger.warning(f"无法加载模型元数据: {e}")
        logger.warning("将使用默认标签 y_ret_5。如果需要训练模型，请先运行 train_ml_model.py")
        model_label = 'y_ret_5'
    except FileNotFoundError as e:
        # 模型文件不存在
        logger.warning(f"模型文件缺失: {e}")
        logger.warning("请先运行 train_ml_model.py 训练模型")
        model_label = 'y_ret_5'
    except Exception as e:
        # 其他未预期错误
        logger.error(f"加载模型元数据时发生未预期错误: {e}")
        logger.warning("将使用默认标签 y_ret_5，但回测结果可能不准确")
        model_label = 'y_ret_5'
    
    # 处理 label 参数
    if args.label is not None:
        # 用户显式指定了 label
        # 如果同时指定了 model_version，需要校验一致性
        if args.model_version is not None and model_label != args.label:
            logger.error("=" * 60)
            logger.error("参数错误：模型版本与标签不一致")
            logger.error("=" * 60)
            logger.error(f"模型版本 v{args.model_version} 训练时使用的标签: {model_label}")
            logger.error(f"您指定的标签: {args.label}")
            logger.error("")
            logger.error("解决方案：")
            logger.error(f"1. 移除 --label 参数，使用模型训练时的标签（{model_label}）")
            logger.error(f"2. 移除 --model-version 参数，自动加载使用 {args.label} 训练的最新模型")
            logger.error(f"3. 使用正确的 --model-version，该模型应使用 {args.label} 标签训练")
            logger.error("=" * 60)
            sys.exit(1)
        selected_label = args.label
    else:
        # 用户未指定 label，使用模型训练时的标签
        selected_label = model_label
        logger.info(f"未指定 --label 参数，使用模型训练时的标签: {selected_label}")
    
    # 处理 rebalance_freq 参数：若未指定，根据 label 自动设置
    if args.rebalance_freq is None:
        # 从 label 中提取数字作为默认调仓频率
        args.rebalance_freq = infer_rebalance_freq_from_label(selected_label, default=10)
        logger.info(f"未指定 --rebalance-freq 参数，根据标签 {selected_label} 自动设置为: {args.rebalance_freq}")
    
    logger.info("=" * 60)
    logger.info("ML 信号回测")
    logger.info("=" * 60)
    logger.info(f"回测区间: {args.start_date} 至 {args.end_date}")
    logger.info(f"初始资金: {args.initial_capital}")
    logger.info(f"标签: {selected_label}")
    logger.info(f"调仓频率: {args.rebalance_freq} 个交易日")
    logger.info(f"模型版本: {args.model_version or '最新版本'}")
    logger.info(f"Top N: {args.top_n}")
    logger.info(f"仓位管理: {args.position_sizing}")
    if args.max_weight_per_stock is not None:
        logger.info(f"单股最大权重: {args.max_weight_per_stock:.2%}")
    if args.max_per_industry is not None and args.max_per_industry > 0:
        logger.info(f"单行业最大持仓数量: {args.max_per_industry}")
    logger.info(f"止损功能: {'启用' if args.stop_loss_enabled else '禁用'}")
    if args.stop_loss_enabled:
        logger.info(f"  - 回撤止损: {args.stop_loss_drawdown_pct}%")
        logger.info(f"  - 移动止损: {'启用' if args.stop_loss_trailing_enabled else '禁用'}")
        if args.stop_loss_trailing_enabled:
            logger.info(f"  - 移动止损阈值: {args.stop_loss_trailing_pct}%")
        logger.info(f"  - 连续跌停止损: {args.stop_loss_consecutive_limit_down} 天")
    
    logger.info(f"ECT功能: {'启用' if args.equity_curve_enabled else '禁用'}")
    if args.equity_curve_enabled:
        logger.info(f"  - 回撤阈值: {args.equity_curve_drawdown_thresholds}")
        logger.info(f"  - 仓位系数: {args.equity_curve_exposure_levels}")
        logger.info(f"  - 均线窗口: 短期={args.equity_curve_ma_short}, 长期={args.equity_curve_ma_long}")
        logger.info(f"  - 恢复模式: {args.equity_curve_recovery_mode}")
        logger.info(f"  - 恢复步长: {args.equity_curve_recovery_step}")
        logger.info(f"  - 恢复等待周期: {args.equity_curve_recovery_delay_periods} 个调仓周期")
    if args.take_profit_threshold is not None:
        logger.info(f"整体止盈: 启用 (threshold={args.take_profit_threshold:.2%}, refill={'启用' if args.take_profit_refill else '关闭'})")
    if args.signal_gate_quality_enabled:
        logger.info(
            f"滚动质量监控: 启用 (window={args.signal_gate_quality_window}, "
            f"threshold={args.signal_gate_quality_threshold}, halflife={args.signal_gate_quality_halflife})"
        )
    if args.signal_gate_dynamic_topn:
        logger.info(
            f"动态Top-N: 启用 (high={args.signal_gate_topn_high_multiplier}, "
            f"low={args.signal_gate_topn_low_multiplier})"
        )

    # 构建统一策略配置
    trading_config = TradingConfig.from_args(args)

    try:
        # 初始化组件（registry 已在前面初始化）
        loader = DataLoader(storage)

        # 通过 TradingConfig 创建止损 / ECT 配置
        stop_loss_config = trading_config.create_stop_loss_config()
        equity_curve_config = trading_config.create_equity_curve_config()

        # 1. 加载数据
        trade_cal, stock_basic, daily_data, features_by_date = load_backtest_data(
            loader, storage, args.start_date, args.end_date
        )

        if len(features_by_date) == 0:
            logger.error("，无法运行回测")
            sys.exit(1)

        # 2. 准备价格数据
        price_data = prepare_price_data(daily_data)

        # 3. 创建股票池
        universe = BasicUniverse(
            stock_basic=stock_basic,
            exclude_st=trading_config.exclude_st,
            min_list_days=trading_config.min_list_days,
            markets=['主板'],
            verbose=False,
        )

        # 4. 创建 ML 信号（通过公共工厂函数）
        signal = create_signal(trading_config, models_dir=models_dir)
        
        # 打印模型信息
        model_info = signal.get_model_info()
        logger.info(f"使用模型: {model_info['version_str']}")
        logger.info(f"训练区间: {model_info['train_start_date']} 至 {model_info['train_end_date']}")
        logger.info(f"特征数: {model_info['feature_count']}")
        logger.info(f"训练样本数: {model_info['n_samples']}")
        #logger.info(f"性能指标: \n{model_info['performance_metrics']}")
        # 提取 validation_daily 部分引用
        vd = model_info['performance_metrics']['validation_daily']
        logger.info(
            f"性能指标 (Validation Daily): "
            f"RankIC Mean: {vd['daily_rankic_mean']:.3f}, "
            f"RankIC IR: {vd['daily_rankic_ir']:.3f}, "
            f"Top30 提升均值: {vd['diagnostic_Top30_相对全市场提升_均值']:.3f}"
        )        
        # 5. 准备交易日列表
        trading_dates = trade_cal[
            (trade_cal['cal_date'] >= args.start_date) & 
            (trade_cal['cal_date'] <= args.end_date) & 
            (trade_cal['is_open'] == 1)
        ]['cal_date'].tolist()
        trading_dates = [pd.Timestamp(d) for d in trading_dates]
        
        # 6. 运行回测
        nav_curve, trades = run_ml_backtest(
            signal=signal,
            universe=universe,
            start_date=args.start_date,
            end_date=args.end_date,
            trading_dates=trading_dates,
            price_data=price_data,
            features_by_date=features_by_date,
            initial_capital=trading_config.initial_capital,
            rebalance_freq=trading_config.rebalance_freq,
            stagger_tranches=trading_config.stagger_tranches,
            stop_loss_config=stop_loss_config,
            equity_curve_config=equity_curve_config,
            sell_timing=args.sell_timing,
            max_weight_per_stock=trading_config.max_weight_per_stock,
            max_per_industry=trading_config.max_per_industry,
            stock_basic=stock_basic,
            holding_bonus_enabled=trading_config.holding_bonus_enabled,
            holding_bonus_sigma=trading_config.holding_bonus_sigma,
            profit_extension_mode=trading_config.profit_extension_mode,
            profit_extension_strength_threshold=trading_config.profit_extension_strength_threshold,
            profit_extension_strength_weights=trading_config.profit_extension_strength_weights,
            industry_rotation_enhanced=trading_config.industry_rotation_enhanced,
            industry_rotation_alpha=trading_config.industry_rotation_alpha,
            position_sizing=trading_config.position_sizing,
            kelly_vol_window=trading_config.kelly_vol_window,
            kelly_max_leverage=trading_config.kelly_max_leverage,
            enable_profit_based_holding=trading_config.enable_profit_based_holding,
            early_exit_loss_threshold=trading_config.early_exit_loss_threshold,
            early_exit_holding_ratio=trading_config.early_exit_holding_ratio,
            profit_extension_threshold=trading_config.profit_extension_threshold,
            profit_extension_days=trading_config.profit_extension_days,
            use_atr_for_early_exit=trading_config.use_atr_for_early_exit,
            atr_multiplier=trading_config.atr_multiplier,
            early_exit_mode=trading_config.early_exit_mode,
            early_exit_strength_protect_threshold=trading_config.early_exit_strength_protect_threshold,
            early_exit_max_reprieves=trading_config.early_exit_max_reprieves,
            take_profit_threshold=trading_config.take_profit_threshold,
            take_profit_refill=trading_config.take_profit_refill,
            enable_early_rebalance_on_empty=trading_config.enable_early_rebalance_on_empty,
            signal_gate_quality_enabled=trading_config.signal_gate_quality_enabled,
            signal_gate_quality_window=trading_config.signal_gate_quality_window,
            signal_gate_quality_threshold=trading_config.signal_gate_quality_threshold,
            signal_gate_quality_halflife=trading_config.signal_gate_quality_halflife,
            signal_gate_dynamic_topn=trading_config.signal_gate_dynamic_topn,
            signal_gate_topn_high_multiplier=trading_config.signal_gate_topn_high_multiplier,
            signal_gate_topn_low_multiplier=trading_config.signal_gate_topn_low_multiplier,
            trading_config=trading_config,
            data_storage=storage,
        )

        # 7. 生成报告
        reporter = Reporter(output_dir=reports_dir)
        stats = reporter.generate_report(nav_curve, trades, output_name=args.output_name)
        
        logger.info("=" * 60)
        logger.info("回测完成！")
        logger.info(f"报告已保存到: {reports_dir}")
        logger.info("=" * 60)

        # ------------------ 追加交易记录到累加文件 ------------------
        try:
            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_id = _generate_run_id(args)
            _append_trades_to_cumulative_file(trades, args, reporter, run_id, run_time)
        except Exception as ex:
            logger.exception(f"写交易记录到累加文件失败: {ex}")
        # -------------------------------------------------------------

        # ------------------ 追加写入回测记录到固定 CSV（不会覆盖老数据） ------------------
        try:
            # 构建要写入的一行记录（可按需扩展字段）
            record = {
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "model_version": args.model_version if args.model_version is not None else "latest",
                "top_n": args.top_n,
                "position_sizing": args.position_sizing,
                "rebalance_freq": args.rebalance_freq,
                "initial_capital": args.initial_capital,
                "sell_timing": args.sell_timing,
                "stop_loss_enabled": args.stop_loss_enabled,
                "report_name": args.output_name,
                # 以下尽量从 nav_curve / stats 中提取常用指标（若不存在则写 None）
                "nav_final": None,
                "total_return": None,
                "max_drawdown": None,
                "sharpe": None,
            }

            # 从 nav_curve 尝试取最终净值或组合市值
            if isinstance(nav_curve, pd.DataFrame) and not nav_curve.empty:
                if 'nav' in nav_curve.columns:
                    record["nav_final"] = float(nav_curve['nav'].iloc[-1])
                elif 'portfolio_value' in nav_curve.columns:
                    record["nav_final"] = float(nav_curve['portfolio_value'].iloc[-1])
                else:
                    # 尝试找到第一个数值列作替代
                    numeric_cols = nav_curve.select_dtypes(include='number').columns.tolist()
                    if numeric_cols:
                        record["nav_final"] = float(nav_curve[numeric_cols[-1]].iloc[-1])

            # 从 stats 字典中安全读取指标（字段名以实际 stats 为准）
            if isinstance(stats, dict):
                record["total_return"] = stats.get("total_return") or stats.get("收益率") or stats.get("return")
                record["max_drawdown"] = stats.get("max_drawdown") or stats.get("最大回撤")
                record["sharpe"] = stats.get("sharpe") or stats.get("夏普比率")

            # 写入到 Reporter 的 output_dir（复用已有目录）
            log_file = Path(reporter.output_dir) / "backtest_runs.csv"

            # 指定列顺序，保证稳定性；如果需要新增字段请在这里同步修改
            fieldnames = [
                "run_time", "start_date", "end_date", "model_version", "top_n", "position_sizing",
                "rebalance_freq", "initial_capital", "sell_timing", "stop_loss_enabled",
                "report_name", "nav_final", "total_return", "max_drawdown", "sharpe"
            ]

            _append_dict_to_csv(log_file, record, fieldnames=fieldnames)
            logger.info(f"本次回测记录已追加到: {log_file}")
        except Exception as ex:
            # 记录追加失败不影响回测结果输出，但记录错误信息
            logger.exception(f"写回测记录到 CSV 失败: {ex}")
        # ---------------------------------------------------------------------------
        
        # ------------------ 导出评估面板 ------------------
        if args.export_eval:
            try:
                # 设置 eval_topk，默认使用 top_n
                eval_topk = args.eval_topk if args.eval_topk is not None else args.top_n
                
                export_evaluation_panel(
                    signal=signal,
                    universe_obj=universe,
                    features_by_date=features_by_date,
                    trading_dates=trading_dates,
                    label_column=selected_label,
                    output_dir=reports_dir,
                    output_name=args.output_name,
                    n_groups=args.eval_groups,
                    topk=eval_topk,
                    args=args
                )
            except Exception as ex:
                logger.exception(f"导出评估面板失败: {ex}")
        # -------------------------------------------------

    except Exception as e:
        logger.error(f"回测失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
