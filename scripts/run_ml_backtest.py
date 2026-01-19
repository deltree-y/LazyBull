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
import sys
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.backtest import BacktestEngine, BacktestEngineML, Reporter
from src.lazybull.common.cost import CostModel
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse


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
        'close', 'close_adj',

        # 交易状态相关（用于 is_tradeable / is_limit_up / is_suspended 等）
        'filter_is_suspended', 'is_limit_up', 'is_limit_down',
        'vol', 'pct_chg',

        # 股票池基础过滤可能用到的字段（按存在保留）
        'filter_is_st', 'filter_list_days', 'tradable'
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
    missing_status_cols = [c for c in ['filter_is_suspended', 'is_limit_up', 'is_limit_down'] if c not in price_data.columns]
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
    cost_model: CostModel = None
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
        
    Returns:
        (nav_curve, trades) 元组
    """
    logger.info("开始运行 ML 信号回测...")
    
    # 创建回测引擎（需要稍作调整以支持特征数据）
    engine = BacktestEngineML(
        universe=universe,
        signal=signal,
        features_by_date=features_by_date,
        initial_capital=initial_capital,
        cost_model=cost_model or CostModel(),
        rebalance_freq=rebalance_freq,
        price_type="close",
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行 ML 信号回测")
    
    # 回测参数
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="回测开始日期，格式 YYYYMMDD"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="回测结束日期，格式 YYYYMMDD"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=500000.0,
        help="初始资金，默认 500000"
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=10,
        #choices=["D", "W", "M"],
        help="调仓频率, 单位为交易日天数，默认 10"
    )
    
    # ML 信号参数
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="模型版本号，默认使用最新版本"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="选择 Top N 只股票，默认 5"
    )
    parser.add_argument(
        "--weight-method",
        type=str,
        default="equal",
        choices=["equal", "score"],
        help="权重方法，equal=等权，score=按分数加权，默认 equal"
    )
    
    # 股票池参数
    parser.add_argument(
        "--exclude-st",
        action="store_true",
        default=True,
        help="排除 ST 股票（默认开启）"
    )
    parser.add_argument(
        "--include-st",
        action="store_false",
        dest="exclude_st",
        help="包含 ST 股票"
    )
    parser.add_argument(
        "--min-list-days",
        type=int,
        default=60,
        help="最小上市天数，默认 60"
    )
    
    # 其他参数
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据根目录，默认 ./data"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="ml_backtest",
        help="报告输出名称，默认 ml_backtest"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("ML 信号回测")
    logger.info("=" * 60)
    logger.info(f"回测区间: {args.start_date} 至 {args.end_date}")
    logger.info(f"初始资金: {args.initial_capital}")
    logger.info(f"调仓频率: {args.rebalance_freq}")
    logger.info(f"模型版本: {args.model_version or '最新版本'}")
    logger.info(f"Top N: {args.top_n}")
    logger.info(f"权重方法: {args.weight_method}")
    
    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        
        # 1. 加载数据
        trade_cal, stock_basic, daily_data, features_by_date = load_backtest_data(
            loader, storage, args.start_date, args.end_date
        )
        
        if len(features_by_date) == 0:
            logger.error("没有特征数据，无法运行回测")
            sys.exit(1)
        
        # 2. 准备价格数据
        price_data = prepare_price_data(daily_data)
        
        # 3. 创建股票池
        universe = BasicUniverse(
            stock_basic=stock_basic,
            exclude_st=args.exclude_st,
            min_list_days=args.min_list_days,
            markets=['主板'],  # 可根据需要调整
        )
        
        # 4. 创建 ML 信号
        signal = MLSignal(
            top_n=args.top_n,
            model_version=args.model_version,
            models_dir=f"{args.data_root}/models",
            weight_method=args.weight_method
        )
        
        # 打印模型信息
        model_info = signal.get_model_info()
        logger.info(f"使用模型: {model_info['version_str']}")
        logger.info(f"训练区间: {model_info['train_start_date']} 至 {model_info['train_end_date']}")
        logger.info(f"特征数: {model_info['feature_count']}")
        logger.info(f"训练样本数: {model_info['n_samples']}")
        logger.info(f"性能指标: \n{model_info['performance_metrics']}")
        
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
            initial_capital=args.initial_capital,
            rebalance_freq=args.rebalance_freq
        )
        
        # 7. 生成报告
        reporter = Reporter(output_dir=f"{args.data_root}/reports")
        stats = reporter.generate_report(nav_curve, trades, output_name=args.output_name)
        
        logger.info("=" * 60)
        logger.info("回测完成！")
        logger.info(f"报告已保存到: {args.data_root}/reports/")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
