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

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import csv
from loguru import logger

from src.lazybull.backtest import BacktestEngine, BacktestEngineML, Reporter
from src.lazybull.common.cost import CostModel
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse
from src.lazybull.risk.stop_loss import StopLossConfig, create_stop_loss_config_from_dict


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
    cost_model: CostModel = None,
    stop_loss_config: StopLossConfig = None,
    sell_timing: str = 'open'
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
        stop_loss_config=stop_loss_config,
        sell_timing=sell_timing,
        completion_window_days=5,
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
    params_str = f"{args.start_date}_{args.end_date}_{args.model_version}_{args.top_n}_{args.weight_method}_{args.rebalance_freq}_{args.initial_capital}_{args.sell_timing}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    return f"{timestamp}_{params_hash}"


def _append_trades_to_cumulative_file(
    trades: pd.DataFrame,
    args,
    reporter: 'Reporter',
    run_id: str,
    run_time: str
):
    """将交易记录追加到累加文件中
    
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
            "权重方法",
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
        
        # 遍历每笔交易，添加参数列后写入累加文件
        for _, trade in trades.iterrows():
            # 构建完整的交易记录（参数 + 交易明细）
            trade_with_params = {
                "回测ID": run_id,
                "回测时间": run_time,
                "开始日期": args.start_date,
                "结束日期": args.end_date,
                "模型版本": model_version_str,
                "TopN": args.top_n,
                "权重方法": args.weight_method,
                "调仓频率": args.rebalance_freq,
                "初始资金": args.initial_capital,
                "卖出时机": args.sell_timing,
                # 原有交易字段（处理可能不存在的列）
                "交易日期": trade.get('date', ''),
                "股票代码": trade.get('stock', ''),
                "操作": "买入" if trade.get('action') == 'buy' else "卖出" if trade.get('action') == 'sell' else trade.get('action', ''),
                "成交价格": trade.get('price', ''),
                "成交股数": trade.get('shares', ''),
                "成交金额": trade.get('amount', ''),
                "交易成本": trade.get('cost', ''),
                "买入价格": trade.get('buy_price', ''),
                "收益金额": trade.get('profit_amount', ''),
                "收益率": trade.get('profit_pct', '')
            }
            
            # 追加到累加文件
            _append_dict_to_csv(cumulative_file, trade_with_params, fieldnames=fieldnames)
        
        logger.info(f"本次回测 {len(trades)} 笔交易已追加到累加文件: {cumulative_file}")
        
    except Exception as ex:
        # 记录追加失败不影响回测结果输出，但记录错误信息
        logger.exception(f"写交易记录到累加文件失败: {ex}")


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
    parser.add_argument(
        "--sell-timing",
        type=str,
        default="open",
        choices=["open", "close"],
        help="卖出时机，open=开盘价卖出，close=收盘价卖出，默认 open"
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
    
    # 止损参数
    parser.add_argument(
        "--stop-loss-enabled",
        action="store_true",
        default=False,
        help="启用止损功能"
    )
    parser.add_argument(
        "--stop-loss-drawdown-pct",
        type=float,
        default=20.0,
        help="回撤止损阈值（百分比），默认 20.0"
    )
    parser.add_argument(
        "--stop-loss-trailing-enabled",
        action="store_true",
        default=False,
        help="启用移动止损"
    )
    parser.add_argument(
        "--stop-loss-trailing-pct",
        type=float,
        default=15.0,
        help="移动止损阈值（百分比），默认 15.0"
    )
    parser.add_argument(
        "--stop-loss-consecutive-limit-down",
        type=int,
        default=2,
        help="连续跌停止损天数，默认 2"
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
    logger.info(f"止损功能: {'启用' if args.stop_loss_enabled else '禁用'}")
    if args.stop_loss_enabled:
        logger.info(f"  - 回撤止损: {args.stop_loss_drawdown_pct}%")
        logger.info(f"  - 移动止损: {'启用' if args.stop_loss_trailing_enabled else '禁用'}")
        if args.stop_loss_trailing_enabled:
            logger.info(f"  - 移动止损阈值: {args.stop_loss_trailing_pct}%")
        logger.info(f"  - 连续跌停止损: {args.stop_loss_consecutive_limit_down} 天")
    
    try:
        # 初始化组件
        storage = Storage(root_path=args.data_root)
        loader = DataLoader(storage)
        
        # 创建止损配置
        stop_loss_config = None
        if args.stop_loss_enabled:
            stop_loss_config = StopLossConfig(
                enabled=True,
                drawdown_pct=args.stop_loss_drawdown_pct,
                trailing_stop_enabled=args.stop_loss_trailing_enabled,
                trailing_stop_pct=args.stop_loss_trailing_pct,
                consecutive_limit_down_days=args.stop_loss_consecutive_limit_down,
                post_trigger_action='hold_cash'
            )
        
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
            exclude_st=args.exclude_st,
            min_list_days=args.min_list_days,
            markets=['主板'],  # 可根据需要调整
            verbose=False,
        )
        
        # 4. 创建 ML 信号
        signal = MLSignal(
            top_n=args.top_n,
            model_version=args.model_version,
            models_dir=f"{args.data_root}/models",
            weight_method=args.weight_method,
            verbose=False,
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
            rebalance_freq=args.rebalance_freq,
            stop_loss_config=stop_loss_config,
            sell_timing=args.sell_timing,
        )
        
        # 7. 生成报告
        reporter = Reporter(output_dir=f"{args.data_root}/reports")
        stats = reporter.generate_report(nav_curve, trades, output_name=args.output_name)
        
        logger.info("=" * 60)
        logger.info("回测完成！")
        logger.info(f"报告已保存到: {args.data_root}/reports/")
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
                "weight_method": args.weight_method,
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
                "run_time", "start_date", "end_date", "model_version", "top_n", "weight_method",
                "rebalance_freq", "initial_capital", "sell_timing", "stop_loss_enabled",
                "report_name", "nav_final", "total_return", "max_drawdown", "sharpe"
            ]

            _append_dict_to_csv(log_file, record, fieldnames=fieldnames)
            logger.info(f"本次回测记录已追加到: {log_file}")
        except Exception as ex:
            # 记录追加失败不影响回测结果输出，但记录错误信息
            logger.exception(f"写回测记录到 CSV 失败: {ex}")
        # ---------------------------------------------------------------------------

    except Exception as e:
        logger.error(f"回测失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
