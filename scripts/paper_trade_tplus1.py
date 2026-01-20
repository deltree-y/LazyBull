#!/usr/bin/env python3
"""T+1 纸面交易脚本

本脚本提供两个子命令用于本地纸面交易验证：

1. generate: 在指定交易日生成信号并保存（不下单）
   示例: python scripts/paper_trade_tplus1.py generate --trade-date 20230601 --top-n 10

2. execute: 在指定执行日读取待执行信号，按执行日收盘价进行买卖
   示例: python scripts/paper_trade_tplus1.py execute --exec-date 20230602

使用流程：
  1. 运行 generate 生成T日信号
  2. 次日运行 execute 执行T+1日交易
  3. 查看 data/trading_state.json 中的订单和持仓状态

数据要求：
  - 优先从 data/features/cs_train/{YYYYMMDD}.parquet 读取
  - 若不存在则从 data/clean/daily/{YYYY-MM-DD}.parquet 读取
  - 需包含 ts_code, close 等基础字段

注意事项：
  - 本脚本仅用于本地纸面验证，不涉及真实券商
  - 复权口径：使用不复权价格（close）
  - T+1逻辑：T日信号在T+1日以T+1收盘价执行
  - 等权分配：可用现金等权分配到各个股票
  - 买入数量：向下取整为整数股数
  - 默认配置：初始资金50万，佣金万3，滑点0.1%
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.live import SimplePersistence, MockBroker


def load_data_for_date(trade_date: str) -> pd.DataFrame:
    """加载指定日期的数据
    
    优先从 features/cs_train 加载，若不存在则从 clean/daily 加载
    
    Args:
        trade_date: 交易日期，YYYYMMDD格式
    
    Returns:
        包含当日数据的DataFrame
    
    Raises:
        FileNotFoundError: 如果两个路径都不存在
    """
    # 尝试从 features 加载
    features_path = Path(f"data/features/cs_train/{trade_date}.parquet")
    if features_path.exists():
        logger.info(f"从 features 加载数据: {features_path}")
        df = pd.read_parquet(features_path)
        return df
    
    # 尝试从 clean/daily 加载
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    clean_path = Path(f"data/clean/daily/{date_str}.parquet")
    if clean_path.exists():
        logger.info(f"从 clean/daily 加载数据: {clean_path}")
        df = pd.read_parquet(clean_path)
        return df
    
    raise FileNotFoundError(
        f"未找到日期 {trade_date} 的数据，尝试了:\n"
        f"  - {features_path}\n"
        f"  - {clean_path}"
    )


def generate_simple_score(df: pd.DataFrame) -> pd.DataFrame:
    """生成简化的评分（示例：按当日价格排名）
    
    Args:
        df: 原始数据
    
    Returns:
        添加了 score 列的DataFrame
    """
    # 简单示例：按收盘价降序排序作为score
    # 实际应用中，这里应该使用特征和模型预测
    df = df.copy()
    if 'close' in df.columns:
        # 价格越高score越高（仅示例，实际策略可能相反）
        df['score'] = df['close'].rank(ascending=False)
    else:
        logger.warning("未找到 close 列，使用随机score")
        df['score'] = pd.Series(range(len(df)), index=df.index)
    
    return df


def cmd_generate(args):
    """生成信号命令
    
    Args:
        args: 命令行参数
    """
    logger.info(f"=== 开始生成信号：日期={args.trade_date}, top_n={args.top_n} ===")
    
    # 加载数据
    try:
        df = load_data_for_date(args.trade_date)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # 检查必要字段
    required_cols = ['ts_code', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"数据缺少必要字段: {missing}")
        return 1
    
    # 如果没有score列，生成简化score
    if 'score' not in df.columns:
        logger.info("未找到 score 列，生成简化评分")
        df = generate_simple_score(df)
    
    # 过滤条件（如有）
    # 过滤ST股票
    if 'is_st' in df.columns:
        before_count = len(df)
        df = df[df['is_st'] == 0]
        logger.info(f"过滤ST股票: {before_count} -> {len(df)}")
    
    # 过滤停牌股票
    if 'is_suspended' in df.columns:
        before_count = len(df)
        df = df[df['is_suspended'] == 0]
        logger.info(f"过滤停牌股票: {before_count} -> {len(df)}")
    
    # 按score排序并选择top N
    df = df.sort_values('score', ascending=False)
    top_stocks = df.head(args.top_n)
    
    if len(top_stocks) == 0:
        logger.error("没有符合条件的股票")
        return 1
    
    # 等权分配
    weight = 1.0 / len(top_stocks)
    
    # 构建信号列表
    signals = []
    for _, row in top_stocks.iterrows():
        signal = {
            "symbol": row['ts_code'],
            "weight": weight,
            "signal_meta": {
                "score": float(row.get('score', 0)),
                "close": float(row['close'])
            }
        }
        signals.append(signal)
    
    # 保存到持久化
    persistence = SimplePersistence(args.state_file)
    persistence.add_pending_signals(args.trade_date, signals)
    
    logger.info(f"=== 信号生成完成，共 {len(signals)} 只股票 ===")
    for sig in signals:
        logger.info(f"  {sig['symbol']}: weight={sig['weight']:.4f}, "
                   f"score={sig['signal_meta']['score']:.2f}, "
                   f"close={sig['signal_meta']['close']:.2f}")
    
    return 0


def cmd_execute(args):
    """执行交易命令
    
    Args:
        args: 命令行参数
    """
    logger.info(f"=== 开始执行交易：日期={args.exec_date} ===")
    
    # 初始化持久化和券商
    persistence = SimplePersistence(args.state_file)
    broker = MockBroker(
        persistence=persistence,
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage=args.slippage
    )
    
    # 查找待执行信号
    pending_list = persistence.get_pending_signals(executed=False)
    if not pending_list:
        logger.warning("没有待执行的信号")
        return 1
    
    # 找到最早的未执行信号
    # 理论上应该按照trade_date匹配，这里简化为取第一个
    if len(pending_list) > 1:
        logger.warning(f"有 {len(pending_list)} 个待执行信号，将执行最早的一个")
    
    pending = pending_list[0]
    trade_date = pending['trade_date']
    signals = pending['signals']
    
    logger.info(f"执行信号：trade_date={trade_date}, 信号数={len(signals)}")
    
    # 加载执行日数据（用于获取收盘价）
    try:
        exec_df = load_data_for_date(args.exec_date)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # 构建价格字典
    price_dict = {}
    for _, row in exec_df.iterrows():
        price_dict[row['ts_code']] = row['close']
    
    # 获取当前持仓
    current_positions = broker.positions.copy()
    target_symbols = set(sig['symbol'] for sig in signals)
    
    # 1. 卖出不在目标中的持仓
    for symbol in list(current_positions.keys()):
        if symbol not in target_symbols:
            qty = current_positions[symbol]['qty']
            if symbol not in price_dict:
                logger.warning(f"无法获取 {symbol} 的价格，跳过卖出")
                continue
            
            price = price_dict[symbol]
            logger.info(f"卖出 {symbol}: {qty}股 @{price:.2f}")
            order = broker.place_order(symbol, 'sell', qty, price, args.exec_date)
            
            if order['status'] == 'rejected':
                logger.error(f"卖出失败: {order['reason']}")
    
    # 2. 买入目标股票
    available_cash = broker.cash
    logger.info(f"可用现金: {available_cash:.2f}")
    
    for signal in signals:
        symbol = signal['symbol']
        weight = signal['weight']
        
        if symbol not in price_dict:
            logger.warning(f"无法获取 {symbol} 的价格，跳过买入")
            continue
        
        price = price_dict[symbol]
        target_value = available_cash * weight
        
        # 计算买入数量（向下取整）
        # 考虑佣金和滑点的粗略估算
        cost_factor = 1 + args.commission_rate + args.slippage
        qty = int(target_value / (price * cost_factor))
        
        if qty <= 0:
            logger.warning(f"{symbol}: 资金不足，无法买入")
            continue
        
        logger.info(f"买入 {symbol}: {qty}股 @{price:.2f}, 目标金额: {target_value:.2f}")
        order = broker.place_order(symbol, 'buy', qty, price, args.exec_date)
        
        if order['status'] == 'rejected':
            logger.error(f"买入失败: {order['reason']}")
    
    # 标记信号为已执行
    persistence.pop_pending_signals(trade_date)
    
    # 显示最终账户状态
    account_info = broker.get_account_info()
    logger.info("=== 执行完成，账户状态 ===")
    logger.info(f"现金: {account_info['cash']:.2f}")
    logger.info(f"账户总值（成本价估算）: {account_info['total_value']:.2f}")
    logger.info(f"持仓数: {len(account_info['positions'])}")
    for symbol, pos in account_info['positions'].items():
        logger.info(f"  {symbol}: {pos['qty']}股, 成本价: {pos['cost_price']:.2f}")
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="T+1 纸面交易脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成T日信号
  python scripts/paper_trade_tplus1.py generate --trade-date 20230601 --top-n 10
  
  # T+1日执行交易
  python scripts/paper_trade_tplus1.py execute --exec-date 20230602
  
  # 使用自定义参数
  python scripts/paper_trade_tplus1.py execute --exec-date 20230602 \\
    --initial-cash 1000000 --commission-rate 0.0005 --slippage 0.002
        """
    )
    
    # 公共参数
    parser.add_argument(
        '--state-file',
        type=str,
        default='data/trading_state.json',
        help='持久化状态文件路径（默认: data/trading_state.json）'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # generate 子命令
    parser_gen = subparsers.add_parser('generate', help='生成信号')
    parser_gen.add_argument(
        '--trade-date',
        type=str,
        required=True,
        help='交易日期（YYYYMMDD格式）'
    )
    parser_gen.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='选择前N只股票（默认: 10）'
    )
    
    # execute 子命令
    parser_exec = subparsers.add_parser('execute', help='执行交易')
    parser_exec.add_argument(
        '--exec-date',
        type=str,
        required=True,
        help='执行日期（YYYYMMDD格式）'
    )
    parser_exec.add_argument(
        '--initial-cash',
        type=float,
        default=500000.0,
        help='初始资金（默认: 500000）'
    )
    parser_exec.add_argument(
        '--commission-rate',
        type=float,
        default=0.0003,
        help='佣金率（默认: 0.0003，即万3）'
    )
    parser_exec.add_argument(
        '--slippage',
        type=float,
        default=0.001,
        help='滑点率（默认: 0.001，即0.1%%）'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 执行命令
    try:
        if args.command == 'generate':
            return cmd_generate(args)
        elif args.command == 'execute':
            return cmd_execute(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.exception(f"执行失败: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
