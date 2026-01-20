#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T+1 纸面交易脚本

功能：
1. generate 子命令：在指定日期（T）生成信号并保存到 pending_signals，不执行下单
2. execute 子命令：在执行日（T+1）读取 pending_signals，按次日收盘价执行买入

使用示例：
# 1. 生成当天信号（T），保存到 pending_signals
python scripts/paper_trade_tplus1.py generate --trade-date 20240115 --top-n 5

# 2. 次日（T+1）执行保存的信号，使用次日收盘价
python scripts/paper_trade_tplus1.py execute --exec-date 20240116

# 3. 查看账户状态
python scripts/paper_trade_tplus1.py status

# 4. 重置账户（清空所有数据）
python scripts/paper_trade_tplus1.py reset --initial-cash 500000

数据要求：
- 需要 data/features/cs_train/{YYYYMMDD}.parquet 或 data/clean/daily/{YYYY-MM-DD}.parquet
- 若不存在，请先运行 scripts/build_features.py 构建数据

注意：
- 本脚本为最简化实现，仅用于本地纸面验证
- 生产环境请替换为真实券商接口和数据库持久化
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage
from src.lazybull.live.mock_broker import MockBroker
from src.lazybull.live.persistence import SimplePersistence


def load_daily_data(trade_date: str, storage: Storage) -> pd.DataFrame:
    """加载指定日期的日线数据
    
    优先从 features 加载，其次从 clean 加载
    
    Args:
        trade_date: 交易日期 YYYYMMDD
        storage: Storage 实例
        
    Returns:
        日线数据 DataFrame
        
    Raises:
        FileNotFoundError: 数据文件不存在
    """
    # 尝试从 features 加载
    features_path = storage.features_path / "cs_train" / f"{trade_date}.parquet"
    if features_path.exists():
        logger.info(f"从 features 加载数据: {features_path}")
        return pd.read_parquet(features_path)
    
    # 尝试从 clean 加载
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    clean_path = storage.clean_path / "daily" / f"{date_str}.parquet"
    if clean_path.exists():
        logger.info(f"从 clean 加载数据: {clean_path}")
        return pd.read_parquet(clean_path)
    
    # 都不存在
    raise FileNotFoundError(
        f"未找到日期 {trade_date} 的数据文件。\n"
        f"请先运行 scripts/build_features.py 构建数据。\n"
        f"尝试的路径:\n  - {features_path}\n  - {clean_path}"
    )


def generate_signals(
    trade_date: str,
    top_n: int,
    persistence: SimplePersistence,
    storage: Storage
) -> None:
    """生成信号并保存到 pending_signals
    
    Args:
        trade_date: 交易日期 YYYYMMDD
        top_n: 选择 top N 只股票
        persistence: 持久化实例
        storage: Storage 实例
    """
    logger.info(f"开始生成信号: {trade_date}, top_n={top_n}")
    
    # 加载数据
    try:
        df = load_daily_data(trade_date, storage)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # 过滤可交易股票
    # 1. 过滤 ST 股票
    if 'is_st' in df.columns:
        df = df[df['is_st'] == False].copy()
        logger.info(f"过滤 ST 后剩余 {len(df)} 只股票")
    
    # 2. 过滤停牌股票
    if 'is_suspended' in df.columns:
        df = df[df['is_suspended'] == False].copy()
        logger.info(f"过滤停牌后剩余 {len(df)} 只股票")
    
    # 3. 过滤上市天数不足的股票（可选）
    if 'list_days' in df.columns:
        df = df[df['list_days'] >= 60].copy()
        logger.info(f"过滤上市不足60天后剩余 {len(df)} 只股票")
    
    # 4. 检查是否有足够的股票
    if len(df) < top_n:
        logger.warning(f"可用股票数 {len(df)} 少于 top_n {top_n}，将选择所有可用股票")
        top_n = len(df)
    
    # 简单策略：按市值排序，选择前 N 只（这里可以替换为任何因子排序逻辑）
    # 如果有因子列（如 factor_score），可以按因子排序
    if 'total_mv' in df.columns:
        df = df.sort_values('total_mv', ascending=False)
        logger.info("按总市值排序选股")
    else:
        logger.warning("未找到 total_mv 列，使用原始顺序")
    
    # 选择 top N
    selected = df.head(top_n)
    
    if len(selected) == 0:
        logger.error("未选择到任何股票")
        return
    
    # 生成等权信号
    signals = {}
    weight = 1.0 / len(selected)
    for _, row in selected.iterrows():
        code = row['ts_code'] if 'ts_code' in row else row.name
        signals[code] = weight
    
    logger.info(f"生成信号: {len(signals)} 只股票")
    for code, w in list(signals.items())[:5]:  # 只显示前 5 只
        logger.info(f"  {code}: {w:.4f}")
    if len(signals) > 5:
        logger.info(f"  ... 还有 {len(signals) - 5} 只")
    
    # 计算执行日期（T+1）
    # 注意：这里简化处理，实际生产环境应该查询交易日历获取下一个交易日
    # 当前实现在跨月/跨年时会产生无效日期（如 20241231 + 1 = 20241232）
    # 建议在生产环境中使用 DataLoader 加载交易日历并调用 get_next_trade_date()
    exec_date = str(int(trade_date) + 1)  # 简化实现
    
    # 保存到 pending_signals
    persistence.add_pending_signal(
        trade_date=trade_date,
        exec_date=exec_date,
        signals=signals,
        top_n=top_n
    )
    
    logger.info(f"信号已保存，预期执行日期: {exec_date}")


def execute_signals(
    exec_date: str,
    persistence: SimplePersistence,
    storage: Storage,
    broker: MockBroker
) -> None:
    """执行 pending_signals
    
    Args:
        exec_date: 执行日期 YYYYMMDD
        persistence: 持久化实例
        storage: Storage 实例
        broker: MockBroker 实例
    """
    logger.info(f"开始执行信号: {exec_date}")
    
    # 获取待执行信号
    pending = persistence.get_pending_signals(exec_date=exec_date, executed=False)
    
    if not pending:
        logger.warning(f"没有找到执行日期为 {exec_date} 的未执行信号")
        return
    
    logger.info(f"找到 {len(pending)} 条待执行信号")
    
    # 加载执行日的行情数据
    try:
        df = load_daily_data(exec_date, storage)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # 转换为 {code: price} 字典，使用收盘价
    price_dict = {}
    if 'ts_code' in df.columns and 'close' in df.columns:
        for _, row in df.iterrows():
            price_dict[row['ts_code']] = row['close']
    else:
        logger.error("数据中未找到 ts_code 或 close 列")
        return
    
    # 获取账户信息
    account = persistence.get_account()
    cash = account["cash"]
    logger.info(f"当前账户现金: {cash:.2f}")
    
    # 执行每个信号
    for signal_item in pending:
        trade_date = signal_item["trade_date"]
        signals = signal_item["signals"]
        
        logger.info(f"执行信号: {trade_date} -> {exec_date}, {len(signals)} 只股票")
        
        # 按权重分配资金
        # 注意：这里使用初始现金计算目标仓位，即按照信号生成时的权重等比例分配
        # 这是等权策略的标准做法，确保每只股票的目标仓位相同
        # 实际买入时会因为资金不足、取整等原因导致实际仓位略有偏差
        for code, weight in signals.items():
            if code not in price_dict:
                logger.warning(f"股票 {code} 在 {exec_date} 无行情数据，跳过")
                continue
            
            price = price_dict[code]
            
            # 计算买入股数
            target_amount = cash * weight  # 目标买入金额
            shares = int(target_amount / price / 100) * 100  # 按手（100股）向下取整
            
            if shares < 100:
                logger.warning(f"股票 {code} 资金不足一手，跳过")
                continue
            
            # 下单
            result = broker.place_order(
                code=code,
                direction="buy",
                shares=shares,
                price=price
            )
            
            if result.is_success():
                logger.info(
                    f"✓ 买入成功: {code}, 股数={shares}, 价格={price:.2f}, "
                    f"金额={result.amount:.2f}, 成本={result.cost:.2f}"
                )
            else:
                logger.error(f"✗ 买入失败: {code}, 原因={result.message}")
        
        # 标记信号为已执行
        persistence.mark_signal_executed(trade_date)
    
    # 显示执行后的账户状态
    show_status(broker)


def show_status(broker: MockBroker) -> None:
    """显示账户状态
    
    Args:
        broker: MockBroker 实例
    """
    info = broker.get_account_info()
    
    print("\n" + "="*60)
    print("账户状态")
    print("="*60)
    print(f"初始资金: {info['initial_cash']:,.2f}")
    print(f"当前现金: {info['cash']:,.2f}")
    print(f"持仓市值: {info['position_value']:,.2f}")
    print(f"总资产:   {info['total_value']:,.2f}")
    
    if info['initial_cash'] > 0:
        pnl = info['total_value'] - info['initial_cash']
        pnl_pct = (pnl / info['initial_cash']) * 100
        print(f"盈亏:     {pnl:,.2f} ({pnl_pct:+.2f}%)")
    
    print("\n持仓详情:")
    print("-"*60)
    if info['positions']:
        print(f"{'股票代码':<12} {'股数':>8} {'成本':>10} {'现价':>10} {'市值':>12}")
        print("-"*60)
        for code, pos in info['positions'].items():
            market_value = pos['shares'] * pos['last_price']
            print(
                f"{code:<12} {pos['shares']:>8} {pos['avg_cost']:>10.2f} "
                f"{pos['last_price']:>10.2f} {market_value:>12.2f}"
            )
    else:
        print("无持仓")
    print("="*60 + "\n")


def reset_account(persistence: SimplePersistence, initial_cash: float) -> None:
    """重置账户
    
    Args:
        persistence: 持久化实例
        initial_cash: 初始资金
    """
    logger.warning("即将重置所有数据，包括账户、持仓、订单和信号")
    
    # 重置
    persistence.reset()
    persistence.init_account(initial_cash)
    
    logger.info(f"账户已重置，初始资金: {initial_cash:.2f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="T+1 纸面交易脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成信号
  python scripts/paper_trade_tplus1.py generate --trade-date 20240115 --top-n 5
  
  # 执行信号
  python scripts/paper_trade_tplus1.py execute --exec-date 20240116
  
  # 查看状态
  python scripts/paper_trade_tplus1.py status
  
  # 重置账户
  python scripts/paper_trade_tplus1.py reset --initial-cash 500000
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # generate 子命令
    parser_gen = subparsers.add_parser("generate", help="生成信号")
    parser_gen.add_argument(
        "--trade-date",
        required=True,
        help="交易日期 YYYYMMDD"
    )
    parser_gen.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="选择 top N 只股票（默认 5）"
    )
    
    # execute 子命令
    parser_exec = subparsers.add_parser("execute", help="执行信号")
    parser_exec.add_argument(
        "--exec-date",
        required=True,
        help="执行日期 YYYYMMDD"
    )
    
    # status 子命令
    parser_status = subparsers.add_parser("status", help="查看账户状态")
    
    # reset 子命令
    parser_reset = subparsers.add_parser("reset", help="重置账户")
    parser_reset.add_argument(
        "--initial-cash",
        type=float,
        default=500000.0,
        help="初始资金（默认 500000）"
    )
    
    # 通用参数
    parser.add_argument(
        "--data-path",
        default="./data",
        help="数据目录路径（默认 ./data）"
    )
    parser.add_argument(
        "--state-file",
        default="data/trading_state.json",
        help="状态文件路径（默认 data/trading_state.json）"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    
    # 检查子命令
    if not args.command:
        parser.print_help()
        return
    
    # 初始化组件
    storage = Storage(root_path=args.data_path)
    persistence = SimplePersistence(file_path=args.state_file)
    broker = MockBroker(persistence=persistence)
    
    # 执行命令
    if args.command == "generate":
        generate_signals(
            trade_date=args.trade_date,
            top_n=args.top_n,
            persistence=persistence,
            storage=storage
        )
    
    elif args.command == "execute":
        # 检查账户是否初始化
        account = persistence.get_account()
        if account["initial_cash"] == 0:
            logger.error("账户未初始化，请先运行 reset 命令初始化账户")
            return
        
        execute_signals(
            exec_date=args.exec_date,
            persistence=persistence,
            storage=storage,
            broker=broker
        )
    
    elif args.command == "status":
        show_status(broker)
    
    elif args.command == "reset":
        reset_account(persistence, args.initial_cash)


if __name__ == "__main__":
    main()
