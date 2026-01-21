#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纸面交易脚本 - 重构版

功能：
- config 子命令：设置全局配置（持久化）
- run 子命令：每日运行入口，自动编排执行各项动作
- positions 子命令：查看持仓明细

示例：
  python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5 --initial-capital 500000 --rebalance-freq 5 --weight-method equal
  python scripts/paper_trade.py run --trade-date 20260121
  python scripts/paper_trade.py positions --trade-date 20260122
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.common.print_table import format_row
from src.lazybull.data import DataLoader, Storage
from src.lazybull.paper import PaperTradingRunner, PaperStorage
from src.lazybull.risk.stop_loss import StopLossConfig, StopLossMonitor


def run_config(args):
    """配置命令：持久化全局配置"""
    logger.info("=" * 80)
    logger.info("纸面交易配置设置")
    logger.info("=" * 80)
    
    # 构建配置字典
    config = {
        'buy_price': args.buy_price,
        'sell_price': args.sell_price,
        'top_n': args.top_n,
        'initial_capital': args.initial_capital,
        'rebalance_freq': args.rebalance_freq,
        'weight_method': args.weight_method,
        'model_version': args.model_version,
        'stop_loss_enabled': args.stop_loss_enabled,
        'stop_loss_drawdown_pct': args.stop_loss_drawdown_pct,
        'stop_loss_trailing_enabled': args.stop_loss_trailing_enabled,
        'stop_loss_trailing_pct': args.stop_loss_trailing_pct,
        'stop_loss_consecutive_limit_down': args.stop_loss_consecutive_limit_down,
        'universe': args.universe
    }
    
    # 保存配置
    storage = PaperStorage()
    storage.save_config(config)
    
    logger.info("配置已保存成功！")
    logger.info("")
    logger.info("当前配置：")
    logger.info("-" * 80)
    
    # 格式化输出
    widths = [30, 50]
    aligns = ['left', 'left']
    
    for key, value in config.items():
        row = [key, str(value)]
        logger.info(format_row(row, widths, aligns))
    
    logger.info("=" * 80)


def run_main(args):
    """运行命令：自动编排执行各项动作"""
    logger.info("=" * 80)
    logger.info("纸面交易自动运行")
    logger.info("=" * 80)
    logger.info(f"交易日期: {args.trade_date}")
    
    # 1. 读取配置
    storage = PaperStorage()
    config = storage.load_config()
    
    if config is None:
        logger.error("未找到配置文件，请先运行 config 命令设置配置")
        logger.error("示例: python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5")
        sys.exit(1)
    
    # 允许命令行参数覆盖配置
    if args.model_version is not None:
        config['model_version'] = args.model_version
    if args.weight_method is not None:
        config['weight_method'] = args.weight_method
    
    logger.info("使用配置：")
    logger.info(f"  买入价格类型: {config['buy_price']}")
    logger.info(f"  卖出价格类型: {config['sell_price']}")
    logger.info(f"  持仓数: {config['top_n']}")
    logger.info(f"  调仓频率: {config['rebalance_freq']} 个交易日")
    logger.info(f"  权重方法: {config['weight_method']}")
    logger.info(f"  止损开关: {config['stop_loss_enabled']}")
    logger.info("=" * 80)
    
    # 2. 创建运行器
    runner = PaperTradingRunner(
        initial_capital=config['initial_capital'],
        weight_method=config['weight_method']
    )
    
    # 3. 校正交易日期
    corrected_date = runner._correct_trade_date(args.trade_date)
    
    # 4. 创建止损监控器
    stop_loss_config = StopLossConfig(
        enabled=config['stop_loss_enabled'],
        drawdown_pct=config['stop_loss_drawdown_pct'],
        trailing_stop_enabled=config['stop_loss_trailing_enabled'],
        trailing_stop_pct=config['stop_loss_trailing_pct'],
        consecutive_limit_down_days=config['stop_loss_consecutive_limit_down']
    )
    stop_loss_monitor = StopLossMonitor(stop_loss_config)
    
    # 加载止损状态
    sl_state = storage.load_stop_loss_state()
    if sl_state:
        stop_loss_monitor.position_high_prices = sl_state.get('position_high_prices', {})
        stop_loss_monitor.consecutive_limit_down_days = sl_state.get('consecutive_limit_down_days', {})
    
    # 收集所有待手工操作的信息
    stop_loss_actions = []
    pending_sell_actions = []
    t1_actions = []
    t0_targets = []
    
    # 5. 执行止损检查
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤1: 检查止损触发")
    logger.info("-" * 80)
    
    if config['stop_loss_enabled']:
        stop_loss_actions = _check_stop_loss(
            runner, stop_loss_monitor, corrected_date, config
        )
        
        # 保存止损状态
        sl_state = {
            'position_high_prices': stop_loss_monitor.position_high_prices,
            'consecutive_limit_down_days': stop_loss_monitor.consecutive_limit_down_days
        }
        storage.save_stop_loss_state(sl_state)
    else:
        logger.info("止损功能未启用，跳过")
    
    # 6. 执行延迟卖出队列
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤2: 处理延迟卖出队列")
    logger.info("-" * 80)
    
    pending_sell_actions = _process_pending_sells(runner, corrected_date, config)
    
    # 7. 执行 T1（如果有待执行目标）
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤3: 检查并执行 T1")
    logger.info("-" * 80)
    
    t1_actions = _execute_t1_if_pending(runner, corrected_date, config)
    
    # 8. 判断是否调仓日并执行 T0
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤4: 检查是否调仓日并执行 T0")
    logger.info("-" * 80)
    
    t0_targets = _execute_t0_if_rebalance_day(runner, corrected_date, config)
    
    # 9. 打印手工操作指令汇总
    logger.info("")
    logger.info("=" * 120)
    logger.info("手工操作指令汇总")
    logger.info("=" * 120)
    
    _print_manual_actions(stop_loss_actions, pending_sell_actions, t1_actions, t0_targets)
    
    logger.info("=" * 120)
    logger.info(f"运行完成 - {corrected_date}")
    logger.info("=" * 120)


def _check_stop_loss(
    runner: PaperTradingRunner,
    stop_loss_monitor: StopLossMonitor,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """检查止损触发
    
    Returns:
        止损动作列表 [{ts_code, shares, reason}, ...]
    """
    actions = []
    
    # 获取当前持仓
    positions = runner.account.get_positions()
    
    if not positions:
        logger.info("当前无持仓，跳过止损检查")
        return actions
    
    # 加载价格数据
    loader = DataLoader(runner.storage)
    daily_data = loader.load_clean_daily_by_date(trade_date)
    
    if daily_data is None or daily_data.empty:
        logger.warning(f"无法加载 {trade_date} 的价格数据，跳过止损检查")
        return actions
    
    # 构建价格字典和跌停信息
    prices = {}
    limit_down_info = {}
    for _, row in daily_data.iterrows():
        ts_code = row['ts_code']
        prices[ts_code] = row.get('close', 0.0)
        limit_down_info[ts_code] = row.get('is_limit_down', 0) == 1
    
    # 检查每个持仓
    for ts_code, pos in positions.items():
        if ts_code not in prices:
            logger.warning(f"股票 {ts_code} 无价格数据，跳过")
            continue
        
        current_price = prices[ts_code]
        is_limit_down = limit_down_info.get(ts_code, False)
        
        # 检查是否触发止损
        triggered, trigger_type, reason = stop_loss_monitor.check_stop_loss(
            ts_code,
            pos.buy_price,
            current_price,
            is_limit_down
        )
        
        if triggered:
            # 计算建议卖出股数（按100股规则）
            sell_shares = (pos.shares // 100) * 100
            
            actions.append({
                'ts_code': ts_code,
                'shares': sell_shares,
                'reason': reason,
                'can_execute': not is_limit_down
            })
            
            # 如果不可卖出，加入延迟卖出队列
            if is_limit_down:
                from src.lazybull.paper.models import PendingSell
                pending_sell = PendingSell(
                    ts_code=ts_code,
                    shares=sell_shares,
                    target_weight=0.0,
                    reason=f"止损-{reason}",
                    create_date=trade_date,
                    attempts=0
                )
                runner.broker.pending_sells.append(pending_sell)
                runner.broker.storage.save_pending_sells(runner.broker.pending_sells)
    
    logger.info(f"止损检查完成：触发 {len(actions)} 个止损信号")
    return actions


def _process_pending_sells(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """处理延迟卖出队列
    
    Returns:
        延迟卖出动作列表 [{ts_code, shares, reason, status}, ...]
    """
    actions = []
    
    # 重试延迟卖出
    fills = runner.broker.retry_pending_sells(trade_date, config['sell_price'])
    
    # 收集仍在队列中的订单
    for ps in runner.broker.pending_sells:
        actions.append({
            'ts_code': ps.ts_code,
            'shares': ps.shares,
            'reason': ps.reason,
            'status': f'不可卖出（尝试次数: {ps.attempts}）'
        })
    
    # 收集已成交的订单
    for fill in fills:
        actions.append({
            'ts_code': fill.ts_code,
            'shares': fill.shares,
            'reason': fill.reason,
            'status': '已成交'
        })
    
    if fills:
        # 更新账户状态和净值
        runner.account.update_last_date(trade_date)
        runner.account.save_state()
        
        # 加载价格
        buy_prices, sell_prices = runner._load_prices(trade_date, config['buy_price'], config['sell_price'])
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)
    
    logger.info(f"延迟卖出处理完成：成交 {len(fills)} 笔，剩余 {len(runner.broker.pending_sells)} 笔")
    return actions


def _execute_t1_if_pending(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """执行 T1（如果有待执行目标）
    
    Returns:
        T1 动作列表 [{ts_code, action, shares, reason}, ...]
    """
    actions = []
    
    # 检查幂等性
    if runner.paper_storage.check_run_exists("t1", trade_date):
        logger.info(f"T1 工作流已在 {trade_date} 执行过，跳过")
        return actions
    
    # 检查是否有待执行目标
    targets = runner.paper_storage.load_pending_weights(trade_date)
    
    if not targets:
        logger.info(f"未找到 {trade_date} 的待执行目标，跳过 T1")
        return actions
    
    logger.info(f"找到 {len(targets)} 个待执行目标，执行 T1")
    
    # 加载价格数据
    buy_prices, sell_prices = runner._load_prices(trade_date, config['buy_price'], config['sell_price'])
    
    if not buy_prices and not sell_prices:
        logger.error("无法加载价格数据，跳过 T1")
        return actions
    
    # 生成订单
    orders = runner.broker.generate_orders(targets, buy_prices, sell_prices, trade_date)
    
    if orders:
        # 执行订单
        fills = runner.broker.execute_orders(
            orders,
            trade_date,
            config['buy_price'],
            config['sell_price']
        )
        
        # 收集动作
        for fill in fills:
            actions.append({
                'ts_code': fill.ts_code,
                'action': fill.action,
                'shares': fill.shares,
                'reason': fill.reason
            })
        
        # 更新账户状态
        runner.account.update_last_date(trade_date)
        runner.account.save_state()
        
        # 记录净值
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)
        
        # 保存执行记录
        run_record = {
            'trade_date': trade_date,
            'buy_price_type': config['buy_price'],
            'sell_price_type': config['sell_price'],
            'targets_count': len(targets),
            'orders_count': len(orders),
            'fills_count': len(fills),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        runner.paper_storage.save_run_record("t1", trade_date, run_record)
    
    logger.info(f"T1 执行完成：{len(actions)} 个订单")
    return actions


def _execute_t0_if_rebalance_day(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """执行 T0（如果是调仓日）
    
    Returns:
        T0 目标列表 [{ts_code, target_weight, reason, score}, ...]
    """
    targets_info = []
    
    # 检查幂等性
    if runner.paper_storage.check_run_exists("t0", trade_date):
        logger.info(f"T0 工作流已在 {trade_date} 执行过，跳过")
        return targets_info
    
    # 检查是否调仓日
    try:
        is_rebalance_day = runner._check_rebalance_day(trade_date, config['rebalance_freq'])
    except RuntimeError as e:
        logger.info(f"当前不是调仓日：{e}")
        logger.info("非调仓日允许执行卖出和T1，T0跳过")
        return targets_info
    
    if not is_rebalance_day:
        logger.info("非调仓日，跳过 T0")
        return targets_info
    
    logger.info("当前是调仓日，执行 T0")
    
    # 执行 T0
    try:
        runner.run_t0(
            trade_date=trade_date,
            buy_price_type=config['buy_price'],
            universe_type=config['universe'],
            top_n=config['top_n'],
            model_version=config.get('model_version'),
            rebalance_freq=config['rebalance_freq']
        )
        
        # 获取下一交易日
        t1_date = runner._get_next_trade_date(trade_date)
        if t1_date:
            # 读取生成的目标
            targets = runner.paper_storage.load_pending_weights(t1_date)
            if targets:
                for target in targets:
                    targets_info.append({
                        'ts_code': target.ts_code,
                        'target_weight': target.target_weight,
                        'reason': target.reason,
                        'score': None  # 如果信号包含score可以在这里添加
                    })
        
        logger.info(f"T0 执行完成：生成 {len(targets_info)} 个目标")
    except Exception as e:
        logger.error(f"T0 执行失败: {e}")
    
    return targets_info


def _print_manual_actions(
    stop_loss_actions: List[Dict],
    pending_sell_actions: List[Dict],
    t1_actions: List[Dict],
    t0_targets: List[Dict]
):
    """打印手工操作指令汇总"""
    
    # 1. 止损卖出清单
    if stop_loss_actions:
        logger.info("")
        logger.info("【止损卖出清单】")
        logger.info("-" * 120)
        
        widths = [15, 10, 15, 60]
        aligns = ['left', 'right', 'left', 'left']
        header = ["股票代码", "建议股数", "是否可执行", "原因"]
        logger.info(format_row(header, widths, aligns))
        logger.info("-" * 120)
        
        for action in stop_loss_actions:
            row = [
                action['ts_code'],
                str(action['shares']),
                "是" if action['can_execute'] else "否(跌停)",
                action['reason']
            ]
            logger.info(format_row(row, widths, aligns))
    
    # 2. 延迟卖出清单
    if pending_sell_actions:
        logger.info("")
        logger.info("【延迟卖出清单】")
        logger.info("-" * 120)
        
        widths = [15, 10, 15, 60]
        aligns = ['left', 'right', 'left', 'left']
        header = ["股票代码", "待卖股数", "状态", "原因"]
        logger.info(format_row(header, widths, aligns))
        logger.info("-" * 120)
        
        for action in pending_sell_actions:
            row = [
                action['ts_code'],
                str(action['shares']),
                action['status'],
                action['reason']
            ]
            logger.info(format_row(row, widths, aligns))
    
    # 3. T1 调仓订单清单
    if t1_actions:
        logger.info("")
        logger.info("【T1 调仓订单清单】")
        logger.info("-" * 120)
        
        widths = [15, 10, 10, 60]
        aligns = ['left', 'left', 'right', 'left']
        header = ["股票代码", "方向", "股数", "原因"]
        logger.info(format_row(header, widths, aligns))
        logger.info("-" * 120)
        
        for action in t1_actions:
            row = [
                action['ts_code'],
                action['action'],
                str(action['shares']),
                action['reason']
            ]
            logger.info(format_row(row, widths, aligns))
    
    # 4. T0 生成目标清单
    if t0_targets:
        logger.info("")
        logger.info("【T0 生成目标清单】")
        logger.info("-" * 120)
        
        widths = [15, 15, 60]
        aligns = ['left', 'right', 'left']
        header = ["股票代码", "目标权重", "原因/评分"]
        logger.info(format_row(header, widths, aligns))
        logger.info("-" * 120)
        
        for target in t0_targets:
            reason_score = target['reason']
            if target.get('score') is not None:
                reason_score = f"{target['reason']} (score: {target['score']:.4f})"
            
            row = [
                target['ts_code'],
                f"{target['target_weight']:.4f}",
                reason_score
            ]
            logger.info(format_row(row, widths, aligns))
    
    # 汇总
    total_actions = len(stop_loss_actions) + len(pending_sell_actions) + len(t1_actions) + len(t0_targets)
    if total_actions == 0:
        logger.info("")
        logger.info("今日无需手工操作")


def view_positions(args):
    """查看当前持仓"""
    logger.info("=" * 80)
    logger.info("查看纸面交易持仓")
    logger.info("=" * 80)
    logger.info(f"参考日期: {args.trade_date}")
    logger.info("=" * 80)
    
    # 读取配置（可选，用于获取一些参数）
    storage = PaperStorage()
    config = storage.load_config()
    
    # 创建运行器
    runner = PaperTradingRunner()
    
    try:
        # 加载价格数据
        loader = DataLoader(runner.storage)
        
        daily_data = loader.load_clean_daily_by_date(args.trade_date)
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {args.trade_date} 的价格数据")
            sys.exit(1)
        
        # 构建价格字典（使用收盘价）
        prices = {}
        for _, row in daily_data.iterrows():
            prices[row['ts_code']] = row['close']
        
        # 打印持仓明细
        runner.broker.print_positions_summary(prices, args.trade_date)
        
    except Exception as e:
        logger.exception(f"查看持仓失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="纸面交易命令行工具（重构版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # config 子命令
    config_parser = subparsers.add_parser(
        'config',
        help='设置全局配置（持久化）'
    )
    config_parser.add_argument(
        '--buy-price',
        choices=['open', 'close'],
        default='close',
        help='买入价格类型（默认：close）'
    )
    config_parser.add_argument(
        '--sell-price',
        choices=['open', 'close'],
        default='close',
        help='卖出价格类型（默认：close）'
    )
    config_parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='持仓股票数（默认：5）'
    )
    config_parser.add_argument(
        '--initial-capital',
        type=float,
        default=500000.0,
        help='初始资金（默认：500000）'
    )
    config_parser.add_argument(
        '--rebalance-freq',
        type=int,
        default=5,
        help='调仓频率（交易日数，默认：5）'
    )
    config_parser.add_argument(
        '--weight-method',
        choices=['equal', 'score'],
        default='equal',
        help='权重分配方法（默认：equal）'
    )
    config_parser.add_argument(
        '--model-version',
        type=int,
        help='ML模型版本（可选）'
    )
    config_parser.add_argument(
        '--universe',
        choices=['mainboard', 'all'],
        default='mainboard',
        help='股票池类型（默认：mainboard）'
    )
    config_parser.add_argument(
        '--stop-loss-enabled',
        action='store_true',
        help='启用止损功能'
    )
    config_parser.add_argument(
        '--stop-loss-drawdown-pct',
        type=float,
        default=20.0,
        help='回撤止损百分比（默认：20.0）'
    )
    config_parser.add_argument(
        '--stop-loss-trailing-enabled',
        action='store_true',
        help='启用移动止损'
    )
    config_parser.add_argument(
        '--stop-loss-trailing-pct',
        type=float,
        default=15.0,
        help='移动止损百分比（默认：15.0）'
    )
    config_parser.add_argument(
        '--stop-loss-consecutive-limit-down',
        type=int,
        default=2,
        help='连续跌停触发天数（默认：2）'
    )
    
    # run 子命令
    run_parser = subparsers.add_parser(
        'run',
        help='每日运行入口，自动编排执行各项动作'
    )
    run_parser.add_argument(
        '--trade-date',
        required=True,
        help='交易日期，格式YYYYMMDD'
    )
    run_parser.add_argument(
        '--model-version',
        type=int,
        help='ML模型版本（覆盖配置）'
    )
    run_parser.add_argument(
        '--weight-method',
        choices=['equal', 'score'],
        help='权重分配方法（覆盖配置）'
    )
    
    # positions 子命令
    pos_parser = subparsers.add_parser(
        'positions',
        help='查看当前持仓明细'
    )
    pos_parser.add_argument(
        '--trade-date',
        required=True,
        help='参考交易日期（用于获取当前价格），格式YYYYMMDD'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    # 执行命令
    if args.command == 'config':
        run_config(args)
    elif args.command == 'run':
        run_main(args)
    elif args.command == 'positions':
        view_positions(args)


if __name__ == "__main__":
    main()
