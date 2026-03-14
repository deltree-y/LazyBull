#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纸面交易脚本 - 重构版

功能：
- config 子命令：设置全局配置（持久化）
- run 子命令：每日运行入口，自动编排执行各项动作
- positions 子命令：查看持仓明细
- adjust reset-t0 子命令：重置T0日并清空延迟交易订单

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
from src.lazybull.common.trading_config import TradingConfig, add_trading_args
from src.lazybull.common.signal_factory import create_signal
from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml import ModelRegistry
from src.lazybull.paper import PaperTradingRunner, PaperStorage
from src.lazybull.risk.stop_loss import StopLossConfig, StopLossMonitor
from src.lazybull.risk.stop_loss_checker import check_positions_stop_loss
from src.lazybull.risk.equity_curve import EquityCurveConfig, EquityCurveMonitor, create_equity_curve_config_from_dict
import warnings
# 匹配告警信息中的关键字符串，设置为 ignore
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")


def format_model_info(models_dir: str = "./data/models") -> str:
    """获取当前配置使用的模型信息

    Args:
        models_dir: 模型目录

    Returns:
        格式化的模型信息文本
    """
    storage = PaperStorage()
    config = storage.load_config()
    if not config:
        return "未找到配置文件，请先运行 config 命令设置配置。"

    registry = ModelRegistry(models_dir=models_dir)
    models = registry.list_models()

    if not models:
        return "没有已注册的模型。请先使用 train_ml_model.py 训练模型。"

    lines = []

    # 从配置中读取版本
    target_version = config.get('model_version')

    # 查找目标模型
    target_meta = None
    if target_version is not None:
        for m in models:
            if m['version'] == target_version:
                target_meta = m
                break
    else:
        # 配置中未指定版本，使用最新
        target_meta = models[-1]

    if target_meta is None:
        return f"未找到版本 {target_version} 的模型。可用版本: {[m['version'] for m in models]}"

    # 显示当前使用的模型详情
    version_label = target_meta['version_str']
    if target_version is None:
        version_label += " (最新)"
    lines.append(f"当前模型: {version_label}")
    lines.append(f"  模型类型: {target_meta.get('model_type', '未知')}")
    lines.append(f"  训练区间: {target_meta.get('train_start_date', '?')} ~ {target_meta.get('train_end_date', '?')}")
    lines.append(f"  特征数量: {target_meta.get('feature_count', '?')}")
    lines.append(f"  训练样本: {target_meta.get('n_samples', '?')}")
    lines.append(f"  标签列: {target_meta.get('label_column', '?')}")
    lines.append(f"  创建时间: {target_meta.get('created_at', '?')}")

    # 训练参数
    train_params = target_meta.get('train_params', {})
    if train_params:
        lines.append(f"  训练参数:")
        for k, v in train_params.items():
            lines.append(f"    {k}: {v}")

    # 性能指标（只显示关键摘要）
    perf = target_meta.get('performance_metrics', {})
    if perf:
        lines.append(f"  性能指标:")
        # 优先显示 validation 和 test 中的关键指标
        for split in ['validation', 'test']:
            split_data = perf.get(split, {})
            if isinstance(split_data, dict) and split_data:
                ic = split_data.get('ic') or split_data.get('rank_ic')
                r2 = split_data.get('r2')
                rmse = split_data.get('rmse')
                parts = []
                if ic is not None:
                    parts.append(f"IC={ic:.4f}")
                rank_ic = split_data.get('rank_ic')
                if rank_ic is not None:
                    parts.append(f"RankIC={rank_ic:.4f}")
                if r2 is not None:
                    parts.append(f"R2={r2:.4f}")
                if rmse is not None:
                    parts.append(f"RMSE={rmse:.4f}")
                if parts:
                    lines.append(f"    {split}: {', '.join(parts)}")
        # validation_daily 关键指标
        vd = perf.get('validation_daily', {})
        if isinstance(vd, dict) and vd:
            rankic_mean = vd.get('daily_rankic_mean')
            rankic_ir = vd.get('daily_rankic_ir')
            top30_ret = vd.get('top30_return_mean')
            parts = []
            if rankic_mean is not None:
                parts.append(f"DailyRankIC={rankic_mean:.4f}")
            if rankic_ir is not None:
                parts.append(f"IR={rankic_ir:.4f}")
            if top30_ret is not None:
                parts.append(f"Top30Ret={top30_ret:.4f}")
            if parts:
                lines.append(f"    validation_daily: {', '.join(parts)}")
        # test_daily 关键指标
        td = perf.get('test_daily', {})
        if isinstance(td, dict) and td:
            rankic_mean = td.get('daily_rankic_mean')
            rankic_ir = td.get('daily_rankic_ir')
            top30_ret = td.get('top30_return_mean')
            parts = []
            if rankic_mean is not None:
                parts.append(f"DailyRankIC={rankic_mean:.4f}")
            if rankic_ir is not None:
                parts.append(f"IR={rankic_ir:.4f}")
            if top30_ret is not None:
                parts.append(f"Top30Ret={top30_ret:.4f}")
            if parts:
                lines.append(f"    test_daily: {', '.join(parts)}")

    # 双模型集成信息
    storage = PaperStorage()
    config = storage.load_config()
    if config and config.get('model_version_b') is not None:
        mv_b = config['model_version_b']
        weight_a = config.get('ensemble_weight_a', 0.5)
        lines.append("")
        lines.append(f"集成模式: 双模型 Ensemble")
        lines.append(f"  模型A: v{config.get('model_version', '最新')} (权重 {weight_a})")
        lines.append(f"  模型B: v{mv_b} (权重 {1 - weight_a})")

    return "\n".join(lines)


def run_model_info(_args):
    """模型信息命令"""
    logger.info("=" * 80)
    logger.info("模型信息")
    logger.info("=" * 80)

    text = format_model_info()
    for line in text.split("\n"):
        logger.info(line)

    logger.info("=" * 80)


def run_config(args):
    """配置命令：持久化全局配置"""
    logger.info("=" * 80)
    logger.info("纸面交易配置设置")
    logger.info("=" * 80)

    # 通过 TradingConfig 统一构建配置
    trading_config = TradingConfig.from_args(args)
    config = trading_config.to_dict()

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
    
    # 设置默认 horizon，如果配置中不存在
    if 'horizon' not in config:
        config['horizon'] = 20  # 默认持仓周期20天
    
    logger.info("使用配置：")
    logger.info(f"  买入价格类型: {config['buy_price']}")
    logger.info(f"  卖出价格类型: {config['sell_price']}")
    logger.info(f"  持仓数: {config['top_n']}")
    logger.info(f"  调仓频率: {config['rebalance_freq']} 个交易日")
    logger.info(f"  权重方法: {config['weight_method']}")
    logger.info(f"  特征预测周期（horizon）: {config['horizon']} 天")
    logger.info(f"  止损开关: {config['stop_loss_enabled']}")
    logger.info(f"  ECT开关: {config.get('equity_curve_enabled', False)}")
    if config.get('max_per_industry'):
        logger.info(f"  单行业最大持仓: {config['max_per_industry']}")
    if config.get('max_weight_per_stock'):
        logger.info(f"  单股最大权重: {config['max_weight_per_stock']:.2%}")
    logger.info(f"  排除ST: {config.get('exclude_st', True)}")
    logger.info(f"  最少上市天数: {config.get('min_list_days', 365)}")
    logger.info("=" * 80)
    
    # 2. 创建运行器（通过公共工厂函数创建 signal）
    trading_config = TradingConfig.from_dict(config)
    signal = create_signal(trading_config) if trading_config.model_version_b is not None else None

    runner = PaperTradingRunner(
        signal=signal,
        initial_capital=config['initial_capital'],
        weight_method=config['weight_method'],
        horizon=config['horizon'],
    )

    # 3. 校正交易日期
    corrected_date = runner._correct_trade_date(args.trade_date)

    # 4. 日期回退检测
    account_state = storage.load_account_state()
    if account_state and account_state.last_update:
        if corrected_date < account_state.last_update:
            logger.error(
                f"日期回退：输入日期 {corrected_date} 早于账户最后更新日期 {account_state.last_update}，"
                f"不允许回退执行"
            )
            sys.exit(1)
    
    # 4. 创建止损监控器（通过 TradingConfig）
    stop_loss_config = trading_config.create_stop_loss_config() or StopLossConfig()
    stop_loss_monitor = StopLossMonitor(stop_loss_config)
    
    # 加载止损状态
    sl_state = storage.load_stop_loss_state()
    if sl_state:
        stop_loss_monitor.position_high_prices = sl_state.get('position_high_prices', {})
        stop_loss_monitor.consecutive_limit_down_days = sl_state.get('consecutive_limit_down_days', {})
    
    # 5. 执行止损检查
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤1: 检查止损触发")
    logger.info("-" * 80)

    if config['stop_loss_enabled']:
        _check_stop_loss(runner, stop_loss_monitor, corrected_date, config)

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

    _process_pending_sells(runner, corrected_date, config)

    # 7. 执行 T1（如果有待执行目标）
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤3: 检查并执行 T1")
    logger.info("-" * 80)

    _execute_t1_if_pending(runner, corrected_date, config)

    # 8. 判断是否调仓日并执行 T0
    logger.info("")
    logger.info("-" * 80)
    logger.info("步骤4: 检查是否调仓日并执行 T0")
    logger.info("-" * 80)

    _execute_t0_if_rebalance_day(runner, corrected_date, config)

    # 9. 打印持仓
    print_positions(corrected_date)

    logger.info("=" * 120)
    logger.info(f"运行完成 - {corrected_date}, 下个交易日: [{runner._get_next_trade_date(corrected_date)}]")
    logger.info("=" * 120)


def _check_stop_loss(
    runner: PaperTradingRunner,
    stop_loss_monitor: StopLossMonitor,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """检查止损触发（委托公共模块 check_positions_stop_loss）

    Returns:
        止损动作列表 [{ts_code, shares, reason, can_execute}, ...]
    """
    from src.lazybull.common.suspend_calendar import SuspendCalendar

    actions = []

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

    # 初始化停牌日历
    suspend_calendar = SuspendCalendar(runner.storage)

    # 调用公共止损检查
    sl_actions = check_positions_stop_loss(
        positions=positions,
        stop_loss_monitor=stop_loss_monitor,
        prices=prices,
        limit_down_info=limit_down_info,
        suspend_calendar=suspend_calendar,
        trade_date=trade_date,
    )

    # 转换为脚本层格式，并处理跌停延迟卖出队列
    for sl in sl_actions:
        pos = positions.get(sl.ts_code)
        sell_shares = (pos.shares // 100) * 100 if pos else 0

        actions.append({
            'ts_code': sl.ts_code,
            'shares': sell_shares,
            'reason': sl.reason,
            'can_execute': sl.can_execute,
        })

        if sl.is_limit_down:
            from src.lazybull.paper.models import PendingSell
            pending_sell = PendingSell(
                ts_code=sl.ts_code,
                shares=sell_shares,
                target_weight=0.0,
                reason=f"止损-{sl.reason}",
                create_date=trade_date,
                attempts=0,
            )
            runner.broker.pending_sells.append(pending_sell)
            runner.broker.storage.save_pending_sells(runner.broker.pending_sells)

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


def _process_pending_buys(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """处理延迟买入队列（补位计划）
    
    Returns:
        延迟买入动作列表 [{ts_code, target_weight, reason, status}, ...]
    """
    actions = []
    
    # 重试延迟买入
    fills, remaining_buys = runner.broker.retry_pending_buys(trade_date, config['buy_price'])
    
    # 收集仍在队列中的订单
    for pb in remaining_buys:
        actions.append({
            'ts_code': pb.ts_code,
            'target_weight': pb.target_weight,
            'reason': pb.reason,
            'status': f'不可买入（尝试次数: {pb.attempts}/5）'
        })
    
    # 收集已成交的订单
    for fill in fills:
        actions.append({
            'ts_code': fill.ts_code,
            'target_weight': 0.0,
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
    
    logger.info(f"延迟买入处理完成：成交 {len(fills)} 笔，剩余 {len(remaining_buys)} 笔")
    return actions


def _execute_t1_if_pending(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict
) -> List[Dict]:
    """执行 T1（如果有交易指令或补位买入计划）
    
    执行 instructions（指令驱动）和 pending_buys（补位队列）
    
    Returns:
        T1 动作列表 [{ts_code, action, shares, reason}, ...]
    """
    actions = []
    
    # 检查幂等性
    if runner.paper_storage.check_run_exists("t1", trade_date):
        logger.info(f"T1 工作流已在 {trade_date} 执行过，跳过")
        return actions

    # 向前搜索未执行的交易指令（支持跳日期场景）
    instructions = None
    inst_date = trade_date
    found = runner.paper_storage.find_pending_instructions(trade_date)
    if found:
        inst_date, instructions = found
        if inst_date != trade_date:
            # 检查是否过期：信号日期与当前日期间隔 >= rebalance_freq * 0.5
            source_date = instructions[0].source_date if instructions else inst_date
            try:
                trade_cal = runner.loader.load_clean_trade_cal()
                trade_dates_list = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
                src_idx = trade_dates_list.index(source_date)
                cur_idx = trade_dates_list.index(trade_date)
                gap = cur_idx - src_idx
                threshold = int(config['rebalance_freq'] * 0.5)
                if gap >= threshold:
                    logger.warning(
                        f"发现 {inst_date} 的未执行指令（信号日 {source_date}），"
                        f"但距今已 {gap} 个交易日，超过阈值 {threshold}（rebalance_freq*0.5），"
                        f"指令已过期，丢弃"
                    )
                    # 标记过期指令为已执行，防止后续重复拾取
                    runner.paper_storage.save_run_record("t1", inst_date, {
                        'trade_date': inst_date,
                        'note': f'指令过期丢弃（距信号日 {gap} 个交易日，阈值 {threshold}）',
                        'expired': True,
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
                    instructions = None
                else:
                    logger.info(
                        f"发现 {inst_date} 的未执行指令（延迟 {gap} 个交易日），"
                        f"将在 {trade_date} 补充执行"
                    )
            except (ValueError, Exception) as e:
                logger.warning(f"检查指令过期失败: {e}，按原日期执行")

    # 检查是否有补位买入计划
    pending_buys = runner.paper_storage.load_pending_buys()

    if not instructions and not pending_buys:
        logger.info(f"未找到 {trade_date} 的交易指令或补位买入计划，跳过 T1")
        return actions
    
    # 输出清晰的模式标识
    if instructions:
        logger.info("=" * 80)
        logger.info(f"【T1 指令驱动】读取到 {len(instructions)} 条交易指令")
        logger.info("=" * 80)
    
    if pending_buys:
        logger.info(f"找到 {len(pending_buys)} 个补位买入计划（将在指令执行后处理）")
    
    # 加载价格数据
    buy_prices, sell_prices = runner._load_prices(trade_date, config['buy_price'], config['sell_price'])
    
    if not buy_prices and not sell_prices:
        logger.error("无法加载价格数据，跳过 T1")
        return actions
    
    fills_count = 0
    orders_count = 0
    
    # 执行交易指令
    if instructions:
        logger.info("执行交易指令")
        fills = runner.broker.execute_instructions(
            instructions,
            buy_prices,
            sell_prices,
            trade_date
        )
        fills_count += len(fills) if fills else 0
        orders_count += len(instructions)
        
        # 收集动作
        for fill in fills:
            actions.append({
                'ts_code': fill.ts_code,
                'action': fill.action,
                'shares': fill.shares,
                'reason': fill.reason
            })
        
        logger.info(f"指令执行完成：{len(instructions)} 条指令，{len(fills)} 笔成交")
    
    # 获取买入失败的目标
    failed_buy_targets = runner.broker.get_failed_buy_targets()

    # 处理买入失败：生成补位计划
    if failed_buy_targets:
        _handle_failed_buys(runner, trade_date, config, failed_buy_targets, attempt_count=0)
    
    # 处理补位买入（如果有pending_buys）
    if pending_buys:
        logger.info("执行补位买入计划")
        replenishment_fills = runner._execute_pending_buys(
            pending_buys,
            buy_prices,
            trade_date,
            config['buy_price']
        )
        
        if replenishment_fills:
            fills_count += len(replenishment_fills)
            orders_count += len(replenishment_fills)
            
            # 收集动作
            for fill in replenishment_fills:
                actions.append({
                    'ts_code': fill.ts_code,
                    'action': fill.action,
                    'shares': fill.shares,
                    'reason': fill.reason
                })
        
        # 检查是否有新的失败买入（从_execute_pending_buys生成）
        new_failed_buy_targets = runner.broker.get_failed_buy_targets()
        if new_failed_buy_targets:
            # 获取当前最大尝试次数（从pending_buys中获取）
            max_attempt = max([pb.attempts for pb in pending_buys], default=0)
            _handle_failed_buys(runner, trade_date, config, new_failed_buy_targets, attempt_count=max_attempt)
    
    # 更新账户状态
    if fills_count > 0:
        runner.account.update_last_date(trade_date)
        runner.account.save_state()
        
        # 记录净值
        all_prices = {**sell_prices, **buy_prices}
        runner._record_nav(trade_date, all_prices)
    
    # 保存执行记录
    if instructions or pending_buys:
        run_record = {
            'trade_date': trade_date,
            'buy_price_type': config['buy_price'],
            'sell_price_type': config['sell_price'],
            'instructions_count': len(instructions) if instructions else 0,
            'pending_buys_count': len(pending_buys) if pending_buys else 0,
            'orders_count': orders_count,
            'fills_count': fills_count,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        runner.paper_storage.save_run_record("t1", trade_date, run_record)
        # 如果指令来自其他日期，也标记原日期已执行，防止重复拾取
        if inst_date != trade_date and instructions:
            runner.paper_storage.save_run_record("t1", inst_date, {
                **run_record,
                'note': f'指令延迟执行，实际执行日期 {trade_date}'
            })
    
    logger.info(f"T1 执行完成：{len(actions)} 个订单")
    return actions


def _handle_failed_buys(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
    failed_buy_targets: List,
    attempt_count: int
) -> None:
    """处理买入失败：生成补位计划
    
    Args:
        runner: 运行器
        trade_date: 当前交易日期
        config: 配置
        failed_buy_targets: 失败的买入目标列表
        attempt_count: 当前尝试次数
    """
    MAX_REPLENISHMENT_ATTEMPTS = 5
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"检测到 {len(failed_buy_targets)} 个买入失败目标")
    
    # 检查补位尝试次数
    next_attempt = attempt_count + 1
    if next_attempt > MAX_REPLENISHMENT_ATTEMPTS:
        logger.warning(f"补位尝试次数已达上限 ({MAX_REPLENISHMENT_ATTEMPTS})，不再继续补位")
        logger.info("=" * 80)
        runner.broker.clear_failed_buy_targets()
        return
    
    logger.info(f"基于当日 {trade_date} 数据重新生成下一交易日补位目标（第 {next_attempt} 次补位尝试）")
    logger.info("=" * 80)
    
    # 获取下一交易日
    next_trade_date = runner._get_next_trade_date(trade_date)
    if next_trade_date:
        # 基于当日 Tn 数据重新生成补位信号，用于下一交易日 Tn+1 买入
        replacement_targets = runner.generate_replacement_targets(
            trade_date=trade_date,
            failed_count=len(failed_buy_targets),
            universe_type=config['universe'],
            model_version=config.get('model_version'),
            buy_price_type=config['buy_price'],
            original_signal_date=trade_date,
            max_per_industry=config.get('max_per_industry'),
            exclude_st=config.get('exclude_st', True),
            min_list_days=config.get('min_list_days', 365),
        )
        
        if replacement_targets:
            # 转换为 PendingBuy 对象（增量买入计划）
            from src.lazybull.paper.models import PendingBuy
            pending_buys = []
            for target in replacement_targets:
                pending_buys.append(PendingBuy(
                    ts_code=target.ts_code,
                    target_weight=target.target_weight,
                    reason=target.reason,
                    create_date=trade_date,
                    attempts=next_attempt,
                    last_attempt_date="",
                    original_signal_date=trade_date
                ))
            
            # 保存到 pending_buys 队列
            runner.paper_storage.save_pending_buys(pending_buys)
            
            logger.info(f"已生成 {len(replacement_targets)} 个补位目标，保存到独立的补位买入队列")
            logger.info(f"下一交易日 {next_trade_date} 将自动读取并执行补位买入（第 {next_attempt}/{MAX_REPLENISHMENT_ATTEMPTS} 次尝试）")
            logger.info(f"补位买入不会触发现有持仓的卖出")
        else:
            logger.warning("无法生成补位目标，候选可能已耗尽")
    else:
        logger.error("无法获取下一交易日，补位计划生成失败")
    
    # 清空失败目标列表
    runner.broker.clear_failed_buy_targets()


def _execute_t0_if_rebalance_day(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict
) -> Tuple[List[Dict], float, str]:
    """执行 T0（如果是调仓日）
    
    Returns:
        (T0 目标列表, ECT系数, ECT原因) 元组
        - T0 目标列表: [{ts_code, target_weight, reason, score}, ...]
        - ECT系数: exposure_multiplier
        - ECT原因: ECT 计算原因
    """
    targets_info = []
    ect_exposure = 1.0
    ect_reason = "ECT 未启用"
    
    # 检查幂等性
    if runner.paper_storage.check_run_exists("t0", trade_date):
        logger.info(f"T0 工作流已在 {trade_date} 执行过，跳过")
        return targets_info, ect_exposure, ect_reason
    
    # 检查是否调仓日
    try:
        is_rebalance_day = runner._check_rebalance_day(trade_date, config['rebalance_freq'])
    except RuntimeError as e:
        logger.info(f"当前不是调仓日：{e}")
        #logger.info("非调仓日允许执行卖出和T1，T0跳过")
        return targets_info, ect_exposure, ect_reason
    
    if not is_rebalance_day:
        logger.info("非调仓日，跳过 T0")
        return targets_info, ect_exposure, ect_reason
    
    logger.info("当前是调仓日，执行 T0")
    
    # 计算 ECT 系数（在生成信号前计算）
    if config.get('equity_curve_enabled', False):
        logger.info("-" * 80)
        logger.info("计算 ECT 仓位系数")
        logger.info("-" * 80)
        
        # 创建 ECT 配置和监控器
        ect_config = create_equity_curve_config_from_dict(config)
        ect_monitor = EquityCurveMonitor(ect_config)
        
        # 加载历史 NAV
        nav_df = runner.paper_storage.load_all_nav()
        if nav_df is not None and len(nav_df) > 0:
            # 转为 Series (index=date, values=nav)
            nav_series = nav_df.set_index('trade_date')['nav']
            
            # 计算 exposure
            ect_exposure, ect_reason = ect_monitor.calculate_exposure(
                nav_series, 
                current_date=trade_date
            )
            
            logger.info(f"ECT 计算结果: {ect_reason}")
            logger.info(f"ECT 仓位系数: {ect_exposure:.2f}")
        else:
            logger.warning("NAV 历史为空，使用默认系数 1.0")
            ect_exposure = 1.0
            ect_reason = "NAV 历史为空"
        
        logger.info("-" * 80)
    
    # 执行 T0
    try:
        runner.run_t0(
            trade_date=trade_date,
            buy_price_type=config['buy_price'],
            sell_price_type=config['sell_price'],
            universe_type=config['universe'],
            top_n=config['top_n'],
            model_version=config.get('model_version'),
            rebalance_freq=config['rebalance_freq'],
            max_per_industry=config.get('max_per_industry'),
            max_weight_per_stock=config.get('max_weight_per_stock'),
            exclude_st=config.get('exclude_st', True),
            min_list_days=config.get('min_list_days', 365),
        )
        
        # 获取下一交易日
        t1_date = runner._get_next_trade_date(trade_date)
        if t1_date:
            # 读取生成的交易指令
            instructions = runner.paper_storage.load_instructions(t1_date)
            if instructions:
                # 应用 ECT 系数到目标权重（仅对买入指令调整股数）
                if ect_exposure < 1.0:
                    logger.info(f"应用 ECT 系数 {ect_exposure:.2f} 到买入指令")
                    valid_instructions = []
                    for inst in instructions:
                        if inst.action == 'buy':
                            original_shares = inst.shares
                            inst.shares = int(inst.shares * ect_exposure)
                            # 确保是100的倍数
                            inst.shares = (inst.shares // 100) * 100
                            
                            # 如果调整后股数为0，跳过该指令
                            if inst.shares == 0:
                                logger.warning(f"ECT调整后 {inst.ts_code} 股数为0，跳过该买入指令（原 {original_shares} 股）")
                                continue
                                
                            if inst.shares != original_shares:
                                inst.reason = f"{inst.reason} (ECT调整: {original_shares} -> {inst.shares}股)"
                        valid_instructions.append(inst)
                    
                    # 重新保存调整后的指令（仅保存有效指令）
                    runner.paper_storage.save_instructions(t1_date, valid_instructions)
                    logger.info(f"已将 ECT 系数应用到买入指令：{len(valid_instructions)}/{len(instructions)} 条有效")
                
                # 收集目标信息用于显示（仅买入指令）
                for inst in instructions:
                    if inst.action == 'buy':
                        targets_info.append({
                            'ts_code': inst.ts_code,
                            'target_weight': inst.target_weight,
                            'reason': inst.reason,
                            'score': None
                        })
        
        logger.info(f"T0 执行完成：生成 {len(targets_info)} 个目标")
    except Exception as e:
        logger.error(f"T0 执行失败: {e}")
    
    return targets_info, ect_exposure, ect_reason


def view_positions(args):
    """查看当前持仓"""
    logger.info("=" * 80)
    logger.info("查看纸面交易持仓")
    print_positions(args.trade_date)
    

def print_positions(trade_date: str):
    logger.info("=" * 80)
    logger.info(f"[{trade_date}]持仓情况")
    logger.info("=" * 80)
    # 读取配置（可选，用于获取一些参数）
    #storage = PaperStorage()
    #config = storage.load_config()
    
    # 创建运行器
    runner = PaperTradingRunner(verbose=False)
    
    try:
        # 加载价格数据
        loader = DataLoader(runner.storage, verbose=False)
        
        daily_data = loader.load_clean_daily_by_date(trade_date)
        if daily_data is None or daily_data.empty:
            logger.error(f"无法加载 {trade_date} 的价格数据")
            sys.exit(1)
        
        # 构建价格字典（使用收盘价）
        prices = {}
        for _, row in daily_data.iterrows():
            prices[row['ts_code']] = row['close']
        
        # 从 stock_basic 构建股票名称字典（使用 DataLoader 公共方法）
        stock_names = loader.build_stock_names_dict()

        # 打印持仓明细（传入股票名称字典）
        runner.broker.print_positions_summary(prices, trade_date, stock_names=stock_names)
        
    except Exception as e:
        logger.exception(f"打印持仓失败: {e}")
        sys.exit(1)

def run_adjust_delete_position(args):
    """删除持仓并按成本价释放资金"""
    logger.info("=" * 80)
    logger.info("手工修正：删除持仓")
    logger.info("=" * 80)
    logger.info(f"Cut-off 日期: {args.trade_date}")
    logger.info(f"股票代码: {args.ts_code}")
    
    # 加载账户
    storage = PaperStorage()
    account_state = storage.load_account_state()
    
    if account_state is None:
        logger.error("账户状态文件不存在，无法执行修正")
        sys.exit(1)
    
    # 检查持仓是否存在
    if args.ts_code not in account_state.positions:
        logger.error(f"持仓 {args.ts_code} 不存在，无法删除")
        sys.exit(1)
    
    position = account_state.positions[args.ts_code]
    
    # 按买入价格释放资金
    released_cash = position.shares * position.buy_price
    account_state.cash += released_cash
    
    # 删除持仓
    del account_state.positions[args.ts_code]
    
    logger.info(f"删除持仓: {args.ts_code}")
    logger.info(f"  股数: {position.shares}")
    logger.info(f"  买入价格: {position.buy_price:.2f}")
    logger.info(f"  释放资金: {released_cash:,.2f}")
    logger.info(f"  更新后现金: {account_state.cash:,.2f}")
    
    # 设置 last_update 为 cut-off 日期
    account_state.last_update = args.trade_date
    
    # 保存账户状态
    storage.save_account_state(account_state)
    
    # 执行清理
    logger.info("")
    storage.truncate_since(args.trade_date)
    
    logger.info("")
    logger.info("持仓删除完成")
    logger.info("=" * 80)


def run_adjust_update_position(args):
    """更新持仓股数和买入价格"""
    logger.info("=" * 80)
    logger.info("手工修正：更新持仓")
    logger.info("=" * 80)
    logger.info(f"Cut-off 日期: {args.trade_date}")
    logger.info(f"股票代码: {args.ts_code}")
    logger.info(f"新股数: {args.shares}")
    logger.info(f"新买入价格: {args.buy_price:.2f}")
    
    # 加载账户
    storage = PaperStorage()
    account_state = storage.load_account_state()
    
    if account_state is None:
        logger.error("账户状态文件不存在，无法执行修正")
        sys.exit(1)
    
    # 检查持仓是否存在
    if args.ts_code not in account_state.positions:
        logger.error(f"持仓 {args.ts_code} 不存在，无法更新")
        sys.exit(1)
    
    position = account_state.positions[args.ts_code]
    
    # 计算现金变动
    old_cost = position.shares * position.buy_price
    new_cost = args.shares * args.buy_price
    delta_cash = old_cost - new_cost
    
    logger.info(f"旧持仓: {position.shares} 股 @ {position.buy_price:.2f} = {old_cost:,.2f}")
    logger.info(f"新持仓: {args.shares} 股 @ {args.buy_price:.2f} = {new_cost:,.2f}")
    logger.info(f"现金变动: {delta_cash:+,.2f}")
    
    # 更新现金
    account_state.cash += delta_cash
    
    # 更新持仓
    position.shares = args.shares
    position.buy_price = args.buy_price
    position.buy_cost = args.shares * args.buy_price
    
    logger.info(f"更新后现金: {account_state.cash:,.2f}")
    
    # 设置 last_update 为 cut-off 日期
    account_state.last_update = args.trade_date
    
    # 保存账户状态
    storage.save_account_state(account_state)
    
    # 执行清理
    logger.info("")
    storage.truncate_since(args.trade_date)
    
    logger.info("")
    logger.info("持仓更新完成")
    logger.info("=" * 80)


def run_adjust_add_shares(args):
    """对已有持仓加仓"""
    logger.info("=" * 80)
    logger.info("手工修正：加仓")
    logger.info("=" * 80)
    logger.info(f"Cut-off 日期: {args.trade_date}")
    logger.info(f"股票代码: {args.ts_code}")
    logger.info(f"加仓股数: {args.shares}")
    logger.info(f"加仓价格: {args.price:.2f}")
    
    # 加载账户
    storage = PaperStorage()
    account_state = storage.load_account_state()
    
    if account_state is None:
        logger.error("账户状态文件不存在，无法执行修正")
        sys.exit(1)
    
    # 检查持仓是否存在
    if args.ts_code not in account_state.positions:
        logger.error(f"持仓 {args.ts_code} 不存在，无法加仓")
        logger.error("提示：add-shares 仅允许对已存在持仓加仓")
        logger.error("      如需新建持仓，请使用 update-position 命令")
        sys.exit(1)
    
    position = account_state.positions[args.ts_code]
    
    # 计算加仓成本
    add_cost = args.shares * args.price
    
    # 检查现金是否足够
    if add_cost > account_state.cash:
        logger.error(f"现金不足：需要 {add_cost:,.2f}，可用 {account_state.cash:,.2f}")
        sys.exit(1)
    
    # 扣减现金
    account_state.cash -= add_cost
    
    # 加权更新买入价格
    old_shares = position.shares
    old_buy_price = position.buy_price
    new_total_shares = old_shares + args.shares
    new_buy_price = (old_buy_price * old_shares + args.price * args.shares) / new_total_shares
    
    logger.info(f"旧持仓: {old_shares} 股 @ {old_buy_price:.2f}")
    logger.info(f"加仓: {args.shares} 股 @ {args.price:.2f}")
    logger.info(f"新持仓: {new_total_shares} 股 @ {new_buy_price:.2f}")
    logger.info(f"现金变动: -{add_cost:,.2f}")
    logger.info(f"更新后现金: {account_state.cash:,.2f}")
    
    # 更新持仓
    position.shares = new_total_shares
    position.buy_price = new_buy_price
    position.buy_cost = new_total_shares * new_buy_price
    
    # 设置 last_update 为 cut-off 日期
    account_state.last_update = args.trade_date
    
    # 保存账户状态
    storage.save_account_state(account_state)
    
    # 执行清理
    logger.info("")
    storage.truncate_since(args.trade_date)
    
    logger.info("")
    logger.info("加仓完成")
    logger.info("=" * 80)


def run_adjust_cash(args):
    """设置现金金额"""
    logger.info("=" * 80)
    logger.info("手工修正：设置现金")
    logger.info("=" * 80)
    logger.info(f"Cut-off 日期: {args.trade_date}")
    logger.info(f"新现金金额: {args.set:,.2f}")
    
    # 加载账户
    storage = PaperStorage()
    account_state = storage.load_account_state()
    
    if account_state is None:
        logger.error("账户状态文件不存在，无法执行修正")
        sys.exit(1)
    
    old_cash = account_state.cash
    
    # 设置现金
    account_state.cash = args.set
    
    logger.info(f"旧现金: {old_cash:,.2f}")
    logger.info(f"新现金: {account_state.cash:,.2f}")
    logger.info(f"变动: {account_state.cash - old_cash:+,.2f}")
    
    # 设置 last_update 为 cut-off 日期
    account_state.last_update = args.trade_date
    
    # 保存账户状态
    storage.save_account_state(account_state)
    
    # 执行清理
    logger.info("")
    storage.truncate_since(args.trade_date)
    
    logger.info("")
    logger.info("现金设置完成")
    logger.info("=" * 80)


def _print_realtime_profit_only(
    runner: 'PaperTradingRunner',
    prices: Dict[str, float],
    current_date: str,
    display_time: str
) -> None:
    """打印精简版实时收益统计"""
    df = runner.broker.get_positions_detail(prices, current_date)
    if df.empty:
        logger.info(f"[{display_time}] 当前无持仓")
        return

    total_cost = df['买入成本'].sum() + (df['持仓股数'] * df['买入均价']).sum()
    total_value = df['当前市值'].sum()
    total_profit = df['浮动盈亏'].sum()
    profit_rate = (total_profit / total_cost * 100) if total_cost > 0 else 0.0

    cash = runner.account.get_cash()
    total_assets = cash + total_value

    storage = PaperStorage()
    config = storage.load_config()
    initial_capital = (
        config.get('initial_capital', runner.account.initial_capital)
        if config else runner.account.initial_capital
    )
    total_pnl_pct = (
        (total_assets - initial_capital) / initial_capital * 100
        if initial_capital > 0 else 0.0
    )

    annualized = runner.broker._calculate_annualized_return(
        initial_capital, total_assets, current_date
    )
    ann_str = f"{annualized:.2f}%" if annualized is not None else "N/A"

    p_sign = "+" if total_profit >= 0 else ""
    t_sign = "+" if total_pnl_pct >= 0 else ""

    logger.info(
        f"[{display_time}] {len(df)}只 | "
        f"现金:{cash:,.2f} | 市值:{total_value:,.2f} | 总资产:{total_assets:,.2f} | "
        f"浮盈:{p_sign}{total_profit:,.2f}({p_sign}{profit_rate:.2f}%) | "
        f"总盈亏:{t_sign}{total_pnl_pct:.2f}% | 年化:{ann_str}"
    )


def run_real(args):
    """实时行情命令：获取持仓实时数据并展示"""
    from src.lazybull.data.tushare_client import TushareClient

    runner = PaperTradingRunner(verbose=False)
    positions = runner.account.get_positions()

    if not positions:
        logger.info("当前无持仓")
        return

    ts_codes = ','.join(positions.keys())

    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(ts_codes)
    except Exception as e:
        logger.error(f"获取实时行情失败: {e}")
        return

    if rt_df is None or rt_df.empty:
        logger.error("实时行情数据为空")
        return

    # 构建价格字典，记录报价时间（realtime_quote 返回大写列名）
    prices: Dict[str, float] = {}
    quote_time = ""
    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', ''))
        price = row.get('PRICE', None)
        if ts_code and price is not None:
            try:
                prices[ts_code] = float(price)
            except (ValueError, TypeError):
                pass
        if not quote_time:
            t = row.get('TIME', '')
            if t:
                quote_time = str(t)

    # 警告缺失行情的持仓
    missing = [c for c in positions if c not in prices]
    if missing:
        logger.warning(f"以下持仓未获取到实时行情: {', '.join(missing)}")

    current_date = pd.Timestamp.today().strftime('%Y%m%d')
    display_time = quote_time or pd.Timestamp.now().strftime('%H:%M:%S')

    if not args.ret_profit_only:
        loader = DataLoader(runner.storage, verbose=False)
        stock_names = loader.build_stock_names_dict()

        logger.info("=" * 140)
        logger.info(f"实时持仓  [{display_time}]")
        logger.info("=" * 140)
        runner.broker.print_positions_summary(prices, current_date, stock_names=stock_names)
    else:
        _print_realtime_profit_only(runner, prices, current_date, display_time)


def get_realtime_portfolio_summary() -> Optional[Dict]:
    """获取实时持仓摘要，供树莓派 LED 显示使用。

    通过 Tushare realtime_quote 接口获取当前持仓的实时价格，
    计算 6 项关键指标。

    Returns:
        dict: {
            pos_count: int       - 持仓数量
            market_value: float  - 持仓市值（元）
            total_assets: float  - 总资产（元）
            float_pnl_pct: float - 浮盈率（%，当前仓位未实现盈亏）
            total_pnl_pct: float - 总盈亏率（%，相对初始资金）
            annual_return_pct: float - 年化收益率（%）
            quote_time: str      - 行情时间
        }
        None if data unavailable
    """
    from src.lazybull.data.tushare_client import TushareClient

    runner = PaperTradingRunner(verbose=False)
    positions = runner.account.get_positions()
    cash = runner.account.get_cash()

    storage = PaperStorage()
    config = storage.load_config()
    initial_capital = (
        config.get('initial_capital', runner.account.initial_capital)
        if config else runner.account.initial_capital
    )

    current_date = pd.Timestamp.today().strftime('%Y%m%d')

    if not positions:
        total_assets = cash
        total_pnl_pct = (
            (total_assets - initial_capital) / initial_capital * 100
            if initial_capital > 0 else 0.0
        )
        return {
            'pos_count': 0,
            'market_value': 0.0,
            'total_assets': total_assets,
            'float_pnl_pct': 0.0,
            'total_pnl_pct': total_pnl_pct,
            'annual_return_pct': 0.0,
            'quote_time': '',
        }

    ts_codes_str = ','.join(positions.keys())
    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(ts_codes_str)
    except Exception as e:
        logger.warning(f"获取实时行情失败: {e}")
        return None

    if rt_df is None or rt_df.empty:
        return None

    prices: Dict[str, float] = {}
    quote_time = ""
    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', ''))
        price = row.get('PRICE', None)
        if ts_code and price is not None:
            try:
                prices[ts_code] = float(price)
            except (ValueError, TypeError):
                pass
        if not quote_time:
            t = row.get('TIME', '')
            if t:
                quote_time = str(t)

    # 计算市值和浮盈（基于买入均价）
    market_value = 0.0
    total_float_pnl = 0.0
    total_buy_value = 0.0
    for ts_code, pos in positions.items():
        current_price = prices.get(ts_code, pos.buy_price)
        market_value += current_price * pos.shares
        total_float_pnl += (current_price - pos.buy_price) * pos.shares
        total_buy_value += pos.buy_price * pos.shares

    float_pnl_pct = (total_float_pnl / total_buy_value * 100) if total_buy_value > 0 else 0.0
    total_assets = cash + market_value
    total_pnl_pct = (
        (total_assets - initial_capital) / initial_capital * 100
        if initial_capital > 0 else 0.0
    )

    annual_return_pct = 0.0
    try:
        if hasattr(runner.broker, '_calculate_annualized_return'):
            result = runner.broker._calculate_annualized_return(
                initial_capital, total_assets, current_date
            )
            if result is not None:
                annual_return_pct = float(result)
    except Exception:
        pass

    return {
        'pos_count': len(positions),
        'market_value': market_value,
        'total_assets': total_assets,
        'float_pnl_pct': float_pnl_pct,
        'total_pnl_pct': total_pnl_pct,
        'annual_return_pct': annual_return_pct,
        'quote_time': quote_time,
    }


def run_reset_t0(args):
    """重置最新T0日并清空所有延迟交易订单"""
    storage = PaperStorage()

    # 自动查找最新T0日期
    t0_date = storage.find_latest_t0()
    if t0_date is None:
        logger.error("未找到任何T0运行记录，无需重置")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"重置T0日: {t0_date}")
    logger.info("=" * 80)

    # 执行重置（内部调用 truncate_since 完成所有清理）
    storage.reset_t0(t0_date)

    logger.info("")
    logger.info("重置完成，可重新执行 run 命令")
    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="纸面交易命令行工具（重构版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # config 子命令 — 使用公共参数注册函数
    config_parser = subparsers.add_parser(
        'config',
        help='设置全局配置（持久化）'
    )
    add_trading_args(config_parser, include_price=True)
    
    # run 子命令
    run_parser = subparsers.add_parser(
        'run',
        help='每日运行入口，自动编排执行各项动作'
    )
    run_parser.add_argument(
        '--trade-date',
        default=pd.Timestamp.today().strftime('%Y%m%d'),
        help='交易日期，格式YYYYMMDD（默认：当前日期）'
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
    
    # model-info 子命令
    subparsers.add_parser(
        'model-info',
        help='查看当前使用的模型信息'
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
    
    # real 子命令
    real_parser = subparsers.add_parser(
        'real',
        help='实时行情：获取持仓的实时数据并展示'
    )
    real_parser.add_argument(
        '--ret-profit-only',
        action='store_true',
        help='仅显示收益统计（精简单行输出）'
    )

    # adjust 子命令
    adjust_parser = subparsers.add_parser(
        'adjust',
        help='手工修正账户状态（修正发生在 cut-off 日期的 run 之前）'
    )
    adjust_subparsers = adjust_parser.add_subparsers(dest='adjust_command', help='修正类型')
    
    # adjust delete-position
    delete_pos_parser = adjust_subparsers.add_parser(
        'delete-position',
        help='删除持仓并按买入价格释放资金'
    )
    delete_pos_parser.add_argument(
        '--trade-date',
        required=True,
        help='Cut-off 日期（修正生效的日期），格式YYYYMMDD'
    )
    delete_pos_parser.add_argument(
        '--ts-code',
        required=True,
        help='股票代码'
    )
    
    # adjust update-position
    update_pos_parser = adjust_subparsers.add_parser(
        'update-position',
        help='更新持仓股数和买入价格'
    )
    update_pos_parser.add_argument(
        '--trade-date',
        required=True,
        help='Cut-off 日期（修正生效的日期），格式YYYYMMDD'
    )
    update_pos_parser.add_argument(
        '--ts-code',
        required=True,
        help='股票代码'
    )
    update_pos_parser.add_argument(
        '--shares',
        type=int,
        required=True,
        help='新持仓股数'
    )
    update_pos_parser.add_argument(
        '--buy-price',
        type=float,
        required=True,
        help='新买入价格'
    )
    
    # adjust add-shares
    add_shares_parser = adjust_subparsers.add_parser(
        'add-shares',
        help='对已有持仓加仓'
    )
    add_shares_parser.add_argument(
        '--trade-date',
        required=True,
        help='Cut-off 日期（修正生效的日期），格式YYYYMMDD'
    )
    add_shares_parser.add_argument(
        '--ts-code',
        required=True,
        help='股票代码'
    )
    add_shares_parser.add_argument(
        '--shares',
        type=int,
        required=True,
        help='加仓股数'
    )
    add_shares_parser.add_argument(
        '--price',
        type=float,
        required=True,
        help='加仓价格'
    )
    
    # adjust reset-t0
    adjust_subparsers.add_parser(
        'reset-t0',
        help='重置最新T0日并清空所有延迟交易订单（允许重新执行T0）'
    )

    # adjust cash
    cash_parser = adjust_subparsers.add_parser(
        'cash',
        help='设置账户现金'
    )
    cash_parser.add_argument(
        '--trade-date',
        required=True,
        help='Cut-off 日期（修正生效的日期），格式YYYYMMDD'
    )
    cash_parser.add_argument(
        '--set',
        type=float,
        required=True,
        help='新现金金额'
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
    elif args.command == 'model-info':
        run_model_info(args)
    elif args.command == 'run':
        run_main(args)
    elif args.command == 'positions':
        view_positions(args)
    elif args.command == 'real':
        run_real(args)
    elif args.command == 'adjust':
        if args.adjust_command is None:
            adjust_parser.print_help()
            sys.exit(1)
        
        if args.adjust_command == 'delete-position':
            run_adjust_delete_position(args)
        elif args.adjust_command == 'update-position':
            run_adjust_update_position(args)
        elif args.adjust_command == 'add-shares':
            run_adjust_add_shares(args)
        elif args.adjust_command == 'cash':
            run_adjust_cash(args)
        elif args.adjust_command == 'reset-t0':
            run_reset_t0(args)
        else:
            logger.error(f"未知的 adjust 子命令: {args.adjust_command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
