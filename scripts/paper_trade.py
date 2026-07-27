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
  python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5 --initial-capital 500000 --rebalance-freq 5 --position-sizing equal
  python scripts/paper_trade.py run --trade-date 20260121
  python scripts/paper_trade.py positions --trade-date 20260122
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.common.print_table import format_row
from src.lazybull.common.trading_config import TradingConfig, add_trading_args
from src.lazybull.data import DataLoader, Storage
from src.lazybull.paper import (
    PaperStorage,
    PaperTradingRunner,
    create_paper_trade_runtime,
    execute_trade_workflow,
    format_next_day_instructions,
)
from src.lazybull.paper import format_model_info as shared_format_model_info
from src.lazybull.paper import (
    load_position_snapshot,
)
from src.lazybull.paper.runtime import _check_stop_loss as shared_check_stop_loss
from src.lazybull.paper.runtime import (
    _execute_t0_if_rebalance_day as shared_execute_t0_if_rebalance_day,
)
from src.lazybull.paper.runtime import _execute_t1_if_pending as shared_execute_t1_if_pending
from src.lazybull.paper.runtime import _handle_failed_buys as shared_handle_failed_buys
from src.lazybull.paper.runtime import _process_pending_buys as shared_process_pending_buys
from src.lazybull.paper.runtime import _process_pending_sells as shared_process_pending_sells

# 匹配告警信息中的关键字符串，设置为 ignore
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")


def format_model_info(models_dir: Optional[str] = None) -> str:
    """获取当前配置使用的模型信息

    Args:
        models_dir: 模型目录

    Returns:
        格式化的模型信息文本
    """
    return shared_format_model_info(models_dir=models_dir)


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
    new_config = trading_config.to_dict()

    # 加载已有配置作为基础，仅覆盖与默认值不同的字段（即用户明确指定的参数）
    storage = PaperStorage()
    existing_config = storage.load_config() or {}
    default_config = TradingConfig().to_dict()
    config = dict(existing_config)
    for k, v in new_config.items():
        if v != default_config.get(k):
            config[k] = v

    storage.save_config(config)

    logger.info("配置已保存成功！")
    logger.info("")
    logger.info("当前配置：")
    logger.info("-" * 80)

    # 格式化输出
    widths = [30, 50]
    aligns = ["left", "left"]

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

    try:
        runtime = create_paper_trade_runtime(args.model_version)
    except RuntimeError as exc:
        logger.error(str(exc))
        logger.error(
            "示例: python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5"
        )
        sys.exit(1)

    config = runtime.config

    logger.info("使用配置：")
    logger.info(f"  买入价格类型: {config['buy_price']}")
    logger.info(f"  卖出价格类型: {config['sell_price']}")
    logger.info(f"  持仓数: {config['top_n']}")
    logger.info(f"  调仓频率: {config['rebalance_freq']} 个交易日")
    logger.info(f"  仓位管理: {config.get('position_sizing', 'equal')}")
    logger.info(f"  特征预测周期（horizon）: {config['horizon']} 天")
    logger.info(
        f"  最小买入市值比例: {config.get('min_buy_value_ratio', 0.2):.0%}（相对平均仓位）"
    )
    logger.info(f"  止损开关: {config['stop_loss_enabled']}")
    logger.info(f"  ECT开关: {config.get('equity_curve_enabled', False)}")
    if config.get("market_regime_enabled"):
        logger.info(
            f"  市场择时: 启用 (模式={config.get('market_regime_mode', 'binary')})"
        )
    if config.get("market_regime_ma250_hard_stop"):
        logger.info(
            f"  MA250硬条件: 启用 (阈值={config.get('market_regime_ma250_threshold', 1.0)}"
            f", 仓位={config.get('market_regime_ma250_exposure', 0.0):.0%})"
        )
    if config.get("industry_momentum_filter"):
        logger.info(
            f"  行业动量过滤: 启用 (剔除后{config.get('industry_momentum_bottom_pct', 0.2):.0%})"
        )
        logger.info(
            f", 亏损换出={early_exit_text})"
        )
    if config.get("max_per_industry"):
        logger.info(f"  单行业最大持仓: {config['max_per_industry']}")
    if config.get("max_weight_per_stock"):
        logger.info(f"  单股最大权重: {config['max_weight_per_stock']:.2%}")
    logger.info(f"  排除ST: {config.get('exclude_st', True)}")
    logger.info(f"  最少上市天数: {config.get('min_list_days', 365)}")
    logger.info("=" * 80)

    try:
        result = execute_trade_workflow(args.trade_date, runtime=runtime)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print_positions(result.corrected_date, runner=result.runner)

    next_trade_date = result.runner._get_next_trade_date(result.corrected_date)
    logger.info("=" * 120)
    logger.info(
        f"运行完成 - {result.corrected_date}, 下个交易日: [{next_trade_date or '无'}]"
    )
    logger.info("=" * 120)


def _check_stop_loss(
    runner: PaperTradingRunner,
    stop_loss_monitor,
    trade_date: str,
    config: dict,
) -> List[Dict]:
    """检查止损触发。"""
    return shared_check_stop_loss(runner, stop_loss_monitor, trade_date, config)


def _check_early_exit(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
) -> List[Dict]:
    """检查亏损提前换出触发。"""


def _check_take_profit(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
) -> List[Dict]:
    """检查整体止盈触发。"""


def _process_pending_sells(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
) -> List[Dict]:
    """处理延迟卖出队列。"""
    return shared_process_pending_sells(runner, trade_date, config)


def _process_pending_buys(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
) -> List[Dict]:
    """处理延迟买入队列。"""
    return shared_process_pending_buys(runner, trade_date, config)


def _execute_t1_if_pending(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
) -> List[Dict]:
    """执行 T1（如果有交易指令或补位买入计划）。"""
    return shared_execute_t1_if_pending(runner, trade_date, config)


def _handle_failed_buys(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
    failed_buy_targets: List,
    attempt_count: int,
) -> None:
    """处理买入失败：生成补位计划。"""
    shared_handle_failed_buys(runner, trade_date, config, failed_buy_targets, attempt_count)


def _execute_t0_if_rebalance_day(
    runner: PaperTradingRunner,
    trade_date: str,
    config: dict,
) -> Tuple[List[Dict], float, str, str]:
    """执行 T0（如果是调仓日）。"""
    return shared_execute_t0_if_rebalance_day(runner, trade_date, config)


def view_positions(args):
    """查看当前持仓。"""
    logger.info("=" * 80)
    logger.info("查看纸面交易持仓")
    print_positions(args.trade_date)


def print_positions(trade_date: str, runner: Optional[PaperTradingRunner] = None):
    """打印当前持仓。"""
    try:
        snapshot = load_position_snapshot(trade_date, runner=runner)
        logger.info("=" * 80)
        next_day_text = format_next_day_instructions(
            snapshot.trade_date,
            runner=snapshot.runner,
            stock_names=snapshot.stock_names,
        )
        for line in next_day_text.splitlines():
            logger.info(line)
        logger.info("-" * 80)
        logger.info(f"[{snapshot.trade_date}]持仓情况")
        logger.info("=" * 80)
        snapshot.runner.broker.print_positions_summary(
            snapshot.prices,
            snapshot.trade_date,
            stock_names=snapshot.stock_names,
        )
    except Exception as exc:
        logger.exception(f"打印持仓失败: {exc}")
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
    runner: "PaperTradingRunner", prices: Dict[str, float], current_date: str, display_time: str
) -> None:
    """打印精简版实时收益统计"""
    df = runner.broker.get_positions_detail(prices, current_date)
    if df.empty:
        logger.info(f"[{display_time}] 当前无持仓")
        return

    total_cost = df["买入成本"].sum() + (df["持仓股数"] * df["买入均价"]).sum()
    total_value = df["当前市值"].sum()
    total_profit = df["浮动盈亏"].sum()
    profit_rate = (total_profit / total_cost * 100) if total_cost > 0 else 0.0

    cash = runner.account.get_cash()
    total_assets = cash + total_value

    storage = PaperStorage()
    config = storage.load_config()
    initial_capital = (
        config.get("initial_capital", runner.account.initial_capital)
        if config
        else runner.account.initial_capital
    )
    total_pnl_pct = (
        (total_assets - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
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


def _create_realtime_runner() -> Tuple[PaperTradingRunner, Dict]:
    """创建实时查询所需的 runner，并对齐持久化配置中的初始资金口径。"""
    storage = PaperStorage()
    config = storage.load_config() or {}

    try:
        initial_capital = float(config.get("initial_capital", 500000.0))
    except (TypeError, ValueError):
        initial_capital = 500000.0

    try:
        horizon = int(config.get("horizon", 20))
    except (TypeError, ValueError):
        horizon = 20

    position_sizing = str(config.get("position_sizing", "equal"))
    runner = PaperTradingRunner(
        initial_capital=initial_capital,
        position_sizing=position_sizing,
        horizon=horizon,
        verbose=False,
    )
    return runner, config


def run_real(args):
    """实时行情命令：获取持仓实时数据并展示"""
    from src.lazybull.data.tushare_client import TushareClient

    runner, _ = _create_realtime_runner()
    positions = runner.account.get_positions()

    if not positions:
        logger.info("当前无持仓")
        return

    ts_codes = ",".join(positions.keys())

    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(ts_codes)
    except Exception as e:
        logger.error(f"获取实时行情失败: {e}")
        return

    if rt_df is None or rt_df.empty:
        logger.error("实时行情数据为空")
        return

    # 构建价格字典（realtime_quote 返回大写列名）
    prices: Dict[str, float] = {}
    for _, row in rt_df.iterrows():
        ts_code = str(row.get("TS_CODE", ""))
        price = row.get("PRICE", None)
        if ts_code and price is not None:
            try:
                prices[ts_code] = float(price)
            except (ValueError, TypeError):
                pass

    # 警告缺失行情的持仓
    missing = [c for c in positions if c not in prices]
    if missing:
        logger.warning(f"以下持仓未获取到实时行情: {', '.join(missing)}")

    current_date = pd.Timestamp.today().strftime("%Y%m%d")
    quote_time = _extract_latest_quote_time(rt_df)
    display_time = quote_time or pd.Timestamp.now().strftime("%H:%M:%S")

    if not args.ret_profit_only:
        loader = DataLoader(runner.storage, verbose=False)
        stock_names = loader.build_stock_names_dict()

        logger.info("=" * 140)
        logger.info(f"实时持仓  [{display_time}]")
        logger.info("=" * 140)
        runner.broker.print_positions_summary(prices, current_date, stock_names=stock_names)
    else:
        _print_realtime_profit_only(runner, prices, current_date, display_time)


def _resolve_realtime_quote_price(row, fallback_price: float) -> float:
    """规范化实时价格；盘前/无效价格回退昨收，仍无效则回退买入价。"""
    price = row.get("PRICE", row.get("price"))
    pre_close = row.get("PRE_CLOSE", row.get("pre_close"))
    try:
        price_float = float(price)
    except (ValueError, TypeError):
        price_float = None
    try:
        pre_close_float = float(pre_close)
    except (ValueError, TypeError):
        pre_close_float = None

    if pre_close_float is not None and (not math.isfinite(pre_close_float) or pre_close_float <= 0):
        pre_close_float = None
    if price_float is not None and (not math.isfinite(price_float) or price_float <= 0):
        price_float = None

    if price_float is not None:
        return price_float
    if pre_close_float is not None:
        return pre_close_float
    return fallback_price


def _extract_latest_quote_time(rt_df: Optional[pd.DataFrame]) -> str:
    """从实时行情表提取最新 TIME（HH:MM:SS）。"""
    if rt_df is None or rt_df.empty:
        return ""

    latest_seconds = -1
    latest_time = ""
    for _, row in rt_df.iterrows():
        quote_time_raw = str(row.get("TIME", "")).strip()
        if not quote_time_raw:
            continue

        parts = quote_time_raw.split(":")
        if len(parts) < 2:
            continue

        hour_text = parts[0].strip()
        minute_text = parts[1].strip()
        second_text = parts[2].strip() if len(parts) >= 3 else "0"
        if not (hour_text.isdigit() and minute_text.isdigit() and second_text.isdigit()):
            continue

        hour = int(hour_text)
        minute = int(minute_text)
        second = int(second_text)
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            continue

        total_seconds = hour * 3600 + minute * 60 + second
        if total_seconds >= latest_seconds:
            latest_seconds = total_seconds
            latest_time = f"{hour:02d}:{minute:02d}:{second:02d}"

    return latest_time


def build_realtime_portfolio_summary_from_quotes(
    positions: Dict[str, object],
    cash: float,
    initial_capital: float,
    current_date: str,
    rt_df: Optional[pd.DataFrame],
    annualized_return_func: Optional[Callable[[float, float, str], Optional[float]]] = None,
) -> Optional[Dict]:
    """基于已获取的实时行情 DataFrame 计算持仓摘要。"""
    if not positions:
        total_assets = cash
        total_pnl_pct = (
            (total_assets - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
        )
        return {
            "pos_count": 0,
            "market_value": 0.0,
            "total_assets": total_assets,
            "float_pnl_pct": 0.0,
            "total_pnl_pct": total_pnl_pct,
            "annual_return_pct": 0.0,
            "quote_time": "",
        }

    if rt_df is None or rt_df.empty:
        return None

    prices: Dict[str, float] = {}
    quote_time = _extract_latest_quote_time(rt_df)
    for _, row in rt_df.iterrows():
        ts_code = str(row.get("TS_CODE", ""))
        if ts_code:
            pos = positions.get(ts_code)
            fallback_price = getattr(pos, "buy_price", 0.0) if pos is not None else 0.0
            prices[ts_code] = _resolve_realtime_quote_price(row, fallback_price)

    market_value = 0.0
    total_float_pnl = 0.0
    total_buy_value = 0.0
    for ts_code, pos in positions.items():
        current_price = prices.get(ts_code, getattr(pos, "buy_price", 0.0))
        buy_price = getattr(pos, "buy_price", 0.0)
        shares = getattr(pos, "shares", 0)
        market_value += current_price * shares
        total_float_pnl += (current_price - buy_price) * shares
        total_buy_value += buy_price * shares

    float_pnl_pct = (total_float_pnl / total_buy_value * 100) if total_buy_value > 0 else 0.0
    total_assets = cash + market_value
    total_pnl_pct = (
        (total_assets - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
    )

    annual_return_pct = 0.0
    try:
        if annualized_return_func is not None:
            result = annualized_return_func(initial_capital, total_assets, current_date)
            if result is not None:
                annual_return_pct = float(result)
    except Exception:
        pass

    return {
        "pos_count": len(positions),
        "market_value": market_value,
        "total_assets": total_assets,
        "float_pnl_pct": float_pnl_pct,
        "total_pnl_pct": total_pnl_pct,
        "annual_return_pct": annual_return_pct,
        "quote_time": quote_time,
    }


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

    runner, config = _create_realtime_runner()
    positions = runner.account.get_positions()
    cash = runner.account.get_cash()

    try:
        initial_capital = float(config.get("initial_capital", runner.account.initial_capital))
    except (TypeError, ValueError):
        initial_capital = float(runner.account.initial_capital)

    current_date = pd.Timestamp.today().strftime("%Y%m%d")

    if not positions:
        return build_realtime_portfolio_summary_from_quotes(
            positions=positions,
            cash=cash,
            initial_capital=initial_capital,
            current_date=current_date,
            rt_df=None,
        )

    ts_codes_str = ",".join(positions.keys())
    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(ts_codes_str)
    except Exception as e:
        logger.warning(f"获取实时行情失败: {e}")
        return None

    if rt_df is None or rt_df.empty:
        return None

    annualized_return_func = getattr(runner.broker, "_calculate_annualized_return", None)
    if not callable(annualized_return_func):
        annualized_return_func = None

    return build_realtime_portfolio_summary_from_quotes(
        positions=positions,
        cash=cash,
        initial_capital=initial_capital,
        current_date=current_date,
        rt_df=rt_df,
        annualized_return_func=annualized_return_func,
    )


def run_reset_t0(args):
    """重置纸面交易，清空所有交易数据恢复为新账户"""
    storage = PaperStorage()

    logger.info("=" * 80)
    logger.info("重置纸面交易：清空所有交易数据，恢复为新账户")
    logger.info("=" * 80)

    # 执行重置（清空所有数据，仅保留 config.yaml）
    storage.reset_t0()

    logger.info("")
    logger.info("重置完成，账户已恢复初始状态，可重新执行 run 命令")
    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="纸面交易命令行工具（重构版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # config 子命令 — 使用公共参数注册函数
    config_parser = subparsers.add_parser("config", help="设置全局配置（持久化）")
    add_trading_args(config_parser, include_price=True)

    # run 子命令
    run_parser = subparsers.add_parser("run", help="每日运行入口，自动编排执行各项动作")
    run_parser.add_argument(
        "--trade-date",
        default=pd.Timestamp.today().strftime("%Y%m%d"),
        help="交易日期，格式YYYYMMDD（默认：当前日期）",
    )
    run_parser.add_argument("--model-version", type=int, help="ML模型版本（覆盖配置）")
    # model-info 子命令
    subparsers.add_parser("model-info", help="查看当前使用的模型信息")

    # positions 子命令
    pos_parser = subparsers.add_parser("positions", help="查看当前持仓明细")
    pos_parser.add_argument(
        "--trade-date", required=True, help="参考交易日期（用于获取当前价格），格式YYYYMMDD"
    )

    # real 子命令
    real_parser = subparsers.add_parser("real", help="实时行情：获取持仓的实时数据并展示")
    real_parser.add_argument(
        "--ret-profit-only", action="store_true", help="仅显示收益统计（精简单行输出）"
    )

    # adjust 子命令
    adjust_parser = subparsers.add_parser(
        "adjust", help="手工修正账户状态（修正发生在 cut-off 日期的 run 之前）"
    )
    adjust_subparsers = adjust_parser.add_subparsers(dest="adjust_command", help="修正类型")

    # adjust delete-position
    delete_pos_parser = adjust_subparsers.add_parser(
        "delete-position", help="删除持仓并按买入价格释放资金"
    )
    delete_pos_parser.add_argument(
        "--trade-date", required=True, help="Cut-off 日期（修正生效的日期），格式YYYYMMDD"
    )
    delete_pos_parser.add_argument("--ts-code", required=True, help="股票代码")

    # adjust update-position
    update_pos_parser = adjust_subparsers.add_parser(
        "update-position", help="更新持仓股数和买入价格"
    )
    update_pos_parser.add_argument(
        "--trade-date", required=True, help="Cut-off 日期（修正生效的日期），格式YYYYMMDD"
    )
    update_pos_parser.add_argument("--ts-code", required=True, help="股票代码")
    update_pos_parser.add_argument("--shares", type=int, required=True, help="新持仓股数")
    update_pos_parser.add_argument("--buy-price", type=float, required=True, help="新买入价格")

    # adjust add-shares
    add_shares_parser = adjust_subparsers.add_parser("add-shares", help="对已有持仓加仓")
    add_shares_parser.add_argument(
        "--trade-date", required=True, help="Cut-off 日期（修正生效的日期），格式YYYYMMDD"
    )
    add_shares_parser.add_argument("--ts-code", required=True, help="股票代码")
    add_shares_parser.add_argument("--shares", type=int, required=True, help="加仓股数")
    add_shares_parser.add_argument("--price", type=float, required=True, help="加仓价格")

    # adjust reset-t0
    adjust_subparsers.add_parser(
        "reset-t0", help="重置最新T0日并清空所有延迟交易订单（允许重新执行T0）"
    )

    # adjust cash
    cash_parser = adjust_subparsers.add_parser("cash", help="设置账户现金")
    cash_parser.add_argument(
        "--trade-date", required=True, help="Cut-off 日期（修正生效的日期），格式YYYYMMDD"
    )
    cash_parser.add_argument("--set", type=float, required=True, help="新现金金额")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载

    # 执行命令
    if args.command == "config":
        run_config(args)
    elif args.command == "model-info":
        run_model_info(args)
    elif args.command == "run":
        run_main(args)
    elif args.command == "positions":
        view_positions(args)
    elif args.command == "real":
        run_real(args)
    elif args.command == "adjust":
        if args.adjust_command is None:
            adjust_parser.print_help()
            sys.exit(1)

        if args.adjust_command == "delete-position":
            run_adjust_delete_position(args)
        elif args.adjust_command == "update-position":
            run_adjust_update_position(args)
        elif args.adjust_command == "add-shares":
            run_adjust_add_shares(args)
        elif args.adjust_command == "cash":
            run_adjust_cash(args)
        elif args.adjust_command == "reset-t0":
            run_reset_t0(args)
        else:
            logger.error(f"未知的 adjust 子命令: {args.adjust_command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
