"""纸面交易展示与格式化公共能力。"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..common.config import get_models_root
from ..data import DataLoader, Storage
from ..ml import ModelRegistry
from .runner import PaperTradingRunner
from .runtime import PaperTradeExecutionResult
from .storage import PaperStorage


@dataclass
class PaperPositionSnapshot:
    """纸面交易持仓快照。"""

    trade_date: str
    runner: PaperTradingRunner
    stock_names: Dict[str, str]
    prices: Dict[str, float]
    positions_df: pd.DataFrame
    cash: float
    total_cost: float
    total_value: float
    total_profit: float
    total_assets: float
    round_pnl_pct: float
    total_pnl_pct: float
    initial_capital: float
    rebalance_info: str


def format_model_info(models_dir: Optional[str] = None) -> str:
    """获取当前配置使用的模型信息。"""
    storage = PaperStorage()
    config = storage.load_config()
    if not config:
        return "未找到配置文件，请先编辑 data/paper/config.yaml 或运行 config 命令设置配置。"

    registry = ModelRegistry(models_dir=models_dir or get_models_root())
    models = registry.list_models()
    if not models:
        return "没有已注册的模型。请先使用 train_ml_model.py 训练模型。"

    target_version = config.get("model_version")
    target_meta = None
    if target_version is not None:
        for model in models:
            if model["version"] == target_version:
                target_meta = model
                break
    else:
        target_meta = models[-1]

    if target_meta is None:
        available_versions = [model["version"] for model in models]
        return f"未找到版本 {target_version} 的模型。可用版本: {available_versions}"

    lines = []
    version_label = target_meta["version_str"]
    if target_version is None:
        version_label += " (最新)"
    lines.append(f"当前模型: {version_label}")
    lines.append(f"  模型类型: {target_meta.get('model_type', '未知')}")
    lines.append(
        f"  训练区间: {target_meta.get('train_start_date', '?')} ~ {target_meta.get('train_end_date', '?')}"
    )
    lines.append(f"  特征数量: {target_meta.get('feature_count', '?')}")
    lines.append(f"  训练样本: {target_meta.get('n_samples', '?')}")
    lines.append(f"  标签列: {target_meta.get('label_column', '?')}")
    lines.append(f"  创建时间: {target_meta.get('created_at', '?')}")

    train_params = target_meta.get("train_params", {})
    if train_params:
        lines.append("  训练参数:")
        for key, value in train_params.items():
            lines.append(f"    {key}: {value}")

    performance = target_meta.get("performance_metrics", {})
    if performance:
        lines.append("  性能指标:")
        for split in ["validation", "test"]:
            split_data = performance.get(split, {})
            if isinstance(split_data, dict) and split_data:
                ic = split_data.get("ic") or split_data.get("rank_ic")
                rank_ic = split_data.get("rank_ic")
                r2 = split_data.get("r2")
                rmse = split_data.get("rmse")
                parts = []
                if ic is not None:
                    parts.append(f"IC={ic:.4f}")
                if rank_ic is not None:
                    parts.append(f"RankIC={rank_ic:.4f}")
                if r2 is not None:
                    parts.append(f"R2={r2:.4f}")
                if rmse is not None:
                    parts.append(f"RMSE={rmse:.4f}")
                if parts:
                    lines.append(f"    {split}: {', '.join(parts)}")

        for split in ["validation_daily", "test_daily"]:
            split_data = performance.get(split, {})
            if isinstance(split_data, dict) and split_data:
                rankic_mean = split_data.get("daily_rankic_mean")
                rankic_ir = split_data.get("daily_rankic_ir")
                top30_return = split_data.get("top30_return_mean")
                parts = []
                if rankic_mean is not None:
                    parts.append(f"DailyRankIC={rankic_mean:.4f}")
                if rankic_ir is not None:
                    parts.append(f"IR={rankic_ir:.4f}")
                if top30_return is not None:
                    parts.append(f"Top30Ret={top30_return:.4f}")
                if parts:
                    lines.append(f"    {split}: {', '.join(parts)}")

    if config.get("model_version_b") is not None:
        weight_a = config.get("ensemble_weight_a", 0.5)
        lines.append("")
        lines.append("集成模式: 双模型 Ensemble")
        lines.append(f"  模型A: v{config.get('model_version', '最新')} (权重 {weight_a})")
        lines.append(f"  模型B: v{config['model_version_b']} (权重 {1 - weight_a})")

    return "\n".join(lines)


def load_position_snapshot(
    trade_date: str,
    runner: Optional[PaperTradingRunner] = None,
) -> PaperPositionSnapshot:
    """加载持仓快照。"""
    active_runner = runner or PaperTradingRunner(verbose=False)
    corrected_trade_date = active_runner._correct_trade_date(trade_date)
    loader = DataLoader(active_runner.storage, verbose=False)
    daily_data = loader.load_clean_daily_by_date(corrected_trade_date)
    if daily_data is None or daily_data.empty:
        raise ValueError(f"无法加载 {corrected_trade_date} 的价格数据")

    prices = {row["ts_code"]: row["close"] for _, row in daily_data.iterrows()}
    stock_names = loader.build_stock_names_dict()
    positions_df = active_runner.broker.get_positions_detail(
        prices, corrected_trade_date, stock_names
    )

    storage = PaperStorage()
    config = storage.load_config()
    initial_capital = (
        config.get("initial_capital", active_runner.account.initial_capital)
        if config
        else active_runner.account.initial_capital
    )
    cash = active_runner.account.get_cash()

    if positions_df.empty:
        total_cost = 0.0
        total_value = 0.0
        total_profit = 0.0
    else:
        total_cost = float(
            positions_df["买入成本"].sum()
            + (positions_df["持仓股数"] * positions_df["买入均价"]).sum()
        )
        total_value = float(positions_df["当前市值"].sum())
        total_profit = float(positions_df["浮动盈亏"].sum())

    total_assets = cash + total_value
    total_pnl_pct = (
        (total_assets - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
    )
    round_pnl_pct = total_profit / total_cost * 100 if total_cost > 0 else 0.0

    return PaperPositionSnapshot(
        trade_date=corrected_trade_date,
        runner=active_runner,
        stock_names=stock_names,
        prices=prices,
        positions_df=positions_df,
        cash=float(cash),
        total_cost=total_cost,
        total_value=total_value,
        total_profit=total_profit,
        total_assets=float(total_assets),
        round_pnl_pct=round_pnl_pct,
        total_pnl_pct=total_pnl_pct,
        initial_capital=float(initial_capital),
        rebalance_info=_get_rebalance_status(corrected_trade_date),
    )


def format_positions_mobile(
    trade_date: str,
    runner: Optional[PaperTradingRunner] = None,
) -> str:
    """生成手机友好的持仓 Markdown 文本。"""
    try:
        snapshot = load_position_snapshot(trade_date, runner=runner)
    except Exception:
        return f"无法加载 {trade_date} 的价格数据"

    if snapshot.positions_df.empty:
        return "当前无持仓"

    total_sign = "+" if snapshot.total_pnl_pct >= 0 else ""
    round_sign = "+" if snapshot.round_pnl_pct >= 0 else ""

    lines = [
        f"持仓概览 ({snapshot.trade_date})",
        f"总资产: {snapshot.total_assets:,.0f}",
        f"现金: {snapshot.cash:,.0f} | 市值: {snapshot.total_value:,.0f}",
        f"本轮: {round_sign}{snapshot.round_pnl_pct:.2f}% | 总: {total_sign}{snapshot.total_pnl_pct:.2f}%",
    ]
    if snapshot.rebalance_info:
        lines.append(snapshot.rebalance_info)
    lines.append("---")

    df_sorted = snapshot.positions_df.sort_values(by="收益率(%)", ascending=False)
    for index, (_, row) in enumerate(df_sorted.iterrows(), 1):
        code_display = row["股票代码"]
        if "(" in code_display and code_display.endswith(")"):
            ts_part = code_display[: code_display.index("(")]
            name_part = code_display[code_display.index("(") + 1 : -1]
            code_display = f"{name_part}({ts_part})"

        pnl_pct = row["收益率(%)"]
        pnl_sign = "+" if pnl_pct >= 0 else ""
        lines.append(f"{index}. {code_display}")
        lines.append(f"   {row['持仓股数']:.0f}股({row['买入均价']:.2f}->{row['当前价格']:.2f})")
        lines.append(f"   {pnl_sign}{pnl_pct:.2f}% ({pnl_sign}{row['浮动盈亏']:,.0f})")

    return _md_join(lines)


def format_trade_result(result: PaperTradeExecutionResult) -> str:
    """生成手机友好的交易执行结果 Markdown。"""
    lines: List[str] = [f"交易执行完成 ({result.corrected_date})", ""]

    stop_loss_count = len(result.stop_loss_actions)
    early_exit_count = len(result.early_exit_actions)
    take_profit_count = len(result.take_profit_actions)
    pending_sell_count = len(result.pending_sell_actions)
    t1_buy_count = sum(1 for action in result.t1_actions if action["action"] == "buy")
    t1_sell_count = sum(1 for action in result.t1_actions if action["action"] == "sell")

    lines.append(f"止损: {'无触发' if stop_loss_count == 0 else f'{stop_loss_count}笔'}")
    lines.append(f"提前换出: {'无触发' if early_exit_count == 0 else f'{early_exit_count}笔'}")
    lines.append(f"整体止盈: {'无触发' if take_profit_count == 0 else f'{take_profit_count}笔'}")
    lines.append(f"延迟卖出: {pending_sell_count}笔")
    if result.t1_actions:
        lines.append(f"T1执行: 买{t1_buy_count}笔 卖{t1_sell_count}笔")
    else:
        lines.append("T1执行: 无待执行指令")

    if result.t0_targets:
        lines.append(f"T0信号: {len(result.t0_targets)}个新目标(明日执行)")
    elif result.t0_status == "already_run":
        lines.append("T0信号: 今日已执行过")
    elif result.t0_status == "not_rebalance_day":
        lines.append("T0信号: 非调仓日")
    elif result.t0_status.startswith("error:"):
        lines.append(f"T0信号: 执行失败 - {result.t0_status[6:]}")
    elif result.t0_status == "no_targets":
        lines.append("T0信号: 调仓日但未生成目标(数据可能不足)")
    else:
        lines.append("T0信号: 非调仓日或无新目标")

    if result.ect_reason and "未启用" not in result.ect_reason and "为空" not in result.ect_reason:
        lines.append(f"ECT系数: {result.ect_exposure:.2f} ({result.ect_reason})")

    rebalance_info = _get_rebalance_status(result.corrected_date)
    if rebalance_info:
        lines.append(rebalance_info)

    if result.missing_factors:
        total_factor_count = 5
        loaded_count = total_factor_count - len(result.missing_factors)
        lines.append("")
        lines.append(f"⚠ 因子覆盖: {loaded_count}/{total_factor_count}")
        lines.append(f"缺失: {', '.join(result.missing_factors)}")

    _append_trigger_section(lines, "止损卖出", result.stop_loss_actions, result.stock_names)
    _append_trigger_section(lines, "亏损提前换出", result.early_exit_actions, result.stock_names)
    _append_trigger_section(lines, "整体止盈", result.take_profit_actions, result.stock_names)

    if result.pending_sell_actions:
        lines.append("")
        lines.append("--- 延迟卖出 ---")
        for index, action in enumerate(result.pending_sell_actions, 1):
            name = result.stock_names.get(str(action["ts_code"]), "")
            lines.append(f"{index}. {name}({action['ts_code']})")
            lines.append(f"   量{action['shares']}, {action['status']}")
            lines.append(f"   因{action['reason']}")

    if result.t1_actions:
        t1_buys = [action for action in result.t1_actions if action["action"] == "buy"]
        t1_sells = [action for action in result.t1_actions if action["action"] == "sell"]
        lines.append("")
        lines.append("--- T1 今日操作明细 ---")
        if t1_buys:
            lines.append("买入-")
            for index, action in enumerate(t1_buys, 1):
                name = result.stock_names.get(str(action["ts_code"]), "")
                lines.append(f"{index}. {name}({action['ts_code']})")
                weight_str = _extract_weight(str(action.get("reason", "")))
                parts = [f"量{action['shares']}"]
                if weight_str:
                    parts.append(weight_str)
                lines.append(f"   {', '.join(parts)}")
        if t1_sells:
            lines.append("")
            lines.append("卖出-")
            for index, action in enumerate(t1_sells, 1):
                name = result.stock_names.get(str(action["ts_code"]), "")
                lines.append(f"{index}. {name}({action['ts_code']})")
                lines.append(f"   量{action['shares']}, 因{action['reason']}")

    if result.t0_instructions:
        buy_instructions = [
            instruction for instruction in result.t0_instructions if instruction.action == "buy"
        ]
        sell_instructions = [
            instruction for instruction in result.t0_instructions if instruction.action == "sell"
        ]
        lines.append("")
        lines.append("--- T0 明日交易指令 ---")
        if buy_instructions:
            lines.append("买入-")
            for index, instruction in enumerate(buy_instructions, 1):
                name = result.stock_names.get(instruction.ts_code, "")
                lines.append(f"{index}. {name}({instruction.ts_code})")
                weight_str = _extract_weight(instruction.reason, instruction.target_weight)
                parts = [f"量{instruction.shares}"]
                if weight_str:
                    parts.append(weight_str)
                lines.append(f"   {', '.join(parts)}")
        if sell_instructions:
            lines.append("")
            lines.append("卖出-")
            for index, instruction in enumerate(sell_instructions, 1):
                name = result.stock_names.get(instruction.ts_code, "")
                lines.append(f"{index}. {name}({instruction.ts_code})")
                lines.append(f"   量{instruction.shares}, 因{instruction.reason}")
    elif result.t0_targets:
        lines.append("")
        lines.append("--- T0 明日目标 ---")
        for index, target in enumerate(result.t0_targets, 1):
            name = result.stock_names.get(str(target["ts_code"]), "")
            lines.append(f"{index}. {name}({target['ts_code']})")
            lines.append(f"   权{target['target_weight']:.2%}, 因{target['reason']}")

    lines.append("")
    lines.append("---")
    try:
        snapshot = load_position_snapshot(result.corrected_date, runner=result.runner)
        lines.append(f"持仓: {len(snapshot.positions_df)}只 | 现金: {snapshot.cash:,.0f}")
        lines.append(f"总资产: {snapshot.total_assets:,.0f}")
        total_sign = "+" if snapshot.total_pnl_pct >= 0 else ""
        round_sign = "+" if snapshot.round_pnl_pct >= 0 else ""
        lines.append(
            f"本轮: {round_sign}{snapshot.round_pnl_pct:.2f}% | 总: {total_sign}{snapshot.total_pnl_pct:.2f}%"
        )
    except Exception:
        lines.append(
            f"持仓: {len(result.runner.account.get_positions())}只 | 现金: {result.runner.account.get_cash():,.0f}"
        )

    return _md_join(lines)


def _build_stock_names(loader: DataLoader) -> Dict[str, str]:
    """从 stock_basic 构建股票名称映射。"""
    return loader.build_stock_names_dict()


def _md_join(lines: List[str]) -> str:
    """拼接钉钉 Markdown 行文本。"""
    result = []
    for line in lines:
        if line.strip():
            result.append(line + "  ")
        else:
            result.append("")
    return "\n".join(result)


def _extract_weight(reason: str, target_weight: float = 0.0) -> str:
    """从文本中提取权重。"""
    if target_weight > 0:
        return f"权{target_weight:.2%}"
    match = re.search(r"权重[=:]([\d.]+)", reason)
    if match:
        return f"权{float(match.group(1)):.2%}"
    return ""


def _get_rebalance_status(trade_date: str) -> str:
    """计算已持仓交易日和距下次调仓剩余交易日。"""
    try:
        paper_storage = PaperStorage()
        rebalance_state = paper_storage.load_rebalance_state()
        config = paper_storage.load_config()
        if rebalance_state is None or config is None:
            return ""

        last_date = rebalance_state.get("last_rebalance_date")
        rebalance_freq = config.get("rebalance_freq", 20)
        if not last_date:
            return ""

        storage = Storage()
        loader = DataLoader(storage, verbose=False)
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            return ""

        trade_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
        if last_date not in trade_dates or trade_date not in trade_dates:
            return ""

        last_idx = trade_dates.index(last_date)
        current_idx = trade_dates.index(trade_date)
        days_held = current_idx - last_idx
        days_remaining = max(0, int(rebalance_freq) - days_held)
        return f"已持 {days_held}d 剩 {days_remaining}d"
    except Exception:
        return ""


def _append_trigger_section(
    lines: List[str],
    title: str,
    actions: List[Dict[str, object]],
    stock_names: Dict[str, str],
) -> None:
    """追加风控触发明细区块。"""
    if not actions:
        return

    lines.append("")
    lines.append(f"--- {title} ---")
    for index, action in enumerate(actions, 1):
        name = stock_names.get(str(action["ts_code"]), "")
        lines.append(f"{index}. {name}({action['ts_code']})")
        parts = [f"量{action['shares']}"]
        can_execute = action.get("can_execute")
        if can_execute is not None:
            parts.append("可执行" if can_execute else "跌停无法卖")
        lines.append(f"   {', '.join(parts)}")
        lines.append(f"   因{action['reason']}")
