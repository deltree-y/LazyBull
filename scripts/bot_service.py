import sys, os
import logging
import subprocess
import shlex  # 用于智能拆分字符串，处理包含引号的路径等复杂情况
import threading
import traceback
from pathlib import Path

import dingtalk_stream as dts   # type: ignore
from dingtalk_stream import DingTalkStreamClient, Credential, AckMessage    # type: ignore

# 添加项目路径（与 paper_trade.py 保持一致）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv  # type: ignore
load_dotenv(project_root / ".env")

import pandas as pd
from src.lazybull.data import DataLoader, Storage
from src.lazybull.paper import PaperTradingRunner, PaperStorage
from src.lazybull.risk.stop_loss import StopLossConfig, StopLossMonitor
from src.lazybull.risk.equity_curve import EquityCurveMonitor, create_equity_curve_config_from_dict

# 1. 填入你的凭证
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# 辅助函数：构建股票名称字典（复用 paper_trade.py 逻辑）
# ---------------------------------------------------------------------------
def _build_stock_names(loader: DataLoader) -> dict:
    """从 stock_basic 构建 {ts_code: name} 映射"""
    stock_names = {}
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None or stock_basic.empty:
        stock_basic = loader.load_stock_basic()
    if stock_basic is None or stock_basic.empty:
        return stock_names
    if 'ts_code' not in stock_basic.columns or 'name' not in stock_basic.columns:
        return stock_names
    for _, row in stock_basic.iterrows():
        if pd.notna(row.get('ts_code')) and pd.notna(row.get('name')) and row['name']:
            stock_names[row['ts_code']] = row['name']
    return stock_names


def _short_code(ts_code: str) -> str:
    """600519.SH -> 600519"""
    return ts_code.split('.')[0] if '.' in ts_code else ts_code


# ---------------------------------------------------------------------------
# 格式化函数：手机友好的持仓展示
# ---------------------------------------------------------------------------
def format_positions_mobile(runner: PaperTradingRunner, trade_date: str) -> str:
    """生成手机友好的持仓 Markdown 文本"""
    loader = DataLoader(runner.storage, verbose=False)
    daily_data = loader.load_clean_daily_by_date(trade_date)
    if daily_data is None or daily_data.empty:
        return f"无法加载 {trade_date} 的价格数据"

    prices = {}
    for _, row in daily_data.iterrows():
        prices[row['ts_code']] = row['close']

    stock_names = _build_stock_names(loader)
    df = runner.broker.get_positions_detail(prices, trade_date, stock_names)

    if df.empty:
        return "当前无持仓"

    # 账户汇总
    total_cost = df['买入成本'].sum() + (df['持仓股数'] * df['买入均价']).sum()
    total_value = df['当前市值'].sum()
    total_profit = df['浮动盈亏'].sum()
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
    round_pnl_pct = (total_profit / total_cost * 100) if total_cost > 0 else 0.0

    sign = "+" if total_pnl_pct >= 0 else ""
    r_sign = "+" if round_pnl_pct >= 0 else ""

    lines = []
    lines.append(f"持仓概览 ({trade_date})")
    lines.append("")
    lines.append(f"总资产: {total_assets:,.0f}")
    lines.append(f"现金: {cash:,.0f} | 市值: {total_value:,.0f}")
    lines.append(f"本轮: {r_sign}{round_pnl_pct:.2f}% | 总: {sign}{total_pnl_pct:.2f}%")
    lines.append("")
    lines.append("---")

    # 按收益率排序
    df_sorted = df.sort_values(by='收益率(%)', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        # 股票代码列已经是 "ts_code(name)" 格式
        code_display = row['股票代码']
        pnl_pct = row['收益率(%)']
        p_sign = "+" if pnl_pct >= 0 else ""

        lines.append(f"{i}. {code_display}")
        lines.append(f"  {row['持仓股数']}股 现价{row['当前价格']:.2f} 成本{row['买入均价']:.2f}")
        lines.append(f"  盈亏: {p_sign}{pnl_pct:.2f}% ({p_sign}{row['浮动盈亏']:,.0f})")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 格式化函数：交易执行结果（重点突出调仓信号）
# ---------------------------------------------------------------------------
def format_trade_result(
    trade_date: str,
    corrected_date: str,
    stop_loss_actions: list,
    pending_sell_actions: list,
    t1_actions: list,
    t0_targets: list,
    t0_instructions: list,
    ect_exposure: float,
    ect_reason: str,
    runner: PaperTradingRunner,
    stock_names: dict,
) -> str:
    """生成手机友好的交易执行结果 Markdown"""
    lines = []
    lines.append(f"交易执行完成 ({corrected_date})")
    lines.append("")

    # --- 摘要 ---
    sl_count = len(stop_loss_actions)
    ps_count = len(pending_sell_actions)
    t1_buy = sum(1 for a in t1_actions if a['action'] == 'buy')
    t1_sell = sum(1 for a in t1_actions if a['action'] == 'sell')

    lines.append(f"止损: {'无触发' if sl_count == 0 else f'{sl_count}笔'}")
    lines.append(f"延迟卖出: {ps_count}笔")
    if t1_actions:
        lines.append(f"T1执行: 买{t1_buy}笔 卖{t1_sell}笔")
    else:
        lines.append("T1执行: 无待执行指令")
    if t0_targets:
        lines.append(f"T0信号: {len(t0_targets)}个新目标(明日执行)")
    else:
        lines.append("T0信号: 非调仓日或无新目标")

    # ECT 信息
    if ect_reason and "未启用" not in ect_reason and "为空" not in ect_reason:
        lines.append(f"ECT系数: {ect_exposure:.2f} ({ect_reason})")

    # --- 止损卖出明细 ---
    if stop_loss_actions:
        lines.append("")
        lines.append("--- 止损卖出 ---")
        for a in stop_loss_actions:
            name = stock_names.get(a['ts_code'], '')
            can = "可执行" if a['can_execute'] else "跌停无法卖"
            lines.append(f"卖 {name} {_short_code(a['ts_code'])}")
            lines.append(f"  {a['shares']}股 | {can}")
            lines.append(f"  原因: {a['reason']}")
            lines.append("")

    # --- 延迟卖出明细 ---
    if pending_sell_actions:
        lines.append("")
        lines.append("--- 延迟卖出 ---")
        for a in pending_sell_actions:
            name = stock_names.get(a['ts_code'], '')
            lines.append(f"卖 {name} {_short_code(a['ts_code'])}")
            lines.append(f"  {a['shares']}股 | {a['status']}")
            lines.append(f"  原因: {a['reason']}")
            lines.append("")

    # --- T1 已执行操作明细（重点） ---
    if t1_actions:
        lines.append("")
        lines.append("--- T1 今日操作明细 ---")
        for a in t1_actions:
            name = stock_names.get(a['ts_code'], '')
            direction = "买" if a['action'] == 'buy' else "卖"
            lines.append(f"{direction} {name} {_short_code(a['ts_code'])}")
            lines.append(f"  {a['shares']}股")
            lines.append(f"  原因: {a['reason']}")
            lines.append("")

    # --- T0 新目标/明日交易指令（核心：用户据此手工下单） ---
    if t0_instructions:
        lines.append("")
        lines.append("--- T0 明日交易指令 ---")
        for inst in t0_instructions:
            name = stock_names.get(inst.ts_code, '')
            direction = "买" if inst.action == 'buy' else "卖"
            lines.append(f"{direction} {name} {_short_code(inst.ts_code)}")
            lines.append(f"  {inst.shares}股 | 价格:{inst.price_type}")
            if inst.target_weight > 0:
                lines.append(f"  目标权重: {inst.target_weight:.2%}")
            lines.append(f"  原因: {inst.reason}")
            lines.append("")
    elif t0_targets:
        # 兜底：如果没有 instructions 但有 targets
        lines.append("")
        lines.append("--- T0 明日目标 ---")
        for t in t0_targets:
            name = stock_names.get(t['ts_code'], '')
            lines.append(f"买 {name} {_short_code(t['ts_code'])}")
            lines.append(f"  目标权重: {t['target_weight']:.2%}")
            lines.append(f"  原因: {t['reason']}")
            lines.append("")

    # --- 执行后持仓概要 ---
    lines.append("---")
    positions = runner.account.get_positions()
    cash = runner.account.get_cash()
    lines.append(f"持仓: {len(positions)}只 | 现金: {cash:,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心：执行交易逻辑（从 paper_trade.py run_main 提取）
# ---------------------------------------------------------------------------
def execute_trade(trade_date: str) -> str:
    """执行交易流程，返回格式化结果文本

    复用 paper_trade.py run_main 的核心逻辑，但不使用 logger 输出，
    而是收集结果后格式化为手机友好文本。
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")

    # 1. 读取配置
    storage = PaperStorage()
    config = storage.load_config()
    if config is None:
        return "错误: 未找到配置文件，请先运行 config 命令"

    if 'horizon' not in config:
        config['horizon'] = 5

    # 2. 创建运行器
    signal = None
    model_version_b = config.get('model_version_b')
    if model_version_b is not None:
        from src.lazybull.signals import EnsembleMLSignal
        signal = EnsembleMLSignal(
            model_version_a=config.get('model_version'),
            model_version_b=model_version_b,
            ensemble_weight_a=config.get('ensemble_weight_a', 0.5),
            top_n=config['top_n'],
            weight_method=config['weight_method'],
            verbose=False,
        )

    runner = PaperTradingRunner(
        signal=signal,
        initial_capital=config['initial_capital'],
        weight_method=config['weight_method'],
        horizon=config['horizon'],
    )

    # 3. 校正交易日期
    corrected_date = runner._correct_trade_date(trade_date)

    # 4. 创建止损监控器
    stop_loss_config = StopLossConfig(
        enabled=config['stop_loss_enabled'],
        drawdown_pct=config['stop_loss_drawdown_pct'],
        trailing_stop_enabled=config['stop_loss_trailing_enabled'],
        trailing_stop_pct=config['stop_loss_trailing_pct'],
        consecutive_limit_down_days=config['stop_loss_consecutive_limit_down']
    )
    stop_loss_monitor = StopLossMonitor(stop_loss_config)

    sl_state = storage.load_stop_loss_state()
    if sl_state:
        stop_loss_monitor.position_high_prices = sl_state.get('position_high_prices', {})
        stop_loss_monitor.consecutive_limit_down_days = sl_state.get('consecutive_limit_down_days', {})

    # 收集动作结果
    stop_loss_actions = []
    pending_sell_actions = []
    t1_actions = []
    t0_targets = []
    t0_instructions = []
    ect_exposure = 1.0
    ect_reason = "ECT 未启用"

    # 5. 止损检查（复用 paper_trade.py 的 _check_stop_loss）
    from scripts.paper_trade import _check_stop_loss, _process_pending_sells
    from scripts.paper_trade import _execute_t1_if_pending, _execute_t0_if_rebalance_day

    if config['stop_loss_enabled']:
        stop_loss_actions = _check_stop_loss(runner, stop_loss_monitor, corrected_date, config)
        sl_state = {
            'position_high_prices': stop_loss_monitor.position_high_prices,
            'consecutive_limit_down_days': stop_loss_monitor.consecutive_limit_down_days
        }
        storage.save_stop_loss_state(sl_state)

    # 6. 延迟卖出
    pending_sell_actions = _process_pending_sells(runner, corrected_date, config)

    # 7. T1 执行
    t1_actions = _execute_t1_if_pending(runner, corrected_date, config)

    # 8. T0 执行
    t0_targets, ect_exposure, ect_reason = _execute_t0_if_rebalance_day(
        runner, corrected_date, config
    )

    # 9. 获取 T0 生成的明日交易指令（这是用户手工下单的依据）
    t1_date = runner._get_next_trade_date(corrected_date)
    if t1_date:
        t0_instructions = runner.paper_storage.load_instructions(t1_date) or []

    # 10. 获取股票名称
    loader = DataLoader(runner.storage, verbose=False)
    stock_names = _build_stock_names(loader)

    # 11. 格式化结果
    return format_trade_result(
        trade_date=trade_date,
        corrected_date=corrected_date,
        stop_loss_actions=stop_loss_actions,
        pending_sell_actions=pending_sell_actions,
        t1_actions=t1_actions,
        t0_targets=t0_targets,
        t0_instructions=t0_instructions,
        ect_exposure=ect_exposure,
        ect_reason=ect_reason,
        runner=runner,
        stock_names=stock_names,
    )


# ===========================================================================
# 钉钉消息处理器
# ===========================================================================
class SimpleHandler(dts.ChatbotHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 注册表：key 为主命令，value 为对应的处理方法
        self.commands = {
            "ping": self.handle_ping,
            "run":  self.handle_run,    # 通用运行命令
            "date": self.handle_date,
            "temp": lambda args, inc: self.run_shell(["vcgencmd", "measure_temp"], inc),
            "ip": lambda args, inc: self.run_shell(["curl", "ifconfig.me"], inc),
            "positions": self.handle_positions,
            "trade": self.handle_trade,
            "help": self.handle_help,
        }

    async def process(self, callback):
        incoming = dts.ChatbotMessage.from_dict(callback.data)
        raw_content = incoming.text.content.strip()
        
        # 1. 解析命令与参数
        # 例如："run ls -l" -> parts = ['run', 'ls', '-l']
        try:
            parts = shlex.split(raw_content)
        except ValueError:
            parts = raw_content.split() # 回退到普通拆分

        if not parts:
            return AckMessage.STATUS_OK, 'Empty'

        cmd = parts[0].lower()    # 主命令
        args = parts[1:]          # 参数列表

        # 2. 路由分发
        handler = self.commands.get(cmd)
        
        if handler:
            # 执行对应的函数，并传入解析好的参数列表
            handler(args, incoming)
        else:
            self.reply_text(f"未知命令: {cmd}\n输入 'help' 查看可用指令", incoming)

        return AckMessage.STATUS_OK, 'OK'

    # --- 处理函数定义 ---

    def handle_ping(self, args, incoming):
        # 如果有参数，比如 "ping 8.8.8.8"，则执行系统 ping
        if args:
            self.run_shell(["ping", "-c", "4", args[0]], incoming)
        else:
            self.reply_text("pong! (请提供 IP 地址以进行真实 ping)", incoming)

    def handle_run(self, args, incoming):
        """处理类似 'run ls -la' 的通用命令"""
        if not args:
            self.reply_text("错误: 'run' 需要配合具体命令，如 'run ls'", incoming)
            return
        # 直接执行 args 里的内容
        self.run_shell(args, incoming)

    def handle_date(self, args, incoming):
        # date 命令也可以带格式化参数
        self.run_shell(["date"] + args, incoming)

    def handle_positions(self, args, incoming):
        """查看持仓 — 手机友好格式
        用法: positions [日期]  例: positions 20260304
        """
        trade_date = args[0] if args else pd.Timestamp.today().strftime('%Y%m%d')
        try:
            runner = PaperTradingRunner(verbose=False)
            text = format_positions_mobile(runner, trade_date)
            self.reply_markdown("持仓概览", text, incoming)
        except Exception as e:
            self.reply_text(f"查询持仓失败: {e}", incoming)

    def handle_trade(self, args, incoming):
        """执行交易 — 异步执行，完成后返回结果
        用法: trade [日期]  例: trade 20260304
        """
        trade_date = args[0] if args else pd.Timestamp.today().strftime('%Y%m%d')
        self.reply_text(f"开始执行交易 ({trade_date})，请稍候...", incoming)

        def _run():
            import time
            start = time.monotonic()
            try:
                result_text = execute_trade(trade_date)
                elapsed = time.monotonic() - start
                result_text += f"\n\n(耗时: {elapsed:.0f}秒)"
                self.reply_markdown("交易结果", result_text, incoming)
            except Exception as e:
                elapsed = time.monotonic() - start
                tb = traceback.format_exc()
                short_tb = "\n".join(tb.strip().split("\n")[-5:])
                self.reply_text(
                    f"交易执行失败 (耗时{elapsed:.0f}秒):\n{short_tb}", incoming
                )

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def handle_help(self, args, incoming):
        """显示帮助信息"""
        text = (
            "可用命令\n\n"
            "positions [日期]\n"
            "  查看持仓（默认今天）\n\n"
            "trade [日期]\n"
            "  执行交易（默认今天）\n\n"
            "ping [IP]\n"
            "  测试连通性\n\n"
            "date\n"
            "  查看服务器时间\n\n"
            "temp\n"
            "  查看CPU温度\n\n"
            "ip\n"
            "  查看公网IP\n\n"
            "run [命令]\n"
            "  执行shell命令"
        )
        self.reply_markdown("帮助", text, incoming)

    def run_shell(self, cmd_list, incoming):
        """底层的安全执行逻辑"""
        try:
            # 使用列表形式调用 subprocess 更加安全，避免 Shell 注入
            result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip() or "执行完毕（无输出）"
            self.reply_text(f"[{' '.join(cmd_list)}]:\n{output}", incoming)
        except Exception as e:
            self.reply_text(f"执行失败: {str(e)}", incoming)


def main():
    if APP_KEY is None or APP_SECRET is None:
        print("错误: 请在环境变量中设置 APP_KEY 和 APP_SECRET")
        return
    credential = Credential(APP_KEY, APP_SECRET)
    client = DingTalkStreamClient(credential)
    # ChatbotMessage.TOPIC 就是 "/v1.0/im/bot/messages/get"
    client.register_callback_handler(dts.ChatbotMessage.TOPIC, SimpleHandler())
    print("启动中，等待消息...")
    client.start_forever()

if __name__ == "__main__":
    main()