import logging
import os
import shlex  # 用于智能拆分字符串，处理包含引号的路径等复杂情况
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import dingtalk_stream as dts  # type: ignore
import requests  # type: ignore
from dingtalk_stream import AckMessage, Credential, DingTalkStreamClient  # type: ignore

# 添加项目路径（与 paper_trade.py 保持一致）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv  # type: ignore

load_dotenv(project_root / ".env")

import pandas as pd

from src.lazybull.data import DataLoader, Storage
from src.lazybull.paper import (
    PaperStorage,
    execute_trade_workflow,
    format_model_info,
    format_positions_mobile,
    format_trade_result,
)

# 凭证与 Webhook
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK")

logging.basicConfig(level=logging.INFO)


class ProgressReporter:
    """定时向钉钉发送进度消息，防止长时间任务无响应

    用法:
        reporter = ProgressReporter(reply_func, incoming, interval=60)
        reporter.start()
        reporter.update("步骤1: 加载数据")
        ...
        reporter.stop()
    """

    def __init__(self, reply_func, incoming, interval: int = 60):
        self._reply = reply_func
        self._incoming = incoming
        self._interval = interval
        self._step = "初始化"
        self._start_time: float = 0
        self._timer: Optional[threading.Timer] = None
        self._stopped = False
        self._lock = threading.Lock()

    def update(self, step: str):
        """更新当前步骤描述"""
        with self._lock:
            self._step = step

    def start(self):
        """启动定时报告"""
        self._start_time = time.monotonic()
        self._schedule()

    def stop(self):
        """停止定时报告"""
        self._stopped = True
        if self._timer:
            self._timer.cancel()

    def _schedule(self):
        if self._stopped:
            return
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self):
        if self._stopped:
            return
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            step = self._step
        try:
            self._reply(f"仍在执行中... ({elapsed:.0f}秒)\n当前: {step}", self._incoming)
        except Exception:
            pass
        self._schedule()


def execute_trade(
    trade_date: str, progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[str, str]:
    """执行交易流程，返回 (格式化结果文本, 校正后交易日期)。"""
    result = execute_trade_workflow(trade_date, progress_callback=progress_callback)
    return format_trade_result(result), result.corrected_date


# ---------------------------------------------------------------------------
# 日志拦截器：检测 dingtalk_stream 库内部吞掉的回复失败
# ---------------------------------------------------------------------------
class _ReplyFailureDetector(logging.Handler):
    """临时挂载到 dingtalk_stream.handler logger，检测 reply 失败日志"""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.failed = False
        self.error_msg = ""

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        if "reply" in msg.lower() and "failed" in msg.lower():
            self.failed = True
            self.error_msg = msg


# ===========================================================================
# 钉钉消息处理器
# ===========================================================================
class SimpleHandler(dts.AsyncChatbotHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 注册表：key 为主命令，value 为对应的处理方法
        self.commands = {
            "ping": self.handle_ping,
            "run": self.handle_run,  # 通用运行命令
            "paper": self.handle_paper,
            "date": self.handle_date,
            "temp": lambda args, inc: self.run_shell(["vcgencmd", "measure_temp"], inc),
            "ip": lambda args, inc: self.run_shell(["curl", "ifconfig.me"], inc),
            "positions": self.handle_positions,
            "trade": self.handle_trade,
            "model": self.handle_model,
            "help": self.handle_help,
            "reboot": self.handle_reboot,
            "reset-t0": self.handle_reset_t0,
        }

    def process(self, callback):
        """处理钉钉消息 — 在线程池中执行，不阻塞 event loop

        整个方法包裹在顶层 try/except 中，确保任何异常都不会导致静默失败。
        如果通过 Stream 回复失败，则降级到 Webhook 发送错误信息。
        """
        incoming = None
        raw_content = "<未解析>"
        try:
            incoming = dts.ChatbotMessage.from_dict(callback.data)
            raw_content = incoming.text.content.strip()
            logging.info(f"收到消息: {raw_content}")

            # 1. 解析命令与参数
            try:
                parts = shlex.split(raw_content)
            except ValueError:
                parts = raw_content.split()

            if not parts:
                return AckMessage.STATUS_OK, "Empty"

            cmd = parts[0].lower()
            args = parts[1:]

            # 2. 路由分发
            handler = self.commands.get(cmd)
            try:
                if handler:
                    handler(args, incoming)
                else:
                    self._safe_reply(f"未知命令: {cmd}\n输入 'help' 查看可用指令", incoming)
            except Exception as e:
                logging.error(f"命令 '{cmd}' 执行异常: {traceback.format_exc()}")
                self._safe_reply(f"命令 '{cmd}' 执行异常: {e}", incoming)

        except Exception as e:
            # 顶层兜底：消息解析失败或其他意外异常
            logging.error(f"process 顶层异常: {traceback.format_exc()}")
            self._safe_reply(f"消息处理异常: {e}\n原始内容: {raw_content}", incoming)

        return AckMessage.STATUS_OK, "OK"

    def _safe_reply(self, text: str, incoming, max_retries: int = 2):
        """安全回复文本：带重试 + Webhook 降级

        dingtalk_stream 库的 reply_text 可能内部吞掉异常只打日志，
        因此通过拦截 logging 记录来检测失败，并自动重试或降级到 Webhook。
        """
        if incoming is not None:
            if self._try_stream_reply(lambda: self.reply_text(text, incoming), max_retries):
                return

        # Stream 全部失败，降级到 Webhook
        self._webhook_notify(text)

    def _safe_reply_markdown(self, title: str, text: str, incoming, max_retries: int = 2):
        """安全回复 Markdown：带重试 + Webhook 降级"""
        if incoming is not None:
            if self._try_stream_reply(
                lambda: self.reply_markdown(title, text, incoming), max_retries
            ):
                return

        # Stream 全部失败，降级到 Webhook（Webhook 不支持 Markdown，发纯文本）
        self._webhook_notify(f"[{title}]\n{text}")

    def _try_stream_reply(self, reply_fn, max_retries: int) -> bool:
        """尝试通过 Stream 回复，带重试

        通过拦截 logging 检测库内部吞掉的异常。

        Returns:
            True 表示成功，False 表示所有重试均失败
        """
        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait = min(2**attempt, 8)
                logging.info(f"Stream 回复重试 ({attempt}/{max_retries})，等待 {wait}s")
                time.sleep(wait)
            try:
                # 拦截 dingtalk_stream 库的 error 日志来检测吞掉的异常
                handler = _ReplyFailureDetector()
                dt_logger = logging.getLogger("dingtalk_stream.handler")
                dt_logger.addHandler(handler)
                try:
                    reply_fn()
                finally:
                    dt_logger.removeHandler(handler)

                if not handler.failed:
                    return True
                logging.warning(f"Stream 回复被库内部标记为失败: {handler.error_msg}")
            except Exception as e:
                logging.warning(f"Stream 回复抛出异常: {e}")

        return False

    def _webhook_notify(self, text: str):
        """通过 Webhook 发送消息（降级通道）"""
        if not DINGTALK_WEBHOOK:
            logging.error(f"无法回复用户（Webhook 未配置）: {text}")
            return
        # 钉钉 Webhook 文本长度限制约 20000 字符，截断防止超限
        if len(text) > 2000:
            text = text[:1950] + "\n...(内容过长已截断)"
        try:
            payload = {
                "msgtype": "text",
                "text": {"content": f"[Webhook] {text}"},
            }
            resp = requests.post(DINGTALK_WEBHOOK, json=payload, timeout=15)
            if resp.ok:
                logging.info("已通过 Webhook 发送消息")
            else:
                logging.error(f"Webhook 发送失败: {resp.text}")
        except Exception as e:
            logging.error(f"Webhook 发送异常: {e}")

    # --- 处理函数定义 ---

    def handle_ping(self, args, incoming):
        # 如果有参数，比如 "ping 8.8.8.8"，则执行系统 ping
        if args:
            self.run_shell(["ping", "-c", "4", args[0]], incoming)
        else:
            self._safe_reply("pong! (请提供 IP 地址以进行真实 ping)", incoming)

    def handle_run(self, args, incoming):
        """处理类似 'run ls -la' 的通用命令"""
        if not args:
            self._safe_reply("错误: 'run' 需要配合具体命令，如 'run ls'", incoming)
            return
        # 直接执行 args 里的内容
        self.run_shell(args, incoming)

    def handle_paper(self, args, incoming):
        """透传 paper_trade CLI 子命令。"""
        if not args:
            self._safe_reply(
                "错误: 请提供 paper_trade 子命令，例如 paper positions --trade-date 20260314",
                incoming,
            )
            return

        command = [sys.executable, str(project_root / "scripts" / "paper_trade.py"), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(project_root),
            )
        except subprocess.TimeoutExpired:
            self._safe_reply("paper_trade 执行超时，请优先使用 trade 命令运行长任务", incoming)
            return
        except Exception as exc:
            self._safe_reply(f"paper_trade 执行失败: {exc}", incoming)
            return

        output = result.stdout.strip() or result.stderr.strip() or "执行完毕（无输出）"
        if len(output) > 4000:
            output = output[:3900] + "\n...(输出过长已截断)"

        if result.returncode == 0:
            self._safe_reply_markdown("paper_trade", output, incoming)
        else:
            self._safe_reply_markdown(
                "paper_trade",
                f"执行失败 (退出码 {result.returncode})\n\n{output}",
                incoming,
            )

    def handle_date(self, args, incoming):
        # date 命令也可以带格式化参数
        self.run_shell(["date"] + args, incoming)

    def handle_positions(self, args, incoming):
        """查看持仓 — 手机友好格式
        用法: positions [日期]  例: positions 20260304
        """
        trade_date = args[0] if args else pd.Timestamp.today().strftime("%Y%m%d")
        try:
            text = format_positions_mobile(trade_date)
            self._safe_reply_markdown("持仓概览", text, incoming)
        except Exception as e:
            self._safe_reply(f"查询持仓失败: {e}", incoming)

    def handle_trade(self, args, incoming):
        """执行交易 — 异步执行，完成后返回结果
        用法: trade <日期|next>  例: trade 20260304 / trade next
        """
        if not args:
            self._safe_reply("错误: 请指定交易日期或 next，如 trade 20260314", incoming)
            return

        if args[0].lower() == "next":
            trade_date = self._resolve_next_trade_date()
            if trade_date is None:
                self._safe_reply("错误: 无法获取下一个交易日", incoming)
                return
        else:
            trade_date = args[0]
        self._safe_reply(f"开始执行交易 ({trade_date})，请稍候...", incoming)

        # 启动进度报告器，每60秒推送一次当前步骤
        reporter = ProgressReporter(self._safe_reply, incoming, interval=60)
        reporter.start()

        start = time.monotonic()
        try:
            result_text, corrected_date = execute_trade(
                trade_date, progress_callback=reporter.update
            )
            reporter.stop()
            # 记录本次实际执行日期，供 trade next 推算下一交易日
            PaperStorage().save_last_trade_date(corrected_date)
            elapsed = time.monotonic() - start
            result_text += f"\n\n(耗时: {elapsed:.0f}秒)"
            self._safe_reply_markdown("交易结果", result_text, incoming)
        except Exception as e:
            reporter.stop()
            elapsed = time.monotonic() - start
            tb = traceback.format_exc()
            short_tb = "\n".join(tb.strip().split("\n")[-5:])
            self._safe_reply(f"交易执行失败 (耗时{elapsed:.0f}秒):\n{short_tb}", incoming)

    def handle_model(self, args, incoming):
        """查看当前使用的模型信息
        用法: model
        """
        try:
            text = format_model_info()
            self._safe_reply_markdown("模型信息", text, incoming)
        except Exception as e:
            self._safe_reply(f"查询模型信息失败: {e}", incoming)

    def handle_help(self, args, incoming):
        """显示帮助信息"""
        text = (
            "可用命令\n\n"
            "positions [日期]\n"
            "  查看持仓（默认今天）\n\n"
            "trade <日期|next>\n"
            "  执行交易（next=下一交易日）\n\n"
            "paper <paper_trade 子命令...>\n"
            "  透传执行 paper_trade.py 的其他子命令\n\n"
            "model\n"
            "  查看当前模型信息\n\n"
            "ping [IP]\n"
            "  测试连通性\n\n"
            "date\n"
            "  查看服务器时间\n\n"
            "temp\n"
            "  查看CPU温度\n\n"
            "ip\n"
            "  查看公网IP\n\n"
            "run [命令]\n"
            "  执行shell命令\n\n"
            "reset-t0\n"
            "  重置交易数据恢复新账户\n\n"
            "reboot\n"
            "  重启系统"
        )
        self._safe_reply_markdown("帮助", text, incoming)

    def handle_reboot(self, _args, incoming):
        """重启树莓派系统
        用法: reboot
        """
        self._safe_reply("系统即将重启...", incoming)
        try:
            subprocess.Popen(
                ["sudo", "reboot"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._safe_reply(f"重启失败: {e}", incoming)

    def handle_reset_t0(self, _args, incoming):
        """重置纸面交易，清空所有交易数据恢复为新账户
        用法: reset-t0
        """
        try:
            ps = PaperStorage()
            ps.reset_t0()
            self._safe_reply("reset-t0 完成，账户已恢复初始状态", incoming)
        except Exception as e:
            self._safe_reply(f"reset-t0 失败: {e}", incoming)

    def _resolve_next_trade_date(self) -> str | None:
        """获取上次交易日之后的下一个交易日

        基于 last_trade_date 推算；若无记录则回退到今天起。
        """
        try:
            storage = Storage()
            loader = DataLoader(storage, verbose=False)
            trade_cal = loader.load_clean_trade_cal()
            if trade_cal is None:
                return None
            trade_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()

            last_date = PaperStorage().load_last_trade_date() or ""

            if last_date:
                future = [d for d in trade_dates if d > last_date]
            else:
                today = pd.Timestamp.today().strftime("%Y%m%d")
                future = [d for d in trade_dates if d >= today]
            return future[0] if future else None
        except Exception:
            return None

    def run_shell(self, cmd_list, incoming):
        """底层的安全执行逻辑"""
        try:
            # 使用列表形式调用 subprocess 更加安全，避免 Shell 注入
            result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip() or "执行完毕（无输出）"
            self._safe_reply(f"[{' '.join(cmd_list)}]:\n{output}", incoming)
        except Exception as e:
            self._safe_reply(f"执行失败: {str(e)}", incoming)


def _notify_startup():
    """通过 Webhook 发送启动通知到钉钉群"""
    if not DINGTALK_WEBHOOK:
        print("未配置 DINGTALK_WEBHOOK，跳过启动通知")
        return
    try:
        payload = {
            "msgtype": "text",
            "text": {"content": "Bot 启动完毕，等待接收指令"},
        }
        resp = requests.post(DINGTALK_WEBHOOK, json=payload, timeout=5)
        if resp.ok:
            print("启动通知已发送")
        else:
            print(f"启动通知发送失败: {resp.text}")
    except Exception as e:
        print(f"启动通知发送异常: {e}")


def main():
    if APP_KEY is None or APP_SECRET is None:
        print("错误: 请在环境变量中设置 APP_KEY 和 APP_SECRET")
        return
    credential = Credential(APP_KEY, APP_SECRET)
    client = DingTalkStreamClient(credential)
    # ChatbotMessage.TOPIC 就是 "/v1.0/im/bot/messages/get"
    client.register_callback_handler(dts.ChatbotMessage.TOPIC, SimpleHandler())
    _notify_startup()
    print("启动中，等待消息...")
    client.start_forever()


if __name__ == "__main__":
    main()
