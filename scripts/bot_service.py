import sys, os
import logging
import subprocess
import shlex  # 用于智能拆分字符串，处理包含引号的路径等复杂情况
import dingtalk_stream as dts   # type: ignore
from dingtalk_stream import DingTalkStreamClient, Credential, AckMessage    # type: ignore

# 1. 填入你的凭证
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")

logging.basicConfig(level=logging.INFO)

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
    credential = Credential(APP_KEY, APP_SECRET)
    client = DingTalkStreamClient(credential)
    # ChatbotMessage.TOPIC 就是 "/v1.0/im/bot/messages/get"
    client.register_callback_handler(dts.ChatbotMessage.TOPIC, SimpleHandler())
    print("启动中，等待消息...")
    client.start_forever()

if __name__ == "__main__":
    main()