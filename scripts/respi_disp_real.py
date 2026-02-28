#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派 mini-LED 实时持仓显示

每10分钟通过 Tushare realtime_quote 刷新持仓数据，
将以下6项指标显示在128x64 OLED 屏幕上：
  Pos   - 持仓数量
  MktV  - 持仓市值（万元）
  Tot   - 总资产（万元）
  Flt   - 浮盈率（%）
  Gain  - 总盈亏率（%）
  Ann   - 年化收益率（%）

屏幕分区：
  黄色区（page 0-1，上16px）：当前时间标题（8x16 字体）
  蓝色区（page 2-7，下48px）：6项指标（6x8 字体，每行一项）

自动息屏：23:00 - 6:00 关闭显示
"""

import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime

# ---------- 路径设置 ----------
project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent
driver_dir = project_root / 'src' / 'lazybull' / 'drv' / 'mini_led'

# 项目根目录（供 src.lazybull.* 导入）
sys.path.insert(0, str(project_root))
# scripts 目录（供导入 paper_trade）
sys.path.insert(0, str(scripts_dir))
# 驱动目录（供 device.py 找到 fonts.py）
sys.path.insert(0, str(driver_dir))

# ---------- OLED 驱动 ----------
from device import OledDevice, CMD  # noqa: E402

# ---------- 项目日志 ----------
from src.lazybull.common.logger import setup_logger  # noqa: E402
from src.lazybull.common.config import get_config    # noqa: E402

# ---------- 常量 ----------
REFRESH_INTERVAL = 600  # 秒，10 分钟


# ---------- 格式化工具 ----------

def _fmt_wan(value: float) -> str:
    """将元值转换为万元字符串，如 48.5W / 123W。"""
    wan = value / 10000.0
    abs_wan = abs(wan)
    if abs_wan >= 100:
        return f"{wan:.0f}W"
    elif abs_wan >= 10:
        return f"{wan:.1f}W"
    else:
        return f"{wan:.2f}W"


def _fmt_pct(value: float) -> str:
    """格式化百分比，带正负号，如 +1.5% / -2.3%。"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


# ---------- 显示逻辑 ----------

def _render(oled: OledDevice, summary: dict | None) -> None:
    """将持仓摘要渲染到 OLED 缓冲区并刷新。"""
    oled.clear_buffer()

    # 黄色区（page 0-1）：当前时间，8x16 字体，居中
    now_str = datetime.now().strftime("%H:%M")
    title = f"LB  {now_str}"
    x = max(0, (128 - len(title) * 8) // 2)
    oled.draw_8x16(x, 0, title)

    if summary is None:
        oled.draw_6x8(0, 2, "No data - retry...")
        oled.refresh()
        return

    pos_count  = summary['pos_count']
    mkt_val    = summary['market_value']
    total_ast  = summary['total_assets']
    flt_pct    = summary['float_pnl_pct']
    gain_pct   = summary['total_pnl_pct']
    ann_pct    = summary['annual_return_pct']

    # 蓝色区（page 2-7）：每行一项指标，6x8 字体
    lines = [
        f"Pos:  {pos_count}",
        f"MktV: {_fmt_wan(mkt_val)}",
        f"Tot:  {_fmt_wan(total_ast)}",
        f"Flt:  {_fmt_pct(flt_pct)}",
        f"Gain: {_fmt_pct(gain_pct)}",
        f"Ann:  {_fmt_pct(ann_pct)}",
    ]
    for page_offset, line in enumerate(lines):
        oled.draw_6x8(0, 2 + page_offset, line)

    oled.refresh()


# ---------- 后台刷新线程 ----------

def _worker(oled: OledDevice, stop_event: threading.Event) -> None:
    """每 REFRESH_INTERVAL 秒获取一次实时行情并更新显示。

    非交易时段（23:00-6:00）自动息屏，恢复后重新点亮。
    数据获取失败时保留上次数据继续显示。
    """
    # 延迟导入，避免启动时的重量级依赖影响 OLED 初始化速度
    from paper_trade import get_realtime_portfolio_summary

    is_screen_on = True
    last_summary: dict | None = None

    while not stop_event.is_set():
        hour = datetime.now().hour

        # ---- 息屏逻辑（23:00 - 6:00）----
        if hour >= 23 or hour < 6:
            if is_screen_on:
                oled.write_byte(0xAE, CMD)  # display off
                is_screen_on = False
            stop_event.wait(60)
            continue

        if not is_screen_on:
            oled.write_byte(0xAF, CMD)  # display on
            is_screen_on = True

        # ---- 获取实时数据 ----
        try:
            summary = get_realtime_portfolio_summary()
            if summary is not None:
                last_summary = summary
        except Exception:
            pass  # 保留 last_summary，下次循环重试

        # ---- 渲染 ----
        _render(oled, last_summary)

        # ---- 等待下次刷新 ----
        stop_event.wait(REFRESH_INTERVAL)


# ---------- 入口 ----------

def main() -> None:
    setup_logger(log_level="WARNING")
    get_config()

    oled = OledDevice()
    stop_event = threading.Event()

    def _shutdown(sig, frame):  # noqa: ANN001
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    t = threading.Thread(target=_worker, args=(oled, stop_event), daemon=True)
    t.start()

    # 主线程只做看门狗
    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        oled.close()


if __name__ == '__main__':
    main()
