#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派 mini-LED 实时持仓显示

每10分钟通过 Tushare realtime_quote 刷新持仓数据，
显示在128x64 OLED 屏幕上：

屏幕分区（8x16 字体）：
  黄色区（page 0-1，上16px）：<最后更新时间>/<距下次调仓剩余交易日>
  蓝色区（page 2-7，下48px）：
    第1行：<市值总额>/<浮盈率>
    第2行：<总资产>/<总盈亏率>
    第3行：<持仓数量>/<年化收益率>

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
from device import OledDevice, CMD  # type: ignore # noqa: E402

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


# ---------- 调仓日计算 ----------

def _calc_days_to_rebalance() -> int | None:
    """计算距下次调仓还剩多少交易日。

    Returns:
        剩余交易日数（0 表示今天是调仓日，负数表示已超期），
        None 表示无法计算（无调仓记录或数据加载失败）。
    """
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data import DataLoader, Storage

    rebalance_state = PaperStorage(
        root_path=str(project_root / "data" / "paper")
    ).load_rebalance_state()
    if rebalance_state is None:
        return None

    last_rebalance_date = rebalance_state.get('last_rebalance_date')
    rebalance_freq = rebalance_state.get('rebalance_freq')
    if not last_rebalance_date or not rebalance_freq:
        return None

    try:
        loader = DataLoader(storage=Storage(root_path=str(project_root / "data")))
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            return None

        trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()

        today_str = datetime.now().strftime("%Y%m%d")
        current_date = today_str if today_str in trade_dates else next(
            (d for d in reversed(trade_dates) if d <= today_str), None
        )
        if current_date is None:
            return None

        last_idx = trade_dates.index(last_rebalance_date)
        current_idx = trade_dates.index(current_date)
        return rebalance_freq - (current_idx - last_idx)
    except Exception:
        return None


# ---------- 显示逻辑 ----------

def _render(oled: OledDevice, summary: dict | None, last_update_time: str,
            days_to_rebalance: int | None) -> None:
    """将持仓摘要渲染到 OLED 缓冲区并刷新。"""
    oled.clear_buffer()

    days_str = "--" if days_to_rebalance is None else f"{days_to_rebalance}d"

    # 黄色区（page 0-1）：最后更新时间/距下次调仓剩余交易日，8x16 字体
    oled.draw_8x16(0, 0, f"{last_update_time}/{days_str}")

    if summary is None:
        oled.draw_8x16(0, 2, "--/--")
        oled.draw_8x16(0, 4, "--/--")
        oled.draw_8x16(0, 6, "--/--")
        oled.refresh()
        return

    pos_count = summary['pos_count']
    mkt_val   = summary['market_value']
    total_ast = summary['total_assets']
    flt_pct   = summary['float_pnl_pct']
    gain_pct  = summary['total_pnl_pct']
    ann_pct   = summary['annual_return_pct']

    # 蓝色区（page 2-7）：三行 8x16 字体
    oled.draw_8x16(0, 2, f"{_fmt_wan(mkt_val)}/{_fmt_pct(flt_pct)}")
    oled.draw_8x16(0, 4, f"{_fmt_wan(total_ast)}/{_fmt_pct(gain_pct)}")
    oled.draw_8x16(0, 6, f"{pos_count}/{_fmt_pct(ann_pct)}")

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
    last_update_time = "--:--"
    last_days_to_rebalance: int | None = None

    while not stop_event.is_set():
        hour = datetime.now().hour
        weekday = datetime.now().weekday()

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
        # 仅在工作日的交易时段获取数据，非交易时段保留上次数据继续显示，避免频繁请求失败。
        try:
            if weekday < 5:  # 仅工作日获取数据
                if (hour > 9 and hour < 12) or (hour > 13 and hour < 15): # 仅交易时段获取数据
                    summary = get_realtime_portfolio_summary()
                    if summary is not None:
                        last_summary = summary
                        last_update_time = datetime.now().strftime("%H:%M")
        except Exception:
            pass  # 保留 last_summary，下次循环重试

        # ---- 计算距调仓剩余交易日 ----
        try:
            days = _calc_days_to_rebalance()
            if days is not None:
                last_days_to_rebalance = days
        except Exception:
            pass

        # ---- 渲染 ----
        _render(oled, last_summary, last_update_time, last_days_to_rebalance)

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
