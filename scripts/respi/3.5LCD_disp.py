#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派 3.5寸 LCD 实时持仓显示

适配微雪 3.5inch RPi LCD (C)，480x320 RGB565，通过 /dev/fb1 framebuffer 输出。
每10分钟通过 Tushare realtime_quote 刷新持仓数据。

屏幕布局（480x320）：
  顶部状态栏：更新时间 | 距调仓天数
  主区域：
    市值总额 / 浮盈率
    总资产  / 总盈亏率
    持仓数量 / 年化收益率
  底部：当前日期时间

自动息屏：23:00 - 6:00 写入全黑画面
"""

import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from datetime import time as dt_time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- 路径设置 ----------
project_root = Path(__file__).parent.parent.parent
scripts_dir = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# ---------- 项目日志 ----------
from src.lazybull.common.logger import setup_logger  # noqa: E402
from src.lazybull.common.config import get_config    # noqa: E402

# ---------- 常量 ----------
FB_PATH = "/dev/fb1"
WIDTH, HEIGHT = 480, 320
REFRESH_INTERVAL = 600  # 秒，10分钟

# 颜色定义 (R, G, B)
COLOR_BG = (15, 15, 25)            # 深色背景
COLOR_HEADER_BG = (30, 30, 50)     # 顶栏背景
COLOR_FOOTER_BG = (30, 30, 50)     # 底栏背景
COLOR_TEXT = (220, 220, 220)       # 主文字
COLOR_LABEL = (140, 140, 160)      # 标签文字（灰色）
COLOR_GREEN = (50, 205, 50)        # 涨 / 正收益
COLOR_RED = (220, 50, 50)          # 跌 / 负收益
COLOR_YELLOW = (255, 200, 50)      # 强调色
COLOR_DIVIDER = (60, 60, 80)       # 分隔线


# ---------- 字体加载 ----------

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """加载中文字体，按优先级尝试多个路径。"""
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",       # 文泉驿正黑
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",     # 文泉驿微米黑
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",    # 备用（无中文）
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------- 格式化工具 ----------

def _fmt_wan(value: float) -> str:
    """将元值转换为万元字符串，如 48.5万 / 123万。"""
    wan = value / 10000.0
    abs_wan = abs(wan)
    if abs_wan >= 100:
        return f"{wan:.0f}万"
    elif abs_wan >= 10:
        return f"{wan:.1f}万"
    else:
        return f"{wan:.2f}万"


def _fmt_pct(value: float) -> str:
    """格式化百分比，带正负号，如 +1.5% / -2.3%。"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _pct_color(value: float) -> tuple:
    """根据盈亏正负返回颜色。"""
    if value > 0:
        return COLOR_GREEN
    elif value < 0:
        return COLOR_RED
    return COLOR_TEXT


# ---------- framebuffer 输出 ----------

def _write_fb(img: Image.Image) -> None:
    """将 PIL Image 转为 RGB565 并写入 framebuffer。"""
    img_array = np.array(img).astype(np.uint16)
    r = (img_array[:, :, 0] >> 3) << 11
    g = (img_array[:, :, 1] >> 2) << 5
    b = img_array[:, :, 2] >> 3
    rgb565 = r | g | b
    try:
        with open(FB_PATH, "wb") as f:
            f.write(rgb565.tobytes())
    except Exception:
        pass


def _clear_screen() -> None:
    """写入全黑画面（息屏用）。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    _write_fb(img)


# ---------- 调仓日计算 ----------

def _calc_days_to_rebalance() -> int | None:
    """计算距下次调仓还剩多少交易日。"""
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


# ---------- 市场时间判断 ----------

def _is_market_open() -> bool:
    """判断当前是否在交易时段。"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False

    current_time = now.time()
    morning_start = dt_time(9, 15)
    morning_end = dt_time(11, 45)
    afternoon_start = dt_time(12, 45)
    afternoon_end = dt_time(15, 15)

    return (morning_start <= current_time <= morning_end
            or afternoon_start <= current_time <= afternoon_end)


# ---------- 渲染逻辑 ----------

def _render(summary: dict | None, last_update_time: str,
            days_to_rebalance: int | None) -> None:
    """将持仓摘要渲染到 PIL Image 并写入 framebuffer。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(img)

    font_lg = _load_font(38)   # 大数字
    font_md = _load_font(24)   # 中号
    font_sm = _load_font(18)   # 小号/标签

    # ===== 顶部状态栏 (0~44) =====
    draw.rectangle([0, 0, WIDTH, 44], fill=COLOR_HEADER_BG)

    days_str = "--" if days_to_rebalance is None else f"{days_to_rebalance}天"
    header_left = f"更新: {last_update_time}"
    header_right = f"距调仓: {days_str}"
    draw.text((12, 10), header_left, fill=COLOR_YELLOW, font=font_sm)
    # 右对齐
    bbox = draw.textbbox((0, 0), header_right, font=font_sm)
    rw = bbox[2] - bbox[0]
    draw.text((WIDTH - rw - 12, 10), header_right, fill=COLOR_YELLOW, font=font_sm)

    # ===== 主数据区域 =====
    if summary is None:
        draw.text((WIDTH // 2 - 60, HEIGHT // 2 - 20), "等待数据...",
                  fill=COLOR_LABEL, font=font_md)
        # 底部时间
        _draw_footer(draw, font_sm)
        _write_fb(img)
        return

    pos_count = summary['pos_count']
    mkt_val = summary['market_value']
    total_ast = summary['total_assets']
    flt_pct = summary['float_pnl_pct']
    gain_pct = summary['total_pnl_pct']
    ann_pct = summary['annual_return_pct']

    # 三行数据，每行：标签 + 值(左) / 标签 + 值(右)
    # 行1: y=60, 行2: y=140, 行3: y=220
    rows = [
        {
            'y': 60,
            'left_label': "持仓市值",
            'left_value': _fmt_wan(mkt_val),
            'left_color': COLOR_TEXT,
            'right_label': "浮盈率",
            'right_value': _fmt_pct(flt_pct),
            'right_color': _pct_color(flt_pct),
        },
        {
            'y': 140,
            'left_label': "总资产",
            'left_value': _fmt_wan(total_ast),
            'left_color': COLOR_TEXT,
            'right_label': "总盈亏率",
            'right_value': _fmt_pct(gain_pct),
            'right_color': _pct_color(gain_pct),
        },
        {
            'y': 220,
            'left_label': "持仓数量",
            'left_value': str(pos_count),
            'left_color': COLOR_TEXT,
            'right_label': "年化收益",
            'right_value': _fmt_pct(ann_pct),
            'right_color': _pct_color(ann_pct),
        },
    ]

    mid_x = WIDTH // 2

    for row in rows:
        y = row['y']
        # 分隔线
        draw.line([(20, y - 4), (WIDTH - 20, y - 4)], fill=COLOR_DIVIDER, width=1)

        # 左侧
        draw.text((20, y), row['left_label'], fill=COLOR_LABEL, font=font_sm)
        draw.text((20, y + 24), row['left_value'], fill=row['left_color'], font=font_lg)

        # 右侧
        draw.text((mid_x + 20, y), row['right_label'], fill=COLOR_LABEL, font=font_sm)
        draw.text((mid_x + 20, y + 24), row['right_value'],
                  fill=row['right_color'], font=font_lg)

    # ===== 底部状态栏 =====
    _draw_footer(draw, font_sm)

    _write_fb(img)


def _draw_footer(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> None:
    """绘制底部时间栏。"""
    draw.rectangle([0, HEIGHT - 36, WIDTH, HEIGHT], fill=COLOR_FOOTER_BG)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bbox = draw.textbbox((0, 0), now_str, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT - 30), now_str,
              fill=COLOR_TEXT, font=font)


# ---------- 后台刷新线程 ----------

def _worker(stop_event: threading.Event) -> None:
    """每 REFRESH_INTERVAL 秒获取一次实时行情并更新显示。

    非交易时段（23:00-6:00）自动息屏，恢复后重新点亮。
    数据获取失败时保留上次数据继续显示。
    """
    from paper_trade import get_realtime_portfolio_summary

    is_screen_on = True
    last_summary: dict | None = None
    last_update_time = "--:--"
    last_days_to_rebalance: int | None = None

    while not stop_event.is_set():
        hour = datetime.now().hour

        # ---- 息屏逻辑（23:00 - 6:00）----
        if hour >= 23 or hour < 6:
            if is_screen_on:
                _clear_screen()
                is_screen_on = False
            stop_event.wait(60)
            continue

        if not is_screen_on:
            is_screen_on = True

        # ---- 获取实时数据 ----
        try:
            if _is_market_open():
                summary = get_realtime_portfolio_summary()
                if summary is not None:
                    last_summary = summary
                    last_update_time = datetime.now().strftime("%H:%M")
        except Exception:
            pass

        # ---- 计算距调仓剩余交易日 ----
        try:
            days = _calc_days_to_rebalance()
            if days is not None:
                last_days_to_rebalance = days
        except Exception:
            pass

        # ---- 渲染 ----
        _render(last_summary, last_update_time, last_days_to_rebalance)

        # ---- 等待下次刷新 ----
        stop_event.wait(REFRESH_INTERVAL)


# ---------- 入口 ----------

def main() -> None:
    setup_logger(log_level="WARNING")
    get_config()

    stop_event = threading.Event()

    def _shutdown(sig, frame):  # noqa: ANN001
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    t = threading.Thread(target=_worker, args=(stop_event,), daemon=True)
    t.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        _clear_screen()


if __name__ == '__main__':
    main()
