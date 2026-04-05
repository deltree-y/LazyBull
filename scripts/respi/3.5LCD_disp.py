#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派 3.5寸 LCD 实时持仓显示

适配微雪 3.5inch RPi LCD (C)，480x320 RGB565，通过 /dev/fb1 framebuffer 输出。

架构：
  数据线程：每10分钟获取实时行情（启动时立即获取一次，非交易日也能显示最近收盘数据）
  显示线程：每秒刷新画面（底部时间实时更新），每60秒随机偏移显示位置（屏保防烧屏）

屏幕布局（480x320）：
  顶部状态栏：更新时间 | 距调仓天数
  主区域：
    市值总额 / 浮盈率
    总资产  / 总盈亏率
    持仓数量 / 年化收益率
  底部：当前日期 星期 时间（每秒刷新）

自动息屏：23:00 - 6:00 写入全黑画面
"""

import sys
import time
import signal
import random
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
REFRESH_INTERVAL = 600       # 数据刷新间隔（秒），10分钟
SCREENSAVER_RANGE = 10       # 屏保偏移范围（±像素）
SCREENSAVER_INTERVAL = 60    # 屏保偏移更新间隔（秒）

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

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


# ---------- 字体加载（带缓存）----------

_font_cache: dict = {}


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


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """获取字体（缓存，避免每秒重复加载磁盘）。"""
    if size not in _font_cache:
        _font_cache[size] = _load_font(size)
    return _font_cache[size]


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


# ---------- 共享显示状态 ----------

class DisplayState:
    """数据线程与显示线程之间的共享状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.summary: dict | None = None
        self.update_time: str = "--:--"
        self.days_to_rebalance: int | None = None
        # 屏保偏移
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.is_screen_on: bool = True


# ---------- 渲染逻辑 ----------

def _render(state: DisplayState) -> None:
    """将持仓摘要渲染到 PIL Image 并写入 framebuffer。

    所有绘制坐标加上屏保偏移量 (offset_x, offset_y)。
    """
    with state.lock:
        summary = state.summary
        last_update_time = state.update_time
        days_to_rebalance = state.days_to_rebalance
        ox = state.offset_x
        oy = state.offset_y

    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(img)

    font_lg = _get_font(38)   # 大数字
    font_md = _get_font(24)   # 中号
    font_sm = _get_font(18)   # 小号/标签

    # ===== 顶部状态栏 =====
    header_y = oy
    draw.rectangle([0, header_y, WIDTH, header_y + 44], fill=COLOR_HEADER_BG)

    days_str = "--" if days_to_rebalance is None else f"{days_to_rebalance}天"
    header_left = f"更新: {last_update_time}"
    header_right = f"距调仓: {days_str}"
    draw.text((12 + ox, header_y + 10), header_left,
              fill=COLOR_YELLOW, font=font_sm)
    bbox = draw.textbbox((0, 0), header_right, font=font_sm)
    rw = bbox[2] - bbox[0]
    draw.text((WIDTH - rw - 12 + ox, header_y + 10), header_right,
              fill=COLOR_YELLOW, font=font_sm)

    # ===== 主数据区域 =====
    if summary is None:
        bbox_wait = draw.textbbox((0, 0), "等待数据...", font=font_md)
        ww = bbox_wait[2] - bbox_wait[0]
        draw.text(((WIDTH - ww) // 2 + ox, HEIGHT // 2 - 20 + oy), "等待数据...",
                  fill=COLOR_LABEL, font=font_md)
        _draw_footer(draw, font_sm)
        _write_fb(img)
        return

    pos_count = summary['pos_count']
    mkt_val = summary['market_value']
    total_ast = summary['total_assets']
    flt_pct = summary['float_pnl_pct']
    gain_pct = summary['total_pnl_pct']
    ann_pct = summary['annual_return_pct']

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
        y = row['y'] + oy
        # 分隔线
        draw.line([(20 + ox, y - 4), (WIDTH - 20 + ox, y - 4)],
                  fill=COLOR_DIVIDER, width=1)
        # 左侧
        draw.text((20 + ox, y), row['left_label'],
                  fill=COLOR_LABEL, font=font_sm)
        draw.text((20 + ox, y + 24), row['left_value'],
                  fill=row['left_color'], font=font_lg)
        # 右侧
        draw.text((mid_x + 20 + ox, y), row['right_label'],
                  fill=COLOR_LABEL, font=font_sm)
        draw.text((mid_x + 20 + ox, y + 24), row['right_value'],
                  fill=row['right_color'], font=font_lg)

    # ===== 底部时间栏（固定位置，不参与屏保偏移）=====
    _draw_footer(draw, font_sm)

    _write_fb(img)


def _draw_footer(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> None:
    """绘制底部时间栏（含星期几，固定在屏幕底部）。"""
    draw.rectangle([0, HEIGHT - 36, WIDTH, HEIGHT], fill=COLOR_FOOTER_BG)
    now = datetime.now()
    weekday = WEEKDAY_NAMES[now.weekday()]
    now_str = now.strftime(f"%Y-%m-%d {weekday} %H:%M:%S")
    bbox = draw.textbbox((0, 0), now_str, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT - 30), now_str,
              fill=COLOR_TEXT, font=font)


# ---------- 数据获取线程 ----------

def _data_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """每 REFRESH_INTERVAL 秒获取一次实时行情，更新共享状态。

    启动时立即获取一次（非交易日也会返回最近一个交易日的收盘数据）。
    """
    from paper_trade import get_realtime_portfolio_summary

    def _fetch_data() -> None:
        """获取行情和调仓天数，更新共享状态。"""
        try:
            summary = get_realtime_portfolio_summary()
            if summary is not None:
                with state.lock:
                    state.summary = summary
                    state.update_time = datetime.now().strftime("%H:%M")
        except Exception:
            pass

        try:
            days = _calc_days_to_rebalance()
            if days is not None:
                with state.lock:
                    state.days_to_rebalance = days
        except Exception:
            pass

    # 启动时立即获取一次（非交易日也能显示最近收盘数据）
    _fetch_data()

    while not stop_event.is_set():
        stop_event.wait(REFRESH_INTERVAL)
        if stop_event.is_set():
            break

        # 仅交易时段刷新实时数据
        if _is_market_open():
            _fetch_data()


# ---------- 显示刷新线程 ----------

def _display_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """每秒刷新画面，每 SCREENSAVER_INTERVAL 秒更新屏保偏移。

    23:00-6:00 自动息屏。
    """
    last_offset_time = 0.0

    while not stop_event.is_set():
        hour = datetime.now().hour

        # ---- 息屏逻辑（23:00 - 6:00）----
        if hour >= 23 or hour < 6:
            if state.is_screen_on:
                _clear_screen()
                state.is_screen_on = False
            stop_event.wait(10)
            continue

        if not state.is_screen_on:
            state.is_screen_on = True

        # ---- 屏保：每分钟随机偏移 ----
        now_ts = time.monotonic()
        if now_ts - last_offset_time >= SCREENSAVER_INTERVAL:
            with state.lock:
                state.offset_x = random.randint(-SCREENSAVER_RANGE, SCREENSAVER_RANGE)
                state.offset_y = random.randint(-SCREENSAVER_RANGE, SCREENSAVER_RANGE)
            last_offset_time = now_ts

        # ---- 渲染（含实时时间）----
        _render(state)

        # ---- 每秒刷新 ----
        stop_event.wait(1)


# ---------- 入口 ----------

def main() -> None:
    setup_logger(log_level="WARNING")
    get_config()

    state = DisplayState()
    stop_event = threading.Event()

    def _shutdown(sig, frame):  # noqa: ANN001
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 数据获取线程（10分钟间隔）
    data_t = threading.Thread(target=_data_worker, args=(state, stop_event), daemon=True)
    data_t.start()

    # 显示刷新线程（每秒）
    disp_t = threading.Thread(target=_display_worker, args=(state, stop_event), daemon=True)
    disp_t.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        _clear_screen()


if __name__ == '__main__':
    main()
