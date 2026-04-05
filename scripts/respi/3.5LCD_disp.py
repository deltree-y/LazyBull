#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派 3.5寸 LCD 实时持仓显示

适配微雪 3.5inch RPi LCD (C)，480x320 RGB565，通过 /dev/fb1 framebuffer 输出。

架构：
  数据线程：每10分钟获取实时行情+图表数据（启动时立即获取一次）
  显示线程：每秒刷新画面（底部时间实时更新），每60秒随机偏移数据区（屏保防烧屏）

屏幕布局（480x320）：
  顶部状态栏（固定）：更新时间 | 距调仓天数
  数据区（屏保偏移）：
    市值 / 浮盈率 | 总资产 / 总盈亏率 | 持仓 / 年化收益
  图表区（固定）：持仓周期内上证指数 vs 持仓组合涨跌幅
  底部时间栏（固定）：日期 星期 时间（每秒刷新）

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
from typing import Optional

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
SCREENSAVER_RANGE_X = 10     # 屏保水平偏移范围（±像素）
SCREENSAVER_RANGE_Y = 6      # 屏保垂直偏移范围（±像素）
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
COLOR_CHART_BG = (22, 22, 38)      # 图表背景
COLOR_CHART_GRID = (45, 45, 65)    # 图表网格线


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
    """将元值转换为万元字符串，如 48.5 / 123。"""
    wan = value / 10000.0
    abs_wan = abs(wan)
    if abs_wan >= 100:
        return f"{wan:.0f}"
    elif abs_wan >= 10:
        return f"{wan:.1f}"
    else:
        return f"{wan:.2f}"


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


# ---------- 图表数据获取 ----------

def _fetch_chart_data() -> Optional[dict]:
    """获取持仓周期内的上证指数和持仓组合涨跌幅数据。

    Returns:
        dict: {
            'dates': list[str],          # 交易日期列表
            'index_pct': list[float],    # 上证指数累计涨跌幅(%)
            'portfolio_pct': list[float] # 持仓组合累计涨跌幅(%)
        }
        None: 数据不可用
    """
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data.tushare_client import TushareClient

    paper_storage = PaperStorage(
        root_path=str(project_root / "data" / "paper"), verbose=False
    )

    # 获取上次调仓日期作为周期起点
    rebalance_state = paper_storage.load_rebalance_state()
    if rebalance_state is None:
        return None
    start_date = rebalance_state.get('last_rebalance_date')
    if not start_date:
        return None

    today_str = datetime.now().strftime("%Y%m%d")

    # 持仓组合净值
    nav_df = paper_storage.load_all_nav()
    if nav_df is None or nav_df.empty:
        return None
    nav_df = nav_df[nav_df['trade_date'] >= start_date].sort_values('trade_date')
    if len(nav_df) < 2:
        return None

    base_nav = nav_df.iloc[0]['total_value']
    if base_nav <= 0:
        return None
    portfolio_pct = ((nav_df['total_value'] / base_nav - 1) * 100).tolist()
    dates = nav_df['trade_date'].tolist()

    # 上证指数日线
    try:
        client = TushareClient(verbose=False)
        index_df = client.query(
            "index_daily", ts_code="000001.SH",
            start_date=start_date, end_date=today_str,
            fields="trade_date,close"
        )
        if index_df is None or index_df.empty:
            return None
        index_map = dict(zip(index_df['trade_date'], index_df['close']))
    except Exception:
        return None

    # 按持仓净值的日期序列对齐上证数据
    base_close = index_map.get(dates[0])
    if base_close is None or base_close <= 0:
        return None

    index_pct = []
    last_pct = 0.0
    for d in dates:
        close = index_map.get(d)
        if close is not None:
            last_pct = (close / base_close - 1) * 100
        index_pct.append(last_pct)

    return {
        'dates': dates,
        'index_pct': index_pct,
        'portfolio_pct': portfolio_pct,
    }


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
        self.chart_data: dict | None = None
        # 屏保偏移（仅数据行参与）
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.is_screen_on: bool = True


# ---------- 渲染逻辑 ----------

# 布局常量
HEADER_H = 42          # 顶栏高度
FOOTER_H = 38          # 底栏高度
DATA_ROW_Y = [50, 88, 126]  # 3行数据的基准 y（参与屏保偏移）
CHART_Y = 178          # 图表区起始 y（固定）
CHART_H = HEIGHT - FOOTER_H - CHART_Y  # 图表区高度


def _render(state: DisplayState) -> None:
    """将持仓摘要和图表渲染到 PIL Image 并写入 framebuffer。

    顶部状态栏、图表区和底部时间栏固定不动，仅数据行参与屏保偏移。
    """
    with state.lock:
        summary = state.summary
        last_update_time = state.update_time
        days_to_rebalance = state.days_to_rebalance
        chart_data = state.chart_data
        ox = state.offset_x
        oy = state.offset_y

    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(img)

    font_val = _get_font(24)   # 数值
    font_sm = _get_font(14)    # 标签
    font_md = _get_font(20)    # 中号（等待提示用）
    font_footer = _get_font(18)

    # ===== 顶部状态栏（固定）=====
    draw.rectangle([0, 0, WIDTH, HEADER_H], fill=COLOR_HEADER_BG)

    days_str = "--" if days_to_rebalance is None else f"{days_to_rebalance}天"
    header_left = f"更新: {last_update_time}"
    header_right = f"距调仓: {days_str}"
    draw.text((12, 11), header_left, fill=COLOR_YELLOW, font=font_sm)
    bbox = draw.textbbox((0, 0), header_right, font=font_sm)
    rw = bbox[2] - bbox[0]
    draw.text((WIDTH - rw - 12, 11), header_right, fill=COLOR_YELLOW, font=font_sm)

    # ===== 数据区（参与屏保偏移）=====
    # 行基准 y: 50, 88, 126，每行高度约38px（标签14px + 值24px）
    # 最大底部 = 126+16+28 = 170，偏移 ±6 后范围 164~176，图表起始178，安全
    if summary is None:
        bbox_wait = draw.textbbox((0, 0), "等待数据...", font=font_md)
        ww = bbox_wait[2] - bbox_wait[0]
        draw.text(((WIDTH - ww) // 2 + ox, 100 + oy), "等待数据...",
                  fill=COLOR_LABEL, font=font_md)
        _draw_chart(draw, chart_data)
        _draw_footer(draw, font_footer)
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
            'y': DATA_ROW_Y[0],
            'left_label': "持仓市值(万)",
            'left_value': _fmt_wan(mkt_val),
            'left_color': COLOR_TEXT,
            'right_label': "浮盈率",
            'right_value': _fmt_pct(flt_pct),
            'right_color': _pct_color(flt_pct),
        },
        {
            'y': DATA_ROW_Y[1],
            'left_label': "总资产(万)",
            'left_value': _fmt_wan(total_ast),
            'left_color': COLOR_TEXT,
            'right_label': "总盈亏率",
            'right_value': _fmt_pct(gain_pct),
            'right_color': _pct_color(gain_pct),
        },
        {
            'y': DATA_ROW_Y[2],
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
        draw.line([(20 + ox, y - 2), (WIDTH - 20 + ox, y - 2)],
                  fill=COLOR_DIVIDER, width=1)
        # 左侧
        draw.text((20 + ox, y), row['left_label'],
                  fill=COLOR_LABEL, font=font_sm)
        draw.text((20 + ox, y + 16), row['left_value'],
                  fill=row['left_color'], font=font_val)
        # 右侧
        draw.text((mid_x + 20 + ox, y), row['right_label'],
                  fill=COLOR_LABEL, font=font_sm)
        draw.text((mid_x + 20 + ox, y + 16), row['right_value'],
                  fill=row['right_color'], font=font_val)

    # ===== 图表区（固定）=====
    _draw_chart(draw, chart_data)

    # ===== 底部时间栏（固定）=====
    _draw_footer(draw, font_footer)

    _write_fb(img)


def _draw_chart(draw: ImageDraw.ImageDraw, chart_data: Optional[dict]) -> None:
    """绘制持仓周期涨跌幅对比折线图。"""
    chart_x = 10
    chart_w = WIDTH - 20
    font_xs = _get_font(11)

    # 图表区背景
    draw.rectangle([chart_x, CHART_Y, chart_x + chart_w, CHART_Y + CHART_H],
                   fill=COLOR_CHART_BG)

    if not chart_data or len(chart_data.get('dates', [])) < 2:
        # 无数据提示
        txt = "暂无图表数据"
        bbox = draw.textbbox((0, 0), txt, font=_get_font(14))
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, CHART_Y + CHART_H // 2 - 8), txt,
                  fill=COLOR_LABEL, font=_get_font(14))
        return

    dates = chart_data['dates']
    idx_pct = chart_data['index_pct']
    ptf_pct = chart_data['portfolio_pct']
    n = len(dates)

    # Y轴范围
    all_vals = idx_pct + ptf_pct
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_margin = max((y_max - y_min) * 0.15, 0.5)
    y_min -= y_margin
    y_max += y_margin
    y_range = y_max - y_min
    if y_range < 0.01:
        y_range = 1.0

    # 内部绘图区域
    label_w = 44       # Y轴标签空间
    legend_h = 16      # 顶部图例高度
    bottom_pad = 4
    cx = chart_x + label_w
    cy = CHART_Y + legend_h
    cw = chart_w - label_w - 6
    ch = CHART_H - legend_h - bottom_pad

    # 绘制边框
    draw.rectangle([cx, cy, cx + cw, cy + ch], outline=COLOR_DIVIDER)

    # 零线
    if y_min < 0 < y_max:
        zero_py = cy + ch - int((0 - y_min) / y_range * ch)
        draw.line([(cx + 1, zero_py), (cx + cw - 1, zero_py)],
                  fill=COLOR_DIVIDER, width=1)

    # 水平网格（3条）
    for i in range(1, 4):
        gy = cy + ch * i // 4
        draw.line([(cx + 1, gy), (cx + cw - 1, gy)],
                  fill=COLOR_CHART_GRID, width=1)

    # Y轴标签（上/中/下）
    for val, align_top in [(y_max, True), ((y_max + y_min) / 2, False), (y_min, False)]:
        py = cy + ch - int((val - y_min) / y_range * ch)
        label = f"{val:+.1f}%"
        ty = py - 12 if align_top else py - 6
        draw.text((chart_x + 1, ty), label, fill=COLOR_LABEL, font=font_xs)

    # 绘制折线
    def _to_points(values):
        pts = []
        for i, v in enumerate(values):
            px = cx + int(i / max(n - 1, 1) * cw)
            py = cy + ch - int((v - y_min) / y_range * ch)
            pts.append((px, py))
        return pts

    idx_pts = _to_points(idx_pct)
    ptf_pts = _to_points(ptf_pct)

    for i in range(n - 1):
        draw.line([idx_pts[i], idx_pts[i + 1]], fill=COLOR_YELLOW, width=2)
    for i in range(n - 1):
        draw.line([ptf_pts[i], ptf_pts[i + 1]], fill=COLOR_GREEN, width=2)

    # 图例 + 末尾数值
    lx = cx + 6
    ly = CHART_Y + 2
    # 上证
    draw.line([(lx, ly + 6), (lx + 14, ly + 6)], fill=COLOR_YELLOW, width=2)
    draw.text((lx + 18, ly), "上证", fill=COLOR_YELLOW, font=font_xs)
    idx_last_str = f"{idx_pct[-1]:+.1f}%"
    draw.text((lx + 50, ly), idx_last_str, fill=COLOR_YELLOW, font=font_xs)
    # 持仓
    sx = lx + 100
    draw.line([(sx, ly + 6), (sx + 14, ly + 6)], fill=COLOR_GREEN, width=2)
    draw.text((sx + 18, ly), "持仓", fill=COLOR_GREEN, font=font_xs)
    ptf_last_str = f"{ptf_pct[-1]:+.1f}%"
    draw.text((sx + 50, ly), ptf_last_str, fill=COLOR_GREEN, font=font_xs)

    # X轴：起止日期
    start_label = f"{dates[0][4:6]}/{dates[0][6:]}"
    end_label = f"{dates[-1][4:6]}/{dates[-1][6:]}"
    draw.text((cx + 2, cy + ch + 1), start_label, fill=COLOR_LABEL, font=font_xs)
    bbox_end = draw.textbbox((0, 0), end_label, font=font_xs)
    ew = bbox_end[2] - bbox_end[0]
    draw.text((cx + cw - ew - 2, cy + ch + 1), end_label,
              fill=COLOR_LABEL, font=font_xs)


def _draw_footer(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> None:
    """绘制底部时间栏（含星期几，固定在屏幕底部）。"""
    footer_y = HEIGHT - FOOTER_H
    draw.rectangle([0, footer_y, WIDTH, HEIGHT], fill=COLOR_FOOTER_BG)
    now = datetime.now()
    weekday = WEEKDAY_NAMES[now.weekday()]
    now_str = now.strftime(f"%Y-%m-%d {weekday} %H:%M:%S")
    bbox = draw.textbbox((0, 0), now_str, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((WIDTH - tw) // 2, footer_y + (FOOTER_H - th) // 2), now_str,
              fill=COLOR_TEXT, font=font)


# ---------- 数据获取线程 ----------

def _data_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """每 REFRESH_INTERVAL 秒获取一次实时行情和图表数据，更新共享状态。

    启动时立即获取一次（非交易日也会返回最近一个交易日的收盘数据）。
    """
    from paper_trade import get_realtime_portfolio_summary

    def _fetch_data() -> None:
        """获取行情、调仓天数和图表数据，更新共享状态。"""
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

        try:
            cd = _fetch_chart_data()
            if cd is not None:
                with state.lock:
                    state.chart_data = cd
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

        # ---- 屏保：每分钟随机偏移数据区 ----
        now_ts = time.monotonic()
        if now_ts - last_offset_time >= SCREENSAVER_INTERVAL:
            with state.lock:
                state.offset_x = random.randint(-SCREENSAVER_RANGE_X, SCREENSAVER_RANGE_X)
                state.offset_y = random.randint(-SCREENSAVER_RANGE_Y, SCREENSAVER_RANGE_Y)
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
