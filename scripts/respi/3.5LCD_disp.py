#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
树莓派 3.5寸 LCD 实时持仓显示

适配微雪 3.5inch RPi LCD (C)，480x320 RGB565，通过 /dev/fb1 framebuffer 输出。

架构：
    数据线程：盘中每2分钟刷新摘要/排行/日内图，周期图与非交易时段补数按10分钟按需获取（启动时立即获取一次）
    显示线程：每秒刷新画面（底部时间实时更新，顶部 CPU/内存双血条每2秒采样一次），
                            每60秒随机偏移数据区（屏保防烧屏）

屏幕布局（480x320）：
    顶部状态栏（固定）：更新时间 | 距调仓天数
    顶栏底部占用一整行：5px 左右的双血条（左 CPU，占右内存）
        数据区（屏保偏移）：
    市值 / 浮盈率 | 总资产 / 总盈亏率 | 持仓 / 年化收益
    图表区（固定）：持仓周期内上证/深证指数 vs 持仓组合涨跌幅
        底部时间栏（固定）：日期 星期 时间（每秒刷新）

自动息屏：23:00 - 6:00 写入全黑画面
"""

import sys
import time
import signal
import random
import threading
import json
import tempfile
import os
import copy
from pathlib import Path
from datetime import datetime, timedelta
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
from src.lazybull.common.config import get_config, get_data_root, get_paper_root    # noqa: E402
from respi.set_backlight import cleanup_backlight_state as _cleanup_backlight_state_helper  # noqa: E402
from respi.set_backlight import get_pwm_hardware_note as _get_pwm_hardware_note_helper  # noqa: E402
from respi.set_backlight import set_backlight as _set_backlight_helper  # noqa: E402
from respi.set_backlight import update_pwm_backlight_state as _update_pwm_backlight_state_helper  # noqa: E402

# ---------- 常量 ----------
DEFAULT_FB_PATH = "/dev/fb1"
WIDTH, HEIGHT = 480, 320
REFRESH_INTERVAL = 600       # 周期图/非交易时段补数间隔（秒），10分钟
REALTIME_REFRESH_INTERVAL = 90  # 盘中摘要/排行/日内图刷新间隔（秒）
USAGE_REFRESH_INTERVAL = 2.0  # 顶部 CPU/内存双血条采样间隔（秒）
CPU_USAGE_STALE_RESET_SECONDS = 10.0  # 息屏恢复后避免拿超长时间窗平均值
MORNING_CLOSE_INTRADAY_GRACE_SECONDS = 120  # 午休前补齐 11:30 最后一格的宽限时长（秒）
POST_CLOSE_INTRADAY_GRACE_SECONDS = 600  # 收盘后继续补齐日内尾点的宽限时长（秒）
BACKLIGHT_PIN = 18           # 背光 GPIO 引脚（硬件 PWM）
BACKLIGHT_BRIGHTNESS = 10    # 背光亮度 0~100（默认40%，可按需调整）
SCREENSAVER_RANGE_X = 4      # 屏保水平偏移范围（±像素）
SCREENSAVER_RANGE_Y = 3      # 屏保垂直偏移范围（±像素）
SCREENSAVER_INTERVAL = 60    # 屏保偏移更新间隔（秒）
A_SHARE_MORNING_OPEN = dt_time(9, 30)
A_SHARE_MORNING_CLOSE = dt_time(11, 30)
A_SHARE_AFTERNOON_OPEN = dt_time(13, 0)
A_SHARE_AFTERNOON_CLOSE = dt_time(15, 0)
INTRADAY_WINDOW_START = A_SHARE_MORNING_OPEN
INTRADAY_WINDOW_END = A_SHARE_AFTERNOON_CLOSE
INTRADAY_SLOT_MINUTES = 10
INTRADAY_MORNING_SLOT_COUNT = (
    ((A_SHARE_MORNING_CLOSE.hour * 60 + A_SHARE_MORNING_CLOSE.minute)
     - (A_SHARE_MORNING_OPEN.hour * 60 + A_SHARE_MORNING_OPEN.minute))
    // INTRADAY_SLOT_MINUTES
    + 1
)
INTRADAY_AFTERNOON_SLOT_COUNT = (
    ((A_SHARE_AFTERNOON_CLOSE.hour * 60 + A_SHARE_AFTERNOON_CLOSE.minute)
     - (A_SHARE_AFTERNOON_OPEN.hour * 60 + A_SHARE_AFTERNOON_OPEN.minute))
    // INTRADAY_SLOT_MINUTES
    + 1
)
INTRADAY_SLOT_COUNT = INTRADAY_MORNING_SLOT_COUNT + INTRADAY_AFTERNOON_SLOT_COUNT
INTRADAY_INDEX_PCT_ABS_LIMIT = 20.0
INTRADAY_PORTFOLIO_PCT_ABS_LIMIT = 35.0
INTRADAY_STOCK_PCT_ABS_LIMIT = 35.0
SHANGHAI_INDEX_CODE = "000001.SH"
SHENZHEN_INDEX_CODE = "399001.SZ"
CSI800_INDEX_CODE = "000906.SH"
INTRADAY_CHART_STATE_DIRNAME = "respi_35lcd_intraday"
DIAG_LOG_FILENAME = "respi_35lcd_runtime.log"

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 颜色定义 (R, G, B)
COLOR_BG = (15, 15, 25)            # 深色背景
COLOR_HEADER_BG = (30, 30, 50)     # 顶栏背景
COLOR_FOOTER_BG = (30, 30, 50)     # 底栏背景
COLOR_TEXT = (220, 220, 220)       # 主文字
COLOR_LABEL = (140, 140, 160)      # 标签文字（灰色）
COLOR_GREEN = (50, 205, 50)        # 涨 / 正收益
COLOR_RED = (220, 50, 50)          # 跌 / 负收益
COLOR_YELLOW = (170, 130, 255)     # 强调色（上证折线）
COLOR_ORANGE = (255, 150, 60)      # 橘黄色（持仓折线）
COLOR_CYAN = (70, 205, 255)        # 青蓝色（深证折线）
COLOR_DIVIDER = (60, 60, 80)       # 分隔线
COLOR_CHART_BG = (22, 22, 38)      # 图表背景
COLOR_CHART_GRID = (45, 45, 65)    # 图表网格线
COLOR_CHART_BREAK = (78, 88, 108)  # 午休分隔标记
COLOR_CHART_ZERO_LINE = (140, 135, 120)  # 0%参考线
COLOR_PANEL_LEFT = (25, 28, 48)    # 左面板背景（偏蓝）
COLOR_PANEL_RIGHT = (28, 35, 38)   # 右面板背景（偏青）
COLOR_CHART_SHANGHAI = COLOR_YELLOW
COLOR_CHART_SHENZHEN = COLOR_CYAN
COLOR_CHART_HOLDINGS = COLOR_ORANGE
COLOR_CHART_CSI800 = (245, 245, 245)
ZERO_LINE_Y_OFFSET = 1
COLOR_USAGE_BAR_OUTLINE = (132, 140, 168)
COLOR_USAGE_BAR_EMPTY = (55, 58, 82)
COLOR_USAGE_BAR_LOW = (75, 205, 105)
COLOR_USAGE_BAR_MID = (240, 190, 82)
COLOR_USAGE_BAR_HIGH = (225, 95, 95)

_diag_lock = threading.Lock()
_diag_once_keys: set[str] = set()


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


def _pick_fitting_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    preferred_size: int,
    min_size: int,
    max_width: int,
) -> ImageFont.FreeTypeFont:
    """按最大可用宽度选择字号，优先保留较大的字体。"""
    if max_width <= 0:
        return _get_font(preferred_size)

    for size in range(preferred_size, min_size - 1, -1):
        font = _get_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return font

    return _get_font(min_size)


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
    """根据盈亏正负返回颜色（A股惯例：涨红跌绿）。"""
    if value > 0:
        return COLOR_RED
    elif value < 0:
        return COLOR_GREEN
    return COLOR_TEXT


def _read_cpu_stat_sample() -> Optional[tuple[int, int]]:
    """读取 /proc/stat 的总 tick 与 idle tick。"""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return None

    parts = first_line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None

    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None

    total_ticks = sum(values)
    idle_ticks = values[3] + (values[4] if len(values) > 4 else 0)
    return total_ticks, idle_ticks


def _calc_cpu_usage_pct(
    previous_sample: tuple[int, int],
    current_sample: tuple[int, int],
) -> Optional[float]:
    """基于两次 /proc/stat 采样计算 CPU 占用率。"""
    prev_total, prev_idle = previous_sample
    curr_total, curr_idle = current_sample
    total_delta = curr_total - prev_total
    idle_delta = curr_idle - prev_idle
    if total_delta <= 0 or idle_delta < 0:
        return None

    usage_pct = (total_delta - idle_delta) / total_delta * 100.0
    return max(0.0, min(100.0, usage_pct))


def _read_memory_usage_pct() -> Optional[float]:
    """读取 /proc/meminfo 并计算当前内存占用率。"""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                try:
                    values[key] = int(parts[1])
                except ValueError:
                    continue
    except OSError:
        return None

    total_kb = values.get("MemTotal")
    if not total_kb or total_kb <= 0:
        return None

    available_kb = values.get("MemAvailable")
    if available_kb is None:
        available_kb = (
            values.get("MemFree", 0)
            + values.get("Buffers", 0)
            + values.get("Cached", 0)
            + values.get("SReclaimable", 0)
            - values.get("Shmem", 0)
        )

    available_kb = max(0, min(total_kb, available_kb))
    used_kb = total_kb - available_kb
    usage_pct = used_kb / total_kb * 100.0
    return max(0.0, min(100.0, usage_pct))


def _refresh_system_usage_sample(
    state: "DisplayState",
    now_ts: Optional[float] = None,
) -> tuple[float, float]:
    """按固定节流周期刷新 CPU 与内存占用率缓存。"""
    sample_ts = now_ts if now_ts is not None else time.monotonic()

    with state.lock:
        previous_sample = getattr(state, "cpu_usage_sample", None)
        previous_sampled_at = getattr(state, "usage_sampled_at", 0.0)
        previous_cpu_usage_pct = float(getattr(state, "cpu_usage_pct", 0.0) or 0.0)
        previous_memory_usage_pct = float(getattr(state, "memory_usage_pct", 0.0) or 0.0)

    if (
        previous_sample is not None
        and sample_ts - previous_sampled_at < USAGE_REFRESH_INTERVAL
    ):
        return previous_cpu_usage_pct, previous_memory_usage_pct

    current_sample = _read_cpu_stat_sample()
    current_memory_usage_pct = _read_memory_usage_pct()
    if current_sample is None:
        if current_memory_usage_pct is None:
            return previous_cpu_usage_pct, previous_memory_usage_pct

    cpu_usage_pct = previous_cpu_usage_pct
    memory_usage_pct = previous_memory_usage_pct
    if current_memory_usage_pct is not None:
        memory_usage_pct = current_memory_usage_pct

    elapsed_seconds = sample_ts - previous_sampled_at
    if (
        previous_sample is not None
        and current_sample is not None
        and 0.0 < elapsed_seconds <= CPU_USAGE_STALE_RESET_SECONDS
    ):
        computed_usage_pct = _calc_cpu_usage_pct(previous_sample, current_sample)
        if computed_usage_pct is not None:
            cpu_usage_pct = computed_usage_pct

    with state.lock:
        if current_sample is not None:
            state.cpu_usage_sample = current_sample
        if current_sample is not None or current_memory_usage_pct is not None:
            state.usage_sampled_at = sample_ts
        state.cpu_usage_pct = cpu_usage_pct
        state.memory_usage_pct = memory_usage_pct

    return cpu_usage_pct, memory_usage_pct


def _get_usage_bar_fill_color(usage_pct: float) -> tuple[int, int, int]:
    """根据占用率返回绿黄红分段颜色。"""
    if usage_pct >= 80.0:
        return COLOR_USAGE_BAR_HIGH
    if usage_pct >= 55.0:
        return COLOR_USAGE_BAR_MID
    return COLOR_USAGE_BAR_LOW


def _draw_usage_bar_section(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    usage_pct: float,
) -> None:
    """在单个分区中绘制一段占用率填充。"""
    section_width = max(0, x1 - x0 + 1)
    fill_width = int(round(section_width * max(0.0, min(100.0, float(usage_pct))) / 100.0))
    if usage_pct > 0.0 and fill_width == 0:
        fill_width = 1
    if fill_width <= 0:
        return

    fill_x1 = x0 + fill_width - 1
    if fill_width >= 3:
        draw.rounded_rectangle(
            [x0, y0, fill_x1, y1],
            radius=1,
            fill=_get_usage_bar_fill_color(usage_pct),
        )
        return

    draw.rectangle(
        [x0, y0, fill_x1, y1],
        fill=_get_usage_bar_fill_color(usage_pct),
    )


def _draw_system_usage_bar(
    draw: ImageDraw.ImageDraw,
    cpu_usage_pct: float,
    memory_usage_pct: float,
) -> None:
    """在顶栏底部绘制左右双槽血条：左 CPU，右内存。"""
    body_height = USAGE_BAR_H
    bar_y0 = HEADER_H - USAGE_BAR_BOTTOM_GAP - body_height
    bar_y1 = bar_y0 + body_height - 1
    body_x0 = USAGE_BAR_MARGIN_X
    body_x1 = WIDTH - USAGE_BAR_MARGIN_X - USAGE_BAR_CAP_W - 2
    cap_x0 = body_x1 + 2
    cap_y0 = bar_y0 + 1
    cap_x1 = cap_x0 + USAGE_BAR_CAP_W - 1
    cap_y1 = bar_y1 - 1

    draw.rounded_rectangle(
        [body_x0, bar_y0, body_x1, bar_y1],
        radius=2,
        fill=COLOR_USAGE_BAR_EMPTY,
        outline=COLOR_USAGE_BAR_OUTLINE,
    )
    draw.rounded_rectangle(
        [cap_x0, cap_y0, cap_x1, cap_y1],
        radius=1,
        fill=COLOR_USAGE_BAR_EMPTY,
        outline=COLOR_USAGE_BAR_OUTLINE,
    )

    inner_x0 = body_x0 + 2
    inner_y0 = bar_y0 + 1
    inner_x1 = body_x1 - 2
    inner_y1 = bar_y1 - 1
    divider_left = inner_x0 + (inner_x1 - inner_x0 + 1) // 2 - USAGE_BAR_SECTION_GAP // 2
    divider_right = divider_left + USAGE_BAR_SECTION_GAP - 1
    left_x0 = inner_x0
    left_x1 = divider_left - 1
    right_x0 = divider_right + 1
    right_x1 = inner_x1

    draw.line(
        [(divider_left - 1, inner_y0), (divider_left - 1, inner_y1)],
        fill=COLOR_USAGE_BAR_OUTLINE,
    )
    draw.line(
        [(divider_right + 1, inner_y0), (divider_right + 1, inner_y1)],
        fill=COLOR_USAGE_BAR_OUTLINE,
    )

    _draw_usage_bar_section(draw, left_x0, left_x1, inner_y0, inner_y1, cpu_usage_pct)
    _draw_usage_bar_section(draw, right_x0, right_x1, inner_y0, inner_y1, memory_usage_pct)


def _coerce_float(value: object) -> Optional[float]:
    """尽力将值转为 float，失败则返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_mmdd(date_str: str) -> str:
    """将 YYYYMMDD 转为 MM/DD。"""
    return f"{date_str[4:6]}/{date_str[6:]}"


def _format_display_time(now: datetime) -> str:
    """格式化顶部显示时间，如 4月7日(周二) 14:40:32。"""
    weekday = WEEKDAY_NAMES[now.weekday()]
    return f"{now.month}月{now.day}日({weekday}) {now:%H:%M:%S}"


def _format_quote_update_time(summary: Optional[dict]) -> Optional[str]:
    """从摘要中提取顶部“更新:HH:MM”应显示的行情时间。"""
    if not isinstance(summary, dict):
        return None

    quote_time = str(summary.get('quote_time', '')).strip()
    if not quote_time:
        return None

    parts = quote_time.split(':')
    if len(parts) < 2:
        return None

    hour_text = parts[0].strip()
    minute_text = parts[1].strip()
    if not (hour_text.isdigit() and minute_text.isdigit()):
        return None

    return f"{int(hour_text):02d}:{int(minute_text):02d}"


def _format_rebalance_status(next_rebalance_date: Optional[str], days_to_rebalance: Optional[int]) -> str:
    """格式化顶部下次调仓文案。"""
    if next_rebalance_date and len(next_rebalance_date) == 8 and next_rebalance_date.isdigit():
        date_str = _format_mmdd(next_rebalance_date)
    else:
        date_str = "--/--"

    if days_to_rebalance is None:
        days_str = "--"
    else:
        days_str = str(max(int(days_to_rebalance), 0))

    return f"下次调仓:{date_str}/剩{days_str}天"


def _sanitize_intraday_pct(value: object, abs_limit: float) -> Optional[float]:
    """校验日内涨跌幅，过滤 NaN、无穷和明显脏点。"""
    pct = _coerce_float(value)
    if pct is None or not np.isfinite(pct):
        return None
    if abs(pct) > abs_limit:
        return None
    return pct


def _normalize_intraday_price(price: object, pre_close: object, abs_limit: float) -> Optional[float]:
    """规范化实时价；异常或无效价格回退到昨收，避免生成脏点。"""
    pre_close_float = _coerce_float(pre_close)
    if pre_close_float is None or not np.isfinite(pre_close_float) or pre_close_float <= 0:
        return None

    price_float = _coerce_float(price)
    if price_float is None or not np.isfinite(price_float) or price_float <= 0:
        return pre_close_float

    pct = (price_float / pre_close_float - 1) * 100
    if _sanitize_intraday_pct(pct, abs_limit) is None:
        return pre_close_float
    return price_float


def _parse_intraday_point_time(trade_date: str, label: object) -> Optional[datetime]:
    """解析日内点标签时间，兼容 HH:MM 和 HH:MM:SS。"""
    label_text = str(label).strip()
    if not label_text:
        return None

    label_text = label_text[:8]
    for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d %H:%M"):
        try:
            return datetime.strptime(f"{trade_date} {label_text}", fmt)
        except ValueError:
            continue
    return None


def _is_intraday_trading_time(point_time: datetime) -> bool:
    """判断时间点是否落在 A 股实际盘中交易时段。"""
    current_time = point_time.time()
    return (
        A_SHARE_MORNING_OPEN <= current_time <= A_SHARE_MORNING_CLOSE
        or A_SHARE_AFTERNOON_OPEN <= current_time <= A_SHARE_AFTERNOON_CLOSE
    )


def _format_intraday_point_label(point_time: datetime) -> str:
    """格式化日内点标签，秒级时间保留到持久化层。"""
    if point_time.second:
        return point_time.strftime("%H:%M:%S")
    return point_time.strftime("%H:%M")


def _get_intraday_x_position(point_time: datetime) -> float:
    """按真实盘中时间计算折叠午休后的横坐标位置。"""
    current_seconds = point_time.hour * 3600 + point_time.minute * 60 + point_time.second
    morning_open_seconds = A_SHARE_MORNING_OPEN.hour * 3600 + A_SHARE_MORNING_OPEN.minute * 60
    morning_close_seconds = A_SHARE_MORNING_CLOSE.hour * 3600 + A_SHARE_MORNING_CLOSE.minute * 60
    afternoon_open_seconds = A_SHARE_AFTERNOON_OPEN.hour * 3600 + A_SHARE_AFTERNOON_OPEN.minute * 60

    if current_seconds <= morning_close_seconds:
        offset_minutes = (current_seconds - morning_open_seconds) / 60.0
        position = offset_minutes / INTRADAY_SLOT_MINUTES
    else:
        offset_minutes = (current_seconds - afternoon_open_seconds) / 60.0
        position = INTRADAY_MORNING_SLOT_COUNT + offset_minutes / INTRADAY_SLOT_MINUTES

    return max(0.0, min(position, float(INTRADAY_SLOT_COUNT - 1)))


def _resolve_intraday_point_axis(
    trade_date: str,
    label: object,
    slot_idx: object,
) -> Optional[tuple[int, float, str]]:
    """解析日内点的槽位与绘图横坐标，兼容旧版持久化数据。"""
    point_time = _parse_intraday_point_time(trade_date, label)
    if point_time is not None:
        if not _is_intraday_trading_time(point_time):
            return None
        return (
            _get_intraday_slot_index(point_time),
            _get_intraday_x_position(point_time),
            _format_intraday_point_label(point_time),
        )

    try:
        slot_int = int(slot_idx)
    except (TypeError, ValueError):
        return None
    if slot_int < 0 or slot_int >= INTRADAY_SLOT_COUNT:
        return None
    return slot_int, float(slot_int), str(label).strip()


def _resolve_intraday_slot_index(
    trade_date: str,
    label: object,
    slot_idx: object,
) -> Optional[int]:
    """解析或修正日内槽位，兼容旧版持久化文件并过滤盘前点。"""
    resolved = _resolve_intraday_point_axis(trade_date, label, slot_idx)
    if resolved is None:
        return None
    slot_int, _, _ = resolved
    return slot_int


def _get_diag_log_paths() -> list[Path]:
    """返回诊断日志落盘路径，优先项目目录，失败时兜底系统临时目录。"""
    primary = Path(get_paper_root()) / "state" / DIAG_LOG_FILENAME
    fallback = Path(tempfile.gettempdir()) / DIAG_LOG_FILENAME
    if primary == fallback:
        return [primary]
    return [primary, fallback]


def _emit_diag(message: str, stderr: bool = True) -> None:
    """输出 LCD 运行诊断信息到文件，并尽量同步到 stderr。"""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    for log_path in _get_diag_log_paths():
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            break
        except OSError:
            continue
    if stderr:
        try:
            print(f"[3.5LCD_disp] {message}", file=sys.stderr, flush=True)
        except OSError:
            pass


def _emit_diag_once(key: str, message: str, stderr: bool = True) -> None:
    """同一类诊断信息仅记录一次，避免持续刷屏。"""
    with _diag_lock:
        if key in _diag_once_keys:
            return
        _diag_once_keys.add(key)
    _emit_diag(message, stderr=stderr)


def _describe_framebuffer_candidates() -> str:
    """返回当前系统可见的 framebuffer 设备列表。"""
    try:
        candidates = sorted(path.name for path in Path("/dev").glob("fb*"))
    except OSError:
        return "无法枚举 /dev/fb*"
    if not candidates:
        return "未发现 /dev/fb*"
    return ", ".join(candidates)


def _resolve_framebuffer_path() -> str:
    """解析当前应写入的 framebuffer 设备路径。"""
    env_path = os.getenv("LAZYBULL_LCD_FB_PATH")
    if env_path:
        return env_path

    if Path(DEFAULT_FB_PATH).exists():
        return DEFAULT_FB_PATH

    fallback_path = "/dev/fb0"
    if Path(fallback_path).exists():
        return fallback_path

    return DEFAULT_FB_PATH


def _render_bootstrap_screen(message: str) -> None:
    """在正式进入刷新线程前先画一张启动测试页。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(img)
    font_title = _get_font(24)
    font_body = _get_font(16)
    font_tip = _get_font(13)

    title = "LCD启动中"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_x = (WIDTH - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, 108), title, fill=COLOR_YELLOW, font=font_title)

    body_bbox = draw.textbbox((0, 0), message, font=font_body)
    body_x = (WIDTH - (body_bbox[2] - body_bbox[0])) // 2
    draw.text((body_x, 148), message, fill=COLOR_TEXT, font=font_body)

    fb_text = f"FB:{_resolve_framebuffer_path()}"
    tip_bbox = draw.textbbox((0, 0), fb_text, font=font_tip)
    tip_x = (WIDTH - (tip_bbox[2] - tip_bbox[0])) // 2
    draw.text((tip_x, 188), fb_text, fill=COLOR_LABEL, font=font_tip)

    _write_fb(img)
    

def _format_error_lines(
    message: str, line_width: int = 26, max_lines: int = 4
) -> list[str]:
    """将错误消息裁剪成适合屏幕显示的多行文本。"""
    text = (message or "未知异常").replace("\n", " ").strip()
    if not text:
        text = "未知异常"

    lines = []
    remain = text
    while remain and len(lines) < max_lines:
        lines.append(remain[:line_width])
        remain = remain[line_width:]

    if remain and lines:
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def _resolve_cycle_slot_count(rebalance_freq: object, point_count: int) -> int:
    """持仓周期图的 x 轴槽位数固定为 max(调仓周期, 当前持仓天数)。"""
    slot_count = max(point_count, 2)
    try:
        slot_count = max(slot_count, int(rebalance_freq))
    except (TypeError, ValueError):
        pass
    return slot_count


def _build_cycle_chart_payload(
    dates: list[str],
    index_pct: list[float],
    shenzhen_pct: list[float],
    portfolio_pct: list[float],
    rebalance_freq: object,
    base_value: float,
    csi800_pct: Optional[list[float]] = None,
) -> Optional[dict]:
    """构建持仓周期图负载，并固定 x 轴槽位。"""
    if csi800_pct is None:
        csi800_pct = list(index_pct)
    point_count = min(
        len(dates),
        len(index_pct),
        len(shenzhen_pct),
        len(csi800_pct),
        len(portfolio_pct),
    )
    if point_count == 0:
        return None

    dates = dates[:point_count]
    index_pct = index_pct[:point_count]
    shenzhen_pct = shenzhen_pct[:point_count]
    csi800_pct = csi800_pct[:point_count]
    portfolio_pct = portfolio_pct[:point_count]
    slot_count = _resolve_cycle_slot_count(rebalance_freq, point_count)

    return {
        'mode': 'cycle',
        'dates': dates,
        'index_pct': index_pct,
        'shenzhen_pct': shenzhen_pct,
        'csi800_pct': csi800_pct,
        'portfolio_pct': portfolio_pct,
        'slot_indices': list(range(point_count)),
        'slot_count': slot_count,
        'x_start_label': _format_mmdd(dates[0]),
        'x_end_label': f"{slot_count}天" if slot_count > point_count else _format_mmdd(dates[-1]),
        'index_label': '上证',
        'shenzhen_label': '深证',
        'portfolio_label': '持仓',
        'csi800_label': '中证800',
        'base_value': base_value,
    }


def _get_intraday_slot_index(point_time: datetime) -> int:
    """将盘中时间映射到固定的 10 分钟槽位，并折叠午休时段。"""
    current_minutes = point_time.hour * 60 + point_time.minute

    morning_open_minutes = A_SHARE_MORNING_OPEN.hour * 60 + A_SHARE_MORNING_OPEN.minute
    morning_close_minutes = A_SHARE_MORNING_CLOSE.hour * 60 + A_SHARE_MORNING_CLOSE.minute
    afternoon_open_minutes = A_SHARE_AFTERNOON_OPEN.hour * 60 + A_SHARE_AFTERNOON_OPEN.minute

    if current_minutes <= morning_close_minutes:
        slot_idx = (current_minutes - morning_open_minutes) // INTRADAY_SLOT_MINUTES
    else:
        slot_idx = INTRADAY_MORNING_SLOT_COUNT + (
            (current_minutes - afternoon_open_minutes) // INTRADAY_SLOT_MINUTES
        )

    return max(0, min(slot_idx, INTRADAY_SLOT_COUNT - 1))


def _get_chart_y_range(values: list[float]) -> tuple[float, float]:
    """计算图表 y 轴范围，并始终保证 0% 参考线可见。"""
    if not values:
        return -1.0, 1.0

    value_min = min(values)
    value_max = max(values)
    y_min = min(value_min, 0.0)
    y_max = max(value_max, 0.0)
    y_margin = max((y_max - y_min) * 0.15, 0.5)
    y_min -= y_margin
    y_max += y_margin
    if y_max - y_min < 0.01:
        y_min -= 0.5
        y_max += 0.5
    return y_min, y_max


def _get_intraday_break_slot_position(chart_data: Optional[dict]) -> Optional[float]:
    """返回午休折叠边界所在的虚拟槽位位置。"""
    if not isinstance(chart_data, dict) or chart_data.get('mode') != 'intraday':
        return None
    return float(INTRADAY_MORNING_SLOT_COUNT) - 0.5


def _draw_vertical_dashed_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    top_y: int,
    bottom_y: int,
    color: tuple,
    segment_length: int = 3,
    gap_length: int = 3,
) -> None:
    """绘制一条竖向虚线。"""
    current_y = top_y
    while current_y <= bottom_y:
        end_y = min(current_y + segment_length - 1, bottom_y)
        draw.line([(x, current_y), (x, end_y)], fill=color, width=1)
        current_y += segment_length + gap_length


def _draw_intraday_break_marker(
    draw: ImageDraw.ImageDraw,
    chart_data: Optional[dict],
    cx: int,
    cy: int,
    cw: int,
    ch: int,
    slot_count: int,
    font_xs: ImageFont.FreeTypeFont,
) -> None:
    """在日内图中绘制午休折叠分隔标记。"""
    break_slot_position = _get_intraday_break_slot_position(chart_data)
    if break_slot_position is None or slot_count < 2:
        return

    break_px = cx + int(break_slot_position / max(slot_count - 1, 1) * cw)
    _draw_vertical_dashed_line(
        draw,
        break_px,
        cy + 2,
        cy + ch - 2,
        COLOR_CHART_BREAK,
    )

    lunch_label = "午休"
    bbox = draw.textbbox((0, 0), lunch_label, font=font_xs)
    label_w = bbox[2] - bbox[0]
    label_x = max(cx + 2, min(break_px - label_w // 2, cx + cw - label_w - 2))
    draw.text((label_x, cy + ch + 1), lunch_label, fill=COLOR_CHART_BREAK, font=font_xs)


def _smooth_intraday_series_for_display(values: list[float]) -> list[float]:
    """对日内图做轻度三点平滑，仅影响显示，不改原始数据。"""
    if len(values) < 3:
        return list(values)

    smoothed = list(values)
    for idx in range(1, len(values) - 1):
        smoothed[idx] = (values[idx - 1] + values[idx] * 2.0 + values[idx + 1]) / 4.0
    return smoothed


def _draw_chart_series(
    draw: ImageDraw.ImageDraw,
    series_points: list[tuple[list[tuple[float, float]], tuple[int, int, int]]],
    cx: int,
    cy: int,
    cw: int,
    ch: int,
) -> None:
    """绘制图表折线；有底层图像时使用超采样以减轻锯齿。"""
    point_radius = 2
    line_width = 2

    base_image = getattr(draw, "_image", None)
    if not isinstance(base_image, Image.Image):
        for points, color in series_points:
            for idx in range(len(points) - 1):
                draw.line([points[idx], points[idx + 1]], fill=color, width=line_width)
            if points:
                px, py = points[-1]
                draw.ellipse(
                    [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                    fill=color,
                )
        return

    scale = 3
    pad = 3
    overlay = Image.new(
        "RGBA",
        ((cw + pad * 2) * scale, (ch + pad * 2) * scale),
        (0, 0, 0, 0),
    )
    overlay_draw = ImageDraw.Draw(overlay)

    for points, color in series_points:
        rgba_color = color + (255,)
        scaled_points = [
            ((px - cx + pad) * scale, (py - cy + pad) * scale)
            for px, py in points
        ]
        for idx in range(len(scaled_points) - 1):
            overlay_draw.line(
                [scaled_points[idx], scaled_points[idx + 1]],
                fill=rgba_color,
                width=line_width * scale,
            )
        if scaled_points:
            px, py = scaled_points[-1]
            overlay_draw.ellipse(
                [
                    px - point_radius * scale,
                    py - point_radius * scale,
                    px + point_radius * scale,
                    py + point_radius * scale,
                ],
                fill=rgba_color,
            )

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    smoothed_overlay = overlay.resize((cw + pad * 2, ch + pad * 2), resample=resampling)
    base_image.paste(smoothed_overlay, (cx - pad, cy - pad), smoothed_overlay)


def _get_intraday_display_x_positions(
    chart_data: Optional[dict],
    dates: list[str],
    slot_indices: list[int],
    x_positions: list[float],
) -> list[float]:
    """返回用于绘制的日内图横坐标，边界时刻吸附到视觉边界。"""
    display_positions = [float(position) for position in x_positions]
    if not isinstance(chart_data, dict) or chart_data.get('mode') != 'intraday':
        return display_positions

    trade_date = str(chart_data.get('trade_date', '')).strip()
    if len(trade_date) != 8 or len(display_positions) != len(dates):
        return display_positions

    break_position = float(INTRADAY_MORNING_SLOT_COUNT) - 0.5
    close_position = float(INTRADAY_SLOT_COUNT - 1)
    boundary_times = {
        A_SHARE_MORNING_CLOSE: break_position,
        A_SHARE_AFTERNOON_OPEN: break_position,
        A_SHARE_AFTERNOON_CLOSE: close_position,
    }

    for idx, label in enumerate(dates):
        point_time = _parse_intraday_point_time(trade_date, label)
        if point_time is None:
            continue
        snapped_position = boundary_times.get(point_time.time())
        if snapped_position is not None:
            display_positions[idx] = snapped_position

    return display_positions


def _draw_zero_reference_line(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    cw: int,
    ch: int,
    y_min: float,
    y_range: float,
    font_xs: ImageFont.FreeTypeFont,
) -> None:
    """绘制 0% 参考线及标签。"""
    zero_py = cy + ch - int((0 - y_min) / y_range * ch) + ZERO_LINE_Y_OFFSET
    zero_py = max(cy + 1, min(zero_py, cy + ch - 1))
    draw.line(
        [(cx + 1, zero_py), (cx + cw - 1, zero_py)],
        fill=COLOR_CHART_ZERO_LINE,
        width=1,
    )

    zero_label = "0%"
    bbox = draw.textbbox((0, 0), zero_label, font=font_xs)
    label_w = bbox[2] - bbox[0]
    label_h = bbox[3] - bbox[1]
    tag_x = cx + 4
    tag_y = max(cy + 2, min(zero_py - label_h - 2, cy + ch - label_h - 2))
    draw.rounded_rectangle(
        [tag_x, tag_y, tag_x + label_w + 6, tag_y + label_h + 2],
        radius=2,
        fill=COLOR_CHART_BG,
        outline=COLOR_CHART_ZERO_LINE,
    )
    draw.text((tag_x + 3, tag_y), zero_label, fill=COLOR_CHART_ZERO_LINE, font=font_xs)


def _empty_intraday_chart(trade_date: str) -> dict:
    """创建新的日内图负载。"""
    return {
        'mode': 'intraday',
        'trade_date': trade_date,
        'dates': [],
        'x_positions': [],
        'raw_index_pct': [],
        'raw_shenzhen_pct': [],
        'raw_csi800_pct': [],
        'raw_portfolio_pct': [],
        'index_pct': [],
        'shenzhen_pct': [],
        'csi800_pct': [],
        'portfolio_pct': [],
        'slot_indices': [],
        'slot_count': INTRADAY_SLOT_COUNT,
        'x_start_label': INTRADAY_WINDOW_START.strftime("%H:%M"),
        'x_end_label': INTRADAY_WINDOW_END.strftime("%H:%M"),
        'index_label': '上证',
        'shenzhen_label': '深证',
        'portfolio_label': '持仓',
        'csi800_label': '中证800',
    }


def _compose_intraday_chart(
    chart_data: dict,
    trade_date: str,
    dates: list[str],
    x_positions: list[float],
    slot_indices: list[int],
    raw_index_values: list[float],
    raw_shenzhen_values: list[float],
    raw_csi800_values: list[float],
    raw_portfolio_values: list[float],
) -> dict:
    """生成日内图负载，同时保留原始值和首点归零后的显示值。"""
    return {
        **chart_data,
        'trade_date': trade_date,
        'dates': dates,
        'x_positions': x_positions,
        'raw_index_pct': raw_index_values,
        'raw_shenzhen_pct': raw_shenzhen_values,
        'raw_csi800_pct': raw_csi800_values,
        'raw_portfolio_pct': raw_portfolio_values,
        'index_pct': list(raw_index_values),
        'shenzhen_pct': list(raw_shenzhen_values),
        'csi800_pct': list(raw_csi800_values),
        'portfolio_pct': list(raw_portfolio_values),
        'slot_indices': slot_indices,
    }


def _upsert_intraday_chart(
    chart_data: Optional[dict],
    point_time: datetime,
    index_pct: float,
    shenzhen_pct: float,
    portfolio_pct: float,
    csi800_pct: Optional[float] = None,
) -> dict:
    """向日内图追加一个采样点，保留每次刷新历史。"""
    if csi800_pct is None:
        csi800_pct = shenzhen_pct
    trade_date = point_time.strftime("%Y%m%d")
    if chart_data is None or chart_data.get('trade_date') != trade_date:
        chart_data = _empty_intraday_chart(trade_date)

    slot_indices = list(chart_data.get('slot_indices', []))
    x_positions = list(chart_data.get('x_positions', []))
    raw_index_values = list(chart_data.get('raw_index_pct', chart_data.get('index_pct', [])))
    raw_shenzhen_values = list(
        chart_data.get('raw_shenzhen_pct', chart_data.get('shenzhen_pct', []))
    )
    raw_csi800_values = list(
        chart_data.get('raw_csi800_pct', chart_data.get('csi800_pct', []))
    )
    raw_portfolio_values = list(
        chart_data.get('raw_portfolio_pct', chart_data.get('portfolio_pct', []))
    )
    dates = list(chart_data.get('dates', []))
    if not x_positions and len(slot_indices) == len(dates):
        for label, slot_idx in zip(dates, slot_indices):
            resolved = _resolve_intraday_point_axis(trade_date, label, slot_idx)
            x_positions.append(float(slot_idx) if resolved is None else resolved[1])
    if (
        len(slot_indices) != len(raw_index_values)
        or len(slot_indices) != len(raw_shenzhen_values)
        or len(slot_indices) != len(raw_csi800_values)
        or len(slot_indices) != len(raw_portfolio_values)
        or len(slot_indices) != len(dates)
        or len(slot_indices) != len(x_positions)
    ):
        chart_data = _empty_intraday_chart(trade_date)
        slot_indices = []
        x_positions = []
        raw_index_values = []
        raw_shenzhen_values = []
        raw_csi800_values = []
        raw_portfolio_values = []
        dates = []
    slot_idx = _get_intraday_slot_index(point_time)
    point_label = _format_intraday_point_label(point_time)
    x_position = _get_intraday_x_position(point_time)

    slot_indices.append(slot_idx)
    x_positions.append(x_position)
    raw_index_values.append(index_pct)
    raw_shenzhen_values.append(shenzhen_pct)
    raw_csi800_values.append(csi800_pct)
    raw_portfolio_values.append(portfolio_pct)
    dates.append(point_label)

    return _compose_intraday_chart(
        chart_data,
        trade_date,
        dates,
        x_positions,
        slot_indices,
        raw_index_values,
        raw_shenzhen_values,
        raw_csi800_values,
        raw_portfolio_values,
    )


def _get_intraday_chart_state_dir() -> Path:
    """返回 3.5 寸 LCD 日内图持久化目录。"""
    return Path(get_paper_root()) / "state" / INTRADAY_CHART_STATE_DIRNAME


def _get_intraday_chart_state_path(trade_date: str) -> Path:
    """返回指定交易日的日内图持久化文件路径。"""
    return _get_intraday_chart_state_dir() / f"{trade_date}.json"


def _normalize_intraday_chart(chart_data: object, trade_date: Optional[str] = None) -> Optional[dict]:
    """规范化日内图持久化数据。"""
    if not isinstance(chart_data, dict):
        return None

    payload_trade_date = str(chart_data.get('trade_date', ''))
    if not payload_trade_date:
        return None
    if trade_date is not None and payload_trade_date != trade_date:
        return None

    normalized = _empty_intraday_chart(payload_trade_date)
    raw_dates = chart_data.get('dates', [])
    raw_index = chart_data.get('raw_index_pct', chart_data.get('index_pct', []))
    raw_shenzhen = chart_data.get('raw_shenzhen_pct', chart_data.get('shenzhen_pct', []))
    raw_csi800 = chart_data.get('raw_csi800_pct', chart_data.get('csi800_pct', raw_index))
    raw_portfolio = chart_data.get('raw_portfolio_pct', chart_data.get('portfolio_pct', []))
    raw_slots = chart_data.get('slot_indices', [])
    if not all(
        isinstance(items, list)
        for items in (raw_dates, raw_index, raw_shenzhen, raw_csi800, raw_portfolio, raw_slots)
    ):
        return normalized

    normalized_points: list[tuple[float, int, str, float, float, float, float]] = []
    for original_idx, (label, index_val, shenzhen_val, csi800_val, portfolio_val, slot_idx) in enumerate(
        zip(raw_dates, raw_index, raw_shenzhen, raw_csi800, raw_portfolio, raw_slots)
    ):
        resolved_axis = _resolve_intraday_point_axis(payload_trade_date, label, slot_idx)
        if resolved_axis is None:
            continue
        slot_int, x_position, normalized_label = resolved_axis
        index_float = _sanitize_intraday_pct(index_val, INTRADAY_INDEX_PCT_ABS_LIMIT)
        shenzhen_float = _sanitize_intraday_pct(shenzhen_val, INTRADAY_INDEX_PCT_ABS_LIMIT)
        csi800_float = _sanitize_intraday_pct(csi800_val, INTRADAY_INDEX_PCT_ABS_LIMIT)
        portfolio_float = _sanitize_intraday_pct(
            portfolio_val,
            INTRADAY_PORTFOLIO_PCT_ABS_LIMIT,
        )
        if (
            index_float is None
            or shenzhen_float is None
            or csi800_float is None
            or portfolio_float is None
        ):
            continue
        normalized_points.append(
            (
                x_position,
                original_idx,
                normalized_label,
                slot_int,
                index_float,
                shenzhen_float,
                csi800_float,
                portfolio_float,
            )
        )

    normalized_points.sort(key=lambda item: (item[0], item[1]))

    for (
        x_position,
        _,
        label,
        slot_int,
        index_float,
        shenzhen_float,
        csi800_float,
        portfolio_float,
    ) in normalized_points:
        normalized['dates'].append(label)
        normalized['x_positions'].append(x_position)
        normalized['raw_index_pct'].append(index_float)
        normalized['raw_shenzhen_pct'].append(shenzhen_float)
        normalized['raw_csi800_pct'].append(csi800_float)
        normalized['raw_portfolio_pct'].append(portfolio_float)
        normalized['slot_indices'].append(slot_int)

    return _compose_intraday_chart(
        normalized,
        payload_trade_date,
        list(normalized['dates']),
        list(normalized['x_positions']),
        list(normalized['slot_indices']),
        list(normalized['raw_index_pct']),
        list(normalized['raw_shenzhen_pct']),
        list(normalized['raw_csi800_pct']),
        list(normalized['raw_portfolio_pct']),
    )


def _save_intraday_chart(chart_data: Optional[dict]) -> None:
    """将当日日内图历史点持久化到 data/paper/state。"""
    normalized = _normalize_intraday_chart(chart_data)
    if normalized is None:
        return

    state_dir = _get_intraday_chart_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    file_path = _get_intraday_chart_state_path(normalized['trade_date'])
    tmp_path = file_path.with_suffix(".json.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    tmp_path.replace(file_path)


def _load_intraday_chart(now: Optional[datetime] = None) -> Optional[dict]:
    """读取当日日内图历史点，脚本重启后可续接。"""
    current_dt = now or datetime.now()
    trade_date = current_dt.strftime("%Y%m%d")
    file_path = _get_intraday_chart_state_path(trade_date)
    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return _normalize_intraday_chart(payload, trade_date=trade_date)


_trade_date_set_cache: Optional[frozenset[str]] = None
_trade_dates_cache: Optional[tuple[str, ...]] = None
_trade_date_set_lock = threading.Lock()


def _load_trade_dates() -> tuple[str, ...]:
    """加载交易日列表并缓存，失败时返回空元组。"""
    global _trade_date_set_cache, _trade_dates_cache

    if _trade_dates_cache is not None:
        return _trade_dates_cache

    with _trade_date_set_lock:
        if _trade_dates_cache is not None:
            return _trade_dates_cache

        result: tuple[str, ...] = ()
        try:
            from src.lazybull.data import DataLoader, Storage

            loader = DataLoader(storage=Storage(root_path=get_data_root()))
            trade_cal = loader.load_clean_trade_cal()
            if trade_cal is not None and not trade_cal.empty:
                result = tuple(
                    trade_cal.loc[trade_cal['is_open'] == 1, 'cal_date']
                    .astype(str)
                    .sort_values()
                    .tolist()
                )
        except Exception:
            result = ()

        _trade_dates_cache = result
        _trade_date_set_cache = frozenset(result)
        return _trade_dates_cache


def _load_trade_date_set() -> frozenset[str]:
    """加载交易日集合并缓存，失败时返回空集合。"""
    if _trade_date_set_cache is not None:
        return _trade_date_set_cache
    return frozenset(_load_trade_dates())


def _get_cached_trade_date_set() -> Optional[frozenset[str]]:
    """返回已加载的交易日集合；若尚未加载则返回 None。"""
    return _trade_date_set_cache


def _get_cached_trade_dates() -> Optional[tuple[str, ...]]:
    """返回已加载的交易日列表；若尚未加载则返回 None。"""
    return _trade_dates_cache


def _is_trade_day(now: Optional[datetime] = None, allow_load: bool = False) -> bool:
    """判断当前日期是否为交易日。

    默认不在显示线程首帧触发交易日历加载，避免与数据线程导入过程互相阻塞。
    """
    current_dt = now or datetime.now()
    trade_dates = _load_trade_date_set() if allow_load else _get_cached_trade_date_set()
    if trade_dates:
        return current_dt.strftime("%Y%m%d") in trade_dates
    return current_dt.weekday() < 5


def _is_intraday_chart_window(now: Optional[datetime] = None) -> bool:
    """判断是否处于盘中图显示窗口（交易日 9:30-15:00）。"""
    current_dt = now or datetime.now()
    if not _is_trade_day(current_dt):
        return False
    current_time = current_dt.time()
    return INTRADAY_WINDOW_START <= current_time <= INTRADAY_WINDOW_END


def _is_realtime_quote_window(now: Optional[datetime] = None) -> bool:
    """判断当前是否处于可用实时行情窗口。"""
    current_dt = now or datetime.now()
    if not _is_trade_day(current_dt):
        return False

    current_time = current_dt.time()
    return (
        A_SHARE_MORNING_OPEN <= current_time <= A_SHARE_MORNING_CLOSE
        or A_SHARE_AFTERNOON_OPEN <= current_time <= A_SHARE_AFTERNOON_CLOSE
    )


def _get_target_cycle_data_date(
    now: Optional[datetime] = None,
    allow_load: bool = False,
) -> Optional[str]:
    """返回当前应具备的最近一个交易日周期图数据日期。"""
    current_dt = now or datetime.now()
    today_str = current_dt.strftime("%Y%m%d")
    trade_dates = _load_trade_dates() if allow_load else _get_cached_trade_dates()

    if trade_dates:
        if today_str in trade_dates:
            today_index = trade_dates.index(today_str)
            if current_dt.time() >= INTRADAY_WINDOW_END:
                return today_str
            if today_index > 0:
                return trade_dates[today_index - 1]
            return None
        return next(
            (trade_date for trade_date in reversed(trade_dates) if trade_date < today_str),
            None,
        )

    if current_dt.weekday() >= 5:
        days_back = current_dt.weekday() - 4
        return (current_dt - timedelta(days=days_back)).strftime("%Y%m%d")

    if current_dt.time() >= INTRADAY_WINDOW_END:
        return today_str

    prev_business_days = 3 if current_dt.weekday() == 0 else 1
    return (current_dt - timedelta(days=prev_business_days)).strftime("%Y%m%d")


def _select_chart_data(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """根据当前时段选择显示周期图或盘中图。"""
    current_dt = now or datetime.now()
    if _should_show_intraday_chart(cycle_chart_data, intraday_chart_data, current_dt):
        return intraday_chart_data
    return cycle_chart_data


def _get_cycle_last_data_date(cycle_chart_data: Optional[dict]) -> Optional[str]:
    """返回周期图最后一个数据日（YYYYMMDD）。"""
    if not cycle_chart_data:
        return None
    dates = cycle_chart_data.get('dates', [])
    if not isinstance(dates, list) or not dates:
        return None
    last_date = str(dates[-1])
    if len(last_date) == 8 and last_date.isdigit():
        return last_date
    return None


def _format_cycle_last_data_label(cycle_chart_data: Optional[dict]) -> Optional[str]:
    """格式化周期图最后数据日角标。"""
    last_date = _get_cycle_last_data_date(cycle_chart_data)
    if last_date is None:
        return None
    return f"数据日:{_format_mmdd(last_date)}"


def _has_cycle_data_for_target(
    cycle_chart_data: Optional[dict],
    target_cycle_date: Optional[str],
) -> bool:
    """判断周期图是否已经覆盖目标交易日。"""
    if target_cycle_date is None:
        return False
    last_date = _get_cycle_last_data_date(cycle_chart_data)
    if last_date is None:
        return False
    return last_date >= target_cycle_date


def _has_intraday_chart_for_today(
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断是否已经有当日日内图数据。"""
    if intraday_chart_data is None:
        return False
    current_dt = now or datetime.now()
    return intraday_chart_data.get('trade_date') == current_dt.strftime("%Y%m%d")


def _is_intraday_chart_complete(
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断当日日内图是否已经补齐收盘最后一个槽位。"""
    if not _has_intraday_chart_for_today(intraday_chart_data, now):
        return False

    slot_indices = intraday_chart_data.get('slot_indices', [])
    if not isinstance(slot_indices, list) or not slot_indices:
        return False
    return int(slot_indices[-1]) >= INTRADAY_SLOT_COUNT - 1


def _is_morning_intraday_chart_complete(
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断当日日内图是否已经补齐上午 11:30 最后一格。"""
    if not _has_intraday_chart_for_today(intraday_chart_data, now):
        return False

    slot_indices = intraday_chart_data.get('slot_indices', [])
    if not isinstance(slot_indices, list) or not slot_indices:
        return False
    return int(slot_indices[-1]) >= INTRADAY_MORNING_SLOT_COUNT - 1


def _should_show_intraday_chart(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断当前是否应继续显示日内图。"""
    current_dt = now or datetime.now()
    if not _has_intraday_chart_for_today(intraday_chart_data, current_dt):
        return False
    if _is_intraday_chart_window(current_dt):
        return True
    if not _is_trade_day(current_dt, allow_load=True):
        return False

    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    return not _has_cycle_data_for_target(cycle_chart_data, target_cycle_date)


def _get_refresh_policy(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> dict:
    """返回当前时段的数据刷新策略。"""
    current_dt = now or datetime.now()
    realtime_active = _is_realtime_quote_window(current_dt)
    if realtime_active:
        return {
            'refresh_cycle': False,
            'refresh_realtime': True,
        }

    need_morning_completion = _should_keep_morning_close_completion_active(
        intraday_chart_data,
        current_dt,
    )

    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    need_cycle_refresh = not _has_cycle_data_for_target(cycle_chart_data, target_cycle_date)
    need_intraday_completion = (
        _is_trade_day(current_dt, allow_load=True)
        and current_dt.strftime("%Y%m%d") == target_cycle_date
        and not _is_intraday_chart_complete(intraday_chart_data, current_dt)
    )
    return {
        'refresh_cycle': need_cycle_refresh,
        'refresh_realtime': need_intraday_completion or need_morning_completion,
    }


def _get_realtime_session_key(now: Optional[datetime] = None) -> Optional[str]:
    """返回当前实时行情窗口标识，用于开盘和午后开盘时立即刷新。"""
    current_dt = now or datetime.now()
    if not _is_realtime_quote_window(current_dt):
        return None
    session = "am" if current_dt.time() <= A_SHARE_MORNING_CLOSE else "pm"
    return f"{current_dt:%Y%m%d}-{session}"


def _is_interval_due(
    last_refresh_at: Optional[datetime],
    interval_seconds: int,
    now: Optional[datetime] = None,
) -> bool:
    """判断距离上次刷新是否已达到指定间隔。"""
    if last_refresh_at is None:
        return True
    current_dt = now or datetime.now()
    return (current_dt - last_refresh_at).total_seconds() >= float(interval_seconds)


def _is_realtime_refresh_due(
    refresh_allowed: bool,
    last_refresh_at: Optional[datetime],
    last_session_key: Optional[str],
    now: Optional[datetime] = None,
) -> tuple[bool, Optional[str]]:
    """判断盘中实时面板是否应立即刷新。"""
    current_dt = now or datetime.now()
    session_key = _get_realtime_session_key(current_dt)
    if not refresh_allowed:
        return False, session_key
    morning_close_dt = datetime.combine(current_dt.date(), A_SHARE_MORNING_CLOSE)
    morning_close_deadline = morning_close_dt + timedelta(seconds=MORNING_CLOSE_INTRADAY_GRACE_SECONDS)
    if (
        last_refresh_at is not None
        and last_refresh_at < morning_close_dt <= current_dt <= morning_close_deadline
    ):
        return True, session_key
    if session_key is None:
        return _is_interval_due(last_refresh_at, REALTIME_REFRESH_INTERVAL, current_dt), None
    if last_session_key != session_key:
        return True, session_key
    return _is_interval_due(last_refresh_at, REALTIME_REFRESH_INTERVAL, current_dt), session_key


def _is_cycle_refresh_due(
    cycle_chart_data: Optional[dict],
    refresh_allowed: bool,
    last_refresh_at: Optional[datetime],
    last_target_date: Optional[str],
    now: Optional[datetime] = None,
) -> tuple[bool, Optional[str]]:
    """判断周期图是否应补抓最近一个交易日数据。"""
    current_dt = now or datetime.now()
    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    if not refresh_allowed or target_cycle_date is None:
        return False, target_cycle_date
    if _has_cycle_data_for_target(cycle_chart_data, target_cycle_date):
        return False, target_cycle_date
    if last_target_date != target_cycle_date:
        return True, target_cycle_date
    return _is_interval_due(last_refresh_at, REFRESH_INTERVAL, current_dt), target_cycle_date


def _get_data_worker_wait_seconds(
    now: Optional[datetime] = None,
    cycle_chart_data: Optional[dict] = None,
    intraday_chart_data: Optional[dict] = None,
) -> float:
    """返回数据线程下次唤醒间隔。

    盘中按 2 分钟节奏唤醒，非交易时段按 10 分钟节奏唤醒；若即将跨过
    开盘、午休结束或收盘切图边界，
    则缩短本次等待，确保关键时点能尽快刷新。
    """
    current_dt = now or datetime.now()
    morning_close_completion_active = _should_keep_morning_close_completion_active(
        intraday_chart_data,
        current_dt,
    )
    wait_seconds = float(
        REALTIME_REFRESH_INTERVAL
        if (
            morning_close_completion_active
            or _should_keep_realtime_completion_active(cycle_chart_data, intraday_chart_data, current_dt)
        )
        else REFRESH_INTERVAL
    )

    if not _is_trade_day(current_dt):
        return wait_seconds

    current_time = current_dt.time()
    if A_SHARE_MORNING_CLOSE < current_time < A_SHARE_AFTERNOON_OPEN and not morning_close_completion_active:
        afternoon_open_dt = datetime.combine(current_dt.date(), A_SHARE_AFTERNOON_OPEN)
        seconds_to_boundary = (afternoon_open_dt - current_dt).total_seconds() + 1.0
        return max(1.0, seconds_to_boundary)

    boundary_times = [
        INTRADAY_WINDOW_START,
        A_SHARE_MORNING_CLOSE,
        A_SHARE_AFTERNOON_OPEN,
        INTRADAY_WINDOW_END,
    ]
    future_boundaries = [
        datetime.combine(current_dt.date(), boundary_time)
        for boundary_time in boundary_times
        if boundary_time >= current_time
    ]
    if not future_boundaries:
        return wait_seconds

    seconds_to_boundary = min(
        (boundary_dt - current_dt).total_seconds() + 1.0
        for boundary_dt in future_boundaries
    )
    if seconds_to_boundary <= 0:
        return 1.0
    return min(wait_seconds, max(1.0, seconds_to_boundary))


def _get_snapshot_quote_time(snapshot: Optional[dict]) -> Optional[datetime]:
    """从实时快照中解析行情时间，优先用于收盘后补齐最后一格。"""
    if snapshot is None:
        return None

    trade_date = str(snapshot.get('current_date', '')).strip()
    if len(trade_date) != 8 or not trade_date.isdigit():
        return None

    rt_df = snapshot.get('quotes')
    if rt_df is None or rt_df.empty:
        return None

    quote_time_text = ''
    for _, row in rt_df.iterrows():
        candidate = str(row.get('TIME', row.get('time', ''))).strip()
        if candidate:
            quote_time_text = candidate[:8]
            break

    if not quote_time_text:
        return None

    for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d %H:%M"):
        try:
            point_time = datetime.strptime(f"{trade_date} {quote_time_text}", fmt)
            break
        except ValueError:
            point_time = None
    if point_time is None:
        return None

    if not (INTRADAY_WINDOW_START <= point_time.time() <= INTRADAY_WINDOW_END):
        return None
    return point_time


# ---------- 背光控制 ----------

_backlight_state: Optional[dict] = None


def _init_backlight() -> None:
    """初始化背光 PWM，设置为 BACKLIGHT_BRIGHTNESS 亮度。

    优先尝试 sysfs 接口，失败则回退到 PWM（优先 lgpio）。
    在非树莓派环境静默跳过。
    """
    global _backlight_state

    try:
        _backlight_state = _set_backlight_helper(
            BACKLIGHT_BRIGHTNESS,
            method="auto",
            pin=BACKLIGHT_PIN,
            frequency=1000,
        )
        if _backlight_state.get("method") == "sysfs":
            backlight_name = _backlight_state.get("backlight_name")
            if backlight_name:
                _emit_diag_once(
                    "backlight_sysfs_ok",
                    f"背光初始化完成: 使用 sysfs 接口({backlight_name})",
                )
            else:
                _emit_diag_once("backlight_sysfs_ok", "背光初始化完成: 使用 sysfs 接口")
            return

        backend = _backlight_state.get("backend", "pwm")
        _emit_diag_once("backlight_pwm_ok", f"背光初始化完成: 使用 {backend} PWM")
        _emit_diag_once(
            "backlight_pwm_hardware_note",
            _get_pwm_hardware_note_helper(_backlight_state.get("pin", BACKLIGHT_PIN)),
        )
    except Exception as exc:
        _backlight_state = None
        _emit_diag_once(
            "backlight_unavailable",
            f"背光初始化未生效: {type(exc).__name__}: {exc}，若屏幕无背光请检查驱动/权限",
        )


def _set_backlight(brightness: int) -> None:
    """设置背光亮度（0~100）。"""
    global _backlight_state

    try:
        if isinstance(_backlight_state, dict):
            if _backlight_state.get("method") == "pwm":
                _update_pwm_backlight_state_helper(_backlight_state, brightness)
                return

            _backlight_state = _set_backlight_helper(
                brightness,
                method="sysfs",
                backlight_name=_backlight_state.get("backlight_name"),
                pin=BACKLIGHT_PIN,
                frequency=1000,
            )
            return

        _backlight_state = _set_backlight_helper(
            brightness,
            method="auto",
            pin=BACKLIGHT_PIN,
            frequency=1000,
        )
    except Exception:
        pass


def _cleanup_backlight() -> None:
    """清理背光 PWM 资源。"""
    global _backlight_state
    _cleanup_backlight_state_helper(_backlight_state)
    _backlight_state = None


# ---------- framebuffer 输出 ----------

def _write_fb(img: Image.Image) -> None:
    """将 PIL Image 转为 RGB565 并写入 framebuffer。"""
    img_array = np.array(img).astype(np.uint16)
    r = (img_array[:, :, 0] >> 3) << 11
    g = (img_array[:, :, 1] >> 2) << 5
    b = img_array[:, :, 2] >> 3
    rgb565 = r | g | b
    fb_path = _resolve_framebuffer_path()
    try:
        with open(fb_path, "wb") as f:
            f.write(rgb565.tobytes())
        _emit_diag_once(
            f"fb_write_ok::{fb_path}",
            f"framebuffer 写入正常: {fb_path} | 可用设备: {_describe_framebuffer_candidates()}",
        )
    except Exception as exc:
        _emit_diag_once(
            f"fb_write_error::{fb_path}",
            f"framebuffer 写入失败: {fb_path} | {type(exc).__name__}: {exc} | 可用设备: {_describe_framebuffer_candidates()}",
        )


def _clear_screen() -> None:
    """写入全黑画面（息屏用）。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    _write_fb(img)
    
def _render_error_screen(message: str) -> None:
    """在屏幕上直接显示异常信息，避免无提示黑屏。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), (28, 12, 12))
    draw = ImageDraw.Draw(img)
    font_title = _get_font(22)
    font_body = _get_font(16)
    font_time = _get_font(12)

    draw.text((12, 14), "LCD显示异常", fill=(255, 130, 130), font=font_title)
    y = 54
    for line in _format_error_lines(message):
        draw.text((12, y), line, fill=COLOR_TEXT, font=font_body)
        y += 24
    draw.text((12, HEIGHT - 22), datetime.now().strftime("%H:%M:%S"), fill=COLOR_LABEL, font=font_time)
    _write_fb(img)


# ---------- 调仓日计算 ----------

def _calc_rebalance_status() -> tuple[Optional[str], Optional[int]]:
    """计算下次调仓日期及剩余交易日。"""
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data import DataLoader, Storage

    rebalance_state = PaperStorage(
        root_path=get_paper_root()
    ).load_rebalance_state()
    if rebalance_state is None:
        return None, None

    last_rebalance_date = rebalance_state.get('last_rebalance_date')
    rebalance_freq = rebalance_state.get('rebalance_freq')
    if not last_rebalance_date or not rebalance_freq:
        return None, None

    try:
        rebalance_freq_int = int(rebalance_freq)
        loader = DataLoader(storage=Storage(root_path=get_data_root()))
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            return None, None

        trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()

        today_str = datetime.now().strftime("%Y%m%d")
        current_date = today_str if today_str in trade_dates else next(
            (d for d in reversed(trade_dates) if d <= today_str), None
        )
        if current_date is None:
            return None, None

        last_idx = trade_dates.index(last_rebalance_date)
        current_idx = trade_dates.index(current_date)
        next_idx = last_idx + rebalance_freq_int
        next_rebalance_date = trade_dates[next_idx] if next_idx < len(trade_dates) else None
        days_to_rebalance = max(next_idx - current_idx, 0)
        return next_rebalance_date, days_to_rebalance
    except Exception:
        return None, None


def _calc_days_to_rebalance() -> Optional[int]:
    """兼容旧调用，仅返回距下次调仓还剩多少交易日。"""
    _, days_to_rebalance = _calc_rebalance_status()
    return days_to_rebalance


# ---------- 图表数据获取 ----------

_cycle_chart_cache_lock = threading.Lock()
_cycle_chart_cache_scope_date: Optional[str] = None
_cycle_chart_cache: dict[tuple, dict] = {}


def _sync_cycle_chart_cache_scope(cache_scope_date: str) -> None:
    """按自然日维护周期图缓存作用域，跨天时自动清空。"""
    global _cycle_chart_cache_scope_date

    if _cycle_chart_cache_scope_date == cache_scope_date:
        return

    _cycle_chart_cache.clear()
    _cycle_chart_cache_scope_date = cache_scope_date


def _build_cycle_chart_cache_key(
    cache_scope_date: str,
    target_cycle_date: Optional[str],
    start_date: str,
    rebalance_freq: object,
    cash: object,
    positions: dict,
) -> tuple:
    """构建周期图当日缓存键，状态变化时自动失效。"""
    cash_float = _coerce_float(cash)
    if cash_float is None:
        cash_float = 0.0

    positions_signature = tuple(
        sorted(
            (
                ts_code,
                int(getattr(pos, 'shares', 0)),
                round(float(getattr(pos, 'buy_price', 0.0)), 6),
            )
            for ts_code, pos in positions.items()
        )
    )

    return (
        cache_scope_date,
        target_cycle_date or "",
        str(start_date),
        str(rebalance_freq),
        round(cash_float, 6),
        positions_signature,
    )


def _get_cached_cycle_chart_data(cache_key: tuple, cache_scope_date: str) -> Optional[dict]:
    """读取周期图当日缓存。"""
    with _cycle_chart_cache_lock:
        _sync_cycle_chart_cache_scope(cache_scope_date)
        cached = _cycle_chart_cache.get(cache_key)
        if cached is None:
            return None
        return copy.deepcopy(cached)


def _save_cycle_chart_data_cache(cache_key: tuple, cache_scope_date: str, chart_data: dict) -> None:
    """保存周期图当日缓存。"""
    with _cycle_chart_cache_lock:
        _sync_cycle_chart_cache_scope(cache_scope_date)
        _cycle_chart_cache[cache_key] = copy.deepcopy(chart_data)

def _fetch_cycle_chart_data() -> Optional[dict]:
    """获取持仓周期内的上证/深证/中证800指数和持仓组合涨跌幅数据。

    基于账户持仓状态 + TuShare daily API 计算每日组合市值，
    不依赖 NAV 记录（NAV 可能不完整）。

    Returns:
        dict: {
            'dates': list[str],          # 交易日期列表
            'index_pct': list[float],    # 上证指数累计涨跌幅(%)
            'shenzhen_pct': list[float], # 深证指数累计涨跌幅(%)
            'csi800_pct': list[float],   # 中证800累计涨跌幅(%)
            'portfolio_pct': list[float] # 持仓组合累计涨跌幅(%)
        }
        None: 数据不可用
    """
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data.tushare_client import TushareClient

    paper_storage = PaperStorage(
        root_path=get_paper_root(), verbose=False
    )

    # 获取上次调仓日期作为周期起点
    rebalance_state = paper_storage.load_rebalance_state()
    if rebalance_state is None:
        return None
    start_date = rebalance_state.get('last_rebalance_date')
    rebalance_freq = rebalance_state.get('rebalance_freq')
    if not start_date:
        return None

    # 获取账户持仓
    account_state = paper_storage.load_account_state()
    if account_state is None or not account_state.positions:
        return None

    positions = account_state.positions  # {ts_code: Position}
    cash = account_state.cash
    current_dt = datetime.now()
    today_str = current_dt.strftime("%Y%m%d")
    cache_scope_date = today_str
    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    cycle_cache_key = _build_cycle_chart_cache_key(
        cache_scope_date,
        target_cycle_date,
        start_date,
        rebalance_freq,
        cash,
        positions,
    )
    cached_chart_data = _get_cached_cycle_chart_data(cycle_cache_key, cache_scope_date)
    if cached_chart_data is not None:
        return cached_chart_data

    try:
        client = TushareClient(verbose=False)

        # 上证与深证指数日线（以此确定交易日序列）
        shanghai_df = client.query(
            "index_daily", ts_code=SHANGHAI_INDEX_CODE,
            start_date=start_date, end_date=today_str,
            fields="trade_date,close"
        )
        shenzhen_df = client.query(
            "index_daily", ts_code=SHENZHEN_INDEX_CODE,
            start_date=start_date, end_date=today_str,
            fields="trade_date,close"
        )
        if shanghai_df is None or shanghai_df.empty or shenzhen_df is None or shenzhen_df.empty:
            return None
        shanghai_df = shanghai_df.sort_values('trade_date').reset_index(drop=True)
        shenzhen_df = shenzhen_df.sort_values('trade_date').reset_index(drop=True)
        shanghai_close_map = dict(zip(shanghai_df['trade_date'], shanghai_df['close']))
        shenzhen_close_map = dict(zip(shenzhen_df['trade_date'], shenzhen_df['close']))
        trade_dates = [d for d in shanghai_df['trade_date'].tolist() if d in shenzhen_close_map]
        if len(trade_dates) < 1:
            return None

        csi800_close_map = _fetch_csi800_daily_close_map_akshare(start_date=start_date, end_date=today_str)
        if not csi800_close_map:
            return None
        trade_dates = [d for d in trade_dates if d in csi800_close_map]
        if len(trade_dates) < 1:
            return None

        # 逐股获取日线收盘价
        stock_closes: dict[str, dict[str, float]] = {}
        for ts_code in positions:
            df = client.query(
                "daily", ts_code=ts_code,
                start_date=start_date, end_date=today_str,
                fields="trade_date,close"
            )
            if df is not None and not df.empty:
                stock_closes[ts_code] = dict(zip(df['trade_date'], df['close']))
    except Exception:
        return None

    # 计算每日组合市值
    base_value: Optional[float] = None
    portfolio_pct: list[float] = []
    for d in trade_dates:
        market_value = 0.0
        for ts_code, pos in positions.items():
            closes = stock_closes.get(ts_code, {})
            price = closes.get(d, pos.buy_price)  # 停牌等无数据时用买入价
            market_value += price * pos.shares
        total_value = market_value + cash
        if base_value is None:
            base_value = total_value
        portfolio_pct.append((total_value / base_value - 1) * 100)

    # 上证/深证指数涨跌幅
    shanghai_base_close = shanghai_close_map[trade_dates[0]]
    shenzhen_base_close = shenzhen_close_map[trade_dates[0]]
    csi800_base_close = csi800_close_map[trade_dates[0]]
    index_pct = [(shanghai_close_map[d] / shanghai_base_close - 1) * 100 for d in trade_dates]
    shenzhen_pct = [(shenzhen_close_map[d] / shenzhen_base_close - 1) * 100 for d in trade_dates]
    csi800_pct = [(csi800_close_map[d] / csi800_base_close - 1) * 100 for d in trade_dates]

    chart_data = _build_cycle_chart_payload(
        dates=trade_dates,
        index_pct=index_pct,
        shenzhen_pct=shenzhen_pct,
        csi800_pct=csi800_pct,
        portfolio_pct=portfolio_pct,
        rebalance_freq=rebalance_freq,
        base_value=base_value,
    )

    if chart_data is not None and (
        target_cycle_date is None or _has_cycle_data_for_target(chart_data, target_cycle_date)
    ):
        _save_cycle_chart_data_cache(cycle_cache_key, cache_scope_date, chart_data)

    return chart_data


# ---------- 个股盈亏排名 ----------

def _fetch_realtime_holdings_snapshot() -> Optional[dict]:
    """获取当前持仓实时行情快照。"""
    from src.lazybull.paper import PaperStorage, PaperTradingRunner
    from src.lazybull.data.tushare_client import TushareClient

    runner = PaperTradingRunner(verbose=False)
    positions = runner.account.get_positions()
    cash = runner.account.get_cash()
    paper_storage = PaperStorage(
        root_path=get_paper_root(), verbose=False
    )
    config = paper_storage.load_config()
    initial_capital = (
        config.get('initial_capital', runner.account.initial_capital)
        if config else runner.account.initial_capital
    )
    annualized_return_func = getattr(runner.broker, '_calculate_annualized_return', None)
    if not callable(annualized_return_func):
        annualized_return_func = None

    snapshot = {
        'positions': positions,
        'cash': cash,
        'initial_capital': initial_capital,
        'current_date': datetime.now().strftime("%Y%m%d"),
        'annualized_return_func': annualized_return_func,
        'index_pct_map': {},
        'quotes': None,
    }

    if not positions:
        return snapshot

    ts_codes = list(positions.keys()) + [SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE]
    ts_codes_str = ','.join(dict.fromkeys(ts_codes))

    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(ts_codes_str)
    except Exception:
        return snapshot

    if rt_df is None or rt_df.empty:
        return snapshot

    snapshot['quotes'] = rt_df
    snapshot['index_pct_map'] = _extract_index_pct_map_from_quote_df(rt_df)
    return snapshot


def _build_realtime_portfolio_summary(snapshot: Optional[dict]) -> Optional[dict]:
    """基于实时快照构建持仓摘要，复用已获取的持仓行情。"""
    if snapshot is None:
        return None

    from paper_trade import build_realtime_portfolio_summary_from_quotes

    cash = _coerce_float(snapshot.get('cash'))
    initial_capital = _coerce_float(snapshot.get('initial_capital'))
    if cash is None or initial_capital is None:
        return None

    annualized_return_func = snapshot.get('annualized_return_func')
    if not callable(annualized_return_func):
        annualized_return_func = None

    return build_realtime_portfolio_summary_from_quotes(
        positions=snapshot.get('positions', {}),
        cash=cash,
        initial_capital=initial_capital,
        current_date=str(snapshot.get('current_date', datetime.now().strftime("%Y%m%d"))),
        rt_df=snapshot.get('quotes'),
        annualized_return_func=annualized_return_func,
    )


def _build_stock_rankings(snapshot: Optional[dict]) -> Optional[list]:
    """基于实时快照构建个股总盈亏排名（按持仓成本）。"""
    if snapshot is None:
        return None

    positions = snapshot.get('positions', {})
    rt_df = snapshot.get('quotes')
    if rt_df is None or rt_df.empty:
        return None

    stocks = []
    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', ''))
        name = str(row.get('NAME', ''))
        price = row.get('PRICE', None)
        if not ts_code or price is None:
            continue
        pos = positions.get(ts_code)
        if pos is None:
            continue
        current_price = _normalize_intraday_price(
            price,
            row.get('PRE_CLOSE', row.get('pre_close')),
            INTRADAY_STOCK_PCT_ABS_LIMIT,
        )
        if current_price is None or pos.buy_price <= 0:
            continue
        pnl_pct = (current_price - pos.buy_price) / pos.buy_price * 100
        code = ts_code.split('.')[0]
        stocks.append({'name': name[:4], 'code': code, 'pnl_pct': pnl_pct})

    if not stocks:
        return None

    stocks.sort(key=lambda x: x['pnl_pct'], reverse=True)
    return stocks


def _fetch_stock_rankings() -> Optional[list]:
    """获取个股盈亏排名（盈利前2 + 亏损前2）。

    Returns:
        list[dict]: 按盈亏排序的个股列表，每项包含:
            name: str       - 股票名称
            code: str       - 6位股票代码
            pnl_pct: float  - 盈亏比率(%)
        None: 数据不可用
    """
    return _build_stock_rankings(_fetch_realtime_holdings_snapshot())


# ---------- 日内图数据获取 ----------

def _extract_pct_from_quote_row(row) -> Optional[float]:
    """从单条实时行情记录中提取当日涨跌幅。"""
    if row is None:
        return None
    price = _coerce_float(row.get('PRICE', row.get('price')))
    pre_close = _coerce_float(row.get('PRE_CLOSE', row.get('pre_close')))
    if price is None or not np.isfinite(price) or price <= 0 or pre_close in (None, 0):
        return None
    return _sanitize_intraday_pct(
        (price / pre_close - 1) * 100,
        INTRADAY_INDEX_PCT_ABS_LIMIT,
    )


def _extract_index_pct_map_from_quote_df(rt_df) -> dict[str, float]:
    """从实时行情表中提取上证、深证与中证800当日涨跌幅。"""
    pct_map: dict[str, float] = {}
    if rt_df is None or rt_df.empty:
        return pct_map

    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', row.get('ts_code', '')))
        if ts_code not in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
            continue
        pct = _extract_pct_from_quote_row(row)
        if pct is not None:
            pct_map[ts_code] = pct

    return pct_map


def _extract_index_pct_from_akshare(df, target_code: str) -> Optional[float]:
    """从 akshare 指数现货表中提取指定指数当日涨跌幅。"""
    if df is None or df.empty:
        return None

    code_columns = ['代码', 'symbol', 'ts_code']
    matched = None
    target_aliases = {target_code}
    if target_code == SHANGHAI_INDEX_CODE:
        target_aliases.update({'000001', 'sh000001'})
    elif target_code == SHENZHEN_INDEX_CODE:
        target_aliases.update({'399001', 'sz399001'})
    elif target_code == CSI800_INDEX_CODE:
        target_aliases.update({'000906', 'sh000906'})
    for col in code_columns:
        if col not in df.columns:
            continue
        code_series = df[col].astype(str)
        mask = code_series.isin(target_aliases)
        if mask.any():
            matched = df.loc[mask].iloc[0]
            break

    if matched is None:
        return None

    pct = _coerce_float(matched.get('涨跌幅', matched.get('pct_chg')))
    if pct is not None:
        return _sanitize_intraday_pct(pct, INTRADAY_INDEX_PCT_ABS_LIMIT)

    price = _coerce_float(matched.get('最新价', matched.get('最新')))
    pre_close = _coerce_float(
        matched.get('昨收', matched.get('昨收盘', matched.get('pre_close')))
    )
    if price is None or not np.isfinite(price) or price <= 0 or pre_close in (None, 0):
        return None
    return _sanitize_intraday_pct(
        (price / pre_close - 1) * 100,
        INTRADAY_INDEX_PCT_ABS_LIMIT,
    )


def _fetch_realtime_index_pcts(snapshot: Optional[dict] = None) -> dict[str, float]:
    """获取上证、深证与中证800当日实时涨跌幅。"""
    pct_map: dict[str, float] = {}

    if snapshot is not None:
        snapshot_pct_map = snapshot.get('index_pct_map')
        if isinstance(snapshot_pct_map, dict):
            for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
                pct = _sanitize_intraday_pct(
                    snapshot_pct_map.get(code),
                    INTRADAY_INDEX_PCT_ABS_LIMIT,
                )
                if pct is not None:
                    pct_map[code] = pct
        if len(pct_map) < 3:
            pct_map.update(
                _extract_index_pct_map_from_quote_df(snapshot.get('quotes'))
            )
        if len(pct_map) == 3:
            return pct_map

    try:
        import akshare as ak  # type: ignore

        for getter_name in ('stock_zh_index_spot_em', 'stock_zh_index_spot_sina'):
            getter = getattr(ak, getter_name, None)
            if getter is None:
                continue
            df = getter()
            for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
                if code in pct_map:
                    continue
                pct = _extract_index_pct_from_akshare(df, code)
                if pct is not None:
                    pct_map[code] = pct
            if len(pct_map) == 3:
                break
    except Exception:
        pass

    return pct_map


def _fetch_csi800_daily_close_map_akshare(start_date: str, end_date: str) -> dict[str, float]:
    """使用 AKShare 获取中证800日线收盘价，返回 YYYYMMDD -> close。"""
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return {}

    index_hist_getter = getattr(ak, 'index_zh_a_hist', None)
    if callable(index_hist_getter):
        try:
            df = index_hist_getter(
                symbol='000906',
                period='daily',
                start_date=start_date,
                end_date=end_date,
            )
            close_map = _extract_daily_close_map_from_akshare_df(df)
            if close_map:
                return close_map
        except Exception:
            pass

    daily_getter = getattr(ak, 'stock_zh_index_daily_em', None)
    if callable(daily_getter):
        try:
            df = daily_getter(symbol='sh000906')
            close_map = _extract_daily_close_map_from_akshare_df(df)
            if close_map:
                return {
                    trade_date: close
                    for trade_date, close in close_map.items()
                    if start_date <= trade_date <= end_date
                }
        except Exception:
            pass

    return {}


def _extract_daily_close_map_from_akshare_df(df) -> dict[str, float]:
    """将 AKShare 指数日线 DataFrame 归一化为 YYYYMMDD -> close。"""
    if df is None or getattr(df, 'empty', True):
        return {}

    date_col = None
    close_col = None
    for candidate in ('日期', 'date', 'trade_date'):
        if candidate in df.columns:
            date_col = candidate
            break
    for candidate in ('收盘', 'close', '收盘价'):
        if candidate in df.columns:
            close_col = candidate
            break
    if date_col is None or close_col is None:
        return {}

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        raw_date = str(row.get(date_col, '')).strip()
        if not raw_date:
            continue
        try:
            trade_date = pd.to_datetime(raw_date).strftime('%Y%m%d')
        except Exception:
            raw_digits = ''.join(ch for ch in raw_date if ch.isdigit())
            if len(raw_digits) != 8:
                continue
            trade_date = raw_digits

        close = _coerce_float(row.get(close_col))
        if close is None or not np.isfinite(close) or close <= 0:
            continue
        result[trade_date] = close

    return result


def _compute_holdings_intraday_pct(snapshot: Optional[dict]) -> Optional[float]:
    """计算当前持仓股票相对昨收的实时涨跌幅（不含现金）。"""
    if snapshot is None:
        return None

    positions = snapshot.get('positions', {})
    rt_df = snapshot.get('quotes')
    if rt_df is None or rt_df.empty or not positions:
        return None

    quote_map = {}
    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', ''))
        if ts_code:
            quote_map[ts_code] = row

    current_value = 0.0
    prev_close_value = 0.0
    valid_count = 0
    for ts_code, pos in positions.items():
        row = quote_map.get(ts_code)
        if row is None:
            continue
        pre_close = _coerce_float(row.get('PRE_CLOSE', row.get('pre_close')))
        current_price = _normalize_intraday_price(
            row.get('PRICE', row.get('price')),
            pre_close,
            INTRADAY_STOCK_PCT_ABS_LIMIT,
        )
        if pre_close in (None, 0) or current_price is None:
            continue
        current_value += current_price * pos.shares
        prev_close_value += pre_close * pos.shares
        valid_count += 1

    if valid_count == 0 or prev_close_value <= 0:
        return None
    return _sanitize_intraday_pct(
        (current_value / prev_close_value - 1) * 100,
        INTRADAY_PORTFOLIO_PCT_ABS_LIMIT,
    )


def _build_intraday_chart(
    chart_data: Optional[dict],
    snapshot: Optional[dict],
    point_time: Optional[datetime] = None,
) -> Optional[dict]:
    """基于上证/深证/中证800实时涨跌与持仓股当日实时涨跌构建盘中图。"""
    current_time = point_time or _get_snapshot_quote_time(snapshot) or datetime.now()
    if snapshot is None or not _is_intraday_trading_time(current_time):
        return chart_data

    index_pct_map = _fetch_realtime_index_pcts(snapshot)
    holdings_pct = _compute_holdings_intraday_pct(snapshot)
    shanghai_pct = index_pct_map.get(SHANGHAI_INDEX_CODE)
    shenzhen_pct = index_pct_map.get(SHENZHEN_INDEX_CODE)
    csi800_pct = index_pct_map.get(CSI800_INDEX_CODE)
    if (
        shanghai_pct is None
        or shenzhen_pct is None
        or csi800_pct is None
        or holdings_pct is None
    ):
        return chart_data

    return _upsert_intraday_chart(
        chart_data,
        current_time,
        index_pct=shanghai_pct,
        shenzhen_pct=shenzhen_pct,
        portfolio_pct=holdings_pct,
        csi800_pct=csi800_pct,
    )


def _should_keep_realtime_completion_active(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断收盘后是否仍需继续补齐日内图最后一格。"""
    current_dt = now or datetime.now()
    if _is_realtime_quote_window(current_dt):
        return True
    if not _is_trade_day(current_dt, allow_load=True):
        return False
    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    if target_cycle_date != current_dt.strftime("%Y%m%d"):
        return False
    close_deadline = datetime.combine(current_dt.date(), INTRADAY_WINDOW_END) + timedelta(
        seconds=POST_CLOSE_INTRADAY_GRACE_SECONDS
    )
    if current_dt > close_deadline:
        return False
    if _has_cycle_data_for_target(cycle_chart_data, target_cycle_date):
        return False
    return not _is_intraday_chart_complete(intraday_chart_data, current_dt)


def _should_keep_morning_close_completion_active(
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断午休开始后是否仍需补齐上午 11:30 最后一格。"""
    current_dt = now or datetime.now()
    if not _is_trade_day(current_dt, allow_load=True):
        return False

    morning_close_dt = datetime.combine(current_dt.date(), A_SHARE_MORNING_CLOSE)
    grace_deadline = morning_close_dt + timedelta(seconds=MORNING_CLOSE_INTRADAY_GRACE_SECONDS)
    if not (morning_close_dt < current_dt <= grace_deadline):
        return False

    return not _is_morning_intraday_chart_complete(intraday_chart_data, current_dt)


def _refresh_display_state(
    state: "DisplayState",
    refresh_realtime: bool = False,
    refresh_cycle: bool = False,
) -> None:
    """按需刷新共享显示状态。"""
    with state.lock:
        state.is_updating = refresh_realtime or refresh_cycle

    holdings_snapshot = None
    latest_update_time: Optional[str] = None

    if refresh_realtime:
        try:
            holdings_snapshot = _fetch_realtime_holdings_snapshot()
        except Exception:
            holdings_snapshot = None

        try:
            summary = _build_realtime_portfolio_summary(holdings_snapshot)
            if summary is not None:
                with state.lock:
                    state.summary = summary
                latest_update_time = _format_quote_update_time(summary) or datetime.now().strftime("%H:%M")
        except Exception:
            pass

        try:
            with state.lock:
                current_intraday_chart = state.intraday_chart_data
            intraday_chart_data = _build_intraday_chart(
                current_intraday_chart,
                holdings_snapshot,
            )
            if intraday_chart_data is not None:
                with state.lock:
                    state.intraday_chart_data = intraday_chart_data
                _save_intraday_chart(intraday_chart_data)
        except Exception:
            pass

        try:
            ranks = _build_stock_rankings(holdings_snapshot)
            if ranks is not None:
                with state.lock:
                    state.stock_rankings = ranks
        except Exception:
            pass

    try:
        next_rebalance_date, days_to_rebalance = _calc_rebalance_status()
        with state.lock:
            state.next_rebalance_date = next_rebalance_date
            state.days_to_rebalance = days_to_rebalance
    except Exception:
        pass

    if refresh_cycle:
        try:
            cycle_chart_data = _fetch_cycle_chart_data()
            if cycle_chart_data is not None:
                with state.lock:
                    state.chart_data = cycle_chart_data
                if latest_update_time is None:
                    latest_update_time = datetime.now().strftime("%H:%M")
        except Exception:
            pass

    with state.lock:
        if latest_update_time is not None:
            state.update_time = latest_update_time
        state.is_updating = False


# ---------- 共享显示状态 ----------

class DisplayState:
    """数据线程与显示线程之间的共享状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.summary: Optional[dict] = None
        self.update_time: str = "--:--"
        self.is_updating: bool = False
        self.next_rebalance_date: Optional[str] = None
        self.days_to_rebalance: Optional[int] = None
        self.chart_data: Optional[dict] = None
        self.intraday_chart_data: Optional[dict] = _load_intraday_chart()
        self.stock_rankings: Optional[list] = None  # 个股盈亏排名
        self.cpu_usage_pct: float = 0.0
        self.memory_usage_pct: float = 0.0
        self.cpu_usage_sample: Optional[tuple[int, int]] = None
        self.usage_sampled_at: float = 0.0
        # 屏保偏移（仅数据行参与）
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.is_screen_on: bool = True


# ---------- 渲染逻辑 ----------

# 布局常量
HEADER_H = 34          # 顶栏高度（含时间、更新、调仓）
HEADER_TIME_FONT_SIZE = 15
HEADER_META_FONT_SIZE = 15
USAGE_BAR_H = 5
USAGE_BAR_MARGIN_X = 8
USAGE_BAR_CAP_W = 4
USAGE_BAR_SECTION_GAP = 4
USAGE_BAR_BOTTOM_GAP = 1
PANEL_MARGIN = 6       # 面板区左右外边距
PANEL_TOP = HEADER_H + 4  # 面板区顶部 y，给顶栏留出更大时间字号空间
PANEL_H = 140          # 面板区高度（去掉底栏后加大）
PANEL_GAP = 6          # 左右面板间距
PANEL_AREA_W = WIDTH - 2 * PANEL_MARGIN  # 面板总可用宽度 = 468
LEFT_W = int(PANEL_AREA_W * 0.575)        # 左面板宽度，略收窄给右侧排行更多空间
RIGHT_W = PANEL_AREA_W - LEFT_W - PANEL_GAP  # 右面板宽度
RIGHT_SUB_GAP = 4      # 右上/右下子面板间距
RIGHT_SUB_H = (PANEL_H - RIGHT_SUB_GAP) // 2  # 每个子面板高度 = 68
CHART_Y = PANEL_TOP + PANEL_H + 4  # 图表区起始 y
CHART_H = HEIGHT - CHART_Y          # 图表区高度（底部到屏幕边缘）


def _render(state: DisplayState) -> None:
    """将持仓摘要、个股排名和图表渲染到 PIL Image 并写入 framebuffer。

    布局：
      顶部状态栏（固定）
            左面板 55%（屏保偏移）：2行×3列
        行1: 持仓市值 | 浮盈率   | 持仓/仓位
        行2: 总资产   | 总盈亏率 | 年化收益
      右上面板（屏保偏移）：盈利 Top3（右对齐）
      右下面板（屏保偏移）：亏损 Top3（右对齐）
      图表区（固定）
      底部时间栏（固定）
    """
    cpu_usage_pct, memory_usage_pct = _refresh_system_usage_sample(state)

    with state.lock:
        summary = state.summary
        last_update_time = state.update_time
        is_updating = getattr(state, 'is_updating', False)
        next_rebalance_date = getattr(state, 'next_rebalance_date', None)
        days_to_rebalance = state.days_to_rebalance
        cycle_chart_data = state.chart_data
        intraday_chart_data = state.intraday_chart_data
        rankings = state.stock_rankings
        ox = state.offset_x
        oy = state.offset_y

    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(img)

    font_val = _get_font(24)   # 数值
    font_val_sm = _get_font(20) # 数值（小号，持仓/仓位用）
    font_label = _get_font(HEADER_META_FONT_SIZE) # 标签
    font_header_time = _get_font(HEADER_TIME_FONT_SIZE)
    font_rank = _get_font(15)  # 排名列表
    font_md = _get_font(20)    # 等待提示

    # ===== 顶部状态栏（固定）：时间 | 更新 | 下次调仓 + CPU/内存双血条 =====
    draw.rectangle([0, 0, WIDTH, HEADER_H], fill=COLOR_HEADER_BG)

    now = datetime.now()
    chart_data = _select_chart_data(cycle_chart_data, intraday_chart_data, now)
    time_str = _format_display_time(now)
    header_mid = "更新中..." if is_updating else f"更新:{last_update_time}"
    header_right = _format_rebalance_status(next_rebalance_date, days_to_rebalance)

    time_bbox = draw.textbbox((0, 0), time_str, font=font_header_time)
    time_h = time_bbox[3] - time_bbox[1]
    header_text_area_h = HEADER_H - USAGE_BAR_H - USAGE_BAR_BOTTOM_GAP - 1
    time_y = max(1, (header_text_area_h - time_h) // 2 - 2)
    meta_bbox = draw.textbbox((0, 0), header_mid, font=font_label)
    meta_h = meta_bbox[3] - meta_bbox[1]
    meta_y = max(1, (header_text_area_h - meta_h) // 2 - 1)

    draw.text((8, time_y), time_str, fill=COLOR_TEXT, font=font_header_time)
    # 居中
    mw = meta_bbox[2] - meta_bbox[0]
    draw.text(((WIDTH - mw) // 2, meta_y), header_mid, fill=COLOR_YELLOW, font=font_label)
    # 右对齐
    bbox_r = draw.textbbox((0, 0), header_right, font=font_label)
    rw = bbox_r[2] - bbox_r[0]
    draw.text((WIDTH - rw - 8, meta_y), header_right, fill=COLOR_YELLOW, font=font_label)
    _draw_system_usage_bar(draw, cpu_usage_pct, memory_usage_pct)

    # ===== 左面板：总览 2行×3列（参与屏保偏移）=====
    lp_x = PANEL_MARGIN + ox
    lp_y = PANEL_TOP + oy
    draw.rounded_rectangle(
        [lp_x, lp_y, lp_x + LEFT_W, lp_y + PANEL_H],
        radius=6, fill=COLOR_PANEL_LEFT
    )

    if summary is None:
        bbox_wait = draw.textbbox((0, 0), "等待数据...", font=font_md)
        ww = bbox_wait[2] - bbox_wait[0]
        draw.text((lp_x + (LEFT_W - ww) // 2, lp_y + PANEL_H // 2 - 12),
                  "等待数据...", fill=COLOR_LABEL, font=font_md)
    else:
        mkt_val = summary['market_value']
        total_ast = summary['total_assets']
        flt_pct = summary['float_pnl_pct']
        gain_pct = summary['total_pnl_pct']
        pos_count = summary['pos_count']
        ann_pct = summary['annual_return_pct']
        pos_ratio = int(mkt_val / total_ast * 100) if total_ast > 0 else 0

        col_w = LEFT_W // 3
        pad = 7
        row_h = (PANEL_H - 2 * pad) // 2
        annual_return_text = _fmt_pct(ann_pct)
        annual_return_font = _pick_fitting_font(
            draw,
            annual_return_text,
            preferred_size=24,
            min_size=16,
            max_width=col_w - pad - 4,
        )

        cells = [
            # (行, 列, 标签, 值, 颜色, 值字体)
            (0, 0, "持仓市值", _fmt_wan(mkt_val), COLOR_TEXT, font_val),
            (0, 1, "浮盈率", _fmt_pct(flt_pct), _pct_color(flt_pct), font_val),
            (0, 2, "持仓/仓位", f"{pos_count}/{pos_ratio}%", COLOR_TEXT, font_val_sm),
            (1, 0, "总资产", _fmt_wan(total_ast), COLOR_TEXT, font_val),
            (1, 1, "总盈亏率", _fmt_pct(gain_pct), _pct_color(gain_pct), font_val),
            (1, 2, "年化收益", annual_return_text, _pct_color(ann_pct), annual_return_font),
        ]
        content_h = 15 + 24  # 标签到数值间距 + 数值字体高度
        v_pad = (row_h - content_h) // 2  # 行内垂直居中偏移
        for r, c, label, value, color, vfont in cells:
            cx = lp_x + pad + c * col_w
            cy = lp_y + pad + r * row_h + v_pad
            draw.text((cx, cy), label, fill=COLOR_LABEL, font=font_label)
            draw.text((cx, cy + 15), value, fill=color, font=vfont)

        # 行间水平分隔线
        sep_y = lp_y + pad + row_h
        draw.line([(lp_x + pad, sep_y), (lp_x + LEFT_W - pad, sep_y)],
                  fill=COLOR_DIVIDER, width=1)

    # ===== 右面板：个股盈亏排名（参与屏保偏移）=====
    rp_x = PANEL_MARGIN + LEFT_W + PANEL_GAP + ox
    rp_y = PANEL_TOP + oy
    rp_right = rp_x + RIGHT_W  # 右边界（用于右对齐）

    # 右上子面板：盈利 Top3
    draw.rounded_rectangle(
        [rp_x, rp_y, rp_right, rp_y + RIGHT_SUB_H],
        radius=5, fill=COLOR_PANEL_RIGHT
    )
    # 右下子面板：亏损 Top3
    rp_y2 = rp_y + RIGHT_SUB_H + RIGHT_SUB_GAP
    draw.rounded_rectangle(
        [rp_x, rp_y2, rp_right, rp_y2 + RIGHT_SUB_H],
        radius=5, fill=COLOR_PANEL_RIGHT
    )

    pad = 6
    line_h = 20

    if not rankings or len(rankings) < 2:
        txt = "暂无排名"
        bbox_r = draw.textbbox((0, 0), txt, font=font_label)
        tw = bbox_r[2] - bbox_r[0]
        draw.text((rp_x + (RIGHT_W - tw) // 2, rp_y + RIGHT_SUB_H // 2 - 7),
                  txt, fill=COLOR_LABEL, font=font_label)
    else:
        top3 = rankings[:3]
        bottom3 = rankings[-3:] if len(rankings) >= 6 else rankings[len(top3):]
        # 避免重复（持仓少于6只时）
        top_codes = {s['code'] for s in top3}
        bottom3 = [s for s in bottom3 if s['code'] not in top_codes]

        def _draw_rank_items(items, panel_y):
            """右对齐绘制排名条目到指定子面板。"""
            y = panel_y + pad
            for s in items:
                pct_str = _fmt_pct(s['pnl_pct'])
                color = _pct_color(s['pnl_pct'])
                line = f"{s['name']} {s['code']} {pct_str}"
                bbox_l = draw.textbbox((0, 0), line, font=font_rank)
                lw = bbox_l[2] - bbox_l[0]
                draw.text((rp_right - pad - lw, y), line,
                          fill=color, font=font_rank)
                y += line_h

        _draw_rank_items(top3, rp_y)
        _draw_rank_items(bottom3, rp_y2)

    # ===== 图表区（固定）=====
    cycle_last_data_label = None
    if chart_data is not None and chart_data.get('mode') == 'cycle':
        cycle_last_data_label = _format_cycle_last_data_label(cycle_chart_data)
    _draw_chart(draw, chart_data, cycle_last_data_label)

    _write_fb(img)


def _draw_chart(
    draw: ImageDraw.ImageDraw,
    chart_data: Optional[dict],
    cycle_last_data_label: Optional[str] = None,
) -> None:
    """绘制持仓周期图或盘中图。"""
    chart_x = 10
    chart_w = WIDTH - 20
    font_xs = _get_font(11)

    # 图表区背景
    draw.rectangle([chart_x, CHART_Y, chart_x + chart_w, CHART_Y + CHART_H],
                   fill=COLOR_CHART_BG)

    if not chart_data:
        # 无数据提示
        txt = "暂无图表数据"
        bbox = draw.textbbox((0, 0), txt, font=_get_font(14))
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, CHART_Y + CHART_H // 2 - 8), txt,
                  fill=COLOR_LABEL, font=_get_font(14))
        return

    dates = list(chart_data.get('dates', []))
    idx_pct = list(chart_data.get('index_pct', []))
    sz_pct = list(chart_data.get('shenzhen_pct', []))
    csi800_pct = list(chart_data.get('csi800_pct', idx_pct))
    ptf_pct = list(chart_data.get('portfolio_pct', []))
    slot_indices = list(chart_data.get('slot_indices', range(len(idx_pct))))
    x_positions = chart_data.get('x_positions', slot_indices)
    if not isinstance(x_positions, list):
        x_positions = slot_indices
    n = min(
        len(dates),
        len(idx_pct),
        len(sz_pct),
        len(csi800_pct),
        len(ptf_pct),
        len(slot_indices),
        len(x_positions),
    )
    if n == 0:
        txt = "暂无图表数据"
        bbox = draw.textbbox((0, 0), txt, font=_get_font(14))
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, CHART_Y + CHART_H // 2 - 8), txt,
                  fill=COLOR_LABEL, font=_get_font(14))
        return

    dates = dates[:n]
    idx_pct = idx_pct[:n]
    sz_pct = sz_pct[:n]
    csi800_pct = csi800_pct[:n]
    ptf_pct = ptf_pct[:n]
    slot_indices = slot_indices[:n]
    x_positions = x_positions[:n]
    display_x_positions = _get_intraday_display_x_positions(
        chart_data,
        dates,
        slot_indices,
        x_positions,
    )
    slot_count = max(int(chart_data.get('slot_count', n)), 2)
    chart_mode = str(chart_data.get('mode', ''))

    if chart_mode == 'intraday':
        idx_pct = _smooth_intraday_series_for_display(idx_pct)
        sz_pct = _smooth_intraday_series_for_display(sz_pct)
        csi800_pct = _smooth_intraday_series_for_display(csi800_pct)
        ptf_pct = _smooth_intraday_series_for_display(ptf_pct)

    # Y轴范围
    all_vals = idx_pct + sz_pct + csi800_pct + ptf_pct
    y_min, y_max = _get_chart_y_range(all_vals)
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

    # 水平网格（3条）
    for i in range(1, 4):
        gy = cy + ch * i // 4
        draw.line([(cx + 1, gy), (cx + cw - 1, gy)],
                  fill=COLOR_CHART_GRID, width=1)

    _draw_intraday_break_marker(draw, chart_data, cx, cy, cw, ch, slot_count, font_xs)

    # Y轴标签（上/中/下）
    for val, align_top in [(y_max, True), ((y_max + y_min) / 2, False), (y_min, False)]:
        py = cy + ch - int((val - y_min) / y_range * ch)
        label = f"{val:+.1f}%"
        ty = py - 12 if align_top else py - 6
        draw.text((chart_x + 1, ty), label, fill=COLOR_LABEL, font=font_xs)

    # 绘制折线
    def _to_points(values):
        pts = []
        for x_position, v in zip(display_x_positions, values):
            px = cx + float(x_position) / max(slot_count - 1, 1) * cw
            py = cy + ch - (v - y_min) / y_range * ch
            pts.append((px, py))
        return pts

    idx_pts = _to_points(idx_pct)
    sz_pts = _to_points(sz_pct)
    csi800_pts = _to_points(csi800_pct)
    ptf_pts = _to_points(ptf_pct)

    _draw_chart_series(
        draw,
        [
            (idx_pts, COLOR_CHART_SHANGHAI),
            (sz_pts, COLOR_CHART_SHENZHEN),
            (ptf_pts, COLOR_CHART_HOLDINGS),
            (csi800_pts, COLOR_CHART_CSI800),
        ],
        cx,
        cy,
        cw,
        ch,
    )

    if y_min <= 0 <= y_max:
        _draw_zero_reference_line(draw, cx, cy, cw, ch, y_min, y_range, font_xs)

    # 图例 + 末尾数值
    def _short_legend_label(label: str) -> str:
        mapping = {
            '上证': '上',
            '深证': '深',
            '持仓': '持',
            '中证800': '中',
        }
        return mapping.get(label, label[:1] if label else '')

    def _draw_legend_item(x: int, label: str, color: tuple, value: str) -> int:
        draw.line([(x, ly + 6), (x + 9, ly + 6)], fill=color, width=2)
        label_x = x + 12
        draw.text((label_x, ly), label, fill=color, font=font_xs)
        bbox_label = draw.textbbox((0, 0), label, font=font_xs)
        value_x = label_x + (bbox_label[2] - bbox_label[0]) + 2
        draw.text((value_x, ly), value, fill=color, font=font_xs)
        bbox_value = draw.textbbox((0, 0), value, font=font_xs)
        return value_x + (bbox_value[2] - bbox_value[0]) + 8

    lx = cx + 6
    ly = CHART_Y + 2
    idx_last_str = f"{idx_pct[-1]:+.1f}%"
    sz_last_str = f"{sz_pct[-1]:+.1f}%"
    csi800_last_str = f"{csi800_pct[-1]:+.1f}%"
    ptf_last_str = f"{ptf_pct[-1]:+.1f}%"
    legend_x = _draw_legend_item(
        lx,
        _short_legend_label(chart_data.get('index_label', '上证')),
        COLOR_CHART_SHANGHAI,
        idx_last_str,
    )
    legend_x = _draw_legend_item(
        legend_x,
        _short_legend_label(chart_data.get('shenzhen_label', '深证')),
        COLOR_CHART_SHENZHEN,
        sz_last_str,
    )
    legend_x = _draw_legend_item(
        legend_x,
        _short_legend_label(chart_data.get('portfolio_label', '持仓')),
        COLOR_CHART_HOLDINGS,
        ptf_last_str,
    )
    _draw_legend_item(
        legend_x,
        _short_legend_label(chart_data.get('csi800_label', '中证800')),
        COLOR_CHART_CSI800,
        csi800_last_str,
    )

    if cycle_last_data_label:
        bbox_last = draw.textbbox((0, 0), cycle_last_data_label, font=font_xs)
        last_w = bbox_last[2] - bbox_last[0]
        draw.text(
            (chart_x + chart_w - last_w - 4, CHART_Y + 2),
            cycle_last_data_label,
            fill=COLOR_LABEL,
            font=font_xs,
        )

    # X轴：起止日期
    start_label = str(chart_data.get('x_start_label', dates[0]))
    end_label = str(chart_data.get('x_end_label', dates[-1]))
    draw.text((cx + 2, cy + ch + 1), start_label, fill=COLOR_LABEL, font=font_xs)
    bbox_end = draw.textbbox((0, 0), end_label, font=font_xs)
    ew = bbox_end[2] - bbox_end[0]
    draw.text((cx + cw - ew - 2, cy + ch + 1), end_label,
              fill=COLOR_LABEL, font=font_xs)


# ---------- 数据获取线程 ----------

def _data_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """按分频策略获取实时行情和图表数据，更新共享状态。

    启动时立即获取一次（非交易日也会返回最近一个交易日的收盘数据）。
    """
    _emit_diag_once("data_worker_start", "数据线程已启动")

    try:
        # 启动时立即获取一次（非交易日也能显示最近收盘数据）
        startup_dt = datetime.now()
        _refresh_display_state(state, refresh_realtime=True, refresh_cycle=True)
        last_realtime_refresh_at: Optional[datetime] = startup_dt
        last_realtime_session_key = _get_realtime_session_key(startup_dt)
        last_cycle_refresh_at: Optional[datetime] = startup_dt
        last_cycle_target_date = _get_target_cycle_data_date(startup_dt, allow_load=True)

        while not stop_event.is_set():
            with state.lock:
                current_cycle_chart = state.chart_data
                current_intraday_chart = state.intraday_chart_data
            wait_seconds = _get_data_worker_wait_seconds(
                cycle_chart_data=current_cycle_chart,
                intraday_chart_data=current_intraday_chart,
            )
            stop_event.wait(wait_seconds)
            if stop_event.is_set():
                break

            current_dt = datetime.now()
            with state.lock:
                current_cycle_chart = state.chart_data
                current_intraday_chart = state.intraday_chart_data
            refresh_policy = _get_refresh_policy(
                current_cycle_chart,
                intraday_chart_data=current_intraday_chart,
                now=current_dt,
            )
            refresh_realtime, realtime_session_key = _is_realtime_refresh_due(
                bool(refresh_policy['refresh_realtime']),
                last_realtime_refresh_at,
                last_realtime_session_key,
                current_dt,
            )
            refresh_cycle, cycle_target_date = _is_cycle_refresh_due(
                current_cycle_chart,
                bool(refresh_policy['refresh_cycle']),
                last_cycle_refresh_at,
                last_cycle_target_date,
                current_dt,
            )

            if refresh_cycle or refresh_realtime:
                _refresh_display_state(
                    state,
                    refresh_realtime=refresh_realtime,
                    refresh_cycle=refresh_cycle,
                )
                if refresh_realtime:
                    last_realtime_refresh_at = current_dt
                    last_realtime_session_key = realtime_session_key
                if refresh_cycle:
                    last_cycle_refresh_at = current_dt
                    last_cycle_target_date = cycle_target_date
    except Exception as exc:
        _emit_diag(f"数据线程异常退出: {type(exc).__name__}: {exc}")


# ---------- 显示刷新线程 ----------

def _display_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """每秒刷新画面，每 SCREENSAVER_INTERVAL 秒更新屏保偏移。

    23:00-6:00 自动息屏。
    """
    last_offset_time = 0.0
    _emit_diag_once("display_worker_start", "显示线程已启动")

    while not stop_event.is_set():
        try:
            hour = datetime.now().hour

            # ---- 息屏逻辑（23:00 - 6:00）----
            if hour >= 23 or hour < 6:
                _emit_diag_once(
                    "sleep_window_active",
                    f"当前命中自动息屏时段({hour:02d}:xx)，LCD 将保持黑屏直到 06:00",
                )
                if state.is_screen_on:
                    _clear_screen()
                    _set_backlight(0)
                    state.is_screen_on = False
                stop_event.wait(10)
                continue

            if not state.is_screen_on:
                _set_backlight(BACKLIGHT_BRIGHTNESS)
                with state.lock:
                    state.cpu_usage_pct = 0.0
                    state.memory_usage_pct = 0.0
                    state.cpu_usage_sample = None
                    state.usage_sampled_at = 0.0
                state.is_screen_on = True

            # ---- 屏保：每分钟随机偏移数据区 ----
            now_ts = time.monotonic()
            if now_ts - last_offset_time >= SCREENSAVER_INTERVAL:
                with state.lock:
                    state.offset_x = random.randint(-SCREENSAVER_RANGE_X, SCREENSAVER_RANGE_X)
                    state.offset_y = random.randint(-SCREENSAVER_RANGE_Y, SCREENSAVER_RANGE_Y)
                last_offset_time = now_ts

            # ---- 渲染（含实时时间）----
            _emit_diag_once("render_first_frame_start", "显示线程开始首帧渲染")
            _render(state)
            _emit_diag_once("render_first_frame_done", "显示线程已写出首帧")

            # ---- 每秒刷新 ----
            stop_event.wait(1)
        except Exception as exc:
            _render_error_screen(f"{type(exc).__name__}: {exc}")
            print(f"[3.5LCD_disp] 渲染异常: {type(exc).__name__}: {exc}", file=sys.stderr)
            stop_event.wait(2)


# ---------- 入口 ----------

def main() -> None:
    _emit_diag("主程序启动")
    try:
        setup_logger(log_level="WARNING")
        _emit_diag_once("logger_ready", "日志初始化完成")
        get_config()
        _emit_diag_once("config_ready", "配置加载完成")

        selected_fb = _resolve_framebuffer_path()
        _emit_diag_once(
            f"fb_target::{selected_fb}",
            f"当前 framebuffer 目标: {selected_fb} | 可用设备: {_describe_framebuffer_candidates()}",
        )

        _init_backlight()
        _emit_diag_once("backlight_phase_done", "背光初始化阶段完成")
        _render_bootstrap_screen("准备启动数据与显示线程")
        _emit_diag_once("bootstrap_screen_written", "已尝试写入启动测试页")

        state = DisplayState()
        stop_event = threading.Event()

        def _shutdown(sig, frame):  # noqa: ANN001
            _emit_diag(f"收到退出信号: {sig}", stderr=False)
            stop_event.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # 数据获取线程（盘中实时 2 分钟，周期图与补数 10 分钟按需）
        data_t = threading.Thread(target=_data_worker, args=(state, stop_event), daemon=True)
        data_t.start()

        # 显示刷新线程（每秒）
        disp_t = threading.Thread(target=_display_worker, args=(state, stop_event), daemon=True)
        disp_t.start()
        _emit_diag_once("threads_started", "数据线程和显示线程已启动")

        try:
            while not stop_event.is_set():
                time.sleep(1)
        finally:
            _clear_screen()
            _cleanup_backlight()
            _emit_diag("主程序退出")
    except Exception as exc:
        _emit_diag(f"主程序启动失败: {type(exc).__name__}: {exc}")
        raise


if __name__ == '__main__':
    main()
