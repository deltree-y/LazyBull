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
DEFAULT_FB_PATH = "/dev/fb1"
WIDTH, HEIGHT = 480, 320
REFRESH_INTERVAL = 600       # 数据刷新间隔（秒），10分钟
BACKLIGHT_PIN = 18           # 背光 GPIO 引脚（硬件 PWM）
BACKLIGHT_BRIGHTNESS = 10    # 背光亮度 0~100（默认40%，可按需调整）
SCREENSAVER_RANGE_X = 4      # 屏保水平偏移范围（±像素）
SCREENSAVER_RANGE_Y = 3      # 屏保垂直偏移范围（±像素）
SCREENSAVER_INTERVAL = 60    # 屏保偏移更新间隔（秒）
INTRADAY_WINDOW_START = dt_time(8, 30)
INTRADAY_WINDOW_END = dt_time(15, 30)
INTRADAY_SLOT_MINUTES = 10
INTRADAY_SLOT_COUNT = (
    ((INTRADAY_WINDOW_END.hour * 60 + INTRADAY_WINDOW_END.minute)
     - (INTRADAY_WINDOW_START.hour * 60 + INTRADAY_WINDOW_START.minute))
    // INTRADAY_SLOT_MINUTES
    + 1
)
INTRADAY_INDEX_PCT_ABS_LIMIT = 20.0
INTRADAY_PORTFOLIO_PCT_ABS_LIMIT = 35.0
INTRADAY_STOCK_PCT_ABS_LIMIT = 35.0
SHANGHAI_INDEX_CODE = "000001.SH"
SHENZHEN_INDEX_CODE = "399001.SZ"
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
COLOR_YELLOW = (255, 200, 50)      # 强调色
COLOR_ORANGE = (255, 150, 60)      # 橘黄色（持仓折线）
COLOR_CYAN = (70, 205, 255)        # 青蓝色（深证折线）
COLOR_DIVIDER = (60, 60, 80)       # 分隔线
COLOR_CHART_BG = (22, 22, 38)      # 图表背景
COLOR_CHART_GRID = (45, 45, 65)    # 图表网格线
COLOR_PANEL_LEFT = (25, 28, 48)    # 左面板背景（偏蓝）
COLOR_PANEL_RIGHT = (28, 35, 38)   # 右面板背景（偏青）
COLOR_CHART_SHANGHAI = COLOR_YELLOW
COLOR_CHART_SHENZHEN = COLOR_CYAN
COLOR_CHART_HOLDINGS = COLOR_ORANGE

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


def _get_diag_log_paths() -> list[Path]:
    """返回诊断日志落盘路径，优先项目目录，失败时兜底系统临时目录。"""
    primary = project_root / "data" / "paper" / "state" / DIAG_LOG_FILENAME
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
) -> Optional[dict]:
    """构建持仓周期图负载，并固定 x 轴槽位。"""
    point_count = min(len(dates), len(index_pct), len(shenzhen_pct), len(portfolio_pct))
    if point_count == 0:
        return None

    dates = dates[:point_count]
    index_pct = index_pct[:point_count]
    shenzhen_pct = shenzhen_pct[:point_count]
    portfolio_pct = portfolio_pct[:point_count]
    slot_count = _resolve_cycle_slot_count(rebalance_freq, point_count)

    return {
        'mode': 'cycle',
        'dates': dates,
        'index_pct': index_pct,
        'shenzhen_pct': shenzhen_pct,
        'portfolio_pct': portfolio_pct,
        'slot_indices': list(range(point_count)),
        'slot_count': slot_count,
        'x_start_label': _format_mmdd(dates[0]),
        'x_end_label': f"{slot_count}天" if slot_count > point_count else _format_mmdd(dates[-1]),
        'index_label': '上证',
        'shenzhen_label': '深证',
        'portfolio_label': '持仓',
        'base_value': base_value,
    }


def _get_intraday_slot_index(point_time: datetime) -> int:
    """将盘中时间映射到固定的 10 分钟槽位。"""
    start_minutes = INTRADAY_WINDOW_START.hour * 60 + INTRADAY_WINDOW_START.minute
    current_minutes = point_time.hour * 60 + point_time.minute
    slot_idx = (current_minutes - start_minutes) // INTRADAY_SLOT_MINUTES
    return max(0, min(slot_idx, INTRADAY_SLOT_COUNT - 1))


def _empty_intraday_chart(trade_date: str) -> dict:
    """创建新的日内图负载。"""
    return {
        'mode': 'intraday',
        'trade_date': trade_date,
        'dates': [],
        'raw_index_pct': [],
        'raw_shenzhen_pct': [],
        'raw_portfolio_pct': [],
        'index_pct': [],
        'shenzhen_pct': [],
        'portfolio_pct': [],
        'slot_indices': [],
        'slot_count': INTRADAY_SLOT_COUNT,
        'x_start_label': INTRADAY_WINDOW_START.strftime("%H:%M"),
        'x_end_label': INTRADAY_WINDOW_END.strftime("%H:%M"),
        'index_label': '上证',
        'shenzhen_label': '深证',
        'portfolio_label': '持仓',
    }


def _rebase_intraday_values(values: list[float]) -> list[float]:
    """将日内序列改写为相对首个有效点的变化量。"""
    if not values:
        return []

    base_value = values[0]
    rebased_values: list[float] = []
    for value in values:
        rebased = round(value - base_value, 6)
        rebased_values.append(0.0 if abs(rebased) < 1e-9 else rebased)
    return rebased_values


def _compose_intraday_chart(
    chart_data: dict,
    trade_date: str,
    dates: list[str],
    slot_indices: list[int],
    raw_index_values: list[float],
    raw_shenzhen_values: list[float],
    raw_portfolio_values: list[float],
) -> dict:
    """生成日内图负载，同时保留原始值和首点归零后的显示值。"""
    return {
        **chart_data,
        'trade_date': trade_date,
        'dates': dates,
        'raw_index_pct': raw_index_values,
        'raw_shenzhen_pct': raw_shenzhen_values,
        'raw_portfolio_pct': raw_portfolio_values,
        'index_pct': _rebase_intraday_values(raw_index_values),
        'shenzhen_pct': _rebase_intraday_values(raw_shenzhen_values),
        'portfolio_pct': _rebase_intraday_values(raw_portfolio_values),
        'slot_indices': slot_indices,
    }


def _upsert_intraday_chart(
    chart_data: Optional[dict],
    point_time: datetime,
    index_pct: float,
    shenzhen_pct: float,
    portfolio_pct: float,
) -> dict:
    """向固定槽位的日内图中追加或覆盖一个采样点。"""
    trade_date = point_time.strftime("%Y%m%d")
    if chart_data is None or chart_data.get('trade_date') != trade_date:
        chart_data = _empty_intraday_chart(trade_date)

    slot_indices = list(chart_data.get('slot_indices', []))
    raw_index_values = list(chart_data.get('raw_index_pct', chart_data.get('index_pct', [])))
    raw_shenzhen_values = list(
        chart_data.get('raw_shenzhen_pct', chart_data.get('shenzhen_pct', []))
    )
    raw_portfolio_values = list(
        chart_data.get('raw_portfolio_pct', chart_data.get('portfolio_pct', []))
    )
    dates = list(chart_data.get('dates', []))
    if (
        len(slot_indices) != len(raw_index_values)
        or len(slot_indices) != len(raw_shenzhen_values)
        or len(slot_indices) != len(raw_portfolio_values)
        or len(slot_indices) != len(dates)
    ):
        chart_data = _empty_intraday_chart(trade_date)
        slot_indices = []
        raw_index_values = []
        raw_shenzhen_values = []
        raw_portfolio_values = []
        dates = []
    slot_idx = _get_intraday_slot_index(point_time)
    point_label = point_time.strftime("%H:%M")

    if slot_indices and slot_idx == slot_indices[-1]:
        raw_index_values[-1] = index_pct
        raw_shenzhen_values[-1] = shenzhen_pct
        raw_portfolio_values[-1] = portfolio_pct
        dates[-1] = point_label
    elif slot_idx in slot_indices:
        replace_idx = slot_indices.index(slot_idx)
        raw_index_values[replace_idx] = index_pct
        raw_shenzhen_values[replace_idx] = shenzhen_pct
        raw_portfolio_values[replace_idx] = portfolio_pct
        dates[replace_idx] = point_label
    else:
        slot_indices.append(slot_idx)
        raw_index_values.append(index_pct)
        raw_shenzhen_values.append(shenzhen_pct)
        raw_portfolio_values.append(portfolio_pct)
        dates.append(point_label)

    return _compose_intraday_chart(
        chart_data,
        trade_date,
        dates,
        slot_indices,
        raw_index_values,
        raw_shenzhen_values,
        raw_portfolio_values,
    )


def _get_intraday_chart_state_dir() -> Path:
    """返回 3.5 寸 LCD 日内图持久化目录。"""
    return project_root / "data" / "paper" / "state" / INTRADAY_CHART_STATE_DIRNAME


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
    raw_portfolio = chart_data.get('raw_portfolio_pct', chart_data.get('portfolio_pct', []))
    raw_slots = chart_data.get('slot_indices', [])
    if not all(isinstance(items, list) for items in (raw_dates, raw_index, raw_shenzhen, raw_portfolio, raw_slots)):
        return normalized

    dedup_points: dict[int, tuple[str, float, float, float]] = {}
    for label, index_val, shenzhen_val, portfolio_val, slot_idx in zip(
        raw_dates, raw_index, raw_shenzhen, raw_portfolio, raw_slots
    ):
        try:
            slot_int = int(slot_idx)
        except (TypeError, ValueError):
            continue
        if slot_int < 0 or slot_int >= INTRADAY_SLOT_COUNT:
            continue
        index_float = _sanitize_intraday_pct(index_val, INTRADAY_INDEX_PCT_ABS_LIMIT)
        shenzhen_float = _sanitize_intraday_pct(shenzhen_val, INTRADAY_INDEX_PCT_ABS_LIMIT)
        portfolio_float = _sanitize_intraday_pct(
            portfolio_val,
            INTRADAY_PORTFOLIO_PCT_ABS_LIMIT,
        )
        if index_float is None or shenzhen_float is None or portfolio_float is None:
            continue
        dedup_points[slot_int] = (str(label), index_float, shenzhen_float, portfolio_float)

    for slot_int in sorted(dedup_points):
        label, index_float, shenzhen_float, portfolio_float = dedup_points[slot_int]
        normalized['dates'].append(label)
        normalized['raw_index_pct'].append(index_float)
        normalized['raw_shenzhen_pct'].append(shenzhen_float)
        normalized['raw_portfolio_pct'].append(portfolio_float)
        normalized['slot_indices'].append(slot_int)

    return _compose_intraday_chart(
        normalized,
        payload_trade_date,
        list(normalized['dates']),
        list(normalized['slot_indices']),
        list(normalized['raw_index_pct']),
        list(normalized['raw_shenzhen_pct']),
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
_trade_date_set_lock = threading.Lock()


def _load_trade_date_set() -> frozenset[str]:
    """加载交易日集合并缓存，失败时返回空集合。"""
    global _trade_date_set_cache

    if _trade_date_set_cache is not None:
        return _trade_date_set_cache

    with _trade_date_set_lock:
        if _trade_date_set_cache is not None:
            return _trade_date_set_cache

        result = frozenset()
        try:
            from src.lazybull.data import DataLoader, Storage

            loader = DataLoader(storage=Storage(root_path=str(project_root / "data")))
            trade_cal = loader.load_clean_trade_cal()
            if trade_cal is not None and not trade_cal.empty:
                result = frozenset(trade_cal.loc[trade_cal['is_open'] == 1, 'cal_date'].astype(str))
        except Exception:
            result = frozenset()

        _trade_date_set_cache = result
        return _trade_date_set_cache


def _get_cached_trade_date_set() -> Optional[frozenset[str]]:
    """返回已加载的交易日集合；若尚未加载则返回 None。"""
    return _trade_date_set_cache


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
    """判断是否处于盘中图显示与刷新窗口（交易日 8:30-15:30）。"""
    current_dt = now or datetime.now()
    if not _is_trade_day(current_dt):
        return False
    current_time = current_dt.time()
    return INTRADAY_WINDOW_START <= current_time <= INTRADAY_WINDOW_END


def _select_chart_data(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """根据当前时段选择显示周期图或盘中图。"""
    current_dt = now or datetime.now()
    if (
        intraday_chart_data is not None
        and intraday_chart_data.get('trade_date') == current_dt.strftime("%Y%m%d")
        and _is_intraday_chart_window(current_dt)
    ):
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
    return f"周期图最后数据日:{_format_mmdd(last_date)}"


def _get_refresh_policy(
    cycle_chart_data: Optional[dict], now: Optional[datetime] = None
) -> dict:
    """返回当前时段的数据刷新策略。"""
    current_dt = now or datetime.now()
    if _is_intraday_chart_window(current_dt):
        return {
            'refresh_cycle': True,
            'refresh_realtime': True,
        }

    if not _is_trade_day(current_dt):
        return {
            'refresh_cycle': False,
            'refresh_realtime': False,
        }

    if current_dt.time() <= INTRADAY_WINDOW_END:
        return {
            'refresh_cycle': False,
            'refresh_realtime': False,
        }

    has_today_cycle_data = _get_cycle_last_data_date(cycle_chart_data) == current_dt.strftime("%Y%m%d")
    return {
        'refresh_cycle': not has_today_cycle_data,
        'refresh_realtime': False,
    }


def _get_data_worker_wait_seconds(now: Optional[datetime] = None) -> float:
    """返回数据线程下次唤醒间隔。

    常规情况下按 10 分钟轮询；若即将跨过 15:30 的图表切换边界，
    则缩短本次等待，确保离开日内窗口后立刻补一次周期图刷新。
    """
    current_dt = now or datetime.now()
    wait_seconds = float(REFRESH_INTERVAL)

    if not _is_trade_day(current_dt):
        return wait_seconds

    if current_dt.time() > INTRADAY_WINDOW_END:
        return wait_seconds

    intraday_end_dt = datetime.combine(current_dt.date(), INTRADAY_WINDOW_END)
    seconds_to_boundary = (intraday_end_dt - current_dt).total_seconds() + 1.0
    if seconds_to_boundary <= 0:
        return 1.0
    return min(wait_seconds, max(1.0, seconds_to_boundary))


# ---------- 背光控制 ----------

_backlight_pwm = None


def _init_backlight() -> None:
    """初始化背光 PWM，设置为 BACKLIGHT_BRIGHTNESS 亮度。

    优先尝试 sysfs 接口，失败则使用 RPi.GPIO 硬件 PWM。
    在非树莓派环境静默跳过。
    """
    global _backlight_pwm

    # 方式1：sysfs 背光接口
    bl_path = "/sys/class/backlight/soc:backlight/brightness"
    max_path = "/sys/class/backlight/soc:backlight/max_brightness"
    try:
        with open(max_path, "r") as f:
            max_br = int(f.read().strip())
        target = int(max_br * BACKLIGHT_BRIGHTNESS / 100)
        with open(bl_path, "w") as f:
            f.write(str(target))
        _emit_diag_once("backlight_sysfs_ok", "背光初始化完成: 使用 sysfs 接口")
        return
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # 方式2：RPi.GPIO 硬件 PWM（GPIO 18）
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BACKLIGHT_PIN, GPIO.OUT)
        _backlight_pwm = GPIO.PWM(BACKLIGHT_PIN, 1000)  # 1kHz
        _backlight_pwm.start(BACKLIGHT_BRIGHTNESS)
        _emit_diag_once("backlight_pwm_ok", "背光初始化完成: 使用 GPIO PWM")
    except (ImportError, RuntimeError):
        _emit_diag_once(
            "backlight_unavailable",
            "背光初始化未生效: 未找到可用背光控制接口，若屏幕无背光请检查驱动/权限",
        )


def _set_backlight(brightness: int) -> None:
    """设置背光亮度（0~100）。"""
    # sysfs
    bl_path = "/sys/class/backlight/soc:backlight/brightness"
    max_path = "/sys/class/backlight/soc:backlight/max_brightness"
    try:
        with open(max_path, "r") as f:
            max_br = int(f.read().strip())
        with open(bl_path, "w") as f:
            f.write(str(int(max_br * brightness / 100)))
        return
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # PWM
    if _backlight_pwm is not None:
        _backlight_pwm.ChangeDutyCycle(brightness)


def _cleanup_backlight() -> None:
    """清理背光 PWM 资源。"""
    if _backlight_pwm is not None:
        _backlight_pwm.stop()
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.cleanup(BACKLIGHT_PIN)
    except (ImportError, RuntimeError):
        pass


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

def _calc_days_to_rebalance() -> Optional[int]:
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

def _fetch_cycle_chart_data() -> Optional[dict]:
    """获取持仓周期内的上证/深证指数和持仓组合涨跌幅数据。

    基于账户持仓状态 + TuShare daily API 计算每日组合市值，
    不依赖 NAV 记录（NAV 可能不完整）。

    Returns:
        dict: {
            'dates': list[str],          # 交易日期列表
            'index_pct': list[float],    # 上证指数累计涨跌幅(%)
            'shenzhen_pct': list[float], # 深证指数累计涨跌幅(%)
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
    rebalance_freq = rebalance_state.get('rebalance_freq')
    if not start_date:
        return None

    # 获取账户持仓
    account_state = paper_storage.load_account_state()
    if account_state is None or not account_state.positions:
        return None

    positions = account_state.positions  # {ts_code: Position}
    cash = account_state.cash
    today_str = datetime.now().strftime("%Y%m%d")

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
    index_pct = [(shanghai_close_map[d] / shanghai_base_close - 1) * 100 for d in trade_dates]
    shenzhen_pct = [(shenzhen_close_map[d] / shenzhen_base_close - 1) * 100 for d in trade_dates]

    return _build_cycle_chart_payload(
        dates=trade_dates,
        index_pct=index_pct,
        shenzhen_pct=shenzhen_pct,
        portfolio_pct=portfolio_pct,
        rebalance_freq=rebalance_freq,
        base_value=base_value,
    )


# ---------- 个股盈亏排名 ----------

def _fetch_realtime_holdings_snapshot() -> Optional[dict]:
    """获取当前持仓实时行情快照。"""
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data.tushare_client import TushareClient

    paper_storage = PaperStorage(
        root_path=str(project_root / "data" / "paper"), verbose=False
    )
    account_state = paper_storage.load_account_state()
    if account_state is None or not account_state.positions:
        return None

    positions = account_state.positions
    ts_codes_str = ','.join(positions.keys())

    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(ts_codes_str)
    except Exception:
        return None

    if rt_df is None or rt_df.empty:
        return None

    return {
        'positions': positions,
        'quotes': rt_df,
    }


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
        current_price = _coerce_float(price)
        if current_price is None:
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


def _fetch_realtime_index_pcts() -> dict[str, float]:
    """获取上证与深证指数当日实时涨跌幅。"""
    from src.lazybull.data.tushare_client import TushareClient

    pct_map: dict[str, float] = {}

    try:
        client = TushareClient(verbose=False)
        rt_df = client.get_realtime_quote(f"{SHANGHAI_INDEX_CODE},{SHENZHEN_INDEX_CODE}")
        if rt_df is not None and not rt_df.empty:
            for _, row in rt_df.iterrows():
                ts_code = str(row.get('TS_CODE', row.get('ts_code', '')))
                if ts_code not in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE):
                    continue
                pct = _extract_pct_from_quote_row(row)
                if pct is not None:
                    pct_map[ts_code] = pct
        if len(pct_map) == 2:
            return pct_map
    except Exception:
        pass

    try:
        import akshare as ak  # type: ignore

        for getter_name in ('stock_zh_index_spot_em', 'stock_zh_index_spot_sina'):
            getter = getattr(ak, getter_name, None)
            if getter is None:
                continue
            df = getter()
            for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE):
                if code in pct_map:
                    continue
                pct = _extract_index_pct_from_akshare(df, code)
                if pct is not None:
                    pct_map[code] = pct
            if len(pct_map) == 2:
                break
    except Exception:
        pass

    return pct_map


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
    """基于上证/深证实时涨跌与持仓股当日实时涨跌构建盘中图。"""
    if snapshot is None:
        return chart_data

    index_pct_map = _fetch_realtime_index_pcts()
    holdings_pct = _compute_holdings_intraday_pct(snapshot)
    shanghai_pct = index_pct_map.get(SHANGHAI_INDEX_CODE)
    shenzhen_pct = index_pct_map.get(SHENZHEN_INDEX_CODE)
    if (
        shanghai_pct is None
        or shenzhen_pct is None
        or holdings_pct is None
    ):
        return chart_data

    current_time = point_time or datetime.now()
    return _upsert_intraday_chart(
        chart_data,
        current_time,
        shanghai_pct,
        shenzhen_pct,
        holdings_pct,
    )


# ---------- 共享显示状态 ----------

class DisplayState:
    """数据线程与显示线程之间的共享状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.summary: Optional[dict] = None
        self.update_time: str = "--:--"
        self.days_to_rebalance: Optional[int] = None
        self.chart_data: Optional[dict] = None
        self.intraday_chart_data: Optional[dict] = _load_intraday_chart()
        self.stock_rankings: Optional[list] = None  # 个股盈亏排名
        # 屏保偏移（仅数据行参与）
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.is_screen_on: bool = True


# ---------- 渲染逻辑 ----------

# 布局常量
HEADER_H = 34          # 顶栏高度（含时间、更新、调仓）
HEADER_TIME_FONT_SIZE = 15
HEADER_META_FONT_SIZE = 15
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
    with state.lock:
        summary = state.summary
        last_update_time = state.update_time
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

    # ===== 顶部状态栏（固定）：时间 | 更新 | 距调仓 =====
    draw.rectangle([0, 0, WIDTH, HEADER_H], fill=COLOR_HEADER_BG)

    now = datetime.now()
    chart_data = _select_chart_data(cycle_chart_data, intraday_chart_data, now)
    time_str = _format_display_time(now)
    days_str = "--" if days_to_rebalance is None else f"{days_to_rebalance}天"
    header_mid = f"更新:{last_update_time}"
    header_right = f"待调仓:{days_str}"

    time_bbox = draw.textbbox((0, 0), time_str, font=font_header_time)
    time_h = time_bbox[3] - time_bbox[1]
    time_y = (HEADER_H - time_h) // 2 - 1
    meta_bbox = draw.textbbox((0, 0), header_mid, font=font_label)
    meta_h = meta_bbox[3] - meta_bbox[1]
    meta_y = (HEADER_H - meta_h) // 2

    draw.text((8, time_y), time_str, fill=COLOR_TEXT, font=font_header_time)
    # 居中
    mw = meta_bbox[2] - meta_bbox[0]
    draw.text(((WIDTH - mw) // 2, meta_y), header_mid, fill=COLOR_YELLOW, font=font_label)
    # 右对齐
    bbox_r = draw.textbbox((0, 0), header_right, font=font_label)
    rw = bbox_r[2] - bbox_r[0]
    draw.text((WIDTH - rw - 8, meta_y), header_right, fill=COLOR_YELLOW, font=font_label)

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
        cells = [
            # (行, 列, 标签, 值, 颜色, 值字体)
            (0, 0, "持仓市值", _fmt_wan(mkt_val), COLOR_TEXT, font_val),
            (0, 1, "浮盈率", _fmt_pct(flt_pct), _pct_color(flt_pct), font_val),
            (0, 2, "持仓/仓位", f"{pos_count}/{pos_ratio}%", COLOR_TEXT, font_val_sm),
            (1, 0, "总资产", _fmt_wan(total_ast), COLOR_TEXT, font_val),
            (1, 1, "总盈亏率", _fmt_pct(gain_pct), _pct_color(gain_pct), font_val),
            (1, 2, "年化收益", _fmt_pct(ann_pct), _pct_color(ann_pct), font_val),
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
    ptf_pct = list(chart_data.get('portfolio_pct', []))
    slot_indices = list(chart_data.get('slot_indices', range(len(idx_pct))))
    n = min(len(dates), len(idx_pct), len(sz_pct), len(ptf_pct), len(slot_indices))
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
    ptf_pct = ptf_pct[:n]
    slot_indices = slot_indices[:n]
    slot_count = max(int(chart_data.get('slot_count', n)), 2)

    # Y轴范围
    all_vals = idx_pct + sz_pct + ptf_pct
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
        for slot_idx, v in zip(slot_indices, values):
            px = cx + int(slot_idx / max(slot_count - 1, 1) * cw)
            py = cy + ch - int((v - y_min) / y_range * ch)
            pts.append((px, py))
        return pts

    idx_pts = _to_points(idx_pct)
    sz_pts = _to_points(sz_pct)
    ptf_pts = _to_points(ptf_pct)

    for i in range(n - 1):
        draw.line([idx_pts[i], idx_pts[i + 1]], fill=COLOR_CHART_SHANGHAI, width=2)
    for i in range(n - 1):
        draw.line([sz_pts[i], sz_pts[i + 1]], fill=COLOR_CHART_SHENZHEN, width=2)
    for i in range(n - 1):
        draw.line([ptf_pts[i], ptf_pts[i + 1]], fill=COLOR_CHART_HOLDINGS, width=2)
    if idx_pts:
        px, py = idx_pts[-1]
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=COLOR_CHART_SHANGHAI)
    if sz_pts:
        px, py = sz_pts[-1]
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=COLOR_CHART_SHENZHEN)
    if ptf_pts:
        px, py = ptf_pts[-1]
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=COLOR_CHART_HOLDINGS)

    # 图例 + 末尾数值
    def _draw_legend_item(x: int, label: str, color: tuple, value: str) -> int:
        draw.line([(x, ly + 6), (x + 14, ly + 6)], fill=color, width=2)
        label_x = x + 18
        draw.text((label_x, ly), label, fill=color, font=font_xs)
        bbox_label = draw.textbbox((0, 0), label, font=font_xs)
        value_x = label_x + (bbox_label[2] - bbox_label[0]) + 4
        draw.text((value_x, ly), value, fill=color, font=font_xs)
        bbox_value = draw.textbbox((0, 0), value, font=font_xs)
        return value_x + (bbox_value[2] - bbox_value[0]) + 16

    lx = cx + 6
    ly = CHART_Y + 2
    idx_last_str = f"{idx_pct[-1]:+.1f}%"
    sz_last_str = f"{sz_pct[-1]:+.1f}%"
    ptf_last_str = f"{ptf_pct[-1]:+.1f}%"
    legend_x = _draw_legend_item(
        lx,
        chart_data.get('index_label', '上证'),
        COLOR_CHART_SHANGHAI,
        idx_last_str,
    )
    legend_x = _draw_legend_item(
        legend_x,
        chart_data.get('shenzhen_label', '深证'),
        COLOR_CHART_SHENZHEN,
        sz_last_str,
    )
    _draw_legend_item(
        legend_x,
        chart_data.get('portfolio_label', '持仓'),
        COLOR_CHART_HOLDINGS,
        ptf_last_str,
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
    """每 REFRESH_INTERVAL 秒获取一次实时行情和图表数据，更新共享状态。

    启动时立即获取一次（非交易日也会返回最近一个交易日的收盘数据）。
    """
    _emit_diag_once("data_worker_start", "数据线程已启动")

    try:
        from paper_trade import get_realtime_portfolio_summary

        def _fetch_data(refresh_realtime: bool = True) -> None:
            """获取行情、调仓天数、图表数据和个股排名，更新共享状态。"""
            summary = None
            cycle_chart_data = None
            holdings_snapshot = None

            if refresh_realtime:
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
                cycle_chart_data = _fetch_cycle_chart_data()
                if cycle_chart_data is not None:
                    with state.lock:
                        state.chart_data = cycle_chart_data
            except Exception:
                pass

            if refresh_realtime:
                try:
                    holdings_snapshot = _fetch_realtime_holdings_snapshot()
                except Exception:
                    holdings_snapshot = None

            if refresh_realtime and holdings_snapshot is not None:
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

            if refresh_realtime:
                try:
                    ranks = _build_stock_rankings(holdings_snapshot)
                    if ranks is not None:
                        with state.lock:
                            state.stock_rankings = ranks
                except Exception:
                    pass

        # 启动时立即获取一次（非交易日也能显示最近收盘数据）
        _fetch_data(refresh_realtime=True)

        while not stop_event.is_set():
            wait_seconds = _get_data_worker_wait_seconds()
            stop_event.wait(wait_seconds)
            if stop_event.is_set():
                break

            # 周期图始终按 10 分钟刷新；实时行情只在盘中刷新
            with state.lock:
                current_cycle_chart = state.chart_data
            refresh_policy = _get_refresh_policy(current_cycle_chart)
            if refresh_policy['refresh_cycle'] or refresh_policy['refresh_realtime']:
                _fetch_data(refresh_realtime=bool(refresh_policy['refresh_realtime']))
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

        # 数据获取线程（10分钟间隔）
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
