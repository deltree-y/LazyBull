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
from functools import lru_cache
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
BACKLIGHT_PIN = 18           # 背光 GPIO 引脚（硬件 PWM）
BACKLIGHT_BRIGHTNESS = 20    # 背光亮度 0~100（默认40%，可按需调整）
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
SHANGHAI_INDEX_CODE = "000001.SH"
SHENZHEN_INDEX_CODE = "399001.SZ"
INTRADAY_CHART_STATE_DIRNAME = "respi_35lcd_intraday"

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
        'index_pct': [],
        'shenzhen_pct': [],
        'portfolio_pct': [],
        'slot_indices': [],
        'slot_count': INTRADAY_SLOT_COUNT,
        'x_start_label': INTRADAY_WINDOW_START.strftime("%H:%M"),
        'x_end_label': INTRADAY_WINDOW_END.strftime("%H:%M"),
        'index_label': '上证日内',
        'shenzhen_label': '深证日内',
        'portfolio_label': '持仓当日',
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
    index_values = list(chart_data.get('index_pct', []))
    shenzhen_values = list(chart_data.get('shenzhen_pct', []))
    portfolio_values = list(chart_data.get('portfolio_pct', []))
    dates = list(chart_data.get('dates', []))
    if len(slot_indices) != len(index_values) or len(slot_indices) != len(shenzhen_values) or len(slot_indices) != len(portfolio_values) or len(slot_indices) != len(dates):
        chart_data = _empty_intraday_chart(trade_date)
        slot_indices = []
        index_values = []
        shenzhen_values = []
        portfolio_values = []
        dates = []
    slot_idx = _get_intraday_slot_index(point_time)
    point_label = point_time.strftime("%H:%M")

    if slot_indices and slot_idx == slot_indices[-1]:
        index_values[-1] = index_pct
        shenzhen_values[-1] = shenzhen_pct
        portfolio_values[-1] = portfolio_pct
        dates[-1] = point_label
    elif slot_idx in slot_indices:
        replace_idx = slot_indices.index(slot_idx)
        index_values[replace_idx] = index_pct
        shenzhen_values[replace_idx] = shenzhen_pct
        portfolio_values[replace_idx] = portfolio_pct
        dates[replace_idx] = point_label
    else:
        slot_indices.append(slot_idx)
        index_values.append(index_pct)
        shenzhen_values.append(shenzhen_pct)
        portfolio_values.append(portfolio_pct)
        dates.append(point_label)

    return {
        **chart_data,
        'trade_date': trade_date,
        'dates': dates,
        'index_pct': index_values,
        'shenzhen_pct': shenzhen_values,
        'portfolio_pct': portfolio_values,
        'slot_indices': slot_indices,
    }


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
    raw_index = chart_data.get('index_pct', [])
    raw_shenzhen = chart_data.get('shenzhen_pct', [])
    raw_portfolio = chart_data.get('portfolio_pct', [])
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
        index_float = _coerce_float(index_val)
        shenzhen_float = _coerce_float(shenzhen_val)
        portfolio_float = _coerce_float(portfolio_val)
        if index_float is None or shenzhen_float is None or portfolio_float is None:
            continue
        dedup_points[slot_int] = (str(label), index_float, shenzhen_float, portfolio_float)

    for slot_int in sorted(dedup_points):
        label, index_float, shenzhen_float, portfolio_float = dedup_points[slot_int]
        normalized['dates'].append(label)
        normalized['index_pct'].append(index_float)
        normalized['shenzhen_pct'].append(shenzhen_float)
        normalized['portfolio_pct'].append(portfolio_float)
        normalized['slot_indices'].append(slot_int)

    return normalized


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


@lru_cache(maxsize=1)
def _load_trade_date_set() -> frozenset[str]:
    """加载交易日集合并缓存，失败时返回空集合。"""
    from src.lazybull.data import DataLoader, Storage

    try:
        loader = DataLoader(storage=Storage(root_path=str(project_root / "data")))
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None or trade_cal.empty:
            return frozenset()
        return frozenset(trade_cal.loc[trade_cal['is_open'] == 1, 'cal_date'].astype(str))
    except Exception:
        return frozenset()


def _is_trade_day(now: Optional[datetime] = None) -> bool:
    """判断当前日期是否为交易日。"""
    current_dt = now or datetime.now()
    trade_dates = _load_trade_date_set()
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


def _get_refresh_policy(now: Optional[datetime] = None) -> dict:
    """返回当前时段的数据刷新策略。"""
    current_dt = now or datetime.now()
    return {
        'refresh_cycle': True,
        'refresh_realtime': _is_intraday_chart_window(current_dt),
    }


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
    except (ImportError, RuntimeError):
        pass  # 非树莓派环境，静默跳过


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
    if price is None or pre_close in (None, 0):
        return None
    return (price / pre_close - 1) * 100


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
        return pct

    price = _coerce_float(matched.get('最新价', matched.get('最新')))
    pre_close = _coerce_float(
        matched.get('昨收', matched.get('昨收盘', matched.get('pre_close')))
    )
    if price is None or pre_close in (None, 0):
        return None
    return (price / pre_close - 1) * 100


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
        current_price = _coerce_float(row.get('PRICE', row.get('price')))
        pre_close = _coerce_float(row.get('PRE_CLOSE', row.get('pre_close')))
        if pre_close in (None, 0):
            continue
        if current_price is None:
            current_price = pre_close
        current_value += current_price * pos.shares
        prev_close_value += pre_close * pos.shares
        valid_count += 1

    if valid_count == 0 or prev_close_value <= 0:
        return None
    return (current_value / prev_close_value - 1) * 100


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
    if shanghai_pct is None or shenzhen_pct is None or holdings_pct is None:
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
HEADER_H = 30          # 顶栏高度（含时间、更新、调仓）
PANEL_MARGIN = 6       # 面板区左右外边距
PANEL_TOP = 34         # 面板区顶部 y
PANEL_H = 140          # 面板区高度（去掉底栏后加大）
PANEL_GAP = 6          # 左右面板间距
PANEL_AREA_W = WIDTH - 2 * PANEL_MARGIN  # 面板总可用宽度 = 468
LEFT_W = int(PANEL_AREA_W * 0.60)        # 左面板宽度 ≈ 280
RIGHT_W = PANEL_AREA_W - LEFT_W - PANEL_GAP  # 右面板宽度
RIGHT_SUB_GAP = 4      # 右上/右下子面板间距
RIGHT_SUB_H = (PANEL_H - RIGHT_SUB_GAP) // 2  # 每个子面板高度 = 68
CHART_Y = PANEL_TOP + PANEL_H + 4  # 图表区起始 y
CHART_H = HEIGHT - CHART_Y          # 图表区高度（底部到屏幕边缘）


def _render(state: DisplayState) -> None:
    """将持仓摘要、个股排名和图表渲染到 PIL Image 并写入 framebuffer。

    布局：
      顶部状态栏（固定）
      左面板 60%（屏保偏移）：2行×3列
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
    font_label = _get_font(13) # 标签
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

    hy = (HEADER_H - 13) // 2  # 垂直居中（字体13px）
    draw.text((8, hy), time_str, fill=COLOR_TEXT, font=font_label)
    # 居中
    bbox_m = draw.textbbox((0, 0), header_mid, font=font_label)
    mw = bbox_m[2] - bbox_m[0]
    draw.text(((WIDTH - mw) // 2, hy), header_mid, fill=COLOR_YELLOW, font=font_label)
    # 右对齐
    bbox_r = draw.textbbox((0, 0), header_right, font=font_label)
    rw = bbox_r[2] - bbox_r[0]
    draw.text((WIDTH - rw - 8, hy), header_right, fill=COLOR_YELLOW, font=font_label)

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
    _draw_chart(draw, chart_data)

    _write_fb(img)


def _draw_chart(draw: ImageDraw.ImageDraw, chart_data: Optional[dict]) -> None:
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
        stop_event.wait(REFRESH_INTERVAL)
        if stop_event.is_set():
            break

        # 周期图始终按 10 分钟刷新；实时行情只在盘中刷新
        refresh_policy = _get_refresh_policy()
        if refresh_policy['refresh_cycle'] or refresh_policy['refresh_realtime']:
            _fetch_data(refresh_realtime=bool(refresh_policy['refresh_realtime']))


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
        _render(state)

        # ---- 每秒刷新 ----
        stop_event.wait(1)


# ---------- 入口 ----------

def main() -> None:
    setup_logger(log_level="WARNING")
    get_config()

    _init_backlight()

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
        _cleanup_backlight()


if __name__ == '__main__':
    main()
