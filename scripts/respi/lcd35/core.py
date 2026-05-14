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
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ---------- 路径设置 ----------
project_root = Path(__file__).parent.parent.parent
scripts_dir = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# ---------- 项目日志 ----------
from src.lazybull.common.logger import setup_logger  # noqa: E402
from src.lazybull.common.config import (  # noqa: E402
    get_config,
    get_data_root,
    get_paper_root,
    get_shenwan_level,
)
from src.lazybull.portfolio.industry_constraint import load_industry_mapping  # noqa: E402
from scripts.respi.set_backlight import cleanup_backlight_state as _cleanup_backlight_state_helper  # noqa: E402
from scripts.respi.set_backlight import get_pwm_hardware_note as _get_pwm_hardware_note_helper  # noqa: E402
from scripts.respi.set_backlight import set_backlight as _set_backlight_helper  # noqa: E402
from scripts.respi.set_backlight import update_pwm_backlight_state as _update_pwm_backlight_state_helper  # noqa: E402


def _resolve_realtime_snapshot_timeout_seconds() -> float:
    """解析实时快照总超时秒数，并限制最小值避免过短截断。"""
    raw = os.getenv("LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS")
    if raw is not None and str(raw).strip() != "":
        try:
            parsed = float(str(raw).strip())
            # 低于 30 秒会频繁中断慢源抓取，导致每轮都重来。
            return max(parsed, 30.0)
        except (TypeError, ValueError):
            pass
    return 45.0


def _resolve_efinance_connect_timeout_seconds() -> float:
    """解析 efinance 连接超时。"""
    raw = os.getenv("LAZYBULL_EFINANCE_CONNECT_TIMEOUT_SECONDS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(float(str(raw).strip()), 3.0)
        except (TypeError, ValueError):
            pass
    return 8.0


def _resolve_efinance_read_timeout_seconds() -> float:
    """解析 efinance 读取超时，默认拉长到 30 秒。"""
    raw = os.getenv("LAZYBULL_EFINANCE_READ_TIMEOUT_SECONDS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(float(str(raw).strip()), 10.0)
        except (TypeError, ValueError):
            pass
    return 30.0


def _resolve_realtime_index_async_timeout_seconds() -> float:
    """解析后台指数抓取超时。"""
    raw = os.getenv("LAZYBULL_REALTIME_INDEX_ASYNC_TIMEOUT_SECONDS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(float(str(raw).strip()), 10.0)
        except (TypeError, ValueError):
            pass
    return 60.0

# ---------- 常量 ----------
DEFAULT_FB_PATH = "/dev/fb1"
WIDTH, HEIGHT = 480, 320
REFRESH_INTERVAL = 600       # 周期图/非交易时段补数间隔（秒），10分钟
REALTIME_REFRESH_INTERVAL = 180  # 盘中摘要/排行/日内图刷新间隔（秒，从300缩短到120以应对代理延迟）
REALTIME_SNAPSHOT_TIMEOUT_SECONDS = _resolve_realtime_snapshot_timeout_seconds()
try:
    REALTIME_SNAPSHOT_CACHE_MAX_AGE_SECONDS = float(
        os.getenv("LAZYBULL_REALTIME_SNAPSHOT_CACHE_MAX_AGE_SECONDS", "1800")
    )
except (TypeError, ValueError):
    REALTIME_SNAPSHOT_CACHE_MAX_AGE_SECONDS = 1800.0
if REALTIME_SNAPSHOT_CACHE_MAX_AGE_SECONDS < 60.0:
    REALTIME_SNAPSHOT_CACHE_MAX_AGE_SECONDS = 60.0
REALTIME_INTRADAY_TIMEOUT_SECONDS = 20.0  # 单次盘中图构建超时（秒）
REALTIME_INDEX_ASYNC_TIMEOUT_SECONDS = _resolve_realtime_index_async_timeout_seconds()
REALTIME_RETRY_WAIT_SECONDS = 15.0  # 实时抓取失败且无缓存时的快速重试间隔（秒）
EFINANCE_RETRY_COUNT = 1  # efinance 失败后的重试次数（总尝试次数=1+重试次数）
EFINANCE_RETRY_MIN_INTERVAL_SECONDS = 2.0  # efinance 重试最小间隔（秒）
EFINANCE_CONNECT_TIMEOUT_SECONDS = _resolve_efinance_connect_timeout_seconds()
EFINANCE_READ_TIMEOUT_SECONDS = _resolve_efinance_read_timeout_seconds()
EFINANCE_USER_AGENT = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1"  # 伪装手机UA避免风控
UPDATE_STUCK_RESET_SECONDS = 30.0  # 顶栏更新状态强制脱困阈值（秒）
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
CHART_PAGE_CHART_SECONDS = 30.0
CHART_PAGE_INDUSTRY_SECONDS = 30.0
CHART_PAGE_CYCLE_SECONDS = CHART_PAGE_CHART_SECONDS + CHART_PAGE_INDUSTRY_SECONDS
CHART_PROGRESS_BAR_H = 3

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 颜色定义 (R, G, B)
COLOR_BG = (15, 15, 25)            # 深色背景
COLOR_HEADER_BG = (30, 30, 50)     # 顶栏背景
COLOR_FOOTER_BG = (30, 30, 50)     # 底栏背景
COLOR_TEXT = (220, 220, 220)       # 主文字
COLOR_LABEL = (140, 140, 160)      # 标签文字（灰色）
COLOR_NEUTRAL = (188, 192, 206)    # 中性数值（浅灰，弱化 0 值视觉）
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
COLOR_PROGRESS_BAR_BG = (36, 40, 60)
COLOR_PROGRESS_BAR_FILL = (240, 184, 72)
COLOR_INDUSTRY_HEADER_BG = (46, 52, 74)
COLOR_INDUSTRY_TABLE_LINE = (86, 94, 122)
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
_proxy_guard_lock = threading.RLock()
_snapshot_cache_lock = threading.Lock()
_snapshot_fetch_lock = threading.Lock()
_latest_holdings_snapshot_cache: Optional[dict] = None
_latest_holdings_snapshot_cached_at: float = 0.0
_realtime_index_cache_lock = threading.Lock()
_realtime_index_fetch_lock = threading.Lock()
_latest_realtime_index_pct_cache: dict[str, float] = {}
_latest_realtime_index_pct_cached_at: float = 0.0
_industry_mapping_cache_lock = threading.Lock()
_industry_mapping_cache: dict[str, dict[str, str]] = {}
_industry_levels_cache_lock = threading.Lock()
_industry_levels_cache: Optional[dict[str, tuple[str, str, str]]] = None

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _should_bypass_proxy_for_fetch() -> bool:
    """判断抓数阶段是否临时禁用代理。"""
    raw = str(os.getenv("LAZYBULL_FETCH_BYPASS_PROXY", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _is_runtime_trace_enabled() -> bool:
    """是否启用运行时诊断追踪日志。"""
    raw = str(os.getenv("LAZYBULL_LCD_TRACE", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _trace_diag(message: str) -> None:
    """按开关输出运行时追踪日志。"""
    if _is_runtime_trace_enabled():
        _emit_diag(message, stderr=True)


def _clone_holdings_snapshot(snapshot: Optional[dict]) -> Optional[dict]:
    """复制快照时仅克隆必要字段，避免递归 deepcopy 带来的额外内存峰值。"""
    if not isinstance(snapshot, dict):
        return None

    cloned: dict = dict(snapshot)
    positions = snapshot.get("positions")
    quotes = snapshot.get("quotes")
    index_pct_map = snapshot.get("index_pct_map")
    if isinstance(positions, dict):
        cloned["positions"] = dict(positions)
    if quotes is not None and hasattr(quotes, "copy"):
        cloned["quotes"] = quotes.copy(deep=True)
    if isinstance(index_pct_map, dict):
        cloned["index_pct_map"] = dict(index_pct_map)
    return cloned


def _set_cached_holdings_snapshot(snapshot: Optional[dict]) -> None:
    """更新最近一次有效持仓快照缓存。"""
    if not isinstance(snapshot, dict):
        return
    quote_df = snapshot.get("quotes")
    if quote_df is None or getattr(quote_df, "empty", True):
        return

    cloned_snapshot = _clone_holdings_snapshot(snapshot)
    if cloned_snapshot is None:
        return

    with _snapshot_cache_lock:
        global _latest_holdings_snapshot_cache
        global _latest_holdings_snapshot_cached_at
        _latest_holdings_snapshot_cache = cloned_snapshot
        _latest_holdings_snapshot_cached_at = time.monotonic()


def _get_cached_holdings_snapshot(max_age_seconds: Optional[float] = None) -> Optional[dict]:
    """读取最近一次有效持仓快照缓存（按年龄过滤）。"""
    age_limit = (
        float(max_age_seconds)
        if max_age_seconds is not None
        else REALTIME_SNAPSHOT_CACHE_MAX_AGE_SECONDS
    )
    with _snapshot_cache_lock:
        cached = _latest_holdings_snapshot_cache
        cached_at = _latest_holdings_snapshot_cached_at
        if cached is None or cached_at <= 0:
            return None
        age = time.monotonic() - cached_at
        if age > age_limit:
            return None
        return _clone_holdings_snapshot(cached)


def _set_cached_realtime_index_pcts(pct_map: Optional[dict[str, float]]) -> None:
    """更新最近一次有效的实时指数涨跌幅缓存。"""
    if not isinstance(pct_map, dict):
        return

    normalized: dict[str, float] = {}
    for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
        pct = _sanitize_intraday_pct(pct_map.get(code), INTRADAY_INDEX_PCT_ABS_LIMIT)
        if pct is not None:
            normalized[code] = pct
    if not normalized:
        return

    with _realtime_index_cache_lock:
        global _latest_realtime_index_pct_cache
        global _latest_realtime_index_pct_cached_at
        _latest_realtime_index_pct_cache = dict(normalized)
        _latest_realtime_index_pct_cached_at = time.monotonic()


def _get_cached_realtime_index_pcts(max_age_seconds: float = 900.0) -> dict[str, float]:
    """读取最近一次有效的实时指数涨跌幅缓存。"""
    with _realtime_index_cache_lock:
        if not _latest_realtime_index_pct_cache or _latest_realtime_index_pct_cached_at <= 0:
            return {}
        age = time.monotonic() - _latest_realtime_index_pct_cached_at
        if age > float(max_age_seconds):
            return {}
        return dict(_latest_realtime_index_pct_cache)


def _is_realtime_index_cache_stale(max_age_seconds: Optional[float] = None) -> bool:
    """判断实时指数缓存是否已过期。"""
    age_limit = (
        float(max_age_seconds)
        if max_age_seconds is not None
        else float(REALTIME_REFRESH_INTERVAL)
    )
    with _realtime_index_cache_lock:
        if _latest_realtime_index_pct_cached_at <= 0:
            return True
        age = time.monotonic() - _latest_realtime_index_pct_cached_at
        return age > age_limit


def _refresh_realtime_index_pcts_async() -> None:
    """后台刷新实时指数缓存，避免阻塞快照主流程。"""
    if not _realtime_index_fetch_lock.acquire(blocking=False):
        _trace_diag("后台指数刷新跳过: 上一轮仍在执行")
        return

    def _runner() -> None:
        try:
            _trace_diag(
                f"后台指数刷新开始: timeout={REALTIME_INDEX_ASYNC_TIMEOUT_SECONDS:.1f}s"
            )
            timeout_sentinel = object()
            pct_map = _call_with_timeout(
                _fetch_realtime_index_pcts_from_akshare,
                REALTIME_INDEX_ASYNC_TIMEOUT_SECONDS,
                fallback=timeout_sentinel,
            )
            if pct_map is timeout_sentinel:
                _emit_diag(
                    "后台指数抓取超时: "
                    f"timeout={REALTIME_INDEX_ASYNC_TIMEOUT_SECONDS:.1f}s，"
                    "本轮沿用已有指数缓存"
                )
                return
            if pct_map:
                _set_cached_realtime_index_pcts(pct_map)
                _trace_diag(
                    f"后台指数缓存更新成功: codes={sorted(pct_map.keys())}"
                )
            else:
                _trace_diag("后台指数缓存更新为空")
        except Exception as exc:  # noqa: BLE001
            _emit_diag(f"后台指数缓存更新失败: {type(exc).__name__}: {exc}")
        finally:
            _realtime_index_fetch_lock.release()

    threading.Thread(target=_runner, daemon=True).start()


@contextmanager
def _fetch_network_context():
    """抓数专用网络上下文：按配置临时禁用代理，退出时恢复。"""
    if not _should_bypass_proxy_for_fetch():
        yield
        return

    with _proxy_guard_lock:
        backup = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS}
        try:
            for key in _PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            # requests/urllib 在 NO_PROXY=* 下会直接绕过代理
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
            yield
        finally:
            for key in _PROXY_ENV_KEYS:
                value = backup.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _configure_efinance_session() -> None:
    """为 efinance 配置自定义 requests Session，增强网络稳定性。
    
    配置内容：
    - 设置连接/读取超时，避免长期挂起
    - 配置 User-Agent 伪装成手机客户端，降低被风控风险
    - 配置连接池和重试机制
    """
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # 创建自定义 session
        session = requests.Session()
        
        # 配置超时和 User-Agent
        session.timeout = (EFINANCE_CONNECT_TIMEOUT_SECONDS, EFINANCE_READ_TIMEOUT_SECONDS)
        session.headers.update({
            'User-Agent': EFINANCE_USER_AGENT,
        })
        
        # 配置连接池：减少连接复用，降低断连风险
        adapter_config = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=5,
            max_retries=Retry(
                total=0,  # 由 efinance 外层控制重试
                backoff_factor=0.5,
            )
        )
        session.mount('http://', adapter_config)
        session.mount('https://', adapter_config)
        
        # 尝试将 session 注入到 efinance 全局
        # efinance 的 requests 调用可能会用到全局的 requests.Session
        import efinance as ef  # type: ignore
        if hasattr(ef, 'requests'):
            ef.requests.Session = lambda: session
        
        _emit_diag_once(
            "efinance_session_configured",
            f"efinance Session 配置完成: timeout=({EFINANCE_CONNECT_TIMEOUT_SECONDS:.1f}s,{EFINANCE_READ_TIMEOUT_SECONDS:.1f}s), "
            f"pool_size=5, User-Agent=iPhone"
        )
    except Exception as exc:
        _emit_diag_once(
            "efinance_session_config_error",
            f"efinance Session 配置失败: {type(exc).__name__}: {exc}"
        )


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
    )
    draw.rounded_rectangle(
        [cap_x0, cap_y0, cap_x1, cap_y1],
        radius=1,
        fill=COLOR_USAGE_BAR_EMPTY,
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
        fill=COLOR_USAGE_BAR_EMPTY,
    )
    draw.line(
        [(divider_right + 1, inner_y0), (divider_right + 1, inner_y1)],
        fill=COLOR_USAGE_BAR_EMPTY,
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


def _get_chart_panel_cycle_state(now_ts: Optional[float] = None) -> tuple[str, float, float]:
    """返回图表区轮播页状态。"""
    current_ts = now_ts if now_ts is not None else time.monotonic()
    phase = current_ts % CHART_PAGE_CYCLE_SECONDS
    if phase < CHART_PAGE_CHART_SECONDS:
        return "chart", phase, CHART_PAGE_CHART_SECONDS
    elapsed = phase - CHART_PAGE_CHART_SECONDS
    return "industry", elapsed, CHART_PAGE_INDUSTRY_SECONDS


def _get_shenwan_industry_mapping() -> dict[str, str]:
    """按配置的申万主口径获取 ts_code -> 行业名称映射（带进程内缓存）。"""
    shenwan_level = get_shenwan_level()

    with _industry_mapping_cache_lock:
        cached_mapping = _industry_mapping_cache.get(shenwan_level)
    if cached_mapping is not None:
        return cached_mapping

    try:
        from src.lazybull.data import DataLoader, Storage

        loader = DataLoader(storage=Storage(root_path=get_data_root()))
        shenwan_industry = loader.load_shenwan_industry()
        if shenwan_industry is None or shenwan_industry.empty:
            mapping: dict[str, str] = {}
        else:
            mapping = load_industry_mapping(
                shenwan_industry=shenwan_industry,
                verbose=False,
                shenwan_level=shenwan_level,
            )
    except Exception:
        mapping = {}

    with _industry_mapping_cache_lock:
        _industry_mapping_cache[shenwan_level] = mapping
    return mapping


def _industry_name_color(contribution_pnl: float) -> tuple[int, int, int]:
    """行业贡献为正红、负绿、零浅灰。"""
    if contribution_pnl > 0:
        return COLOR_RED
    if contribution_pnl < 0:
        return COLOR_GREEN
    return COLOR_NEUTRAL


def _get_shenwan_levels_mapping() -> dict[str, tuple[str, str, str]]:
    """获取 ts_code 到申万 L1/L2/L3 名称的映射（缓存）。"""
    global _industry_levels_cache

    with _industry_levels_cache_lock:
        if _industry_levels_cache is not None:
            return _industry_levels_cache

    result: dict[str, tuple[str, str, str]] = {}
    try:
        from src.lazybull.data import DataLoader, Storage

        loader = DataLoader(storage=Storage(root_path=get_data_root()))
        shenwan_industry = loader.load_shenwan_industry()
        if shenwan_industry is not None and not shenwan_industry.empty:
            for _, row in shenwan_industry.iterrows():
                ts_code = str(row.get('ts_code', '')).strip()
                if not ts_code:
                    continue

                l1 = str(row.get('sw_l1', row.get('sw_name', '未知行业')) or '未知行业')
                l2 = str(
                    row.get(
                        'sw_l2',
                        row.get('sw_industry', row.get('sw_name', '未知行业')),
                    )
                    or '未知行业'
                )
                l3 = str(
                    row.get(
                        'sw_l3',
                        row.get('sw_industry', row.get('sw_name', '未知行业')),
                    )
                    or '未知行业'
                )
                result[ts_code] = (l1, l2, l3)
    except Exception:
        result = {}

    with _industry_levels_cache_lock:
        _industry_levels_cache = result
    return result


def _draw_text_segments(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    segments: list[tuple[str, tuple[int, int, int]]],
    font: ImageFont.FreeTypeFont,
) -> int:
    """按分段颜色绘制文本，并返回绘制结束 x 坐标。"""
    current_x = x
    for text, color in segments:
        draw.text((current_x, y), text, fill=color, font=font)
        bbox = draw.textbbox((0, 0), text, font=font)
        current_x += bbox[2] - bbox[0]
    return current_x


