import copy

from scripts.respi.lcd35._context import (
    Optional,
    Path,
    Image,
    ImageDraw,
    ImageFont,
    datetime,
    np,
    _cleanup_backlight_state_helper,
    _get_pwm_hardware_note_helper,
    _set_backlight_helper,
    _update_pwm_backlight_state_helper,
)
from scripts.respi.lcd35.charting import _describe_framebuffer_candidates, _emit_diag_once, _resolve_framebuffer_path
from scripts.respi.lcd35.core import *  # noqa: F401,F403


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
    img_array = np.asarray(img, dtype=np.uint8)
    r = (img_array[:, :, 0].astype(np.uint16) >> 3) << 11
    g = (img_array[:, :, 1].astype(np.uint16) >> 2) << 5
    b = img_array[:, :, 2].astype(np.uint16) >> 3
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
    finally:
        del rgb565
        del r
        del g
        del b
        del img_array


def _clear_screen() -> None:
    """写入全黑画面（息屏用）。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    try:
        _write_fb(img)
    finally:
        img.close()
    
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
    try:
        _write_fb(img)
    finally:
        img.close()


# ---------- 调仓日计算 ----------

def _calc_rebalance_status() -> tuple[Optional[str], Optional[int]]:
    """计算下次调仓日期及剩余交易日。"""
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data import DataLoader, Storage

    rebalance_state = PaperStorage(
        root_path=get_paper_root(), smb_reader=_smb_reader
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

