from scripts.respi.lcd35._context import Optional, datetime, np, pd, Image, ImageDraw, ImageFont
from scripts.respi.lcd35.core import *  # noqa: F401,F403
from scripts.respi.lcd35.core import (
    _coerce_float,
    _diag_lock,
    _diag_once_keys,
    _format_mmdd,
    _get_font,
    _trace_diag,
)
from scripts.respi.lcd35.industry import *  # noqa: F401,F403


def _write_fb(img: Image.Image) -> None:
    """占位定义：在组合入口中会被 system_io 同名函数覆盖。"""
    raise RuntimeError("framebuffer writer is not initialized")


def _should_keep_realtime_completion_active(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """占位定义：在组合入口中会被 data_pipeline 同名函数覆盖。"""
    return False


def _should_keep_morning_close_completion_active(
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """占位定义：在组合入口中会被 data_pipeline 同名函数覆盖。"""
    return False

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


def _normalize_cycle_price(price: object, pre_close: object, abs_limit: float) -> Optional[float]:
    """规范化成本口径实时价；昨收缺失时仍允许使用现价。"""
    price_float = _coerce_float(price)
    if price_float is None or not np.isfinite(price_float) or price_float <= 0:
        return None

    pre_close_float = _coerce_float(pre_close)
    if pre_close_float is None or not np.isfinite(pre_close_float) or pre_close_float <= 0:
        return price_float

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
            print(f"[{datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)
        except OSError:
            pass


def _emit_diag_once(key: str, message: str, stderr: bool = True) -> None:
    """同一类诊断信息仅记录一次，避免持续刷屏。"""
    with _diag_lock:
        if key in _diag_once_keys:
            return
        _diag_once_keys.add(key)
    _emit_diag(message, stderr=stderr)


def _call_with_timeout(
    func,
    timeout_seconds: float,
    fallback=None,
    timeout_diag_key: Optional[str] = None,
    timeout_diag_message: Optional[str] = None,
):
    """在后台线程执行函数并施加超时，超时时快速返回 fallback。"""
    started_at = time.monotonic()
    result = {
        "value": fallback,
        "error": None,
    }

    def _runner() -> None:
        try:
            result["value"] = func()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(max(float(timeout_seconds), 0.01))

    if worker.is_alive():
        elapsed = time.monotonic() - started_at
        _trace_diag(
            "超时命中: "
            f"timeout={float(timeout_seconds):.2f}s, elapsed={elapsed:.2f}s, func={getattr(func, '__name__', 'anonymous')}"
        )
        if timeout_diag_key and timeout_diag_message:
            _emit_diag_once(timeout_diag_key, timeout_diag_message)
        return fallback

    if result["error"] is not None:
        raise result["error"]
    return result["value"]


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

    try:
        _write_fb(img)
    finally:
        img.close()
    

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
        _trace_diag(
            f"盘中刷新触发: 午休补齐窗口, current={current_dt.strftime('%H:%M:%S')}"
        )
        return True, session_key
    if session_key is None:
        is_due = _is_interval_due(last_refresh_at, REALTIME_REFRESH_INTERVAL, current_dt)
        return is_due, None
    if last_session_key != session_key:
        _trace_diag(
            f"盘中刷新触发: 开盘/午后开盘切换, session={last_session_key}->{session_key}"
        )
        return True, session_key
    is_due = _is_interval_due(last_refresh_at, REALTIME_REFRESH_INTERVAL, current_dt)
    if is_due:
        _trace_diag(
            f"盘中刷新触发: 间隔到期({REALTIME_REFRESH_INTERVAL}s)"
        )
    return is_due, session_key


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


