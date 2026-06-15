
from scripts.respi.lcd35._context import Optional, datetime, random, time, Image, ImageDraw
from scripts.respi.lcd35.core import *  # noqa: F401,F403
from scripts.respi.lcd35.core import _draw_text_segments, _get_font, _industry_name_color
from scripts.respi.lcd35.industry import *  # noqa: F401,F403
from scripts.respi.lcd35.industry import _value_color
from scripts.respi.lcd35.charting import *  # noqa: F401,F403
from scripts.respi.lcd35.charting import _normalize_cycle_price
from scripts.respi.lcd35.data_pipeline import *  # noqa: F401,F403
from scripts.respi.lcd35.system_io import *  # noqa: F401,F403
from scripts.respi.lcd35.state import DisplayState


# ---------- 渲染逻辑 ----------

# 布局常量
HEADER_H = 34          # 顶栏高度（含时间、更新、调仓）
HEADER_TIME_FONT_SIZE = 15
HEADER_META_FONT_SIZE = 13
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
        update_step = str(getattr(state, 'update_step', '') or '')
        update_started_at = float(getattr(state, 'update_started_at', 0.0) or 0.0)
        quote_source_tag = str(getattr(state, 'quote_source_tag', '-') or '-').upper()
        next_rebalance_date = getattr(state, 'next_rebalance_date', None)
        days_to_rebalance = state.days_to_rebalance
        cycle_chart_data = state.chart_data
        intraday_chart_data = state.intraday_chart_data
        rankings = state.stock_rankings
        industry_panel_cycle = getattr(state, 'industry_panel_cycle', getattr(state, 'industry_panel', None))
        industry_panel_intraday = getattr(state, 'industry_panel_intraday', None)
        ox = state.offset_x
        oy = state.offset_y

    force_reset = False
    stuck_step = update_step[:5] or "未知"
    elapsed = 0.0
    if is_updating and update_started_at > 0:
        elapsed = max(0.0, time.monotonic() - update_started_at)
        if elapsed > UPDATE_STUCK_RESET_SECONDS:
            force_reset = True

    if force_reset:
        with state.lock:
            state.is_updating = False
            state.update_step = ""
            state.update_started_at = 0.0
            is_updating = False
            update_step = ""
        _emit_diag(
            f"刷新状态超时脱困: step={stuck_step} 持续{elapsed:.1f}s，已自动复位顶部状态",
            stderr=False,
        )

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
    source_suffix = f"[{quote_source_tag}]" if quote_source_tag in ('T', 'A', 'D') else ""
    header_mid = (
        f"更:{update_step[:5] or '刷新中'}{source_suffix}"
        if is_updating
        else f"更新:{last_update_time}{source_suffix}"
    )
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
                # name==code 时只显示代码+涨跌幅，避免 "002414 002414" 重复
                if s.get('name') and s['name'] != s['code']:
                    line = f"{s['name']} {s['code']} {pct_str}"
                else:
                    line = f"{s['code']} {pct_str}"
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
    chart_mode = str(chart_data.get('mode', '')) if isinstance(chart_data, dict) else ''
    if chart_mode == 'intraday':
        industry_panel = industry_panel_intraday
    else:
        industry_panel = industry_panel_cycle or industry_panel_intraday
    _draw_chart_panel(draw, chart_data, cycle_last_data_label, industry_panel)

    try:
        _write_fb(img)
    finally:
        img.close()


def _draw_chart(
    draw: ImageDraw.ImageDraw,
    chart_data: Optional[dict],
    cycle_last_data_label: Optional[str] = None,
    chart_top: int = CHART_Y,
    chart_height: int = CHART_H,
) -> None:
    """绘制持仓周期图或盘中图。"""
    chart_height = max(40, int(chart_height))
    chart_x = 10
    chart_w = WIDTH - 20
    font_xs = _get_font(11)

    # 图表区背景
    draw.rectangle([chart_x, chart_top, chart_x + chart_w, chart_top + chart_height],
                   fill=COLOR_CHART_BG)

    if not chart_data:
        # 无数据提示
        txt = "暂无图表数据"
        bbox = draw.textbbox((0, 0), txt, font=_get_font(14))
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, chart_top + chart_height // 2 - 8), txt,
                  fill=COLOR_LABEL, font=_get_font(14))
        return

    dates = list(chart_data.get('dates', []))
    idx_pct = list(chart_data.get('index_pct', []))
    sz_pct = list(chart_data.get('shenzhen_pct', []))
    csi800_pct = list(chart_data.get('csi800_pct', []))
    ptf_pct = list(chart_data.get('portfolio_pct', []))
    slot_indices = list(chart_data.get('slot_indices', range(len(idx_pct))))
    x_positions = chart_data.get('x_positions', slot_indices)
    if not isinstance(x_positions, list):
        x_positions = slot_indices
    n = min(
        len(dates),
        len(idx_pct),
        len(sz_pct),
        len(ptf_pct),
        len(slot_indices),
        len(x_positions),
    )
    if n == 0:
        txt = "暂无图表数据"
        bbox = draw.textbbox((0, 0), txt, font=_get_font(14))
        tw = bbox[2] - bbox[0]
        draw.text(((WIDTH - tw) // 2, chart_top + chart_height // 2 - 8), txt,
                  fill=COLOR_LABEL, font=_get_font(14))
        return

    dates = dates[:n]
    idx_pct = idx_pct[:n]
    sz_pct = sz_pct[:n]
    csi800_available = len(csi800_pct) >= n and n > 0
    csi800_pct = csi800_pct[:n] if csi800_available else []
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
        if csi800_available:
            csi800_pct = _smooth_intraday_series_for_display(csi800_pct)
        ptf_pct = _smooth_intraday_series_for_display(ptf_pct)

    # Y轴范围
    all_vals = idx_pct + sz_pct + ptf_pct
    if csi800_available:
        all_vals += csi800_pct
    y_min, y_max = _get_chart_y_range(all_vals)
    y_range = y_max - y_min
    if y_range < 0.01:
        y_range = 1.0

    # 内部绘图区域
    label_w = 44       # Y轴标签空间
    legend_h = 16      # 顶部图例高度
    bottom_pad = 4
    cx = chart_x + label_w
    cy = chart_top + legend_h
    cw = chart_w - label_w - 6
    ch = max(26, chart_height - legend_h - bottom_pad)

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
    csi800_pts = _to_points(csi800_pct) if csi800_available else []
    ptf_pts = _to_points(ptf_pct)

    series_points = [
        (idx_pts, COLOR_CHART_SHANGHAI),
        (sz_pts, COLOR_CHART_SHENZHEN),
        (ptf_pts, COLOR_CHART_HOLDINGS),
    ]
    if csi800_available:
        series_points.append((csi800_pts, COLOR_CHART_CSI800))

    _draw_chart_series(draw, series_points, cx, cy, cw, ch)

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
    ly = chart_top + 2
    idx_last_str = f"{idx_pct[-1]:+.1f}%"
    sz_last_str = f"{sz_pct[-1]:+.1f}%"
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
        _short_legend_label(chart_data.get('csi800_label', '中证800')),
        COLOR_CHART_CSI800,
        f"{csi800_pct[-1]:+.1f}%",
    ) if csi800_available else legend_x
    _draw_legend_item(
        legend_x,
        _short_legend_label(chart_data.get('portfolio_label', '持仓')),
        COLOR_CHART_HOLDINGS,
        ptf_last_str,
    )

    if cycle_last_data_label:
        bbox_last = draw.textbbox((0, 0), cycle_last_data_label, font=font_xs)
        last_w = bbox_last[2] - bbox_last[0]
        draw.text(
            (chart_x + chart_w - last_w - 4, chart_top + 2),
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


def _draw_industry_panel(
    draw: ImageDraw.ImageDraw,
    industry_panel: Optional[dict],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    elapsed_seconds: float = 0.0,
    duration_seconds: float = CHART_PAGE_INDUSTRY_SECONDS,
) -> None:
    """在图表区域绘制行业统计页。"""
    draw.rectangle([panel_x, panel_y, panel_x + panel_w, panel_y + panel_h], fill=COLOR_CHART_BG)
    font_title = _get_font(14)
    font_body = _get_font(15)
    font_small = _get_font(14)
    font_page = _get_font(10)

    if not industry_panel or not industry_panel.get('industries'):
        tip = "暂无行业统计数据"
        bbox = draw.textbbox((0, 0), tip, font=font_title)
        tip_w = bbox[2] - bbox[0]
        draw.text(
            (panel_x + (panel_w - tip_w) // 2, panel_y + panel_h // 2 - 8),
            tip,
            fill=COLOR_LABEL,
            font=font_title,
        )
        return

    total_positive = int(industry_panel.get('total_positive', 0))
    total_negative = int(industry_panel.get('total_negative', 0))
    position_count = int(industry_panel.get('position_count', 0))
    l1_count = int(industry_panel.get('l1_industry_count', 0))
    l2_count = int(industry_panel.get('l2_industry_count', 0))
    l3_count = int(industry_panel.get('l3_industry_count', 0))
    industries_all = list(industry_panel.get('industries', []))
    total_industries = len(industries_all)

    row_top = panel_y + 24
    rows_per_col = 4
    col_count = 2
    per_page = rows_per_col * col_count
    page_sizes = []
    if total_industries <= 0:
        page_sizes = [0]
    else:
        remain = total_industries
        while remain > 0:
            page_sizes.append(min(per_page, remain))
            remain -= per_page
    page_count = max(1, len(page_sizes))

    # 多页时按页内行业数量占比分配该轮展示时长。
    if duration_seconds > 0 and total_industries > 0:
        page_durations = [duration_seconds * size / total_industries for size in page_sizes]
    else:
        page_durations = [max(1.0, duration_seconds)] * page_count

    elapsed_norm = max(0.0, elapsed_seconds)
    page_idx = page_count - 1
    acc = 0.0
    for idx, seconds in enumerate(page_durations):
        acc += seconds
        if elapsed_norm < acc:
            page_idx = idx
            break

    start_idx = page_idx * per_page
    industries = industries_all[start_idx:start_idx + per_page]

    header_h = 18
    draw.rounded_rectangle(
        [panel_x + 2, panel_y + 1, panel_x + panel_w - 2, panel_y + header_h],
        radius=3,
        fill=COLOR_INDUSTRY_HEADER_BG,
    )

    left_title = f"行业1/2/3:{l1_count}/{l2_count}/{l3_count}"
    draw.text((panel_x + 6, panel_y + 2), left_title, fill=COLOR_TEXT, font=font_title)

    if page_count > 1:
        page_text = f"页{page_idx + 1}/{page_count}"
        bbox_page = draw.textbbox((0, 0), page_text, font=font_page)
        page_w = bbox_page[2] - bbox_page[0]
        draw.text((panel_x + panel_w - page_w - 6, panel_y + 4), page_text, fill=COLOR_LABEL, font=font_page)

    right_label = "正/负收益股票数量:"
    left_bbox = draw.textbbox((0, 0), left_title, font=font_title)
    left_w = left_bbox[2] - left_bbox[0]
    right_x = max(panel_x + left_w + 16, panel_x + panel_w // 2 - 8)
    right_x = min(right_x, panel_x + panel_w - 180)
    current_x = _draw_text_segments(
        draw,
        right_x,
        panel_y + 2,
        [(right_label, COLOR_LABEL)],
        font_title,
    )
    _draw_text_segments(
        draw,
        current_x,
        panel_y + 2,
        [
            (f"+{total_positive}", _value_color(float(total_positive))),
            ("/", COLOR_LABEL),
            (f"-{total_negative}", _value_color(float(-total_negative))),
        ],
        font_title,
    )

    table_top = row_top - 2
    table_bottom_limit = panel_y + panel_h - 6
    available_h = max(rows_per_col * 16, table_bottom_limit - table_top)
    row_h = max(16, available_h // rows_per_col)
    table_bottom = table_top + rows_per_col * row_h

    col_w = panel_w // 2
    name_x_left = panel_x + 8
    metric_x_left = panel_x + 94
    name_x_right = panel_x + col_w + 8
    metric_x_right = panel_x + col_w + 94

    draw.rectangle(
        [panel_x + 3, table_top, panel_x + panel_w - 3, table_bottom],
        outline=COLOR_INDUSTRY_TABLE_LINE,
        width=1,
    )

    divider_x = panel_x + col_w
    draw.line(
        [(divider_x, table_top), (divider_x, table_bottom)],
        fill=COLOR_INDUSTRY_TABLE_LINE,
        width=2,
    )

    for row_idx in range(1, rows_per_col):
        y = table_top + row_idx * row_h
        draw.line(
            [(panel_x + 3, y), (panel_x + panel_w - 3, y)],
            fill=COLOR_INDUSTRY_TABLE_LINE,
            width=1,
        )

    for idx, item in enumerate(industries):
        col_idx = idx // rows_per_col
        row_idx = idx % rows_per_col
        y = row_top + row_idx * row_h
        if col_idx == 0:
            name_x = name_x_left
            metric_x = metric_x_left
        else:
            name_x = name_x_right
            metric_x = metric_x_right

        industry_name = str(item.get('industry', '未知行业'))[:14]
        positive_count = int(item.get('positive_count', 0))
        negative_count = int(item.get('negative_count', 0))
        contribution_ratio = float(item.get('contribution_ratio', 0.0))
        industry_pnl_amount = float(item.get('pnl_amount', 0.0))
        name_color = _industry_name_color(industry_pnl_amount)

        draw.text((name_x, y), industry_name, fill=name_color, font=font_body)
        _draw_text_segments(
            draw,
            metric_x,
            y,
            [
                (f"+{positive_count}", _value_color(float(positive_count))),
                ("/", COLOR_LABEL),
                (f"-{negative_count}", _value_color(float(-negative_count))),
                ("/", COLOR_LABEL),
                (f"{contribution_ratio:+.1f}%", _value_color(contribution_ratio)),
            ],
            font_body,
        )


def _draw_chart_panel(
    draw: ImageDraw.ImageDraw,
    chart_data: Optional[dict],
    cycle_last_data_label: Optional[str],
    industry_panel: Optional[dict],
) -> None:
    """绘制图表区域轮播页（图表页/行业统计页）与顶部进度条。"""
    chart_x = 10
    chart_w = WIDTH - 20
    page_name, elapsed, duration = _get_chart_panel_cycle_state()

    progress_y0 = CHART_Y
    progress_y1 = progress_y0 + CHART_PROGRESS_BAR_H - 1
    draw.rectangle([chart_x, progress_y0, chart_x + chart_w, progress_y1], fill=COLOR_PROGRESS_BAR_BG)
    if duration > 0:
        ratio = max(0.0, min(1.0, elapsed / duration))
    else:
        ratio = 0.0
    fill_width = int(round(chart_w * ratio))
    if ratio > 0 and fill_width <= 0:
        fill_width = 1
    if fill_width > 0:
        draw.rectangle(
            [chart_x, progress_y0, chart_x + fill_width - 1, progress_y1],
            fill=COLOR_PROGRESS_BAR_FILL,
        )

    content_top = progress_y1 + 2
    content_h = HEIGHT - content_top
    if page_name == "industry":
        _draw_industry_panel(
            draw,
            industry_panel,
            chart_x,
            content_top,
            chart_w,
            content_h,
            elapsed_seconds=elapsed,
            duration_seconds=duration,
        )
        return

    _draw_chart(
        draw,
        chart_data,
        cycle_last_data_label=cycle_last_data_label,
        chart_top=content_top,
        chart_height=content_h,
    )


