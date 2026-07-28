import importlib.util
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lcd35_display",
        PROJECT_ROOT / "scripts" / "respi" / "lcd35_display.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_entrypoint_bootstraps_project_root_for_src_import(monkeypatch):
    pruned_path = []
    project_root_resolved = PROJECT_ROOT.resolve()
    for candidate in list(sys.path):
        if not candidate:
            continue
        try:
            if Path(candidate).resolve() == project_root_resolved:
                continue
        except OSError:
            pass
        pruned_path.append(candidate)

    monkeypatch.setattr(sys, "path", pruned_path)

    _load_module()

    resolved_paths = set()
    for candidate in sys.path:
        if not candidate:
            continue
        try:
            resolved_paths.add(Path(candidate).resolve())
        except OSError:
            continue

    assert project_root_resolved in resolved_paths
    assert (PROJECT_ROOT / "scripts").resolve() in resolved_paths


def test_snapshot_timeout_reads_single_env_key(monkeypatch):
    monkeypatch.setenv("LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS", "120")
    monkeypatch.delenv("REALTIME_SNAPSHOT_TIMEOUT_SECONDS", raising=False)

    module = _load_module()

    assert module.REALTIME_SNAPSHOT_TIMEOUT_SECONDS == 120.0


def test_display_state_defaults_screen_on_for_display_worker():
    module = _load_module()

    state = module.DisplayState()

    assert hasattr(state, "is_screen_on")
    assert state.is_screen_on is True


def test_snapshot_timeout_ignores_legacy_env_key(monkeypatch):
    # 先拿“仅主变量为空且无旧变量”时的基准值，再验证设置旧变量不应改变结果。
    monkeypatch.setenv("LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS", "")
    monkeypatch.delenv("REALTIME_SNAPSHOT_TIMEOUT_SECONDS", raising=False)
    baseline_module = _load_module()
    baseline_timeout = baseline_module.REALTIME_SNAPSHOT_TIMEOUT_SECONDS

    monkeypatch.setenv("LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS", "")
    monkeypatch.setenv("REALTIME_SNAPSHOT_TIMEOUT_SECONDS", "120")
    module = _load_module()

    assert module.REALTIME_SNAPSHOT_TIMEOUT_SECONDS == baseline_timeout


def test_fetch_network_context_temporarily_clears_proxy_env(monkeypatch):
    module = _load_module()

    monkeypatch.setenv("LAZYBULL_FETCH_BYPASS_PROXY", "1")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.local:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)

    with module._fetch_network_context():
        assert os.environ.get("HTTP_PROXY") is None
        assert os.environ.get("HTTPS_PROXY") is None
        assert os.environ.get("NO_PROXY") == "*"

    assert os.environ.get("HTTP_PROXY") == "http://proxy.local:8080"
    assert os.environ.get("HTTPS_PROXY") == "http://proxy.local:8080"
    assert os.environ.get("NO_PROXY") is None


def test_fetch_network_context_respects_disable_switch(monkeypatch):
    module = _load_module()

    monkeypatch.setenv("LAZYBULL_FETCH_BYPASS_PROXY", "0")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.local:8080")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    with module._fetch_network_context():
        assert os.environ.get("HTTP_PROXY") == "http://proxy.local:8080"
        assert os.environ.get("NO_PROXY") == "127.0.0.1,localhost"


def test_format_display_time_uses_new_chinese_style():
    module = _load_module()

    formatted = module._format_display_time(datetime(2026, 4, 7, 14, 40, 32))

    assert formatted == "4月7日(周二) 14:40:32"


def test_format_rebalance_status_shows_next_date_and_clamps_negative_days():
    module = _load_module()

    formatted = module._format_rebalance_status("20260410", -2)

    assert formatted == "下次调仓:04/10/剩0天"


def test_format_quote_update_time_prefers_quote_time_hour_and_minute():
    module = _load_module()

    formatted = module._format_quote_update_time({"quote_time": "11:30:05"})

    assert formatted == "11:30"


def test_get_chart_panel_cycle_state_follows_40_20_ratio():
    module = _load_module()

    page, elapsed, duration = module._get_chart_panel_cycle_state(now_ts=5.0)
    assert page == "chart"
    assert elapsed == 5.0
    assert duration == 30.0

    page, elapsed, duration = module._get_chart_panel_cycle_state(now_ts=35.0)
    assert page == "industry"
    assert elapsed == 5.0
    assert duration == 30.0


def test_industry_name_color_uses_red_green_light_gray_by_contribution_sign():
    module = _load_module()

    assert module._industry_name_color(10.0) == module.COLOR_RED
    assert module._industry_name_color(-0.5) == module.COLOR_GREEN
    assert module._industry_name_color(0.0) == module.COLOR_NEUTRAL


def test_value_color_uses_light_gray_for_zero():
    module = _load_module()

    assert module._value_color(1.0) == module.COLOR_RED
    assert module._value_color(-1.0) == module.COLOR_GREEN
    assert module._value_color(0.0) == module.COLOR_NEUTRAL


def test_pick_fitting_font_shrinks_annual_return_when_text_is_too_wide(monkeypatch):
    module = _load_module()

    class _FakeDraw:
        @staticmethod
        def textbbox(_xy, text, font):
            width = len(text) * font["size"]
            return (0, 0, width, font["size"])

    monkeypatch.setattr(module, "_get_font", lambda size: {"size": size})

    fitted_font = module._pick_fitting_font(
        _FakeDraw(),
        "+123.4%",
        preferred_size=24,
        min_size=16,
        max_width=120,
    )
    default_font = module._pick_fitting_font(
        _FakeDraw(),
        "+9.9%",
        preferred_size=24,
        min_size=16,
        max_width=120,
    )

    assert fitted_font["size"] == 17
    assert default_font["size"] == 24

def test_format_error_lines_truncates_message_for_screen():
    module = _load_module()

    lines = module._format_error_lines(
        "RuntimeError: something went wrong while rendering the lcd framebuffer",
        line_width=12,
        max_lines=3,
    )

    assert len(lines) == 3
    assert lines[0] == "RuntimeError"
    assert lines[-1].endswith("…")


def test_write_fb_reports_framebuffer_error(monkeypatch):
    module = _load_module()
    messages = []

    monkeypatch.setattr(
        module,
        "_emit_diag_once",
        lambda key, message, stderr=True: messages.append((key, message, stderr)),
    )

    def _raise_open(*args, **kwargs):
        raise FileNotFoundError("fb missing")

    monkeypatch.setattr("builtins.open", _raise_open)

    module._write_fb(module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0)))

    assert messages
    assert messages[0][0].startswith("fb_write_error::")
    assert module.DEFAULT_FB_PATH in messages[0][1]
    assert "FileNotFoundError" in messages[0][1]


def test_emit_diag_uses_timestamp_prefix_for_stderr(monkeypatch, capsys, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "_get_diag_log_paths", lambda: [tmp_path / "diag.log"])

    module._emit_diag("行业更新成功: cycle=Y, intraday=Y", stderr=True)

    captured = capsys.readouterr()
    assert captured.err.startswith("[")
    assert captured.err[1:9].count(":") == 2
    assert captured.err.endswith("行业更新成功: cycle=Y, intraday=Y\n")


def test_resolve_framebuffer_path_uses_env_override(monkeypatch):
    monkeypatch.setenv("LAZYBULL_LCD_FB_PATH", "/dev/fb9")
    module = _load_module()

    assert module._resolve_framebuffer_path() == "/dev/fb9"


def test_init_backlight_uses_helper_and_records_state(monkeypatch):
    module = _load_module()
    messages = []

    monkeypatch.setattr(
        module,
        "_emit_diag_once",
        lambda key, message, stderr=True: messages.append((key, message)),
    )
    monkeypatch.setattr(
        module,
        "_set_backlight_helper",
        lambda *args, **kwargs: {
            "method": "pwm",
            "backend": "lgpio",
            "percent": 10,
            "pin": 18,
            "frequency": 1000,
        },
    )

    module._backlight_state = None
    module._init_backlight()

    assert module._backlight_state is not None
    assert module._backlight_state["backend"] == "lgpio"
    assert ("backlight_pwm_ok", "背光初始化完成: 使用 lgpio PWM") in messages
    assert any(key == "backlight_pwm_hardware_note" and "焊" in message for key, message in messages)


def test_set_backlight_updates_existing_pwm_state(monkeypatch):
    module = _load_module()
    state = {"method": "pwm", "backend": "lgpio", "percent": 10}
    updates = []

    monkeypatch.setattr(
        module,
        "_update_pwm_backlight_state_helper",
        lambda current_state, brightness: updates.append((current_state, brightness)) or current_state,
    )

    module._backlight_state = state
    module._set_backlight(5)

    assert updates == [(state, 5)]


def test_cleanup_backlight_uses_helper_cleanup(monkeypatch):
    module = _load_module()
    cleaned = []

    monkeypatch.setattr(
        module,
        "_cleanup_backlight_state_helper",
        lambda state: cleaned.append(state),
    )

    module._backlight_state = {"method": "pwm", "backend": "lgpio"}
    module._cleanup_backlight()

    assert cleaned == [{"method": "pwm", "backend": "lgpio"}]
    assert module._backlight_state is None


def test_layout_constants_use_taller_header_and_narrower_left_panel():
    module = _load_module()

    assert module.HEADER_H == 34
    assert module.USAGE_BAR_H == 5
    assert module.USAGE_BAR_BOTTOM_GAP == 1
    assert module.USAGE_BAR_SECTION_GAP >= 2
    assert module.HEADER_TIME_FONT_SIZE >= module.HEADER_META_FONT_SIZE
    assert module.PANEL_TOP == module.HEADER_H + 4
    assert module.LEFT_W == int(module.PANEL_AREA_W * 0.575)
    assert module.RIGHT_W == module.PANEL_AREA_W - module.LEFT_W - module.PANEL_GAP


def test_refresh_system_usage_sample_throttles_to_two_seconds(monkeypatch):
    module = _load_module()
    cpu_calls = []
    cpu_samples = [(100, 40), (140, 52)]
    memory_calls = []
    memory_samples = [48.0, 62.5]

    def _fake_read_cpu_stat_sample():
        sample = cpu_samples[len(cpu_calls)]
        cpu_calls.append(sample)
        return sample

    def _fake_read_memory_usage_pct():
        memory_pct = memory_samples[len(memory_calls)]
        memory_calls.append(memory_pct)
        return memory_pct

    monkeypatch.setattr(module, "_read_cpu_stat_sample", _fake_read_cpu_stat_sample)
    monkeypatch.setattr(module, "_read_memory_usage_pct", _fake_read_memory_usage_pct)

    state = module.DisplayState()

    assert module._refresh_system_usage_sample(state, now_ts=0.0) == (0.0, 48.0)
    assert module._refresh_system_usage_sample(state, now_ts=1.9) == (0.0, 48.0)

    cpu_usage_pct, memory_usage_pct = module._refresh_system_usage_sample(state, now_ts=2.0)

    assert cpu_calls == [(100, 40), (140, 52)]
    assert memory_calls == [48.0, 62.5]
    assert round(cpu_usage_pct, 4) == 70.0
    assert round(memory_usage_pct, 4) == 62.5
    assert round(state.cpu_usage_pct, 4) == 70.0
    assert round(state.memory_usage_pct, 4) == 62.5


def test_draw_system_usage_bar_fills_left_and_right_portions_with_independent_colors():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), module.COLOR_HEADER_BG)
    draw = module.ImageDraw.Draw(image)

    module._draw_system_usage_bar(draw, 30.0, 85.0)

    body_x1 = module.WIDTH - module.USAGE_BAR_MARGIN_X - module.USAGE_BAR_CAP_W - 2
    inner_x0 = module.USAGE_BAR_MARGIN_X + 2
    inner_x1 = body_x1 - 2
    divider_left = inner_x0 + (inner_x1 - inner_x0 + 1) // 2 - module.USAGE_BAR_SECTION_GAP // 2
    divider_right = divider_left + module.USAGE_BAR_SECTION_GAP - 1
    left_x0 = inner_x0
    left_x1 = divider_left - 1
    right_x0 = divider_right + 1
    right_x1 = inner_x1
    inner_y = module.HEADER_H - module.USAGE_BAR_BOTTOM_GAP - module.USAGE_BAR_H + 2
    left_fill_x = left_x0 + max(2, (left_x1 - left_x0) // 5)
    left_empty_x = min(left_x1 - 2, left_x0 + int(round((left_x1 - left_x0 + 1) * 0.6)))
    right_fill_x = right_x0 + max(2, (right_x1 - right_x0) // 2)

    assert image.getpixel((left_fill_x, inner_y)) == module.COLOR_USAGE_BAR_LOW
    assert image.getpixel((left_empty_x, inner_y)) == module.COLOR_USAGE_BAR_EMPTY
    assert image.getpixel((right_fill_x, inner_y)) == module.COLOR_USAGE_BAR_HIGH


def test_draw_system_usage_bar_has_no_outline_border():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), module.COLOR_HEADER_BG)
    draw = module.ImageDraw.Draw(image)

    module._draw_system_usage_bar(draw, 30.0, 85.0)

    body_y0 = module.HEADER_H - module.USAGE_BAR_BOTTOM_GAP - module.USAGE_BAR_H
    body_x0 = module.USAGE_BAR_MARGIN_X
    assert image.getpixel((body_x0, body_y0)) != module.COLOR_USAGE_BAR_OUTLINE


def test_is_trade_day_uses_weekday_fallback_without_loading_calendar(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_trade_date_set_cache", None)

    def _should_not_load():
        raise AssertionError("首帧判断不应触发交易日历加载")

    monkeypatch.setattr(module, "_load_trade_date_set", _should_not_load)

    assert module._is_trade_day(datetime(2026, 4, 7, 10, 0, 0)) is True
    assert module._is_trade_day(datetime(2026, 4, 11, 10, 0, 0)) is False


def test_build_cycle_chart_payload_keeps_fixed_slots_before_holding_exceeds_period():
    module = _load_module()

    payload = module._build_cycle_chart_payload(
        dates=[f"202604{day:02d}" for day in range(1, 12)],
        index_pct=[float(i) for i in range(11)],
        shenzhen_pct=[float(i) * 0.8 for i in range(11)],
        portfolio_pct=[float(i) * 0.5 for i in range(11)],
        rebalance_freq=20,
        base_value=100.0,
    )

    assert payload is not None
    assert payload["slot_count"] == 20
    assert payload["slot_indices"][-1] == 10
    assert payload["x_end_label"] == "20天"
    assert round(payload["shenzhen_pct"][3], 4) == 2.4
    assert payload["shenzhen_label"] == "深证"


def test_build_cycle_chart_payload_expands_slots_after_holding_exceeds_period():
    module = _load_module()

    payload = module._build_cycle_chart_payload(
        dates=[f"202604{day:02d}" for day in range(1, 24)],
        index_pct=[float(i) for i in range(23)],
        shenzhen_pct=[float(i) * 0.8 for i in range(23)],
        portfolio_pct=[float(i) * 0.5 for i in range(23)],
        rebalance_freq=20,
        base_value=100.0,
    )

    assert payload is not None
    assert payload["slot_count"] == 23
    assert payload["x_end_label"] == "04/23"


def test_build_cycle_chart_payload_includes_csi800_series():
    module = _load_module()

    payload = module._build_cycle_chart_payload(
        dates=[f"202604{day:02d}" for day in range(1, 6)],
        index_pct=[0.0, 0.5, 1.0, 1.5, 2.0],
        shenzhen_pct=[0.0, 0.4, 0.8, 1.2, 1.6],
        csi800_pct=[0.0, 0.3, 0.6, 0.9, 1.2],
        portfolio_pct=[0.0, 0.2, 0.5, 0.7, 1.0],
        rebalance_freq=5,
        base_value=100.0,
    )

    assert payload is not None
    assert payload["csi800_pct"] == [0.0, 0.3, 0.6, 0.9, 1.2]
    assert payload["csi800_label"] == "中证800"


def test_upsert_intraday_chart_keeps_every_refresh_point_with_time_based_x_axis():
    module = _load_module()

    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 34, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=1.2,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 39, 0),
        index_pct=0.8,
        shenzhen_pct=0.6,
        portfolio_pct=1.5,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 41, 0),
        index_pct=1.1,
        shenzhen_pct=0.9,
        portfolio_pct=1.8,
    )

    assert chart["slot_count"] == module.INTRADAY_SLOT_COUNT
    assert chart["slot_indices"] == [0, 0, 1]
    assert chart["x_positions"] == [0.4, 0.9, 1.1]
    assert chart["raw_index_pct"] == [0.5, 0.8, 1.1]
    assert chart["raw_shenzhen_pct"] == [0.3, 0.6, 0.9]
    assert chart["raw_csi800_pct"] == [0.3, 0.6, 0.9]
    assert chart["raw_portfolio_pct"] == [1.2, 1.5, 1.8]
    assert chart["index_pct"] == [0.5, 0.8, 1.1]
    assert chart["shenzhen_pct"] == [0.3, 0.6, 0.9]
    assert chart["csi800_pct"] == [0.3, 0.6, 0.9]
    assert chart["portfolio_pct"] == [1.2, 1.5, 1.8]
    assert chart["x_start_label"] == "09:30"
    assert chart["x_end_label"] == "15:00"
    assert chart["portfolio_label"] == "持仓"
    assert chart["shenzhen_label"] == "深证"


def test_upsert_intraday_chart_accepts_explicit_csi800_pct():
    module = _load_module()

    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 34, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=1.2,
        csi800_pct=0.4,
    )

    assert chart["raw_csi800_pct"] == [0.4]
    assert chart["csi800_pct"] == [0.4]


def test_intraday_slots_collapse_lunch_break_into_continuous_line():
    module = _load_module()

    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 11, 30, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=1.2,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 13, 0, 0),
        index_pct=0.8,
        shenzhen_pct=0.6,
        portfolio_pct=1.5,
    )

    assert module.INTRADAY_SLOT_COUNT == 26
    assert chart["slot_indices"] == [12, 13]
    assert chart["x_positions"] == [12.0, 13.0]


def test_chart_y_range_always_keeps_zero_reference_visible():
    module = _load_module()

    positive_y_min, positive_y_max = module._get_chart_y_range([1.2, 1.8, 2.1])
    negative_y_min, negative_y_max = module._get_chart_y_range([-2.1, -1.3, -0.8])

    assert positive_y_min < 0 < positive_y_max
    assert negative_y_min < 0 < negative_y_max


def test_smooth_intraday_series_for_display_preserves_endpoints_and_softens_jitter():
    module = _load_module()

    smoothed = module._smooth_intraday_series_for_display([0.0, 1.0, -1.0, 1.0, 0.0])

    assert smoothed == [0.0, 0.25, 0.0, 0.25, 0.0]


def test_get_intraday_display_x_positions_snaps_session_boundaries():
    module = _load_module()

    chart = {
        "mode": "intraday",
        "trade_date": "20260407",
    }
    display_x_positions = module._get_intraday_display_x_positions(
        chart,
        ["11:30", "13:00", "15:00"],
        [module.INTRADAY_MORNING_SLOT_COUNT - 1, module.INTRADAY_MORNING_SLOT_COUNT, module.INTRADAY_SLOT_COUNT - 1],
        [12.0, 13.0, 24.95],
    )

    assert display_x_positions == [12.5, 12.5, float(module.INTRADAY_SLOT_COUNT - 1)]


def test_draw_chart_supports_antialiased_intraday_rendering_on_real_image():
    module = _load_module()
    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 34, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=1.2,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 36, 0),
        index_pct=0.6,
        shenzhen_pct=0.2,
        portfolio_pct=1.1,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 38, 0),
        index_pct=0.4,
        shenzhen_pct=0.5,
        portfolio_pct=1.3,
    )

    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    draw = module.ImageDraw.Draw(image)

    module._draw_chart(draw, chart)

    assert image.getbbox() is not None


def test_draw_chart_shows_lunch_marker_and_zero_label_for_intraday():
    module = _load_module()
    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 11, 30, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=1.2,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 13, 0, 0),
        index_pct=0.8,
        shenzhen_pct=0.6,
        portfolio_pct=1.5,
    )

    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)
    captured_texts = []
    captured_lines = []

    class DrawProxy:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            captured_lines.append(kwargs.get("fill"))
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            captured_texts.append((position, text, kwargs.get("fill")))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

        def ellipse(self, *args, **kwargs):
            return real_draw.ellipse(*args, **kwargs)

    module._draw_chart(DrawProxy(), chart)

    assert any(text == "午休" for _, text, _ in captured_texts)
    assert any(text == "0%" for _, text, _ in captured_texts)
    assert module.COLOR_CHART_BREAK in captured_lines
    assert module.COLOR_CHART_ZERO_LINE in captured_lines


def test_draw_chart_legacy_intraday_without_csi800_does_not_draw_white_series():
    module = _load_module()
    chart = {
        "mode": "intraday",
        "trade_date": "20260407",
        "dates": ["09:30", "09:40", "09:50"],
        "index_pct": [0.2, 0.3, 0.5],
        "shenzhen_pct": [0.1, 0.2, 0.4],
        "portfolio_pct": [0.0, 0.1, 0.2],
        "slot_indices": [0, 1, 2],
        "x_positions": [0.0, 1.0, 2.0],
        "slot_count": module.INTRADAY_SLOT_COUNT,
    }

    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)
    captured_line_colors = []

    class DrawProxy:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            captured_line_colors.append(kwargs.get("fill"))
            return real_draw.line(*args, **kwargs)

        def text(self, *args, **kwargs):
            return real_draw.text(*args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

        def ellipse(self, *args, **kwargs):
            return real_draw.ellipse(*args, **kwargs)

    module._draw_chart(DrawProxy(), chart)

    assert module.COLOR_CHART_CSI800 not in captured_line_colors


def test_draw_zero_reference_line_applies_offset():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)
    captured_lines = []

    class DrawProxy:
        def line(self, points, *args, **kwargs):
            captured_lines.append(points)
            return real_draw.line(points, *args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def text(self, *args, **kwargs):
            return real_draw.text(*args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    module._draw_zero_reference_line(
        DrawProxy(),
        cx=20,
        cy=40,
        cw=100,
        ch=40,
        y_min=-2.0,
        y_range=4.0,
        font_xs=module._get_font(11),
    )

    assert captured_lines[0] == [(21, 61), (119, 61)]


def test_compute_holdings_intraday_pct_uses_pre_close_weighting():
    module = _load_module()

    snapshot = {
        "positions": {
            "000001.SZ": SimpleNamespace(shares=100),
            "000002.SZ": SimpleNamespace(shares=200),
        },
        "quotes": pd.DataFrame(
            [
                {"TS_CODE": "000001.SZ", "PRICE": 11.0, "PRE_CLOSE": 10.0},
                {"TS_CODE": "000002.SZ", "PRICE": 18.0, "PRE_CLOSE": 20.0},
            ]
        ),
    }

    pct = module._compute_holdings_intraday_pct(snapshot)

    assert pct is not None
    assert round(pct, 4) == round(((11.0 * 100 + 18.0 * 200) / (10.0 * 100 + 20.0 * 200) - 1) * 100, 4)


def test_compute_holdings_intraday_pct_falls_back_to_pre_close_for_invalid_price():
    module = _load_module()

    snapshot = {
        "positions": {
            "000001.SZ": SimpleNamespace(shares=100),
            "000002.SZ": SimpleNamespace(shares=100),
        },
        "quotes": pd.DataFrame(
            [
                {"TS_CODE": "000001.SZ", "PRICE": 0.0, "PRE_CLOSE": 10.0},
                {"TS_CODE": "000002.SZ", "PRICE": 22.0, "PRE_CLOSE": 20.0},
            ]
        ),
    }

    pct = module._compute_holdings_intraday_pct(snapshot)

    assert pct is not None
    assert round(pct, 4) == round(((10.0 * 100 + 22.0 * 100) / (10.0 * 100 + 20.0 * 100) - 1) * 100, 4)


def test_build_industry_panel_aggregates_counts_tops_and_contribution(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_shenwan_industry_mapping",
        lambda: {
            "000001.SZ": "银行",
            "000002.SZ": "银行",
            "000003.SZ": "电子",
        },
    )
    monkeypatch.setattr(
        module,
        "_get_shenwan_levels_mapping",
        lambda: {
            "000001.SZ": ("金融", "银行", "城商行"),
            "000002.SZ": ("金融", "银行", "股份行"),
            "000003.SZ": ("科技", "电子", "消费电子"),
        },
    )

    panel = module._build_industry_panel(
        {
            "positions": {
                "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                "000002.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                "000003.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            },
            "quotes": pd.DataFrame(
                [
                    {"TS_CODE": "000001.SZ", "NAME": "平安银行", "PRICE": 11.0, "PRE_CLOSE": 10.5},
                    {"TS_CODE": "000002.SZ", "NAME": "招商银行", "PRICE": 9.0, "PRE_CLOSE": 9.5},
                    {"TS_CODE": "000003.SZ", "NAME": "立讯精密", "PRICE": 10.5, "PRE_CLOSE": 10.2},
                ]
            ),
        }
    )

    assert panel is not None
    assert panel["total_positive"] == 2
    assert panel["total_negative"] == 1
    assert panel["position_count"] == 3
    assert panel["l1_industry_count"] == 2
    assert panel["l2_industry_count"] == 2
    assert panel["l3_industry_count"] == 3
    assert panel["industries"][0]["industry"] == "科技"
    finance_item = next(item for item in panel["industries"] if item["industry"] == "金融")
    assert finance_item["positive_count"] == 1
    assert finance_item["negative_count"] == 1
    assert "top_positive" not in finance_item
    assert "top_negative" not in finance_item


def test_build_industry_panel_intraday_uses_pre_close_instead_of_buy_price(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_shenwan_industry_mapping",
        lambda: {"000001.SZ": "银行"},
    )
    monkeypatch.setattr(
        module,
        "_get_shenwan_levels_mapping",
        lambda: {"000001.SZ": ("金融", "银行", "城商行")},
    )

    snapshot = {
        "positions": {
            "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
        },
        "quotes": pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "PRICE": 11.0,
                    "PRE_CLOSE": 12.0,
                }
            ]
        ),
    }

    cycle_panel = module._build_industry_panel(snapshot, mode="cycle")
    intraday_panel = module._build_industry_panel(snapshot, mode="intraday")

    assert cycle_panel is not None
    assert intraday_panel is not None
    # 周期口径: (11-10)*100
    assert round(cycle_panel["total_pnl_amount"], 4) == 100.0
    # 盘内口径: (11-12)*100
    assert round(intraday_panel["total_pnl_amount"], 4) == -100.0

def test_build_industry_panel_intraday_derives_pre_close_from_pct_chg_when_missing(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_shenwan_industry_mapping",
        lambda: {"000001.SZ": "银行"},
    )
    monkeypatch.setattr(
        module,
        "_get_shenwan_levels_mapping",
        lambda: {"000001.SZ": ("金融", "银行", "城商行")},
    )

    snapshot = {
        "positions": {
            "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
        },
        "quotes": pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "PRICE": 11.0,
                    "PCT_CHG": -8.3333333333,
                }
            ]
        ),
    }

    intraday_panel = module._build_industry_panel(snapshot, mode="intraday")

    assert intraday_panel is not None
    assert intraday_panel["total_positive"] == 0
    assert intraday_panel["total_negative"] == 1
    assert round(intraday_panel["total_pnl_amount"], 4) == -100.0


def test_build_industry_panel_intraday_contribution_uses_intraday_total(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_shenwan_industry_mapping",
        lambda: {"000001.SZ": "银行", "000002.SZ": "电子"},
    )
    monkeypatch.setattr(
        module,
        "_get_shenwan_levels_mapping",
        lambda: {
            "000001.SZ": ("金融", "银行", "城商行"),
            "000002.SZ": ("科技", "电子", "半导体"),
        },
    )

    snapshot = {
        "positions": {
            "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            "000002.SZ": SimpleNamespace(shares=100, buy_price=10.0),
        },
        "quotes": pd.DataFrame(
            [
                {"TS_CODE": "000001.SZ", "PRICE": 11.0, "PRE_CLOSE": 12.0},
                {"TS_CODE": "000002.SZ", "PRICE": 10.5, "PRE_CLOSE": 10.0},
            ]
        ),
    }

    cycle_panel = module._build_industry_panel(snapshot, mode="cycle")
    intraday_panel = module._build_industry_panel(snapshot, mode="intraday")

    assert cycle_panel is not None
    assert intraday_panel is not None
    assert cycle_panel["contribution_basis"] == "cycle_total_pnl"
    assert intraday_panel["contribution_basis"] == "intraday_total_pnl"

    cycle_by_industry = {item["industry"]: item for item in cycle_panel["industries"]}
    intraday_by_industry = {item["industry"]: item for item in intraday_panel["industries"]}

    assert round(cycle_panel["total_pnl_amount"], 4) == 150.0
    assert round(intraday_panel["total_pnl_amount"], 4) == -50.0
    assert round(cycle_by_industry["金融"]["contribution_ratio"], 1) == 66.7
    assert round(cycle_by_industry["科技"]["contribution_ratio"], 1) == 33.3
    assert round(intraday_by_industry["金融"]["contribution_ratio"], 1) == -100.0
    assert round(intraday_by_industry["科技"]["contribution_ratio"], 1) == 100.0


def test_build_industry_panel_contribution_normalizes_positive_and_negative_separately(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_shenwan_levels_mapping",
        lambda: {
            "000001.SZ": ("金融", "银行", "城商行"),
            "000002.SZ": ("科技", "电子", "半导体"),
            "000003.SZ": ("消费", "食品饮料", "白酒"),
            "000004.SZ": ("医药", "化学制药", "创新药"),
        },
    )

    panel = module._build_industry_panel(
        {
            "positions": {
                "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                "000002.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                "000003.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                "000004.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            },
            "quotes": pd.DataFrame(
                [
                    {"TS_CODE": "000001.SZ", "PRICE": 11.5, "PRE_CLOSE": 11.0},
                    {"TS_CODE": "000002.SZ", "PRICE": 11.0, "PRE_CLOSE": 10.8},
                    {"TS_CODE": "000003.SZ", "PRICE": 9.0, "PRE_CLOSE": 9.2},
                    {"TS_CODE": "000004.SZ", "PRICE": 9.5, "PRE_CLOSE": 9.7},
                ]
            ),
        },
        mode="cycle",
    )

    assert panel is not None
    by_industry = {item["industry"]: item for item in panel["industries"]}
    positive_sum = (
        max(by_industry["金融"]["contribution_ratio"], 0.0)
        + max(by_industry["科技"]["contribution_ratio"], 0.0)
    )
    negative_sum = (
        min(by_industry["消费"]["contribution_ratio"], 0.0)
        + min(by_industry["医药"]["contribution_ratio"], 0.0)
    )
    assert round(positive_sum, 4) == 100.0
    assert round(negative_sum, 4) == -100.0


def test_build_industry_panel_counts_flat_position_as_positive(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_shenwan_levels_mapping",
        lambda: {
            "000001.SZ": ("金融", "银行", "城商行"),
            "000002.SZ": ("科技", "电子", "半导体"),
        },
    )

    panel = module._build_industry_panel(
        {
            "positions": {
                "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                "000002.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            },
            "quotes": pd.DataFrame(
                [
                    {"TS_CODE": "000001.SZ", "PRICE": 10.0, "PRE_CLOSE": 10.0},
                    {"TS_CODE": "000002.SZ", "PRICE": 9.8, "PRE_CLOSE": 10.0},
                ]
            ),
        },
        mode="cycle",
    )

    assert panel is not None
    assert panel["position_count"] == 2
    assert panel["total_positive"] == 1
    assert panel["total_negative"] == 1
    by_industry = {item["industry"]: item for item in panel["industries"]}
    assert by_industry["金融"]["positive_count"] == 1
    assert by_industry["金融"]["negative_count"] == 0


def test_normalize_intraday_chart_drops_abnormal_points():
    module = _load_module()

    normalized = module._normalize_intraday_chart(
        {
            "trade_date": "20260407",
            "dates": ["09:30", "09:40", "09:50"],
            "index_pct": [0.6, 99.0, 0.8],
            "shenzhen_pct": [0.4, 0.5, 0.6],
            "portfolio_pct": [1.1, 1.2, -80.0],
            "slot_indices": [0, 1, 2],
        }
    )

    assert normalized is not None
    assert normalized["slot_indices"] == [0]
    assert normalized["dates"] == ["09:30"]
    assert normalized["x_positions"] == [0.0]
    assert normalized["raw_index_pct"] == [0.6]
    assert normalized["raw_shenzhen_pct"] == [0.4]
    assert normalized["raw_portfolio_pct"] == [1.1]
    assert normalized["index_pct"] == [0.6]
    assert normalized["shenzhen_pct"] == [0.4]
    assert normalized["portfolio_pct"] == [1.1]


def test_intraday_chart_persistence_restores_same_day_history(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "project_root", tmp_path)

    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 34, 0),
        index_pct=0.6,
        shenzhen_pct=0.4,
        portfolio_pct=1.1,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 39, 0),
        index_pct=0.9,
        shenzhen_pct=0.7,
        portfolio_pct=1.3,
    )

    module._save_intraday_chart(chart)
    restored = module._load_intraday_chart(now=datetime(2026, 4, 7, 14, 0, 0))
    next_day = module._load_intraday_chart(now=datetime(2026, 4, 8, 9, 30, 0))

    assert restored is not None
    assert restored["slot_indices"] == [0, 0]
    assert restored["dates"] == ["09:34", "09:39"]
    assert restored["x_positions"] == [0.4, 0.9]
    assert restored["raw_index_pct"] == [0.6, 0.9]
    assert restored["raw_shenzhen_pct"] == [0.4, 0.7]
    assert restored["raw_portfolio_pct"] == [1.1, 1.3]
    assert restored["index_pct"] == [0.6, 0.9]
    assert restored["shenzhen_pct"] == [0.4, 0.7]
    assert restored["portfolio_pct"] == [1.1, 1.3]
    assert next_day is None
    assert module._get_intraday_chart_state_path("20260407").exists()


def test_normalize_intraday_chart_keeps_previous_close_based_legacy_payload():
    module = _load_module()

    normalized = module._normalize_intraday_chart(
        {
            "trade_date": "20260407",
            "dates": ["09:30", "09:40"],
            "index_pct": [0.8, 1.1],
            "shenzhen_pct": [0.6, 0.9],
            "portfolio_pct": [1.5, 1.8],
            "slot_indices": [0, 1],
        }
    )

    assert normalized is not None
    assert normalized["x_positions"] == [0.0, 1.0]
    assert normalized["raw_index_pct"] == [0.8, 1.1]
    assert normalized["raw_shenzhen_pct"] == [0.6, 0.9]
    assert normalized["raw_portfolio_pct"] == [1.5, 1.8]
    assert normalized["index_pct"] == [0.8, 1.1]
    assert normalized["shenzhen_pct"] == [0.6, 0.9]
    assert normalized["portfolio_pct"] == [1.5, 1.8]


def test_normalize_intraday_chart_drops_pre_open_legacy_points():
    module = _load_module()

    normalized = module._normalize_intraday_chart(
        {
            "trade_date": "20260407",
            "dates": ["09:05", "09:30", "09:40"],
            "index_pct": [-2.7, 0.0, -0.3],
            "shenzhen_pct": [-4.8, 0.0, -0.4],
            "portfolio_pct": [-3.5, 0.0, -0.2],
            "slot_indices": [3, 6, 7],
        }
    )

    assert normalized is not None
    assert normalized["dates"] == ["09:30", "09:40"]
    assert normalized["slot_indices"] == [0, 1]
    assert normalized["x_positions"] == [0.0, 1.0]
    assert normalized["raw_index_pct"] == [0.0, -0.3]
    assert normalized["index_pct"] == [0.0, -0.3]


def test_build_stock_rankings_falls_back_to_pre_close_for_invalid_price():
    module = _load_module()

    rankings = module._build_stock_rankings(
        {
            "positions": {
                "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            },
            "quotes": pd.DataFrame(
                [
                    {
                        "TS_CODE": "000001.SZ",
                        "NAME": "平安银行",
                        "PRICE": 0.0,
                        "PRE_CLOSE": 11.0,
                    }
                ]
            ),
        }
    )

    assert rankings is not None
    assert len(rankings) == 1
    assert rankings[0]["code"] == "000001"
    assert round(rankings[0]["pnl_pct"], 4) == 10.0


def test_build_stock_rankings_matches_codes_when_quote_without_suffix():
    module = _load_module()

    rankings = module._build_stock_rankings(
        {
            "positions": {
                "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            },
            "quotes": pd.DataFrame(
                [
                    {
                        "TS_CODE": "000001",
                        "NAME": "平安银行",
                        "PRICE": 10.5,
                        "PRE_CLOSE": 10.2,
                    }
                ]
            ),
        }
    )

    assert rankings is not None
    assert len(rankings) == 1
    assert rankings[0]["code"] == "000001"
    assert round(rankings[0]["pnl_pct"], 4) == 5.0


def test_fetch_realtime_holdings_snapshot_prefers_efinance_over_akshare(monkeypatch):
    module = _load_module()

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {"initial_capital": 100000.0}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                cash=5000.0,
            )

    efinance_df = pd.DataFrame(
        [
            {
                "TS_CODE": "000001.SZ",
                "NAME": "平安银行",
                "PRICE": 11.0,
                "PRE_CLOSE": 10.0,
                "TIME": "09:32:00",
            }
        ]
    )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_efinance", lambda ts_codes: efinance_df)
    monkeypatch.setattr(
        module,
        "_fetch_realtime_quotes_akshare",
        lambda ts_codes: (_ for _ in ()).throw(AssertionError("efinance可用时不应触发AKShare兜底")),
    )

    snapshot = module._fetch_realtime_holdings_snapshot()

    assert snapshot is not None
    assert snapshot["quotes"] is not None
    assert snapshot["quote_source"] == "E"
    assert len(snapshot["quotes"]) == 1
    assert snapshot["quotes"].iloc[0]["TS_CODE"] == "000001.SZ"


def test_fetch_realtime_holdings_snapshot_falls_back_to_akshare_when_efinance_empty(monkeypatch):
    module = _load_module()

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {"initial_capital": 100000.0}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                cash=5000.0,
            )

    akshare_df = pd.DataFrame(
        [
            {
                "TS_CODE": "000001.SZ",
                "NAME": "平安银行",
                "PRICE": 11.0,
                "PRE_CLOSE": 10.0,
                "TIME": "09:35:00",
            }
        ]
    )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_efinance", lambda ts_codes: pd.DataFrame())
    monkeypatch.setattr(module, "_fetch_realtime_quotes_akshare", lambda ts_codes: akshare_df)

    snapshot = module._fetch_realtime_holdings_snapshot()

    assert snapshot is not None
    assert snapshot["quotes"] is not None
    assert snapshot["quote_source"] == "A"
    assert len(snapshot["quotes"]) == 1
    assert snapshot["quotes"].iloc[0]["TS_CODE"] == "000001.SZ"


def test_fetch_realtime_holdings_snapshot_returns_empty_when_efinance_and_akshare_unavailable(monkeypatch):
    module = _load_module()

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {"initial_capital": 100000.0}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                cash=5000.0,
            )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_efinance", lambda ts_codes: None)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_akshare", lambda ts_codes: None)
    monkeypatch.setattr(module, "_build_post_close_daily_snapshot", lambda snapshot, now=None: None)

    snapshot = module._fetch_realtime_holdings_snapshot()

    assert snapshot is not None
    assert snapshot["quotes"] is None
    assert snapshot["quote_source"] == "-"


def test_fetch_realtime_holdings_snapshot_falls_back_to_daily_close_post_close(monkeypatch):
    module = _load_module()
    query_calls = []

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {"initial_capital": 100000.0}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                    "000002.SZ": SimpleNamespace(shares=200, buy_price=20.0),
                },
                cash=5000.0,
            )

    class DummyClient:
        def __init__(self, verbose=False):
            pass

        def query(self, api_name, **kwargs):
            query_calls.append((api_name, kwargs.get("ts_code"), kwargs.get("trade_date")))
            if api_name == "daily":
                return pd.DataFrame(
                    [
                        {
                            "ts_code": "000001.SZ",
                            "trade_date": "20260407",
                            "close": 11.0,
                            "pre_close": 10.5,
                            "pct_chg": 4.7619,
                        },
                        {
                            "ts_code": "000002.SZ",
                            "trade_date": "20260407",
                            "close": 19.0,
                            "pre_close": 20.0,
                            "pct_chg": -5.0,
                        }
                    ]
                )
            if kwargs.get("ts_code") == module.SHANGHAI_INDEX_CODE:
                closes = [3000.0, 3030.0]
            elif kwargs.get("ts_code") == module.SHENZHEN_INDEX_CODE:
                closes = [10000.0, 10100.0]
            else:
                closes = [5000.0, 5100.0]
            return pd.DataFrame(
                {
                    "trade_date": ["20260403", "20260407"],
                    "close": closes,
                }
            )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr("src.lazybull.data.tushare_client.TushareClient", DummyClient)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_efinance", lambda ts_codes: None)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_akshare", lambda ts_codes: None)
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    snapshot = module._fetch_realtime_holdings_snapshot()

    assert snapshot is not None
    assert snapshot["quote_source"] == "D"
    assert snapshot["current_date"] == "20260407"
    assert snapshot["quotes"] is not None
    assert list(snapshot["quotes"]["TS_CODE"]) == ["000001.SZ", "000002.SZ"]
    assert list(snapshot["quotes"]["TIME"]) == ["15:00:00", "15:00:00"]
    assert round(snapshot["index_pct_map"][module.SHANGHAI_INDEX_CODE], 6) == 1.0
    assert round(snapshot["index_pct_map"][module.SHENZHEN_INDEX_CODE], 6) == 1.0
    assert round(snapshot["index_pct_map"][module.CSI800_INDEX_CODE], 6) == 2.0
    assert query_calls.count(("daily", None, "20260407")) == 1


def test_fetch_realtime_holdings_snapshot_prefers_daily_snapshot_before_open(monkeypatch):
    module = _load_module()

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {"initial_capital": 100000.0}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                cash=5000.0,
            )

    fallback_snapshot = {
        "positions": {"000001.SZ": SimpleNamespace(shares=100, buy_price=10.0)},
        "cash": 5000.0,
        "initial_capital": 100000.0,
        "current_date": "20260407",
        "annualized_return_func": None,
        "quote_source": "D",
        "index_pct_map": {},
        "quotes": pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "NAME": "000001",
                    "PRICE": 11.0,
                    "PRE_CLOSE": 10.5,
                    "PCT_CHG": 4.7619,
                    "TIME": "15:00:00",
                }
            ]
        ),
    }

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr(module, "_should_prefer_daily_holdings_snapshot", lambda now=None: True)
    monkeypatch.setattr(module, "_build_post_close_daily_snapshot", lambda snapshot, now=None: fallback_snapshot)
    monkeypatch.setattr(
        module,
        "_fetch_realtime_quotes_efinance",
        lambda ts_codes: (_ for _ in ()).throw(AssertionError("不应触发 efinance")),
    )
    monkeypatch.setattr(
        module,
        "_fetch_realtime_quotes_akshare",
        lambda ts_codes: (_ for _ in ()).throw(AssertionError("不应触发 AKShare")),
    )

    snapshot = module._fetch_realtime_holdings_snapshot()

    assert snapshot is fallback_snapshot
    assert snapshot["quote_source"] == "D"


def test_refresh_display_state_populates_panels_from_daily_fallback_snapshot(monkeypatch):
    module = _load_module()
    state = module.DisplayState()
    fallback_snapshot = {
        "positions": {
            "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
            "000002.SZ": SimpleNamespace(shares=200, buy_price=20.0),
        },
        "cash": 5000.0,
        "initial_capital": 100000.0,
        "current_date": "20260407",
        "annualized_return_func": None,
        "quote_source": "D",
        "index_pct_map": {
            module.SHANGHAI_INDEX_CODE: 1.0,
            module.SHENZHEN_INDEX_CODE: 1.0,
            module.CSI800_INDEX_CODE: 2.0,
        },
        "quotes": pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "NAME": "000001",
                    "PRICE": 11.0,
                    "PRE_CLOSE": 10.5,
                    "PCT_CHG": 4.7619,
                    "TIME": "15:00:00",
                },
                {
                    "TS_CODE": "000002.SZ",
                    "NAME": "000002",
                    "PRICE": 19.0,
                    "PRE_CLOSE": 20.0,
                    "PCT_CHG": -5.0,
                    "TIME": "15:00:00",
                },
            ]
        ),
    }

    monkeypatch.setattr(module, "_fetch_realtime_holdings_snapshot", lambda: fallback_snapshot)
    monkeypatch.setattr(
        module,
        "_build_industry_panel",
        lambda snapshot, mode="cycle": {
            "mode": mode,
            "industries": [
                {
                    "industry": "银行",
                    "positive_count": 1,
                    "negative_count": 0,
                    "pnl_amount": 100.0,
                    "contribution_ratio": 100.0,
                }
            ],
            "total_positive": 1,
            "total_negative": 0,
            "position_count": 2,
            "l1_industry_count": 1,
            "l2_industry_count": 1,
            "l3_industry_count": 1,
            "total_pnl_amount": 100.0,
            "contribution_basis": "cycle_total_pnl",
        },
    )
    monkeypatch.setattr(module, "_calc_rebalance_status", lambda: ("20260410", 2))
    monkeypatch.setattr(module, "_fetch_cycle_chart_data", lambda: None)

    module._refresh_display_state(state, refresh_realtime=True, refresh_cycle=False)

    assert state.summary is not None
    assert state.stock_rankings is not None
    assert len(state.stock_rankings) == 2
    assert state.industry_panel is not None
    assert state.quote_source_tag == "D"
    assert state.update_time == "15:00"


def test_fetch_realtime_quotes_akshare_matches_symbol_prefixed_codes(monkeypatch):
    module = _load_module()

    class DummyAK:
        @staticmethod
        def stock_zh_a_spot():
            return pd.DataFrame(
                [
                    {
                        "symbol": "sz000001",
                        "名称": "平安银行",
                        "最新价": 11.0,
                        "昨收": 10.0,
                        "时间": "09:35:00",
                    },
                    {
                        "symbol": "sh600746",
                        "名称": "江苏索普",
                        "最新价": 8.0,
                        "昨收": 7.8,
                        "时间": "09:35:00",
                    },
                ]
            )

    monkeypatch.setitem(sys.modules, "akshare", DummyAK)

    quotes = module._fetch_realtime_quotes_akshare(["000001.SZ", "600746.SH"])

    assert quotes is not None
    assert len(quotes) == 2
    assert set(quotes["TS_CODE"].tolist()) == {"000001.SZ", "600746.SH"}


def test_fetch_realtime_quotes_efinance_retry_waits_at_least_two_seconds(monkeypatch):
    module = _load_module()

    class DummyStockApi:
        call_count = 0

        @classmethod
        def get_latest_quote(cls, codes):
            cls.call_count += 1
            if cls.call_count == 1:
                raise ConnectionError("temporary efinance failure")
            return pd.DataFrame(
                [
                    {
                        "代码": "000001",
                        "名称": "平安银行",
                        "最新价": 11.0,
                        "昨收": 10.0,
                        "更新时间": "09:35:00",
                    }
                ]
            )

    class DummyEF:
        stock = DummyStockApi

    sleep_calls = []

    monkeypatch.setitem(sys.modules, "efinance", DummyEF)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    quotes = module._fetch_realtime_quotes_efinance(["000001.SZ"])

    assert quotes is not None
    assert len(quotes) == 1
    assert DummyStockApi.call_count == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 2.0


def test_fetch_realtime_holdings_snapshot_builds_annualized_func_from_config(monkeypatch):
    module = _load_module()

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {
                "initial_capital": 6000.0,
                "account_start_date": "20250101",
            }

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                cash=5000.0,
            )

        def load_all_nav(self):
            return None

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr(
        module,
        "_fetch_realtime_quotes_akshare",
        lambda ts_codes: pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "NAME": "平安银行",
                    "PRICE": 12.0,
                    "PRE_CLOSE": 11.5,
                    "TIME": "10:05:00",
                }
            ]
        ),
    )

    snapshot = module._fetch_realtime_holdings_snapshot()
    assert snapshot is not None
    assert callable(snapshot.get("annualized_return_func"))

    summary = module._build_realtime_portfolio_summary(snapshot)
    assert summary is not None
    assert summary["total_pnl_pct"] > 0
    assert summary["annual_return_pct"] > 0


def test_fetch_realtime_index_pcts_prefers_snapshot_data():
    module = _load_module()
    module._emit_diag_once = lambda *args, **kwargs: None

    pct_map = module._fetch_realtime_index_pcts(
        {
            "index_pct_map": {
                module.SHANGHAI_INDEX_CODE: 0.7,
                module.SHENZHEN_INDEX_CODE: -0.4,
            }
        }
    )

    assert pct_map[module.SHANGHAI_INDEX_CODE] == 0.7
    assert pct_map[module.SHENZHEN_INDEX_CODE] == -0.4


def test_fetch_realtime_index_pcts_triggers_async_refresh_when_complete_cache_is_stale(monkeypatch):
    module = _load_module()
    module._emit_diag_once = lambda *args, **kwargs: None
    async_calls = []

    monkeypatch.setattr(
        module,
        "_get_cached_realtime_index_pcts",
        lambda max_age_seconds=900.0: {
            module.SHANGHAI_INDEX_CODE: 0.7,
            module.SHENZHEN_INDEX_CODE: -0.4,
            module.CSI800_INDEX_CODE: 0.2,
        },
    )
    monkeypatch.setattr(module, "_is_realtime_index_cache_stale", lambda max_age_seconds=None: True)
    monkeypatch.setattr(module, "_refresh_realtime_index_pcts_async", lambda: async_calls.append("refresh"))

    pct_map = module._fetch_realtime_index_pcts({"quotes": pd.DataFrame()})

    assert pct_map[module.SHANGHAI_INDEX_CODE] == 0.7
    assert pct_map[module.SHENZHEN_INDEX_CODE] == -0.4
    assert pct_map[module.CSI800_INDEX_CODE] == 0.2
    assert async_calls == ["refresh"]


def test_fetch_realtime_holdings_snapshot_prewarms_index_refresh_before_quote_fetch(monkeypatch):
    module = _load_module()
    call_order = []
    refresh_state = {"started": False}

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_config(self):
            return {"initial_capital": 100000.0}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                cash=5000.0,
            )

        def load_all_nav(self):
            return None

    def fake_refresh_async():
        call_order.append("refresh")
        refresh_state["started"] = True

    def fake_fetch_quotes(ts_codes):
        call_order.append("quotes")
        assert refresh_state["started"] is True
        return pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "NAME": "平安银行",
                    "PRICE": 11.0,
                    "PRE_CLOSE": 10.5,
                    "TIME": "10:05:00",
                }
            ]
        )

    def fake_get_cached_index_pcts(max_age_seconds=900.0):
        if refresh_state["started"]:
            return {
                module.SHANGHAI_INDEX_CODE: 0.7,
                module.SHENZHEN_INDEX_CODE: -0.4,
                module.CSI800_INDEX_CODE: 0.2,
            }
        return {}

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr(module, "_should_prefer_daily_holdings_snapshot", lambda now=None: False)
    monkeypatch.setattr(module, "_is_realtime_index_cache_stale", lambda max_age_seconds=None: True)
    monkeypatch.setattr(module, "_refresh_realtime_index_pcts_async", fake_refresh_async)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_efinance", fake_fetch_quotes)
    monkeypatch.setattr(module, "_fetch_realtime_quotes_akshare", lambda ts_codes: None)
    monkeypatch.setattr(module, "_get_cached_realtime_index_pcts", fake_get_cached_index_pcts)

    snapshot = module._fetch_realtime_holdings_snapshot()

    assert snapshot is not None
    assert call_order[:2] == ["refresh", "quotes"]
    assert snapshot["index_pct_map"] == {
        module.SHANGHAI_INDEX_CODE: 0.7,
        module.SHENZHEN_INDEX_CODE: -0.4,
        module.CSI800_INDEX_CODE: 0.2,
    }


def test_select_chart_data_switches_by_intraday_window(monkeypatch):
    module = _load_module()

    cycle_chart = {"mode": "cycle", "dates": ["20260407"]}
    intraday_chart = {"mode": "intraday", "trade_date": "20260407"}
    point_time = datetime(2026, 4, 7, 14, 40, 0)

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "intraday"

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )
    assert module._select_chart_data(
        cycle_chart,
        intraday_chart,
        datetime(2026, 4, 7, 15, 1, 0),
    )["mode"] == "cycle"


def test_select_chart_data_keeps_intraday_after_close_until_cycle_updates(monkeypatch):
    module = _load_module()

    cycle_chart = {"mode": "cycle", "dates": ["20260401", "20260406"]}
    intraday_chart = {"mode": "intraday", "trade_date": "20260407", "slot_indices": [0, 25]}
    point_time = datetime(2026, 4, 7, 15, 1, 0)

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    selected = module._select_chart_data(cycle_chart, intraday_chart, point_time)

    assert selected is intraday_chart


def test_is_intraday_chart_window_starts_after_open(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    assert module._is_intraday_chart_window(datetime(2026, 4, 7, 9, 5, 0)) is False
    assert module._is_intraday_chart_window(datetime(2026, 4, 7, 9, 30, 0)) is True


def test_build_intraday_chart_skips_pre_open_snapshot(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_fetch_realtime_index_pcts",
        lambda snapshot=None: {module.SHANGHAI_INDEX_CODE: 0.1, module.SHENZHEN_INDEX_CODE: 0.2},
    )
    monkeypatch.setattr(module, "_compute_holdings_intraday_pct", lambda snapshot: 0.3)

    chart = module._build_intraday_chart(
        None,
        {"positions": {}, "quotes": pd.DataFrame()},
        point_time=datetime(2026, 4, 7, 9, 5, 0),
    )

    assert chart is None


def test_build_intraday_chart_uses_snapshot_quote_time_after_close(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_fetch_realtime_index_pcts",
        lambda snapshot=None: {
            module.SHANGHAI_INDEX_CODE: 0.1,
            module.SHENZHEN_INDEX_CODE: 0.2,
            module.CSI800_INDEX_CODE: 0.15,
        },
    )
    monkeypatch.setattr(module, "_compute_holdings_intraday_pct", lambda snapshot: 0.3)

    chart = module._build_intraday_chart(
        None,
        {
            "current_date": "20260407",
            "positions": {"000001.SZ": SimpleNamespace(shares=100)},
            "quotes": pd.DataFrame(
                [
                    {"TS_CODE": "000001.SZ", "PRICE": 11.0, "PRE_CLOSE": 10.0, "TIME": "15:00:00"}
                ]
            ),
        },
        point_time=None,
    )

    assert chart is not None
    assert chart["slot_indices"] == [module.INTRADAY_SLOT_COUNT - 1]
    assert chart["dates"] == ["15:00"]


def test_build_intraday_chart_keeps_refreshing_when_csi800_temporarily_missing(monkeypatch):
    module = _load_module()
    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 30, 0),
        index_pct=0.1,
        shenzhen_pct=0.2,
        portfolio_pct=0.3,
        csi800_pct=0.4,
    )

    monkeypatch.setattr(
        module,
        "_fetch_realtime_index_pcts",
        lambda snapshot=None: {
            module.SHANGHAI_INDEX_CODE: 0.5,
            module.SHENZHEN_INDEX_CODE: 0.6,
            # 故意缺失 CSI800，模拟 AKShare 暂时不可用
        },
    )
    monkeypatch.setattr(module, "_compute_holdings_intraday_pct", lambda snapshot: 0.7)

    updated = module._build_intraday_chart(
        chart,
        {"positions": {"000001.SZ": SimpleNamespace(shares=100)}, "quotes": pd.DataFrame()},
        point_time=datetime(2026, 4, 7, 9, 40, 0),
    )

    assert updated is not None
    assert len(updated["dates"]) == 2
    assert updated["raw_csi800_pct"][-1] == 0.4


def test_format_cycle_last_data_label_uses_last_cycle_date():
    module = _load_module()

    label = module._format_cycle_last_data_label(
        {"dates": ["20260401", "20260407"]}
    )

    assert label == "数据日:04/07"


def test_render_hides_cycle_last_data_label_in_intraday_mode(monkeypatch):
    module = _load_module()
    captured = {}
    intraday_chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 0, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=0.8,
    )
    state = SimpleNamespace(
        lock=module.threading.Lock(),
        summary=None,
        update_time="14:40",
        days_to_rebalance=1,
        chart_data={"mode": "cycle", "dates": ["20260401", "20260407"]},
        intraday_chart_data=intraday_chart,
        stock_rankings=None,
        offset_x=0,
        offset_y=0,
    )

    monkeypatch.setattr(module, "_select_chart_data", lambda cycle, intraday, now: intraday)
    monkeypatch.setattr(module, "_format_cycle_last_data_label", lambda chart: "数据日:04/07")
    monkeypatch.setattr(
        module,
        "_draw_chart_panel",
        lambda draw, chart_data, cycle_last_data_label=None, industry_panel=None: captured.update(
            {"mode": chart_data.get("mode"), "label": cycle_last_data_label}
        ),
    )
    monkeypatch.setattr(module, "_write_fb", lambda img: None)

    module._render(state)

    assert captured == {"mode": "intraday", "label": None}

def test_render_intraday_does_not_fallback_to_cycle_industry_panel(monkeypatch):
    module = _load_module()
    captured = {}
    intraday_chart = {"mode": "intraday", "trade_date": "20260407", "slot_indices": [0]}
    state = SimpleNamespace(
        lock=module.threading.Lock(),
        summary=None,
        update_time="14:40",
        days_to_rebalance=1,
        chart_data={"mode": "cycle", "dates": ["20260401", "20260407"]},
        intraday_chart_data=intraday_chart,
        stock_rankings=None,
        industry_panel_cycle={"industries": [{"industry": "金融"}], "contribution_basis": "cycle_total_pnl"},
        industry_panel_intraday=None,
        offset_x=0,
        offset_y=0,
    )

    monkeypatch.setattr(module, "_select_chart_data", lambda cycle, intraday, now: intraday)
    monkeypatch.setattr(
        module,
        "_draw_chart_panel",
        lambda draw, chart_data, cycle_last_data_label=None, industry_panel=None: captured.update(
            {"mode": chart_data.get("mode"), "industry_panel": industry_panel}
        ),
    )
    monkeypatch.setattr(module, "_write_fb", lambda img: None)

    module._render(state)

    assert captured == {"mode": "intraday", "industry_panel": None}


def test_render_shows_updating_text_and_new_rebalance_status(monkeypatch):
    module = _load_module()
    captured_texts = []
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)

    class DrawProxy:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            captured_texts.append(text)
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

    state = SimpleNamespace(
        lock=module.threading.Lock(),
        summary=None,
        update_time="14:40",
        is_updating=True,
        next_rebalance_date="20260410",
        days_to_rebalance=2,
        chart_data=None,
        intraday_chart_data=None,
        stock_rankings=None,
        offset_x=0,
        offset_y=0,
    )

    monkeypatch.setattr(module.ImageDraw, "Draw", lambda img: DrawProxy())
    monkeypatch.setattr(module, "_select_chart_data", lambda cycle, intraday, now: None)
    monkeypatch.setattr(
        module,
        "_draw_chart_panel",
        lambda draw, chart_data, cycle_last_data_label=None, industry_panel=None: None,
    )
    monkeypatch.setattr(module, "_write_fb", lambda img: None)

    module._render(state)

    assert "更:刷新中" in captured_texts
    assert "下次调仓:04/10/剩2天" in captured_texts


def test_draw_chart_panel_switches_between_chart_and_industry(monkeypatch):
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    draw = module.ImageDraw.Draw(image)
    called = {"chart": 0, "industry": 0}

    monkeypatch.setattr(
        module,
        "_draw_chart",
        lambda *args, **kwargs: called.__setitem__("chart", called["chart"] + 1),
    )
    monkeypatch.setattr(
        module,
        "_draw_industry_panel",
        lambda *args, **kwargs: called.__setitem__("industry", called["industry"] + 1),
    )

    monkeypatch.setattr(module, "_get_chart_panel_cycle_state", lambda now_ts=None: ("chart", 10.0, 30.0))
    module._draw_chart_panel(
        draw,
        {
            "dates": ["20260407"],
            "index_pct": [0.1],
            "shenzhen_pct": [0.1],
            "portfolio_pct": [0.1],
            "slot_indices": [0],
            "slot_count": 2,
        },
        None,
        None,
    )

    monkeypatch.setattr(module, "_get_chart_panel_cycle_state", lambda now_ts=None: ("industry", 5.0, 30.0))
    module._draw_chart_panel(draw, None, None, {"industries": []})

    assert called == {"chart": 1, "industry": 1}


def test_draw_industry_panel_shows_summary_rows_without_top_stock_fields():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)
    captured_texts = []

    class DrawProxy:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            captured_texts.append(str(text))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    panel = {
        "total_positive": 1,
        "total_negative": 1,
        "position_count": 2,
        "l1_industry_count": 1,
        "l2_industry_count": 1,
        "l3_industry_count": 1,
        "industries": [
            {
                "industry": "金融",
                "positive_count": 1,
                "negative_count": 1,
                "contribution_ratio": 12.3,
                "pnl_amount": 100.0,
            }
        ],
    }

    module._draw_industry_panel(
        DrawProxy(),
        panel,
        panel_x=10,
        panel_y=100,
        panel_w=460,
        panel_h=120,
    )

    assert any(text.startswith("行业1/2/3:") for text in captured_texts)
    assert any(text == "金融" for text in captured_texts)
    assert any(text == "+1" for text in captured_texts)
    assert any(text == "-1" for text in captured_texts)
    assert any(text == "+12.3%" for text in captured_texts)


def test_draw_industry_panel_draws_table_and_thicker_middle_divider():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    draw = module.ImageDraw.Draw(image)
    panel = {
        "total_positive": 2,
        "total_negative": 1,
        "position_count": 3,
        "l1_industry_count": 2,
        "l2_industry_count": 2,
        "l3_industry_count": 3,
        "industries": [
            {
                "industry": "银行",
                "positive_count": 1,
                "negative_count": 1,
                "contribution_ratio": 12.3,
                "pnl_amount": 100.0,
            }
        ],
    }

    panel_x = 10
    panel_y = 100
    panel_w = 460
    panel_h = 120
    module._draw_industry_panel(draw, panel, panel_x, panel_y, panel_w, panel_h)

    col_w = panel_w // 2
    divider_x = panel_x + col_w
    assert image.getpixel((divider_x, panel_y + 40)) == module.COLOR_INDUSTRY_TABLE_LINE


def test_draw_industry_panel_paginates_to_cover_all_industries():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)
    panel = {
        "total_positive": 6,
        "total_negative": 6,
        "position_count": 12,
        "l1_industry_count": 3,
        "l2_industry_count": 6,
        "l3_industry_count": 12,
        "industries": [
            {
                "industry": f"行业{i}",
                "positive_count": 1,
                "negative_count": 1,
                "contribution_ratio": float(i),
                "pnl_amount": float(i),
            }
            for i in range(12)
        ],
    }

    captured_page_1 = []
    captured_page_2 = []
    captured_page_last = []

    class DrawProxyPage1:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            captured_page_1.append(str(text))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    class DrawProxyPage2:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            captured_page_2.append(str(text))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    class DrawProxyPageLast:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            captured_page_last.append(str(text))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    module._draw_industry_panel(
        DrawProxyPage1(),
        panel,
        panel_x=10,
        panel_y=100,
        panel_w=460,
        panel_h=120,
        elapsed_seconds=0.0,
        duration_seconds=30.0,
    )
    module._draw_industry_panel(
        DrawProxyPage2(),
        panel,
        panel_x=10,
        panel_y=100,
        panel_w=460,
        panel_h=120,
        elapsed_seconds=19.0,
        duration_seconds=30.0,
    )
    module._draw_industry_panel(
        DrawProxyPageLast(),
        panel,
        panel_x=10,
        panel_y=100,
        panel_w=460,
        panel_h=120,
        elapsed_seconds=21.0,
        duration_seconds=30.0,
    )

    assert any(text.startswith("页1/") for text in captured_page_1)
    assert any(text.startswith("页1/") for text in captured_page_2)
    assert any(text == "行业0" for text in captured_page_1)
    assert any(text.startswith("页2/") for text in captured_page_last)
    assert any(text == "行业11" for text in captured_page_last)


def test_draw_industry_panel_page_duration_is_proportional_to_page_size():
    module = _load_module()
    image = module.Image.new("RGB", (module.WIDTH, module.HEIGHT), (0, 0, 0))
    real_draw = module.ImageDraw.Draw(image)
    panel = {
        "total_positive": 6,
        "total_negative": 6,
        "position_count": 12,
        "l1_industry_count": 3,
        "l2_industry_count": 6,
        "l3_industry_count": 12,
        "industries": [
            {
                "industry": f"行业{i}",
                "positive_count": 1,
                "negative_count": 1,
                "contribution_ratio": float(i),
                "pnl_amount": float(i),
            }
            for i in range(12)
        ],
    }

    early_texts = []
    late_texts = []

    class DrawProxyEarly:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            early_texts.append(str(text))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    class DrawProxyLate:
        def rectangle(self, *args, **kwargs):
            return real_draw.rectangle(*args, **kwargs)

        def rounded_rectangle(self, *args, **kwargs):
            return real_draw.rounded_rectangle(*args, **kwargs)

        def line(self, *args, **kwargs):
            return real_draw.line(*args, **kwargs)

        def text(self, position, text, *args, **kwargs):
            late_texts.append(str(text))
            return real_draw.text(position, text, *args, **kwargs)

        def textbbox(self, *args, **kwargs):
            return real_draw.textbbox(*args, **kwargs)

    # 12 个行业，8+4 分页，30 秒窗口下应分配为 20 秒 + 10 秒。
    module._draw_industry_panel(
        DrawProxyEarly(),
        panel,
        panel_x=10,
        panel_y=100,
        panel_w=460,
        panel_h=120,
        elapsed_seconds=19.9,
        duration_seconds=30.0,
    )
    module._draw_industry_panel(
        DrawProxyLate(),
        panel,
        panel_x=10,
        panel_y=100,
        panel_w=460,
        panel_h=120,
        elapsed_seconds=20.1,
        duration_seconds=30.0,
    )

    assert any(text.startswith("页1/") for text in early_texts)
    assert any(text.startswith("页2/") for text in late_texts)


def test_get_refresh_policy_stops_outside_refresh_after_today_cycle_data(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    refresh_waiting = module._get_refresh_policy(
        {"dates": ["20260401", "20260406"]},
        datetime(2026, 4, 7, 20, 0, 0),
    )
    refresh_done = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        datetime(2026, 4, 7, 20, 0, 0),
    )

    assert refresh_waiting == {"refresh_cycle": True, "refresh_realtime": False}
    assert refresh_done == {"refresh_cycle": False, "refresh_realtime": False}


def test_get_refresh_policy_keeps_cycle_and_realtime_refresh_intraday(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )
    outside_policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260406"]},
        datetime(2026, 4, 7, 20, 0, 0),
    )

    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: True)
    intraday_policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        datetime(2026, 4, 7, 10, 0, 0),
    )

    assert outside_policy == {"refresh_cycle": True, "refresh_realtime": False}
    assert intraday_policy == {"refresh_cycle": False, "refresh_realtime": True}


def test_get_refresh_policy_keeps_realtime_refresh_after_close_until_intraday_complete(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260406"]},
        intraday_chart_data={"trade_date": "20260407", "slot_indices": [0, 24]},
        now=datetime(2026, 4, 7, 15, 1, 0),
    )

    assert policy == {"refresh_cycle": True, "refresh_realtime": True}


def test_get_refresh_policy_stops_realtime_after_intraday_complete(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260406"]},
        intraday_chart_data={"trade_date": "20260407", "slot_indices": [0, 25]},
        now=datetime(2026, 4, 7, 15, 1, 0),
    )

    assert policy == {"refresh_cycle": True, "refresh_realtime": False}


def test_get_refresh_policy_stops_post_close_realtime_once_cycle_catches_up(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        intraday_chart_data={"trade_date": "20260407", "slot_indices": [0, 24]},
        now=datetime(2026, 4, 7, 15, 1, 0),
    )

    assert policy == {"refresh_cycle": False, "refresh_realtime": False}


def test_get_refresh_policy_stops_post_close_realtime_after_grace_deadline(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260406"]},
        intraday_chart_data={"trade_date": "20260407", "slot_indices": [0, 24]},
        now=datetime(2026, 4, 7, 15, 11, 0),
    )

    assert policy == {"refresh_cycle": True, "refresh_realtime": False}


def test_should_prefer_daily_holdings_snapshot_only_before_open_and_after_close(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    assert module._should_prefer_daily_holdings_snapshot(datetime(2026, 4, 7, 8, 0, 0)) is True
    assert module._should_prefer_daily_holdings_snapshot(datetime(2026, 4, 7, 12, 0, 0)) is False
    assert module._should_prefer_daily_holdings_snapshot(datetime(2026, 4, 7, 15, 1, 0)) is True


def test_get_refresh_policy_pauses_realtime_during_lunch(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260406",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        datetime(2026, 4, 7, 12, 0, 0),
    )

    assert policy == {"refresh_cycle": False, "refresh_realtime": False}


def test_get_refresh_policy_keeps_realtime_refresh_right_after_morning_close_until_last_slot_complete(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260406",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        intraday_chart_data={"trade_date": "20260407", "slot_indices": [0, 11]},
        now=datetime(2026, 4, 7, 11, 30, 1),
    )

    assert policy == {"refresh_cycle": False, "refresh_realtime": True}


def test_get_data_worker_wait_seconds_keeps_regular_interval_away_from_close(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 14, 0, 0))

    assert wait_seconds == float(module.REALTIME_REFRESH_INTERVAL)


def test_get_data_worker_wait_seconds_keeps_ten_minutes_outside_trading(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 20, 0, 0))

    assert wait_seconds == float(module.REFRESH_INTERVAL)


def test_get_data_worker_wait_seconds_keeps_short_interval_during_post_close_completion(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260408",
    )

    wait_seconds = module._get_data_worker_wait_seconds(
        datetime(2026, 4, 8, 15, 1, 0),
        cycle_chart_data={"dates": ["20260401", "20260407"]},
        intraday_chart_data={"trade_date": "20260408", "slot_indices": [0, 24]},
    )

    assert wait_seconds == float(module.REALTIME_REFRESH_INTERVAL)


def test_get_data_worker_wait_seconds_shortens_before_cycle_switch(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 14, 59, 58))

    assert wait_seconds == 3.0


def test_get_data_worker_wait_seconds_keeps_short_interval_during_morning_close_completion(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    wait_seconds = module._get_data_worker_wait_seconds(
        datetime(2026, 4, 8, 11, 30, 1),
        intraday_chart_data={"trade_date": "20260408", "slot_indices": [0, 11]},
    )

    assert wait_seconds == float(module.REALTIME_REFRESH_INTERVAL)


def test_get_data_worker_wait_seconds_wakes_immediately_after_1500_fetch(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 15, 0, 0))

    assert wait_seconds == 1.0


def test_get_refresh_policy_retries_latest_trade_day_on_weekend_when_missing(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260410",
    )

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260409"]},
        datetime(2026, 4, 11, 20, 0, 0),
    )

    assert policy == {"refresh_cycle": True, "refresh_realtime": False}


def test_is_realtime_refresh_due_respects_new_session_and_refresh_interval(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None, allow_load=False: True)
    last_refresh_at = datetime(2026, 4, 7, 9, 30, 0)

    due_on_open, session_key = module._is_realtime_refresh_due(
        True,
        datetime(2026, 4, 7, 9, 29, 0),
        None,
        datetime(2026, 4, 7, 9, 30, 0),
    )
    not_due_yet, _ = module._is_realtime_refresh_due(
        True,
        last_refresh_at,
        session_key,
        last_refresh_at + timedelta(seconds=module.REALTIME_REFRESH_INTERVAL - 1),
    )
    due_after_interval, _ = module._is_realtime_refresh_due(
        True,
        last_refresh_at,
        session_key,
        last_refresh_at + timedelta(seconds=module.REALTIME_REFRESH_INTERVAL),
    )

    assert due_on_open is True
    assert session_key == "20260407-am"
    assert not_due_yet is False
    assert due_after_interval is True


def test_is_realtime_refresh_due_continues_interval_after_close_when_allowed():
    module = _load_module()

    last_refresh_at = datetime(2026, 4, 7, 15, 0, 0)
    now = last_refresh_at + timedelta(seconds=module.REALTIME_REFRESH_INTERVAL + 30)
    due_now, session_key = module._is_realtime_refresh_due(
        True,
        last_refresh_at,
        "20260407-pm",
        now,
    )

    assert due_now is True
    assert session_key is None


def test_is_realtime_refresh_due_forces_refresh_right_after_morning_close_boundary():
    module = _load_module()

    due_now, session_key = module._is_realtime_refresh_due(
        True,
        datetime(2026, 4, 7, 11, 29, 0),
        "20260407-am",
        datetime(2026, 4, 7, 11, 30, 1),
    )

    assert due_now is True
    assert session_key is None


def test_is_cycle_refresh_due_refreshes_immediately_when_target_trade_day_changes(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260408",
    )

    due_now, target_date = module._is_cycle_refresh_due(
        {"dates": ["20260401", "20260407"]},
        True,
        datetime(2026, 4, 8, 14, 58, 0),
        "20260407",
        datetime(2026, 4, 8, 15, 0, 1),
    )

    assert due_now is True
    assert target_date == "20260408"


def test_fetch_cycle_chart_data_uses_same_day_cache_when_target_available(monkeypatch):
    module = _load_module()
    query_calls = []

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_rebalance_state(self):
            return {"last_rebalance_date": "20260401", "rebalance_freq": 5}

        def load_account_state(self):
            return SimpleNamespace(
                positions={"000001.SZ": SimpleNamespace(shares=100, buy_price=10.0)},
                cash=5000.0,
            )

    class DummyClient:
        def __init__(self, verbose=False):
            pass

        def query(self, api_name, **kwargs):
            query_calls.append((api_name, kwargs.get("ts_code")))
            if api_name == "index_daily":
                if kwargs.get("ts_code") == module.SHANGHAI_INDEX_CODE:
                    closes = [3000.0, 3030.0]
                elif kwargs.get("ts_code") == module.SHENZHEN_INDEX_CODE:
                    closes = [10000.0, 10100.0]
                else:
                    closes = [5000.0, 5050.0]
                return pd.DataFrame(
                    {
                        "trade_date": ["20260401", "20260407"],
                        "close": closes,
                    }
                )
            return pd.DataFrame(
                {
                    "trade_date": ["20260401", "20260407"],
                    "close": [10.0, 11.0],
                }
            )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr("src.lazybull.data.tushare_client.TushareClient", DummyClient)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    first_chart = module._fetch_cycle_chart_data()
    second_chart = module._fetch_cycle_chart_data()

    assert first_chart is not None
    assert second_chart is not None
    assert len(query_calls) == 4
    assert first_chart == second_chart


def test_fetch_cycle_chart_data_retries_until_target_trade_day_available(monkeypatch):
    module = _load_module()
    query_call_count = {"value": 0}

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_rebalance_state(self):
            return {"last_rebalance_date": "20260401", "rebalance_freq": 5}

        def load_account_state(self):
            return SimpleNamespace(
                positions={"000001.SZ": SimpleNamespace(shares=100, buy_price=10.0)},
                cash=5000.0,
            )

    class DummyClient:
        def __init__(self, verbose=False):
            pass

        def query(self, api_name, **kwargs):
            fetch_round = query_call_count["value"] // 4
            query_call_count["value"] += 1
            if api_name == "index_daily":
                trade_dates = ["20260401", "20260407"] if fetch_round == 0 else ["20260401", "20260408"]
                if kwargs.get("ts_code") == module.SHANGHAI_INDEX_CODE:
                    closes = [3000.0, 3030.0] if fetch_round == 0 else [3000.0, 3040.0]
                elif kwargs.get("ts_code") == module.SHENZHEN_INDEX_CODE:
                    closes = [10000.0, 10100.0] if fetch_round == 0 else [10000.0, 10150.0]
                else:
                    closes = [5000.0, 5050.0] if fetch_round == 0 else [5000.0, 5100.0]
                return pd.DataFrame(
                    {
                        "trade_date": trade_dates,
                        "close": closes,
                    }
                )
            trade_dates = ["20260401", "20260407"] if fetch_round == 0 else ["20260401", "20260408"]
            closes = [10.0, 11.0] if fetch_round == 0 else [10.0, 11.5]
            return pd.DataFrame(
                {
                    "trade_date": trade_dates,
                    "close": closes,
                }
            )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr("src.lazybull.data.tushare_client.TushareClient", DummyClient)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260408",
    )

    first_chart = module._fetch_cycle_chart_data()
    second_chart = module._fetch_cycle_chart_data()
    third_chart = module._fetch_cycle_chart_data()

    assert first_chart is not None
    assert second_chart is not None
    assert third_chart is not None
    assert first_chart["dates"][-1] == "20260407"
    assert second_chart["dates"][-1] == "20260408"
    assert third_chart["dates"][-1] == "20260408"
    assert query_call_count["value"] == 8


def test_fetch_cycle_chart_data_tolerates_string_account_numbers(monkeypatch):
    module = _load_module()

    class DummyStorage:
        def __init__(self, root_path=None, verbose=False, smb_reader=None):
            pass

        def load_rebalance_state(self):
            return {"last_rebalance_date": "20260401", "rebalance_freq": "5"}

        def load_account_state(self):
            return SimpleNamespace(
                positions={
                    "000001.SZ": SimpleNamespace(shares="100", buy_price="10.0"),
                    "000002.SZ": SimpleNamespace(shares="200", buy_price="20.0"),
                },
                cash="5000.0",
            )

    class DummyClient:
        def __init__(self, verbose=False):
            pass

        def query(self, api_name, **kwargs):
            if api_name == "index_daily":
                if kwargs.get("ts_code") == module.SHANGHAI_INDEX_CODE:
                    closes = [3000.0, 3030.0]
                elif kwargs.get("ts_code") == module.SHENZHEN_INDEX_CODE:
                    closes = [10000.0, 10150.0]
                else:
                    closes = [5000.0, 5075.0]
                return pd.DataFrame(
                    {
                        "trade_date": ["20260401", "20260407"],
                        "close": closes,
                    }
                )
            if kwargs.get("ts_code") == "000001.SZ":
                closes = [10.0, 11.0]
            else:
                closes = [20.0, 22.0]
            return pd.DataFrame(
                {
                    "trade_date": ["20260401", "20260407"],
                    "close": closes,
                }
            )

    monkeypatch.setattr("src.lazybull.paper.PaperStorage", DummyStorage)
    monkeypatch.setattr("src.lazybull.data.tushare_client.TushareClient", DummyClient)
    monkeypatch.setattr(
        module,
        "_get_target_cycle_data_date",
        lambda now=None, allow_load=False: "20260407",
    )

    chart = module._fetch_cycle_chart_data()

    assert chart is not None
    assert chart["dates"] == ["20260407"]
    assert chart["portfolio_pct"] == [0.0]


def test_refresh_display_state_reuses_single_holdings_snapshot(monkeypatch):
    module = _load_module()
    state = module.DisplayState()
    snapshot = {
        "positions": {"000001.SZ": SimpleNamespace(shares=100, buy_price=10.0)},
        "cash": 5000.0,
        "initial_capital": 100000.0,
        "current_date": "20260407",
        "annualized_return_func": None,
        "quotes": pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "PRICE": 11.0,
                    "PRE_CLOSE": 10.5,
                    "TIME": "09:32:00",
                }
            ]
        ),
    }
    fetch_calls = []
    seen_snapshots = []

    monkeypatch.setattr(
        module,
        "_fetch_realtime_holdings_snapshot",
        lambda: fetch_calls.append(state.is_updating) or snapshot,
    )
    monkeypatch.setattr(
        module,
        "_build_realtime_portfolio_summary",
        lambda payload: seen_snapshots.append(("summary", payload))
        or {
            "market_value": 1100.0,
            "total_assets": 6100.0,
            "float_pnl_pct": 10.0,
            "total_pnl_pct": -93.9,
            "annual_return_pct": 0.0,
            "pos_count": 1,
            "quote_time": "11:30:00",
        },
    )
    monkeypatch.setattr(
        module,
        "_build_stock_rankings",
        lambda payload: seen_snapshots.append(("rank", payload))
        or [{"name": "平安", "code": "000001", "pnl_pct": 10.0}],
    )
    monkeypatch.setattr(
        module,
        "_build_industry_panel",
        lambda payload, mode="cycle": seen_snapshots.append((f"industry-{mode}", payload))
        or {"industries": [{"industry": "银行"}], "mode": mode},
    )
    monkeypatch.setattr(
        module,
        "_build_intraday_chart",
        lambda chart_data, payload, point_time=None: seen_snapshots.append(("intraday", payload))
        or {"mode": "intraday", "dates": ["09:32"]},
    )
    monkeypatch.setattr(module, "_save_intraday_chart", lambda chart: None)
    monkeypatch.setattr(module, "_calc_rebalance_status", lambda: ("20260410", 3))

    module._refresh_display_state(state, refresh_realtime=True, refresh_cycle=False)

    assert fetch_calls == [True]
    assert [name for name, _ in seen_snapshots] == [
        "summary",
        "intraday",
        "rank",
        "industry-cycle",
        "industry-intraday",
    ]
    assert all(payload is snapshot for _, payload in seen_snapshots)
    assert state.summary is not None
    assert state.stock_rankings == [{"name": "平安", "code": "000001", "pnl_pct": 10.0}]
    assert state.industry_panel == {"industries": [{"industry": "银行"}], "mode": "cycle"}
    assert state.industry_panel_cycle == {"industries": [{"industry": "银行"}], "mode": "cycle"}
    assert state.industry_panel_intraday == {"industries": [{"industry": "银行"}], "mode": "intraday"}
    assert state.intraday_chart_data == {"mode": "intraday", "dates": ["09:32"]}
    assert state.next_rebalance_date == "20260410"
    assert state.days_to_rebalance == 3
    assert state.is_updating is False
    assert state.update_time == "11:30"


def test_call_with_timeout_returns_fallback_on_timeout(monkeypatch):
    module = _load_module()
    messages = []

    monkeypatch.setattr(
        module,
        "_emit_diag_once",
        lambda key, message, stderr=True: messages.append((key, message)),
    )

    def _slow_call():
        time.sleep(0.05)
        return 1

    result = module._call_with_timeout(
        _slow_call,
        timeout_seconds=0.001,
        fallback=99,
        timeout_diag_key="timeout-key",
        timeout_diag_message="timeout-message",
    )

    assert result == 99
    assert messages == [("timeout-key", "timeout-message")]


def test_refresh_display_state_timeout_does_not_stuck_updating(monkeypatch):
    module = _load_module()
    state = module.DisplayState()

    monkeypatch.setattr(module, "REALTIME_SNAPSHOT_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(module, "_calc_rebalance_status", lambda: ("20260510", 2))

    def _slow_snapshot():
        time.sleep(0.05)
        return None

    monkeypatch.setattr(module, "_fetch_realtime_holdings_snapshot", _slow_snapshot)

    module._refresh_display_state(state, refresh_realtime=True, refresh_cycle=False)

    assert state.is_updating is False
    assert state.next_rebalance_date == "20260510"
    assert state.days_to_rebalance == 2


def test_refresh_display_state_uses_cached_snapshot_when_timeout(monkeypatch):
    module = _load_module()
    state = module.DisplayState()

    cached_snapshot = {
        "quote_source": "A",
        "positions": {"000001.SZ": SimpleNamespace(shares=100, buy_price=10.0)},
        "quotes": pd.DataFrame(
            [
                {
                    "TS_CODE": "000001.SZ",
                    "NAME": "平安银行",
                    "PRICE": 11.0,
                    "PRE_CLOSE": 10.5,
                    "TIME": "10:01:00",
                }
            ]
        ),
    }

    monkeypatch.setattr(module, "REALTIME_SNAPSHOT_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(module, "_fetch_realtime_holdings_snapshot", lambda: time.sleep(0.05) or None)
    monkeypatch.setattr(module, "_get_cached_holdings_snapshot", lambda max_age_seconds=None: cached_snapshot)
    monkeypatch.setattr(
        module,
        "_build_realtime_portfolio_summary",
        lambda payload: {
            "market_value": 1100.0,
            "total_assets": 6100.0,
            "float_pnl_pct": 10.0,
            "total_pnl_pct": -93.9,
            "annual_return_pct": 0.0,
            "pos_count": 1,
            "quote_time": "10:01:00",
        }
        if payload is not None
        else None,
    )
    monkeypatch.setattr(module, "_build_intraday_chart", lambda chart_data, payload, point_time=None: chart_data)
    monkeypatch.setattr(module, "_save_intraday_chart", lambda chart: None)
    monkeypatch.setattr(
        module,
        "_build_stock_rankings",
        lambda payload: [{"name": "平安", "code": "000001", "pnl_pct": 10.0}] if payload else None,
    )
    monkeypatch.setattr(
        module,
        "_build_industry_panel",
        lambda payload, mode="cycle": {"industries": [{"industry": "银行"}], "mode": mode}
        if payload
        else None,
    )
    monkeypatch.setattr(module, "_calc_rebalance_status", lambda: ("20260510", 2))

    module._refresh_display_state(state, refresh_realtime=True, refresh_cycle=False)

    assert state.summary is not None
    assert state.stock_rankings is not None
    assert state.industry_panel is not None
    assert state.quote_source_tag == "A"


def test_render_header_uses_short_update_step(monkeypatch):
    module = _load_module()
    state = module.DisplayState()
    state.is_updating = True
    state.update_step = "抓快照阶段很长"

    captured = []

    def _capture_text(self, xy, text, fill=None, font=None, anchor=None, *args, **kwargs):
        captured.append(str(text))
        return None

    monkeypatch.setattr(module.ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(module, "_write_fb", lambda img: None)
    monkeypatch.setattr(module, "_refresh_system_usage_sample", lambda _state: (0.0, 0.0))
    monkeypatch.setattr(module, "_draw_system_usage_bar", lambda draw, cpu, mem: None)
    monkeypatch.setattr(module, "_select_chart_data", lambda cycle, intraday, now: None)
    monkeypatch.setattr(module, "_draw_chart_panel", lambda draw, chart_data, cycle_label, industry_panel: None)

    module._render(state)

    assert "更:抓快照阶段" in captured


def test_refresh_display_state_clears_update_step_after_done(monkeypatch):
    module = _load_module()
    state = module.DisplayState()

    monkeypatch.setattr(module, "_fetch_realtime_holdings_snapshot", lambda: None)
    monkeypatch.setattr(module, "_build_realtime_portfolio_summary", lambda snapshot: None)
    monkeypatch.setattr(module, "_build_intraday_chart", lambda chart_data, snapshot: chart_data)
    monkeypatch.setattr(module, "_build_stock_rankings", lambda snapshot: None)
    monkeypatch.setattr(module, "_build_industry_panel", lambda snapshot, mode="cycle": None)
    monkeypatch.setattr(module, "_calc_rebalance_status", lambda: ("20260510", 2))

    module._refresh_display_state(state, refresh_realtime=True, refresh_cycle=False)

    assert state.is_updating is False
    assert state.update_step == ""


def test_refresh_display_state_updates_time_when_summary_missing(monkeypatch):
    module = _load_module()
    state = module.DisplayState()

    snapshot = {
        "quote_source": "T",
        "positions": {},
        "quotes": pd.DataFrame(),
    }
    monkeypatch.setattr(module, "_fetch_realtime_holdings_snapshot", lambda: snapshot)
    monkeypatch.setattr(module, "_build_realtime_portfolio_summary", lambda payload: None)
    monkeypatch.setattr(module, "_build_intraday_chart", lambda chart_data, payload: chart_data)
    monkeypatch.setattr(module, "_build_stock_rankings", lambda payload: None)
    monkeypatch.setattr(module, "_build_industry_panel", lambda payload, mode="cycle": None)
    monkeypatch.setattr(module, "_calc_rebalance_status", lambda: ("20260510", 2))

    module._refresh_display_state(state, refresh_realtime=True, refresh_cycle=False)

    assert state.update_time != "--:--"


def test_render_watchdog_resets_stuck_updating_state(monkeypatch):
    module = _load_module()
    state = module.DisplayState()
    state.is_updating = True
    state.update_step = "抓快照"
    state.update_started_at = time.monotonic() - (module.UPDATE_STUCK_RESET_SECONDS + 5.0)

    monkeypatch.setattr(module, "_write_fb", lambda img: None)
    monkeypatch.setattr(module, "_refresh_system_usage_sample", lambda _state: (0.0, 0.0))
    monkeypatch.setattr(module, "_draw_system_usage_bar", lambda draw, cpu, mem: None)
    monkeypatch.setattr(module, "_select_chart_data", lambda cycle, intraday, now: None)
    monkeypatch.setattr(module, "_draw_chart_panel", lambda draw, chart_data, cycle_label, industry_panel: None)
    monkeypatch.setattr(module, "_emit_diag", lambda *args, **kwargs: None)

    module._render(state)

    assert state.is_updating is False
    assert state.update_step == ""
    assert state.update_started_at == 0.0


def test_render_header_shows_source_tag(monkeypatch):
    module = _load_module()
    state = module.DisplayState()
    state.is_updating = True
    state.update_step = "抓快照"
    state.quote_source_tag = "A"

    captured = []

    def _capture_text(self, xy, text, fill=None, font=None, anchor=None, *args, **kwargs):
        captured.append(str(text))
        return None

    monkeypatch.setattr(module.ImageDraw.ImageDraw, "text", _capture_text)
    monkeypatch.setattr(module, "_write_fb", lambda img: None)
    monkeypatch.setattr(module, "_refresh_system_usage_sample", lambda _state: (0.0, 0.0))
    monkeypatch.setattr(module, "_draw_system_usage_bar", lambda draw, cpu, mem: None)
    monkeypatch.setattr(module, "_select_chart_data", lambda cycle, intraday, now: None)
    monkeypatch.setattr(module, "_draw_chart_panel", lambda draw, chart_data, cycle_label, industry_panel: None)

    module._render(state)

    assert "更:抓快照[A]" in captured
