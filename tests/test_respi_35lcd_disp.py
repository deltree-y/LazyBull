import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lcd_35_disp",
        PROJECT_ROOT / "scripts" / "respi" / "3.5LCD_disp.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_format_display_time_uses_new_chinese_style():
    module = _load_module()

    formatted = module._format_display_time(datetime(2026, 4, 7, 14, 40, 32))

    assert formatted == "4月7日(周二) 14:40:32"

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


def test_resolve_framebuffer_path_uses_env_override(monkeypatch):
    monkeypatch.setenv("LAZYBULL_LCD_FB_PATH", "/dev/fb9")
    module = _load_module()

    assert module._resolve_framebuffer_path() == "/dev/fb9"


def test_layout_constants_use_taller_header_and_narrower_left_panel():
    module = _load_module()

    assert module.HEADER_H == 34
    assert module.HEADER_TIME_FONT_SIZE >= module.HEADER_META_FONT_SIZE
    assert module.PANEL_TOP == module.HEADER_H + 4
    assert module.LEFT_W == int(module.PANEL_AREA_W * 0.575)
    assert module.RIGHT_W == module.PANEL_AREA_W - module.LEFT_W - module.PANEL_GAP


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


def test_upsert_intraday_chart_uses_fixed_slots_and_replaces_same_slot():
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
    assert chart["slot_indices"] == [0, 1]
    assert chart["raw_index_pct"] == [0.8, 1.1]
    assert chart["raw_shenzhen_pct"] == [0.6, 0.9]
    assert chart["raw_portfolio_pct"] == [1.5, 1.8]
    assert chart["index_pct"] == [0.0, 0.3]
    assert chart["shenzhen_pct"] == [0.0, 0.3]
    assert chart["portfolio_pct"] == [0.0, 0.3]
    assert chart["x_start_label"] == "09:30"
    assert chart["x_end_label"] == "15:00"
    assert chart["portfolio_label"] == "持仓"
    assert chart["shenzhen_label"] == "深证"


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
    assert normalized["raw_index_pct"] == [0.6]
    assert normalized["raw_shenzhen_pct"] == [0.4]
    assert normalized["raw_portfolio_pct"] == [1.1]
    assert normalized["index_pct"] == [0.0]
    assert normalized["shenzhen_pct"] == [0.0]
    assert normalized["portfolio_pct"] == [0.0]


def test_intraday_chart_persistence_restores_same_day_history(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "project_root", tmp_path)

    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 30, 0),
        index_pct=0.6,
        shenzhen_pct=0.4,
        portfolio_pct=1.1,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 40, 0),
        index_pct=0.9,
        shenzhen_pct=0.7,
        portfolio_pct=1.3,
    )

    module._save_intraday_chart(chart)
    restored = module._load_intraday_chart(now=datetime(2026, 4, 7, 14, 0, 0))
    next_day = module._load_intraday_chart(now=datetime(2026, 4, 8, 9, 30, 0))

    assert restored is not None
    assert restored["slot_indices"] == [0, 1]
    assert restored["raw_index_pct"] == [0.6, 0.9]
    assert restored["raw_shenzhen_pct"] == [0.4, 0.7]
    assert restored["raw_portfolio_pct"] == [1.1, 1.3]
    assert restored["index_pct"] == [0.0, 0.3]
    assert restored["shenzhen_pct"] == [0.0, 0.3]
    assert restored["portfolio_pct"] == [0.0, 0.2]
    assert next_day is None
    assert module._get_intraday_chart_state_path("20260407").exists()


def test_normalize_intraday_chart_rebases_legacy_payload_to_zero_start():
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
    assert normalized["raw_index_pct"] == [0.8, 1.1]
    assert normalized["raw_shenzhen_pct"] == [0.6, 0.9]
    assert normalized["raw_portfolio_pct"] == [1.5, 1.8]
    assert normalized["index_pct"] == [0.0, 0.3]
    assert normalized["shenzhen_pct"] == [0.0, 0.3]
    assert normalized["portfolio_pct"] == [0.0, 0.3]


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


def test_select_chart_data_switches_by_intraday_window(monkeypatch):
    module = _load_module()

    cycle_chart = {"mode": "cycle"}
    intraday_chart = {"mode": "intraday", "trade_date": "20260407"}
    point_time = datetime(2026, 4, 7, 14, 40, 0)

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "intraday"

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "cycle"


def test_is_intraday_chart_window_starts_after_open(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None: True)

    assert module._is_intraday_chart_window(datetime(2026, 4, 7, 9, 5, 0)) is False
    assert module._is_intraday_chart_window(datetime(2026, 4, 7, 9, 30, 0)) is True


def test_build_intraday_chart_skips_pre_open_snapshot(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)
    monkeypatch.setattr(
        module,
        "_fetch_realtime_index_pcts",
        lambda: {module.SHANGHAI_INDEX_CODE: 0.1, module.SHENZHEN_INDEX_CODE: 0.2},
    )
    monkeypatch.setattr(module, "_compute_holdings_intraday_pct", lambda snapshot: 0.3)

    chart = module._build_intraday_chart(
        None,
        {"positions": {}, "quotes": pd.DataFrame()},
        point_time=datetime(2026, 4, 7, 9, 5, 0),
    )

    assert chart is None


def test_format_cycle_last_data_label_uses_last_cycle_date():
    module = _load_module()

    label = module._format_cycle_last_data_label(
        {"dates": ["20260401", "20260407"]}
    )

    assert label == "周期图最后数据日:04/07"


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
    monkeypatch.setattr(module, "_format_cycle_last_data_label", lambda chart: "周期图最后数据日:04/07")
    monkeypatch.setattr(
        module,
        "_draw_chart",
        lambda draw, chart_data, cycle_last_data_label=None: captured.update(
            {"mode": chart_data.get("mode"), "label": cycle_last_data_label}
        ),
    )
    monkeypatch.setattr(module, "_write_fb", lambda img: None)

    module._render(state)

    assert captured == {"mode": "intraday", "label": None}


def test_get_refresh_policy_stops_outside_refresh_after_today_cycle_data(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None: True)

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

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None: True)
    outside_policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260406"]},
        datetime(2026, 4, 7, 20, 0, 0),
    )

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: True)
    intraday_policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        datetime(2026, 4, 7, 10, 0, 0),
    )

    assert outside_policy == {"refresh_cycle": True, "refresh_realtime": False}
    assert intraday_policy == {"refresh_cycle": True, "refresh_realtime": True}


def test_get_refresh_policy_pauses_realtime_during_lunch(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    monkeypatch.setattr(module, "_is_realtime_quote_window", lambda now=None: False)

    policy = module._get_refresh_policy(
        {"dates": ["20260401", "20260407"]},
        datetime(2026, 4, 7, 12, 0, 0),
    )

    assert policy == {"refresh_cycle": False, "refresh_realtime": False}


def test_get_data_worker_wait_seconds_keeps_regular_interval_away_from_close(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 14, 0, 0))

    assert wait_seconds == float(module.REFRESH_INTERVAL)


def test_get_data_worker_wait_seconds_shortens_before_cycle_switch(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 14, 59, 58))

    assert wait_seconds == 3.0


def test_get_data_worker_wait_seconds_wakes_immediately_after_1500_fetch(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_is_trade_day", lambda now=None: True)

    wait_seconds = module._get_data_worker_wait_seconds(datetime(2026, 4, 8, 15, 0, 0))

    assert wait_seconds == 1.0