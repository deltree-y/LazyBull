import importlib.util
from datetime import datetime
from pathlib import Path


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


def test_build_cycle_chart_payload_keeps_fixed_slots_before_holding_exceeds_period():
    module = _load_module()

    payload = module._build_cycle_chart_payload(
        dates=[f"202604{day:02d}" for day in range(1, 12)],
        index_pct=[float(i) for i in range(11)],
        portfolio_pct=[float(i) * 0.5 for i in range(11)],
        rebalance_freq=20,
        base_value=100.0,
    )

    assert payload is not None
    assert payload["slot_count"] == 20
    assert payload["slot_indices"][-1] == 10
    assert payload["x_end_label"] == "20天"


def test_build_cycle_chart_payload_expands_slots_after_holding_exceeds_period():
    module = _load_module()

    payload = module._build_cycle_chart_payload(
        dates=[f"202604{day:02d}" for day in range(1, 24)],
        index_pct=[float(i) for i in range(23)],
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
        datetime(2026, 4, 7, 8, 34, 0),
        index_pct=0.5,
        portfolio_pct=1.2,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 8, 39, 0),
        index_pct=0.8,
        portfolio_pct=1.5,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 8, 41, 0),
        index_pct=1.1,
        portfolio_pct=1.8,
    )

    assert chart["slot_count"] == module.INTRADAY_SLOT_COUNT
    assert chart["slot_indices"] == [0, 1]
    assert chart["index_pct"] == [0.8, 1.1]
    assert chart["portfolio_pct"] == [1.5, 1.8]
    assert chart["x_start_label"] == "08:30"
    assert chart["x_end_label"] == "15:30"


def test_select_chart_data_switches_by_intraday_window(monkeypatch):
    module = _load_module()

    cycle_chart = {"mode": "cycle"}
    intraday_chart = {"mode": "intraday", "trade_date": "20260407"}
    point_time = datetime(2026, 4, 7, 14, 40, 0)

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "intraday"

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "cycle"