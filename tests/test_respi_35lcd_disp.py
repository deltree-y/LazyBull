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
        datetime(2026, 4, 7, 8, 34, 0),
        index_pct=0.5,
        shenzhen_pct=0.3,
        portfolio_pct=1.2,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 8, 39, 0),
        index_pct=0.8,
        shenzhen_pct=0.6,
        portfolio_pct=1.5,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 8, 41, 0),
        index_pct=1.1,
        shenzhen_pct=0.9,
        portfolio_pct=1.8,
    )

    assert chart["slot_count"] == module.INTRADAY_SLOT_COUNT
    assert chart["slot_indices"] == [0, 1]
    assert chart["index_pct"] == [0.8, 1.1]
    assert chart["shenzhen_pct"] == [0.6, 0.9]
    assert chart["portfolio_pct"] == [1.5, 1.8]
    assert chart["x_start_label"] == "08:30"
    assert chart["x_end_label"] == "15:30"
    assert chart["portfolio_label"] == "持仓当日"
    assert chart["shenzhen_label"] == "深证日内"


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


def test_intraday_chart_persistence_restores_same_day_history(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "project_root", tmp_path)

    chart = module._upsert_intraday_chart(
        None,
        datetime(2026, 4, 7, 9, 0, 0),
        index_pct=0.6,
        shenzhen_pct=0.4,
        portfolio_pct=1.1,
    )
    chart = module._upsert_intraday_chart(
        chart,
        datetime(2026, 4, 7, 9, 10, 0),
        index_pct=0.9,
        shenzhen_pct=0.7,
        portfolio_pct=1.3,
    )

    module._save_intraday_chart(chart)
    restored = module._load_intraday_chart(now=datetime(2026, 4, 7, 14, 0, 0))
    next_day = module._load_intraday_chart(now=datetime(2026, 4, 8, 9, 30, 0))

    assert restored is not None
    assert restored["slot_indices"] == [3, 4]
    assert restored["index_pct"] == [0.6, 0.9]
    assert restored["shenzhen_pct"] == [0.4, 0.7]
    assert restored["portfolio_pct"] == [1.1, 1.3]
    assert next_day is None
    assert module._get_intraday_chart_state_path("20260407").exists()


def test_select_chart_data_switches_by_intraday_window(monkeypatch):
    module = _load_module()

    cycle_chart = {"mode": "cycle"}
    intraday_chart = {"mode": "intraday", "trade_date": "20260407"}
    point_time = datetime(2026, 4, 7, 14, 40, 0)

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "intraday"

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    assert module._select_chart_data(cycle_chart, intraday_chart, point_time)["mode"] == "cycle"


def test_get_refresh_policy_keeps_cycle_refresh_outside_intraday(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: False)
    outside_policy = module._get_refresh_policy(datetime(2026, 4, 7, 20, 0, 0))

    monkeypatch.setattr(module, "_is_intraday_chart_window", lambda now=None: True)
    intraday_policy = module._get_refresh_policy(datetime(2026, 4, 7, 10, 0, 0))

    assert outside_policy == {"refresh_cycle": True, "refresh_realtime": False}
    assert intraday_policy == {"refresh_cycle": True, "refresh_realtime": True}