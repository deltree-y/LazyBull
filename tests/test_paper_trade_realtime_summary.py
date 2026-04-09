import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "paper_trade_script",
        PROJECT_ROOT / "scripts" / "paper_trade.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_realtime_portfolio_summary_falls_back_to_pre_close(monkeypatch):
    module = _load_module()

    class DummyRunner:
        def __init__(self, verbose=False):
            self.account = SimpleNamespace(
                initial_capital=100000.0,
                get_positions=lambda: {
                    "000001.SZ": SimpleNamespace(shares=100, buy_price=10.0),
                },
                get_cash=lambda: 5000.0,
            )
            self.broker = SimpleNamespace(
                _calculate_annualized_return=lambda initial, total, current_date: 12.3
            )

    class DummyStorage:
        def load_config(self):
            return {"initial_capital": 100000.0}

    class DummyClient:
        def __init__(self, verbose=False):
            pass

        def get_realtime_quote(self, ts_codes_str):
            return pd.DataFrame(
                [
                    {
                        "TS_CODE": "000001.SZ",
                        "PRICE": 0.0,
                        "PRE_CLOSE": 11.0,
                        "TIME": "09:01:00",
                    }
                ]
            )

    monkeypatch.setattr(module, "PaperTradingRunner", DummyRunner)
    monkeypatch.setattr(module, "PaperStorage", DummyStorage)
    monkeypatch.setattr("src.lazybull.data.tushare_client.TushareClient", DummyClient)

    summary = module.get_realtime_portfolio_summary()

    assert summary is not None
    assert summary["market_value"] == 1100.0
    assert summary["total_assets"] == 6100.0
    assert round(summary["float_pnl_pct"], 4) == 10.0
    assert summary["quote_time"] == "09:01:00"