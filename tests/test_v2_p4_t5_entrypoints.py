# -*- coding: utf-8 -*-
"""CLI/Bot 共用路由、v2 展示与回退防呆的合成测试。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import dumps_blob  # noqa: E402
from src.lazybull.v2.hosts.paper import entrypoints  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402


def _config(root: Path) -> None:
    storage = PaperStorage(root_path=str(root))
    storage.save_config(
        {
            "model_version": 24022,
            "initial_capital": 1000.0,
            "buy_price": "close",
            "sell_price": "open",
            "top_n": 20,
            "rebalance_freq": 20,
            "horizon": 20,
        }
    )


def test_resolve_entry_engine_defaults_legacy_and_accepts_override(tmp_path: Path) -> None:
    _config(tmp_path / "paper")
    assert entrypoints.resolve_entry_engine(tmp_path / "paper") == "legacy"
    assert entrypoints.resolve_entry_engine(tmp_path / "paper", cli_override="v2") == "v2"


def test_resolve_entry_engine_rejects_invalid_override(tmp_path: Path) -> None:
    _config(tmp_path / "paper")
    with pytest.raises(RuntimeError, match="engine 切换键非法"):
        entrypoints.resolve_entry_engine(tmp_path / "paper", cli_override="other")


def test_load_v2_account_state_avoids_legacy_runner(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    store._atomic_write_json(
        store.state_path / "account.json",
        {
            "cash": 100.0,
            "last_update": "20261030",
            "positions": {
                "A": {
                    "shares": 100, "buy_price": 10.0, "buy_cost": 5.0,
                    "buy_date": "20261001", "buy_pnl_price": 12.0,
                }
            },
        },
    )
    account = entrypoints.load_v2_account_state(store)
    assert account.cash == 100.0
    assert account.positions["A"].buy_cost == 5.0
    assert account.positions["A"].buy_pnl_price == 12.0


def test_format_v2_positions_uses_clean_close(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    store = PaperStateStore(root_path=str(data_root / "paper"))
    store._atomic_write_json(
        store.state_path / "account.json",
        {
            "cash": 100.0,
            "last_update": "20261030",
            "positions": {
                "A": {"shares": 100, "buy_price": 10.0, "buy_cost": 5.0, "buy_date": "20261001"}
            },
        },
    )
    day_dir = data_root / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["A"], "close": [11.0], "pre_close": [10.0]}).to_parquet(
        day_dir / "2026-10-30.parquet"
    )
    text = entrypoints.format_v2_positions("20261030", store=store)
    assert "总资产: 1,200.00" in text
    assert "现价11.00" in text


def test_legacy_guard_allows_pre_v2_state(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    entrypoints.assert_legacy_rollback_allowed(store)


def test_legacy_guard_rejects_v2_blob_without_manifest(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    store._atomic_write_bytes(store.state_path / "engine_state.bin", b"v2")
    with pytest.raises(RuntimeError, match="拒绝 legacy 直接接回"):
        entrypoints.assert_legacy_rollback_allowed(store)


def test_legacy_guard_validates_both_identities(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    store._atomic_write_bytes(store.state_path / "engine_state.bin", b"v2")
    store._atomic_write_json(
        store.state_path / "account.json", {"cash": 1.0, "last_update": "20261030"}
    )
    manifest = {
        "validated": True,
        "rollback_date": "20261030",
        "engine_state_sha256": hashlib.sha256(b"v2").hexdigest(),
        "account_sha256": hashlib.sha256((store.state_path / "account.json").read_bytes()).hexdigest(),
    }
    store._atomic_write_json(store.state_path / "rollback_manifest.json", manifest)
    entrypoints.assert_legacy_rollback_allowed(store)
    store._atomic_write_json(
        store.state_path / "account.json", {"cash": 2.0, "last_update": "20261030"}
    )
    with pytest.raises(RuntimeError, match="哈希不符"):
        entrypoints.assert_legacy_rollback_allowed(store)


def test_legacy_guard_allows_post_rollback_run_marker(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    store._atomic_write_bytes(store.state_path / "engine_state.bin", b"v2")
    store._atomic_write_json(
        store.state_path / "account.json", {"cash": 3.0, "last_update": "20261102"}
    )
    manifest = {
        "validated": True,
        "rollback_date": "20261030",
        "engine_state_sha256": hashlib.sha256(b"v2").hexdigest(),
        "account_sha256": "rollback-day-hash",
    }
    store._atomic_write_json(store.state_path / "rollback_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="后续日期缺 legacy 运行标记"):
        entrypoints.assert_legacy_rollback_allowed(store)
    store._atomic_write_json(store.runs_path / "t1_20261102.json", {"status": "ok"})
    entrypoints.assert_legacy_rollback_allowed(store)


def test_complete_generation_rejects_half_commit(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    raw = dumps_blob({"fields": {}, "paper_generation": {"trade_date": "20261030", "run_kind": "t1"}})
    store._atomic_write_bytes(store.state_path / "engine_state.bin", raw)
    with pytest.raises(RuntimeError, match="半提交代"):
        entrypoints.assert_complete_generation(store)


def test_complete_generation_accepts_published_run_marker(tmp_path: Path) -> None:
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    raw = dumps_blob({"fields": {}, "paper_generation": {"trade_date": "20261030", "run_kind": "t1"}})
    store._atomic_write_bytes(store.state_path / "engine_state.bin", raw)
    store._atomic_write_json(store.runs_path / "t1_20261030.json", {"status": "ok"})
    entrypoints.assert_complete_generation(store)


def test_run_v2_day_rejects_not_ready(monkeypatch) -> None:
    class _Host:
        last_day_result = None

        def run_t1(self, day):  # noqa: ANN001, ANN202
            return None

    monkeypatch.setattr(entrypoints, "create_paper_host", lambda **kwargs: _Host())
    with pytest.raises(RuntimeError, match="未真实驱动"):
        entrypoints.run_v2_day("20261030")


def test_run_v2_day_passes_model_override_to_host(monkeypatch) -> None:
    captured = {}

    class _Result:
        driven = True

    class _Host:
        def __init__(self) -> None:
            self.last_day_result = None

        def run_t1(self, day):  # noqa: ANN001, ANN202
            self.last_day_result = _Result()

    host = _Host()

    def factory(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return host

    monkeypatch.setattr(entrypoints, "load_paper_config", lambda root: {"model_version": 1})
    monkeypatch.setattr(entrypoints, "create_paper_host", factory)
    entrypoints.run_v2_day("20261030", model_version=24022)
    assert captured["config_flat"]["model_version"] == 24022


def test_production_switch_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="显式授权"):
        entrypoints.assert_production_switch_allowed(tmp_path, authorized=False)


def test_production_switch_rejects_unpassed_gates(tmp_path: Path) -> None:
    (tmp_path / "closeout_status.json").write_text(
        json.dumps({"gates": {"gate1": "PASS", "gate5": "FAIL", "gate6": "PASS", "gate7": "PASS"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="gate5"):
        entrypoints.assert_production_switch_allowed(tmp_path, authorized=True)


def test_production_switch_accepts_all_required_gates(tmp_path: Path) -> None:
    (tmp_path / "closeout_status.json").write_text(
        json.dumps(
            {
                "verdict": "READY_FOR_FINAL_AUTHORIZATION",
                "gates": {
                    name: "PASS"
                    for name in ("gate1", "gate5", "gate6", "gate7", "gate8")
                },
                "statuses": {"technical_conditions_ready": True},
                "blocked_reasons": [],
            }
        ),
        encoding="utf-8",
    )
    entrypoints.assert_production_switch_allowed(tmp_path, authorized=True)


def test_production_switch_rejects_gate8_blocked(tmp_path: Path) -> None:
    (tmp_path / "closeout_status.json").write_text(
        json.dumps(
            {
                "verdict": "BLOCKED",
                "gates": {
                    "gate1": "PASS", "gate5": "PASS", "gate6": "PASS",
                    "gate7": "PASS", "gate8": "BLOCKED",
                },
                "statuses": {"technical_conditions_ready": False},
                "blocked_reasons": ["gate8"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="gate8"):
        entrypoints.assert_production_switch_allowed(tmp_path, authorized=True)
