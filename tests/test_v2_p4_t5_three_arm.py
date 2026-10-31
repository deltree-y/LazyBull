# -*- coding: utf-8 -*-
"""三臂隔离（首次迁移起跑）与 exposure 断言的合成测试。"""

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

from scripts.v2_p4 import t5_three_arm as arms  # noqa: E402
from scripts.v2_p4.t5_gate7_rehearsal import (  # noqa: E402
    _DAYS,
    _make_data,
    _make_host,
    _make_legacy_source,
)
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.paper.models import AccountState, Position  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config  # noqa: E402


def _frozen_account_hash(root: Path) -> str:
    return hashlib.sha256((root / "state" / "account.json").read_bytes()).hexdigest()


def _gate1(frozen: Path, **overrides) -> dict:
    payload = {
        "gate1_frozen": True,
        "verdict": "PASS",
        "account": {"sha256": _frozen_account_hash(frozen), "last_update": _DAYS[0]},
        "data_state": {"data_state_id": "state-1"},
        "model": {"content_fingerprint": "model-1"},
        "execution": {"fingerprint": "exec-1"},
        "q11a_restart": {"selected_days": list(_DAYS[:1])},
    }
    payload.update(overrides)
    return payload


def _prepare(tmp_path: Path, frozen: Path, gate1: dict) -> dict:
    return arms.prepare_three_arms(
        frozen_paper_root=frozen,
        work_root=tmp_path / "work",
        gate1=gate1,
        data_root=tmp_path / "data",
        evidence=tmp_path / "evidence" / "initial_migration",
        host_builder=_make_host,
    )


def _fixture(tmp_path: Path) -> Path:
    """非空旧账户 + 无 engine_state.bin 的冻结源（含持仓/成交/pending/分批锚/稀疏 NAV）。"""
    _make_data(tmp_path / "data")
    frozen = tmp_path / "frozen"
    _make_legacy_source(frozen)
    storage = PaperStorage(root_path=str(frozen))
    config = storage.load_config() or {}
    config.update({"exposure_policy": "arm=combined,lambda=0.5", "exposure_replenish": True})
    storage.save_config(config)
    storage.save_account_state(
        AccountState(
            cash=60000.0,
            positions={
                "A": Position(
                    ts_code="A",
                    shares=4000,
                    buy_price=10.0,
                    buy_cost=5.0,
                    buy_date=_DAYS[0],
                    buy_pnl_price=10.0,
                )
            },
            last_update=_DAYS[0],
        )
    )
    pd.DataFrame(
        {
            "trade_date": [_DAYS[0]],
            "ts_code": ["A"],
            "action": ["buy"],
            "shares": [4000],
            "price": [10.0],
            "total_cost": [5.0],
            "signal_date": [_DAYS[0]],
        }
    ).to_parquet(storage.trades_path / "trades.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": [_DAYS[0]],
            "cash": [60000.0],
            "position_value": [40000.0],
            "total_value": [100000.0],
            "nav": [1.0],
        }
    ).to_parquet(storage.nav_path / "nav.parquet", index=False)
    return frozen


def test_prepare_three_arms_migrates_v2_and_matches_frozen_account(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    specs = _prepare(tmp_path, frozen, _gate1(frozen))
    assert set(specs) == set(arms.ARM_NAMES)
    assert load_paper_config(specs[arms.ARM_ACTIVE].root)["engine"] == "legacy"
    assert load_paper_config(specs[arms.ARM_NEUTRAL].root)["engine"] == "legacy"
    assert load_paper_config(specs[arms.ARM_V2].root)["engine"] == "v2"
    legacy_hashes = {
        (specs[name].root / "state" / "account.json").read_bytes()
        for name in (arms.ARM_ACTIVE, arms.ARM_NEUTRAL)
    }
    assert len(legacy_hashes) == 1
    frozen_account = json.loads((frozen / "state" / "account.json").read_text(encoding="utf-8"))
    migrated_account = json.loads(
        (specs[arms.ARM_V2].root / "state" / "account.json").read_text(encoding="utf-8")
    )
    assert migrated_account["cash"] == frozen_account["cash"]
    assert set(migrated_account["positions"]) == set(frozen_account["positions"])
    assert (specs[arms.ARM_V2].root / "state" / "engine_state.bin").is_file()
    marker = arms.load_init_marker(tmp_path / "work")
    assert marker is not None
    assert marker["migration"]["account_projection_equivalent"] is True
    assert marker["arms"][arms.ARM_V2]["complete_generation"] is True


def test_prepare_three_arms_nulls_only_validation_arm_exposure(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    specs = _prepare(tmp_path, frozen, _gate1(frozen))
    active = load_paper_config(specs[arms.ARM_ACTIVE].root)
    neutral = load_paper_config(specs[arms.ARM_NEUTRAL].root)
    v2 = load_paper_config(specs[arms.ARM_V2].root)
    assert active["exposure_policy"] is not None
    assert neutral["exposure_policy"] is None
    assert v2["exposure_policy"] is None


def test_prepare_three_arms_rejects_unfrozen_gate1(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="门 1 未冻结"):
        arms.prepare_three_arms(
            frozen_paper_root=frozen,
            work_root=tmp_path / "work",
            gate1={"gate1_frozen": False, "verdict": "BLOCKED_Q11A"},
            data_root=tmp_path / "data",
        )


def test_prepare_three_arms_never_overwrites_existing_generation(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    (tmp_path / "work" / arms.ARM_V2).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        _prepare(tmp_path, frozen, _gate1(frozen))


def test_second_day_resumes_by_marker_instead_of_remigrating(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    gate1 = _gate1(frozen)
    first = _prepare(tmp_path, frozen, gate1)
    marker_before = arms.load_init_marker(tmp_path / "work")
    engine_state_sha = (first[arms.ARM_V2].root / "state" / "engine_state.bin").read_bytes()
    second = _prepare(tmp_path, frozen, gate1)
    assert {name: spec.root for name, spec in first.items()} == {
        name: spec.root for name, spec in second.items()
    }
    assert (
        second[arms.ARM_V2].root / "state" / "engine_state.bin"
    ).read_bytes() == engine_state_sha
    assert arms.load_init_marker(tmp_path / "work") == marker_before


def test_frozen_source_drift_rejects_initialization(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    gate1 = _gate1(frozen)
    storage = PaperStorage(root_path=str(frozen))
    account = storage.load_account_state()
    account.cash += 1.0
    storage.save_account_state(account)
    with pytest.raises(RuntimeError, match="冻结源账户与门 1 身份不一致"):
        _prepare(tmp_path, frozen, gate1)


def test_gate1_identity_drift_rejects_resume(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    _prepare(tmp_path, frozen, _gate1(frozen))
    changed = _gate1(frozen, data_state={"data_state_id": "state-2"})
    with pytest.raises(RuntimeError, match="run 身份"):
        _prepare(tmp_path, frozen, changed)


def test_half_initialized_root_is_rejected_without_marker(tmp_path: Path) -> None:
    frozen = _fixture(tmp_path)
    work = tmp_path / "work"
    (work / arms.ARM_NEUTRAL).mkdir(parents=True)
    (work / arms.ARM_NEUTRAL / "state").mkdir()
    with pytest.raises(FileExistsError, match="半初始化"):
        _prepare(tmp_path, frozen, _gate1(frozen))
    assert (work / arms.ARM_NEUTRAL).is_dir()


def test_neutral_exposure_assertion_fail_fast() -> None:
    with pytest.raises(RuntimeError, match="exposure 非 null"):
        arms.assert_neutral_exposure({"exposure_policy": "arm=combined"})
    arms.assert_neutral_exposure({"exposure_policy": None})
    arms.assert_neutral_exposure({"exposure_policy": ""})


def test_effective_identity_excludes_implementation_arm_name() -> None:
    legacy = arms._effective_identity(
        {"engine": "legacy", "buy_price": "close", "sell_price": "open"}
    )
    v2 = arms._effective_identity({"engine": "v2", "buy_price": "close", "sell_price": "open"})
    assert legacy == v2
    assert "engine" not in legacy
    assert legacy["exposure_policy"] is None


def test_legacy_pre_state_uses_v2_position_field_semantics(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    storage = PaperStorage(root_path=str(root))
    (root / "state" / "account.json").write_text(
        json.dumps(
            {
                "cash": 100.0,
                "last_update": "20261030",
                "positions": {
                    "A": {
                        "ts_code": "A",
                        "shares": 100,
                        "buy_price": 10.0,
                        "buy_cost": 5.0,
                        "buy_date": "20261001",
                        "buy_pnl_price": 10.0,
                        "buy_atr_pct": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    state = arms._legacy_pre_state(storage)
    position = state["account"]["positions"]["A"]
    assert position["buy_trade_price"] == 10.0
    assert position["buy_cost_cash"] == 1005.0


def test_features_for_day_uses_actual_yyyymmdd_key() -> None:
    frame = pd.DataFrame({"ts_code": ["A"], "f": [1.0]})
    assert arms._features_for_day({"20260717": frame}, pd.Timestamp("2026-07-17")) is frame
    assert arms._features_for_day({pd.Timestamp("2026-07-17"): frame}, pd.Timestamp("2026-07-17")) is None


def test_v2_capture_shares_merger_target_list() -> None:
    class _Host:
        pass

    targets = []
    capture = arms.V2DecisionCapture(_Host(), "20260717", {}, targets=targets)
    assert capture.targets is targets
