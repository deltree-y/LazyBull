# -*- coding: utf-8 -*-
"""P7 T6 预登记 / 噪声带 / 裁定 / 门 / 黄金例回归（桩件驱动，零真实数据依赖）。

覆盖点
======
- 预登记：判据编号、阈值、verdict 规则、边界声明齐全；摘要稳定（两次调用逐位相同）；
- 噪声带：路径复算与 ``calculate_chain_metrics`` 一致；同 seed 重采样逐位一致；
  精确二项符号检验取值；
- 裁定：pass / fail / inconclusive 三种映射与「越带 = |Δ|≥δ 且带不含 0」的合取语义；
- 机械门：产物 SHA 不符、未登记文件、预登记晚于运行三类失败；
- 黄金例③：死局 / 迟回收手算数字逐项复现；
- 写域：run 根已存在 / 越界均拒绝（零副作用）。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.common import WriteScopeError, sha256_file
from scripts.v2_p7.t6 import calibration
from scripts.v2_p7.t6.gates import _g5, _g7, _pre_registration_ordering
from scripts.v2_p7.t6.golden import _deadlock_late_recovery_handcalc
from scripts.v2_p7.t6.noise import (
    _metrics_path,
    build_noise,
    returns_from_nav,
    same_split_mask,
    sign_test,
)
from scripts.v2_p7.t6.runner import _prepare_run_root
from scripts.v2_p7.t6.verdict import _over_band, build_verdict
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

# ────────────────────────── 预登记 ──────────────────────────


def test_calibration_payload_is_complete_and_stable() -> None:
    payload = calibration.payload()
    assert [item["id"] for item in payload["criteria"]] == ["C1", "C2", "C3", "C4", "C5"]
    assert payload["delta"] == {
        "total_return": 0.005,
        "max_drawdown": 0.005,
        "cagr": 0.001,
        "volatility": 0.005,
    }
    assert payload["band"]["block"] == 20
    assert payload["band"]["replicates"] == 4000
    assert payload["band"]["seed"] == 42
    assert payload["over_band_rule"] == "|Δ| ≥ δ 且 95% 区间不含 0"
    assert set(payload["verdict_rule"]) == {"pass", "fail", "inconclusive", "order"}
    assert payload["boundary"] and payload["not_in_scope"]
    assert payload["alpha_consumed"] is True
    assert calibration.digest() == calibration.digest()
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert "待预登记" not in text


# ────────────────────────── 噪声带 ──────────────────────────


def _random_nav(seed: int, days: int = 120, drift: float = 0.0004) -> List[float]:
    rng = np.random.default_rng(seed)
    values = [1.0]
    for _ in range(days - 1):
        values.append(values[-1] * (1.0 + rng.normal(drift, 0.01)))
    return values


def test_metrics_path_matches_chain_metrics_entry() -> None:
    """两个宇宙各自与唯一入口同源：全序列 → TR/MDD/cagr；同折 → vol/sharpe。"""
    nav = _random_nav(7)
    split = [0] * 40 + [1] * (len(nav) - 40)
    chain = calculate_chain_metrics(pd.DataFrame({"nav": nav, "split_index": split}))
    returns = returns_from_nav(nav)
    mask = same_split_mask(split)
    full = _metrics_path(returns[None, :], cagr_days=int(mask.sum()))
    same = _metrics_path(returns[mask][None, :])
    assert float(full["total_return"][0]) == pytest.approx(chain["total_return"], abs=1e-9)
    assert float(full["max_drawdown"][0]) == pytest.approx(chain["max_drawdown"], abs=1e-9)
    assert float(full["cagr"][0]) == pytest.approx(chain["cagr"], abs=1e-9)
    assert float(same["volatility"][0]) == pytest.approx(chain["volatility"], abs=1e-9)
    assert float(same["sharpe"][0]) == pytest.approx(chain["sharpe"], abs=1e-9)


def test_same_split_mask_matches_entry_convention() -> None:
    mask = same_split_mask([0, 0, 0, 1, 1, 0])
    assert mask.tolist() == [True, True, False, True, False]
    assert same_split_mask([0]).size == 0


def test_bootstrap_is_deterministic_for_same_seed() -> None:
    nav = {"vvh": _random_nav(11), "vvs": _random_nav(12)}
    split = [0] * 60 + [1] * (len(nav["vvh"]) - 60)
    points = {
        "total_return": 0.001,
        "max_drawdown": 0.0,
        "cagr": 0.0,
        "volatility": 0.0,
        "sharpe": 0.0,
    }
    first = build_noise(
        nav=nav,
        split_index=split,
        point_deltas=points,
        calibration=calibration.payload(),
        deltas_by_year=[0.01, -0.02],
        deltas_by_month=[0.001, -0.002],
    )
    second = build_noise(
        nav=nav,
        split_index=split,
        point_deltas=points,
        calibration=calibration.payload(),
        deltas_by_year=[0.01, -0.02],
        deltas_by_month=[0.001, -0.002],
    )
    assert first["primary"] == second["primary"]
    assert first["sensitivity"] == second["sensitivity"]
    assert first["days"]["full"] == len(nav["vvh"]) - 1
    assert first["days"]["same_split"] == len(nav["vvh"]) - 2
    assert first["days"]["dropped_at_split_boundary"] == 1
    for metric in ("total_return", "max_drawdown", "cagr", "volatility", "sharpe"):
        item = first["primary"]["metrics"][metric]
        assert item["low"] <= item["high"]
        assert item["band_width"] >= 0.0
        assert item["mde"] == pytest.approx(1.96 * item["se"])
        assert item["point"] == pytest.approx(points[metric])
    assert first["primary"]["metrics"]["volatility"]["universe"] == "same_split"
    assert first["primary"]["metrics"]["total_return"]["universe"] == "full"


def test_sign_test_exact_binomial() -> None:
    payload = sign_test([0.1, 0.2, 0.3, -0.1, -0.2, 0.4, 0.5, -0.3], unit="year")
    assert payload["positive"] == 5
    assert payload["negative"] == 3
    assert payload["same_sign_share"] == pytest.approx(5 / 8)
    assert payload["p_value"] == pytest.approx(0.7265625)
    empty = sign_test([], unit="month")
    assert empty["non_zero"] == 0 and empty["same_sign_share"] is None


# ────────────────────────── 裁定 ──────────────────────────


def _metrics_payload(
    *,
    delta_tr: float,
    delta_mdd: float,
    delta_cagr: float,
    delta_vol: float,
    delta_gross: float,
    turnover_delta: float,
    fee_delta: float,
    monthly: List[Dict[str, Any]],
    yearly: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def pair(control: float, delta: float) -> Dict[str, Any]:
        return {
            "control": control,
            "treatment": control + delta,
            "delta": delta,
            "relative_delta": delta / abs(control) if control else None,
        }

    return {
        "paired": {
            "total_return": pair(1.8, delta_tr),
            "max_drawdown": pair(-0.23, delta_mdd),
            "volatility": pair(0.16, delta_vol),
            "nav_end": pair(2.8, delta_tr),
            "cagr": pair(0.16, delta_cagr),
            "sharpe": pair(0.83, 0.0005),
            "fees": {"control": 188_856.5, "treatment": 188_856.5 + fee_delta, "delta": fee_delta},
            "turnover": {
                "control": 196.1,
                "treatment": 196.1 + turnover_delta,
                "delta": turnover_delta,
            },
            "attribution": {
                "physical_capital": 1_000_000.0,
                "delta_tr": delta_tr,
                "delta_fee_over_pc": fee_delta / 1_000_000.0,
                "delta_tr_fee": -(fee_delta / 1_000_000.0),
                "delta_tr_gross": delta_gross,
                "delta_turnover": turnover_delta,
                "same_sign_turnover_fee": turnover_delta * fee_delta >= 0.0,
            },
        },
        "monthly_paired": monthly,
        "by_year_paired": yearly,
    }


def _noise_payload(excludes: Dict[str, bool]) -> Dict[str, Any]:
    metrics = {
        key: {
            "point": 0.0,
            "boot_mean": 0.0,
            "low": -0.01 if not excludes.get(key) else 0.001,
            "high": 0.01 if not excludes.get(key) else 0.02,
            "se": 0.005,
            "excludes_zero": bool(excludes.get(key)),
            "band_width": 0.02,
        }
        for key in ("total_return", "max_drawdown", "cagr", "volatility", "sharpe")
    }
    return {"primary": {"block": 20, "replicates": 4000, "seed": 42, "metrics": metrics}}


def _monthly(deltas: List[float]) -> List[Dict[str, Any]]:
    return [
        {
            "month": f"2024{index + 1:02d}",
            "control": 0.01,
            "treatment": 0.01 + delta,
            "delta": delta,
        }
        for index, delta in enumerate(deltas)
    ]


def _yearly(deltas: List[float]) -> List[Dict[str, Any]]:
    return [
        {"year": 2020 + index, "control": 0.1, "treatment": 0.1 + delta, "delta": delta}
        for index, delta in enumerate(deltas)
    ]


def test_over_band_requires_both_economic_and_statistical() -> None:
    band = {"excludes_zero": True}
    assert _over_band(delta=0.008, band=band, threshold=0.005) is True
    assert _over_band(delta=0.002, band=band, threshold=0.005) is False
    assert _over_band(delta=0.008, band={"excludes_zero": False}, threshold=0.005) is False


def test_verdict_pass_requires_all_criteria() -> None:
    metrics = _metrics_payload(
        delta_tr=0.008,
        delta_mdd=0.008,
        delta_cagr=0.002,
        delta_vol=-0.006,
        delta_gross=0.004,
        turnover_delta=1.0,
        fee_delta=10.0,
        monthly=_monthly([0.0016] * 6),
        yearly=_yearly([0.002, 0.002, 0.002, 0.002]),
    )
    noise = _noise_payload(
        {
            "total_return": True,
            "max_drawdown": True,
            "cagr": True,
            "volatility": True,
        }
    )
    verdict = build_verdict(calibration=calibration.payload(), metrics=metrics, noise=noise)
    assert verdict["verdict"] == "pass"
    assert verdict["passed_ids"] == ["C1", "C2", "C3", "C4", "C5"]


def test_verdict_fail_on_opposite_over_band() -> None:
    metrics = _metrics_payload(
        delta_tr=-0.02,
        delta_mdd=0.001,
        delta_cagr=-0.001,
        delta_vol=0.001,
        delta_gross=-0.02,
        turnover_delta=0.0,
        fee_delta=0.0,
        monthly=_monthly([-0.01] * 6),
        yearly=_yearly([-0.01, -0.01, -0.01, -0.01]),
    )
    noise = _noise_payload(
        {"total_return": True, "max_drawdown": False, "cagr": True, "volatility": False}
    )
    verdict = build_verdict(calibration=calibration.payload(), metrics=metrics, noise=noise)
    assert verdict["verdict"] == "fail"
    assert "C1" in verdict["opposite_over_band_ids"]


def test_verdict_inconclusive_when_delta_below_materiality() -> None:
    metrics = _metrics_payload(
        delta_tr=0.0022,
        delta_mdd=0.0002,
        delta_cagr=0.0001,
        delta_vol=0.00002,
        delta_gross=0.002,
        turnover_delta=-0.22,
        fee_delta=-200.0,
        monthly=_monthly([0.0002, -0.0001, 0.0003, 0.0]),
        yearly=_yearly([0.0001, -0.002, 0.0002, 0.001, -0.001]),
    )
    noise = _noise_payload(
        {key: False for key in ("total_return", "max_drawdown", "cagr", "volatility")}
    )
    verdict = build_verdict(calibration=calibration.payload(), metrics=metrics, noise=noise)
    assert verdict["verdict"] == "inconclusive"
    assert verdict["opposite_over_band_ids"] == []
    failed = {item["id"] for item in verdict["criteria"] if not item["passed"]}
    assert {"C1", "C2", "C3"} <= failed
    assert any("C5" == item for item in failed)


def test_verify_paired_detects_flipped_delta() -> None:
    from scripts.v2_p7.t6.verify import _verify_paired

    metrics = _metrics_payload(
        delta_tr=0.002,
        delta_mdd=0.0001,
        delta_cagr=0.0001,
        delta_vol=0.00001,
        delta_gross=0.0018,
        turnover_delta=-0.2,
        fee_delta=-200.0,
        monthly=_monthly([0.001]),
        yearly=_yearly([0.001]),
    )
    metrics["paired"]["fees"]["treatment"] = metrics["paired"]["fees"]["control"] - 200.0
    metrics["paired"]["turnover"]["treatment"] = metrics["paired"]["turnover"]["control"] - 0.2
    assert _verify_paired(metrics)["ok"] is True
    metrics["paired"]["total_return"]["delta"] = -0.002
    assert _verify_paired(metrics)["ok"] is False


class _StubEnvelope:
    def state_digest(self) -> str:
        return "same-state"

    def digest(self) -> str:
        return "same-envelope"


class _StubArm:
    """最小宿主结果桩件（只覆盖 G5 聚合用到的字段）。"""

    def __init__(self, replay: Any, restart: Any) -> None:
        self.digest_series = [("20240102", "d1"), ("20240103", "d2")]
        self.nav_series = {"v20": [{"trade_date": "20240102", "nav": 1.0}]}
        self.trades = [{"trade_date": "20240102", "fee": 1.0}]
        self.conservation_rows = [{"trade_date": "20240102", "qty": 100}]
        self.balance_rows = [{"trade_date": "20240102", "sleeve": "v20"}]
        self.credit_payload = {"totals": {"granted": 0.0}}
        self.envelope = _StubEnvelope()
        self.replay = replay
        self.restart = restart


_OK_CHECK = {"checked": True, "all_bitwise_identical": True}


def test_g5_requires_host_self_checks() -> None:
    """C6-01：重跑逐位一致但宿主 replay/restart 自检失败/未执行时，G5 必须失败。"""
    from scripts.v2_p7.t6.arms import arm_determinism_payload, self_check_rows, self_checks_ok

    healthy = arm_determinism_payload(
        _StubArm(_OK_CHECK, _OK_CHECK), _StubArm(_OK_CHECK, _OK_CHECK)
    )
    assert healthy["bitwise_ok"] is True
    assert healthy["self_checks_ok"] is True
    assert healthy["ok"] is True

    replay_failed = arm_determinism_payload(
        _StubArm({"checked": True, "all_bitwise_identical": False}, _OK_CHECK),
        _StubArm(_OK_CHECK, _OK_CHECK),
    )
    assert replay_failed["bitwise_ok"] is True
    assert replay_failed["self_checks_ok"] is False
    assert replay_failed["ok"] is False

    restart_skipped = arm_determinism_payload(
        _StubArm(_OK_CHECK, _OK_CHECK),
        _StubArm(_OK_CHECK, {"checked": False, "note": "无检查点"}),
    )
    assert restart_skipped["ok"] is False

    missing = arm_determinism_payload(_StubArm(_OK_CHECK, None), _StubArm(_OK_CHECK, _OK_CHECK))
    assert missing["self_checks"]["first"]["restart"]["present"] is False
    assert missing["ok"] is False

    assert self_checks_ok(self_check_rows(_StubArm(_OK_CHECK, _OK_CHECK))) is True
    assert self_checks_ok({}) is False

    gate = _g5(
        {
            "ok": False,
            "failures": ["vvh 宿主重放/续跑自检未通过（checked/identical 不全为真）"],
            "arms": {"vvh": replay_failed},
        }
    )
    assert gate["ok"] is False
    assert gate["self_checks"]["vvh"]["ok"] is False


# ────────────────────────── 机械门 ──────────────────────────


def _touch(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8"))
    return sha256_file(path)


def test_g7_detects_sha_mismatch_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "run"
    calibration_digest = _touch(root / "calibration.json", "{}")
    digest = _touch(root / "metrics.json", "{}")
    arm_vvh = _touch(root / "arms" / "vvh" / "daily_vvh.parquet", "x")
    arm_vvs = _touch(root / "arms" / "vvs" / "daily_vvs.parquet", "x")
    artifacts = {
        "calibration.json": {"sha256": calibration_digest},
        "metrics.json": {"sha256": digest},
        "arms/vvh/daily_vvh.parquet": {"sha256": arm_vvh},
        "arms/vvs/daily_vvs.parquet": {"sha256": arm_vvs},
    }
    prereg = {
        "all_equal": True,
        "covers_current_run": True,
        "runs": {"run": {"criteria_digest": "x"}},
        "digests": ["x"],
    }
    good = _g7(artifacts=artifacts, run_root=root, prereg=prereg)
    assert good["ok"] is True
    assert good["extra_files"] == []
    fewer = {key: value for key, value in artifacts.items() if key != "metrics.json"}
    partial = _g7(artifacts=fewer, run_root=root, prereg=prereg)
    assert partial["ok"] is False
    assert partial["extra_files"] == ["metrics.json"]
    (root / "metrics.json").write_bytes(b'{"a":1}')
    bad = _g7(artifacts=artifacts, run_root=root, prereg=prereg)
    assert bad["ok"] is False
    assert bad["mismatch"]
    missing_prereg = _g7(artifacts=artifacts, run_root=root)
    assert missing_prereg["ok"] is False
    assert missing_prereg["prereg_stability"]["ok"] is False
    drifted = _g7(artifacts=artifacts, run_root=root, prereg={**prereg, "all_equal": False})
    assert drifted["ok"] is False


def test_pre_registration_ordering(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _touch(root / "calibration.json", "{}")
    _touch(root / "arms" / "vvh" / "daily_vvh.parquet", "x")
    _touch(root / "arms" / "vvs" / "daily_vvs.parquet", "x")
    ordering = _pre_registration_ordering(root)
    assert ordering["ok"] is True
    (root / "calibration.json").unlink()
    assert _pre_registration_ordering(root)["ok"] is False


# ────────────────────────── 黄金例③ ──────────────────────────


def test_golden_deadlock_and_late_recovery_hand_calc() -> None:
    case = _deadlock_late_recovery_handcalc(params=SimpleNamespace(physical_capital=1_000_000.0))
    assert case["ok"] is True, case["checks"]
    assert case["observed"]["credit_after_late"] == {"A": 0.0, "B": 0.0}
    assert case["observed"]["totals"]["late_recovery_total"] == 1_000.0
    assert case["observed"]["totals"]["deadlocked_total"] == 1_000.0
    assert set(case["observed"]["nav_effects"]) == {"grant", "deadlock", "late_recovery"}
    assert all(value == 0.0 for value in case["observed"]["nav_effects"].values())
    assert case["observed"]["equity_after_deadlock"] == {"A": 5000.0, "B": 1000.0}


# ────────────────────────── 写域 ──────────────────────────


def test_prepare_run_root_refuses_existing_and_outside(tmp_path: Path) -> None:
    with pytest.raises(WriteScopeError):
        _prepare_run_root(tmp_path / "outside")
    existing = Path("data/paper_shadow/p7_testbed/p7_t5_20261113_r15")
    if existing.exists():
        with pytest.raises(FileExistsError):
            _prepare_run_root(existing)
