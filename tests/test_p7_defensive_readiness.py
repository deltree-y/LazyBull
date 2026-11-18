"""DEF-ML-60-v1 R0 装配测试：隔离根端到端，缺件与单位错误必须显式暴露。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import readiness, spec

PROTOCOL_TEXT = "# synthetic protocol\n\nDEF-ML-60-v1 test fixture\n"
CODES = ["000001.SZ", "600000.SH", "600087.SH"]


def _build_minimal_root(tmp_path: Path, amount_is_yuan: bool = False) -> Path:
    """构造最小合成仓库根（只含 R0 会读的输入）。"""
    root = tmp_path / "repo"
    protocol = root / spec.PROTOCOL_RELATIVE
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(PROTOCOL_TEXT, encoding="utf-8")

    daily_dir = root / "data" / "clean" / "daily"
    basic_dir = root / "data" / "clean" / "daily_basic"
    daily_dir.mkdir(parents=True, exist_ok=True)
    basic_dir.mkdir(parents=True, exist_ok=True)
    amount = 1000.0 * (1000.0 if amount_is_yuan else 1.0)
    for date_dash in ("2019-04-01", "2026-10-29"):
        date8 = date_dash.replace("-", "")
        pd.DataFrame(
            {
                "ts_code": CODES,
                "trade_date": [date8] * 3,
                "open": [10.0] * 3,
                "high": [10.2] * 3,
                "low": [9.8] * 3,
                "close": [10.0] * 3,
                "vol": [1000.0] * 3,
                "amount": [amount] * 3,
                "adj_factor": [2.0] * 3,
                "close_adj": [20.0] * 3,
                "is_st": [0] * 3,
                "tradable": [1] * 3,
            }
        ).to_parquet(daily_dir / f"{date_dash}.parquet")
        pd.DataFrame(
            {
                "ts_code": CODES,
                "trade_date": [date8] * 3,
                "close": [10.0] * 3,
                "total_mv": [100000.0] * 3,
                "circ_mv": [100000.0] * 3,
                "total_share": [10000.0] * 3,
                "float_share": [8000.0] * 3,
                "turnover_rate": [3.0] * 3,
            }
        ).to_parquet(basic_dir / f"{date_dash}.parquet")

    clean = root / "data" / "clean"
    pd.DataFrame(
        {
            "exchange": ["SSE"] * 3,
            "cal_date": ["20130401", "20190401", "20261029"],
            "is_open": [1, 1, 1],
        }
    ).to_parquet(clean / "trade_cal.parquet")
    basic = pd.DataFrame(
        {
            "ts_code": CODES,
            "symbol": ["000001", "600000", "600087"],
            "name": ["a", "b", "c"],
            "area": ["x"] * 3,
            "industry": ["y"] * 3,
            "market": ["主板"] * 3,
            "list_date": ["19910403", "19991110", "19970101"],
        }
    )
    basic.to_parquet(clean / "stock_basic.parquet")
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    basic.to_parquet(root / "data" / "raw" / "stock_basic.parquet")
    pd.DataFrame(
        {
            "code6": ["000001", "600000", "600087"],
            "ts_code": CODES,
            "in_date": ["20140101"] * 3,
            "out_date": [None, "20180102", None],
            "ind_code": ["801780"] * 3,
            "l1_code": ["801780", "801780", None],
            "l2_code": ["801781", "801781", None],
            "l1_name": ["银行", "银行", None],
            "l2_name": ["银行", "银行", None],
            "l3_name": ["银行", "银行", None],
            "index_code": ["x", "x", None],
            "sw_version": ["SW2021", "SW2021", "UNMAPPED"],
            "source_updated_at": pd.Timestamp("2026-11-12"),
            "code_alias": [True, True, False],
        }
    ).to_parquet(clean / "sw_industry_pit.parquet")
    for name in ("cashflow", "dividend", "suspend", "stk_limit", "stock_st"):
        (root / "data" / "raw" / name).mkdir(parents=True, exist_ok=True)
    return root


def _write_a_baseline(root: Path) -> str:
    """写合成 A 基准并返回 manifest 字节 SHA。"""
    base = root / spec.A_BASELINE["relative"]
    base.mkdir(parents=True, exist_ok=True)
    manifest = base / "manifest.json"
    manifest.write_bytes(b'{"files": {}}\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for arm in spec.A_BASELINE["required_arm_dirs"]:
        arm_dir = base / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "initial_capital": 1_000_000,
            "initial_point": spec.WINDOW["cash_anchor"],
            "start": spec.WINDOW["start"],
            "end": spec.WINDOW["end"],
            "position_sizing": "half_kelly",
            "not_virgin_holdout": True,
            "cross_sleeve_netting": False,
            "capital_scaled_from_other_nav": False,
        }
        (arm_dir / "baseline_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False), encoding="utf-8"
        )
    return digest


def test_readiness_blocks_when_a_baseline_missing(tmp_path: Path) -> None:
    root = _build_minimal_root(tmp_path)
    result = readiness.run_readiness(
        repo_root=root, campaign="camp_test", run_id="readiness_t1",
        hash_content=False, artifact_base=tmp_path / "artifacts",
    )
    report = result["readiness"]
    assert report["status"] == "blocked"
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["a_baseline_identity"] == "fail"
    assert report["formal_admission"] == "not_evaluated"
    assert report["production_changed"] is False
    assert Path(result["readiness_path"]["path"]).exists()
    assert (Path(result["run_dir"]) / "manifest.json").exists()


def test_readiness_seals_protocol_and_spec(tmp_path: Path) -> None:
    root = _build_minimal_root(tmp_path)
    result = readiness.run_readiness(
        repo_root=root, campaign="camp_test", run_id="readiness_t2",
        hash_content=False, artifact_base=tmp_path / "artifacts",
    )
    camp = Path(result["run_dir"]).parents[1]
    snapshot = camp / "specification" / "protocol_snapshot.md"
    assert snapshot.read_text(encoding="utf-8") == PROTOCOL_TEXT
    # Windows 文本模式会把 LF 写成 CRLF，身份必须取磁盘字节
    expected = hashlib.sha256((root / spec.PROTOCOL_RELATIVE).read_bytes()).hexdigest()
    assert result["readiness"]["protocol_sha256"] == expected
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
    spec_doc = json.loads((camp / "specification" / "spec.json").read_text(encoding="utf-8"))
    assert spec_doc["candidate_id"] == spec.CANDIDATE_ID
    assert spec_doc["protocol_sha256"] == expected
    auth = json.loads((camp / "specification" / "authorization.json").read_text(encoding="utf-8"))
    assert auth["authorized_stages"] == "not_filled_by_user"
    assert auth["data_purchase_budget_yuan"] == 0


def test_readiness_flags_unit_mismatch_as_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_minimal_root(tmp_path, amount_is_yuan=True)
    digest = _write_a_baseline(root)
    monkeypatch.setitem(spec.A_BASELINE, "expected_manifest_sha256", digest)
    monkeypatch.setattr(
        readiness.sources,
        "warmup_availability",
        lambda _root: {
            "warmup_ok": True,
            "partitions_before_first_fold": 252,
            "warmup_required": 252,
            "oldest_partition": "2011-10-10",
            "latest_partition": "2026-10-29",
            "first_fold_signal_start": "20130401",
            "anchor_for_refresh": "20190401",
        },
    )
    result = readiness.run_readiness(
        repo_root=root, campaign="camp_test", run_id="readiness_t3",
        hash_content=False, artifact_base=tmp_path / "artifacts",
    )
    report = result["readiness"]
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["units"] == "fail"
    assert report["status"] == "blocked"


def test_code_identity_degrades_without_git_or_files(tmp_path: Path) -> None:
    identity = readiness._code_identity(tmp_path)
    assert identity["git_head"].startswith("not_available")
    assert set(identity["closure_sha256"].values()) == {"missing"}


def test_status_aggregation_prefers_worst() -> None:
    ok = [{"name": "a", "status": "pass"}, {"name": "b", "status": "warn"}]
    assert readiness._status_of(ok) == "ready_with_limits"
    ok.append({"name": "c", "status": "fail"})
    assert readiness._status_of(ok) == "blocked"
    assert readiness._status_of([{"name": "a", "status": "pass"}]) == "ready"
    assert readiness._status_of([{"name": "a", "status": "not_estimable"}]) == "ready_with_limits"
