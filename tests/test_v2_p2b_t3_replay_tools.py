"""P2b-T3：装配、runs 证据、三条比较边与 NAV 前置的合成测试。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p2a import replay_b0
from scripts.v2_p2b import replay_merger
from scripts.v2_p2b.check_frozen_inputs import (
    canonical_digest_note,
    digest_b0_raw_reference,
    digest_p2a_runs_reference,
    resolve_b0_raw_reference,
)
from scripts.v2_p2b.check_workspace_state import canonical_json
from scripts.v2_p2b.compare_merger_batches import (
    _validate_raw_reference,
    compare_merger_batches,
    validate_nav_finite,
)
from scripts.v2_p2b.replay_merger import P2B_SOURCE
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.runs_loader import RunsBatch, load_runs_batch
from src.lazybull.v2.hosts.backtest import runs_writer
from src.lazybull.v2.hosts.backtest.runtime import (
    LEGACY_BYPASS,
    MERGER_SLEEVE_ID,
    SINGLE_SLEEVE_PASSTHROUGH,
    get_merger_call_count,
    validate_merger_mode,
)

SPLITS = (0, 1)
BASE_CONFIG = {
    "bt_top_n": 20,
    "label_column": "neu_y_ret_20",
    "enable_fundamental": True,
    "stock_domain": "main",
    "wf_start_date": "20240102",
    "wf_end_date": "20241231",
}


def _nav(dates, values):
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(value) for value in dates],
            "portfolio_value": values,
            "capital": [value * 0.4 for value in values],
            "market_value": [value * 0.6 for value in values],
        }
    )
    frame["nav"] = frame["portfolio_value"] / 1_000_000.0
    frame["return"] = frame["nav"] - 1.0
    return frame


def _metrics(nav, start, end):
    return {
        "bt_total_return": round(float(nav["return"].iloc[-1]), 6),
        "bt_annual_return": 0.02,
        "bt_max_drawdown": -0.01,
        "bt_volatility": 0.1,
        "bt_sharpe": 1.0,
        "bt_calmar": 2.0,
        "bt_trading_days": len(nav),
        "bt_start": start,
        "bt_end": end,
        "bt_top_n": 20,
    }


def _fold(split):
    if split == 0:
        dates = ["2024-01-02", "2024-01-03"]
        stock = "A"
        train = ("20180101", "20231229")
        test = ("20240102", "20240628")
        bt_end = "20240702"
        version = 24008
    else:
        dates = ["2024-07-01", "2024-07-02"]
        stock = "C"
        train = ("20180701", "20240628")
        test = ("20240701", "20241231")
        bt_end = "20250101"
        version = 24009
    nav = _nav(dates, [1_000_000.0, 1_010_000.0 if split == 0 else 1_020_000.0])
    trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(dates[0]),
                "signal_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                "stock": stock,
                "action": "buy",
                "price": 10.0 + split,
                "shares": 100,
                "amount": 1000.0 + split * 100,
                "cost": 5.0,
                "buy_date": pd.Timestamp(dates[0]),
                "buy_price": 10.0 + split,
                "buy_type": "rebalance",
                "buy_reason": "signal",
            }
        ]
    )
    attribution = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                "ranking_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                "execution_date": pd.Timestamp(dates[0]),
                "execution_stage": "t1",
                "tranche_idx": 0,
                "planned_stock": stock,
                "actual_stock": stock,
                "planned_rank": 1,
                "actual_rank": 1,
                "pred_score": 0.9,
                "target_weight": 0.05,
                "status": "filled",
                "reason": None,
                "buy_price": 10.0 + split,
                "signal_price": 9.9 + split,
                "signal_to_buy_return": 0.01,
            }
        ]
    )
    snapshot = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(day),
                "ts_code": stock,
                "shares": 100,
                "market_value": 1000.0,
                "weight": 0.001,
                "portfolio_value": portfolio_value,
                "buy_date": pd.Timestamp(dates[0]),
                "signal_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                "holding_days": index,
                "planned_exit_date": None,
                "remaining_intervals": None,
            }
            for index, (day, portfolio_value) in enumerate(zip(dates, nav["portfolio_value"]))
        ]
    )
    return {
        "split_index": split,
        "model_version": version,
        "train_start": train[0],
        "train_end": train[1],
        "test_start": test[0],
        "test_end": test[1],
        "metrics": _metrics(nav, dates[0].replace("-", ""), bt_end),
        "nav_curve": nav,
        "trades": trades,
        "attribution": attribution,
        "holdings_snapshot": snapshot,
    }


def _workspace_document():
    payload = {
        "scope": ["src/lazybull", "scripts"],
        "files": {"src/lazybull/example.py": "a" * 64},
        "deleted_files": [],
    }
    return {
        **payload,
        "digest": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
    }


def _evidence_notes(mode, counts, p2a_digest, raw_digest):
    return {
        "merger.mode": mode,
        "merger.sleeve_id": MERGER_SLEEVE_ID,
        "merger.call_count": str(sum(counts.values())),
        "merger.call_count_by_fold": json.dumps(counts, sort_keys=True, separators=(",", ":")),
        "workspace.digest": canonical_json(_workspace_document()),
        "environment.versions": canonical_json({"python": "3.12", "packages": {"pandas": "3"}}),
        "reference.p2a_runs.digest": canonical_digest_note(p2a_digest),
        "reference.b0_raw.digest": canonical_digest_note(raw_digest),
    }


def _write_batch(
    root, batch_id, *, mode=None, counts=None, baseline_ref=None, notes=None, folds=SPLITS
):
    config = {**BASE_CONFIG, "batch_run_id": batch_id}
    if mode is not None:
        config.update({"merger_mode": mode, "merger_sleeve_id": MERGER_SLEEVE_ID})
    return runs_writer.write_runs_batch(
        [_fold(split) for split in folds],
        batch_id=batch_id,
        runs_root=Path(root) / "runs",
        config=config,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={
            "data_state_id": "syn_data",
            "wf_run_id": batch_id,
            "batch_run_id": batch_id,
            "sources": {"daily": "2024-07-02"},
        },
        baseline_ref=baseline_ref,
        arms=() if mode is None else (mode,),
        source="v2_p2a_native_replay" if mode is None else P2B_SOURCE,
        extra_meta_notes=notes,
    )


def _make_raw(reference_dir, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    batch = load_runs_batch(reference_dir)
    batch.summary.drop(columns=["batch_id"]).to_csv(
        raw_dir / "walk_forward_summary_0101_0001.csv", index=False, encoding="utf-8-sig"
    )
    batch.chain_nav.to_csv(raw_dir / "chain_nav_wf_syn.csv", index=False, encoding="utf-8-sig")
    for split, fold in batch.folds.items():
        tag = f"split{split:02d}"
        trades = fold.trades.rename(columns={"trade_date": "date", "ts_code": "stock"})
        trades = trades.drop(columns=["lot_id", "tranche_idx"])
        trades.to_csv(
            raw_dir / f"walk_forward_trades_syn_{tag}.csv", index=False, encoding="utf-8-sig"
        )
        attr = fold.attribution.rename(
            columns={"planned_ts_code": "planned_stock", "actual_ts_code": "actual_stock"}
        )
        attr.to_csv(
            raw_dir / f"walk_forward_execution_attribution_syn_{tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return raw_dir


def _make_pair(tmp_path, folds=SPLITS):
    reference = _write_batch(tmp_path / "reference", "p2a_reference")
    raw = _make_raw(reference, tmp_path / "raw")
    p2a_digest = digest_p2a_runs_reference(reference, SPLITS)
    raw_refs = resolve_b0_raw_reference(raw, SPLITS)
    raw_digest = digest_b0_raw_reference(raw, raw_refs)
    closed_counts = {str(split): 0 for split in folds}
    enabled_counts = {str(split): split + 2 for split in folds}
    closed = _write_batch(
        tmp_path / "closed",
        "p2b_closed",
        mode=LEGACY_BYPASS,
        baseline_ref="p2a_reference",
        notes=_evidence_notes(LEGACY_BYPASS, closed_counts, p2a_digest, raw_digest),
        folds=folds,
    )
    enabled = _write_batch(
        tmp_path / "enabled",
        "p2b_enabled",
        mode=SINGLE_SLEEVE_PASSTHROUGH,
        baseline_ref="p2b_closed",
        notes=_evidence_notes(SINGLE_SLEEVE_PASSTHROUGH, enabled_counts, p2a_digest, raw_digest),
        folds=folds,
    )
    return closed, enabled, reference, raw


def _compare(closed, enabled, reference, raw, **kwargs):
    return compare_merger_batches(
        closed,
        enabled,
        reference,
        raw,
        input_identity_verified=True,
        expected_splits=SPLITS,
        **kwargs,
    )


def _edit_meta(batch_dir, mutate):
    path = Path(batch_dir) / "batch_meta.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def _edit_meta_and_rehash(batch_dir, mutate):
    def _rehash(document):
        mutate(document)
        document["config_fingerprint"] = fingerprint(document["config"])

    _edit_meta(batch_dir, _rehash)


def test_runtime_default_bypass_and_enabled_assembly():
    assert validate_merger_mode(None) == SINGLE_SLEEVE_PASSTHROUGH
    assert validate_merger_mode(LEGACY_BYPASS) == LEGACY_BYPASS
    assert get_merger_call_count(SimpleNamespace()) == 0
    with pytest.raises(ValueError, match="未知 P2b 合并模式"):
        validate_merger_mode("silent-bypass")


def test_replay_b0_config_snapshot_excludes_mode_and_notes_use_actual_counts(monkeypatch):
    split = replay_b0.WalkForwardSplit(0, "20120103", "20180101", "20180102", "20180702")
    monkeypatch.setattr(replay_b0, "get_cost_settings", lambda: {"commission_rate": 0.0002})
    snapshot = replay_b0.build_manifest_config_snapshot([split])
    config = dict(snapshot["driver_config"])
    replay_b0._apply_merger_mode_to_config(config, SINGLE_SLEEVE_PASSTHROUGH)
    assert config["merger_mode"] == SINGLE_SLEEVE_PASSTHROUGH
    assert "merger_mode" not in snapshot["driver_config"]
    results = [
        {"split_index": 0, "merger_call_count": 2},
        {"split_index": 1, "merger_call_count": 3},
    ]
    notes = replay_b0._build_replay_meta_notes(results, SINGLE_SLEEVE_PASSTHROUGH, None)
    assert notes["merger.call_count"] == "5"
    assert json.loads(notes["merger.call_count_by_fold"]) == {"0": 2, "1": 3}
    default_notes = replay_b0._build_replay_meta_notes(results, None, None)
    assert default_notes["merger.mode"] == SINGLE_SLEEVE_PASSTHROUGH
    assert default_notes["merger.call_count"] == "5"
    assert default_notes["merger.net_call_count"] == "0"


def test_replay_b0_unknown_mode_and_note_conflict_fail_before_real_config(monkeypatch):
    monkeypatch.setattr(
        replay_b0,
        "get_data_root",
        lambda: pytest.fail("不得在参数校验前读取真实配置"),
    )
    with pytest.raises(ValueError, match="未知 P2b 合并模式"):
        replay_b0.replay_b0(merger_mode="unknown")
    with pytest.raises(ValueError, match="合并内建证据键冲突"):
        replay_b0.replay_b0(
            merger_mode=LEGACY_BYPASS,
            extra_meta_notes={"merger.call_count": "999"},
        )
    with pytest.raises(ValueError, match="合并内建证据键冲突"):
        replay_b0.replay_b0(extra_meta_notes={"merger.mode": LEGACY_BYPASS})


def test_writer_default_notes_and_extra_conflict(tmp_path):
    reference = _write_batch(tmp_path, "default_notes")
    meta = json.loads((reference / "batch_meta.json").read_text(encoding="utf-8"))
    assert set(meta["meta_notes"]) == {"topk_detail.missing", "topk_detail.missing_folds"}
    with pytest.raises(ValueError, match="内建证据键冲突"):
        runs_writer.write_runs_batch(
            [_fold(0)],
            batch_id="conflict",
            runs_root=tmp_path / "runs2",
            config=dict(BASE_CONFIG),
            code_state={"git_commit": "syn", "git_dirty": False},
            data_state={"data_state_id": "syn"},
            extra_meta_notes={"topk_detail.missing": "false"},
        )
    with pytest.raises(ValueError, match="内建证据键冲突"):
        runs_writer.write_runs_batch(
            [_fold(0)],
            batch_id="merger_conflict",
            runs_root=tmp_path / "runs3",
            config=dict(BASE_CONFIG),
            code_state={"git_commit": "syn", "git_dirty": False},
            data_state={"data_state_id": "syn"},
            extra_meta_notes={"merger.mode": LEGACY_BYPASS},
        )


def test_replay_merger_arm_wires_source_notes_and_detects_post_change(monkeypatch, tmp_path):
    preflight = {
        "workspace": {
            "workspace": _workspace_document(),
            "environment": {"python": "3.12", "packages": {}},
        },
        "frozen": {
            "model_report": {"selected": {"24008": {"model_path": "model"}}},
            "p2a_runs": {"digest": "p2a"},
            "b0_raw": {"digest": "raw"},
        },
    }
    calls = []
    monkeypatch.setattr(replay_merger, "run_preflight", lambda **kwargs: preflight)

    def _replay(**kwargs):
        calls.append(kwargs)
        return tmp_path / "out"

    monkeypatch.setattr(replay_merger.b0_driver, "replay_b0", _replay)
    out = replay_merger.replay_merger_arm(
        merger_mode=SINGLE_SLEEVE_PASSTHROUGH,
        baseline_ref="closed",
        data_root=str(tmp_path / "data"),
        repo_root=str(replay_merger.PROJECT_ROOT),
    )
    assert out == tmp_path / "out"
    assert calls[0]["source"] == P2B_SOURCE
    assert calls[0]["arms"] == (SINGLE_SLEEVE_PASSTHROUGH,)
    assert calls[0]["extra_meta_notes"]["reference.p2a_runs.digest"] == canonical_digest_note(
        preflight["frozen"]["p2a_runs"]
    )

    changed = {
        **preflight,
        "workspace": {
            "workspace": {**_workspace_document(), "digest": "changed"},
            "environment": {},
        },
    }
    responses = iter([preflight, changed])
    monkeypatch.setattr(replay_merger, "run_preflight", lambda **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="运行前后证据变化"):
        replay_merger.replay_merger_arm(
            merger_mode=LEGACY_BYPASS,
            data_root=str(tmp_path / "data"),
            repo_root=str(replay_merger.PROJECT_ROOT),
        )


def test_replay_merger_rejects_mismatched_repo_root_before_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(
        replay_merger,
        "run_preflight",
        lambda **kwargs: pytest.fail("异根拒绝前不得执行预检"),
    )
    with pytest.raises(ValueError, match="repo_root 与实际执行代码根不一致"):
        replay_merger.replay_merger_arm(
            merger_mode=LEGACY_BYPASS,
            data_root=str(tmp_path / "data"),
            repo_root=str(tmp_path / "other_checkout"),
        )


def test_three_edges_positive(tmp_path):
    report = _compare(*_make_pair(tmp_path))
    assert report.ok, report.errors
    assert all(report.edge_results.values())


@pytest.mark.parametrize(
    "values",
    ([1.0, np.nan, 1.0], [np.nan, np.nan, np.nan], [np.inf, np.inf, np.inf], [np.nan, np.inf, 1.0]),
    ids=["nan", "all_nan", "all_inf", "nan_inf"],
)
def test_nav_nonfinite_errors_have_location(values):
    batch = RunsBatch(
        batch_id="syn_batch",
        directory=Path("syn_batch"),
        batch_meta={},
        summary=pd.DataFrame(),
        chain_nav=pd.DataFrame({"date": [0, 1, 2], "nav": values, "split_index": [0, 0, 0]}),
        folds={},
    )
    errors = validate_nav_finite(batch)
    assert errors
    assert all(
        "syn_batch" in message and "split00" in message and "列=nav" in message
        for message in errors
    )


def test_three_edges_reject_common_nan_and_common_snapshot_regression(tmp_path):
    closed, enabled, reference, raw = _make_pair(tmp_path / "nan")
    for batch_dir in (closed, enabled):
        path = batch_dir / "chain_nav.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.index[0], "nav"] = np.nan
        frame.to_parquet(path, index=False)
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok
    assert any("非有限值" in message and "p2b_closed" in message for message in report.errors)

    closed, enabled, reference, raw = _make_pair(tmp_path / "snapshot")
    for batch_dir in (closed, enabled):
        path = batch_dir / "folds" / "split00" / "holdings_snapshot.parquet"
        frame = pd.read_parquet(path)
        frame["market_value"] += 123.0
        frame.to_parquet(path, index=False)
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok
    assert report.edge_results["edge1_closed_vs_p2a_runs"] is False
    assert any("holdings_snapshot" in message for message in report.errors)


def test_identity_duplicate_fold_and_call_count_mutations_rejected(tmp_path):
    closed, enabled, reference, raw = _make_pair(tmp_path)
    path = closed / "folds" / "split00" / "holdings_snapshot.parquet"
    frame = pd.read_parquet(path)
    frame["run_id"] = "wrong"
    frame.to_parquet(path, index=False)
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok and any("run_id" in message for message in report.errors)

    closed, enabled, reference, raw = _make_pair(tmp_path / "duplicate")
    path = closed / "summary.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok and any("重复 split_index" in message for message in report.errors)

    closed, enabled, reference, raw = _make_pair(tmp_path / "count")
    _edit_meta(enabled, lambda meta: meta["meta_notes"].__setitem__("merger.call_count", "99"))
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok and any("sum(call_count_by_fold)" in message for message in report.errors)


def test_extra_business_key_and_unregistered_note_rejected(tmp_path):
    closed, enabled, reference, raw = _make_pair(tmp_path / "business")
    path = enabled / "summary.parquet"
    frame = pd.read_parquet(path)
    frame["bt_top_n"] = 30
    frame.to_parquet(path, index=False)
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok and any("bt_top_n" in message for message in report.errors)

    closed, enabled, reference, raw = _make_pair(tmp_path / "note")
    for batch_dir in (closed, enabled):
        _edit_meta(batch_dir, lambda meta: meta["meta_notes"].__setitem__("rogue", "x"))
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok and any("未登记 meta_notes" in message for message in report.errors)


def test_subset_requires_explicit_flag(tmp_path):
    closed, enabled, reference, raw = _make_pair(tmp_path, folds=(0,))
    strict = _compare(closed, enabled, reference, raw)
    assert not strict.ok
    subset = _compare(closed, enabled, reference, raw, allow_subset=True)
    assert subset.ok, subset.errors


def test_input_identity_proof_is_mandatory(tmp_path):
    closed, enabled, reference, raw = _make_pair(tmp_path)
    report = compare_merger_batches(closed, enabled, reference, raw, expected_splits=SPLITS)
    assert not report.ok
    assert all("门 1" in message for message in report.errors)


def test_raw_nonfinite_nav_rejected_before_tolerance_comparison(tmp_path):
    closed, enabled, reference, raw = _make_pair(tmp_path)
    path = raw / "chain_nav_wf_syn.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "nav"] = np.nan
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    report = _compare(closed, enabled, reference, raw)
    assert not report.ok
    assert any("raw chain_nav" in message and "非有限值" in message for message in report.errors)


def test_meta_config_summary_projection_mismatch_rejected(tmp_path):
    for case in ("wrong_value", "rogue_key", "wrong_batch_id"):
        closed, enabled, reference, raw = _make_pair(tmp_path / case)

        def _mutate(meta, case=case):
            if case == "wrong_value":
                meta["config"]["bt_top_n"] = 30
            elif case == "rogue_key":
                meta["config"]["rogue_business_key"] = "unregistered"
            else:
                meta["config"]["batch_id"] = "some_other_batch"

        _edit_meta_and_rehash(enabled, _mutate)
        report = _compare(closed, enabled, reference, raw)
        assert not report.ok, case
        assert any("summary 首行投影" in message for message in report.errors), case


def test_raw_reference_requires_expected_fold_set():
    summary = pd.DataFrame({"split_index": [0]})
    chain = pd.DataFrame({"date": [0], "nav": [1.0], "split_index": [0]})
    errors = _validate_raw_reference(summary, chain, Path("raw"), expected=(0, 1))
    assert any("折集必须等于预期" in message for message in errors)
