# -*- coding: utf-8 -*-
"""P7 T0.5 评审整改（R1-T05-01~06）的定点测试：不依赖真实生产数据。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "v2_p7"
sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"p7_t05_{name}", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def precheck():
    return _load("precheck_t05_inputs")


@pytest.fixture(scope="module")
def common():
    return _load("common")


@pytest.fixture(scope="module")
def crash_spec():
    return _load("crash_spec")


@pytest.fixture(scope="module")
def governance():
    return _load("build_t05_governance")


@pytest.fixture(scope="module")
def index_source():
    return _load("index_source")


# ---------------------------------------------------------------- R1-T05-01 日历截断


def test_truncate_calendar_drops_tail_and_reports(precheck):
    calendar = ["20240101", "20240102", "20240103", "20240104"]
    bounded, info = precheck.truncate_calendar(calendar, "20240102")
    assert bounded == ["20240101", "20240102"]
    assert info["dropped_trading_days"] == 2
    assert info["calendar_last_date"] == "20240104"
    assert info["bounded_last_date"] == "20240102"
    assert info["truncated_at_cutoff"] is True


def test_truncate_calendar_rejects_empty(precheck):
    with pytest.raises(ValueError):
        precheck.truncate_calendar(["20240102"], "20240101")


def test_entry_positions_beyond_cutoff_counted_missing(precheck):
    valid, missing = precheck.entry_positions_within([0, 3, 4], 5)
    assert valid == [1, 4]
    assert missing == 1


def test_effective_blocks_ignore_calendar_tail(precheck):
    """块内成熟才计有效；guard 与未成熟入场不得被日历尾端「补成」有效。"""
    bounded = precheck._block_table(np.asarray([0, 25], dtype=np.int64), 30, 10)
    assert bounded["blocks_total"] == 1
    assert bounded["effective_blocks"] == 1
    assert bounded["events_in_guard_or_outside"] == 1
    assert bounded["events_matured_in_blocks"] == 1
    late = precheck._block_table(np.asarray([18], dtype=np.int64), 25, 10)
    assert late["events_right_censored"] == 1
    assert late["effective_blocks"] == 0


def test_analyze_events_rejects_untruncated_calendar(precheck):
    with pytest.raises(ValueError):
        precheck.analyze_events(["20261029", "20261030"], "20261029")


# ---------------------------------------------------------------- R1-T05-02 输入门


def _gate_inputs(**overrides):
    folds = {
        "frozen_folds_match": True,
        "test_domain_contiguous": True,
        "fold_count": 14,
        "data_state_ids": ["b6d1eb61"],
        "model_versions": [24008 + index for index in range(14)],
        "rows": [{} for _ in range(14)],
    }
    models = {"identity_match": True}
    panel = {"signal_days_complete": True}
    phase = {"phase_offset_matches": True, "runoff_complete": True}
    watermarks = {"fina_watermark_derivable": True}
    calendar = {"truncated_at_cutoff": True, "bounded_last_date": "20261029"}
    events = {"evaluation_end": "20261029"}
    payload = {
        "folds": folds,
        "models": models,
        "panel": panel,
        "phase": phase,
        "watermarks": watermarks,
        "calendar": calendar,
        "events": events,
    }
    for key, patch in overrides.items():
        payload[key].update(patch)
    return payload


def test_input_gate_passes_on_clean_inputs(precheck):
    payload = _gate_inputs()
    gate = precheck.build_input_gate(
        payload["folds"],
        payload["models"],
        payload["panel"],
        payload["phase"],
        payload["watermarks"],
        payload["calendar"],
        payload["events"],
    )
    assert gate["input_gate_pass"] is True
    assert gate["failed_checks"] == []


@pytest.mark.parametrize(
    "section, patch, expected",
    [
        ("models", {"identity_match": False}, "models_identity_match"),
        ("panel", {"signal_days_complete": False}, "signal_days_complete"),
        ("phase", {"phase_offset_matches": False}, "phase_offset_matches"),
        ("phase", {"runoff_complete": False}, "runoff_complete"),
        ("folds", {"frozen_folds_match": False}, "folds_match_frozen_table"),
        ("folds", {"data_state_ids": ["a", "b"]}, "single_data_state"),
        ("watermarks", {"fina_watermark_derivable": False}, "fina_watermark_derivable"),
        ("calendar", {"truncated_at_cutoff": False}, "calendar_truncated_at_cutoff"),
    ],
)
def test_input_gate_fails_closed_per_check(precheck, section, patch, expected):
    payload = _gate_inputs(**{section: patch})
    gate = precheck.build_input_gate(
        payload["folds"],
        payload["models"],
        payload["panel"],
        payload["phase"],
        payload["watermarks"],
        payload["calendar"],
        payload["events"],
    )
    assert gate["input_gate_pass"] is False
    assert expected in gate["failed_checks"]


def test_governance_refuses_when_input_gate_false(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck", input_gate_pass=False)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(precheck_dir)


def test_governance_refuses_when_core_gate_false(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck", core_gate_pass=False)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(precheck_dir)


# ---------------------------------------------------------------- R1-T05-04 附件校验


def _write_canonical(path: Path, payload: dict) -> str:
    import hashlib

    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _gate_checks(**overrides) -> dict:
    checks = {name: True for name in governance_gate_keys()}
    checks.update(overrides)
    return checks


def governance_gate_keys():
    module = Path(__file__).resolve().parents[1] / "scripts" / "v2_p7" / "build_t05_governance.py"
    text = module.read_text(encoding="utf-8")
    start = text.index("REQUIRED_GATE_CHECKS = (")
    stop = text.index(")", start)
    body = text[start:stop]
    return [line.strip().strip('",') for line in body.splitlines()[1:] if line.strip()]


def _make_precheck_dir(
    root: Path, *, input_gate_pass=True, core_gate_pass=True, blocks=None, min_blocks=60
) -> Path:
    """构造与真实预检同构的六件附件 + 汇总（默认全部通过）。"""
    blocks = blocks or {"forecast": 121, "express": 72, "fina_report": 121}
    attachments = {
        "fold_check.json": {
            "frozen_folds_match": True,
            "fold_count": 14,
            "rows": [{} for _ in range(14)],
            "test_domain_contiguous": True,
            "data_state_ids": ["b6d1eb61"],
            "model_versions": [24008 + index for index in range(14)],
        },
        "model_identity.json": {"identity_match": True, "mismatch": []},
        "panel_coverage.json": {"signal_days_complete": True},
        "phase_calendar.json": {
            "phase_offset_matches": True,
            "runoff_complete": True,
            "v10_shifted_matches_v20": True,
            "v10_no_wraparound": True,
            "runoff_trading_days": 20,
            "runoff_required_trading_days": 20,
        },
        "source_watermarks.json": {
            "common_data_cutoff": "20261029",
            "fina_watermark_derivable": True,
        },
        "event_blocks.json": {
            "common_data_cutoff": "20261029",
            "evaluation_end": "20261029",
            "calendar_bounded_by_cutoff": True,
            "cells": {
                name: {
                    "core_10d": {
                        "effective_blocks": count,
                        "gate_pass": count >= min_blocks,
                        "min_effective_blocks": min_blocks,
                    }
                }
                for name, count in blocks.items()
            },
        },
    }
    shas = {name: _write_canonical(root / name, payload) for name, payload in attachments.items()}
    summary = {
        "schema": "p7_t05_precheck/v2",
        "evidence_id": root.parent.name,
        "common_data_cutoff": "20261029",
        "calendar_boundary": {
            "truncated_at_cutoff": True,
            "bounded_last_date": "20261029",
        },
        "folds_match_frozen_table": True,
        "test_domain_contiguous": True,
        "models_identity_match": True,
        "signal_days_complete": True,
        "phase_offset_matches": True,
        "fina_watermark_derivable": True,
        "input_gate": _gate_checks(),
        "input_gate_pass": input_gate_pass,
        "input_gate_failed_checks": [] if input_gate_pass else ["models_identity_match"],
        "core_gate_all_pass": core_gate_pass,
        "min_effective_blocks": min_blocks,
        "evaluation_window": ["20111107", "20261029"],
        "block_rule": {
            "block_days": 20,
            "guard_days": 10,
            "assignment_key": "planned_entry_day=T+1",
            "guard_entries_unassigned": True,
            "window": "entry+horizon",
            "bounded_by": "common_data_cutoff",
        },
        "core_cells": {
            name: {
                "effective_blocks": count,
                "block_years": 14.52,
                "gate_pass": count >= min_blocks,
                "sensitivity_5d_effective_blocks": count,
            }
            for name, count in blocks.items()
        },
        "outputs_sha256": shas,
    }
    _write_canonical(root / "precheck_summary.json", summary)
    return root


def test_precheck_inputs_accept_valid_fixture(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    loaded = governance.load_precheck_inputs(precheck_dir)
    assert loaded["input_gate_pass"] is True


def test_precheck_inputs_reject_missing_attachment(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    (precheck_dir / "fold_check.json").unlink()
    with pytest.raises(FileNotFoundError):
        governance.load_precheck_inputs(precheck_dir)


def test_precheck_inputs_reject_byte_mismatch(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    (precheck_dir / "fold_check.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(precheck_dir)


def test_precheck_inputs_reject_summary_attachment_disagreement(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    summary_path = precheck_dir / "precheck_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["models_identity_match"] = False
    _write_canonical(summary_path, summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(precheck_dir)


def test_precheck_inputs_reject_old_schema(tmp_path, governance):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    summary_path = precheck_dir / "precheck_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema"] = "p7_t05_precheck/v1"
    _write_canonical(summary_path, summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(precheck_dir)


# ------------------------------------------------- R1R-01 失败事实 + 过期总 pass 必须拒绝


def _rewrite(case: Path, name: str, payload: dict, summary: dict) -> None:
    sha = _write_canonical(case / name, payload)
    summary["outputs_sha256"][name] = sha
    _write_canonical(case / "precheck_summary.json", summary)


def test_gate_rejects_failure_fact_with_stale_pass(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    models = json.loads((case / "model_identity.json").read_text(encoding="utf-8"))
    models["identity_match"] = False
    models["mismatch"] = ["SHA 不符 models/x"]
    summary["models_identity_match"] = False
    summary["input_gate"]["models_identity_match"] = False
    _rewrite(case, "model_identity.json", models, summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_string_false(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    summary["input_gate_pass"] = "false"
    _write_canonical(case / "precheck_summary.json", summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_missing_check_key(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    summary["input_gate"].pop("runoff_complete")
    _write_canonical(case / "precheck_summary.json", summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_extra_check_key(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    summary["input_gate"]["legacy_check"] = True
    _write_canonical(case / "precheck_summary.json", summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_core_cell_summary_vs_attachment(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    summary["core_cells"]["forecast"]["effective_blocks"] = 130
    _write_canonical(case / "precheck_summary.json", summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_governance_main_rejects_failure_fixture(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck", input_gate_pass=False)
    out = tmp_path / "out"
    exit_code = governance.main(
        [
            "--root",
            str(out),
            "--precheck-dir",
            str(case),
            "--allow-root",
            str(tmp_path),
        ]
    )
    assert exit_code == governance.EXIT_INPUT_GATE_FAIL
    assert not (out / "prereg").exists()


# ---------------------------------------------------------------- R1-T05-05 写域守卫


def test_write_guard_allows_default_roots(common):
    roots = common.resolve_write_roots()
    target = common.REPO_ROOT / "temp" / "p7_guard_probe" / "x.json"
    assert common.assert_within_write_roots(target, roots) == target.resolve()


@pytest.mark.parametrize(
    "relative",
    ["data/paper/x.json", "data/runs/x.json", "data/raw/x.json", "data/frozen_reference/v2_p65/x"],
)
def test_write_guard_rejects_forbidden_roots(common, relative):
    roots = common.resolve_write_roots()
    with pytest.raises(common.WriteScopeError):
        common.assert_within_write_roots(common.REPO_ROOT / relative, roots)


def test_write_guard_rejects_outside_repo_without_allowance(common, tmp_path):
    roots = common.resolve_write_roots()
    with pytest.raises(common.WriteScopeError):
        common.assert_within_write_roots(tmp_path / "x.json", roots)


def test_write_guard_allows_explicit_temp_root(common, tmp_path):
    roots = common.resolve_write_roots([str(tmp_path)])
    assert common.assert_within_write_roots(tmp_path / "x.json", roots).exists() is False


def test_governance_main_refuses_outside_scope_with_zero_side_effect(governance, tmp_path):
    target = tmp_path / "governance_out"
    exit_code = governance.main(["--root", str(target)])
    assert exit_code == governance.EXIT_WRITE_SCOPE
    assert not target.exists()


def test_precheck_main_refuses_forbidden_out_dir(precheck):
    exit_code = precheck.main(["--out-dir", "data/runs/p7_probe"])
    assert exit_code == precheck.EXIT_WRITE_SCOPE
    assert not (precheck._REPO_ROOT / "data" / "runs" / "p7_probe").exists()


# ---------------------------------------------------------------- R1-T05-06 相位与 runoff


def _days(count: int) -> list[str]:
    return [f"2024{index // 28 + 1:02d}{index % 28 + 1:02d}"[:8] for index in range(count)]


def test_phase_second_tranche_shifts_not_wraps(precheck):
    days = [f"d{index:03d}" for index in range(101)]
    phases = precheck.derive_phase_tranches(days)
    assert phases["v20_t1"][0] == "d000"
    assert phases["v20_t2"][0] == "d010"
    assert phases["v10_t1"][0] == "d010"
    assert phases["v10_t2"][0] == "d020"
    assert phases["shifted_matches"] is True
    assert phases["no_wraparound"] is True
    assert phases["v10_positions"][0] == 10
    assert len(phases["v10_union"]) == len(phases["v20_union"]) - 1


def test_phase_wraparound_would_be_detected(precheck):
    """把第二档写成绕回起点（0,20,…）时，独立平移比对必须判否。"""
    days = [f"d{index:03d}" for index in range(101)]
    phases = precheck.derive_phase_tranches(days)
    wrapped_union = sorted(set(phases["v10_t1"]) | set(phases["v20_t1"]))
    assert wrapped_union != phases["v10_union"]


def test_phase_odd_tail_keeps_shift_identity(precheck):
    days = [f"d{index:03d}" for index in range(97)]
    phases = precheck.derive_phase_tranches(days)
    assert phases["shifted_matches"] is True


def test_runoff_requires_exactly_block_days(precheck):
    days = [f"d{index:03d}" for index in range(30)]
    complete_days, complete = precheck.runoff_window(days, 5)
    assert len(complete_days) == 20
    assert complete is True
    short_days, short = precheck.runoff_window(days, 25)
    assert len(short_days) == 4
    assert short is False


# ---------------------------------------------------------------- R1-T05-03 崩盘门


def _valid_interval() -> dict:
    return _load("crash_spec").build_interval_spec()


def _spec_payload(**overrides):
    payload = {
        "schema": "p7_crash_windows_spec/v2",
        "basis": "market_index",
        "algorithm": {
            "rule": "year_peak_trough",
            "module": "scripts/v2_p7/crash_spec.py",
            "module_sha256": "0" * 64,
        },
        "windows": [
            {
                "start_date": "20150608",
                "end_date": "20150826",
                "depth_pp": -57.1,
                "trading_days": 57,
            },
            {
                "start_date": "20180124",
                "end_date": "20181227",
                "depth_pp": -38.4,
                "trading_days": 226,
            },
        ],
        "tolerance": {
            "tolerance_pp": 0.0,
            "decision_rule": _load("crash_spec").DECISION_RULE,
        },
        "source": {"api": "index_daily", "index_code": "000300.SH", "snapshot_sha256": "1" * 64},
        "interval": _valid_interval(),
        "window_union_trading_days": 283,
    }
    payload.update(overrides)
    return payload


def test_crash_spec_valid_payload(crash_spec):
    validated = crash_spec.validate_spec(_spec_payload(), spec_sha256="a" * 64)
    assert validated["tolerance_pp"] == 0.0
    assert validated["basis"] == "market_index"
    assert len(validated["windows"]) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"basis": "unknown_basis"},
        {"windows": []},
        {"windows": [{"start_date": "20220113", "end_date": "20220426", "depth_pp": -1.0}]},
        {"tolerance": {"tolerance_pp": 1.0, "decision_rule": "95% 区间下界 > 0"}},
        {
            "algorithm": {
                "rule": "year_peak_trough",
                "module": "other.py",
                "module_sha256": "0" * 64,
            }
        },
        {"algorithm": {"rule": "unknown_rule", "module": "m.py", "module_sha256": "0" * 64}},
        {"source": {"api": "index_daily", "index_code": "000300.SH"}},
    ],
)
def test_crash_spec_rejects_invalid_payloads(crash_spec, overrides):
    with pytest.raises(crash_spec.CrashSpecError):
        crash_spec.validate_spec(_spec_payload(**overrides), spec_sha256="a" * 64)


def test_crash_spec_rejects_p5b_rule_on_index_basis(crash_spec):
    series = {"source": {"date_range": ["20050104", "20261110"], "api": "index_daily"}}
    with pytest.raises(crash_spec.CrashSpecError):
        crash_spec.build_spec(series, [], "market_index", "p5b_episode", "20111107")


def test_crash_spec_requires_two_episodes(crash_spec):
    series = {"log_returns": np.array([-0.2, 0.05, 0.05]), "dates": ["d0", "d1", "d2"]}
    with pytest.raises(crash_spec.CrashSpecError):
        crash_spec.derive_windows(series)


def _year_series(drawdowns: dict, days_per_year: int = 40):
    dates = []
    closes = []
    for year, depth in drawdowns.items():
        for index in range(days_per_year):
            dates.append(f"{year}{index + 100:03d}")
            closes.append(100.0 if index < 5 else 100.0 * (1.0 - depth) ** ((index - 4) / 35.0))
    return dates, closes


def test_year_rule_picks_deepest_in_window(crash_spec):
    dates, closes = _year_series({"2013": 0.22, "2015": 0.45, "2018": 0.32, "2022": 0.28})
    windows = crash_spec.derive_year_windows(dates, closes, "20111107", 3)
    assert [item["year"] for item in windows] == ["2015", "2018", "2022"]
    assert windows[0]["depth_pp"] < windows[1]["depth_pp"]


def test_year_rule_requires_eval_window_coverage(crash_spec):
    dates, closes = _year_series({"2013": 0.30})
    with pytest.raises(crash_spec.CrashSpecError):
        crash_spec.derive_year_windows(dates, closes, "20111107", 3)


def test_year_rule_excludes_pre_window_year(crash_spec):
    dates, closes = _year_series({"2008": 0.70, "2015": 0.45, "2018": 0.32})
    windows = crash_spec.derive_year_windows(dates, closes, "20130101", 3)
    assert "2008" not in [item["year"] for item in windows]


def test_prereg_identity_binds_crash_spec(governance):
    base = {
        "spec_path": "specs/crash_windows_v1.json",
        "spec_sha256": "a" * 64,
        "basis": "A0_strategy_chain_nav",
        "algorithm_module": "m.py",
        "algorithm_module_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "windows": [{"start_date": "20190423", "end_date": "20190809"}],
        "tolerance_pp": 0.0,
        "decision_rule": "95% 区间下界 > 0",
    }
    other = dict(base, spec_sha256="d" * 64)
    assert governance._prereg_crash(base) != governance._prereg_crash(other)


# ------------------------------------------------- R1R-04/R1R-05 指数取数与有限性


class _PagedClient:
    def __init__(self, pages, always=None):
        self.pages = pages
        self.always = always
        self.calls = []

    def query(self, api, fields=None, **kwargs):
        self.calls.append(dict(kwargs))
        if self.always is not None:
            return self.always.copy()
        index = len(self.calls) - 1
        if index < len(self.pages):
            return self.pages[index]
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])


def _frame(rows):
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"])


def test_fetch_reads_until_empty_even_if_short_page(index_source):
    client = _PagedClient(
        [
            _frame([["000300.SH", "20240102", 100.0]]),
            _frame([["000300.SH", "20240103", 101.0]]),
            _frame([]),
        ]
    )
    raw = index_source.fetch_index_daily(client, "000300.SH", "20240101", "20240131")
    assert len(raw) == 2
    assert len(client.calls) == 3
    assert [call["offset"] for call in client.calls] == [0, 1, 2]


def test_fetch_rejects_non_advancing_pagination(index_source):
    same = _frame([["000300.SH", "20240102", 100.0], ["000300.SH", "20240103", 101.0]])
    client = _PagedClient([], always=same)
    with pytest.raises(index_source.IndexSourceError):
        index_source.fetch_index_daily(client, "000300.SH", "20240101", "20240131", max_pages=6)


def test_normalize_dedupes_identical_duplicates(index_source):
    raw = _frame(
        [
            ["000300.SH", "20240102", 100.0],
            ["000300.SH", "20240102", 100.0],
            ["000300.SH", "20240103", 101.0],
        ]
    )
    frame = index_source.normalize_index_frame(raw)
    assert len(frame) == 2


def test_normalize_rejects_conflicting_duplicates(index_source):
    raw = _frame(
        [
            ["000300.SH", "20240102", 100.0],
            ["000300.SH", "20240102", 101.0],
        ]
    )
    with pytest.raises(index_source.IndexSourceError):
        index_source.normalize_index_frame(raw)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 0.0, -1.0])
def test_normalize_rejects_non_positive_or_non_finite(index_source, bad):
    raw = _frame([["000300.SH", "20240102", 100.0], ["000300.SH", "20240103", bad]])
    with pytest.raises(index_source.IndexSourceError):
        index_source.normalize_index_frame(raw)


def test_small_window_rejects_value_mismatch(index_source):
    reference = index_source.normalize_index_frame(
        _frame([["000300.SH", "20240102", 100.0], ["000300.SH", "20240103", 101.0]])
    )
    client = _PagedClient(
        [_frame([["000300.SH", "20240102", 99.0], ["000300.SH", "20240103", 101.0]])]
    )
    with pytest.raises(index_source.IndexSourceError):
        index_source.verify_small_window(client, reference, "000300.SH", "20240102", "20240103")


def test_small_window_rejects_missing_rows(index_source):
    reference = index_source.normalize_index_frame(
        _frame([["000300.SH", "20240102", 100.0], ["000300.SH", "20240103", 101.0]])
    )
    client = _PagedClient([_frame([["000300.SH", "20240102", 100.0]])])
    with pytest.raises(index_source.IndexSourceError):
        index_source.verify_small_window(client, reference, "000300.SH", "20240102", "20240103")


# ------------------------------------------------- R1R-03 子目录 junction 逃逸


@pytest.mark.skipif(os.name != "nt", reason="junction 仅 Windows")
def test_write_guard_rejects_subdirectory_junction(common, tmp_path):
    campaign = tmp_path / "data/paper_shadow/p7_testbed/campaign"
    campaign.mkdir(parents=True)
    forbidden = tmp_path / "data/paper"
    forbidden.mkdir(parents=True)
    link = campaign / "governance"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(forbidden)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"无法创建 junction：{created.stderr.strip()[:60]}")
    roots = (tmp_path / "data/paper_shadow/p7_testbed",)
    with pytest.raises(common.WriteScopeError):
        common.assert_targets_within_write_roots([link / "budget_ledger.json"], roots)
    assert not (forbidden / "budget_ledger.json").exists()


# ------------------------------------------------- R2R-02 冻结下限不可被输入降低


def test_gate_rejects_lowered_minimum(tmp_path, governance):
    case = _make_precheck_dir(
        tmp_path / "precheck",
        blocks={"forecast": 1, "express": 1, "fina_report": 1},
        min_blocks=1,
    )
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_minimum_changed_in_summary_only(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    summary["min_effective_blocks"] = 1
    _write_canonical(case / "precheck_summary.json", summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_minimum_changed_in_cell_only(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    events = json.loads((case / "event_blocks.json").read_text(encoding="utf-8"))
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    events["cells"]["forecast"]["core_10d"]["min_effective_blocks"] = 1
    _rewrite(case, "event_blocks.json", events, summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


def test_gate_rejects_inconsistent_cell_gate_pass(tmp_path, governance):
    case = _make_precheck_dir(tmp_path / "precheck")
    events = json.loads((case / "event_blocks.json").read_text(encoding="utf-8"))
    summary = json.loads((case / "precheck_summary.json").read_text(encoding="utf-8"))
    events["cells"]["express"]["core_10d"]["gate_pass"] = False
    _rewrite(case, "event_blocks.json", events, summary)
    with pytest.raises(ValueError):
        governance.load_precheck_inputs(case)


# ------------------------------------------------- R2R-03 统计程序必须齐备


def _mutate_interval(mutator):
    payload = _spec_payload()
    interval = json.loads(json.dumps(payload["interval"]))
    mutator(interval)
    payload["interval"] = interval
    return payload


def _drop_interval(payload):
    payload.pop("interval")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda data: data.pop("e3_paired_delta_mdd"),
        lambda data: data["e3_paired_delta_mdd"].update({"sample_unit": "day"}),
        lambda data: data["e3_paired_delta_mdd"].update({"n_boot": 10}),
        lambda data: data["e3_paired_delta_mdd"].update({"implementation": "scripts/none.py"}),
        lambda data: data["diagnostics"]["daily_block_bootstrap"].update(
            {"decision_role": "decision"}
        ),
        lambda data: data["diagnostics"]["daily_block_bootstrap"].update({"block_length": 0}),
        lambda data: data["correlation"]["definitions"].pop("crash_segment"),
        lambda data: data["correlation"]["definitions"]["full_sample"].update({"block_length": 0}),
        lambda data: data["correlation"]["definitions"]["full_sample"].update(
            {"min_observations": 1}
        ),
        lambda data: data["alignment"].pop("min_union_pairable_days"),
    ],
)
def test_crash_spec_rejects_incomplete_interval(crash_spec, mutator):
    with pytest.raises(crash_spec.CrashSpecError):
        crash_spec.validate_spec(_mutate_interval(mutator), spec_sha256="a" * 64)


def test_crash_spec_rejects_missing_interval(crash_spec):
    payload = _spec_payload()
    _drop_interval(payload)
    with pytest.raises(crash_spec.CrashSpecError):
        crash_spec.validate_spec(payload, spec_sha256="a" * 64)


# ------------------------------------------------- R2R-04 等长页不得误判为重复


def test_fetch_accepts_equal_length_distinct_pages(index_source):
    first = [["000300.SH", f"2024{index:04d}", 100.0 + index] for index in range(1, 5001)]
    second = [["000300.SH", f"2025{index:04d}", 200.0 + index] for index in range(1, 5001)]
    client = _PagedClient([_frame(first), _frame(second), _frame([])])
    raw = index_source.fetch_index_daily(client, "000300.SH", "20240101", "20251231")
    assert len(raw) == 10000
    assert len(client.calls) == 3


def test_fetch_rejects_repeated_page(index_source):
    page = _frame([["000300.SH", f"2024{index:04d}", 100.0] for index in range(1, 5001)])
    client = _PagedClient([page, page, _frame([])])
    with pytest.raises(index_source.IndexSourceError):
        index_source.fetch_index_daily(client, "000300.SH", "20240101", "20251231")


# ------------------------------------------------- R3R-01 最终 prereg 方向一致性


def _crash_from_spec(crash_spec):
    return crash_spec.validate_spec(_spec_payload(), spec_sha256="a" * 64)


def test_prereg_e4_matches_crash_rule(tmp_path, governance, crash_spec):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    precheck = governance.load_precheck_inputs(precheck_dir)
    prereg = governance.build_prereg(precheck, _crash_from_spec(crash_spec))
    governance.assert_prereg_consistency(prereg)
    e4 = {item["id"]: item for item in prereg["gates"]["E4_vetoes"]}
    assert e4["E4-1"]["rule"] == prereg["crash_gate"]["decision_rule"]
    assert "arm − base" in e4["E4-1"]["rule"]
    assert "base − arm" not in e4["E4-1"]["rule"]


def test_prereg_consistency_rejects_reverse_mirror(tmp_path, governance, crash_spec):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    precheck = governance.load_precheck_inputs(precheck_dir)
    prereg = governance.build_prereg(precheck, _crash_from_spec(crash_spec))
    prereg["gates"]["E4_vetoes"][0]["rule"] = "ΔMaxDD = base − arm（改善为正）"
    with pytest.raises(ValueError):
        governance.assert_prereg_consistency(prereg)


def test_prereg_consistency_rejects_tolerance_drift(tmp_path, governance, crash_spec):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    precheck = governance.load_precheck_inputs(precheck_dir)
    prereg = governance.build_prereg(precheck, _crash_from_spec(crash_spec))
    prereg["gates"]["E4_vetoes"][0]["tolerance_pp"] = 1.0
    with pytest.raises(ValueError):
        governance.assert_prereg_consistency(prereg)


def test_prereg_consistency_rejects_day_unit(tmp_path, governance, crash_spec):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    precheck = governance.load_precheck_inputs(precheck_dir)
    prereg = governance.build_prereg(precheck, _crash_from_spec(crash_spec))
    prereg["crash_gate"]["interval"]["e3_paired_delta_mdd"]["sample_unit"] = "day"
    with pytest.raises(ValueError):
        governance.assert_prereg_consistency(prereg)


def test_main_writes_consistent_prereg(tmp_path, governance, crash_spec):
    precheck_dir = _make_precheck_dir(tmp_path / "precheck")
    module_rel = "src/lazybull/v2/hosts/testbed/referee/drawdown/crash_window.py"
    module_sha = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / module_rel).read_bytes()
    ).hexdigest()
    spec_payload = _spec_payload(
        basis="A0_strategy_chain_nav",
        source={"chain_nav_sha256": "1" * 64},
        algorithm={"rule": "p5b_episode", "module": module_rel, "module_sha256": module_sha},
    )
    spec_payload["tolerance"] = {
        "tolerance_pp": 0.0,
        "decision_rule": (
            "ΔMaxDD = arm − base（signed 非正数；改善为正、pp），容差 0；"
            "配对 ΔMaxDD 的 95% 区间下界 > 0 才算改善，否则 E4-1 否决"
        ),
    }
    spec_path = tmp_path / "spec.json"
    _write_canonical(spec_path, spec_payload)
    out = tmp_path / "out"
    exit_code = governance.main(
        [
            "--root",
            str(out),
            "--precheck-dir",
            str(precheck_dir),
            "--crash-spec",
            str(spec_path),
            "--allow-root",
            str(tmp_path),
        ]
    )
    assert exit_code == governance.EXIT_OK
    written = json.loads((out / "prereg" / "b0_composite_prereg_v1.json").read_text("utf-8"))
    governance.assert_prereg_consistency(written)
