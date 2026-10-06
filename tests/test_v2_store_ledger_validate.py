# -*- coding: utf-8 -*-
"""v2 台账 schema 校验测试（合成 jsonl，不触碰真实 data/ledger/）。

覆盖：合法 prereg / conclusion、占位 prereg 豁免（F3 §8）、缺字段、坏 kind、
坏 schema_version、坏 verdict、空证据引用、data_state 缺键、坏 registered_at、
JSON 解析失败、重复 hypothesis_id，以及 CLI 退出码与报告产物。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.lazybull.v2.store.ledger_validate import validate_ledger

_DATA_STATE = {
    "raw_partitions": {"daily": "20260731"},
    "features_partition_max": "20260702",
    "config_digest": "cfg",
    "code_digest": "abc1234",
}


def _prereg_payload(**overrides):
    payload = {
        "title": "测试假设",
        "vehicle": "factor_family",
        "tags": ["t"],
        "statement": "可证伪陈述",
        "arms": [{"arm": "A1"}],
        "metrics": {"primary": "配对 ΔCAGR", "guards": []},
        "decision_rule": "裁决规则",
        "noise_band": {"口径": "x"},
        "candidate_pool_denominator": 8,
        "pool_description": "8 选 1",
        "alpha_budget": 0.025,
        "stop_rule": "止损条件",
        "reopen_condition": "重开条件",
    }
    payload.update(overrides)
    return payload


def _conclusion_payload(**overrides):
    payload = {
        "title": "测试结论",
        "hypothesis_id": "H-test-prereg",
        "verdict": "pass",
        "result_metrics": {"delta": 0.01},
        "evidence_refs": ["data/reports/x.json"],
        "supersedes": [],
        "notes": "已知代价",
    }
    payload.update(overrides)
    return payload


def _entry(hypothesis_id, kind, payload, **overrides):
    entry = {
        "schema_version": 1,
        "hypothesis_id": hypothesis_id,
        "kind": kind,
        "payload": payload,
        "data_state": dict(_DATA_STATE),
        "registered_at": "2026-10-03T10:00:00",
    }
    entry.update(overrides)
    return entry


def _write_ledger(tmp_path, lines) -> Path:
    path = tmp_path / "hypotheses.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if isinstance(line, str) else json.dumps(line, ensure_ascii=False))
            f.write("\n")
    return path


def _by_line(report, line_no):
    return next(e for e in report["entries"] if e["line"] == line_no)


class TestValidEntries:
    def test_prereg_pass(self, tmp_path):
        path = _write_ledger(tmp_path, [_entry("H-1", "prereg", _prereg_payload())])
        report = validate_ledger(path)
        assert report["failed"] == 0 and report["passed"] == 1

    def test_conclusion_pass(self, tmp_path):
        path = _write_ledger(tmp_path, [_entry("H-2", "conclusion", _conclusion_payload())])
        report = validate_ledger(path)
        assert report["failed"] == 0

    def test_placeholder_prereg_exemption(self, tmp_path):
        """F3 §8：占位 prereg 允许 arms/metrics/noise_band/stop_rule 空值 + alpha_budget null。"""
        payload = _prereg_payload(
            retrospective_placeholder=True,
            arms=[],
            metrics={},
            noise_band=None,
            stop_rule="",
            alpha_budget=None,
        )
        path = _write_ledger(tmp_path, [_entry("H-3", "prereg", payload)])
        report = validate_ledger(path)
        assert report["failed"] == 0, report["entries"][0]["reasons"]


class TestInvalidEntries:
    def _report(self, tmp_path, lines):
        return validate_ledger(_write_ledger(tmp_path, lines))

    def test_missing_payload_fields(self, tmp_path):
        payload = _prereg_payload()
        del payload["arms"], payload["statement"]
        report = self._report(tmp_path, [_entry("H-1", "prereg", payload)])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("缺字段" in r for r in entry["reasons"])

    def test_non_placeholder_empty_arms_fails(self, tmp_path):
        report = self._report(tmp_path, [_entry("H-1", "prereg", _prereg_payload(arms=[]))])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("arms" in r for r in entry["reasons"])

    def test_non_placeholder_null_alpha_budget_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(alpha_budget=None))]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_metrics_without_primary_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(metrics={"guards": []}))]
        )
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("primary" in r for r in entry["reasons"])

    def test_bad_vehicle_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(vehicle="rocket"))]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_bad_pool_denominator_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(candidate_pool_denominator=0))]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_bad_kind_fails(self, tmp_path):
        report = self._report(tmp_path, [_entry("H-1", "draft", _prereg_payload())])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("kind 非法" in r for r in entry["reasons"])

    def test_bad_schema_version_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(), schema_version=2)]
        )
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("schema_version" in r for r in entry["reasons"])

    def test_empty_hypothesis_id_fails(self, tmp_path):
        report = self._report(tmp_path, [_entry("  ", "prereg", _prereg_payload())])
        assert _by_line(report, 1)["status"] == "fail"

    def test_conclusion_bad_verdict_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-2", "conclusion", _conclusion_payload(verdict="maybe"))]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_conclusion_empty_evidence_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-2", "conclusion", _conclusion_payload(evidence_refs=[]))]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_conclusion_missing_link_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-2", "conclusion", _conclusion_payload(hypothesis_id=""))]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_data_state_missing_key_fails(self, tmp_path):
        bad_state = {k: v for k, v in _DATA_STATE.items() if k != "code_digest"}
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(), data_state=bad_state)]
        )
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("data_state" in r for r in entry["reasons"])

    def test_bad_registered_at_fails(self, tmp_path):
        report = self._report(
            tmp_path, [_entry("H-1", "prereg", _prereg_payload(), registered_at="不是日期")]
        )
        assert _by_line(report, 1)["status"] == "fail"

    def test_json_parse_error_fails(self, tmp_path):
        report = self._report(tmp_path, ["{broken json"])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("JSON 解析失败" in r for r in entry["reasons"])

    def test_duplicate_id_fails(self, tmp_path):
        report = self._report(
            tmp_path,
            [
                _entry("H-1", "prereg", _prereg_payload()),
                _entry("H-1", "conclusion", _conclusion_payload()),
            ],
        )
        assert _by_line(report, 1)["status"] == "pass"
        entry = _by_line(report, 2)
        assert entry["status"] == "fail"
        assert any("重复" in r for r in entry["reasons"])

    def test_summary_counts(self, tmp_path):
        report = self._report(
            tmp_path,
            [
                _entry("H-1", "prereg", _prereg_payload()),
                _entry("H-2", "draft", _prereg_payload()),
                "{broken",
            ],
        )
        assert report["total"] == 3
        assert report["passed"] == 1
        assert report["failed"] == 2


class TestCli:
    def _run_cli(self, ledger: Path, out: Path) -> subprocess.CompletedProcess:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "v2_p1"
            / "backfill_ledger_validate.py"
        )
        return subprocess.run(
            [sys.executable, str(script), "--ledger", str(ledger), "--out", str(out)],
            capture_output=True,
            text=True,
        )

    def test_cli_all_pass_exit_0(self, tmp_path):
        ledger = _write_ledger(tmp_path, [_entry("H-1", "prereg", _prereg_payload())])
        out = tmp_path / "report.json"
        proc = self._run_cli(ledger, out)
        assert proc.returncode == 0, proc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["failed"] == 0

    def test_cli_fail_exit_1(self, tmp_path):
        ledger = _write_ledger(tmp_path, [_entry("H-1", "draft", _prereg_payload())])
        out = tmp_path / "report.json"
        proc = self._run_cli(ledger, out)
        assert proc.returncode == 1
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["failed"] == 1

    def test_cli_type_error_entries_exit_1_no_crash(self, tmp_path):
        """R3-09：类型错误条目（kind/vehicle/verdict 非 str）逐条 fail，不中断整批。"""
        ledger = _write_ledger(
            tmp_path,
            [
                _entry("H-bad", [], _prereg_payload()),  # kind=list
                _entry("H-ok", "prereg", _prereg_payload()),
            ],
        )
        out = tmp_path / "report.json"
        proc = self._run_cli(ledger, out)
        assert proc.returncode == 1
        assert "TypeError" not in proc.stderr and "Traceback" not in proc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["total"] == 2 and report["failed"] == 1 and report["passed"] == 1


class TestTypeGuards:
    """R3-09：合法 JSON 但类型错误的 kind/vehicle/verdict 记为该条目失败原因，不抛 TypeError。"""

    def test_non_str_kind_fails_batch_continues(self, tmp_path):
        report = validate_ledger(
            _write_ledger(
                tmp_path,
                [
                    _entry("H-bad", [], _prereg_payload()),
                    _entry("H-ok", "prereg", _prereg_payload()),
                ],
            )
        )
        bad = _by_line(report, 1)
        assert bad["status"] == "fail"
        assert any("kind 非法" in r for r in bad["reasons"])
        assert _by_line(report, 2)["status"] == "pass"

    def test_non_str_vehicle_fails(self, tmp_path):
        report = validate_ledger(
            _write_ledger(tmp_path, [_entry("H-1", "prereg", _prereg_payload(vehicle={}))])
        )
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("vehicle 非法" in r for r in entry["reasons"])

    def test_non_str_verdict_fails(self, tmp_path):
        report = validate_ledger(
            _write_ledger(tmp_path, [_entry("H-2", "conclusion", _conclusion_payload(verdict=[]))])
        )
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("verdict 非法" in r for r in entry["reasons"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
