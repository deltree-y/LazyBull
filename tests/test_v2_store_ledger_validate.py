# -*- coding: utf-8 -*-
"""v2 台账 schema 校验测试（合成 jsonl，不触碰真实 data/ledger/）。

覆盖：合法 prereg / conclusion / p1_conclusion / research_decision（F9）、
占位 prereg 豁免（F3 §8）、缺字段、坏 kind、
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


def _p1_conclusion_payload(**overrides):
    payload = {
        "title": "P-1 否决门结论",
        "target_hypothesis_id": "B0-composite-20261111",
        "prereg_id": "p7-p1-redesign-20261113",
        "prereg_sha256": "a" * 64,
        "criteria_digest": "b" * 64,
        "disposition": "p1_verdict",
        "verdict": "negative",
        "reason_code": "negative",
        "reason": "最终包络上界 < 0 且负向功效 ≥80%",
        "envelope": {"lo_bp": -30.0, "hi_bp": -5.0, "unit": "bp/event"},
        "power": {"negative_detection": 0.85, "theta_bp": -10.0, "samples": 200},
        "calibration": {"passed": True, "coverage": 0.92, "false_trigger_rate": 0.03},
        "alpha_consumed": False,
        "evidence_refs": ["data/frozen_reference/v2_p7/p7_p1r_20261113_r1/verdict.json"],
        "supersedes": [],
        "notes": "只许否决，不动 α",
    }
    payload.update(overrides)
    return payload


def _research_decision_payload(**overrides):
    payload = {
        "title": "测试资源决定",
        "decision": "stop_new_investment",
        "target_hypothesis_id": "B0-composite-20261111",
        "scope": "仅当前定义，不外推",
        "reason": "证据不足，研究优先级下调",
        "evidence_refs": ["docs/reports/x.md"],
        "reopen_condition": "新信息或实质不同候选，另立 prereg",
        "alpha_consumed": False,
        "notes": "统计状态不变，非经济失败",
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

    def test_p1_conclusion_pass(self, tmp_path):
        path = _write_ledger(
            tmp_path, [_entry("H-P1R-cons-1", "p1_conclusion", _p1_conclusion_payload())]
        )
        report = validate_ledger(path)
        assert report["failed"] == 0, report["entries"][0]["reasons"]

    def test_p1_conclusion_superseded_not_estimable_pass(self, tmp_path):
        """F8：旧条目改记 not_estimable 用追加条目表达，verdict 为 null。"""
        payload = _p1_conclusion_payload(
            disposition="superseded_not_estimable",
            verdict=None,
            reason_code="superseded_not_estimable",
        )
        path = _write_ledger(tmp_path, [_entry("H-P1R-old", "p1_conclusion", payload)])
        report = validate_ledger(path)
        assert report["failed"] == 0, report["entries"][0]["reasons"]

    def test_research_decision_pass(self, tmp_path):
        """F9：资源决定与统计结论分层；停投条目不动 α、不改 verdict。"""
        path = _write_ledger(
            tmp_path,
            [_entry("H-P7-stop-1", "research_decision", _research_decision_payload())],
        )
        report = validate_ledger(path)
        assert report["failed"] == 0, report["entries"][0]["reasons"]

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

    def test_p1_conclusion_alpha_consumed_fails(self, tmp_path):
        payload = _p1_conclusion_payload(alpha_consumed=True)
        report = self._report(tmp_path, [_entry("H-P1R-cons-1", "p1_conclusion", payload)])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("alpha_consumed" in r for r in entry["reasons"])

    def test_p1_conclusion_bad_verdict_fails(self, tmp_path):
        payload = _p1_conclusion_payload(verdict="pass")
        report = self._report(tmp_path, [_entry("H-P1R-cons-1", "p1_conclusion", payload)])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("verdict 非法" in r for r in entry["reasons"])

    def test_p1_conclusion_superseded_with_verdict_fails(self, tmp_path):
        payload = _p1_conclusion_payload(
            disposition="superseded_not_estimable",
            reason_code="superseded_not_estimable",
        )
        report = self._report(tmp_path, [_entry("H-P1R-old", "p1_conclusion", payload)])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("verdict 必须为 null" in r for r in entry["reasons"])

    def test_p1_conclusion_empty_evidence_fails(self, tmp_path):
        payload = _p1_conclusion_payload(evidence_refs=[])
        report = self._report(tmp_path, [_entry("H-P1R-cons-1", "p1_conclusion", payload)])
        assert _by_line(report, 1)["status"] == "fail"

    def test_research_decision_bad_decision_fails(self, tmp_path):
        payload = _research_decision_payload(decision="pause_thinking")
        report = self._report(tmp_path, [_entry("H-P7-stop-1", "research_decision", payload)])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("decision 非法" in r for r in entry["reasons"])

    def test_research_decision_alpha_consumed_fails(self, tmp_path):
        payload = _research_decision_payload(alpha_consumed=True)
        report = self._report(tmp_path, [_entry("H-P7-stop-1", "research_decision", payload)])
        entry = _by_line(report, 1)
        assert entry["status"] == "fail"
        assert any("alpha_consumed" in r for r in entry["reasons"])

    def test_research_decision_empty_evidence_fails(self, tmp_path):
        payload = _research_decision_payload(evidence_refs=[])
        report = self._report(tmp_path, [_entry("H-P7-stop-1", "research_decision", payload)])
        assert _by_line(report, 1)["status"] == "fail"

    def test_research_decision_empty_reopen_condition_fails(self, tmp_path):
        payload = _research_decision_payload(reopen_condition="")
        report = self._report(tmp_path, [_entry("H-P7-stop-1", "research_decision", payload)])
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
