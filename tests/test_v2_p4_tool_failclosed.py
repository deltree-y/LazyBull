# -*- coding: utf-8 -*-
"""验收/迁移工具的 fail-closed 出口锁定（reviewer3 R2-P1-1/2/3/4）。

这些用例冻结"工具不得默认成功"的纪律：
- 补录 no-op 前必须证明双份一致（split-brain 不得假成功）；
- 影子对照的无效日/未归因阻断/无 rate error 必须使出口 FAIL；
- 种子对损坏历史默认拒绝覆盖。
"""

import json
from pathlib import Path

import pytest

import scripts.v2_p4.backfill_feature_missing_rate as backfill
from scripts.v2_p4 import seed_coverage_history as seed
from scripts.v2_p4.t5_coverage_shadow_compare import DayRow, _summarize


def _row(date: str, **kwargs) -> DayRow:
    base = dict(
        date=date,
        ok=True,
        blocked=False,
        errors=0,
        warnings=0,
        rates_domain={},
        rates_all={},
        error_columns=[],
        gaps=[],
        c_blocked=False,
        c_reasons=[],
    )
    base.update(kwargs)
    return DayRow(**base)


def _summary(rows, expected):
    return _summarize(
        rows,
        {"version": 24022, "model_type": "xgb"},
        "20260101",
        "20260102",
        expected_days=expected,
        fixed_evidence={},
        identity={},
    )


class TestShadowExit:
    def test_invalid_day_forces_failure(self):
        rows = [_row("20260101"), _row("20260102", error="RuntimeError: boom")]
        summary = _summary(rows, expected=2)
        assert summary["n_errors"] == 1
        assert summary["b1_exit_ok"] is False

    def test_fewer_days_than_expected_forces_failure(self):
        summary = _summary([_row("20260101")], expected=2)
        assert summary["b1_exit_ok"] is False

    def test_uncovered_error_counts_in_frequency(self):
        row = _row(
            "20260101",
            error_columns=[{"column": "ghost", "rate": None, "code": "uncovered"}],
            errors=1,
        )
        summary = _summary([row], expected=1)
        freq = {item["column"]: item for item in summary["freq"]}
        assert freq["ghost"]["error_days"] == 1
        assert freq["ghost"]["over_20pct"] is True
        assert summary["b1_2_no_column_over_20pct"] is False

    def test_blocked_without_fixed_evidence_fails_b1(self):
        row = _row(
            "20260101",
            blocked=True,
            errors=20,
            rates_all={"x": 0.7},
            error_columns=[{"column": "x", "rate": 0.7, "code": "rate_exceeded"}],
        )
        summary = _summary([row], expected=1)
        assert summary["b1_1_blocked_zero_or_attributed"] is False
        assert summary["b1_exit_ok"] is False

    def test_blocked_with_real_column_and_evidence_passes(self):
        row = _row(
            "20260101",
            blocked=True,
            errors=20,
            rates_all={"x": 0.7},
            error_columns=[{"column": "x", "rate": 0.7, "code": "rate_exceeded"}],
        )
        summary = _summarize(
            [row] + [_row(f"2026010{i}") for i in range(2, 7)],
            {"version": 24022, "model_type": "xgb"},
            "20260101",
            "20260101",
            expected_days=6,
            fixed_evidence={"20260101": "data/fix/x.json"},
            identity={},
        )
        assert summary["b1_1_blocked_zero_or_attributed"] is True
        assert summary["b1_exit_ok"] is True


def _write_split_brain_models(tmp_path: Path) -> tuple[Path, int]:
    models = tmp_path / "models"
    models.mkdir()
    params = {
        "feature_missing_rate_schema_version": 2,
        "feature_missing_rate": {
            "a": {
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "n_days": 100,
                "p99_unstable": True,
            }
        },
        "feature_missing_rate_stat": {"coverage_n_samples": 1},
    }
    (models / "v7_metadata.json").write_text(json.dumps({"train_params": params}), encoding="utf-8")
    params_b = json.loads(json.dumps(params))
    params_b["feature_missing_rate_stat"]["coverage_n_samples"] = 2
    (models / "model_registry.json").write_text(
        json.dumps({"models": [{"version": 7, "train_params": params_b}]}), encoding="utf-8"
    )
    return models, 7


class TestBackfillFailClosed:
    def test_noop_on_split_brain_raises(self, tmp_path):
        models, version = _write_split_brain_models(tmp_path)
        spec = backfill.BackfillSpec(
            version=version,
            feature_columns=("a",),
            label_column="y_ret_20",
            train_start_date="20200101",
            train_end_date="20260101",
            n_samples_registered=1,
        )
        with pytest.raises(RuntimeError, match="双份不一致"):
            backfill.write_backfill(models, spec, {}, {"feature_missing_rate_sha256": "x"})

    def test_verify_requires_keys_present(self, tmp_path):
        models, version = _write_split_brain_models(tmp_path)
        registry = json.loads((models / "model_registry.json").read_text(encoding="utf-8"))
        registry["models"][0]["train_params"].pop("feature_missing_rate_stat")
        (models / "model_registry.json").write_text(json.dumps(registry), encoding="utf-8")
        with pytest.raises(RuntimeError, match="registry 缺"):
            backfill.verify_backfill(models, [version])


class TestSeedFailClosed:
    def _shadow(self, tmp_path: Path) -> Path:
        path = tmp_path / "shadow.json"
        path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "date": f"202601{i:02d}",
                            "rates_domain": {"a": 0.1},
                            "blocked": i == 2,
                        }
                        for i in range(1, 22)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_corrupt_existing_history_refused(self, tmp_path):
        shadow = self._shadow(tmp_path)
        paper = tmp_path / "paper"
        (paper / "runs").mkdir(parents=True)
        (paper / "runs" / "coverage_history.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit, match="无法严格解析"):
            seed.main(["--shadow-json", str(shadow), "--paper-root", str(paper), "--write"])

    def test_blocked_day_not_seeded(self, tmp_path):
        shadow = self._shadow(tmp_path)
        paper = tmp_path / "paper"
        seed.main(["--shadow-json", str(shadow), "--paper-root", str(paper), "--write"])
        payload = json.loads((paper / "runs" / "coverage_history.json").read_text(encoding="utf-8"))
        assert "20260102" not in payload["days"]
        assert len(payload["days"]) == 20
        assert payload["days"]["20260101"]["a"] == 0.1
