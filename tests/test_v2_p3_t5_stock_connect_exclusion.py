# -*- coding: utf-8 -*-
"""P3 T5 北向闭市日窄口径排除测试（规划 §19.18，2026-10-24 用户裁决）。

覆盖：raw/源侧双条件判定、默认不排除反例、指纹结果登记、折级覆盖门、
双跑排除集合一致性、臂级视图拍平。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.v2_p3.t5_arm_artifacts import build_batch_fingerprint_view
from scripts.v2_p3.t5_arm_executors import _assert_fold_coverage
from scripts.v2_p3.t05b_evidence import _finalize_comparison
from scripts.v2_p3.t05b_fingerprint import (
    STOCK_CONNECT_CLOSED_REASON,
    _build_fingerprint_result,
    resolve_stock_connect_closure_days,
)

CLOSURE_DAY = "20211224"


def _write_raw(root: Path, day: str, *, hgt=None, sgt=None, north=None, south=1.0) -> None:
    path = root / "raw" / "moneyflow_hsgt" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": [day],
            "hgt": [hgt],
            "sgt": [sgt],
            "north_money": [north],
            "south_money": [south],
        }
    ).to_parquet(path, index=False)


def _write_cs_train(root: Path, day: str, north_values) -> None:
    path = root / "features" / "cs_train" / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": [day] * len(north_values),
            "ts_code": [f"00000{i}.SZ" for i in range(len(north_values))],
            "north_net_buy": north_values,
        }
    ).to_parquet(path, index=False)


class TestResolveClosureDays:
    def test_closure_detected_when_raw_and_feature_all_null(self, tmp_path):
        _write_raw(tmp_path, CLOSURE_DAY)
        _write_cs_train(tmp_path, CLOSURE_DAY, [None, None])
        result = resolve_stock_connect_closure_days(
            tmp_path, [CLOSURE_DAY], ["north_net_buy", "other"]
        )
        assert result == {CLOSURE_DAY: STOCK_CONNECT_CLOSED_REASON}

    def test_model_without_north_column_never_excluded(self, tmp_path):
        _write_raw(tmp_path, CLOSURE_DAY)
        _write_cs_train(tmp_path, CLOSURE_DAY, [None, None])
        assert resolve_stock_connect_closure_days(tmp_path, [CLOSURE_DAY], ["other"]) == {}

    def test_raw_has_values_not_excluded(self, tmp_path):
        _write_raw(tmp_path, CLOSURE_DAY, hgt=1.0, sgt=2.0, north=3.0)
        _write_cs_train(tmp_path, CLOSURE_DAY, [None, None])
        assert resolve_stock_connect_closure_days(tmp_path, [CLOSURE_DAY], ["north_net_buy"]) == {}

    def test_missing_raw_partition_not_excluded(self, tmp_path):
        _write_cs_train(tmp_path, CLOSURE_DAY, [None, None])
        assert resolve_stock_connect_closure_days(tmp_path, [CLOSURE_DAY], ["north_net_buy"]) == {}

    def test_feature_not_all_null_not_excluded(self, tmp_path):
        _write_raw(tmp_path, CLOSURE_DAY)
        _write_cs_train(tmp_path, CLOSURE_DAY, [None, 0.5])
        assert resolve_stock_connect_closure_days(tmp_path, [CLOSURE_DAY], ["north_net_buy"]) == {}

    def test_panel_frame_all_null_excluded(self, tmp_path):
        _write_raw(tmp_path, CLOSURE_DAY)
        panel_frames = {
            CLOSURE_DAY: pd.DataFrame(
                {"trade_date": [CLOSURE_DAY], "ts_code": ["000001.SZ"], "north_net_buy": [None]}
            )
        }
        result = resolve_stock_connect_closure_days(
            tmp_path, [CLOSURE_DAY], ["north_net_buy"], panel_frames
        )
        assert result == {CLOSURE_DAY: STOCK_CONNECT_CLOSED_REASON}

    def test_panel_frame_not_all_null_not_excluded(self, tmp_path):
        _write_raw(tmp_path, CLOSURE_DAY)
        panel_frames = {
            CLOSURE_DAY: pd.DataFrame(
                {"trade_date": [CLOSURE_DAY], "ts_code": ["000001.SZ"], "north_net_buy": [1.0]}
            )
        }
        assert (
            resolve_stock_connect_closure_days(
                tmp_path, [CLOSURE_DAY], ["north_net_buy"], panel_frames
            )
            == {}
        )


class TestFingerprintResultRegistration:
    def _signal(self):
        return SimpleNamespace(metadata={}, model=SimpleNamespace(models=[]))

    def test_excluded_days_registered(self):
        result = _build_fingerprint_result(
            self._signal(),
            model_version=1,
            used_days=["20211223"],
            ensemble_rows=[("20211223", "000001.SZ", 0.1)],
            seed_rows=[],
            feature_columns=["north_net_buy"],
            producer_identity=None,
            source_mode="cs_train_verify",
            excluded_days={CLOSURE_DAY: STOCK_CONNECT_CLOSED_REASON},
        )
        assert result["excluded_days"] == [CLOSURE_DAY]
        assert result["excluded_day_reasons"] == {CLOSURE_DAY: STOCK_CONNECT_CLOSED_REASON}

    def test_default_path_has_no_exclusion_keys(self):
        result = _build_fingerprint_result(
            self._signal(),
            model_version=1,
            used_days=["20211223"],
            ensemble_rows=[("20211223", "000001.SZ", 0.1)],
            seed_rows=[],
            feature_columns=["north_net_buy"],
            producer_identity=None,
            source_mode="cs_train_verify",
        )
        assert "excluded_days" not in result
        assert "excluded_day_reasons" not in result


class TestFoldCoverageGate:
    def _fingerprint(self, used, excluded, n_rows=1, fp="a" * 64):
        out = {"score_days": list(used), "n_rows": n_rows, "fingerprint": fp}
        if excluded:
            out["excluded_days"] = list(excluded)
        return out

    def test_used_plus_excluded_equals_expected(self):
        _assert_fold_coverage(
            self._fingerprint(["20211223"], [CLOSURE_DAY]),
            ["20211223", CLOSURE_DAY],
            0,
        )

    def test_missing_day_still_fails(self):
        with pytest.raises(RuntimeError, match="闭市排除日与独立预期日不一致"):
            _assert_fold_coverage(
                self._fingerprint(["20211223"], [CLOSURE_DAY]),
                ["20211223", CLOSURE_DAY, "20211227"],
                0,
            )

    def test_overlap_rejected(self):
        with pytest.raises(RuntimeError, match="重叠"):
            _assert_fold_coverage(
                self._fingerprint([CLOSURE_DAY], [CLOSURE_DAY]),
                [CLOSURE_DAY],
                0,
            )


class TestDoubleRunExcludedMatch:
    def _run(self, excluded):
        return {
            "fingerprint": {
                "fingerprint": "a" * 64,
                "per_seed_diagnostic": [],
                "score_days": ["20211223"],
                "n_rows": 1,
                "excluded_days": list(excluded),
            }
        }

    def test_equal_exclusion_sets_pass_coverage(self):
        comparison = _finalize_comparison(
            [self._run([CLOSURE_DAY]), self._run([CLOSURE_DAY])], [], []
        )
        assert comparison["excluded_days_match"] is True
        assert comparison["coverage_match"] is True
        assert comparison["verdict"] == "PASS"

    def test_asymmetric_exclusion_sets_fail_coverage(self):
        comparison = _finalize_comparison([self._run([CLOSURE_DAY]), self._run([])], [], [])
        assert comparison["excluded_days_match"] is False
        assert comparison["coverage_match"] is False
        assert comparison["verdict"] == "FAIL"

    def test_default_absent_exclusion_keys_pass(self):
        run = self._run([])
        run["fingerprint"].pop("excluded_days")
        comparison = _finalize_comparison([run, self._run([])], [], [])
        assert comparison["excluded_days_match"] is True
        assert comparison["coverage_match"] is True


class TestBatchViewExcludedDays:
    def test_view_flattens_excluded_days(self, tmp_path):
        fold = tmp_path / "folds" / "split00"
        fold.mkdir(parents=True)
        (fold / "fingerprint.json").write_text(
            json.dumps(
                {
                    "fingerprint": "f" * 64,
                    "per_seed_diagnostic": [],
                    "score_days": ["20211223"],
                    "excluded_days": [CLOSURE_DAY],
                    "n_rows": 1,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "batch_fingerprint.json").write_text(
            json.dumps({"folds": {"split00": "f" * 64}, "batch_fingerprint": "b" * 64}),
            encoding="utf-8",
        )
        view = build_batch_fingerprint_view(tmp_path)
        assert view["excluded_days"] == [f"split00:{CLOSURE_DAY}"]
        assert view["score_days"] == ["split00:20211223"]
