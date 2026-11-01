# -*- coding: utf-8 -*-
"""门 4（B′）分级判据矩阵锁定测试。

用户裁决（2026-10-31）：
- 基线 = 该列训练期**逐日缺失率分布**的 p95/p99；去掉 τ 容差；
- 分级：≤p95 pass / (p95, p99] warning / >p99 error / 100% error；
- **三档均不阻断**；唯一阻断 = 聚合升级（error 列数 ≥ max(8, ceil(10%×列数))）；
- 分母两侧同域（主板域；域缺失 ⇒ error 且不判定）；
- 滚动分位（近 60 日、最少 20 日）在线自校准，不足回落静态分布。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.common.model_schema import (
    FEATURE_MISSING_RATE_SCHEMA_VERSION,
    read_drift_exempt,
    read_feature_missing_rate,
    summarize_daily_missing_rates,
)
from src.lazybull.v2.hosts.paper.coverage_gate import (
    AGG_MIN_ERROR_COLUMNS,
    COVERAGE_ERR_ALL_NAN,
    COVERAGE_ERR_DOMAIN_EMPTY,
    COVERAGE_ERR_DOMAIN_MISSING,
    COVERAGE_ERR_EXEMPT_UNKNOWN,
    COVERAGE_ERR_KEY_MISSING,
    COVERAGE_ERR_RATE_EXCEEDED,
    COVERAGE_ERR_ROW_DOMAIN,
    COVERAGE_ERR_UNCOVERED,
    COVERAGE_RATE_FLOOR,
    ROLLING_MIN_DAYS,
    CoverageMember,
    classify_severity,
    evaluate_coverage,
)

_DOMAIN = frozenset({"600000.SH", "600001.SH"})


def _dist(p95: float, p99: float | None = None) -> dict:
    p99 = p95 if p99 is None else p99
    return {
        "mean": p95,
        "p50": p95,
        "p90": p95,
        "p95": p95,
        "p99": p99,
        "max": max(p99, p95),
        "n_days": 1454,
        "p99_unstable": False,
    }


def _member(columns, distribution: dict, member_id: str = "m1") -> CoverageMember:
    return CoverageMember(
        member_id=member_id,
        columns=list(columns),
        train_params={
            "feature_missing_rate_schema_version": FEATURE_MISSING_RATE_SCHEMA_VERSION,
            "feature_missing_rate": distribution,
            "feature_missing_rate_stat": {"row_domain": "train_filtered_main_board"},
        },
    )


def _frame(columns_rates: dict, *, n: int = 100, extra_rows: int = 0) -> pd.DataFrame:
    """构造当日帧：每列按给定缺失率布 NaN（域内 n 行 + 域外 extra_rows 行全 NaN）。"""
    data = {}
    codes = [f"60000{i}.SH" for i in range(min(n, 2))]
    codes = (codes * ((n // 2) + 1))[:n]
    for col, rate in columns_rates.items():
        nan_count = int(round(rate * n))
        values = np.array([0.01] * n, dtype=float)
        values[:nan_count] = np.nan
        data[col] = values
    if extra_rows:
        pad = {col: [np.nan] * extra_rows for col in columns_rates}
        for col, values in pad.items():
            data[col] = [*data[col], *values]
        codes = [*codes, *[f"30000{i}.SZ" for i in range(extra_rows)]]
    data["ts_code"] = codes
    return pd.DataFrame(data)


class TestSeverityGrades:
    def test_at_or_below_p95_is_pass(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.10}),
            [(_member(["bp"], {"bp": _dist(0.10)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.verdicts[0]["severity"] == "pass"
        assert outcome.blocked is False and outcome.gaps == []

    def test_between_p95_and_p99_is_warning_not_block(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.15}),
            [(_member(["bp"], {"bp": _dist(0.10, 0.20)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        verdict = outcome.verdicts[0]
        assert verdict["severity"] == "warning"
        assert verdict["error_code"] == COVERAGE_ERR_RATE_EXCEEDED
        assert outcome.blocked is False and outcome.gaps == []
        assert any(a["severity"] == "warning" and a["column"] == "bp" for a in outcome.alerts)

    def test_above_p99_is_error_but_not_block(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.30}),
            [(_member(["bp"], {"bp": _dist(0.10, 0.20)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.verdicts[0]["severity"] == "error"
        assert outcome.blocked is False and outcome.gaps == []
        assert outcome.error_columns == 1

    def test_all_nan_today_is_error_not_block(self):
        outcome = evaluate_coverage(
            _frame({"bp": 1.0}),
            [(_member(["bp"], {"bp": _dist(0.95)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.verdicts[0]["error_code"] == COVERAGE_ERR_ALL_NAN
        assert outcome.blocked is False

    def test_p95_p99_thresholds_recorded_with_source(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.15}),
            [(_member(["bp"], {"bp": _dist(0.10, 0.20)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        verdict = outcome.verdicts[0]
        assert verdict["p95"] == pytest.approx(0.10)
        assert verdict["p99"] == pytest.approx(0.20)
        assert verdict["baseline_source"] == "static"
        assert verdict["baseline_days"] == 1454


class TestAggregateEscalation:
    def test_many_error_columns_block(self):
        columns = [f"c{i:02d}" for i in range(AGG_MIN_ERROR_COLUMNS)]
        frame = _frame({col: 0.50 for col in columns}, n=1200)
        members = [(_member(columns, {col: _dist(0.05) for col in columns}), columns)]
        outcome = evaluate_coverage(frame, members, domain_codes=_DOMAIN)
        assert outcome.error_columns >= AGG_MIN_ERROR_COLUMNS
        assert outcome.blocked is True
        assert outcome.gaps and "聚合升级" in outcome.gaps[0]

    def test_threshold_scales_with_column_count(self):
        # 100 列时阈值 = max(15, ceil(20)) = 20；9 列 error ⇒ 不阻断
        columns = [f"c{i:03d}" for i in range(100)]
        rates = {col: (0.50 if i < 9 else 0.0) for i, col in enumerate(columns)}
        frame = _frame(rates, n=1200)
        members = [(_member(columns, {col: _dist(0.05) for col in columns}), columns)]
        outcome = evaluate_coverage(frame, members, domain_codes=_DOMAIN)
        assert outcome.error_columns == 9
        assert outcome.error_threshold == 20
        assert outcome.blocked is False

    def test_min_floor_dominates_for_small_column_sets(self):
        # 20 列时阈值 = max(15, ceil(4)) = 15；15 列 error ⇒ 阻断
        columns = [f"d{i:02d}" for i in range(20)]
        rates = {col: (0.50 if i < 15 else 0.0) for i, col in enumerate(columns)}
        frame = _frame(rates, n=1200)
        members = [(_member(columns, {col: _dist(0.05) for col in columns}), columns)]
        outcome = evaluate_coverage(frame, members, domain_codes=_DOMAIN)
        assert outcome.error_threshold == 15
        assert outcome.blocked is True


class TestMetadataErrorsAreAlerts:
    def test_missing_metadata_is_error_alert_without_verdicts(self):
        frame = _frame({"bp": 0.4})
        member = CoverageMember(member_id="legacy", columns=["bp"], train_params={})
        outcome = evaluate_coverage(frame, [(member, ["bp"])], domain_codes=_DOMAIN)
        assert outcome.verdicts == []
        assert outcome.blocked is False
        assert any(
            a["code"] == "metadata_missing" and a["severity"] == "error" for a in outcome.alerts
        )

    def test_v1_schema_is_unsupported_alert(self):
        frame = _frame({"bp": 0.4})
        member = CoverageMember(
            member_id="v1",
            columns=["bp"],
            train_params={
                "feature_missing_rate_schema_version": 1,
                "feature_missing_rate": {"bp": 0.1},
            },
        )
        outcome = evaluate_coverage(frame, [(member, ["bp"])], domain_codes=_DOMAIN)
        assert any(a["code"] == "metadata_schema_unsupported" for a in outcome.alerts)
        assert outcome.blocked is False

    def test_invalid_distribution_is_alert(self):
        frame = _frame({"bp": 0.4})
        bad = dict(_dist(0.10))
        bad["p95"] = 0.30  # p95 > p99 ⇒ 分位序非法
        member = _member(["bp"], {"bp": bad})
        outcome = evaluate_coverage(frame, [(member, ["bp"])], domain_codes=_DOMAIN)
        assert any(a["code"] == "metadata_invalid" for a in outcome.alerts)

    def test_column_without_key_is_alert_not_block(self):
        frame = _frame({"bp": 0.0, "pe": 0.9})
        outcome = evaluate_coverage(
            frame, [(_member(["bp", "pe"], {"bp": _dist(0.1)}), ["bp", "pe"])], domain_codes=_DOMAIN
        )
        assert any(
            a["code"] == COVERAGE_ERR_KEY_MISSING and a["column"] == "pe" for a in outcome.alerts
        )
        assert outcome.blocked is False

    def test_extra_keys_are_legal_superset(self):
        frame = _frame({"bp": 0.0}, n=1200)
        outcome = evaluate_coverage(
            frame,
            [(_member(["bp"], {"bp": _dist(0.1), "ht_x": _dist(0.2)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.extra_keys == ["ht_x"]
        assert outcome.alerts == []


class TestSameDomain:
    def test_out_of_domain_rows_excluded_from_rate(self):
        # 域内 2 行全有值；域外 98 行全 NaN ⇒ 按域内计算 rate=0（否则 0.98 会 error）
        frame = _frame({"bp": 0.0}, n=2, extra_rows=98)
        outcome = evaluate_coverage(
            frame, [(_member(["bp"], {"bp": _dist(0.1)}), ["bp"])], domain_codes=_DOMAIN
        )
        assert outcome.verdicts[0]["rate"] == pytest.approx(0.0)
        assert outcome.rate_denominator == "main_board_rows"
        assert outcome.n_rows_used == 2
        assert outcome.n_rows_total == 100

    def test_domain_missing_skips_judging_with_alert(self):
        frame = _frame({"bp": 0.9})
        outcome = evaluate_coverage(
            frame, [(_member(["bp"], {"bp": _dist(0.1)}), ["bp"])], domain_codes=None
        )
        assert outcome.verdicts == []
        assert outcome.blocked is False
        assert any(a["code"] == COVERAGE_ERR_DOMAIN_MISSING for a in outcome.alerts)


class TestRollingBaseline:
    def test_rolling_used_when_history_sufficient(self):
        days = [f"202610{i:02d}" for i in range(1, ROLLING_MIN_DAYS + 1)]
        history = {day: {"bp": 0.05} for day in days}
        outcome = evaluate_coverage(
            _frame({"bp": 0.20}),
            [(_member(["bp"], {"bp": _dist(0.01)}), ["bp"])],
            domain_codes=_DOMAIN,
            rate_history=history,
        )
        verdict = outcome.verdicts[0]
        assert verdict["baseline_source"] == "rolling"
        assert verdict["p95"] == pytest.approx(0.05)
        assert verdict["severity"] == "error"  # 0.20 > rolling p99(0.05)

    def test_rolling_falls_back_when_history_insufficient(self):
        history = {"20261001": {"bp": 0.05}}
        outcome = evaluate_coverage(
            _frame({"bp": 0.20}),
            [(_member(["bp"], {"bp": _dist(0.01, 0.2)}), ["bp"])],
            domain_codes=_DOMAIN,
            rate_history=history,
        )
        verdict = outcome.verdicts[0]
        assert verdict["baseline_source"] == "static"
        assert verdict["severity"] == "warning"

    def test_rolling_buffer_degrades_gracefully(self):
        # 滚动窗口内自身有高值 ⇒ 基线自适应抬升（自校准语义）
        days = [f"202610{i:02d}" for i in range(1, ROLLING_MIN_DAYS + 1)]
        history = {day: {"bp": 0.18} for day in days}
        outcome = evaluate_coverage(
            _frame({"bp": 0.20}),
            [(_member(["bp"], {"bp": _dist(0.01)}), ["bp"])],
            domain_codes=_DOMAIN,
            rate_history=history,
        )
        assert outcome.verdicts[0]["baseline_source"] == "rolling"
        assert outcome.verdicts[0]["severity"] in ("warning", "error")
        assert outcome.blocked is False


class TestRateFloor:
    """决策 b（2026-10-31 用户裁决）：绝对率下限 max(分位, 2%)。"""

    def test_near_zero_baseline_below_floor_passes(self):
        # 训练分位 0.08%/0.15%（fund_* 形态），当日 0.6% ≤ 下限 2% ⇒ pass
        outcome = evaluate_coverage(
            _frame({"bp": 0.006}),
            [(_member(["bp"], {"bp": _dist(0.0008, 0.0015)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        verdict = outcome.verdicts[0]
        assert verdict["severity"] == "pass"
        assert verdict["error_threshold"] == pytest.approx(COVERAGE_RATE_FLOOR)
        assert verdict["rate_floor"] == pytest.approx(COVERAGE_RATE_FLOOR)

    def test_near_zero_baseline_above_floor_is_error(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.03}),
            [(_member(["bp"], {"bp": _dist(0.0008, 0.0015)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.verdicts[0]["severity"] == "error"

    def test_floor_at_boundary_is_inclusive(self):
        # 恰好等于下限 ⇒ pass（`>` 语义；`>=` 漂移将被锁定）
        assert classify_severity(COVERAGE_RATE_FLOOR, 0.0, 0.001) == "pass"
        assert classify_severity(COVERAGE_RATE_FLOOR + 1e-9, 0.0, 0.001) == "error"

    def test_floor_does_not_loosen_sparse_columns(self):
        # 稀疏列分位远高于下限 ⇒ 仍按分位判定（不会因下限放宽）
        outcome = evaluate_coverage(
            _frame({"cons_target_upside": 0.61}),
            [
                (
                    _member(["cons_target_upside"], {"cons_target_upside": _dist(0.60, 0.61)}),
                    ["cons_target_upside"],
                )
            ],
            domain_codes=_DOMAIN,
        )
        verdict = outcome.verdicts[0]
        assert verdict["severity"] == "warning"
        assert verdict["error_threshold"] == pytest.approx(0.61)

    def test_all_nan_still_error_regardless_of_floor(self):
        outcome = evaluate_coverage(
            _frame({"bp": 1.0}),
            [(_member(["bp"], {"bp": _dist(0.0, 0.0)}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.verdicts[0]["severity"] == "error"
        assert outcome.verdicts[0]["error_code"] == COVERAGE_ERR_ALL_NAN


class TestDriftExempt:
    """决策 a（2026-11-01 用户批准）：结构性漂移单列豁免（error 封顶 warning）。"""

    def _exempt_member(self, columns, distribution, exempt) -> CoverageMember:
        return CoverageMember(
            member_id="m1",
            columns=list(columns),
            train_params={
                "feature_missing_rate_schema_version": FEATURE_MISSING_RATE_SCHEMA_VERSION,
                "feature_missing_rate": distribution,
            },
            drift_exempt=frozenset(exempt),
        )

    def test_exempt_column_error_capped_to_warning(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.50}),
            [(self._exempt_member(["bp"], {"bp": _dist(0.30)}, {"bp"}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        verdict = outcome.verdicts[0]
        assert verdict["severity"] == "warning"
        assert verdict["drift_exempt"] is True
        assert outcome.error_columns == 0
        assert outcome.blocked is False

    def test_exempt_column_still_counts_warning(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.50}),
            [(self._exempt_member(["bp"], {"bp": _dist(0.30)}, {"bp"}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert any(a["severity"] == "warning" and a["column"] == "bp" for a in outcome.alerts)

    def test_all_nan_not_exempted(self):
        outcome = evaluate_coverage(
            _frame({"bp": 1.0}),
            [(self._exempt_member(["bp"], {"bp": _dist(0.30)}, {"bp"}), ["bp"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.verdicts[0]["severity"] == "error"
        assert outcome.verdicts[0]["error_code"] == COVERAGE_ERR_ALL_NAN

    def test_non_exempt_columns_unaffected(self):
        outcome = evaluate_coverage(
            _frame({"bp": 0.50, "pe": 0.50}),
            [
                (
                    self._exempt_member(
                        ["bp", "pe"], {"bp": _dist(0.30), "pe": _dist(0.30)}, {"bp"}
                    ),
                    ["bp", "pe"],
                )
            ],
            domain_codes=_DOMAIN,
        )
        by_col = {v["column"]: v for v in outcome.verdicts}
        assert by_col["bp"]["severity"] == "warning"
        assert by_col["pe"]["severity"] == "error"
        assert outcome.error_columns == 1


class TestSignalLayerStillRejectsAllNaN:
    """reviewer1 P1-A：门 4「不阻断」≠ 信号层继续预测。

    门 4 对全空列只记 error；信号层 `_check_feature_column_quality` 仍硬拒全空列
    （行为冻结副本）——本用例锁定该端到端语义：全空 ⇒ 信号层拒预测；高缺失
    （非全空）⇒ 仅 warning 不阻断。
    """

    def test_all_nan_column_is_rejected_by_signal_layer(self):
        from src.lazybull.v2.core.signal.ml_signal import _classify_feature_column_quality

        series = pd.Series([np.nan] * 100)
        reject, warn = _classify_feature_column_quality("bp", series, 0.5)
        assert reject == "全空" and warn is None

    def test_high_missing_non_all_nan_only_warns(self):
        from src.lazybull.v2.core.signal.ml_signal import _classify_feature_column_quality

        series = pd.Series([np.nan] * 80 + [0.1] * 20)
        reject, warn = _classify_feature_column_quality("bp", series, 0.5)
        assert reject is None and warn is not None


class TestDriftExemptSchema:
    def test_valid_registration_read(self):
        params = {
            "feature_missing_rate_drift_exempt": {
                "columns": ["a", "b"],
                "registered_at": "2026-11-01",
                "review_condition": "观察期 10 日复核",
            }
        }
        assert read_drift_exempt(params) == frozenset({"a", "b"})

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            [],
            {"columns": ["a"]},
            {"columns": ["a"], "registered_at": "x"},
            {"columns": "a", "registered_at": "x", "review_condition": "y"},
            {"columns": [1, ""], "registered_at": "x", "review_condition": "y"},
        ],
    )
    def test_invalid_registration_never_enables_exemption(self, bad):
        params = {} if bad is None else {"feature_missing_rate_drift_exempt": bad}
        assert read_drift_exempt(params) == frozenset()


class TestAggregationPerMember:
    """reviewer3 P1-1：分子/分母同集合（逐成员阈值与逐成员 error 数）。"""

    def _member(self, columns, dist, member_id):
        return _member(columns, {c: dist for c in columns}, member_id=member_id)

    def test_shared_columns_not_double_counted(self):
        columns = [f"s{i:02d}" for i in range(20)]
        frame = _frame({c: (0.50 if i < 8 else 0.0) for i, c in enumerate(columns)}, n=1200)
        dist = _dist(0.05)
        outcome = evaluate_coverage(
            frame,
            [
                (self._member(columns, dist, "A"), columns),
                (self._member(columns, dist, "B"), columns),
            ],
            domain_codes=_DOMAIN,
        )
        assert outcome.error_columns == 8
        assert outcome.error_threshold == 15
        assert outcome.blocked is False

    def test_single_member_over_threshold_blocks(self):
        columns = [f"t{i:02d}" for i in range(20)]
        frame = _frame({c: (0.50 if i < 15 else 0.0) for i, c in enumerate(columns)}, n=1200)
        outcome = evaluate_coverage(
            frame, [(self._member(columns, _dist(0.05), "A"), columns)], domain_codes=_DOMAIN
        )
        assert outcome.blocked is True
        assert "成员 A" in outcome.blocked_reason


class TestAsOfRolling:
    """reviewer1 P1-B / reviewer3 P1-2：滚动窗口严格早于判定日。"""

    def test_current_and_future_days_excluded(self):
        history = {f"202609{i:02d}": {"bp": 0.01} for i in range(1, 26)}
        history["20261020"] = {"bp": 0.99}  # 判定日本身（不得参与）
        history["20261021"] = {"bp": 0.99}  # 未来日（不得参与）
        outcome = evaluate_coverage(
            _frame({"bp": 0.02}),
            [(_member(["bp"], {"bp": _dist(0.001, 0.001)}), ["bp"])],
            domain_codes=_DOMAIN,
            rate_history=history,
            current_date="20261020",
        )
        verdict = outcome.verdicts[0]
        assert verdict["baseline_source"] == "rolling"
        assert verdict["p99"] < 0.5  # 未被当日/未来 0.99 抬升
        assert verdict["baseline_days"] == 25


class TestDomainAndRateGuards:
    """reviewer3 P1-3：空域/非有限率不得判 pass。"""

    def test_disjoint_domain_is_error_and_not_judged(self):
        frame = _frame({"bp": 0.0})
        frame["ts_code"] = ["000001.SZ"] * len(frame)  # 与 _DOMAIN 无交集
        outcome = evaluate_coverage(
            frame, [(_member(["bp"], {"bp": _dist(0.1)}), ["bp"])], domain_codes=_DOMAIN
        )
        assert outcome.verdicts == []
        assert outcome.blocked is False
        assert any(a["code"] == COVERAGE_ERR_DOMAIN_EMPTY for a in outcome.alerts)

    def test_nan_history_values_filtered(self):
        import json
        import tempfile

        from src.lazybull.v2.hosts.paper.coverage_history import load_coverage_history

        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "coverage_history.json").write_text(
                json.dumps({"days": {"20261020": {"bp": float("nan"), "pe": 0.5}}}),
                encoding="utf-8",
            )
            history = load_coverage_history(runs)
            assert "bp" not in history["20261020"]
            assert history["20261020"]["pe"] == 0.5


class TestStrictMetadata:
    """reviewer3 P1-4：schema/分位/豁免全字段一致拒绝。"""

    @pytest.mark.parametrize("bad_version", [2.9, "2", True, None, 1])
    def test_version_must_be_exact_v2_int(self, bad_version):
        coverage, error = read_feature_missing_rate(
            {
                "feature_missing_rate_schema_version": bad_version,
                "feature_missing_rate": {"bp": _dist(0.1)},
            }
        )
        assert coverage is None and error == "metadata_schema_unsupported"

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: d.pop("p50"),
            lambda d: d.update({"p90": 0.9}),
            lambda d: d.update({"n_days": 2.0}),
            lambda d: d.update({"n_days": True}),
            lambda d: d.update({"p99_unstable": "no"}),
            lambda d: d.update({"mean": 1.5}),
        ],
    )
    def test_invalid_distribution_rejected(self, mutate):
        stats = _dist(0.1)
        mutate(stats)
        _, error = read_feature_missing_rate(
            {
                "feature_missing_rate_schema_version": 2,
                "feature_missing_rate": {"bp": stats},
            }
        )
        assert error == "metadata_invalid"

    def test_mixed_exempt_columns_block_entire_exemption(self):
        params = {
            "feature_missing_rate_drift_exempt": {
                "columns": ["bp", 1],
                "registered_at": "2026-11-01",
                "review_condition": "x",
            }
        }
        assert read_drift_exempt(params) == frozenset()

    def test_unknown_exempt_column_disables_whole_block(self):
        frame = _frame({"bp": 0.50})
        member = CoverageMember(
            member_id="m1",
            columns=["bp"],
            train_params={
                "feature_missing_rate_schema_version": 2,
                "feature_missing_rate": {"bp": _dist(0.30)},
                "feature_missing_rate_stat": {"row_domain": "train_filtered_main_board"},
            },
            drift_exempt=frozenset({"bp", "ghost"}),
        )
        outcome = evaluate_coverage(frame, [(member, ["bp"])], domain_codes=_DOMAIN)
        assert any(a["code"] == COVERAGE_ERR_EXEMPT_UNKNOWN for a in outcome.alerts)
        assert outcome.verdicts[0]["severity"] == "error"  # 豁免整块失效


class TestRowDomainBinding:
    def test_declared_domain_mismatch_skips_member(self):
        frame = _frame({"bp": 0.0})
        member = CoverageMember(
            member_id="m1",
            columns=["bp"],
            train_params={
                "feature_missing_rate_schema_version": 2,
                "feature_missing_rate": {"bp": _dist(0.1)},
                "feature_missing_rate_stat": {"row_domain": "all_market_rows"},
            },
        )
        outcome = evaluate_coverage(frame, [(member, ["bp"])], domain_codes=_DOMAIN)
        assert outcome.verdicts == []
        assert any(a["code"] == COVERAGE_ERR_ROW_DOMAIN for a in outcome.alerts)


class TestUncoveredCountsAsError:
    def test_missing_column_counts_toward_aggregate(self):
        frame = _frame({"bp": 0.50})
        outcome = evaluate_coverage(
            frame,
            [(_member(["bp", "ghost"], {"bp": _dist(0.05)}), ["bp", "ghost"])],
            domain_codes=_DOMAIN,
        )
        assert outcome.member_aggregates[0]["error_columns"] == 2


class TestDistributionSummary:
    def test_quantiles_and_short_window_degradation(self):
        rates = {"bp": [0.0] * 500 + [0.5] * 4}
        summary = summarize_daily_missing_rates(rates)["bp"]
        assert summary["n_days"] == 504
        assert summary["p95"] <= summary["p99"] <= summary["max"]
        assert summary["p99_unstable"] is False

        short = summarize_daily_missing_rates({"bp": [0.0] * 10 + [0.5]})["bp"]
        assert short["p99_unstable"] is True
        assert short["p99"] == pytest.approx(short["p95"])

    def test_multi_member_judgment_keeps_member_identity(self):
        frame = _frame({"bp": 0.30, "pe": 0.10})
        members = [
            (_member(["bp"], {"bp": _dist(0.10)}, member_id="vA"), ["bp"]),
            (
                _member(["bp", "pe"], {"bp": _dist(0.10), "pe": _dist(0.20)}, member_id="vB"),
                ["bp", "pe"],
            ),
        ]
        outcome = evaluate_coverage(frame, members, domain_codes=_DOMAIN)
        bp = [v for v in outcome.verdicts if v["column"] == "bp"]
        assert {v["member"] for v in bp} == {"vA", "vB"}
        assert all(v["severity"] == "error" for v in bp)
        assert {m["member_id"] for m in outcome.members_echo} == {"vA", "vB"}
