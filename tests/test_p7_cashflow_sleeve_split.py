# -*- coding: utf-8 -*-
"""CF-SPLIT-20-v1 定向测试（审点一 + r0_001 复审整改 1–7）。

测试纪律：全部使用合成夹具或真实仓储的**只读**结构探针，不依赖生产产物内容，
不写生产目录；真实数据路径只在 `test_real_*` 里以只读方式建立索引。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7.cashflow_sleeve_split import (
    artifacts,
    budget,
    feature_routes,
    fold_clock,
    gates,
    identity,
    probes,
    readiness,
    scoring,
    spec,
)

#: 审点一显式延后的账户级测试项（与 readiness.not_run 对应）
DEFERRED_AT_CHECKPOINT_1 = (
    "整手后跌破 V 被拒（账户接线）",
    "最低佣金/印花税逐笔核对",
    "停牌/涨跌停成交阻塞",
    "锁股退出重试",
    "跨袖重仓与权益相加（账户层）",
    "跨折持仓与挂单连续",
    "空排名日仍有估值",
    "同日财报修订与公告日回退",
)


def _feature_list(total: int, group=spec.CASHFLOW_GROUP) -> list[str]:
    head = [f"head_{index:03d}" for index in range(60)]
    tail = [f"tail_{index:03d}" for index in range(total - 60 - len(group))]
    return [*head, *group, *tail]


def _replace_column(columns: list[str], old: str, new: str) -> list[str]:
    return [new if column == old else column for column in columns]


# ── ① 列路由 ─────────────────────────────────────────────────────────


class TestFeatureRoutes:
    def test_split_140(self):
        route = feature_routes.verify_route(24008, _feature_list(140), [], "fold")
        assert (route["r_count"], route["b_count"]) == (129, 11)
        assert route["b"] == list(spec.CASHFLOW_GROUP)
        assert route["checks"]["intersection_empty"]
        assert route["checks"]["union_equals_f"]

    def test_split_154(self):
        route = feature_routes.verify_route(24022, _feature_list(154), [], "deploy")
        assert route["r_count"] == 143

    def test_order_is_inherited_not_sorted(self):
        columns = list(reversed(_feature_list(148)))
        r, b = feature_routes.split_route(columns)
        assert b == [column for column in columns if column in set(spec.CASHFLOW_GROUP)]
        assert r + b != sorted(r + b)

    def test_cf_sales_pair_goes_to_b(self):
        route = feature_routes.verify_route(24020, _feature_list(154), [], "fold")
        assert "zscore_cf_sales" in route["b"]
        assert "zscore_cf_sales" not in route["r"]

    def test_cf_nm_alias_rejected(self):
        columns = _replace_column(_feature_list(148), "zscore_cf_sales", "zscore_cf_nm")
        with pytest.raises(feature_routes.RouteError):
            feature_routes.verify_route(24013, columns, [], "fold")

    def test_forbidden_alias_in_full_count_rejected(self):
        columns = _replace_column(_feature_list(148), "head_000", "zscore_cf_nm")
        with pytest.raises(feature_routes.RouteError, match="禁用列"):
            feature_routes.verify_route(24013, columns, [], "fold")

    def test_group_column_missing_is_rejected_not_patched(self):
        columns = _replace_column(_feature_list(148), "zscore_cf_sales", "head_extra")
        with pytest.raises(feature_routes.RouteError, match="现金流组列数不符"):
            feature_routes.verify_route(24013, columns, [], "fold")

    def test_count_mismatch_not_trimmed(self):
        with pytest.raises(feature_routes.RouteError, match="列数不符冻结值"):
            feature_routes.verify_route(24008, _feature_list(141), [], "fold")

    def test_build_routes_from_synthetic_root(self, tmp_path: Path):
        model_root = tmp_path / spec.MODEL_ROOT_RELATIVE
        model_root.mkdir(parents=True)
        for version in list(spec.FOLD_VERSIONS) + [spec.DEPLOY_VERSION]:
            count = spec.EXPECTED_FEATURE_COUNTS[version]
            columns = _feature_list(count)
            (model_root / f"v{version}_features.json").write_text(
                json.dumps(columns), encoding="utf-8"
            )
            metadata = {
                "train_params": {
                    "cashflow_quality_feature_columns": [
                        column
                        for column in columns
                        if column.startswith("zscore_") and "cf_sales" not in column
                    ]
                }
            }
            (model_root / f"v{version}_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        protocol = tmp_path / spec.PROTOCOL_RELATIVE
        protocol.parent.mkdir(parents=True)
        protocol.write_text("protocol-fixture\n", encoding="utf-8")

        routes = feature_routes.build_routes(tmp_path)
        assert routes["summary"]["versions"] == 15
        assert routes["summary"]["all_b_equal_group"] is True
        table = feature_routes.route_table(routes)
        assert len(table[24008]["B"]) == 11


# ── ② 无标签/哨兵泄漏 ────────────────────────────────────────────────


class TestNoLeakage:
    def test_labels_and_sentinel_rejected(self):
        for column in ("y_ret_20", "neu_y_ret_20", spec.CASHFLOW_SENTINEL_COLUMN):
            columns = _replace_column(_feature_list(148), "head_000", column)
            with pytest.raises(feature_routes.RouteError, match="禁用列"):
                feature_routes.verify_route(24013, columns, [], "fold")

    def test_projection_is_label_independent(self):
        columns = _feature_list(148)
        frame = pd.DataFrame({column: [0.1, 0.2] for column in columns})
        frame["y_ret_20"] = [0.05, None]
        route = feature_routes.verify_route(24013, columns, [], "fold")
        with_labels = frame[route["r"] + route["b"]].copy()
        without_labels = frame.drop(columns=["y_ret_20"])[route["r"] + route["b"]].copy()
        pd.testing.assert_frame_equal(with_labels, without_labels)


# ── ③ 资格探针：真实选股域与分步排除（整改 2）──────────────────────


def _universe_fixtures(codes_n: int = 12) -> dict:
    codes = [f"{index:06d}.SZ" for index in range(codes_n)]
    mother = pd.DataFrame(
        {
            "ts_code": codes,
            "trade_date": ["20260106"] * codes_n,
            "list_days": [800] * codes_n,
            "is_st": [0] * codes_n,
            "is_suspended": [0] * codes_n,
            "tradable": [1] * codes_n,
        }
    )
    features = pd.DataFrame(
        {
            "ts_code": codes,
            "amount_ma20": [80000.0] * codes_n,
            "total_mv": [800000.0] * codes_n,
            "sw_l1_code": ["801010.SI"] * codes_n,
        }
    )
    basic = pd.DataFrame(
        {
            "ts_code": codes,
            "market": ["主板"] * codes_n,
            "name": [f"股票{index}" for index in range(codes_n)],
            "list_date": ["20100101"] * codes_n,
        }
    )
    industry = pd.DataFrame({"ts_code": codes, "l1_code": ["110000"] * codes_n})
    return {"mother": mother, "features": features, "basic": basic, "industry": industry}


def test_selection_excludes_gem_and_star():
    data = _universe_fixtures(12)
    data["basic"].loc[:3, "market"] = "创业板"
    data["basic"].loc[4:5, "market"] = "科创板"
    result = probes.apply_selection(
        data["mother"], data["features"], data["basic"], data["industry"], "20260106"
    )
    steps = {step["id"]: step for step in result["steps"]}
    assert steps["main_board"]["excluded_n"] == 6
    assert result["eligible_n"] if "eligible_n" in result else len(result["eligible"]) == 6


def test_selection_excludes_st_listing_days_and_suspension():
    data = _universe_fixtures(10)
    data["mother"].loc[0, "list_days"] = 100
    data["mother"].loc[1, "is_st"] = 1
    data["normal"] = None
    data["mother"].loc[2, "is_st"] = 1
    data["mother"].loc[3, "is_suspended"] = 1
    data["mother"].loc[4, "tradable"] = 0
    result = probes.apply_selection(
        data["mother"], data["features"], data["basic"], data["industry"], "20260106"
    )
    steps = {step["id"]: step for step in result["steps"]}
    assert steps["listed_days"]["excluded_n"] == 1
    assert steps["not_st_pit"]["excluded_n"] == 2
    assert steps["not_suspended"]["excluded_n"] == 2
    assert len(result["eligible"]) == 5


def test_selection_missing_industry_is_source_not_ready():
    data = _universe_fixtures(6)
    result = probes.apply_selection(
        data["mother"], data["features"], data["basic"], None, "20260106"
    )
    assert result["status"] == "source_not_ready"
    steps = {step["id"]: step for step in result["steps"]}
    assert steps["industry_known"]["status"] == "source_not_ready"


def test_selection_industry_unknown_is_counted():
    data = _universe_fixtures(8)
    data["industry"] = data["industry"].iloc[:5]
    result = probes.apply_selection(
        data["mother"], data["features"], data["basic"], data["industry"], "20260106"
    )
    steps = {step["id"]: step for step in result["steps"]}
    assert steps["industry_known"]["unknown_n"] == 3
    assert len(result["eligible"]) == 5


def test_selection_financial_uses_pit_codes_not_legacy_snapshot():
    data = _universe_fixtures(6)
    data["industry"].loc[0:1, "l1_code"] = "490000"  # 非银金融（PIT 旧码）
    data["features"].loc[0, "sw_l1_code"] = "801890.SI"  # 旧截面也标成非银
    result = probes.apply_selection(
        data["mother"], data["features"], data["basic"], data["industry"], "20260106"
    )
    steps = {step["id"]: step for step in result["steps"]}
    assert steps["not_financial_pit"]["excluded_n"] == 2
    # PIT 剔除先执行，冲突行已离场；桥接计数只反映留下的行
    assert result["industry_bridge"]["legacy_snapshot_financial_n"] == 0
    assert len(result["eligible"]) == 4


def test_selection_structural_gap_counts_rows_without_feature():
    data = _universe_fixtures(6)
    data["features"] = data["features"].iloc[:4]
    result = probes.apply_selection(
        data["mother"], data["features"], data["basic"], data["industry"], "20260106"
    )
    assert result["structural_gap_n"] == 2
    assert len(result["eligible"]) == 4


def test_industry_interval_uses_isna_not_string_replace():
    pit = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 2,
            "in_date": ["20100101", "20210101"],
            "out_date": pd.array(["20201231", None], dtype="string"),
            "l1_code": ["110000", "220000"],
        }
    )
    hit = probes.industry_for_day(pit, "20260106")
    assert len(hit) == 1
    assert hit.iloc[0]["l1_code"] == "220000"


# ── ④ 三臂来源取证（整改 3）────────────────────────────────────────


def _route_fixture() -> dict:
    return {
        24022: {
            "F": ["amount_ma20", "zscore_ocf_to_revenue", "cashflow_freshness_days"],
            "R": ["amount_ma20"],
            "B": list(spec.CASHFLOW_GROUP),
        }
    }


def test_source_evidence_counts_arms_separately():
    frame = pd.DataFrame(
        {
            "amount_ma20": [80000.0, 80000.0, None],
            "zscore_ocf_to_revenue": [0.5, None, None],
            "cashflow_freshness_days": [10.0, None, None],
            "ocf_to_revenue": [0.2, None, None],
        }
    )
    result = probes.source_evidence(frame, 24022, _route_fixture(), list(frame.columns))
    assert result["F"]["known_n"] == 2  # 第二行有 amount_ma20
    assert result["R"]["known_n"] == 2
    assert result["B"]["known_n"] == 1  # 只有第一行有原始比率


def test_source_evidence_label_only_row_is_not_a_source():
    """复审反例：一行只有代码与 neu_y_ret_20，不得算「有经济来源」。"""
    frame = pd.DataFrame({"neu_y_ret_20": [0.1], "amount_ma20": [None]})
    result = probes.source_evidence(frame, 24022, _route_fixture(), list(frame.columns))
    assert result["F"]["known_n"] == 0
    assert result["R"]["known_n"] == 0
    assert result["B"]["known_n"] == 0


def test_source_evidence_zscore_only_is_not_b_source():
    frame = pd.DataFrame(
        {
            "zscore_ocf_to_revenue": [0.7],
            "zscore_fcf_yield_sz": [0.2],
            # 无原始比率、无 freshness
        }
    )
    result = probes.source_evidence(frame, 24022, _route_fixture(), list(frame.columns))
    assert result["B"]["known_n"] == 0
    assert result["B_standardized_only"]["n"] == 1


def test_source_evidence_flags_forbidden_route_column():
    routes = {24022: {"F": ["y_ret_20"], "R": ["y_ret_20"], "B": []}}
    result = probes.source_evidence(pd.DataFrame({"y_ret_20": [0.1]}), 24022, routes, ["y_ret_20"])
    assert result["status"] == "leak"
    assert "y_ret_20" in result["forbidden_in_route"]


# ── ⑤ 时钟：原日志分段 + T+21（整改 1）─────────────────────────────


def _clock_fixture(tmp_path: Path) -> list[str]:
    cal_dir = tmp_path / "data/clean"
    cal_dir.mkdir(parents=True)
    days = [
        f"{year}{month:02d}{day:02d}"
        for year in (2020, 2021)
        for month in range(1, 13)
        for day in range(1, 21)
    ]
    pd.DataFrame({"cal_date": days, "is_open": [1] * len(days)}).to_parquet(
        cal_dir / "trade_cal.parquet", index=False
    )
    return days


class TestClock:
    def test_maturity_is_t_plus_21(self, tmp_path: Path):
        days = _clock_fixture(tmp_path)
        assert fold_clock.maturity_day(days[0], days, 21) == days[21]
        assert fold_clock.maturity_day(days[0], days, 20) == days[20]

    def test_maturity_beyond_calendar_raises(self, tmp_path: Path):
        days = _clock_fixture(tmp_path)
        with pytest.raises(fold_clock.ClockError, match="不足"):
            fold_clock.maturity_day(days[-3], days, 21)

    def test_log_physical_line_convention(self, tmp_path: Path):
        rows = [
            {
                "model_version": version,
                "timestamp": f"2026-09-07 11:0{index}:00",
                "start_date": "20120101",
                "end_date": "20180101",
                "val_start_date": "20160101",
                "val_end_date": "20160601",
                "val_calib_start_date": "20160602",
                "val_calib_end_date": "20161101",
                "test_start_date": "20161102",
                "test_end_date": "20171001",
                "split_index": index,
                "best_iteration": 100,
            }
            for index, version in enumerate((*spec.FOLD_VERSIONS, spec.DEPLOY_VERSION))
        ]
        frame = pd.DataFrame(rows)
        log_dir = tmp_path / "data/models/stock_selection"
        log_dir.mkdir(parents=True)
        frame.to_csv(log_dir / "ml_train_runs.csv", index=False, encoding="utf-8")

        table = fold_clock.load_training_log(tmp_path)
        assert table[24008]["physical_line"] == 2
        assert table[24009]["physical_line"] == 3
        assert table[24008]["es_start"] == "20160101"

    def test_split_derivation_requires_enough_room(self, tmp_path: Path):
        cal_dir = tmp_path / "data/clean"
        cal_dir.mkdir(parents=True)
        pd.DataFrame({"cal_date": ["20200101", "20200102"], "is_open": [1, 1]}).to_parquet(
            cal_dir / "trade_cal.parquet", index=False
        )
        with pytest.raises(fold_clock.ClockError, match="训练窗过短"):
            fold_clock._resolve_split_dates("20200101", "20200102", 0.2, 20, tmp_path)

    def test_derived_delta_vs_logged_is_recorded(self, tmp_path: Path):
        days = _clock_fixture(tmp_path)
        summary_row = {
            "split_index": 0,
            "model_version": 24008,
            "val_ratio": 0.2,
            "test_start": days[210],
            "test_end": days[-30],
        }
        log_row = {
            "physical_line": 2,
            "timestamp": "2026-09-07 11:20:27",
            "outer_train_start": days[0],
            "outer_train_end": days[200],
            "es_start": days[80],
            "es_end": days[100],
            "calib_start": days[101],
            "calib_end": days[130],
            "test_start": days[210],
            "test_end": days[-1],
            "split_index": 0,
            "best_iteration": 699,
        }
        row = fold_clock.build_fold_row(summary_row, log_row, days, tmp_path)
        assert row["maturity_offset"] == 21
        assert row["es_last_usable_info"] == days[121]
        assert row["calib_last_usable_info"] == days[151]
        assert row["tail_isolation_start"] == days[131]
        # 原日志边界把早停末端标签跨进校准段：登记但不致命
        assert "es_maturity>=calib_start" in row["maturity_breach"]
        assert row["maturity_breach_purified"] == ""
        assert row["es_last_usable_sample"] and row["es_last_usable_sample"] < row["es_end"]
        assert row["es_purged_samples"] > 0
        assert isinstance(row["derived_val_start_delta_days"], int)

    def test_maturity_breach_is_detected(self, tmp_path: Path):
        days = _clock_fixture(tmp_path)
        summary_row = {
            "split_index": 0,
            "model_version": 24008,
            "val_ratio": 0.2,
            "test_start": days[210],
            "test_end": days[-30],
        }
        log_row = {
            "physical_line": 2,
            "timestamp": "t",
            "outer_train_start": days[0],
            "outer_train_end": days[200],
            "es_start": days[80],
            "es_end": days[100],
            "calib_start": days[101],
            "calib_end": days[150],  # 150 + 21 = 171 ≥ test_start(160) ⇒ 跨段
            "test_start": days[160],
            "test_end": days[-1],
            "split_index": 0,
            "best_iteration": 1,
        }
        row = fold_clock.build_fold_row(summary_row, log_row, days, tmp_path)
        assert "calib_maturity>=test_start" in row["maturity_breach"]

    def test_timeline_overlap_rejected(self, tmp_path: Path, monkeypatch):
        calendar = ["20190329", "20190401", "20190402", "20190403"]
        monkeypatch.setattr(fold_clock, "account_calendar", lambda root: list(calendar))
        monkeypatch.setattr(
            fold_clock,
            "load_trade_days",
            lambda root, start, end: [d for d in calendar if str(start) <= d <= str(end)],
        )
        folds = [
            {
                "split_index": 0,
                "model_version": 24008,
                "test_start": "20190401",
                "test_end": "20190402",
            },
            {
                "split_index": 1,
                "model_version": 24009,
                "test_start": "20190402",
                "test_end": "20190403",
            },
        ]
        with pytest.raises(fold_clock.ClockError, match="模型时间线重叠"):
            fold_clock.build_timeline(folds, tmp_path)

    def test_deploy_window_constants(self):
        assert spec.WINDOW["seam_days"] == 20
        assert spec.WINDOW["deploy_days"] == 176
        assert spec.WINDOW["nav_points"] == 1839
        assert spec.SPLIT_CLOCK["label_maturity_offset"] == 21


# ── ⑥ 判据：符号与轴（整改 5）──────────────────────────────────────


class TestGates:
    def test_negative_loss_is_rejected(self):
        with pytest.raises(gates.GateError, match="为负"):
            gates.portfolio_gate(0.26, -0.30)
        with pytest.raises(gates.GateError, match="为负"):
            gates.r_gate(0.30, -0.30, -0.20)

    def test_mdd_loss_from_raw(self):
        assert gates.mdd_loss_from_raw(-0.317443) == pytest.approx(0.317443)
        assert gates.mdd_loss_from_raw(0.0) == 0.0
        with pytest.raises(gates.GateError, match="为正"):
            gates.mdd_loss_from_raw(0.317443)

    def test_portfolio_boundaries(self):
        assert gates.portfolio_gate(0.25, 0.10)["booleans"]["cagr_gt_25pct"] is False
        assert gates.portfolio_gate(0.2500000001, 0.10)["passed"] is True
        assert gates.portfolio_gate(0.30, 0.20)["booleans"]["mdd_loss_lt_20pct"] is False
        assert gates.portfolio_gate(0.30, 0.1999999)["passed"] is True

    def test_r_boundaries(self):
        assert gates.r_gate(0.26, 0.30, 0.30)["booleans"]["cagr_ge_26pct"] is True
        assert gates.r_gate(0.26, 0.30, 0.30)["passed"] is False
        assert gates.r_gate(0.2599, 0.20, 0.30)["passed"] is False
        assert gates.r_gate(0.30, 0.317443, 0.317443)["passed"] is False

    def test_nav_frame_requires_date_column(self):
        frame = pd.DataFrame({"nav": [1.0, 1.1]})
        with pytest.raises(gates.GateError, match="缺 trade_date"):
            gates.validate_nav_frame(frame)

    def test_nav_frame_rejects_inf(self):
        frame = pd.DataFrame({"trade_date": ["20260105", "20260106"], "nav": [1.0, float("inf")]})
        with pytest.raises(gates.GateError, match="非有限"):
            gates.validate_nav_frame(frame)

    def test_nav_frame_rejects_duplicate_and_single_point(self):
        duplicated = pd.DataFrame({"trade_date": ["20260105", "20260105"], "nav": [1.0, 1.1]})
        with pytest.raises(gates.GateError, match="重复日期"):
            gates.validate_nav_frame(duplicated)
        single = pd.DataFrame({"trade_date": ["20260105"], "nav": [1.0]})
        with pytest.raises(gates.GateError, match="不足以"):
            gates.validate_nav_frame(single)

    def test_nav_frame_strict_axis(self):
        frame = pd.DataFrame({"trade_date": ["20260105", "20260106"], "nav": [1.0, 1.1]})
        assert gates.validate_nav_frame(frame, expected_days=["20260105", "20260106"])["rows"] == 2
        with pytest.raises(gates.GateError, match="冻结轴"):
            gates.validate_nav_frame(frame, expected_days=["20260105", "20260107"])

    def test_align_axes_requires_two_points_and_strict_axis(self):
        with pytest.raises(gates.GateError, match="只有一个点|只有 1 个点"):
            gates.align_axes({"A": ["20260105"]})
        with pytest.raises(gates.GateError, match="日期轴不一致"):
            gates.align_axes({"A": ["20260105", "20260106"], "B": ["20260105"]})
        with pytest.raises(gates.GateError, match="冻结轴"):
            gates.align_axes(
                {"A": ["20260105", "20260106"]}, expected_days=["20260105", "20260106", "20260107"]
            )

    def test_conjunction_requires_both(self):
        results = {"1000000": {"passed": True}, "2000000": {"passed": False}}
        assert gates.portfolio_conjunction(results)["passed"] is False
        with pytest.raises(gates.GateError):
            gates.portfolio_conjunction({"1000000": {"passed": True}})


# ── ⑦ 账户角色与金额（整改 4）──────────────────────────────────────


class TestAccountRoles:
    def test_big_leg_is_top20_with_15pct_cap(self):
        slots = readiness.slot_amounts()
        assert slots["big_leg_uses_top20"] is True
        assert slots["big_leg_cap_by_capital"]["1000000"] == pytest.approx(120_000.0)
        assert slots["big_leg_cap_by_capital"]["2000000"] == pytest.approx(240_000.0)

    def test_small_leg_is_six_slots(self):
        slots = readiness.slot_amounts()
        assert slots["small_leg_uses_six_slots"] is True
        assert slots["small_leg_slot_by_capital"]["1000000"] == pytest.approx(33_333.3333, abs=1e-3)
        assert slots["small_leg_slot_by_capital"]["2000000"] == pytest.approx(66_666.6667, abs=1e-3)
        assert slots["all_above_v"] is True

    def test_roles_cover_required_accounts(self):
        roles = {row["role"] for row in readiness.execution_configs()}
        assert {"F100", "R100", "F80", "R80", "B20", "F20"} <= roles
        assert all(
            row["top_n"] == 20 for row in readiness.execution_configs() if row["signal"] == "big"
        )

    def test_min_ticket_and_lot_price(self):
        assert spec.min_ticket_value() == pytest.approx(25_588.5363, abs=1e-3)
        rows = [row for row in readiness.execution_configs() if row["signal"] == "small"]
        assert all(
            row["min_price_for_one_lot_at_v"] == pytest.approx(255.885, abs=1e-2) for row in rows
        )

    def test_add_then_divide_not_normalized_sum(self):
        k = 1_000_000.0
        leg_a = pd.Series([800_000.0, 880_000.0])
        leg_b = pd.Series([200_000.0, 210_000.0])
        correct = ((leg_a + leg_b) / k).tolist()
        wrong = ((leg_a / leg_a.iloc[0] + leg_b / leg_b.iloc[0]) / 2).tolist()
        assert correct[1] == pytest.approx(1.09)
        assert wrong[1] != pytest.approx(correct[1])


# ── ⑧ 工件、身份与可还原（整改 6）──────────────────────────────────


class TestArtifacts:
    def test_production_root_rejected(self):
        assert artifacts.require_write_scope_probe()["rejected"] is True
        with pytest.raises(Exception):
            artifacts.guard(artifacts.REPO_ROOT / "data/models/stock_selection/x.json")

    def test_create_run_dir_refuses_overwrite(self, tmp_path: Path):
        camp = tmp_path / "camp"
        camp.mkdir()
        artifacts.create_run_dir(camp, "r0_001", [str(tmp_path)])
        with pytest.raises(artifacts.ArtifactError, match="禁止覆盖"):
            artifacts.create_run_dir(camp, "r0_001", [str(tmp_path)])

    def test_seal_and_require_roundtrip(self, tmp_path: Path):
        camp = tmp_path / "camp"
        camp.mkdir()
        run_dir = artifacts.create_run_dir(camp, "r0_002", [str(tmp_path)])
        names = ("a.json", "b.csv")
        for name in names:
            (run_dir / "work" / name).write_text("x", encoding="utf-8")
            artifacts.publish_file(run_dir / "work" / name, run_dir, name, [str(tmp_path)])
        sealed = artifacts.seal_run(run_dir, "prepare", {"ok": True}, names, [str(tmp_path)])
        assert sealed["manifest_sha256"]
        manifest = artifacts.require_published(camp, "r0_002", "prepare")
        assert manifest["payload"]["ok"] is True

    def test_publish_refuses_different_bytes(self, tmp_path: Path):
        camp = tmp_path / "camp"
        camp.mkdir()
        run_dir = artifacts.create_run_dir(camp, "r0_005", [str(tmp_path)])
        (run_dir / "work" / "a.json").write_text("x", encoding="utf-8")
        artifacts.publish_file(run_dir / "work" / "a.json", run_dir, "a.json", [str(tmp_path)])
        (run_dir / "work" / "a.json").write_text("y", encoding="utf-8")
        with pytest.raises(artifacts.ArtifactError, match="字节不同"):
            artifacts.publish_file(run_dir / "work" / "a.json", run_dir, "a.json", [str(tmp_path)])


class TestIdentity:
    def test_code_inventory_is_recursive(self, tmp_path: Path):
        package = tmp_path / "scripts/pkg"
        package.mkdir(parents=True)
        (package / "a.py").write_text("x", encoding="utf-8")
        (package / "b.py").write_text("y", encoding="utf-8")
        (package / "noise.log").write_text("z", encoding="utf-8")
        inventory = identity._code_inventory(tmp_path, ["scripts/pkg"])
        paths = sorted(item["path"] for item in inventory)
        assert paths == ["scripts/pkg/a.py", "scripts/pkg/b.py"]
        assert all(item["sha256"] for item in inventory)

    def test_frozen_files_require_every_entry(self, tmp_path: Path):
        anchor = tmp_path / "anchor"
        anchor.mkdir()
        (anchor / "one.json").write_text("{}", encoding="utf-8")
        frozen = {"one.json": identity.sha256_file(anchor / "one.json")}
        assert identity._verify_frozen_files(anchor, frozen)["verified_n"] == 1
        with pytest.raises(identity.IdentityError, match="缺件"):
            identity._verify_frozen_files(anchor, {**frozen, "two.json": "deadbeef"})
        with pytest.raises(identity.IdentityError, match="SHA 不符"):
            identity._verify_frozen_files(anchor, {"one.json": "0" * 64})

    def test_input_identity_drift_detected(self):
        registered = {"sources": [{"name": "a", "digest": "aaa"}]}
        current = {"sources": [{"name": "a", "digest": "bbb"}]}
        result = identity.verify_registered_manifest(registered, current)
        assert result["status"] == "drift"

    def test_baseline_rejects_wrong_manifest(self, tmp_path: Path):
        anchor = tmp_path / spec.A_BASELINE["relative"]
        anchor.mkdir(parents=True)
        (anchor / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(identity.IdentityError, match="manifest 字节不符"):
            identity.verify_a_baseline(tmp_path)


# ── ⑨ 规格、预算、卫生 ──────────────────────────────────────────────


class TestSpec:
    def test_spec_sha_is_deterministic(self):
        assert spec.spec_sha256() == spec.spec_sha256()

    def test_machine_spec_marks_boundaries(self):
        machine = spec.machine_spec_document()
        assert machine["formal_admission"] == "not_evaluated"
        assert machine["alpha_consumed"] is False
        assert machine["authorized_stages"] == ["prepare"]
        assert machine["split_clock"]["label_maturity_offset"] == 21
        assert len(machine["signal_execution"]) == 2
        assert machine["account_roles"]["B20"]["top_n"] == 6
        assert machine["account_roles"]["R80"]["top_n"] == 20


class TestBudget:
    def test_run_a_scope_excludes_b(self):
        payload = budget.build_pilot_costs(
            {"total_seconds": 40.0}, {"samples": {}}, {"mb_per_second": 100.0}
        )
        run_a = next(item for item in payload["projections"] if item["batch"] == "run-a")
        run_b = next(item for item in payload["projections"] if item["batch"] == "run-b")
        assert "不含 B" in " ".join(run_a["what_it_does"])
        assert run_b["status"] == "not_authorized"
        assert "审点二" in run_b["gate"]

    def test_structural_rebuild_requires_measurement(self):
        payload = budget.build_pilot_costs(
            {"total_seconds": 1.0}, {"samples": {}}, {"mb_per_second": 1.0}
        )
        assert payload["structural_rebuild"]["status"] == "not_measured"

    def test_structural_measurement_extrapolates(self):
        measured = {
            "block": ["20251203", "20260703"],
            "kept_written_days": 140,
            "elapsed_seconds": 465.146,
            "peak_rss_mb": 4129.1,
            "rows": 5166,
            "columns_n": 407,
        }
        payload = budget.build_pilot_costs(
            {"total_seconds": 1.0}, {"samples": {}}, {"mb_per_second": 1.0}, structural=measured
        )
        block = payload["structural_rebuild"]
        assert block["status"] == "measured"
        assert block["quotes"]["planned_blocks"] == budget._month_span(
            budget.REBUILD_MONTH_START, budget.REBUILD_MONTH_END
        )
        assert block["quotes"]["lower_hours"] > 1.5
        assert block["quotes"]["upper_hours"] > block["quotes"]["lower_hours"]
        assert block["rss_is_true_peak"] is False
        assert block["disk_bytes"] == "not_measured"

    def test_authorization_threshold(self):
        assert spec.BUDGET_REFERENCE["approval_required_over_minutes"] == 20
        assert spec.BUDGET_REFERENCE["structural_rebuild"]["status"] == "separate_line_item"


class TestHygieneParsing:
    def test_baseline_and_waived_markers_are_not_new(self):
        text = "\n".join(
            [
                "[hard] a.py func_lines: 函数 f 200 行 > 硬上限 150（存量基线）",
                "[hard] b.py mccabe: 'g' is too complex (20)（已豁免）",
                "[hard] c.py file_lines: 文件 1300 行 > 硬上限 1200",
                "检查文件 10 个；软超限 3 条，新增硬超限 1 条，存量基线 1 条，豁免 1 条",
            ]
        )
        parsed = readiness.parse_complexity_output(text)
        assert parsed["new_hard_overruns"] == 1
        assert parsed["hard_total"] == 3
        assert len(parsed["new_hard_samples"]) == 1


class TestCli:
    def test_requires_stage(self):
        from scripts.v2_p7.cashflow_sleeve_split import run as cli

        with pytest.raises(SystemExit):
            cli.main([])

    @pytest.mark.parametrize("stage", ["run-a", "run-b", "verify"])
    def test_unauthorized_stages_refuse(self, stage: str, capsys):
        from scripts.v2_p7.cashflow_sleeve_split import run as cli

        code = cli.main([stage, "--campaign", "cf_split_v1", "--parent", "x", "--run", "y"])
        assert code == cli.EXIT_BLOCKED
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "blocked"


class TestDeferredRegistration:
    def test_deferred_items_documented(self):
        assert len(DEFERRED_AT_CHECKPOINT_1) >= 8
        assert "explicitly_not_run" in spec.machine_spec_document()

    def test_recompute_rules_tolerance(self):
        rules = readiness.recompute_rules()
        assert rules["equity_identity_abs_tolerance_yuan"] == 1e-6
        assert rules["metric_abs_tolerance_ratio"] == 1e-10


# ── ⑩ r0_004 整改：行业注入 / 缺失位置反事实 / 有限值来源 / 部署净化 / 键数 ──


def _pit_fixture(tmp_path: Path) -> None:
    """合成行业 PIT + 码表（600783.SH 在 2019-04-01 属 SW2014 非银金融）。"""
    clean = tmp_path / "data/clean"
    clean.mkdir(parents=True, exist_ok=True)
    pit = pd.DataFrame(
        {
            "code6": ["600783", "600783", "000001"],
            "ts_code": ["600783.SH", "600783.SH", "000001.SZ"],
            "in_date": ["20140221", "20210730", "19910403"],
            "out_date": ["20210730", "20220729", None],
            "ind_code": ["490301", "510101", "480000"],
            "l1_code": ["490000", "510000", "480000"],
            "l2_code": ["490300", "510100", "480100"],
            "l1_name": ["非银金融", "综合", "银行"],
            "l2_name": ["多元金融", "综合Ⅱ", "银行Ⅱ"],
            "l3_name": ["多元金融", "综合Ⅲ", "股份制银行Ⅲ"],
            "sw_version": ["SW2014", "SW2021", "SW2021"],
            "source_updated_at": [None, None, None],
            "code_alias": [False, False, False],
        }
    )
    pit.to_parquet(clean / "sw_industry_pit.parquet", index=False)
    code_table = pd.DataFrame(
        {
            "src": ["SW2014", "SW2014", "SW2014", "SW2021", "SW2021", "SW2021"],
            "industry_code": ["490000", "490300", "490301", "510000", "510100", "480000"],
            "level": ["L1", "L2", "L3", "L1", "L2", "L1"],
            "industry_name": ["非银金融", "多元金融", "多元金融", "综合", "综合Ⅱ", "银行"],
            "index_code": [
                "801790.SI",
                "851900.SI",
                "851921.SI",
                "801230.SI",
                "801231.SI",
                "801780.SI",
            ],
        }
    )
    code_table.to_parquet(clean / "sw_industry_code_table.parquet", index=False)


class TestIndustryPitInjection:
    def test_snapshot_matches_production_schema_and_day_industry(self, tmp_path: Path):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        _pit_fixture(tmp_path)
        snapshot = structbuild.industry_pit_snapshot("20190401", tmp_path)
        assert snapshot["stats"]["schema"] == spec.INDUSTRY_PIT_INJECTION["snapshot_schema"]
        assert snapshot["stats"]["codes"] == 2
        row = snapshot["frame"][snapshot["frame"]["ts_code"] == "600783.SH"].iloc[0]
        # r0_003 复核身份探针：2019-04-01 是非银金融（801790.SI），不是快照里的机械设备
        assert row["sw_l1_code"] == "801790.SI"
        assert row["sw_l1"] == "非银金融"
        assert snapshot["stats"]["identity_probe"]["match"] is True

    def test_identity_probe_mismatch_reported(self, tmp_path: Path):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        _pit_fixture(tmp_path)
        # 让 600783 在 20190401 无命中 → 探针应报 not match（而不是静默通过）
        clean = tmp_path / "data/clean"
        pit = pd.read_parquet(clean / "sw_industry_pit.parquet")
        pit.loc[pit["ts_code"] == "600783.SH", "in_date"] = "20200101"
        pit.to_parquet(clean / "sw_industry_pit.parquet", index=False)
        snapshot = structbuild.industry_pit_snapshot("20190401", tmp_path)
        assert snapshot["stats"]["identity_probe"]["match"] is False

    def test_unresolved_code_mapping_is_counted(self, tmp_path: Path):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        _pit_fixture(tmp_path)
        clean = tmp_path / "data/clean"
        pit = pd.read_parquet(clean / "sw_industry_pit.parquet")
        # 把 000001 的版本改成 UNMAPPED（码表查不到）→ sw_l1_code 反查失败并登记
        pit.loc[pit["ts_code"] == "000001.SZ", "sw_version"] = "UNMAPPED"
        pit.to_parquet(clean / "sw_industry_pit.parquet", index=False)
        snapshot = structbuild.industry_pit_snapshot("20190401", tmp_path)
        assert snapshot["stats"]["l1_unresolved"] == 1


class TestLabelReshapeCounterfactual:
    def test_reshape_changes_missing_positions_deterministically(self):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        frame = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(9)],
                "y_ret_20": [0.1, 0.2, None, 0.3, None, None, 0.4, None, None],
            }
        )
        out = structbuild._reshape_label_frame(frame.copy())
        values = out["y_ret_20"]
        stats = out.attrs["label_reshape_changed"]
        # 有→无：有限值的前 k 个（位置 0/1/3）被置 NaN
        assert values.iloc[0] is pd.NA or pd.isna(values.iloc[0])
        assert pd.isna(values.iloc[1])
        assert pd.isna(values.iloc[3])
        # 无→有：缺失位置的最后 k 个（位置 5/7/8）被填 −0.5
        assert values.iloc[5] == -0.5 and values.iloc[7] == -0.5 and values.iloc[8] == -0.5
        # 缺失位置确实改变了：原缺失 [2,4,5,7,8] → 现缺失 [0,1,2,3,4]
        assert stats["to_nan"] == 3 and stats["filled"] == 3
        again = structbuild._reshape_label_frame(frame.copy())
        assert again["y_ret_20"].equals(values), "重排必须确定性可复现"

    def test_non_label_columns_untouched(self):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        frame = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(9)],
                "head_000": [1.5] * 9,
                "y_ret_20": [None, 0.1, 0.2, None, 0.3, None, 0.4, None, None],
            }
        )
        out = structbuild._reshape_label_frame(frame.copy())
        assert out["head_000"].tolist() == frame["head_000"].tolist()

    def test_signature_excludes_label_value_and_missing_position_changes(self):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        def _result(frame: pd.DataFrame):
            return {
                "frames": [("20200101", frame)],
                "rows": len(frame),
                "dates": ["20200101"],
            }

        base = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(6)],
                "head_000": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "y_ret_20": [0.01, None, 0.02, None, 0.03, None],
            }
        )
        value_perturbed = base.copy()
        value_perturbed["y_ret_20"] = value_perturbed["y_ret_20"] * -3.0 + 0.5
        position_reshaped = structbuild._reshape_label_frame(base.copy())
        assert structbuild._signature(_result(base.copy())) == structbuild._signature(
            _result(value_perturbed)
        ), "值改写不应改变特征签名"
        assert structbuild._signature(_result(base.copy())) == structbuild._signature(
            _result(position_reshaped)
        ), "缺失位置重排不应改变特征签名"

    def test_finite_source_mask_excludes_inf(self):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        frame = pd.DataFrame(
            {
                "ts_code": ["000000.SZ", "000001.SZ", "000002.SZ"],
                "ocf_to_revenue": [0.5, float("inf"), None],
            }
        )
        mask = structbuild.finite_source_mask(
            {"frames": [("20200101", frame)]}, ["ocf_to_revenue"]
        )
        assert mask["per_frame"]["20200101"]["finite_rows"] == 1, "inf 不算有值"

    def test_counterfactual_requires_days(self):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        with pytest.raises(structbuild.BuildError):
            structbuild.counterfactual()


class TestFiniteSourceEvidence:
    def test_notna_would_count_inf_but_finite_mask_does_not(self):
        frame = pd.DataFrame({"head_000": [0.5, float("inf"), None]})
        assert int(frame["head_000"].notna().sum()) == 2
        assert probes._mask_counts(frame, ["head_000"])["known_n"] == 1

    def test_absolute_amount_only_is_not_b_source(self):
        frame = pd.DataFrame(
            {
                "ocf": [100.0, None, 200.0],
                "ocf_to_revenue": [None, 0.3, None],
                "fcf_yield": [None, 0.1, None],
            }
        )
        result = probes._absolute_only(frame, ["ocf_to_revenue", "fcf_yield"], ["ocf"])
        assert result["n"] == 2, "只有 ocf 绝对额的行不算 B 来源"

    def test_source_evidence_ignores_inf_and_absolute_only_counted(self):
        routes = {24008: {"F": ["head_000"], "R": ["head_000"], "B": []}}
        frame = pd.DataFrame(
            {
                "head_000": [0.5, float("inf"), None],
                "ocf": [10.0, None, 20.0],
                "neu_y_ret_20": [0.1, 0.1, 0.1],
            }
        )
        result = probes.source_evidence(frame, 24008, routes, list(frame.columns))
        assert result["F"]["known_n"] == 1

    def test_b_policy_mentions_absolute_amount_rule(self):
        payload = spec.SOURCE_EVIDENCE["B"]
        assert "cf_sales" in payload["raw_columns"]
        assert "ocf" not in payload["raw_columns"]
        assert payload["absolute_amount_columns"] == ["ocf", "fcf"]


class TestDeployPurification:
    def test_last_sample_before_purges_21_days(self, tmp_path: Path):
        days = _clock_fixture(tmp_path)
        # es_end=days[300]、校准始于 days[301] ⇒ 最后可用信号日 = days[279]，剔除 21 日
        result = fold_clock.last_sample_before(days[300], days[301], days, 21)
        assert result["last_sample"] == days[279]
        assert result["purged_n"] == 21

    def test_spec_freezes_deploy_purification_expectation(self):
        assert spec.SPLIT_CLOCK["deploy_es_last_usable_expected"] == "20250730"
        assert spec.SPLIT_CLOCK["deploy_es_purged_expected"] == 21

    def test_deploy_row_carries_purification_fields(self, tmp_path: Path):
        days = _clock_fixture(tmp_path)
        rows = []
        for index, version in enumerate((*spec.FOLD_VERSIONS, spec.DEPLOY_VERSION)):
            es_end = days[300] if version == spec.DEPLOY_VERSION else days[100]
            rows.append(
                {
                    "model_version": version,
                    "timestamp": "t",
                    "start_date": days[0],
                    "end_date": days[340],
                    "val_start_date": days[80],
                    "val_end_date": es_end,
                    "val_calib_start_date": (
                        days[301] if version == spec.DEPLOY_VERSION else days[101]
                    ),
                    "val_calib_end_date": days[330],
                    "test_start_date": days[340],
                    "test_end_date": days[450],
                    "split_index": index,
                    "best_iteration": 100,
                }
            )
        log_dir = tmp_path / "data/models/stock_selection"
        log_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(log_dir / "ml_train_runs.csv", index=False, encoding="utf-8")

        deploy = fold_clock.deploy_row(tmp_path)
        assert deploy["es_last_usable_sample"] == days[279]
        assert deploy["es_purged_samples"] == 21
        assert deploy["es_purification_status"] == "purified"
        assert deploy["es_purification_breach"] == ""


# ── ⑭ 段缓存路由列覆盖断言（2026-11-22 share_float 整列消失事故防复发）──


class TestRouteColumnsCoverage:
    """scoring.assert_route_columns_coverage 的 fail-closed 行为。"""

    @staticmethod
    def _routes() -> dict:
        return {24008: {"F": ["col_f"], "R": ["col_r"], "B": ["col_b"]}}

    @staticmethod
    def _write_cache(tmp_path: Path, columns: list[str], leaf_missing: bool = False) -> Path:
        """两个缓存叶各写一个单帧段（leaf_missing 时 history 叶缺列）。"""
        digest = spec.spec_sha256()[:16]
        for leaf in ("structural_full", "structural_history"):
            seg = tmp_path / "cache" / digest / leaf / "20200101_20200101"
            seg.mkdir(parents=True)
            frame = pd.DataFrame(
                {"trade_date": ["20200101"], **{col: [1.0] for col in columns}}
            )
            if leaf_missing and leaf == "structural_history":
                frame = frame.drop(columns=["col_r"])
            frame.to_parquet(seg / "20200101.parquet", index=False)
        return tmp_path

    def test_full_coverage_passes(self, tmp_path: Path):
        camp = self._write_cache(tmp_path, ["col_f", "col_r", "col_b"])
        read = scoring.assert_route_columns_coverage(camp, self._routes(), lambda _msg: None)
        assert read == {"required_columns": 3, "files_checked": 2}

    def test_missing_column_raises(self, tmp_path: Path):
        camp = self._write_cache(tmp_path, ["col_f", "col_r", "col_b"], leaf_missing=True)
        with pytest.raises(scoring.ScoringError, match="缺列"):
            scoring.assert_route_columns_coverage(camp, self._routes(), lambda _msg: None)

    def test_missing_cache_raises(self, tmp_path: Path):
        with pytest.raises(scoring.ScoringError, match="结构缓存不存在"):
            scoring.assert_route_columns_coverage(tmp_path, self._routes(), lambda _msg: None)

    def test_backup_digest_dirs_are_ignored(self, tmp_path: Path):
        """cache/ 下的备份目录（旧代际产物）不参与断言（2026-11-22 a_007 误扫教训）。"""
        camp = self._write_cache(tmp_path, ["col_f", "col_r", "col_b"])
        digest = spec.spec_sha256()[:16]
        # 备份目录：不同代际名 + 缺列文件
        bak = camp / "cache" / f"{digest}_bak_20261122" / "structural_history" / "old_seg"
        bak.mkdir(parents=True)
        pd.DataFrame({"trade_date": ["20200101"]}).to_parquet(
            bak / "20200101.parquet", index=False
        )
        read = scoring.assert_route_columns_coverage(camp, self._routes(), lambda _msg: None)
        assert read["files_checked"] == 2

    def test_prod_intersection_allows_historical_evolution(self, tmp_path: Path):
        """段帧缺「生产同日也缺」的路由列属既定演化，不报错（2026-11-23 口径修订）。"""
        camp = self._write_cache(tmp_path, ["col_f", "col_r", "col_b"], leaf_missing=False)
        digest = spec.spec_sha256()[:16]
        # 段帧（历史日 20120101）缺 col_b（早期无筹码类数据源）
        seg = camp / "cache" / digest / "structural_full" / "20200101_20200101"
        frame = pd.read_parquet(seg / "20200101.parquet").drop(columns=["col_b"])
        frame.to_parquet(seg / "20120101.parquet", index=False)
        (seg / "20200101.parquet").unlink()
        # 生产 20120101 分区同样无 col_b（数据起点演化）
        prod = tmp_path / "data" / "features" / "cs_train"
        prod.mkdir(parents=True)
        pd.DataFrame({"trade_date": ["20120101"]}).to_parquet(
            prod / "20120101.parquet", index=False
        )
        read = scoring.assert_route_columns_coverage(
            camp, self._routes(), lambda _m: None, prod_root=tmp_path
        )
        assert read["files_checked"] == 2

    def test_prod_intersected_missing_still_raises(self, tmp_path: Path):
        """段帧缺「生产存在」的路由列仍 fail-closed（整列消失缺陷主判据）。"""
        camp = self._write_cache(tmp_path, ["col_f", "col_r", "col_b"])
        # full 叶段帧改为缺 col_b（模拟整列消失）
        seg = camp / "cache" / spec.spec_sha256()[:16] / "structural_full" / "20200101_20200101"
        frame = pd.read_parquet(seg / "20200101.parquet").drop(columns=["col_b"])
        frame.to_parquet(seg / "20200101.parquet", index=False)
        prod = tmp_path / "data" / "features" / "cs_train"
        prod.mkdir(parents=True)
        pd.DataFrame(
            {"trade_date": ["20200101"], "col_b": [1.0]}
        ).to_parquet(prod / "20200101.parquet", index=False)
        with pytest.raises(scoring.ScoringError, match="缺列"):
            scoring.assert_route_columns_coverage(
                camp, self._routes(), lambda _m: None, prod_root=tmp_path
            )


class TestShareFloatFullLookbackAnchor:
    """share_float 加载起点契约锚：段捕获窗命不中年分区导致整列消失的事故修复。"""

    def test_pit_full_lookback_constant(self):
        from src.lazybull.data.loader_announcement import ANNOUNCEMENT_PIT_LOOKBACK_START

        assert ANNOUNCEMENT_PIT_LOOKBACK_START == "20100101"

    def test_pipeline_share_float_uses_full_lookback(self):
        import inspect

        from src.lazybull.features import pipeline

        source = inspect.getsource(pipeline)
        # share_float 与 pledge 同口径：从 ANNOUNCEMENT_PIT_LOOKBACK_START 起加载
        # （若回退为 start_dt 预热窗，段缓存将再次出现解禁列整列消失）
        assert source.count("ANNOUNCEMENT_PIT_LOOKBACK_START") >= 2
        assert (
            "loader.load_share_float(\n            ANNOUNCEMENT_PIT_LOOKBACK_START,"
            in source
        )

    def test_pipeline_lookup_none_fallbacks(self):
        """公告/北向 lookup 为 None 时兜底空 dict（handler 走 NaN 占位列分支）。

        数据起点晚于段捕获窗（北向 2014-11、筹码 2018）时窗内无分区文件，
        加载返回 None ⇒ handler 被跳过 ⇒ 列整列消失（2026-11-23 事故第三例）。
        """
        import inspect

        from src.lazybull.features import pipeline

        source = inspect.getsource(pipeline)
        for anchor in (
            "north_flow_lookup = {}",
            "pledge_lookup = {}",
            "share_float_lookup = {}",
            "block_trade_lookup = {}",
        ):
            assert anchor in source, f"缺少兜底锚：{anchor}"


def _write_key_fixture(tmp_path: Path) -> Path:
    day_dir = tmp_path / "structural"
    day_dir.mkdir()
    cs_dir = tmp_path / "data/features/cs_train"
    cs_dir.mkdir(parents=True)
    for date, extras in (("20200101", ["600783.SH"]), ("20200102", ["600783.SH", "300001.SZ"])):
        base_codes = [f"00000{i}.SZ" for i in range(5)]
        pd.DataFrame({"ts_code": [*base_codes, *extras]}).to_parquet(day_dir / f"{date}.parquet")
        pd.DataFrame({"ts_code": base_codes}).to_parquet(cs_dir / f"{date}.parquet")
    return day_dir


class TestKeyControlRecount:
    def test_key_control_sums_per_day_and_sha_rule(self, tmp_path: Path):
        import hashlib

        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        day_dir = _write_key_fixture(tmp_path)
        result = structbuild.key_control_from_files(day_dir, tmp_path)
        assert result["totals"]["only_structural"] == 3
        assert result["totals"]["only_cs_train"] == 0
        first = result["days_sample"][0]
        # 键摘要口径：升序 + 每行一个代码 + 末尾换行
        expected = hashlib.sha256(
            ("\n".join([f"00000{i}.SZ" for i in range(5)] + ["600783.SH"]) + "\n").encode("utf-8")
        ).hexdigest()
        assert first["structural_keys_sha256"] == expected

    def test_missing_day_raises(self, tmp_path: Path):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        day_dir = _write_key_fixture(tmp_path)
        (tmp_path / "data/features/cs_train" / "20200102.parquet").unlink()
        with pytest.raises(structbuild.BuildError, match="旧训练域缺当日分区"):
            structbuild.key_control_from_files(day_dir, tmp_path)

    def test_rebuild_corrects_warmup_and_digest(self, tmp_path: Path):
        import json

        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        day_dir = _write_key_fixture(tmp_path)
        cal_dir = tmp_path / "data/clean"
        cal_dir.mkdir(parents=True, exist_ok=True)
        days = [f"2020010{index}" for index in range(1, 9)]
        pd.DataFrame({"cal_date": days, "is_open": [1] * len(days)}).to_parquet(
            cal_dir / "trade_cal.parquet", index=False
        )
        original = {
            "schema": "p7_cashflow_split_structural_block/v2",
            "block": ["20200103", "20200104"],
            "capture_window": ["20200101", "20200104"],
            "computed_days": 2,
            "kept_written_days": 1,
            "warmup_trading_days": 148,
            "wall_seconds": 10.0,
            "key_control": {"total_only_structural": 3},
        }
        original_path = tmp_path / "structural_block_hist_2019.json"
        original_path.write_text(json.dumps(original), encoding="utf-8")

        corrected = structbuild.rebuild_block_measurement(original_path, day_dir, tmp_path)
        # 预热按捕获窗重算：20200101~20200104 = 4 交易日 - 1 输出日 = 3（修正 148）
        assert corrected["window_trading_days"] == 4
        assert corrected["warmup_trading_days"] == 3
        assert corrected["key_control"]["sum_matches_total"] is True
        assert structbuild.verify_measurement(corrected)
        # 重入（v3 → v3 重算）：兼容键保证 sum_matches_total 仍为 True
        (tmp_path / "v3.json").write_text(json.dumps(corrected), encoding="utf-8")
        rerun = structbuild.rebuild_block_measurement(tmp_path / "v3.json", day_dir, tmp_path)
        assert rerun["key_control"]["sum_matches_total"] is True
        assert rerun["warmup_trading_days"] == 3
        # 逐日相加与全块总数不一致 → 标记 False（run.py 会拒绝发布）
        original["key_control"]["total_only_structural"] = 5
        original_path.write_text(json.dumps(original), encoding="utf-8")
        mismatched = structbuild.rebuild_block_measurement(original_path, day_dir, tmp_path)
        assert mismatched["key_control"]["sum_matches_total"] is False


class TestMeasurementDigest:
    def test_digest_changes_with_content(self):
        from scripts.v2_p7.cashflow_sleeve_split import structbuild

        payload = {"a": 1}
        digest = structbuild.measurement_content_digest(payload)
        assert structbuild.verify_measurement({**payload, "content_digest": digest})
        assert not structbuild.verify_measurement({**payload, "content_digest": "0" * 64})
        assert not structbuild.verify_measurement({**payload, "a": 2, "content_digest": digest})

    def test_load_structural_measurement_rejects_bad_digest(self, tmp_path: Path):
        campaign = tmp_path / "campaign"
        (campaign / "measurements").mkdir(parents=True)
        payload = {"schema": "x", "wall_seconds": 1.0, "kept_written_days": 1}
        path = campaign / "measurements" / "structural_block_hist_2019.json"
        path.write_text(json.dumps({**payload, "content_digest": "bad"}), encoding="utf-8")
        with pytest.raises(budget.BudgetError, match="自校验失败"):
            budget.load_structural_measurement(campaign)


class TestBlockBudgetQuotes:
    def test_quotes_use_planned_blocks_and_warmup_repeat(self):
        measured = {
            "block": ["20190104", "20190401"],
            "wall_seconds": 495.6,
            "kept_written_days": 57,
            "warmup_trading_days": 145,
            "window_trading_days": 202,
            "planned_blocks": 97,
            "disk_bytes": 408440059,
            "rss_after_mb": 1554.4,
        }
        payload = budget.structural_measurement(measured)
        quotes = payload["quotes"]
        assert quotes["planned_blocks"] == 97
        expected_upper = 97 * (145 + 1838 / 97) * (495.6 / 202) / 3600
        assert abs(quotes["upper_hours"] - round(expected_upper, 2)) < 0.02
        expected_lower = 1838 * (495.6 / 57) / 3600
        assert abs(quotes["lower_hours"] - round(expected_lower, 2)) < 0.02
        # r0_003 复核示意口径：单大块摊薄 ≈ 4.44h
        assert 4.3 < quotes["lower_hours"] < 4.6
        assert quotes["upper_hours"] > quotes["lower_hours"]

    def test_missing_wall_is_unknown_not_zero(self):
        payload = budget.structural_measurement({"kept_written_days": 57})
        assert payload["status"] == "unknown"


class TestSingleDayProbeConsumption:
    def test_probe_day_uses_structural_frame_and_reports_three_layers(self, tmp_path: Path):
        clean = tmp_path / "data/clean/daily"
        clean.mkdir(parents=True)
        mother = pd.DataFrame(
            {
                "ts_code": ["000000.SZ"],
                "list_days": [400],
                "is_st": [0],
                "is_suspended": [0],
                "tradable": [1],
            }
        )
        mother.to_parquet(clean / "20200101.parquet", index=False)
        basic = tmp_path / "data/clean/stock_basic.parquet"
        basic.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ts_code": ["000000.SZ"], "market": ["主板"]}).to_parquet(basic, index=False)
        features = pd.DataFrame(
            {"ts_code": ["000000.SZ"], "amount_ma20": [60000.0], "total_mv": [800000.0]}
        )
        record = probes.probe_day(
            tmp_path, "20200101", features_frame=features, features_label="structural_test"
        )
        assert record["feature_source"] == "structural_test"
        assert record["structural_domain_observed_n"] == 1
        assert record["old_domain_observed_n"] is None
        assert record["scored_n"] is None
        assert record["scored_n_status"] == "not_estimable_at_checkpoint_1"

    def test_industry_alignment_flags_mismatch(self, tmp_path: Path):
        _pit_fixture(tmp_path)
        # 特征帧写生产快照的旧值（801890.SI 机械设备），PIT 当日是非银金融 → mismatch
        features = pd.DataFrame(
            {"ts_code": ["600783.SH"], "sw_l1_code": ["801890.SI"]}
        )
        alignment = probes._industry_alignment(features, None, "20190401", tmp_path)
        assert alignment["status"] == "mismatch"
        assert alignment["mismatch_examples"][0]["expected"] == "801790.SI"

    def test_industry_alignment_passes_when_injected(self, tmp_path: Path):
        _pit_fixture(tmp_path)
        features = pd.DataFrame(
            {"ts_code": ["600783.SH"], "sw_l1_code": ["801790.SI"]}
        )
        alignment = probes._industry_alignment(features, None, "20190401", tmp_path)
        assert alignment["status"] == "ok"
        assert alignment["identity_probe"]["feature_match"] is True
