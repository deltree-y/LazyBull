# -*- coding: utf-8 -*-
"""run-a 定向测试（审点二；全合成夹具，不触生产数据）。

覆盖协议 §9 已知回归风险的 run-a 切面：
- 折缝换算（Δ=21 显式缝与生产编排器同源）与 fail-fast；
- 行业变更日分段（非交易日吸附）与逐日行业核验；
- 打分并列决胜（代码升序）与有效宽度；B 绝对额不算来源的负例沿用审点一口径；
- 账户组件（BaselineInputs 装配、本金换算、resume 守卫）；
- §5.2-6 小夹具：训练后直接预测 → 隔离注册 → 重载 → 实际信号入口；
- 读数组件（链式指标、B 终值门槛、R 双门、块自举确定性）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.cashflow_sleeve_split import (
    accounts,
    artifacts,
    deliver_seal,
    metrics,
    run_a,
    scoring,
    training,
)


def test_delivery_seals_the_final_batch_state(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "a_test"
    work = run_dir / "work"
    work.mkdir(parents=True)
    (work / "a_review.md").write_text("fixture\n", encoding="utf-8")
    state = {
        "parent": "r0_test",
        "parent_manifest_sha256": "ab" * 32,
        "batches_done": list(run_a.BATCHES[:-1]),
    }
    ctx = SimpleNamespace(
        base=tmp_path, run_dir=run_dir, work=work, state=state, emit=lambda message: None,
        extra_roots=[str(tmp_path)],
    )
    monkeypatch.setitem(run_a._BATCH_DISPATCH, "deliver", lambda ctx: {"gates": {}})
    monkeypatch.setattr(deliver_seal, "PUBLISH_MAP", (("a_review.md", "a_review.md"),))

    run_a._run_batch("deliver", ctx, time.perf_counter())

    manifest = artifacts.require_published(tmp_path, "a_test", "run-a")
    payload = manifest["payload"]
    assert payload["batches_done"] == list(run_a.BATCHES)
    assert payload["parent_manifest_sha256"] == "ab" * 32
    actual_sha = run_a.sha256_file(run_a._state_path(run_dir))
    assert payload["work_digests"][run_a.STATE_NAME] == actual_sha


# ── checkpoint-2 修复：es 净化 / PIT 叠加 / 压力费率 / 未定日两端测算 ────


class TestEsMaturityPurge:
    def test_purge_trims_es_tail_and_keeps_calib(self):
        """冻结可用日截断 es 尾部；校准/embargo 不变；统计回写。"""
        import importlib

        prepare = importlib.import_module("src.lazybull.v2.train.core.prepare")
        days = [f"202501{index:02d}" for index in range(1, 25)]
        df = pd.DataFrame({"trade_date": days, "ts_code": "000001.SZ", "value": range(len(days))})
        with training.es_maturity_purge("20250110") as state:
            es, calib, embargo, stats = prepare.split_val_for_selection_protocol_by_date(
                df, embargo_days=4
            )
        assert es["trade_date"].max() <= "20250110"
        assert stats["val_es_end_date"] == "20250110"
        assert stats["val_es_purged_rows"] > 0
        assert state["stats"]["val_es_end_date"] == "20250110"
        assert all(len(value) == 64 for value in state["keys"].values())
        # 校准/embargo 与原始拆分一致（剔除行绝不并入校准）
        assert calib["trade_date"].min() >= "20250110"

    def test_assert_fails_on_untouched_split(self):
        with pytest.raises(training.TrainingError, match="早停净化"):
            training.assert_es_purged({"stats": None}, "20250110", lambda message: None)


def test_training_input_digests_cover_order_labels_and_weights():
    rows = pd.DataFrame(
        {"trade_date": ["20240102", "20250102"], "ts_code": ["000001.SZ", "000002.SZ"],
         "target": [0.1, 0.2]}
    )
    args = SimpleNamespace(rank_weight_enabled=False, time_decay_half_life=1.0)
    result = {
        "label_column": "target",
        "df_train_split": rows,
        "df_val_split": rows.iloc[:1],
        "df_val_split_original": rows.iloc[1:],
    }
    original = training._training_input_digests(result, args)
    reversed_train = training._training_input_digests(
        {**result, "df_train_split": rows.iloc[::-1]}, args
    )
    revised_val = training._training_input_digests(
        {**result, "df_val_split": rows.iloc[:1].assign(target=0.9)}, args
    )
    args.time_decay_half_life = 2.0
    reweighted = training._training_input_digests(result, args)
    assert reversed_train["train"]["keys_sha256"] != original["train"]["keys_sha256"]
    original_es_hash = original["early_stopping"]["labels_sha256"]
    assert revised_val["early_stopping"]["labels_sha256"] != original_es_hash
    assert reweighted["train"]["weights_sha256"] != original["train"]["weights_sha256"]
    with pytest.raises(training.TrainingError, match="样本、标签或权重"):
        training._assert_peer_inputs(
            {"version": 24008, "input_digests": original}, {"input_digests": reweighted}, "R"
        )


def test_delayed_buy_and_completion_share_actual_position_capacity(monkeypatch):
    from scripts.v2_p7.cashflow_sleeve_split.account_engine import CashflowAccountEngine
    from scripts.v2_p7.operating_monthly_baseline import OperatingBaselineEngine

    fills = []

    def buy(engine, date, stock, target_value, signal_date=None, requested_shares=None):
        fills.append((stock, requested_shares))
        engine.positions.setdefault(stock, {})

    monkeypatch.setattr(OperatingBaselineEngine, "_buy_stock_direct", buy)
    engine = object.__new__(CashflowAccountEngine)
    engine.signal = SimpleNamespace(top_n=20)
    engine.positions = {f"{key:06d}.SZ": {} for key in range(1, 21)}
    engine.position_limit_rejections = []
    del engine.positions["000020.SZ"]
    date = pd.Timestamp("2025-01-13")
    engine._buy_stock_direct(date, "600551.SH", 54000)
    engine._buy_stock_direct(date, "600496.SH", 48000)
    engine._buy_stock_direct(date, "600551.SH", 54000, requested_shares=100)

    assert fills == [("600551.SH", None), ("600551.SH", 100)]
    assert len(engine.positions) == 20
    assert engine.position_limit_rejections[0]["ts_code"] == "600496.SH"
    assert engine.position_limit_rejections[0]["reason"] == "position_capacity_full"


class TestOverlayPitFrame:
    def _day_frame(self, day: str, codes: Sequence[str], industry: float) -> pd.DataFrame:
        rows = []
        for code in codes:
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "ret_1": 0.01,
                    "neu_y_ret_20": industry + 0.001,
                    "neu_ret_20": industry,
                    "sw_l1_code": "110000000",
                }
            )
        return pd.DataFrame(rows)

    def test_overlay_replaces_and_counts(self, tmp_path: Path):
        days = ["20220104", "20220105"]
        index = {}
        for day in days:
            path = tmp_path / f"{day}.parquet"
            self._day_frame(day, ["000001.SZ", "000002.SZ"], 0.5).to_parquet(path)
            index[day] = path

        class _Frame:
            pass

        frame = _Frame()
        frame.df = pd.DataFrame(
            {
                "trade_date": ["20220104", "20220104", "20220105"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
                "ret_1": [0.01, 0.01, 0.01],
                "neu_y_ret_20": [0.9, 0.9, 0.9],
                "neu_ret_20": [0.8, 0.8, 0.8],
            }
        )
        emit = []
        summary = training.overlay_pit_frame(
            frame,
            index,
            "20220104",
            "20220105",
            label_columns=("neu_y_ret_20",),
            emit=emit.append,
        )
        # 标签与行业列被 PIT 值覆写；原始列不变
        assert frame.df["neu_y_ret_20"].tolist() == [0.501, 0.501, 0.501]
        assert frame.df["neu_ret_20"].tolist() == [0.5, 0.5, 0.5]
        assert frame.df["ret_1"].tolist() == [0.01, 0.01, 0.01]
        assert summary["changed_by_column"]["neu_y_ret_20"] == 3
        assert summary["changed_by_column"]["neu_ret_20"] == 3

    def test_overlay_fails_closed_on_missing_day(self, tmp_path: Path):
        index = {}

        class _Frame:
            pass

        frame = _Frame()
        frame.df = pd.DataFrame(
            {"trade_date": ["20220104"], "ts_code": ["000001.SZ"], "neu_y_ret_20": [0.9]}
        )
        with pytest.raises(training.TrainingError, match="rebuild-history"):
            training.overlay_pit_frame(
                frame,
                index,
                "20220104",
                "20220104",
                label_columns=("neu_y_ret_20",),
                emit=lambda message: None,
            )

    def test_overlay_pit_nan_overrides_and_missing_row_kept(self, tmp_path: Path):
        """行存在于 PIT：标签 NaN 一并采用（与评分域同口径）；整行缺失保留生产值。"""
        day_frame = self._day_frame("20220104", ["000001.SZ", "000002.SZ"], 0.5)
        # 000001.SZ 的 PIT 标签为 NaN（评分域同口径）
        day_frame.loc[day_frame["ts_code"] == "000001.SZ", "neu_y_ret_20"] = float("nan")
        path = tmp_path / "20220104.parquet"
        day_frame.to_parquet(path)

        class _Frame:
            pass

        frame = _Frame()
        frame.df = pd.DataFrame(
            {
                "trade_date": ["20220104", "20220104", "20220104"],
                "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
                "neu_y_ret_20": [0.9, 0.9, 0.9],
            }
        )
        summary = training.overlay_pit_frame(
            frame,
            {"20220104": path},
            "20220104",
            "20220104",
            label_columns=("neu_y_ret_20",),
            emit=lambda message: None,
        )
        # 行在 PIT：000001.SZ 标签被覆写为 NaN（下游 dropna 落样本）
        assert pd.isna(frame.df.loc[0, "neu_y_ret_20"])
        # 000002.SZ 在 PIT 有值（0.501）但帧里没有该键？——有：覆写为 PIT 值
        assert frame.df.loc[1, "neu_y_ret_20"] == pytest.approx(0.501)
        # 600000.SH 整行不在 PIT：保留生产值
        assert frame.df.loc[2, "neu_y_ret_20"] == 0.9
        assert summary["rows_missing_in_pit"] == 1


class TestCheckpoint2Fixes:
    def test_pit_day_index_prefers_structural_full(self, tmp_path: Path):
        """重叠日（full/history 两根都有）以 structural_full 为准（checkpoint-2 第 2 项）。"""
        full = scoring.cache_root(tmp_path)
        history = scoring.cache_root(tmp_path, history=True)
        backup = history.parent.with_name(f"{history.parent.name}_bak") / history.name
        for leaf, days in (
            (full, ["20120104", "20120105"]),
            (history, ["20120104", "20130104"]),
            (backup, ["20120103"]),
        ):
            seg = scoring.plan_segments(days, [])[0]
            seg_dir = leaf / f"{seg['first']}_{seg['last']}"
            seg_dir.mkdir(parents=True)
            for day in days:
                (seg_dir / f"{day}.parquet").write_bytes(day.encode())
            scoring._write_segment_manifest(seg_dir, seg, [])
        unfinished = history / "20130105_20130105"
        unfinished.mkdir()
        (unfinished / "20130105.parquet").write_bytes(b"x")
        index = training.build_pit_day_index(tmp_path)
        assert index["20120104"].parts[-3] == "structural_full"
        assert index["20130104"].parts[-3] == "structural_history"
        assert "20120103" not in index
        assert "20130105" not in index

    def test_stress_cost_keeps_base_stamp_tax(self):
        """协议 §7.6：压力臂只提滑点；印花税维持基准 5bp（checkpoint-2 第 5 项）。"""
        from src.lazybull.common.cost import CostModel

        assert tuple(run_a.STRESS_COST_KWARGS) == (0.0001954, 5.0, 0.0005, 0.0010)
        cost = CostModel(*run_a.STRESS_COST_KWARGS)
        assert cost.stamp_tax == 0.0005
        assert cost.slippage == 0.0010

    def test_undated_bounds_from_trades(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        for role in ("R100", "F100"):
            acc = run_dir / "accounts" / "1000000" / role
            acc.mkdir(parents=True)
            rows = []
            if role == "R100":
                rows.append(
                    {"stock": "600602.SH", "action": "buy", "shares": 7600, "date": "2021-10-08"}
                )
            (acc / "trades.json").write_text(json.dumps(rows), encoding="utf-8")
        work = run_dir / "work" / "corporate_resolved"
        work.mkdir(parents=True)
        (work / "undated_audit.json").write_text(
            json.dumps({"records": [{"ts_code": "600602.SH"}]}), encoding="utf-8"
        )
        bounds = deliver_seal.undated_bounds(run_dir)
        assert set(bounds) == {"R100@1000000"}
        assert bounds["R100@1000000"] == {"max_shares": 7600, "upper_bound_yuan": 60800.0}

    def test_ending_value_uses_total_value(self):
        rows = [
            {"trade_date": "20190329", "nav": 1.0, "total_value": 1000000.0},
            {"trade_date": "20261029", "nav": 1.9807, "total_value": 1980739.73},
        ]
        result = metrics.account_metrics(rows, "F100@1000000")
        assert result["ending_value"] == pytest.approx(1980739.73)
        assert result["ending_nav"] == pytest.approx(1.9807)


# ── 折缝换算 ──────────────────────────────────────────────────────────


class TestFoldSeams:
    def _calendar(self, start: str, count: int) -> list[str]:
        base = pd.Timestamp(start)
        days = []
        while len(days) < count:
            if base.weekday() < 5:
                days.append(base.strftime("%Y%m%d"))
            base += pd.Timedelta(days=1)
        return days

    def test_seams_follow_production_rule(self):
        days = self._calendar("20120101", 300)
        fold = {"outer_train_start": days[0], "outer_train_end": days[200], "es_start": None}
        seams = training.resolve_fold_seams(fold, days)
        # 生产规则：n_val=ceil(n*0.2)、train_end=D[n_train-Δ-1]、val_start=D[n_train+Δ]
        last_outer = days.index(days[200]) + 1
        window = days[0:last_outer]
        n = len(window)
        n_val = int(np.ceil(n * 0.2))
        n_train = n - n_val
        assert seams["val_start"] == window[n_train + 21]
        assert seams["train_end"] == window[n_train - 21 - 1]
        assert seams["train_start"] == window[0]
        assert seams["val_end"] == window[-1]

    def test_seam_mismatch_is_fail_fast(self):
        days = self._calendar("20120101", 300)
        fold = {
            "outer_train_start": days[0],
            "outer_train_end": days[200],
            "es_start": days[50],  # 与换算必然不符
        }
        with pytest.raises(training.TrainingError, match="显式缝与日志不符"):
            training.resolve_fold_seams(fold, days)

    def test_train_end_maturity_is_before_es_start(self):
        from scripts.v2_p7.cashflow_sleeve_split import fold_clock

        days = self._calendar("20120101", 300)
        # 先用空 es_start 换算出生产口径缝，再回填为日志值（自洽夹具）
        base = {"outer_train_start": days[0], "outer_train_end": days[200], "es_start": None}
        seams = training.resolve_fold_seams(base, days)
        fold = {**base, "es_start": seams["val_start"]}
        seams = training.resolve_fold_seams(fold, days)
        maturity = fold_clock.maturity_day(seams["train_end"], days, 21)
        assert maturity < seams["val_start"]


# ── 行业变更日分段 ────────────────────────────────────────────────────


class TestIndustrySegments:
    def _pit(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code6": ["000001", "000001", "000002"],
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "in_date": ["20180101", "20200115", "20180101"],
                "out_date": ["20200114", None, None],
                "ind_code": ["I1", "I2", "I1"],
                "l1_code": ["110000", "220000", "110000"],
                "l2_code": ["A1", "B1", "A1"],
                "l3_code": ["x", "y", "x"],
                "l1_name": ["甲", "乙", "甲"],
                "l2_name": ["A", "B", "A"],
                "l3_name": ["x", "y", "x"],
                "index_code": ["801010.SI", "801030.SI", "801010.SI"],
                "sw_version": ["SW2014", "SW2014", "SW2014"],
                "source_updated_at": [None, None, None],
                "code_alias": ["a", "b", "c"],
            }
        )

    def test_change_dates_and_snapping(self, tmp_path: Path):
        days = [f"2020010{index}" for index in range(1, 10)]  # 20200101..20200109
        clean = tmp_path / "data/clean"
        clean.mkdir(parents=True, exist_ok=True)
        pit_path = clean / "sw_industry_pit.parquet"
        pit = self._pit()
        pit.loc[1, "in_date"] = "20200105"  # 交易日变更
        pit.to_parquet(pit_path)
        # stock_basic：两只股票都在窗前已上市 >365 天（首区间不属于晚覆盖）
        pd.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "list_date": ["19900101", "19900101"]}
        ).to_parquet(clean / "stock_basic.parquet")
        changes = scoring.industry_change_dates(tmp_path, days)
        assert changes == ["20200105"]
        # 非交易日变更 → 吸附下一交易日
        pit.loc[1, "in_date"] = "20200104"  # 周六
        pit.to_parquet(pit_path)
        calendar = [d for d in pd.date_range("2020-01-01", periods=12, freq="D").strftime("%Y%m%d")]
        calendar = [d for d in calendar if pd.Timestamp(d).weekday() < 5][:9]
        changes = scoring.industry_change_dates(tmp_path, calendar)
        assert changes[0] == "20200106"

    def test_late_coverage_boundary(self, tmp_path: Path):
        """老股票 PIT 覆盖窗内才出现（代码复用）⇒ 首现日为分块边界。"""
        clean = tmp_path / "data/clean"
        clean.mkdir(parents=True, exist_ok=True)
        pit = self._pit()
        pit.loc[0, "in_date"] = "20200108"  # 000001.SZ 首区间（窗内才出现）
        pit.to_parquet(clean / "sw_industry_pit.parquet")
        pd.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "list_date": ["19940928", "19900101"]}
        ).to_parquet(clean / "stock_basic.parquet")
        days = [f"2020010{index}" for index in range(1, 10)]
        changes = scoring.industry_change_dates(tmp_path, days)
        assert changes == ["20200108"]

    def test_historical_holiday_splits_capture_within_existing_cache_segment(self, tmp_path: Path):
        clean = tmp_path / "data/clean"
        clean.mkdir(parents=True)
        pit = self._pit()
        pit["in_date"] = ["20100101", "20130101", "20100101"]
        pit.to_parquet(clean / "sw_industry_pit.parquet")
        pd.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "list_date": ["19900101", "19900101"]}
        ).to_parquet(clean / "stock_basic.parquet")
        days = ["20121228", "20121231", "20130104", "20130107"]
        cache_cuts = scoring.industry_change_dates(tmp_path, days, snap_from="20190401")
        capture_cuts = scoring.industry_change_dates(tmp_path, days)
        assert [seg["days"] for seg in scoring.plan_segments(days, cache_cuts)] == [days]
        assert [seg["days"] for seg in scoring.plan_segments(days, capture_cuts)] == [
            days[:2], days[2:]
        ]

    def test_segments_split_at_changes(self):
        days = [f"2020010{index}" for index in range(1, 9)]
        segments = scoring.plan_segments(days, ["20200104"])
        assert [(seg["first"], seg["last"]) for seg in segments] == [
            ("20200101", "20200103"),
            ("20200104", "20200108"),
        ]
        assert sum(seg["n"] for seg in segments) == len(days)


class TestLoadDayFrame:
    """load_day_frame 段目录定位（回归：glob 生成器不可用 / 拼接路径）。"""

    def _base(self, tmp_path: Path) -> Path:
        return scoring.cache_root(tmp_path)

    def test_locates_segment_parquet(self, tmp_path: Path):
        seg_dir = self._base(tmp_path) / "20191010_20200221"
        seg_dir.mkdir(parents=True)
        frame = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [1.0]})
        frame.to_parquet(seg_dir / "20200108.parquet")
        loaded = scoring.load_day_frame(tmp_path, "20200108")
        assert list(loaded["ts_code"]) == ["000001.SZ"]

    def test_missing_day_raises(self, tmp_path: Path):
        self._base(tmp_path).mkdir(parents=True)
        with pytest.raises(scoring.ScoringError, match="缺当日分区"):
            scoring.load_day_frame(tmp_path, "20200901")

    def test_duplicate_day_raises(self, tmp_path: Path):
        base = self._base(tmp_path)
        for name in ("20191010_20200221", "20200224_20200430"):
            seg_dir = base / name
            seg_dir.mkdir(parents=True)
            pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(seg_dir / "20200108.parquet")
        with pytest.raises(scoring.ScoringError, match="重复"):
            scoring.load_day_frame(tmp_path, "20200108")


def test_segment_cache_requires_complete_manifest_and_all_hashes(tmp_path: Path):
    seg = scoring.plan_segments(["20200102", "20200103"], [])[0]
    for day in seg["days"]:
        (tmp_path / f"{day}.parquet").write_bytes(day.encode())

    assert scoring._segment_cache_status(tmp_path, seg) == "missing"
    scoring._write_segment_manifest(tmp_path, seg, [])
    assert scoring._segment_cache_status(tmp_path, seg) == "verified"

    manifest_path = tmp_path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["20200103.parquet"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert scoring._segment_cache_status(tmp_path, seg) == "mismatch"

    scoring._write_segment_manifest(tmp_path, seg, [])
    (tmp_path / "20200103.parquet").write_bytes(b"different")
    assert scoring._segment_cache_status(tmp_path, seg) == "mismatch"


# ── 打分组件 ──────────────────────────────────────────────────────────


class _StubSignal:
    """打分夹具：固定分数表（含并列），验证决胜与有效宽度。"""

    def __init__(self, scores: dict[str, float], top_n: int = 20) -> None:
        self.scores = scores
        self.top_n = top_n
        self.min_amount_ma20 = 50000.0
        self.min_total_mv = 500000.0
        self.max_total_mv = 15000000.0
        self.exclude_financial = True
        self._FINANCIAL_SW_L1_CODES = ["801780", "801790"]

    def generate_ranked(self, date, universe, data):
        return [(code, self.scores[code]) for code in universe if code in self.scores]

    def _apply_selection_filters(self, features_df):
        return features_df, len(features_df), len(features_df)


class TestScoreDay:
    def _fixtures(self) -> dict:
        codes = [f"{index:06d}.SZ" for index in range(6)]
        mother = pd.DataFrame(
            {
                "ts_code": codes,
                "list_days": [400] * 6,
                "is_st": [0] * 6,
                "is_suspended": [0] * 6,
                "tradable": [1] * 6,
            }
        )
        basic = pd.DataFrame({"ts_code": codes, "market": ["主板"] * 6})
        industry = pd.DataFrame({"ts_code": codes, "l1_code": ["110000"] * 6})
        features = pd.DataFrame(
            {
                "ts_code": codes,
                "amount_ma20": [60000.0] * 6,
                "total_mv": [800000.0] * 6,
                "sw_l1_code": ["801010.SI"] * 6,
                "head_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        return mother, basic, industry, features

    def test_tie_break_by_code_ascending(self):
        mother, basic, industry, features = self._fixtures()
        codes = list(features["ts_code"])
        signal = {"F": _StubSignal({code: 1.0 for code in codes})}
        result = scoring.score_day(
            Path("."),
            features,
            "20200101",
            signal,
            basic,
            industry,
            mother,
            {},
            1,
            {"F": ["head_a"]},
        )
        ranked = result["arm_outputs"]["F"]
        # 同分并列：按代码升序
        assert [code for code, _ in ranked] == sorted(codes)
        assert all(rank[1] == 1.0 for rank in ranked)

    def test_effective_width_requires_real_source(self):
        mother, basic, industry, features = self._fixtures()
        codes = list(features["ts_code"])
        signal = {"F": _StubSignal({code: float(index) for index, code in enumerate(codes)})}
        result = scoring.score_day(
            Path("."),
            features,
            "20200101",
            signal,
            basic,
            industry,
            mother,
            {},
            1,
            {"F": ["head_a"]},
        )
        assert result["per_arm"]["F"]["effective_n"] == 6
        # 臂列全部缺失 ⇒ 有效宽度为 0（分数不算来源）
        result2 = scoring.score_day(
            Path("."),
            features,
            "20200101",
            signal,
            basic,
            industry,
            mother,
            {},
            1,
            {"F": ["no_such_column"]},
        )
        assert result2["per_arm"]["F"]["effective_n"] == 0
        assert result2["per_arm"]["F"]["ranked_rows"] == 6


# ── 账户组件 ──────────────────────────────────────────────────────────


class TestAccountComponents:
    def test_capital_for_role(self):
        assert accounts.capital_for_role("F100", 1_000_000) == 1_000_000
        assert accounts.capital_for_role("F80", 1_000_000) == 800_000
        assert accounts.capital_for_role("R80", 2_000_000) == 1_600_000

    def test_baseline_inputs_rejects_gap(self, tmp_path: Path):
        days = ["20200101", "20200102", "20200103"]
        ranked = {"20200101": [("000001.SZ", 1.0)], "20200103": [("000001.SZ", 1.0)]}
        with pytest.raises(accounts.AccountError, match="缺口"):
            accounts.build_baseline_inputs(ranked, days, "20191231", {})

    def test_baseline_inputs_allows_registered_empty_days(self, tmp_path: Path):
        """空排名日（score 台账登记）按生产语义落空单；未登记缺口仍 fail-closed。"""
        days = ["20200101", "20200102", "20200103"]
        ranked = {"20200101": [("000001.SZ", 1.0)], "20200103": [("000001.SZ", 1.0)]}
        # 未登记的缺口 → 拒绝
        with pytest.raises(accounts.AccountError, match="缺口"):
            accounts.build_baseline_inputs(ranked, days, "20191231", {}, allowed_empty=["20200105"])
        # 登记的空排名日 → 空单
        inputs = accounts.build_baseline_inputs(
            ranked, days, "20191231", {}, allowed_empty=["20200102"]
        )
        assert inputs.ranked["20200102"] == []
        assert inputs.ranked["20200101"] == [("000001.SZ", 1.0)]
        assert set(inputs.ranked) == set(days)
        # 登记为空却出现排名行 → 口径冲突
        ranked_conflict = {
            **ranked,
            "20200102": [("000001.SZ", 1.0)],
        }
        with pytest.raises(accounts.AccountError, match="口径冲突"):
            accounts.build_baseline_inputs(
                ranked_conflict, days, "20191231", {}, allowed_empty=["20200102"]
            )

    def test_baseline_inputs_rejects_out_of_window(self):
        days = ["20200101"]
        ranked = {"20200101": [("000001.SZ", 1.0)], "20191231": [("000001.SZ", 1.0)]}
        with pytest.raises(accounts.AccountError, match="超出执行窗口"):
            accounts.build_baseline_inputs(ranked, days, "20191231", {})

    def test_open_run_dir_refuses_published(self, tmp_path: Path):
        run_dir = tmp_path / "runs" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "PUBLISHED.json").write_text("{}", encoding="utf-8")
        with pytest.raises(Exception, match="已发布"):
            run_a._open_run_dir(tmp_path, "r1", tmp_path, "p", [])

    def test_open_run_dir_refuses_parent_mismatch(self, tmp_path: Path):
        work = tmp_path / "runs" / "r1" / "work"
        work.mkdir(parents=True)
        state = {
            "parent": "other",
            "package_digest": run_a._package_digest(
                run_a._code_identity(tmp_path)["package_files"]
            ),
        }
        (work / run_a.STATE_NAME).write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(Exception, match="parent"):
            run_a._open_run_dir(tmp_path, "r1", tmp_path, "prepare_x", [])

    def test_batch_code_identity_refuses_drift(self, tmp_path: Path):
        state = {"batch_digests": {"train-f": "deadbeef"}}
        with pytest.raises(Exception, match="代码身份"):
            run_a._assert_batch_code_identity(state, "train-f", tmp_path)
        # 未登记过身份的批次（含已完成批后的新批）不阻塞
        digest = run_a._assert_batch_code_identity({"batch_digests": {}}, "train-f", tmp_path)
        assert digest == run_a._package_digest(run_a._code_identity(tmp_path)["package_files"])

    def test_open_run_dir_refuses_code_drift(self, tmp_path: Path):
        work = tmp_path / "runs" / "r1x" / "work"
        work.mkdir(parents=True)
        state = {"parent": "prepare_x"}
        (work / run_a.STATE_NAME).write_text(json.dumps(state), encoding="utf-8")
        run_dir, resume = run_a._open_run_dir(
            tmp_path, "r1x", tmp_path, "prepare_x", [str(tmp_path)]
        )
        assert resume is True

    def test_open_run_dir_creates_fresh(self, tmp_path: Path):
        run_dir, resume = run_a._open_run_dir(
            tmp_path, "run_x1", tmp_path, "prepare_x", [str(tmp_path)]
        )
        assert resume is False
        assert (run_dir / "work").exists()


# ── §5.2-6 小夹具：训练后直接预测 → 隔离注册 → 重载 → 实际信号入口 ────


class TestRegisterReloadSignalFixture:
    def test_register_reload_signal(self, tmp_path: Path):
        from scripts.v2_p3.t5_arm_executors import _signal_selection_mapping
        from scripts.v2_p3.t05b_fingerprint import _configure_signal_selection
        from src.lazybull.v2.common.model_registry import ModelRegistry
        from src.lazybull.v2.core.signal.ml_signal import MLSignal
        from src.lazybull.v2.train.ensemble import EnsembleModel

        rng = np.random.default_rng(7)
        columns = [f"f{index}" for index in range(5)]
        X = pd.DataFrame(rng.normal(size=(80, 5)), columns=columns)
        y = pd.Series(rng.normal(size=80))
        sub_models = []
        for seed in (42, 61):
            model = __import__("xgboost").XGBRegressor(
                n_estimators=3, max_depth=2, random_state=seed
            )
            model.fit(X, y)
            sub_models.append(model)
        ensemble = EnsembleModel(sub_models)
        direct = ensemble.predict(X)

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        version = registry.register_model(
            model=ensemble,
            model_type="xgboost_regression_wf",
            train_start_date="20120101",
            train_end_date="20180101",
            feature_columns=columns,
            label_column="neu_y_ret_20",
            n_samples=80,
            train_params={"task": "regression", "freshness_strategy": "state_keep_event_decay"},
            performance_metrics={},
        )
        signal = MLSignal(
            top_n=20, model_version=int(version), models_dir=str(tmp_path / "models"), verbose=False
        )
        _configure_signal_selection(signal, _signal_selection_mapping())
        signal._load_model()
        assert list(signal.feature_columns) == columns
        frame = X.copy()
        frame["ts_code"] = [f"{index:06d}.SZ" for index in range(80)]
        frame["amount_ma20"] = 60000.0
        frame["total_mv"] = 800000.0
        universe = frame["ts_code"].tolist()
        ranked = signal.generate_ranked(pd.Timestamp("20200101"), universe, {"features": frame})
        # 信号入口分数 = 集成直接预测（逐位一致，含并列时按代码升序）
        by_code = {code: score for code, score in ranked}
        for index, code in enumerate(universe):
            assert by_code[code] == pytest.approx(float(direct[index]), abs=1e-12)


# ── 读数组件 ──────────────────────────────────────────────────────────


class TestMetricsComponents:
    def _daily(self, nav_values: list[float], start: str = "20190329") -> list[dict]:
        dates = pd.bdate_range(start, periods=len(nav_values)).strftime("%Y%m%d")
        return [
            {"trade_date": str(day), "nav": value, "total_value": value * 1_000_000}
            for day, value in zip(dates, nav_values)
        ]

    def test_chain_and_segments(self):
        rows = self._daily([1.0, 1.1, 0.99, 1.2, 0.9, 1.05])
        payload = metrics.account_metrics(rows, "x")
        assert payload["chain"]["cagr"] is not None
        assert payload["segments"]["full"]["mdd_loss"] == pytest.approx(
            -payload["chain"]["max_drawdown"]
        )
        # 分段继承基点：第二段起点权益 = 前段末行
        seg = payload["segments"]["s2019_2021"]
        assert seg["nav_points"] >= 2

    def test_chain_rejects_nonpositive_nav(self):
        rows = self._daily([1.0, 0.0, 1.1])
        with pytest.raises(metrics.MetricsError, match="非有限或非正"):
            metrics.chain_frame(rows, "x")

    def test_b_required_exact_line_and_gap(self):
        row = metrics.b_required_row(1_000_000, 1_000_000.0 * 1.25 ** (252 / 252), 253)
        assert row["b_required_ending_value"] == pytest.approx(0.0)
        assert row["status"] == "no_gap"
        row2 = metrics.b_required_row(1_000_000, 0.0, 1839)
        assert row2["status"] == "positive_gap"
        assert row2["b_required_cagr_from_0.2k"] > 0

    def test_r_gate_equality_semantics(self):
        gate = metrics.r_gate_verdict(0.26, 0.30, 0.31)
        assert gate["cagr_gate_ge_26"] is True
        assert gate["distance_to_26_pp"] == 0.0
        assert gate["mdd_strictly_below_f"] is True
        assert gate["both_gates"] is True
        gate2 = metrics.r_gate_verdict(0.2599999, 0.30, 0.31)
        assert gate2["cagr_gate_ge_26"] is False
        gate3 = metrics.r_gate_verdict(0.27, 0.31, 0.31)
        assert gate3["mdd_strictly_below_f"] is False

    def test_bootstrap_deterministic(self):
        dates = pd.bdate_range("20190101", periods=60).strftime("%Y%m%d")
        # 左（F）：等比 1%/日且每 5 日一个 −8% 回撤点 ⇒ 任意重抽序下 mdd_loss > 0
        nav_a = [1.01**index * (0.92 if index % 5 == 3 else 1.0) for index in range(60)]
        # 右（R）：等比 2%/日、无回撤（乘积与重抽序无关 ⇒ ΔCAGR 确定性为正）
        nav_b = [1.02**index for index in range(60)]
        frame_a = pd.DataFrame({"trade_date": dates, "nav": nav_a})
        frame_b = pd.DataFrame({"trade_date": dates, "nav": nav_b})
        first = metrics.bootstrap_diff(frame_a, frame_b, repetitions=50, seed=42)
        second = metrics.bootstrap_diff(frame_a, frame_b, repetitions=50, seed=42)
        assert first["cagr_diff"] == second["cagr_diff"]
        assert first["cagr_diff"]["prob_positive"] == pytest.approx(1.0)
        assert first["mdd_loss_diff"]["mean"] < 0
        assert first["mdd_loss_diff"]["prob_negative"] == pytest.approx(1.0)

    def test_b_required_row_writer_roundtrip(self, tmp_path: Path):
        rows = [metrics.b_required_row(1_000_000, 2_500_000.0, 1839)]
        target = tmp_path / "b_required.csv"
        metrics.write_b_required_csv(target, rows)
        text = target.read_text(encoding="utf-8")
        assert "target_total_value" in text
        assert rows[0]["capital"] == 1_000_000


# ── 训练组装（注册参数与部署标记）────────────────────────────────────


class TestFullTrainParams:
    def _config_and_params(self):
        config = training.build_model_config()
        from src.lazybull.v2.train.trainer import params_from_config

        return config, params_from_config(config)

    @staticmethod
    def _ensemble() -> Any:
        import xgboost as xgb

        from src.lazybull.v2.train.ensemble import EnsembleModel

        return EnsembleModel([xgb.XGBRegressor(n_estimators=1, max_depth=1)])

    @staticmethod
    def _coverage(columns: Sequence[str]) -> dict:
        from src.lazybull.v2.common.model_schema import SELECTION_FILTER_COLUMNS

        keys = list(dict.fromkeys([*columns, *SELECTION_FILTER_COLUMNS]))

        def entry() -> dict:
            return {
                "mean": 0.05,
                "p50": 0.02,
                "p90": 0.1,
                "p95": 0.15,
                "p99": 0.3,
                "max": 0.5,
                "n_days": 100,
                "p99_unstable": False,
            }

        return {key: entry() for key in keys}

    def test_assemble_includes_spec_versions_and_ensemble(self):
        config, params = self._config_and_params()
        full = training._assemble_full_train_params(
            config,
            params,
            {"n_estimators": training.RESOLVED_N_ESTIMATORS},
            ["a", "b"],
            self._ensemble(),
            ensemble_seeds=[42, 61, 82],
            data_stats={
                "feature_missing_rate_distribution": self._coverage(["a", "b"]),
                "feature_missing_rate_stat_basis": {"domain": "train_filtered_main_board"},
            },
            train_start="20120101",
            val_end="20180101",
            is_deploy=False,
        )
        assert full["ensemble_seeds"] == [42, 61, 82]
        assert full["ensemble_n_models"] == 1
        assert full["label_spec"]["version"] == 1
        assert full["feature_recipe"]["version"] == 1
        assert full["split_spec"]["version"] == 1
        assert full["domain"] == "main"
        assert "is_deploy" not in full
        assert "feature_missing_rate" in full

    def test_deploy_flag_and_live_columns(self):
        config, params = self._config_and_params()
        full = training._assemble_full_train_params(
            config,
            params,
            {},
            ["zscore_ocf_to_revenue", "head_a"],
            self._ensemble(),
            ensemble_seeds=[42, 61, 82],
            data_stats={
                "feature_missing_rate_distribution": self._coverage(
                    ["zscore_ocf_to_revenue", "head_a"]
                ),
                "feature_missing_rate_stat_basis": {"domain": "train_filtered_main_board"},
            },
            train_start="20200106",
            val_end="20260105",
            is_deploy=True,
        )
        assert full["is_deploy"] is True
        # 现金流 live 元数据按本臂最终列生成（协议 §5.2-6）
        assert full["cashflow_quality_feature_columns"] == ["zscore_ocf_to_revenue"]

    def test_missing_coverage_is_fail_closed(self):
        config, params = self._config_and_params()
        with pytest.raises(ValueError, match="feature_missing_rate"):
            training._assemble_full_train_params(
                config,
                params,
                {},
                ["a"],
                self._ensemble(),
                ensemble_seeds=[42, 61, 82],
                data_stats=None,
                train_start="20120101",
                val_end="20180101",
                is_deploy=False,
            )
