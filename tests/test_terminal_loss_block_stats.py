"""分块重采样统计测试（合成数据，不依赖真实数据）"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.terminal_loss import (
    BLOCK_DAYS_SENSITIVITY,
    DAY_NORM_SCORE_COL,
    BootstrapConfig,
    add_day_percentile_score,
    block_paired_delta,
    fold_level_gate,
    moving_block_metric_ci,
    moving_block_metric_sensitivity,
)
from src.lazybull.risk.terminal_loss.block_stats import lift


def _es_frame(n_days: int = 60, n_stocks: int = 30, seed: int = 0) -> pd.DataFrame:
    """合成 ES 评估行：p_loss 带真实信号（负收益更可能事件）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        for j in range(n_stocks):
            p = float(np.clip(rng.normal(0.3, 0.15), 0.01, 0.99))
            y = int(rng.random() < p)
            rows.append(
                {
                    "trade_date": f"2024{(i // 20) + 1:02d}{(i % 20) + 1:02d}",
                    "ts_code": f"0000{j:02d}.SZ",
                    "h": (j % 20) + 1,
                    "loss_label": y,
                    "p_loss": p,
                }
            )
    return pd.DataFrame(rows)


class TestLiftMetric:
    def test_day_percentile_is_within_day_and_monotone(self):
        frame = _es_frame(n_days=3, n_stocks=10, seed=5)
        out = add_day_percentile_score(frame)
        assert DAY_NORM_SCORE_COL in out.columns
        for _, g in out.groupby("trade_date"):
            assert g[DAY_NORM_SCORE_COL].between(0.0, 1.0, inclusive="right").all()
            assert g.sort_values("p_loss")[DAY_NORM_SCORE_COL].is_monotonic_increasing

    def test_day_percentile_keeps_nan_and_rejects_bad_input(self):
        frame = _es_frame(n_days=3, n_stocks=5, seed=6)
        frame.loc[frame.index[:3], "p_loss"] = np.nan
        out = add_day_percentile_score(frame)
        # 缺失值保持缺失（不得填 0.5 把"无分数"伪装成中性排序）
        assert out[DAY_NORM_SCORE_COL].isna().sum() >= 3
        with pytest.raises(ValueError, match="缺少源分数列"):
            add_day_percentile_score(frame, src_col="ghost")
        with pytest.raises(ValueError, match="为空"):
            add_day_percentile_score(pd.DataFrame())

    def test_lift_equals_pr_auc_over_event_rate(self):
        y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
        from sklearn.metrics import average_precision_score

        assert lift(y, p) == pytest.approx(average_precision_score(y, p) / y.mean())

    def test_lift_none_when_no_event(self):
        assert lift(np.zeros(5, dtype=int), np.linspace(0.1, 0.9, 5)) is None


class TestMovingBlockCi:
    def test_daynorm_score_isolates_cross_sectional_ranking(self):
        """daynorm 口径不受跨日水平平移影响，raw 口径受影响（v0.110.0 双判据）。"""
        frame = _es_frame(n_days=40, n_stocks=20, seed=8)
        shifted = frame.copy()
        days = sorted(shifted["trade_date"].unique())
        upper = set(days[len(days) // 2 :])
        mask = shifted["trade_date"].isin(upper)
        shifted.loc[mask, "p_loss"] = shifted.loc[mask, "p_loss"] + 5.0
        cfg = BootstrapConfig(block_days=10, n_resamples=20, seed=3)
        raw_a = moving_block_metric_ci(frame, metric="lift", config=cfg)["point"]
        raw_b = moving_block_metric_ci(shifted, metric="lift", config=cfg)["point"]
        dn_a = moving_block_metric_ci(
            add_day_percentile_score(frame),
            metric="lift",
            config=cfg,
            score_col=DAY_NORM_SCORE_COL,
        )["point"]
        dn_b = moving_block_metric_ci(
            add_day_percentile_score(shifted),
            metric="lift",
            config=cfg,
            score_col=DAY_NORM_SCORE_COL,
        )["point"]
        # raw 池化口径把跨日水平也当信息；daynorm 只保留当日截面排序
        assert raw_a != pytest.approx(raw_b)
        assert dn_a == pytest.approx(dn_b)

    def test_missing_score_col_rejected(self):
        frame = _es_frame(n_days=30, n_stocks=10, seed=9)
        with pytest.raises(ValueError, match="缺少分数列"):
            moving_block_metric_ci(frame, metric="lift", score_col=DAY_NORM_SCORE_COL)

    def test_point_matches_lift_and_seed_reproducible(self):
        frame = _es_frame()
        cfg = BootstrapConfig(block_days=10, n_resamples=50, seed=7)
        first = moving_block_metric_ci(frame, metric="lift", config=cfg)
        second = moving_block_metric_ci(frame, metric="lift", config=cfg)
        assert first["point"] == pytest.approx(lift(frame["loss_label"], frame["p_loss"]))
        assert first["ci_low"] == pytest.approx(second["ci_low"])
        assert first["block_days"] == 10
        assert first["n_days"] == frame["trade_date"].nunique()

    def test_blocks_are_whole_day_cross_sections(self):
        """块单位必须是完整交易日截面：重采样后每日行数不变。"""
        frame = _es_frame(n_days=10, n_stocks=25, seed=3)
        cfg = BootstrapConfig(block_days=5, n_resamples=5, seed=1)
        result = moving_block_metric_ci(frame, metric="event_rate", config=cfg)
        assert result["n_days"] == 10
        # 行数按天整块放大：重采样行数 = 天数 × 每日行数
        assert result["point"] == pytest.approx(frame["loss_label"].mean())

    def test_sensitivity_defaults_to_es_block_sizes(self):
        """ES 段默认 5/10/20 日块；方案第 6 节的 40 日块属组合级多年 OOS 口径。"""
        frame = _es_frame(n_days=70, seed=5)
        results = moving_block_metric_sensitivity(
            frame, metric="lift", config=BootstrapConfig(n_resamples=20, seed=2)
        )
        assert [r["block_days"] for r in results] == [5, 10, 20]
        # lift 是比率统计量（PR-AUC/事件率），百分位区间不保证包含点估计，
        # 只要求区间有序且有限
        for r in results:
            assert r["ci_low"] <= r["ci_high"]
            assert np.isfinite(r["point"]) and np.isfinite(r["ci_low"])

    def test_sensitivity_composite_blocks_optional(self):
        frame = _es_frame(n_days=70, seed=5)
        results = moving_block_metric_sensitivity(
            frame,
            metric="lift",
            block_days_list=BLOCK_DAYS_SENSITIVITY,
            config=BootstrapConfig(n_resamples=20, seed=2),
        )
        assert [r["block_days"] for r in results] == [20, 40, 60]

    def test_infeasible_block_sizes_are_skipped_not_shrunk(self):
        """块长 ≥ 交易日数的项必须跳过（不得静默缩小块长）。"""
        frame = _es_frame(n_days=13, seed=6)
        results = moving_block_metric_sensitivity(
            frame, metric="lift", config=BootstrapConfig(n_resamples=10, seed=2)
        )
        assert [r["block_days"] for r in results] == [5, 10]

    def test_all_infeasible_blocks_raise(self):
        frame = _es_frame(n_days=13, seed=6)
        with pytest.raises(ValueError, match="不可行"):
            moving_block_metric_sensitivity(
                frame, block_days_list=(20, 40), config=BootstrapConfig(n_resamples=5)
            )

    def test_block_not_smaller_than_days_raises(self):
        """块长 ≥ 交易日数时重采样退化为原样本，必须显式报错。"""
        frame = _es_frame(n_days=8, seed=7)
        with pytest.raises(ValueError, match="≥ 交易日数"):
            moving_block_metric_ci(
                frame, metric="lift", config=BootstrapConfig(block_days=8, n_resamples=5)
            )
        one_day = moving_block_metric_ci(
            frame, metric="lift", config=BootstrapConfig(block_days=5, n_resamples=5)
        )
        assert one_day["ci_low"] <= one_day["ci_high"]
        assert one_day["std"] >= 0.0

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="评估行为空"):
            moving_block_metric_ci(pd.DataFrame(), metric="lift")


class TestBlockPairedDelta:
    def test_identical_arms_give_zero_delta(self):
        frame = _es_frame(seed=9)
        cfg = BootstrapConfig(block_days=10, n_resamples=30, seed=4)
        result = block_paired_delta(frame, frame.copy(), metric="lift", config=cfg)
        assert result["delta_point"] == pytest.approx(0.0)
        assert result["delta_mean"] == pytest.approx(0.0)
        assert result["prob_positive"] == 0.0

    def test_better_arm_detected(self):
        """A 组的概率含真实信号、B 组为常数：A 的 lift 应显著为正差值。"""
        frame = _es_frame(seed=11)
        a = frame.copy()
        a["p_loss"] = np.where(a["loss_label"] == 1, 0.8, 0.2)
        b = frame.copy()
        b["p_loss"] = 0.5
        cfg = BootstrapConfig(block_days=10, n_resamples=30, seed=4)
        result = block_paired_delta(a, b, metric="lift", config=cfg)
        assert result["delta_point"] > 0
        assert result["prob_positive"] > 0.9

    def test_row_count_mismatch_raises(self):
        a = _es_frame(n_stocks=10)
        b = _es_frame(n_stocks=9)
        with pytest.raises(ValueError, match="行数一致"):
            block_paired_delta(a, b, metric="lift")

    def test_key_misalignment_raises(self):
        a = _es_frame(seed=1)
        b = a.copy()
        b.loc[0, "ts_code"] = "999999.SZ"
        with pytest.raises(ValueError, match="逐行对齐"):
            block_paired_delta(a, b, metric="lift")

    def test_label_mismatch_raises(self):
        a = _es_frame(seed=1)
        b = a.copy()
        b.loc[0, "loss_label"] = 1 - b.loc[0, "loss_label"]
        with pytest.raises(ValueError, match="loss_label 必须一致"):
            block_paired_delta(a, b, metric="lift")


class TestFoldLevelGate:
    def test_reports_distribution_and_mean_interval(self):
        lifts = [1.15, 1.25, 1.46, 2.24, 1.65, 2.60, 1.25, 1.28]
        gate = fold_level_gate(lifts, lift_min_threshold=1.1, n_resamples=500, seed=1)
        assert gate["n_folds"] == 8
        assert gate["point_min"] == pytest.approx(1.15)
        assert gate["point_gate_pass"] is True
        assert gate["fold_min_pass_share"] == 1.0
        assert gate["ci_mean"][0] <= gate["point_mean"] <= gate["ci_mean"][1]

    def test_min_bootstrap_is_degenerate_by_construction(self):
        """重采样抽不到比观测最小值更差的折：min 区间下界恒等于观测最小值。

        该行为是口径边界而非缺陷：本函数不得被当作"最差折是否高于阈值"的
        检验，故显式标记 min_ci_degenerate 并由调用方改看逐折区间。
        """
        lifts = [1.15, 2.0, 3.0]
        gate = fold_level_gate(lifts, lift_min_threshold=1.1, n_resamples=200, seed=2)
        assert gate["min_ci_degenerate"] is True
        assert gate["ci_min_resampled"][0] == pytest.approx(1.15)

    def test_threshold_not_met_marks_fold_share(self):
        lifts = [1.05, 1.5, 1.6]
        gate = fold_level_gate(lifts, lift_min_threshold=1.1, n_resamples=100, seed=3)
        assert gate["point_gate_pass"] is False
        assert gate["fold_min_pass_share"] == pytest.approx(2 / 3)

    def test_empty_lifts_raise(self):
        with pytest.raises(ValueError, match="没有有效折 lift"):
            fold_level_gate([None, float("nan")])

    def test_prob_mean_pass_uses_mean_not_min(self):
        """均值口径的通过率必须来自均值分布，而不是 min 分布。"""
        lifts = [1.12, 1.12, 1.12, 1.12]
        gate = fold_level_gate(lifts, lift_min_threshold=1.1, n_resamples=100, seed=5)
        assert gate["prob_mean_pass"] == 1.0
