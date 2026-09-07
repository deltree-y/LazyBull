"""期末异常亏损数据集构建测试（合成数据，不依赖真实配置与真实数据）"""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.terminal_loss.dataset import (
    BASE_FEATURES,
    DERIVED_FEATURES,
    META_COLUMNS,
    TERMINAL_LOSS_FEATURES,
    DatasetConfig,
    StageSpec,
    add_pct_features,
    attach_horizon_features,
    build_training_matrix,
    split_stages_with_label_isolation,
    subsample_dates,
    subsample_h_per_group,
    validate_feature_manifest,
)
from src.lazybull.risk.terminal_loss.labels import (
    LABEL_STATUS_VALID,
    TerminalLossLabelConfig,
    build_terminal_loss_labels,
)

CAL: List[str] = [
    "20240102", "20240103", "20240104", "20240105",
    "20240108", "20240109", "20240110", "20240111", "20240112", "20240115",
]


def _panels():
    open_panel = pd.DataFrame(
        {"A": [100.0] * len(CAL), "B": [100.0] * len(CAL)}, index=CAL
    )
    open_panel.iloc[2, 0] = 90.0  # A 股 idx2 开盘大跌
    sigma_panel = pd.DataFrame(
        {"A": [0.05] * len(CAL), "B": [0.05] * len(CAL)}, index=CAL
    )
    return open_panel, sigma_panel


def _day_features(codes, seed=0):
    """单日完整母截面：manifest 基础列全量给值。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts_code": list(codes),
            **{c: rng.normal(0, 1, len(codes)) for c in BASE_FEATURES},
        }
    )


class TestManifest:
    def test_manifest_is_33_columns_frozen(self):
        assert len(TERMINAL_LOSS_FEATURES) == 33
        assert set(TERMINAL_LOSS_FEATURES) == set(DERIVED_FEATURES) | set(
            BASE_FEATURES
        ) | {"pct_ret_20", "pct_cvar_95_20", "pct_max_drawdown_20", "pct_amihud_illiq_20"}

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="缺少 manifest"):
            validate_feature_manifest(["ts_code"] + BASE_FEATURES[:-1])


class TestPctFeatures:
    def test_pct_from_full_cross_section(self):
        """百分位基于完整母截面：分母含全部证券。"""
        day = _day_features(["A", "B", "C"])
        day["ret_20"] = [0.01, 0.02, 0.03]
        out = add_pct_features(day)
        assert out["pct_ret_20"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_pct_not_recomputed_on_filtered_subset(self):
        """先筛行再排名会改变百分位（反例语义锁定）：全截面 2/3 vs 子集 1.0。"""
        day = _day_features(["A", "B", "C"])
        day["ret_20"] = [0.01, 0.02, 0.03]
        full = add_pct_features(day)
        assert full.loc[1, "pct_ret_20"] == pytest.approx(2 / 3)
        # 若只拿 B/C 两行排名，B 会变成 1/2——本函数要求调用方传完整截面
        subset_rank = day.iloc[1:]["ret_20"].rank(pct=True).iloc[0]
        assert subset_rank == pytest.approx(1 / 2)


class TestHorizonFeatures:
    def test_sigma_sqrt_h(self):
        matrix = pd.DataFrame({"h": [1, 4, 19], "sigma_daily_20": [0.02, 0.02, 0.02]})
        out = attach_horizon_features(matrix)
        assert out["remaining_intervals"].tolist() == [1, 4, 19]
        assert out["expected_vol_over_horizon"].iloc[0] == pytest.approx(0.02)
        assert out["expected_vol_over_horizon"].iloc[1] == pytest.approx(0.04)
        assert out["expected_vol_over_horizon"].iloc[2] == pytest.approx(0.02 * np.sqrt(19))


class TestBuildTrainingMatrix:
    def _labels(self):
        open_panel, sigma_panel = _panels()
        return (
            build_terminal_loss_labels(
                open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=2)
            ),
            sigma_panel,
        )

    def test_matrix_schema_weights_and_sigma(self):
        labels, sigma_panel = self._labels()
        features_by_date = {d: _day_features(["A", "B"], seed=i) for i, d in enumerate(CAL)}
        matrix = build_training_matrix(
            labels, features_by_date, sigma_panel,
            DatasetConfig(horizon_grid_size=20),
        )
        # 列契约：元数据 + 33 特征
        assert list(matrix.columns) == META_COLUMNS + TERMINAL_LOSS_FEATURES
        # 仅 valid 行：A 股每股 10 T × 2 h - immature(2+3) = 15；B 股 15
        assert len(matrix) == 30
        # 权重 = 1/20（组内总权重归一，未成熟不重归一）
        assert matrix["sample_weight"].eq(1 / 20).all()
        # sigma 关联正确（全为 0.05）
        assert matrix["sigma_daily_20"].eq(0.05).all()
        # h 与 σ√h 派生
        assert matrix["expected_vol_over_horizon"].eq(0.05 * np.sqrt(matrix["h"])).all()
        # 标签数学：A 股 T=idx0,h=1 的 R=-0.1 < -0.05 → 1
        row = matrix[
            (matrix["ts_code"] == "A")
            & (matrix["trade_date"] == CAL[0])
            & (matrix["h"] == 1)
        ].iloc[0]
        assert row["loss_label"] == 1

    def test_missing_feature_day_skipped(self):
        labels, sigma_panel = self._labels()
        features_by_date = {d: _day_features(["A", "B"]) for d in CAL[:5]}
        matrix = build_training_matrix(labels, features_by_date, sigma_panel)
        assert set(matrix["trade_date"]) <= set(CAL[:5])
        assert len(matrix) > 0


class TestStageIsolation:
    def _matrix(self):
        """构造带 label_end_date 的矩阵：CAL 日期 + 手工 label_end_date。"""
        return pd.DataFrame(
            {
                "ts_code": ["A"] * 6,
                "trade_date": [CAL[0], CAL[0], CAL[2], CAL[2], CAL[4], CAL[4]],
                "h": [1, 2, 1, 2, 1, 2],
                # T+1+h 落点：idx2/idx3/idx4/idx5/idx6/idx7
                "label_end_date": [CAL[2], CAL[3], CAL[4], CAL[5], CAL[6], CAL[7]],
                "loss_label": [1, 0, 1, 0, 1, 0],
            }
        )

    def test_immature_rows_dropped_from_earlier_stage(self):
        """label_end_date >= 下一阶段起点的行必须从当前阶段剔除。"""
        matrix = self._matrix()
        stages = [
            StageSpec("train", CAL[0], CAL[2]),
            StageSpec("es", CAL[4], CAL[5]),
        ]
        result = split_stages_with_label_isolation(matrix, stages)
        # train 内 T=idx0 两行：h=1 的 label_end=idx2 < idx4 保留；
        # h=2 的 label_end=idx3 < idx4 保留；T=idx2 两行 label_end idx4/idx5 >= idx4 剔除
        train = result.stages["train"]
        assert set(train["h"]) == {1, 2}
        assert (train["trade_date"] == CAL[0]).all()
        assert result.isolation_dropped["train"] == 2
        # es：T=idx4 的行不受隔离（无下一阶段），h=1/2 都保留
        es = result.stages["es"]
        assert len(es) == 2
        assert result.isolation_dropped["es"] == 0

    def test_same_ts_date_group_stays_together(self):
        """同一 (ts_code, trade_date) 的不同 h 随 T 进入同一阶段。"""
        matrix = self._matrix()
        stages = [
            StageSpec("train", CAL[0], CAL[4]),
            StageSpec("oos", CAL[8], CAL[9]),
        ]
        result = split_stages_with_label_isolation(matrix, stages)
        # T=idx4 两行 label_end idx6/idx7 < CAL[8]，整组保留
        train = result.stages["train"]
        assert set(train["trade_date"]) == {CAL[0], CAL[2], CAL[4]}
        assert result.isolation_dropped["train"] == 0

    def test_empty_matrix(self):
        matrix = pd.DataFrame(columns=["trade_date", "label_end_date"])
        result = split_stages_with_label_isolation(
            matrix, [StageSpec("train", CAL[0], CAL[2])]
        )
        assert result.stages["train"].empty


class TestSubsampling:
    def _matrix(self, n_codes=50, n_days=30):
        codes = [f"C{i:03d}" for i in range(n_codes)]
        rows = []
        for d in range(n_days):
            for c in codes:
                for h in range(1, 21):
                    rows.append((c, f"2024{d:04d}", h))
        return pd.DataFrame(rows, columns=["ts_code", "trade_date", "h"])

    def test_h_subsample_keeps_n_per_group_and_covers_all_h(self):
        matrix = self._matrix()
        out = subsample_h_per_group(matrix, n_h=2)
        per_group = out.groupby(["ts_code", "trade_date"]).size()
        assert (per_group == 2).all()
        # 跨组覆盖全部 1..20（偏移轮转保证）
        assert set(out["h"]) == set(range(1, 21))

    def test_h_subsample_deterministic(self):
        matrix = self._matrix(n_codes=20, n_days=10)
        a = subsample_h_per_group(matrix, n_h=2)
        b = subsample_h_per_group(matrix, n_h=2)
        pd.testing.assert_frame_equal(a, b)

    def test_h_subsample_no_op_when_grid_small(self):
        small = pd.DataFrame(
            {"ts_code": ["A", "A"], "trade_date": ["20240102"] * 2, "h": [1, 2]}
        )
        out = subsample_h_per_group(small, n_h=2)
        assert len(out) == 2

    def test_date_subsample_keeps_full_cross_section(self):
        matrix = self._matrix(n_codes=3, n_days=9)
        out = subsample_dates(matrix, every_n=3)
        kept_dates = sorted(out["trade_date"].unique())
        # 9 个交易日等距取 3 个；同日全截面整组保留
        assert len(kept_dates) == 3
        for d in kept_dates:
            day_rows = out[out["trade_date"] == d]
            assert set(day_rows["ts_code"]) == {"C000", "C001", "C002"}
            assert set(day_rows["h"]) == set(range(1, 21))
        assert len(subsample_dates(matrix, every_n=1)) == len(matrix)
