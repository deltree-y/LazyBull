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
    empty_training_matrix,
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
    "20240102",
    "20240103",
    "20240104",
    "20240105",
    "20240108",
    "20240109",
    "20240110",
    "20240111",
    "20240112",
    "20240115",
]


def _panels():
    open_panel = pd.DataFrame({"A": [100.0] * len(CAL), "B": [100.0] * len(CAL)}, index=CAL)
    open_panel.iloc[2, 0] = 90.0  # A 股 idx2 开盘大跌
    sigma_panel = pd.DataFrame({"A": [0.05] * len(CAL), "B": [0.05] * len(CAL)}, index=CAL)
    return open_panel, sigma_panel


def _day_features(codes, seed=0):
    """单日训练行：manifest 基础列全量给值。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts_code": list(codes),
            **{c: rng.normal(0, 1, len(codes)) for c in BASE_FEATURES},
        }
    )


def _mother(codes, seed=0, ret20=None):
    """单日完整母截面（pct 基列）。"""
    rng = np.random.default_rng(seed + 100)
    frame = pd.DataFrame(
        {
            "ts_code": list(codes),
            "ret_20": ret20 if ret20 is not None else rng.normal(0, 1, len(codes)),
            "cvar_95_20": rng.normal(-0.02, 0.01, len(codes)),
            "max_drawdown_20": rng.normal(-0.05, 0.01, len(codes)),
            "amihud_illiq_20": rng.random(len(codes)),
        }
    )
    return frame


def _mother_by_date(codes, dates, seed=0):
    return {d: _mother(codes, seed=seed + i) for i, d in enumerate(dates)}


class TestManifest:
    def test_manifest_is_33_columns_frozen(self):
        assert len(TERMINAL_LOSS_FEATURES) == 33
        assert set(TERMINAL_LOSS_FEATURES) == set(DERIVED_FEATURES) | set(BASE_FEATURES) | {
            "pct_ret_20",
            "pct_cvar_95_20",
            "pct_max_drawdown_20",
            "pct_amihud_illiq_20",
        }

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="缺少 manifest"):
            validate_feature_manifest(["ts_code"] + BASE_FEATURES[:-1])


class TestPctFeatures:
    def test_pct_ranked_on_full_mother_section(self):
        """百分位分母为完整母截面：A/B 在 3 只母截面下为 1/3、2/3。"""
        day = _day_features(["A", "B"])
        mother = _mother(["A", "B", "C"], ret20=[0.01, 0.02, 0.03])
        out = add_pct_features(day, mother)
        assert out["pct_ret_20"].tolist() == pytest.approx([1 / 3, 2 / 3])
        assert list(out["ts_code"]) == ["A", "B"]

    def test_denominator_not_taken_from_day_rows(self):
        """分母只能是母截面：只传子集行排名会得到 1/2 与 1.0，必须是 2/3 与 1.0。"""
        day = _day_features(["B", "C"])
        mother = _mother(["A", "B", "C"], ret20=[0.01, 0.02, 0.03])
        out = add_pct_features(day, mother)
        assert out["pct_ret_20"].tolist() == pytest.approx([2 / 3, 1.0])
        subset_rank = day.copy()
        subset_rank["ret_20"] = [0.02, 0.03]
        assert subset_rank["ret_20"].rank(pct=True).tolist() == pytest.approx([0.5, 1.0])

    def test_day_base_values_do_not_affect_pct(self):
        """pct 只取决于母截面取值：训练行自带的基列（cs_train 口径）不参与排名。"""
        day = _day_features(["A", "B"])
        day["ret_20"] = [999.0, -999.0]
        mother = _mother(["A", "B", "C"], ret20=[0.01, 0.02, 0.03])
        out = add_pct_features(day, mother)
        assert out["pct_ret_20"].tolist() == pytest.approx([1 / 3, 2 / 3])

    def test_day_row_absent_from_mother_raises(self):
        day = _day_features(["A", "Z"])
        mother = _mother(["A", "B", "C"])
        with pytest.raises(ValueError, match="不在完整母截面中"):
            add_pct_features(day, mother)

    def test_empty_mother_raises(self):
        with pytest.raises(ValueError, match="母截面为空"):
            add_pct_features(_day_features(["A"]), pd.DataFrame())

    def test_mother_missing_base_column_raises(self):
        mother = _mother(["A"]).drop(columns=["ret_20"])
        with pytest.raises(ValueError, match="缺少 pct 基列"):
            add_pct_features(_day_features(["A"]), mother)

    def test_missing_manifest_column_raises(self):
        day = _day_features(["A", "B"]).drop(columns=["ret_5"])
        with pytest.raises(ValueError, match="缺少 manifest 基础列"):
            add_pct_features(day, _mother(["A", "B"]))


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
            labels,
            features_by_date,
            sigma_panel,
            _mother_by_date(["A", "B", "C"], CAL),
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
            (matrix["ts_code"] == "A") & (matrix["trade_date"] == CAL[0]) & (matrix["h"] == 1)
        ].iloc[0]
        assert row["loss_label"] == 1

    def test_missing_feature_day_skipped(self):
        labels, sigma_panel = self._labels()
        features_by_date = {d: _day_features(["A", "B"]) for d in CAL[:5]}
        matrix = build_training_matrix(
            labels, features_by_date, sigma_panel, _mother_by_date(["A", "B", "C"], CAL)
        )
        assert set(matrix["trade_date"]) <= set(CAL[:5])
        assert len(matrix) > 0

    def test_mother_missing_for_feature_day_raises(self):
        """有训练行但缺母截面时必须失败：不得退化为子集排名。"""
        labels, sigma_panel = self._labels()
        features_by_date = {d: _day_features(["A", "B"]) for d in CAL}
        with pytest.raises(ValueError, match="缺少完整母截面"):
            build_training_matrix(labels, features_by_date, sigma_panel, {})

    def test_pct_columns_come_from_mother(self):
        labels, sigma_panel = self._labels()
        features_by_date = {d: _day_features(["A", "B"]) for d in CAL}
        mother = _mother_by_date(["A", "B", "C"], CAL)
        matrix = build_training_matrix(labels, features_by_date, sigma_panel, mother)
        for d in CAL[:3]:
            expected = mother[d].set_index("ts_code")["ret_20"].rank(pct=True)
            rows = matrix[matrix["trade_date"] == d]
            if rows.empty:
                continue
            for _, row in rows.iterrows():
                assert row["pct_ret_20"] == pytest.approx(expected[row["ts_code"]])

    def test_empty_matrix_keeps_numeric_schema(self):
        """空标签的返回必须与非空路径同列同 dtype（分块 concat 契约）。"""
        labels, sigma_panel = self._labels()
        matrix = build_training_matrix(labels.iloc[0:0], {}, sigma_panel, {})
        assert matrix.empty
        assert list(matrix.columns) == META_COLUMNS + TERMINAL_LOSS_FEATURES
        for col in TERMINAL_LOSS_FEATURES:
            assert pd.api.types.is_numeric_dtype(matrix[col]), col
        assert matrix["h"].dtype == np.int64
        assert matrix["remaining_intervals"].dtype == np.int64

    def test_all_immature_piece_concat_keeps_numeric_dtypes(self):
        """回归（WF 2024H2 折失败）：数据末端整块 immature 产生的空块
        与正常块 concat 后，特征列不得被提升为 object（XGBoost 拒绝输入）。"""
        labels, sigma_panel = self._labels()
        features_by_date = {d: _day_features(["A", "B"], seed=i) for i, d in enumerate(CAL)}
        matrix = build_training_matrix(
            labels,
            features_by_date,
            sigma_panel,
            _mother_by_date(["A", "B", "C"], CAL),
            DatasetConfig(horizon_grid_size=20),
        )
        assert len(matrix) > 0
        # 末日标签全部 immature（E 超出日历末端）→ 空块
        last_day = labels[labels["trade_date"] == CAL[-1]]
        assert (last_day["label_status"] == "immature").all()
        empty_piece = build_training_matrix(last_day, features_by_date, sigma_panel, {})
        assert empty_piece.empty
        combined = pd.concat([matrix, empty_piece], ignore_index=True)
        for col in TERMINAL_LOSS_FEATURES:
            assert pd.api.types.is_numeric_dtype(combined[col]), col
        assert len(combined) == len(matrix)

    def test_empty_training_matrix_helper_schema(self):
        """empty_training_matrix 直接产出的空帧即数值 dtype。"""
        empty = empty_training_matrix()
        assert empty.empty
        assert list(empty.columns) == META_COLUMNS + TERMINAL_LOSS_FEATURES
        for col in TERMINAL_LOSS_FEATURES:
            assert pd.api.types.is_numeric_dtype(empty[col]), col


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
        result = split_stages_with_label_isolation(matrix, [StageSpec("train", CAL[0], CAL[2])])
        assert result.stages["train"].empty

    def test_three_stage_isolation_val_before_es(self):
        """三段（train/val/es）逐段隔离：Val 是早停段，ES 是纯评估段（v0.109.0）。"""
        matrix = pd.DataFrame(
            {
                "ts_code": ["A"] * 6,
                "trade_date": [CAL[0], CAL[0], CAL[3], CAL[3], CAL[6], CAL[6]],
                "h": [1, 5, 1, 5, 1, 5],
                # label_end 落点：idx2/idx6/idx5/idx9/idx8/idx9
                "label_end_date": [CAL[2], CAL[6], CAL[5], CAL[9], CAL[8], CAL[9]],
                "loss_label": [1, 0, 1, 0, 1, 0],
            }
        )
        stages = [
            StageSpec("train", CAL[0], CAL[0]),
            StageSpec("val", CAL[3], CAL[3]),
            StageSpec("es", CAL[6], CAL[6]),
        ]
        result = split_stages_with_label_isolation(matrix, stages)
        # Train：h=1 的 label_end=CAL[2] < Val 起点 CAL[3] 保留；
        # h=5 的 label_end=CAL[6] >= CAL[3] 剔除
        assert len(result.stages["train"]) == 1
        assert result.isolation_dropped["train"] == 1
        # Val：h=1 的 label_end=CAL[5] < ES 起点 CAL[6] 保留；
        # h=5 的 label_end=CAL[9] >= CAL[6] 剔除
        assert len(result.stages["val"]) == 1
        assert result.isolation_dropped["val"] == 1
        # ES 评估段是最后一段：无下一阶段，行不受隔离
        assert len(result.stages["es"]) == 2
        assert result.isolation_dropped["es"] == 0


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
        small = pd.DataFrame({"ts_code": ["A", "A"], "trade_date": ["20240102"] * 2, "h": [1, 2]})
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
