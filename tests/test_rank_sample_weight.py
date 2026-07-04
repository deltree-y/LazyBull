"""测试 rank-weight sample_weight 构造逻辑

验证：
- 单日截面 Top/Bottom K 样本按 linear_decay 权重正确（默认）
- Top/Bottom K 末位权重为 2.0，中间样本权重为 1.0
- 多日分组不串（各日独立排名）
- K 大于样本数时全部样本加权（退化处理）
- 返回数组长度与输入一致
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml.train_core import build_rank_sample_weights


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _make_single_day(n: int = 20, label_col: str = 'neu_y_ret_20',
                     date: str = '20230102') -> pd.DataFrame:
    """创建单日截面数据（标签值等间隔，方便验证排名）"""
    return pd.DataFrame({
        'ts_code': [f'{i:06d}.SZ' for i in range(n)],
        'trade_date': [date] * n,
        label_col: [float(i) for i in range(n)],  # 0,1,...,n-1
    })


def _make_multi_day(n_per_day: int = 20, n_days: int = 3,
                    label_col: str = 'neu_y_ret_20') -> pd.DataFrame:
    """创建多日截面数据"""
    frames = []
    for d in range(n_days):
        date = f'202301{d + 2:02d}'
        df = _make_single_day(n=n_per_day, label_col=label_col, date=date)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. 单日 Top/Bottom K 权重测试
# ---------------------------------------------------------------------------

class TestBuildRankSampleWeightsSingleDay:
    """单日截面 rank-weight 测试"""

    def test_topk_linear_decay_weights_correct(self):
        """Top/Bottom K 样本应按 linear_decay 递减赋权，末位为2，中间样本为1"""
        n = 20
        topk = 5
        top_weight = 5.0
        df = _make_single_day(n=n)

        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=top_weight)

        assert len(weights) == n, "权重数组长度应与输入行数相同"

        expected = np.ones(n, dtype=float)
        # Top: 标签 19,18,17,16,15 对应 rank 1..5
        expected[19] = 5.0
        expected[18] = 4.25
        expected[17] = 3.5
        expected[16] = 2.75
        expected[15] = 2.0
        # Bottom: 标签 0,1,2,3,4 对应 rank 1..5
        expected[0] = 5.0
        expected[1] = 4.25
        expected[2] = 3.5
        expected[3] = 2.75
        expected[4] = 2.0
        np.testing.assert_allclose(weights, expected)

    def test_topk_flat_mode_weights_correct(self):
        """flat 模式下 Top/Bottom K 样本同权 top_weight"""
        n = 20
        topk = 5
        top_weight = 5.0
        df = _make_single_day(n=n)

        weights = build_rank_sample_weights(
            df,
            'neu_y_ret_20',
            topk=topk,
            top_weight=top_weight,
            topk_weight_mode='flat',
        )

        expected = np.ones(n, dtype=float)
        expected[0:5] = top_weight
        expected[15:20] = top_weight
        np.testing.assert_allclose(weights, expected)

    def test_weight_count_correct(self):
        """默认 linear_decay 下，加权样本数应为 2*topk（Top/Bottom 末位权重=2）"""
        n = 30
        topk = 5
        df = _make_single_day(n=n)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk)
        heavy_count = int((weights > 1.0).sum())
        assert heavy_count == (2 * topk), f"加权样本数应为 {2 * topk}，实际={heavy_count}"

    def test_default_topk_30(self):
        """默认 topk=30 且 linear_decay 时加权样本数应为 60（Top/Bottom 各30）"""
        n = 100
        df = _make_single_day(n=n)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20')
        heavy_count = int((weights > 1.0).sum())
        assert heavy_count == 60, f"默认 topk=30 时加权样本数应为 60，实际={heavy_count}"

    def test_returns_numpy_array(self):
        """返回值应为 numpy 数组"""
        df = _make_single_day(n=20)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20')
        assert isinstance(weights, np.ndarray), "返回值类型应为 np.ndarray"


# ---------------------------------------------------------------------------
# 2. K 大于样本数时的退化处理
# ---------------------------------------------------------------------------

class TestBuildRankSampleWeightsDegenerateCase:
    """K 大于等于样本数时的边界处理"""

    def test_k_greater_than_n_all_weighted(self):
        """当 n <= topk 时，整组样本均赋 top_weight（退化处理）"""
        n = 8
        topk = 8
        top_weight = 4.0
        df = _make_single_day(n=n)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=top_weight)

        # 全部应为 top_weight
        assert (weights == top_weight).all(), "样本数 <= topk 时全部应赋 top_weight"

    def test_k_equal_n(self):
        """n == topk 时全部赋 top_weight（边界情况）"""
        n = 10
        topk = 10
        df = _make_single_day(n=n)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=3.0)
        assert (weights == 3.0).all()

    def test_empty_dataframe(self):
        """空 DataFrame 时返回长度为 0 的数组，不报错"""
        df = pd.DataFrame(columns=['ts_code', 'trade_date', 'neu_y_ret_20'])
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=30)
        assert len(weights) == 0


# ---------------------------------------------------------------------------
# 3. 多日分组不串
# ---------------------------------------------------------------------------

class TestBuildRankSampleWeightsMultiDay:
    """多日截面不串权重测试"""

    def test_multiday_groups_independent(self):
        """多日数据中，每日独立排名，互不影响"""
        n_per_day = 20
        n_days = 3
        topk = 3
        df = _make_multi_day(n_per_day=n_per_day, n_days=n_days)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=5.0)

        assert len(weights) == n_per_day * n_days

        # linear_decay 下每日应有 2*topk 个加权样本（Top/Bottom 末位均为2）
        for d in range(n_days):
            start = d * n_per_day
            end = start + n_per_day
            day_weights = weights[start:end]
            heavy_count = int((day_weights > 1.0).sum())
            assert heavy_count == (2 * topk), (
                f"第 {d+1} 日加权样本数应为 {2 * topk}，实际={heavy_count}"
            )

    def test_total_weighted_count(self):
        """多日场景：总加权样本数 = n_days * 2 * topk"""
        n_per_day = 20
        n_days = 3
        topk = 4
        df = _make_multi_day(n_per_day=n_per_day, n_days=n_days)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=5.0)

        expected_heavy = n_days * 2 * topk
        actual_heavy = int((weights > 1.0).sum())
        assert actual_heavy == expected_heavy, (
            f"总加权样本数应为 {expected_heavy}，实际={actual_heavy}"
        )


# ---------------------------------------------------------------------------
# 4. 异常/鲁棒性测试
# ---------------------------------------------------------------------------

class TestBuildRankSampleWeightsRobust:
    """异常和鲁棒性测试"""

    def test_missing_label_column(self):
        """标签列不存在时，返回全为 1 的权重（不报错）"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20230102', '20230102'],
            'other_col': [1.0, 2.0],
        })
        weights = build_rank_sample_weights(df, 'neu_y_ret_20')
        assert (weights == 1.0).all()

    def test_missing_date_column(self):
        """日期列不存在时，返回全为 1 的权重（不报错）"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'neu_y_ret_20': [0.01, 0.02],
        })
        weights = build_rank_sample_weights(df, 'neu_y_ret_20')
        assert (weights == 1.0).all()

    def test_nan_labels_handled(self):
        """标签含 NaN 时不应崩溃，NaN 样本不被计入排名"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'trade_date': ['20230102'] * 10,
            'neu_y_ret_20': [float(i) if i < 8 else float('nan') for i in range(10)],
        })
        # topk=2 时应正常处理（8 有效样本 > 2）
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=2, top_weight=3.0)
        assert len(weights) == 10
        # NaN 样本（索引 8,9）不应被设为 top_weight
        assert weights[8] == 1.0
        assert weights[9] == 1.0

    def test_weight_array_length_matches_df(self):
        """返回数组长度始终与输入 DataFrame 行数相同"""
        for n in [5, 20, 100]:
            df = _make_single_day(n=n)
            weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=3)
            assert len(weights) == n, f"n={n} 时权重数组长度不匹配"
