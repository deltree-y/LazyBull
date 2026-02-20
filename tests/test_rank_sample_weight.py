"""测试 rank-weight sample_weight 构造逻辑

验证：
- 单日截面 Top K / Bottom K 样本权重正确
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
# 1. 单日 Top K / Bottom K 权重测试
# ---------------------------------------------------------------------------

class TestBuildRankSampleWeightsSingleDay:
    """单日截面 rank-weight 测试"""

    def test_topk_bottomk_weights_correct(self):
        """Top K 和 Bottom K 样本权重应为 top_weight，其余为 1.0"""
        n = 20
        topk = 5
        top_weight = 5.0
        df = _make_single_day(n=n)

        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=top_weight)

        assert len(weights) == n, "权重数组长度应与输入行数相同"

        # Top 5：标签值最大的 5 个（索引 15-19）
        top_indices = list(range(n - topk, n))
        # Bottom 5：标签值最小的 5 个（索引 0-4）
        bottom_indices = list(range(topk))
        # 其余
        middle_indices = list(range(topk, n - topk))

        for i in top_indices:
            assert weights[i] == top_weight, f"Top K 样本 {i} 权重应为 {top_weight}，实际={weights[i]}"
        for i in bottom_indices:
            assert weights[i] == top_weight, f"Bottom K 样本 {i} 权重应为 {top_weight}，实际={weights[i]}"
        for i in middle_indices:
            assert weights[i] == 1.0, f"中间样本 {i} 权重应为 1.0，实际={weights[i]}"

    def test_weight_count_correct(self):
        """加权样本数应恰好为 2 * topk（当 n > 2*topk 时）"""
        n = 30
        topk = 5
        df = _make_single_day(n=n)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk)
        heavy_count = int((weights > 1.0).sum())
        assert heavy_count == 2 * topk, f"加权样本数应为 {2*topk}，实际={heavy_count}"

    def test_default_topk_30(self):
        """默认 topk=30 时加权样本数应为 2*30=60（n>60）"""
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
        """当 n <= 2*topk 时，整组样本均赋 top_weight（退化处理）"""
        n = 8
        topk = 5  # 2*5=10 > 8
        top_weight = 4.0
        df = _make_single_day(n=n)
        weights = build_rank_sample_weights(df, 'neu_y_ret_20', topk=topk, top_weight=top_weight)

        # 全部应为 top_weight
        assert (weights == top_weight).all(), "样本数 <= 2*topk 时全部应赋 top_weight"

    def test_k_equal_half_n(self):
        """n == 2*topk 时全部赋 top_weight（边界情况）"""
        n = 10
        topk = 5
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

        # 每日应有 2*topk 个加权样本
        for d in range(n_days):
            start = d * n_per_day
            end = start + n_per_day
            day_weights = weights[start:end]
            heavy_count = int((day_weights > 1.0).sum())
            assert heavy_count == 2 * topk, (
                f"第 {d+1} 日加权样本数应为 {2*topk}，实际={heavy_count}"
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
        # topk=2 时应正常处理（8 有效样本 > 2*2=4）
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
