"""测试申万三级行业分层回退中性化

覆盖：
- L3 不足样本 → L2 回退
- L2 不足样本 → L1 回退
- L1 不足样本 → 全市场回退
- 仅 tradable==1 参与统计
- hierarchical_zscore 与 hierarchical_demean 正确性
- FeatureBuilder._apply_industry_neutralization 在当前实现中使用 L2 主口径回退路径
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.hierarchical_industry_neutralization import (
    hierarchical_demean,
    hierarchical_zscore,
)
from src.lazybull.features.builder import FeatureBuilder


# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------

def _make_l3_df(
    n_l3a=5, n_l3b=3, n_l3c=2,
    tradable_l3a=5, tradable_l3b=3, tradable_l3c=2,
    seed=42,
):
    """构造含 L1/L2/L3 三层行业信息的测试 DataFrame。

    L3 结构：
      A→(l1=R1, l2=R1a): l3a 支股票
      B→(l1=R1, l2=R1b): l3b 支股票
      C→(l1=R2, l2=R2a): l3c 支股票
    """
    rng = np.random.default_rng(seed)
    rows = []

    def _make_group(l3_code, l3_name, l2_code, l2_name, l1_code, l1_name,
                    n, tradable_n, start_idx):
        for i in range(n):
            rows.append({
                'ts_code': f'{(start_idx + i):06d}.SZ',
                'value': float(rng.normal(10.0 + start_idx, 2.0)),
                'ret': float(rng.normal(0, 0.02)),
                'sw_industry_code': l3_code,
                'sw_industry': l3_name,
                'sw_l2_code': l2_code,
                'sw_l2': l2_name,
                'sw_l1_code': l1_code,
                'sw_l1': l1_name,
                'tradable': 1 if i < tradable_n else 0,
            })

    _make_group('L3A', '子行业A', 'L2R1a', '二级R1a', 'L1R1', '一级R1',
                n_l3a, tradable_l3a, 0)
    _make_group('L3B', '子行业B', 'L2R1a', '二级R1a', 'L1R1', '一级R1',
                n_l3b, tradable_l3b, 100)
    _make_group('L3C', '子行业C', 'L2R2a', '二级R2a', 'L1R2', '一级R2',
                n_l3c, tradable_l3c, 200)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestHierarchicalZscore
# ---------------------------------------------------------------------------

class TestHierarchicalZscore:
    """测试 hierarchical_zscore 分层回退逻辑"""

    def test_l3_sufficient_uses_l3_stats(self):
        """L3 行业内可交易样本 >=5，应使用 L3 统计量（行业内均值接近 0）"""
        df = _make_l3_df(n_l3a=6, tradable_l3a=6)
        result = hierarchical_zscore(
            df, columns=['value'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        assert 'zscore_value' in result.columns
        # L3A 组均值应接近 0（使用 L3 统计）
        l3a_vals = result[result['sw_industry_code'] == 'L3A']['zscore_value'].dropna()
        assert len(l3a_vals) > 0
        assert abs(float(l3a_vals.mean())) < 0.5  # 使用 L3 均值去中心化后应接近0

    def test_l3_insufficient_falls_back_to_l2(self):
        """L3B 行业只有 3 个可交易样本（<5），应回退到 L2（L2R1a 共有 L3A+L3B=8 支）"""
        df = _make_l3_df(n_l3a=5, n_l3b=3, tradable_l3a=5, tradable_l3b=3)
        min_gs = 5
        result = hierarchical_zscore(
            df, columns=['value'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=min_gs,
        )
        # L3B 3 个可交易 < 5，回退到 L2R1a
        # 手动计算 L2R1a 的统计量（L3A+L3B 共 8 个可交易样本）
        l2_tradable = df[
            (df['sw_l2_code'] == 'L2R1a') & (df['tradable'] == 1)
        ]['value'].dropna()
        assert len(l2_tradable) >= min_gs
        l2_mean = float(l2_tradable.mean())
        l2_std = float(l2_tradable.std())

        # L3B 行（应使用 L2 统计）
        l3b_rows = df[df['sw_industry_code'] == 'L3B']
        for idx in l3b_rows.index:
            x = df.loc[idx, 'value']
            expected = (x - l2_mean) / l2_std
            actual = result.loc[idx, 'zscore_value']
            assert not np.isnan(actual), f"L3B 行 {idx} zscore 不应为 NaN"
            assert abs(actual - expected) < 1e-9, (
                f"L3B 行 {idx} 应使用 L2 统计: 期望={expected:.6f}，实际={actual:.6f}"
            )

    def test_l2_insufficient_falls_back_to_l1(self):
        """L3C（L2=L2R2a，L1=L1R2）样本数不足，L2 也不足，应回退到 L1"""
        # L3C 2个可交易，L2R2a 只有 L3C=2个可交易 → L2 也不足
        # L1R2 只有 L3C=2个可交易 → L1 也不足 → 全市场兜底
        df = _make_l3_df(n_l3a=5, n_l3b=5, n_l3c=2,
                         tradable_l3a=5, tradable_l3b=5, tradable_l3c=2)
        result = hierarchical_zscore(
            df, columns=['value'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        # L3C 回退到全市场（L1R2 只有 2 个可交易，L1 也不足）
        global_tradable = df[df['tradable'] == 1]['value'].dropna()
        global_mean = float(global_tradable.mean())
        global_std = float(global_tradable.std())

        l3c_rows = df[df['sw_industry_code'] == 'L3C']
        for idx in l3c_rows.index:
            x = df.loc[idx, 'value']
            expected = (x - global_mean) / global_std
            actual = result.loc[idx, 'zscore_value']
            assert abs(actual - expected) < 1e-9, (
                f"L3C 行 {idx} 应使用全市场统计: 期望={expected:.6f}，实际={actual:.6f}"
            )

    def test_l1_sufficient_falls_back_to_l1(self):
        """当 L3/L2 均不足但 L1 足够时，应使用 L1 统计"""
        # 构造 L3A(5 tradable, L2=L2A, L1=L1) L3B(2 tradable, L2=L2A, L1=L1)
        # L2A 共 7 tradable(<5? No, 7>=5 so it'd use L2... let me make L2 insufficient too)
        # Let's use a custom structure: L3B only 2, L2B only 2, L1 has all 10+ 
        rng = np.random.default_rng(0)
        rows = []
        # Group A: L3=LA, L2=M, L1=TOP → 5 tradable (L3 sufficient)
        for i in range(5):
            rows.append({'ts_code': f'{i:06d}.SZ', 'value': float(rng.normal(10, 2)),
                         'sw_industry_code': 'LA', 'sw_industry': 'l3A',
                         'sw_l2_code': 'M', 'sw_l2': 'l2M',
                         'sw_l1_code': 'TOP', 'sw_l1': 'l1T', 'tradable': 1})
        # Group B: L3=LB, L2=N, L1=TOP → 2 tradable, L2=N only 2 tradable, but L1=TOP has 5+2=7 tradable
        for i in range(5, 9):
            rows.append({'ts_code': f'{i:06d}.SZ', 'value': float(rng.normal(20, 2)),
                         'sw_industry_code': 'LB', 'sw_industry': 'l3B',
                         'sw_l2_code': 'N', 'sw_l2': 'l2N',
                         'sw_l1_code': 'TOP', 'sw_l1': 'l1T',
                         'tradable': 1 if i < 7 else 0})  # 只有 5,6 可交易 = 2 tradable

        df = pd.DataFrame(rows)
        result = hierarchical_zscore(
            df, columns=['value'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        # LB: L3=2(<5)，L2=N 2tradable(<5)，L1=TOP=7tradable(>=5) → 使用 L1 统计
        l1_tradable = df[(df['sw_l1_code'] == 'TOP') & (df['tradable'] == 1)]['value'].dropna()
        l1_mean = float(l1_tradable.mean())
        l1_std = float(l1_tradable.std())

        lb_rows = df[df['sw_industry_code'] == 'LB']
        for idx in lb_rows.index:
            x = df.loc[idx, 'value']
            expected = (x - l1_mean) / l1_std
            actual = result.loc[idx, 'zscore_value']
            assert abs(actual - expected) < 1e-9, (
                f"LB 行 {idx} 应使用 L1 统计: 期望={expected:.6f}，实际={actual:.6f}"
            )

    def test_tradable_only_affects_stats(self):
        """非可交易股票不参与统计量计算，但仍会得到 zscore"""
        df = _make_l3_df(n_l3a=6, tradable_l3a=5)  # L3A 有1只不可交易
        result = hierarchical_zscore(
            df, columns=['value'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        # 不可交易股票也应有 zscore 值（用可交易的统计量计算）
        non_tradable = result[result['tradable'] == 0]
        if len(non_tradable) > 0:
            assert not non_tradable['zscore_value'].isna().all(), \
                "不可交易股票也应得到 zscore 值"


# ---------------------------------------------------------------------------
# TestHierarchicalDemean
# ---------------------------------------------------------------------------

class TestHierarchicalDemean:
    """测试 hierarchical_demean 分层回退逻辑"""

    def test_l3_sufficient_uses_l3_mean(self):
        """L3 样本充足时，去均值后行业内均值应接近 0"""
        df = _make_l3_df(n_l3a=6, tradable_l3a=6)
        result = hierarchical_demean(
            df, columns=['ret'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        assert 'neu_ret' in result.columns
        l3a_tradable = result[
            (result['sw_industry_code'] == 'L3A') & (result['tradable'] == 1)
        ]['neu_ret'].dropna()
        # 均值应近似为 0（因为减去了该组均值）
        assert abs(float(l3a_tradable.mean())) < 1e-9

    def test_l3_insufficient_falls_back_to_l2_demean(self):
        """L3B 不足 → 回退到 L2 均值去均值"""
        df = _make_l3_df(n_l3a=5, n_l3b=3, tradable_l3a=5, tradable_l3b=3)
        result = hierarchical_demean(
            df, columns=['ret'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        # 手动计算 L2R1a 均值
        l2_tradable = df[
            (df['sw_l2_code'] == 'L2R1a') & (df['tradable'] == 1)
        ]['ret'].dropna()
        l2_mean = float(l2_tradable.mean())

        l3b_rows = df[df['sw_industry_code'] == 'L3B']
        for idx in l3b_rows.index:
            expected = df.loc[idx, 'ret'] - l2_mean
            actual = result.loc[idx, 'neu_ret']
            assert abs(actual - expected) < 1e-9, \
                f"L3B 行 {idx} demean 应使用 L2 均值: 期望={expected:.9f}，实际={actual:.9f}"

    def test_tradable_only_affects_mean(self):
        """仅 tradable==1 的样本参与均值计算"""
        rng = np.random.default_rng(1)
        rows = []
        for i in range(7):
            rows.append({
                'ts_code': f'{i:06d}.SZ',
                'value': float(rng.normal(0, 1)),
                'sw_industry_code': 'LA', 'sw_l2_code': 'MA', 'sw_l1_code': 'TA',
                'tradable': 1 if i < 5 else 0,
            })
        df = pd.DataFrame(rows)
        result = hierarchical_demean(
            df, columns=['value'],
            l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
            tradable_col='tradable', min_group_size=5,
        )
        # 统计量应仅基于前 5 个可交易样本
        tradable_mean = float(df[df['tradable'] == 1]['value'].mean())
        for idx in df.index:
            expected = df.loc[idx, 'value'] - tradable_mean
            actual = result.loc[idx, 'neu_value']
            assert abs(actual - expected) < 1e-9, \
                f"行 {idx} demean 应仅使用可交易样本均值"


# ---------------------------------------------------------------------------
# TestFeatureBuilderHierarchicalNeutralization
# ---------------------------------------------------------------------------

class TestFeatureBuilderHierarchicalNeutralization:
    """测试 FeatureBuilder._apply_industry_neutralization 在 L2 主口径下使用分层路径。"""

    def _make_l2_features(self, n=10):
        """构造含 L2/L1 层级信息的特征 DataFrame。"""
        rng = np.random.default_rng(99)
        rows = []
        for i in range(n):
            l2 = 'L2A' if i < n // 2 else 'L2B'
            l1 = 'L1A'
            rows.append({
                'ts_code': f'{i:06d}.SZ',
                'trade_date': '20230101',
                'y_ret_20': float(rng.normal(0, 0.05)),
                'ret_5': float(rng.normal(0, 0.02)),
                'sw_industry': l2,
                'sw_industry_code': l2,
                'sw_l2': l2,
                'sw_l2_code': l2,
                'sw_l1': l1,
                'sw_l1_code': l1,
                'tradable': 1,
            })
        return pd.DataFrame(rows)

    def test_uses_hierarchical_path_when_l2_info_present(self):
        """当 sw_industry_code/sw_l1_code 存在时，应走二级到一级的回退路径。"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5])
        features = self._make_l2_features(n=10)
        result = builder._apply_industry_neutralization(features)

        # 应生成 neu_ 列（去均值）
        assert 'neu_y_ret_20' in result.columns
        assert 'neu_ret_5' in result.columns

    def test_hierarchical_demean_result_in_industry_mean_near_zero(self):
        """每个二级行业内（样本数 >=5）去均值后均值应接近 0。"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5])
        features = self._make_l2_features(n=10)
        result = builder._apply_industry_neutralization(features)

        if 'neu_y_ret_20' in result.columns:
            for industry_code in features['sw_industry_code'].unique():
                grp = result[result['sw_industry_code'] == industry_code]['neu_y_ret_20'].dropna()
                if len(grp) >= 5:
                    assert abs(float(grp.mean())) < 1e-9, \
                        f"二级行业 {industry_code} 去均值后均值应接近 0，实际={grp.mean()}"

    def test_fallback_to_single_level_when_no_l3_codes(self):
        """当 DataFrame 中没有 sw_l2_code/sw_l1_code 时，回退到单层中性化"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5])
        features = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'trade_date': ['20230101'] * 10,
            'y_ret_20': [0.01] * 5 + [0.02] * 5,
            'ret_5': [0.005] * 10,
            'sw_industry': ['L3A'] * 5 + ['L3B'] * 5,
            'tradable': [1] * 10,
        })
        # 无 sw_industry_code/sw_l2_code/sw_l1_code → 单层路径
        result = builder._apply_industry_neutralization(features)
        assert 'neu_y_ret_20' in result.columns
