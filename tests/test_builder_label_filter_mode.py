"""测试 FeatureBuilder 的 label_filter_mode 双模式过滤逻辑

覆盖场景：
- single 模式：仅按主 horizon 对应的 y_ret_N 非空过滤
- all 模式：按所有 horizons 对应的 y_ret_N 同时非空过滤（AND 语义）
- 参数校验：非法 label_filter_mode 抛 ValueError
- horizon 推断告警：传入 horizon 不在 horizons 中时自动追加
"""

import pandas as pd
import pytest

from src.lazybull.features import FeatureBuilder


def _make_fake_df() -> pd.DataFrame:
    """构造一个覆盖停牌场景的测试样本

    4 条样本，全部符合 is_st=0 / list_days 足够 / is_suspended=0：
    - 样本 A: y_ret_5/10/20 全有值 → 两种模式都应保留
    - 样本 B: y_ret_10 缺失（模拟 T+10 停牌），y_ret_5 和 y_ret_20 有值
      → single(20) 保留，all 丢弃
    - 样本 C: y_ret_20 缺失（模拟 T+20 停牌）
      → single(20) 丢弃，all 丢弃
    - 样本 D: y_ret_5 缺失（模拟 T+5 停牌），y_ret_10 和 y_ret_20 有值
      → single(20) 保留，all 丢弃
    """
    return pd.DataFrame({
        'ts_code': ['A', 'B', 'C', 'D'],
        'is_st': [0, 0, 0, 0],
        'list_days': [1000, 1000, 1000, 1000],
        'is_suspended': [0, 0, 0, 0],
        'y_ret_5':  [0.01, 0.02, 0.03, None],
        'y_ret_10': [0.01, None, 0.03, 0.04],
        'y_ret_20': [0.01, 0.02, None, 0.04],
    })


class TestLabelFilterMode:
    """测试 single / all 两种标签过滤模式的行为差异"""

    def test_single_mode_only_filters_primary_horizon(self):
        """single 模式下只按主 horizon (y_ret_20) 过滤，停牌导致的辅助标签缺失应被保留"""
        builder = FeatureBuilder(
            horizon=20,
            horizons=[5, 10, 20],
            require_label=True,
            label_filter_mode="single",
        )
        df = _make_fake_df()
        result = builder._apply_filters(df)

        # A/B/D 的 y_ret_20 都有值 → 保留；C 的 y_ret_20 缺失 → 丢弃
        assert set(result['ts_code'].tolist()) == {'A', 'B', 'D'}

    def test_all_mode_requires_all_horizons_notna(self):
        """all 模式下要求所有 horizons 对应的 y_ret_N 同时非空"""
        builder = FeatureBuilder(
            horizon=20,
            horizons=[5, 10, 20],
            require_label=True,
            label_filter_mode="all",
        )
        df = _make_fake_df()
        result = builder._apply_filters(df)

        # 只有 A 三列都非空
        assert set(result['ts_code'].tolist()) == {'A'}

    def test_default_mode_is_all(self):
        """默认 label_filter_mode 应为 'all'（保持向后兼容）"""
        builder = FeatureBuilder(
            horizon=20,
            horizons=[5, 10, 20],
            require_label=True,
        )
        assert builder.label_filter_mode == "all"

        df = _make_fake_df()
        result = builder._apply_filters(df)
        assert set(result['ts_code'].tolist()) == {'A'}

    def test_single_mode_with_nonstandard_primary_horizon(self):
        """single 模式用非默认主 horizon（如 10）时按对应列过滤"""
        builder = FeatureBuilder(
            horizon=10,
            horizons=[5, 10, 20],
            require_label=True,
            label_filter_mode="single",
        )
        df = _make_fake_df()
        result = builder._apply_filters(df)

        # y_ret_10 非空的是 A/C/D
        assert set(result['ts_code'].tolist()) == {'A', 'C', 'D'}

    def test_invalid_label_filter_mode_raises(self):
        """非法 label_filter_mode 值应抛 ValueError"""
        with pytest.raises(ValueError, match="label_filter_mode"):
            FeatureBuilder(
                horizon=20,
                horizons=[5, 10, 20],
                require_label=True,
                label_filter_mode="invalid_mode",
            )

    def test_horizon_not_in_horizons_auto_appended(self):
        """传入 horizon 不在 horizons 中时，自动追加到 horizons"""
        builder = FeatureBuilder(
            horizon=15,
            horizons=[5, 10, 20],
            require_label=False,
        )
        # horizons 应被追加并排序
        assert 15 in builder.horizons
        assert builder.horizons == sorted(builder.horizons)
        assert builder.horizon == 15

    def test_require_label_false_bypasses_label_filter(self):
        """require_label=False 时标签缺失样本应全部保留（推理/实盘模式）"""
        builder = FeatureBuilder(
            horizon=20,
            horizons=[5, 10, 20],
            require_label=False,
            label_filter_mode="single",  # 即便设了 single 也应被 require_label=False 短路
        )
        df = _make_fake_df()
        result = builder._apply_filters(df)

        # 所有样本 is_st=0 / 足够上市天数 / 未停牌，标签过滤被跳过 → 全保留
        assert set(result['ts_code'].tolist()) == {'A', 'B', 'C', 'D'}
