"""测试 FeatureBuilder.clear_caches() 方法"""

import pandas as pd
import pytest

from src.lazybull.features import FeatureBuilder


class TestClearCaches:
    """测试缓存清理"""

    def test_clear_caches_sets_all_to_none(self):
        """clear_caches 应将所有缓存属性设为 None"""
        builder = FeatureBuilder(horizon=20, require_label=False)

        # 模拟填充缓存
        builder._market_state_cache = pd.DataFrame({'a': [1]})
        builder._tech_factor_cache = pd.DataFrame({'b': [2]})
        builder._tech_factor_cache_dict = {'20230101': pd.DataFrame()}
        builder._trading_dates_cache = ['20230101', '20230102']
        builder._trading_date_index = {'20230101': 0, '20230102': 1}
        builder._daily_adj_precomputed = pd.DataFrame({'c': [3]})
        builder._daily_adj_dict = {'20230101': pd.DataFrame()}

        builder.clear_caches()

        assert builder._market_state_cache is None
        assert builder._tech_factor_cache is None
        assert builder._tech_factor_cache_dict is None
        assert builder._trading_dates_cache is None
        assert builder._trading_date_index is None
        assert builder._daily_adj_precomputed is None
        assert builder._daily_adj_dict is None

    def test_clear_caches_noop_when_empty(self):
        """缓存本就为空时 clear_caches 应安全执行"""
        builder = FeatureBuilder(horizon=20, require_label=False)
        # 所有缓存默认为 None，clear_caches 不应报错
        builder.clear_caches()

        assert builder._market_state_cache is None
        assert builder._tech_factor_cache is None

    def test_clear_caches_partial(self):
        """仅部分缓存有值时也能正确清理"""
        builder = FeatureBuilder(horizon=20, require_label=False)
        builder._market_state_cache = pd.DataFrame({'a': [1]})
        builder._trading_dates_cache = ['20230101']

        builder.clear_caches()

        assert builder._market_state_cache is None
        assert builder._trading_dates_cache is None
        # 其余本就是 None
        assert builder._tech_factor_cache is None
