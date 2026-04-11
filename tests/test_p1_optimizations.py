# -*- coding: utf-8 -*-
"""P1优化测试: 因子增强(2.2)"""
import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml.train_core import (
    ENHANCED_FEATURE_COLUMNS,
)


# ── 2.2 因子增强 ──────────────────────────────────────────────

class TestEnhancedFeatures:
    """测试增强因子常量定义"""

    def test_enhanced_feature_columns_count(self):
        """增强因子应包含5个列"""
        assert len(ENHANCED_FEATURE_COLUMNS) == 5

    def test_enhanced_feature_columns_content(self):
        """增强因子应包含开盘强度、日内波动结构和委托不平衡"""
        assert "zscore_opening_strength" in ENHANCED_FEATURE_COLUMNS
        assert "zscore_intraday_vol_structure" in ENHANCED_FEATURE_COLUMNS
        assert "zscore_order_imbalance" in ENHANCED_FEATURE_COLUMNS
        assert "order_imbalance_mean_5" in ENHANCED_FEATURE_COLUMNS
        assert "order_imbalance_mean_20" in ENHANCED_FEATURE_COLUMNS
