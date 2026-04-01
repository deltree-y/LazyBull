#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_clean_features.py 参数辅助逻辑测试。"""

import types

from scripts.build_clean_features import (
    OPTIONAL_FEATURE_FLAG_ATTRS,
    apply_build_all_feature_flags,
)


class TestBuildAllFeatureFlags:
    """测试 --build-all 开关行为。"""

    def test_build_all_turns_on_all_optional_feature_flags(self):
        args = types.SimpleNamespace(
            build_all=True,
            enable_fundamental_features=False,
            enable_alt_features=False,
            enable_margin_features=False,
            enable_cyq_features=False,
            enable_fund_features=False,
            enable_express_features=False,
            enable_industry_neutralization=False,
        )

        result = apply_build_all_feature_flags(args)

        for attr in OPTIONAL_FEATURE_FLAG_ATTRS:
            assert getattr(result, attr) is True
        assert result.enable_industry_neutralization is False

    def test_without_build_all_keeps_existing_flag_values(self):
        args = types.SimpleNamespace(
            build_all=False,
            enable_fundamental_features=True,
            enable_alt_features=False,
            enable_margin_features=True,
            enable_cyq_features=False,
            enable_fund_features=True,
            enable_express_features=False,
        )

        result = apply_build_all_feature_flags(args)

        assert result.enable_fundamental_features is True
        assert result.enable_alt_features is False
        assert result.enable_margin_features is True
        assert result.enable_cyq_features is False
        assert result.enable_fund_features is True
        assert result.enable_express_features is False