# -*- coding: utf-8 -*-
"""可选因子组缓存 schema 契约测试。"""

import pandas as pd
import pytest

from src.lazybull.data import Storage
from src.lazybull.factors.cashflow_quality import (
    CASHFLOW_QUALITY_SCHEMA_VERSION,
    CASHFLOW_QUALITY_VERSION_COL,
)
from src.lazybull.factors.consensus_revision import (
    CONSENSUS_REVISION_SCHEMA_VERSION,
    CONSENSUS_REVISION_VERSION_COL,
)
from src.lazybull.factors.dividend import (
    DIVIDEND_POLICY_SCHEMA_VERSION,
    DIVIDEND_POLICY_VERSION_COL,
)
from src.lazybull.features.ensure.schema import (
    _BASE_REQUIRED_FACTOR_COLS,
    _OPTIONAL_FACTOR_REQUIRED_COLS,
    OPTIONAL_FACTOR_GROUP_CASHFLOW_QUALITY,
    OPTIONAL_FACTOR_GROUP_CONSENSUS_REVISION,
    OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY,
    _check_features_schema,
)

_GROUP_CASES = [
    (
        OPTIONAL_FACTOR_GROUP_CASHFLOW_QUALITY,
        CASHFLOW_QUALITY_VERSION_COL,
        CASHFLOW_QUALITY_SCHEMA_VERSION,
    ),
    (
        OPTIONAL_FACTOR_GROUP_CONSENSUS_REVISION,
        CONSENSUS_REVISION_VERSION_COL,
        CONSENSUS_REVISION_SCHEMA_VERSION,
    ),
    (
        OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY,
        DIVIDEND_POLICY_VERSION_COL,
        DIVIDEND_POLICY_SCHEMA_VERSION,
    ),
]


def _write_cache(storage: Storage, frame: pd.DataFrame) -> None:
    cache_dir = storage.features_path / "cs_train"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_dir / "20240610.parquet", index=False)


def test_disabled_optional_groups_do_not_invalidate_cache(tmp_path):
    """构建开关关闭时，缓存不应被未产出的可选因子列永久淘汰。"""
    storage = Storage(str(tmp_path))
    frame = pd.DataFrame({column: [0.0] for column in _BASE_REQUIRED_FACTOR_COLS})
    _write_cache(storage, frame)

    assert _check_features_schema(
        storage,
        "20240610",
        required_optional_groups=set(),
    )
    assert not _check_features_schema(storage, "20240610")


@pytest.mark.parametrize("group,version_col,current_version", _GROUP_CASES)
def test_enabled_optional_group_requires_current_sentinel(
    tmp_path,
    group,
    version_col,
    current_version,
):
    """三类可选组都必须同时满足列完整和哨兵版本契约。"""
    storage = Storage(str(tmp_path))
    frame = pd.DataFrame({column: [0.0] for column in _BASE_REQUIRED_FACTOR_COLS})
    for column in _OPTIONAL_FACTOR_REQUIRED_COLS[group]:
        frame[column] = 0.0
    frame[version_col] = current_version - 1
    _write_cache(storage, frame)

    required_groups = {group}
    assert not _check_features_schema(
        storage,
        "20240610",
        required_optional_groups=required_groups,
    )

    frame[version_col] = current_version
    _write_cache(storage, frame)
    assert _check_features_schema(
        storage,
        "20240610",
        required_optional_groups=required_groups,
    )


def test_dividend_group_requires_all_output_and_training_columns(tmp_path):
    """当前哨兵不能掩盖残缺的分红原始列或训练实际使用列。"""
    required_dividend_columns = {
        "dividend_continuity_5y",
        "dividend_stability_5y",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_payout_ratio",
        "dividend_yield_hist_12m",
        "dividend_days_to_ex_date",
        "dividend_recent_imp_ann_10d",
        "zscore_dividend_continuity_5y",
        "zscore_dividend_stability_5y",
        "zscore_dividend_growth_3y",
        "zscore_dividend_growth_5y",
        "zscore_dividend_payout_ratio",
        "zscore_dividend_yield_hist_12m",
        "dividend_freshness_days",
        "dividend_hist_missing",
        DIVIDEND_POLICY_VERSION_COL,
    }
    declared_columns = set(_OPTIONAL_FACTOR_REQUIRED_COLS[OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY])
    assert required_dividend_columns == declared_columns

    storage = Storage(str(tmp_path))
    frame = pd.DataFrame({column: [0.0] for column in _BASE_REQUIRED_FACTOR_COLS})
    for column in declared_columns:
        frame[column] = 0.0
    frame[DIVIDEND_POLICY_VERSION_COL] = DIVIDEND_POLICY_SCHEMA_VERSION

    for missing_column in required_dividend_columns:
        _write_cache(storage, frame.drop(columns=[missing_column]))
        assert not _check_features_schema(
            storage,
            "20240610",
            required_optional_groups={OPTIONAL_FACTOR_GROUP_DIVIDEND_POLICY},
        )
