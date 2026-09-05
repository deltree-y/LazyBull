"""Walk-forward 批量最小因子实验配置测试。"""

import json
from pathlib import Path
from typing import List

from src.lazybull.ml.train_core.constants import (
    CASHFLOW_QUALITY_FEATURE_COLUMNS,
    DIVIDEND_POLICY_FEATURE_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_excludes(relative_path: str) -> List[str]:
    payload = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    assert payload["exclude_count"] == len(payload["exclude_factors"])
    return payload["exclude_factors"]


def test_minimal_factor_exclude_lists_leave_only_intended_base_features() -> None:
    """两个候选清单应分别只保留股息率和两组现金流基础列。"""
    dividend_excludes = set(_load_excludes("configs/factor_exclude_dividend_yield_only_v1.json"))
    cashflow_excludes = set(_load_excludes("configs/factor_exclude_cashflow_keep_2pairs_v1.json"))

    dividend_live = [
        column for column in DIVIDEND_POLICY_FEATURE_COLUMNS if column not in dividend_excludes
    ]
    cashflow_live = [
        column for column in CASHFLOW_QUALITY_FEATURE_COLUMNS if column not in cashflow_excludes
    ]

    assert dividend_live == ["zscore_dividend_yield_hist_12m"]
    assert cashflow_live == ["zscore_ocf_to_revenue", "zscore_fcf_yield"]
