"""实际资金合计、分段与负债敏感性的定向验证。"""

import pandas as pd
import pytest

from scripts.v2_p7.operating_monthly_analysis import (
    analyze_capital,
    cash_daily,
    normalize_monthly_daily,
    paired_uncertainty,
    validate_path,
)


def test_cash_identity_and_unknown_probability():
    days = ["20260130", "20260202", "20260203", "20260204"]
    result, _ = analyze_capital(
        capital=1000,
        a_full=cash_daily(days, 1000),
        a_partial=cash_daily(days, 800),
        b_accounts={"signal": cash_daily(days, 200)},
        bootstrap=False,
    )
    assert result["differences_vs_A100"]["A80_B_signal"]["late"]["cagr"] == 0
    assert result["metrics"]["A100"]["late"]["trading_days"] == 2
    assert result["future_success_probability"] is None


def test_liability_not_counted_as_assets():
    row = {
        **cash_daily(["20260130"], 100)[0],
        "receivable": 10,
        "dividend_haircut_liability": 2,
        "total_value": 108,
        "nav": 1.08,
    }
    normalized = normalize_monthly_daily(pd.DataFrame([row]))[0]
    assert normalized["receivable"] == 8
    assert normalized["receivable_gross"] == 10


def test_fail_closed_different_calendars_and_bad_nav():
    rows = cash_daily(["20260130", "20260202"], 100)
    rows[1]["nav"] = 2
    with pytest.raises(ValueError, match="NAV"):
        validate_path(rows)


def test_identical_paths_have_zero_paired_ranges():
    import numpy as np

    returns = np.array([0.01, -0.01, 0.02, -0.015])
    result = paired_uncertainty(returns, returns, repeats=4)
    assert set(result["delta_cagr_quantiles"].values()) == {0.0}
    assert set(result["delta_mdd_quantiles"].values()) == {0.0}
