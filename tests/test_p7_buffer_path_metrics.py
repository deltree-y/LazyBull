"""续持路径指标的分母、同域、跨年和配对重抽样语义测试。"""

import numpy as np
import pytest

from scripts.v2_p7.buffer_path_metrics import (
    annual_rows,
    compare_accounts,
    goals,
    paired_bootstrap,
    validate_daily,
)


def rows():
    """小型合成连续账户，日期跨年与模型折界无关。"""
    return [
        dict(
            trade_date=day,
            nav=nav,
            cash=100,
            total_value=1000 * nav,
            market_value=1000 * nav - 100,
            receivable=0,
            fees=0,
            buy_amount=0,
            sell_amount=0,
            split_index=split,
        )
        for day, nav, split in [
            ("20241230", 1.0, 0),
            ("20241231", 1.1, 0),
            ("20250102", 0.99, 1),
            ("20250103", 1.188, 1),
        ]
    ]


def test_continuous_account_keeps_fold_boundary_return():
    """连续账户跨折的-10%不能被通用指标的split过滤掉。"""
    result = compare_accounts(rows(), rows(), bootstrap=False)
    assert result["base"]["trading_days"] == 3
    assert result["base"]["max_drawdown"] == pytest.approx(-0.1)
    assert result["delta"]["cagr"] == 0


def test_annual_includes_new_year_first_return():
    """次年以去年末而非次年首日为分母。"""
    annual = annual_rows(validate_daily(rows()))
    assert annual[0]["return"] == pytest.approx(0.1)
    assert annual[1]["return"] == pytest.approx(0.08)


def test_unpaired_domains_and_negative_cash_rejected():
    """不允许缩日期交集或用借款伪装现货现金账户。"""
    changed = rows()
    changed[-1]["trade_date"] = "20250106"
    with pytest.raises(ValueError, match="日期"):
        compare_accounts(rows(), changed, bootstrap=False)
    changed[-1]["cash"] = -1
    with pytest.raises(ValueError, match="现金为负"):
        validate_daily(changed)


def test_shared_bootstrap_identical_arms_stay_equal():
    """相同路径同块抽样后差值必须仍为零，不能两臂独立抽样。"""
    values = np.array([0.01, -0.02, 0.03, -0.01, 0.01])
    result = paired_bootstrap(values, values, repeats=20, block=3)
    assert set(result["delta_cagr_quantiles"].values()) == {0}
    assert set(result["delta_mdd_quantiles"].values()) == {0}
    assert result["future_success_probability"] is None


def test_goal_signs_and_zero_drawdown():
    """回撤负数的改善为正差；零回撤基准不能相对降低25%。"""
    assert goals(0.005, -0.01, -0.2, -0.21)["return_goal"]
    assert not goals(0.005, -0.011, -0.2, -0.211)["return_goal"]
    assert goals(-0.02, 0.05, -0.2, -0.15)["drawdown_goal"]
    assert not goals(0, 0, 0, 0)["drawdown_goal"]
