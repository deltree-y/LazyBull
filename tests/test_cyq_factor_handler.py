"""CyqPerfFactorHandler 单元测试（v0.95.4 未复权口径修复）。

覆盖:
- weight_avg_bias 使用未复权 close 计算（与 weight_avg 同口径）；
- 后复权 close_adj 数值再大也不影响结果（防回归为 adj_factor 代理）；
- current_data 缺 close 时兜底不产出 weight_avg_bias。
"""

import numpy as np
import pandas as pd

from src.lazybull.features.factor_handlers import CyqPerfFactorHandler


def _make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260731"] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "vol": [1.0, 1.0, 1.0],
        }
    )


def _make_cyq_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "winner_rate": [80.0, 20.0, 50.0],
            "weight_avg": [10.0, 20.0, 25.0],
            "cost_concentration": [0.2, 0.3, 0.4],
            "winner_rate_chg_5": [1.0, 2.0, 3.0],
            "winner_rate_chg_20": [4.0, 5.0, 6.0],
        }
    )


def _make_current_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260731"] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            # 未复权收盘价：与 weight_avg 同口径
            "close": [12.0, 24.0, 30.0],
            # 后复权收盘价：历史分红送转放大后与成本价不可比
            "close_adj": [1200.0, 2400.0, 3000.0],
        }
    )


def test_weight_avg_bias_uses_unadjusted_close():
    """偏离度按未复权 close 计算：(close - weight_avg) / weight_avg"""
    handler = CyqPerfFactorHandler()
    result = handler.apply(_make_features(), _make_cyq_data(), "20260731", _make_current_data())
    bias = result["weight_avg_bias"].values
    # (12-10)/10=0.2, (24-20)/20=0.2, (30-25)/25=0.2
    np.testing.assert_allclose(bias, [0.2, 0.2, 0.2], atol=1e-9)


def test_weight_avg_bias_not_corrupted_by_close_adj():
    """后复权价在 features/current_data 中恒存在，但不能混入口径。

    若误用 close_adj 计算，偏离度会是数十倍量级（1200/10-1=119 等）。
    """
    handler = CyqPerfFactorHandler()
    result = handler.apply(_make_features(), _make_cyq_data(), "20260731", _make_current_data())
    bias = result["weight_avg_bias"].values
    assert (np.abs(bias) < 1.0).all()


def test_fallback_without_close_drops_weight_avg_bias():
    """current_data 缺 close 时兜底：不产出 weight_avg_bias，其余列正常。"""
    handler = CyqPerfFactorHandler()
    current_data = _make_current_data().drop(columns=["close"])
    result = handler.apply(_make_features(), _make_cyq_data(), "20260731", current_data)
    assert "weight_avg_bias" not in result
    assert "winner_rate" in result
    assert "cost_concentration" in result
