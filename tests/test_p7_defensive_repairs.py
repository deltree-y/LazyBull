"""评审 F01/F02/C2/C4/F04-1/F06 的反例测试（修复前应失败，修复后必须通过）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import b_replay, evaluation, features, labels, universe


def _pit_row(code: str, in_date: str, out_date, l1_code: str, l1_name) -> dict:
    return {
        "ts_code": code,
        "in_date": in_date,
        "out_date": out_date,
        "l1_code": l1_code,
        "l1_name": l1_name,
    }


def test_unmapped_legacy_code_is_unknown_not_non_financial() -> None:
    """F01：旧码无可核一级名称 ⇒ 身份未知（不得当作非金融入域）。"""
    pit = pd.DataFrame(
        [
            _pit_row("000001.SZ", "20100101", None, "440000", None),
            _pit_row("600030.SH", "20100101", None, "440000", None),
        ]
    )
    names = universe.industry_l1_names(None, pit)
    known, financial = universe._industry_masks(
        pit, ["20200102"], ["000001.SZ", "600030.SH"], names
    )
    assert not known.any()
    assert not financial.any()


def test_financial_code_is_flagged_even_without_name() -> None:
    """F01：已知金融一级代码必须判金融（不能被 UNKNOWN 吞掉）。"""
    pit = pd.DataFrame([_pit_row("000001.SZ", "20100101", None, "480000", None)])
    names = universe.industry_l1_names(None, pit)
    known, financial = universe._industry_masks(pit, ["20200102"], ["000001.SZ"], names)
    assert financial.all()
    assert known.all()


def test_code_table_resolves_name_when_pit_name_missing() -> None:
    """F01：代码表可核名称优先，非金融旧记录不得被排除。"""
    pit = pd.DataFrame([_pit_row("000002.SZ", "20100101", None, "450000", None)])
    table = pd.DataFrame(
        {
            "src": ["SW2014"],
            "industry_code": ["450000"],
            "level": ["L1"],
            "industry_name": ["商业贸易"],
            "index_code": ["x"],
        }
    )
    names = universe.industry_l1_names(table, pit)
    known, financial = universe._industry_masks(pit, ["20200102"], ["000002.SZ"], names)
    assert known.all()
    assert not financial.any()


def _paired_series(n: int = 160, seed: int = 7, time_varying: bool = True):
    rng = np.random.default_rng(seed)
    bench = rng.normal(0.0, 0.01, size=(n, 1))
    beta = np.linspace(0.5, 1.8, n).reshape(-1, 1) if time_varying else np.full((n, 1), 1.1)
    noise = rng.normal(0.0, 0.02, size=(n, 1))
    stock = 0.0003 + beta * bench + noise
    index = [f"d{i:04d}" for i in range(n)]
    return (
        pd.DataFrame(stock, index=index, columns=["A"]),
        pd.DataFrame(bench, index=index, columns=["A"]),
    )


def _direct_ols_idiovol(stock: np.ndarray, bench: np.ndarray) -> float:
    design = np.column_stack([np.ones_like(bench), bench])
    coef, *_ = np.linalg.lstsq(design, stock, rcond=None)
    resid = stock - design @ coef
    return float(np.std(resid, ddof=1))


def test_idiovol_uses_fixed_window_coefficients() -> None:
    """F02：必须用同窗固定 alpha_T/beta_T 的残差标准差（β 时变时与逐日口径不同）。"""
    stock, bench = _paired_series(time_varying=True)
    ols = features._ols_beta(stock, bench, features.MIN_OBS["w120"])
    got = features._idiovol(stock, bench, ols, features.MIN_OBS["w120"])
    row = len(stock) - 1
    expected = _direct_ols_idiovol(
        stock.iloc[-120:, 0].to_numpy(), bench.iloc[-120:, 0].to_numpy()
    )
    assert got.iloc[row, 0] == pytest.approx(expected, rel=1e-10)

    # 用另一时点的固定系数（T-10 窗口）算同窗残差，数值必须不同 ⇒ 测试能抓住错系数
    alt = _direct_ols_idiovol_with(
        stock.iloc[-130:-10, 0].to_numpy(),
        bench.iloc[-130:-10, 0].to_numpy(),
        stock.iloc[-120:, 0].to_numpy(),
        bench.iloc[-120:, 0].to_numpy(),
    )
    assert abs(alt - expected) > 1e-6


def _direct_ols_idiovol_with(fit_s, fit_b, apply_s, apply_b) -> float:
    design = np.column_stack([np.ones_like(fit_b), fit_b])
    coef, *_ = np.linalg.lstsq(design, fit_s, rcond=None)
    resid = apply_s - (coef[0] + coef[1] * apply_b)
    return float(np.std(resid, ddof=1))


def test_idiovol_nan_below_min_pairs() -> None:
    """F02：配对样本不足 96 时必须 NaN。"""
    stock, bench = _paired_series(n=90)
    ols = features._ols_beta(stock, bench, features.MIN_OBS["w120"])
    got = features._idiovol(stock, bench, ols, features.MIN_OBS["w120"])
    assert np.isnan(got.iloc[-1, 0])


def test_amihud_requires_sixteen_observations() -> None:
    """C2：20 日窗口 80% 门槛是 16，不是 15。"""
    index = [f"d{i:03d}" for i in range(40)]
    returns = pd.DataFrame(0.01, index=index, columns=["A"])
    amount = pd.DataFrame(1.0e8, index=index, columns=["A"])
    returns.iloc[20:24, 0] = np.nan  # 最近 20 日剩 16 个有效 -> 可算
    assert np.isfinite(features._amihud(returns, amount).iloc[-1, 0])
    returns.iloc[24, 0] = np.nan  # 剩 15 个有效 -> NaN
    assert np.isnan(features._amihud(returns, amount).iloc[-1, 0])


def test_label_dates_not_clamped_to_watermark() -> None:
    """C4：超出日历的计划日必须显式未知，不得截断成水位末日。"""
    calendar = ["20260101", "20260102", "20260105"]
    book = {
        "y_def_ret60": pd.DataFrame([[np.nan]]),
        "label_status": pd.DataFrame([[labels.STATUSES.index("right_censored")]]),
        "entry_execution_blocked": pd.DataFrame([[False]]),
        "exit_execution_blocked": pd.DataFrame([[False]]),
    }
    frame = labels.extract_label_frame(
        book, np.array([[True]]), ["20260101"], ["A"], calendar
    )
    assert frame.loc[0, "entry_date"] == "20260102"
    assert frame.loc[0, "label_end_date"] == ""
    assert not bool(frame.loc[0, "label_dates_in_calendar"])


def test_rank_lag_uses_past_ranks_not_future() -> None:
    """F04-1：延迟压力必须在 T 日使用 T-lag 的排名（禁止把未来排名搬到过去）。"""
    ranks = pd.DataFrame(
        {
            "trade_date": ["20200102", "20200103"],
            "ts_code": ["PAST", "FUTURE"],
            "rank": [1, 1],
        }
    )
    shifted = b_replay._rank_lists(ranks, 1, ["20200102", "20200103"])
    assert "20200102" not in shifted
    assert shifted["20200103"] == ["PAST"]


def test_bootstrap_keeps_first_return() -> None:
    """F06：重排后的 NAV 必须保留期初基点，否则首个收益被漏计。"""
    metrics = evaluation._metrics_from_returns(np.array([-0.2, 0.0, 0.0]))
    assert metrics["max_drawdown"] == pytest.approx(-0.2, abs=1e-9)
