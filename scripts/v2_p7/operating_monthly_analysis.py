"""经营月报研究的实际资金账合计与无硬门槛读数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.v2_p7.buffer_path_metrics import arm_metrics
from scripts.v2_p7.operating_monthly_baseline import combine_independent_daily
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics


def normalize_monthly_daily(frame: pd.DataFrame) -> list[dict]:
    """压力负债从应收中扣除，合计保持净资产恒等式；保留原始毛账供核验。"""
    result = frame.copy()
    result["receivable_gross"] = result.receivable
    result["receivable"] = result.receivable - result.dividend_haircut_liability
    return result.to_dict("records")


def cash_daily(calendar: list[str], capital: int) -> list[dict]:
    """独立不计息现金账户，不缩放任何交易账户。"""
    return [
        {
            "trade_date": day,
            "cash": capital,
            "total_value": capital,
            "nav": 1.0,
            "market_value": 0.0,
            "receivable": 0.0,
            "fees": 0.0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
        }
        for day in calendar
    ]


def validate_path(rows: list[dict]) -> pd.DataFrame:
    """逐日资产与现金非负检查；拒绝交集对齐或删除坏点。"""
    frame = pd.DataFrame(rows)
    days = frame.trade_date.astype(str).tolist()
    if len(days) < 2 or days != sorted(set(days)):
        raise ValueError("资金路径必须包含现金锚点且日期唯一升序")
    required = [
        "cash",
        "market_value",
        "receivable",
        "total_value",
        "nav",
        "fees",
        "buy_amount",
        "sell_amount",
    ]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("资金路径含非有限数")
    if frame.cash.min() < -1e-6 or frame.receivable.min() < -1e-6 or frame.nav.min() <= 0:
        raise ValueError("资金路径现金/净应收为负或净值非正")
    residual = frame.total_value - frame[["cash", "market_value", "receivable"]].sum(axis=1)
    if residual.abs().max() > 1e-6:
        raise ValueError("资金路径资产恒等式不成立")
    capital = float(frame.total_value.iloc[0])
    if frame.nav.iloc[0] != 1 or abs(frame.cash.iloc[0] - capital) > 1e-6:
        raise ValueError("缺少初始全现金锚点")
    if not np.allclose(frame.nav, frame.total_value / capital, rtol=0, atol=1e-12):
        raise ValueError("NAV与实际资金不一致")
    return frame


def _windows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split = frame.index[frame.trade_date.ge("20260203")]
    if len(split) == 0 or int(split[0]) == 0:
        raise ValueError("路径缺少已登记历史分界及前一收盘点")
    at = int(split[0])
    return {"full": frame, "early": frame.iloc[:at], "late": frame.iloc[at - 1 :]}


def _metrics(frame: pd.DataFrame) -> dict:
    # 分段首行仅作收益基点，不能把前一天成交费用和换手记入新段。
    work = frame.copy()
    for name in ("fees", "buy_amount", "sell_amount"):
        work.loc[work.index[0], name] = 0.0
    return arm_metrics(work)


def paired_uncertainty(left: np.ndarray, right: np.ndarray, repeats: int = 2000) -> dict:
    """固定20日环形块，同索引配对；仅描述样本依赖，绝非未来成功概率。"""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("配对日收益必须同域")
    if not np.isfinite(np.r_[left, right]).all() or min(left.min(), right.min()) <= -1:
        raise ValueError("配对日收益非法")
    rng = np.random.default_rng(20261116)
    n, block, rows = len(left), 20, []
    for _ in range(repeats):
        starts = rng.integers(0, n, (n + block - 1) // block)
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        metrics = [
            calculate_chain_metrics(pd.DataFrame({"nav": np.r_[1.0, np.cumprod(1 + x[idx])]}))
            for x in (left, right)
        ]
        rows.append([metrics[1][key] - metrics[0][key] for key in ("cagr", "max_drawdown")])
    quantiles = np.quantile(rows, [0.025, 0.5, 0.975], axis=0)
    return {
        "method": "paired_circular_block",
        "block": block,
        "repeats": repeats,
        "seed": 20261116,
        "delta_cagr_quantiles": dict(zip(("q025", "q50", "q975"), quantiles[:, 0].tolist())),
        "delta_mdd_quantiles": dict(zip(("q025", "q50", "q975"), quantiles[:, 1].tolist())),
        "future_success_probability": None,
        "calibrated_for_admission": False,
    }


def analyze_capital(
    *,
    capital: int,
    a_full: list[dict],
    a_partial: list[dict],
    b_accounts: dict[str, list[dict]],
    bootstrap: bool = True,
) -> tuple[dict, dict[str, list[dict]]]:
    """同资本真实A账户对照，不依据阈值返回通过/失败。"""
    paths = {"A100": a_full}
    for name, rows in b_accounts.items():
        paths[f"A80_B_{name}"] = combine_independent_daily(
            a_partial,
            rows,
            total_capital=capital,
        )
    frames = {name: validate_path(rows) for name, rows in paths.items()}
    expected = frames["A100"].trade_date.tolist()
    if any(frame.trade_date.tolist() != expected for frame in frames.values()):
        raise ValueError("对照日期不完全一致")
    metrics = {
        name: {period: _metrics(part) for period, part in _windows(frame).items()}
        for name, frame in frames.items()
    }
    differences = {}
    for name in metrics:
        if name == "A100":
            continue
        differences[name] = {
            period: {
                key: metrics[name][period][key] - metrics["A100"][period][key]
                for key in ("cagr", "max_drawdown", "total_return", "fees_yuan")
            }
            for period in ("full", "early", "late")
        }
    result: dict[str, Any] = {
        "capital": capital,
        "metrics": metrics,
        "differences_vs_A100": differences,
        "formal_admission": "not_evaluated",
        "hard_economic_thresholds": None,
        "historical_sample_reused": True,
        "future_success_probability": None,
    }
    result["b_standalone"] = {
        name: _metrics(validate_path(rows)) for name, rows in b_accounts.items()
    }
    if bootstrap:
        main = frames["A80_B_signal"].nav.pct_change().dropna().to_numpy()
        result["uncertainty"] = {
            name: paired_uncertainty(frame.nav.pct_change().dropna().to_numpy(), main)
            for name, frame in frames.items()
            if name in ("A100", "A80_B_cash", "A80_B_equal")
        }
    return result, paths
