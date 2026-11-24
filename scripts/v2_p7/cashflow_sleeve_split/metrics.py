# -*- coding: utf-8 -*-
"""run-a 读数与交付（协议 §7.1/§7.3/§7.6；审点二 A 批）。

口径（冻结）
============
- 链式指标唯一实现 = ``v2/common/chain_metrics.py::calculate_chain_metrics``；
  调用前断言日期轴完整（锚..末日）、NAV 有限且为正；**不带 ``split_index``**；
  主表 ``mdd_loss = -max_drawdown``（底层为负值）。
- 分段继承前一开市日权益（切片含基点）；短段主报实际区间收益，年化仅描述。
- 桥接三行：旧 A100（冻结底账）→ 同态 F100 → R100；差异按未舍入浮点。
- B 终值门槛（§7.3）：目标总权益等线 ``K × 1.25^(N/252)`` − R80 实际期末权益；
  N = 含锚后的有效收益区间数（与链式指标交易日数同口径）。
- 自举（§7.6）：主窗完整配对日收益、20 日循环块、2,000 次、seed 42；R−F 的
  CAGR/MDD 差分位数（组合对属 run-b，``not_run``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT

#: 审点二交付（协议 §8 审点二行）
DELIVERABLES: Tuple[str, ...] = ("a_review.md", "a_metrics.csv", "b_required.csv")

#: 分段（协议 §7.2；含锚基点继承）
SEGMENTS: Dict[str, Tuple[str, str]] = {k: (v[0], v[1]) for k, v in spec.WINDOW["segments"].items()}


class MetricsError(RuntimeError):
    """读数输入缺件或断言失败（fail-fast）。"""


def chain_frame(daily_rows: Sequence[Mapping[str, Any]], label: str) -> pd.DataFrame:
    """账户逐日权益 → 链式净值帧（日期轴完整性断言）。"""
    if not daily_rows:
        raise MetricsError(f"{label} 逐日权益为空")
    frame = pd.DataFrame(
        {
            "trade_date": [str(row["trade_date"]) for row in daily_rows],
            "nav": [float(row["nav"]) for row in daily_rows],
        }
    )
    dates = frame["trade_date"].tolist()
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise MetricsError(f"{label} 日期轴重复或乱序")
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    if not np.isfinite(nav).all() or (nav <= 0).any():
        raise MetricsError(f"{label} NAV 存在非有限或非正值")
    return frame


def chain_metrics(frame: pd.DataFrame, label: str) -> Dict[str, Optional[float]]:
    """链式指标（唯一实现；无 split_index；含锚日）。"""
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    metrics = calculate_chain_metrics(frame, annual_trading_days=252, annual_risk_free_rate=0.03)
    if metrics.get("trading_days") is None:
        raise MetricsError(f"{label} 链式指标不可计算")
    return metrics


def segment_metrics(frame: pd.DataFrame, start: str, end: str) -> Dict[str, Any]:
    """分段读数（切片含前一开市日基点；继承权益与持仓）。"""
    dates = frame["trade_date"].tolist()
    inside = [i for i, day in enumerate(dates) if start <= day <= end]
    if not inside:
        raise MetricsError(f"分段 {start}~{end} 无净值点")
    base_index = inside[0] - 1 if inside[0] > 0 else inside[0]
    last_index = inside[-1] + 1
    window = frame.iloc[base_index:last_index].reset_index(drop=True)
    nav = window["nav"].astype(float)
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cumulative = nav.cummax()
    mdd = float(((nav - cumulative) / cumulative).min())
    intervals = max(len(nav) - 1, 1)
    years = intervals / 252.0
    cagr_descriptive = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    return {
        "segment_start": str(start),
        "segment_end": str(end),
        "base_date": str(window["trade_date"].iloc[0]),
        "nav_points": int(len(nav)),
        "trading_days": int(intervals),
        "total_return": total_return,
        "mdd_loss": -mdd,
        "cagr_descriptive": cagr_descriptive,
    }


def year_segments(dates: Sequence[str]) -> List[Tuple[str, str, str]]:
    """日历年分段（2019~2026；按净值点实际覆盖裁剪）。"""
    out: List[Tuple[str, str, str]] = []
    for year in range(2019, 2027):
        start = f"{year}0101"
        end = f"{year}1231"
        inside = [day for day in dates if start <= day <= end]
        if inside:
            out.append((f"y{year}", inside[0], inside[-1]))
    return out


def account_metrics(daily_rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    """单账户全窗 + 全部分段读数。"""
    frame = chain_frame(daily_rows, label)
    full = chain_metrics(frame, label)
    segments = {
        "full": {
            "trading_days": full["trading_days"],
            "cagr": full["cagr"],
            "mdd_loss": -full["max_drawdown"],
            "sharpe": full["sharpe"],
            "volatility": full["volatility"],
            "total_return": full["total_return"],
        }
    }
    for name, (start, end) in SEGMENTS.items():
        if name == "full":
            continue
        dates_in = [day for day in frame["trade_date"].tolist() if start <= day <= end]
        if not dates_in:
            continue  # 合成短序列/未覆盖段：缺席段不产出读数（真实账户窗全覆盖）
        segments[name] = segment_metrics(frame, start, end)
    for name, start, end in year_segments(frame["trade_date"].tolist()):
        segments[name] = segment_metrics(frame, start, end)
    return {
        "label": label,
        "chain": full,
        "segments": segments,
        "nav_points": int(len(frame)),
        "first_date": str(frame["trade_date"].iloc[0]),
        "last_date": str(frame["trade_date"].iloc[-1]),
        "ending_nav": float(frame["nav"].iloc[-1]),
        # 期末总权益（元）＝ daily 末行 total_value；checkpoint-2 Codex 第 6 项：
        # 此前误写单位净值
        "ending_value": float(daily_rows[-1].get("total_value") or frame["nav"].iloc[-1]),
    }


def read_old_anchor(capital: int) -> Dict[str, Any]:
    """冻结旧 A100 逐日权益（协议 §4 冻结底账；只读）。"""
    path = (
        Path(REPO_ROOT)
        / spec.A_BASELINE["relative"]
        / spec.A_BASELINE["harness"][str(capital)]["a100_daily"]
    )
    if not path.exists():
        raise MetricsError(f"旧 A 锚逐日件缺件：{path}")
    frame = pd.read_parquet(path)
    rows = [
        {"trade_date": str(row["trade_date"]), "nav": float(row["nav"])}
        for row in frame.to_dict("records")
    ]
    return account_metrics(rows, f"oldA100@{capital}")


def _daily_returns(frame: pd.DataFrame) -> np.ndarray:
    nav = frame["nav"].astype(float).to_numpy()
    return nav[1:] / nav[:-1] - 1.0


def bootstrap_diff(
    frame_left: pd.DataFrame,
    frame_right: pd.DataFrame,
    *,
    block: int = 20,
    repetitions: int = 2000,
    seed: int = 42,
    label: str = "",
) -> Dict[str, Any]:
    """配对日收益的 20 日循环块自举（同一组索引；CAGR/MDD 差分位数）。

    两臂日期轴必须逐点一致；重抽只描述路径不确定性。
    """
    dates_l = frame_left["trade_date"].tolist()
    dates_r = frame_right["trade_date"].tolist()
    if dates_l != dates_r:
        raise MetricsError(f"{label} 配对自举日期轴不一致")
    returns_l = _daily_returns(frame_left)
    returns_r = _daily_returns(frame_right)
    n = len(returns_l)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    cagr_diffs = np.empty(repetitions)
    mdd_diffs = np.empty(repetitions)
    for rep in range(repetitions):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]
        cagr_l = float(np.prod(1.0 + returns_l[indices]) ** (252.0 / n) - 1.0)
        cagr_r = float(np.prod(1.0 + returns_r[indices]) ** (252.0 / n) - 1.0)
        curve_l = np.cumprod(1.0 + returns_l[indices])
        curve_r = np.cumprod(1.0 + returns_r[indices])
        mdd_l = float(((curve_l / np.maximum.accumulate(curve_l)) - 1.0).min())
        mdd_r = float(((curve_r / np.maximum.accumulate(curve_r)) - 1.0).min())
        cagr_diffs[rep] = cagr_r - cagr_l
        mdd_diffs[rep] = -mdd_r - (-mdd_l)
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    return {
        "label": label,
        "block": block,
        "repetitions": repetitions,
        "seed": seed,
        "n": n,
        "cagr_diff": {
            "mean": float(cagr_diffs.mean()),
            **{f"p{int(q * 100)}": float(np.quantile(cagr_diffs, q)) for q in quantiles},
            "prob_positive": float((cagr_diffs > 0).mean()),
        },
        "mdd_loss_diff": {
            "mean": float(mdd_diffs.mean()),
            **{f"p{int(q * 100)}": float(np.quantile(mdd_diffs, q)) for q in quantiles},
            "prob_negative": float((mdd_diffs < 0).mean()),
        },
    }


def b_required_row(
    capital: int,
    r80_ending_value: float,
    nav_points: int,
) -> Dict[str, Any]:
    """B 终值门槛算术（协议 §7.3；等线 N 与链式交易日数同口径）。"""
    n = max(nav_points - 1, 1)
    target = float(capital) * (1.25 ** (n / 252.0))
    required_end = target - float(r80_ending_value)
    row = {
        "capital": int(capital),
        "n_intervals": int(n),
        "target_total_value": target,
        "r80_ending_value": float(r80_ending_value),
        "b_required_ending_value": required_end,
        "b_initial_capital": int(capital * 0.2),
        "rule": "目标等线 K×1.25^(N/252) − R80 实际期末权益；CAGR 恰 25% 未达标",
    }
    if required_end > 0:
        row["b_required_cagr_from_0.2k"] = float(
            (required_end / (capital * 0.2)) ** (252.0 / n) - 1.0
        )
        row["status"] = "positive_gap"
    else:
        row["b_required_cagr_from_0.2k"] = None
        row["status"] = "no_gap"
        row["strict_note"] = (
            "R80 期末权益已 ≥ 目标等线；严格不等式（组合 CAGR > 25%）要求 B 期末权益 "
            f"> {required_end:.2f} 元（可为负值上限，即 B 不得拖累组合跌破等线）"
        )
    return row


def r_gate_verdict(cagr: float, mdd_loss: float, f_mdd_loss: float) -> Dict[str, Any]:
    """R 单袖双门判定（协议 §1：净 CAGR ≥26% 且回撤损失严格低于 F）。"""
    return {
        "cagr": float(cagr),
        "mdd_loss": float(mdd_loss),
        "f_mdd_loss": float(f_mdd_loss),
        "cagr_gate_ge_26": bool(cagr >= 0.26),
        "distance_to_26_pp": float(cagr - 0.26),
        "mdd_strictly_below_f": bool(mdd_loss < f_mdd_loss),
        "mdd_gap_pp_vs_f": float(mdd_loss - f_mdd_loss),
        "both_gates": bool(cagr >= 0.26 and mdd_loss < f_mdd_loss),
    }


def write_a_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """a_metrics.csv（机器键；浮点全精度，不预四舍五入）。"""
    columns = [
        "capital",
        "account",
        "segment",
        "metric",
        "value",
        "extra",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    [
                        str(row.get("capital", "")),
                        str(row.get("account", "")),
                        str(row.get("segment", "")),
                        str(row.get("metric", "")),
                        repr(row.get("value")) if row.get("value") is not None else "",
                        str(row.get("extra", "")).replace(",", ";"),
                    ]
                )
                + "\n"
            )


def write_b_required_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """b_required.csv（§7.3 算术原值）。"""
    columns = [
        "capital",
        "n_intervals",
        "target_total_value",
        "r80_ending_value",
        "b_required_ending_value",
        "b_initial_capital",
        "b_required_cagr_from_0_2k",
        "status",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            cagr = row.get("b_required_cagr_from_0.2k")
            handle.write(
                ",".join(
                    [
                        str(row["capital"]),
                        str(row["n_intervals"]),
                        repr(row["target_total_value"]),
                        repr(row["r80_ending_value"]),
                        repr(row["b_required_ending_value"]),
                        str(row["b_initial_capital"]),
                        repr(cagr) if cagr is not None else "",
                        str(row["status"]),
                    ]
                )
                + "\n"
            )


__all__ = [
    "DELIVERABLES",
    "MetricsError",
    "account_metrics",
    "b_required_row",
    "bootstrap_diff",
    "chain_frame",
    "chain_metrics",
    "read_old_anchor",
    "r_gate_verdict",
    "segment_metrics",
    "write_a_metrics_csv",
    "write_b_required_csv",
]
