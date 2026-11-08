# -*- coding: utf-8 -*-
"""趋势指标适配层（P6 T4：五类指标按证据类分列，只调既有唯一实现）。

硬口径（设计 §5.5 / §5.7 / §8.2）：

- 年化/回撤**只调既有唯一实现**：research 走 ``experiment_sources.validate_chain_rows``
  → ``calculate_chain_metrics``；production 走 ``calculate_annualized_return``（纸面年化唯一
  实现，自然日百分数）；calibration 的 log-MDD/pp 由上游回撤裁判换算后原样读出（不二次换算）；
- 金额（双边成交金额 / 成本）只做**同窗口求和**：任一消费行缺失或含 NaN/±Inf 即整项不可用
  （不补 0、不丢行），聚合溢出同样降级；
- 单位/日期语义按证据类分列（``trend_contract.unit_for``），**禁止跨域复用或互转**；
- 本模块不做 I/O，不读快照，不写任何文件。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.compare.aggregate import aggregate_run
from src.lazybull.v2.hosts.paper.performance import calculate_annualized_return
from src.lazybull.v2.reporting import experiment_sources, status
from src.lazybull.v2.reporting.authority import normalize_date
from src.lazybull.v2.reporting.experiment_metrics import TOP20_COLUMN

__all__ = [
    "chain_stats",
    "legacy_topk_mean",
    "money_total",
    "paper_annualized",
    "status_row",
]


def status_row(
    word: str,
    reason: str | None,
    *,
    detail: str,
    source_ref: str | None = None,
) -> dict[str, Any]:
    """统一状态行（闭集构造器；禁止自由文本冒充原因码）。"""
    return status.make_status(
        word, reason_code=reason, source_ref=source_ref, detail=detail
    )


def chain_stats(chain: pd.DataFrame | None) -> dict[str, Any]:
    """research 链式指标：全行门（唯一实现）→ cagr / max_drawdown（小数）。"""
    if chain is None or chain.empty:
        row = status_row(
            "not_available", "chain_source_missing", detail="链式净值为空，指标不可计算"
        )
        return {"status": row, "cagr": None, "max_drawdown": None}
    gate = experiment_sources.validate_chain_rows(chain)
    metrics = dict(gate.get("metrics") or {})
    if not status.is_usable(gate):
        return {"status": gate, "cagr": None, "max_drawdown": None}
    return {
        "status": gate,
        "cagr": _finite(metrics.get("cagr")),
        "max_drawdown": _finite(metrics.get("max_drawdown")),
    }


def money_total(
    frames: Sequence[pd.DataFrame], column: str
) -> dict[str, Any]:
    """同窗口金额求和（缺失/非有限/溢出 ⇒ 值 ``None`` + 显式原因，不补 0）。"""
    if not frames:
        return {
            "value": None,
            "status": status_row("not_available", "no_source", detail=f"无金额来源（{column}）"),
        }
    for frame in frames:
        if column not in frame.columns:
            return {
                "value": None,
                "status": status_row(
                    "not_available", "missing_precondition", detail=f"金额列缺失：{column}"
                ),
            }
    merged = pd.concat([frame[[column]] for frame in frames], ignore_index=True)
    values = pd.to_numeric(merged[column], errors="coerce").astype(float)
    invalid = ~np.isfinite(values.to_numpy(dtype=float))
    if bool(invalid.any()):
        return {
            "value": None,
            "status": status_row(
                "degraded",
                "nonfinite_value",
                detail=f"{column} 有 {int(invalid.sum())} 行缺失/非有限（不补零）",
            ),
        }
    with np.errstate(over="ignore"):
        total = float(values.sum())
    if not np.isfinite(total):
        return {
            "value": None,
            "status": status_row(
                "degraded", "nonfinite_value", detail=f"{column} 聚合后非有限（不补零）"
            ),
        }
    return {
        "value": total,
        "status": status_row(
            "available", None, detail=f"{column} 按同窗口求和（{len(values)} 行）"
        ),
    }


def legacy_topk_mean(summary: pd.DataFrame | None) -> dict[str, Any]:
    """旧 WF TopK 正日占比（仅 ``KEY_Top20_hit_rate`` 改名 + ``aggregate_run`` 均值）。"""
    if summary is None or summary.empty:
        return {
            "value": None,
            "status": status_row("not_available", "no_source", detail="旧 WF summary 为空"),
        }
    if TOP20_COLUMN not in summary.columns:
        return {
            "value": None,
            "status": status_row(
                "not_available",
                "no_source",
                detail=f"旧 WF summary 无 {TOP20_COLUMN}（不拿其它命中率替代）",
            ),
        }
    aggregated = aggregate_run(summary)
    value = _finite(aggregated.get("key_top20_hit_rate_mean"))
    if value is None:
        return {
            "value": None,
            "status": status_row(
                "degraded",
                "no_source",
                detail=f"{TOP20_COLUMN} 存在但无有限均值（不拿其它命中率替代）",
            ),
        }
    return {
        "value": value,
        "status": status_row(
            "available",
            None,
            detail=f"旧 WF {TOP20_COLUMN} 仅字段改名；跨折均值取自 aggregate_run",
        ),
    }


def paper_annualized(
    nav: pd.DataFrame | None,
    *,
    run_day: str,
    initial_capital: Any,
    account_start_date: str | None,
) -> dict[str, Any]:
    """production 年化（自然日百分数）：只用既有 ``calculate_annualized_return``。"""
    if nav is None or "trade_date" not in getattr(nav, "columns", []):
        return {
            "value": None,
            "status": status_row("not_available", "nav_source_missing", detail="NAV 来源缺失"),
        }
    # 日期归一走既有 ``normalize_date``（YYYYMMDD/ISO/Timestamp 一致；不可解析不匹配）
    matched = nav[nav["trade_date"].map(normalize_date) == str(run_day)]
    if matched.empty:
        return {
            "value": None,
            "status": status_row(
                "not_available", "nav_row_missing", detail=f"NAV 无 {run_day} 行（不插值）"
            ),
        }
    row = matched.iloc[-1]
    try:
        initial = float(initial_capital)
        current = float(row.get("total_value"))
    except (TypeError, ValueError):
        return {
            "value": None,
            "status": status_row(
                "invalid", "annualization_input_invalid", detail="初资或 NAV 总值不可解析"
            ),
        }
    if not np.isfinite(initial) or not np.isfinite(current) or initial <= 0 or current <= 0:
        return {
            "value": None,
            "status": status_row(
                "invalid",
                "annualization_input_invalid",
                detail=f"初资 {initial} / 总资产 {current} 非有限或非正",
            ),
        }
    if not account_start_date:
        return {
            "value": None,
            "status": status_row("not_available", "no_source", detail="缺 account_start_date"),
        }
    if str(account_start_date) > str(run_day):
        return {
            "value": None,
            "status": status_row(
                "invalid",
                "annualization_start_after_nav",
                detail=f"账户起始日 {account_start_date} 晚于运行日 {run_day}",
            ),
        }
    value = calculate_annualized_return(initial, current, str(run_day), str(account_start_date))
    if value is None:
        return {
            "value": None,
            "status": status_row(
                "not_available", "no_source", detail="既有年化实现返回 None（不可计算）"
            ),
        }
    return {
        "value": float(value),
        "status": status_row(
            "available",
            None,
            detail="已核对 NAV + 展平 config 经既有实现计算（自然日百分数）",
        ),
    }


def _finite(value: Any) -> float | None:
    """有限浮点（``None``/NaN/±Inf/非数值一律 ``None``，不改取值）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def metric_value_of(result: Mapping[str, Any]) -> float | None:
    """从指标结果取 ``value``（缺失写 ``None``）。"""
    return _finite(result.get("value"))
