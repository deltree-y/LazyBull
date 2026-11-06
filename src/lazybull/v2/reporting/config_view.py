# -*- coding: utf-8 -*-
"""配置/模型/年化投影（设计 §5.2 对应四行）。

硬口径：
- ``account_start_date`` 冲突检测必须在**原始 config 展平前**完成：``extra.account_start_date``
  与顶层 ``account_start_date`` 不同即 ``account_start_date_conflict``，相同则取该值；
  YAML 块顺序不影响判定；再把确定值置入展平 dict 后调**既有** ``load_account_start_date``；
- 展平只喂既有函数（``config_adapter.load_paper_config``），本模块不复制展平规则；
- 年化只调 ``v2/hosts/paper/performance.py::calculate_annualized_return``（唯一实现）；
  初资/总值非正、起始日晚于 NAV 日一律 ``invalid``（不得静默返回 0）；
- 实跑模型：当前 v2 run record 无该字段 ⇒ 固定 ``not_available/no_persisted_actual_model``，
  **禁止**用 legacy 配置或 Registry 回填；latest 解析固定 ``no_readonly_resolution``
  （本模块绝不构造 ModelRegistry）。
"""

from __future__ import annotations

from typing import Any, Mapping

from src.lazybull.v2.hosts.paper.config_adapter import PaperHostError, load_paper_config
from src.lazybull.v2.hosts.paper.performance import (
    calculate_annualized_return,
    load_account_start_date,
)
from src.lazybull.v2.reporting import status

__all__ = [
    "project_annualization",
    "project_config",
    "project_model",
    "resolve_account_start_date",
]

_START_DATE_KEY = "account_start_date"


def _config_text(value: Any) -> str | None:
    """配置起始日规整（空白视为缺省；非字符串按文本处理，不猜语义）。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_account_start_date(
    raw_config: Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """展平前冲突检测；返回 ``(确定值, 冲突状态行或 None)``。"""
    if not isinstance(raw_config, Mapping):
        return None, None
    top_level = _config_text(raw_config.get(_START_DATE_KEY))
    extra_block = raw_config.get("extra")
    nested = (
        _config_text(extra_block.get(_START_DATE_KEY))
        if isinstance(extra_block, Mapping)
        else None
    )
    if top_level is not None and nested is not None and top_level != nested:
        return None, status.make_status(
            "invalid",
            reason_code="account_start_date_conflict",
            detail=(
                "extra.account_start_date 与顶层 account_start_date 不一致："
                f"{nested!r} ≠ {top_level!r}（展平前拒绝，不猜值）"
            ),
        )
    return top_level or nested, None


def project_model(flat: Mapping[str, Any]) -> dict[str, Any]:
    """配置模型 A/B/权重投影 + 实跑模型固定降级（禁止 legacy/Registry 回填）。"""
    model_version = flat.get("model_version")
    model_version_b = flat.get("model_version_b")
    weight_a = flat.get("ensemble_weight_a")
    weight_b = None
    if model_version_b is not None and weight_a is not None:
        try:
            weight_b = 1.0 - float(weight_a)
        except (TypeError, ValueError):
            weight_b = None
    follows_latest = model_version is None
    row = status.make_status(
        "degraded" if follows_latest else "available",
        reason_code="no_readonly_resolution" if follows_latest else None,
        detail=(
            "model_version 为 null：显示跟随最新，但 latest 只读不可解析（不构造 Registry）"
            if follows_latest
            else "配置模型原值（不做最新解析）"
        ),
        extra={
            "model_version": model_version,
            "model_version_b": model_version_b,
            "ensemble_weight_a": weight_a,
            "ensemble_weight_b": weight_b,
            "follows_latest": follows_latest,
        },
    )
    row["actual_model"] = status.make_status(
        "not_available",
        reason_code="no_persisted_actual_model",
        detail="当前 v2 代不持久化实跑模型版本（禁止用配置/legacy 回填）",
    )
    return row


def project_annualization(
    *,
    flat: Mapping[str, Any],
    nav_result: Mapping[str, Any],
    account_start_date: str | None,
) -> dict[str, Any]:
    """纸面年化投影：只用已核对 NAV + 展平 config，调既有唯一实现。"""
    nav_date = nav_result.get("nav_date")
    if not nav_result.get("annualization_allowed"):
        return status.make_status(
            "not_available",
            reason_code="authority_mismatch",
            detail="NAV 未通过权威原值核对，年化不得计算（不回调较早历史点）",
        )
    if nav_date is None or not account_start_date:
        return status.make_status(
            "not_available",
            reason_code="no_source",
            detail=f"年化输入缺失（nav_date={nav_date!r}, account_start_date={account_start_date!r}）",
        )
    row = _nav_row_of(nav_result, str(nav_date))
    if row is None:
        return status.make_status(
            "not_available", reason_code="nav_row_missing", detail=f"NAV 日 {nav_date} 无核对行"
        )
    try:
        initial = float(flat.get("initial_capital"))
        current = float(row.get("total_value"))
    except (TypeError, ValueError):
        return status.make_status(
            "invalid", reason_code="annualization_input_invalid", detail="初始资金或总资产不可解析"
        )
    if initial <= 0 or current <= 0:
        return status.make_status(
            "invalid",
            reason_code="annualization_input_invalid",
            detail=f"初资 {initial} / 总资产 {current} 非正，年化不可计算",
        )
    start_text = str(account_start_date)
    if start_text > str(nav_date):
        return status.make_status(
            "invalid",
            reason_code="annualization_start_after_nav",
            detail=f"账户起始日 {start_text} 晚于 NAV 日 {nav_date}",
        )
    value = calculate_annualized_return(initial, current, str(nav_date), start_text)
    if value is None:
        return status.make_status(
            "not_available",
            reason_code="no_source",
            detail="既有年化实现返回 None（不可计算）",
        )
    return status.make_status(
        "available",
        detail="已核对 NAV + 展平 config 经既有实现计算（百分数）",
        verification_level="authority_verified",
        extra={
            "annualized_return": value,
            "initial_capital": initial,
            "current_value": current,
            "nav_date": nav_date,
            "account_start_date": start_text,
        },
    )


def _nav_row_of(nav_result: Mapping[str, Any], nav_date: str) -> Mapping[str, Any] | None:
    """从 NAV 核对结果取指定日期的持久 NAV 行（``rows`` 只含核对明细，值另存）。"""
    values = nav_result.get("nav_values")
    if isinstance(values, Mapping):
        row = values.get(nav_date)
        if isinstance(row, Mapping):
            return row
    return None


def project_config(
    *,
    store: Any,
    raw_config: Any,
    nav_result: Mapping[str, Any],
) -> dict[str, Any]:
    """配置区投影：展平/起始日/模型/年化的统一入口。"""
    if not isinstance(raw_config, Mapping):
        return status.make_status(
            "not_available", reason_code="config_missing", detail="config.yaml 缺失或顶层非 dict"
        )
    resolved, conflict = resolve_account_start_date(raw_config)
    if conflict is not None:
        return conflict
    try:
        flat = load_paper_config(store)
    except PaperHostError as exc:
        return status.make_status("not_available", reason_code="config_missing", detail=str(exc))
    if resolved is not None:
        flat[_START_DATE_KEY] = resolved
    start_date = load_account_start_date(store, config=flat)
    row = status.make_status(
        "available" if start_date else "degraded",
        reason_code=None if start_date else "no_source",
        detail=(
            f"起始日 {start_date}（展平后交既有 load_account_start_date）"
            if start_date
            else "config 与 NAV 均无起始日来源"
        ),
        extra={"account_start_date": start_date},
    )
    row["model"] = project_model(flat)
    row["annualization"] = project_annualization(
        flat=flat, nav_result=nav_result, account_start_date=start_date
    )
    return row
