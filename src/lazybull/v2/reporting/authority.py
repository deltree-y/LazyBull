# -*- coding: utf-8 -*-
"""权威内容核对：代形态、账户投影、NAV 原值与当前日成交（设计 §2.3.4 / §2.4 / §5.2）。

稳定哈希只能证明“读取未变”，不能证明“内容正确”。本模块只做**核对**：
- 任何不一致一律 ``invalid`` 并**零修复**（不改权威、不回写投影、不猜值）；
- 历史已核对点保留（不让整段历史因末日异常而消失），但当前年化**不得**回退到较早历史点；
- 权威已裁剪的历史成交、缺权威对应值的字段只能标 ``projection_only``。

本模块为纯函数层，不做 I/O、不构造 Store/Registry、不调用恢复。
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date as date_type
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from src.lazybull.v2.reporting import status

__all__ = [
    "NAV_AUTHORITY_MAP",
    "NAV_VALUE_TOLERANCE",
    "TRADE_IDENTITY_COLUMNS",
    "classify_run_record",
    "compare_account_projection",
    "normalize_date",
    "trade_identity",
    "trade_identity_multiset",
    "verify_current_day_trades",
    "verify_nav_authority",
]

#: 权威 ``portfolio_values`` → 持久 NAV 列的三原值映射（设计 §2.4）
NAV_AUTHORITY_MAP: tuple[tuple[str, str], ...] = (
    ("portfolio_value", "total_value"),
    ("capital", "cash"),
    ("market_value", "position_value"),
)
#: NAV 原值比对容差（与 P4 一致性第三腿同值）
NAV_VALUE_TOLERANCE = 1e-6
#: 当前日成交身份七元组（成本分解不参与身份判断）
TRADE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "ts_code",
    "action",
    "shares",
    "price",
    "amount",
    "reason",
)

_ENGINE_V2 = "v2"
#: 卖出类型 → 发布 reason（与宿主 ``_SELL_TYPE_REASONS`` 同表，parity 测试守卫）
_SELL_TYPE_REASONS: dict[str, str] = {
    "rebalance": "调仓卖出",
    "holding_period": "持有期到期",
    "stop_loss": "止损卖出",
}


def normalize_date(value: Any) -> str | None:
    """任意日期表示 → ``YYYYMMDD``；不可解析返回 None（不猜日期）。"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date_type):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] in ("-", "/"):
        text = text[:10].replace("-", "").replace("/", "")
    return text if len(text) == 8 and text.isdigit() else None


def classify_run_record(
    record: Mapping[str, Any] | None,
    *,
    run_kind: str,
    trade_date: str,
) -> dict[str, Any]:
    """当前权威代 run record 形态判定（正常 v2 / 切换前记录形态 / 非法）。

    设计 §2.3.4：``run_kind`` 必须等于权威相位；``engine`` 若存在必须为 v2（它只是配置解析
    结果，不能替代 ``paper_generation``）；**同时缺** ``run_kind`` 与 ``engine`` 才允许
    账户保留为 ``degraded/run_record_pre_switch_shape``；**只缺其一**按
    ``invalid/authority_mismatch``，不得补写。
    """
    if record is None:
        return status.make_status(
            "invalid",
            reason_code="run_record_missing",
            detail=f"缺 runs/{run_kind}_{trade_date}.json",
        )
    if not isinstance(record, Mapping):
        return status.make_status(
            "invalid", reason_code="invalid_json", detail="run record 顶层不是对象"
        )
    recorded_date = normalize_date(record.get("trade_date"))
    if recorded_date != trade_date:
        return status.make_status(
            "invalid",
            reason_code="authority_mismatch",
            detail=f"run record 日期 {record.get('trade_date')!r} 与权威日 {trade_date} 不一致",
        )
    has_kind = "run_kind" in record
    has_engine = "engine" in record
    if not has_kind and not has_engine:
        return status.make_status(
            "degraded",
            reason_code="run_record_pre_switch_shape",
            detail="记录同时缺 run_kind 与 engine：非可证明的 v2 驱动代（不补写）",
        )
    if has_kind != has_engine:
        missing = "engine" if has_kind else "run_kind"
        return status.make_status(
            "invalid",
            reason_code="authority_mismatch",
            detail=f"记录只缺 {missing}（不得补写，按形态冲突处理）",
        )
    if str(record.get("run_kind")) != run_kind:
        return status.make_status(
            "invalid",
            reason_code="authority_mismatch",
            detail=f"run_kind={record.get('run_kind')!r} 与权威相位 {run_kind!r} 不一致",
        )
    if str(record.get("engine")) != _ENGINE_V2:
        return status.make_status(
            "invalid",
            reason_code="authority_mismatch",
            detail=f"engine={record.get('engine')!r} 不是 {_ENGINE_V2}",
        )
    return status.make_status("available", detail="正常 v2 记录形态")


def _compare_scalar(
    mismatches: list[dict[str, Any]],
    *,
    path: str,
    expected: Any,
    actual: Any,
    numeric: bool,
    tolerance: float,
) -> None:
    """逐字段比对（数值走容差，其余严格相等）；不一致即登记，不修复。"""
    if numeric:
        try:
            left = float(expected)
            right = float(actual)
        except (TypeError, ValueError):
            left = right = None
        finite = (
            left is not None
            and right is not None
            and math.isfinite(left)
            and math.isfinite(right)
        )
        if not finite or abs(left - right) > tolerance:
            mismatches.append({"path": path, "expected": expected, "actual": actual})
        return
    if expected != actual:
        mismatches.append({"path": path, "expected": expected, "actual": actual})


def compare_account_projection(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any] | None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """权威重建投影与 ``state/account.json`` 逐字段全等核对（设计 §2.3.3）。

    金额字段走容差，其余（日期、代码、状态、备注）严格相等；缺文件 →
    ``not_available``，任一差异 → ``invalid/authority_mismatch``（零修复）。
    """
    if actual is None:
        return status.make_status(
            "not_available", reason_code="account_projection_missing", detail="缺 state/account.json"
        )
    if not isinstance(actual, Mapping):
        return status.make_status(
            "invalid", reason_code="invalid_json", detail="account.json 顶层不是对象"
        )
    mismatches: list[dict[str, Any]] = []
    extra_keys = sorted(set(str(key) for key in actual) - set(str(key) for key in expected))
    if extra_keys:
        mismatches.append(
            {"path": "extra_keys", "expected": [], "actual": extra_keys}
        )
    _compare_scalar(
        mismatches, path="cash", expected=expected.get("cash"), actual=actual.get("cash"),
        numeric=True, tolerance=tolerance,
    )
    _compare_scalar(
        mismatches,
        path="last_update",
        expected=expected.get("last_update"),
        actual=actual.get("last_update"),
        numeric=False,
        tolerance=tolerance,
    )
    expected_positions = dict(expected.get("positions") or {})
    actual_positions = actual.get("positions")
    if not isinstance(actual_positions, Mapping):
        mismatches.append(
            {"path": "positions", "expected": "dict", "actual": type(actual_positions).__name__}
        )
        actual_positions = {}
    for code in sorted(set(expected_positions) | set(actual_positions)):
        left = expected_positions.get(code)
        right = actual_positions.get(code)
        if left is None or right is None:
            mismatches.append({"path": f"positions.{code}", "expected": left, "actual": right})
            continue
        for field_name in sorted(set(left) | set(right)):
            _compare_scalar(
                mismatches,
                path=f"positions.{code}.{field_name}",
                expected=left.get(field_name),
                actual=right.get(field_name),
                numeric=field_name in ("shares", "buy_price", "buy_cost", "buy_pnl_price"),
                tolerance=max(tolerance, 1e-9),
            )
    if mismatches:
        return status.make_status(
            "invalid",
            reason_code="authority_mismatch",
            detail=f"{len(mismatches)} 处字段与权威重建投影不一致（零修复）",
            extra={"mismatches": mismatches[:20]},
        )
    return status.make_status(
        "available",
        detail=f"账户投影逐字段一致（{len(expected_positions)} 只持仓）",
    )


def _authority_nav_rows(
    portfolio_values: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """权威 ``portfolio_values`` → 规范化行；返回 ``(行, 问题描述)``。"""
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    previous: str | None = None
    for index, entry in enumerate(portfolio_values):
        if not isinstance(entry, Mapping):
            problems.append(f"portfolio_values[{index}] 不是对象")
            continue
        day = normalize_date(entry.get("date"))
        if day is None:
            problems.append(f"portfolio_values[{index}] 日期不可解析: {entry.get('date')!r}")
            continue
        if previous is not None and day <= previous:
            problems.append(f"portfolio_values 日期非严格递增: {previous} → {day}")
        previous = day
        rows.append(
            {
                "trade_date": day,
                "portfolio_value": entry.get("portfolio_value"),
                "capital": entry.get("capital"),
                "market_value": entry.get("market_value"),
            }
        )
    return rows, problems


def _to_float(value: Any) -> float | None:
    """数值规整为 float（不可解析 ⇒ None，不猜值）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nav_row_index(nav_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict], list[str]]:
    """持久 NAV 行 → ``{日期: 行}``（同日稳定原行序取最后）；返回 ``(索引, 问题描述)``。

    行值统一转 float（``None`` = 不可解析），保证 JSON 输出可序列化且比对口径一致。
    """
    index: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for position, row in enumerate(nav_rows):
        if not isinstance(row, Mapping):
            problems.append(f"nav.parquet 第 {position} 行不是对象")
            continue
        day = normalize_date(row.get("trade_date"))
        if day is None:
            problems.append(f"nav.parquet 第 {position} 行 trade_date 不可解析")
            continue
        index[day] = {
            "trade_date": day,
            "cash": _to_float(row.get("cash")),
            "position_value": _to_float(row.get("position_value")),
            "total_value": _to_float(row.get("total_value")),
            "nav": _to_float(row.get("nav")),
        }
    return index, problems


def _diff_nav_values(
    authority: Mapping[str, Any], nav_row: Mapping[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    """三原值逐项比对（authority 键 → nav 列键）；返回差异清单。"""
    diffs: list[dict[str, Any]] = []
    for source_key, nav_key in NAV_AUTHORITY_MAP:
        try:
            left = float(authority.get(source_key))
            right = float(nav_row.get(nav_key))
        except (TypeError, ValueError):
            diffs.append(
                {
                    "field": nav_key,
                    "authority": authority.get(source_key),
                    "nav": nav_row.get(nav_key),
                }
            )
            continue
        if not (math.isfinite(left) and math.isfinite(right)):
            diffs.append({"field": nav_key, "authority": left, "nav": right,
                          "reason": "nonfinite"})
            continue
        if abs(left - right) > tolerance:
            diffs.append({"field": nav_key, "authority": left, "nav": right})
    return diffs


def verify_nav_authority(
    *,
    portfolio_values: Sequence[Mapping[str, Any]],
    nav_rows: Sequence[Mapping[str, Any]] | None,
    authority_date: str,
    current_day_has_trades: bool,
    tolerance: float = NAV_VALUE_TOLERANCE,
) -> dict[str, Any]:
    """NAV 三原值权威核对 + 逐行状态（设计 §2.4 / §5.2 NAV 行）。

    返回整区状态、逐行核对明细、断线与年化可用性。**历史已核对点全部保留**；末日不匹配
    只把该点标 invalid 并中断其有效连线，不使整图历史消失，也不把年化回退到较早历史点。
    """
    authorities, authority_problems = _authority_nav_rows(portfolio_values)
    early = _nav_precondition(authorities, authority_problems, nav_rows)
    if early is not None:
        return early
    index, nav_problems = _nav_row_index(nav_rows or ())
    if nav_problems:
        return _nav_result(
            status.make_status(
                "invalid",
                reason_code="nav_row_unparsable",
                detail="; ".join(nav_problems[:5]),
            )
        )
    rows, broken, nav_only = _compare_nav_rows(authorities, index, tolerance)
    candidates = [day for day in index if day <= authority_date]
    nav_date = max(candidates) if candidates else None
    details = {
        "rows": rows,
        "verified_rows": [row["trade_date"] for row in rows if row["present"] and row["matched"]],
        "broken_links": broken,
        "nav_only_days": nav_only,
        "nav_date": nav_date,
        "authority_date": authority_date,
        "current_row": next((row for row in rows if row["trade_date"] == authority_date), None),
        "nav_values": {
            str(row["trade_date"]): dict(index[str(row["trade_date"])])
            for row in rows
            if row["present"]
        },
    }
    if nav_only:
        return _nav_result(
            status.make_status(
                "invalid",
                reason_code="authority_mismatch",
                detail=(
                    f"nav.parquet 含权威 portfolio_values 不存在的业务日 {nav_only[:5]}"
                    "（两载体业务日集合冲突，禁止静默处置）"
                ),
                extra={"nav_only_days": nav_only},
            ),
            nav_date_status="same_day" if nav_date == authority_date else "unknown",
            annualization_allowed=False,
            details=details,
        )
    return _decide_nav_status(
        details=details,
        nav_date=nav_date,
        nav_date_row=next((row for row in rows if row["trade_date"] == nav_date), None),
        authority_date=authority_date,
        current_day_has_trades=current_day_has_trades,
        broken=broken,
    )


def _nav_precondition(
    authorities: Sequence[Mapping[str, Any]],
    authority_problems: Sequence[str],
    nav_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """权威侧前置门：权威不可读/日期倒退/NAV 来源缺失时提前返回状态。"""
    if not authorities:
        return _nav_result(
            status.make_status(
                "invalid",
                reason_code="authority_unreadable",
                detail=f"权威 portfolio_values 不可用：{list(authority_problems) or '空'}",
            )
        )
    if authority_problems:
        return _nav_result(
            status.make_status(
                "invalid",
                reason_code="date_regression",
                detail="; ".join(list(authority_problems)[:5]),
            )
        )
    if nav_rows is None:
        return _nav_result(
            status.make_status(
                "not_available", reason_code="nav_source_missing", detail="缺 nav.parquet"
            ),
            nav_date_status="missing",
            annualization_allowed=False,
            details={"rows": [], "verified_rows": [], "broken_links": [], "nav_only_days": []},
        )
    return None


def _compare_nav_rows(
    authorities: Sequence[Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """三原值比对：**只比对两侧都有行**的业务日 + 反向差集。

    跳空语义（生产侧 `state_store`/`nav_valuation` 明写）：``nav.parquet`` 只在有成交日追加行，
    ``portfolio_values`` 每个处理日追加 ⇒ **权威有业务日而 nav 无行是合法跳空**，不得判断线
    （否则正常根每天 degraded）。真正的冲突只有两类：
    ① 两侧都有行但三原值不一致；② nav 有行而权威无该业务日（两载体业务日集合冲突）。

    Returns:
        ``(逐行明细, 数值不一致日, nav 独有日)``。
    """
    rows: list[dict[str, Any]] = []
    broken: list[str] = []
    authority_days = [str(entry["trade_date"]) for entry in authorities]
    for entry in authorities:
        day = str(entry["trade_date"])
        nav_row = index.get(day)
        if nav_row is None:
            rows.append({"trade_date": day, "present": False, "matched": True, "diffs": []})
            continue
        diffs = _diff_nav_values(entry, nav_row, tolerance)
        rows.append({"trade_date": day, "present": True, "matched": not diffs, "diffs": diffs})
        if diffs:
            broken.append(day)
    nav_only = sorted(set(str(day) for day in index) - set(authority_days))
    return rows, broken, nav_only


def _decide_nav_status(
    *,
    details: dict[str, Any],
    nav_date: str | None,
    nav_date_row: Mapping[str, Any] | None,
    authority_date: str,
    current_day_has_trades: bool,
    broken: Sequence[str],
) -> dict[str, Any]:
    """按 NAV 日/当前行核对结果给出整区状态与年化可用性。"""
    early = _nav_gate_failure(
        details=details,
        nav_date=nav_date,
        nav_date_row=nav_date_row,
        authority_date=authority_date,
        current_day_has_trades=current_day_has_trades,
    )
    if early is not None:
        return early
    assert nav_date is not None
    if nav_date < authority_date:
        return _nav_result(
            status.make_status(
                "stale",
                reason_code="nav_before_account_date",
                detail=f"NAV 最新行 {nav_date} 早于账户日 {authority_date}（保留原日期）",
            ),
            nav_date_status="stale",
            details=details,
        )
    if broken:
        return _nav_result(
            status.make_status(
                "degraded",
                reason_code="authority_mismatch",
                detail=f"当前行已核对；历史断线日 {list(broken)[:5]}（连线中断，不修复）",
            ),
            nav_date_status="same_day",
            details=details,
        )
    return _nav_result(
        status.make_status(
            "available", detail=f"{len(details.get('rows') or [])} 个业务日三原值与权威一致"
        ),
        nav_date_status="same_day",
        details=details,
    )


def _nav_gate_failure(
    *,
    details: dict[str, Any],
    nav_date: str | None,
    nav_date_row: Mapping[str, Any] | None,
    authority_date: str,
    current_day_has_trades: bool,
) -> dict[str, Any] | None:
    """NAV 日缺失/未核对/当前日无行三类“不可用”判定；无问题返回 None。"""
    if nav_date is None or nav_date_row is None:
        where = "无 ≤ {} 的 NAV 行".format(authority_date) if nav_date is None else (
            f"NAV 日 {nav_date} 在权威 portfolio_values 中无对应行"
        )
        return _nav_result(
            status.make_status(
                "invalid" if current_day_has_trades else "not_available",
                reason_code="nav_row_missing",
                detail=where,
            ),
            nav_date_status="missing",
            details=details,
        )
    if not nav_date_row.get("matched"):
        return _nav_result(
            status.make_status(
                "invalid",
                reason_code="authority_mismatch",
                detail=f"NAV 日 {nav_date} 三原值与权威不一致（零修复；年化不得回退）",
                extra={"diffs": list(nav_date_row.get("diffs") or [])},
            ),
            nav_date_status="same_day" if nav_date == authority_date else "stale",
            annualization_allowed=False,
            details=details,
        )
    current_row = next(
        (row for row in details.get("rows") or [] if row["trade_date"] == authority_date), None
    )
    if current_row is not None and not current_row.get("present") and current_day_has_trades:
        return _nav_result(
            status.make_status(
                "invalid",
                reason_code="nav_row_missing",
                detail=f"当前日 {authority_date} 有成交但缺同日 NAV 行（不修复）",
            ),
            nav_date_status="missing",
            annualization_allowed=False,
            details=details,
        )
    return None


def _nav_result(
    row: dict[str, Any],
    *,
    nav_date_status: str = "unknown",
    annualization_allowed: bool | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """NAV 区结果装配（状态行在前，明细在后）。

    ``annualization_allowed`` 缺省时由状态推导：只有 ``available``/``degraded``/``stale``
    （即当前 NAV 点已与权威核对过）才允许年化；``invalid``/``not_available`` 一律 False
    ——标志位不得与“NAV 未通过核对”自相矛盾。
    """
    result = dict(row)
    verified = row.get("report_status") in ("available", "degraded", "stale")
    if annualization_allowed is True and not verified:
        raise ValueError(
            f"annualization_allowed=True 与 NAV 状态 {row.get('report_status')!r} 矛盾（禁止放行）"
        )
    result["verification_level"] = "authority_verified" if verified else "projection_only"
    result["nav_date_status"] = nav_date_status
    result["annualization_allowed"] = (
        verified if annualization_allowed is None else bool(annualization_allowed)
    )
    if details:
        result.update(details)
    return result


def project_authority_trade_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """权威成交记录 → 七元组口径（与宿主 ``_project_trade_row``/``_trade_reason`` 逐字段一致）。

    权威 ``EngineState.trades`` 是**引擎记录形态**（``date/stock/action/price/shares/amount/cost``，
    卖出另有 ``sell_type/sell_reason``），而 ``trades.parquet`` 是 11 列投影形态。本函数是报告包内
    的**只读镜像**（禁止 import 宿主写侧），映射口径由
    ``tests/test_v2_p6_t1_authority.py::test_authority_trade_projection_parity`` 与宿主实现逐字段比对；
    对已投影行幂等（``trade_date/ts_code/reason`` 存在时原值优先）。

    Raises:
        ValueError: 缺必需字段或类型非法（带上下文，不猜值）。
    """
    try:
        action = str(record.get("action") or "")
        return {
            "trade_date": normalize_date(_first_present(record, "date", "trade_date")) or "",
            "ts_code": str(_first_present(record, "stock", "ts_code") or ""),
            "action": action,
            "shares": int(_first_present(record, "shares")),
            "price": float(_first_present(record, "price")),
            "amount": float(_first_present(record, "amount")),
            "reason": _trade_reason(record, action),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"权威成交记录不可投影为七元组：{dict(record)}（{exc}）") from exc


def _first_present(record: Mapping[str, Any], *names: str) -> Any:
    """按顺序取第一个非 None 字段（引擎形态优先，已投影形态兜底）。"""
    for name in names:
        if record.get(name) is not None:
            return record[name]
    raise KeyError(f"缺字段 {names}")


def _trade_reason(record: Mapping[str, Any], action: str) -> str:
    """成交 reason 投影（已投影行原值优先，其余口径与宿主 ``_trade_reason`` 一致）。"""
    explicit = record.get("reason")
    if explicit is not None and str(explicit) != "":
        return str(explicit)
    if action == "sell":
        if record.get("sell_reason"):
            return str(record["sell_reason"])
        sell_type = str(record.get("sell_type") or "")
        return _SELL_TYPE_REASONS.get(sell_type, sell_type or "卖出")
    if record.get("buy_reason"):
        return str(record["buy_reason"])
    return "信号生成"


def trade_identity(row: Mapping[str, Any]) -> tuple[str, str, str, int, float, float, str]:
    """成交七元组身份（成本分解不参与；缺字段/类型非法即报错）。"""
    projected = project_authority_trade_record(row)
    return (
        projected["trade_date"],
        projected["ts_code"],
        projected["action"],
        projected["shares"],
        projected["price"],
        projected["amount"],
        projected["reason"],
    )


def trade_identity_multiset(rows: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """排序后的多重集（**保留重复次数**，禁止去重为 set 或只比行数）。"""
    return sorted(trade_identity(row) for row in rows)


def _duplicate_count(identities: Sequence[tuple]) -> int:
    """重复行数（多重集长度 − 去重长度）。"""
    return len(identities) - len(set(identities))


def _trade_mismatch_result(
    authority_ids: Sequence[tuple],
    projection_ids: Sequence[tuple],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """成交多重集差异结果（保留缺失/多余元组，便于定位且不修复）。"""
    authority_counter = Counter(authority_ids)
    projection_counter = Counter(projection_ids)
    missing = sorted((authority_counter - projection_counter).elements())
    extra = sorted((projection_counter - authority_counter).elements())
    return status.make_status(
        "invalid",
        reason_code="authority_mismatch",
        detail=(
            f"当前日成交多重集不一致（权威 {len(authority_ids)} 行 / 投影 "
            f"{len(projection_ids)} 行）"
        ),
        extra={
            **dict(details),
            "missing_rows": [list(item) for item in missing[:10]],
            "extra_rows": [list(item) for item in extra[:10]],
        },
    )


def verify_current_day_trades(
    *,
    authority_rows: Sequence[Mapping[str, Any]] | None,
    projection_rows: Sequence[Mapping[str, Any]] | None,
    record_count: int | None,
    authority_verifiable: bool = True,
    history_rows: int = 0,
) -> dict[str, Any]:
    """当前日成交按七元组全集合核对（设计 §2.4 / §5.2 当前代成交）。

    - 计数存在时必须等于文件行数；不等即 ``invalid/count_mismatch``；
    - 权威可核对时逐元组多重集比对（保留重复次数），不等即 ``invalid/authority_mismatch``；
    - 恢复代缺计数不当作 0：能由权威空 trades 支持则 ``empty_valid``，否则
      ``not_available/recovery_counts_missing``。
    """
    if not authority_verifiable or authority_rows is None:
        return status.make_status(
            "not_available",
            reason_code="not_authority_verifiable",
            detail="当前日权威成交不可核对（不得宣称完成内容认证）",
            extra={"history_rows": history_rows},
        )
    projection = list(projection_rows or [])
    if record_count is not None and int(record_count) != len(projection):
        return status.make_status(
            "invalid",
            reason_code="count_mismatch",
            detail=f"记录 fills_count={record_count} 与 trades 行数 {len(projection)} 不一致",
            extra={"history_rows": history_rows},
        )
    try:
        authority_ids = trade_identity_multiset(list(authority_rows))
        projection_ids = trade_identity_multiset(projection)
    except ValueError as exc:
        return status.make_status("invalid", reason_code="missing_precondition", detail=str(exc))
    details = {
        "authority_rows": len(authority_ids),
        "projection_rows": len(projection_ids),
        "authority_duplicates": _duplicate_count(authority_ids),
        "projection_duplicates": _duplicate_count(projection_ids),
        "history_rows": history_rows,
    }
    if authority_ids != projection_ids:
        return _trade_mismatch_result(authority_ids, projection_ids, details)
    if record_count is None:
        if not projection_ids:
            return status.make_status(
                "empty_valid",
                detail="恢复代缺计数；权威当前日无成交，支持零成交语义",
                verification_level="authority_verified",
                extra=details,
            )
        return status.make_status(
            "degraded",
            reason_code="recovered_generation_count_missing",
            detail="恢复代缺 fills_count；当前日成交已由权威多重集核对",
            verification_level="authority_verified",
            extra=details,
        )
    if not projection_ids:
        return status.make_status(
            "empty_valid",
            detail="当前日无成交（权威确认）",
            verification_level="authority_verified",
            extra=details,
        )
    return status.make_status(
        "available",
        detail=f"{len(projection_ids)} 笔当前日成交与权威多重集一致",
        verification_level="authority_verified",
        extra=details,
    )
