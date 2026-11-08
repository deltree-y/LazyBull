# -*- coding: utf-8 -*-
"""驾驶舱视图原语（状态视图 / 键值事实 / 表格 / 区段结构）。

本模块只做**防御读取与展示格式化**，不含任何指标计算，也不做 HTML 转义
（转义由模板过滤器 ``p6text`` 统一负责）。设计约束：

- T1 投影允许“不可用态字段缺席”，故一律用 ``row_of``/``rows_of`` 取；缺席显式写
  ``（字段缺席）``/``（未读取）``，**禁止把缺席当 0 或空串**；
- 中性空列表（如 ``nav.rows=[]``）表示“未读取”，零成交只看 ``empty_valid`` 状态词；
- 展示格式化固定小数位与千分位，不受 locale 影响（服务字节一致性门）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import status

__all__ = [
    "MAX_DEGRADATION_ROWS",
    "MAX_INSTRUCTION_ROWS",
    "MAX_PENDING_ROWS",
    "MAX_POSITION_ROWS",
    "MAX_QUALITY_ROWS",
    "MAX_TRADE_ROWS",
    "MISSING_VALUE",
    "NOT_READ_VALUE",
    "STATUS_LABELS",
    "VIEW_SCHEMA",
    "EMPTY_TABLE_NOTE",
    "EMPTY_VALID_TABLE_NOTE",
    "STALE_TABLE_NOTE",
    "UNAVAILABLE_TABLE_NOTE",
    "bool_of",
    "cap_note",
    "combine_status",
    "empty_note",
    "fact_view",
    "join_text",
    "number_of",
    "percent_of",
    "row_of",
    "rows_of",
    "section_view",
    "status_view",
    "table_view",
    "text_of",
]

#: 视图模型 schema（不进入 canonical 报告，仅渲染入口校验）
VIEW_SCHEMA = "p6_dashboard_view/v1"
#: 字段缺席（键不存在或状态区未提供）——不得替换为 0/空串
MISSING_VALUE = "（字段缺席）"
#: 中性空列表/未读取来源的显式说明
NOT_READ_VALUE = "（未读取）"
#: 六状态的中文标签（闭集，与 ``status.STATUS_WORDS`` 一一对应）
STATUS_LABELS: dict[str, str] = {
    "available": "可用",
    "empty_valid": "合法空",
    "degraded": "降级",
    "not_available": "不可用",
    "stale": "陈旧",
    "invalid": "无效",
}
#: 各区可见行上限（超限只减可见行并显式说明；完整值见 projection.json）
MAX_POSITION_ROWS = 200
MAX_INSTRUCTION_ROWS = 200
MAX_PENDING_ROWS = 100
MAX_TRADE_ROWS = 100
MAX_QUALITY_ROWS = 100
MAX_DEGRADATION_ROWS = 100
#: 状态严重度（合并多路状态时取最严重；``stale`` 视为轻于 ``degraded``）
_SEVERITY: dict[str, int] = {
    "available": 0,
    "empty_valid": 0,
    "stale": 1,
    "degraded": 2,
    "not_available": 3,
    "invalid": 4,
}
#: 必须携带 reason_code 的状态（与 ``status`` 内部闭集同口径，合并时兜底用）
_REASON_REQUIRED: frozenset[str] = frozenset({"degraded", "not_available", "invalid", "stale"})


def row_of(value: Any) -> dict[str, Any]:
    """取子对象（非 Mapping 一律视为缺席，不抛裸异常）。"""
    return dict(value) if isinstance(value, Mapping) else {}


def rows_of(value: Any) -> list[dict[str, Any]]:
    """取行列表（只保留 Mapping 行；非列表视为空，调用方须自行区分“未读取”）。"""
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def text_of(value: Any, *, missing: str = MISSING_VALUE) -> str:
    """来源文本原值（``None`` 显式标缺席，不写空串冒充）。"""
    return missing if value is None else str(value)


def number_of(value: Any, *, digits: int = 2, missing: str = MISSING_VALUE) -> str:
    """数值原文格式化（固定小数位；非数值/非有限显式标缺席）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return missing
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return missing
    return f"{number:,.{digits}f}"


def percent_of(value: Any, *, digits: int = 4, missing: str = MISSING_VALUE) -> str:
    """百分数原文（唯一实现返回百分数，渲染侧不再换算）。"""
    text = number_of(value, digits=digits, missing=missing)
    return text if text == missing else f"{text}%"


def join_text(values: Sequence[Any]) -> str:
    """字符串列表拼接（空显式写“无”，不用空串掩盖）。"""
    items = [str(item) for item in values if str(item)]
    return "、".join(items) if items else "无"


def bool_of(value: Any) -> str:
    """三值布尔文本：``True`` → 是、``False`` → 否、其余（含键缺席）→ 字段缺席。"""
    if value is True:
        return "是"
    if value is False:
        return "否"
    return MISSING_VALUE


def status_view(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """状态行 → 视图（状态词/原因码/明细/验证级别；未知状态词按 invalid 处理）。"""
    material = row_of(row)
    word = str(material.get("report_status") or "not_available")
    if word not in status.STATUS_WORDS:
        word = "invalid"
    reason = material.get("degradation_reason_code")
    return {
        "word": word,
        "label": STATUS_LABELS[word],
        "reason": str(reason) if reason else "—",
        "detail": str(material.get("detail") or "") or "—",
        "verification": str(material.get("verification_level") or NOT_READ_VALUE),
        "usable": word in ("available", "empty_valid"),
        "css": f"status status--{word}",
    }


def combine_status(
    rows: Sequence[Mapping[str, Any]], *, detail: str, reason_code: str = "no_source"
) -> dict[str, Any]:
    """把若干状态行合并成一条区级状态行（状态词与原因码**成对**取最严重那一路）。

    ``word`` 取最严重状态词；``reason_code`` 只从**与该状态词同级**的第一行取
    （否则会出现 ``invalid`` + ``no_source`` 这类自相矛盾，T2-R1-N1）。
    """
    materials = [row_of(row) for row in rows]
    words = [str(item.get("report_status") or "not_available") for item in materials]
    word = max(words or ["not_available"], key=lambda item: _SEVERITY.get(item, 3))
    reason = next(
        (
            str(item.get("degradation_reason_code"))
            for item in materials
            if str(item.get("report_status") or "") == word
            and item.get("degradation_reason_code")
        ),
        None,
    )
    if word in _REASON_REQUIRED and reason is None:
        reason = reason_code
    verified = bool(materials) and all(
        item.get("verification_level") == "authority_verified" for item in materials
    )
    return status.make_status(
        word,
        reason_code=reason,
        detail=detail,
        verification_level="authority_verified" if verified else "projection_only",
    )


def fact_view(
    label: str, value: str, *, row: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """键值事实（``row`` 提供时同时展示该值自己的状态与原因码）。"""
    return {"label": label, "value": value, "status": status_view(row) if row is not None else None}


#: 表格无行且状态未知时的固定文案（禁止留空段落掩盖“无行”）
EMPTY_TABLE_NOTE = "无可见行（状态见本表状态与区级说明）"
#: ``empty_valid`` 时的空表文案（语义与状态词一致：**已核实**为零）
EMPTY_VALID_TABLE_NOTE = "已核实为空：本表无行（零值不是缺值）"
#: 非可用态（缺来源/损坏/降级）时的空表文案
UNAVAILABLE_TABLE_NOTE = "来源不可用或已降级：不伪造空表（原因见本表状态）"
#: ``stale`` 时的空表文案（保留旧值 ≠ 缺来源；不承诺本表状态里有日期，T2-R1-N8）
STALE_TABLE_NOTE = "保留最近一次已核对值（陈旧）：本表无新行，日期与原因见本节事实与本表状态"


def empty_note(row: Mapping[str, Any] | None) -> str:
    """按本表状态词选空表文案（``empty_valid``/``stale``/不可用必须区分，T2-R1-06）。"""
    word = str(row_of(row).get("report_status") or "")
    if word == "empty_valid":
        return EMPTY_VALID_TABLE_NOTE
    if word == "stale":
        return STALE_TABLE_NOTE
    if word in _REASON_REQUIRED:
        return UNAVAILABLE_TABLE_NOTE
    return EMPTY_TABLE_NOTE


def table_view(
    caption: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    total: int,
    note: str = "",
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """表格视图（``rows`` 已按上限裁剪，``total`` 为真实行数；空表按状态词给文案）。"""
    material = [list(item) for item in rows]
    return {
        "caption": caption,
        "columns": list(columns),
        "rows": material,
        "total": int(total),
        "note": note or ("" if material else empty_note(row)),
        "status": status_view(row) if row is not None else None,
    }


def cap_note(total: int, shown: int, what: str) -> str:
    """可见行裁剪说明（超限先减可见行；完整值仍在 projection.json）。"""
    if total <= shown:
        return ""
    return f"{what}共 {total} 行，页面仅显示前 {shown} 行（完整值见 projection.json）"


def section_view(
    section_id: str,
    title: str,
    row: Mapping[str, Any] | None,
    *,
    facts: Sequence[Mapping[str, Any]] = (),
    tables: Sequence[Mapping[str, Any]] = (),
    charts: Sequence[Mapping[str, Any]] = (),
    note: str = "",
) -> dict[str, Any]:
    """区段统一结构（状态块 + 键值事实 + 表格 + 图表 + 说明）。"""
    return {
        "id": section_id,
        "title": title,
        "status": status_view(row),
        "facts": [dict(item) for item in facts],
        "tables": [dict(item) for item in tables],
        "charts": [dict(item) for item in charts],
        "note": note,
    }
