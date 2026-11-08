# -*- coding: utf-8 -*-
"""驾驶舱视图模型门面（设计 §4.1：六区 + 证据横幅 + 降级清单 + research 三端卡）。

分层（避免单文件膨胀，同时保持公开入口稳定）：

- ``view_primitives``：状态视图 / 键值事实 / 表格 / 区段结构等**原语**；
- ``dashboard_sections``：六区与附属面板（账户 / 持仓 / 指令 / pending / 调仓 /
  模型·质量·新鲜度 / 降级清单 / 三端卡）；
- 本模块：``build_dashboard_view`` 门面 + 常量与常用原语转导出。

职责边界（P6 T2）：

- 只**读** T1 投影（``cockpit.build_dashboard_projection``），不做任何新指标、不补值、不重算；
- 渲染侧防御读取：字段缺席显示 ``（字段缺席）``/``（未读取）``，**禁止把缺席当 0 或空串**；
- 全部来源文本原样交给模板，由过滤器 ``p6text`` 做 NFC + 转义（不在此处预转义）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import evidence, text_safety
from src.lazybull.v2.reporting.dashboard_sections import (
    account_section,
    all_sections,
    degradations_section,
    instructions_section,
    model_quality_section,
    pending_section,
    positions_section,
    project_three_ends,
    rebalance_section,
)
from src.lazybull.v2.reporting.view_primitives import (
    MISSING_VALUE,
    NOT_READ_VALUE,
    STATUS_LABELS,
    VIEW_SCHEMA,
    row_of,
    status_view,
    text_of,
)

__all__ = [
    "MISSING_VALUE",
    "NOT_READ_VALUE",
    "STATUS_LABELS",
    "VIEW_SCHEMA",
    "account_section",
    "all_sections",
    "build_dashboard_view",
    "degradations_section",
    "instructions_section",
    "model_quality_section",
    "pending_section",
    "positions_section",
    "project_three_ends",
    "rebalance_section",
    "status_view",
]


def build_dashboard_view(
    projection: Mapping[str, Any],
    *,
    report_id: str,
    generated_at: str,
    input_evidence_classes: Sequence[str],
    generator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """装配驾驶舱视图（六区 + 降级清单 + research 三端卡 + 固定 companion 链接）。

    三端身份**只从 projection 读**（T2-R3-03）：额外注入会破坏“只读 projection + 包内可复算”，
    一律在 ``generate_dashboard`` 入口拒绝。
    """
    material = row_of(projection)
    evidence_class = str(material.get("evidence_class") or "")
    banner = evidence.banner_for(evidence_class, row_of(material.get("record")))
    return {
        "schema": VIEW_SCHEMA,
        "lang": "zh-CN",
        "title": f"LazyBull 纸面驾驶舱（{evidence_class}）",
        "report": {
            "report_id": report_id,
            "report_kind": "dashboard",
            "generated_at": generated_at,
            "evidence_class": evidence_class,
            "input_evidence_classes": "、".join(
                sorted(str(item) for item in input_evidence_classes)
            ),
            "projection_schema": text_of(material.get("schema")),
            "projection_generated_at": text_of(material.get("generated_at")),
        },
        "banner": banner,
        "sections": all_sections(material),
        "three_ends": (
            project_three_ends(material)
            if evidence_class == "research"
            else {"visible": False, "title": "", "rows": [], "note": ""}
        ),
        "companions": text_safety.companion_links(),
        "footer": {
            "generator": "、".join(
                f"{key}={value}" for key, value in sorted(dict(generator or {}).items())
            )
            or "—",
            "note": (
                "本页为只读静态投影：不产生交易指令、不推进 T0/T1、不修改账本或模型；"
                "历史回测与纸面读数不代表未来收益。"
            ),
        },
    }
