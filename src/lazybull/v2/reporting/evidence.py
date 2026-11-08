# -*- coding: utf-8 -*-
"""证据域隔离与固定横幅文案（设计 §2.2 / §6）。

硬口径（P6 T2）：

- 三类证据域**物理与视觉隔离**，不提供跨类总览：``production`` 只接受 production 输入，
  ``calibration_only`` 只接受 calibration 输入，``research`` 可含 production/research 输入
  但**不得混入 calibration**（calibration_only 不与其他类混合）；
- 任何未声明类、跨类组合一律 **fail-closed**（非零退出），不生成“有效混合报告”；
- 横幅文案是**固定常量**（唯一来源），模板不得自造或省略：production 蓝、research 灰、
  calibration_only 琥珀整页边框；
- 切换前记录形态必须在横幅区**可见**（``PRE_SWITCH_NOTE``），不得只在降级清单里提一句。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "BANNER_CLASSES",
    "EVIDENCE_BANNERS",
    "EvidenceIsolationError",
    "PRE_SWITCH_NOTE",
    "assert_input_evidence",
    "banner_for",
    "mixed_evidence_classes",
]

#: 每类证据域的固定横幅（``text`` 即设计原文口径，不得改写）
EVIDENCE_BANNERS: dict[str, dict[str, str]] = {
    "production": {
        "text": "纸面生产投影；非实盘；不产生决策",
        "class": "banner--production",
        "note": "只读投影：不推进 T0/T1、不下单、不调账、不修改账本。",
    },
    "research": {
        "text": "研究/对账；非生产结论",
        "class": "banner--research",
        "note": "含纸面/回测/回放读数时逐块标来源，不得升格为生产摘要。",
    },
    "calibration_only": {
        "text": "仅校准；非前向；不得生产裁决",
        "class": "banner--calibration",
        "note": "仅陈列经封存校验的 P5b bundle；不得替代用户闸门判定。",
    },
}
#: 允许出现在 CSS/模板中的横幅类名（闭集，防模板自造）
BANNER_CLASSES: tuple[str, ...] = tuple(item["class"] for item in EVIDENCE_BANNERS.values())
#: 切换前记录形态的固定可见说明（横幅区强制展示）
PRE_SWITCH_NOTE = (
    "非可证明的 v2 驱动代：当前 run record 同时缺 run_kind 与 engine（切换前记录形态），"
    "账户值仅降级展示、不作认证，未来指令与实跑模型不推断。"
)
#: 触发上述说明的原因码（与 ``status.REASON_CODES`` 同名）
PRE_SWITCH_REASON = "run_record_pre_switch_shape"
#: 每类证据域**允许**的输入类（跨类即 fail-closed）
ALLOWED_INPUT_CLASSES: dict[str, frozenset[str]] = {
    "production": frozenset({"production"}),
    "research": frozenset({"production", "research"}),
    "calibration_only": frozenset({"calibration_only"}),
}


class EvidenceIsolationError(ValueError):
    """证据域混合或未声明（必须非零退出，禁止生成“有效”混合报告）。"""


def mixed_evidence_classes(
    evidence_class: str, input_evidence_classes: Sequence[str]
) -> list[str]:
    """返回越界/未声明的输入证据类（空表示合法）。"""
    if evidence_class not in ALLOWED_INPUT_CLASSES:
        raise EvidenceIsolationError(f"未知 evidence_class：{evidence_class!r}")
    allowed = ALLOWED_INPUT_CLASSES[evidence_class]
    return sorted({str(item) for item in input_evidence_classes} - allowed)


def assert_input_evidence(evidence_class: str, input_evidence_classes: Sequence[str]) -> None:
    """证据域隔离硬门：混合即抛 ``EvidenceIsolationError``（不降级、不裁剪输入）。"""
    if not list(input_evidence_classes):
        raise EvidenceIsolationError(
            f"{evidence_class} 面必须显式声明至少一个输入证据类"
            "（空声明会让 manifest 的 input_evidence_classes 失去意义）"
        )
    offenders = mixed_evidence_classes(evidence_class, input_evidence_classes)
    if offenders:
        raise EvidenceIsolationError(
            f"{evidence_class} 面不得接收 {offenders} 类输入"
            f"（允许 {sorted(ALLOWED_INPUT_CLASSES[evidence_class])}）；"
            "禁止混合证据域生成报告"
        )


def banner_for(evidence_class: str, record_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """返回横幅视图（固定文案 + 是否展示切换前形态说明）。"""
    if evidence_class not in EVIDENCE_BANNERS:
        raise EvidenceIsolationError(f"未知 evidence_class：{evidence_class!r}")
    banner = dict(EVIDENCE_BANNERS[evidence_class])
    row = record_row if isinstance(record_row, Mapping) else {}
    pre_switch = str(row.get("degradation_reason_code") or "") == PRE_SWITCH_REASON
    return {
        "text": banner["text"],
        "css": banner["class"],
        "note": banner["note"],
        "pre_switch_visible": pre_switch,
        "pre_switch_note": PRE_SWITCH_NOTE if pre_switch else "",
    }
