# -*- coding: utf-8 -*-
"""P7 预登记跨字段一致性检查（R3R-01）。

作用对象是**将要序列化或已序列化的完整 prereg**：同一 ΔMaxDD 方向只允许 signed
``arm − base`` 一处来源（``crash_gate.decision_rule``），E4-1 必须引用而非重写。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

REVERSE_TOKENS = ("base − arm", "base - arm", "base−arm", "base_arm")
FORWARD_TOKEN = "arm − base"


def _iter_strings(node: Any) -> Iterator[str]:
    """递归遍历 JSON 结构中的所有字符串。"""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _iter_strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_strings(item)


def _assert_no_reverse_terms(payload: Dict[str, Any]) -> None:
    for text in _iter_strings(payload):
        for token in REVERSE_TOKENS:
            if token in text:
                raise ValueError(f"prereg 出现反向 ΔMaxDD 定义：{text[:80]}")


def _assert_e4_entry(payload: Dict[str, Any]) -> None:
    gates = payload.get("gates") or {}
    crash = payload.get("crash_gate") or {}
    entries = {
        item.get("id"): item for item in gates.get("E4_vetoes") or [] if isinstance(item, dict)
    }
    entry = entries.get("E4-1") or {}
    if entry.get("rule") != crash.get("decision_rule"):
        raise ValueError("E4-1.rule 必须引用 crash_gate.decision_rule（唯一来源）")
    if FORWARD_TOKEN not in str(entry.get("rule")):
        raise ValueError("E4-1.rule 未采用 signed arm − base")
    if float(entry.get("tolerance_pp", -1)) != float(crash.get("tolerance_pp", -2)):
        raise ValueError("E4-1 容差与崩盘门不一致")
    if float(crash.get("tolerance_pp", -1)) != 0.0:
        raise ValueError("崩盘门容差必须为 0")
    if FORWARD_TOKEN not in str(crash.get("delta_sign")):
        raise ValueError("crash_gate.delta_sign 未采用 signed arm − base")


def _assert_e3_entry(payload: Dict[str, Any]) -> None:
    crash = payload.get("crash_gate") or {}
    interval = crash.get("interval") or {}
    e3 = interval.get("e3_paired_delta_mdd") or {}
    if e3.get("sample_unit") != "fold":
        raise ValueError("E3 必须是折单位制度重排")
    if FORWARD_TOKEN not in str(e3.get("pass_rule")):
        raise ValueError("E3 pass_rule 未采用 signed arm − base")
    examples = e3.get("sign_examples") or {}
    improve = examples.get("improve") or {}
    worsen = examples.get("worsen") or {}
    if round(float(improve.get("delta", 0.0)), 6) != 0.10:
        raise ValueError("E3 正例必须是 +0.10")
    if round(float(worsen.get("delta", 0.0)), 6) != -0.10:
        raise ValueError("E3 反例必须是 −0.10")


def assert_prereg_consistency(payload: Dict[str, Any]) -> None:
    """校验最终 prereg 的 ΔMaxDD 方向与 E3/E4 字段一致。"""
    _assert_no_reverse_terms(payload)
    _assert_e4_entry(payload)
    _assert_e3_entry(payload)
