# -*- coding: utf-8 -*-
"""P6 T2 合成夹具：投影/视图/渲染辅助（**不引用任何真实生产数据**）。

提供三类能力：

1. **真实形态投影**：复用 T1 合成纸面根（``tests.p6_t1_env``）走一遍稳定快照 + 投影，
   得到与生产同构的驾驶舱 projection（六区三态的可产出形态）；
2. **渲染直入口**：``render_projection`` 绕过快照直接渲染给定 projection，用于
   “渲染侧防御读取 / 空态 / 降级态”等只与视图有关的用例；
3. **恶意语料**：``malicious_projection`` 把 P6-T2-01 的注入串放进每个来源字段
   （代码/原因/明细/日期/配置/质量/降级），用于 DOM 快照证据。
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import dashboard_view, evidence, render, status

#: 字段缺席文案（与视图模型同源，测试直接引用避免各写一份）
MISSING = dashboard_view.MISSING_VALUE
#: 未读取文案
NOT_READ = dashboard_view.NOT_READ_VALUE

#: 视为“固定身份”的节点报告号（形态合法，避免下游形态校验误判）
SAMPLE_REPORT_ID = "p6_dashboard_20261106T000000Z_0123456789ab"
#: 固定生成的显式输入（新墙钟即新 run）
SAMPLE_GENERATED_AT = "20261106T000000Z"
#: P6-T2-01 注入串（HTML/SVG/script/URL/事件属性/**属性逃逸**/Unicode 分隔符/NFC 分解）
MALICIOUS_TEXTS: tuple[str, ...] = (
    '<script>alert("t2-01")</script>',
    '<img src=x onerror="alert(1)">',
    "javascript:alert(1)",
    "https://evil.example/payload",
    "<iframe src='https://evil.example'></iframe>",
    '">&lt;/textarea&gt;<svg onload=alert(1)>',
    "nfc-e\u0301-decomposed",
    "sep\u2028line\u2029para",
    "onclick=alert(1)",
    "<svg><script>alert(2)</script></svg>",
    # 属性逃逸形态：若来源文本被放进引号内的属性，会闭合属性并注入新属性
    'x" hidden data-x="y',
    'x\' onmouseover=\'alert(1)',
    "//evil.example/protocol-relative",
    "content=\"0;url=//evil.example/x\"",
)


def _status_row(word: str, **kwargs: Any) -> dict[str, Any]:
    """合成状态行（复用闭集构造器，避免测试自造状态词）。"""
    return status.make_status(word, **kwargs)


def render_projection(
    projection: Mapping[str, Any],
    *,
    input_evidence_classes: Sequence[str] | None = None,
    generated_at: str = SAMPLE_GENERATED_AT,
) -> str:
    """直接渲染给定 projection（不经快照；只测视图/模板层，含强制安全自检）。

    视图**不接受**额外三端输入（T2-R3-03）：三端身份只从 projection 读。
    """
    evidence_class = str(projection.get("evidence_class") or "production")
    view = dashboard_view.build_dashboard_view(
        projection,
        report_id=SAMPLE_REPORT_ID,
        generated_at=generated_at,
        input_evidence_classes=tuple(
            input_evidence_classes or (evidence_class,)
        ),
        generator={"project": "lazybull", "synthetic": "true"},
    )
    return render.render_dashboard_html(view)


def build_view(
    projection: Mapping[str, Any],
    *,
    generated_at: str = SAMPLE_GENERATED_AT,
) -> dict[str, Any]:
    """构造视图模型（供需要断言 view 结构的用例）。"""
    evidence_class = str(projection.get("evidence_class") or "production")
    return dashboard_view.build_dashboard_view(
        projection,
        report_id=SAMPLE_REPORT_ID,
        generated_at=generated_at,
        input_evidence_classes=(evidence_class,),
        generator={"project": "lazybull", "synthetic": "true"},
    )


def section_slice(html: str, section_id: str) -> str:
    """截取单个区段 HTML（区段不嵌套，非贪婪匹配安全）。"""
    pattern = rf'<section id="{re.escape(section_id)}".*?</section>'
    match = re.search(pattern, html, flags=re.S)
    if match is None:
        raise AssertionError(f"HTML 中找不到区段 {section_id}")
    return match.group(0)


def fact_value(html: str, label: str) -> str:
    """取视图里某个键值事实的 ``<dd>`` 纯文本（剔除内嵌状态 span 与标签）。"""
    pattern = rf"<dt>{re.escape(label)}</dt>\s*<dd>(.*?)</dd>"
    match = re.search(pattern, html, flags=re.S)
    if match is None:
        raise AssertionError(f"HTML 中找不到键值事实 {label}")
    body = re.sub(r'<span class="mini">.*?</span>', "", match.group(1), flags=re.S)
    return re.sub(r"<[^>]+>", "", body).strip()


def status_of(view: Mapping[str, Any], section_id: str) -> dict[str, Any]:
    """取视图里某区段的状态视图（断言三态用）。"""
    for section in view.get("sections") or []:
        if section.get("id") == section_id:
            return dict(section["status"])
    raise AssertionError(f"视图里找不到区段 {section_id}")


def clone(projection: Mapping[str, Any]) -> dict[str, Any]:
    """深拷贝 projection（用于逐区替换状态/删字段）。"""
    return copy.deepcopy(dict(projection))


def with_section(projection: Mapping[str, Any], dotted: str, row: Mapping[str, Any]) -> dict:
    """替换（或新增）某个区段的状态行；``dotted`` 支持 ``pending.buys`` 形式。"""
    material = clone(projection)
    cursor: dict[str, Any] = material
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = dict(row)
    return material


def without_keys(projection: Mapping[str, Any], dotted: str, *keys: str) -> dict:
    """删除区段内的指定键（模拟“不可用态字段缺席”，验证渲染侧防御读取）。"""
    material = clone(projection)
    cursor: Any = material
    for part in dotted.split("."):
        cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if cursor is None:
            return material
    if isinstance(cursor, dict):
        for key in keys:
            cursor.pop(key, None)
    return material


def malicious_projection() -> dict[str, Any]:
    """P6-T2-01 夹具：每个来源文本字段都带注入串（状态词/原因码保持闭集合法）。"""
    bad = MALICIOUS_TEXTS
    degraded = _status_row(
        "degraded",
        reason_code="authority_mismatch",
        detail=bad[1],
        source_ref=bad[3],
        verification_level="projection_only",
    )
    positions = {
        bad[1]: {
            "ts_code": bad[0],
            "shares": 7,
            "buy_price": 1.5,
            "buy_cost": 0.5,
            "buy_date": bad[5],
            "buy_pnl_price": 1.6,
            "status": bad[8],
        },
        f"code-{bad[9]}": {
            "ts_code": bad[9],
            "shares": 9,
            "buy_price": 2.5,
            "buy_cost": 0.2,
            "buy_date": bad[2],
            "buy_pnl_price": 2.6,
            "status": bad[7],
        },
    }
    authority = {**degraded, "trade_date": bad[7], "run_kind": bad[4]}
    account = {
        **degraded,
        "account": {"cash": 10.0, "last_update": bad[6], "positions": positions},
    }
    nav = {
        **degraded,
        "nav_date": bad[7],
        "nav_date_status": bad[8],
        "rows": [
            {"trade_date": bad[7], "present": True, "matched": True, "diffs": []},
            {"trade_date": bad[6], "present": True, "matched": False, "diffs": []},
        ],
        "nav_values": {
            bad[7]: {"total_value": 12.5, "position_value": 10.0, "cash": 2.5},
            bad[6]: {"total_value": 11.5, "position_value": 9.0, "cash": 2.5},
        },
        "verified_rows": [bad[7]],
        "broken_links": [bad[6]],
        "nav_only_days": [bad[9]],
        "annualization_allowed": False,
    }
    trades = {
        **degraded,
        "trades": [
            {
                "trade_date": bad[7],
                "ts_code": bad[0],
                "action": bad[8],
                "shares": 1,
                "price": 1.0,
                "amount": 1.0,
                "reason": bad[5],
            }
        ],
        "broken_rows": [bad[9]],
    }
    instructions = {
        **degraded,
        "execution_date": bad[7],
        "future_files": [bad[7]],
        "history_files": 1,
        "instruction_count": 1,
        "instructions": [
            {
                "ts_code": bad[0],
                "action": bad[8],
                "shares": 1,
                "price_type": bad[4],
                "target_weight": 0.5,
                "source_date": bad[7],
                "reason": bad[5],
            }
        ],
    }
    pending_row = {
        **degraded,
        "pending_count": 1,
        "pending": [
            {"ts_code": bad[10], "action": bad[8], "shares": 1, "price": 1.0, "reason": bad[9]}
        ],
    }
    rebalance = {
        **degraded,
        "rebalance": {
            "last_rebalance_date": bad[5],
            "last_scheduled_rebalance_date": bad[6],
            "tranche_anchor_date": bad[11],
            "rebalance_freq": bad[8],
            "stagger_tranches": bad[9],
        },
    }
    model = {
        **degraded,
        "model_version": bad[0],
        "model_version_b": bad[1],
        "ensemble_weight_a": 0.7,
        "ensemble_weight_b": 0.3,
        "follows_latest": False,
        "actual_model": _status_row(
            "not_available",
            reason_code="no_persisted_actual_model",
            detail=bad[3],
        ),
    }
    config = {**degraded, "account_start_date": bad[6], "model": model}
    config["annualization"] = _status_row(
        "invalid", reason_code="annualization_input_invalid", detail=bad[2]
    )
    quality = _status_row(
        "available",
        detail=f"{bad[5]} 行",
        extra={
            "metric_count": 3,
            "error_count": 1,
            "warning_count": 1,
            "datasets": [
                {
                    "layer": bad[4],
                    "dataset": bad[12],
                    "metric_count": 3,
                    "error_count": 2,
                    "warning_count": 1,
                    "partition_count": 4,
                    "latest_partition": bad[6],
                    "sync_watermark": bad[7],
                }
            ],
        },
    )
    quality["scan_time"] = _status_row("not_available", reason_code="no_source", detail=bad[3])
    record = _status_row(
        "degraded", reason_code="run_record_pre_switch_shape", detail=bad[1]
    )
    return {
        "schema": "p6_dashboard_projection/v1",
        "generated_at": SAMPLE_GENERATED_AT,
        "evidence_class": "research",
        "root_map": {"paper": f"合成根 {bad[3]}"},
        "authority": authority,
        "record": record,
        "account": account,
        "nav": nav,
        "trades": trades,
        "instructions": instructions,
        "pending": {"buys": pending_row, "sells": copy.deepcopy(pending_row)},
        "rebalance": rebalance,
        "config": config,
        "quality": quality,
        "degradations": [
            {
                "field_path": bad[13],
                "report_status": "invalid",
                "reason_code": "authority_mismatch",
                "source_ref": bad[3],
                "detail": bad[0],
            }
        ],
    }


def forced_section_status(
    word: str, *, reason_code: str | None = None, detail: str = "合成状态行"
) -> dict[str, Any]:
    """构造指定状态词的最小状态行（三态覆盖用例）。"""
    return status.make_status(word, reason_code=reason_code, detail=detail)


def banner_view(evidence_class: str, record_row: Mapping[str, Any] | None = None) -> dict:
    """取固定横幅视图（文案唯一来源 ``evidence``）。"""
    return evidence.banner_for(evidence_class, record_row)


def case_paper_root(tmp_path: Path, profile: str = "standard") -> tuple[Path, Path]:
    """用 T2 语料构造器生成本地合成纸面根与质量根（返回 ``(paper, quality)``）。"""
    from scripts.v2_p6.build_t2_corpus import build_case

    payload = build_case(tmp_path / profile, profile)
    return Path(payload["paper"]), Path(payload["quality_root"])
