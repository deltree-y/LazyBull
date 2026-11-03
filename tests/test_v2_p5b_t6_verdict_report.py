# -*- coding: utf-8 -*-
"""P5b T6 判定表与报告测试（tests/ 豁免复杂度）。

覆盖：固定 claim 字段、无「闸门通过」字段、H1/H6/H7 由证据推导（含 ``not_executed``）、
``partial`` 不得变 pass、§7.2 报告字段齐备、markdown 渲染。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t6_env as t6env  # noqa: E402

from src.lazybull.v2.hosts.testbed.rehearsal.assembly import (  # noqa: E402
    forbidden_gate_fields,
)
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (  # noqa: E402
    arm_effect_plan,
)
from src.lazybull.v2.hosts.testbed.rehearsal.report import (  # noqa: E402
    build_report,
    render_markdown,
)
from src.lazybull.v2.hosts.testbed.rehearsal.verdict import (  # noqa: E402
    build_verdict,
    h1_conditions,
    h6_conditions,
    h7_conditions,
)

WINDOW = {
    "d0_date": "20260706",
    "first_return_date": "20260707",
    "virtual_days": [f"202607{d:02d}" for d in range(6, 16)],
    "day_count": 10,
    "return_days": 9,
    "selection_rule": "测试",
    "deletion_forbidden": True,
}
ROW_DOMAIN = {"declared": "structural", "verified": True, "per_day": {"20260706": {}}}
ARMS = {
    arm: {
        "driven_days": 10,
        "authoritative_values": 10,
        "complete_generations": 10,
        "row_counts_all_one": True,
        "row_counts_not_one": [],
    }
    for arm in t6env.ARMS
}
AA = {
    "identical": True,
    "compared_days": 10,
    "compared_fields": 60,
    "mismatches": [],
    "source": "shard_recompute",
}


def test_verdict_fixed_claims_and_no_gate_pass_field():
    verdict = build_verdict(
        run_id="r",
        campaign_id="c",
        manifest_sha256="a" * 64,
        calibrator_sha256="b" * 64,
        artifact_hashes={"rehearsal_report.json": "c" * 64},
        h1={"conditions": [], "author_side_status": "pass"},
        h6={"conditions": [], "author_side_status": "not_executed"},
        h7={"conditions": [], "author_side_status": "pass"},
        effects={},
    )
    assert verdict["h1_claim"] == "evaluated_not_declared_pass"
    assert verdict["h6_claim"] == "evaluated_not_declared_pass"
    assert verdict["h7_claim"] == "evaluated_not_declared_pass"
    assert verdict["gate_owner"] == "user_at_merge_acceptance"
    assert forbidden_gate_fields(verdict) == []
    assert "gate_passed" not in verdict and "passed" not in verdict


def test_h1_status_derives_from_evidence():
    passing = h1_conditions(
        window=WINDOW,
        row_domain=ROW_DOMAIN,
        aa=AA,
        arm_summaries=ARMS,
        timings={"production_zero_write": True},
    )
    assert passing["author_side_status"] == "pass"
    failing = h1_conditions(
        window=WINDOW,
        row_domain=ROW_DOMAIN,
        aa={**AA, "identical": False},
        arm_summaries=ARMS,
        timings={"production_zero_write": True},
    )
    assert failing["author_side_status"] == "fail"
    # 逐日「恰一条」被破坏（合计仍为 10）⇒ 必须失败（合计口径不足以发现 0+2）
    uneven = {
        **ARMS,
        "base": {
            **ARMS["base"],
            "row_counts_all_one": False,
            "row_counts_not_one": [
                {"virtual_date": "20260706", "row_count": 0},
                {"virtual_date": "20260707", "row_count": 2},
            ],
        },
    }
    assert (
        h1_conditions(
            window=WINDOW,
            row_domain=ROW_DOMAIN,
            aa=AA,
            arm_summaries=uneven,
            timings={"production_zero_write": True},
        )["author_side_status"]
        == "fail"
    )
    unknown = h1_conditions(
        window=WINDOW,
        row_domain=ROW_DOMAIN,
        aa=AA,
        arm_summaries=ARMS,
        timings={"production_zero_write": False},
    )
    assert unknown["author_side_status"] == "not_executed"
    short_window = h1_conditions(
        window={**WINDOW, "virtual_days": WINDOW["virtual_days"][:9], "day_count": 9},
        row_domain=ROW_DOMAIN,
        aa=AA,
        arm_summaries=ARMS,
        timings={"production_zero_write": True},
    )
    assert short_window["author_side_status"] == "fail"


def test_h6_partial_and_not_constructed_are_not_executed():
    items = [
        {"name": "income_minus_5d", "status": "partial", "target": "−5δ", "judgment": "退化方差"},
        {
            "name": "dd_improve_12pp",
            "status": "not_executed",
            "target": "+12pp",
            "judgment": "不可构造",
            "constructibility": "not_constructible",
        },
        {"name": "dd_worsen_12pp", "status": "fail", "target": "−12pp", "judgment": "未触发"},
    ]
    conditions = h6_conditions(items)
    assert conditions["author_side_status"] == "fail"
    statuses = [item["status"] for item in conditions["conditions"]]
    assert statuses == ["fail", "not_executed", "fail"]
    all_partial = h6_conditions(
        [{**items[0]}, {**items[1]}, {**items[1], "name": "dd_worsen_12pp"}]
    )
    assert all_partial["author_side_status"] == "fail"


def test_h7_status_requires_upstream_runbook_version_and_boundary():
    manifest = {"limitation_count": 9, "no_gate_pass_field": True}
    passing = h7_conditions(
        manifest=manifest,
        upstream={"verified": True},
        runbook={"verified": True, "path": "x", "sha256": "d" * 64},
        version={"synchronised": True},
    )
    assert passing["author_side_status"] == "pass"
    missing_runbook = h7_conditions(
        manifest=manifest,
        upstream={"verified": True},
        runbook={"verified": False},
        version={"synchronised": True},
    )
    assert missing_runbook["author_side_status"] == "not_executed"
    bad_version = h7_conditions(
        manifest=manifest,
        upstream={"verified": True},
        runbook={"verified": True},
        version={"synchronised": False},
    )
    assert bad_version["author_side_status"] == "fail"


def _document(tmp_path: Path) -> dict:
    env = SimpleNamespace(
        data_root=tmp_path / "data", base_root=tmp_path / "testbed", days=WINDOW["virtual_days"]
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    repo = t6env.build_repo(tmp_path)
    document = t6env.manifest_document(env, repo)
    document["window"]["virtual_days"] = WINDOW["virtual_days"]
    return document


def test_report_carries_required_fields(tmp_path):
    document = _document(tmp_path)
    thresholds = {
        "look_days": [120, 240, 360, 500],
        "look_blocks": [6, 12, 18, 25],
        "h": 20,
        "c_k": [4.7, 3.4, 3.0, 2.9],
        "c_w": [-17.8, -30.5, -31.0, -31.1],
        "c_i": [7.9, 6.5, 5.5, 5.1],
        "delta_log_daily": 0.0001936117626,
        "anchor": 500,
    }
    readings = {
        "h6_items": t6env_not_executed(),
        "income": {"sample_days": 0, "rows": []},
        "drawdown": {},
        "thresholds": thresholds,
        "effects": arm_effect_plan(document, 9),
    }
    report = build_report(
        campaign_id="c",
        run_id="r",
        manifest=document,
        manifest_sha256="a" * 64,
        calibrator={"implementation_id": "i", "sha256": "b" * 64, "modules": {"x.py": "c" * 64}},
        upstream=dict(document["upstream"]),
        run_report={
            "row_domain": ROW_DOMAIN,
            "identity": {"manifest_sha256": "a" * 64},
            "label_cutoff": {"status": "not_consumed_by_mechanical_layer"},
            "model_availability": {},
            "arms": ARMS,
            "transactions": [],
            "published": {"published_generation": 10},
            "recovery": {},
            "fault_injections": [],
        },
        aa=AA,
        h6_items=readings["h6_items"],
        income_reading=readings["income"],
        drawdown_readings=readings["drawdown"],
        thresholds=thresholds,
        effects={arm: effect.to_dict() for arm, effect in readings["effects"].items()},
        artifact_hashes={"rehearsal_manifest.json": "a" * 64},
        machine_time={"virtual_days": 10, "per_day_seconds": 1.0, "batch_seconds": 10.0},
        authorization=None,
        production={"unchanged": True},
        referenced={"note": "引用 T4/T5"},
        t4_reference={"c_k": thresholds["c_k"]},
        t5_reference={"c_w": thresholds["c_w"], "c_i": thresholds["c_i"]},
    )
    for key in (
        "scope",
        "evidence_class",
        "virtual_clock",
        "not_forward_evidence",
        "not_sla_evidence",
        "row_domain",
        "history",
        "label_cutoff",
        "model_availability",
        "identity",
        "arms",
        "mechanical_aa",
        "transactions",
        "published",
        "recovery",
        "fault_injections",
        "effects",
        "income",
        "drawdown",
        "h6",
        "referenced_readings",
        "artifact_hashes",
        "machine_time",
        "long_task_authorization",
        "production_zero_write",
        "limitations",
    ):
        assert key in report, key
    assert report["h6"]["claim"] == "evaluated_not_declared_pass"
    assert report["h6"]["mutual_offset_forbidden"] is True
    assert len(report["h6"]["items"]) == 3
    markdown = render_markdown(report)
    assert "H6 三条单条数值" in markdown
    assert "闸门判定归用户" in markdown


def t6env_not_executed():
    return [
        {"name": name, "status": "not_executed", "judgment": "未执行", "target_pp": None}
        for name in ("income_minus_5d", "dd_improve_12pp", "dd_worsen_12pp")
    ]


def test_referenced_readings_drop_gate_fields():
    """引用 T4/T5 读数时不得带入闸门语义键（否则 `aa_gate_passed` 会被误读）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal.assembly import (
        _numeric_only,
        forbidden_gate_fields,
    )

    payload = {
        "aa_gate_passed": True,
        "mde_pp": 8.0,
        "registered": True,
        "gate_scope": "方向级",
        "nested": {"x_gate_passed": True},
    }
    kept = _numeric_only(payload)
    assert kept == {"mde_pp": 8.0, "registered": True}
    assert forbidden_gate_fields(kept) == []
    assert forbidden_gate_fields({"a": {"b_gate_passed": True}}) == ["a.b_gate_passed"]


def test_report_h6_items_have_target_and_evidence_paths(tmp_path):
    document = _document(tmp_path)
    items = document["h6_numeric"]["items"]
    assert len(items) == 3
    assert all(item["evidence_fields"] for item in items)
    assert all(item["judgment"] and item["forbidden"] for item in items)
    assert document["h6_numeric"]["frequency_source"].startswith("T4/T5")
