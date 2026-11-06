# -*- coding: utf-8 -*-
"""P6 T1 manifest 自封存与源/projection 复算（P6-T1 自 seal/复算 1 项）。

覆盖：``record_sha256`` 自封存先行；篡改 degradations/字段即失败；projection/outputs/inputs
逐环复算定位；同输入且同 ``generated_at`` 的记录哈希逐字节一致；输入身份取自稳定快照回执。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting.manifest import (  # noqa: E402
    MANIFEST_SCHEMA,
    build_manifest,
    build_report_id,
    recompute_chain,
    seal_manifest,
    verify_manifest,
)
from tests.p6_t1_env import build_paper_root, build_projection  # noqa: E402


def _manifest(projection: dict, inputs: list[dict], outputs: dict, degradations: list) -> dict:
    return build_manifest(
        report_kind="dashboard",
        evidence_class="production",
        input_evidence_classes=["production"],
        generated_at="20261105T000000Z",
        data_window={"start": "20260506", "end": "20260603"},
        source_watermarks={"paper": "20260603"},
        root_map={"paper": "v2 纸面生产根（只读副本）"},
        inputs=inputs,
        projection=projection,
        generator={"project": "lazybull", "python": "3.12.10"},
        outputs=outputs,
        csv_schemas={},
        degradations=degradations,
    )


def test_p6_t1_self_seal_and_recompute(tmp_path: Path) -> None:
    """自封存 + 逐环复算 + 篡改定位 + 快照回执集成。"""
    built = build_paper_root(tmp_path / "paper")
    projection, snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    receipts = list(snapshot.receipts.values())
    inputs = [receipt.as_dict() for receipt in receipts]
    outputs = {"projection.json": "b" * 64, "report.html": "c" * 64}
    manifest = _manifest(projection, inputs, outputs, projection["degradations"])

    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["input_manifest_sha256"] == manifest["input_manifest_sha256"].lower()
    assert manifest["report_id"] == build_report_id(
        report_kind="dashboard",
        generated_at="20261105T000000Z",
        input_sha256=manifest["input_manifest_sha256"],
    )
    assert [row["path"] for row in manifest["inputs"]] == sorted(
        row["path"] for row in manifest["inputs"]
    )
    assert all(row["path"].startswith("paper:") for row in manifest["inputs"])
    assert verify_manifest(manifest)["report_status"] == "available"

    repeat = _manifest(projection, inputs, outputs, projection["degradations"])
    assert repeat["record_sha256"] == manifest["record_sha256"]
    assert repeat == manifest

    ok = recompute_chain(
        manifest=manifest, projection=projection, outputs=outputs, inputs=inputs
    )
    assert ok["report_status"] == "available"
    assert ok["ok"] is True
    assert len(ok["checks"]) == len(manifest["outputs"]) + 3  # 输入/投影/键集 + 逐产出

    tampered = dict(manifest)
    tampered["degradations"] = [{"field_path": "nav", "reason_code": "authority_mismatch"}]
    row = verify_manifest(tampered)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "manifest_seal_mismatch"

    row = recompute_chain(
        manifest=manifest,
        projection={**projection, "generated_at": "20261106T000000Z"},
        outputs=outputs,
        inputs=inputs,
    )
    assert row["degradation_reason_code"] == "projection_sha_mismatch"

    row = recompute_chain(
        manifest=manifest,
        projection=projection,
        outputs={"projection.json": "d" * 64, "report.html": outputs["report.html"]},
        inputs=inputs,
    )
    assert row["degradation_reason_code"] == "output_sha_mismatch"

    row = recompute_chain(
        manifest=manifest,
        projection=projection,
        outputs={**outputs, "unregistered.json": "f" * 64},
        inputs=inputs,
    )
    assert row["degradation_reason_code"] == "output_sha_mismatch"
    assert "outputs.keys" in row["detail"]

    row = recompute_chain(
        manifest=manifest,
        projection=projection,
        outputs=outputs,
        inputs=[{**inputs[0], "sha256": "e" * 64}] + inputs[1:],
    )
    assert row["degradation_reason_code"] == "input_drift"

    unsupported = verify_manifest({**manifest, "schema_version": "p6_report_manifest/v0"})
    assert unsupported["degradation_reason_code"] == "manifest_schema_unsupported"

    unsealed = {key: value for key, value in manifest.items() if key != "record_sha256"}
    assert verify_manifest(unsealed)["degradation_reason_code"] == "manifest_seal_mismatch"


def test_p6_t1_manifest_rejects_duplicate_inputs_and_bad_hashes(tmp_path: Path) -> None:
    """R1-12：重复输入路径与非法哈希形态一律拒绝（不自造修正）。"""
    projection = {"schema": "x"}
    receipt = {"path": "paper:state/account.json", "sha256": "a" * 64, "size": 1}
    with pytest.raises(ValueError):
        _manifest(projection, [receipt, dict(receipt)], {"o.json": "b" * 64}, [])
    with pytest.raises(ValueError):
        _manifest(projection, [{**receipt, "sha256": "NOT-A-SHA"}], {"o.json": "b" * 64}, [])

    manifest = _manifest(projection, [receipt], {"o.json": "b" * 64}, [])
    bad = dict(manifest)
    bad["outputs"] = {"o.json": "XYZ"}
    bad = seal_manifest(bad)
    row = verify_manifest(bad)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "missing_precondition"
