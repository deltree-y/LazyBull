# -*- coding: utf-8 -*-
"""P5b T6 终审 P2 回归（tests/ 豁免复杂度）。

覆盖：授权必须非空且严格匹配 campaign、手册「正式件声明」精确比对、墙钟顶层摘要语义、
限制声明 10 条、占位产物下线、报告逐字节确定性。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t6_env as t6env  # noqa: E402

from src.lazybull.v2.hosts.testbed.rehearsal import artifacts  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal.assembly import (  # noqa: E402
    verify_runbook,
)
from src.lazybull.v2.hosts.testbed.rehearsal.runner import (  # noqa: E402
    RehearsalError,
    append_machine_time,
    load_authorization,
)

DAYS = [f"202607{d:02d}" for d in range(6, 16)]


def _authorization(tmp_path: Path, **scope) -> Path:
    payload = {
        "authorized_at": "2026-11-03T08:00:00",
        "authorized_by": "user",
        "quote": "同意",
        "basis": "单日实测 × 日数",
        "scope": {"campaign_id": "c1", "max_virtual_days": 10, **scope},
    }
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_authorization_requires_non_empty_campaign_and_positive_days(tmp_path):
    with pytest.raises(RehearsalError):
        load_authorization(str(_authorization(tmp_path, campaign_id="")))
    with pytest.raises(RehearsalError):
        load_authorization(str(_authorization(tmp_path, max_virtual_days=0)))
    loaded = load_authorization(str(_authorization(tmp_path)))
    assert loaded["scope"]["campaign_id"] == "c1"


def test_runbook_formal_declaration_must_match_campaign(tmp_path):
    repo = t6env.build_repo(tmp_path)
    hashes = {artifacts.ARTIFACT_REPORT: "a" * 64}
    text = t6env.runbook_text(hashes).replace(
        "正式演练根 `c1`", "正式演练根 `p5b_t6_rehearsal_20260101`"
    )
    (repo / "docs" / "guide" / "p5b_testbed_runbook.md").write_text(text, encoding="utf-8")
    result = verify_runbook(repo, artifact_hashes=hashes, campaign_id="c1", lineage_sequence=None)
    assert result["verified"] is False
    assert any("formal_campaign_declared" in item for item in result["mismatches"])


def test_runbook_matching_declaration_passes(tmp_path):
    repo = t6env.build_repo(tmp_path)
    hashes = {artifacts.ARTIFACT_REPORT: "a" * 64}
    (repo / "docs" / "guide" / "p5b_testbed_runbook.md").write_text(
        t6env.runbook_text(hashes), encoding="utf-8"
    )
    result = verify_runbook(repo, artifact_hashes=hashes, campaign_id="c1", lineage_sequence=1)
    assert result["verified"] is True, result["mismatches"]


def test_machine_time_top_level_summary_is_explicit(tmp_path):
    path = tmp_path / "machine_time_run.json"
    append_machine_time(
        path,
        {"stage": "run", "days": ["20260706"], "per_day_seconds": {"20260706": 21.0}},
    )
    append_machine_time(
        path,
        {
            "stage": "run",
            "days": ["20260707", "20260708"],
            "per_day_seconds": {"20260707": 21.0, "20260708": 21.0},
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["virtual_days_total"] == 3
    assert [batch["days"] for batch in payload["batches"]] == [1, 2]
    assert "最后一次" in payload["top_level_semantics"]


def test_limitations_have_ten_entries(tmp_path):
    env = SimpleNamespace(data_root=tmp_path / "data", base_root=tmp_path / "testbed", days=DAYS)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    document = t6env.manifest_document(env, t6env.build_repo(tmp_path))
    assert len(document["limitations"]) == 10
    assert any("INJECTED_ARM_VALUATION_DECOMPOSITION" in item for item in document["limitations"])


def test_placeholder_freeze_wallclock_artifact_is_gone():
    assert artifacts.ARTIFACT_MACHINE_TIME_FREEZE not in artifacts.ARTIFACT_NAMES
    with pytest.raises(Exception):
        artifacts.artifact_path(Path("x"), artifacts.ARTIFACT_MACHINE_TIME_FREEZE)


def test_report_bytes_are_deterministic(tmp_path):
    from src.lazybull.v2.hosts.testbed.rehearsal.report import build_report

    env = SimpleNamespace(data_root=tmp_path / "data", base_root=tmp_path / "testbed", days=DAYS)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    document = t6env.manifest_document(env, t6env.build_repo(tmp_path))
    kwargs = t6env.report_kwargs(document)
    first = artifacts.json_text(build_report(**kwargs))
    second = artifacts.json_text(build_report(**kwargs))
    assert first == second
