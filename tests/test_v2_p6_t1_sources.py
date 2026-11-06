# -*- coding: utf-8 -*-
"""P6 T1 链式净值有限性门与 P5b 原生 bundle 兼容性（P6-T1-07/09/13）。

**正向依据必须是上游原生管线产出的合成 bundle**（R2-20261106-01）：
``tests.p5b_t4_env.full_pipeline`` 与 ``tests.p5b_t5_env.full_t5`` 真实冻结/训练/holdout 后，
``verify_p5b_bundle`` 必须判 ``available``；自造 JSON 只能作负例。
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed.rehearsal import artifacts as t6_artifacts  # noqa: E402
from src.lazybull.v2.hosts.testbed.referee import artifacts as t4_artifacts  # noqa: E402
from src.lazybull.v2.hosts.testbed.referee.drawdown import artifacts as t5_artifacts  # noqa: E402
from src.lazybull.v2.hosts.testbed.storage import seal_record  # noqa: E402
from src.lazybull.v2.reporting import bundle as bundle_module  # noqa: E402
from src.lazybull.v2.reporting.bundle import BUNDLE_FAMILIES, verify_p5b_bundle  # noqa: E402
from src.lazybull.v2.reporting.sources import verify_chain_rows  # noqa: E402


def _chain(splits: list[int], navs: list[float], days: list[str]) -> pd.DataFrame:
    """构造链式净值帧（``date/nav/split_index``）。"""
    return pd.DataFrame({"date": days, "nav": navs, "split_index": splits})


LEGAL_CHAIN = _chain(
    [1, 1, 1, 2, 2, 2],
    [1.0, 1.02, 1.05, 1.05, 1.08, 1.10],
    ["20260105", "20260106", "20260107", "20260108", "20260109", "20260112"],
)


def test_p6_t1_13_chain_nan_row_rejects_whole_arm() -> None:
    """注入 1 行 NaN ⇒ nonfinite_chain_nav，且不产出任何指标（禁止少一行后照算）。"""
    legal = verify_chain_rows(LEGAL_CHAIN, expected_splits=2)
    assert legal["report_status"] == "available"
    assert legal["rows_total"] == 6
    assert legal["rows_used"] == 6
    assert legal["trading_days"] == 6 - 2

    dirty = LEGAL_CHAIN.copy()
    dirty.loc[2, "nav"] = float("nan")
    row = verify_chain_rows(dirty, expected_splits=2)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "nonfinite_chain_nav"
    assert "metrics" not in row
    assert "第 2 行" in row["detail"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda frame: frame.assign(nav=frame["nav"].astype(str).where(frame.index != 1, "abc")),
         "nonfinite_chain_nav"),
        (lambda frame: frame.assign(date=frame["date"].where(frame.index != 3, "not-a-date")),
         "chain_rows_unparsable"),
        (lambda frame: frame.assign(nav=frame["nav"].where(frame.index != 4, 0.0)),
         "nonfinite_chain_nav"),
        (lambda frame: frame.assign(nav=frame["nav"].where(frame.index != 5, float("inf"))),
         "nonfinite_chain_nav"),
    ],
)
def test_p6_t1_13b_chain_bad_rows_fail_closed(mutate: Any, reason: str) -> None:
    """不可解析/非正/Inf 行都按非有限值拒绝（函数不得静默过滤）。"""
    row = verify_chain_rows(mutate(LEGAL_CHAIN.copy()))
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == reason


def test_p6_t1_13c_chain_missing_source_and_empty() -> None:
    """缺来源 ⇒ not_available；空表 ⇒ not_available/chain_rows_empty。"""
    missing = verify_chain_rows(None)
    assert missing["degradation_reason_code"] == "chain_source_missing"
    empty = verify_chain_rows(pd.DataFrame(columns=["date", "nav"]))
    assert empty["degradation_reason_code"] == "chain_rows_empty"
    no_nav = verify_chain_rows(pd.DataFrame(columns=["date"]))
    assert no_nav["degradation_reason_code"] == "missing_precondition"


# ---------------------------------------------------------------------------
# 原生 bundle 正向契约（上游管线真实产出，禁止自造正向夹具）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def native_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """一次构建原生 T4/T5 合成 campaign（module 级，避免重复训练）。"""
    base = tmp_path_factory.mktemp("p6_t1_native")
    from tests import p5b_t4_env, p5b_t5_env

    p5b_t4_env.full_pipeline(base / "t4", campaign_id="p6_native_t4")
    p5b_t5_env.full_t5(base / "t5", campaign_id="p6_native_t5")
    return base


def _t4_dir(base: Path) -> Path:
    from tests import p5b_t4_env  # noqa: F401  兼容导入以便重建
    from src.lazybull.v2.hosts.testbed.referee import artifacts

    return artifacts.reports_dir(base / "t4" / "campaigns" / "p6_native_t4")


def _t5_dir(base: Path) -> Path:
    from src.lazybull.v2.hosts.testbed.referee.drawdown import artifacts

    return artifacts.reports_dir(
        base / "t5" / "fake_repo" / "campaigns" / "p6_native_t5"
    )


def test_p6_t1_07_native_t4_bundle_is_available(native_base: Path) -> None:
    """上游原生 T4 产物 ⇒ ``verify_p5b_bundle`` 必须 available（协议兼容）。"""
    row = verify_p5b_bundle("t4_seq10", _t4_dir(native_base), scope_base=native_base)
    assert row["report_status"] == "available", row.get("failures")
    assert row["failures"] == []
    assert row["protocol"] == "native_p5b_artifacts/v1"
    assert row["campaign_identity"].count("@") == 1


def test_p6_t1_07_native_t5_bundle_is_available(native_base: Path) -> None:
    """上游原生 T5 产物 ⇒ available；``manifest.files`` 上游引用只登记不误判缺失。"""
    row = verify_p5b_bundle("t5_seq7", _t5_dir(native_base), scope_base=native_base)
    assert row["report_status"] == "available", row.get("failures")
    assert any(item["key"] == "parent_manifest" for item in row["external_refs"])


def _reseal(path: Path, mutate: Any) -> None:
    """改值后**重新封存**（模拟“会重封存的真实对手”）。"""
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(
        json.dumps(seal_record(document), ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def test_p6_t1_07b_resealed_verdict_tamper_is_invalid(native_base: Path) -> None:
    """改判定表并重封存 ⇒ 报告 ``identity.verdict_sha256`` 断链（原生双向绑定）。"""
    directory = _t4_dir(native_base)
    target = directory / "referee_verdict.json"
    original = target.read_bytes()
    try:
        _reseal(target, lambda doc: doc.__setitem__("limitations", ["tampered"]))
        row = verify_p5b_bundle("t4_seq10", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "bundle_chain_broken"
    assert any("verdict_sha256" in item for item in row["failures"])


def test_p6_t1_07c_resealed_thresholds_tamper_is_invalid(native_base: Path) -> None:
    """改临界值产物并重封存 ⇒ 报告/判定表 ``thresholds_sha256`` 断链。"""
    directory = _t5_dir(native_base)
    target = directory / "drawdown_thresholds.json"
    original = target.read_bytes()
    try:
        _reseal(target, lambda doc: doc.__setitem__("alpha_per_look", 0.5))
        row = verify_p5b_bundle("t5_seq7", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "bundle_chain_broken"
    assert any("thresholds_sha256" in item for item in row["failures"])


def test_p6_t1_07d_missing_document_is_invalid(native_base: Path) -> None:
    """缺主报告 ⇒ invalid/bundle_field_missing（不进入部分可信）。"""
    directory = _t4_dir(native_base)
    target = directory / "income_referee_report.json"
    original = target.read_bytes()
    try:
        target.unlink()
        row = verify_p5b_bundle("t4_seq10", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "bundle_field_missing"
    assert any("文件缺失" in item for item in row["failures"])


def test_p6_t1_07e_scope_escape_and_missing_dir(native_base: Path) -> None:
    """bundle 目录在 scope 之外 ⇒ bundle_path_escape；目录不存在 ⇒ not_available。"""
    escaped = verify_p5b_bundle("t4_seq10", _t4_dir(native_base), scope_base=native_base / "t5")
    assert escaped["report_status"] == "invalid"
    assert escaped["degradation_reason_code"] == "bundle_path_escape"
    missing = verify_p5b_bundle("t4_seq10", native_base / "nope", scope_base=native_base)
    assert missing["report_status"] == "not_available"
    assert missing["degradation_reason_code"] == "no_source"


def test_p6_t1_07f_spec_matches_native_artifact_names() -> None:
    """三族��明文件名必须与上游 ``ARTIFACT_*`` 常量逐一致（含 T6 结构对照）。"""
    expected = {
        "t4_seq10": (
            t4_artifacts.ARTIFACT_MANIFEST,
            t4_artifacts.ARTIFACT_REPORT,
            t4_artifacts.ARTIFACT_VERDICT,
            t4_artifacts.ARTIFACT_THRESHOLDS,
            t4_artifacts.ARTIFACT_HOLDOUT,
            t4_artifacts.ARTIFACT_POWER,
        ),
        "t5_seq7": (
            t5_artifacts.ARTIFACT_MANIFEST,
            t5_artifacts.ARTIFACT_REPORT,
            t5_artifacts.ARTIFACT_VERDICT,
            t5_artifacts.ARTIFACT_THRESHOLDS,
            t5_artifacts.ARTIFACT_HOLDOUT,
            t5_artifacts.ARTIFACT_POWER,
        ),
        "t6_seq21": (
            t6_artifacts.ARTIFACT_MANIFEST,
            t6_artifacts.ARTIFACT_REPORT,
            t6_artifacts.ARTIFACT_VERDICT,
        ),
    }
    for family, names in expected.items():
        spec = BUNDLE_FAMILIES[family]
        declared = [spec["manifest"], spec["report"], spec["verdict"], *spec["aux"].values()]
        assert set(declared) == set(names), family
        assert spec["loader"].__module__.startswith(
            "src.lazybull.v2.hosts.testbed"
        ), family


def test_p6_t1_07g_empty_chain_is_rejected(native_base: Path) -> None:
    """R2-20261106-01（复审）：链为空不得放行（空链 = 未验证）。"""
    directory = _t4_dir(native_base)
    target = directory / "income_referee_report.json"
    original = target.read_bytes()
    try:
        _reseal(target, lambda doc: doc.__setitem__("hash_chain", {}))
        row = verify_p5b_bundle("t4_seq10", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "bundle_chain_broken"
    assert any("链为空" in item for item in row["failures"])


def test_p6_t1_07h_ghost_chain_key_is_rejected(native_base: Path) -> None:
    """R2-20261106-01（复审）：幽灵键（有合法 SHA 但无载体）必须失败。"""
    directory = _t4_dir(native_base)
    target = directory / "income_referee_report.json"
    original = target.read_bytes()
    try:
        _reseal(
            target,
            lambda doc: doc["hash_chain"].__setitem__("ghost_sha256", "f" * 64),
        )
        row = verify_p5b_bundle("t4_seq10", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert any("无载体" in item for item in row["failures"])


def test_p6_t1_07i_missing_requirement_is_degraded_not_raised(native_base: Path) -> None:
    """缺 manifest 必须按区降级为 invalid（不得抛裸异常）。"""
    directory = _t4_dir(native_base)
    target = directory / "calibration_manifest.json"
    original = target.read_bytes()
    try:
        target.unlink()
        row = verify_p5b_bundle("t4_seq10", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert any("文件缺失" in item for item in row["failures"])


def test_p6_t1_07j_unsupported_schema_is_rejected(native_base: Path) -> None:
    """schema 与原生常量不符必须拒绝（同一代码路径覆盖三族）。"""
    directory = _t5_dir(native_base)
    target = directory / "drawdown_manifest.json"
    original = target.read_bytes()
    try:
        _reseal(target, lambda doc: doc.__setitem__("schema", "unsupported_schema/v999"))
        row = verify_p5b_bundle("t5_seq7", directory, scope_base=native_base)
    finally:
        target.write_bytes(original)
    assert row["report_status"] == "invalid"
    assert any("schema" in item for item in row["failures"])


def _synthetic_t6(directory: Path, *, campaign: str = "p6_t6_synthetic") -> Path:
    """由原生构造与封存函数生成 T6 合成包（不执行虚拟日或读取真实 campaign）。"""
    from types import SimpleNamespace

    from src.lazybull.v2.hosts.testbed.rehearsal import report as t6_report
    from src.lazybull.v2.hosts.testbed.rehearsal import verdict as t6_verdict
    from tests import p5b_t6_env

    env = SimpleNamespace(
        data_root=directory.parent / "data",
        base_root=directory.parent / "testbed",
        days=(
            "20260706", "20260707", "20260708", "20260709", "20260710",
            "20260713", "20260714", "20260715", "20260716", "20260717",
        ),
    )
    repo = p5b_t6_env.build_repo(directory.parent / "upstream")
    manifest = p5b_t6_env.manifest_document(env, repo, campaign_id=campaign)
    frozen = t6_artifacts.write_json_once(
        directory / t6_artifacts.ARTIFACT_MANIFEST, manifest, what="合成 T6 manifest"
    )
    manifest_sha = frozen["sha256"]
    kwargs = p5b_t6_env.report_kwargs(manifest)
    kwargs["manifest_sha256"] = manifest_sha
    kwargs["calibrator"] = manifest["files"]["calibrator"]
    kwargs["run_report"]["identity"]["manifest_sha256"] = manifest_sha
    kwargs["artifact_hashes"] = {t6_artifacts.ARTIFACT_MANIFEST: manifest_sha}
    report = t6_report.build_report(**kwargs)
    written = t6_artifacts.write_json_once(
        directory / t6_artifacts.ARTIFACT_REPORT, report, what="合成 T6 报告"
    )
    markdown = t6_artifacts.write_text_once(
        directory / t6_artifacts.ARTIFACT_REPORT_MD,
        t6_report.render_markdown(report),
        what="合成 T6 Markdown",
    )
    checks = {"conditions": [], "author_side_status": "not_executed"}
    verdict = t6_verdict.build_verdict(
        run_id=manifest["run_id"],
        campaign_id=campaign,
        manifest_sha256=manifest_sha,
        calibrator_sha256=manifest["files"]["calibrator"]["sha256"],
        artifact_hashes={
            t6_artifacts.ARTIFACT_REPORT: written["sha256"],
            t6_artifacts.ARTIFACT_REPORT_MD: markdown["sha256"],
        },
        h1=checks,
        h6=checks,
        h7=checks,
        effects={},
    )
    t6_artifacts.write_json_once(
        directory / t6_artifacts.ARTIFACT_VERDICT, verdict, what="合成 T6 判定表"
    )
    return directory


def test_p6_t1_07k_t6_schema_and_linked_markdown(tmp_path: Path) -> None:
    """T6：原生 schema 常量生效；被 ``hash_chain`` 引用的 md 缺失即失败。"""
    from src.lazybull.v2.hosts.testbed.rehearsal import report as t6_report
    from src.lazybull.v2.hosts.testbed.rehearsal import verdict as t6_verdict

    assert BUNDLE_FAMILIES["t6_seq21"]["schemas"][t6_artifacts.ARTIFACT_REPORT] == (
        t6_report.REPORT_SCHEMA
    )
    assert BUNDLE_FAMILIES["t6_seq21"]["schemas"][t6_artifacts.ARTIFACT_VERDICT] == (
        t6_verdict.VERDICT_SCHEMA
    )
    directory = _synthetic_t6(tmp_path / "t6")
    ok = verify_p5b_bundle("t6_seq21", directory, scope_base=tmp_path)
    assert ok["report_status"] == "available", ok.get("failures")

    md = directory / t6_artifacts.ARTIFACT_REPORT_MD
    md.unlink()
    broken = verify_p5b_bundle("t6_seq21", directory, scope_base=tmp_path)
    assert broken["report_status"] == "invalid"
    assert any("引用的文件缺失" in item for item in broken["failures"])


@pytest.mark.parametrize(
    "artifact_name", (t6_artifacts.ARTIFACT_REPORT, t6_artifacts.ARTIFACT_VERDICT)
)
def test_p6_t1_07l_t6_unsupported_schema_is_rejected(
    tmp_path: Path, artifact_name: str
) -> None:
    """T6 主报告与判定表分别使用不支持的 schema 时，原生读口必须拒绝。"""
    directory = _synthetic_t6(tmp_path / "t6")
    _reseal(directory / artifact_name, lambda doc: doc.__setitem__("schema", "unsupported/v999"))
    row = verify_p5b_bundle("t6_seq21", directory, scope_base=tmp_path)
    assert row["report_status"] == "invalid"
    assert any("schema 非法" in item for item in row["failures"])


def test_p6_t1_09_no_hardcoded_campaign_name_or_target() -> None:
    """campaign 身份来自 manifest 或目录名；不得硬编码真实 campaign/target_pp。"""
    source = inspect.getsource(bundle_module)
    assert "p5b_t6_rehearsal_20261104_r21" not in source
    assert "28.088587217293803" not in source
    assert "30.876549316416398" not in source


def test_p6_t1_bundle_unknown_family_raises(tmp_path: Path) -> None:
    """未知族直接报错（禁止静默返回可信）。"""
    with pytest.raises(ValueError):
        verify_p5b_bundle("t9_unknown", tmp_path, scope_base=tmp_path)
