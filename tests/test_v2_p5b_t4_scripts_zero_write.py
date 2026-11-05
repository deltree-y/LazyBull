# -*- coding: utf-8 -*-
"""P5b T4 验收工具入口、零写入、规模与确定性（设计 v4 §7.1/§7.2/§9）。

覆盖：三个薄入口（冻结/训练、holdout、导出判定表）的默认 dry-run 与写域守卫、
产物只落 campaign ``reports/``、重复运行拒绝覆盖、生产根零写入、
同 seed 重跑逐字节一致（墙钟单独落盘）、实现数下限与长任务阈值登记、
源码级守卫（不写 LedgerEntry、不联网）。
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t4_env as env  # noqa: E402

from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    ARTIFACT_HOLDOUT,
    ARTIFACT_MANIFEST,
    ARTIFACT_MACHINE_TIME,
    ARTIFACT_NAMES,
    ARTIFACT_POWER,
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
    ARTIFACT_THRESHOLDS,
    ARTIFACT_VERDICT,
    SCALE_FLOOR,
    artifact_path,
    reports_dir,
)

SUBPROCESS_TIMEOUT = 900


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    import os

    # 子进程按 UTF-8 输出：否则非 UTF-8 控制台（如 Windows GBK）的中文断言文本会被
    # 父侧 `encoding="utf-8"` 替换成 U+FFFD，导致「已存在/生产根/只允许」类断言假失败。
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "v2_p5b" / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SUBPROCESS_TIMEOUT,
        env=environment,
    )


def _freeze_args(prepared: env.SimpleNamespace, psd_path: Path, psd_sha: str) -> list:
    """冻结参数（仓库冻结件与假仓库同根，绝不写真实 ``docs/plans/``）。"""
    return [
        "--campaign-root",
        str(prepared.root),
        "--freeze",
        "--apply",
        "--repo-root",
        str(prepared.env.repo_root),
        "--repo-manifest",
        str(Path(prepared.env.repo_root) / "docs/plans/v2/t4_frozen.json"),
        "--psd-manifest",
        str(psd_path),
        "--psd-manifest-sha256",
        psd_sha,
        "--allow-external-root",
        "--json",
    ]


def _prepare_floor_campaign(tmp_path: Path) -> tuple:
    """按 §5.3 下限实现数准备 campaign（合成源对；向量化后总墙钟数秒）。"""
    built = env.fast_manifest(tmp_path, counts=env.FLOOR_COUNTS)
    psd_path, psd_sha = env.write_psd_manifest(built.env)
    root = Path(tmp_path) / "campaigns" / "p5b_t4_floor"
    root.mkdir(parents=True, exist_ok=True)
    return built, psd_path, psd_sha, root


def test_freeze_dry_run_writes_nothing(tmp_path):
    built, psd_path, psd_sha, root = _prepare_floor_campaign(tmp_path)
    result = _run(
        "referee_calibration.py",
        "--campaign-root",
        str(root),
        "--freeze",
        "--repo-root",
        str(built.env.repo_root),
        "--psd-manifest",
        str(psd_path),
        "--psd-manifest-sha256",
        psd_sha,
        "--allow-external-root",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["campaign_manifest"]["written"] is False
    assert not reports_dir(root).exists()


def test_train_dry_run_writes_nothing(tmp_path):
    built, psd_path, psd_sha, root = _prepare_floor_campaign(tmp_path)
    frozen = _run(
        "referee_calibration.py",
        *_freeze_args(env.SimpleNamespace(root=root, env=built.env), psd_path, psd_sha),
    )
    assert frozen.returncode == 0, frozen.stderr
    before = env.tree_state(root)
    result = _run(
        "referee_calibration.py",
        "--campaign-root",
        str(root),
        "--train",
        "--repo-root",
        str(built.env.repo_root),
        "--repo-manifest",
        str(Path(built.env.repo_root) / "docs/plans/v2/t4_frozen.json"),
        "--allow-external-root",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert env.tree_state(root) == before
    assert not artifact_path(root, ARTIFACT_THRESHOLDS).exists()


def test_full_pipeline_scripts_end_to_end(tmp_path):
    built, psd_path, psd_sha, root = _prepare_floor_campaign(tmp_path)
    frozen = _run(
        "referee_calibration.py",
        *_freeze_args(env.SimpleNamespace(root=root, env=built.env), psd_path, psd_sha),
    )
    assert frozen.returncode == 0, frozen.stderr
    trained = _run(
        "referee_calibration.py",
        "--campaign-root",
        str(root),
        "--train",
        "--apply",
        "--repo-root",
        str(built.env.repo_root),
        "--repo-manifest",
        str(Path(built.env.repo_root) / "docs/plans/v2/t4_frozen.json"),
        "--allow-external-root",
        "--json",
    )
    assert trained.returncode == 0, trained.stderr
    held = _run(
        "referee_holdout.py",
        "--campaign-root",
        str(root),
        "--apply",
        "--repo-root",
        str(built.env.repo_root),
        "--repo-manifest",
        str(Path(built.env.repo_root) / "docs/plans/v2/t4_frozen.json"),
        "--allow-external-root",
        "--json",
    )
    assert held.returncode in (0, 2), held.stderr
    payload = json.loads(held.stdout)
    assert payload["unit_count"] == 8
    assert payload["long_task_threshold_exceeded"] is False
    for name in (
        ARTIFACT_MANIFEST,
        ARTIFACT_THRESHOLDS,
        ARTIFACT_HOLDOUT,
        ARTIFACT_POWER,
        ARTIFACT_VERDICT,
        ARTIFACT_REPORT,
        ARTIFACT_REPORT_MD,
        ARTIFACT_MACHINE_TIME,
    ):
        assert artifact_path(root, name).is_file(), name
    machine = json.loads(artifact_path(root, ARTIFACT_MACHINE_TIME).read_text(encoding="utf-8"))
    assert machine["long_task_threshold_seconds"] == 1200.0
    assert machine["phase"] == "t4_income_referee_holdout"


def test_rerun_refuses_to_overwrite(tmp_path):
    built, psd_path, psd_sha, root = _prepare_floor_campaign(tmp_path)
    args = _freeze_args(env.SimpleNamespace(root=root, env=built.env), psd_path, psd_sha)
    assert _run("referee_calibration.py", *args).returncode == 0
    second = _run("referee_calibration.py", *args)
    assert second.returncode != 0
    assert "已存在" in (second.stdout + second.stderr)


def test_scripts_refuse_production_root():
    for script, extra in (
        ("referee_calibration.py", ["--freeze"]),
        ("referee_holdout.py", []),
    ):
        result = _run(script, "--campaign-root", str(ROOT / "data" / "paper"), *extra, "--json")
        assert result.returncode != 0, script
        assert "生产根" in (result.stdout + result.stderr), script


def test_export_table_refuses_production_output(tmp_path):
    pipeline = env.full_pipeline(tmp_path)
    result = _run(
        "referee_export_table.py",
        "--campaign-root",
        str(pipeline.prepared.root),
        "--out",
        str(ROOT / "data" / "paper" / "t4_verdict.md"),
    )
    assert result.returncode != 0
    assert "生产根" in (result.stdout + result.stderr)


def test_export_table_enforces_whitelist_and_refuses_overwrite(tmp_path):
    """导出是旁路产物：默认只允许白名单目录，且独占创建（不得静默覆盖）。"""
    pipeline = env.full_pipeline(tmp_path)
    outside = _run(
        "referee_export_table.py",
        "--campaign-root",
        str(pipeline.prepared.root),
        "--out",
        str(Path(tmp_path) / "verdict.md"),
    )
    assert outside.returncode != 0
    assert "只允许" in (outside.stdout + outside.stderr)
    allowed = ROOT / "temp" / "t4_export_probe.md"
    if allowed.exists():
        allowed.unlink()
    try:
        first = _run(
            "referee_export_table.py",
            "--campaign-root",
            str(pipeline.prepared.root),
            "--out",
            str(allowed),
        )
        assert first.returncode == 0, first.stderr
        assert allowed.is_file()
        second = _run(
            "referee_export_table.py",
            "--campaign-root",
            str(pipeline.prepared.root),
            "--out",
            str(allowed),
        )
        assert second.returncode != 0
        assert "禁止覆盖" in (second.stdout + second.stderr)
    finally:
        if allowed.exists():
            allowed.unlink()


def test_testbed_root_and_frozen_docs_untouched(tmp_path):
    """试车场根与 docs/plans 冻结件不得被测试链路写入。"""
    watched = [
        ROOT / "data" / "paper_shadow" / "p5b_testbed",
        ROOT / "docs" / "plans" / "v2" / "p5b_t4_calibration_manifest_20261102.json",
    ]

    def digest(path: Path):
        if path.is_file():
            return (path.stat().st_size, env._sha256(path))
        if path.is_dir():
            # 目录用元数据快照（size+mtime_ns）：任何写入必改其一，"被写入"检测语义
            # 与内容哈希等价（production_snapshot 对 data/models 的既有先例），且不受
            # 试车场根体积增长影响——曾因 77 万文件/25.5 GB 演练产物堆积，单测试
            # 全量哈希 51 GB 令全量回归在 ~80% 处假性卡死（2026-11-05 根因裁决）。
            return env.metadata_state(path)
        return None

    before = {path.as_posix(): digest(path) for path in watched}
    env.full_pipeline(tmp_path)
    after = {path.as_posix(): digest(path) for path in watched}
    assert after == before


def test_export_table_is_read_only_and_lists_units(tmp_path):
    pipeline = env.full_pipeline(tmp_path)
    root = pipeline.prepared.root
    before = env.tree_state(root)
    result = _run("referee_export_table.py", "--campaign-root", str(root))
    assert result.returncode == 0, result.stderr
    assert "holdout 逐单元" in result.stdout
    assert env.tree_state(root) == before
    summary = _run("referee_export_table.py", "--campaign-root", str(root), "--json")
    payload = json.loads(summary.stdout)
    assert payload["holdout"]["unit_count"] == 8
    assert "c_by_look" in payload["thresholds"]
    assert payload["h4_boundary"]["t4_claim"] == "evaluated_not_declared_pass"


def test_production_roots_untouched_by_pipeline(tmp_path):
    before = env.production_snapshot()
    env.full_pipeline(tmp_path)
    assert env.production_unchanged(before, env.production_snapshot())


def test_artifacts_stay_inside_campaign_reports(tmp_path):
    pipeline = env.full_pipeline(tmp_path)
    root = pipeline.prepared.root
    reports = reports_dir(root)
    written = {path.name for path in reports.iterdir() if path.is_file()}
    # 库层链路不写墙钟；machine_time.json 只由 holdout 脚本入口写（见端到端用例）
    assert written == set(ARTIFACT_NAMES) - {ARTIFACT_MACHINE_TIME}
    assert set(ARTIFACT_NAMES) >= {ARTIFACT_MANIFEST, ARTIFACT_THRESHOLDS, ARTIFACT_REPORT}
    outside = [path for path in root.rglob("*") if path.is_file() and reports not in path.parents]
    assert outside == []


def test_same_seed_rerun_is_byte_identical(tmp_path):
    first = env.full_pipeline(Path(tmp_path) / "a", campaign_id="p5b_t4_same")
    second = env.full_pipeline(Path(tmp_path) / "b", campaign_id="p5b_t4_same")
    for name in ARTIFACT_NAMES:
        if name == ARTIFACT_MACHINE_TIME:
            continue
        left = artifact_path(first.prepared.root, name).read_bytes()
        right = artifact_path(second.prepared.root, name).read_bytes()
        assert left == right, f"{name} 重跑不一致"


def test_counts_floor_and_manifest_recorded(tmp_path):
    pipeline = env.full_pipeline(tmp_path, counts=env.FLOOR_COUNTS)
    doc = json.loads(
        artifact_path(pipeline.prepared.root, ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    )
    report = json.loads(
        artifact_path(pipeline.prepared.root, ARTIFACT_REPORT).read_text(encoding="utf-8")
    )
    assert doc["counts"]["train_per_unit"] >= SCALE_FLOOR["train_per_unit"]
    assert doc["counts"]["holdout_per_unit"] >= SCALE_FLOOR["holdout_per_unit"]
    assert doc["counts"]["effect_per_scene_per_level"] >= SCALE_FLOOR["effect_per_scene_per_level"]
    assert report["counts"]["holdout_per_unit"] >= SCALE_FLOOR["holdout_per_unit"]
    assert report["holdout"]["unit_count"] == 8
    assert report["machine_time"]["excluded_from_determinism"] is True


def test_report_carries_required_scope_fields(tmp_path):
    pipeline = env.full_pipeline(tmp_path)
    report = json.loads(
        artifact_path(pipeline.prepared.root, ARTIFACT_REPORT).read_text(encoding="utf-8")
    )
    assert report["scope"] == "historical_testbed"
    assert report["evidence_class"] == "calibration_only"
    assert report["virtual_clock"] is True
    assert report["not_forward_evidence"] is True
    assert report["not_sla_evidence"] is True
    assert report["row_domain"] == "structural"
    assert report["schedule_mode"] == "replay"
    assert report["drawdown_referee"]["status"] == "not_implemented_in_t4"
    assert report["h6_numeric"]["status"] == "placeholder_not_run"


def test_verdict_carries_grid_upper_bound_flags(tmp_path):
    """判定表也带 MDE 上界标记，避免下游只读 verdict 时把上界档当精确 MDE。"""
    pipeline = env.full_pipeline(tmp_path)
    verdict = json.loads(
        artifact_path(pipeline.prepared.root, ARTIFACT_VERDICT).read_text(encoding="utf-8")
    )
    referee = verdict["income_referee"]
    assert "mde_at_grid_upper_bound" in referee
    assert set(referee["mde_at_grid_upper_bound"]) == set(referee["mde_by_scene"])
    for scene, flagged in referee["mde_at_grid_upper_bound"].items():
        if referee["mde_by_scene"][scene] is None:
            assert flagged is False, scene


def test_modules_have_no_ledger_or_network_entrypoints():
    """源码级守卫：裁判子包全模块不写 LedgerEntry、不联网、不触碰补齐入口。"""
    import src.lazybull.v2.hosts.testbed.referee as package
    from src.lazybull.v2.hosts.testbed.referee import (
        artifacts,
        calibration,
        holdout,
        ledger_source,
        manifest,
        power,
        report,
        sampling,
        scenes,
        statistics,
        training,
    )

    forbidden = (
        "tushare",
        "akshare",
        "efinance",
        "requests",
        "urllib",
        "features.ensure",
        "append_ledger_entry",
        "LedgerEntry(",
        "pending_buys",
        "pending_sells",
    )
    from src.lazybull.v2.hosts.testbed.referee import manifest_sections

    modules = (
        package,
        artifacts,
        calibration,
        holdout,
        ledger_source,
        manifest,
        manifest_sections,
        power,
        report,
        sampling,
        scenes,
        statistics,
        training,
    )
    for module in modules:
        source = inspect.getsource(module)
        for token in forbidden:
            assert token not in source, f"{module.__name__} 命中禁用依赖: {token}"
