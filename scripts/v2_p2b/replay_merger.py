# -*- coding: utf-8 -*-
"""P2b 单袖合并器重放薄驱动（关闭/开启两臂共用，不复制 B0 配置）。

每次调用只运行一个显式模式，并独立创建 B0 驱动内的 storage、loader、
persistent signal 与引擎。每臂前后均复核完整输入清单、模型实际路径、
参照摘要、工作树和环境；真实重放仅在 T4/T5 对应授权后执行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p2a import replay_b0 as b0_driver
from scripts.v2_p2a.input_manifest import validate_manifest
from scripts.v2_p2a.replay_b0 import _load_splits_for_snapshot, build_manifest_config_snapshot
from scripts.v2_p2b.check_frozen_inputs import (
    DEFAULT_B0_RAW,
    DEFAULT_MANIFEST,
    DEFAULT_P2A_RUNS,
    canonical_digest_note,
    check_all_frozen_inputs,
)
from scripts.v2_p2b.check_workspace_state import canonical_json, check_workspace_state
from src.lazybull.v2.hosts.backtest.runtime import (
    LEGACY_BYPASS,
    MERGER_SLEEVE_ID,
    validate_merger_mode,
)

P2B_SOURCE = "v2_p2b_merger_replay"


def _repo_path(value: str, repo_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / path


def build_evidence_notes(preflight: Dict[str, object]) -> Dict[str, str]:
    """把预检证据转换为 writer 接受的字符串 notes。"""
    frozen = preflight["frozen"]
    return {
        "workspace.digest": canonical_json(preflight["workspace"]["workspace"]),
        "environment.versions": canonical_json(preflight["workspace"]["environment"]),
        "reference.p2a_runs.digest": canonical_digest_note(frozen["p2a_runs"]),
        "reference.b0_raw.digest": canonical_digest_note(frozen["b0_raw"]),
    }


def evidence_signature(preflight: Dict[str, object]) -> str:
    """提取开跑前后必须完全一致的证据（不含采集时间或运行 ID）。"""
    frozen = preflight["frozen"]
    payload = {
        "workspace": preflight["workspace"]["workspace"],
        "environment": preflight["workspace"]["environment"],
        "selected": preflight["frozen"]["model_report"]["selected"],
        "p2a_runs": frozen["p2a_runs"],
        "b0_raw": frozen["b0_raw"],
    }
    return canonical_json(payload)


def run_preflight(
    *,
    data_root: Path,
    manifest_path: Path,
    repo_root: Path,
    p2a_runs_dir: Path,
    b0_raw_dir: Path,
) -> Dict[str, object]:
    """执行一臂的单次完整前置/后置核验；任何差异都 fail-fast。"""
    workspace = check_workspace_state(repo_root)
    if not workspace["ok"]:
        raise RuntimeError(f"P2b 工作树门 2 失败: {workspace['violations']}")
    splits = _load_splits_for_snapshot(str(data_root))
    config = build_manifest_config_snapshot(splits)
    manifest_diffs = validate_manifest(data_root, manifest_path, config=config)
    if manifest_diffs:
        raise RuntimeError(f"P2b 门 1 完整输入清单失败: {manifest_diffs}")
    frozen = check_all_frozen_inputs(
        data_root=data_root,
        manifest_path=manifest_path,
        p2a_runs_dir=p2a_runs_dir,
        b0_raw_dir=b0_raw_dir,
    )
    if not frozen["ok"]:
        raise RuntimeError(f"P2b 冻结模型/参照核验失败: {frozen['model_report']['errors']}")
    return {"workspace": workspace, "frozen": frozen}


def _baseline_for_mode(mode: str, baseline_ref: Optional[str], p2a_runs_dir: Path) -> str:
    if mode == LEGACY_BYPASS:
        expected = p2a_runs_dir.name
        if baseline_ref and baseline_ref != expected:
            raise ValueError(f"legacy_bypass 的 baseline_ref 必须指向 P2a 冻结批 {expected}")
        return expected
    if not baseline_ref:
        raise ValueError("single_sleeve_passthrough 必须以 --baseline-ref 指向当前关闭批")
    return baseline_ref


def _execution_repo_root(repo_root: Optional[str]) -> Path:
    """取证根必须等于实际执行驱动所在根，禁止核验 B 却执行 A。"""
    requested = Path(repo_root).resolve() if repo_root else PROJECT_ROOT.resolve()
    execution = Path(b0_driver.__file__).resolve().parents[2]
    if requested != execution:
        raise ValueError(
            f"repo_root 与实际执行代码根不一致: 取证={requested}，执行={execution}；"
            "禁止用另一份检出的摘要为本驱动背书"
        )
    return execution


def replay_merger_arm(
    *,
    merger_mode: str,
    splits_subset: Optional[Sequence[int]] = None,
    data_root: Optional[str] = None,
    runs_root: Optional[str] = None,
    batch_id: Optional[str] = None,
    baseline_ref: Optional[str] = None,
    manifest_path: Optional[str] = None,
    p2a_runs_dir: Optional[str] = None,
    b0_raw_dir: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Path:
    """运行一个显式 P2b 模式；本函数本身不授权真实机器任务。"""
    validate_merger_mode(merger_mode)
    if merger_mode is None:
        raise ValueError("P2b 薄驱动要求显式 merger_mode，禁止缺省回退")
    root = _execution_repo_root(repo_root)
    data = _repo_path(data_root or "data", root)
    runs = _repo_path(runs_root, root) if runs_root else data / "runs"
    manifest = _repo_path(manifest_path or DEFAULT_MANIFEST, root)
    p2a_runs = _repo_path(p2a_runs_dir or DEFAULT_P2A_RUNS, root)
    b0_raw = _repo_path(b0_raw_dir or DEFAULT_B0_RAW, root)
    baseline = _baseline_for_mode(merger_mode, baseline_ref, p2a_runs)

    before = run_preflight(
        data_root=data,
        manifest_path=manifest,
        repo_root=root,
        p2a_runs_dir=p2a_runs,
        b0_raw_dir=b0_raw,
    )
    signature_before = evidence_signature(before)
    notes = build_evidence_notes(before)
    out_dir = b0_driver.replay_b0(
        splits_subset=splits_subset,
        data_root=str(data),
        runs_root=str(runs),
        batch_id=batch_id,
        merger_mode=merger_mode,
        extra_meta_notes=notes,
        baseline_ref=baseline,
        arms=(merger_mode,),
        source=P2B_SOURCE,
    )
    after = run_preflight(
        data_root=data,
        manifest_path=manifest,
        repo_root=root,
        p2a_runs_dir=p2a_runs,
        b0_raw_dir=b0_raw,
    )
    if evidence_signature(after) != signature_before:
        raise RuntimeError(f"P2b 臂运行前后证据变化，批次拒绝放行: {out_dir}")
    logger.info(f"P2b {merger_mode}（sleeve={MERGER_SLEEVE_ID}）完成并通过前后核验: {out_dir}")
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, help="legacy_bypass 或 single_sleeve_passthrough")
    parser.add_argument("--splits", default=None, help="预检折子集，如 0,1,2；终验省略")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--baseline-ref", default=None)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--p2a-runs", default=DEFAULT_P2A_RUNS)
    parser.add_argument("--b0-raw", default=DEFAULT_B0_RAW)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)
    splits = (
        [int(token) for token in args.splits.split(",") if token.strip()] if args.splits else None
    )
    replay_merger_arm(
        merger_mode=args.mode,
        splits_subset=splits,
        data_root=args.data_root,
        runs_root=args.runs_root,
        batch_id=args.batch_id,
        baseline_ref=args.baseline_ref,
        manifest_path=args.manifest,
        p2a_runs_dir=args.p2a_runs,
        b0_raw_dir=args.b0_raw,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
