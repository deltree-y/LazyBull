# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判薄入口：冻结回撤 manifest / 训练 c^W、c^I（默认 dry-run）。

用法::

    # 1) 冻结回撤仪器 manifest（先冻结，之后不可改；只读 B0 盲态选窗）
    python scripts/v2_p5b/drawdown_calibration.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261102 \\
        --freeze --apply

    # 2) 训练两个方向的临界值（只读训练池；写 drawdown_thresholds.json）
    python scripts/v2_p5b/drawdown_calibration.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261102 \\
        --train --apply

默认 **dry-run**（只打印计划/数值，不写盘）；``--apply`` 才写。写域守卫：只允许
``data/paper_shadow/p5b_testbed/`` 或 ``temp/``（``--allow-external-root`` 仅测试放开，
永不解除生产根禁令）。重复运行拒绝覆盖（已存在产物即失败）。

上游父证据：默认指向 T4 收益仪器（仓库冻结件 + campaign 临界值），sha256 必须与登记值一致，
否则拒绝运行（``--parent-manifest``/``--parent-thresholds`` 仅测试可覆盖）。

退出码：0 = 完成/可继续；1 = 参数或校验错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed.paths import (  # noqa: E402
    assert_write_scope,
    forbidden_roots,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.artifacts import (  # noqa: E402
    ARTIFACT_MANIFEST,
    artifact_path,
    sha256_file,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.calibration import (  # noqa: E402
    build_drawdown_manifest,
    freeze_drawdown_manifest,
    source_from_pair_document,
    train_drawdown_calibration,
    write_machine_time,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.manifest import (  # noqa: E402
    SCALE_FLOOR,
    calibrator_identity,
    freeze_manifest,
    inspect_parent,
    scale_floor_errors,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (  # noqa: E402
    HOLD_H,
    WINDOW_SAMPLES,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.sections import lineage_section  # noqa: E402

#: 上游收益仪器 manifest（仓库冻结件；正式证据以 campaign 副本为准）
DEFAULT_PARENT_MANIFEST = ROOT / "docs/plans/v2/p5b_t4_calibration_manifest_20261102.json"
#: 上游收益仪器 manifest 的冻结 sha256（T4 闭合报告 §4.4）
DEFAULT_PARENT_MANIFEST_SHA256 = "545d2be43683c0fb4c95ad964717f5b110527a7d8dc33f0b94d72ad117853450"
#: 上游收益临界值产物（T4 campaign）
DEFAULT_PARENT_THRESHOLDS = (
    ROOT
    / "data/paper_shadow/p5b_testbed/p5b_t4_referee_20261102/reports/calibration_thresholds.json"
)
#: 上游收益临界值产物的冻结 sha256（T4 闭合报告 §4.4）
DEFAULT_PARENT_THRESHOLDS_SHA256 = (
    "af37b21786c68fc1729a805522d0ef93c2fcf8dd7b6a7948f2cbfc719b55cdaf"
)
#: 仓库内回撤冻结件默认落点
DEFAULT_REPO_MANIFEST = ROOT / "docs/plans/v2/p5b_t5_drawdown_manifest_20261102.json"
#: 默认 run 身份
DEFAULT_RUN_ID = "p5b_t5_drawdown_20261102"
#: T0.5b §5「回撤注入预检」已批读数：B0 最后 500 收益样本的基准 MDD（pp）
T05B_WINDOW_MDD_PP = -20.7076
#: 上述读数的对照容差（pp）
T05B_WINDOW_MDD_TOLERANCE = 1e-4


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录（产物落 reports/）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="冻结回撤仪器 manifest")
    group.add_argument("--train", action="store_true", help="训练 c^W、c^I")
    parser.add_argument("--apply", action="store_true", help="实际写盘（缺省 dry-run）")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="本次校准身份")
    parser.add_argument(
        "--repo-manifest", default=str(DEFAULT_REPO_MANIFEST), help="仓库内回撤冻结件路径"
    )
    parser.add_argument(
        "--parent-manifest",
        default=str(DEFAULT_PARENT_MANIFEST),
        help="上游收益仪器 manifest（仅测试可覆盖）",
    )
    parser.add_argument(
        "--parent-manifest-sha256",
        default=DEFAULT_PARENT_MANIFEST_SHA256,
        help="上游收益 manifest 的期望 sha256（覆盖 --parent-manifest 时必须显式给出）",
    )
    parser.add_argument(
        "--parent-thresholds",
        default=str(DEFAULT_PARENT_THRESHOLDS),
        help="上游收益临界值产物（仅测试可覆盖）",
    )
    parser.add_argument(
        "--parent-thresholds-sha256",
        default=DEFAULT_PARENT_THRESHOLDS_SHA256,
        help="上游收益临界值的期望 sha256（覆盖 --parent-thresholds 时必须显式给出）",
    )
    parser.add_argument(
        "--repo-root", default="", help="仓库根（缺省 = 本脚本所在仓库；仅测试用隔离假仓库）"
    )
    parser.add_argument(
        "--expected-window-mdd",
        type=float,
        default=T05B_WINDOW_MDD_PP,
        help="B0 冻结窗口基准 MDD 的对照值（T0.5b §5 已批读数；仅测试可覆盖）",
    )
    parser.add_argument(
        "--window-mdd-tolerance",
        type=float,
        default=T05B_WINDOW_MDD_TOLERANCE,
        help="上述对照的绝对容差（pp）",
    )
    parser.add_argument(
        "--sequence", type=int, default=1, help="冻结序号（首次 = 1；重冻必须递增）"
    )
    parser.add_argument("--supersedes", default="", help="被重冻前身的 manifest sha256（重冻必填）")
    parser.add_argument(
        "--supersedes-calibrator", default="", help="被重冻前身的校准器 sha256（重冻必填）"
    )
    parser.add_argument("--supersede-reason", default="", help="重冻原因（重冻必填）")
    parser.add_argument("--seed", type=int, default=None, help="命名种子根（缺省 = 上游冻结值）")
    parser.add_argument("--prefix", default=None, help="realization 前缀（缺省 = 上游冻结值）")
    parser.add_argument("--train-per-unit", type=int, default=None, help="训练实现数/单元")
    parser.add_argument("--holdout-per-unit", type=int, default=None, help="holdout 实现数/单元")
    parser.add_argument("--effect-per-unit", type=int, default=None, help="效应档实现数/场景")
    parser.add_argument("--chunk", type=int, default=2000, help="分块实现数（不影响结果）")
    parser.add_argument(
        "--allow-external-root",
        action="store_true",
        help="仅测试/临时脚本：放开写域白名单（不解除生产根禁令）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    return parser.parse_args(argv)


def _guard_repo_manifest(path: Path, *, repo_root: Path, allow_external: bool) -> Path:
    """仓库冻结件写域守卫：生产根**硬禁**，其余只允许 ``docs/plans/v2/``。

    仓库冻结件按项目惯例落 ``docs/plans/v2/``（本地归档、不入 git），它既不是试车场产物根
    也不是 ``temp/``，因此单靠 ``assert_write_scope`` 会误拒；这里显式放开这一目录，同时
    **保留**生产根禁令（``--repo-manifest data/paper/x.json`` 仍被拒绝）。
    """
    target = Path(path).resolve()
    for forbidden in forbidden_roots(None):
        resolved = Path(forbidden).resolve()
        if target == resolved or resolved in target.parents:
            raise SystemExit(
                f"仓库冻结件禁止写生产根: {target}（命中 {resolved}）；"
                "P5b 只允许 data/paper_shadow/p5b_testbed/、temp/ 或 docs/plans/v2/"
            )
    if allow_external:
        return target
    allowed = (Path(repo_root).resolve() / "docs" / "plans" / "v2",)
    if not any(target.parent == root for root in allowed):
        raise SystemExit(
            f"仓库冻结件只允许写 {allowed[0]}（实际 {target.parent}）；"
            "其他位置须显式 --allow-external-root（仅测试）"
        )
    return target


def _counts(args: argparse.Namespace, parent_doc: Dict[str, Any]) -> Dict[str, Any]:
    """实现数/种子（缺省 = 上游 T4 冻结值；只允许上调）。"""
    counts = dict((parent_doc or {}).get("counts") or {})
    seed = args.seed if args.seed is not None else int(counts.get("seed", 20261103))
    prefix = args.prefix or str(counts.get("realization_prefix", DEFAULT_RUN_ID))
    merged = {
        "train_per_unit": int(args.train_per_unit or SCALE_FLOOR["train_per_unit"]),
        "holdout_per_unit": int(args.holdout_per_unit or SCALE_FLOOR["holdout_per_unit"]),
        "effect_per_scene_per_level": int(
            args.effect_per_unit or SCALE_FLOOR["effect_per_scene_per_level"]
        ),
        "seed": int(seed),
        "realization_prefix": str(prefix),
    }
    floor = scale_floor_errors(merged)
    if floor:
        raise SystemExit("实现数低于冻结下限（禁止下调）: " + "; ".join(floor))
    return merged


def _freeze(args: argparse.Namespace, campaign_root: Path) -> Dict[str, Any]:
    repo_root = Path(args.repo_root) if args.repo_root else ROOT
    info = inspect_parent(
        Path(args.parent_manifest),
        Path(args.parent_thresholds),
        expected_manifest_sha256=args.parent_manifest_sha256,
        expected_thresholds_sha256=args.parent_thresholds_sha256,
        require_scale_floor=True,
    )
    source = source_from_pair_document(
        info["parent_source_pair"],
        dgp_doc=info["parent_calibration"],
        repo_root=repo_root,
        h=HOLD_H,
        window_samples=WINDOW_SAMPLES,
        expected_window_mdd=float(args.expected_window_mdd),
        mdd_tolerance=float(args.window_mdd_tolerance),
    )
    counts = _counts(args, {"counts": dict(info["parent_counts"])})
    doc = build_drawdown_manifest(
        run_id=str(args.run_id),
        source=source,
        windows=source.windows,
        counts=counts,
        upstream=info,
        calibrator=calibrator_identity(),
        repo_root=repo_root,
        lineage=lineage_section(
            sequence=int(args.sequence),
            superseded_manifest_sha256=str(args.supersedes) or None,
            superseded_calibrator_sha256=str(args.supersedes_calibrator) or None,
            supersede_reason=str(args.supersede_reason) or None,
        ),
    )
    frozen = freeze_drawdown_manifest(
        campaign_root, doc, apply=args.apply, allow_external_root=args.allow_external_root
    )
    result: Dict[str, Any] = {
        "mode": "freeze",
        "campaign_root": campaign_root.as_posix(),
        "counts": counts,
        "calibrator_sha256": calibrator_identity()["sha256"],
        "campaign_manifest": frozen,
        "repo_manifest": str(Path(args.repo_manifest).as_posix()),
        "window_mdd_percent": float(source.window.mdd_percent),
        "crash_windows": [item.as_dict() for item in source.windows],
        "parent_manifest_sha256": info["parent_manifest"]["sha256"],
        "parent_thresholds_sha256": info["parent_thresholds"]["sha256"],
    }
    if args.apply:
        repo_target = _guard_repo_manifest(
            Path(args.repo_manifest),
            repo_root=repo_root,
            allow_external=args.allow_external_root,
        )
        written = freeze_manifest(doc, repo_target)
        result["repo_written"] = written
        if written["sha256"] != frozen.get("sha256"):
            raise SystemExit(
                "campaign 副本与仓库冻结件哈希不一致（序列化口径必须相同）: "
                f"{frozen.get('sha256')} vs {written['sha256']}"
            )
    return result


def _train(args: argparse.Namespace, campaign_root: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    outcome = train_drawdown_calibration(
        campaign_root,
        repo_root=Path(args.repo_root) if args.repo_root else ROOT,
        repo_manifest=Path(args.repo_manifest) if args.repo_manifest else None,
        apply=args.apply,
        allow_external_root=args.allow_external_root,
        require_scale_floor=True,
        chunk=args.chunk,
    )
    outcome = {**outcome, "mode": "train", "campaign_root": campaign_root.as_posix()}
    outcome["wall_seconds"] = time.perf_counter() - started
    outcome["manifest_sha256"] = sha256_file(artifact_path(campaign_root, ARTIFACT_MANIFEST))
    return outcome


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_root = Path(args.campaign_root)
    assert_write_scope(campaign_root, allow_external_root=args.allow_external_root)
    outcome = _freeze(args, campaign_root) if args.freeze else _train(args, campaign_root)
    if args.apply:
        write_machine_time(
            campaign_root,
            {
                "mode": outcome["mode"],
                "campaign_id": campaign_root.name,
                "wall_seconds": outcome.get("wall_seconds", 0.0),
                "machine_seconds": outcome.get("machine_seconds"),
                "note": "墙钟不属于产物身份；同 seed 重跑逐字节一致（分阶段独立落盘）",
            },
            stage=str(outcome["mode"]),
        )
    if args.json:
        print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"mode={outcome['mode']} campaign={outcome['campaign_root']} "
            f"dry_run={outcome.get('dry_run')}"
        )
        if outcome["mode"] == "freeze":
            print(f"manifest_target={outcome['campaign_manifest']['target']}")
            print(
                f"window_mdd={outcome['window_mdd_percent']:.4f}pp "
                f"windows={len(outcome['crash_windows'])}"
            )
            print(f"parent={outcome['parent_manifest_sha256'][:8]}")
        else:
            print(f"c_w={[round(v, 6) for v in outcome['c_w_by_look']]}")
            print(f"c_i={[round(v, 6) for v in outcome['c_i_by_look']]}")
            print(f"wall_seconds={outcome['wall_seconds']:.2f}")
            if outcome.get("thresholds_sha256"):
                print(f"thresholds_sha256={outcome['thresholds_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
