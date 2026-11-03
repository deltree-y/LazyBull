# -*- coding: utf-8 -*-
"""P5b T4 收益裁判校准薄入口：冻结 §5.5 manifest / 训练低尾临界值（默认 dry-run）。

用法::

    # 1) 冻结校准 manifest（先冻结，之后不可改）
    python scripts/v2_p5b/referee_calibration.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t4_referee_20261102 \\
        --freeze --apply

    # 2) 训练临界值（只读训练池；写 calibration_thresholds.json）
    python scripts/v2_p5b/referee_calibration.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t4_referee_20261102 \\
        --train --apply

默认 **dry-run**（只打印计划/数值，不写盘）；``--apply`` 才写。写域守卫：只允许
``data/paper_shadow/p5b_testbed/`` 或 ``temp/``（``--allow-external-root`` 仅测试放开，
永不解除生产根禁令）。重复运行拒绝覆盖（已存在产物即失败）。

退出码：0 = 完成/可继续；2 = 配置硬失败（无效率超门/阈值不可估计）；1 = 参数或校验错误。
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

from src.lazybull.v2.hosts.testbed.paths import assert_write_scope  # noqa: E402
from src.lazybull.v2.hosts.testbed.referee.artifacts import (  # noqa: E402
    ARTIFACT_MANIFEST,
    artifact_path,
    sha256_file,
)
from src.lazybull.v2.hosts.testbed.referee.calibration import (  # noqa: E402
    DESIGN_DEFAULTS,
    build_manifest_document,
    freeze_calibration_manifest,
    source_pair_for_manifest,
    train_calibration,
)
from src.lazybull.v2.hosts.testbed.referee.manifest_sections import (  # noqa: E402
    lineage_section,
)
from src.lazybull.v2.hosts.testbed.referee.manifest import (  # noqa: E402
    SCALE_FLOOR,
    calibrator_identity,
    freeze_manifest,
    scale_floor_errors,
)

#: T0.5b PSD r3 冻结 manifest（种子/前缀/源折/源对哈希的唯一来源）
PSD_MANIFEST = ROOT / "docs/plans/v2/p5b_calibration_manifest_psd_20261101_r3.json"
#: 上述 manifest 的文件哈希（冻结身份，防换源）
PSD_MANIFEST_SHA256 = "3b371593811031f5e921abda198ca18895e3040dc21b12a857759111d3314ca4"
#: 仓库内 T4 冻结件默认落点
DEFAULT_REPO_MANIFEST = ROOT / "docs/plans/v2/p5b_t4_calibration_manifest_20261102.json"
#: 默认 run 身份
DEFAULT_RUN_ID = "p5b_t4_referee_20261102"


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    """仓库内文件登记为相对路径；仓库外（仅测试）保留绝对路径并显式告警。"""
    try:
        return Path(path).resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        print(
            f"WARNING: 引用文件在仓库外（仅测试可用）: {path}；repo_root={repo_root}",
            file=sys.stderr,
        )
        return Path(path).resolve().as_posix()


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录（产物落 reports/）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="冻结 §5.5 校准 manifest")
    group.add_argument("--train", action="store_true", help="训练低尾临界值 c_k")
    parser.add_argument("--apply", action="store_true", help="实际写盘（缺省 dry-run）")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="本次校准身份")
    parser.add_argument(
        "--repo-manifest", default=str(DEFAULT_REPO_MANIFEST), help="仓库冻结件路径"
    )
    parser.add_argument(
        "--psd-manifest",
        default=str(PSD_MANIFEST),
        help="T0.5b PSD 冻结 manifest（种子/前缀/源折/源对哈希来源；仅测试可覆盖）",
    )
    parser.add_argument(
        "--psd-manifest-sha256",
        default=PSD_MANIFEST_SHA256,
        help="上述 manifest 的期望 sha256（覆盖 --psd-manifest 时必须显式给出）",
    )
    parser.add_argument(
        "--repo-root",
        default="",
        help="仓库根（缺省 = 本脚本所在仓库；仅测试用隔离假仓库）",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="命名种子根（缺省 = T0.5b 冻结值 20261103）"
    )
    parser.add_argument("--prefix", default=None, help="realization 前缀（缺省 = T0.5b 冻结值）")
    parser.add_argument("--train-per-unit", type=int, default=None, help="训练实现数/单元")
    parser.add_argument("--holdout-per-unit", type=int, default=None, help="holdout 实现数/单元")
    parser.add_argument("--effect-per-unit", type=int, default=None, help="效应档实现数/场景")
    parser.add_argument(
        "--chunk", type=int, default=2000, help="分块路径数（不影响结果，只影响峰值内存）"
    )
    parser.add_argument(
        "--allow-external-root",
        action="store_true",
        help="仅测试/临时脚本：放开写域白名单（不解除生产根禁令）",
    )
    parser.add_argument(
        "--sequence", type=int, default=1, help="冻结序号（首次 = 1；重冻必须递增）"
    )
    parser.add_argument("--supersedes", default="", help="重冻时被废止的 manifest sha256（64 位）")
    parser.add_argument(
        "--supersedes-calibrator",
        default="",
        help="重冻时被废止的 calibrator sha256（64 位）",
    )
    parser.add_argument(
        "--supersede-reason", default="", help="重冻原因（>=8 字；必须写清动了什么）"
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    return parser.parse_args(argv)


def _load_psd_manifest(path: Path, expected_sha256: str) -> Dict[str, Any]:
    """读取并核验 T0.5b 冻结 manifest（种子/前缀/源折/源对哈希的唯一来源）。"""
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"T0.5b PSD manifest 缺失: {source}")
    actual = sha256_file(source)
    if actual != str(expected_sha256):
        raise SystemExit(
            f"T0.5b PSD manifest 哈希不符（禁止改用其它源/参数）: {source} "
            f"实际={actual} 期望={expected_sha256}"
        )
    return json.loads(source.read_text(encoding="utf-8"))


def _freeze(args: argparse.Namespace, campaign_root: Path) -> Dict[str, Any]:
    repo_root = Path(args.repo_root) if args.repo_root else ROOT
    psd_path = Path(args.psd_manifest)
    psd = _load_psd_manifest(psd_path, args.psd_manifest_sha256)
    pair_doc = psd["calibration"]["source_pair"]
    counts = {
        "train_per_unit": int(args.train_per_unit or SCALE_FLOOR["train_per_unit"]),
        "holdout_per_unit": int(args.holdout_per_unit or SCALE_FLOOR["holdout_per_unit"]),
        "effect_per_scene_per_level": int(
            args.effect_per_unit or SCALE_FLOOR["effect_per_scene_per_level"]
        ),
        "seed": int(args.seed if args.seed is not None else psd["calibration"]["counts"]["seed"]),
        "realization_prefix": str(
            args.prefix if args.prefix else psd["calibration"]["counts"]["realization_prefix"]
        ),
    }
    floor = scale_floor_errors(counts)
    if floor:
        raise SystemExit("实现数低于 §5.3 冻结下限（禁止下调）: " + "; ".join(floor))
    # 用冻结源对构造「只含 manifest 侧声明」的预读（哈希在此已核验）
    probe = {
        "source_pair": {
            "base_chain_nav": pair_doc["base_chain_nav"],
            "noise_chain_nav": pair_doc["noise_chain_nav"],
            "base_sha256": pair_doc["base_sha256"],
            "noise_sha256": pair_doc["noise_sha256"],
            "train_folds": pair_doc["train_folds"],
            "holdout_folds": pair_doc["validation_source_folds"],
        },
        "files": {
            "prereg": {
                "path": psd["prereg"]["path"],
                "sha256": psd["prereg"]["sha256"],
            },
            "psd_manifest": {
                "path": _relative_or_absolute(psd_path, repo_root),
                "sha256": str(args.psd_manifest_sha256),
            },
        },
        "income_referee": {"sample": {"H": int(DESIGN_DEFAULTS["H"]), "n_max": 503}},
        "calibration": {
            "dgp": psd["calibration"]["dgp"],
            "block_resample": psd["calibration"]["block_resample"],
        },
    }
    source = source_pair_for_manifest(probe, repo_root=repo_root)
    lineage = lineage_section(
        sequence=int(args.sequence),
        superseded_manifest_sha256=str(args.supersedes) or None,
        superseded_calibrator_sha256=str(args.supersedes_calibrator) or None,
        supersede_reason=str(args.supersede_reason) or None,
    )
    doc = build_manifest_document(
        run_id=str(args.run_id),
        source=source,
        prereg=probe["files"]["prereg"],
        psd_manifest=probe["files"]["psd_manifest"],
        counts=counts,
        calibrator=calibrator_identity(),
        repo_root=repo_root,
        lineage=lineage,
    )
    frozen = freeze_calibration_manifest(
        campaign_root, doc, apply=args.apply, allow_external_root=args.allow_external_root
    )
    result: Dict[str, Any] = {
        "mode": "freeze",
        "campaign_root": campaign_root.as_posix(),
        "counts": counts,
        "calibrator_sha256": calibrator_identity()["sha256"],
        "campaign_manifest": frozen,
        "repo_manifest": str(Path(args.repo_manifest).as_posix()),
    }
    if args.apply:
        repo_target = Path(args.repo_manifest)
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
    outcome = train_calibration(
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
    if args.json:
        print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"mode={outcome['mode']} campaign={outcome['campaign_root']} "
            f"dry_run={outcome.get('dry_run')}"
        )
        if outcome["mode"] == "freeze":
            manifest = outcome["campaign_manifest"]
            print(f"manifest_target={manifest['target']} written={manifest['written']}")
        else:
            ordered = [round(value, 10) for value in outcome["c_by_look"]]
            print(f"passed={outcome['passed']} c_by_look={ordered}")
            print(f"failures={len(outcome['failures'])} wall_seconds={outcome['wall_seconds']:.2f}")
            if outcome.get("thresholds_sha256"):
                print(f"thresholds_sha256={outcome['thresholds_sha256']}")
    if outcome["mode"] == "train" and not outcome["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
