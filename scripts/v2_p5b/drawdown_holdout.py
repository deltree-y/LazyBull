# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判 holdout + 功效薄入口（默认 dry-run）。

用法::

    python scripts/v2_p5b/drawdown_holdout.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261102 \\
        --apply

跑 ``holdout``（逐单元 Wilson 上界门，两个方向分别判定）+ 效应网格（纯注入档
``{1,2,3,5,8,12}pp``，逐方向 × 逐档 × 逐场景求解 λ/ζ）+ 两个 MDE，并写
``drawdown_holdout.json`` / ``drawdown_power.json`` / ``drawdown_verdict.json`` /
``drawdown_report.json`` / ``drawdown_report.md`` / ``machine_time.json``。

前置（任一不符即拒绝运行）：回撤 manifest 已冻结且三环哈希互链、上游收益 manifest 与收益
临界值逐字节匹配登记值、临界值产物 ``passed=true``。

退出码：0 = 两个方向均过门且均登记 MDE；3 = A/A 过门但至少一个方向未登记 MDE；
4 = 任一单元未过门（该方向裁判不可用）；2 = argparse 用法错误；1 = 校验/写域错误。
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
from src.lazybull.v2.hosts.testbed.referee.drawdown.calibration import (  # noqa: E402
    run_drawdown_holdout,
    write_machine_time,
)

#: 仓库内回撤冻结件默认路径
DEFAULT_REPO_MANIFEST = ROOT / "docs/plans/v2/p5b_t5_drawdown_manifest_20261102.json"


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录（产物落 reports/）")
    parser.add_argument("--apply", action="store_true", help="实际写盘（缺省 dry-run）")
    parser.add_argument(
        "--repo-manifest", default=str(DEFAULT_REPO_MANIFEST), help="仓库内回撤冻结件路径"
    )
    parser.add_argument(
        "--repo-root", default="", help="仓库根（缺省 = 本脚本所在仓库；仅测试用隔离假仓库）"
    )
    parser.add_argument("--chunk", type=int, default=2000, help="分块实现数（不影响结果）")
    parser.add_argument(
        "--allow-external-root",
        action="store_true",
        help="仅测试/临时脚本：放开写域白名单（不解除生产根禁令）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_root = Path(args.campaign_root)
    assert_write_scope(campaign_root, allow_external_root=args.allow_external_root)
    started = time.perf_counter()
    outcome: Dict[str, Any] = run_drawdown_holdout(
        campaign_root,
        repo_root=Path(args.repo_root) if args.repo_root else ROOT,
        repo_manifest=Path(args.repo_manifest) if args.repo_manifest else None,
        apply=args.apply,
        allow_external_root=args.allow_external_root,
        chunk=args.chunk,
        require_scale_floor=True,
    )
    outcome["mode"] = "holdout"
    outcome["campaign_root"] = campaign_root.as_posix()
    outcome["wall_seconds"] = time.perf_counter() - started
    if args.apply:
        written = write_machine_time(
            campaign_root,
            {
                "mode": "holdout",
                "campaign_id": campaign_root.name,
                "wall_seconds": outcome["wall_seconds"],
                "holdout_power_seconds": outcome.get("machine_seconds"),
                "chunk": args.chunk,
                "note": "墙钟不属于产物身份；同 seed 重跑逐字节一致（分阶段独立落盘）",
            },
            stage="holdout",
        )
        outcome["machine_time"] = written
    if args.json:
        print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
    else:
        usable = outcome["usable_by_direction"]
        print(
            f"mode=holdout campaign={outcome['campaign_root']} dry_run={outcome.get('dry_run')} "
            f"units={outcome['unit_count']} failed={len(outcome['failed_units'])}"
        )
        print(f"usable dd_worsen={usable.get('dd_worsen')} dd_improve={usable.get('dd_improve')}")
        print(
            f"MDE_drawdown_worsen={outcome['mde_worsen']['by_scene']} "
            f"registered={outcome['mde_worsen'].get('registered')}"
        )
        print(
            f"MDE_drawdown_improve={outcome['mde_improve']['by_scene']} "
            f"registered={outcome['mde_improve'].get('registered')}"
        )
        print(f"wall_seconds={outcome['wall_seconds']:.2f}")
    if not outcome["referee_usable"]:
        # 4 = 单元门失败（避免与 argparse 的用法错误退出码 2 冲突）
        return 4
    if not all(
        bool(item.get("registered")) for item in (outcome["mde_worsen"], outcome["mde_improve"])
    ):
        # A/A 过门但至少一个方向在网格内未登记 MDE：不是失败，但必须显式区分
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
