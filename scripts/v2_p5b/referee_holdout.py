# -*- coding: utf-8 -*-
"""P5b T4 收益裁判 holdout 薄入口（逐单元 Wilson 门 + 功效/MDE；默认 dry-run）。

用法::

    python scripts/v2_p5b/referee_holdout.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t4_referee_20261102 --apply

前置：``referee_calibration.py --freeze --apply`` 与 ``--train --apply`` 已完成。
本入口先校验 **manifest ↔ 校准器 ↔ 临界值** 三环哈希（不符即拒绝运行），再跑 holdout 与
功效，产出 ``income_holdout.json`` / ``income_power.json`` / ``referee_verdict.json`` /
``income_referee_report.{json,md}`` / ``machine_time.json``。

默认 **dry-run**；``--apply`` 才写。产物已存在 ⇒ 拒绝（换新 run 身份目录重跑）。

退出码：0 = 裁判可用（holdout 全单元过门）；2 = 裁判不可用（任一单元不过，不得删场景/调阈值）；
1 = 参数或校验错误。
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
    project_root,
)
from src.lazybull.v2.hosts.testbed.referee.calibration import (  # noqa: E402
    run_holdout,
    write_machine_time,
)

#: 单批长任务授权阈值（秒；设计 §4.4「预计 >20 分钟须单独授权」）
LONG_TASK_THRESHOLD_SECONDS = 1200.0


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录")
    parser.add_argument("--apply", action="store_true", help="实际写盘（缺省 dry-run）")
    parser.add_argument("--chunk", type=int, default=2000, help="分块路径数")
    parser.add_argument(
        "--repo-root",
        default="",
        help="仓库根（缺省 = 本脚本所在仓库；仅测试用隔离假仓库）",
    )
    parser.add_argument(
        "--repo-manifest",
        default=str(
            project_root() / "docs" / "plans" / "v2" / "p5b_t4_calibration_manifest_20261102.json"
        ),
        help="仓库冻结件路径（缺省 = 本仓库冻结件；给出时核验与 campaign 副本逐字节一致）",
    )
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
    outcome = run_holdout(
        campaign_root,
        repo_root=Path(args.repo_root) if args.repo_root else project_root(),
        repo_manifest=Path(args.repo_manifest) if args.repo_manifest else None,
        apply=args.apply,
        allow_external_root=args.allow_external_root,
        chunk=args.chunk,
    )
    wall = time.perf_counter() - started
    payload: Dict[str, Any] = {
        "campaign_root": campaign_root.as_posix(),
        "dry_run": outcome["dry_run"],
        "referee_usable": outcome["referee_usable"],
        "unit_count": outcome["unit_count"],
        "failed_units": outcome["failed_units"],
        "mde_registered": outcome["mde"]["registered"],
        "mde_by_scene": outcome["mde"]["by_scene"],
        "mde_status_by_scene": outcome["mde"]["status_by_scene"],
        "machine_seconds": outcome["machine_seconds"],
        "wall_seconds": wall,
        "long_task_threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
        "long_task_threshold_exceeded": bool(wall > LONG_TASK_THRESHOLD_SECONDS),
    }
    for key in ("holdout", "power", "verdict", "report", "report_md"):
        if outcome.get(key):
            payload[key] = outcome[key]
    if args.apply:
        machine = write_machine_time(
            campaign_root,
            {
                "phase": "t4_income_referee_holdout",
                "wall_seconds": wall,
                "machine_seconds": outcome["machine_seconds"],
                "chunk": args.chunk,
                "long_task_threshold_seconds": LONG_TASK_THRESHOLD_SECONDS,
                "long_task_threshold_exceeded": bool(wall > LONG_TASK_THRESHOLD_SECONDS),
                "authorization": (
                    "not_required" if wall <= LONG_TASK_THRESHOLD_SECONDS else "required_before_run"
                ),
            },
        )
        payload["machine_time"] = machine
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"campaign={payload['campaign_root']} dry_run={payload['dry_run']}")
        print(
            f"referee_usable={payload['referee_usable']} units={payload['unit_count']} "
            f"failed={len(payload['failed_units'])}"
        )
        print(f"mde_registered={payload['mde_registered']} mde={payload['mde_by_scene']}")
        print(
            f"wall_seconds={wall:.2f} long_task_exceeded={payload['long_task_threshold_exceeded']}"
        )
        for key in ("holdout", "power", "verdict", "report", "report_md", "machine_time"):
            if payload.get(key):
                print(f"{key}={payload[key]['path']}")
    return 0 if outcome["referee_usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
