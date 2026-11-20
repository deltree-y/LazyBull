# -*- coding: utf-8 -*-
"""``python -m scripts.v2_p7.peer_return_gap.run`` 阶段入口。

# 复杂度豁免：P7 候选一阶段专用研究工具，不入生产链路（AGENTS §7.5）。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Sequence

from . import spec
from .common import ResearchError, StageContext, log_stage, run_root
from .stages import run_r0, run_r1, run_r2, run_r3, run_r4, run_r5, run_r6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P7 候选一 PRG-M1-v1 研究阶段")
    parser.add_argument("--campaign", default=spec.CAMPAIGN)
    parser.add_argument("--stage", required=True, choices=list(spec.STAGES))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--upstream", action="append", default=[])
    parser.add_argument(
        "--max-months",
        type=int,
        default=3,
        help="R1/R2 本次处理的月份上限（默认 3；全窗需经济运行授权）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage not in ("R0", "R1", "R2", "R3", "R4", "R5", "R6"):
        print(f"未知阶段：{args.stage}", file=sys.stderr)
        return 3
    ctx = StageContext(
        campaign=args.campaign,
        stage=args.stage,
        run_id=args.run_id,
        upstream=tuple(args.upstream),
    )
    if run_root(args.campaign, args.run_id).exists():
        print(f"run 已存在，拒绝覆盖：{ctx.run_dir}", file=sys.stderr)
        return 5
    try:
        ctx.require_upstream()
        if args.stage == "R0":
            result = run_r0(ctx)
        elif args.stage == "R1":
            result = run_r1(ctx, max_months=args.max_months)
        elif args.stage == "R2":
            result = run_r2(ctx, max_months=args.max_months)
        elif args.stage == "R3":
            result = run_r3(ctx)
        elif args.stage == "R4":
            result = run_r4(ctx)
        elif args.stage == "R5":
            result = run_r5(ctx)
        else:
            result = run_r6(ctx)
    except ResearchError as error:
        ctx.preserve_failure()
        log_stage(args.stage, f"FAILED {args.run_id}: {error}")
        print(f"{args.stage} 拒绝/失败：{error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - 保留失败现场
        ctx.preserve_failure()
        log_stage(args.stage, f"CRASH {args.run_id}: {error!r}")
        traceback.print_exc()
        return 1
    finally:
        ctx.cleanup()
    manifest = result.get("manifest", {})
    print(f"stage={args.stage} run={args.run_id}")
    print(f"wall_clock_seconds={manifest.get('wall_clock_seconds')}")
    print(f"manifest_sha256={manifest.get('manifest_sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
