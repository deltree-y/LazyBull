# -*- coding: utf-8 -*-
"""P7 T5 薄入口：``python scripts/v2_p7/t5/run_t5.py --run-id <id>``。

用法
====
``python scripts/v2_p7/t5/run_t5.py``                             全窗三臂（T4 互证 / VV-H / VV-S）
``python scripts/v2_p7/t5/run_t5.py --run-id p7_t5_20261113_r2`` 新 run 身份重取
``python scripts/v2_p7/t5/run_t5.py --window 20240102 20240628`` 分段验证子窗口
``python scripts/v2_p7/t5/run_t5.py --arms vvs``                  只跑单臂（调试）
``python scripts/v2_p7/t5/run_t5.py --resume``                    续跑（完成根仍拒绝）

证据落 ``data/paper_shadow/p7_testbed/<run_id>/``；不写任何生产/冻结目录。
本 task ``alpha_consumed=false``：不产生任何 pass/fail 经济判定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

from scripts.v2_p7.t5.params import DEFAULT_RUN_ID  # noqa: E402
from scripts.v2_p7.t5.runner import run  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 T5 配额/借用/召回的机械证据生成")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="新 run 身份（整改一律新 ID）")
    parser.add_argument("--resume", action="store_true", help="续跑（完成根会被拒绝）")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["parity", "vvh", "vvs"],
        choices=["parity", "vvh", "vvs"],
        help="要运行的臂（默认三臂全跑）",
    )
    parser.add_argument(
        "--window",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="分段验证子窗口（YYYYMMDD YYYYMMDD；不替代全窗证据）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    manifest = run(
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        resume=bool(args.resume),
        arms=tuple(args.arms),
        windows=args.window,
    )
    checks = manifest["checks"]
    logger.info(
        f"run={manifest['run_id']} 构造例={checks['constructed_cases']['passed']}"
        f"/{checks['constructed_cases']['total']} T4互证={checks['t4_parity']['ok']} "
        f"VV-H参照={checks['vvh_reference']['ok']} "
        f"非负门={checks['nonnegative_gate']['ok']}"
        f"(违例日={checks['nonnegative_gate']['violation_days']}) "
        f"恒等式破缺日={checks['identities']['failure_days']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
