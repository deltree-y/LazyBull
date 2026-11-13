# -*- coding: utf-8 -*-
"""P7 T6 薄入口：``python scripts/v2_p7/t6/run_t6.py --run-id <id>``。

用法
====
``python scripts/v2_p7/t6/run_t6.py``                             全窗两臂（VV-H / VV-S）
``python scripts/v2_p7/t6/run_t6.py --run-id p7_t6_20261113_r2``  新 run 身份（整改一律新 ID）

证据落 ``data/paper_shadow/p7_testbed/<run_id>/``；不写任何生产/冻结目录。
本 task ``alpha_consumed=true``：产出 ``verdict``（pass / fail / inconclusive），
但只对**本窗口、本机械与费用口径**负责，不外推为阶段授权或验证通过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

from scripts.v2_p7.t6.params import DEFAULT_RUN_ID  # noqa: E402
from scripts.v2_p7.t6.runner import run  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 T6 两臂运行与经济裁定")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="新 run 身份（整改一律新 ID）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    manifest = run(repo_root=REPO_ROOT, run_id=args.run_id)
    gates = manifest["checks"]["gates_summary"]
    logger.info(
        f"run={manifest['run_id']} 门={gates['passed']}/{gates['total']} "
        f"verdict={manifest['verdict']} reason={manifest['verdict_reason']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
