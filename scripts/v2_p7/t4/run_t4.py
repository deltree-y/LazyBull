# -*- coding: utf-8 -*-
"""P7 T4 薄入口：``python scripts/v2_p7/t4/run_t4.py --run-id <id>``。

用法
====
``python scripts/v2_p7/t4/run_t4.py``                                 全窗（默认 run_id）
``python scripts/v2_p7/t4/run_t4.py --run-id p7_t4_20261112_r2``      新 run 身份重取
``python scripts/v2_p7/t4/run_t4.py --window 20240102 20240628``      分段验证子窗口
``python scripts/v2_p7/t4/run_t4.py --resume``                        续跑（完成根仍拒绝）

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

from scripts.v2_p7.t4.params import DEFAULT_RUN_ID  # noqa: E402
from scripts.v2_p7.t4.runner import CONSTRUCTED_SAMPLE_DAY, run  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 T4 生命周期 / 价源 / 双口径证据生成")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="新 run 身份（整改一律新 ID）")
    parser.add_argument("--resume", action="store_true", help="续跑（完成根会被拒绝）")
    parser.add_argument(
        "--sample-day",
        default=CONSTRUCTED_SAMPLE_DAY,
        help="构造例取样交易日（真实历史价源）",
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
    windows = None
    if args.window:
        from scripts.v2_p7.t4.params import build_params

        params = build_params(REPO_ROOT)
        days = [day for day in params.extended_days if args.window[0] <= day <= args.window[1]]
        if not days:
            logger.error(f"子窗口内无交易日：{args.window}")
            return 2
        windows = [days]
    manifest = run(
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        resume=bool(args.resume),
        sample_day=args.sample_day,
        windows=windows,
    )
    checks = manifest["checks"]
    logger.info(
        f"run={manifest['run_id']} 自然={checks['natural_instances']['counts']} "
        f"执行={checks['executed_scenarios']['counts']} "
        f"费用闭合={checks['fees']['closed']} 恒等式破缺日={checks['identities']['failure_days']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
