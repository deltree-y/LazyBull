# -*- coding: utf-8 -*-
"""P7 T2.5 薄入口：A 逐日信号再生、相位目标与原始意图预检。

用法（工作根固定 ``data/paper_shadow/p7_testbed/<run_id>/``）::

    python scripts/v2_p7/t25/run_t25.py --run-id p7_t25_20261111_r1
    python scripts/v2_p7/t25/run_t25.py --splits 0 1            # 分段验证
    python scripts/v2_p7/t25/run_t25.py --resume --splits 2     # 续跑同 run

本入口只推理不训练（``alpha_consumed=false``），不写生产根，不产生 pass/fail 判定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p7.t25.params import DEFAULT_RUN_ID  # noqa: E402
from scripts.v2_p7.t25.runner import run  # noqa: E402


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 T2.5 A 袖信号再生与原始意图预检")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--splits", nargs="*", type=int, default=None, help="限定折子集")
    parser.add_argument("--resume", action="store_true", help="复用同一 run_id 的已完成分片")
    parser.add_argument("--no-anchor-plans", action="store_true", help="跳过锚批次折内计划复算")
    parser.add_argument(
        "--verify-splits",
        nargs="*",
        type=int,
        default=(),
        help="逐位重放复核抽样折（建议 ≥3 折）",
    )
    parser.add_argument("--days-per-fold", type=int, default=3)
    parser.add_argument(
        "--counterpart-top-n",
        type=int,
        default=None,
        help="构造对手袖（v10）总目标持仓数；缺省取冻结值 20",
    )
    parser.add_argument(
        "--counterpart-capital",
        type=float,
        default=None,
        help="构造对手袖（v10）虚拟初资；缺省取冻结值 300000",
    )
    parser.add_argument(
        "--accept-input-drift",
        nargs="*",
        default=(),
        help="显式接受的实际输入漂移项，形如 data/clean/stock_basic.parquet:<sha256>",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifest = run(
        run_id=args.run_id,
        splits=args.splits,
        resume=bool(args.resume),
        anchor_plans=not bool(args.no_anchor_plans),
        verify_splits=tuple(args.verify_splits or ()),
        days_per_fold=int(args.days_per_fold),
        counterpart_top_n=args.counterpart_top_n,
        counterpart_capital=args.counterpart_capital,
        accepted_drift=tuple(args.accept_input_drift or ()),
    )
    summary = {
        "run_id": manifest["run_id"],
        "signal_days": manifest["params"]["signal_days"],
        "folds": len(manifest["folds"]),
        "total_seconds": manifest["timing"]["total_seconds"],
        "checks": manifest["checks"],
        "natural_observations": manifest["natural_observations"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
