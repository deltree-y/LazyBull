# -*- coding: utf-8 -*-
"""P5b T5 回撤判定表导出薄入口（只读；默认打印，不写盘）。

用法::

    # 只读打印判定表（默认）
    python scripts/v2_p5b/drawdown_export_table.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261102

    # 复制一份到白名单目录（独占创建，不覆盖）
    python scripts/v2_p5b/drawdown_export_table.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261102 \\
        --out temp/h5_table.json

只读语义：本脚本**不改写**任何既有产物；``--out`` 只能落在试车场根或 ``temp/`` 下，
且必须独占创建（已存在即拒绝）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed.paths import assert_write_scope  # noqa: E402
from src.lazybull.v2.hosts.testbed.referee.drawdown.artifacts import (  # noqa: E402
    ARTIFACT_VERDICT,
    artifact_path,
    load_artifact,
    write_json_once,
)


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录")
    parser.add_argument("--out", default="", help="导出路径（必须在试车场根或 temp/ 下）")
    parser.add_argument(
        "--allow-external-root",
        action="store_true",
        help="仅测试/临时脚本：放开写域白名单（不解除生产根禁令）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出判定表")
    return parser.parse_args(argv)


def _summary(verdict: Dict[str, Any]) -> List[str]:
    referee = dict(verdict["drawdown_referee"])
    lines = [
        f"run_id={verdict['run_id']} campaign={verdict['campaign_id']}",
        f"scope={verdict['scope']} evidence_class={verdict['evidence_class']}",
        f"h5_claim={verdict['h5_boundary']['h5_claim']} "
        f"gate_owner={verdict['h5_boundary']['gate_owner']}",
        f"c_w={[round(v, 6) for v in referee['c_w_by_look']]}",
        f"c_i={[round(v, 6) for v in referee['c_i_by_look']]}",
        f"units={referee['unit_count']} failed={len(referee['units_failed'])} "
        f"usable={referee['usable_by_direction']}",
        f"MDE_drawdown_worsen={referee['MDE_drawdown_worsen']['by_scene']}",
        f"MDE_drawdown_improve={referee['MDE_drawdown_improve']['by_scene']}",
    ]
    for row in referee["rows"]:
        lines.append(
            f"  {row['unit']}: {row['successes']}/{row['total']} "
            f"rate={row['rate']:.4%} wilson_high={row['wilson_high']:.4%} pass={row['pass']}"
        )
    return lines


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_root = Path(args.campaign_root)
    verdict = load_artifact(artifact_path(campaign_root, ARTIFACT_VERDICT), what="回撤判定表")
    if args.out:
        target = assert_write_scope(Path(args.out), allow_external_root=args.allow_external_root)
        written = write_json_once(target, verdict, what="回撤判定表导出")
        print(f"exported={written['path']} sha256={written['sha256']}")
    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        for line in _summary(verdict):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
