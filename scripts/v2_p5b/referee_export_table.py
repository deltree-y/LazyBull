# -*- coding: utf-8 -*-
"""P5b T4 判定表导出薄入口（只读：从既有校准产物重新渲染表格）。

用法::

    # 读 campaign reports/ 里的判定表与报告，渲染 markdown 到 stdout
    python scripts/v2_p5b/referee_export_table.py \\
        --campaign-root data/paper_shadow/p5b_testbed/p5b_t4_referee_20261102

    # 精简 JSON（逐单元 + MDE + 哈希链）
    python scripts/v2_p5b/referee_export_table.py --campaign-root <root> --json

    # 落文件（写域守卫：生产根硬禁）
    python scripts/v2_p5b/referee_export_table.py --campaign-root <root> --out temp/t4_verdict.md

只读输入：不重跑、不改任何校准产物。
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
from src.lazybull.v2.hosts.testbed.referee.artifacts import (  # noqa: E402
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
    ARTIFACT_VERDICT,
    artifact_path,
    load_artifact,
)
from src.lazybull.v2.hosts.testbed.referee.report import (  # noqa: E402
    REPORT_SCHEMA,
    VERDICT_SCHEMA,
)


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录")
    parser.add_argument("--out", default="", help="输出路径（缺省打印到 stdout）")
    parser.add_argument("--json", action="store_true", help="输出精简 JSON 汇总")
    return parser.parse_args(argv)


def _summary(report: Dict[str, Any], verdict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "campaign_id": report["campaign_id"],
        "run_id": report["run_id"],
        "scope": report["scope"],
        "evidence_class": report["evidence_class"],
        "referee": report["referee"],
        "hash_chain": report["hash_chain"],
        "sample": {
            "H": report["sample"]["H"],
            "look_blocks": report["sample"]["look_blocks"],
            "look_days": report["sample"]["look_days"],
            "K": report["sample"]["K"],
            "block_length": report["sample"]["block_length"],
            "statistic": report["sample"]["statistic"],
        },
        "thresholds": {
            "c_by_look": report["thresholds"]["c_by_look"],
            "determining_units": report["thresholds"]["determining_units"],
            "alpha_train": report["thresholds"]["alpha_train"],
            "alpha_per_look": report["thresholds"]["alpha_per_look"],
        },
        "holdout": {
            "unit_count": report["holdout"]["unit_count"],
            "referee_usable": report["holdout"]["referee_usable"],
            "failed_units": report["holdout"]["failed_units"],
            "units": report["holdout"]["units"],
        },
        "mde": report["mde"],
        "counts": report["counts"],
        "h4_boundary": report["h4_boundary"],
        "drawdown_referee": report["drawdown_referee"],
        "verdict_income": verdict["income_referee"],
    }


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_root = Path(args.campaign_root)
    report = load_artifact(
        artifact_path(campaign_root, ARTIFACT_REPORT), what="收益裁判报告", schema=REPORT_SCHEMA
    )
    verdict = load_artifact(
        artifact_path(campaign_root, ARTIFACT_VERDICT), what="判定表", schema=VERDICT_SCHEMA
    )
    if args.json:
        text = json.dumps(_summary(report, verdict), ensure_ascii=False, indent=2, default=str)
    else:
        markdown_path = artifact_path(campaign_root, ARTIFACT_REPORT_MD)
        text = markdown_path.read_text(encoding="utf-8")
    if args.out:
        target = Path(args.out)
        # 与其它入口同一口径：默认只允许白名单（temp/ 与试车场根），生产根硬禁；
        # 且独占创建——导出是旁路产物，不得静默覆盖既有文件
        assert_write_scope(target, allow_external_root=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = text + ("" if text.endswith("\n") else "\n")
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
        except FileExistsError as exc:
            raise SystemExit(f"导出目标已存在（禁止覆盖）: {target}") from exc
        print(f"written={target.as_posix()}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
