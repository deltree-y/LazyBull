# -*- coding: utf-8 -*-
"""P5b T3 判定表导出薄入口（只读；从既有矩阵结果重新渲染表格）。

用法::

    python scripts/v2_p5b/export_verdict_table.py \\
        --matrix-json temp/p5b_t3_fm/audit/fault_matrix.json --out temp/p5b_t3_fm_verdict.md

默认输出到 stdout；``--out`` 落文件。只读输入，不改任何 campaign 产物。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    INVARIANTS,
    assert_write_scope,
    table_markdown,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix-json", required=True, help="fault_matrix.json 路径")
    parser.add_argument("--out", default="", help="输出路径（缺省打印到 stdout）")
    parser.add_argument("--json", action="store_true", help="改为输出精简 JSON 汇总")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = Path(args.matrix_json)
    table = json.loads(source.read_text(encoding="utf-8"))
    cases = list(table.get("cases") or [])
    if args.json:
        payload = {
            "source": source.as_posix(),
            "all_passed": table.get("all_passed"),
            "case_count": table.get("case_count"),
            "passed_count": table.get("passed_count"),
            "invariants": list(table.get("invariants") or INVARIANTS),
            "rows": [
                {
                    "case_id": row["case_id"],
                    "fault_point": row["fault_point"],
                    "terminal": row["terminal"],
                    "expected_terminal": row["expected_terminal"],
                    "recovery": row["recovery"],
                    "invariants": row["invariants"],
                    "passed": row["passed"],
                }
                for row in cases
            ],
            "containment": table.get("containment"),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        header = (
            f"campaign={table.get('campaign_id')} cases={table.get('case_count')} "
            f"passed={table.get('passed_count')} all_passed={table.get('all_passed')}"
        )
        text = header + "\n\n" + table_markdown(cases)
    if args.out:
        # 写域守卫：生产根硬禁（导出目录不受试车场白名单限制，但永不可落生产面）
        assert_write_scope(Path(args.out), allow_external_root=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"written={Path(args.out).as_posix()} rows={len(cases)}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
