# -*- coding: utf-8 -*-
"""v2 P1 单元 1：假设台账全量 schema 校验（偿还 hypothesis_ledger_schema F3 §8 技术债）。

对 `data/ledger/hypotheses.jsonl` 全量条目逐条校验（schema F3：JSON 可解析 /
schema_version==1 / hypothesis_id 非空 / kind 合法 / payload 分类必填（含占位 prereg
字段豁免）/ data_state 四键 / registered_at ISO 可解析 / id 唯一）。**只登记不改写**
（append-only），本脚本不动台账文件本身。

用法：
    python scripts/v2_p1/backfill_ledger_validate.py [--ledger PATH] [--out PATH]

产物：`data/reports/v2_p1_ledger_validate_<YYYYMMDD>.json`（逐条 pass/fail + 原因 + 汇总），
控制台打印汇总；存在 fail 条目时退出码 1。业务逻辑在 `src/lazybull/v2/store/ledger_validate.py`
（scripts 薄入口契约）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.v2.store.ledger_validate import validate_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="假设台账全量 schema 校验（只登记不改写）")
    parser.add_argument(
        "--ledger",
        default=str(ROOT / "data" / "ledger" / "hypotheses.jsonl"),
        help="台账 jsonl 路径（默认 data/ledger/hypotheses.jsonl）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="报告输出路径（默认 data/reports/v2_p1_ledger_validate_<YYYYMMDD>.json）",
    )
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        print(f"台账文件不存在: {ledger_path}")
        return 1
    out_path = (
        Path(args.out)
        if args.out
        else (ROOT / "data" / "reports" / f"v2_p1_ledger_validate_{datetime.now():%Y%m%d}.json")
    )

    report = validate_ledger(ledger_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)

    print(f"台账校验完成: {ledger_path}")
    print(f"  总条目 {report['total']}，pass {report['passed']}，fail {report['failed']}")
    for entry in report["entries"]:
        if entry["status"] == "fail":
            print(f"  [FAIL] 第 {entry['line']} 行 {entry['hypothesis_id']}（{entry['kind']}）")
            for reason in entry["reasons"]:
                print(f"         - {reason}")
    print(f"  报告: {out_path}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
