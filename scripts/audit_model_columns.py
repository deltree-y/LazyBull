#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型列集审计脚本（薄入口）。

回答两个问题：
  1. **版本漂移**：同一模型家族的 `v{N}_features.json` 之间，列集如何变化
     （哪一列被移除、哪一列反复进出）——列集本身就是模型契约，掉列会让线上推理
     与训练矩阵不再是同一套特征；
  2. **跨来源差异**：部署模型与训练折目录（walk_forward batch 折）的列集是否一致；
     不一致说明折内训练与线上推理存在 schema 偏差。

使用示例：
    python scripts/audit_model_columns.py --last 40
    python scripts/audit_model_columns.py --source deploy=data/models/stock_selection
    python scripts/audit_model_columns.py ^
        --source deploy=data/models/stock_selection ^
        --source fold0=data/walk_forward/batches/<batch>/fold_0 --last 0

本文件为薄入口：CLI 参数解析 + 编排 scripts/model_audit/ 子包；只读特征清单 JSON，
不写回任何模型或特征产物。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger  # noqa: E402

from scripts.model_audit import (  # noqa: E402
    DEFAULT_LAST_VERSIONS,
    DEFAULT_MODEL_DIR,
    build_audit_tables,
    build_markdown,
    iter_source_items,
    parse_source_arg,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模型列集审计（只读）")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=("审计来源，格式 LABEL=PATH（可重复）；缺省使用 " f"deploy={DEFAULT_MODEL_DIR}"),
    )
    parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST_VERSIONS,
        help=f"每个来源只审计末尾 N 个版本（0=全部），默认 {DEFAULT_LAST_VERSIONS}",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="不输出列集存在矩阵（版本多时矩阵较大）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="产物目录（默认 data/reports/model_column_audit/<时间戳>）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_args = args.source or [f"deploy={DEFAULT_MODEL_DIR}"]
    specs = [parse_source_arg(text) for text in source_args]

    items = iter_source_items(specs, last=args.last if args.last > 0 else None)
    tables = build_audit_tables(items, include_matrix=not args.no_matrix)
    report = build_markdown(tables, items, last=args.last)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("data/reports/model_column_audit") / timestamp
    written = write_outputs(out_dir, tables, report)

    logger.info(f"列集审计完成: {len(items)} 个特征清单")
    for item in items:
        logger.info(f"  {item.label}/{item.name}: {len(item.columns)} 列")
    logger.info(f"产物目录: {out_dir}")
    for name, path in written.items():
        logger.info(f"  {name} -> {path}")

    ledger = tables.get("列集漂移台账.csv")
    if ledger is not None:
        changed = ledger[(ledger["相对上一版本新增数"] > 0) | (ledger["相对上一版本移除数"] > 0)]
        logger.info(f"列集发生变化版本数: {len(changed)} / {len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
