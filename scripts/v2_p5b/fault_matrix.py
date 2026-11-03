# -*- coding: utf-8 -*-
"""P5b T3 正式故障注入矩阵薄入口（验收专用，默认只读；只在 ``--root`` 下新建合成 campaign）。

用法::

    python scripts/v2_p5b/fault_matrix.py --root temp/p5b_t3_fault_matrix_20261102
    python scripts/v2_p5b/fault_matrix.py --root temp/p5b_t3_fm --cases T3-FM-04,T3-FM-11

产物：``<root>/audit/fault_matrix.json``（判定表：注入点 × 终态 × 恢复动作 × 五不变量 × 通过/失败，
含 containment 表）；``--markdown`` 另存 Markdown 表格。

边界：本脚本属**阶段验收工具**（按项目约定视作临时开发文件）；合成环境由
``tests/p5b_t3_env.py`` 提供，只读复用 T2 合成输入构造器，不读生产 ``data/paper``、不联网。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.lazybull.v2.hosts.testbed import MATRIX, build_verdict_table, table_markdown  # noqa: E402
from src.lazybull.v2.hosts.testbed.paths import assert_write_scope  # noqa: E402
from src.lazybull.v2.hosts.testbed.storage import write_json_atomic  # noqa: E402

import p5b_t3_env as envmod  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, help="合成 campaign 落点（须不存在或为空目录）")
    parser.add_argument("--cases", default="", help="只跑指定用例（逗号分隔 case_id）")
    parser.add_argument("--markdown", default="", help="另存判定表 Markdown 的路径")
    parser.add_argument("--json", action="store_true", help="打印完整判定表 JSON")
    return parser.parse_args(argv)


def _select_cases(raw: str):
    wanted = [item.strip() for item in raw.split(",") if item.strip()]
    if not wanted:
        return list(MATRIX)
    known = {case.case_id: case for case in MATRIX}
    unknown = [item for item in wanted if item not in known]
    if unknown:
        raise SystemExit(f"未知用例: {unknown}；可选: {sorted(known)}")
    return [known[item] for item in wanted]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    # 生产根硬禁（--root 视为临时脚本外部根，仍不解除生产禁令）
    assert_write_scope(root, allow_external_root=True)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"拒绝非空 root（避免覆盖既有证据）: {root}")
    root.mkdir(parents=True, exist_ok=True)

    results = []
    started = time.perf_counter()
    for case in _select_cases(args.cases):
        case_root = root / case.case_id.lower().replace("-", "_")
        env = envmod.build_env(case_root)
        prepared = envmod.prepare(env)
        result = envmod.run_matrix_case(env, prepared, case)
        results.append(result["record"])
        status = "PASS" if result["record"]["passed"] else "FAIL"
        terminal = result["record"]["terminal"]
        print(f"{case.case_id} {case.fault_point:<24} terminal={terminal:<11} {status}")
    elapsed = time.perf_counter() - started

    table = build_verdict_table(results, campaign_id=root.name)
    table = {
        **table,
        "machine_time": {
            "total_seconds": elapsed,
            "case_count": len(results),
            "long_task_threshold_seconds": 1200.0,
            "long_task_threshold_exceeded": bool(elapsed > 1200.0),
        },
    }
    target = root / "audit" / "fault_matrix.json"
    write_json_atomic(target, table)
    if args.markdown:
        Path(args.markdown).write_text(table_markdown(results), encoding="utf-8")
    print(f"all_passed={table['all_passed']} cases={table['case_count']} seconds={elapsed:.1f}")
    print(f"verdict_table={target.as_posix()}")
    if len(results) != len(MATRIX):
        print(f"note=PARTIAL_MATRIX（{len(results)}/{len(MATRIX)}）；完整判定需跑全量用例")
    if args.json:
        print(json.dumps(table, ensure_ascii=False, indent=2))
    return 0 if table["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
