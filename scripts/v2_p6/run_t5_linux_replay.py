# -*- coding: utf-8 -*-
"""P6 T5 跨平台复算包（自包含：按载荷记录的**同一仓库相对路径**重建语料 → 三面复算 → 逐字节比对）。

本脚本是 ``dual_platform.linux_commands`` 的唯一执行入口（一条命令跑完）：

1. 从 Windows 侧 ``t5_acceptance.json`` 读三面的 ``faces.<kind>.<profile>.paths``（仓库相对
   POSIX），在**同一相对路径**重建语料与 trend 上一快照——``root_id`` 由来源根的仓库相对路径
   派生（README「身份」条），路径不同则 trend 的每行 ``run_key``/``source`` 都会变，
   所以跨机对照必须“同路径重建”，不能换 ``--work`` 前缀；
2. 按 ``run_t5_acceptance._face_publish_args`` 的同一协议（各面冻结 ``generated_at``）复算三面，
   输出到 ``<work>/<kind>_replay``（输出根不进字节，不影响比对）；
3. 与载荷 ``checks.dual_platform.windows`` 逐文件 SHA 与 canonical 摘要比对，三面 MATCH 才是
   ``verdict=pass``。

写域：只写载荷已记录的工作树（``temp/...``）与输出根，不写任何生产根、不写 ``data/reports``；
目标工作树**已存在即拒绝**（在 Windows 上二次执行会 fail-closed，这是预期行为）。

用法（干净 Linux 机器）::

    python scripts/v2_p6/run_t5_linux_replay.py \
        --windows data/reports/p6/validations/research/<id>/t5_acceptance.json --json

退出码：0 = 三面全 MATCH；1 = 任一面不符；2 = 设置错误（工作树已存在/载荷缺字段等）。
"""

# 复杂度豁免：阶段专用验收工具（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.8）；
# 拆分计划：语料构建复用 build_t2/t3/t4_corpus，协议复用 run_t5_acceptance._face_publish_args，
# 本文件只保留“路径复原 + 编排 + 比对”三段；超过 800 行时再拆 compare 子模块。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __name__ == "__main__":  # 验收进程禁写 __pycache__
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import (  # noqa: E402
    build_t2_corpus,
    build_t3_corpus,
    build_t4_corpus,
    run_t4_acceptance,
    run_t5_acceptance,
)

REPORT_SCRIPT = Path("scripts/p6_report.py")
PROFILE = "standard"


class ReplaySetupError(RuntimeError):
    """复算前置条件不满足（工作树已存在、载荷缺字段等）。"""


def _run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    return run_t5_acceptance._run(command, cwd=cwd)


def _file_hashes(run_dir: Path) -> dict[str, str]:
    return dict(run_t5_acceptance._file_hashes(run_dir))


def _canonical_of(files: Mapping[str, str]) -> str:
    return run_t5_acceptance._canonical_of(files)


def _face_block(payload: Mapping[str, Any], kind: str) -> dict[str, Any]:
    faces = dict((payload.get("checks") or {}).get("three_faces") or {}).get("faces") or {}
    row = dict((faces.get(kind) or {}).get(PROFILE) or {})
    if not row or not row.get("paths"):
        raise ReplaySetupError(f"载荷缺 faces.{kind}.{PROFILE}.paths（无法复原语料路径）")
    return row


def _windows_reference(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Windows 载荷里的三面 canonical 参照（缺即设置错误，不得静默跳过）。"""
    block = dict((payload.get("checks") or {}).get("dual_platform") or {})
    windows = dict(block.get("windows") or {})
    missing = [kind for kind in run_t5_acceptance.KINDS if not windows.get(kind)]
    if missing:
        raise ReplaySetupError(f"Windows 载荷缺 dual_platform.windows：{missing}")
    return windows


def _abs(relative: str) -> Path:
    return ROOT / str(relative)


def _prepare_corpora(payload: Mapping[str, Any]) -> dict[str, Any]:
    """在载荷记录的同一相对路径重建三面语料（工作树必须不存在）。"""
    dashboard = _face_block(payload, "dashboard")
    experiment = _face_block(payload, "experiment")
    trend = _face_block(payload, "trend")

    t2_corpus = _abs(dashboard["paths"]["quality"]).parent
    t3_root = _abs(experiment["paths"]["base"]).parents[2]
    t4_corpus = _abs(trend["paths"]["new_run"]).parents[1]
    t4_work = t4_corpus.parent
    existing = [path for path in (t2_corpus, t3_root, t4_work) if path.exists()]
    if existing:
        raise ReplaySetupError(
            "目标工作树已存在（跨平台复算必须在干净机器上按记录路径重建）："
            + ", ".join(path.relative_to(ROOT).as_posix() for path in existing)
        )

    case = build_t2_corpus.build_case(t2_corpus, PROFILE)
    build_t3_corpus.build(t3_root, profile=PROFILE)
    roots = run_t4_acceptance._profile_roots(t4_work, PROFILE)
    previous = run_t4_acceptance.publish_previous(roots, PROFILE)
    if not previous.get("passed"):
        raise ReplaySetupError(f"trend 上一快照发布失败：{previous.get('checks')}")
    build_t4_corpus.build(roots["corpus"], PROFILE)
    previous_dir = Path(str(previous["run_dir"]))
    # 只校验**落点根**（上一快照 run_id 由输入+stamp 决定，载荷里的 id 是同一协议的产物；
    # 若语料/路径有偏差，后续三面比对必然不符，故不在此做 id 硬等值）
    expected_root = _abs(trend["previous"]["run_dir"]).parents[2]
    if previous_dir.parents[2] != expected_root:
        raise ReplaySetupError(
            f"上一快照落点根与载荷不一致：{previous_dir.parents[2]} != {expected_root}"
        )
    new_run = run_t4_acceptance._new_run_dir(roots["corpus"], PROFILE)
    expected_new_run = _abs(trend["paths"]["new_run"])
    # 只校验父目录（``runs/``）：run 名由 ``PROFILE_PLAN['runs_previous']`` 决定，若与载荷不同，
    # 复算会以实际 run 为来源、三面比对必然不符，无需在此做叶子名硬等值
    if new_run.parent != expected_new_run.parent:
        raise ReplaySetupError(
            f"新 run 落点目录与载荷不一致：{new_run.parent} != {expected_new_run.parent}"
        )

    return {
        "work": t4_work,
        "paths": {
            "dashboard": {
                "paper": Path(str(case["paper"])),
                "quality": Path(str(case["quality_root"])),
            },
            "experiment": {
                "base": t3_root / PROFILE / "runs" / "base",
                "arm": t3_root / PROFILE / "runs" / "arm",
            },
            "trend": {"new_run": new_run, "previous": previous_dir},
        },
    }


def _replay_one(kind: str, paths: Mapping[str, Any], out_root: Path) -> dict[str, Any]:
    """按 T5 同一协议复算一面，返回重建目录与文件哈希。"""
    args = run_t5_acceptance._face_publish_args(kind, PROFILE, paths)
    target = out_root / kind
    command = [
        sys.executable,
        str(REPORT_SCRIPT),
        "generate",
        *[str(item) for item in args],
        "--reports-root",
        run_t5_acceptance._rel_text(target),
        "--attempt-id",
        f"linux_replay_{kind}",
        "--json",
    ]
    result = _run(command)
    payload = run_t5_acceptance._last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    run_dir = Path(str(payload["report_path"])) if payload.get("report_path") else None
    files = _file_hashes(run_dir) if run_dir is not None else {}
    return {
        "kind": kind,
        "command": run_t5_acceptance._posix_command(run_t5_acceptance._relativize(command)),
        "exit_code": result["exit_code"],
        "verdict": payload.get("verdict"),
        "report_id": payload.get("report_id"),
        "run_dir": run_dir.as_posix() if run_dir is not None else None,
        "files": files,
        "canonical_sha256": _canonical_of(files) if files else None,
        "stderr_tail": result["stderr_tail"],
    }


def replay(payload: Mapping[str, Any]) -> dict[str, Any]:
    """同路径重建语料 → 三面复算 → 与 Windows 参照逐字节比对。"""
    reference = _windows_reference(payload)
    prepared = _prepare_corpora(payload)
    out_root = Path(prepared["work"]) / "replay_out"
    rows: dict[str, Any] = {}
    for kind in run_t5_acceptance.KINDS:
        row = _replay_one(kind, prepared["paths"][kind], out_root)
        expected_files = dict((reference.get(kind) or {}).get("files") or {})
        expected_canonical = (reference.get(kind) or {}).get("canonical_sha256")
        row["expected_files"] = expected_files
        row["expected_canonical_sha256"] = expected_canonical
        row["files_byte_identical"] = bool(expected_files) and row["files"] == expected_files
        row["canonical_identical"] = bool(expected_canonical) and (
            row["canonical_sha256"] == expected_canonical
        )
        row["passed"] = bool(
            row["exit_code"] == 0
            and row["verdict"] == "pass"
            and row["files_byte_identical"]
            and row["canonical_identical"]
        )
        rows[kind] = row
    passed = all(item["passed"] for item in rows.values())
    return {
        "schema": "p6_t5_linux_replay/v1",
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "work": Path(prepared["work"]).relative_to(ROOT).as_posix(),
        "generated_at_by_kind": dict(run_t5_acceptance.FACE_GENERATED_AT),
        "kinds": rows,
        "verdict": "pass" if passed else "fail",
        "failed_kinds": sorted(kind for kind, item in rows.items() if not item["passed"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T5 跨平台复算（同路径语料重建 + 三面比对）")
    parser.add_argument("--windows", required=True, help="Windows 侧 t5_acceptance.json")
    parser.add_argument("--json", action="store_true", help="打印 JSON 结果")
    args = parser.parse_args(argv)
    payload = run_t5_acceptance._read_json(ROOT / args.windows)
    if not isinstance(payload, Mapping):
        print(f"[P6-T5] Windows 载荷不可解析：{args.windows}", file=sys.stderr)
        return 2
    try:
        result = replay(payload)
    except ReplaySetupError as exc:
        print(f"[P6-T5] 复算未执行：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        for kind, row in result["kinds"].items():
            print(
                f"{kind}: {'MATCH' if row['passed'] else 'MISMATCH'} "
                f"(exit={row['exit_code']}, canonical={row['canonical_sha256']})"
            )
        print(f"verdict={result['verdict']}")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
