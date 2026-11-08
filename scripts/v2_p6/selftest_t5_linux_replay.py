# -*- coding: utf-8 -*-
"""P6 T5 跨平台复算包的**本机自检**（独立前缀：参照生成 → 删树 → 按记录路径重建 → 三面比对）。

为什么需要它：``run_t5_linux_replay.py`` 依赖“语料重建在与 Windows 完全相同的仓库相对路径”
（``root_id`` 由相对路径派生）。真实 Linux 执行在本阶段未达成（无合格环境），为了让准备包
不是“纸面命令”，本脚本在同一台机器上用**另一个前缀**跑通同一条闭环：

1. 用假载荷驱动 ``run_t5_linux_replay._prepare_corpora`` 在 ``--work`` 下建语料与前序快照；
2. 按同一协议生成三面参照（与驱动 ``_face_publish_args`` 同源）；
3. **删除整棵工作树**，把参照写进 stub 载荷；
4. 调 ``run_t5_linux_replay.py --windows <stub>`` 让其自行重建并比对，要求三面 MATCH。

产物：``--json`` 时输出一行 JSON（``verdict``/``failed_kinds``/三面 report_id 与 canonical）。
写域：只写 ``--work``（默认 ``temp/p6_t5_linux_selftest``）与同目录 stub 载荷。

退出码：0 = 自检通过（三面 MATCH）；1 = 不符；2 = 设置错误。
"""

# 复杂度豁免：阶段专用验收工具（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.8）；
# 拆分计划：语料/协议全部复用 run_t5_linux_replay 与 run_t5_acceptance，本文件只做“参照生成 +
# stub 组装 + 子进程调用”三段；不承载任何生产逻辑。

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

if __name__ == "__main__":  # 验收进程禁写 __pycache__
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import run_t5_acceptance as t5  # noqa: E402
from scripts.v2_p6 import run_t5_linux_replay as replay  # noqa: E402


class SelftestSetupError(RuntimeError):
    """自检前置条件不满足。"""


#: 唯一允许的工作域根（其**严格子目录**才可被删除/写入）
TEMP_ROOT = (ROOT / "temp").resolve()


def _is_reparse_point(path: Path) -> bool:
    """是否 reparse point / 符号链接 / junction（POSIX 上即 symlink）。"""
    try:
        info = path.lstat()
    except OSError:
        return False
    if path.is_symlink():
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _check_chain_links(start: Path) -> None:
    """沿**原始（未 resolve）**路径自 ``start`` 向上逐级拒绝链接/重解析点。

    评审 R3-T5-R1-01：只检查 `resolve()` 之后的路径会漏掉“别名”本身——别名解析后消失，
    于是“原路径是 symlink/junction”这一事实被抹掉。这里在解析**之前**检查原路径链。
    """
    node = start
    while True:
        if _is_reparse_point(node):
            raise SelftestSetupError(f"--work 原路径含链接/重解析点，拒绝操作：{node}")
        parent = node.parent
        if parent == node:
            return
        node = parent


def safe_work_dir(raw: str | Path) -> Path:
    """把 ``--work`` 归一化，并在**任何删除/写入之前**校验工作域（评审 R3-T5-01/R1-01）。

    规则：①**解析前**先沿原路径链拒绝 symlink/junction/重解析点（别名不得被抹平）；
    ②解析后必须**严格位于** ``<repo>/temp/`` 之下；拒绝 ``temp/`` 本身、仓库根、生产根、
    路径逃逸（``..``/绝对路径）；③再对解析后的链复查一次链接。
    违规即 ``SelftestSetupError``，调用方必须在此之前**不执行任何** ``rmtree``/写入。
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    _check_chain_links(candidate)
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - 解析失败属环境异常
        raise SelftestSetupError(f"--work 无法解析：{raw}（{exc}）") from exc
    if resolved == TEMP_ROOT or TEMP_ROOT not in resolved.parents:
        raise SelftestSetupError(
            f"--work 必须严格位于 {TEMP_ROOT} 之下（拒绝仓库根/生产根/逃逸）：{resolved}"
        )
    for node in (resolved, *resolved.parents):
        if _is_reparse_point(node):
            raise SelftestSetupError(f"--work 路径含重解析点，拒绝操作：{node}")
    return resolved


def safe_stub_write(stub: Path, text: str) -> None:
    """安全写 stub 载荷文件（评审 R3-T5-R1-01 ②）。

    兄弟文件 ``<work>_payload.json`` 不在 ``work`` 目录检查范围内：若它是指向工作域外的
    symlink 或硬链接，直接 ``write_text`` 会改写目标文件。做法：已存在且为链接/硬链接即拒绝；
    其余情况用**同目录临时文件 + ``os.replace``** 原子替换（替换目录项，不沿链接写目标）。
    """
    if stub.exists() or stub.is_symlink():
        info = stub.lstat()
        if stub.is_symlink() or int(getattr(info, "st_nlink", 1)) > 1:
            raise SelftestSetupError(f"stub 载荷已存在且为链接/硬链接，拒绝写入：{stub}")
    tmp = stub.with_name(f"{stub.name}.tmp{os.getpid()}")
    if tmp.exists() or tmp.is_symlink():  # pragma: no cover - 同进程重入保护
        raise SelftestSetupError(f"临时文件已存在，拒绝覆盖：{tmp}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, stub)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _fake_payload(work_rel: str) -> dict[str, Any]:
    """只带 ``faces.*.paths`` 的假载荷（驱动 ``_prepare_corpora`` 复原路径用）。"""
    return {
        "checks": {
            "three_faces": {
                "faces": {
                    "dashboard": {
                        "standard": {
                            "paths": {
                                "paper": f"{work_rel}/dashboard_standard/corpus/paper",
                                "quality": f"{work_rel}/dashboard_standard/corpus/quality",
                            }
                        }
                    },
                    "experiment": {
                        "standard": {
                            "paths": {
                                "base": f"{work_rel}/experiment_standard/corpus/"
                                f"{replay.PROFILE}/runs/base",
                                "arm": f"{work_rel}/experiment_standard/corpus/"
                                f"{replay.PROFILE}/runs/arm",
                            }
                        }
                    },
                    "trend": {
                        "standard": {
                            "paths": {
                                "new_run": f"{work_rel}/trend_standard/corpus_standard/runs/x"
                            },
                            "previous": {
                                "run_dir": f"{work_rel}/trend_standard/reports_previous_"
                                f"{replay.PROFILE}/research/trend/placeholder"
                            },
                        }
                    },
                }
            }
        }
    }


def _fill_paths(payload: dict[str, Any], prepared: dict[str, Any], work_rel: str) -> None:
    faces = payload["checks"]["three_faces"]["faces"]
    faces["dashboard"]["standard"]["paths"] = {
        key: _rel(value) for key, value in prepared["paths"]["dashboard"].items()
    }
    faces["experiment"]["standard"]["paths"] = {
        key: _rel(value) for key, value in prepared["paths"]["experiment"].items()
    }
    faces["trend"]["standard"]["paths"]["new_run"] = _rel(prepared["paths"]["trend"]["new_run"])
    # 上一快照落点根必须与 ``--work`` 结构一致（id 由协议决定，不预写）
    faces["trend"]["standard"]["previous"][
        "run_dir"
    ] = f"{work_rel}/trend_standard/reports_previous_{replay.PROFILE}/research/trend/unknown"


def selftest(work: Path) -> dict[str, Any]:
    """跑完整闭环并返回结构化结果（工作域校验先于任何删除/写入）。"""
    work = safe_work_dir(work)
    if work.exists():
        shutil.rmtree(work)
    work_rel = _rel(work)
    payload = _fake_payload(work_rel)
    prepared = replay._prepare_corpora(payload)
    _fill_paths(payload, prepared, work_rel)

    windows: dict[str, Any] = {}
    for kind in t5.KINDS:
        row = replay._replay_one(kind, prepared["paths"][kind], work / "ref_out")
        windows[kind] = {
            "files": row["files"],
            "canonical_sha256": row["canonical_sha256"],
            "report_id": row["report_id"],
            "exit_code": row["exit_code"],
        }
        if row["exit_code"] != 0:
            return {
                "schema": "p6_t5_linux_selftest/v1",
                "verdict": "fail",
                "stage": "reference",
                "kind": kind,
                "stderr_tail": row["stderr_tail"],
            }
    payload["checks"]["dual_platform"] = {"windows": windows}

    shutil.rmtree(work)
    stub = work.parent / f"{work.name}_payload.json"
    safe_stub_write(stub, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if not work.exists():
        pass  # 工作树已删除：replay 必须自行重建
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "scripts/v2_p6/run_t5_linux_replay.py",
            "--windows",
            _rel(stub),
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload_out: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            payload_out = json.loads(result.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload_out = {}
    rows = {
        kind: {
            "passed": bool((payload_out.get("kinds") or {}).get(kind, {}).get("passed")),
            "report_id": (payload_out.get("kinds") or {}).get(kind, {}).get("report_id"),
            "canonical_sha256": (payload_out.get("kinds") or {})
            .get(kind, {})
            .get("canonical_sha256"),
        }
        for kind in t5.KINDS
    }
    return {
        "schema": "p6_t5_linux_selftest/v1",
        "work": work_rel,
        "replay_exit_code": result.returncode,
        "reference_ids": {kind: windows[kind].get("report_id") for kind in t5.KINDS},
        "kinds": rows,
        "failed_kinds": sorted(kind for kind, row in rows.items() if not row["passed"]),
        "stderr_tail": [line for line in result.stderr.splitlines() if line.strip()][-5:],
        "verdict": (
            "pass" if result.returncode == 0 and all(r["passed"] for r in rows.values()) else "fail"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T5 跨平台复算包本机自检")
    parser.add_argument(
        "--work", default="temp/p6_t5_linux_selftest", help="自检工作树（须可覆盖）"
    )
    parser.add_argument("--json", action="store_true", help="打印 JSON 结果")
    args = parser.parse_args(argv)
    try:
        result = selftest(ROOT / args.work)
    except SelftestSetupError as exc:
        print(f"[P6-T5] 自检未执行：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        for kind, row in result.get("kinds", {}).items():
            print(f"{kind}: {'MATCH' if row['passed'] else 'MISMATCH'} ({row['report_id']})")
        print(f"verdict={result['verdict']}")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
