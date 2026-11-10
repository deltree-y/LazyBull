# -*- coding: utf-8 -*-
"""P6.5 T4 验收驱动：三件套（S1/S4/自拆一致）+ G4 位模式门证据包。

一次运行产出 ``data/frozen_reference/v2_p65/<run_id>/``：

1. ``zero_turnover.json``（S1 零换手：物理成交/费用恒 0，贡献 NAV 和 1.048）；
2. ``allocation_conservation.json``（S4 分摊守恒：Q18 股数 + Q13 费用瀑布）；
3. ``self_split.json``（自拆双臂：黄金值、generation 身份、G4 位模式门）；
4. ``self_split_diff.json``（逐位比较：差异清单、文件状态、规范化 SHA-256、排除表）；
5. ``normalized_sha256.json`` + ``normalized/``（两臂 runs 规范化字节**本体**与 SHA-256，
   每条摘要都能定位本体并重算校验）；
6. ``arms/``（两臂 runs_root 与根外 generation manifest；**不写 data/runs/**）；
7. ``manifest.json``（HEAD/脏标记、输入、配置、费率、容差、排除表、runs writer 身份、
   文件哈希、命令）。

边界：本驱动只产出**合成对账边界内**的候选门证据，不宣称 P6.5 出口、B 区转正或
生产多袖能力；已存在的证据目录一律拒绝（无覆盖）。
"""

# 复杂度豁免：阶段专用验收驱动（scripts/v2_p65，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.lazybull.common.config import get_logs_dir  # noqa: E402
from src.lazybull.v2.core.accounting.g4_nav import G4NavMismatchError  # noqa: E402
from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError  # noqa: E402
from src.lazybull.v2.evidence.arm_comparator import (  # noqa: E402
    ArmComparisonError,
    ComparisonInputError,
)
from src.lazybull.v2.evidence.p65_t4_evidence import (  # noqa: E402
    EVIDENCE_ROOT,
    EvidenceVerdictError,
    P65EvidenceError,
    collect_t4_evidence,
    default_run_id,
)
from src.lazybull.v2.hosts.order_chain import OrderChainError  # noqa: E402
from src.lazybull.v2.hosts.self_split_arms import SelfSplitArmError  # noqa: E402
from src.lazybull.v2.hosts.virtual_arm_runs import ArmIsolationError, require_ascii_id  # noqa: E402

#: 退出码约定（0 = 全部候选门通过；3 = 证据/断言失败；2 = 前置条件不满足，含输入/链层拒绝）
EXIT_OK = 0
EXIT_FAIL = 3
EXIT_SETUP_ERROR = 2


def _git_state(repo_root: Path) -> dict:
    """读取 HEAD 与脏标记（不改 git 配置、不做写操作）。"""
    head = subprocess.run(  # noqa: S603 - 固定参数、只读
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(  # noqa: S603 - 固定参数、只读
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_commit": head.stdout.strip() or "<unknown>",
        "git_dirty": bool(status.stdout.strip()),
        "git_head_ok": head.returncode == 0,
    }


def _configure_log(run_id: str) -> Path:
    """把运行日志写到 ``logs/``（项目唯一临时日志目录；文件不带 ANSI 转义）。"""
    log_path = Path(get_logs_dir()) / f"p65_t4_acceptance_{run_id}.log"
    logger.add(
        str(log_path),
        encoding="utf-8",
        level="INFO",
        enqueue=False,
        colorize=False,
        format="{time:YYYY-MM-DDTHH:mm:ss} | {level} | {message}",
    )
    return log_path


#: 参与 run_id 内容指纹的 T4 核心文件（内容变了 run_id 就变，禁止跨版本复用证据名）
_CONTENT_FILES = (
    "src/lazybull/v2/core/decision/target_synthesis.py",
    "src/lazybull/v2/core/accounting/g4_nav.py",
    "src/lazybull/v2/hosts/self_split_arms.py",
    "src/lazybull/v2/evidence/arm_comparator.py",
    "src/lazybull/v2/evidence/p65_t4_evidence.py",
)


def _content_salt() -> str:
    """短内容指纹（阶段内核心文件哈希 + 脏标记；不含主机名/墙钟）。"""
    import hashlib

    digest = hashlib.sha256()
    for relative in _CONTENT_FILES:
        path = ROOT / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
        else:  # pragma: no cover - 文件缺失即环境异常
            digest.update(f"{relative}:missing".encode("utf-8"))
    return digest.hexdigest()[:8]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6.5 T4 三件套 + G4 候选门证据驱动")
    parser.add_argument(
        "--root", default=str(EVIDENCE_ROOT), help="证据根（默认 data/frozen_reference/v2_p65）"
    )
    parser.add_argument("--run-id", default="", help="ASCII 运行标识（缺省自动生成；已存在即拒绝）")
    parser.add_argument("--date", default="", help="run_id 日期段（YYYYMMDD；缺省今天）")
    args = parser.parse_args(argv)

    try:
        run_id = require_ascii_id(
            args.run_id or default_run_id(date=args.date or None, salt=_content_salt()),
            "run_id",
        )
    except ValueError as error:
        logger.error(f"run_id 非法（拒绝在证据根之外创建目录）：{error}")
        return EXIT_SETUP_ERROR
    log_path = _configure_log(run_id)
    command = f"python scripts/v2_p65/run_t4_acceptance.py --root {args.root} --run-id {run_id}"
    logger.info(f"P6.5 T4 证据驱动启动：run_id={run_id} 证据根={args.root} 日志={log_path}")

    try:
        bundle = collect_t4_evidence(
            root=Path(args.root),
            run_id=run_id,
            code_state=_git_state(ROOT),
            command=command,
            scenario_notes={
                "boundary": "合成对账边界内的候选门结果；不构成 P6.5 阶段出口或生产多袖能力",
                "three_piece": [
                    "S1_zero_turnover",
                    "S4_allocation_conservation",
                    "self_split_consistency",
                ],
                "writer": "复用 hosts/backtest/runs_writer.py::write_runs_batch；不写 data/runs/",
                "acceptance_log": str(log_path).replace("\\", "/"),
            },
        )
    except EvidenceVerdictError as error:
        logger.error(f"候选门未通过（证据已落盘，差异清单见 {error.root}）：{error.verdict}")
        return EXIT_FAIL
    except (
        P65EvidenceError,
        ArmIsolationError,
        SelfSplitArmError,
        G4NavMismatchError,
        ArmComparisonError,
        ComparisonInputError,
        OrderChainError,
        ReconciliationError,
    ) as error:
        logger.error(
            f"证据装配拒绝（零覆盖，本次创建的目录已清理）：{type(error).__name__}: {error}"
        )
        return EXIT_SETUP_ERROR

    verdict = dict(bundle.verdict)
    logger.info(f"证据产物根={bundle.root}（{len(bundle.files)} 个文件）")
    for key in ("zero_turnover", "allocation_conservation", "self_split", "self_split_bitwise"):
        logger.info(f"候选门 {key} = {verdict.get(key)}")
    logger.info(f"未比较项（不得写成逐位一致）：{verdict.get('not_compared')}")
    logger.info(f"边界：{verdict.get('boundary')}")

    passed = all(
        verdict.get(key) == "PASS"
        for key in ("zero_turnover", "allocation_conservation", "self_split", "self_split_bitwise")
    )
    if not passed:
        logger.error(f"候选门未全部通过：{verdict}")
        return EXIT_FAIL
    logger.info("三件套与 G4 位模式门在合成对账边界内全部通过（不外推阶段出口）")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
