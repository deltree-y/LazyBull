# -*- coding: utf-8 -*-
"""P6.5 T5 验收驱动：G5~G7 证据包（恢复矩阵 + 同代信封本体 + 回归/复杂度 + manifest）。

一次运行产出 ``data/frozen_reference/v2_p65/<run_id>/``：

1. ``recovery_matrix.json``（五类恢复场景 + 无中断参照，逐字段位级结论）；
2. ``state_envelopes/``（参照链**两代及以上**信封本体 + ``index.json`` 逐文件 SHA 与
   上一代摘要衔接）；
3. ``pytest_targeted.txt`` / ``pytest_full.txt`` / ``complexity.json`` / ``flake8.txt``
   （回归与静态门；全量缺省写"待跑声明"，**不冒充已运行**）；
4. ``envelope_chains/``（每条链的完整信封目录，可独立复核）；
5. ``manifest.json``（HEAD/脏标记、输入、费率、容差、排除表、runs writer 身份、
   信封版本与提交点链、恢复场景清单、文件哈希、命令）。

边界：本驱动只产出**合成对账边界内**的候选门证据；不宣称 P6.5 出口、G5~G7 阶段
验收通过、B 区转正或生产多袖能力；已存在的证据目录一律拒绝（无覆盖）。
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
from src.lazybull.v2.core.accounting.envelope_store import EnvelopeStoreError  # noqa: E402
from src.lazybull.v2.core.accounting.state_envelope import (  # noqa: E402
    StateEnvelopeCorruptError,
    StateEnvelopeError,
    StateEnvelopeIntegrityError,
)
from src.lazybull.v2.core.accounting.state_owners import StateOwnerError  # noqa: E402
from src.lazybull.v2.evidence.p65_t5_evidence import (  # noqa: E402
    CODE_SURFACE,
    EVIDENCE_ROOT,
    P65T5EvidenceError,
    T5EvidenceVerdictError,
    collect_t5_evidence,
    content_salt as _content_salt,
    default_run_id,
    reproduce_command as _reproduce_command,
)
from src.lazybull.v2.hosts.recovery_runtime import (  # noqa: E402
    RecoveryError,
    RecoveryIntegrityError,
)
from src.lazybull.v2.hosts.virtual_arm_runs import require_ascii_id  # noqa: E402

#: 退出码约定（0 = G5 候选门通过；3 = 证据已落盘但判定 FAIL；2 = 前置条件不满足）
EXIT_OK = 0
EXIT_FAIL = 3
EXIT_SETUP_ERROR = 2

#: 参与 run_id 内容指纹的 T5 面：与 manifest ``code_surface`` **共用同一清单**
#: （`p65_t5_evidence.CODE_SURFACE`，8 个新增交付模块 + 5 个被修改模块）。
_CONTENT_FILES = CODE_SURFACE


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
    log_path = Path(get_logs_dir()) / f"p65_t5_acceptance_{run_id}.log"
    logger.add(
        str(log_path),
        encoding="utf-8",
        level="INFO",
        enqueue=False,
        colorize=False,
        format="{time:YYYY-MM-DDTHH:mm:ss} | {level} | {message}",
    )
    return log_path


def content_salt(files: tuple[str, ...] = _CONTENT_FILES) -> str:
    """短内容指纹（委托证据层唯一实现 `p65_t5_evidence.content_salt`）。"""
    return _content_salt(files, root=ROOT)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """命令行参数（路径缺省指向 ``logs/`` 下的回归日志）。"""
    parser = argparse.ArgumentParser(description="P6.5 T5 恢复矩阵 + G5~G7 证据驱动")
    parser.add_argument(
        "--root", default=str(EVIDENCE_ROOT), help="证据根（默认 data/frozen_reference/v2_p65）"
    )
    parser.add_argument("--run-id", default="", help="ASCII 运行标识（缺省自动生成；已存在即拒绝）")
    parser.add_argument("--date", default="", help="run_id 日期段（YYYYMMDD；缺省今天）")
    parser.add_argument(
        "--pytest-targeted",
        default=str(Path(get_logs_dir()) / "p65_t5_targeted.log"),
        help="定向回归日志",
    )
    parser.add_argument(
        "--pytest-full", default="", help="全量回归日志（缺省视为复审态，写待跑声明）"
    )
    parser.add_argument(
        "--complexity",
        default=str(Path(get_logs_dir()) / "p65_t5_complexity_src_scripts.log"),
        help="复杂度 `src+scripts` 口径日志",
    )
    parser.add_argument(
        "--complexity-src",
        default="",
        help="复杂度 `src` 口径日志（缺省取同目录 p65_t5_complexity_src.log）",
    )
    parser.add_argument("--flake8", default="", help="flake8 日志（可缺省）")
    return parser.parse_args(argv)


def reproduce_command(args: argparse.Namespace, run_id: str) -> str:
    """按**实际参数**拼接可逐字复跑的命令（委托证据层唯一实现）。"""
    return _reproduce_command(
        root=args.root,
        run_id=run_id,
        pytest_targeted=args.pytest_targeted,
        complexity=args.complexity,
        complexity_src=complexity_src_of(args),
        pytest_full=args.pytest_full,
        flake8=args.flake8,
    )


def complexity_src_of(args: argparse.Namespace) -> str:
    """``src`` 口径复杂度日志路径（缺省取 ``--complexity`` 同目录的 ``..._src.log``）。"""
    if args.complexity_src:
        return args.complexity_src
    return str(Path(args.complexity).with_name("p65_t5_complexity_src.log"))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_id = require_ascii_id(
            args.run_id
            or default_run_id(date=args.date or None, salt=content_salt()),
            "run_id",
        )
    except (ValueError, FileNotFoundError) as error:
        logger.error(f"run_id 非法或内容指纹不可用（拒绝创建证据目录）：{error}")
        return EXIT_SETUP_ERROR
    log_path = _configure_log(run_id)
    command = reproduce_command(args, run_id)
    logger.info(f"P6.5 T5 证据驱动启动：run_id={run_id} 证据根={args.root} 日志={log_path}")
    try:
        bundle = collect_t5_evidence(
            root=Path(args.root),
            run_id=run_id,
            code_state=_git_state(ROOT),
            command=command,
            pytest_targeted=Path(args.pytest_targeted),
            complexity_log=Path(args.complexity),
            complexity_src_log=Path(complexity_src_of(args)),
            pytest_full=Path(args.pytest_full) if args.pytest_full else None,
            flake8_log=Path(args.flake8) if args.flake8 else None,
            scenario_notes={
                "boundary": "合成对账边界内的候选门结果；不构成 P6.5 阶段出口或生产多袖能力",
                "gates": ["G5_recovery_matrix", "G6_regression", "G7_evidence_identity"],
                "writer": "T5 不写 runs；writer 身份只读登记",
                "acceptance_log": str(log_path).replace("\\", "/"),
            },
        )
    except T5EvidenceVerdictError as error:
        logger.error(f"G5 候选门未通过（证据已落盘，差异清单见 {error.root}）：{error.verdict}")
        return EXIT_FAIL
    except (
        P65T5EvidenceError,
        StateOwnerError,
        StateEnvelopeError,
        StateEnvelopeIntegrityError,
        StateEnvelopeCorruptError,
        EnvelopeStoreError,
        RecoveryError,
        RecoveryIntegrityError,
    ) as error:
        logger.error(
            f"证据装配拒绝（零覆盖，本次创建的目录已清理）：{type(error).__name__}: {error}"
        )
        return EXIT_SETUP_ERROR

    verdict = dict(bundle.verdict)
    logger.info(f"证据产物根={bundle.root}（{len(bundle.files)} 个文件）")
    logger.info(f"G5 恢复矩阵 = {verdict.get('recovery_matrix')}（{verdict.get('case_count')} 场景）")
    logger.info(f"信封代数 = {verdict.get('envelope_generations')}；全量回归 = {verdict.get('pytest_full')}")
    if not bundle.complete:
        logger.warning(
            "证据包为**复审态**（全量 pytest 未运行，按 CLAUDE.md §7.6 于复审通过后执行）；"
            "收口时必须用真实全量日志重新生成证据包"
        )
    logger.info(f"边界：{verdict.get('boundary')}")
    if verdict.get("recovery_matrix") != "PASS":
        logger.error(f"G5 候选门未通过：{verdict}")
        return EXIT_FAIL
    logger.info("G5 恢复矩阵在合成对账边界内全部逐位一致（不外推阶段出口）")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
