# -*- coding: utf-8 -*-
"""验收门 2（代码态预检）：``git diff b9d866e..HEAD`` 限重放执行闭包逐文件核对。

规划 §5 门 2（B4 处置）：重放执行闭包 = backtest/ trading/ signals/
portfolio/ common/ execution/ universe/ data/ + ml/model_registry +
ml/train_core（ml_signal 只读依赖侧）。闭包内文件相对冻结提交 b9d866e
的任何改动都必须落在**冻结白名单**内：

- 白名单外文件有 diff ⇒ 失败（非零退出）；
- 白名单内文件输出完整 diff 供人工确认（当前已核：仅 trading_config help
  文本与 downside_penalty docstring 改动，均不影响重放行为）。

白名单为内嵌常量（冻结登记，改动需显式编辑本文件并登记）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 冻结基点提交（B0/B1 双臂冻结时的代码态，baseline_freeze §4）
FROZEN_COMMIT = "b9d866e"

#: 重放执行闭包路径前缀（git pathspec，相对仓库根）
CLOSURE_PATHS: List[str] = [
    "src/lazybull/backtest",
    "src/lazybull/trading",
    "src/lazybull/signals",
    "src/lazybull/portfolio",
    "src/lazybull/common",
    "src/lazybull/execution",
    "src/lazybull/universe",
    "src/lazybull/data",
    "src/lazybull/ml/model_registry.py",
    "src/lazybull/ml/train_core",
]

#: 冻结白名单（闭包内已核对的无行为改动文件；逐文件附改动性质登记）
#: T7 实跑扩充（2026-10-08）：data/*_raw 四件 + universe/domains 为 docs/
#: 目录重组后的 docstring 引用路径同步（diff 实测仅注释行，无行为改动）。
DIFF_WHITELIST: Dict[str, str] = {
    "src/lazybull/common/trading_config.py": "help 文本变更（无行为改动，已核）",
    "src/lazybull/signals/downside_penalty.py": "docstring 变更（退役登记，无行为改动，已核）",
    "src/lazybull/data/holdertrade_raw.py": "docstring 文档路径同步（无行为改动，T7 实跑扩充）",
    "src/lazybull/data/repurchase_raw.py": "docstring 文档路径同步（无行为改动，T7 实跑扩充）",
    "src/lazybull/data/top10_floatholders_raw.py": "docstring 文档路径同步（无行为改动，T7 实跑扩充）",
    "src/lazybull/data/top_inst_raw.py": "docstring 文档路径同步（无行为改动，T7 实跑扩充）",
    "src/lazybull/universe/domains.py": "docstring 文档路径同步（无行为改动，T7 实跑扩充）",
}


def _git_diff(args: List[str]) -> str:
    result = subprocess.run(
        ["git", "diff", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff 失败: {result.stderr.strip()}")
    return result.stdout


def check_closure_diff(
    frozen_commit: str = FROZEN_COMMIT,
) -> Dict[str, object]:
    """执行门 2 核对。

    Returns:
        {"ok": bool, "violations": [白名单外有 diff 的文件],
         "whitelisted_diffs": {白名单内文件: 完整 diff 文本},
         "unseen_whitelist": [白名单内但无 diff 的登记项]}
    """
    name_only = _git_diff([f"{frozen_commit}..HEAD", "--name-only", "--", *CLOSURE_PATHS])
    changed = [line.strip() for line in name_only.splitlines() if line.strip()]
    violations = sorted(f for f in changed if f not in DIFF_WHITELIST)
    whitelisted_diffs: Dict[str, str] = {}
    for path in sorted(f for f in changed if f in DIFF_WHITELIST):
        whitelisted_diffs[path] = _git_diff([f"{frozen_commit}..HEAD", "--", path])
    return {
        "ok": not violations,
        "violations": violations,
        "whitelisted_diffs": whitelisted_diffs,
        "unseen_whitelist": sorted(set(DIFF_WHITELIST) - set(whitelisted_diffs)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="验收门 2：重放执行闭包代码态预检")
    parser.add_argument("--frozen-commit", default=FROZEN_COMMIT)
    args = parser.parse_args(argv)

    report = check_closure_diff(args.frozen_commit)
    for path, diff_text in report["whitelisted_diffs"].items():
        logger.info(f"白名单内改动（{DIFF_WHITELIST[path]}）: {path}\n{diff_text}")
    for path in report["unseen_whitelist"]:
        logger.info(f"白名单登记项当前无 diff（保留登记）: {path}")
    if not report["ok"]:
        logger.error("白名单外文件存在 diff（门 2 失败）:")
        for path in report["violations"]:
            logger.error(f"  {path}")
        return 1
    logger.info("门 2 通过：闭包内无白名单外改动")
    return 0


if __name__ == "__main__":
    sys.exit(main())
