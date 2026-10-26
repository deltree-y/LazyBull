# -*- coding: utf-8 -*-
"""P3 T5 terminal_loss 家族处置清单核验（P2-6；只读核验，不改任何被验对象）。

核验对象 = ``docs/reports/terminal_loss_disposal_20261023.md``（处置档案，gitignore
内的本地文档）所列资产与引用面实测值：

- §2 家族代码 12 个 .py 文件存在且行数合计 == 6,136（行数口径 = ``wc -l``，
  即字节 ``\\n`` 计数；逐文件行数一并锁定）；
- §4 训练/分析链路 12 个脚本存在（家族专用 9 + 主链路/通用 3）；
- §5.2 tests 引用面 19 个 test_*.py 存在（+ P2a 在用闭包基线 fixture 1 个）；
- §5 引用计数复算（按行口径，pattern ``terminal_loss|terminal_risk``，排除
  ``__pycache__`` 与本核验测试自身——本文件晚于实测日存在）：src 39 文件 /
  85 行（家族自身 46 行 + 家族外 39 行）、tests 19 个 test_*.py、scripts
  12 个（仅 .py/.ps1 脚本文件入计——``scripts/check/complexity_baseline.json``
  等数据产物非脚本，不在文档口径内）。

**断言锁定 2026-10-23 实测值**（文档 §5.3：规划期旧口径已过时，以清单实测为准）。
处置文档为 append-only——未来追加复审注记不改清单数值，不破本测试；家族文件
在观察期内行为冻结（§9.1），行数变动须先过评审再同步本表。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "reports" / "terminal_loss_disposal_20261023.md"

#: §2 家族代码行数表（2026-10-23 实测，wc -l 口径）
FAMILY_DIR = Path("src/lazybull/risk/terminal_loss")
FAMILY_FILE_LINES = {
    "__init__.py": 200,
    "artifacts.py": 240,
    "block_stats.py": 511,
    "coverage_audit.py": 409,
    "dataset.py": 665,
    "exposure_gate.py": 900,
    "exposure_online.py": 923,
    "labels.py": 314,
    "model.py": 183,
    "mother_section.py": 627,
    "policy_sidecar.py": 634,
    "train.py": 530,
}
FAMILY_TOTAL_LINES = 6136

#: §4 训练/分析链路脚本（9 家族专用 + 3 主链路/通用）
DISPOSAL_SCRIPTS = (
    "scripts/train_terminal_risk_model.py",
    "scripts/batch/batch_terminal_risk_wf.ps1",
    "scripts/calibrate_exposure_gate.py",
    "scripts/analyze_terminal_risk_gate.py",
    "scripts/summarize_terminal_risk_wf.py",
    "scripts/analyze_drawdown_sweep.py",
    "scripts/analyze_policy_sidecar.py",
    "scripts/prepare_paper_exposure_warmup.py",
    "scripts/batch/batch_policy_sidecar.ps1",
    "scripts/batch/batch_walk_forward.ps1",
    "scripts/compare/signal_metrics.py",
    "scripts/compare/drawdown_sweep.py",
)

#: §5.2 tests 引用面（10 家族专测 + 7 政策层/旁路 + 2 复用/配置）
DISPOSAL_TESTS = tuple(
    f"tests/test_terminal_loss_{name}.py"
    for name in (
        "artifacts",
        "block_stats",
        "coverage_audit",
        "dataset",
        "gate_analysis",
        "labels",
        "mother_section",
        "rank_early_stop",
        "script",
        "train_model",
    )
) + (
    "tests/test_exposure_gate.py",
    "tests/test_exposure_online.py",
    "tests/test_backtest_exposure_override.py",
    "tests/test_exposure_budget_discount_replenish.py",
    "tests/test_policy_sidecar.py",
    "tests/test_summarize_terminal_risk_wf.py",
    "tests/test_paper_config_cleanup.py",
    "tests/test_signal_metrics.py",
    "tests/test_config.py",
)
DISPOSAL_FIXTURE = "tests/fixtures/inuse_closure_baseline_20261006.json"

#: §5 引用面实测值（按行口径，pattern 见下，排除 __pycache__）
_REFERENCE_PATTERN = re.compile(r"terminal_loss|terminal_risk")
SRC_REF_FILES = 39
SRC_REF_LINES = 85
SRC_REF_FAMILY_LINES = 46
SRC_REF_OUTSIDE_LINES = 39
TESTS_REF_FILES = 19
SCRIPTS_REF_FILES = 12


def _wc_lines(path: Path) -> int:
    """``wc -l`` 口径行数（字节 ``\\n`` 计数）。"""
    return path.read_bytes().count(b"\n")


def _reference_counts(root: Path, suffixes: tuple[str, ...]) -> tuple[int, int, dict[Path, int]]:
    """(命中文件数, 命中行数, 逐文件命中行数)，按行口径，排除 __pycache__。

    排除本核验测试自身（文件 2026-10-23 实测后才存在，不在文档实测面内；
    其常量表必然命中 pattern）。
    """
    self_path = Path(__file__).resolve()
    per_file: dict[Path, int] = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file() or path.suffix not in suffixes:
            continue
        if path.resolve() == self_path:
            continue
        hits = sum(
            1
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if _REFERENCE_PATTERN.search(line)
        )
        if hits:
            per_file[path] = hits
    return len(per_file), sum(per_file.values()), per_file


class TestDisposalDoc:
    def test_doc_exists(self):
        """处置档案在位（gitignore 内本地文档，核验前提）。"""
        assert DOC_PATH.is_file(), f"处置清单缺失: {DOC_PATH}"


class TestFamilyCode:
    """§2：12 个家族文件存在且行数锁定 2026-10-23 实测值。"""

    def test_family_files_exist_with_locked_line_counts(self):
        assert len(FAMILY_FILE_LINES) == 12
        for name, expected_lines in FAMILY_FILE_LINES.items():
            path = REPO_ROOT / FAMILY_DIR / name
            assert path.is_file(), f"家族文件缺失: {path}"
            assert _wc_lines(path) == expected_lines, f"{name} 行数漂移（观察期行为冻结）"

    def test_family_total_lines(self):
        total = sum(_wc_lines(REPO_ROOT / FAMILY_DIR / name) for name in FAMILY_FILE_LINES)
        assert total == FAMILY_TOTAL_LINES


class TestDisposalScripts:
    """§4：12 个训练/分析链路脚本存在。"""

    def test_scripts_exist(self):
        assert len(DISPOSAL_SCRIPTS) == 12
        missing = [rel for rel in DISPOSAL_SCRIPTS if not (REPO_ROOT / rel).is_file()]
        assert not missing, f"处置清单 §4 脚本缺失: {missing}"


class TestDisposalTests:
    """§5.2：19 个引用面测试文件 + 1 个 P2a 在用闭包基线 fixture 存在。"""

    def test_referring_tests_exist(self):
        assert len(DISPOSAL_TESTS) == 19
        missing = [rel for rel in DISPOSAL_TESTS if not (REPO_ROOT / rel).is_file()]
        assert not missing, f"处置清单 §5.2 测试文件缺失: {missing}"

    def test_inuse_closure_fixture_exists(self):
        assert (REPO_ROOT / DISPOSAL_FIXTURE).is_file()


class TestReferenceCounts:
    """§5 引用计数复算（按行口径 ``terminal_loss|terminal_risk``，排除 __pycache__）。"""

    def test_src_reference_counts(self):
        files, lines, per_file = _reference_counts(REPO_ROOT / "src", (".py",))
        assert files == SRC_REF_FILES
        assert lines == SRC_REF_LINES
        family_lines = sum(
            hits
            for path, hits in per_file.items()
            if FAMILY_DIR.parts == path.relative_to(REPO_ROOT).parts[: len(FAMILY_DIR.parts)]
        )
        assert family_lines == SRC_REF_FAMILY_LINES
        assert lines - family_lines == SRC_REF_OUTSIDE_LINES

    def test_tests_reference_counts(self):
        files, _lines, per_file = _reference_counts(REPO_ROOT / "tests", (".py",))
        assert files == TESTS_REF_FILES
        # 命中文件集 == §5.2 清单（引用面双向锁定）
        assert {p.relative_to(REPO_ROOT).as_posix() for p in per_file} == set(DISPOSAL_TESTS)

    def test_scripts_reference_counts(self):
        files, _lines, per_file = _reference_counts(REPO_ROOT / "scripts", (".py", ".ps1"))
        assert files == SCRIPTS_REF_FILES
        assert {p.relative_to(REPO_ROOT).as_posix() for p in per_file} == set(DISPOSAL_SCRIPTS)
