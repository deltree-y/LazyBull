"""P2b-T3：实际工作树、稳定摘要与 AST 归属的临时 Git 反例。"""

import subprocess

from scripts.v2_p2b import check_workspace_state as tool


def _git(repo, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write(repo, relative, content):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "syn@example.invalid")
    _git(repo, "config", "user.name", "syn")
    _write(repo, "scripts/v2_p2b/tool.py", "VALUE = 1\n")
    _write(repo, "src/lazybull/backtest/engine.py", "OLD = 1\n")
    _write(repo, "src/lazybull/v2/core/execution/base.py", "VALUE = 1\n")
    _write(repo, "src/lazybull/v2/hosts/native_engine.py", "# native boundary\n")
    _write(repo, "src/lazybull/v2/common/types.py", "# types\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_four_git_states_and_allowed_changes(tmp_path):
    repo, base = _init_repo(tmp_path)
    _write(repo, "scripts/v2_p2b/tool.py", "VALUE = 2\n")
    _git(repo, "add", "scripts/v2_p2b/tool.py")
    _git(repo, "commit", "-m", "committed")
    _write(repo, "scripts/v2_p2b/staged.py", "STAGED = 1\n")
    _git(repo, "add", "scripts/v2_p2b/staged.py")
    _write(repo, "scripts/v2_p2b/tool.py", "VALUE = 3\n")
    _write(repo, "scripts/v2_p2b/untracked.py", "UNTRACKED = 1\n")

    report = tool.check_workspace_state(repo, old_production_base=base, modification_base=base)
    assert report["ok"], report["violations"]
    assert report["changes"]["scripts/v2_p2b/tool.py"] == ["committed", "unstaged"]
    assert report["changes"]["scripts/v2_p2b/staged.py"] == ["staged"]
    assert report["changes"]["scripts/v2_p2b/untracked.py"] == ["untracked"]


def test_unstaged_old_production_and_out_of_surface_files_rejected(tmp_path):
    repo, base = _init_repo(tmp_path)
    _write(repo, "src/lazybull/backtest/engine.py", "OLD = 2\n")
    _write(repo, "docs/rogue.txt", "rogue")
    report = tool.check_workspace_state(repo, old_production_base=base, modification_base=base)
    assert not report["ok"]
    assert any(
        "src/lazybull/backtest/engine.py" in item for item in report["old_production_violations"]
    )
    assert any("docs/rogue.txt" in item for item in report["modification_surface_violations"])


def test_untracked_v2_file_outside_registered_surface_rejected(tmp_path):
    repo, base = _init_repo(tmp_path)
    _write(repo, "src/lazybull/v2/core/execution/rogue.py", "ROGUE = 1\n")
    report = tool.check_workspace_state(repo, old_production_base=base, modification_base=base)
    assert not report["ok"]
    assert any("rogue.py" in item for item in report["modification_surface_violations"])


def test_digest_tracks_edit_new_init_and_deletion(tmp_path):
    repo, _ = _init_repo(tmp_path)
    first = tool.collect_workspace_digest(repo)
    assert tool.validate_workspace_digest(first)
    _write(repo, "scripts/v2_p2b/tool.py", "VALUE = 2\n")
    second = tool.collect_workspace_digest(repo)
    assert second["digest"] != first["digest"]
    _write(repo, "scripts/v2_p2b/__init__.py", "# init\n")
    third = tool.collect_workspace_digest(repo)
    assert "scripts/v2_p2b/__init__.py" in third["files"]
    assert third["digest"] != second["digest"]
    (repo / "scripts/v2_p2b/tool.py").unlink()
    fourth = tool.collect_workspace_digest(repo)
    assert "scripts/v2_p2b/tool.py" in fourth["deleted_files"]
    assert fourth["digest"] != third["digest"]
    tampered = {**fourth, "files": dict(fourth["files"])}
    tampered["files"]["scripts/v2_p2b/__init__.py"] = "0" * 64
    assert not tool.validate_workspace_digest(tampered)


def test_ast_catches_each_dynamic_import_form(tmp_path):
    cases = {
        "absolute_import": "import src.lazybull.v2.hosts.backtest.runtime\n",
        "absolute_from": "from src.lazybull.v2.hosts import backtest\n",
        "direct_import_module": (
            "from importlib import import_module\n"
            "module = import_module('src.lazybull.v2.hosts.backtest.runtime')\n"
        ),
        "alias_import_module": (
            "from importlib import import_module as load_module\n"
            "module = load_module('src.lazybull.v2.hosts.backtest.runtime')\n"
        ),
        "attribute_import_module": (
            "import importlib\n"
            "module = importlib.import_module('src.lazybull.v2.hosts.backtest.runtime')\n"
        ),
        "relative_from": "from ...hosts import native_engine\n",
    }
    for name, source in cases.items():
        repo, _ = _init_repo(tmp_path / name)
        _write(repo, "src/lazybull/v2/core/execution/rogue.py", source)
        violations = tool.check_ast_boundaries(repo)
        assert any("core 禁止 import hosts" in item for item in violations), name


def test_ast_catches_constructor_and_nav_aliases(tmp_path):
    repo, _ = _init_repo(tmp_path)
    _write(
        repo,
        "src/lazybull/v2/hosts/backtest/rogue.py",
        "from ...common.types import Account as A\n"
        "ctor: object = A\n"
        "account = ctor()\n"
        "initial: float = self.initial_capital\n"
        "nav = total_value / initial\n",
    )
    violations = tool.check_ast_boundaries(repo)
    assert any("值对象构造" in item for item in violations)
    assert any("不得自行除以 initial_capital" in item for item in violations)


def test_nav_exemptions_are_function_scoped_and_keep_positive_detection(tmp_path):
    repo, _ = _init_repo(tmp_path)
    for relative, allowed in (
        ("src/lazybull/v2/hosts/paper/performance.py", "calculate_annualized_return"),
        ("src/lazybull/v2/hosts/paper/reporting.py", "load_position_snapshot"),
    ):
        _write(
            repo,
            relative,
            f"def {allowed}(current_value, initial_capital):\n"
            "    return current_value / initial_capital\n"
            "def rogue_new_formula(total_value, initial_capital):\n"
            "    return total_value / initial_capital\n",
        )
    violations = tool.check_ast_boundaries(repo)
    nav_hits = [item for item in violations if "不得自行除以 initial_capital" in item]
    assert len(nav_hits) == 2
    assert any("performance.py:4" in item for item in nav_hits)
    assert any("reporting.py:4" in item for item in nav_hits)


def test_environment_versions_are_separate_from_code_digest(monkeypatch, tmp_path):
    repo, _ = _init_repo(tmp_path)
    monkeypatch.setattr(tool.metadata, "version", lambda name: f"{name}-version")
    environment = tool.collect_environment_versions()
    assert environment["packages"]["pandas"] == "pandas-version"
    assert environment["python"]
    digest = tool.canonical_json(tool.collect_workspace_digest(repo))
    assert "pandas-version" not in digest
