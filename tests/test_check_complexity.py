# -*- coding: utf-8 -*-
"""check_complexity 脚本单元测试（合成文件，不依赖真实代码库）"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check" / "check_complexity.py"
_spec = importlib.util.spec_from_file_location("check_complexity", _SCRIPT)
check_complexity = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_complexity  # dataclass 处理需要模块已注册
_spec.loader.exec_module(check_complexity)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_clean_file_no_violation(tmp_path):
    p = _write(tmp_path, "ok.py", "def f():\n    return 1\n")
    report = check_complexity.check_file(p, tmp_path)
    assert report.violations == []


def test_file_hard_limit(tmp_path):
    body = "\n".join(f"# line {i}" for i in range(check_complexity.FILE_LINES_HARD + 1))
    p = _write(tmp_path, "big.py", body)
    report = check_complexity.check_file(p, tmp_path)
    assert any(v.kind == "file_lines" and v.level == "hard" for v in report.violations)


def test_file_soft_limit(tmp_path):
    body = "\n".join(f"# line {i}" for i in range(check_complexity.FILE_LINES_SOFT + 1))
    p = _write(tmp_path, "mid.py", body)
    report = check_complexity.check_file(p, tmp_path)
    assert any(v.kind == "file_lines" and v.level == "soft" for v in report.violations)
    assert not any(v.level == "hard" for v in report.violations)


def test_func_lines_and_nesting(tmp_path):
    long_body = "\n".join(f"    x{i} = {i}" for i in range(check_complexity.FUNC_LINES_SOFT + 2))
    nested = "def g():\n"
    for d in range(check_complexity.NESTING_HARD + 1):
        nested += "    " * (d + 1) + ("if True:\n" if d % 2 == 0 else "for _ in [1]:\n")
    nested += "    " * (check_complexity.NESTING_HARD + 2) + "pass\n"
    p = _write(tmp_path, "funcs.py", f"def f():\n{long_body}\n\n{nested}")
    report = check_complexity.check_file(p, tmp_path)
    kinds = {(v.kind, v.level) for v in report.violations}
    assert ("func_lines", "soft") in kinds
    assert ("nesting", "hard") in kinds


def test_waiver_marker_downgrades(tmp_path):
    header = "# 复杂度豁免：历史遗留文件，拆分计划见契约 §4.4 备案\n"
    body = "\n".join(f"# line {i}" for i in range(check_complexity.FILE_LINES_HARD + 1))
    p = _write(tmp_path, "waived.py", header + body)
    report = check_complexity.check_file(p, tmp_path)
    assert report.waived
    assert all(v.waived for v in report.violations)


def test_syntax_error_reported(tmp_path):
    p = _write(tmp_path, "bad.py", "def broken(:\n")
    report = check_complexity.check_file(p, tmp_path)
    assert any(v.kind == "syntax" and v.level == "hard" for v in report.violations)


def test_main_exit_code(tmp_path, monkeypatch):
    _write(tmp_path, "ok.py", "def f():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    rc = check_complexity.main(["--paths", ".", "--skip-mccabe"])
    assert rc == 0

    big = "\n".join(f"# line {i}" for i in range(check_complexity.FILE_LINES_HARD + 1))
    _write(tmp_path, "big.py", big)
    rc = check_complexity.main(["--paths", ".", "--skip-mccabe"])
    assert rc == 1


def test_collect_files_skips_pycache(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    _write(cache_dir, "b.py", "y = 2\n")
    files = check_complexity.collect_py_files([tmp_path])
    assert len(files) == 1 and files[0].name == "a.py"


def test_baseline_ratchet(tmp_path, monkeypatch):
    """存量棘轮：基线内硬超限不阻塞；新增硬超限失败；基线清掉后收缩"""
    big = "\n".join(f"# line {i}" for i in range(check_complexity.FILE_LINES_HARD + 1))
    _write(tmp_path, "big.py", big)
    monkeypatch.chdir(tmp_path)

    # 写基线 → 基线模式下通过
    assert check_complexity.main(["--paths", ".", "--skip-mccabe", "--write-baseline", "base.json"]) == 0
    assert check_complexity.main(["--paths", ".", "--skip-mccabe", "--baseline", "base.json"]) == 0

    # 新增一个硬超限文件 ⇒ 失败
    _write(tmp_path, "big2.py", big)
    assert check_complexity.main(["--paths", ".", "--skip-mccabe", "--baseline", "base.json"]) == 1

    # 回到基线规模 ⇒ 通过（基线只缩不扩：big.py 修复后仍通过）
    (tmp_path / "big2.py").unlink()
    (tmp_path / "big.py").write_text("x = 1\n", encoding="utf-8")
    assert check_complexity.main(["--paths", ".", "--skip-mccabe", "--baseline", "base.json"]) == 0


def test_mccabe_c901_actually_collected(tmp_path):
    """C901 非零诊断必须被采集（R2-T1-01 回归：旧格式串缺 %(code)s 导致静默漏报）。"""
    pytest.importorskip("flake8")
    branches = "\n".join(f"    if x == {i}:\n        y = {i}" for i in range(15))
    p = _write(tmp_path, "complex_mod.py", f"def f(x):\n{branches}\n    return y\n")
    violations = check_complexity.check_mccabe([tmp_path], tmp_path)
    assert any(
        v.kind == "mccabe" and v.level == "hard" and "too complex" in v.detail
        for v in violations
    ), f"C901 诊断未被采集（漏报回归）：{violations}"
    assert any(p.name in v.path for v in violations)


_COMPLEX_BODY = "\n".join(f"    if x == {i}:\n        y = {i}" for i in range(15))


def test_mccabe_baseline_keys_repo_relative_posix(tmp_path, monkeypatch):
    """R2-T1-R2-01 回归：基线键一律仓库相对 + 正斜杠，且新增超限仍失败。"""
    pytest.importorskip("flake8")
    import json

    _write(tmp_path, "complex_mod.py", f"def f(x):\n{_COMPLEX_BODY}\n    return y\n")
    monkeypatch.chdir(tmp_path)
    assert check_complexity.main(["--paths", ".", "--write-baseline", "base.json"]) == 0
    keys = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))["keys"]
    assert any(k.startswith("complex_mod.py|mccabe|") for k in keys)
    for k in keys:
        path_part = k.split("|")[0]
        assert "\\" not in path_part, f"基线键含反斜杠：{k}"
        assert not Path(path_part).is_absolute(), f"基线键为绝对路径：{k}"
    # 基线复跑通过；新增真实超限仍返回非零
    assert check_complexity.main(["--paths", ".", "--baseline", "base.json"]) == 0
    _write(tmp_path, "complex_mod2.py", f"def g(x):\n{_COMPLEX_BODY}\n    return y\n")
    assert check_complexity.main(["--paths", ".", "--baseline", "base.json"]) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 盘符大小写语义")
def test_gate_matches_under_case_variant_paths(tmp_path, monkeypatch):
    """R2-T1-R2-01 回归：入口路径盘符大小写变化（d: vs D:）不影响存量命中。"""
    pytest.importorskip("flake8")
    target = _write(tmp_path, "complex_mod.py", f"def f(x):\n{_COMPLEX_BODY}\n    return y\n")
    monkeypatch.chdir(tmp_path)
    assert check_complexity.main(["--paths", ".", "--write-baseline", "base.json"]) == 0
    # 用大小写翻转的绝对路径输入（模拟小写 d: 入口），同一存量必须仍命中基线
    variant = str(target).swapcase()
    rc = check_complexity.main(["--paths", variant, "--baseline", "base.json"])
    assert rc == 0, "盘符/路径大小写变化导致存量被误判为新增"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
