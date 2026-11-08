# -*- coding: utf-8 -*-
"""P6 T5 复算包自检脚本的工作域守卫测试（评审 R3-T5-01）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import selftest_t5_linux_replay as selftest_module  # noqa: E402


def test_selftest_work_dir_must_be_under_temp() -> None:
    """`--work` 必须严格位于 `temp/` 之下：仓库根/生产根/逃逸/`temp` 本身一律拒绝。"""
    for bad in (".", "data/paper", "data/runs", "temp", "../outside", "src"):
        with pytest.raises(selftest_module.SelftestSetupError):
            selftest_module.safe_work_dir(bad)
    absolute_outside = (ROOT.parent / "outside_work").as_posix()
    with pytest.raises(selftest_module.SelftestSetupError):
        selftest_module.safe_work_dir(absolute_outside)
    ok = selftest_module.safe_work_dir("temp/p6_t5_selftest_unit")
    assert ok == (selftest_module.TEMP_ROOT / "p6_t5_selftest_unit").resolve()


def test_selftest_refuses_to_delete_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--work .` 必须在任何删除之前被拒（rmtree 不被调用）。"""
    calls: list[Path] = []
    monkeypatch.setattr(selftest_module.shutil, "rmtree", lambda path: calls.append(Path(path)))
    code = selftest_module.main(["--work", ".", "--json"])
    assert code == 2
    assert calls == []


def test_selftest_refuses_to_delete_production_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--work data/paper` 同样必须在删除之前被拒。"""
    calls: list[Path] = []
    monkeypatch.setattr(selftest_module.shutil, "rmtree", lambda path: calls.append(Path(path)))
    code = selftest_module.main(["--work", "data/paper", "--json"])
    assert code == 2
    assert calls == []


def test_selftest_rejects_original_path_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """R3-T5-R1-01①：解析**之前**就要拒绝原路径上的链接/重解析点（别名不得被 resolve 抹平）。"""
    alias = (selftest_module.TEMP_ROOT / "alias_dir").resolve()
    original = selftest_module._is_reparse_point

    def fake_is_reparse(path: Path) -> bool:
        return path == alias or original(path)

    monkeypatch.setattr(selftest_module, "_is_reparse_point", fake_is_reparse)
    with pytest.raises(selftest_module.SelftestSetupError):
        selftest_module.safe_work_dir(alias / "child")


def test_selftest_stub_writer_rejects_hardlink(tmp_path: Path) -> None:
    """R3-T5-R1-01②：stub 是硬链接到工作域外文件时，不得沿链接写穿目标。"""
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    stub = tmp_path / "work_payload.json"
    try:
        stub.hardlink_to(sentinel)
    except (OSError, NotImplementedError):  # pragma: no cover - 平台不支持硬链接
        pytest.skip("本机不支持硬链接")
    with pytest.raises(selftest_module.SelftestSetupError):
        selftest_module.safe_stub_write(stub, "overwritten\n")
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_selftest_stub_writer_uses_atomic_replace(tmp_path: Path) -> None:
    """正常情况：stub 以原子替换写出，内容正确且无残留临时文件。"""
    stub = tmp_path / "work_payload.json"
    stub.write_text("old\n", encoding="utf-8")
    selftest_module.safe_stub_write(stub, "new\n")
    assert stub.read_text(encoding="utf-8") == "new\n"
    leftovers = [path.name for path in tmp_path.iterdir() if ".tmp" in path.name]
    assert leftovers == []
