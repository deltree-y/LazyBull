# -*- coding: utf-8 -*-
"""P7 T1 发布入口的写域、禁覆盖与身份校验。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7.t1 import params as t1_params  # noqa: E402
from scripts.v2_p7.t1 import run_p1  # noqa: E402


def _args(out_root: Path, run_id: str = "unit_run", allow: bool = True):
    argv = ["--run-id", run_id, "--out-root", str(out_root)]
    if allow:
        argv.extend(["--allow-root", str(out_root)])
    return run_p1.parse_args(argv)


def test_existing_evidence_root_is_never_overwritten(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    target = out_root / "unit_run"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("original", encoding="utf-8")
    code = run_p1.run(_args(out_root))
    assert code == run_p1.EXIT_WRITE_SCOPE
    assert (target / "keep.txt").read_text(encoding="utf-8") == "original"
    assert not list(out_root.glob("*.partial-*"))


def test_write_scope_rejects_unlisted_root(tmp_path: Path) -> None:
    out_root = tmp_path / "outside"
    code = run_p1.run(_args(out_root, allow=False))
    assert code == run_p1.EXIT_WRITE_SCOPE
    assert not out_root.exists()


def test_parse_defaults_follow_prereg() -> None:
    args = run_p1.parse_args([])
    assert args.run_id == t1_params.DEFAULT_RUN_ID
    assert args.implementations == 0
    assert args.limit_signal_days == 0


def test_prereg_identity_prefix_is_enforced(tmp_path: Path) -> None:
    payload = {
        "schema": "p7_b0_composite_prereg/v2",
        "status": "effective",
        "prereg_id": "fake",
    }
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(t1_params.ParamsError):
        t1_params.load_prereg(PROJECT_ROOT, path)


def test_script_identity_covers_t1_modules() -> None:
    identity = run_p1.script_identity(run_p1._REPO_ROOT)
    for name in (
        "scripts/v2_p7/event_builder.py",
        "scripts/v2_p7/t1/params.py",
        "scripts/v2_p7/t1/run_p1.py",
        "scripts/v2_p7/t1/randomization.py",
    ):
        assert name in identity
        assert len(identity[name]) == 64


def test_git_state_is_read_only() -> None:
    state = run_p1.git_state(run_p1._REPO_ROOT)
    assert "git_head" in state and "dirty" in state
    assert isinstance(state["dirty_files"], list)
