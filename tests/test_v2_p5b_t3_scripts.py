# -*- coding: utf-8 -*-
"""P5b T3 验收工具薄入口测试（tests/ 全量豁免复杂度）。

覆盖：账本审计工具（只读，零写盘）、判定表导出工具（写域守卫 + 输出）与
故障矩阵工具的非空 root 拒写（防覆盖既有证据）。用子进程调用，等价于运维实际用法。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t3_env as envmod  # noqa: E402


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "v2_p5b" / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


def test_ledger_audit_tool_is_read_only_and_reports_status(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env)
    envmod.run(env, prepared, env.days[:2])
    before = envmod.tree_state(prepared.root)

    result = _run("ledger_audit.py", "--campaign-root", str(prepared.root), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["three_way_status"] == "PASS"
    assert payload["published_generation"] == 2
    assert payload["repair"] is None
    # 只读：不带 --repair/--report 时零写盘
    assert envmod.tree_state(prepared.root) == before


def test_ledger_audit_tool_returns_2_on_invalidated(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env)
    envmod.run(env, prepared, env.days[:2])
    envmod.tamper_shard(prepared, "aa_1", env.days[0])
    repair = _run(
        "ledger_audit.py",
        "--campaign-root",
        str(prepared.root),
        "--repair",
        "--allow-external-root",
        "--json",
    )
    assert repair.returncode == 2, (repair.returncode, repair.stdout, repair.stderr)
    payload = json.loads(repair.stdout)
    assert payload["status"] == "INVALIDATED"
    assert payload["repair"]["blocked"] is True


def test_export_verdict_table_tool_writes_markdown(tmp_path):
    matrix = {
        "campaign_id": "probe",
        "matrix_fingerprint": "sha-probe",
        "all_passed": True,
        "passed_count": 1,
        "case_count": 1,
        "invariants": [
            "no_double_booking",
            "unpublished_not_in_generation",
            "deterministic_recompute",
            "read_side_published_only",
            "parent_backup_verifiable",
        ],
        "cases": [
            {
                "case_id": "T3-FM-09",
                "fault_point": "virtual_date_regression",
                "terminal": "REJECTED",
                "expected_terminal": "REJECTED",
                "recovery": "advance_refused",
                "invariants": {
                    "no_double_booking": True,
                    "unpublished_not_in_generation": True,
                    "deterministic_recompute": True,
                    "read_side_published_only": True,
                    "parent_backup_verifiable": True,
                },
                "passed": True,
            }
        ],
    }
    source = tmp_path / "fault_matrix.json"
    source.write_text(json.dumps(matrix, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "verdict.md"
    result = _run("export_verdict_table.py", "--matrix-json", str(source), "--out", str(out))
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "T3-FM-09" in text and "advance_refused" in text and "通过" in text
    assert "no_double_booking" in text


def test_export_verdict_table_refuses_production_output(tmp_path):
    """`--out` 受生产根硬禁：导出工具不得把结果落 data/paper 等生产面。"""
    matrix = {
        "campaign_id": "probe",
        "all_passed": True,
        "case_count": 0,
        "cases": [],
        "invariants": [],
    }
    source = tmp_path / "fault_matrix.json"
    source.write_text(json.dumps(matrix, ensure_ascii=False), encoding="utf-8")
    target = ROOT / "data" / "paper" / "t3_verdict_probe.md"
    result = _run("export_verdict_table.py", "--matrix-json", str(source), "--out", str(target))
    assert result.returncode != 0
    assert not target.exists()


def test_fault_matrix_tool_refuses_non_empty_root(tmp_path):
    root = tmp_path / "existing"
    (root / "audit").mkdir(parents=True)
    (root / "audit" / "keep.json").write_text("{}", encoding="utf-8")
    result = _run("fault_matrix.py", "--root", str(root), "--cases", "T3-FM-09")
    assert result.returncode != 0
    # 子进程控制台编码随平台而异，只断言「未写入/未覆盖」这一可移植事实
    assert (root / "audit" / "keep.json").is_file()
    assert not (root / "audit" / "fault_matrix.json").exists()
