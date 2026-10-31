# -*- coding: utf-8 -*-
"""T0.5b spike：门 2b 数据态绑定留证（P4 规划 §2.1-3 / 门 2b，阶段专用临时开发文件）。

留证规则（冻结）：门 2b「基线保持门」的每次执行，其比较输入必须绑定到
**基线批次自身登记的输入 manifest / 冻结数据态**（本文件采集的指纹），
**禁止读当前 data root 现状后比较**；数据态漂移单独归因、不得计入代码回归。

采集对象（门 2b 载体精确分列，v4 S-01 修正）：
- 冻结参照 runs 批：P2a 基线批 ``p2a_b0replay_20261008_144307_d79f3f``、
  P2b 关闭批 ``p2b_t5_closed_20261017_203508``、
  P3 收口复跑批 ``p2a_b0replay_20261026_122848_979330``（历史 PASS verdict 指向的实际 runs）；
- 冻结 B0 raw 目录：``data/walk_forward/batches/wf_batch_20260930_171221/raw``
  （``compare_replay_vs_b0`` 默认参照）及其数据态文件（``raw/data_state_*.json`` glob 定位）；
- 历史 PASS 证据目录：``data/frozen_reference/v2_p3/t5_regression_edge_final_20261026``
  （仅证据：consumption 快照 + verdict JSON，非比较输入）；
- 比较工具两件：``compare_replay_vs_b0.py`` / ``regression_edge_runner.py``（源码哈希登记）。

**fail-closed（T0.5b 评审 P1-1/R2-01 处置）**：必选对象缺失或必备字段缺位
一律记入顶层 ``errors`` 且 ``verdict=FAIL``（子对象错误上冒），禁止 null 静默通过。

只读采集，不写任何被采集对象。

复杂度豁免依据：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

FROZEN_RUNS_BATCHES: Tuple[str, ...] = (
    "p2a_b0replay_20261008_144307_d79f3f",
    "p2b_t5_closed_20261017_203508",
    "p2a_b0replay_20261026_122848_979330",
)

B0_RAW_DIR = Path("data/walk_forward/batches/wf_batch_20260930_171221/raw")
# 数据态文件在 raw/ 子目录内（T0.5b 评审 P1-1：初版常量漏 raw/ 层），glob 定位防文件名漂移
B0_DATA_STATE_GLOB = "data_state_*.json"

HISTORICAL_EVIDENCE_DIR = Path("data/frozen_reference/v2_p3/t5_regression_edge_final_20261026")
HISTORICAL_EVIDENCE_FILES: Tuple[str, ...] = (
    "consumption_before.json",
    "regression_edge_verdict.json",
)

COMPARE_TOOLS: Tuple[str, ...] = (
    "scripts/v2_p2a/compare_replay_vs_b0.py",
    "scripts/v2_p3/regression_edge_runner.py",
)

BINDING_RULE = (
    "门 2b 每次执行：比较输入 = 本文件登记的冻结参照批次（其 batch_meta.data_state 为绑定数据态）；"
    "禁止读当前 data root 现状后比较；数据态漂移（当前水位 ≠ 绑定水位）单独归因，"
    "不得计入代码回归。"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch_binding(repo_root: Path, batch_id: str) -> Dict:
    meta_path = repo_root / "data" / "runs" / batch_id / "batch_meta.json"
    if not meta_path.exists():
        return {"batch_id": batch_id, "error": f"batch_meta.json 不存在: {meta_path}"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ds = meta.get("data_state", {})
    entry = {
        "batch_id": batch_id,
        "host_mode": meta.get("host_mode"),
        "source": meta.get("source"),
        "created_at": meta.get("created_at"),
        "git_commit": ds.get("git_commit"),
        "data_state_id": ds.get("data_state_id"),
        "raw_latest_partitions": ds.get("raw_latest_partitions"),
        "features_cs_train_latest": ds.get("features_cs_train_latest"),
        "batch_meta_sha256": _sha256_file(meta_path),
    }
    if not ds.get("data_state_id"):
        entry["error"] = f"{batch_id} batch_meta 缺 data_state.data_state_id"
    return entry


def _b0_raw_binding(repo_root: Path) -> Dict:
    raw_dir = repo_root / B0_RAW_DIR
    if not raw_dir.exists():
        return {"path": str(B0_RAW_DIR), "error": f"B0 raw 目录不存在: {raw_dir}"}
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    entry: Dict = {
        "path": B0_RAW_DIR.as_posix(),
        "file_count": len(files),
        "data_state_glob": f"{B0_RAW_DIR.as_posix()}/{B0_DATA_STATE_GLOB}",
    }
    candidates = sorted(raw_dir.glob(B0_DATA_STATE_GLOB))
    if len(candidates) != 1:
        entry["error"] = f"数据态文件 glob 命中数 = {len(candidates)}（期望 1）: {candidates}"
        return entry
    ds_path = candidates[0]
    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    entry["data_state_file"] = str(ds_path.relative_to(repo_root))
    entry["data_state_sha256"] = _sha256_file(ds_path)
    entry["data_state_id"] = ds.get("data_state_id")
    entry["raw_latest_partitions"] = ds.get("raw_latest_partitions")
    entry["features_cs_train_latest"] = ds.get("features_cs_train_latest")
    if not ds.get("data_state_id"):
        entry["error"] = f"数据态文件缺 data_state_id: {ds_path}"
    return entry


def _historical_evidence_binding(repo_root: Path) -> Dict:
    ev_dir = repo_root / HISTORICAL_EVIDENCE_DIR
    if not ev_dir.exists():
        return {"path": str(HISTORICAL_EVIDENCE_DIR), "error": f"历史 PASS 证据目录不存在: {ev_dir}"}
    entry: Dict = {"path": str(HISTORICAL_EVIDENCE_DIR)}
    missing = []
    for name in HISTORICAL_EVIDENCE_FILES:
        fp = ev_dir / name
        entry[f"{name}_sha256"] = _sha256_file(fp) if fp.exists() else None
        if not fp.exists():
            missing.append(name)
    verdict_path = ev_dir / "regression_edge_verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        entry["verdict"] = verdict.get("verdict")
        entry["verdict_new_batch"] = verdict.get("new_batch")
        entry["verdict_frozen_batch"] = verdict.get("frozen_batch")
    if missing:
        entry["error"] = f"历史证据目录缺文件: {missing}"
    return entry


def collect_binding_evidence(repo_root: Path) -> Dict:
    """只读采集门 2b 绑定证据（冻结参照身份 + 数据态 + 工具哈希）。

    fail-closed：任一必选对象缺失/字段缺位 ⇒ ``errors`` 非空且 ``verdict=FAIL``。
    """
    repo_root = Path(repo_root)
    evidence: Dict = {
        "gate": "gate2b_data_state_binding",
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "binding_rule": BINDING_RULE,
        "frozen_runs_batches": [],
        "b0_raw_dir": None,
        "historical_pass_evidence": None,
        "compare_tools": [],
        "errors": [],
    }

    for batch_id in FROZEN_RUNS_BATCHES:
        entry = _batch_binding(repo_root, batch_id)
        if "error" in entry:
            evidence["errors"].append(entry["error"])
        evidence["frozen_runs_batches"].append(entry)

    b0_entry = _b0_raw_binding(repo_root)
    if "error" in b0_entry:
        evidence["errors"].append(b0_entry["error"])
    evidence["b0_raw_dir"] = b0_entry

    hist_entry = _historical_evidence_binding(repo_root)
    if "error" in hist_entry:
        evidence["errors"].append(hist_entry["error"])
    evidence["historical_pass_evidence"] = hist_entry

    for tool in COMPARE_TOOLS:
        tool_path = repo_root / tool
        if not tool_path.exists():
            evidence["compare_tools"].append({"path": tool, "sha256": None})
            evidence["errors"].append(f"比较工具不存在: {tool}")
        else:
            evidence["compare_tools"].append({"path": tool, "sha256": _sha256_file(tool_path)})

    evidence["verdict"] = "PASS" if not evidence["errors"] else "FAIL"
    return evidence


def write_binding_evidence(repo_root: Path, out_dir: Path) -> Path:
    """采集并落盘 binding 证据，返回写出路径。"""
    evidence = collect_binding_evidence(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate2b_binding.json"
    out_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path
