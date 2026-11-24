# -*- coding: utf-8 -*-
"""run-a deliver 批次的封存/发布/未定日两端测算（checkpoint-2 Codex 第 4、6 项）。

run_a.py 超 1200 行硬上限 ⇒ 本模块从 run_a 抽出：
- ``_render_undated_bounds`` / ``_undated_bounds`` / ``_load_undated_records``：
  600602.SH 未定日事件的逐账户两端测算（上界 = 税后每股 8.0 元 × 最大持有股数；
  状态 unbounded）；
- ``_PUBLISH_MAP`` / ``_seal_deliverables`` / ``_digest_of_digests``：三件交付 +
  宽度/训练/审计件发布到 run 根并封存；work/ 未发布文件逐件 SHA 登记进
  manifest payload。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.common import sha256_file


class DeliverSealError(RuntimeError):
    """封存/发布失败（fail-fast）。"""


def load_undated_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "work" / "corporate_resolved" / "undated_audit.json"
    if not path.exists():
        return []
    audit = json.loads(path.read_text(encoding="utf-8"))
    return list(audit.get("records") or [])


def undated_bounds(run_dir: Path) -> Dict[str, Any]:
    """600602.SH 未定日事件的逐账户两端测算（checkpoint-2 Codex 第 4 项）。

    上界 = 税后每股 8.0 元 × 该账户最大持有股数（现金一次性到账）；下界 = 0
    （隔离回放即未授予）。若账户未持有该股 ⇒ 不列。
    """
    records = load_undated_records(run_dir)
    if not any(record.get("ts_code") == "600602.SH" for record in records):
        return {}
    per_share = 8.0
    bounds: Dict[str, Any] = {}
    for path in sorted((run_dir / "accounts").glob("*/*/trades.json")):
        trades = json.loads(path.read_text(encoding="utf-8"))
        rows = trades if isinstance(trades, list) else trades.get("rows") or []
        shares = [
            int(row.get("shares") or 0)
            for row in rows
            if str(row.get("stock")) == "600602.SH" and row.get("action") == "buy"
        ]
        if not shares:
            continue
        capital = path.parents[1].name
        role = path.parent.name
        max_shares = max(shares)
        bounds[f"{role}@{capital}"] = {
            "max_shares": max_shares,
            "upper_bound_yuan": per_share * max_shares,
        }
    return bounds


def render_undated_bounds(lines: List[str], undated_bounds: Mapping[str, Any]) -> None:
    if not undated_bounds:
        return
    lines.append("## 9. 未定日事件两端测算（600602.SH，1991 年实施行权益状态不可判定）")
    lines.append("")
    lines.append(
        "- 引擎严格检查按前裁决保留；结清凭证不可得（本地 adj_factor 原始分区自 2005 年起）；"
        "回放按事件未授予（隔离）执行，逐受影响账户登记如下区间；状态 **unbounded**。"
    )
    lines.append("")
    lines.append("| 账户 | 最大持有股数 | 下界（隔离回放） | 上界增量（税后 8.0 元/股） |")
    lines.append("|---|---|---|---|")
    for key in sorted(undated_bounds):
        item = undated_bounds[key]
        lines.append(
            f"| {key} | {item['max_shares']} | 报告值 | +{item['upper_bound_yuan']:.2f} 元 |"
        )
    lines.append("")


#: 发布到 run 根的工件（work/ 相对路径 → 根文件名）
PUBLISH_MAP: Sequence[Tuple[str, str]] = (
    ("a_review.md", "a_review.md"),
    ("a_metrics.csv", "a_metrics.csv"),
    ("b_required.csv", "b_required.csv"),
    ("metrics_full.json", "metrics_full.json"),
    ("stress_metrics.json", "stress_metrics.json"),
    ("scoring_summary.json", "scoring_summary.json"),
    ("width_by_day.csv", "width_by_day.csv"),
    ("structural_manifest.json", "structural_manifest.json"),
    ("training_F.json", "training_F.json"),
    ("training_R.json", "training_R.json"),
    ("account_inputs.json", "account_inputs.json"),
    ("ranked/ranked_F.parquet", "ranked_F.parquet"),
    ("ranked/ranked_R.parquet", "ranked_R.parquet"),
    ("corporate_resolved/dedupe_audit.json", "dedupe_audit.json"),
    ("corporate_resolved/resolutions.json", "resolutions.json"),
    ("corporate_resolved/undated_audit.json", "undated_audit.json"),
)


def seal_deliverables(
    seal_entry: Callable[..., Dict[str, Any]],
    run_dir: Path,
    work: Path,
    gates: Mapping[str, Any],
    state: Mapping[str, Any],
    extra_roots: Sequence[str],
) -> Dict[str, Any]:
    """发布三件交付 + 宽度/训练/审计件到 run 根并封存（checkpoint-2 第 6 项）。

    work/ 未发布文件（账户原始账、逐日评分帧等）以逐件 SHA256 登记进
    manifest payload（``work_digests``），不动字节位置。
    """
    from scripts.v2_p7.cashflow_sleeve_split.artifacts import publish_file

    for relative, name in PUBLISH_MAP:
        source = work / relative
        if not source.exists():
            raise DeliverSealError(f"封存缺件：{source}")
        publish_file(source, run_dir, name, extra_roots)
    work_digests = {
        path.relative_to(work).as_posix(): sha256_file(path)
        for path in sorted(work.rglob("*"))
        if path.is_file()
    }
    payload = {
        "parent": state["parent"],
        "parent_manifest_sha256": state["parent_manifest_sha256"],
        "authorized_by": state.get("authorized_by"),
        "code_restore": "work/code_restore/code_restore.json",
        "batches_done": list(state.get("batches_done") or []),
        "batch_digests": dict(state.get("batch_digests") or {}),
        "gates": {capital: dict(gate) for capital, gate in gates.items()},
        "work_digests_n": len(work_digests),
        "work_digests_sha256": digest_of_digests(work_digests),
        "work_digests": work_digests,
    }
    return seal_entry(Path(run_dir).parent.parent, run_dir, run_dir.name, payload, extra_roots)


def digest_of_digests(digests: Mapping[str, str]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(digests):
        hasher.update(f"{name}:{digests[name]}\n".encode("utf-8"))
    return hasher.hexdigest()


__all__ = [
    "DeliverSealError",
    "PUBLISH_MAP",
    "digest_of_digests",
    "load_undated_records",
    "render_undated_bounds",
    "seal_deliverables",
    "undated_bounds",
]
