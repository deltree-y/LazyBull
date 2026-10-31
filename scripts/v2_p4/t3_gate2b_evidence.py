# -*- coding: utf-8 -*-
"""P4-T3 门 2b：replay/回归边与工作树前后内容身份绑定（阶段专用）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v2_p2a.compare_replay_vs_b0 import compare_replay_vs_b0  # noqa: E402
from scripts.v2_p4.t1_worktree_digest import compute_worktree_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-batch", required=True)
    parser.add_argument(
        "--frozen-dir", default="data/walk_forward/batches/wf_batch_20260930_171221/raw"
    )
    parser.add_argument("--regression-verdict", required=True)
    parser.add_argument("--worktree-before", required=True)
    parser.add_argument("--worktree-after", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    new_batch = Path(args.new_batch)
    regression_path = Path(args.regression_verdict)
    before = _load(Path(args.worktree_before))
    after = _load(Path(args.worktree_after))
    current = compute_worktree_digest(ROOT)
    digest_keys = ("scope", "file_count", "sha256")
    digest_same = all(before.get(key) == after.get(key) for key in digest_keys)
    current_same = all(current.get(key) == after.get(key) for key in digest_keys)
    report = compare_replay_vs_b0(
        new_batch,
        Path(args.frozen_dir),
        migration_only_values={
            "merger_mode": "single_sleeve_passthrough",
            "merger_sleeve_id": "value",
        },
    )
    regression = _load(regression_path)
    batch_meta = _load(new_batch / "batch_meta.json")
    meta_notes = batch_meta.get("meta_notes", {})
    native_evidence_ok = (
        batch_meta.get("config", {}).get("merger_mode") == "single_sleeve_passthrough"
        and batch_meta.get("config", {}).get("merger_sleeve_id") == "value"
        and int(meta_notes.get("merger.call_count", 0)) > 0
        and int(meta_notes.get("merger.net_call_count", 0)) > 0
    )
    # Q6-B 原生模式登记键：只允许新侧多这两个键；其他配置/成交/净值差异不豁免。
    registered_gate3 = {
        "门3 多键: merger_mode（新侧有、冻结侧无）",
        "门3 多键: merger_sleeve_id（新侧有、冻结侧无）",
    }
    unexpected_gate3 = [diff for diff in report.gate3_diffs if diff not in registered_gate3]
    replay_ok = (
        native_evidence_ok
        and not unexpected_gate3
        and not report.gate4_diffs
        and not report.gate5_diffs
    )
    passed = replay_ok and regression.get("verdict") == "PASS" and digest_same and current_same
    payload = {
        "run_id": Path(args.out).parent.name,
        "verdict": "PASS" if passed else "FAIL",
        "new_batch": str(new_batch),
        "frozen_dir": args.frozen_dir,
        "replay": {
            "verdict": "PASS" if replay_ok else "FAIL",
            "gate3_diffs": report.gate3_diffs,
            "registered_gate3_diffs": sorted(registered_gate3 & set(report.gate3_diffs)),
            "unexpected_gate3_diffs": unexpected_gate3,
            "gate4_diffs": report.gate4_diffs,
            "gate5_diffs": report.gate5_diffs,
            "native_evidence_ok": native_evidence_ok,
            "merger_call_count": int(meta_notes.get("merger.call_count", 0)),
            "merger_net_call_count": int(meta_notes.get("merger.net_call_count", 0)),
            "native_evidence_order_count": int(
                meta_notes.get("merger.native_evidence_order_count", 0)
            ),
            "notes": report.notes,
            "batch_meta_sha256": _sha256(new_batch / "batch_meta.json"),
        },
        "regression_edge": regression,
        "regression_verdict_sha256": _sha256(regression_path),
        "worktree_binding": {
            "before": before,
            "after": after,
            "before_after_same": digest_same,
            "current_matches_after": current_same,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "out": str(out),
                "worktree": payload["worktree_binding"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
