# -*- coding: utf-8 -*-
"""P-1 O1 读数重算（R05 修正；**只读**源证据根，写入全新根）。

背景
====
P7 整体检视（2026-11-15）R05 指出旧 O1 有四处失真：回款用包络中点 ``d_point``、
按 ``signal_pos+1`` 占资、忽略 ``holding=OPEN``、R3 按逐笔固定 min_ticket 分位。
本脚本用**已冻结的逐意图路径表**（``intents/intents.parquet``，含
``fill/holding/entry_pos/exit_pos/event_net``）重算约束账户读数，不重跑 P-1、
不改任何 envelope / verdict / B0 资源决定。

边界
====
- ``decision_role=readings_only``、``auto_veto=false``、``alpha_consumed=false``；
- 旧根只读；输出根已存在即拒绝（新 run 身份重取）；
- 不重开 B0、不改统计 verdict，只更正 O1 读数。

用法
====
``python scripts/v2_p7/t1/o1_recompute.py --out data/frozen_reference/v2_p7/<new_root>``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p7.t1.p1_o1 import constraint_readings  # noqa: E402

#: 默认源证据根（R05 检视对象）
DEFAULT_SOURCE = "data/frozen_reference/v2_p7/p7_p1r_20261114_r6"

#: 参与读数口径的源码（身份登记）
CODE_FILES = (
    "scripts/v2_p7/t1/p1_o1.py",
    "scripts/v2_p7/t1/p1_bounds.py",
    "scripts/v2_p7/t1/p1_params.py",
)

SCHEMA = "p7_p1r_o1_recompute/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _namespaces(frame: pd.DataFrame) -> tuple[Any, Any]:
    intents = SimpleNamespace(
        stock=frame["stock"].to_numpy(dtype=np.int64),
        signal_pos=frame["signal_pos"].to_numpy(dtype=np.int64),
        source_code=frame["source_code"].to_numpy(dtype=np.int64),
        size=int(len(frame)),
    )
    result = SimpleNamespace(
        fill=frame["fill"].to_numpy(dtype=np.int64),
        holding=frame["holding"].to_numpy(dtype=np.int64),
        entry_pos=frame["entry_pos"].to_numpy(dtype=np.int64),
        exit_pos=frame["exit_pos"].to_numpy(dtype=np.int64),
        event_net=frame["event_net"].to_numpy(dtype=np.float64),
        d_point=frame["d_point"].to_numpy(dtype=np.float64),
    )
    return intents, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P-1 O1 读数重算（R05）")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="源证据根（只读）")
    parser.add_argument("--out", required=True, help="新输出根（禁止覆盖）")
    parser.add_argument("--cutoff-pos", type=int, default=None, help="截止面板位置（默认取源 n_days−1）")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(f"输出根已存在（禁止覆盖）：{out}；整改请用新根")

    started = time.time()
    params_payload = json.loads((source / "params.json").read_text(encoding="utf-8"))
    noise = json.loads((source / "noise_fidelity.json").read_text(encoding="utf-8"))
    default_cutoff = int(noise["actual"]["n_days"]) - 1
    cutoff_pos = int(args.cutoff_pos if args.cutoff_pos is not None else default_cutoff)
    intents_path = source / "intents" / "intents.parquet"
    frame = pd.read_parquet(intents_path)
    required = {
        "stock",
        "signal_pos",
        "source_code",
        "fill",
        "holding",
        "entry_pos",
        "exit_pos",
        "event_net",
        "d_point",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"源 intents 缺列：{missing}")

    intents, result = _namespaces(frame)
    params = SimpleNamespace(
        initial_capital=float(params_payload["initial_capital"]),
        min_ticket=float(params_payload["min_ticket"]),
        max_concurrent_positions=int(params_payload["max_concurrent_positions"]),
        horizon_primary=int(params_payload["horizon_primary"]),
    )
    readings = constraint_readings(intents, result, params, cutoff_pos=cutoff_pos)

    old_path = source / "o1" / "readings.json"
    previous = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    payload = {
        "schema": SCHEMA,
        "run_id": out.name,
        "task": "P-1 O1 读数重算（R05 修正）",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(source.relative_to(ROOT).as_posix()),
        "source_intents_sha256": _sha256(intents_path),
        "source_params_sha256": _sha256(source / "params.json"),
        "prereg_id": params_payload["prereg_id"],
        "prereg_sha256": params_payload["prereg_sha256"],
        "criteria_digest": params_payload["criteria_digest"],
        "code_sha256": {name: _sha256(ROOT / name) for name in CODE_FILES},
        "cutoff_pos": cutoff_pos,
        "readings": readings,
        "old_readings": previous,
        "delta": _delta(previous, readings),
        "boundaries": {
            "readings_only": True,
            "auto_veto": False,
            "alpha_consumed": False,
            "does_not_reopen_b0": True,
            "does_not_change_verdict": True,
            "note": (
                "O1 只报不判：本根只更正资金利用率/锁定/终局读数，不产生任何通过、"
                "否决或 B0 复活结论；统计 verdict 仍以源根为准。"
            ),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    readings_sha = _write_json(out / "readings.json", payload)
    manifest = {
        "schema": "p7_p1r_o1_manifest/v1",
        "run_id": out.name,
        "created_at": payload["created_at"],
        "source_run": payload["source_run"],
        "source_intents_sha256": payload["source_intents_sha256"],
        "code_sha256": payload["code_sha256"],
        "written_files_sha256": {"readings.json": readings_sha},
        "recompute_rule": (
            "回款 = event_net；建仓 = entry_pos；释放仅 holding=CLOSED；"
            "R3 = 逐日超期仓位市值/本金 95 分位"
        ),
        "not_forward_evidence": True,
        "not_real_clock_evidence": True,
        "not_production_authorization": True,
    }
    manifest_sha = _write_json(out / "manifest.json", manifest)
    verify = {
        "schema": "p7_p1r_o1_verify/v1",
        "readings_sha256_match": _sha256(out / "readings.json") == readings_sha,
        "manifest_sha256": manifest_sha,
        "files": sorted(path.name for path in out.iterdir() if path.is_file()),
    }
    _write_json(out / "verify.json", verify)
    print(json.dumps(readings, ensure_ascii=False, indent=2))
    return 0


def _delta(previous: Mapping[str, Any], readings: Mapping[str, Any]) -> Dict[str, Any]:
    """新旧读数对照（只登记差异，不解释为经济结论）。"""
    if not previous:
        return {"note": "源根无 O1 读数"}
    keys = (
        "R1_uncapped_capital_demand",
        "R2_mean_capital_utilization",
        "R3_overdue_lock_share_p95",
        "R4_exit_delay_p95_days",
        "R5_terminal_loss_stress",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        old = previous.get(key)
        new = readings.get(key)
        out[key] = {
            "old": old,
            "new": new,
            "delta": (None if old is None or new is None else float(new) - float(old)),
        }
    old_counts = dict(previous.get("counts") or {})
    new_counts = dict(readings.get("counts") or {})
    out["counts"] = {
        key: {"old": old_counts.get(key), "new": new_counts.get(key)}
        for key in sorted(set(old_counts) | set(new_counts))
    }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
