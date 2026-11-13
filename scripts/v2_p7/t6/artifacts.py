# -*- coding: utf-8 -*-
"""P7 T6 证据落盘：``arms/{vvh,vvs}/*.parquet`` + 四件套 JSON + manifest。

落点（任务书 §9）
================
``data/paper_shadow/p7_testbed/<run_id>/``
├── manifest.json          身份 / 参数摘要 / 门结果 / 产物 SHA / 耗时（**最后写**）
├── arms/{vvh,vvs}/daily_*.parquet
├── metrics.json           收益/回撤/稳定性/链式指标（两臂 + 配对差）
├── calibration.json       §6.2 预登记判据（**运行前**落盘，含登记时刻）
├── verdict.json           verdict + 逐条判据证据
├── golden.json            §4.6 黄金例三条
└── identity.json          上游 SHA、参数摘要、重跑/续跑一致性

写盘纪律
========
- 只写 ``data/paper_shadow/p7_testbed/<run_id>/``（写域守卫在 runner 预检，拒绝零副作用）；
- 全部 JSON 走 ``scripts.v2_p7.common.write_json``（规范字节 + LF，SHA 可复核）；
- Parquet 嵌套列规范化为 JSON 字符串（与 T5 证据同一约定）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.common import sha256_file, write_json
from scripts.v2_p7.t4.runner import _flatten_row
from scripts.v2_p7.t6 import calibration
from scripts.v2_p7.t6.metrics import METRICS_SCHEMA
from scripts.v2_p7.t6.noise import NOISE_SCHEMA
from scripts.v2_p7.t6.params import T6_IDENTITY_SCHEMA, T6Params, params_digest

#: 臂目录
ARMS: Sequence[str] = ("vvh", "vvs")


def write_calibration(run_root: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """**运行前**落盘预登记（返回 SHA/字节；runner 必须先调用本函数）。"""
    digest = write_json(run_root / "calibration.json", dict(payload))
    return {
        "sha256": digest,
        "bytes": int((run_root / "calibration.json").stat().st_size),
        "file": "calibration.json",
    }


def write_arm_frames(
    run_root: Path, frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
) -> Dict[str, Dict[str, Any]]:
    """两臂帧落盘（daily / sleeve / trades）。"""
    written: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        directory = run_root / "arms" / arm
        directory.mkdir(parents=True, exist_ok=True)
        for kind in ("daily", "sleeve", "trades"):
            name = f"arms/{arm}/{kind}_{arm}.parquet"
            written[name] = _write_frames(directory / f"{kind}_{arm}.parquet", frames[arm][kind])
    return written


def write_payloads(
    *,
    run_root: Path,
    metrics: Mapping[str, Any],
    noise: Mapping[str, Any],
    verdict: Mapping[str, Any],
    golden: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """四件套 + 噪声带（并入 metrics.json，避免第二套口径文件）。"""
    metrics_payload = dict(metrics)
    metrics_payload["noise"] = dict(noise)
    metrics_payload["schema"] = METRICS_SCHEMA
    metrics_payload["noise_schema"] = NOISE_SCHEMA
    written: Dict[str, Dict[str, Any]] = {}
    written["metrics.json"] = _write_json_counted(run_root / "metrics.json", metrics_payload)
    written["verdict.json"] = _write_json_counted(run_root / "verdict.json", dict(verdict))
    written["golden.json"] = _write_json_counted(run_root / "golden.json", dict(golden))
    written["identity.json"] = _write_json_counted(run_root / "identity.json", dict(identity))
    return written


def build_identity(
    *,
    params: T6Params,
    run_id: str,
    run_root: Path,
    t6_modules: Mapping[str, Any],
    determinism: Mapping[str, Any],
    rerun_replay: Mapping[str, Any],
    timings: Mapping[str, Any],
) -> Dict[str, Any]:
    """身份件：上游 SHA + 本 task 模块 SHA + 参数摘要 + 预登记摘要 + 重跑一致性。"""
    self_sha = {
        str(key): str((value or {}).get("sha256") or "")
        for key, value in sorted(t6_modules.items())
    }
    return {
        "schema": T6_IDENTITY_SCHEMA,
        "run_id": run_id,
        "task": "P7 T6",
        "alpha_consumed": True,
        "params_digest": params_digest(params),
        "calibration": {
            "schema": calibration.CALIBRATION_SCHEMA,
            "digest": params.calibration_digest,
            "registered_date": calibration.REGISTERED_DATE,
        },
        "upstream": {
            "t5_root": params.t5_reference.payload(),
            "t25_input": params.t25.payload(),
            "t4_modules": {
                key: str((value or {}).get("sha256") or "")
                for key, value in sorted(params.t4_module_sha.items())
            },
            "t5_modules_baseline": {
                key: str((value or {}).get("sha256") or "")
                for key, value in sorted(params.t5_module_sha.items())
            },
        },
        "t6_modules": self_sha,
        "t6_module_digest": _digest(self_sha),
        "environment": environment_versions(),
        "prereg_stability": prereg_stability(run_root),
        "determinism": dict(determinism),
        "rerun_replay": dict(rerun_replay),
        "timings": dict(timings),
        "note": (
            "``t5_modules_baseline`` 是本 task 首采的 T5 模块 SHA（T5 完成根未登记自身 SHA，"
            "已登记缺口）；determinism = 同参独立重跑逐位比对；rerun_replay = 宿主内部重放抽样；"
            "prereg_stability = 同前缀全部 run 的预登记摘要（跨 run 是否改过阈值）"
        ),
    }


def environment_versions() -> Dict[str, Any]:
    """运行环境版本（bootstrap 与 parquet 读取的可复现性依赖）。"""
    import platform
    import sys

    import numpy
    import pandas

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "platform": platform.platform(),
    }


def write_verification(run_root: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    """落盘只读复算报告（``verify.json``；G4 的 ``artifact_recompute`` 引用它）。"""
    return _write_json_counted(run_root / "verify.json", dict(report))


def prereg_stability(run_root: Path) -> Dict[str, Any]:
    """扫描同前缀 run 的 ``calibration.json``，核对**跨 run 预登记摘要不变**。

    这只能证明「同前缀证据根之间的判据一致」，不能证明「阈值在第一次运行前就已冻结」——
    该残余边界已在 ``docs/reports/p7_t6_closure_20261113.md`` 登记为已知限制。
    """
    import json

    base = Path(run_root)
    prefix = base.name.split("_r")[0]
    rows: Dict[str, Any] = {}
    for path in sorted(base.parent.glob(f"{prefix}_r*")):
        calibration_path = path / "calibration.json"
        if not calibration_path.exists():
            continue
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - 防御分支
            rows[path.name] = {"status": "unreadable"}
            continue
        rows[path.name] = {
            "criteria_digest": str(payload.get("criteria_digest") or ""),
            "registered_at": str(payload.get("registered_at") or ""),
            "sha256": sha256_file(calibration_path),
        }
    digests = {item.get("criteria_digest") for item in rows.values() if item.get("criteria_digest")}
    return {
        "prefix": prefix,
        "runs": rows,
        "digests": sorted(str(item) for item in digests),
        "all_equal": bool(len(digests) <= 1),
        "covers_current_run": base.name in rows,
        "boundary": (
            "只能证明同前缀 run 之间预登记一致；「阈值在首跑前已冻结」由 "
            "calibration.json 的 mtime 顺序 + 任务书预登记文本共同保证，外部不可完全复核"
        ),
    }


def build_manifest(
    *,
    params: T6Params,
    run_id: str,
    run_root: Path,
    days: Sequence[str],
    checks: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    verdict: Mapping[str, Any],
    timings: Mapping[str, Any],
    calibration_sha: str,
    written_before_run: bool,
    created_at: str,
) -> Dict[str, Any]:
    """manifest（**最后写**；登记其余全部产物的 SHA 与行数）。"""
    outputs = {str(name): str(item.get("sha256") or "") for name, item in sorted(artifacts.items())}
    return {
        "schema": "p7_t6/v1",
        "run_id": run_id,
        "task": "P7 T6 — 两臂运行、绩效与三连指标经济裁定",
        "created_at": created_at,
        "alpha_consumed": True,
        "verdict": verdict.get("verdict"),
        "verdict_reason": verdict.get("verdict_reason"),
        "calibration": {
            "file": "calibration.json",
            "sha256": calibration_sha,
            "digest": params.calibration_digest,
            "written_before_run": bool(written_before_run),
            "written_before_run_source": "G7.pre_registration（文件 mtime 顺序），非字面量",
        },
        "params_digest": params_digest(params),
        "params": params.params_payload(),
        "calendar": {
            "window": [str(days[0]), str(days[-1])],
            "days": len(days),
            "right_censored": "全窗最后一个交易日之后不再推进（右删失如实登记）",
        },
        "checks": dict(checks),
        "arms": {
            "names": list(ARMS),
            "treatment": "vvs",
            "control": "vvh",
            "difference": "credit 开关（其余逐位相同）",
        },
        "artifacts": {str(name): dict(item) for name, item in sorted(artifacts.items())},
        "outputs": outputs,
        "timing": dict(timings),
        "write_scope": {
            "root": run_root.relative_to(params.repo_root).as_posix(),
            "forbidden": [
                "data/raw",
                "data/clean",
                "data/features",
                "data/models",
                "data/runs",
                "data/paper",
                "data/frozen_reference",
                "src/lazybull",
            ],
        },
        "boundary": [
            "verdict 只对本窗口、本机械与费用口径成立",
            "不构成阶段授权、B 区转正、生产多袖、前向/日历验证或实盘可用结论",
            "不得用事后挑选的窗口/子集支撑 verdict；判据与阈值运行前冻结（calibration.json）",
        ],
    }


def _write_json_counted(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    digest = write_json(path, dict(payload))
    return {"sha256": digest, "bytes": int(path.stat().st_size), "file": path.name}


def _write_frames(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_flatten_row(row) for row in rows])
    frame.to_parquet(path, index=False)
    return {
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "file": path.name,
    }


def dumps(payload: Mapping[str, Any]) -> str:
    """调试/日志用稳定序列化。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _digest(payload: Mapping[str, Any]) -> str:
    """模块 SHA 集合的稳定摘要（规范化 JSON 的 SHA256）。"""
    import hashlib

    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
