# -*- coding: utf-8 -*-
"""P7 T6 编排器：预登记落盘 → 两臂运行 → 逐日序列 → 指标/噪声 → 裁定 → 门 → 证据。

顺序（**写死**）
==============
1. 冻结输入校验（``params.build_params``；漂移即停，早于任何写盘）；
2. 写域预检（``common.assert_targets_within_write_roots``；拒绝路径零副作用）；
3. **预登记落盘**（``calibration.json``，含 ``registered_at``；必须先于两臂运行）；
4. 两臂全窗运行（VV-H / VV-S，除 ``credit`` 外逐位同参）；
5. 逐日 / 逐袖 / 交易帧；
6. 同参重跑（G5 确定性）；
7. 指标 + 噪声带（配对块 bootstrap）+ 分年/分月序列；
8. 经济裁定（C1–C5，预登记判据）；
9. 黄金例（§4.6 三条）；
10. 证据落盘（帧 + 四件套）→ 机械门 G1–G7 → manifest（最后写）。

边界
====
``verdict`` 只对本窗口、本机械与费用口径成立；不构成阶段授权、B 区转正、
生产多袖能力、前向/日历验证或实盘可用结论。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from loguru import logger

from scripts.v2_p7.common import (
    REPO_ROOT,
    assert_targets_within_write_roots,
    resolve_write_roots,
    write_json,
)
from scripts.v2_p7.t6 import SCHEMA, TASK, calibration
from scripts.v2_p7.t6.arms import determinism_check, intent_index, run_arms
from scripts.v2_p7.t6.artifacts import (
    ARMS,
    build_identity,
    build_manifest,
    write_arm_frames,
    write_calibration,
    write_payloads,
    write_verification,
)
from scripts.v2_p7.t6.daily import build_arm_frames
from scripts.v2_p7.t6.gates import build_gates, stop_on_hard_failures
from scripts.v2_p7.t6.golden import build_golden
from scripts.v2_p7.t6.metrics import build_metrics
from scripts.v2_p7.t6.noise import build_noise
from scripts.v2_p7.t6.params import (
    DEFAULT_RUN_ID,
    T6_MODULE_GLOB,
    TESTBED_REL,
    T6Params,
    build_params,
    module_identity,
    params_digest,
)
from scripts.v2_p7.t6.verdict import build_verdict
from scripts.v2_p7.t6.verify import verify_run


class RunnerError(RuntimeError):
    """T6 编排前置条件或硬门不满足（禁止发布）。"""


def run(
    repo_root: Path = REPO_ROOT,
    run_id: str = DEFAULT_RUN_ID,
) -> Dict[str, Any]:
    """全窗两臂 → 指标 → 裁定 → 证据（返回 manifest）。"""
    started = time.time()
    params = build_params(repo_root)
    days = tuple(params.extended_days)
    run_root = Path(repo_root) / TESTBED_REL / run_id
    _prepare_run_root(run_root)
    registration = _register_calibration(params=params, run_root=run_root, run_id=run_id)

    analysis = _analyse(params=params, days=days, run_id=run_id, run_root=run_root)
    written = _write_evidence(
        run_root=run_root,
        params=params,
        run_id=run_id,
        analysis=analysis,
        calibration_sha=registration["calibration_sha"],
    )
    verification = verify_run(run_root)
    written["verify.json"] = write_verification(run_root, verification)
    gates = build_gates(
        params=params,
        frames=analysis["frames"],
        results=analysis["results"],
        metrics=analysis["metrics"],
        determinism=analysis["determinism"],
        golden=analysis["golden"],
        artifacts=written,
        run_root=run_root,
        verification=verification,
        prereg=analysis["identity"].get("prereg_stability") or {},
    )
    stop_on_hard_failures(gates)
    manifest = _publish_manifest(
        params=params,
        run_id=run_id,
        run_root=run_root,
        days=days,
        analysis=analysis,
        written=written,
        gates=gates,
        registration=registration,
        elapsed=round(time.time() - started, 1),
    )
    logger.info(
        f"T6 证据已落盘：{run_root}（{manifest['timing']['total_seconds']}s；"
        f"verdict={manifest['verdict']}；门={gates['summary']['passed']}/7）"
    )
    return manifest


# ────────────────────────── 阶段 ──────────────────────────


def _register_calibration(*, params: T6Params, run_root: Path, run_id: str) -> Dict[str, Any]:
    """**运行前**落盘预登记（运行后修改即视为无效裁定）。"""
    registered_at = datetime.now().isoformat(timespec="seconds")
    payload = dict(calibration.payload())
    payload["registered_at"] = registered_at
    payload["run_id"] = run_id
    payload["criteria_digest"] = params.calibration_digest
    written = write_calibration(run_root, payload)
    run_started_at = datetime.now().isoformat(timespec="seconds")
    logger.info(f"T6 预登记已落盘（{registered_at}）：{run_root}/calibration.json")
    return {
        "registered_at": registered_at,
        "run_started_at": run_started_at,
        "calibration_sha": written,
        "payload_digest": params.calibration_digest,
    }


def _analyse(
    *, params: T6Params, days: Sequence[str], run_id: str, run_root: Path
) -> Dict[str, Any]:
    """两臂 → 帧 → 确定性 → 指标/噪声 → 裁定 → 黄金例 → 身份摘要。"""
    results, arm_seconds = _timed(lambda: run_arms(params=params, days=days, run_id=run_id))
    frames, frames_seconds = _timed(
        lambda: {arm: build_arm_frames(results[arm], params) for arm in ARMS}
    )
    determinism, determinism_seconds = _timed(
        lambda: determinism_check(params=params, days=days, first=results, run_id=run_id)
    )
    if not determinism["ok"]:
        raise RunnerError(f"G5 确定性失败，禁止发布：{determinism['failures']}")
    metrics, metrics_seconds = _timed(lambda: build_metrics(params=params, frames=frames))
    noise = _noise(params=params, frames=frames, metrics=metrics)
    verdict = build_verdict(calibration=calibration.payload(), metrics=metrics, noise=noise)
    golden = _golden(params=params, frames=frames, results=results)
    timings = {
        "arm_seconds": arm_seconds,
        "frames_seconds": frames_seconds,
        "determinism_seconds": determinism_seconds,
        "metrics_seconds": metrics_seconds,
    }
    identity = build_identity(
        params=params,
        run_id=run_id,
        run_root=run_root,
        t6_modules=module_identity(params.repo_root, T6_MODULE_GLOB),
        determinism=determinism,
        rerun_replay={arm: dict(results[arm].replay) for arm in ARMS},
        timings=timings,
    )
    return {
        "results": results,
        "frames": frames,
        "determinism": determinism,
        "metrics": metrics,
        "noise": noise,
        "verdict": verdict,
        "golden": golden,
        "identity": identity,
        "timings": timings,
    }


def _noise(
    *,
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """配对块 bootstrap 噪声带（点估计取自 ``metrics.paired``，与裁定同源）。"""
    del params
    paired = metrics["paired"]
    return build_noise(
        nav={arm: [float(row["nav"]) for row in frames[arm]["daily"]] for arm in ARMS},
        split_index=[int(row["split_index"]) for row in frames["vvh"]["daily"]],
        point_deltas={
            "total_return": float(paired["total_return"]["delta"]),
            "max_drawdown": float(paired["max_drawdown"]["delta"]),
            "cagr": float(paired["cagr"]["delta"]),
            "volatility": float(paired["volatility"]["delta"]),
            "sharpe": float(paired["sharpe"]["delta"]),
        },
        calibration=calibration.payload(),
        deltas_by_year=[float(row["delta"]) for row in metrics["by_year_paired"]],
        deltas_by_month=[float(row["delta"]) for row in metrics["monthly_paired"]],
    )


def _golden(
    *,
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    results: Mapping[str, Any],
) -> Dict[str, Any]:
    golden = build_golden(
        params=params, frames=frames, results=results, buy_index=intent_index(params)[0]
    )
    if not golden["ok"]:
        raise RunnerError(f"G6 黄金例未通过，禁止发布：{golden['failed']}")
    return golden


def _write_evidence(
    *,
    run_root: Path,
    params: T6Params,
    run_id: str,
    analysis: Mapping[str, Any],
    calibration_sha: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """帧 + 四件套落盘（返回逐件 SHA/行数；manifest 不在其中）。"""
    del params, run_id
    written: Dict[str, Dict[str, Any]] = {}
    written.update(write_arm_frames(run_root, analysis["frames"]))
    written.update(
        write_payloads(
            run_root=run_root,
            metrics=analysis["metrics"],
            noise=analysis["noise"],
            verdict=analysis["verdict"],
            golden=analysis["golden"],
            identity=analysis["identity"],
        )
    )
    written["calibration.json"] = dict(calibration_sha)
    return written


def _publish_manifest(
    *,
    params: T6Params,
    run_id: str,
    run_root: Path,
    days: Sequence[str],
    analysis: Mapping[str, Any],
    written: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
    registration: Mapping[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    """manifest（**最后写**；登记其余全部产物的 SHA 与行数）。"""
    timings = dict(analysis["timings"])
    timings.update(
        {
            "total_seconds": elapsed,
            "calibration_registered_at": registration["registered_at"],
            "run_started_at": registration["run_started_at"],
        }
    )
    manifest = build_manifest(
        params=params,
        run_id=run_id,
        run_root=run_root,
        days=days,
        checks=_checks_payload(
            params=params,
            gates=gates,
            verdict=analysis["verdict"],
            metrics=analysis["metrics"],
            noise=analysis["noise"],
        ),
        artifacts=written,
        verdict=analysis["verdict"],
        timings=timings,
        calibration_sha=str(registration["calibration_sha"]["sha256"]),
        written_before_run=bool((gates.get("G7") or {}).get("pre_registration", {}).get("ok")),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    manifest["schema"] = SCHEMA
    manifest["task"] = TASK
    write_json(run_root / "manifest.json", manifest)
    return manifest


def _timed(action: Callable[[], Any]) -> Tuple[Any, float]:
    """计时包装（返回 ``(结果, 秒)``；只用于 manifest 的耗时登记）。"""
    started = time.time()
    value = action()
    return value, round(time.time() - started, 1)


def _checks_payload(
    *,
    params: T6Params,
    gates: Mapping[str, Any],
    verdict: Mapping[str, Any],
    metrics: Mapping[str, Any],
    noise: Mapping[str, Any],
) -> Dict[str, Any]:
    """manifest 内的门与裁定摘要（明细见 verdict.json / metrics.json）。"""
    return {
        "gates": {key: gates[key] for key in ("G1", "G2", "G3", "G4", "G5", "G6", "G7")},
        "gates_summary": dict(gates["summary"]),
        "verdict": verdict["verdict"],
        "verdict_reason": verdict["verdict_reason"],
        "criteria": {
            item["id"]: {
                "passed": bool(item["passed"]),
                "over_band": bool(item.get("over_band", False)),
            }
            for item in verdict["criteria"]
        },
        "paired_delta": {
            "total_return": float(metrics["paired"]["total_return"]["delta"]),
            "max_drawdown": float(metrics["paired"]["max_drawdown"]["delta"]),
            "cagr": float(metrics["paired"]["cagr"]["delta"]),
            "volatility": float(metrics["paired"]["volatility"]["delta"]),
        },
        "noise_band": {
            "method": noise["method"],
            "block": noise["block"],
            "replicates": noise["replicates"],
            "seed": noise["seed"],
            "primary_excludes_zero": {
                key: bool(value["excludes_zero"])
                for key, value in noise["primary"]["metrics"].items()
            },
        },
        "params_digest": str(params_digest(params)),
    }


def _prepare_run_root(run_root: Path) -> None:
    """写域预检 + 已存在根只读（拒绝路径零副作用）。"""
    roots = resolve_write_roots()
    assert_targets_within_write_roots([run_root], roots)
    if run_root.exists():
        raise FileExistsError(f"输出根已存在（禁止覆盖；整改请用新 run_id）：{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)


__all__: Sequence[str] = ("run", "RunnerError")
