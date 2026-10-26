"""P3 T0.5b spike 证据构造与双跑判定（S11 拆文件）。

职责：证据骨架（观察身份顶层、生产者身份随各 run 分列）、双跑比对与判定、
判定结果日志。纯函数，无副作用（落盘由编排器负责）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger

from scripts.v2_p3.t05b_fingerprint import environment_identity


def _build_evidence(
    run_name: str,
    production_data_root: Path,
    baselines: Dict[str, object],
    config_digest: str,
    repo_root: Path,
) -> Dict[str, object]:
    """证据骨架（顶层为观察身份；生产者身份随各 run 分列）。"""
    from src.lazybull.v2.store.store_identity import workspace_identity

    return {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "production_data_root": str(production_data_root.resolve()),
        "spike_config": {
            "fold": "split 0（首折）",
            "seeds": [42, 61, 82],
            "runs": 2,
            "deploy_train": False,
            "deploy_off_reason": "Q10-B：P3 回放显式关闭 deploy 训练，生产部署训练留 P4 验证",
            "device_thread_candidate": "生产现役硬编码 tree_method=hist + device=cuda；"
            "n_jobs=-1 解析为本机全核（cpu_count=20 固定值，同机同码双跑即固定）",
            "config_digest": config_digest,
        },
        "observer_environment_identity": environment_identity(),
        "observer_workspace_identity": workspace_identity(repo_root),
        "baselines": {
            "guard_snapshot": str(baselines["guard_snapshot_path"]),
            "consumption_snapshot": str(baselines["consumption_snapshot_path"]),
            "consumption_counts": baselines["consumption_counts"],
            "consumption_files_count": baselines["consumption_files_count"],
            "input_snapshot_digest": baselines["input_snapshot_digest"],
            "resumed": baselines["resumed"],
        },
    }


def _finalize_comparison(
    runs_evidence: List[Dict[str, object]],
    guard_drift: List[str],
    consumption_drift: List[str],
) -> Dict[str, object]:
    """双跑比对与判定：指纹 / 覆盖 / 守卫 / 消费集合四门。"""
    fp1 = str(runs_evidence[0]["fingerprint"]["fingerprint"])  # type: ignore[index]
    fp2 = str(runs_evidence[1]["fingerprint"]["fingerprint"])  # type: ignore[index]
    seed_fp1 = [
        str(item["fingerprint"])
        for item in runs_evidence[0]["fingerprint"]["per_seed_diagnostic"]  # type: ignore[index]
    ]
    seed_fp2 = [
        str(item["fingerprint"])
        for item in runs_evidence[1]["fingerprint"]["per_seed_diagnostic"]  # type: ignore[index]
    ]
    fingerprint_match = fp1 == fp2
    seed_match = seed_fp1 == seed_fp2
    fp0 = runs_evidence[0]["fingerprint"]  # type: ignore[index]
    fp1_runs = runs_evidence[1]["fingerprint"]  # type: ignore[index]
    excluded1 = [str(day) for day in fp0.get("excluded_days") or []]
    excluded2 = [str(day) for day in fp1_runs.get("excluded_days") or []]
    excluded_match = excluded1 == excluded2
    coverage_match = (
        runs_evidence[0]["fingerprint"].get("score_days")  # type: ignore[index]
        == runs_evidence[1]["fingerprint"].get("score_days")  # type: ignore[index]
        and runs_evidence[0]["fingerprint"]["n_rows"]  # type: ignore[index]
        == runs_evidence[1]["fingerprint"]["n_rows"]  # type: ignore[index]
        and excluded_match
    )
    passed = fingerprint_match and coverage_match and not guard_drift and not consumption_drift
    return {
        "ensemble_fingerprint_run1": fp1,
        "ensemble_fingerprint_run2": fp2,
        "ensemble_fingerprint_match": fingerprint_match,
        "per_seed_fingerprint_match": seed_match,
        "coverage_match": coverage_match,
        "excluded_days_run1": excluded1,
        "excluded_days_run2": excluded2,
        "excluded_days_match": excluded_match,
        "guard_drift": guard_drift,
        "consumption_drift": consumption_drift,
        "verdict": "PASS" if passed else "FAIL",
    }


def _log_verdict(comparison: Dict[str, object]) -> None:
    """判定结果日志（停止规则文案与 §7.4 对齐）。"""
    if not comparison["ensemble_fingerprint_match"]:
        logger.error("确定性 spike 不通过：同码双跑预测指纹不一致 ⇒ 停止（§7.4-2）")
    if not comparison["coverage_match"]:
        logger.error("双跑覆盖不一致（打分日清单或行数不同）⇒ 停止（R2-T05B-03）")
    if comparison["guard_drift"]:
        logger.error(
            f"生产模型目录漂移 ⇒ 停止并回滚（§7.4-3）: "
            f"{comparison['guard_drift'][:5]}"  # type: ignore[index]
        )
    if comparison["consumption_drift"]:
        logger.error(
            f"消费集合漂移 ⇒ 停止（§13.3.4 唯一硬门）: "
            f"{comparison['consumption_drift'][:5]}"  # type: ignore[index]
        )
    if comparison["verdict"] == "PASS":
        fp = str(comparison["ensemble_fingerprint_run1"])
        logger.info(f"确定性 spike 通过：指纹逐位一致（{fp[:16]}…），守卫与消费集合零漂移")
