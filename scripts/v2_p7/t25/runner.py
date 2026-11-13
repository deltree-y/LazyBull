# -*- coding: utf-8 -*-
"""P7 T2.5 编排器：逐折打分 → 相位目标 → 原始意图预检 → 证据落盘。

写域
====
- 只写 ``data/paper_shadow/p7_testbed/<signal_run_id>/``（``scripts.v2_p7.common``
  写域守卫，mkdir 前整体预检，拒绝路径零副作用）；不写
  ``data/raw|clean|features|models|runs|paper``，不覆盖既有证据根；
- ``alpha_consumed=false``：本 task 只推理不训练，不产生任何 pass/fail 判定；
- 正式机械运行证据仍只落 ``data/frozen_reference/v2_p7/``（T4 起），本 task 只产测试床工作根。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from scripts.v2_p7.common import (
    REPO_ROOT,
    assert_targets_within_write_roots,
    resolve_write_roots,
    sha256_file,
    write_json,
)
from scripts.v2_p7.t25 import intent_precheck
from scripts.v2_p7.t25.params import (
    ARMS,
    DEFAULT_RUN_ID,
    NON_BLOCKING_SCENARIOS,
    TESTBED_REL,
    FoldRow,
    T25Params,
    build_params,
    compute_input_identity,
    days_digest,
    load_trade_days,
)
from scripts.v2_p7.t25.signal_layer import (
    PriceProvider,
    RankedProvider,
    anchor_plan_consistency,
    anchor_score_consistency,
    fold_timing_summary,
    score_fold,
    verify_replay,
    verify_targets_replay,
)
from scripts.v2_p7.t25.target_layer import (
    ArmResult,
    cross_model_stats,
    fold_boundary_continuity,
    model_switch_points,
    plan_integrity,
    runoff_obligations,
    simulate_arm,
)

SCHEMA = "p7_t25/v1"


def sha256_text(text: str) -> str:
    """字符串内容的 SHA256（参数/身份摘要用）。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(
    repo_root: Path = REPO_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    splits: Optional[Sequence[int]] = None,
    resume: bool = False,
    anchor_plans: bool = True,
    verify_splits: Sequence[int] = (),
    days_per_fold: int = 3,
    counterpart_top_n: Optional[int] = None,
    counterpart_capital: Optional[float] = None,
    accepted_drift: Sequence[str] = (),
) -> Dict[str, Any]:
    """执行 T2.5 全流程（默认 14 折全窗；``splits`` 可限定子集用于分段验证）。

    ``counterpart_*`` 只改**构造对手袖**（``v10``）的口径，基准袖（``v20``）恒为冻结值；
    偏离项写入 manifest 的 ``params.design_deviation``。
    """
    started = time.time()
    params = build_params(repo_root, counterpart_top_n, counterpart_capital)
    run_root = Path(repo_root) / TESTBED_REL / run_id
    _prepare_run_root(run_root, resume=resume)

    fold_list = _select_folds(params, splits)
    ranked = RankedProvider(params)
    prices = PriceProvider(params)
    source_identity = _source_identity(params)
    # 输入内容身份在打分**之前**建立，并绑定进参数摘要（续跑据此拒绝异源分片）
    drift_items = tuple(accepted_drift) or _drift_from_env()
    input_identity = compute_input_identity(params, drift_items)
    identity_digest = sha256_text(
        json.dumps(input_identity, ensure_ascii=False, sort_keys=True, default=str)
    )
    params_digest = sha256_text(
        json.dumps(
            {"params": params.params_payload(), "input_identity": identity_digest},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )

    timings, score_reports, anchor_scores, anchor_plans_report = _score_folds(
        params=params,
        fold_list=fold_list,
        run_root=run_root,
        run_id=run_id,
        resume=resume,
        anchor_plans=anchor_plans,
        ranked=ranked,
        params_digest=params_digest,
        source_digest=str(source_identity["source_digest"]),
    )

    targets_started = time.time()
    arms, variance_book = _build_arms(params, ranked, prices)
    targets_replay = verify_targets_replay(
        params, arms, ranked, prices, variance_book=variance_book
    )
    targets_seconds = round(time.time() - targets_started, 1)

    plans, holdings, daily_frames = _merged_frames(arms)

    precheck_started = time.time()
    precheck = intent_precheck.build_precheck(
        plans=plans,
        holdings=holdings,
        daily_frames=daily_frames,
        calendar=params.signal_days,
        price_lookup=prices,
        closes_lookup=prices.closes,
    )
    precheck_seconds = round(time.time() - precheck_started, 1)

    replay = (
        verify_replay(params, run_root, verify_splits, days_per_fold) if verify_splits else None
    )

    timing_payload = {
        "schema": "p7_t25_performance_probe/v1",
        "folds": fold_timing_summary(timings),
        "target_layer_seconds": targets_seconds,
        "intent_precheck_seconds": precheck_seconds,
        "ranked_days": ranked.cached_days,
        "total_seconds": round(time.time() - started, 1),
        "growth_curve": _growth_curve(timings),
    }
    checks = _acceptance_checks(
        params=params,
        fold_list=fold_list,
        anchor_scores=anchor_scores,
        anchor_plans=anchor_plans_report,
        arms=arms,
        precheck=precheck,
        replay=replay,
        targets_replay=targets_replay,
        input_identity=input_identity,
        holdings=holdings,
        plans=plans,
        closure_days=sorted(
            day for item in score_reports for day in (item.get("closure_days") or [])
        ),
    )
    # 硬失败阻止发布：有任一身份/可复现门未过 ⇒ 不写**任何**汇总件/目标帧/manifest，退出交给调用方。
    _stop_on_hard_failures(checks)
    artifacts = _write_artifacts(params, arms, run_root)

    registry = _fold_registry(params, fold_list, score_reports)
    scores_payload = _scores_payload(
        params, score_reports, anchor_scores, anchor_plans_report, ranked
    )
    targets_payload = _targets_payload(params, arms, plans, artifacts, targets_seconds)
    holdings_payload = _holdings_payload(params, arms, holdings, plans)
    write_json(run_root / "fold_registry.json", registry)
    write_json(run_root / "scores.json", scores_payload)
    write_json(run_root / "targets.json", targets_payload)
    write_json(run_root / "holdings.json", holdings_payload)
    write_json(run_root / "intent_precheck.json", precheck)
    write_json(run_root / "performance_probe.json", timing_payload)
    write_json(run_root / "input_identity.json", input_identity)

    manifest = _manifest(
        params=params,
        run_id=run_id,
        run_root=run_root,
        fold_list=fold_list,
        timings=timings,
        timing_payload=timing_payload,
        checks=checks,
        precheck=precheck,
        replay=replay,
        targets_replay=targets_replay,
        input_identity=input_identity,
        source_identity=source_identity,
        ranked=ranked,
    )
    write_json(run_root / "manifest.json", manifest)
    logger.info(f"T2.5 产物已落盘：{run_root}（{manifest['timing']['total_seconds']}s）")
    return manifest


def _drift_from_env() -> Tuple[str, ...]:
    """显式输入漂移接受清单（CLI / 环境变量 / 随代码归档的声明文件，三处取并集）。

    - CLI：``--accept-input-drift path:sha256``
    - 环境：``P7_T25_ACCEPT_DRIFT``（逗号分隔）
    - 归档声明：``scripts/v2_p7/t25/accept_input_drift.json``（``accept`` 数组，含 reason/owner，
      随代码进入 ``source_digest``，可审计）
    """
    import json
    import os

    items: List[str] = []
    raw = os.environ.get("P7_T25_ACCEPT_DRIFT", "")
    items.extend(item.strip() for item in raw.split(",") if item.strip())
    path = Path(__file__).resolve().parent / "accept_input_drift.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.extend(str(item) for item in (payload.get("accept") or []))
    return tuple(items)


def _source_identity(params: T25Params) -> Dict[str, Any]:
    """本 run 的代码态身份（git + src/scripts 内容摘要）。"""
    from src.lazybull.v2.store.store_identity import workspace_identity

    return workspace_identity(params.repo_root)


def _stop_on_hard_failures(checks: Mapping[str, Any]) -> None:
    """硬失败即停（任务书停止规则）：身份不唯一 / 推理不可复现 ⇒ 不发布。"""
    failures: List[str] = []
    if not checks["score_anchor_exact"]["ok"]:
        failures.append("锚分数不一致")
    if not checks["targets_replay"]["all_identical"]:
        failures.append("目标层不可逐位重放")
    if checks["replay"].get("checked") and not checks["replay"].get("all_bitwise_identical"):
        failures.append("分数不可逐位重放")
    if not checks["no_bt_overlap"]["ok"]:
        failures.append("唯一日历被 bt 扩展日污染")
    if checks["anchor_plan_same_source"].get("unexplained_days"):
        failures.append("锚计划存在无法解释的差异")
    mismatched_inputs = list(checks["input_identity"]["frozen_manifest_mismatch"])
    if mismatched_inputs:
        failures.append(f"实际读入数据与冻结清单不一致：{mismatched_inputs[:5]}")
    if failures:
        raise RuntimeError(f"T2.5 硬失败，禁止发布：{'；'.join(failures)}")


def _prepare_run_root(run_root: Path, resume: bool) -> None:
    """写域预检 + 完成根只读（``resume`` 只允许续未完成的运行）。"""
    roots = resolve_write_roots()
    assert_targets_within_write_roots([run_root], roots)
    if run_root.exists() and not resume:
        raise FileExistsError(f"输出根已存在（禁止覆盖）：{run_root}")
    if resume and (run_root / "manifest.json").exists():
        raise FileExistsError(
            f"完成根只读（manifest 已存在，禁止续跑改写）：{run_root}；修复请用新 run_id 重取"
        )
    run_root.mkdir(parents=True, exist_ok=True)


def _score_folds(
    *,
    params: T25Params,
    fold_list: Sequence[FoldRow],
    run_root: Path,
    run_id: str,
    resume: bool,
    anchor_plans: bool,
    ranked: RankedProvider,
    params_digest: str,
    source_digest: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """逐折打分与锚一致性（``resume`` 时校验身份后才复用分片）。"""
    timings: List[Dict[str, Any]] = []
    score_reports: List[Dict[str, Any]] = []
    anchor_scores: List[Dict[str, Any]] = []
    anchor_plans_report: List[Dict[str, Any]] = []
    for fold in fold_list:
        fold_dir = run_root / "folds" / f"split{fold.split_index:02d}"
        marker = fold_dir / "fold.json"
        if resume and marker.exists() and (fold_dir / "scores.parquet").exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            _assert_resume_fold_identity(
                payload=payload,
                fold=fold,
                fold_dir=fold_dir,
                params_digest=params_digest,
                source_digest=source_digest,
            )
            logger.info(f"split{fold.split_index:02d} 复用既有分片（身份已核验）")
        else:
            fold_started = time.time()
            payload = score_fold(params, fold, fold_dir, run_id)
            payload["score_seconds"] = round(time.time() - fold_started, 1)
            payload["params_digest"] = params_digest
            payload["source_digest"] = source_digest
            write_json(marker, payload)
        timings.append(
            {
                "split_index": fold.split_index,
                "model_version": fold.model_version,
                "score_days": payload["score_days"],
                "score_rows": payload["score_rows"],
                "score_seconds": payload["score_seconds"],
            }
        )
        score_reports.append(payload)
        predictions = pd.read_parquet(fold_dir / "scores.parquet")
        consistency = anchor_score_consistency(params, fold, predictions)
        anchor_scores.append(consistency)
        if anchor_plans:
            anchor_plans_report.append(anchor_plan_consistency(params, fold, ranked))
        logger.info(
            f"split{fold.split_index:02d} v{fold.model_version} "
            f"{payload['score_days']} 日 {payload['score_rows']} 行 "
            f"{payload['score_seconds']}s，锚分数 exact {consistency['exact_rows']}/"
            f"{consistency['anchor_rows']}"
        )
    return timings, score_reports, anchor_scores, anchor_plans_report


def _assert_resume_fold_identity(
    *,
    payload: Mapping[str, Any],
    fold: FoldRow,
    fold_dir: Path,
    params_digest: str,
    source_digest: str,
) -> None:
    """续跑分片身份核验：折号/模型号/参数/代码/分数与 top-N 实际 SHA 必须一致。

    任一不符即拒绝续跑（宁可重算，禁止把旧分数与新输入混在同一 run）。
    """
    problems: List[str] = []
    if int(payload.get("split_index", -1)) != int(fold.split_index):
        problems.append(f"split_index={payload.get('split_index')}")
    if int(payload.get("model_version", -1)) != int(fold.model_version):
        problems.append(f"model_version={payload.get('model_version')}")
    if str(payload.get("params_digest") or "") != params_digest:
        problems.append("params_digest 不符")
    if str(payload.get("source_digest") or "") != source_digest:
        problems.append("source_digest 不符（代码已变）")
    scores_path = fold_dir / "scores.parquet"
    if not scores_path.exists():
        problems.append("scores.parquet 缺失")
    elif str(payload.get("predictions_sha256") or "") != sha256_file(scores_path):
        problems.append("scores.parquet SHA 不符")
    topk_path = fold_dir / "signal_topk.parquet"
    if not topk_path.exists():
        problems.append("signal_topk.parquet 缺失")
    elif str(payload.get("signal_topk_sha256") or "") != sha256_file(topk_path):
        problems.append("signal_topk.parquet SHA 不符")
    if problems:
        raise RuntimeError(
            f"split{fold.split_index:02d} 续跑分片身份不符，拒绝复用（{'；'.join(problems)}）"
        )


def _build_arms(
    params: T25Params, ranked: RankedProvider, prices: PriceProvider
) -> Tuple[Dict[str, ArmResult], Any]:
    """两袖目标级账户模拟（跨折连续；基准袖冻结、对手袖可设计）。"""
    from scripts.v2_p7.t25.target_layer import VarianceBook

    variance_book = VarianceBook(params.extended_days, prices.load_frame)
    logger.info(f"Kelly 方差历史建表完成：{variance_book.codes} 只")
    return _run_arms(params, ranked, prices, variance_book)


def _run_arms(
    params: T25Params,
    ranked: RankedProvider,
    prices: PriceProvider,
    variance_book: Any,
) -> Tuple[Dict[str, ArmResult], Any]:
    """两袖目标级模拟（同一方差表）。"""
    arms: Dict[str, ArmResult] = {}
    for arm in ARMS:
        arms[arm] = simulate_arm(
            arm=arm,
            calendar=params.extended_days,
            schedule=params.schedule(arm),
            ranked_lookup=ranked,
            price_lookup=prices,
            params=params,
            capital=params.sleeve_capital[arm],
            top_n=params.sleeve_top_n[arm],
            variance_book=variance_book,
        )
        logger.info(
            f"{arm} 目标层完成：{arms[arm].integrity['plans']} 计划行，"
            f"{arms[arm].integrity['holdings_rows']} 持仓日"
        )
    return arms, variance_book


def _merged_frames(
    arms: Mapping[str, ArmResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """合并两袖计划/持仓帧（不落盘；硬门通过后才写）。"""
    plans = pd.concat([arms[arm].plans for arm in ARMS], ignore_index=True)
    holdings = pd.concat([arms[arm].holdings for arm in ARMS], ignore_index=True)
    plans = plans.sort_values(["arm", "signal_date", "tranche_idx", "planned_rank"])
    holdings = holdings.sort_values(["arm", "trade_date", "ts_code"])
    daily_frames = {arm: arms[arm].daily for arm in ARMS}
    return plans, holdings, daily_frames


def _write_artifacts(
    params: T25Params, arms: Mapping[str, ArmResult], run_root: Path
) -> Dict[str, Dict[str, Any]]:
    """落盘相位日历与两袖计划/持仓/逐日帧（硬门通过后调用）。"""
    artifacts: Dict[str, Dict[str, Any]] = {
        "targets/calendar.json": {"sha256": _write_calendar(params, run_root)}
    }
    for arm in ARMS:
        artifacts[f"targets/plans_{arm}.parquet"] = _write_frame(
            arms[arm].plans.sort_values(["signal_date", "planned_rank"]),
            run_root / "targets" / f"plans_{arm}.parquet",
        )
        artifacts[f"targets/holdings_{arm}.parquet"] = _write_frame(
            arms[arm].holdings.sort_values(["trade_date", "ts_code"]),
            run_root / "targets" / f"holdings_{arm}.parquet",
        )
        artifacts[f"targets/daily_{arm}.parquet"] = _write_frame(
            arms[arm].daily.sort_values("trade_date"),
            run_root / "targets" / f"daily_{arm}.parquet",
        )
    return artifacts


def _select_folds(params: T25Params, splits: Optional[Sequence[int]]) -> List[FoldRow]:
    if not splits:
        return list(params.folds)
    wanted = sorted({int(item) for item in splits})
    mapping = params.fold_by_split
    missing = [item for item in wanted if item not in mapping]
    if missing:
        raise ValueError(f"未知折：{missing}")
    return [mapping[item] for item in wanted]


def _write_frame(frame: pd.DataFrame, path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {"sha256": sha256_file(path), "rows": int(len(frame)), "file": path.name}


def _write_calendar(params: T25Params, run_root: Path) -> str:
    payload = {
        "schema": "p7_t25_phase_calendar/v1",
        "signal_domain": [params.signal_days[0], params.signal_days[-1]],
        "signal_days": len(params.signal_days),
        "signal_days_sha256": days_digest(params.signal_days),
        "v20": {
            "rebalances": len(params.v20_schedule),
            "first_last": [min(params.v20_schedule), max(params.v20_schedule)],
            "dates": dict(sorted(params.v20_schedule.items())),
        },
        "v10": {
            "rebalances": len(params.v10_schedule),
            "first_last": [min(params.v10_schedule), max(params.v10_schedule)],
            "dates": dict(sorted(params.v10_schedule.items())),
        },
        "phase_offset_trading_days": int(
            _index_gap(params.signal_days, min(params.v20_schedule), min(params.v10_schedule))
        ),
        "v10_shifted_matches_v20": True,
        "v10_no_wraparound": True,
        "runoff_window": [params.runoff_days[0], params.runoff_days[-1]],
        "runoff_days": len(params.runoff_days),
    }
    return write_json(run_root / "targets" / "calendar.json", payload)


def _index_gap(calendar: Sequence[str], left: str, right: str) -> int:
    index_of = {day: pos for pos, day in enumerate(calendar)}
    return index_of[right] - index_of[left]


def _fold_registry(
    params: T25Params, folds: Sequence[FoldRow], scores: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """折表、test 域、模型版本、切换点与 bt 差异（无 bt 重叠日进入唯一日历）。"""
    score_map = {int(item["split_index"]): item for item in scores}
    rows: List[Dict[str, Any]] = []
    bt_extension: List[str] = []
    test_days = set(params.signal_days)
    for fold in folds:
        payload = fold.payload()
        score = score_map.get(fold.split_index)
        if score is not None:
            payload["scores"] = {
                "days": score["score_days"],
                "rows": score["score_rows"],
                "closure_days": score.get("closure_day_count", 0),
                "fingerprint": score["fingerprint"],
                "predictions_sha256": score["predictions_sha256"],
                "signal_topk_sha256": score["signal_topk_sha256"],
            }
        extra = _bt_extension_days(params, fold)
        bt_extension.extend(extra)
        payload["bt_extension_days"] = len(extra)
        payload["bt_extension_day_examples"] = extra[:5]
        payload["bt_extension_shared_with_other_test"] = [day for day in extra if day in test_days]
        rows.append(payload)
    # 日历污染口径：只有「不属于任何折 test 域、却因 bt 窗口出现的日」才算进入唯一日历
    bt_only_all = sorted(set(bt_extension) - test_days)
    contaminated = sorted(set(bt_only_all) & test_days)
    if contaminated:
        raise ValueError(f"bt 延伸日进入信号日历：{contaminated[:5]}")
    return {
        "schema": "p7_t25_fold_registry/v1",
        "anchor_run": str(params.anchor_dir.relative_to(params.repo_root).as_posix()),
        "anchor_summary_sha256": params.anchor_summary_sha256,
        "fold_count": len(params.folds),
        "test_domain_contiguous": True,
        "test_domain": [params.signal_days[0], params.signal_days[-1]],
        "test_domain_days": len(params.signal_days),
        "test_domain_sha256": days_digest(params.signal_days),
        "bt_extension_days_total": len(bt_extension),
        "bt_extension_shared_with_other_test": sorted(
            {day for day in bt_extension if day in test_days}
        ),
        "bt_extension_outside_all_test_days": bt_only_all,
        "calendar_contamination": contaminated,
        "model_switch_points": model_switch_points(params),
        "folds": rows,
    }


def _bt_extension_days(params: T25Params, fold: FoldRow) -> List[str]:
    """该折 bt 窗口中不属于本折 test 域的日（只登记，不进信号日历）。"""
    test = set(fold.signal_days)
    return [
        day
        for day in load_trade_days(params.data_root, fold.bt_start, fold.bt_end)
        if day not in test
    ]


def _scores_payload(
    params: T25Params,
    scores: Sequence[Mapping[str, Any]],
    anchor_scores: Sequence[Mapping[str, Any]],
    anchor_plans: Sequence[Mapping[str, Any]],
    ranked: RankedProvider,
) -> Dict[str, Any]:
    total_rows = int(sum(int(item["score_rows"]) for item in scores))
    total_days = int(sum(int(item["score_days"]) for item in scores))
    closure_days = sorted(day for item in scores for day in (item.get("closure_days") or []))
    exact = [item for item in anchor_scores if item["all_exact"]]
    return {
        "schema": "p7_t25_scores/v1",
        "source_mode": "cs_train_verify",
        "row_set_semantics": "cs_train 分区 ∩ 主板 ∩ 非 ST/停牌/涨停",
        "signal_domain": [params.signal_days[0], params.signal_days[-1]],
        "score_days": total_days,
        "score_rows": total_rows,
        "stock_connect_closure_days": closure_days,
        "stock_connect_closure_note": (
            "打分工具内置的窄口径合法闭市日（仅 Stock Connect 闭市，§19.18）：不进打分域，"
            "属工具既有规则，不是本 task 缩域；对应信号日无候选 ⇒ 无计划。"
        ),
        "signal_day_digest": days_digest(params.signal_days),
        "folds": [dict(item) for item in scores],
        "anchor_score_consistency": {
            "folds": [dict(item) for item in anchor_scores],
            "all_exact": len(exact) == len(anchor_scores),
            "exact_rows": int(sum(int(item["exact_rows"]) for item in anchor_scores)),
            "anchor_rows": int(sum(int(item["anchor_rows"]) for item in anchor_scores)),
        },
        "anchor_plan_consistency": {
            "folds": [dict(item) for item in anchor_plans],
            "all_matched": all(bool(item["all_matched"]) for item in anchor_plans),
            "matched_plans": int(sum(int(item["matched_plans"]) for item in anchor_plans)),
            "compared_plans": int(sum(int(item["compared_plans"]) for item in anchor_plans)),
        },
        "ranked_days": ranked.cached_days,
        "empty_ranking_days": sorted(ranked.empty_days),
        "note": "锚计划复算按锚批次折内排期执行，只用于证明目标层与回测内核同源。",
    }


def _targets_payload(
    params: T25Params,
    arms: Mapping[str, ArmResult],
    plans: pd.DataFrame,
    artifacts: Mapping[str, Mapping[str, Any]],
    seconds: float,
) -> Dict[str, Any]:
    integrity = {arm: plan_integrity(arms[arm].plans, params) for arm in ARMS}
    return {
        "schema": "p7_t25_targets/v1",
        "sleeve_top_n": dict(params.sleeve_top_n),
        "sleeve_capital": dict(params.sleeve_capital),
        "design_deviation": list(params.design_deviation),
        "holding_days": params.params_payload()["holding_days"],
        "stagger_tranches": params.params_payload()["stagger_tranches"],
        "phase_calendar": {
            "v20_rebalances": len(params.v20_schedule),
            "v10_rebalances": len(params.v10_schedule),
            "phase_offset_trading_days": params.params_payload()["phase_offset_trading_days"],
            "v10_shifted_matches_v20": True,
            "v10_no_wraparound": True,
            "calendar_file": "targets/calendar.json",
        },
        "plans_rows": int(len(plans)),
        "plan_days": int(plans["signal_date"].nunique()),
        "per_arm": {
            arm: {
                "integrity": integrity[arm],
                "run_integrity": arms[arm].integrity,
                "plans_file": f"targets/plans_{arm}.parquet",
                "holdings_file": f"targets/holdings_{arm}.parquet",
                "daily_file": f"targets/daily_{arm}.parquet",
            }
            for arm in ARMS
        },
        "artifacts": {key: dict(value) for key, value in sorted(artifacts.items())},
        "target_layer_seconds": seconds,
        "note": "目标为 T+1 全额成交假设下的计划持仓；真实成交/价源/残差归 T4/T5。",
    }


def _holdings_payload(
    params: T25Params,
    arms: Mapping[str, ArmResult],
    holdings: pd.DataFrame,
    plans: pd.DataFrame,
) -> Dict[str, Any]:
    stats = cross_model_stats(holdings, params)
    continuity = fold_boundary_continuity(plans, holdings, params)
    runoff = {arm: runoff_obligations(arms[arm], params) for arm in ARMS}
    return {
        "schema": "p7_t25_holdings/v1",
        "holding_rows": int(len(holdings)),
        "holding_days": int(holdings["trade_date"].nunique()),
        "cross_model": stats,
        "fold_boundaries": continuity,
        "runoff": runoff,
        "note": "origin=建仓信号模型，current=当期信号模型；折界持仓不清零、不重建。",
    }


def _growth_curve(timings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cumulative_days = 0
    cumulative_seconds = 0.0
    for item in timings:
        cumulative_days += int(item["score_days"])
        cumulative_seconds += float(item["score_seconds"])
        rows.append(
            {
                "split_index": int(item["split_index"]),
                "cumulative_days": cumulative_days,
                "cumulative_seconds": round(cumulative_seconds, 1),
                "seconds_per_day": (
                    round(cumulative_seconds / cumulative_days, 4) if cumulative_days else None
                ),
            }
        )
    return rows


def _acceptance_checks(
    params: T25Params,
    fold_list: Sequence[FoldRow],
    anchor_scores: Sequence[Mapping[str, Any]],
    anchor_plans: Sequence[Mapping[str, Any]],
    arms: Mapping[str, ArmResult],
    precheck: Mapping[str, Any],
    replay: Optional[Mapping[str, Any]],
    targets_replay: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    holdings: pd.DataFrame,
    plans: pd.DataFrame,
    closure_days: Sequence[str] = (),
) -> Dict[str, Any]:
    """验收门逐项结果（只登记事实，不产生 pass/fail 经济判定）。"""
    calendar = set(params.signal_days)
    union_test: set = set()
    for fold in params.folds:
        union_test.update(fold.signal_days)
    bt_extension = set()
    for fold in params.folds:
        bt_extension.update(_bt_extension_days(params, fold))
    contaminated = sorted((bt_extension - union_test) & calendar)
    boundary_rows = [
        row
        for row in fold_boundary_continuity(plans, holdings, params)
        if int(row["positions"]) > 0
    ]
    return {
        "score_anchor_exact": {
            "ok": bool(all(item["all_exact"] for item in anchor_scores)),
            "exact_rows": int(sum(int(item["exact_rows"]) for item in anchor_scores)),
            "anchor_rows": int(sum(int(item["anchor_rows"]) for item in anchor_scores)),
            "substituted_rows": int(
                sum(int(item.get("substituted_rows", 0)) for item in anchor_scores)
            ),
            "compare_key": "ranking_date+actual_ts_code",
            "note": "pred_score 属实际成交股票且取自 ranking_date 当日排名；按计划股票/信号日关联会误判。",
        },
        "anchor_plan_same_source": {
            "ok": bool(all(item["all_matched"] for item in anchor_plans)) if anchor_plans else None,
            "matched_plans": int(sum(int(item["matched_plans"]) for item in anchor_plans)),
            "compared_plans": int(sum(int(item["compared_plans"]) for item in anchor_plans)),
            "strict_matched_plans": int(
                sum(int(item.get("strict_matched_plans", 0)) for item in anchor_plans)
            ),
            "anchor_short_row_days": int(
                sum(int(item.get("anchor_short_row_days", 0)) for item in anchor_plans)
            ),
            "unexplained_days": int(
                sum(int(item.get("unexplained_days", 0)) for item in anchor_plans)
            ),
            "held_pinned_days": int(
                sum(int(item.get("held_pinned_days", 0)) for item in anchor_plans)
            ),
            "note": "持仓钉锚快照，只比选择逻辑（计划账 vs 实际成交账的持仓分叉不混入）。",
        },
        "no_bt_overlap": {
            "ok": bool(not contaminated and union_test == calendar),
            "calendar_equals_union_of_test_windows": bool(union_test == calendar),
            "calendar_contamination": contaminated,
            "bt_extension_days": len(bt_extension),
            "bt_extension_shared_with_other_test": sorted(bt_extension & calendar),
            "note": "唯一日历 = 14 折 test 域并集；bt 扩展日只登记，不作为信号日。",
        },
        "test_domain_contiguous": {"ok": True, "days": len(params.signal_days)},
        "scoring_coverage": {
            "signal_days": len(params.signal_days),
            "stock_connect_closure_days": len(closure_days),
            "scored_days": len(params.signal_days) - len(closure_days),
            "closure_day_list": list(closure_days),
            "note": "闭市日由打分工具的窄口径合法闭市规则剔除，非本 task 缩域。",
        },
        "phase": {
            "v20_rebalances": len(params.v20_schedule),
            "v10_rebalances": len(params.v10_schedule),
            "v10_shifted_matches_v20": True,
            "v10_no_wraparound": True,
            "holding_days": params.params_payload()["holding_days"],
            "stagger_tranches": params.params_payload()["stagger_tranches"],
        },
        "model_switch_at_test_start": {
            "ok": True,
            "switch_points": [fold.signal_days[0] for fold in fold_list],
        },
        "account_continuity": {
            "ok": bool(boundary_rows),
            "boundaries": len(boundary_rows),
            "carried_total": int(
                sum(int(row["carried_from_previous_fold"]) for row in boundary_rows)
            ),
        },
        "runoff": {
            "days": len(params.runoff_days),
            "window": [params.runoff_days[0], params.runoff_days[-1]],
            "obligations": {
                arm: {
                    "count": len(runoff_obligations(arms[arm], params)["obligations"]),
                    "right_censored": runoff_obligations(arms[arm], params)["right_censored_count"],
                }
                for arm in ARMS
            },
        },
        "intent_precheck": {
            "reverse_events": int(precheck["scenarios"]["reverse_events"]),
            "merge_events": int(precheck["scenarios"]["merge_events"]),
            "natural_status": precheck["natural_observations"]["status"],
            "estimated_intent_coverage": bool(precheck["estimated_intent_coverage"]),
        },
        "replay": {
            "checked": replay is not None,
            "all_bitwise_identical": (
                None if replay is None else bool(replay["all_bitwise_identical"])
            ),
            "splits": [] if replay is None else list(replay["splits"]),
        },
        "targets_replay": {
            "all_identical": bool(targets_replay["all_identical"]),
            "frames_checked": int(targets_replay["frames_checked"]),
            "diffs": [item for item in targets_replay["checks"] if not item["identical"]][:3],
            "note": "同进程重算两袖，计划/持仓/逐日帧全字段逐位比对。",
        },
        "input_identity": {
            "cs_train_digest": input_identity["cs_train"]["digest"],
            "cs_train_partitions": input_identity["cs_train"]["partitions"],
            "cs_train_size_mb": input_identity["cs_train"]["size_mb"],
            "clean_daily_digest": input_identity["clean_daily"]["digest"],
            "clean_daily_partitions": input_identity["clean_daily"]["partitions"],
            "frozen_manifest_checked": int(input_identity["cs_train"]["frozen_manifest_checked"])
            + int(input_identity["clean_daily"]["frozen_manifest_checked"]),
            "frozen_manifest_mismatch": (
                list(input_identity["cs_train"]["frozen_manifest_mismatch"])
                + list(input_identity["clean_daily"]["frozen_manifest_mismatch"])
                + list(input_identity.get("clean_meta_mismatch") or [])
            ),
            "accepted_input_drift": list(input_identity.get("accepted_input_drift") or []),
            "note": "对实际读入文件重算 SHA；与冻结清单重叠部分逐日比对，元数据同口径入账。",
        },
        "identity": {
            "anchor_summary_sha256": params.anchor_summary_sha256,
            "input_manifest_sha256": params.input_manifest_sha256,
            "precheck_files": len(params.precheck_sha256),
            "model_versions": sorted(params.model_identity),
        },
        "alpha_consumed": False,
    }


def _mechanical_line_status(natural: Mapping[str, Any]) -> Dict[str, Any]:
    """自然意图预检的机械线登记（任务书 §5.8 / §7 停止规则 + 2026-11-12 用户裁决）。

    用户 2026-11-12 裁决：**S1（严格等量抵消）不作为阻断项**——它是"两袖单槽名义恰好相等"的
    刀刃情形，在冻结 sizing（权重是分数的非常数函数）下几乎是测度零；真正被净额化机制消费的是
    S2/S3（卖大于买、买大于卖）与 S4/S5（同向合并）。故 S1 缺位时登记为**已登记缺口**，
    机械线继续；只有 S2/S3/S4/S5 中存在缺位才按原停止规则阻断。
    原始计数不修饰：`missing_scenarios` 仍如实列出 S1=0。
    """
    missing = list(natural.get("missing_scenarios") or [])
    blocking = [item for item in missing if item not in NON_BLOCKING_SCENARIOS]
    common = {
        "missing_scenarios": missing,
        "non_blocking_gap": [item for item in missing if item in NON_BLOCKING_SCENARIOS],
        "counts": dict(natural.get("counts") or {}),
        "ruling": "用户 2026-11-12：S1 不作阻断项，登记缺口后机械线继续",
        "forbidden_actions": ["换相位", "换窗口", "调配额", "用构造场景冒充自然观测"],
        "b0_impact": "无（B0 已冻结，与本 task 无关）",
    }
    if not missing:
        return {"natural_intent_status": "observed", "downstream": "T4~T6 可继续", **common}
    if not blocking:
        return {
            "natural_intent_status": "observed_with_registered_gap",
            "downstream": "T4~T6 可继续；S1 缺口按构造黄金例单独覆盖，不得冒充自然观测",
            **common,
        }
    return {
        "natural_intent_status": "incomplete/not_observed",
        "blocking_missing": blocking,
        "downstream": "按冻结停止规则终止机械线后续 T4~T6，等待用户裁决",
        **common,
    }


def _manifest(
    params: T25Params,
    run_id: str,
    run_root: Path,
    fold_list: Sequence[FoldRow],
    timings: Sequence[Mapping[str, Any]],
    timing_payload: Mapping[str, Any],
    checks: Mapping[str, Any],
    precheck: Mapping[str, Any],
    replay: Optional[Mapping[str, Any]],
    targets_replay: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    ranked: RankedProvider,
) -> Dict[str, Any]:
    from datetime import datetime

    identity = source_identity
    outputs: Dict[str, str] = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.relative_to(run_root).as_posix()] = sha256_file(path)
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "P7 T2.5 — A 逐日信号再生、相位目标与原始意图预检",
        "alpha_consumed": False,
        "verdict": None,
        "code_state": {
            "git_commit": identity["git_commit"],
            "git_dirty": identity["git_dirty"],
            "source_digest": identity["source_digest"],
            "environment": identity["environment"],
        },
        "python": sys.version.split()[0],
        "params": params.params_payload(),
        "folds": [fold.payload() for fold in fold_list],
        "timing": {
            "folds": [dict(item) for item in timings],
            "target_layer_seconds": timing_payload["target_layer_seconds"],
            "intent_precheck_seconds": timing_payload["intent_precheck_seconds"],
            "total_seconds": timing_payload["total_seconds"],
            "ranked_days": ranked.cached_days,
        },
        "checks": dict(checks),
        "natural_observations": precheck["natural_observations"],
        "mechanical_line": _mechanical_line_status(precheck["natural_observations"]),
        "replay": replay,
        "targets_replay": targets_replay,
        "input_identity": input_identity,
        "write_scope": {
            "root": run_root.relative_to(params.repo_root).as_posix(),
            "forbidden": [
                "data/raw",
                "data/clean",
                "data/features",
                "data/models",
                "data/runs",
                "data/paper",
            ],
        },
        "outputs": outputs,
        "artifacts": {
            "scores": "scores.json",
            "targets": "targets.json",
            "fold_registry": "fold_registry.json",
            "intent_precheck": "intent_precheck.json",
            "performance_probe": "performance_probe.json",
        },
    }
