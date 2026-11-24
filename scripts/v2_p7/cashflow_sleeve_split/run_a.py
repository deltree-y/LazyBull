# -*- coding: utf-8 -*-
"""run-a 批次编排（协议 §8 审点二；用户 2026-11-20 指令授权的命名批次）。

批次状态机（各批可单独恢复；中断分片不算完成）
================================================
``rebuild``  结构域全窗重建（行业变更日分段；缓存 campaign/cache/<spec16>/structural_full）
``train-f``  F 臂 14 折 + 部署（隔离注册 campaign/models/f；逐折持久化断点）
``train-r``  R 臂 14 折 + 部署（campaign/models/r）
``score``    1838 日 × F/R 逐日打分（日期绑定模型 + 宽度台账 + 整窗 ranked）
``accounts`` F100/F80/R100/R80 × 两本金主账户（+ F/R 压力臂 10bp）
``deliver``  链式指标/桥接/B 门槛/自举 → a_metrics.csv、b_required.csv、a_review.md → 封存

断点续跑：run 目录存在且无 PUBLISHED.json、batch_state.json 的代码身份一致 ⇒ 继续；
同 run 重写、代码身份漂移、已发布 run 重入一律拒绝。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.cashflow_sleeve_split import (
    accounts,
    artifacts,
    deliver_seal,
    feature_routes,
    fold_clock,
    identity,
    metrics,
    scoring,
    training,
)
from scripts.v2_p7.cashflow_sleeve_split.artifacts import ArtifactError
from scripts.v2_p7.common import sha256_file, write_json

BATCHES: Sequence[str] = (
    "rebuild",
    "rebuild-history",
    "train-f",
    "train-r",
    "score",
    "accounts",
    "deliver",
)

#: 训练历史结构覆盖起点（checkpoint-2 第 2 项：训练帧 PIT 叠加需全历史结构帧）
HISTORY_START = "20120101"

STATE_NAME = "batch_state.json"

#: 主账户角色 × 本金档（协议 §6：100/80 与 200/160）
SCENARIOS: Sequence[int] = (1_000_000, 2_000_000)

#: 压力臂滑点（协议 §7.6：每边 5bp → 10bp）
#: 压力臂唯一差异 = 双边滑点 5bp→10bp（协议 §7.6）；印花税维持基准 5bp
#: （checkpoint-2 Codex 第 5 项：此前误将 stamp_tax 一并提到 10bp）
STRESS_COST_KWARGS = (0.0001954, 5.0, 0.0005, 0.0010)


class RunAError(RuntimeError):
    """run-a 编排失败（fail-fast）。"""


def _log(run_id: str) -> Callable[[str], None]:
    def emit(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}][run-a/{run_id}] {message}", flush=True)

    return emit


def _state_path(run_dir: Path) -> Path:
    return run_dir / "work" / STATE_NAME


def _code_identity(base: Path) -> Dict[str, Any]:
    """run-a 代码身份（包源码 SHA；续跑一致性判据）。"""
    package = sorted((base / "scripts/v2_p7/cashflow_sleeve_split").glob("*.py"))
    files = {path.name: sha256_file(path) for path in package}
    return {"package_files": files, "package_digest": _package_digest(files)}


def _package_digest(files: Mapping[str, str]) -> str:
    import hashlib

    payload = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_run_dir(
    camp: Path, run_id: str, base: Path, parent: str, extra_roots: Sequence[str]
) -> tuple:
    """创建或续开 run 目录；返回 (run_dir, resume)。"""
    run_dir = Path(camp) / "runs" / str(run_id)
    if run_dir.exists():
        if (run_dir / artifacts.PUBLISHED_NAME).exists():
            raise ArtifactError(f"run {run_id} 已发布，禁止重入（新结果用新 run，协议 §2.3）")
        state_path = _state_path(run_dir)
        if not state_path.exists():
            raise ArtifactError(
                f"run {run_id} 存在但缺批次状态（{state_path}）；视为中断残片，请用新 run"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("parent") != str(parent):
            raise ArtifactError(f"续跑 parent 不符：{state.get('parent')} != {parent}")
        return run_dir, True
    run_dir = artifacts.create_run_dir(camp, run_id, extra_roots)
    return run_dir, False


def _assert_batch_code_identity(state: Mapping[str, Any], batch: str, base: Path) -> str:
    """待执行批次的代码身份守卫（同批续跑必须同码）；返回当前摘要并登记。"""
    current = _package_digest(_code_identity(base)["package_files"])
    digests = state.get("batch_digests") or {}
    recorded = digests.get(batch)
    if recorded is not None and recorded != current:
        raise RunAError(
            f"批次 {batch} 的代码身份与上次执行不一致（{str(recorded)[:12]}… != "
            f"{current[:12]}…）：批内断点续跑禁止跨代码漂移，请用新 run 重取该批（协议 §2.3）"
        )
    return current


def _load_state(run_dir: Path) -> Dict[str, Any]:
    state = json.loads(_state_path(run_dir).read_text(encoding="utf-8"))
    state.setdefault("batches_done", [])
    state.setdefault("batch_digests", {})
    state.setdefault("folds", {})
    state.setdefault("deploy", {})
    state.setdefault("segments", {})
    state.setdefault("ledger", [])
    return state


def _persist_state(run_dir: Path, state: Dict[str, Any]) -> None:
    write_json(_state_path(run_dir), state)


def _write_code_evidence(base: Path, work: Path) -> None:
    """在首批开始前存实际代码差异，随 run 的原始工件一起封存。"""
    from shutil import copyfile

    restore = identity.code_restore_payload(base)
    code_dir = work / "code_restore"
    code_dir.mkdir(parents=True, exist_ok=True)
    patch = identity._run_git(["diff", "HEAD", "--binary", "--no-color"], base)
    patch_path = code_dir / "tracked.patch"
    patch_path.write_text(patch + "\n", encoding="utf-8", newline="\n")
    for record in restore["files"]:
        relative = Path(record["path"])
        target = code_dir / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(base / relative, target)
        if sha256_file(target) != record["sha256"]:
            raise RunAError(f"初始代码快照与登记字节不符：{relative}")
    restore["tracked_patch_sha256"] = sha256_file(patch_path)
    restore["tracked_patch_bytes"] = patch_path.stat().st_size
    restore["package"] = _code_identity(base)
    write_json(code_dir / "code_restore.json", restore)


def run_run_a(args: argparse.Namespace) -> Dict[str, Any]:
    base = Path(args.repo_root)
    camp = artifacts.campaign_root(args.campaign, args.artifact_base and Path(args.artifact_base))
    extra = list(args.allow_root or [])
    run_id = args.run_id or artifacts.new_run_id("run-a")
    emit = _log(run_id)
    started = time.perf_counter()

    try:
        artifacts.require_published(camp, args.parent, "prepare")
    except ArtifactError as error:
        # 未授权/上游缺件：只报告前置条件，不写任何工件、不污染 stdout（协议 §2.4、§8）
        return {
            "status": "blocked",
            "stage": "run-a",
            "reason": "not_authorized",
            "required_upstream": "prepare",
            "parent": args.parent,
            "run_id": run_id,
            "message": (
                f"阶段 run-a 未授权或上游缺件：{error}。"
                "审点一材料须由用户转交 Codex 审核通过，并由用户按命名批次批准；"
                "本次未写任何工件（协议 §8）。"
            ),
        }
    parent_manifest = artifacts.require_published(camp, args.parent, "prepare")
    emit(f"1/6 上游核验 OK：prepare run {args.parent}")
    artifacts.ensure_layout(camp)
    parent_dir = Path(camp) / "runs" / str(args.parent)
    baseline = identity.verify_a_baseline(base)

    run_dir, resume = _open_run_dir(camp, run_id, base, args.parent, extra)
    if resume:
        state = _load_state(run_dir)
        if not state.get("parent_manifest_sha256"):
            # checkpoint-2 第 6 项：父 manifest 实际字节 SHA（require_published 返回体不含自 SHA）
            state["parent_manifest_sha256"] = sha256_file(parent_dir / artifacts.MANIFEST_NAME)
    else:
        state = _fresh_state(
            args.parent,
            {
                **parent_manifest,
                "manifest_sha256": sha256_file(parent_dir / artifacts.MANIFEST_NAME),
            },
            base,
        )
    if resume:
        emit(f"续跑 run {run_id}；已完成批次 = {state['batches_done']}")
    else:
        _write_code_evidence(base, run_dir / "work")
    _persist_state(run_dir, state)
    work = run_dir / "work"

    routes_doc = json.loads((parent_dir / "feature_routes.json").read_text(encoding="utf-8"))
    routes = feature_routes.route_table(routes_doc)
    clock = fold_clock.build_fold_clock(base)
    batch_filter = list(args.batch) if args.batch else list(BATCHES)
    unknown = [item for item in batch_filter if item not in BATCHES]
    if unknown:
        raise RunAError(f"未知批次：{unknown}（合法 {list(BATCHES)}）")

    ctx = _BatchContext(
        base,
        camp,
        run_dir,
        work,
        state,
        emit,
        resume,
        routes,
        clock,
        baseline,
        extra,
        workers=int(getattr(args, "workers", 1) or 1),
    )
    results: Dict[str, Any] = {}
    for batch in batch_filter:
        results[batch] = _run_batch(batch, ctx, started)

    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "run_id": run_id,
        "batches": list(batch_filter),
        "batches_done": list(state["batches_done"]),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def _fresh_state(parent: str, parent_manifest: Mapping[str, Any], base: Path) -> Dict[str, Any]:
    """新 run 的批次状态（含授权登记与逐批代码身份字典）。"""
    return {
        "schema": "p7_cashflow_split_run_a_state/v1",
        "parent": str(parent),
        "parent_manifest_sha256": parent_manifest.get("manifest_sha256"),
        "authorized_by": "user instruction 2026-11-20: 按 CF-SPLIT-20-v1 实现至审点二",
        "batches_done": [],
        "batch_digests": {},
        "folds": {},
        "deploy": {},
        "segments": {},
        "ledger": [],
    }


class _BatchContext:
    """批次共享上下文（一个 run 内固定不变）。"""

    def __init__(
        self,
        base: Path,
        camp: Path,
        run_dir: Path,
        work: Path,
        state: Dict[str, Any],
        emit: Callable[[str], None],
        resume: bool,
        routes: Mapping[int, Mapping[str, Any]],
        clock: Mapping[str, Any],
        baseline: Mapping[str, Any],
        extra_roots: Sequence[str],
        workers: int = 1,
    ) -> None:
        self.base = base
        self.camp = camp
        self.run_dir = run_dir
        self.work = work
        self.state = state
        self.emit = emit
        self.resume = resume
        self.routes = routes
        self.clock = clock
        self.baseline = baseline
        self.extra_roots = extra_roots
        self.workers = workers


def _dispatch_rebuild(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_rebuild(
        ctx.base,
        ctx.camp,
        ctx.run_dir,
        ctx.work,
        ctx.state,
        ctx.emit,
        ctx.resume,
        routes=ctx.routes,
        workers=ctx.workers,
    )


def _dispatch_rebuild_history(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_rebuild(
        ctx.base,
        ctx.camp,
        ctx.run_dir,
        ctx.work,
        ctx.state,
        ctx.emit,
        ctx.resume,
        routes=ctx.routes,
        workers=ctx.workers,
        history=True,
    )


def _dispatch_train_f(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_train(
        ctx.base, ctx.camp, ctx.work, ctx.state, ctx.emit, ctx.routes, ctx.clock, "F", ctx.resume
    )


def _dispatch_train_r(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_train(
        ctx.base, ctx.camp, ctx.work, ctx.state, ctx.emit, ctx.routes, ctx.clock, "R", ctx.resume
    )


def _dispatch_score(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_score(
        ctx.base, ctx.camp, ctx.run_dir, ctx.work, ctx.state, ctx.emit, ctx.routes, ctx.resume
    )


def _dispatch_accounts(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_accounts(
        ctx.base, ctx.camp, ctx.run_dir, ctx.work, ctx.state, ctx.emit, ctx.resume
    )


def _dispatch_deliver(ctx: _BatchContext) -> Dict[str, Any]:
    return _batch_deliver(ctx.run_dir, ctx.work, ctx.emit, ctx.baseline)


#: 批次分发表（入口已按 BATCHES 白名单校验）
_BATCH_DISPATCH = {
    "rebuild": _dispatch_rebuild,
    "rebuild-history": _dispatch_rebuild_history,
    "train-f": _dispatch_train_f,
    "train-r": _dispatch_train_r,
    "score": _dispatch_score,
    "accounts": _dispatch_accounts,
    "deliver": _dispatch_deliver,
}


def _run_batch(batch: str, ctx: _BatchContext, started: float) -> Dict[str, Any]:
    """单批次执行 + 完成登记 + 持久化（deliver 每轮都重算）。

    批次开始前登记本批代码身份：同批断点续跑要求同码；跨批改码不阻塞
    （已完成批次的代码身份已随其产物封存）。
    """
    if batch in ctx.state["batches_done"] and batch != "deliver":
        ctx.emit(f"2/6 批次 {batch} 已完成（跳过）")
        return {}
    import sys

    batch_started = time.perf_counter()
    digest = _assert_batch_code_identity(ctx.state, batch, ctx.base)
    ctx.state.setdefault("batch_digests", {})[batch] = digest
    ctx.state.setdefault("batch_commands", {})[batch] = [
        sys.executable, "-m", "scripts.v2_p7.cashflow_sleeve_split.run", *sys.argv[1:]
    ]
    _persist_state(ctx.run_dir, ctx.state)
    ctx.emit(f"2/6 批次 {batch} 开始（代码 {digest[:12]}…）")
    handler = _BATCH_DISPATCH.get(batch)
    if handler is None:  # pragma: no cover - 入口已白名单校验
        raise RunAError(f"未知批次：{batch}")
    result = handler(ctx)
    if batch not in ctx.state["batches_done"]:
        ctx.state["batches_done"].append(batch)
    ctx.state.setdefault("batch_elapsed_seconds", {})[batch] = round(
        time.perf_counter() - batch_started, 1
    )
    _persist_state(ctx.run_dir, ctx.state)
    if batch == "deliver":
        # 完成状态先进入清单，PUBLISHED 最后提交；发布后不再写 work 中的任何文件。
        result["seal"] = deliver_seal.seal_deliverables(
            seal, ctx.run_dir, ctx.work, result["gates"], ctx.state, ctx.extra_roots
        )
        ctx.emit("    封存完成（manifest.json + PUBLISHED.json；work/ 逐件 SHA 入 payload）")
    ctx.emit(f"2/6 批次 {batch} 完成；累计 {time.perf_counter() - started:.0f}s")
    return result


# ── 批次实现 ──────────────────────────────────────────────────────────


def _batch_rebuild(
    base: Path,
    camp: Path,
    run_dir: Path,
    work: Path,
    state: Dict[str, Any],
    emit: Callable[[str], None],
    resume: bool,
    routes: Mapping[int, Mapping[str, Any]],
    workers: int = 1,
    history: bool = False,
) -> Dict[str, Any]:
    result = scoring.rebuild_segments(
        base,
        camp,
        state,
        emit,
        resume=resume,
        persist=lambda: _persist_state(run_dir, state),
        workers=workers,
        history_start=HISTORY_START if history else None,
    )
    manifest_name = "structural_history_manifest.json" if history else "structural_manifest.json"
    write_json(work / manifest_name, result)
    # 段缓存路由列覆盖断言（fail-closed；整列消失类缺陷在训练期才暴露的教训）
    coverage = scoring.assert_route_columns_coverage(camp, routes, emit, prod_root=base)
    write_json(work / "route_columns_coverage.json", coverage)
    return {"segments": result["segments_n"], "days": result["totals"]["days"]}


def _batch_train(
    base: Path,
    camp: Path,
    work: Path,
    state: Dict[str, Any],
    emit: Callable[[str], None],
    routes: Mapping[int, Mapping[str, Any]],
    clock: Mapping[str, Any],
    arm: str,
    resume: bool,
) -> Dict[str, Any]:
    result = training.train_arm_folds(
        base,
        camp,
        arm,
        routes,
        clock,
        state,
        emit,
        resume=resume,
        persist=lambda: _persist_state(work.parent, state),
    )
    write_json(work / f"training_{arm}.json", result)
    folds = result["records"]
    emit(f"    臂 {arm} 完成：注册记录 {len(folds)} 件（14 折 + 部署）")
    return {"arm": arm, "records": len(folds)}


def _batch_score(
    base: Path,
    camp: Path,
    run_dir: Path,
    work: Path,
    state: Dict[str, Any],
    emit: Callable[[str], None],
    routes: Mapping[int, Mapping[str, Any]],
    resume: bool,
) -> Dict[str, Any]:
    version_map: Dict[str, Dict[str, int]] = {}
    for arm in ("F", "R"):
        doc = json.loads((work / f"training_{arm}.json").read_text(encoding="utf-8"))
        version_map[arm] = {
            str(record["version"]): int(record["registered_version"]) for record in doc["records"]
        }
        emit(f"    臂 {arm} 版本映射 {len(version_map[arm])} 项")
    ranked_dir = work / "ranked"
    result = scoring.score_window(
        base,
        camp,
        ranked_dir,
        routes,
        version_map,
        state,
        emit,
        resume=resume,
        persist=lambda: _persist_state(run_dir, state),
    )
    window = scoring.account_calendar(base)
    outputs = scoring.assemble_ranked(ranked_dir, window)
    write_json(work / "scoring_summary.json", result)
    pd.DataFrame(result["ledger"]).to_csv(
        work / "width_by_day.csv", index=False, encoding="utf-8-sig"
    )
    width_f = result["widths"]["F"]
    width_r = result["widths"]["R"]
    emit(
        f"    打分完成：{result['days']} 日；宽度最小值 "
        f"F={width_f['min_effective_n']} R={width_r['min_effective_n']}；"
        f"空排名日 F={len(width_f['empty_days'])} R={len(width_r['empty_days'])}"
    )
    return {
        "days": result["days"],
        "widths": result["widths"],
        "ranked_files": {arm: str(path) for arm, path in outputs.items()},
    }


def _batch_accounts(
    base: Path,
    camp: Path,
    run_dir: Path,
    work: Path,
    state: Dict[str, Any],
    emit: Callable[[str], None],
    resume: bool,
) -> Dict[str, Any]:
    from src.lazybull.common.cost import CostModel
    from src.lazybull.v2.core.execution.price_index import prepare_price_index

    calendar, anchor = accounts.build_calendar(base)
    verify = accounts.verify_corporate_input(base)
    emit(f"    冻结公司行为复核 OK：{verify['dividend_files']} 件")
    ranked_paths = {arm: work / "ranked" / f"ranked_{arm}.parquet" for arm in ("F", "R")}
    for arm, path in ranked_paths.items():
        if not path.exists():
            raise RunAError(f"整窗排名缺件：{path}")
    summary = json.loads((work / "scoring_summary.json").read_text(encoding="utf-8"))
    inputs_by_arm = {}
    for arm, path in ranked_paths.items():
        ranked = scoring.load_ranked_dict(path)
        allowed_empty = list(summary["widths"][arm]["empty_days"])
        inputs_by_arm[arm] = accounts.build_baseline_inputs(
            ranked,
            calendar,
            anchor,
            identity={
                "source": "run-a structural scoring",
                "ranked_parquet": path.name,
                "anchor": anchor,
                "days": len(calendar),
                "not_virgin_holdout": True,
                "allowed_empty_days": allowed_empty,
            },
            allowed_empty=allowed_empty,
        )
        emit(
            f"    臂 {arm} 排名 {len(ranked)} 日装配完成"
            f"（空排名日 {len(allowed_empty)} 按生产语义落空单）"
        )
    prices, price_identity = accounts.load_prices_with_identity(base, inputs_by_arm["F"])
    indexes = prepare_price_index(prices)
    book = accounts.build_corporate_book(base, [anchor, *calendar], work=work)
    write_json(
        work / "account_inputs.json",
        {
            "price_identity": price_identity,
            "corporate_verify": verify,
            "anchor": anchor,
            "calendar_days": len(calendar),
        },
    )
    results: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        for role in accounts.PRIMARY_ROLES:
            capital = accounts.capital_for_role(role, scenario)
            arm = role[0]
            out_dir = run_dir / "accounts" / str(scenario) / role
            done = state.get("accounts", {}).get(f"{role}@{scenario}")
            if resume and done and (out_dir / "ending.json").exists():
                emit(f"    账户 {role}@{scenario} 已完成（跳过）")
                continue
            if out_dir.exists():
                # 崩跑残留的半成品目录（ending.json 缺失即不可用）→ 清除后重放
                shutil.rmtree(out_dir)
            emit(f"    账户 {role}@{scenario}（本金 {capital}）回放中…")
            accounts.run_account(
                base,
                inputs=inputs_by_arm[arm],
                prices=prices,
                indexes=indexes,
                capital=capital,
                corporate_book=book,
                output=out_dir,
                label=f"{role}@{scenario}",
            )
            state.setdefault("accounts", {})[f"{role}@{scenario}"] = {"capital": capital}
            _persist_state(run_dir, state)
            results[f"{role}@{scenario}"] = {"capital": capital}
        emit(f"    本金档 {scenario} 主账户完成")
    emit("    压力臂（滑点 10bp）回放中…")
    stress_cost = CostModel(*STRESS_COST_KWARGS)
    for scenario in SCENARIOS:
        for role in accounts.STRESS_ROLES:
            capital = accounts.capital_for_role(role, scenario)
            arm = role[0]
            out_dir = run_dir / "accounts_stress" / str(scenario) / role
            done = state.get("accounts_stress", {}).get(f"{role}@{scenario}")
            if resume and done and (out_dir / "ending.json").exists():
                continue
            if out_dir.exists():
                shutil.rmtree(out_dir)
            accounts.run_account(
                base,
                inputs=inputs_by_arm[arm],
                prices=prices,
                indexes=indexes,
                capital=capital,
                corporate_book=book,
                output=out_dir,
                cost_override=stress_cost,
                label=f"{role}@{scenario}@stress10bp",
            )
            state.setdefault("accounts_stress", {})[f"{role}@{scenario}"] = {"capital": capital}
            _persist_state(run_dir, state)
    emit("    压力臂完成（现金/主组合属 run-b 范围，标记 not_run）")
    return {"accounts": len(results), "scenarios": list(SCENARIOS)}


def _read_account_daily(run_dir: Path, scenario: int, role: str) -> List[Dict[str, Any]]:
    path = run_dir / "accounts" / str(scenario) / role / "daily.json"
    if not path.exists():
        raise RunAError(f"账户逐日权益缺件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_stress_daily(run_dir: Path, scenario: int, role: str) -> List[Dict[str, Any]]:
    path = run_dir / "accounts_stress" / str(scenario) / role / "daily.json"
    if not path.exists():
        raise RunAError(f"压力账户逐日权益缺件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _batch_deliver(
    run_dir: Path,
    work: Path,
    emit: Callable[[str], None],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    scoring_summary = json.loads((work / "scoring_summary.json").read_text(encoding="utf-8"))
    widths = scoring_summary["widths"]
    ledger: List[Mapping[str, Any]] = scoring_summary["ledger"]
    # 预定调仓日 = 引擎排期共享实现（20 日 2 批 ⇒ 每 10 个交易日一档）
    from src.lazybull.v2.common.rules.stagger import compute_tranche_schedule

    schedule = compute_tranche_schedule([pd.Timestamp(str(row["day"])) for row in ledger], 20, 2)
    scheduled = {day.strftime("%Y%m%d") for day in schedule}
    scheduled_narrow = {
        arm: [
            {"day": row["day"], "effective_n": row["per_arm"][arm]["effective_n"]}
            for row in ledger
            if row["day"] in scheduled and int(row["per_arm"][arm]["effective_n"]) < 500
        ]
        for arm in ("F", "R")
    }
    emit(
        f"    预定调仓日 {len(scheduled)} 个；<500 宽度 F={len(scheduled_narrow['F'])} "
        f"R={len(scheduled_narrow['R'])}"
    )

    account_metrics: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        for role in ("F100", "F80", "R100", "R80"):
            rows = _read_account_daily(run_dir, scenario, role)
            account_metrics[f"{role}@{scenario}"] = metrics.account_metrics(
                rows, f"{role}@{scenario}"
            )
        account_metrics[f"oldA100@{scenario}"] = metrics.read_old_anchor(scenario)
        emit(
            f"    本金档 {scenario}：R100 CAGR="
            f"{account_metrics[f'R100@{scenario}']['chain']['cagr']:.4f} "
            f"F100 CAGR={account_metrics[f'F100@{scenario}']['chain']['cagr']:.4f} "
            f"旧A CAGR={account_metrics[f'oldA100@{scenario}']['chain']['cagr']:.4f}"
        )

    gates: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        r = account_metrics[f"R100@{scenario}"]["chain"]
        f = account_metrics[f"F100@{scenario}"]["chain"]
        gates[str(scenario)] = metrics.r_gate_verdict(
            r["cagr"], -r["max_drawdown"], -f["max_drawdown"]
        )
    # b_required 使用账户实际期末总权益（daily 末行 total_value）
    b_rows: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        rows = _read_account_daily(run_dir, scenario, "R80")
        r80_metrics = account_metrics[f"R80@{scenario}"]
        b_rows.append(
            metrics.b_required_row(
                scenario,
                float(rows[-1]["total_value"]),
                r80_metrics["nav_points"],
            )
        )

    bootstrap: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        for leg in ("100", "80"):
            r_frame = metrics.chain_frame(
                _read_account_daily(run_dir, scenario, f"R{leg}"), f"R{leg}@{scenario}"
            )
            f_frame = metrics.chain_frame(
                _read_account_daily(run_dir, scenario, f"F{leg}"), f"F{leg}@{scenario}"
            )
            bootstrap[f"R{leg}-F{leg}@{scenario}"] = metrics.bootstrap_diff(
                f_frame, r_frame, label=f"R{leg}-F{leg}@{scenario}"
            )

    stress: Dict[str, Any] = {
        "slippage_per_side": 0.0010,
        "accounts": {},
        "not_run": ["cash", "portfolio"],
    }
    for scenario in SCENARIOS:
        for role in ("F100", "F80", "R100", "R80"):
            rows = _read_stress_daily(run_dir, scenario, role)
            stress["accounts"][f"{role}@{scenario}"] = metrics.account_metrics(
                rows, f"{role}@{scenario}@stress"
            )
    write_json(work / "stress_metrics.json", stress)

    a_rows: List[Dict[str, Any]] = []
    _append_metric_rows(a_rows, account_metrics, gates)
    _append_bootstrap_rows(a_rows, bootstrap)
    metrics.write_a_metrics_csv(work / "a_metrics.csv", a_rows)
    metrics.write_b_required_csv(work / "b_required.csv", b_rows)
    write_json(
        work / "metrics_full.json",
        {
            "accounts": account_metrics,
            "gates": gates,
            "b_required": b_rows,
            "bootstrap": bootstrap,
            "stress": stress,
            "widths": widths,
            "scheduled_days": len(scheduled),
            "scheduled_narrow": scheduled_narrow,
        },
    )
    review = _render_review(
        account_metrics,
        gates,
        b_rows,
        bootstrap,
        widths,
        scheduled_narrow,
        stress,
        baseline,
        ledger,
        deliver_seal.undated_bounds(run_dir),
    )
    (work / "a_review.md").write_text(review, encoding="utf-8", newline="\n")
    emit("    交付三件已生成（a_review.md / a_metrics.csv / b_required.csv）")
    return {"gates": gates}


def _append_metric_rows(
    rows: List[Dict[str, Any]],
    account_metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> None:
    segments_order = ["full", "s2019_2021", "s2022_2024", "s2025_end", "post_seam"] + [
        f"y{year}" for year in range(2019, 2027)
    ]
    for key, payload in account_metrics.items():
        capital, role = _split_key(key)
        for segment in segments_order:
            block = payload["segments"].get(segment)
            if block is None:
                continue
            for metric in ("cagr", "mdd_loss", "sharpe", "total_return", "trading_days"):
                value = block.get(metric)
                if value is None:
                    continue
                rows.append(
                    {
                        "capital": capital,
                        "account": role,
                        "segment": segment,
                        "metric": metric,
                        "value": value,
                    }
                )
    for scenario_text, gate in gates.items():
        rows.append(
            {
                "capital": scenario_text,
                "account": "R_GATE",
                "segment": "full",
                "metric": "distance_to_26_pp",
                "value": gate["distance_to_26_pp"],
            }
        )
        rows.append(
            {
                "capital": scenario_text,
                "account": "R_GATE",
                "segment": "full",
                "metric": "mdd_gap_pp_vs_f",
                "value": gate["mdd_gap_pp_vs_f"],
            }
        )
        rows.append(
            {
                "capital": scenario_text,
                "account": "R_GATE",
                "segment": "full",
                "metric": "cagr_gate_ge_26",
                "value": 1.0 if gate["cagr_gate_ge_26"] else 0.0,
            }
        )
        rows.append(
            {
                "capital": scenario_text,
                "account": "R_GATE",
                "segment": "full",
                "metric": "mdd_strictly_below_f",
                "value": 1.0 if gate["mdd_strictly_below_f"] else 0.0,
            }
        )
        rows.append(
            {
                "capital": scenario_text,
                "account": "R_GATE",
                "segment": "full",
                "metric": "both_gates",
                "value": 1.0 if gate["both_gates"] else 0.0,
            }
        )
    for scenario in SCENARIOS:
        for pair, delta in (
            ("R_MINUS_F", ("R100", "F100")),
            ("R_MINUS_OLDA", ("R100", "oldA100")),
            ("R80_MINUS_F80", ("R80", "F80")),
        ):
            left = account_metrics[f"{delta[0]}@{scenario}"]["chain"]
            right = account_metrics[f"{delta[1]}@{scenario}"]["chain"]
            rows.append(
                {
                    "capital": scenario,
                    "account": pair,
                    "segment": "full",
                    "metric": "cagr",
                    "value": left["cagr"] - right["cagr"],
                }
            )
            rows.append(
                {
                    "capital": scenario,
                    "account": pair,
                    "segment": "full",
                    "metric": "mdd_loss",
                    "value": (-left["max_drawdown"]) - (-right["max_drawdown"]),
                }
            )


def _split_key(key: str) -> tuple:
    role, _, capital = key.rpartition("@")
    return capital, role


def _append_bootstrap_rows(rows: List[Dict[str, Any]], bootstrap: Mapping[str, Any]) -> None:
    for pair, payload in bootstrap.items():
        for metric, values in payload.items():
            if not isinstance(values, Mapping):
                continue
            for stat, value in values.items():
                rows.append(
                    {
                        "capital": payload.get("label", ""),
                        "account": f"BOOTSTRAP_{pair}",
                        "segment": metric,
                        "metric": stat,
                        "value": value,
                    }
                )


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.4f}%"


def _render_review(
    account_metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
    b_rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    widths: Mapping[str, Any],
    scheduled_narrow: Mapping[str, Any],
    stress: Mapping[str, Any],
    baseline: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    undated_bounds: Mapping[str, Any],
) -> str:
    lines: List[str] = ["# CF-SPLIT-20-v1 审点二报告（run-a）", ""]
    lines.append(
        "研究目标（协议 §1）：100 万、200 万两档资金下，R 分别做到净 CAGR ≥26%，"
        "且回撤损失幅度严格低于同口径完整 A（F）。B 是否投入由本报告与用户裁决决定。"
    )
    lines.append("")
    _render_main_table(lines, account_metrics)
    _render_gates(lines, gates)
    _render_segments(lines, account_metrics)
    _render_b_table(lines, b_rows)
    _render_bootstrap(lines, bootstrap)
    _render_width(lines, widths, scheduled_narrow)
    _render_stress(lines, stress)
    deliver_seal.render_undated_bounds(lines, undated_bounds)
    _render_status(lines, gates, baseline, scheduled_narrow, undated_bounds)
    return "\n".join(lines) + "\n"


def _render_main_table(lines: List[str], account_metrics: Mapping[str, Any]) -> None:
    lines.append("## 1. 主读数（全窗 2019-04-01 ~ 2026-10-29，含锚 1839 点）")
    lines.append("")
    lines.append(
        "| 本金 | 账户 | CAGR | 回撤损失 mdd_loss | 夏普 | R−F (pp) | R−旧A (pp) | 距 26% (pp) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for scenario in SCENARIOS:
        for role in ("oldA100", "F100", "F80", "R100", "R80"):
            chain = account_metrics[f"{role}@{scenario}"]["chain"]
            extra = _main_table_extra(role, scenario, account_metrics)
            lines.append(
                f"| {scenario} | {role} | {_fmt_pct(chain['cagr'])} | "
                f"{_fmt_pct(-chain['max_drawdown'])} | {chain['sharpe']:.4f} | {extra} |"
            )
    lines.append("")


def _main_table_extra(role: str, scenario: int, account_metrics: Mapping[str, Any]) -> str:
    """主表尾部三列（R100 = R−F / R−旧A / 距 26%；R80 = R−F；其余空）。"""
    chain = account_metrics[f"{role}@{scenario}"]["chain"]
    if role == "R100":
        f_chain = account_metrics[f"F100@{scenario}"]["chain"]
        old_chain = account_metrics[f"oldA100@{scenario}"]["chain"]
        r_f = (chain["cagr"] - f_chain["cagr"]) * 100
        r_old = (chain["cagr"] - old_chain["cagr"]) * 100
        dist = (chain["cagr"] - 0.26) * 100
        return f"{r_f:+.3f} | {r_old:+.3f} | {dist:+.3f}"
    if role == "R80":
        f_chain = account_metrics[f"F80@{scenario}"]["chain"]
        return f"{(chain['cagr'] - f_chain['cagr']) * 100:+.3f} | | |"
    return "| | |"


def _render_gates(lines: List[str], gates: Mapping[str, Any]) -> None:
    lines.append("## 2. R 双门判定（未舍入浮点判定；等号含入 26% 门）")
    lines.append("")
    for scenario_text, gate in gates.items():
        lines.append(
            f"- {scenario_text} 档：CAGR 门（≥26%）= **{gate['cagr_gate_ge_26']}**"
            f"（距离 {gate['distance_to_26_pp'] * 100:+.4f}pp）；"
            f"回撤门（mdd_loss 严格 < F）= **{gate['mdd_strictly_below_f']}**"
            f"（差距 {gate['mdd_gap_pp_vs_f'] * 100:+.4f}pp）；"
            f"双门合取 = **{gate['both_gates']}**"
        )
    lines.append("")


def _render_segments(lines: List[str], account_metrics: Mapping[str, Any]) -> None:
    lines.append("## 3. 分段与分年（区间收益口径，年化仅描述）")
    lines.append("")
    names = ("s2019_2021", "s2022_2024", "s2025_end", "post_seam")
    for scenario in SCENARIOS:
        lines.append(f"### {scenario} 档（区间收益 total_return）")
        lines.append("")
        lines.append(
            "| 账户 | " + " | ".join(["2019-2021", "2022-2024", "2025~末", "接缝后"]) + " |"
        )
        lines.append("|---|---|---|---|---|")
        for role in ("oldA100", "F100", "R100"):
            payload = account_metrics[f"{role}@{scenario}"]["segments"]
            values = [_fmt_pct(payload.get(name, {}).get("total_return")) for name in names]
            lines.append(f"| {role} | " + " | ".join(values) + " |")
        lines.append("")


def _render_b_table(lines: List[str], b_rows: Sequence[Mapping[str, Any]]) -> None:
    lines.append("## 4. B 终值门槛（协议 §7.3；只解释收益缺口）")
    lines.append("")
    lines.append("| 本金 K | N | 目标等线 | R80 期末权益 | B 所需期末 | 折 0.2K 所需年化 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in b_rows:
        cagr = row.get("b_required_cagr_from_0.2k")
        shown = _fmt_pct(cagr) if cagr is not None else "n/a"
        lines.append(
            f"| {row['capital']} | {row['n_intervals']} | {row['target_total_value']:.2f} | "
            f"{row['r80_ending_value']:.2f} | {row['b_required_ending_value']:.2f} | "
            f"{shown} | {row['status']} |"
        )
    lines.append("")


def _render_bootstrap(lines: List[str], bootstrap: Mapping[str, Any]) -> None:
    lines.append("## 5. 路径不确定性（20 日循环块自举，2000 次，seed 42；R−F）")
    lines.append("")
    for pair, payload in bootstrap.items():
        cagr = payload["cagr_diff"]
        mdd = payload["mdd_loss_diff"]
        lines.append(
            f"- {pair}: ΔCAGR p5/p50/p95 = "
            f"{cagr['p5'] * 100:+.3f} / {cagr['p50'] * 100:+.3f} / {cagr['p95'] * 100:+.3f} pp"
            f"（P(>0)={cagr['prob_positive']:.3f}）；"
            f"Δmdd_loss p5/p50/p95 = {mdd['p5'] * 100:+.3f} / {mdd['p50'] * 100:+.3f} / "
            f"{mdd['p95'] * 100:+.3f} pp"
        )
    lines.append("")


def _render_width(
    lines: List[str], widths: Mapping[str, Any], scheduled_narrow: Mapping[str, Any]
) -> None:
    lines.append("## 6. 宽度与覆盖（协议 §5.1：各袖每个预定调仓日 ≥500）")
    lines.append("")
    for arm in ("F", "R"):
        block = widths[arm]
        lines.append(
            f"- {arm}：有效宽度最小值 {block['min_effective_n']}；<500 日 "
            f"{block['narrow_n']} 个（全部日历）；预定调仓日 <500 {len(scheduled_narrow[arm])} 个；"
            f"空排名日 {len(block['empty_days'])} 个"
        )
    lines.append("")


def _render_stress(lines: List[str], stress: Mapping[str, Any]) -> None:
    lines.append("## 7. 压力（滑点 5bp→10bp，其余不变；现金/主组合属 run-b = not_run）")
    lines.append("")
    for scenario in SCENARIOS:
        for role in ("F100", "R100"):
            payload = stress["accounts"][f"{role}@{scenario}"]["chain"]
            lines.append(
                f"- {role}@{scenario}@10bp: CAGR {_fmt_pct(payload['cagr'])}，"
                f"mdd_loss {_fmt_pct(-payload['max_drawdown'])}"
            )
    lines.append("")


def _render_status(
    lines: List[str],
    gates: Mapping[str, Any],
    baseline: Mapping[str, Any],
    scheduled_narrow: Mapping[str, Sequence[Mapping[str, Any]]],
    undated_bounds: Mapping[str, Any],
) -> None:
    lines.append("## 8. 状态词与限制")
    lines.append("")
    # checkpoint-2 Codex 第 6 项：分档独立记录，总体 = 两档均满足（all），禁用 any 聚合
    per_capital = {capital: bool(gate["both_gates"]) for capital, gate in gates.items()}
    all_pass = all(per_capital.values()) if per_capital else False
    if all_pass:
        status_word = "historical_target_met（以未舍入判定为准，见 metrics_full.json）"
    else:
        status_word = "economic_target_not_met"
    lines.append(f"- 状态词（§7.5）：**{status_word}**")
    for capital, passed in sorted(per_capital.items()):
        lines.append(f"- 本金档 {capital}：双门合取 = {'Pass' if passed else 'Fail'}")
    for arm in sorted(scheduled_narrow):
        days = scheduled_narrow[arm]
        lines.append(
            f"- 宽度状态（{arm}）：预定调仓日 <500 宽度 {len(days)} 日"
            + (f"（{[item['day'] for item in days]}）" if days else "")
        )
    if undated_bounds:
        lines.append(
            "- 未定日事件状态：**unbounded**（600602.SH 1991 年事件权益状态引擎不可判定；"
            "结清凭证不可得；两端测算见第 9 节）"
        )
    lines.append(
        "- 旧锚 → 同态 F 的桥接变化按「数据态 / 评分域（structural 恢复）/ 行业时点"
        "（PIT 注入）/ 训练实现（v2 栈 panel）/ 执行与费用」归因，逐项登记于 run manifest；"
        "拆分效应只读 F→R 差值，不把桥接差异记为拆袖收益。"
    )
    lines.append(
        "- 本轮为历史经济预筛（等级 H），formal_admission=not_evaluated、"
        "alpha_consumed=False；不宣称前向或日历验证。"
    )
    lines.append(
        f"- 旧 A 锚核验：{baseline['status']}；manifest SHA {baseline['manifest_sha256'][:16]}…"
    )
    lines.append("- 现金腿与主组合（R80+B20）属 run-b 范围，本轮 not_run。")
    lines.append("")


def seal(
    camp: Path,
    run_dir: Path,
    run_id: str,
    payload: Mapping[str, Any],
    extra_roots: Sequence[str],
) -> Dict[str, Any]:
    """动态封存：声明 run 目录下全部文件（work/attachments/封存件除外）。"""
    names = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and "work" not in path.relative_to(run_dir).parts
        and path.name not in (artifacts.MANIFEST_NAME, artifacts.PUBLISHED_NAME)
        and not path.relative_to(run_dir).parts[0] == artifacts.ATTACHMENTS_DIR
    )
    return artifacts.seal_run(run_dir, "run-a", dict(payload), names, extra_roots)


def build_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch",
        action="append",
        default=None,
        help="只执行指定批次（可多次给出；缺省 = 按序执行全部未完成批次）",
    )


__all__ = ["BATCHES", "RunAError", "run_run_a", "seal"]
