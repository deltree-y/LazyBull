# -*- coding: utf-8 -*-
"""P3 T5 三臂统一批驱动 + 阶段 B 新路径同码双跑（§19.1-A 核心工具；验收专用）。

**机器任务（正式三批 §14.5 重报 11.0~12.1h / 双跑 ≈15~20min），启动前须获用户授权。**

三臂分派（公共流程：隔离根 → 基线快照 → 臂执行 → 逐折产物 → 批指纹 → 链式指标 →
运行后守卫复核 → 证据 JSON；``verdict != PASS ⇒ exit 1``）：

- **臂① legacy_cs_train**：subprocess ``scripts/walk_forward.py``（生产口径逐项镜像 =
  泛化后的 ``build_spike_walk_forward_args``，窗口三参数自 ctx 透传，全折不带折子集
  flag，summary 名按 run_name）→ 训练完成后**追加 v2 内核 skip-training 回放**
  （薄入口组装编排器，``skip_training=True + start_model_version`` 按
  ``model_version = start + split_index`` 算术口径反解并对准隔离根台账版本；
  source_mode=cs_train_verify + g_code_parity）产出与臂②③同构的 runs 批 +
  链式证据；全 14 折时追加冻结 B0'；
- **臂② v2_cs_train_verify / 臂③ v2_panel**：in-process §19.1-B 13 步组装
  （``t5_arm_assembly.assemble_arm_orchestrator``，spec 驱动零硬编码 + 承接①②③门）；
- **runs 批统一产批路径（P3-5）**：三臂训练一律不写 runs 批，全折完成后统一经
  skip-training 回放产批（消除「训练批 vs 回放批」不同构）；续跑补训缺折后同样由
  回放产批；真实执行件（机器路径）在 ``t5_arm_executors``（复杂度门 P2-5 拆分）；
- **消费门臂感知（R2-T5A-02）**：``acquire_arm_baselines`` 传 ``arm``（臂③消费集合
  扩展到 panel 热/冷区 + labels 三表 + manifest 相关性基线）；臂③打分后核对打分
  store 实际读取 ⊆ 消费集合（越界计入 drift 通道 ⇒ verdict FAIL）；``manifest_drift``
  并入单/双跑判定，双跑 ``manifest_verify.json`` 随 run1/run2 证据落盘；预测行
  持久化登记相对臂目录 POSIX 路径（``predictions_anchor`` 接线，R2-T5A-06）；
- **double-run**（``--double-run``，阶段 B）：同一臂配置跑两遍（run1/run2 各自完整
  布局），四门比对（集成指纹逐位 / 逐种子诊断 / 覆盖日集 / 守卫+消费集合零漂移）；
- **折级断点续跑**（``--skip-existing``）：逐折产物齐备 ⇒ 先核验生产者身份 + 模型
  字节后复用；基线缺失拒绝独立于续跑开关、先于任何快照写入（§14.10 口径，
  经 ``t5_arm_artifacts.acquire_arm_baselines`` 复用 ``_acquire_baselines`` 原语义）。

用法（示例；真实执行须先获授权）::

    python scripts/v2_p3/t5_arm_runner.py --help
    python scripts/v2_p3/t5_arm_runner.py --arm v2_cs_train_verify --double-run \
        --selected-split-indices 0            # 阶段 B：1 折 × 3 种子 × 2 次
    python scripts/v2_p3/t5_arm_runner.py --arm legacy_cs_train --run-name t5_legacy_x
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3.t5_arm_artifacts import (  # noqa: E402
    ARM_LEGACY_CS_TRAIN,
    ARM_V2_PANEL,
    ARMS,
    FULL_SPLIT_COUNT,
    T5_BASE_DIR,
    acquire_arm_baselines,
    arm_config_digest,
    build_arm_producer_identity,
    collect_and_write_batch_fingerprint,
    collect_derive_call_site_report,
    compare_double_run,
    compute_and_write_chain_metrics,
    ensure_arm_root,
    freeze_b0_prime,
    load_fold_record_for_resume,
    runs_batch_state,
    verify_after_guards,
    write_arm_meta,
)
from scripts.v2_p3.t5_arm_assembly import (  # noqa: E402
    ACCEPTANCE_G_CODE_PARITY,
    DEPLOY_OFF_REASON,
    build_backtest_host_params,
    build_runs_batch_config,
)
from scripts.v2_p3.t5_arm_executors import (  # noqa: E402
    ArmExecutor,
    ArmRunContext,
    _plan_and_train,
    _score_fold,
    check_score_receipts,
    default_arm_executor,
)
from scripts.v2_p3.t05b_fingerprint import load_main_board_codes  # noqa: E402
from scripts.v2_p3.t05b_spike_runner import build_spike_walk_forward_args  # noqa: E402
from scripts.v2_p3.t05b_stock_connect import raw_read_guard, raw_read_ledger_snapshot  # noqa: E402

#: 默认 WF 口径（§13.3.3 冻结：14 折 / final_date 沿基线 / 测试窗 6 个月）
DEFAULT_SPLIT_COUNT = FULL_SPLIT_COUNT
DEFAULT_FINAL_DATE = "20260105"
DEFAULT_TEST_WINDOW_MONTHS = 6

# 执行上下文（ArmRunContext）/ 执行缝（ArmExecutor）/ 真实执行件（机器路径）拆至
# ``t5_arm_executors``（复杂度门 P2-5：本文件行数软上限收口）；上方再导出保持既有
# import 路径（``from scripts.v2_p3.t5_arm_runner import ArmExecutor ...``）可用。


# ---------------------------------------------------------------------------
# 配置摘要与 arm_meta
# ---------------------------------------------------------------------------


def build_arm_config_payload(
    arm: str,
    split_count: int,
    final_date: str,
    test_window_months: int,
    selected_split_indices: Sequence[int],
) -> Dict[str, object]:
    """臂配置摘要负载（**含本次选定折集**——P3-4 注释修正）。

    ``selected_split_indices`` 直接进入摘要：换子集「续跑」会改变 config_digest，
    被生产者身份校验拒绝（属预期防线——子集变更须换新 run_name 重新取证；
    折级断点续跑的合法语义 = 同一折集下复用已完成折，不属摘要漂移）。

    臂① = walk-forward 参数镜像（窗口三参数随入参透传，路径占位归一化，沿
    spike_config_digest 口径）+ 回放段参数；臂②③ = spec 驱动 runs 批次配置 +
    运行参数（承接②③门先过）。
    """
    payload: Dict[str, object] = {
        "arm": arm,
        "split_count": int(split_count),
        "final_date": str(final_date),
        "test_window_months": int(test_window_months),
        "selected_split_indices": [int(s) for s in selected_split_indices],
    }
    if arm == ARM_LEGACY_CS_TRAIN:
        payload["walk_forward_args"] = build_spike_walk_forward_args(
            Path("<ROOT>"),
            Path("<SUMMARY>"),
            selected_split_indices=None,
            split_count=int(split_count),
            final_date=str(final_date),
            test_window_months=int(test_window_months),
        )
        payload["v2_replay"] = {
            "source_mode": "cs_train_verify",
            "acceptance_purpose": ACCEPTANCE_G_CODE_PARITY,
            "skip_training": True,
            "skip_training_eval": False,
            "deploy_train": False,
        }
    else:
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        source_mode = "panel" if arm == ARM_V2_PANEL else "cs_train_verify"
        payload["runs_batch_config"] = build_runs_batch_config(
            spec, build_backtest_host_params(spec), source_mode
        )
        payload["deploy_train"] = False
    return payload


def _build_arm_meta(
    ctx: ArmRunContext, producer_identity: Mapping[str, object]
) -> Dict[str, object]:
    """arm_meta.json 内容（布局契约字段；config_digest 随 producer_identity 携带）。"""
    sleeve_identity: Optional[Dict[str, object]] = None
    if ctx.arm != ARM_LEGACY_CS_TRAIN:
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        sleeve_identity = {"sleeve_id": spec.sleeve_id, "spec_version": spec.spec_version}
    return {
        "arm": ctx.arm,
        "run_name": ctx.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "producer_identity": dict(producer_identity),
        "sleeve_identity": sleeve_identity,
        "input_snapshot_digest": producer_identity["input_snapshot_digest"],
        "split_count": ctx.split_count,
        "final_date": ctx.final_date,
        "test_window_months": ctx.test_window_months,
        "selected_split_indices": list(ctx.selected_split_indices),
        "deploy_train": {"enabled": False, "reason": DEPLOY_OFF_REASON},
        "layout_contract": "docs/plans/v2/p3_implementation_plan_20261020.md §19 T5 臂产物布局",
    }


# ---------------------------------------------------------------------------
# 折级执行（训练计划 / 覆盖门 / 单折打分采集在 ``t5_arm_executors``，P2-5 拆分）
# ---------------------------------------------------------------------------


def _fold_evidence_record(record: Mapping[str, object]) -> Dict[str, object]:
    """单折证据记录（闭市排除日非空时显式登记，§19.18）。"""
    fingerprint = record["fingerprint"]  # type: ignore[index]
    result: Dict[str, object] = {
        "split_index": record["split_index"],
        "model_version": record["model_version"],
        "fingerprint": fingerprint["fingerprint"],
        "n_days": fingerprint.get("n_days"),
        "n_rows": fingerprint.get("n_rows"),
        "reused": record["reused"],
    }
    excluded = fingerprint.get("excluded_days")
    if excluded:
        result["excluded_days"] = list(excluded)
    return result


def _raw_read_identities_from_baselines(baselines: Mapping[str, object]) -> Dict[str, str]:
    """消费快照正常记录 → 读取门登记身份（绝对 POSIX → sha256；R2-T5AE-R2-01）。

    与快照采集同源（``base_dir`` = 生产根父级）：显示路径经 resolve 后与评分侧
    （隔离根联接）实际读取路径一致；``missing`` 记录不登记（主臂消费集合只含
    采集时现存文件）。快照缺失/解析失败 ⇒ 显式报错（读取门禁用比静默弱门强）。
    """
    snapshot_path = Path(str(baselines["consumption_snapshot_path"]))
    records = json.loads(snapshot_path.read_text(encoding="utf-8"))
    base_dir = Path(str(baselines["base_dir"]))
    registered: Dict[str, str] = {}
    for record in records:
        if record.get("missing"):
            continue
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = base_dir / path
        if path.is_file():
            registered[path.resolve().as_posix()] = str(record["sha256"])
    if not registered:
        raise RuntimeError(f"读取门登记身份构建为空（快照异常）: {snapshot_path}")
    return registered


def _execute_arm_into(
    ctx: ArmRunContext,
    producer_identity: Mapping[str, object],
    executor: ArmExecutor,
    skip_existing: bool,
    link_fn: Optional[Callable[[Path, Path], None]],
    verify_links: bool,
    baselines: Mapping[str, object],
) -> Dict[str, object]:
    """单臂执行主流程（公共流程主体；double-run 时每子 run 各走一遍）。

    R2-T5A-02 后半：逐折打分完成后做打分侧消费核对（臂③实际读取 ⊆ 消费集合），
    报告入 ``score_receipt_check`` 键；越界不抛断流程，由顶层并入 drift 判定。
    """
    root = ensure_arm_root(
        ctx.arm_dir, ctx.production_data_root, link_fn=link_fn, verify_links=verify_links
    )
    if root != ctx.root:
        raise RuntimeError(f"隔离根解析与上下文不一致: {root} vs {ctx.root}")
    write_arm_meta(ctx.arm_dir, _build_arm_meta(ctx, producer_identity))

    reused: Dict[int, Dict[str, object]] = {}
    if skip_existing:
        for split in ctx.selected_split_indices:
            record = load_fold_record_for_resume(ctx.arm_dir, root, split, producer_identity)
            if record is not None:
                reused[split] = record

    train_evidence = _plan_and_train(ctx, executor, reused)
    replay_evidence = None
    if runs_batch_state(ctx.runs_root, ctx.selected_split_indices) != "complete":
        replay_evidence = executor.replay_fn(ctx)
        if runs_batch_state(ctx.runs_root, ctx.selected_split_indices) != "complete":
            raise RuntimeError(f"回放后 runs 批次仍不完整（{ctx.runs_root}），停止")

    main_board_codes = load_main_board_codes(root)
    score_store = None
    if ctx.arm == ARM_V2_PANEL:
        from src.lazybull.v2.store.data_store import PanelDataStore

        score_store = PanelDataStore(ctx.production_data_root)  # 生产 data_root 只读
    # 读取门（R2-T5AE-R2-01，主臂与补评分共用同一读取约束）：评分期间闭市判定
    # 实际读取须为消费快照在册成员且读时内容与跑前登记身份相符——「创建→读取→
    # 移除」/快照外实读/读时替换均在读取动作处立即拒绝。
    registered = _raw_read_identities_from_baselines(baselines)
    with raw_read_guard(missing_paths=(), registered_identities=registered):
        fold_records: List[Dict[str, object]] = []
        for split in ctx.selected_split_indices:
            if split in reused:
                fold_records.append(reused[split])
                continue
            fold_records.append(
                _score_fold(
                    ctx, split, producer_identity, executor.score_fn, main_board_codes, score_store
                )
            )
        # 读取台账证据（R2-T5AE-R3-01）：闭市判定实读记录（路径 + 实际解析字节哈希）
        raw_reads = [
            {"path": record.path, "sha256": record.sha256} for record in raw_read_ledger_snapshot()
        ]

    receipt_report = check_score_receipts(ctx, baselines, score_store)
    batch_doc = collect_and_write_batch_fingerprint(ctx.arm_dir)
    chain_doc = compute_and_write_chain_metrics(ctx.arm_dir)
    return {
        "train": train_evidence,
        "replay": replay_evidence,
        "score_receipt_check": receipt_report,
        "raw_reads": raw_reads,
        "folds": [_fold_evidence_record(record) for record in fold_records],
        "batch_fingerprint": batch_doc,
        "chain_metrics": chain_doc,
    }


# ---------------------------------------------------------------------------
# 顶层编排（单跑 / 双跑）
# ---------------------------------------------------------------------------


def _score_receipt_drift(execution: Mapping[str, object]) -> List[str]:
    """打分侧消费核对 ⇒ drift 清单形态（与 guard/consumption/manifest drift 同通道）。"""
    report = execution.get("score_receipt_check") or {}
    if report.get("skipped") or report.get("pass"):
        return []
    return [
        "打分侧实际读取越出消费集合（R2-T5A-02 后半）: " f"{report.get('error') or '核对未通过'}"
    ]


def _make_ctx(
    arm: str,
    run_name: str,
    arm_dir: Path,
    production_data_root: Path,
    split_count: int,
    final_date: str,
    test_window_months: int,
    selected: Tuple[int, ...],
    logs_dir: Path,
    log_tag: str,
) -> ArmRunContext:
    """执行上下文组装（路径唯一来源；日志命名含 run_name 与 run 标签）。

    R2-T5A-01：全部路径统一 ``Path.resolve()`` 为绝对路径——下游
    ``build_isolated_root`` 返回绝对路径，相对/绝对混合比较必抛
    「隔离根解析与上下文不一致」；入口规范化后只做同形态比较。
    """
    arm_dir_abs = Path(arm_dir).resolve()
    logs_dir_abs = Path(logs_dir).resolve()
    return ArmRunContext(
        arm=arm,
        run_name=run_name,
        arm_dir=arm_dir_abs,
        root=arm_dir_abs / "root",
        production_data_root=Path(production_data_root).resolve(),
        split_count=int(split_count),
        final_date=str(final_date),
        test_window_months=int(test_window_months),
        selected_split_indices=tuple(int(s) for s in selected),
        log_path=logs_dir_abs / f"v2_p3_t5_{run_name}_{log_tag}.log",
        replay_log_path=logs_dir_abs / f"v2_p3_t5_{run_name}_{log_tag}_v2replay.log",
    )


def _derive_gate_report() -> Dict[str, object]:
    """承接④ derive 调用点核验段（漂移不抛断流程——登记进证据并判 FAIL）。"""
    try:
        return collect_derive_call_site_report()
    except ValueError as exc:
        logger.error(f"承接④ derive 调用点白名单漂移: {exc}")
        return {"whitelist_match": False, "error": str(exc)}


def _assert_compare_arm_source_digest(arm: str, compare_arm_dir: Optional[Path]) -> Optional[str]:
    """臂③运行前 source_digest 前置校验（P2-4，§13.3.8；任何训练/快照动作前）。

    仅 ``--arm v2_panel`` 强制要求 ``compare_arm_dir``（臂② run 目录）；其他臂传了
    也校验。读取臂② ``arm_meta.json`` 的 ``producer_identity.source_digest``，与当前
    ``workspace_identity(repo_root)["source_digest"]`` 比较：不一致 ⇒ 拒绝启动并提示
    重跑臂②（代码态漂移下两臂产物不可比，禁止带病起跑）。返回被核验摘要（未启用
    ⇒ None）。
    """
    if compare_arm_dir is None:
        if arm == ARM_V2_PANEL:
            raise ValueError(
                "--arm v2_panel 必须显式提供 --compare-arm-dir（臂② run 目录），"
                "用于运行前 source_digest 前置校验（P2-4，§13.3.8）"
            )
        return None
    meta_path = Path(compare_arm_dir) / "arm_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"compare-arm-dir 缺 arm_meta.json（须为已完成基线获取的臂② run 目录）: {meta_path}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = (meta.get("producer_identity") or {}).get("source_digest")
    if not digest:
        raise ValueError(f"臂② arm_meta.json 缺 producer_identity.source_digest: {meta_path}")
    from src.lazybull.v2.store.store_identity import workspace_identity

    current = workspace_identity(_REPO_ROOT)["source_digest"]
    if str(digest) != str(current):
        raise RuntimeError(
            f"臂② source_digest 与当前代码态不一致（{str(digest)[:16]}… != "
            f"{str(current)[:16]}…），拒绝启动：请重跑臂②（同代码态）后再执行本臂"
            "（P2-4，§13.3.8）"
        )
    logger.info(f"compare-arm 前置校验通过：source_digest {str(digest)[:16]}… 与当前一致")
    return str(digest)


def _normalize_selected(
    selected_split_indices: Optional[Sequence[int]], split_count: int
) -> Tuple[int, ...]:
    """折子集归一：None/空 ⇒ 全折；显式 ⇒ 去重保序 + 界内校验。"""
    if not selected_split_indices:
        return tuple(range(split_count))
    selected = tuple(sorted({int(s) for s in selected_split_indices}))
    out_of_range = [s for s in selected if s < 0 or s >= split_count]
    if out_of_range:
        raise ValueError(
            f"--selected-split-indices 越界（split_count={split_count}）: {out_of_range}"
        )
    return selected


def _maybe_freeze_b0_prime(
    ctx: ArmRunContext,
    producer_identity: Mapping[str, object],
    guard_drift: Sequence[str],
    consumption_drift: Sequence[str],
) -> Optional[str]:
    """B0' 冻结触发判定与执行（臂① + 全 14 折 + 守卫零漂移；非双跑模式）。"""
    is_full = ctx.selected_split_indices == tuple(range(FULL_SPLIT_COUNT))
    if ctx.arm != ARM_LEGACY_CS_TRAIN or not is_full or ctx.split_count != FULL_SPLIT_COUNT:
        return None
    if guard_drift or consumption_drift:
        logger.error("守卫/消费集合存在漂移，跳过 B0' 冻结（先处置漂移再冻结）")
        return None
    target = freeze_b0_prime(
        ctx.arm_dir,
        run_name=ctx.run_name,
        producer_identity=producer_identity,
        config_digest=str(producer_identity["config_digest"]),
    )
    return str(target)


def run_arm(
    *,
    arm: str,
    run_name: str,
    production_data_root: Path,
    split_count: int = DEFAULT_SPLIT_COUNT,
    final_date: str = DEFAULT_FINAL_DATE,
    test_window_months: int = DEFAULT_TEST_WINDOW_MONTHS,
    selected_split_indices: Optional[Sequence[int]] = None,
    double_run: bool = False,
    skip_existing: bool = False,
    out: Optional[Path] = None,
    base_dir: Path = T5_BASE_DIR,
    compare_arm_dir: Optional[Path] = None,
    executor: Optional[ArmExecutor] = None,
    link_fn: Optional[Callable[[Path, Path], None]] = None,
    verify_links: bool = True,
    logs_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """臂运行主入口：基线 → 执行（单跑/双跑）→ 守卫复核 → 证据落盘 + verdict。

    R2-T5A-01：``base_dir`` / ``logs_dir`` / ``out`` 入口统一 ``resolve()`` 为绝对
    路径（默认 ``T5_BASE_DIR`` 为相对路径时同样可执行）；P2-4（§13.3.8）：
    ``compare_arm_dir`` 前置校验先于任何训练/快照动作。
    """
    if arm not in ARMS:
        raise ValueError(f"未知臂: {arm!r}（合法值 {ARMS}）")
    # P2-4：臂③（及其他臂传了也校验）source_digest 前置校验——先于任何目录/快照写入
    compare_source_digest = _assert_compare_arm_source_digest(arm, compare_arm_dir)
    base_dir = Path(base_dir).resolve()
    production_data_root = Path(production_data_root).resolve()
    selected = _normalize_selected(selected_split_indices, split_count)
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is None:
        from src.lazybull.common.config import get_logs_dir

        logs_dir = Path(get_logs_dir())
    logs_dir = Path(logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    if executor is None:
        executor = default_arm_executor(arm)

    # 基线获取（新建/续跑核验；基线缺失拒绝独立于 --skip-existing、先于任何快照写入；
    # R2-T5A-02：臂③消费集合扩展到 panel 热/冷区 + labels 三表 + manifest 相关性基线）
    baselines = acquire_arm_baselines(run_dir, production_data_root, skip_existing, arm=arm)
    config_payload = build_arm_config_payload(
        arm, split_count, final_date, test_window_months, selected
    )
    config_digest = arm_config_digest(config_payload)
    derive_gate = _derive_gate_report()

    evidence: Dict[str, object] = {
        "run_name": run_name,
        "arm": arm,
        "mode": "double_run" if double_run else "single",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "production_data_root": str(production_data_root),
        "params": {
            "split_count": split_count,
            "final_date": final_date,
            "test_window_months": test_window_months,
            "selected_split_indices": list(selected),
            "double_run": double_run,
            "skip_existing": skip_existing,
            "compare_arm_dir": str(compare_arm_dir) if compare_arm_dir else None,
            "compare_source_digest": compare_source_digest,
        },
        "baselines": {
            "guard_snapshot": str(baselines["guard_snapshot_path"]),
            "consumption_snapshot": str(baselines["consumption_snapshot_path"]),
            "consumption_counts": baselines["consumption_counts"],
            "input_snapshot_digest": baselines["input_snapshot_digest"],
            "resumed": baselines["resumed"],
        },
        "config_payload": config_payload,
        "config_digest": config_digest,
        "derive_call_sites": derive_gate,
    }

    if double_run:
        comparison, sub_results = _run_double(
            arm,
            run_name,
            run_dir,
            production_data_root,
            split_count,
            final_date,
            test_window_months,
            selected,
            logs_dir,
            baselines,
            config_digest,
            executor,
            skip_existing,
            link_fn,
            verify_links,
        )
        evidence["runs"] = sub_results
        evidence["comparison"] = comparison
        verdict = comparison["verdict"]
    else:
        verdict = _run_single(
            arm,
            run_name,
            run_dir,
            production_data_root,
            split_count,
            final_date,
            test_window_months,
            selected,
            logs_dir,
            baselines,
            config_digest,
            executor,
            skip_existing,
            link_fn,
            verify_links,
            evidence,
        )
    if not derive_gate.get("whitelist_match"):
        verdict = "FAIL"
    evidence["verdict"] = verdict
    evidence["finished_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = Path(out).resolve() if out else run_dir / "t5_arm_evidence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"T5 臂证据落盘: {out_path}（verdict={verdict}）")
    return evidence


def _run_single(
    arm: str,
    run_name: str,
    run_dir: Path,
    production_data_root: Path,
    split_count: int,
    final_date: str,
    test_window_months: int,
    selected: Tuple[int, ...],
    logs_dir: Path,
    baselines: Mapping[str, object],
    config_digest: str,
    executor: ArmExecutor,
    skip_existing: bool,
    link_fn: Optional[Callable[[Path, Path], None]],
    verify_links: bool,
    evidence: Dict[str, object],
) -> str:
    """单跑模式：执行 → 运行后守卫复核 → B0' 冻结（若臂①全折）→ verdict。

    R2-T5A-02：verdict 并入 ``manifest_drift``（臂③ manifest 相关性段）与打分侧
    消费核对 drift（``score_receipt_check`` 越界 ⇒ FAIL，与既有 drift 同通道）。
    """
    ctx = _make_ctx(
        arm,
        run_name,
        run_dir,
        production_data_root,
        split_count,
        final_date,
        test_window_months,
        selected,
        logs_dir,
        log_tag=arm,
    )
    producer_identity = build_arm_producer_identity(
        arm, config_digest, str(baselines["input_snapshot_digest"])
    )
    execution = _execute_arm_into(
        ctx, producer_identity, executor, skip_existing, link_fn, verify_links, baselines
    )
    evidence["execution"] = execution
    guards = verify_after_guards(ctx.arm_dir, baselines)
    evidence["guards"] = guards
    receipt_drift = _score_receipt_drift(execution)
    b0_prime = _maybe_freeze_b0_prime(
        ctx, producer_identity, guards["guard_drift"], guards["consumption_drift"]
    )
    evidence["b0_prime"] = b0_prime
    passed = (
        not guards["guard_drift"]
        and not guards["consumption_drift"]
        and not guards.get("manifest_drift")
        and not receipt_drift
    )
    return "PASS" if passed else "FAIL"


def _run_double(
    arm: str,
    run_name: str,
    run_dir: Path,
    production_data_root: Path,
    split_count: int,
    final_date: str,
    test_window_months: int,
    selected: Tuple[int, ...],
    logs_dir: Path,
    baselines: Mapping[str, object],
    config_digest: str,
    executor: ArmExecutor,
    skip_existing: bool,
    link_fn: Optional[Callable[[Path, Path], None]],
    verify_links: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """双跑模式（阶段 B）：同码两遍（run1/run2 各自完整布局）→ 四门比对 ⇒ comparison。

    R2-T5A-02：``manifest_drift``（有基线时）与两子 run 打分侧消费核对 drift 一并
    并入 ``compare_double_run`` 的 drift 输入；``manifest_verify.json`` 有基线时随
    run1/run2 证据同步落盘。
    """
    sub_results: List[Dict[str, object]] = []
    for run_index in (1, 2):
        ctx = _make_ctx(
            arm,
            run_name,
            run_dir / f"run{run_index}",
            production_data_root,
            split_count,
            final_date,
            test_window_months,
            selected,
            logs_dir,
            log_tag=f"{arm}_run{run_index}",
        )
        producer_identity = build_arm_producer_identity(
            arm, config_digest, str(baselines["input_snapshot_digest"]), run_index=run_index
        )
        sub_results.append(
            _execute_arm_into(
                ctx, producer_identity, executor, skip_existing, link_fn, verify_links, baselines
            )
        )
    guards = verify_after_guards(run_dir, baselines)
    # run1/run2 各自完整布局：复核结果同步落两份（内容一致，共享基线）
    for run_index in (1, 2):
        sub_dir = run_dir / f"run{run_index}"
        names = ["after_guard.json", "consumption_verify.json"]
        if (run_dir / "manifest_verify.json").exists():  # 臂③ manifest 基线段
            names.append("manifest_verify.json")
        for name in names:
            source = run_dir / name
            (sub_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    drift_all = (
        list(guards["consumption_drift"])
        + list(guards.get("manifest_drift") or [])
        + [item for sub in sub_results for item in _score_receipt_drift(sub)]
    )
    comparison = compare_double_run(
        run_dir / "run1",
        run_dir / "run2",
        guards["guard_drift"],
        drift_all,
    )
    return comparison, sub_results


def main() -> None:
    # 控制台编码：非 UTF-8 流一律重配（仓库先例只判 isatty，Windows 下 /dev/null=NUL
    # 字符设备 isatty=True 会漏配 ⇒ GBK 打印崩溃；按编码判定同时修好交互 GBK 控制台）
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") not in ("utf8",):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(
        description="P3 T5 三臂统一批驱动 + 阶段 B 同码双跑（机器任务，须先获用户授权 §7.4-9）"
    )
    parser.add_argument("--arm", required=True, choices=list(ARMS), help="验收臂")
    parser.add_argument(
        "--run-name",
        default=None,
        help="运行名（默认 t5_<arm>_<时间戳>）；复用同名目录可配合 --skip-existing 续跑",
    )
    parser.add_argument("--production-data-root", default="./data", help="生产数据根")
    parser.add_argument("--split-count", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--final-date", default=DEFAULT_FINAL_DATE)
    parser.add_argument("--test-window-months", type=int, default=DEFAULT_TEST_WINDOW_MONTHS)
    parser.add_argument(
        "--selected-split-indices",
        type=int,
        nargs="*",
        default=None,
        help="折子集（如 0 4 5）；留空 = 全折",
    )
    parser.add_argument(
        "--double-run",
        action="store_true",
        help="阶段 B 模式：同码双跑 run1/run2 + 四门比对",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="折级断点续跑：逐折产物齐备 ⇒ 先核验生产者身份/模型字节后复用",
    )
    parser.add_argument(
        "--out", default=None, help="证据/判定 JSON 落盘路径（默认 <run_dir>/t5_arm_evidence.json）"
    )
    parser.add_argument(
        "--base-dir",
        default=str(T5_BASE_DIR),
        help="冻结参照根（默认 data/frozen_reference/v2_p3；验收固定，测试可注入）",
    )
    parser.add_argument(
        "--compare-arm-dir",
        default=None,
        help="臂② run 目录（运行前 source_digest 前置校验，P2-4/§13.3.8；"
        "--arm v2_panel 强制要求，其他臂传了也校验）",
    )
    args = parser.parse_args()

    run_name = args.run_name or f"t5_{args.arm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evidence = run_arm(
        arm=args.arm,
        run_name=run_name,
        production_data_root=Path(args.production_data_root),
        split_count=args.split_count,
        final_date=args.final_date,
        test_window_months=args.test_window_months,
        selected_split_indices=args.selected_split_indices,
        double_run=args.double_run,
        skip_existing=args.skip_existing,
        out=Path(args.out) if args.out else None,
        base_dir=Path(args.base_dir),
        compare_arm_dir=Path(args.compare_arm_dir) if args.compare_arm_dir else None,
    )
    if evidence["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
