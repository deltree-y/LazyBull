# -*- coding: utf-8 -*-
"""P3 T5 臂运行真实执行件（机器路径；自 ``t5_arm_runner`` 拆出，复杂度门 P2-5 处置）。

本模块承载执行上下文与重操作执行缝的**真实实现**；编排主流程（基线 → 执行 →
守卫复核 → 证据）仍在 ``t5_arm_runner``。测试一律注入合成替身，禁止触达本模块
真实训练路径。

- ``ArmRunContext``：单臂（或双跑单子 run）执行上下文（路径唯一来源；
  ``_make_ctx`` 统一 ``resolve()`` 为绝对路径，R2-T5A-01）；
- ``ArmExecutor``：重操作执行缝（train/replay/score 三 callable）；
- ``_real_legacy_train`` / ``_real_v2_train`` / ``_real_v2_replay``：真实执行件；
- ``default_arm_executor``：按臂装配真实执行缝。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3.t5_arm_artifacts import (  # noqa: E402
    ARM_LEGACY_CS_TRAIN,
    ARM_V2_PANEL,
    assert_score_receipt_within_consumption,
    collect_score_read_receipt_files,
    fold_dir,
    fold_name,
    resolve_fold_model_version,
    resolve_skip_start_model_version,
    write_fold_artifacts,
)
from scripts.v2_p3.t5_arm_assembly import (  # noqa: E402
    ACCEPTANCE_G_CODE_PARITY,
    assemble_arm_orchestrator,
)
from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    _load_model_feature_columns,
    resolve_panel_score_days,
    score_model_fingerprint,
)
from scripts.v2_p3.t05b_spike_runner import (  # noqa: E402
    _EMPTY_SHA256,
    _run_walk_forward,
    build_spike_walk_forward_args,
    collect_model_identity,
    list_expected_oos_days,
    read_split_boundary,
    resolve_score_days,
)


@dataclass(frozen=True)
class ArmRunContext:
    """单臂（或双跑单子 run）执行上下文。"""

    arm: str
    run_name: str
    arm_dir: Path
    root: Path
    production_data_root: Path
    split_count: int
    final_date: str
    test_window_months: int
    selected_split_indices: Tuple[int, ...]
    log_path: Path
    replay_log_path: Path

    @property
    def runs_root(self) -> Path:
        return self.arm_dir / "runs"

    @property
    def models_dir(self) -> Path:
        return self.root / "models" / "stock_selection"


@dataclass(frozen=True)
class ArmExecutor:
    """重操作执行缝（默认真实实现；测试注入合成替身，禁止真实训练）。

    - ``train_fn(ctx, subset, write_batch)``: 训练/注册（臂① = subprocess 旧 runner；
      臂②③ = 编排器训练运行；``write_batch=False`` ⇒ 不写 runs 批，由回放统一产出——
      P3-5 起三臂训练一律不写批，批次统一经 skip-training 回放产出）；
    - ``replay_fn(ctx)``: skip-training 全折回放（三臂统一产批路径）；
    - ``score_fn``: ``score_model_fingerprint`` 同签名可调用件。
    """

    train_fn: Callable[[ArmRunContext, Tuple[int, ...], bool], Dict[str, object]]
    replay_fn: Callable[[ArmRunContext], Dict[str, object]]
    score_fn: Callable[..., Dict[str, object]]


def _log_sink(log_path: Path) -> int:
    """in-process 训练/回放日志落盘（loguru 文件 sink；逐折边界解析依赖该日志）。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return logger.add(str(log_path), level="INFO", encoding="utf-8")


def _real_legacy_train(
    ctx: ArmRunContext, subset: Tuple[int, ...], write_batch: bool
) -> Dict[str, object]:
    """臂①：subprocess ``scripts/walk_forward.py``（生产口径逐项镜像；不写 runs 批）。

    ``write_batch`` 对臂①无意义（旧 runner 不写 runs 批次，批次由 skip 回放产出）。
    全折（subset == 全量）时不带 ``--selected-split-indices``（生产全折语义）。
    窗口三参数（split_count / final_date / test_window_months）自 ctx 透传
    （P2-1 ≡ R2-T5A-08：用户参数与真实命令挂钩，不再硬编码静默无效）。
    """
    del write_batch  # 臂①训练永不写批（§19.1-B 末段口径）
    summary_csv = ctx.root / "walk_forward" / "raw" / f"walk_forward_summary_{ctx.run_name}.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    full = tuple(range(ctx.split_count))
    flag: Optional[Tuple[str, ...]] = (
        None if tuple(subset) == full else tuple(str(i) for i in subset)
    )
    args = build_spike_walk_forward_args(
        ctx.root,
        summary_csv,
        selected_split_indices=flag,
        split_count=ctx.split_count,
        final_date=ctx.final_date,
        test_window_months=ctx.test_window_months,
    )
    wall_seconds = _run_walk_forward(ctx.root, ctx.log_path, args=args)
    return {
        "mode": "subprocess_walk_forward",
        "wall_seconds": round(wall_seconds, 1),
        "summary_csv": str(summary_csv),
        "trained_splits": list(subset),
        "train_log": str(ctx.log_path),
    }


def _run_orchestrator_with_log(ctx: ArmRunContext, assembly: object, **run_kwargs: object) -> str:
    """编排器 run + 日志 sink 生命周期（ sink 移除在 finally，防泄漏）。"""
    sink_id = _log_sink(Path(str(run_kwargs.pop("log_path"))))
    try:
        return assembly.orchestrator.run(  # type: ignore[attr-defined]
            assembly.config,  # type: ignore[attr-defined]
            ctx.split_count,
            ctx.final_date,
            assembly.trainer,  # type: ignore[attr-defined]
            assembly.store,  # type: ignore[attr-defined]
            test_window_months=ctx.test_window_months,
            **run_kwargs,
        )
    finally:
        logger.remove(sink_id)


def _real_v2_train(
    ctx: ArmRunContext, subset: Tuple[int, ...], write_batch: bool
) -> Dict[str, object]:
    """臂②③：编排器训练运行（§19.1-B 组装；P3-5 起训练不写批，批次由回放产出）。"""
    full = tuple(range(ctx.split_count))
    assembly = assemble_arm_orchestrator(
        ctx.arm,
        ctx.root,
        ctx.runs_root,
        run_name=ctx.run_name,
        summary_name=ctx.run_name if write_batch else None,
        export_topk_details=True,
        write_batch=write_batch,
    )
    wf_run_id = _run_orchestrator_with_log(
        ctx,
        assembly,
        log_path=ctx.log_path,
        selected_split_indices=None if tuple(subset) == full else list(subset),
        skip_training=False,
        deploy_train=False,
    )
    return {
        "mode": "v2_orchestrator_train",
        "wf_run_id": wf_run_id,
        "trained_splits": list(subset),
        "write_batch": write_batch,
        "gate_evidence": assembly.gate_evidence,
        "train_log": str(ctx.log_path),
    }


def _real_v2_replay(ctx: ArmRunContext) -> Dict[str, object]:
    """skip-training 全折回放（三臂统一产批路径 = G-code ② 同构证据）。

    ``start_model_version`` 按编排器 skip 路径 ``model_version = start + split_index``
    （orchestrator.py:803）算术口径从隔离根台账反解，全折一致性断言在
    ``resolve_skip_start_model_version`` 内；回放不写 topk 明细（防 G-code ③
    topk glob 与训练产物双命中）。
    """
    start = resolve_skip_start_model_version(ctx.root, ctx.selected_split_indices)
    assembly = assemble_arm_orchestrator(
        ctx.arm,
        ctx.root,
        ctx.runs_root,
        run_name=ctx.run_name,
        summary_name=f"{ctx.run_name}_v2replay",
        export_topk_details=False,
        write_batch=True,
    )
    wf_run_id = _run_orchestrator_with_log(
        ctx,
        assembly,
        log_path=ctx.replay_log_path,
        selected_split_indices=list(ctx.selected_split_indices),
        skip_training=True,
        start_model_version=start,
        skip_training_eval=False,
        deploy_train=False,
    )
    return {
        "mode": "v2_orchestrator_skip_training_replay",
        "wf_run_id": wf_run_id,
        "start_model_version": start,
        "skip_training_eval": False,
        "acceptance_purpose": ACCEPTANCE_G_CODE_PARITY,
        "gate_evidence": assembly.gate_evidence,
        "replay_log": str(ctx.replay_log_path),
    }


def default_arm_executor(arm: str) -> ArmExecutor:
    """真实执行缝（机器路径）；测试注入合成替身。"""
    train_fn = _real_legacy_train if arm == ARM_LEGACY_CS_TRAIN else _real_v2_train
    return ArmExecutor(
        train_fn=train_fn, replay_fn=_real_v2_replay, score_fn=score_model_fingerprint
    )


# ---------------------------------------------------------------------------
# 折级执行段助手（训练计划 / 覆盖门；自 runner 拆出，复杂度门 P2-5）
# ---------------------------------------------------------------------------


def _ledger_versions(root: Path, selected: Sequence[int]) -> Dict[int, Optional[int]]:
    """台账逐折版本扫描（无台账/缺行 ⇒ None = 该折未注册，须训练）。"""
    result: Dict[int, Optional[int]] = {int(split): None for split in selected}
    csv_path = Path(root) / "models" / "stock_selection" / "ml_train_runs.csv"
    if not csv_path.exists():
        return result
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    for split in selected:
        rows = df[df["split_index"].astype(str) == str(split)]
        if not rows.empty:
            result[int(split)] = int(rows.iloc[-1]["model_version"])
    return result


def _assert_fold_coverage(
    fingerprint: Mapping[str, object], expected_days: Sequence[str], split_index: int
) -> None:
    """折级覆盖完整性门（沿 spike R2-T05B-03/N3 口径到折粒度）。

    窄口径合法闭市日（规划 §19.18）：``score_days``（已评分日）∪
    ``excluded_days``（登记闭市日）必须精确等于独立预期日；两者不得重叠。
    未登记闭市日的任何缺失/拒绝仍按原口径失败。
    """
    used = [str(day) for day in fingerprint.get("score_days") or []]
    excluded = [str(day) for day in fingerprint.get("excluded_days") or []]
    overlap = sorted(set(used) & set(excluded))
    if overlap:
        raise RuntimeError(
            f"折 {fold_name(split_index)} 打分日与闭市排除日重叠 {overlap[:10]}，"
            "覆盖口径自相矛盾（§19.18）"
        )
    reconciled = sorted(set(used) | set(excluded))
    if reconciled != sorted(expected_days):
        raise RuntimeError(
            f"折 {fold_name(split_index)} 打分日 + 闭市排除日与独立预期日不一致"
            f"（{len(used)}+{len(excluded)}/{len(expected_days)} 日），禁止缩域后放行（N3）"
        )
    if int(fingerprint.get("n_rows") or 0) <= 0 or fingerprint.get("fingerprint") == _EMPTY_SHA256:
        raise RuntimeError(
            f"折 {fold_name(split_index)} 空预测批次，禁止产出通过指纹（R2-T05B-03）"
        )


def _signal_selection_mapping() -> Dict[str, object]:
    """信号选择参数映射（跨路契约：``score_model_fingerprint(signal_selection=...)``）。

    三臂统一：``top_n`` / ``min_amount_ma20`` / ``exclude_financial`` 自
    ``get_sleeve_spec("value")``（spec = 生产配置代码化单一来源）；
    ``min_total_mv`` / ``max_total_mv`` 自 ``resolve_stock_domain(spec.domain)``
    （``universe/domains.py`` 单一来源，与薄入口组装期 ``build_signal_factory``
    同源同值，防双源漂移）。提供时跨路契约要求 ``persist_predictions=True``
    （``_score_fold`` 恒满足）。
    """
    from src.lazybull.universe.domains import resolve_stock_domain
    from src.lazybull.v2.sleeves._registry import get_sleeve_spec

    spec = get_sleeve_spec("value")
    domain_cfg = resolve_stock_domain(spec.domain)
    return {
        "top_n": int(spec.top_n),
        "min_amount_ma20": float(spec.min_amount_ma20),
        "exclude_financial": bool(spec.exclude_financial),
        "min_total_mv": float(domain_cfg["min_total_mv_wan"]),
        "max_total_mv": float(domain_cfg["max_total_mv_wan"]),
    }


def _signal_universe_config_mapping() -> Dict[str, object]:
    """候选池与真实回放共用宿主参数和域定义，不另行维护上市/ST 过滤阈值。"""
    from scripts.v2_p3.t5_arm_assembly import build_backtest_host_params
    from src.lazybull.universe.domains import domain_market_whitelist
    from src.lazybull.v2.sleeves._registry import get_sleeve_spec

    spec = get_sleeve_spec("value")
    params = build_backtest_host_params(spec)
    return {
        "exclude_st": params.bt_exclude_st,
        "min_list_days": params.bt_min_list_days,
        "markets": tuple(domain_market_whitelist(spec.domain)),
    }


def _score_fold(
    ctx: ArmRunContext,
    split_index: int,
    producer_identity: Mapping[str, object],
    score_fn: Callable[..., Dict[str, object]],
    main_board_codes: object,
    score_store: object,
) -> Dict[str, object]:
    """单折产物采集：版本解析 → 边界 → 预期日 → 打分 → 覆盖门 → 三件套落盘。

    预测行持久化接线（跨路契约 R2-T5A-06 + 信号 top-k 产物）：所有臂统一传
    ``persist_predictions=True`` + ``predictions_path=<fold目录>/predictions.parquet``
    + ``predictions_anchor=ctx.arm_dir``（结果 ``predictions_path`` 登记为相对臂目录
    的 POSIX 形态，如 ``folds/split00/predictions.parquet``——归因消费侧
    ``t5_attribution_io.load_fold_predictions`` 只接受该口径）+
    ``signal_selection``（真实信号路径选择参数映射，spec/域市值同源，见
    ``_signal_selection_mapping``）；打分结果 JSON 自然携带
    ``predictions_*/signal_topk_*`` 键并随 fingerprint.json 落盘（五件套口径，
    续跑核验见 ``load_fold_record_for_resume``）。
    """
    model_version = resolve_fold_model_version(ctx.root, split_index)
    boundary = read_split_boundary(ctx.root, ctx.log_path, split_index)
    test_start, test_end = str(boundary["test_start"]), str(boundary["test_end"])
    expected_days = list_expected_oos_days(ctx.root, test_start, test_end)
    extra_kwargs: Dict[str, object] = {}
    if ctx.arm == ARM_V2_PANEL:
        # 臂③ panel 模式：store = PanelDataStore(生产 data_root 只读)；读取列契约 =
        # 模型登记特征列（t05b_fingerprint._load_model_feature_columns）
        score_days = resolve_panel_score_days(expected_days, score_store, test_start, test_end)
        extra_kwargs = {
            "source_mode": "panel",
            "store": score_store,
            "feature_columns": _load_model_feature_columns(ctx.models_dir, model_version),
        }
    else:
        score_days = resolve_score_days(expected_days, ctx.root / "features")
    fingerprint = score_fn(
        models_dir=ctx.models_dir,
        model_version=model_version,
        data_root=ctx.root,
        score_days=score_days,
        main_board_codes=main_board_codes,
        producer_identity=dict(producer_identity),
        persist_predictions=True,
        predictions_path=fold_dir(ctx.arm_dir, split_index) / "predictions.parquet",
        predictions_anchor=ctx.arm_dir,
        signal_selection=_signal_selection_mapping(),
        signal_universe_config=_signal_universe_config_mapping(),
        **extra_kwargs,
    )
    _assert_fold_coverage(fingerprint, expected_days, split_index)
    fingerprint["model_identity"] = collect_model_identity(ctx.root, model_version)
    write_fold_artifacts(ctx.arm_dir, split_index, model_version, boundary, fingerprint)
    logger.info(
        f"折 {fold_name(split_index)} 产物落盘: v{model_version}，指纹 {fingerprint['fingerprint']}"
    )
    return {
        "split_index": int(split_index),
        "model_version": model_version,
        "boundary": boundary,
        "fingerprint": fingerprint,
        "reused": False,
    }


def _plan_and_train(
    ctx: ArmRunContext,
    executor: ArmExecutor,
    reused: Mapping[int, Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    """训练计划与执行：缺台账行的折 ⇒ 训练；返回训练段证据（无训练 ⇒ None）。

    批次写出策略（P3-5 两臂 runs 产批路径统一）：三臂训练**一律不写批**
    （``write_batch=False``），runs 批统一由随后全折 skip-training 回放产出——
    消除「训练批 vs 回放批」不同构；续跑补训缺折后同样由回放产批（与臂①同一
    路径）。``write_batch`` 形参保留为执行缝协议位（真实执行件透传组装层）。
    """
    ledger_versions = _ledger_versions(ctx.root, ctx.selected_split_indices)
    folds_to_train = [
        split
        for split in ctx.selected_split_indices
        if ledger_versions.get(split) is None and split not in reused
    ]
    if not folds_to_train:
        return None
    logger.info(f"待训练折: {folds_to_train}（训练不写批，runs 批由回放统一产出）")
    return executor.train_fn(ctx, tuple(folds_to_train), False)


def check_score_receipts(
    ctx: ArmRunContext, baselines: Mapping[str, object], score_store: object
) -> Dict[str, object]:
    """臂③打分侧消费核对（R2-T5A-02 后半）：打分实际读取 ⊆ 消费集合。

    核对对象 = ``_execute_arm_into`` 内建于生产根只读的打分 ``PanelDataStore``；
    receipt 采集经 ``collect_score_read_receipt_files``（会话级 ``read_receipts``
    累积，覆盖逐折多次读取）。越界 ⇒ **不抛断流程**，返回 ``pass=False`` 报告
    （越界清单/实际读取数入证据），verdict 判定归 runner（drift 同通道语义）；
    非 panel 臂（无打分 store）⇒ 登记 skipped 原因放行。
    """
    if ctx.arm != ARM_V2_PANEL or score_store is None:
        return {
            "skipped": True,
            "reason": f"臂 {ctx.arm} 非 panel 打分源，打分侧消费核对不适用（仅臂③）",
        }
    try:
        return assert_score_receipt_within_consumption(
            score_store,
            Path(str(baselines["consumption_snapshot_path"])),
            base_dir=Path(str(baselines["base_dir"])),
            production_data_root=ctx.production_data_root,
        )
    except RuntimeError as exc:
        logger.error(f"打分侧消费核对越界（计入 drift 语义，verdict FAIL）: {exc}")
        return {
            "pass": False,
            "error": str(exc),
            "receipt_files": len(collect_score_read_receipt_files(score_store)),
        }
