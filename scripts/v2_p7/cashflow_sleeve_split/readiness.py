# -*- coding: utf-8 -*-
"""审点一就绪清点：数据、行域、配置取证、金额与阻断项（协议 §8 审点一）。

本模块只做**清点与登记**，不解决数据缺口：
- 数据面：分区数/首末/字节/身份口径 + 缺口；
- 行域：``label_filtered``（≤2026-07-02）与 ``structural``（≥07-03）实测
  （标签非空率推断）+ 协议登记的历史事实；
- 配置取证：``summary.parquet``（WF 参数）vs 逐模型 ``metadata``（实际训练参数）
  vs ``value`` 袖 spec（生产配置代码化）三方对账，冲突逐条登记并给出裁决依据；
- 金额：六槽（Top6，两批各 3）在大/小腿上的单槽名义金额与 V 下限核对；
- 阻断项：必须由 run-a 关闭的事项，逐条写 ``unblock`` 与证据路径。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.v2_p7.cashflow_sleeve_split import artifacts as artifacts_mod
from scripts.v2_p7.cashflow_sleeve_split import feature_routes, fold_clock, identity, probes, spec
from scripts.v2_p7.common import REPO_ROOT

#: 生产 value 袖 spec 的代码化配置（对照项；只读引用，不修改）
VALUE_SPEC_RELATIVE = "src/lazybull/v2/sleeves/value/spec.py"
SUMMARY_RELATIVE = f"{fold_clock.ANCHOR_RUN_RELATIVE}/summary.parquet"

#: 训练超参对照项：metadata 键 → (spec 键, 期望是否一致)
HYPERPARAM_PAIRS: Sequence[tuple[str, str]] = (
    ("n_estimators", "n_estimators"),
    ("max_depth", "max_depth"),
    ("learning_rate", "learning_rate"),
    ("subsample", "subsample"),
    ("colsample_bytree", "colsample_bytree"),
    ("min_child_weight", "min_child_weight"),
    ("reg_alpha", "reg_alpha"),
    ("reg_lambda", "reg_lambda"),
    ("gamma", "gamma"),
    ("early_stopping_rounds", "early_stopping_rounds"),
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def value_spec_snapshot(root: Path) -> Dict[str, Any]:
    """读取 value 袖 spec 的代码化配置（运行期 import，不复制数值）。"""
    from src.lazybull.universe.domains import resolve_stock_domain
    from src.lazybull.v2.sleeves._registry import get_sleeve_spec

    sleeve = get_sleeve_spec("value")
    domain = resolve_stock_domain(sleeve.domain)
    return {
        "source": VALUE_SPEC_RELATIVE,
        "n_estimators": int(sleeve.hyperparams["n_estimators"]),
        "max_depth": int(sleeve.hyperparams["max_depth"]),
        "learning_rate": float(sleeve.hyperparams["learning_rate"]),
        "subsample": float(sleeve.hyperparams["subsample"]),
        "colsample_bytree": float(sleeve.hyperparams["colsample_bytree"]),
        "min_child_weight": float(sleeve.hyperparams["min_child_weight"]),
        "reg_alpha": float(sleeve.hyperparams["reg_alpha"]),
        "reg_lambda": float(sleeve.hyperparams["reg_lambda"]),
        "gamma": float(sleeve.hyperparams["gamma"]),
        "early_stopping_rounds": int(sleeve.hyperparams["early_stopping_rounds"]),
        "ensemble_seeds": list(sleeve.ensemble_seeds),
        "rank_weight_enabled": bool(sleeve.hyperparams["rank_weight_enabled"]),
        "rank_weight_topk": int(sleeve.hyperparams["rank_weight_topk"]),
        "rank_weight": float(sleeve.hyperparams["rank_weight"]),
        "rank_weight_topk_weight_mode": str(sleeve.hyperparams["rank_weight_topk_weight_mode"]),
        "top_n": int(sleeve.top_n),
        "holding_period_days": int(sleeve.holding_period_days),
        "min_amount_ma20": float(sleeve.min_amount_ma20),
        "exclude_financial": bool(sleeve.exclude_financial),
        "domain": str(sleeve.domain),
        "domain_config": dict(domain),
        "val_ratio": float(sleeve.split_spec["val_ratio"]),
        "label_name": str(sleeve.label_name),
    }


def summary_snapshot(root: Path) -> Dict[str, Any]:
    """锚批次 summary 的 WF 参数（14 折应逐值相同；只取公共标量）。"""
    import pandas as pd

    path = Path(root) / SUMMARY_RELATIVE
    if not path.exists():
        return {"status": "missing"}
    frame = pd.read_parquet(path)
    keys = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "early_stopping_rounds",
        "ensemble_seeds",
        "val_ratio",
        "rank_weight_enabled",
        "rank_weight_topk",
        "rank_weight",
        "bt_top_n",
        "bt_rebalance_freq",
        "position_sizing",
        "kelly_vol_window",
        "bt_max_weight_per_stock",
        "stagger_tranches",
        "bt_exclude_st",
        "bt_min_list_days",
        "stock_domain",
        "enable_cashflow_quality_features",
    )
    out: Dict[str, Any] = {"status": "ok", "folds": int(len(frame)), "values": {}}
    for key in keys:
        if key not in frame.columns:
            out["values"][key] = None
            continue
        unique = sorted({str(value) for value in frame[key].tolist()})
        out["values"][key] = unique[0] if len(unique) == 1 else unique
    return out


def metadata_snapshot(root: Path, version: int) -> Dict[str, Any]:
    """逐模型元数据的关键训练参数与身份。"""
    path = Path(root) / spec.MODEL_ROOT_RELATIVE / f"v{version}_metadata.json"
    if not path.exists():
        return {"version": version, "status": "missing"}
    payload = _read_json(path)
    params = payload.get("train_params") or {}
    return {
        "version": int(version),
        "status": "ok",
        "created_at": payload.get("created_at"),
        "train_start": payload.get("train_start_date"),
        "train_end": payload.get("train_end_date"),
        "n_samples": payload.get("n_samples"),
        "feature_count": payload.get("feature_count"),
        "n_estimators": params.get("n_estimators"),
        "best_iteration": params.get("best_iteration"),
        "best_iteration_floor_triggered": params.get("best_iteration_floor_triggered"),
        "ensemble_seeds": params.get("ensemble_seeds"),
        "random_state": params.get("random_state"),
        "max_depth": params.get("max_depth"),
        "learning_rate": params.get("learning_rate"),
        "subsample": params.get("subsample"),
        "colsample_bytree": params.get("colsample_bytree"),
        "min_child_weight": params.get("min_child_weight"),
        "reg_alpha": params.get("reg_alpha"),
        "reg_lambda": params.get("reg_lambda"),
        "gamma": params.get("gamma"),
        "early_stopping_rounds": params.get("early_stopping_rounds"),
        "early_stopping_metric": params.get("early_stopping_metric"),
        "label_transform": params.get("label_transform"),
        "freshness_strategy": params.get("freshness_strategy"),
        "cashflow_quality_schema_version": params.get("cashflow_quality_schema_version"),
        "cashflow_quality_feature_columns_count": len(
            params.get("cashflow_quality_feature_columns") or []
        ),
        "device": params.get("device"),
    }


def model_config_evidence(root: Path) -> Dict[str, Any]:
    """三方配置对账 + 1500/500 冲突裁决（协议 §5.2-3）。"""
    values = value_spec_snapshot(root)
    summary = summary_snapshot(root)
    versions = list(spec.FOLD_VERSIONS) + [spec.DEPLOY_VERSION]
    metadata = {str(version): metadata_snapshot(root, version) for version in versions}

    conflicts: List[Dict[str, Any]] = []
    agreements: List[str] = []
    for meta_key, spec_key in HYPERPARAM_PAIRS:
        spec_value = values.get(spec_key)
        key_conflicts = 0
        for version, item in metadata.items():
            meta_value = item.get(meta_key)
            if meta_value is None or spec_value is None:
                continue
            if _num(meta_value) != _num(spec_value):
                key_conflicts += 1
                conflicts.append(
                    {
                        "item": meta_key,
                        "version": version,
                        "spec_value": spec_value,
                        "metadata_value": meta_value,
                    }
                )
        if key_conflicts == 0 and spec_value is not None:
            agreements.append(meta_key)

    n_estimators_resolution = _resolve_n_estimators(metadata, values, summary)
    seeds = {str(item.get("ensemble_seeds")) for item in metadata.values()}
    best_iterations = {version: item.get("best_iteration") for version, item in metadata.items()}
    floors = [
        version for version, item in metadata.items() if item.get("best_iteration_floor_triggered")
    ]
    return {
        "sources": {
            "value_spec": VALUE_SPEC_RELATIVE,
            "wf_summary": SUMMARY_RELATIVE,
            "model_metadata": f"{spec.MODEL_ROOT_RELATIVE}/v{{version}}_metadata.json",
        },
        "value_spec": values,
        "wf_summary": summary,
        "metadata": metadata,
        "agreements": agreements,
        "conflicts": conflicts,
        "n_estimators_resolution": n_estimators_resolution,
        "ensemble_seeds_consistent": len(seeds) == 1,
        "ensemble_seeds": sorted(seeds),
        "best_iterations": best_iterations,
        "best_iteration_floor_triggered": floors,
        "per_arm_requirement": {
            "best_iteration": "每臂在自己的早停段决定轮次，禁止复制旧 best_iteration（协议 §5.2-3）",
            "seeds": "42/61/82 三种子集成、三者全保留",
            "cashflow_live_metadata": (
                "注册器必须按本臂最终列生成 live-column 元数据；旧 9 列口径不得复制给 R，"
                "也不等于 B 的 11 列业务组（协议 §5.2-6）"
            ),
        },
    }


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_n_estimators(
    metadata: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """1500 / 500 冲突的裁决与证据（不按结果挑，只按证据挑）。"""
    declared = sorted(
        {int(item["n_estimators"]) for item in metadata.values() if item.get("n_estimators")}
    )
    best = [
        int(item["best_iteration"])
        for item in metadata.values()
        if isinstance(item.get("best_iteration"), (int, float))
    ]
    max_best = max(best) if best else None
    summary_value = (summary.get("values") or {}).get("n_estimators")
    spec_value = values.get("n_estimators")
    resolution = {
        "values": {
            "metadata_n_estimators": declared,
            "value_spec_n_estimators": spec_value,
            "wf_summary_n_estimators": summary_value,
            "max_best_iteration": max_best,
        },
        "chosen": designated(declared, max_best),
        "rule": "部署/折元数据优先于「跳过训练、复用旧模型」的 500 棵复跑示例（协议 §5.2-3）",
        "evidence": (
            f"best_iteration 最大 {max_best} > 500 ⇒ 500 棵上限不可能产生该模型；"
            f"15 份 metadata 全部声明 n_estimators={declared}"
        ),
        "not_reproducible_if_500": bool(max_best is not None and max_best > 500),
    }
    return resolution


def designated(declared: Sequence[int], max_best: int | None) -> int | None:
    """取元数据声明的棵数（若与 best_iteration 矛盾则返回 None 并要求登记）。"""
    if not declared:
        return None
    if max_best is not None and max_best > max(declared):
        return None
    return int(declared[0])


def execution_configs() -> List[Dict[str, Any]]:
    """逐账户角色的执行配置（复审第 4 项：主腿 Top20/15%，六槽只属小腿）。

    ``per_stock_new_buy_cap_yuan`` 是**新买**单股上限（占该腿当时权益），
    与小腿的``per_slot_nominal_yuan``（等权槽位名义金额）不是同一概念。
    """
    value = spec.min_ticket_value()
    lot = int(spec.machine_spec_document()["lot_size"])
    rows: List[Dict[str, Any]] = []
    for role, config in spec.ACCOUNT_ROLES.items():
        signal = str(config["signal"])
        execution = spec.SIGNAL_EXECUTION[signal]
        for capital in spec.ACCOUNTS["capitals"]:
            capital = int(capital)
            leg_capital = capital * float(config["capital_share"])
            slots = execution["top_n"]
            per_slot = leg_capital / slots
            cap = leg_capital * float(execution["max_weight_per_stock"])
            rows.append(
                {
                    "role": role,
                    "leg": config["leg"],
                    "capital": capital,
                    "leg_capital": leg_capital,
                    "signal": signal,
                    "top_n": slots,
                    "position_sizing": execution["position_sizing"],
                    "kelly_vol_window": execution["kelly_vol_window"],
                    "max_weight_per_stock": execution["max_weight_per_stock"],
                    "per_stock_new_buy_cap_yuan": cap,
                    "stagger_tranches": execution["stagger_tranches"],
                    "slots_per_tranche": execution["slots_per_tranche"],
                    "per_slot_nominal_yuan": per_slot if signal == "small" else None,
                    "min_ticket_value_yuan": value,
                    "slot_above_v": bool(per_slot >= value) if signal == "small" else None,
                    "min_price_for_one_lot_at_v": value / lot,
                    "note": (
                        "小腿：两批各 3 槽、批内等权、整手后再查 ≥V"
                        if signal == "small"
                        else "主腿：half-Kelly 排序 + 单股新买 ≤15% 权益；整手与现金逐笔复核"
                    ),
                }
            )
    return rows


def slot_amounts() -> Dict[str, Any]:
    """金额核对：主腿新买上限 vs 小腿槽位名义金额（复审第 4 项）。"""
    value = spec.min_ticket_value()
    rows = execution_configs()
    small = [row for row in rows if row["signal"] == "small"]
    big = [row for row in rows if row["signal"] == "big"]
    return {
        "min_ticket_value_yuan": value,
        "formula": "V = max(25000, min_commission / commission_rate)",
        "references": {
            "protocol_min_ticket": "协议 §6：V=max(25000,5/0.0001954)",
            "big_leg_min_buy_value_ratio": spec.BIG_LEG["min_buy_value_ratio"],
        },
        "rows": rows,
        "small_leg_rows": small,
        "big_leg_rows": big,
        "all_above_v": all(bool(row["slot_above_v"]) for row in small),
        "big_leg_uses_top20": all(int(row["top_n"]) == 20 for row in big),
        "small_leg_uses_six_slots": all(int(row["top_n"]) == 6 for row in small),
        "big_leg_cap_by_capital": {
            str(row["capital"]): row["per_stock_new_buy_cap_yuan"] for row in big
        },
        "small_leg_slot_by_capital": {
            str(row["capital"]): row["per_slot_nominal_yuan"] for row in small
        },
    }


def recompute_rules() -> Dict[str, Any]:
    """独立复算口径（协议 §9；容差只用于发现错误，不得放宽门槛）。"""
    return {
        "equity_identity_abs_tolerance_yuan": 1e-6,
        "metric_abs_tolerance_ratio": 1e-10,
        "metric_function": spec.CHAIN_METRICS["implementation"],
        "mdd_loss": spec.CHAIN_METRICS["mdd_loss_convention"],
        "first_divergence": "任何失败先给首个分歧样本及其父工件",
        "threshold_relaxation": "禁止：容差只用于发现复算错误",
        "read_paths": "verify 必须从逐日账户/逐笔成交与上游哈希复算，不取 summary 作为输入",
    }


def _inventory(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """输入清单 → 精简清点表（只留身份与规模字段）。"""
    fields = (
        "relative",
        "kind",
        "status",
        "partition_units",
        "files",
        "first",
        "last",
        "total_bytes",
        "digest_kind",
        "digest",
        "full_sha_status",
    )
    return {item["name"]: {key: item.get(key) for key in fields} for item in manifest["sources"]}


def _preflight(
    baseline: Mapping[str, Any],
    routes: Mapping[str, Any],
    clock: Mapping[str, Any],
) -> Dict[str, Any]:
    """协议冻结项预检摘要（旧 A 锚 + 列路由 + 折时钟）。"""
    return {
        "a_baseline_status": baseline["status"],
        "a_baseline_manifest_sha256": baseline["manifest_sha256"],
        "a_baseline_frozen_files": baseline.get("frozen_files"),
        "a_baseline_reference_status": baseline["reference_delta"],
        "a_baseline_account_checks": {
            capital: {
                "rows": item["rows"],
                "first": item["first"],
                "last": item["last"],
                "a80_cash_identity_max_abs_residual": item["a80_cash_identity_max_abs_residual"],
                "recomputed": item["recomputed"],
            }
            for capital, item in baseline["account_checks"].items()
        },
        "folds": len(routes["folds"]),
        "deploy_version": routes["deploy_version"],
        "all_b_equal_group": routes["summary"]["all_b_equal_group"],
        "b_count": routes["summary"]["b_count"],
        "f_counts": routes["summary"]["f_counts"],
        "fold_clock_checks": clock["checks"],
    }


def _probe_records(probe_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """探针逐日摘要（字段白名单，避免把大对象写进就绪件）。"""
    keys = (
        "date",
        "segment",
        "feature_source",
        "status",
        "mother_n",
        "mother_tradable_n",
        "feature_rows_n",
        "feature_columns_n",
        "feature_to_mother_ratio",
        "old_domain_observed_n",
        "structural_domain_observed_n",
        "scored_n",
        "scored_n_status",
        "industry_pit_alignment",
        "universe_n",
        "eligible_n",
        "structural_gap_n",
        "industry_bridge",
        "b_freshness_only_n",
        "b_absolute_amount_only",
        "financial_literal_n",
        "source_known_n",
        "source_known_f_n",
        "source_known_f_r_n",
        "selection_status",
        "group_columns_missing",
        "sentinel",
        "row_domain",
        "width_floor_500",
        "b_source_floor_500",
        "elapsed_seconds",
    )
    return [{key: item.get(key) for key in keys} for item in probe_result.get("records") or []]


def _findings(
    baseline: Mapping[str, Any],
    probe_result: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    clock: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    cf = dict(evidence or {})
    """实测发现（不影响 CANDIDATE 判定，但必须让审核者看到）。"""
    records = list(probe_result.get("records") or [])
    literal = sum(int(record.get("financial_literal_n") or 0) for record in records)
    gaps = {record["date"]: record.get("structural_gap_n") for record in records}
    return [
        {
            "id": "F-01-MLSIGNAL-FINANCIAL-SUFFIX",
            "severity": "pre_existing_production_defect",
            "statement": (
                "MLSignal 的金融剔除常量是裸码 {801780,801790}，而特征 sw_l1_code 实测带 "
                "'.SI' 后缀 ⇒ 生产规则命中 0，银行/非银未被剔除"
            ),
            "measured": {"literal_hits_on_probe_days": literal, "intended_hits_sample": None},
            "handling": "探针按意图去后缀执行并同时记录两口径；不改生产代码（须另案）",
        },
        {
            "id": "F-02-STRUCTURAL-PATH-CALLS-LABELS",
            "severity": "cost_and_dependency",
            "statement": (
                "_run_capture(structural=True) 仍会调用标签构建函数（结果归 labels 列组），"
                "因此「删除标签」不是受支持状态；标签无关性以三遍构建证明：原值 vs 值改写"
                "（缺失位置不变）vs 缺失位置重排（有→无/无→有），股票键、特征列逐值与 "
                "F/R/B 有限值来源掩码均须逐值一致（r0_003 复核第 1 项）"
            ),
            "measured": (
                {
                    "label_removal_supported": False,
                    "counterfactual": cf.get("status"),
                    "identical_value_only": cf.get("identical_value_only"),
                    "identical_position": cf.get("identical_position"),
                    "identical": cf.get("identical"),
                    "known_label_targets": cf.get("known_label_targets"),
                    "source_mask_identical": (cf.get("source_mask") or {}).get("identical"),
                    "day": cf.get("day"),
                    "rows": cf.get("rows"),
                    "columns_n": cf.get("columns_n"),
                    "source": "measurements/structural_counterfactual_hist_2019.json",
                }
                if cf
                else {
                    "label_removal_supported": False,
                    "counterfactual": "unknown",
                    "counterfactual_reason": "本 run 未内嵌标签反事实的实测件",
                }
            ),
            "handling": "run-a 结构重建按含标签计算估时；反事实走改写与缺失位置重排",
        },
        {
            "id": "F-03-STRUCTURAL-GAP",
            "severity": "info",
            "statement": (
                "通过全部非特征资格但当日无特征行的股票数（label_filtered 行域漏行），"
                "逐日登记；这是 run-a 结构重建的直接动因"
            ),
            "measured": gaps,
            "handling": "run-a 按结构通路重建并逐日记录母表/合格/来源宽度",
        },
        {
            "id": "F-04-A-BASELINE-FROZEN-FILES",
            "severity": "info",
            "statement": "旧 A 锚按冻结清单逐件验 SHA（缺件或改一字节即失败）",
            "measured": baseline.get("frozen_files"),
            "handling": "每次 prepare 重跑同一校验",
        },
        {
            "id": "F-05-INDUSTRY-PIT-INJECTION",
            "severity": "corrected_in_r0_004",
            "statement": (
                "r0_003 及以前的结构构建走生产默认行业快照（当前申万分类、无日期维度），"
                "历史日的特征分组与 neu 标签分组被未来行业改写（600783.SH 在 2019-04-01 "
                "属非银金融 490000/801790.SI，快照写机械设备 801890.SI）。r0_004 起研究"
                "构建入口从 sw_industry_pit 区间命中构造当日帧并显式注入，资格、中性化"
                "与标签共用同一输入"
            ),
            "measured": (
                {
                    "identity_probe": (cf.get("industry_injection") or {})
                    .get("stats", {})
                    .get("identity_probe"),
                    "injection_stats": (cf.get("industry_injection") or {}).get("stats"),
                    "probe_alignment": (cf.get("probe_consumption") or {}).get(
                        "industry_pit_alignment"
                    ),
                }
                if cf
                else {"status": "unknown"}
            ),
            "handling": "单日验证（20190401）已注入并逐只对齐；run-a 全窗分块逐日注入当日快照",
        },
        {
            "id": "F-06-DEPLOY-ES-PURIFICATION",
            "severity": "corrected_in_r0_004",
            "statement": (
                "r0_003 折表的部署行只给原始段，未应用早停净化。部署 v24022 早停段截至 "
                "2025-08-28、校准始于 08-29 ⇒ 早停最后可用信号日 = 2025-07-30（T+21="
                "08-28），剔除其后 21 个早停信号日。部署训练入口须同时收到原始段与净化段"
                "两套边界"
            ),
            "measured": {
                "es_last_usable_sample": (clock or {})
                .get("deploy", {})
                .get("es_last_usable_sample"),
                "es_purged_samples": (clock or {}).get("deploy", {}).get("es_purged_samples"),
                "source": "fold_clock.csv 部署行 + checks.deploy_training_history",
            },
            "handling": "run-a 部署训练按净化边界取键并逐键核对（原始边界同封存）",
        },
        {
            "id": "F-07-B-SOURCE-EVIDENCE-REFINED",
            "severity": "corrected_in_r0_004",
            "statement": (
                "r0_003 的来源掩码用 .notna()（单个无穷值也计入）且把 ocf/fcf 绝对额计入 "
                "B 来源。r0_004 起掩码用有限值（np.isfinite），B 来源只认经济比率与销售"
                "现金代理；只有绝对额、没有可用比率的行单独计数（B_absolute_amount_only）"
            ),
            "measured": {
                "policy": spec.SOURCE_EVIDENCE["B"].get("note"),
                "absolute_columns": spec.SOURCE_EVIDENCE["B"].get("absolute_amount_columns"),
            },
            "handling": "探针与夹具同口径；无需训练 B（r0_003 复核）",
        },
        {
            "id": "F-08-BLOCK-KEYS-RECOUNTED",
            "severity": "corrected_in_r0_004",
            "statement": (
                "r0_003 附件的逐日键摘要字段口径错误（477+471=948 > 全块 748）。r0_004 用"
                "已保存的 57 日文件逐日直读重算：only_structural 逐日相加必须等于全块总数，"
                "预热天数按交易日历重算（145 预热 + 57 输出 = 202，原写 148）"
            ),
            "measured": (cf.get("key_control") or {}).get("totals"),
            "handling": "预算报价按修正后的每计算日/每保留日秒数重列（见 pilot_costs.json）",
        },
    ]


def build_readiness(
    root: Path | None = None,
    extra_roots: Sequence[str] = (),
    routes: Mapping[str, Any] | None = None,
    clock: Mapping[str, Any] | None = None,
    probe_result: Mapping[str, Any] | None = None,
    counterfactual_evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """组装 ``readiness.json``（清点 + 依据；不写盘）。"""
    base = Path(root) if root is not None else REPO_ROOT
    started_hashes = identity.measure_hash_throughput(
        [base / spec.PROTOCOL_RELATIVE, base / "data/clean/trade_cal.parquet"]
    )
    baseline = identity.verify_a_baseline(base)
    manifest = identity.build_input_manifest(
        base,
        probe_dates=[item["date"] for item in spec.PROBE_DATES],
        a_baseline={
            "verified": True,
            "manifest_sha256": baseline["manifest_sha256"],
            "consumed_files": baseline["consumed_files"],
        },
    )
    routes = routes or feature_routes.build_routes(base)
    clock = clock or fold_clock.build_fold_clock(base)
    probe_result = probe_result or probes.run_probes(base)

    inventory = _inventory(manifest)
    evidence = model_config_evidence(base)
    slots = slot_amounts()
    blockers = _blockers(manifest, probe_result, evidence, clock, base)

    return {
        "schema": "p7_cashflow_split_readiness/v1",
        "candidate_id": spec.CANDIDATE_ID,
        "generated_at": artifacts_mod.utc_now(),
        "protocol_sha256": spec.protocol_sha256(base),
        "spec_sha256": spec.spec_sha256(),
        "code_identity": identity.git_identity(base),
        "write_scope": {
            "roots": [str(path) for path in artifacts_mod.write_roots(extra_roots)],
            "production_root_probe": artifacts_mod.require_write_scope_probe(),
        },
        "protocol_frozen_preflight": _preflight(baseline, routes, clock),
        "data_inventory": inventory,
        "input_gaps": manifest["gaps"],
        "row_domain": {
            "declared": dict(spec.ROW_DOMAIN),
            "declared_source": "v2/store/row_domain.py::row_domain_for_date（边界单一来源）",
            "measured": {item["date"]: item.get("row_domain") for item in probe_result["records"]},
            "structural_partition_count": inventory.get("features_cs_infer", {}).get(
                "partition_units"
            ),
            "note": (
                "行域是日期的派生量而非分区标记；标签非空率不作判别依据。"
                "cs_infer 仅覆盖尾段，不足以追溯整窗（协议 §5.1）；"
                "历史推理端必须按 run-a 结构缓存重建"
            ),
        },
        "model_config_evidence": evidence,
        "feature_routes_summary": routes["summary"],
        "fold_clock_summary": clock["checks"],
        "probes": probe_result["summary"],
        "probe_records": _probe_records(probe_result),
        "recovery_plan": _recovery_plan(),
        "slot_amounts": slots,
        "recompute_rules": recompute_rules(),
        "costs": {
            "hash_throughput": started_hashes,
            "probe": probes.probe_costs(probe_result),
        },
        "hygiene": hygiene_report(base),
        "findings": _findings(baseline, probe_result, counterfactual_evidence, clock),
        "blockers": blockers,
        "not_run": [
            "真实收益回放（run-a / run-b / verify 未授权）",
            "整窗训练（F/R/B 三臂 14 折 + 部署）",
            "全窗结构特征缓存重建（单日 20190401 验证已做，全窗待 run-a 分块）",
            "内核评分与有效宽度（kernel_scored / effective_scored）",
            "全窗标签反事实（单日 20190401 已做：值改写 + 缺失位置重排三遍一致）",
            "同日财报修订、公告日回退与跨折未成熟标签删除数量",
            "公司行为入账与新持仓事件覆盖核对",
            "账户级规格：整手后跌破 V 被拒、最低佣金、停牌/涨跌停、锁股退出重试",
            "压力滑点重放与 20 日块自举",
        ],
        "verdict": {
            "checkpoint": 1,
            "status": (
                "ready_for_review"
                if not [item for item in blockers if item["severity"] == "blocking_for_review"]
                else "blocked"
            ),
            "review_owner": "用户转交 Codex（编码者不得自行代签，协议 §2.4）",
            "authorized_stages": list(spec.AUTHORIZED_STAGES),
        },
    }


#: 卫生检查命令（prepare 内实测，结果写入 readiness.json.hygiene）
HYGIENE_COMMANDS: Dict[str, Sequence[str]] = {
    "flake8_package": (
        "-m",
        "flake8",
        "--max-line-length=100",
        "scripts/v2_p7/cashflow_sleeve_split",
        "tests/test_p7_cashflow_sleeve_split.py",
        "tests/test_p7_cashflow_sleeve_split_run_a.py",
    ),
    "complexity_package": (
        "scripts/check/check_complexity.py",
        "--paths",
        "scripts/v2_p7/cashflow_sleeve_split",
    ),
    "complexity_scope_src": (
        "scripts/check/check_complexity.py",
        "--paths",
        "src/lazybull",
        "--baseline",
        "scripts/check/complexity_baseline.json",
    ),
    "complexity_scope_all": (
        "scripts/check/check_complexity.py",
        "--paths",
        "src/lazybull",
        "scripts",
        "--baseline",
        "scripts/check/complexity_baseline.json",
    ),
}


def parse_complexity_output(text: str) -> Dict[str, Any]:
    """解析 hook 输出：只有无「存量基线」「已豁免」标记的 ``[hard]`` 才是新增。

    逐行标记不可靠（标记用全角括号），因此同时保留官方汇总数
    （``新增硬超限 N 条``）作权威值，逐行结果只作样本。
    """
    import re

    hard_lines = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("[hard]") and "存量基线" not in line and "已豁免" not in line
    ]
    match = re.search(r"新增硬超限 (\d+) 条", text)
    return {
        "new_hard_overruns": int(match.group(1)) if match else None,
        "new_hard_samples": hard_lines[:5],
        "hard_total": len([line for line in text.splitlines() if line.startswith("[hard]")]),
    }


def hygiene_report(root: Path) -> Dict[str, Any]:
    """实测 lint 与复杂度双口径（只登记，不因仓库既有超限阻断研究批次）。"""
    import os
    import subprocess
    import sys

    out: Dict[str, Any] = {"commands": {}, "results": {}}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for name, argv in HYGIENE_COMMANDS.items():
        command = [sys.executable, *argv]
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        parsed = parse_complexity_output(completed.stdout or "")
        out["commands"][name] = " ".join(argv)
        out["results"][name] = {"exit_code": completed.returncode, **parsed}
    out["policy"] = {
        "flake8": "flake8 --max-line-length=100",
        "complexity": "复杂度双口径须 exit=0、新增硬超限 0；棘轮基线只缩不扩",
        "package_scope": "scripts/v2_p7/cashflow_sleeve_split 属阶段专用研究工具（CLAUDE §7.5/§7.8）",
        "pre_existing_note": (
            "仓库既有硬超限不计入本包责任；本包只在 workspace 内新增文件，"
            "src/lazybull 零改动（代码身份见 code_identity）"
        ),
    }
    return out


def _recovery_plan() -> Dict[str, Any]:
    """推理端特征矩阵的恢复方案（协议 §5.1「恢复路径」；只描述，不执行）。"""
    return {
        "scope": "2026-10-29 之前需要评分的全部日（含账户窗外预热与补买查询日）",
        "path": (
            "复用 V2PanelBuilder._run_capture(..., structural=True)，"
            "或等价 FeatureBuilder(require_label=False) 生产计算路径"
        ),
        "inputs": {
            "cleaning_sources": "共同 clean 源与水位（clean/daily、daily_basic、stk_limit、suspend、stock_st）",
            "industry": "显式注入当日历史行业映射（data/clean/sw_industry_pit.parquet）",
            "warmup": {
                "feature_trading_days": spec.WINDOW["feature_warmup_trading_days"],
                "price_records": spec.WINDOW["price_warmup_records"],
            },
        },
        "layout": "campaign/cache/<input_digest>/ 按月分块；产物只写研究根",
        "validation": [
            "近期真实结构分区核对键与同定义列值（逐列、容差 1e-6）",
            "早期 / 接缝 / 已知拒绝日（20261019）抽查",
            "删除或改写未来标签后，评分行键与特征必须逐值不变",
            "逐日记录 mother_n / eligible_n / source_known_n / kernel_scored_n / effective_scored_n",
        ],
        "prohibited": [
            "给旧分区改名冒充 structural",
            "向母表左连后默认分打满",
            "只数旧排名证明 500 只",
            "静默修改主窗或缩短窗口",
        ],
        "status": "not_run（需 run-a 授权）",
    }


def _blockers(
    manifest: Mapping[str, Any],
    probe_result: Mapping[str, Any],
    evidence: Mapping[str, Any],
    clock: Mapping[str, Any],
    root: Path,
) -> List[Dict[str, Any]]:
    """阻断与登记项（``severity`` 三档：blocking_for_run_a / registered / not_run）。"""
    items: List[Dict[str, Any]] = []

    missing = list(manifest.get("gaps") or [])
    if missing:
        items.append(
            {
                "id": "B-INPUT-GAP",
                "severity": "blocking_for_run_a",
                "statement": f"输入清单存在缺口 {len(missing)} 项",
                "detail": missing,
                "unblock": "run-a 前补齐或登记替代来源",
                "evidence": "input_manifest.json.gaps",
            }
        )

    cs_infer = next(
        (item for item in manifest["sources"] if item["name"] == "features_cs_infer"), {}
    )
    items.append(
        {
            "id": "B-INFER-PANEL",
            "severity": "blocking_for_run_a",
            "statement": (
                "模型矩阵式推理缓存未覆盖整窗：cs_infer 只有 "
                f"{cs_infer.get('partitions')} 个日分区"
                f"（{cs_infer.get('first')}~{cs_infer.get('last')}）。"
                "v2 panel 存档（月分区 2012-01~2025-06 + 日分区 2025-07 起）不是模型矩阵："
                "列组不同、缺选择列与运行时派生列，且 cs_train 是 label_filtered 行域，"
                "不得直接当评分域"
            ),
            "detail": {
                "panel_hot": "data/features/panel（8 个列组，日分区）",
                "panel_archive": "data/features/panel_archive（8 个列组，月分区）",
                "requirement": (
                    "协议 §5.1 恢复路径：V2PanelBuilder._run_capture(structural=True) 或等价 "
                    "FeatureBuilder(require_label=False) 生产计算路径，显式注入当日历史行业映射"
                ),
                "validation_plan": [
                    "先以近期真实结构分区核对键与同定义列值",
                    "再抽早期、接缝与已知拒绝日（20261019）验证",
                    "删除/改写未来标签的反事实测试：评分行键与特征必须不变",
                ],
            },
            "unblock": "run-a 批：按月分块重建（缓存写 campaign/cache/<input_digest>/）",
            "evidence": "readiness.json.data_inventory + recovery_plan",
        }
    )

    items.append(
        {
            "id": "B-KERNEL-WIDTH",
            "severity": "blocking_for_run_a",
            "statement": "内核评分宽度（kernel_scored / effective_scored）在本阶段 not_estimable",
            "detail": probe_result.get("summary", {}).get("kernel_scored_status"),
            "unblock": "run-a 结构缓存 + 日期绑定模型后逐日统计",
            "evidence": "readiness.json.probes",
        }
    )

    items.append(
        {
            "id": "B-VAL-CLOCK",
            "severity": "registered",
            "statement": "模型元数据未记录 val/es 日期，本表按冻结规则重算并标 derived",
            "detail": clock.get("checks", {}).get("val_dates_note"),
            "unblock": "run-a 逐折训练时以实际切分日期回填并复核样本键",
            "evidence": "fold_clock.csv.val_date_status",
        }
    )

    resolution = evidence.get("n_estimators_resolution") or {}
    items.append(
        {
            "id": "B-1500-500",
            "severity": "registered",
            "statement": "树数存在 1500（metadata）与 500（value spec / 复跑示例）冲突，已按证据裁决",
            "detail": resolution,
            "unblock": "run-a 逐臂按元数据 1500 与各自早停段重训",
            "evidence": "readiness.json.model_config_evidence.n_estimators_resolution",
        }
    )

    reference = next((item for item in probe_result.get("records") or []), {})
    if reference.get("sentinel", {}).get("matches_expected") is False:
        items.append(
            {
                "id": "B-CASHFLOW-SENTINEL",
                "severity": "registered",
                "statement": "探针日现金流哨兵列取值不等于 3 或缺失，需按源校验定位",
                "detail": reference.get("sentinel"),
                "unblock": "run-a 前按 cashflow_quality 版本化 PIT 复核",
                "evidence": "readiness.json.probe_records",
            }
        )

    items.append(
        {
            "id": "B-CASHFLOW-RAW",
            "severity": "registered",
            "statement": (
                "现金流的 raw 目录为 data/raw/cashflow（季度分区）+ 两个 revision 目录；"
                "无 data/raw/cashflow_vip 目录，与 ensure 侧命名不同源"
            ),
            "detail": {
                "quarterly_partitions": "data/raw/cashflow",
                "revision": "data/raw/cashflow_revision_*",
            },
            "unblock": "run-a 数据装载时按版本化 PIT 契约读取并登记实际分区",
            "evidence": "input_manifest.json.sources",
        }
    )

    items.append(
        {
            "id": "B-CORPORATE-ACTIONS",
            "severity": "blocking_for_run_a",
            "statement": "新持仓的公司行为覆盖、重复组件与期末权益尚未核对（本阶段不运行账户）",
            "detail": "协议 §6「净成本与账务」要求逐日对账并复核事件覆盖",
            "unblock": "run-a 账户阶段 + run-b 逐日对账",
            "evidence": "readiness.json.not_run",
        }
    )
    return items


__all__ = [
    "build_readiness",
    "hygiene_report",
    "metadata_snapshot",
    "model_config_evidence",
    "parse_complexity_output",
    "recompute_rules",
    "slot_amounts",
    "summary_snapshot",
    "value_spec_snapshot",
]
