# -*- coding: utf-8 -*-
"""实验报告投影装配（P6 T3：显式基线/臂 → 指标/CSV/证据卡 → 投影）。

职责：

- **只做装配与状态判定**：所有数值来自 ``experiment_metrics``（其只调用既有唯一实现），
  本模块不实现任何公式，也不做采纳/闸门结论；
- 五份 CSV 的**行集**在此一次生成（同一源行对象，避免“HTML 与 CSV 两套口径”）；
- 制度摘要只读显式 P5a JSON 的既有分箱计数；无 bins 时固定 ``not_available/no_regime_bins``，
  **不画伪分布、不重采样**；
- 四段证据卡是"信号 / 执行 / β暴露 / 成本"，**不是守恒瀑布**，成本金额差不称 NAV 贡献。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.compare.aggregate import aggregate_run
from src.lazybull.v2.reporting import (
    experiment_csv,
    experiment_metrics,
    experiment_sources,
    experiment_stability,
    status,
    svg,
)
from src.lazybull.v2.reporting.experiment_sources import ArmData

__all__ = [
    "EXPERIMENT_SCHEMA",
    "build_experiment_projection",
    "expected_csv_rows",
    "extract_regime",
]

#: 投影 schema（不进入 manifest 自封存之外的任何派生口径）
EXPERIMENT_SCHEMA = "p6_experiment_projection/v1"
#: 四段证据卡（固定顺序与固定措辞；禁止改称守恒瀑布）
EVIDENCE_CARDS: tuple[str, ...] = ("信号", "执行", "β暴露", "成本")
CARD_NOTE = "四段证据卡（信号/执行/β暴露/成本）不是守恒瀑布：各卡独立列证据，不可相加抵账。"


def _status(word: str, reason: str | None, *, source_ref: str, detail: str) -> dict[str, Any]:
    """状态行（闭集构造器）。"""
    return status.make_status(word, reason_code=reason, source_ref=source_ref, detail=detail)


def extract_regime(
    payload: Mapping[str, Any] | None, *, error: str | None = None
) -> dict[str, Any]:
    """制度重排摘要（只读显式 P5a JSON；有 bins 才陈列计数，不重采样）。"""
    if error is not None:
        row = _status(
            "invalid",
            error,
            source_ref="regime",
            detail="制度重排 JSON 损坏或顶层不是对象（不猜内容）",
        )
        return {"status": row, "results": [], "point_delta": None, "bins": None}
    if not payload:
        row = _status(
            "not_available",
            "no_source",
            source_ref="regime",
            detail="未提供显式 P5a 制度重排 JSON（不伪造分布）",
        )
        return {"status": row, "results": [], "point_delta": None, "bins": None}
    results = payload.get("results")
    rows: list[dict[str, Any]] = []
    if isinstance(results, Mapping):
        for key in sorted(results):
            value = results[key]
            if isinstance(value, Mapping):
                rows.append({"scene": str(key), **{k: _scalar(v) for k, v in sorted(value.items())}})
            else:
                rows.append({"scene": str(key), "value": _scalar(value)})
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, Mapping):
                rows.append({str(k): _scalar(v) for k, v in item.items()})
    bins = payload.get("bins")
    point_delta = payload.get("point_delta")
    if not isinstance(bins, Mapping) or not bins:
        row = _status(
            "not_available",
            "no_regime_bins",
            source_ref="regime",
            detail="来源只有摘要、无分箱计数：不画分布、不重采样（未来有合法 bins 才另评）",
        )
        return {
            "status": row,
            "results": rows,
            "point_delta": _scalar(point_delta),
            "bins": None,
        }
    counts = {
        str(scene): {str(key): _scalar(value) for key, value in sorted(dict(items).items())}
        for scene, items in sorted(bins.items())
        if isinstance(items, Mapping)
    }
    row = _status(
        "available",
        None,
        source_ref="regime",
        detail="只读既有分箱计数（原值陈列，不重采样）",
    )
    return {"status": row, "results": rows, "point_delta": _scalar(point_delta), "bins": counts}


def _scalar(value: Any) -> Any:
    """标量归一（数组/嵌套结构保留 JSON 结构，NaN → ``None``）。"""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _scalar(item) for key, item in sorted(value.items())}
    return value


def _arm_identity(arm: ArmData) -> dict[str, Any]:
    """臂身份（原值；缺失写 ``None``，不猜）。"""
    return {
        "arm_id": arm.spec.arm_id,
        "source_kind": arm.spec.source_kind,
        "root_id": arm.spec.root_id,
        "batch_id": arm.batch_id,
        "host_mode": arm.host_mode,
        "implementation": arm.implementation,
        "data_state_id": arm.data_state_id,
        "config_fingerprint": arm.config_fingerprint,
        "watermarks": dict(arm.watermarks),
    }


def _arm_entry(
    arm: ArmData, baseline_arm: ArmData, splits: Sequence[int]
) -> dict[str, Any]:
    """单臂投影条目（指标 + Δ + 分折 + TopK + 成本 + 归因统计）。"""
    metrics = experiment_metrics.arm_metrics(arm)
    baseline = arm.spec.arm_id == baseline_arm.spec.arm_id
    # 基线也走同一 ``compare_runs`` 实现（空臂路径，Δ 恒为 0）：子集指标/逐折收益/同向计数
    # 全部来自唯一实现，禁止用链式全周期指标冒充子集口径（A-02/A-03/A-09）
    compare = (
        experiment_metrics.compare_baseline(arm, splits)
        if baseline
        else experiment_metrics.compare_pair(baseline_arm, arm, splits)
    )
    topk = experiment_metrics.topk_values(arm)
    reconciliation = None
    if arm.implementation != baseline_arm.implementation and not baseline:
        reconciliation = experiment_metrics.reconcile_implementation(baseline_arm, arm)
    cost = experiment_metrics.cost_turnover(arm)
    attribution = experiment_metrics.attribution_rows(arm)
    folds = experiment_metrics.fold_metric_rows(arm, compare, splits)
    status_row = metrics["status"]
    if status_row.get("report_status") == "available" and compare["status"]["report_status"] != "available":
        status_row = _status(
            "degraded",
            str(compare["status"].get("degradation_reason_code") or "not_comparable"),
            source_ref=f"arm:{arm.spec.arm_id}",
            detail=str(compare["status"].get("detail") or ""),
        )
    return {
        "identity": _arm_identity(arm),
        "is_baseline": baseline,
        "status": status_row,
        "chain": {
            "rows_total": arm.rows_total,
            "rows_used": arm.rows_used,
            "split_count": arm.split_count,
            "trading_days": metrics.get("trading_days"),
            "metrics": metrics["metrics"],
            "chart": _chart_series(arm),
        },
        "compare": {
            "status": compare["status"],
            "summary": compare["summary"],
            "folds": list(compare.get("folds") or []),
        },
        "reconciliation": reconciliation,
        "topk": {
            "status": topk["status"],
            "means": topk["means"],
            "per_fold": {str(key): value for key, value in sorted(topk["per_fold"].items())},
        },
        "cost": {
            "status": cost["status"],
            "turnover_total": cost["turnover_total"],
            "cost_total": cost["cost_total"],
            # 逐 (折, 日期) 明细只进投影计算；projection 只留逐折合计（体积可控）
            "by_fold": cost["turnover_by_fold"],
            "seam_duplicate_days": len(cost["seam_duplicate_days"]),
        },
        "attribution": {
            "row_count": len(attribution),
            "non_success_status": _non_success_counts(attribution),
        },
        "folds": folds,
        # 源侧逐折行数（= 源 summary ∩ 折子集），供验收独立对账
        "fold_row_count": len(folds),
        "degradations": list(arm.degradations),
    }


def _non_success_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """非成功执行状态计数（保留原 ``status`` 值，不映射成报告状态词）。"""
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get("status")
        key = "" if value is None else str(value)
        if key in ("", "filled", "success"):
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _chart_series(arm: ArmData) -> dict[str, Any]:
    """图上点集（确定性等距抽样；**只影响图**，完整值在 chain_nav.csv）。"""
    if arm.chain.empty:
        return {"series": [], "sampled": False}
    values = [float(item) for item in arm.chain["nav"].tolist()]
    indexes = svg.downsample_indices(values)
    rows = [
        {
            "date": _text(arm.chain.iloc[index].get("date")),
            "value": values[index],
            "matched": True,
        }
        for index in indexes
    ]
    return {"series": rows, "sampled": len(indexes) < len(values)}


def _fold_status(
    arm: ArmData, topk: Mapping[str, Any], fold: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """逐折行状态：臂 invalid > 折收益/回撤缺失（``no_source``）> TopK 来源状态。"""
    if arm.status.get("report_status") == "invalid":
        return dict(arm.status)
    material = dict(fold or {})
    if material.get("fold_return") is None or material.get("fold_max_drawdown") is None:
        # 逐折收益/回撤来自唯一实现；任一缺失必须有本行专用原因码（不得借用 TopK 原因，A-03）
        return _status(
            "degraded",
            "no_source",
            source_ref=f"arm:{arm.spec.arm_id}/fold_table",
            detail="该折缺 fold_table 收益/回撤（唯一实现未给出该折数值）",
        )
    topk_status = dict(topk.get("status") or {})
    if not status.is_usable(topk_status):
        return _status(
            str(topk_status.get("report_status") or "degraded"),
            str(topk_status.get("degradation_reason_code") or "no_source"),
            source_ref=f"arm:{arm.spec.arm_id}/summary",
            detail=str(topk_status.get("detail") or "TopK 来源降级"),
        )
    return _status(
        "available",
        None,
        source_ref=f"arm:{arm.spec.arm_id}/summary",
        detail="逐折指标原值（TopK 仅字段改名）",
    )


def _arm_metric_rows(arm: ArmData, entry: Mapping[str, Any]) -> dict[str, Any]:
    """``arm_metrics.csv`` 行（严格等于声明列）。"""
    identity = dict(entry["identity"])
    metrics = dict(entry["chain"]["metrics"])
    compare = dict(entry["compare"].get("summary") or {})
    status_row = dict(entry["status"])
    row = {
        "arm_id": identity["arm_id"],
        "batch_id": identity["batch_id"],
        "host_mode": identity["host_mode"],
        "implementation": identity["implementation"],
        "data_state_id": identity["data_state_id"],
        "config_fingerprint": identity["config_fingerprint"],
        "total_return": metrics.get("total_return"),
        "cagr": metrics.get("cagr"),
        "max_drawdown": metrics.get("max_drawdown"),
        "volatility": metrics.get("volatility"),
        "sharpe": metrics.get("sharpe"),
        "trading_days": entry["chain"].get("trading_days"),
        "subset_cagr": compare.get("subset_cagr"),
        "subset_max_drawdown": compare.get("subset_max_drawdown"),
        "subset_sharpe": compare.get("subset_sharpe"),
        "delta_cagr": compare.get("delta_cagr"),
        "delta_max_drawdown": compare.get("delta_max_drawdown"),
        "delta_sharpe": compare.get("delta_sharpe"),
        "fold_return_aligned_count": compare.get("fold_return_aligned_count"),
        "fold_count": compare.get("fold_count"),
        "report_status": status_row.get("report_status"),
        "degradation_reason_code": status_row.get("degradation_reason_code"),
        "source_ref": _source_ref(arm, "summary"),
    }
    return row


def _source_ref(arm: ArmData, key: str, split: int | None = None) -> str:
    """CSV 来源引用：优先真实相对路径（``root_id:relative``），否则退回语义别名。"""
    root_id = arm.spec.root_id
    paths = dict(arm.source_paths or {})
    if split is not None:
        relative = paths.get(f"{key}:{int(split)}") or paths.get(key)
    else:
        relative = paths.get(key)
    return f"{root_id}:{relative}" if relative else f"{root_id}:{key}"


def _fold_rows(
    arm: ArmData, entry: Mapping[str, Any], splits: Sequence[int]
) -> list[dict[str, Any]]:
    """``fold_metrics.csv`` 行（逐臂逐折；行集 = 报告声明的折子集）。

    行集由**源 summary 的折键**过滤；比较列对未参与配对的折写 ``None``（不删原始折记录）。
    """
    target = {int(item) for item in splits}
    rows: list[dict[str, Any]] = []
    for item in experiment_metrics.fold_metric_rows(arm, entry.get("compare"), sorted(target)):
        if int(item["split_index"]) not in target:
            continue
        status_row = _fold_status(arm, dict(entry.get("topk") or {}), item)
        rows.append(
            {
                "arm_id": entry["identity"]["arm_id"],
                "split_index": int(item["split_index"]),
                "test_start": item.get("test_start"),
                "test_end": item.get("test_end"),
                "model_version": item.get("model_version"),
                "wf_run_id": item.get("wf_run_id"),
                "bt_total_return": item.get("bt_total_return"),
                "bt_annual_return": item.get("bt_annual_return"),
                "bt_max_drawdown": item.get("bt_max_drawdown"),
                "bt_sharpe": item.get("bt_sharpe"),
                "topk20_positive_day_ratio": item.get("topk20_positive_day_ratio"),
                "topk30_positive_day_ratio": item.get("topk30_positive_day_ratio"),
                "fold_return": item.get("fold_return"),
                "fold_max_drawdown": item.get("fold_max_drawdown"),
                "report_status": status_row["report_status"],
                "degradation_reason_code": status_row["degradation_reason_code"],
                "source_ref": _source_ref(arm, "summary"),
            }
        )
    return rows


def _chain_rows(arm: ArmData, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``chain_nav.csv`` 行（全行原序；合法输入须 rows_used=rows_total）。"""
    status_row = dict(entry["status"])
    rows: list[dict[str, Any]] = []
    for item in experiment_metrics.chain_rows_frame(arm):
        rows.append(
            {
                "arm_id": item["arm_id"],
                "split_index": item["split_index"],
                "source_row_index": item["source_row_index"],
                "date": item["date"] if isinstance(item["date"], str) else _text(item["date"]),
                "nav": item["nav"],
                "report_status": status_row.get("report_status"),
                "degradation_reason_code": status_row.get("degradation_reason_code"),
                "source_ref": _source_ref(arm, "chain_nav"),
            }
        )
    return rows


def _text(value: Any) -> str | None:
    """日期文本（int 折内序号原样转字符串；不改变数值语义）。"""
    if value is None:
        return None
    return str(value)


def _attribution_rows(arm: ArmData, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``execution_attribution.csv`` 行（19 列原值 + 原 status/reason）。"""
    rows: list[dict[str, Any]] = []
    for item in experiment_metrics.attribution_rows(arm):
        row = dict(item)
        row["report_status"] = "available"
        row["degradation_reason_code"] = None
        row["source_ref"] = _source_ref(arm, "attribution", int(item["split_index"]))
        rows.append(row)
    return rows


def _stability_records(arms: Sequence[ArmData]) -> list[experiment_stability.StabilityRecord]:
    """稳定性输入行（每臂一期；不可用臂不进入分组，避免空指标拉低得分）。"""
    records: list[experiment_stability.StabilityRecord] = []
    for arm in arms:
        if arm.status.get("report_status") == "invalid" or arm.rows_used <= 0:
            continue
        metrics = experiment_metrics.arm_metrics(arm)["metrics"]
        aggregated = aggregate_run(arm.summary) if not arm.summary.empty else {}
        model_version = None
        if "model_version" in arm.summary.columns:
            values = pd.to_numeric(arm.summary["model_version"], errors="coerce").dropna()
            model_version = str(int(values.max())) if len(values) else None
        config = dict(arm.config)
        run_id = arm.batch_id
        if "wf_run_id" in arm.summary.columns:
            values = arm.summary["wf_run_id"].dropna().astype(str).unique()
            if len(values) == 1:
                # 旧 WF 的 run 身份是 wf_run_id；batch_run_id 另列，二者不得互相冒充
                run_id = str(values[0])
        records.append(
            experiment_stability.StabilityRecord(
                arm_id=arm.spec.arm_id,
                run_id=run_id,
                config=config,
                host_mode=arm.host_mode,
                implementation=arm.implementation,
                data_state_id=arm.data_state_id,
                watermarks=dict(arm.watermarks),
                metrics={
                    "chain_cagr": _as_float(metrics.get("cagr")),
                    "chain_max_drawdown": _as_float(metrics.get("max_drawdown")),
                    "chain_sharpe": _as_float(metrics.get("sharpe")),
                    "oos_cross_split_ir": _as_float(aggregated.get("oos_cross_split_ir")),
                    "bt_win_rate": _as_float(aggregated.get("bt_win_rate")),
                },
                batch_id=arm.batch_id,
                batch_run_id=str(config.get("batch_run_id") or arm.batch_id),
                # 空标签保持为空，由唯一实现按 wf_start~wf_end 回退并对同窗口重跑去重（R3-04）；
                # 不得用 batch_id 填补冒充“不同期间”
                batch_period_label=str(
                    config.get("batch_period_label") or config.get("批次时间段") or ""
                ),
                split_count=arm.split_count,
                final_date=_as_text(config.get("final_date")),
                model_version=model_version,
            )
        )
    return records


def _as_text(value: Any) -> str | None:
    """可选文本（None/NaN → None）。"""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _money(value: float | None) -> str:
    """金额展示（缺失写 ``—``，不写 0）。"""
    return "—" if value is None else f"{value:.2f}"


def _as_float(value: Any) -> float | None:
    """可选浮点（numpy 标量/NaN → Python float 或 ``None``）。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _signal_card(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """信号卡：链式指标原值（唯一实现）。"""
    cagr_values = [
        float(entry["chain"]["metrics"]["cagr"])
        for entry in entries
        if entry["chain"]["metrics"].get("cagr") is not None
    ]
    return {
        "card": EVIDENCE_CARDS[0],
        "status": _status(
            "available" if cagr_values else "not_available",
            None if cagr_values else "no_source",
            source_ref="arms.chain",
            detail="链式指标原值（唯一实现 calculate_chain_metrics）",
        ),
        "facts": [
            {"label": "臂数", "value": str(len(entries))},
            {"label": "有 CAGR 的臂", "value": str(len(cagr_values))},
            {
                "label": "CAGR 区间",
                "value": f"{min(cagr_values):.6f} ~ {max(cagr_values):.6f}" if cagr_values else "—",
            },
        ],
    }


def _execution_card(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """执行卡：归因行数 + 非成功执行状态原值计数。"""
    attribution_total = sum(int(entry["attribution"]["row_count"]) for entry in entries)
    non_success: dict[str, int] = {}
    for entry in entries:
        for key, value in entry["attribution"]["non_success_status"].items():
            non_success[key] = non_success.get(key, 0) + int(value)
    return {
        "card": EVIDENCE_CARDS[1],
        "status": _status(
            "available" if attribution_total else "not_available",
            None if attribution_total else "fold_file_missing",
            source_ref="arms.attribution",
            detail="执行归因原 19 列（非成功状态按原值计数）",
        ),
        "facts": [
            {"label": "归因行数", "value": str(attribution_total)},
            {
                "label": "非成功执行状态",
                "value": "、".join(f"{key}={value}" for key, value in sorted(non_success.items()))
                or "无",
            },
        ],
    }


def _beta_card() -> dict[str, Any]:
    """β 暴露卡：无唯一实现/产物，固定降级（不借相似字段补造）。"""
    return {
        "card": EVIDENCE_CARDS[2],
        "status": _status(
            "not_available",
            "no_source",
            source_ref="arms.beta_exposure",
            detail="当前没有 β 暴露唯一实现或既有产物：按设计降级，不复制公式",
        ),
        "facts": [{"label": "来源", "value": "暂无来源（不借相似字段补造）"}],
    }


def _cost_card(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """成本卡：双边成交金额与成本金额（金额差不等于 NAV 贡献）。"""
    turnover_values = [
        float(entry["cost"]["turnover_total"])
        for entry in entries
        if entry["cost"]["turnover_total"] is not None
    ]
    cost_values = [
        float(entry["cost"]["cost_total"])
        for entry in entries
        if entry["cost"]["cost_total"] is not None
    ]
    turnover = sum(turnover_values) if turnover_values else None
    cost = sum(cost_values) if cost_values else None
    covered = turnover is not None and cost is not None
    seams = sum(int(entry["cost"].get("seam_duplicate_days") or 0) for entry in entries)
    return {
        "card": EVIDENCE_CARDS[3],
        "status": _status(
            "available" if covered else "not_available",
            None if covered else "fold_file_missing",
            source_ref="arms.cost",
            detail="双边成交金额与成本金额（金额差不等于 NAV 贡献）",
        ),
        "facts": [
            {"label": "双边成交金额合计", "value": _money(turnover)},
            {"label": "成本金额合计", "value": _money(cost)},
            {"label": "覆盖臂数", "value": f"{len(turnover_values)}/{len(entries)}"},
            {"label": "折界重复日（各臂合计）", "value": str(seams)},
        ],
        "note": (
            CARD_NOTE
            + " 双边成交金额按 (折, 日期) 求和，含折界重复日（不归并）；"
            "该合计与 trades.amount 的口径不同，不可直接对账。"
        ),
    }


def _stability_card(stability: Mapping[str, Any]) -> dict[str, Any]:
    """稳定性卡：完整等价类分区结果（不是守恒瀑布的一环）。"""
    return {
        "card": "稳定性",
        "status": dict(stability["status"]),
        "facts": [
            {"label": "稳定等价类", "value": str(stability.get("partitions"))},
            {"label": "稳定性行数", "value": str(len(stability.get("rows") or []))},
        ],
        "note": "稳定性由完整等价类分区 + 旧函数逐项对账给出；无前置/不足两期为显式状态行。",
    }


def _make_cards(
    entries: Sequence[Mapping[str, Any]], stability: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """四段证据卡 + 稳定性卡（独立列证据，禁止相加抵账）。"""
    cards = [
        _signal_card(entries),
        _execution_card(entries),
        _beta_card(),
        _cost_card(entries),
        _stability_card(stability),
    ]
    for card in cards:
        card.setdefault("note", CARD_NOTE)
    return cards


def _data_window(arms: Sequence[ArmData]) -> dict[str, Any]:
    """数据窗口（逐折 test_start/test_end 的最小/最大值；缺失写 ``None``）。"""
    starts: list[str] = []
    ends: list[str] = []
    for arm in arms:
        frame = arm.windows
        if frame.empty:
            continue
        if "test_start" in frame.columns:
            starts.extend(str(item) for item in frame["test_start"].dropna().tolist())
        if "test_end" in frame.columns:
            ends.extend(str(item) for item in frame["test_end"].dropna().tolist())
    return {"start": min(starts) if starts else None, "end": max(ends) if ends else None}


def build_experiment_projection(
    *,
    baseline: str,
    arms: Sequence[ArmData],
    generated_at: str,
    evidence_class: str,
    root_map: Mapping[str, str],
    regime_payload: Mapping[str, Any] | None = None,
    regime_error: str | None = None,
    splits: Sequence[int] | None = None,
) -> dict[str, Any]:
    """装配实验投影与五份 CSV 行集（一次生成，HTML 与 CSV 同源）。"""
    by_id = {arm.spec.arm_id: arm for arm in arms}
    if baseline not in by_id:
        raise experiment_sources.ArmSourceError(f"基线臂 {baseline!r} 不在臂清单 {sorted(by_id)}")
    baseline_arm = by_id[baseline]
    explicit = splits is not None
    target_splits = sorted(
        int(item) for item in (splits if splits is not None else _splits_of(baseline_arm))
    )
    # 完整 CSV 默认保留**各臂 summary 的全部折**（含基线子集外的额外折，R3-07）；
    # 显式 --splits 时按声明子集过滤（CSV 行集 = 声明子集）。
    csv_splits = target_splits if explicit else _all_splits(arms, target_splits)
    entries = [_arm_entry(arm, baseline_arm, target_splits) for arm in arms]
    stability = experiment_stability.build_stability_rows(_stability_records(arms))
    regime = extract_regime(regime_payload, error=regime_error)
    csv_rows = {
        "arm_metrics.csv": [
            _arm_metric_rows(arm, entry) for arm, entry in zip(arms, entries)
        ],
        "fold_metrics.csv": [
            row
            for arm, entry in zip(arms, entries)
            for row in _fold_rows(arm, entry, csv_splits)
        ],
        "chain_nav.csv": [
            row for arm, entry in zip(arms, entries) for row in _chain_rows(arm, entry)
        ],
        "execution_attribution.csv": [
            row for arm, entry in zip(arms, entries) for row in _attribution_rows(arm, entry)
        ],
        "stability.csv": [dict(row) for row in stability["rows"]],
    }
    degradations = _collect_degradations(entries, stability, regime)
    projection = {
        "schema": EXPERIMENT_SCHEMA,
        "generated_at": generated_at,
        "evidence_class": evidence_class,
        "baseline_arm": baseline,
        "splits": target_splits,
        "csv_splits": csv_splits,
        "root_map": {str(key): str(value) for key, value in sorted(root_map.items())},
        "data_window": _data_window(arms),
        "source_watermarks": {
            entry["identity"]["arm_id"]: entry["identity"]["watermarks"] for entry in entries
        },
        "arms": entries,
        "compare_mapping": [
            {"source": source, "p6_column": target}
            for source, target in experiment_metrics.COMPARE_SUMMARY_MAP.items()
        ],
        "stability": {
            "status": stability["status"],
            "partitions": stability["partitions"],
            "expected_row_count": stability.get("expected_row_count"),
            "dynamic_columns": stability["dynamic_columns"],
            "rows": [dict(row) for row in stability["rows"]],
        },
        "regime": regime,
        "cards": _make_cards(entries, stability),
        "csv_files": list(experiment_csv.CSV_FILENAMES),
        "degradations": degradations,
        "notes": [
            CARD_NOTE,
            "未覆盖 F23 图（回撤水面序列、制度重排分布）仍明示未达成：无唯一实现或无 bins 即降级。",
            "本报告不产生采纳/闸门结论；Δ 仅为符号约定的展示口径（ΔMaxDD>0 表示改善）。",
        ],
    }
    return {
        "projection": projection,
        "csv_rows": csv_rows,
        "csv_expected_rows": expected_csv_rows(
            arms, csv_rows, csv_splits, int(stability.get("expected_row_count") or 0)
        ),
    }


def expected_csv_rows(
    arms: Sequence[ArmData],
    csv_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    splits: Sequence[int],
    stability_expected: int,
) -> dict[str, int]:
    """五份 CSV 的**独立期望行数**（来自源侧计数，不是同一行集长度自证）。

    - 臂指标：臂数；逐折：Σ 源 summary 中属于折子集的行数；链式：Σ 源链行数；
    - 归因：Σ 源归因帧行数（缺帧记 0）；稳定性：行集长度（状态行也是行）。
    """
    target = {int(item) for item in splits}
    return {
        "arm_metrics.csv": len(arms),
        "fold_metrics.csv": sum(
            int(arm.summary["split_index"].astype(int).isin(target).sum())
            if "split_index" in arm.summary.columns
            else 0
            for arm in arms
        ),
        "chain_nav.csv": sum(int(arm.rows_total) for arm in arms),
        "execution_attribution.csv": sum(
            len(fold.attribution)
            for arm in arms
            for fold in arm.folds.values()
            if fold.attribution is not None
        ),
        "stability.csv": int(stability_expected),
    }


def _all_splits(arms: Sequence[ArmData], fallback: Sequence[int]) -> list[int]:
    """各臂 summary 的折键并集（默认完整 CSV 的行集；仍按源键集过滤）。"""
    found: set[int] = set()
    for arm in arms:
        if "split_index" in arm.summary.columns and not arm.summary.empty:
            found.update(int(item) for item in arm.summary["split_index"].tolist())
    return sorted(found or {int(item) for item in fallback})


def _splits_of(arm: ArmData) -> list[int]:
    """基线臂的可用折集合（升序；无窗口时退回链内折号）。"""
    if not arm.windows.empty and "split_index" in arm.windows.columns:
        return sorted(int(item) for item in set(arm.windows["split_index"].tolist()))
    if not arm.chain.empty:
        return sorted(int(item) for item in set(arm.chain["split_index"].tolist()))
    return []


def _collect_degradations(
    entries: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Any],
    regime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """合并降级登记（按 field_path 稳定排序去重）。"""
    collected: list[dict[str, Any]] = []
    for entry in entries:
        collected.extend(dict(row) for row in entry.get("degradations") or [])
        for key, row in (
            ("compare", entry["compare"]["status"]),
            ("topk", entry["topk"]["status"]),
            ("cost", entry["cost"]["status"]),
        ):
            if status.is_usable(row):
                continue
            collected.append(
                status.degradation_entry(
                    row, field_path=f"arms.{entry['identity']['arm_id']}.{key}"
                )
            )
    if not status.is_usable(stability["status"]):
        collected.append(status.degradation_entry(stability["status"], field_path="stability"))
    if not status.is_usable(regime["status"]):
        collected.append(status.degradation_entry(regime["status"], field_path="regime"))
    return status.collect_degradations(collected)
