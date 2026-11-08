# -*- coding: utf-8 -*-
"""实验指标口径：链式指标、Δ 指标、分折榜与执行归因的**适配层**。

硬口径（设计 §4.2 / §5.3 / §5.3.1 / §5.5）：

- 数学指标只调用既有唯一实现：``calculate_chain_metrics``（链式）、``compare_runs``
  （配对/Δ/分折）、``fold_table``（逐折收益与回撤）、``aggregate_run``（旧 WF TopK 跨折均值）；
- **禁止复制公式**：本模块只做列映射（中文 → P6 列）、状态判定与不可比判定，不实现任何
  收益/回撤/夏普/TopK 计算；
- **禁止调用** ``bootstrap_delta`` / ``bootstrap_table``（生成链拦截由
  :mod:`src.lazybull.v2.reporting.experiment_guard` 的 AST 扫描 + 运行期哨兵承担）；
- 跨实现（runs vs 旧 WF）只做对账陈列：不产 Δ、不合并成同一可比结论。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.compare.aggregate import aggregate_run
from scripts.compare.constants import COL_NAMES
from scripts.compare.fold_subset import RunArtifacts, compare_runs
from src.lazybull.v2.evidence.runs_schema import ATTR_CONTRACT_COLS
from src.lazybull.v2.reporting import contracts, experiment_sources, status
from src.lazybull.v2.reporting.experiment_sources import ArmData

__all__ = [
    "ATTR_COLUMNS",
    "COMPARE_SUMMARY_MAP",
    "TOP20_COLUMN",
    "TOP30_COLUMN",
    "arm_metrics",
    "artifacts_for",
    "attribution_rows",
    "chain_rows_frame",
    "check_compare_mapping",
    "comparability_reasons",
    "compare_baseline",
    "compare_pair",
    "cost_turnover",
    "fold_metric_rows",
    "names",
    "reconcile_implementation",
    "topk_values",
]

#: 逐折 TopK 来源列（旧 WF summary；仅字段改名，不重算）
TOP20_COLUMN = "KEY_Top20_hit_rate"
TOP30_COLUMN = "KEY_Top30_hit_rate"
#: compare_runs 中文输出 → P6 列（设计 §5.3.1 显式映射，禁止按英文同名键猜测）
COMPARE_SUMMARY_MAP: dict[str, str] = {
    "运行": "run_label",
    "子集CAGR": "subset_cagr",
    "子集最大回撤": "subset_max_drawdown",
    "子集夏普": "subset_sharpe",
    "子集总收益": "subset_total_return",
    "子集波动": "subset_volatility",
    "子集交易日数": "subset_trading_days",
    "全周期CAGR": "full_cagr",
    "全周期最大回撤": "full_max_drawdown",
    "全周期夏普": "full_sharpe",
    "ΔCAGR(vs基线)": "delta_cagr",
    "Δ最大回撤(vs基线)": "delta_max_drawdown",
    "Δ夏普(vs基线)": "delta_sharpe",
    "折收益为正(子集)": "fold_positive_count",
    "折收益同向(vs基线)": "fold_return_aligned_count",
    "折数": "fold_count",
}
#: fold_table 中文输出 → P6 列
FOLD_TABLE_MAP: dict[str, str] = {
    "折序号": "split_index",
    "折起始净值": "fold_start_nav",
    "折结束净值": "fold_end_nav",
    "折收益": "fold_return",
    "折最大回撤": "fold_max_drawdown",
    "折交易日数": "fold_trading_days",
}
#: 归因契约全 19 列（唯一列名来源）
ATTR_COLUMNS: list[str] = list(ATTR_CONTRACT_COLS)


def check_compare_mapping() -> list[str]:
    """校验本模块映射与 T0.5 冻结 ``COMPARE_MAPPING`` 的列名一致（禁止静默漂移）。"""
    frozen = {row["p6_column"] for row in contracts.COMPARE_MAPPING}
    declared = set(COMPARE_SUMMARY_MAP.values())
    problems: list[str] = []
    for column in ("subset_cagr", "subset_max_drawdown", "subset_sharpe"):
        if column not in frozen or column not in declared:
            problems.append(f"子集指标列未同时出现在冻结映射与实现映射：{column}")
    for column in ("delta_cagr", "delta_max_drawdown", "delta_sharpe"):
        if column not in frozen or column not in declared:
            problems.append(f"Δ 列未同时出现在冻结映射与实现映射：{column}")
    if "fold_return_aligned_count" not in frozen or "fold_return_aligned_count" not in declared:
        problems.append("折收益同向计数列未同时出现在冻结映射与实现映射")
    if "run_label" not in frozen or "run_label" not in declared:
        problems.append("运行标签列未同时出现在冻结映射与实现映射")
    return problems


def _status(word: str, reason: str | None, *, source_ref: str, detail: str) -> dict[str, Any]:
    """状态行（闭集构造器；禁止自由文本冒充原因码）。"""
    return status.make_status(word, reason_code=reason, source_ref=source_ref, detail=detail)


def arm_metrics(arm: ArmData) -> dict[str, Any]:
    """单臂链式指标：全行门 → 唯一实现（``verify_chain_rows`` 内已算并判定 N−S）。

    读入阶段已判 ``invalid`` 的臂（缺必需文件/多候选/契约违例）**原样返回其原因码**，
    不得改写成 ``chain_source_missing``（A-05）；链路坏行同样保留门禁给出的具体原因码。
    """
    source_ref = f"arm:{arm.spec.arm_id}/chain_nav"
    if arm.status.get("report_status") == "invalid":
        return {"status": dict(arm.status), "metrics": {}, "trading_days": None}
    if arm.chain.empty or arm.rows_total <= 0:
        row = _status("invalid", "chain_source_missing", source_ref=source_ref, detail="链式净值不可用")
        return {"status": row, "metrics": {}, "trading_days": None}
    gate = experiment_sources.validate_chain_rows(arm.chain)
    if not status.is_usable(gate):
        return {"status": dict(gate), "metrics": {}, "trading_days": gate.get("trading_days")}
    metrics = dict(gate.get("metrics") or {})
    trading_days = gate.get("trading_days")
    row = _status(
        "available",
        None,
        source_ref=source_ref,
        detail=f"链式指标 {arm.rows_total} 行 / {arm.split_count} 折（trading_days={trading_days}）",
    )
    return {"status": row, "metrics": metrics, "trading_days": trading_days}


def _confirmed_data_state(arm: ArmData) -> str | None:
    """已确认的数据态身份（``None``/空白串都视为**未知**，不得当作同态）。"""
    value = arm.data_state_id
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def comparability_reasons(baseline: ArmData, arm: ArmData, splits: Sequence[int]) -> list[str]:
    """差值可比性五要素（数据态/折集合/窗口/host+实现/基线确认）。

    数据态必须**两侧均可确认**才比较相等性：``None == None`` 不是同态证据
    （旧 WF 缺数据态 JSON 时不得输出可比较 Δ，R3-01）。
    """
    reasons: list[str] = []
    if baseline.status.get("report_status") == "invalid" or baseline.artifacts is None:
        reasons.append("基线臂不可用（baseline_unconfirmed）")
    if arm.status.get("report_status") == "invalid" or arm.artifacts is None:
        reasons.append("该臂不可用")
    base_state = _confirmed_data_state(baseline)
    arm_state = _confirmed_data_state(arm)
    if base_state is None or arm_state is None:
        reasons.append(
            "数据态不可确认（"
            f"baseline={baseline.data_state_id!r} vs arm={arm.data_state_id!r}；未知不等于同态）"
        )
    elif base_state != arm_state:
        reasons.append(f"数据态不一致（baseline={base_state} vs arm={arm_state}）")
    base_splits = sorted(int(item) for item in set(baseline.windows.get("split_index", [])))
    arm_splits = sorted(int(item) for item in set(arm.windows.get("split_index", [])))
    if base_splits != arm_splits:
        reasons.append(f"折集合不一致（{base_splits} vs {arm_splits}）")
    requested = {int(item) for item in splits}
    if not requested <= set(base_splits) or not requested <= set(arm_splits):
        reasons.append(
            f"请求折子集 {sorted(requested)} 超出可用折（base={base_splits} vs arm={arm_splits}）"
        )
    if not _windows_equal(baseline.windows, arm.windows):
        reasons.append("逐折窗口不一致（test_start/test_end）")
    if baseline.host_mode != arm.host_mode:
        reasons.append(f"host_mode 不一致（{baseline.host_mode} vs {arm.host_mode}）")
    if baseline.implementation != arm.implementation:
        reasons.append(f"实现不一致（{baseline.implementation} vs {arm.implementation}）")
    return reasons


def _windows_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """逐折窗口比较（缺列即视为不一致，不做隐式补齐）。"""
    needed = {"split_index", "test_start", "test_end"}
    if not needed.issubset(left.columns) or not needed.issubset(right.columns):
        return False
    left_frame = left[["split_index", "test_start", "test_end"]].astype(str).sort_values("split_index")
    right_frame = right[["split_index", "test_start", "test_end"]].astype(str).sort_values("split_index")
    return left_frame.reset_index(drop=True).equals(right_frame.reset_index(drop=True))


def _py(value: Any) -> Any:
    """numpy 标量 → Python 标量；非有限浮点一律 ``None``（不改取值，只归一表示）。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def _map_frame(frame: pd.DataFrame, mapping: Mapping[str, str]) -> dict[str, Any]:
    """按显式映射取首个数据行的原值（缺列写 ``None``，不做同名键猜测）。"""
    if frame.empty:
        return {}
    first = frame.iloc[0]
    mapped: dict[str, Any] = {}
    for source, target in mapping.items():
        if source not in frame.columns:
            mapped[target] = None
            continue
        value = first[source]
        mapped[target] = None if pd.isna(value) else _py(value)
    return mapped


def compare_pair(
    baseline: ArmData, arm: ArmData, splits: Sequence[int]
) -> dict[str, Any]:
    """配对对比：调 ``compare_runs`` 并按其中文输出显式映射（不重算 Δ）。"""
    if baseline.artifacts is None or arm.artifacts is None:
        row = _status(
            "not_available",
            "not_comparable",
            source_ref=f"arm:{arm.spec.arm_id}",
            detail="臂不可用，无法配对",
        )
        return {"status": row, "summary": {}, "folds": []}
    reasons = comparability_reasons(baseline, arm, splits)
    if reasons:
        row = _status(
            "not_available",
            "not_comparable",
            source_ref=f"arm:{arm.spec.arm_id}",
            detail="；".join(reasons),
        )
        return {"status": row, "summary": {}, "folds": []}
    try:
        result = compare_runs(baseline.artifacts, [arm.artifacts], list(splits))
    except (ValueError, KeyError) as exc:
        row = _status(
            "not_available",
            "not_comparable",
            source_ref=f"arm:{arm.spec.arm_id}",
            detail=f"既有对比实现拒绝配对：{type(exc).__name__}: {exc}",
        )
        return {"status": row, "summary": {}, "folds": []}
    summary = result["summary"]
    arm_row = summary[summary["运行"] == arm.spec.arm_id]
    mapped = _map_frame(arm_row, COMPARE_SUMMARY_MAP)
    folds = _map_folds(result["folds"], arm.spec.arm_id)
    row = _status(
        "available",
        None,
        source_ref=f"arm:{arm.spec.arm_id}/compare_runs",
        detail=f"配对对比通过（{len(splits)} 折；Δ 由既有实现给出）",
    )
    return {"status": row, "summary": mapped, "folds": folds}


def compare_baseline(baseline: ArmData, splits: Sequence[int]) -> dict[str, Any]:
    """基线自身的配对口径：``compare_runs(baseline, [], splits)``（Δ 恒为 0）。

    基线不能作为自己的“对照臂”传入（``fold_table`` 透视会因同名运行重复而失败），
    故走既有实现的空臂路径：子集指标/逐折收益/同向计数全部来自唯一实现。
    """
    if baseline.status.get("report_status") == "invalid" or baseline.artifacts is None:
        # 坏链/坏来源的基线**禁止**进入旧实现：子集口径也不得泄漏数值（R3-02）；
        # 保留基线自身具体原因码，不降级成笼统的不可比。
        preserved = dict(baseline.status)
        preserved["detail"] = "基线自身不可用（保留原原因码）；子集口径不计算"
        if not preserved.get("degradation_reason_code"):
            preserved.update(_status("not_available", "not_comparable", source_ref=None, detail=""))
        return {"status": preserved, "summary": {}, "folds": []}
    try:
        result = compare_runs(baseline.artifacts, [], list(splits))
    except (ValueError, KeyError) as exc:
        row = _status(
            "not_available",
            "not_comparable",
            source_ref=f"arm:{baseline.spec.arm_id}",
            detail=f"既有对比实现拒绝基线口径：{type(exc).__name__}: {exc}",
        )
        return {"status": row, "summary": {}, "folds": []}
    summary = result["summary"]
    mapped = _map_frame(summary[summary["运行"] == baseline.spec.arm_id], COMPARE_SUMMARY_MAP)
    folds = _map_folds(result["folds"], baseline.spec.arm_id)
    row = _status(
        "available",
        None,
        source_ref=f"arm:{baseline.spec.arm_id}/compare_runs",
        detail=f"基线配对口径（{len(splits)} 折；Δ 恒为 0）",
    )
    return {"status": row, "summary": mapped, "folds": folds}


def _map_folds(frame: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    """逐折明细（既有 ``fold_table`` 输出）映射为 P6 列。"""
    if frame.empty:
        return []
    subset = frame[frame["运行"] == label]
    rows: list[dict[str, Any]] = []
    for _, item in subset.iterrows():
        row: dict[str, Any] = {}
        for source, target in FOLD_TABLE_MAP.items():
            value = item.get(source)
            row[target] = None if value is None or pd.isna(value) else _py(value)
        rows.append(row)
    return sorted(rows, key=lambda item: int(item.get("split_index") or 0))


def reconcile_implementation(left: ArmData, right: ArmData) -> dict[str, Any]:
    """跨实现对账：两边各自用同一函数出指标，只并列为“仅对账”，不产 Δ。"""
    left_metrics = arm_metrics(left)["metrics"]
    right_metrics = arm_metrics(right)["metrics"]
    comparisons: list[dict[str, Any]] = []
    for key in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe", "trading_days"):
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        comparisons.append(
            {
                "metric": key,
                "left": left_value,
                "right": right_value,
                "equal": left_value == right_value,
            }
        )
    row = _status(
        "not_available",
        "not_comparable",
        source_ref=f"arm:{right.spec.arm_id}",
        detail=(
            f"跨实现（{left.implementation} vs {right.implementation}）只允许对账陈列："
            "不计算差值、不合并为同一结论"
        ),
    )
    return {"status": row, "comparisons": comparisons}


def topk_values(arm: ArmData) -> dict[str, Any]:
    """TopK 正日占比：规范化 runs 固定降级；旧 WF 逐折取 ``KEY_*`` 并经聚合对账。"""
    if arm.implementation == "runs":
        row = _status(
            "not_available",
            "no_unique_implementation",
            source_ref=f"arm:{arm.spec.arm_id}/topk",
            detail="规范化 runs 无 TopK 正日占比唯一实现；禁止从 topk_detail 私算",
        )
        return {"status": row, "per_fold": {}, "means": {}}
    summary = arm.summary
    per_fold: dict[int, dict[str, Any]] = {}
    for _, item in summary.iterrows():
        split = int(item["split_index"])
        per_fold[split] = {
            "topk20": _optional_float(item.get(TOP20_COLUMN)),
            "topk30": _optional_float(item.get(TOP30_COLUMN)),
        }
    if TOP20_COLUMN not in summary.columns and TOP30_COLUMN not in summary.columns:
        row = _status(
            "not_available",
            "no_source",
            source_ref=f"arm:{arm.spec.arm_id}/summary",
            detail=f"旧 WF summary 无 {TOP20_COLUMN}/{TOP30_COLUMN}（不拿其它命中率替代）",
        )
        return {"status": row, "per_fold": per_fold, "means": {}}
    aggregated = aggregate_run(summary)
    means = {
        "topk20_positive_day_ratio_mean": _py(aggregated.get("key_top20_hit_rate_mean")),
        "topk30_positive_day_ratio_mean": _py(aggregated.get("key_top30_hit_rate_mean")),
    }
    if all(value is None for value in means.values()):
        # 列存在但全空：不得读成 available（逐折与均值都无值，A-11）
        row = _status(
            "degraded",
            "no_source",
            source_ref=f"arm:{arm.spec.arm_id}/summary",
            detail=f"{TOP20_COLUMN}/{TOP30_COLUMN} 存在但全部为空（不拿其它命中率替代）",
        )
        return {"status": row, "per_fold": per_fold, "means": means}
    row = _status(
        "available",
        None,
        source_ref=f"arm:{arm.spec.arm_id}/summary",
        detail=f"旧 WF {TOP20_COLUMN}/{TOP30_COLUMN} 仅字段改名；跨折均值取自 aggregate_run",
    )
    return {"status": row, "per_fold": per_fold, "means": means}


def _optional_float(value: Any) -> float | None:
    """可选浮点（非数值/NaN 写 ``None``，不补 0）。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def fold_metric_rows(
    arm: ArmData,
    compare: Mapping[str, Any] | None = None,
    splits: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """逐折指标行（源 summary 原值 + TopK 映射 + 配对折收益/回撤；行集 = 折子集）。"""
    pair_folds = {int(item["split_index"]): item for item in (compare or {}).get("folds", [])}
    target = None if splits is None else {int(item) for item in splits}
    topk = topk_values(arm)
    rows: list[dict[str, Any]] = []
    for _, item in arm.summary.iterrows():
        split = int(item["split_index"])
        if target is not None and split not in target:
            continue
        fold_pair = pair_folds.get(split, {})
        topk_row = topk["per_fold"].get(split, {})
        rows.append(
            {
                "arm_id": arm.spec.arm_id,
                "split_index": split,
                "test_start": _text_or_none(item.get("test_start")),
                "test_end": _text_or_none(item.get("test_end")),
                "model_version": _text_or_none(item.get("model_version")),
                "wf_run_id": _text_or_none(item.get("wf_run_id")) or arm.batch_id,
                "bt_total_return": _optional_float(item.get("bt_total_return")),
                "bt_annual_return": _optional_float(item.get("bt_annual_return")),
                "bt_max_drawdown": _optional_float(item.get("bt_max_drawdown")),
                "bt_sharpe": _optional_float(item.get("bt_sharpe")),
                "topk20_positive_day_ratio": topk_row.get("topk20"),
                "topk30_positive_day_ratio": topk_row.get("topk30"),
                "fold_return": fold_pair.get("fold_return"),
                "fold_max_drawdown": fold_pair.get("fold_max_drawdown"),
            }
        )
    return rows


def _text_or_none(value: Any) -> str | None:
    """文本原值（NaN/None → ``None``）。"""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return str(value)


def attribution_rows(arm: ArmData) -> list[dict[str, Any]]:
    """执行归因原始行（全 19 列原值 + 原 status/reason + 源行序）。"""
    rows: list[dict[str, Any]] = []
    for split, fold in sorted(arm.folds.items()):
        frame = fold.attribution
        if frame is None:
            continue
        for index, (_, item) in enumerate(frame.iterrows()):
            row: dict[str, Any] = {
                "arm_id": arm.spec.arm_id,
                "split_index": int(split),
                "source_row_index": int(index),
            }
            for column in ATTR_COLUMNS:
                row[column] = _cell(item.get(column))
            rows.append(row)
    return rows


def _cell(value: Any) -> Any:
    """单元格原值（NaN → ``None``；不把缺失写成 0/空串）。"""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if pd.isna(value):
        return None
    return value


def _collect_cost_frames(
    arm: ArmData,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str]]:
    """逐折收集 daily/trades 帧（缺帧登记，不补造）。"""
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for split, fold in sorted(arm.folds.items()):
        if fold.daily is None:
            missing.append(f"split{split}/daily")
            continue
        frame = fold.daily.copy()
        frame["split_index"] = int(split)
        daily_frames.append(frame)
        if fold.trades is None:
            missing.append(f"split{split}/trades")
            continue
        trades = fold.trades.copy()
        trades["split_index"] = int(split)
        trade_frames.append(trades)
    return daily_frames, trade_frames, missing


def _summarize_turnover(
    daily_frames: Sequence[pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, list[str], list[str]]:
    """按 (折, 日期) 与按折汇总双边成交金额，并给出折界重复日与未知金额行。

    金额缺失/非有限不得补零（设计 §5.1/§5.5）：任一消费行金额未知即整体降级，
    合计写 ``None``，并把坏行位置登记给状态行（R3-06）。
    """
    if not daily_frames:
        return [], [], None, [], []
    daily = pd.concat(daily_frames, ignore_index=True)
    if "turnover_amount" not in daily.columns:
        return [], [], None, [], ["turnover_amount 列缺失"]
    amounts = pd.to_numeric(daily["turnover_amount"], errors="coerce").astype(float)
    # 缺失与 ±Inf 都属金额未知（isna 不识别 Inf，R3-06-R1）
    invalid = ~np.isfinite(amounts.to_numpy(dtype=float))
    unknown = daily[invalid]
    bad_rows = [
        f"split{int(item.split_index)}/{item.trade_date}"
        for item in unknown[["split_index", "trade_date"]].itertuples()
    ]
    day_splits = daily.groupby("trade_date")["split_index"].nunique()
    seam_days = sorted(str(day) for day, count in day_splits.items() if int(count) > 1)
    if bad_rows:
        return [], [], None, seam_days, bad_rows[:5]
    daily = daily.assign(p6_turnover_amount=amounts)
    grouped = (
        daily.groupby(["split_index", "trade_date"], as_index=False)["p6_turnover_amount"]
        .sum()
        .sort_values(["split_index", "trade_date"])
    )
    rows = [
        {
            "split_index": int(item.split_index),
            "trade_date": str(item.trade_date),
            "turnover_amount": float(item.p6_turnover_amount),
        }
        for item in grouped.itertuples()
    ]
    with np.errstate(over="ignore"):
        per_fold = grouped.groupby("split_index", as_index=False)["p6_turnover_amount"].sum()
    by_fold = [
        {
            "split_index": int(item.split_index),
            "turnover_amount": float(item.p6_turnover_amount),
        }
        for item in per_fold.itertuples()
    ]
    with np.errstate(over="ignore"):
        total = float(grouped["p6_turnover_amount"].sum())
    # 逐行有限仍可能在聚合后溢出（如每天 1e308）；聚合结果也必须有限（R3-06-R2）
    aggregates = [total] + [item["turnover_amount"] for item in rows]
    aggregates += [item["turnover_amount"] for item in by_fold]
    if not all(np.isfinite(item) for item in aggregates):
        return [], [], None, seam_days, ["成交金额聚合后非有限"]
    return rows, by_fold, total, seam_days, []


def cost_turnover(arm: ArmData) -> dict[str, Any]:
    """双边成交金额（daily.turnover_amount）与成本（trades.cost），按 fold+date 聚合。"""
    daily_frames, trade_frames, missing = _collect_cost_frames(arm)
    turnover_rows, turnover_by_fold, turnover_total, seam_days, unknown_turnover = (
        _summarize_turnover(daily_frames)
    )
    cost_total: float | None = None
    unknown_cost: list[str] = []
    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        if "cost" in trades.columns:
            values = pd.to_numeric(trades["cost"], errors="coerce").astype(float)
            invalid_cost = ~np.isfinite(values.to_numpy(dtype=float))
            if invalid_cost.any():
                unknown_cost = [f"成交成本存在 {int(invalid_cost.sum())} 行缺失/非有限值"]
            else:
                cost_total = float(values.sum())
                if not np.isfinite(cost_total):
                    cost_total = None
                    unknown_cost = ["成交成本合计非有限"]
        else:
            unknown_cost = ["cost 列缺失"]
    unknown = unknown_turnover + unknown_cost
    if missing and not daily_frames:
        row = _status(
            "not_available",
            "fold_file_missing",
            source_ref=f"arm:{arm.spec.arm_id}/daily",
            detail=f"无逐日来源（{missing[:3]}）——双边成交金额不可得",
        )
    elif unknown:
        # 有源行但金额未知：不得展示成零成本/零成交（R3-06）
        row = _status(
            "degraded",
            "missing_precondition",
            source_ref=f"arm:{arm.spec.arm_id}/daily",
            detail=f"金额缺失或非有限（不补零）：{'；'.join(unknown[:3])}",
        )
    elif missing:
        row = _status(
            "degraded",
            "fold_file_missing",
            source_ref=f"arm:{arm.spec.arm_id}/daily",
            detail=f"部分折缺逐日/成交来源（{missing[:3]}），合计值只覆盖已有折",
        )
    else:
        row = _status(
            "available",
            None,
            source_ref=f"arm:{arm.spec.arm_id}/daily",
            detail="双边成交金额与成本按 fold+date 聚合（折界重复日不归并）",
        )
    return {
        "status": row,
        "turnover_by_fold_date": turnover_rows,
        "turnover_by_fold": turnover_by_fold,
        "turnover_total": turnover_total,
        "cost_total": cost_total,
        "seam_duplicate_days": seam_days,
        "missing": missing,
        "unknown_amounts": unknown,
    }


def artifacts_for(arm: ArmData) -> RunArtifacts | None:
    """臂的 ``RunArtifacts``（compare_runs 入参；不可用臂返回 ``None``）。"""
    return arm.artifacts


def chain_rows_frame(arm: ArmData) -> list[dict[str, Any]]:
    """链式净值全行（原行序；CSV 行键 = arm+split+source_row_index）。"""
    rows: list[dict[str, Any]] = []
    for _, item in arm.chain.iterrows():
        rows.append(
            {
                "arm_id": arm.spec.arm_id,
                "split_index": int(item["split_index"]),
                "source_row_index": int(item["source_row_index"]),
                "date": _text_or_none(item.get("date")),
                "nav": float(item["nav"]),
            }
        )
    return rows


def names() -> dict[str, str]:
    """``COL_NAMES`` 转导出（供稳定性层映射列名，避免二次定义）。"""
    return dict(COL_NAMES)
