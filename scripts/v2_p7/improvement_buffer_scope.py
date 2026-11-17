"""到期前仍居 Top20 的作用面补充：只读计划与既有信号，不构造候选收益。

以到期前一 SSE 交易日的当前模型查询持仓排除前 Top20；缺排名单列不可判定，
给出全包含/全排除边界。预算是原买入计划逐股金额，不能解释成到期市值或节省费用。
复杂度豁免：阶段专用首关诊断，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "data/paper_shadow/p7_testbed/p7_t25_hk_20261112_r6"
PLAN_COLUMNS = [
    "ts_code",
    "signal_date",
    "execution_date",
    "due_date",
    "split_index",
    "model_version",
    "tranche_idx",
    "budget",
]


def sha256(path: Path) -> str:
    """计算原文件字节身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """只读 UTF-8 元信息。"""
    return json.loads(path.read_text(encoding="utf-8"))


def date_key(value: Any) -> str:
    """统一已有日期表示，缺失保留为空。"""
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return pd.Timestamp(str(value)).strftime("%Y%m%d")


def validate_topk(frame: pd.DataFrame, split: int, span: list[str]) -> pd.DataFrame:
    """校验信号唯一键、日期窗口与 Top20 连续名次，不补齐未保存的完整排名。"""
    result = frame.copy(deep=True)
    result["trade_date"] = result.trade_date.map(date_key)
    if result[["trade_date", "ts_code", "rank"]].isna().any().any():
        raise ValueError(f"split{split:02d} Top20 含空身份")
    if result.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError(f"split{split:02d} Top20 股票键重复")
    if not result.trade_date.between(*span).all():
        raise ValueError(f"split{split:02d} Top20 日期越过当期模型信号窗")
    for day, group in result.groupby("trade_date"):
        ranks = group["rank"].sort_values().tolist()
        if len(ranks) > 20 or ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"split{split:02d}/{day} Top20 名次不连续或越界")
    result["split_index"] = split
    return result


def classify_plans(
    plans: pd.DataFrame,
    calendar: list[str],
    folds: list[dict[str, Any]],
    topk: pd.DataFrame,
    closure_days: set[str],
    empty_days: set[str],
) -> pd.DataFrame:
    """按到期前时钟查当前模型，仅作命中/未命中/不可判定三态分类。"""
    if calendar != sorted(set(calendar)):
        raise ValueError("SSE 开市日历必须唯一升序")
    windows = sorted(folds, key=lambda item: item["signal_span"][0])
    if not windows:
        raise ValueError("模型信号窗为空")
    for previous, current in zip(windows, windows[1:]):
        if previous["signal_span"][1] >= current["signal_span"][0]:
            raise ValueError("模型信号窗重叠，无法唯一指定当期模型")
    prior = {day: calendar[index - 1] for index, day in enumerate(calendar) if index > 0}
    ranked = {
        (int(split), str(day)): dict(zip(group.ts_code, group["rank"].astype(int)))
        for (split, day), group in topk.groupby(["split_index", "trade_date"])
    }
    rows = []
    for plan_id, row in enumerate(plans.itertuples(index=False)):
        due = date_key(row.due_date)
        signal = date_key(row.signal_date)
        decision = prior.get(due, "")
        budget = float(row.budget)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError(f"plan{plan_id} 原计划预算非法：{budget}")
        candidates = [
            fold
            for fold in windows
            if decision and fold["signal_span"][0] <= decision <= fold["signal_span"][1]
        ]
        current = candidates[0] if candidates else None
        rank = None
        reason = ""
        if not due:
            reason = "missing_due_date"
        elif not decision:
            reason = "calendar_uncovered"
        elif current is None:
            reason = (
                "runoff" if decision > windows[-1]["signal_span"][1] else "outside_model_window"
            )
        elif decision in closure_days:
            reason = "stock_connect_closure"
        elif decision in empty_days:
            reason = "empty_ranking"
        else:
            day_ranks = ranked.get((int(current["split_index"]), decision))
            if day_ranks is None:
                reason = "missing_ranking"
            else:
                rank = day_ranks.get(str(row.ts_code))
        status = "unknown" if reason else ("hit" if rank is not None else "miss")
        rows.append(
            {
                "plan_id": plan_id,
                "ts_code": str(row.ts_code),
                "signal_date": signal,
                "execution_date": date_key(row.execution_date),
                "due_date": due,
                "decision_date": decision,
                "decision_year": (decision or due or signal)[:4],
                "origin_split_index": int(row.split_index),
                "origin_model_version": int(row.model_version),
                "current_split_index": int(current["split_index"]) if current else None,
                "current_model_version": int(current["model_version"]) if current else None,
                "cross_fold": current is not None
                and int(current["split_index"]) != int(row.split_index),
                "tranche_idx": int(row.tranche_idx),
                "initial_plan_budget": budget,
                "rank_within_top20": rank,
                "status": status,
                "unknown_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def ratio(numerator: float, denominator: float) -> float | None:
    """空分母显式返回不可计算，不补零。"""
    return float(numerator / denominator) if denominator else None


def summarize_scope(rows: pd.DataFrame) -> dict[str, Any]:
    """条数与初始计划预算分别给覆盖、已知命中率和未知全幅边界。"""
    hit = rows.status.eq("hit")
    unknown = rows.status.eq("unknown")
    known = ~unknown
    total_budget = float(rows.initial_plan_budget.sum())
    hit_budget = float(rows.loc[hit, "initial_plan_budget"].sum())
    unknown_budget = float(rows.loc[unknown, "initial_plan_budget"].sum())
    known_budget = total_budget - unknown_budget
    return {
        "plans": len(rows),
        "unique_stocks": int(rows.ts_code.nunique()),
        "known_plans": int(known.sum()),
        "known_unique_stocks": int(rows.loc[known, "ts_code"].nunique()),
        "hit_plans": int(hit.sum()),
        "hit_unique_stocks": int(rows.loc[hit, "ts_code"].nunique()),
        "miss_plans": int(rows.status.eq("miss").sum()),
        "unknown_plans": int(unknown.sum()),
        "unknown_unique_stocks": int(rows.loc[unknown, "ts_code"].nunique()),
        "plan_coverage": ratio(float(known.sum()), len(rows)),
        "hit_rate_known_plans": ratio(float(hit.sum()), float(known.sum())),
        "hit_rate_all_plans_lower": ratio(float(hit.sum()), len(rows)),
        "hit_rate_all_plans_upper": ratio(float(hit.sum() + unknown.sum()), len(rows)),
        "total_initial_plan_budget": total_budget,
        "known_initial_plan_budget": known_budget,
        "hit_initial_plan_budget": hit_budget,
        "unknown_initial_plan_budget": unknown_budget,
        "budget_coverage": ratio(known_budget, total_budget),
        "hit_budget_share_known": ratio(hit_budget, known_budget),
        "hit_budget_share_all_lower": ratio(hit_budget, total_budget),
        "hit_budget_share_all_upper": ratio(hit_budget + unknown_budget, total_budget),
        "cross_fold_plans": int(rows.cross_fold.sum()),
        "unknown_reason_counts": rows.loc[unknown, "unknown_reason"].value_counts().to_dict(),
    }


def input_paths(root: Path) -> list[Path]:
    """先固定只读输入集合；不读取原始 scores 或任何未来收益列。"""
    source = root / SOURCE
    paths = [
        source / "targets/plans_v20.parquet",
        source / "fold_registry.json",
        source / "scores.json",
        root / "data/raw/trade_cal.parquet",
    ]
    for split in range(14):
        paths.extend(
            source / "folds" / f"split{split:02d}" / name
            for name in ("fold.json", "signal_topk.parquet")
        )
    return paths


def load_inputs(
    root: Path,
) -> tuple[
    pd.DataFrame, list[str], list[dict[str, Any]], pd.DataFrame, set[str], set[str], dict[str, Any]
]:
    """核验既有 Top20 身份后读取，保留缺排名及历史特征行域限制。"""
    source = root / SOURCE
    registry, scores = read_json(source / "fold_registry.json"), read_json(source / "scores.json")
    folds = registry["folds"]
    if sorted(int(fold["split_index"]) for fold in folds) != list(range(14)):
        raise ValueError("冻结输入必须恰为 split00 至 split13")
    frames, checks = [], []
    for fold in folds:
        split = int(fold["split_index"])
        folder = source / "folds" / f"split{split:02d}"
        metadata = read_json(folder / "fold.json")
        path = folder / "signal_topk.parquet"
        digest = sha256(path)
        if (
            digest != metadata["signal_topk_sha256"]
            or digest != fold["scores"]["signal_topk_sha256"]
        ):
            raise ValueError(f"split{split:02d} Top20 SHA 与原登记不符")
        if int(metadata["model_version"]) != int(fold["model_version"]):
            raise ValueError(f"split{split:02d} 当前模型身份不一致")
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "rank"])
        if len(frame) != int(metadata["signal_topk_rows"]):
            raise ValueError(f"split{split:02d} Top20 行数与原登记不符")
        frame = validate_topk(frame, split, fold["signal_span"])
        frames.append(frame)
        checks.append(
            {
                "split_index": split,
                "topk_rows": len(frame),
                "topk_days": int(frame.trade_date.nunique()),
                "recorded_empty_days": metadata["signal_topk_empty_days"],
                "original_sha_verified": True,
            }
        )
    plans = pd.read_parquet(source / "targets/plans_v20.parquet", columns=PLAN_COLUMNS)
    if plans.empty or plans.duplicated(["signal_date", "tranche_idx", "ts_code"]).any():
        raise ValueError("目标计划为空或计划身份重复")
    cal = pd.read_parquet(
        root / "data/raw/trade_cal.parquet", columns=["exchange", "cal_date", "is_open"]
    )
    calendar = sorted(
        cal.loc[cal.exchange.eq("SSE") & cal.is_open.eq(1), "cal_date"]
        .map(date_key)
        .unique()
        .tolist()
    )
    return (
        plans,
        calendar,
        folds,
        pd.concat(frames, ignore_index=True),
        set(scores["stock_connect_closure_days"]),
        set(scores["empty_ranking_days"]),
        {
            "fold_checks": checks,
            "source_mode": scores["source_mode"],
            "signal_domain": scores["signal_domain"],
        },
    )


def run(root: Path, output: Path, spec: Path) -> dict[str, Any]:
    """从固定输入运行补充诊断，新证据根拒绝覆盖，前后校验身份。"""
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"证据根已存在，拒绝覆盖：{output}")
    spec_hash = sha256(spec)
    if not spec.read_text(encoding="utf-8").strip():
        raise ValueError("补充规格不能为空")
    sources = {path.relative_to(root).as_posix(): sha256(path) for path in input_paths(root)}
    plans, calendar, folds, topk, closures, empty, coverage = load_inputs(root)
    rows = classify_plans(plans, calendar, folds, topk, closures, empty)
    yearly = [
        {"decision_year": str(year), **summarize_scope(group)}
        for year, group in rows.groupby("decision_year", sort=True)
    ]
    result = {
        "schema": "p7_buffer_scope_prescreen/v1",
        "source": SOURCE,
        "spec_sha256": spec_hash,
        "fixed_top_n": 20,
        "target_layer_capital": 700000,
        "overall": summarize_scope(rows),
        "by_decision_year": yearly,
        "source_coverage": coverage,
        "boundaries": [
            "这是机械回转结果之后、交叉读取前冻结的作用面补充，不追认为原始预登记。",
            "T0 是 due_date 前一 SSE 开市日，跨折使用 T0 所属模型；未进 Top20 不生成完整名次。",
            "闭市、空排名、缺排名、模型窗外和末端 runoff 单列不可判定，边界包含全部未知。",
            "budget 是原买入计划逐股分配额；不再乘 weight，不称到期市值权重或成本节省。",
            "T25 为 70 万目标层，可能现金不足，且沿用 cs_train_verify 历史行域，不是新 PIT 证明。",
            "Top20 命中不证明续持优于换股；没有候选收益、成交、公司行为或组合风险路径。",
            "不把不同源旧 A 的费用乘命中占比，不生成 100 万或 200 万账户收益。",
        ],
    }
    if sha256(spec) != spec_hash or any(
        sha256(root / path) != value for path, value in sources.items()
    ):
        raise ValueError("读取期间补充规格或输入身份变化")
    result["analysis_elapsed_seconds"] = time.perf_counter() - started
    if result["analysis_elapsed_seconds"] >= 120:
        raise TimeoutError("作用面补充分析超过 120 秒，停止发布")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "method_snapshot.py")
    shutil.copyfile(spec, output / "screening_spec.md")
    rows.to_csv(output / "plan_scope.csv", index=False, encoding="utf-8")
    pd.DataFrame(yearly).to_csv(output / "yearly_scope.csv", index=False, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = {
        "spec": str(spec),
        "spec_sha256": spec_hash,
        "inputs": sources,
        "method_sha256": sha256(Path(__file__)),
        "outputs": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    """显式指定补充规格与新产物目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(ROOT, args.output, args.spec)
    print(
        json.dumps(
            {
                "plans": result["overall"]["plans"],
                "elapsed_seconds": result["analysis_elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
