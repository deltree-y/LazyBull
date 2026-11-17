"""成本感知持仓更新首关：只读既有交易队列，不构造候选净值或更改到期规则。

历史纯 A 成交与 P7 目标计划分开计量。固定 0、1、2—5 交易日桶仅描述卖后重买，
配对不使用盈亏；删除配对股数前后逐单重算最低佣金，得到条件费用余量。
跨日配对忽略期间风险暴露和价差，不是可兑现利润；滑点始终只是冻结模型预算。
复杂度豁免：阶段专用首关诊断，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.common.cost import CostModel  # noqa: E402

RATES = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}
ANCHOR = "data/runs/p2a_b0replay_20261008_144307_d79f3f"
PLANS = "data/paper_shadow/p7_testbed/p7_t25_hk_20261112_r6/targets/plans_v20.parquet"
TRADE_COLUMNS = [
    "split_index",
    "trade_date",
    "signal_date",
    "ts_code",
    "action",
    "price",
    "shares",
    "amount",
    "cost",
    "sell_type",
    "sell_timing",
]
PAIR_COLUMNS = [
    "split_index",
    "ts_code",
    "sell_id",
    "buy_id",
    "sell_date",
    "buy_date",
    "gap_sessions",
    "bucket",
    "matched_shares",
    "sell_type",
    "sell_timing",
    "buy_signal_date",
    "sell_previous_session",
    "buy_signal_known_before_sell",
]


def date_key(value: Any) -> str:
    """日期严格转 YYYYMMDD，不将缺失字段伪造成日期。"""
    text = str(value).replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"日期格式非法：{value!r}")
    pd.Timestamp(text)
    return text


def sha256(path: Path) -> str:
    """计算证据身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def fee_parts(amount: float, action: str) -> dict[str, float]:
    """显式复用成本模型；订单删除至零时不再收最低佣金。"""
    if action not in {"buy", "sell"} or not math.isfinite(amount) or amount < 0:
        raise ValueError(f"费用输入非法：action={action}, amount={amount}")
    if amount == 0:
        return {"commission": 0.0, "stamp_tax": 0.0, "slippage_budget": 0.0}
    model = CostModel(**RATES)
    return {
        "commission": model.calculate_commission(amount),
        "stamp_tax": model.calculate_stamp_tax(amount) if action == "sell" else 0.0,
        "slippage_budget": model.calculate_slippage(amount),
    }


def validate_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """检查结构与模型费用，不加载任何未来盈亏字段。"""
    required = set(TRADE_COLUMNS + ["trade_id", "previous_nav"])
    if not required.issubset(frame):
        raise ValueError(f"成交缺列：{sorted(required - set(frame))}")
    result = frame.copy()
    if result.trade_id.duplicated().any():
        raise ValueError("成交 trade_id 重复")
    for col in ("trade_date", "signal_date"):
        result[col] = result[col].map(date_key)
    if not result.action.isin(["buy", "sell"]).all():
        raise ValueError("成交方向只能为 buy/sell")
    for row in result.itertuples():
        if row.shares <= 0 or int(row.shares) != row.shares:
            raise ValueError(f"成交股数非法：{row.trade_id}")
        if not math.isfinite(row.price) or row.price <= 0:
            raise ValueError(f"成交价格非法：{row.trade_id}")
        if not math.isfinite(row.previous_nav) or row.previous_nav <= 0:
            raise ValueError(f"前值分母非法：{row.trade_id}")
        if not math.isclose(row.shares * row.price, row.amount, rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError(f"成交金额不守恒：{row.trade_id}")
        if not math.isclose(
            sum(fee_parts(row.amount, row.action).values()), row.cost, rel_tol=1e-12, abs_tol=1e-7
        ):
            raise ValueError(f"冻结模型费用与记录不符：{row.trade_id}")
        if row.action == "buy" and row.signal_date >= row.trade_date:
            raise ValueError(f"买入未晚于信号：{row.trade_id}")
    return result


def match_sell_rebuy(trades: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """同折同股按时间 FIFO 消耗股数；未知同日时序不算机械往返。

    同日仅接受 open 卖出与后续买入的诊断配对。旧卖出行 signal_date 是原买入
    信号，不可当卖出触发日，因此可知性只比较新买信号与卖出前一交易会话。
    """
    ordinal = {date_key(day): i for i, day in enumerate(calendar)}
    if len(ordinal) != len(calendar) or list(ordinal) != sorted(ordinal):
        raise ValueError("交易日历须唯一且升序")
    rows: list[dict[str, Any]] = []
    for (fold, code), group in trades.groupby(["split_index", "ts_code"], sort=True):
        queue: deque[dict[str, Any]] = deque()
        ordered = group.assign(action_order=group.action.map({"sell": 0, "buy": 1}))
        ordered = ordered.sort_values(["trade_date", "action_order", "trade_id"])
        for row in ordered.to_dict("records"):
            day = row["trade_date"]
            if day not in ordinal:
                raise ValueError(f"成交日不在交易日历：{day}")
            if row["action"] == "sell":
                queue.append({"row": row, "remaining": int(row["shares"])})
                continue
            left = int(row["shares"])
            for pending in queue:
                sell = pending["row"]
                gap = ordinal[day] - ordinal[sell["trade_date"]]
                if gap > 5 or not pending["remaining"]:
                    continue
                if gap == 0 and sell["sell_timing"] != "open":
                    continue
                quantity = min(left, pending["remaining"])
                if not quantity:
                    break
                previous = (
                    calendar[ordinal[sell["trade_date"]] - 1]
                    if ordinal[sell["trade_date"]]
                    else None
                )
                rows.append(
                    {
                        "split_index": int(fold),
                        "ts_code": code,
                        "sell_id": sell["trade_id"],
                        "buy_id": row["trade_id"],
                        "sell_date": sell["trade_date"],
                        "buy_date": day,
                        "gap_sessions": gap,
                        "bucket": (
                            "same_day"
                            if gap == 0
                            else "one_day" if gap == 1 else "two_to_five_days"
                        ),
                        "matched_shares": quantity,
                        "sell_type": sell["sell_type"],
                        "sell_timing": sell["sell_timing"],
                        "buy_signal_date": row["signal_date"],
                        "sell_previous_session": previous,
                        "buy_signal_known_before_sell": previous is not None
                        and row["signal_date"] <= previous,
                    }
                )
                pending["remaining"] -= quantity
                left -= quantity
            while queue and (
                ordinal[day] - ordinal[queue[0]["row"]["trade_date"]] > 5
                or not queue[0]["remaining"]
            ):
                queue.popleft()
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def deletion_costs(trades: pd.DataFrame, pairs: pd.DataFrame, scale: int = 1) -> pd.DataFrame:
    """按订单聚合删除股数后重算费用；部分删除不得按比例虚构最低佣金节省。"""
    if scale not in (1, 2):
        raise ValueError("仅冻结的 100 万/200 万队列尺度")
    removed: dict[str, int] = {}
    known_ids = set(trades.trade_id)
    for pair in pairs.itertuples():
        if pair.matched_shares <= 0 or int(pair.matched_shares) != pair.matched_shares:
            raise ValueError("配对股数须为正整数")
        for key in (pair.sell_id, pair.buy_id):
            if key not in known_ids:
                raise ValueError(f"配对引用未知成交：{key}")
            removed[key] = removed.get(key, 0) + int(pair.matched_shares)
    rows = []
    for trade in trades.itertuples():
        quantity = removed.get(trade.trade_id, 0)
        if quantity > trade.shares:
            raise ValueError(f"配对超出原成交股数：{trade.trade_id}")
        before = fee_parts(trade.amount * scale, trade.action)
        after = fee_parts((trade.shares - quantity) * scale * trade.price, trade.action)
        item = {
            "trade_id": trade.trade_id,
            "trade_date": trade.trade_date,
            "split_index": int(trade.split_index),
            "removed_shares": quantity * scale,
            "previous_nav": trade.previous_nav * scale,
        }
        for key in before:
            item[f"base_{key}"] = before[key]
            item[f"saved_{key}"] = before[key] - after[key]
        rows.append(item)
    return pd.DataFrame(rows)


def cost_summary(rows: pd.DataFrame, intervals: int, capital: float) -> dict[str, Any]:
    """费用/年与费用占前日资产的算术年化，不计算 CAGR 或候选收益。"""
    if intervals <= 0 or capital <= 0:
        raise ValueError("计量间隔/资本须为正")
    result: dict[str, Any] = {"capital_scenario": capital, "effective_intervals": intervals}
    for prefix in ("base", "saved"):
        components = {}
        for part in ("commission", "stamp_tax", "slippage_budget"):
            series = rows[f"{prefix}_{part}"]
            components[part] = {
                "yuan": float(series.sum()),
                "yuan_per_252_sessions": float(series.sum() * 252 / intervals),
                "initial_capital_pp_per_252_sessions": float(
                    series.sum() * 252 / intervals / capital * 100
                ),
                "previous_nav_pp_per_252_sessions": float(
                    (series / rows.previous_nav).sum() * 252 / intervals * 100
                ),
            }
        result[prefix] = components
    base_total = sum(item["yuan"] for item in result["base"].values())
    saved_total = sum(item["yuan"] for item in result["saved"].values())
    result["saved_fraction_of_model_cost"] = saved_total / base_total if base_total else 0.0
    result["not_account_replay_or_cagr_gain"] = True
    return result


def plan_continuation(plans: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """目标层同股相邻计划与前计划到期日的关系，不当成实际成交。"""
    ordinal = {day: i for i, day in enumerate(calendar)}
    rows = []
    for code, group in plans.groupby("ts_code", sort=True):
        previous = None
        for row in group.sort_values(["execution_date", "signal_date", "tranche_idx"]).itertuples():
            gap = None
            if previous is not None:
                if row.execution_date not in ordinal or previous.due_date not in ordinal:
                    raise ValueError("计划执行/到期日期未被日历覆盖")
                gap = ordinal[row.execution_date] - ordinal[previous.due_date]
            label = "new_or_gap_gt5"
            if gap is not None and gap <= 5:
                label = (
                    "overlap_existing"
                    if gap < 0
                    else (
                        "due_same_day"
                        if gap == 0
                        else "due_one_day_ago" if gap == 1 else "due_two_to_five_days_ago"
                    )
                )
            rows.append(
                {
                    "ts_code": code,
                    "signal_date": row.signal_date,
                    "execution_date": row.execution_date,
                    "due_date": row.due_date,
                    "previous_due_date": previous.due_date if previous else None,
                    "gap_sessions": gap,
                    "relationship": label,
                    "cross_fold": previous is not None and previous.split_index != row.split_index,
                    "planned_overlap_shares": (
                        min(row.shares, previous.shares)
                        if previous and gap is not None and gap <= 5
                        else 0
                    ),
                    "not_actual_trade": True,
                }
            )
            previous = row
    return pd.DataFrame(rows)


def input_paths(root: Path) -> list[Path]:
    """在读取分析字段前固定完整输入文件集合。"""
    anchor = root / ANCHOR
    folds = sorted((anchor / "folds").glob("split*"))
    if len(folds) != 14:
        raise ValueError(f"冻结纯 A 应为 14 折，实际 {len(folds)}")
    paths = [anchor / "batch_meta.json", root / "data/raw/trade_cal.parquet", root / PLANS]
    for fold in folds:
        paths.extend(
            fold / name for name in ("trades.parquet", "daily.parquet", "attribution.parquet")
        )
    return paths


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int, dict[str, Any]]:
    """仅读取费用、时间、身份及前值分母列，禁止导入收益标签。"""
    anchor = root / ANCHOR
    meta_path = anchor / "batch_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("config", {}).get("bt_initial_capital") != 1_000_000:
        raise ValueError("纯 A 初资元信息与冻结 100 万场景不符")
    calendar_path = root / "data/raw/trade_cal.parquet"
    cal = pd.read_parquet(calendar_path, columns=["exchange", "cal_date", "is_open"])
    calendar = sorted(
        cal.loc[cal.exchange.eq("SSE") & cal.is_open.eq(1), "cal_date"]
        .map(date_key)
        .unique()
        .tolist()
    )
    frames = []
    intents = []
    intervals = 0
    folds = sorted((anchor / "folds").glob("split*"))
    if len(folds) != 14:
        raise ValueError(f"冻结纯 A 应为 14 折，实际 {len(folds)}")
    for fold in folds:
        trade_path, daily_path = fold / "trades.parquet", fold / "daily.parquet"
        frame = pd.read_parquet(trade_path, columns=TRADE_COLUMNS)
        daily = pd.read_parquet(daily_path, columns=["trade_date", "total_value"])
        daily["trade_date"] = daily.trade_date.map(date_key)
        if daily.trade_date.duplicated().any() or not daily.trade_date.is_monotonic_increasing:
            raise ValueError(f"{fold.name} 每日序列非唯一升序")
        previous = daily.set_index("trade_date").total_value.shift(1, fill_value=1_000_000.0)
        frame["trade_id"] = [f"{fold.name}:{index}" for index in range(len(frame))]
        frame["previous_nav"] = frame.trade_date.map(previous)
        frames.append(frame)
        intervals += len(daily) - 1
        intent = pd.read_parquet(
            fold / "attribution.parquet",
            columns=[
                "split_index",
                "signal_date",
                "execution_date",
                "planned_ts_code",
                "actual_ts_code",
                "status",
                "execution_stage",
            ],
        )
        intents.append(intent)
    plan_path = root / PLANS
    plans = pd.read_parquet(
        plan_path,
        columns=[
            "ts_code",
            "signal_date",
            "execution_date",
            "due_date",
            "tranche_idx",
            "split_index",
            "shares",
        ],
    )
    for col in ("signal_date", "execution_date", "due_date"):
        plans[col] = plans[col].map(date_key)
    combined = validate_trades(pd.concat(frames, ignore_index=True))
    intent_frame = pd.concat(intents, ignore_index=True)
    filled = intent_frame.loc[intent_frame.status.eq("filled")]
    recorded_keys = (
        combined.loc[combined.action.eq("buy")]
        .groupby(["split_index", "trade_date", "ts_code"])
        .size()
    )
    filled_keys = filled.groupby(["split_index", "execution_date", "actual_ts_code"]).size()
    unmatched = recorded_keys.subtract(filled_keys, fill_value=0).abs()
    intent_summary = {
        "rows": len(intent_frame),
        "status_counts": intent_frame.status.value_counts(dropna=False).to_dict(),
        "execution_stage_counts": intent_frame.execution_stage.value_counts(dropna=False).to_dict(),
        "filled_rows": len(filled),
        "filled_vs_buy_key_absolute_count_difference": int(unmatched.sum()),
        "not_full_ranked_candidate_sequence": True,
    }
    return combined, plans, calendar, intervals, intent_summary


def run(root: Path, output: Path, spec: Path) -> dict[str, Any]:
    """运行冻结首关；新根独占创建，不覆盖历史证据。"""
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"证据根已存在，拒绝覆盖：{output}")
    spec_hash = sha256(spec)
    if not spec.read_text(encoding="utf-8").strip():
        raise ValueError("冻结规格不能为空")
    sources = input_paths(root)
    source_hashes = {path.relative_to(root).as_posix(): sha256(path) for path in sources}
    trades, plans, calendar, intervals, intent_summary = load_inputs(root)
    pairs = match_sell_rebuy(trades, calendar)
    plan_rows = plan_continuation(plans, calendar)
    scenarios = []
    detail_tables = {}
    for scale in (1, 2):
        for horizon in (0, 1, 5):
            for scope in ("all_reasons", "nonrisk_known_intent"):
                subset = pairs.loc[pairs.gap_sessions.le(horizon)]
                if scope == "nonrisk_known_intent":
                    subset = subset.loc[
                        subset.sell_type.isin(["holding_period", "rebalance"])
                        & subset.buy_signal_known_before_sell
                    ]
                costs = deletion_costs(trades, subset, scale)
                item = cost_summary(costs, intervals, 1_000_000.0 * scale)
                item.update(
                    {
                        "max_gap_sessions": horizon,
                        "scope": scope,
                        "matched_pairs": len(subset),
                        "matched_shares": int(subset.matched_shares.sum()) * scale,
                    }
                )
                scenarios.append(item)
                detail_tables[f"costs_{scale}m_gap{horizon}_{scope}.csv"] = costs
    result = {
        "schema": "p7_improvement_cost_prescreen/v1",
        "spec_sha256": spec_hash,
        "source": ANCHOR,
        "trade_rows": len(trades),
        "buy_rows": int(trades.action.eq("buy").sum()),
        "sell_rows": int(trades.action.eq("sell").sum()),
        "cost_settings": RATES,
        "effective_intervals": intervals,
        "model_cost_recorded": float(trades.cost.sum()),
        "pair_counts_by_bucket": pairs.groupby("bucket").size().to_dict(),
        "pair_counts_by_sell_type": pairs.groupby("sell_type", dropna=False).size().to_dict(),
        "anchor_intents": intent_summary,
        "scenarios": scenarios,
        "plan_layer_only": {
            "source": PLANS,
            "initial_capital": 700000,
            "rows": len(plans),
            "relationship_counts": plan_rows.groupby("relationship").size().to_dict(),
            "not_cash_feasible_account": True,
        },
        "boundaries": [
            "旧纯 A 历史模拟成交与 P7 70 万目标计划分列；不是券商实际成交。",
            "0/1/2—5 交易日是预先固定诊断，不选择最优窗口或改持有期。",
            "同日 open 字段允许收盘降级，仍未认证真实同步撮合；机械上界有条件。",
            "跨日删除往返意味着改变期间暴露，未计价差/机会成本/公司行为，不是可执行利润。",
            "滑点只是 5bp 模型预算，佣金与印税沿冻结历史研究口径，不重估真实收费。",
            "100/200 万为订单股数及原前值轨迹×1/×2的费用情景；没有账户重放或 CAGR 增益。",
            "前日 NAV 算术年化仅费用强度；不读取候选收益或优化参数。",
            "现有跨袖净额化不作为新增节省；本源为单袖直通，原同目标保留已反映在成交队列。",
            "卖出 signal_date 是原持仓买入信号；可知性只用重买信号<=卖出前一会话代理。",
        ],
        "analysis_elapsed_seconds": time.perf_counter() - started,
    }
    if result["analysis_elapsed_seconds"] >= 120:
        raise TimeoutError("首关费用分析超出 120 秒，停止发布")
    if sha256(spec) != spec_hash or any(
        sha256(root / p) != digest for p, digest in source_hashes.items()
    ):
        raise ValueError("读取期间规格或输入身份变化")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "method_snapshot.py")
    shutil.copyfile(spec, output / "screening_spec.md")
    pairs.to_csv(output / "sell_rebuy_pairs.csv", index=False, encoding="utf-8")
    plan_rows.to_csv(output / "plan_continuation.csv", index=False, encoding="utf-8")
    for name, frame in detail_tables.items():
        frame.to_csv(output / name, index=False, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = {
        "spec": str(spec),
        "spec_sha256": spec_hash,
        "inputs": source_hashes,
        "method_sha256": sha256(Path(__file__)),
        "outputs": {p.name: sha256(p) for p in sorted(output.iterdir())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    """命令行只接受显式规格与新证据根。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(ROOT, args.output, args.spec)
    print(
        json.dumps(
            {
                "trade_rows": result["trade_rows"],
                "elapsed_seconds": result["analysis_elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
