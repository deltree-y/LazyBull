"""R5 诊断：主口径读数、固定四压力、覆盖与工程错误检查。

四个固定压力（协议 §13.3）只诊断、不择优：
1. 研报信息延后一交易日（用原折模型重算 OOS 特征，不重训）；
2. B 滑点加倍（5bp -> 10bp）；
3. B 现金权益额外 20% 折扣（应收确认/到账层一致下调后重放现金）；
4. 未知永久缺价零回收（只报告影响与未决资产规模）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import (
    dataset,
    evaluation,
    native_replay,
    panel,
    prediction,
    spec,
)

ARMS: tuple[str, ...] = tuple(spec.TARGETS["arms"])
SLEEVES: dict[str, str] = {"ML": "A80_AD20", "SIMPLE": "A80_AD_SIMPLE20"}


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def accounts_root(campaign_dir: Path, accounts_dir: Path | None = None) -> Path:
    """账户根：优先用回放 run 的发布件，缺失才退回 campaign 目录。"""
    if accounts_dir is not None:
        return Path(accounts_dir)
    return Path(campaign_dir) / "accounts"


def load_arms(
    campaign_dir: Path, capital: int, accounts_dir: Path | None = None
) -> pd.DataFrame:
    """读入四臂逐日资产（索引=trade_date）。"""
    path = accounts_root(campaign_dir, accounts_dir) / str(capital) / "arms.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺四臂账：{path}")
    frame = pd.read_parquet(path)
    frame.index = frame.index.astype(str)
    missing = [name for name in ARMS if name not in frame.columns]
    if missing:
        raise ValueError(f"四臂账缺列：{missing}")
    return frame


def _normalize_day_column(frame: pd.DataFrame) -> pd.DataFrame:
    """引擎产物用 `date`（带短横线）；研究层统一补 `trade_date`（YYYYMMDD）。"""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame
    work = frame.copy()
    if "trade_date" not in work.columns:
        if "date" not in work.columns:
            return work
        work["trade_date"] = work["date"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str).str.replace("-", "", regex=False)
    return work


def load_sleeve(
    campaign_dir: Path, capital: int, label: str, accounts_dir: Path | None = None
) -> dict[str, Any]:
    """读入 B 袖逐日账、成交与持仓（统一 `trade_date` 口径）。"""
    base = accounts_root(campaign_dir, accounts_dir) / str(capital)
    daily = pd.read_parquet(base / f"{label}_daily.parquet")
    daily = _normalize_day_column(daily)
    trades_path = base / f"{label}_trades.parquet"
    holdings_path = base / f"{label}_holdings.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    holdings = pd.read_parquet(holdings_path) if holdings_path.exists() else pd.DataFrame()
    return {
        "daily": daily,
        "trades": _normalize_day_column(trades),
        "holdings": _normalize_day_column(holdings),
    }


def engineering_errors(arms: pd.DataFrame, sleeve: dict[str, Any]) -> list[dict[str, Any]]:
    """工程层错误：缺失/非正净值、现金为负、单日持仓超槽、单笔低于 V。"""
    issues: list[dict[str, Any]] = []
    for name in arms.columns:
        series = arms[name].astype(float)
        if series.isna().any():
            issues.append({"arm": name, "issue": "missing_nav", "count": int(series.isna().sum())})
        if (series <= 0).any():
            issues.append(
                {"arm": name, "issue": "nonpositive_nav", "count": int((series <= 0).sum())}
            )
    daily = sleeve["daily"]
    if len(daily):
        if (daily["cash"].astype(float) < -1e-6).any():
            issues.append({"arm": "B", "issue": "negative_cash"})
        over = daily["positions"].astype(int) > int(spec.PORTFOLIO["top_n"])
        if over.any():
            issues.append({"arm": "B", "issue": "positions_over_top_n", "days": int(over.sum())})
    trades = sleeve["trades"]
    if len(trades) and "action" in trades.columns and "amount" in trades.columns:
        buys = trades[trades["action"] == "buy"]
        small = pd.to_numeric(buys["amount"], errors="coerce") < spec.min_ticket_value()
        if int(small.sum()):
            issues.append({"arm": "B", "issue": "below_min_ticket", "trades": int(small.sum())})
    return issues


def industry_concentration(
    holdings: pd.DataFrame, pit: pd.DataFrame, names: dict[str, str]
) -> dict[str, Any]:
    """逐日行业桶最大计数（持仓行业按当日 PIT 近似；引擎侧按买入日冻结）。"""
    if holdings.empty or "trade_date" not in holdings.columns or "ts_code" not in holdings.columns:
        return {"status": "not_estimable", "reason": "无持仓快照"}
    frame = pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    worst = 0
    worst_day = None
    breach_days = 0
    for day, part in holdings.groupby("trade_date"):
        active = frame[(frame["in_date"] <= str(day)) & (frame["out_date"] > str(day))]
        mapping = {
            str(code): names.get(str(l1), "未知行业")
            for code, l1 in zip(active["ts_code"], active["l1_code"])
        }
        buckets: dict[str, int] = {}
        for code in part["ts_code"].astype(str):
            bucket = mapping.get(code, "未知行业")
            buckets[bucket] = buckets.get(bucket, 0) + 1
        if not buckets:
            continue
        top = max(buckets.values())
        if top > worst:
            worst, worst_day = top, str(day)
        if top > int(spec.PORTFOLIO["max_per_industry"]):
            breach_days += 1
    return {
        "status": "pass" if breach_days == 0 else "warn",
        "max_bucket_count": int(worst),
        "max_bucket_day": worst_day,
        "cap": int(spec.PORTFOLIO["max_per_industry"]),
        "breach_days": int(breach_days),
        "note": "按当日 PIT 近似；引擎按买入日冻结，口径差异须在报告中说明",
    }


def a_position_overlap(root: Path, capital: int, holdings: pd.DataFrame) -> dict[str, Any]:
    """B 与 A100 的逐日持仓交集（A 的 `states.json` 为逐日真实持仓）。

    A 账户 `holdings_snapshot.json` 为空（历史产物未记录），故以 `states.json` 为准；
    该文件仅在继承的冻结账户内，缺失即 `not_estimable`，不得用价格或收益倒推。
    """
    path = (
        Path(root)
        / spec.A_BASELINE["relative"]
        / str(capital)
        / "A100"
        / "states.json"
    )
    if not path.exists():
        return {"status": "not_estimable", "reason": "缺 A states.json"}
    if holdings.empty or "trade_date" not in holdings.columns:
        return {"status": "not_estimable", "reason": "无 B 持仓快照"}
    states = json.loads(path.read_text(encoding="utf-8"))
    a_positions = {
        str(item.get("trade_date")): set((item.get("positions") or {}).keys())
        for item in states
        if isinstance(item, dict)
    }
    rows = []
    for day, part in holdings.groupby("trade_date"):
        day = str(day)
        codes = set(part["ts_code"].astype(str))
        a_codes = a_positions.get(day)
        if a_codes is None or not codes:
            continue
        rows.append(
            {
                "trade_date": day,
                "b_count": len(codes),
                "overlap": len(codes & a_codes),
            }
        )
    if not rows:
        return {"status": "not_estimable", "reason": "无可比日期"}
    frame = pd.DataFrame(rows)
    return {
        "status": "pass",
        "days": int(len(frame)),
        "mean_overlap": float(frame["overlap"].mean()),
        "max_overlap": int(frame["overlap"].max()),
        "share_days_with_overlap": float((frame["overlap"] > 0).mean()),
        "mean_b_count": float(frame["b_count"].mean()),
        "source": "A100/states.json（逐日真实持仓）",
    }


def zero_recovery_check(sleeve: dict[str, Any], arms: pd.DataFrame) -> dict[str, Any]:
    """压力 4：缺可核回收证据的持仓（全历史扫描，非仅期末）。

    口径与边界：
    - 分母为 **B 袖自身资产**（`A80_AD20 - A80`），不是 A+B 合并资产；
    - "可核回收"要求持仓有正市价；估值缺失/非正即计入未决；
    - 持仓快照没有逐日陈旧价字段，**无法**区分"停牌沿旧价 carry"与真实报价，
      因此该压力只能给下界，必须在报告中明示（不填 -100%，不假装已覆盖）。
    """
    holdings = sleeve["holdings"]
    if holdings.empty or "trade_date" not in holdings.columns:
        return {"status": "not_estimable", "reason": "无持仓快照"}
    value_column = next(
        (name for name in ("market_value", "value", "amount") if name in holdings.columns), None
    )
    if value_column is None:
        return {"status": "not_estimable", "reason": "持仓快照无估值字段"}
    values = pd.to_numeric(holdings[value_column], errors="coerce")
    unresolved_all = holdings[values.isna() | (values <= 0)]
    last_day = str(arms.index[-1])
    final = holdings[holdings["trade_date"].astype(str) == last_day]
    final_values = pd.to_numeric(final[value_column], errors="coerce") if len(final) else None
    unresolved_final = (
        final[final_values.isna() | (final_values <= 0)] if len(final) else final
    )
    daily = sleeve.get("daily")
    if daily is None or len(daily) == 0 or "total_value" not in daily.columns:
        return {"status": "not_estimable", "reason": "缺袖逐日资产，无法给未决占比"}
    total = float(pd.to_numeric(daily["total_value"], errors="coerce").iloc[-1])
    unresolved_value = float(
        pd.to_numeric(unresolved_all[value_column], errors="coerce").fillna(0).sum()
    )
    return {
        "status": "pass",
        "held_at_end": int(len(final)),
        "unresolved_positions": int(len(unresolved_all)),
        "unresolved_at_end": int(len(unresolved_final)),
        "unresolved_value": unresolved_value,
        "unresolved_share_of_sleeve": unresolved_value / total if total else None,
        "denominator": "B 袖自身逐日资产（sleeve daily total_value）",
        "coverage": "全历史逐日持仓扫描",
        "limitation": "无逐日陈旧价字段，停牌沿旧价 carry 的情形不可辨识，读数只是下界",
        "note": "普通短暂停牌不适用；未知回收不填 -100%，只列零回收压力",
    }


def coverage_diagnostics(dataset_dir: Path, predictions_dir: Path) -> dict[str, Any]:
    """标签状态、逐日评分数量与覆盖。"""
    domain = pd.read_parquet(Path(dataset_dir) / "daily_domain.parquet")
    start = str(spec.WINDOW["start"])
    window = domain[domain["trade_date"].astype(str) >= start]
    payload: dict[str, Any] = {
        "dataset_days": int(len(domain)),
        # 全历史最小值可能落在评估窗之外（如 2015 年流动性危机中几乎全市场涨停/停牌）
        "base_eligible_mean": float(domain["base_eligible_count"].mean()),
        "analyst_eligible_mean": float(domain["analyst_eligible_count"].mean()),
        "analyst_eligible_min": int(domain["analyst_eligible_count"].min()),
        "analyst_eligible_min_date": str(
            domain.loc[domain["analyst_eligible_count"].idxmin(), "trade_date"]
        ),
        "analyst_eligible_min_in_window": int(window["analyst_eligible_count"].min()),
        "days_below_100": int((domain["min100_status"] != "ok").sum()),
        "days_below_100_in_window": int((window["min100_status"] != "ok").sum()),
        "window_start": start,
    }
    counts_path = Path(predictions_dir) / "daily_counts.parquet"
    if counts_path.exists():
        counts = pd.read_parquet(counts_path)
        payload.update(
            {
                "scored_mean": float(counts["scored_count"].mean()),
                "scored_min": int(counts["scored_count"].min()),
                "scored_days_below_100": int((counts["min100_status"] != "ok").sum()),
            }
        )
    return payload


def build_lagged_predictions(
    root: Path,
    campaign_dir: Path,
    *,
    work_dir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """压力 1：研报信息延后一个交易日重建 OOS 特征，并用原折模型重新打分。"""
    from src.lazybull.v2.common.model_registry import ModelRegistry

    emit = _emit(log)
    lag = int(spec.STRESS["report_lag_trading_days"])
    built = dataset.build_dataset(
        root,
        feature_start=str(spec.WINDOW["start"]),
        end=str(spec.WINDOW["end"]),
        out_dir=Path(work_dir) / "lag_dataset",
        source_manifest_sha="stress_report_lag",
        lag_trading_days=lag,
        log=emit,
        write=False,
        keep_frame=True,
    )
    frame = built["frame"]
    manifest = json.loads(
        (Path(campaign_dir) / "models" / "fold_models.json").read_text(encoding="utf-8")
    )
    registry = ModelRegistry(str(Path(campaign_dir) / "models"))
    folds = manifest["folds"]
    days = sorted(frame["trade_date"].astype(str).unique())
    pieces = []
    for index, record in enumerate(folds):
        loaded, _meta = registry.load_model(version=int(record["model_version"]))
        next_c = str(folds[index + 1]["C"]) if index + 1 < len(folds) else None
        oos_days = [
            day
            for day in days
            if day >= str(record["C"]) and (next_c is None or day < next_c)
        ]
        subset = frame[frame["trade_date"].astype(str).isin(set(oos_days))]
        if subset.empty:
            continue
        scores = prediction.predict_matrix(loaded, subset)
        ranked = prediction.rank_frame(subset, scores)
        ranked["fold_id"] = record["fold_id"]
        ranked["model_version"] = int(record["model_version"])
        pieces.append(ranked)
    lagged = pd.concat(pieces, ignore_index=True)
    lagged.to_parquet(Path(work_dir) / "predictions_lag.parquet", index=False)
    emit(f"压力1（研报延后 {lag} 交易日）：{len(lagged)} 行")
    return {
        "predictions": lagged,
        "rows": int(len(lagged)),
        "lag_trading_days": lag,
        "note": "使用原折模型不重训；行情/市值/行业/资格仍取 T",
    }


def replay_sleeve(
    root: Path,
    campaign_dir: Path,
    *,
    capital: int,
    ranked: dict[str, list[tuple[str, float]]],
    slippage: float | None = None,
    cash_discount: float = 0.0,
    tag: str = "stress",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """重放单个 B 账户（压力口径），返回逐日总资产。"""
    emit = _emit(log)
    calendar = panel.load_calendar(root)
    days = native_replay.replay_days(root)
    pit, names = native_replay.load_industry(root)
    book, audit = native_replay.prepare_corporate(
        root, campaign_dir, days, cash_discount=cash_discount, tag=tag
    )
    result = native_replay.run_arm(
        root,
        campaign_dir,
        ranked=ranked,
        days=days,
        calendar=calendar,
        capital=float(capital) * float(spec.TARGETS["initial_b_fraction"]),
        corporate_book=book,
        pit=pit,
        l1_names=names,
        anchor=str(spec.WINDOW["cash_anchor"]),
        slippage=slippage,
        log=emit,
    )
    result["corporate_audit"] = audit
    series = result["daily"].set_index("trade_date")["total_value"].astype(float)
    return {"series": series, "result": result, "audit": audit}


def combine_with_sleeve(arms: pd.DataFrame, capital: int, sleeve: pd.Series) -> pd.DataFrame:
    """A80 + 新袖（A80 = A80_CASH20 - 现金腿），日期必须完全同轴。"""
    b_capital = float(capital) * float(spec.TARGETS["initial_b_fraction"])
    a80 = arms["A80_CASH20"].astype(float) - b_capital
    if not sleeve.index.equals(arms.index):
        raise ValueError("压力袖日期轴与主口径不一致")
    out = arms.copy()
    out["A80_AD20"] = a80 + sleeve
    return out


def _stress_block(arms: pd.DataFrame, capital: int, sleeve: pd.Series) -> dict[str, Any]:
    """单压力下的目标差读数（相对 A100 与现金臂）。"""
    combined = combine_with_sleeve(arms, capital, sleeve)
    series = {name: combined[name].astype(float) for name in combined.columns}
    return {
        "target_vs_a100": evaluation.target_check(
            series["A80_AD20"], series["A100"], series["A80_CASH20"]
        ),
        "target_vs_cash": evaluation.target_check(
            series["A80_AD20"], series["A80_CASH20"], series["A80_CASH20"]
        ),
        "cagr_delta": float(
            evaluation.chain_metrics(series["A80_AD20"])["cagr"]
            - evaluation.chain_metrics(series["A100"])["cagr"]
        ),
        "mdd_loss_delta": float(
            evaluation.chain_metrics(series["A80_AD20"])["mdd_loss"]
            - evaluation.chain_metrics(series["A100"])["mdd_loss"]
        ),
    }


def run_stress(
    root: Path,
    campaign_dir: Path,
    *,
    predictions_dir: Path,
    work_dir: Path,
    accounts_dir: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """四个固定压力（每个独立改变一项）。"""
    emit = _emit(log)
    days = native_replay.replay_days(root)
    main_ranked = prediction.rank_table_lookup(
        pd.read_parquet(Path(predictions_dir) / "predictions_all.parquet",
                        columns=["trade_date", "ts_code", "rank"]),
        days,
    )
    lag = build_lagged_predictions(root, campaign_dir, work_dir=work_dir, log=emit)
    lag_ranked = prediction.rank_table_lookup(
        lag["predictions"][["trade_date", "ts_code", "rank"]], days
    )

    out: dict[str, Any] = {"report_lag_1d": {}, "slippage_doubled": {},
                           "cash_equity_discount_20": {}}
    for capital in spec.TARGETS["capital_scenarios"]:
        capital = int(capital)
        arms = load_arms(campaign_dir, capital, accounts_dir)
        lagged = replay_sleeve(
            root, campaign_dir, capital=capital, ranked=lag_ranked,
            tag=f"stress_lag_{capital}", log=emit
        )
        out["report_lag_1d"][str(capital)] = {
            **_stress_block(arms, capital, lagged["series"]),
            "sleeve_metrics": evaluation.chain_metrics(lagged["series"]),
            "note": lag["note"],
        }
        slippage = replay_sleeve(
            root, campaign_dir, capital=capital, ranked=main_ranked,
            slippage=float(spec.STRESS["slippage_doubled"]),
            tag=f"stress_slippage_{capital}", log=emit
        )
        out["slippage_doubled"][str(capital)] = {
            **_stress_block(arms, capital, slippage["series"]),
            "sleeve_metrics": evaluation.chain_metrics(slippage["series"]),
            "cost": slippage["result"]["diagnostics"]["cost_model"],
        }
        discount = replay_sleeve(
            root, campaign_dir, capital=capital, ranked=main_ranked,
            cash_discount=float(spec.STRESS["cash_equity_discount"]),
            tag=f"stress_cash_discount_{capital}", log=emit
        )
        out["cash_equity_discount_20"][str(capital)] = {
            **_stress_block(arms, capital, discount["series"]),
            "sleeve_metrics": evaluation.chain_metrics(discount["series"]),
            "discount_scope": discount["audit"].get("discount_scope"),
        }
    out["zero_recovery_unresolved"] = zero_recovery_check(
        load_sleeve(
            campaign_dir, int(spec.TARGETS["capital_scenarios"][0]), "ML", accounts_dir
        ),
        load_arms(campaign_dir, int(spec.TARGETS["capital_scenarios"][0]), accounts_dir),
    )
    return out


def run_diagnostics(
    root: Path,
    campaign_dir: Path,
    *,
    dataset_dir: Path,
    predictions_dir: Path,
    work_dir: Path,
    accounts_dir: Path | None = None,
    run_stress_flag: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """R5：主口径读数 + 四压力 + 工程检查。"""
    emit = _emit(log)
    pit, names = native_replay.load_industry(root)
    out: dict[str, Any] = {
        "main": {},
        "stress": {},
        "coverage": coverage_diagnostics(Path(dataset_dir), Path(predictions_dir)),
        "engineering": {},
    }
    for capital in spec.TARGETS["capital_scenarios"]:
        capital = int(capital)
        arms = load_arms(campaign_dir, capital, accounts_dir)
        sleeve_ml = load_sleeve(campaign_dir, capital, "ML", accounts_dir)
        sleeve_simple = load_sleeve(campaign_dir, capital, "SIMPLE", accounts_dir)
        series = {name: arms[name].astype(float) for name in arms.columns}
        out["main"][str(capital)] = {
            "metrics": evaluation.arm_metrics_table(series).to_dict("records"),
            "target_vs_a100": evaluation.target_check(
                series[SLEEVES["ML"]], series["A100"], series["A80_CASH20"]
            ),
            "target_vs_cash": evaluation.target_check(
                series[SLEEVES["ML"]], series["A80_CASH20"], series["A80_CASH20"]
            ),
            "simple_vs_a100": evaluation.target_check(
                series[SLEEVES["SIMPLE"]], series["A100"], series["A80_CASH20"]
            ),
            "segments": evaluation.segment_table(
                series, spec.WINDOW["segments"]
            ).to_dict("records"),
            "years": evaluation.segment_table(
                series, evaluation.year_segments(list(arms.index))
            ).to_dict("records"),
            "correlation": evaluation.correlation_table(series).to_dict(),
            "bootstrap": evaluation.bootstrap_deltas(
                series,
                [
                    ("A100", SLEEVES["ML"]),
                    ("A80_CASH20", SLEEVES["ML"]),
                    ("A100", SLEEVES["SIMPLE"]),
                ],
            ),
            "non_overlap_correlation": {
                "ml_vs_a100": float(
                    evaluation.non_overlap_returns(series[SLEEVES["ML"]]).corr(
                        evaluation.non_overlap_returns(series["A100"])
                    )
                ),
                "simple_vs_ml": float(
                    evaluation.non_overlap_returns(series[SLEEVES["SIMPLE"]]).corr(
                        evaluation.non_overlap_returns(series[SLEEVES["ML"]])
                    )
                ),
            },
            "sleeve": {
                label: {
                    "metrics": evaluation.chain_metrics(
                        sleeve["daily"].set_index("trade_date")["total_value"].astype(float)
                    ),
                    "turnover": evaluation.turnover_and_cost(sleeve["daily"]),
                    "exposure": evaluation.exposure_profile(sleeve["daily"]),
                    "trades": int(len(sleeve["trades"])),
                }
                for label, sleeve in (("ML", sleeve_ml), ("SIMPLE", sleeve_simple))
            },
            "engineering": engineering_errors(arms, sleeve_ml),
            "industry": industry_concentration(sleeve_ml["holdings"], pit, names),
            "a_overlap": a_position_overlap(root, capital, sleeve_ml["holdings"]),
            "zero_recovery": zero_recovery_check(sleeve_ml, arms),
        }
        out["engineering"][str(capital)] = out["main"][str(capital)]["engineering"]

    if run_stress_flag:
        out["stress"] = run_stress(
            root,
            campaign_dir,
            predictions_dir=Path(predictions_dir),
            work_dir=Path(work_dir),
            accounts_dir=accounts_dir,
            log=emit,
        )
    return out
