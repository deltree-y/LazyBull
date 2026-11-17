"""共同风险首关：固定规格的样本内诊断，不生成策略收益或交易。

复杂度豁免：阶段专用首关诊断，不入生产链路（CLAUDE.md §7.8）。
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

import numpy as np
import pandas as pd
from scipy.optimize import minimize

WINDOW = 120
MIN_COMMON = 90
CORRELATION_THRESHOLD = 0.6
ROOT = Path(__file__).resolve().parents[2]
OLD_ACCOUNT = Path("data/runs/p2a_b0replay_20261008_144307_d79f3f")
CURRENT_TARGET = Path("data/paper_shadow/p7_testbed/p7_t25_hk_20261112_r6")
FROZEN_INPUT = Path("data/frozen_reference/p2a_b0_input_manifest_20261008_withconfig.json")
DEFAULT_OUTPUT = Path("data/reports/p7_improvement_prescreen_20261116/risk")


def sha256(path: Path) -> str:
    """记录实际读取文件身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simple_returns(prices: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """先对齐完整交易日历，再算简单收益，缺价与停牌缺口不填补。"""
    aligned = prices.reindex(calendar)
    aligned = aligned.where(np.isfinite(aligned) & (aligned > 0))
    return aligned.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def past_window(returns: pd.DataFrame, observation: str) -> pd.DataFrame:
    """截至观察日的最近120个交易日；调用方必须传完整日历轴。"""
    return returns.loc[returns.index <= observation].tail(WINDOW)


def common_factor_metrics(returns: np.ndarray, weights: np.ndarray, market: np.ndarray) -> dict:
    """含截距OLS的共同因子分解；方差口径均为样本方差。"""
    portfolio = returns @ weights
    y = portfolio - portfolio.mean()
    x = market - market.mean()
    denominator = float(x @ x)
    total = float(y @ y)
    if denominator <= 0 or total <= 0:
        raise ValueError("共同因子或组合方差为零，不能估计beta及方差份额")
    beta = float(x @ y / denominator)
    residual = y - beta * x
    return {
        "market_beta": beta,
        "common_factor_variance_share": float(beta * beta * denominator / total),
        "residual_variance_share": float(residual @ residual / total),
        "market_variance": denominator / (len(x) - 1),
        "portfolio_variance": total / (len(x) - 1),
    }


def bounded_minimum_variance(covariance: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """固定50%对角收缩矩阵由调用方传入；只解一次有界最小方差。"""
    scale = float(np.trace(covariance))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("协方差迹非正或非有限，无法优化")
    normalized = covariance / scale
    result = minimize(
        lambda w: float(w @ normalized @ w),
        weights.copy(),
        jac=lambda w: 2 * normalized @ w,
        method="SLSQP",
        bounds=list(zip(0.5 * weights, 1.5 * weights)),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(len(w))},
        options={"ftol": 1e-12, "maxiter": 300},
    )
    if not result.success:
        raise ValueError(f"有界最小方差未收敛：{result.message}")
    fitted = np.asarray(result.x, dtype=float)
    if (
        not np.all(np.isfinite(fitted))
        or abs(fitted.sum() - 1) > 1e-8
        or np.any(fitted < 0.5 * weights - 1e-8)
        or np.any(fitted > 1.5 * weights + 1e-8)
    ):
        raise ValueError("有界最小方差结果违反预算或单股界限")
    return fitted


def correlation_graph(
    correlation: np.ndarray, weights: np.ndarray, codes: list[str], industries: list[str | None]
) -> tuple[dict, list[dict], list[dict]]:
    """相关连通簇及行业解释；未知行业不计为同业或跨业。"""
    parent = list(range(len(codes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges: list[dict] = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            rho = float(correlation[i, j])
            if rho < CORRELATION_THRESHOLD:
                continue
            parent[find(j)] = find(i)
            known = industries[i] is not None and industries[j] is not None
            cross = None if not known else industries[i] != industries[j]
            edges.append(
                {"left": codes[i], "right": codes[j], "correlation": rho, "cross_l1": cross}
            )
    components: dict[int, list[int]] = {}
    for i in range(len(codes)):
        components.setdefault(find(i), []).append(i)
    clusters = [
        {
            "members": "|".join(codes[i] for i in members),
            "stock_count": len(members),
            "weight": float(weights[members].sum()),
        }
        for members in components.values()
    ]
    known_edges = [edge for edge in edges if edge["cross_l1"] is not None]
    cross_count = sum(bool(edge["cross_l1"]) for edge in known_edges)
    return (
        {
            "high_correlation_edges": len(edges),
            "known_industry_edges": len(known_edges),
            "unknown_industry_edges": len(edges) - len(known_edges),
            "cross_l1_edges": cross_count,
            "cross_l1_edge_share": cross_count / len(known_edges) if known_edges else None,
            "largest_cluster_weight": max(cluster["weight"] for cluster in clusters),
            "largest_multistock_cluster_weight": max(
                (cluster["weight"] for cluster in clusters if cluster["stock_count"] > 1),
                default=0.0,
            ),
            "cluster_count": len(clusters),
        },
        edges,
        clusters,
    )


def industry_at(pit: pd.DataFrame, codes: list[str], day: str) -> list[str | None]:
    """按左闭右开区间连接一级行业；冲突或缺失均保持未知。"""
    active = pit.loc[(pit["in_date"] <= day) & (day < pit["out_date"])]
    mapping: dict[str, str | None] = {}
    for code, rows in active.loc[active["ts_code"].isin(codes)].groupby("ts_code"):
        values = rows["l1_code"].dropna().astype(str)
        values = values.loc[values.str.len() > 0].unique()
        mapping[str(code)] = str(values[0]) if len(values) == 1 else None
    return [mapping.get(code) for code in codes]


def diagnose(
    observation: dict, returns: pd.DataFrame, market: pd.Series, pit: pd.DataFrame
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """一组固定名单诊断；缺数据不删股，不把拟合风险变成策略收益。"""
    codes = observation["codes"]
    weights = np.asarray(observation["weights"], dtype=float)
    weights = weights / weights.sum()
    day = observation["trade_date"]
    result = {k: v for k, v in observation.items() if k not in {"codes", "weights"}}
    result.update({"stock_count": len(codes), "status": "not_estimable", "reason": ""})
    window = past_window(returns.reindex(columns=codes), day)
    result["window_start"] = str(window.index.min())
    result["window_end"] = str(window.index.max())
    result["window_days"] = len(window)
    result["missing_stock_days"] = int(window.isna().sum().sum())
    result["missing_returns_by_stock"] = json.dumps(
        {str(code): int(count) for code, count in window.isna().sum().items() if count},
        ensure_ascii=False,
    )
    stock_complete = window.notna().all(axis=1)
    result["stock_common_days"] = int(stock_complete.sum())
    sample = window.loc[stock_complete]
    if len(sample) < MIN_COMMON:
        result["reason"] = "common_days_below_90"
        return result, [], [], []
    market_sample = market.reindex(sample.index)
    valid_market = market_sample.notna()
    result["market_common_days"] = int(valid_market.sum())
    if not valid_market.all():
        result["reason"] = "market_factor_missing_on_common_days"
        return result, [], [], []
    covariance = np.atleast_2d(np.cov(sample.to_numpy(), rowvar=False, ddof=1))
    variance = np.diag(covariance)
    if np.any(variance <= 0) or not np.all(np.isfinite(covariance)):
        result["reason"] = "nonpositive_or_invalid_stock_variance"
        return result, [], [], []
    correlation = covariance / np.sqrt(np.outer(variance, variance))
    shrink = 0.5 * covariance + 0.5 * np.diag(variance)
    try:
        fitted = bounded_minimum_variance(shrink, weights)
        factor = common_factor_metrics(sample.to_numpy(), weights, market_sample.to_numpy())
    except ValueError as exc:
        result["reason"] = str(exc)
        return result, [], [], []
    baseline_raw = float(weights @ covariance @ weights)
    baseline_shrink = float(weights @ shrink @ weights)
    fitted_raw = float(fitted @ covariance @ fitted)
    fitted_shrink = float(fitted @ shrink @ fitted)
    risk_contributions = weights * (covariance @ weights) / baseline_raw
    equal = np.full(len(weights), 1 / len(weights))
    pair_weights = np.outer(weights, weights)
    pair_mask = np.triu(np.ones_like(pair_weights, dtype=bool), k=1)
    known = industry_at(pit, codes, day)
    graph, edges, clusters = correlation_graph(correlation, weights, codes, known)
    result.update(factor)
    result.update(graph)
    result.update(
        {
            "status": "estimated",
            "known_industry_weight": float(sum(w for w, code in zip(weights, known) if code)),
            "known_industry_stocks": sum(code is not None for code in known),
            "weighted_average_correlation": (
                float(
                    np.sum(pair_weights[pair_mask] * correlation[pair_mask])
                    / pair_weights[pair_mask].sum()
                )
                if len(weights) > 1
                else None
            ),
            "top5_risk_contribution_share": float(np.sort(risk_contributions)[-5:].sum()),
            "baseline_raw_variance": baseline_raw,
            "fitted_raw_variance": fitted_raw,
            "raw_variance_reduction": 1 - fitted_raw / baseline_raw,
            "raw_volatility_reduction": 1 - math.sqrt(fitted_raw / baseline_raw),
            "baseline_shrink_variance": baseline_shrink,
            "fitted_shrink_variance": fitted_shrink,
            "shrink_variance_reduction": 1 - fitted_shrink / baseline_shrink,
            "shrink_volatility_reduction": 1 - math.sqrt(fitted_shrink / baseline_shrink),
            "equal_weight_raw_variance": float(equal @ covariance @ equal),
            "equal_weight_raw_volatility_reduction": 1
            - math.sqrt(float(equal @ covariance @ equal) / baseline_raw),
            "weight_l1_change": float(np.abs(fitted - weights).sum()),
            "one_way_weight_change": float(np.abs(fitted - weights).sum() / 2),
        }
    )
    stock_rows = [
        {
            "ts_code": code,
            "l1_code": known[i],
            "original_weight": float(weights[i]),
            "fitted_weight": float(fitted[i]),
            "risk_contribution_share": float(risk_contributions[i]),
        }
        for i, code in enumerate(codes)
    ]
    return result, stock_rows, edges, clusters


def load_observations(root: Path, record: Any) -> list[dict]:
    """纯A旧账和当前目标层分别取冻结观察日，不混并账户。"""
    observations = []
    for path in sorted((root / OLD_ACCOUNT / "folds").glob("split*/holdings_snapshot.parquet")):
        record(path)
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "market_value"])
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["month"] = frame["trade_date"].str[:6]
        for _, month in frame.groupby("month"):
            day = str(month["trade_date"].max())
            selected = (
                month.loc[month["trade_date"] == day].groupby("ts_code")["market_value"].sum()
            )
            if not np.all(np.isfinite(selected)) or np.any(selected <= 0):
                raise ValueError(f"历史持仓存在非法市值：{path} / {day}")
            observations.append(
                {
                    "group": "historical_pure_a",
                    "fold": path.parent.name,
                    "trade_date": day,
                    "observation_id": f"historical_{path.parent.name}_{day}",
                    "codes": selected.index.astype(str).tolist(),
                    "weights": selected.tolist(),
                }
            )
    if not observations:
        raise ValueError(f"未找到历史纯A持仓：{root / OLD_ACCOUNT}")
    record(root / OLD_ACCOUNT / "batch_meta.json")
    path = root / CURRENT_TARGET / "targets/plans_v20.parquet"
    record(path)
    frame = pd.read_parquet(path, columns=["signal_date", "ts_code", "weight"])
    for day, selected in frame.groupby("signal_date"):
        weights = selected.groupby("ts_code")["weight"].sum()
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(f"当前计划存在非法权重：{day}")
        observations.append(
            {
                "group": "current_a_intentions",
                "fold": "",
                "trade_date": str(day),
                "observation_id": f"current_{day}",
                "codes": weights.index.astype(str).tolist(),
                "weights": weights.tolist(),
            }
        )
    record(root / CURRENT_TARGET / "targets.json")
    record(root / CURRENT_TARGET / "input_identity.json")
    return observations


def load_prices(
    root: Path, observations: list[dict], record: Any
) -> tuple[pd.DataFrame, pd.Series, list[dict], dict]:
    """读取全clean截面形成共同因子；全日历留洞，不压缩缺失日。"""
    calendar_path = root / "data/raw/trade_cal.parquet"
    record(calendar_path)
    calendar_frame = pd.read_parquet(calendar_path)
    calendar = sorted(
        set(calendar_frame.loc[calendar_frame["is_open"] == 1, "cal_date"].astype(str))
    )
    first = min(row["trade_date"] for row in observations)
    last = max(row["trade_date"] for row in observations)
    first_index = calendar.index(first)
    start_index = max(0, first_index - WINDOW)
    end_index = calendar.index(last) + 1
    calendar = calendar[start_index:end_index]
    series = []
    coverage = []
    for day in calendar:
        filename = f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        path = root / "data/clean/daily" / filename
        if not path.exists():
            coverage.append({"trade_date": day, "status": "missing_partition", "valid_prices": 0})
            continue
        record(path)
        frame = pd.read_parquet(
            path, columns=["ts_code", "trade_date", "close_adj", "is_suspended"]
        )
        if frame["ts_code"].duplicated().any() or not frame["trade_date"].astype(str).eq(day).all():
            raise ValueError(f"行情主键重复或分区日期不一致：{path}")
        prices = frame.set_index("ts_code")["close_adj"]
        suspended = frame.set_index("ts_code")["is_suspended"].eq(1)
        prices = prices.where((prices > 0) & np.isfinite(prices) & ~suspended)
        prices.name = day
        series.append(prices)
        coverage.append(
            {"trade_date": day, "status": "read", "valid_prices": int(prices.notna().sum())}
        )
    if not series:
        raise ValueError("未读取到任何本地clean行情分区")
    prices = pd.DataFrame(series).reindex(calendar)
    returns = simple_returns(prices, calendar)
    market = returns.mean(axis=1, skipna=True)
    for row in coverage:
        day = row["trade_date"]
        row["market_valid_returns"] = int(returns.loc[day].notna().sum())
        row["market_return"] = None if pd.isna(market.loc[day]) else float(market.loc[day])
    details = {
        "calendar_start": calendar[0],
        "calendar_end": calendar[-1],
        "calendar_days": len(calendar),
        "all_market_codes": len(returns.columns),
    }
    return returns, market, coverage, details


def summarize(rows: list[dict]) -> dict:
    """分别汇总观察日分布，未估计组不进入分位数。"""
    frame = pd.DataFrame(rows)
    fields = [
        "weighted_average_correlation",
        "top5_risk_contribution_share",
        "largest_cluster_weight",
        "largest_multistock_cluster_weight",
        "cross_l1_edge_share",
        "market_beta",
        "common_factor_variance_share",
        "raw_volatility_reduction",
        "shrink_volatility_reduction",
        "raw_variance_reduction",
        "shrink_variance_reduction",
        "one_way_weight_change",
        "equal_weight_raw_volatility_reduction",
        "known_industry_weight",
        "stock_common_days",
    ]
    result = {}
    for group, all_rows in frame.groupby("group"):
        estimated = all_rows.loc[all_rows["status"] == "estimated"]
        distributions = {}
        for field in fields:
            values = pd.to_numeric(estimated.get(field, pd.Series(dtype=float))).dropna()
            distributions[field] = (
                {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "p10": float(values.quantile(0.1)),
                    "median": float(values.median()),
                    "p90": float(values.quantile(0.9)),
                }
                if len(values)
                else {"count": 0}
            )
        result[str(group)] = {
            "observations": len(all_rows),
            "estimated": len(estimated),
            "not_estimable": len(all_rows) - len(estimated),
            "reasons": all_rows.loc[all_rows["status"] != "estimated", "reason"]
            .value_counts()
            .to_dict(),
            "distributions": distributions,
        }
    return result


def run(root: Path, output: Path, spec: Path) -> dict:
    """运行一次冻结首关并保存来源、排除、明细与方法快照。"""
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{output}")
    output.mkdir(parents=True)
    frozen_path = root / FROZEN_INPUT
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["files"]
    inputs: dict[str, dict] = {}

    def record(path: Path) -> None:
        relative = path.relative_to(root).as_posix()
        if relative in inputs:
            return
        digest = sha256(path)
        frozen_key = relative.removeprefix("data/")
        old = frozen.get(frozen_key)
        inputs[relative] = {
            "path": relative,
            "sha256": digest,
            "size": path.stat().st_size,
            "frozen_match": None if old is None else digest == old["sha256"],
            "frozen_reference": "not_registered" if old is None else frozen_key,
        }

    record(frozen_path)
    observations = load_observations(root, record)
    returns, market, price_coverage, price_details = load_prices(root, observations, record)
    pit_path = root / "data/clean/sw_industry_pit.parquet"
    record(pit_path)
    pit = pd.read_parquet(pit_path, columns=["ts_code", "in_date", "out_date", "l1_code"])
    for col in ["in_date", "out_date"]:
        pit[col] = pit[col].fillna("99991231" if col == "out_date" else "").astype(str)
    rows, stock_rows, edges, clusters = [], [], [], []
    for observation in observations:
        result, stocks, graph_edges, graph_clusters = diagnose(observation, returns, market, pit)
        rows.append(result)
        key = {
            "observation_id": observation["observation_id"],
            "group": observation["group"],
            "trade_date": observation["trade_date"],
        }
        stock_rows.extend({**key, **row} for row in stocks)
        edges.extend({**key, **row} for row in graph_edges)
        clusters.extend({**key, **row} for row in graph_clusters)
    summary = summarize(rows)
    for name, content in [
        ("observations", rows),
        ("stock_weights", stock_rows),
        ("high_correlation_edges", edges),
        ("clusters", clusters),
        ("price_coverage", price_coverage),
        ("exclusions", [row for row in rows if row["status"] != "estimated"]),
    ]:
        pd.DataFrame(content).to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    methods = output / "methods"
    methods.mkdir()
    for relative in [
        Path(__file__).resolve().relative_to(root),
        Path("tests/test_p7_improvement_prescreen_risk.py"),
        spec.relative_to(root),
    ]:
        record(root / relative)
        shutil.copy2(root / relative, methods / relative.name)
    manifest = {
        "schema": "p7_improvement_prescreen_risk/v1",
        "inputs": list(inputs.values()),
        "price_details": price_details,
        "elapsed_seconds": time.perf_counter() - started,
        "observations": len(rows),
        "alpha_consumed": False,
        "historical_account_source": OLD_ACCOUNT.as_posix(),
        "current_intention_source": CURRENT_TARGET.as_posix(),
        "risk_inputs_same_state_as_historical_account": "not_claimed",
        "limitations": [
            "历史账户与当前风险输入分别登记；冻结匹配只覆盖登记过的文件。",
            "当前意图为70万元目标层新买批次，不是实际持仓或完整经济账。",
            "收益仅用于截至观察日的协方差，不生成候选收益、成交或MDD。",
            "拟合风险下降为样本内读数，不等于未来或最大回撤改善。",
            "全clean等权共同因子是解释性代理；缺价及停牌缺口不填补。",
            "相关簇是连通分量，不保证同簇任意两股均达到相关阈值。",
            "前五风险贡献使用有符号Euler方差贡献；可能因负贡献超过100%。",
        ],
    }
    for name, payload in [("summary", summary), ("manifest", manifest)]:
        (output / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    return {
        "output": str(output),
        "observations": len(rows),
        "estimated": sum(row["status"] == "estimated" for row in rows),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }


def main() -> None:
    """只读本地输入，默认输出拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "docs/experiments/p7_improvement_prescreen_20261116.md",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / DEFAULT_OUTPUT
    print(json.dumps(run(root, output, args.spec.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
