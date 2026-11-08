# -*- coding: utf-8 -*-
"""P6 T4 五类指标口径（按证据类分列、单位不互转；设计 §5.5 / §5.7）。

覆盖：production 自然日百分数 + 金额 / research 链式 CAGR+MaxDD+金额+旧 WF TopK /
calibration log-MDD/pp / 五类指标键集与单位闭集（P6-T4-04 的降级部分）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.compare.aggregate import aggregate_run
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics
from src.lazybull.v2.hosts.paper.performance import calculate_annualized_return
from src.lazybull.v2.reporting import generate, trend_contract, trend_csv
from tests import p6_t4_env

#: 该类/该指标的合法单位（None = 该类不生成）
EXPECTED_UNITS = {
    (metric["name"], evidence_class): unit
    for metric in trend_contract.METRICS
    for evidence_class, unit in dict(metric["unit_by_class"]).items()
}


def _rows(result: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """解析已生成 run 的两份 CSV（文本行；不重算任何值）。"""
    index = trend_csv.parse_run_index(p6_t4_env.read_csv_text(result, "run_index.csv"))
    trend = trend_csv.parse_trend(p6_t4_env.read_csv_text(result, "trend.csv"))
    return index, trend


def _metric_rows(trend: list[dict[str, str]], run_key: str) -> dict[str, dict[str, str]]:
    return {
        row["metric_name"]: row
        for row in trend
        if row["run_key"] == run_key
    }


def _value(row: dict[str, str]) -> float:
    return float(row["value"])


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p6_t4_metrics") / "corpus"
    p6_t4_env.build_corpus(root)
    return root


@pytest.fixture(scope="module")
def research(corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    reports = tmp_path_factory.mktemp("p6_t4_metrics_reports")
    return generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="metrics_research")
    )


@pytest.fixture(scope="module")
def production(corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    reports = tmp_path_factory.mktemp("p6_t4_metrics_paper")
    return generate.generate_trend(
        p6_t4_env.production_request(reports, corpus, attempt="metrics_paper")
    )


def test_p6_t4_metrics_production_annualized_and_money(corpus: Path, production: dict) -> None:
    """production：年化=自然日百分数（唯一实现）、双边金额/成本=同窗口求和。"""
    index, trend = _rows(production)
    nav = pd.read_parquet(corpus / "paper" / "nav" / "nav.parquet")
    trades = pd.read_parquet(corpus / "paper" / "trades" / "trades.parquet")
    config = (corpus / "paper" / "config.yaml").read_text(encoding="utf-8")
    assert "initial_capital" in config
    for run in index:
        metrics = _metric_rows(trend, run["run_key"])
        assert set(metrics) == set(trend_contract.METRIC_NAMES)
        expected_nav = nav[nav["trade_date"] == run["window_end"]].iloc[-1]
        expected = calculate_annualized_return(
            500_000.0,
            float(expected_nav["total_value"]),
            run["window_end"],
            index[0]["window_start"],
        )
        assert _value(metrics["annualized_return"]) == pytest.approx(expected, rel=1e-12)
        assert metrics["annualized_return"]["unit"] == "percent_natural_day"
        window = trades[
            (trades["trade_date"] >= run["window_start"])
            & (trades["trade_date"] <= run["window_end"])
        ]
        assert _value(metrics["turnover_amount"]) == pytest.approx(float(window["amount"].sum()))
        assert _value(metrics["cost_amount"]) == pytest.approx(float(window["total_cost"].sum()))
        assert metrics["turnover_amount"]["unit"] == "currency"
        assert metrics["cost_amount"]["unit"] == "currency"
        assert metrics["max_drawdown"]["report_status"] == "not_available"
        assert metrics["max_drawdown"]["degradation_reason_code"] == "no_unique_drawdown_series"
        assert metrics["topk_positive_day_ratio"]["degradation_reason_code"] == "no_source"
        assert run["date_semantics"] == "natural_day"


def test_p6_t4_metrics_research_runs_chain_and_money(corpus: Path, research: dict) -> None:
    """research/runs：年化=链式 CAGR（252 日小数）、回撤=链式 MaxDD、金额按折求和。"""
    index, trend = _rows(research)
    run = next(row for row in index if row["implementation"] == "runs")
    batch = corpus / "runs" / run["run_id"]
    chain = pd.read_parquet(batch / "chain_nav.parquet")
    metrics = calculate_chain_metrics(chain[["date", "nav", "split_index"]])
    rows = _metric_rows(trend, run["run_key"])
    assert rows["annualized_return"]["unit"] == "decimal_252d"
    assert _value(rows["annualized_return"]) == pytest.approx(float(metrics["cagr"]), rel=1e-12)
    assert rows["max_drawdown"]["unit"] == "decimal"
    assert _value(rows["max_drawdown"]) == pytest.approx(
        float(metrics["max_drawdown"]), rel=1e-12
    )
    daily = pd.read_parquet(batch / "folds" / "split0" / "daily.parquet")
    trades = pd.read_parquet(batch / "folds" / "split0" / "trades.parquet")
    assert _value(rows["turnover_amount"]) == pytest.approx(float(daily["turnover_amount"].sum()))
    assert _value(rows["cost_amount"]) == pytest.approx(float(trades["cost"].sum()))
    assert rows["topk_positive_day_ratio"]["report_status"] == "not_available"
    assert (
        rows["topk_positive_day_ratio"]["degradation_reason_code"]
        == "no_unique_implementation"
    )


def test_p6_t4_metrics_research_legacy_topk(corpus: Path, research: dict) -> None:
    """research/旧 WF：TopK 仅字段改名 + 跨折既有均值；无 daily/trades ⇒ 金额 not_available。"""
    index, trend = _rows(research)
    run = next(row for row in index if row["implementation"] == "legacy_wf")
    summary = pd.read_csv(
        corpus / "legacy" / run["run_id"] / "walk_forward_summary_wf_corpus.csv"
    )
    expected = aggregate_run(summary)["key_top20_hit_rate_mean"]
    rows = _metric_rows(trend, run["run_key"])
    assert rows["topk_positive_day_ratio"]["unit"] == "ratio"
    assert _value(rows["topk_positive_day_ratio"]) == pytest.approx(float(expected), rel=1e-12)
    assert rows["turnover_amount"]["report_status"] == "not_available"
    assert rows["cost_amount"]["report_status"] == "not_available"
    assert rows["annualized_return"]["unit"] == "decimal_252d"


def test_p6_t4_metrics_calibration_log_mdd_pp(tmp_path_factory: pytest.TempPathFactory) -> None:
    """calibration：只读 sealed bundle 的 log-MDD 换算 pp；其余四类不生成。"""
    base = tmp_path_factory.mktemp("p6_t4_calibration")
    bundle = p6_t4_env.native_t5_bundle(base / "native")
    report = json.loads((bundle / "drawdown_report.json").read_text(encoding="utf-8"))
    expected = report["sample_sources"]["b0_window"]["mdd_percent"]
    result = generate.generate_trend(
        generate.TrendRequest(
            reports_root=base / "reports",
            evidence_class="calibration_only",
            generated_at=p6_t4_env.FIXED_AT,
            attempt_id="calibration_run",
            input_evidence_classes=("calibration_only",),
            bundles=(str(bundle),),
        )
    )
    index, trend = _rows(result)
    assert len(index) == 1
    run = index[0]
    assert run["evidence_class"] == "calibration_only"
    assert run["date_semantics"] == "fold_local_ordinal"
    assert run["window_start"].startswith("fold")
    rows = _metric_rows(trend, run["run_key"])
    assert rows["max_drawdown"]["unit"] == "pp"
    assert _value(rows["max_drawdown"]) == pytest.approx(float(expected), rel=1e-12)
    for name in ("annualized_return", "turnover_amount", "cost_amount",
                 "topk_positive_day_ratio"):
        assert rows[name]["report_status"] == "not_available"
        assert rows[name]["degradation_reason_code"] == "no_source"
    # 显示名防误读（A-05）：calibration 的 mdd 是 B0 冻结源窗口基准，不是策略回撤
    catalog = {item["name"]: item for item in result["projection"]["metric_catalog"]}
    assert "B0 冻结源窗口基准 MDD" in catalog["max_drawdown"]["note"]
    assert "不是策略回撤" in catalog["max_drawdown"]["note"]


def test_p6_t4_metrics_paper_nav_timestamp_dates_are_normalized() -> None:
    """NAV 的 trade_date 为 Timestamp 时按既有 ``normalize_date`` 归一（不猜日期）。"""
    from src.lazybull.v2.reporting import trend_metrics

    nav = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
            "total_value": [500_000.0, 505_000.0],
        }
    )
    result = trend_metrics.paper_annualized(
        nav,
        run_day="20260102",
        initial_capital=500_000.0,
        account_start_date="20260101",
    )
    assert result["status"]["report_status"] == "available"
    assert result["value"] == pytest.approx(
        calculate_annualized_return(500_000.0, 505_000.0, "20260102", "20260101")
    )
    missing = trend_metrics.paper_annualized(
        nav, run_day="20260103", initial_capital=500_000.0, account_start_date="20260101"
    )
    assert missing["status"]["degradation_reason_code"] == "nav_row_missing"


def test_p6_t4_metrics_units_never_mixed(production: dict, research: dict) -> None:
    """两类面各自五类指标的单位必须落在该类的合法值域内，跨类单位不得出现。"""
    _, prod_trend = _rows(production)
    _, res_trend = _rows(research)
    for material, evidence_class in ((prod_trend, "production"), (res_trend, "research")):
        for row in material:
            expected = EXPECTED_UNITS[(row["metric_name"], evidence_class)]
            if row["value"]:
                # 有值必须带该类单位（不得借用别的类的单位）
                assert row["unit"] == expected
            else:
                # 无值不得写单位（缺值不能伪装成 0 或借用单位）
                assert row["unit"] == ""
            assert row["date_semantics"] == "natural_day"
    assert (
        EXPECTED_UNITS[("annualized_return", "production")]
        != EXPECTED_UNITS[("annualized_return", "research")]
    )
