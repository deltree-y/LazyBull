# -*- coding: utf-8 -*-
"""P6 T4 输入/来源边界（Reviewer3 R3-01~R3-08）。

production 权威门、诊断旁路排除、金额来源绑定与窗口自洽、父/子来源根去重、
previous 五件完整性、计时新 run 负载、语料 LF 字节。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p6 import build_t4_corpus
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.reporting import generate, trend_csv, trend_sources
from tests import p6_t4_env


def _rows(result: dict) -> tuple[list[dict], list[dict]]:
    index = trend_csv.parse_run_index(p6_t4_env.read_csv_text(result, "run_index.csv"))
    trend = trend_csv.parse_trend(p6_t4_env.read_csv_text(result, "trend.csv"))
    return index, trend


def _metric(trend: list[dict], run_key: str, name: str) -> dict:
    return next(
        row for row in trend if row["run_key"] == run_key and row["metric_name"] == name
    )


def test_p6_t4_r3_production_requires_authority(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """缺 P4 权威代（engine_state）⇒ production 不得输出 available 年化（R3-01）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_authority")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    (corpus / "paper" / "state" / "engine_state.bin").unlink()
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="no_authority")
    )
    index, trend = _rows(result)
    assert {row["report_status"] for row in index} == {"degraded"}
    assert {row["degradation_reason_code"] for row in index} == {"authority_missing"}
    annual = [_metric(trend, row["run_key"], "annualized_return") for row in index]
    assert {row["report_status"] for row in annual} == {"not_available"}
    assert {row["degradation_reason_code"] for row in annual} == {"authority_missing"}


def test_p6_t4_r3_production_run_kind_conflict_is_invalid(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """run_kind/engine 与权威相位冲突 ⇒ 该 run invalid（不因公式一致而放行，R3-01）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_conflict")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    target = corpus / "paper" / "runs" / "t1_20260102.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.update({"engine": "legacy", "run_kind": "t0"})
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="conflict")
    )
    index, _ = _rows(result)
    bad = next(row for row in index if row["run_id"].startswith("t1_20260102"))
    assert bad["report_status"] == "invalid"
    assert bad["degradation_reason_code"] == "authority_mismatch"


def test_p6_t4_r3_paper_diagnostics_are_not_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """runs/ 下 readiness/gap 诊断旁路不得被当作交易 run（R3-02）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_diagnostics")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    runs = corpus / "paper" / "runs"
    for name, payload in (
        ("readiness_20260103.json", {"date": "20260103", "ok": True, "gaps": []}),
        ("gap_20260102.json", {"date": "20260102", "ok": False, "gaps": ["x"]}),
    ):
        (runs / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="diagnostics")
    )
    index, trend = _rows(result)
    assert len(index) == 3
    assert all(row["report_status"] == "available" for row in index)
    annual = [row for row in trend if row["metric_name"] == "annualized_return"]
    assert all(row["report_status"] == "available" for row in annual)


def test_p6_t4_r3_money_binding_tracks_consumed_files(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """金额行绑定实际消费的逐折文件：文件变化 ⇒ source_sha256 变化（R3-03）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_money_binding")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    first = generate.generate_trend(
        p6_t4_env.research_request(
            root / "reports_a", corpus, attempt="money_a", sources=(f"runs:{corpus / 'runs'}",)
        )
    )
    index_a, trend_a = _rows(first)
    run_key = index_a[0]["run_key"]
    before = _metric(trend_a, run_key, "turnover_amount")
    assert before["source_field"] == "folds/*/daily.parquet:turnover_amount"
    assert before["source_ref"].endswith("p6t4_run_0000")
    daily = corpus / "runs" / "p6t4_run_0000" / "folds" / "split0" / "daily.parquet"
    frame = pd.read_parquet(daily)
    frame["turnover_amount"] = frame["turnover_amount"] * 2
    frame.to_parquet(daily, index=False)
    second = generate.generate_trend(
        p6_t4_env.research_request(
            root / "reports_b", corpus, attempt="money_b", sources=(f"runs:{corpus / 'runs'}",)
        )
    )
    index_b, trend_b = _rows(second)
    after = _metric(trend_b, index_b[0]["run_key"], "turnover_amount")
    assert float(after["value"]) == pytest.approx(float(before["value"]) * 2)
    assert after["source_sha256"] != before["source_sha256"]


def test_p6_t4_r3_money_window_is_filtered(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """金额必须与行窗口自洽：只对闭区间内的行求和（R3-04）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_window")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    daily_path = corpus / "runs" / "p6t4_run_0000" / "folds" / "split0" / "daily.parquet"
    daily = pd.read_parquet(daily_path)
    window = sorted(daily["trade_date"].astype(str))[3:7]
    expected = float(
        daily[daily["trade_date"].astype(str).isin(window)]["turnover_amount"].sum()
    )
    summary_path = corpus / "runs" / "p6t4_run_0000" / "summary.parquet"
    summary = pd.read_parquet(summary_path)
    summary.loc[0, "test_start"] = window[0]
    summary.loc[0, "test_end"] = window[-1]
    summary.to_parquet(summary_path, index=False)
    result = generate.generate_trend(
        p6_t4_env.research_request(
            root / "reports", corpus, attempt="window", sources=(f"runs:{corpus / 'runs'}",)
        )
    )
    index, trend = _rows(result)
    row = _metric(trend, index[0]["run_key"], "turnover_amount")
    assert row["window_start"] == window[0]
    assert row["window_end"] == window[-1]
    assert float(row["value"]) == pytest.approx(expected)


def test_p6_t4_r3_parent_child_sources_rejected(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """父集合根 + 直接批次根同时输入：同一物理 run 不得重复（R3-05）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_overlap")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    child = corpus / "runs" / "p6t4_run_0000"
    with pytest.raises(trend_sources.TrendSourceError, match="同一物理 run"):
        generate.generate_trend(
            p6_t4_env.research_request(
                root / "reports",
                corpus,
                attempt="overlap",
                sources=(f"runs:{corpus / 'runs'}", f"runs:{child}"),
            )
        )


def test_p6_t4_r3_previous_requires_complete_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """previous 五件缺一/被改写必须拒绝继承（R3-06）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_previous")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    reports = root / "reports"
    previous_dir = Path(
        generate.generate_trend(
            p6_t4_env.research_request(reports, corpus, attempt="prev_full", publish=True)
        )["report_path"]
    )
    (previous_dir / "report.html").unlink()
    with pytest.raises(trend_sources.TrendSourceError, match="上一快照缺 report.html"):
        generate.generate_trend(
            p6_t4_env.research_request(
                reports,
                corpus,
                attempt="prev_missing",
                previous=previous_dir,
                generated_at=p6_t4_env.FIXED_AT,
            )
        )
    # 篡改（而非缺失）必须按字节不符拒绝
    tampered = root / "prev_tampered"
    shutil.copytree(previous_dir, tampered)
    (tampered / "report.html").write_text("corrupt\n", encoding="utf-8", newline="\n")
    with pytest.raises(trend_sources.TrendSourceError, match="report.html"):
        generate.generate_trend(
            p6_t4_env.research_request(
                reports,
                corpus,
                attempt="prev_tampered",
                previous=tampered,
                generated_at=p6_t4_env.FIXED_AT,
            )
        )


def test_p6_t4_r3_corpus_new_run_carries_profile_load(tmp_path: Path) -> None:
    """标准/压力档的计时新 run 必须携带设计负载（R3-07）。"""
    for profile in ("standard", "stress"):
        load = dict(build_t4_corpus.PROFILE_LOAD[profile])
        batch = tmp_path / profile / "runs" / "p6t4_run_0000"
        build_t4_corpus._build_run(batch.parent.parent, 0, load)
        chain = pd.read_parquet(batch / "chain_nav.parquet")
        summary = pd.read_parquet(batch / "summary.parquet")
        folds = sorted(path for path in (batch / "folds").iterdir() if path.is_dir())
        trades = sum(len(pd.read_parquet(fold / "trades.parquet")) for fold in folds)
        assert len(chain) == load["chain_rows"]
        assert len(folds) == load["folds"] == len(summary)
        assert trades == load["trades"]
        # 折界重复点必须让 loader 读入成功（R3-07-R1）
        loaded = load_runs_batch(batch)
        assert len(loaded.folds) == load["folds"]


def test_p6_t4_r3_history_mixed_phases_allowed(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """历史 t0 记录不得因当前权威相位是 t1 被判冲突（R3-01-R1）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_mixed_phase")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    t0 = corpus / "paper" / "runs" / "t0_20260101.json"
    t0.write_text(
        json.dumps(
            {"trade_date": "20260101", "run_kind": "t0", "engine": "v2"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="mixed_phase")
    )
    index, _ = _rows(result)
    row = next(item for item in index if item["run_id"].startswith("t0_20260101"))
    assert row["report_status"] == "available"


def test_p6_t4_r3_same_day_other_phase_idempotent_record_allowed(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """同日另一相位的幂等补写记录（t1 权威 + t0 recovery）不得判冲突（R3-01-R2）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_same_day")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    same_day = corpus / "paper" / "runs" / "t0_20260103.json"
    same_day.write_text(
        json.dumps(
            {"trade_date": "20260103", "run_kind": "t0", "engine": "v2"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="same_day")
    )
    index, _ = _rows(result)
    by_id = {row["run_id"]: row for row in index}
    assert by_id["t1_20260103"]["report_status"] == "available"
    assert by_id["t0_20260103"]["report_status"] == "available"


def test_p6_t4_r3_middle_bad_nav_keeps_later_point(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """中间坏 NAV 只断该点，后续已核对点仍出数值（区级 degraded 不吞点）(R3-N1-R1)。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_middle_bad")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    nav_path = corpus / "paper" / "nav" / "nav.parquet"
    nav = pd.read_parquet(nav_path)
    nav.loc[nav["trade_date"] == "20260102", "total_value"] += 1.0
    nav.to_parquet(nav_path, index=False)
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="middle_bad")
    )
    index, trend = _rows(result)
    by_day = {row["run_id"]: row for row in index}
    assert by_day["t1_20260101"]["report_status"] == "available"
    assert by_day["t1_20260102"]["report_status"] == "invalid"
    later = by_day["t1_20260103"]
    assert later["report_status"] == "degraded"
    row = _metric(trend, later["run_key"], "annualized_return")
    assert row["report_status"] == "degraded"
    assert row["value"] not in (None, "")


def test_p6_t4_r3_missing_authority_record_degrades_all(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """当前权威代记录缺失：不得由同日其它相位顶替，整档降级（R3-01-R2 关联）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_missing_auth_record")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    (corpus / "paper" / "runs" / "t1_20260103.json").unlink()
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="missing_auth")
    )
    index, _ = _rows(result)
    assert {row["report_status"] for row in index} == {"degraded"}
    assert {row["degradation_reason_code"] for row in index} == {"authority_missing"}


def test_p6_t4_r3_zero_runs_with_authority_is_not_empty_valid(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """权威存在但交易记录全缺：零行也不得报“合法空”（R3-01-R3）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_zero_runs")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    for path in (corpus / "paper" / "runs").glob("t*_*.json"):
        path.unlink()
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="zero_runs")
    )
    projection = result["projection"]
    assert projection["counts"]["run_total"] == 0
    assert projection["status"]["report_status"] == "degraded"
    assert projection["status"]["degradation_reason_code"] == "authority_missing"
    inputs = [row.get("path") for row in result["manifest"]["inputs"]]
    assert any(str(item).endswith("state/engine_state.bin") for item in inputs)
    reasons = {item.get("reason_code") for item in projection["degradations"]}
    assert "authority_missing" in reasons


def test_p6_t4_r3_annualization_failure_keeps_invalid_and_empty_unit(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """区级 degraded 不得覆盖年化函数自身的失败：空值/空单位/具体原因（R3-N1-R2）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_annual_fail")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    nav_path = corpus / "paper" / "nav" / "nav.parquet"
    nav = pd.read_parquet(nav_path)
    nav.loc[nav["trade_date"] == "20260102", "total_value"] += 1.0
    nav.to_parquet(nav_path, index=False)
    config = corpus / "paper" / "config.yaml"
    config.write_text(
        "engine: v2\ninitial_capital: 0.0\naccount_start_date: '20260101'\n",
        encoding="utf-8",
        newline="\n",
    )
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="annual_fail")
    )
    index, trend = _rows(result)
    later = next(row for row in index if row["run_id"].startswith("t1_20260103"))
    row = _metric(trend, later["run_key"], "annualized_return")
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "annualization_input_invalid"
    assert row["value"] == "" and row["unit"] == ""


def test_p6_t4_r3_all_degraded_values_not_reported_as_status_only(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """有值但区级降级时，汇总/页面不得宣称“全部为状态行”（R3-N1-R3）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_all_degraded")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    nav_path = corpus / "paper" / "nav" / "nav.parquet"
    nav = pd.read_parquet(nav_path)
    nav.loc[nav["trade_date"] == "20260101", "total_value"] += 1.0
    nav.to_parquet(nav_path, index=False)
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="all_degraded")
    )
    summary = next(
        item
        for item in result["projection"]["metrics"]
        if item["name"] == "annualized_return"
    )
    assert summary["usable_rows"] == 0
    assert summary["with_value_rows"] == 2
    assert summary["status_rows"] == 1
    html = result["files"]["report.html"].decode("utf-8")
    assert "2 / 0 / 1" in html
    assert "3 行全部为状态行" not in html


def test_p6_t4_r3_bad_last_nav_keeps_earlier_history(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """末日 N A V 坏点只让该点失效，不反向抹去此前已核对历史（R3-N1）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_bad_last")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    nav_path = corpus / "paper" / "nav" / "nav.parquet"
    nav = pd.read_parquet(nav_path)
    nav.loc[nav["trade_date"] == "20260103", "total_value"] += 1.0
    nav.to_parquet(nav_path, index=False)
    result = generate.generate_trend(
        p6_t4_env.production_request(root / "reports", corpus, attempt="bad_last")
    )
    index, trend = _rows(result)
    by_day = {row["run_id"]: row for row in index}
    assert by_day["t1_20260101"]["report_status"] == "available"
    assert by_day["t1_20260102"]["report_status"] == "available"
    assert by_day["t1_20260103"]["report_status"] == "invalid"


def test_p6_t4_r3_direct_run_root_money_binding(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """直接批次根（相对路径为空）金额也必须绑定实际文件，不得为空集合摘要（R3-03-R1）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_direct_root")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    batch = corpus / "runs" / "p6t4_run_0000"
    first = generate.generate_trend(
        p6_t4_env.research_request(
            root / "reports_a", corpus, attempt="direct_a", sources=(f"runs:{batch}",)
        )
    )
    index_a, trend_a = _rows(first)
    before = _metric(trend_a, index_a[0]["run_key"], "turnover_amount")
    daily = batch / "folds" / "split0" / "daily.parquet"
    frame = pd.read_parquet(daily)
    frame["turnover_amount"] = frame["turnover_amount"] * 2
    frame.to_parquet(daily, index=False)
    second = generate.generate_trend(
        p6_t4_env.research_request(
            root / "reports_b", corpus, attempt="direct_b", sources=(f"runs:{batch}",)
        )
    )
    index_b, trend_b = _rows(second)
    after = _metric(trend_b, index_b[0]["run_key"], "turnover_amount")
    assert float(after["value"]) == pytest.approx(float(before["value"]) * 2)
    assert after["source_sha256"] != before["source_sha256"]


def test_p6_t4_r3_multi_fold_uses_each_fold_window(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """多折金额按各折窗口过滤，不套全批外包窗口（R3-04-R1）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_multifold")
    corpus = root / "corpus"
    batch = corpus / "runs" / "p6t4_run_0000"
    build_t4_corpus._build_run(corpus, 0, {"chain_rows": 24, "folds": 2, "trades": 4})
    chain = pd.read_parquet(batch / "chain_nav.parquet")
    summary = pd.read_parquet(batch / "summary.parquet")
    # 第 0 折窗口缩成单日；第 1 折保持原窗口（全批 min/max 不变）
    summary.loc[summary["split_index"] == 0, "test_end"] = summary.loc[
        summary["split_index"] == 0, "test_start"
    ]
    summary.to_parquet(batch / "summary.parquet", index=False)
    expected_turnover = 0.0
    expected_cost = 0.0
    for split in (0, 1):
        fold_dir = batch / "folds" / f"split{split}"
        daily = pd.read_parquet(fold_dir / "daily.parquet")
        trades = pd.read_parquet(fold_dir / "trades.parquet")
        window = summary[summary["split_index"] == split]
        start = str(window["test_start"].iloc[0])
        end = str(window["test_end"].iloc[0])
        expected_turnover += float(
            daily[
                (daily["trade_date"].astype(str) >= start)
                & (daily["trade_date"].astype(str) <= end)
            ]["turnover_amount"].sum()
        )
        expected_cost += float(
            trades[
                (trades["trade_date"].astype(str) >= start)
                & (trades["trade_date"].astype(str) <= end)
            ]["cost"].sum()
        )
    assert len(chain) == 24
    result = generate.generate_trend(
        p6_t4_env.research_request(
            root / "reports", corpus, attempt="multifold", sources=(f"runs:{corpus / 'runs'}",)
        )
    )
    index, trend = _rows(result)
    row = _metric(trend, index[0]["run_key"], "turnover_amount")
    cost = _metric(trend, index[0]["run_key"], "cost_amount")
    assert float(row["value"]) == pytest.approx(expected_turnover)
    assert float(cost["value"]) == pytest.approx(expected_cost)


def test_p6_t4_r3_unknown_dates_degrade_money() -> None:
    """消费行存在不可解析日期 ⇒ 金额整项降级，不得当成窗内无成交（R3-N2）。"""
    from src.lazybull.v2.reporting import trend_observe

    frame = pd.DataFrame({"trade_date": [None], "cost": [99.0]})
    result = trend_observe._fold_money(
        [(0, frame)],
        "cost",
        {0: ("20240101", "20240102")},
        source_ref="src:run",
        source_sha="a" * 64,
        source_field="folds/*/trades.parquet:cost",
    )
    assert result["status"]["report_status"] == "not_available"
    assert result["status"]["degradation_reason_code"] == "missing_precondition"


def test_p6_t4_r3_legacy_parent_child_sources_rejected(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """无 manifest 旧 WF 的父/子根同样不得重复收录（R3-05-R1）。"""
    root = tmp_path_factory.mktemp("p6_t4_r3_legacy_overlap")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    child = corpus / "legacy" / "wf_0000"
    with pytest.raises(trend_sources.TrendSourceError, match="同一物理 run"):
        generate.generate_trend(
            p6_t4_env.research_request(
                root / "reports",
                corpus,
                attempt="legacy_overlap",
                sources=(
                    f"legacy_wf:{corpus / 'legacy'}",
                    f"legacy_wf:{child}",
                ),
            )
        )


def test_p6_t4_r3_corpus_text_is_lf(tmp_path: Path) -> None:
    """语料文本一律 LF（Linux 重建保持同一身份，R3-08）。"""
    build_t4_corpus.build(tmp_path / "corpus", "tiny")
    for path in (tmp_path / "corpus").rglob("*"):
        if path.suffix in (".json", ".yaml", ".csv"):
            assert b"\r" not in path.read_bytes(), path
