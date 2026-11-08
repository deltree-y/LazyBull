# -*- coding: utf-8 -*-
"""P6 T4 排序 / 500 可见行 / 1,200 点抽样（设计 §4.3 / §5.7）。

P6-T4-01：2,000 run 的完整索引 vs 页面最近 500 行（``visible_rows=min(500,total)``）；
P6-T4-02：抽样只影响图，**不改变任何 CSV 行键/值**；
排序：``(window_end, run_id, run_key)`` 稳定升序，页面取末尾 500 行。
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from src.lazybull.v2.reporting import generate, render, svg, trend_contract, trend_csv
from src.lazybull.v2.reporting import trend_projection, trend_svg, trend_view
from tests import p6_t4_env

_GENERATOR = {
    "project_version": "test",
    "python": "3.12",
    "jinja2": "3.1.6",
    "git_commit": "test",
    "git_dirty": False,
}


def _projection(index_rows: list[dict[str, Any]], trend_rows: list[dict[str, Any]]) -> dict:
    return trend_projection.build_trend_projection(
        evidence_class="research",
        generated_at=p6_t4_env.FIXED_AT,
        root_map={"src0": "合成来源（只读）"},
        index_rows=index_rows,
        trend_rows=trend_rows,
        observations=[],
        previous=None,
    )


def test_p6_t4_01_full_2000_vs_visible_500() -> None:
    """2,000 run：CSV 恒为 2,000 行；页面/投影只见最近 500 行且 index_mode=visible。"""
    index_rows, trend_rows = p6_t4_env.synthetic_rows(2_000)
    bundle = trend_csv.build_trend_bundle(index_rows, trend_rows, expected_index_rows=2_000)
    assert bundle["schemas"]["run_index.csv"]["row_count"] == 2_000
    assert bundle["schemas"]["trend.csv"]["row_count"] == 10_000
    projection = _projection(index_rows, trend_rows)
    assert projection["index_mode"] == "full"
    assert projection["visible"]["index_mode"] == "visible"
    assert projection["visible"]["total_rows"] == 2_000
    assert projection["visible"]["visible_rows"] == 500
    assert len(projection["runs"]) == 500
    view = trend_view.build_trend_view(
        projection,
        report_id="p6_trend_20261106T090000Z_0123456789ab",
        generated_at=p6_t4_env.FIXED_AT,
        input_evidence_classes=("research",),
        generator=_GENERATOR,
    )
    html = render.render_report_bytes(view, report_kind="trend").decode("utf-8")
    assert len(html.encode("utf-8")) <= 3_000_000
    assert "index_mode" in html
    visible_ids = {str(row["run_id"]) for row in projection["runs"]}
    assert "run_01999" in visible_ids
    assert "run_00000" not in visible_ids
    assert "run_01500" in visible_ids  # 最近 500 行 = 1500..1999（升序末尾）
    assert "run_01499" not in visible_ids


def test_p6_t4_01b_empty_index_is_legal_empty() -> None:
    """总数 0：合法空（``empty_valid``，visible_rows=0），不伪造行。"""
    projection = _projection([], [])
    assert projection["visible"]["visible_rows"] == 0
    assert projection["runs"] == []
    assert projection["status"]["report_status"] == "empty_valid"
    bundle = trend_csv.build_trend_bundle([], [])
    assert bundle["schemas"]["run_index.csv"]["row_count"] == 0
    assert bundle["schemas"]["trend.csv"]["row_count"] == 0


def test_p6_t4_02_sampling_does_not_change_csv() -> None:
    """图抽样 ≤1,200 点且保留首尾/极值；CSV 行键/值与抽样前**逐字节一致**。"""
    values = [0.0] * 2_000
    values[777] = 99.0
    values[1_234] = -99.0
    kept = svg.downsample_indices(values, limit=trend_contract.MAX_CHART_POINTS)
    assert len(kept) <= trend_contract.MAX_CHART_POINTS
    assert {0, 1_999, 777, 1_234} <= set(kept)
    points = [
        {"run_id": f"run_{index:05d}", "window_end": f"2024{index % 12:02d}01",
         "value": value}
        for index, value in enumerate(values)
    ]
    chart = trend_svg.metric_chart(points, caption="年化", unit_label="小数")
    coordinates = re.findall(r'points="([^"]*)"', chart)
    drawn = sum(len(item.split()) for item in coordinates)
    assert 0 < drawn <= trend_contract.MAX_CHART_POINTS
    index_rows, trend_rows = p6_t4_env.synthetic_rows(2_000)
    before = trend_csv.build_trend_bundle(index_rows, trend_rows)
    projection = _projection(index_rows, trend_rows)
    view = trend_view.build_trend_view(
        projection,
        report_id="p6_trend_20261106T090000Z_0123456789ab",
        generated_at=p6_t4_env.FIXED_AT,
        input_evidence_classes=("research",),
        generator=_GENERATOR,
    )
    render.render_report_bytes(view, report_kind="trend")
    after = trend_csv.build_trend_bundle(index_rows, trend_rows)
    assert before["digests"] == after["digests"]
    assert before["files"] == after["files"]
    # 图数据面 = 可见 500 行（不是 2,000），但 CSV 仍是 2,000 / 10,000 行
    assert len(projection["series"]["annualized_return"]) == 500
    assert after["schemas"]["run_index.csv"]["row_count"] == 2_000


def test_p6_t4_order_state_row_value_is_labelled_not_missing() -> None:
    """最近点表对不可用行的 value 显示「不可用」，不写「（字段缺席）」（T4-R2-A2-04）。"""
    index_rows, trend_rows = p6_t4_env.synthetic_rows(2)
    for row in trend_rows:
        if row["metric_name"] == "annualized_return":
            row["value"] = None
            row["unit"] = None
            row["report_status"] = "not_available"
            row["degradation_reason_code"] = "no_source"
    projection = _projection(index_rows, trend_rows)
    view = trend_view.build_trend_view(
        projection,
        report_id="p6_trend_20261106T090000Z_0123456789ab",
        generated_at=p6_t4_env.FIXED_AT,
        input_evidence_classes=("research",),
        generator=_GENERATOR,
    )
    html = render.render_report_bytes(view, report_kind="trend").decode("utf-8")
    # 两条年化行不可用：最近点表两格都应显示「不可用」，而不是「（字段缺席）」
    assert html.count("不可用（见状态列）") == 2


def test_p6_t4_order_is_window_end_run_id_run_key() -> None:
    """排序键 ``(window_end, run_id, run_key)``；页面取升序末尾 500 行。"""
    rows = [
        {"window_end": "20240102", "run_id": "b", "run_key": "2"},
        {"window_end": "20240101", "run_id": "z", "run_key": "1"},
        {"window_end": "20240102", "run_id": "a", "run_key": "3"},
        {"window_end": "20240102", "run_id": "a", "run_key": "0"},
    ]
    ordered = trend_csv.sort_index_rows(rows)
    assert [row["run_key"] for row in ordered] == ["1", "0", "3", "2"]
    visible = trend_projection.visible_rows(rows, limit=2)
    assert [row["run_key"] for row in visible] == ["3", "2"]


def test_p6_t4_04_full_build_render_has_charts_and_sampling_note(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """全量构建同样必须出图；抽样上限文案为 ``min(1,200, 可见行)``（A-04）。"""
    root = tmp_path_factory.mktemp("p6_t4_full_render")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    result = generate.generate_trend(
        p6_t4_env.research_request(root / "reports", corpus, attempt="render_full")
    )
    html = result["files"]["report.html"]
    # 仅有链式来源的两类指标可绘；金额类在该 tiny 语料缺逐折来源 ⇒ 合法空图（不补 0）
    assert html.count(b"<polyline") >= 2
    assert html.count('aria-label="chart-empty"'.encode("utf-8")) >= 1
    assert b"min(1200, \xe5\x8f\xaf\xe8\xa7\x81\xe8\xa1\x8c)" in html
