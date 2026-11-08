# -*- coding: utf-8 -*-
"""P6 T4 历史不覆盖与增量生成（设计 §4.3 / §9 / §10）。

同 ``report_id`` 再发布必须拒绝；上一快照字节不变；上一快照 + 新 run 增量生成时
历史行逐字节保留（不重算、不删除）。
"""

from __future__ import annotations

import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from src.lazybull import PROJECT_ROOT
from src.lazybull.v2.reporting import generate, publish, trend_csv, trend_sources
from tests import p6_t4_env


def _digest_tree(root: Path) -> dict[str, str]:
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_p6_t4_history_incremental_and_no_overwrite(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """上一快照 + 新 run：索引 +1、历史行逐字节不变、同 ID 再发布拒绝、坏快照拒绝继承。"""
    root = tmp_path_factory.mktemp("p6_t4_history")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=2)
    reports = root / "reports"
    first = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="history_a", publish=True)
    )
    previous_dir = Path(first["report_path"])
    assert len(trend_csv.parse_run_index(
        p6_t4_env.read_csv_text(first, "run_index.csv")
    )) == 3  # 2 runs + 1 legacy
    before = _digest_tree(previous_dir)
    # 同输入同 generated_at 再发布同一 report_id ⇒ 拒绝覆盖
    with pytest.raises(publish.PublishError):
        generate.generate_trend(
            p6_t4_env.research_request(reports, corpus, attempt="history_dup", publish=True)
        )
    p6_t4_env.build_corpus(corpus, upto=3)
    # 只把 runs 根作为来源：legacy 根的旧行（及其 root_id）必须按快照继承（A-02 场景）
    second = generate.generate_trend(
        p6_t4_env.research_request(
            reports,
            corpus,
            attempt="history_b",
            sources=(f"runs:{corpus / 'runs'}",),
            previous=previous_dir,
            publish=True,
        )
    )
    assert second["projection"]["inheritance"]["mode"] == "incremental"
    assert second["projection"]["inheritance"]["inherited_runs"] == 3
    assert second["projection"]["inheritance"]["new_runs"] == 1
    index = trend_csv.parse_run_index(p6_t4_env.read_csv_text(second, "run_index.csv"))
    trend = trend_csv.parse_trend(p6_t4_env.read_csv_text(second, "trend.csv"))
    assert len(index) == 4
    assert len(trend) == 20
    # 历史行逐字节保留：上一快照的每一条索引/指标行文本都能在新文件中原样找到
    old_index_lines = set(
        p6_t4_env.read_csv_text(first, "run_index.csv").splitlines()[1:]
    )
    new_index_lines = set(p6_t4_env.read_csv_text(second, "run_index.csv").splitlines()[1:])
    assert old_index_lines <= new_index_lines
    old_trend_lines = set(p6_t4_env.read_csv_text(first, "trend.csv").splitlines()[1:])
    new_trend_lines = set(p6_t4_env.read_csv_text(second, "trend.csv").splitlines()[1:])
    assert old_trend_lines <= new_trend_lines
    # 增量页面的图与表必须真的可绘：继承行数值经投影严格转 float（A-01 回归）
    series = second["projection"]["series"]["annualized_return"]
    assert len(series) == 4
    assert all(point["value"] is None or isinstance(point["value"], float) for point in series)
    assert sum(1 for point in series if isinstance(point["value"], float)) >= 3
    html = second["files"]["report.html"]
    assert html.count(b"<polyline") >= 4  # 4 个指标有 ≥2 个数值点
    assert html.count(b'aria-label="chart-empty"') == 1  # topk 仅旧 WF 1 点 ⇒ 合法空图
    # 继承 root_map：上一快照的 legacy1 必须随包继承，否则行内 source_ref 不可回溯（A-02）
    root_map = second["projection"]["root_map"]
    assert any(key.startswith("legacy_") for key in root_map)
    assert any(key.startswith("src_") for key in root_map)
    assert "prev" in root_map
    assert len(second["projection"]["inherited_roots"]) == 1
    assert second["projection"]["inherited_roots"][0].startswith("legacy_")
    assert second["projection"]["inherited_source"]["report_id"] == first["report_id"]
    assert (
        second["projection"]["inheritance"]["previous_manifest_bytes_sha256"]
        == sha256((previous_dir / "report_manifest.json").read_bytes()).hexdigest()
    )
    # 上一快照位置登记（这里在 temp 内 ⇒ 只登记存在性，不写绝对路径）
    previous_record = second["projection"]["inherited_source"]["previous"]
    assert previous_record["run_dir_present"] is True
    assert previous_record["inside_repo"] is False
    assert previous_record["run_dir"] is None
    # 上一快照目录字节不变（历史 run 不被改写）
    assert _digest_tree(previous_dir) == before
    # 篡改上一快照 CSV 后拒绝继承（先验 manifest 登记的产物哈希）
    target = previous_dir / "trend.csv"
    original = target.read_bytes()
    try:
        target.write_bytes(original.replace(b"annualized_return", b"annualized_returns", 1))
        with pytest.raises(trend_sources.TrendSourceError):
            generate.generate_trend(
                p6_t4_env.research_request(
                    reports, corpus, attempt="history_c", previous=previous_dir
                )
            )
    finally:
        target.write_bytes(original)


def test_p6_t4_history_root_id_is_path_derived_not_order(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """root_id 由来源根路径派生：改变 `--source` 顺序不改变身份（T4-R1-06）。"""
    root = tmp_path_factory.mktemp("p6_t4_history_order")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    reports = root / "reports"
    forward = generate.generate_trend(
        p6_t4_env.research_request(
            reports,
            corpus,
            attempt="order_a",
            sources=(f"runs:{corpus / 'runs'}", f"legacy_wf:{corpus / 'legacy'}"),
        )
    )
    reversed_order = generate.generate_trend(
        p6_t4_env.research_request(
            reports,
            corpus,
            attempt="order_b",
            sources=(f"legacy_wf:{corpus / 'legacy'}", f"runs:{corpus / 'runs'}"),
        )
    )
    assert forward["projection"]["root_map"] == reversed_order["projection"]["root_map"]
    assert sorted(
        row["run_key"] for row in forward["projection"]["runs"]
    ) == sorted(row["run_key"] for row in reversed_order["projection"]["runs"])


def test_p6_t4_history_root_token_anchored_to_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """root_id 以仓库根为锚点：切换 cwd 不改变同一来源根的标识（T4-R2-N1）。"""
    repo_relative = Path(PROJECT_ROOT) / "data"
    expected = sha256("data".encode("utf-8")).hexdigest()[:8]
    before = trend_sources._root_token(repo_relative)
    monkeypatch.chdir(Path(PROJECT_ROOT) / "src")
    after = trend_sources._root_token(repo_relative)
    assert before == after == expected


def test_p6_t4_history_absolute_source_bytes_identical_across_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """同一绝对来源、两个 cwd：五件字节一致（同 generated_at 固定）。"""
    root = tmp_path_factory.mktemp("p6_t4_history_cwd_e2e")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    absolute = (corpus / "runs").resolve()
    reports = root / "reports"
    request = lambda attempt: p6_t4_env.research_request(  # noqa: E731
        reports, corpus, attempt=attempt, sources=(f"runs:{absolute}",)
    )
    first = generate.generate_trend(request("cwd_a"))
    monkeypatch.chdir(Path(PROJECT_ROOT) / "src")
    second = generate.generate_trend(request("cwd_b"))
    assert first["projection"]["root_map"] == second["projection"]["root_map"]
    assert first["files"] == second["files"]


def test_p6_t4_history_root_id_collision_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """不同来源根映射到同一 root_id（8 hex 碰撞）⇒ 生成期拒绝（T4-R3-R3A-01）。"""
    root = tmp_path_factory.mktemp("p6_t4_history_collision")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    copied = root / "runs_copy"
    shutil.copytree(corpus / "runs", copied)
    monkeypatch.setattr(trend_sources, "_root_token", lambda path: "deadbeef")
    with pytest.raises(trend_sources.TrendSourceError, match="root_id 冲突"):
        generate.generate_trend(
            p6_t4_env.research_request(
                root / "reports",
                corpus,
                attempt="collision",
                sources=(f"runs:{corpus / 'runs'}", f"runs:{copied}"),
            )
        )


def test_p6_t4_history_unregistered_root_id_rejected() -> None:
    """投影行引用未登记 root_id ⇒ 生成期 fail-closed（T4-R2-A2-03）。"""
    run_key = "a" * 64
    index_rows = [
        {
            "run_key": run_key,
            "source_ref": "ghost:wf_0001",
            "source_sha256": "b" * 64,
            "index_mode": "full",
            "report_status": "available",
            "degradation_reason_code": None,
        }
    ]
    trend_rows = [
        {
            "run_key": run_key,
            "metric_name": "annualized_return",
            "source_ref": "ghost:wf_0001",
            "source_sha256": "b" * 64,
        }
    ]
    with pytest.raises(trend_sources.TrendSourceError, match="未登记的 root_id"):
        generate._assert_trend_root_refs(index_rows, trend_rows, {"src_x": "说明"})


def test_p6_t4_history_duplicate_identity_under_new_root_rejected(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """同一物理 run 换到另一来源根（同身份、不同 root_id）：增量合并拒绝静默重复。"""
    root = tmp_path_factory.mktemp("p6_t4_history_dup")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    reports = root / "reports"
    previous_dir = Path(
        generate.generate_trend(
            p6_t4_env.research_request(
                reports,
                corpus,
                attempt="dup_a",
                sources=(f"legacy_wf:{corpus / 'legacy'}",),
                publish=True,
            )
        )["report_path"]
    )
    copied = root / "legacy_copy"
    shutil.copytree(corpus / "legacy", copied)
    with pytest.raises(trend_sources.TrendSourceError, match="同一来源身份"):
        generate.generate_trend(
            p6_t4_env.research_request(
                reports,
                corpus,
                attempt="dup_b",
                sources=(f"legacy_wf:{copied}",),
                previous=previous_dir,
            )
        )
