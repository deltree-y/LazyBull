# -*- coding: utf-8 -*-
"""P6 T4 三类证据域物理隔离（设计 §2.2 / §4.3）。

P6-T4-05：根、索引、横幅、输入白名单四层隔离——越类输入 fail-closed、跨类上一快照
拒绝继承、三类产物根互不重叠。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.lazybull.v2.reporting import evidence, generate, trend_sources
from tests import p6_t4_env

_CLASSES = ("production", "research", "calibration_only")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p6_t4_isolation") / "corpus"
    p6_t4_env.build_corpus(root)
    return root


def test_p6_t4_05_input_whitelists_fail_closed(corpus: Path) -> None:
    """越类输入在白名单层即失败（production 收 research/calibration、calibration 收 runs）。"""
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources(
            "production", raw_sources=(f"runs:{corpus / 'runs'}",), paper_root=None, bundles=None
        )
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources(
            "production", raw_sources=(), paper_root=corpus / "paper",
            bundles=(str(corpus / "legacy"),),
        )
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources(
            "research", raw_sources=(f"runs:{corpus / 'runs'}",),
            paper_root=corpus / "paper", bundles=None,
        )
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources(
            "research", raw_sources=(), paper_root=None, bundles=(str(corpus / "paper"),)
        )
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources(
            "calibration_only", raw_sources=(f"runs:{corpus / 'runs'}",),
            paper_root=None, bundles=None,
        )
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources(
            "calibration_only", raw_sources=(), paper_root=corpus / "paper",
            bundles=(str(corpus / "paper"),),
        )
    with pytest.raises(trend_sources.TrendSourceError):
        trend_sources.build_sources("research", raw_sources=(), paper_root=None, bundles=None)


def test_p6_t4_05_cross_class_previous_is_rejected(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """跨证据类上一快照不得继承；三类产物根物理隔离。"""
    reports = tmp_path_factory.mktemp("p6_t4_isolation_reports")
    production = generate.generate_trend(
        p6_t4_env.production_request(reports, corpus, attempt="iso_prod", publish=True)
    )
    assert production["report_path"] is not None
    production_dir = Path(production["report_path"])
    assert production_dir.parent.name == "trend"
    assert production_dir.parent.parent.name == "production"
    research = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="iso_research", publish=True)
    )
    research_dir = Path(research["report_path"])
    assert research_dir.parent.parent.name == "research"
    with pytest.raises(trend_sources.TrendSourceError):
        generate.generate_trend(
            p6_t4_env.research_request(
                reports, corpus, attempt="iso_cross", previous=production_dir
            )
        )
    # 跨类横幅常量为固定口径（三类互不替代）
    banners = {name: evidence.EVIDENCE_BANNERS[name]["text"] for name in _CLASSES}
    assert len(set(banners.values())) == 3
    research_html = research["files"]["report.html"].decode("utf-8")
    production_html = production["files"]["report.html"].decode("utf-8")
    assert banners["research"] in research_html
    assert banners["production"] in production_html
    assert banners["production"] not in research_html
    assert banners["calibration_only"] not in production_html
