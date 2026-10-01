# -*- coding: utf-8 -*-
"""glossary_refs 脚本单元测试（合成语料，不依赖真实文档库）"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check" / "glossary_refs.py"
_spec = importlib.util.spec_from_file_location("glossary_refs", _SCRIPT)
glossary_refs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = glossary_refs  # dataclass 处理需要模块已注册
_spec.loader.exec_module(glossary_refs)


def _make_terms(tmp_path: Path) -> Path:
    content = """
version: 1
terms:
  - term: 袖子
    aliases: [sleeve]
    one_liner: x
    layman: x
    example: x
    source: x
    related: []
  - term: 净额化
    aliases: [netting]
    one_liner: x
    layman: x
    example: x
    source: x
    related: []
"""
    p = tmp_path / "terms.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_terms(tmp_path):
    terms = glossary_refs.load_terms(_make_terms(tmp_path))
    assert [t["term"] for t in terms] == ["袖子", "净额化"]


def test_alias_and_word_boundary(tmp_path):
    """ASCII 别名按词边界匹配（sleeve 不应命中 sleeved）；中文按子串"""
    entry = {"term": "袖子", "aliases": ["sleeve"]}
    patterns = glossary_refs._patterns_for(entry)
    text_hit = "每只 sleeve 独立记账；袖子之间零互 import"
    text_miss = "the sleeved version"
    assert any(p.search(text_hit) for p in patterns)
    assert not any(p.search(text_miss) for p in patterns)


def test_count_references_per_file_once(tmp_path):
    """同一文件命中多次只记 1（按文件计数口径）"""
    doc = tmp_path / "a.md"
    doc.write_text("袖子 袖子 袖子 sleeve", encoding="utf-8")
    entry = {"term": "袖子", "aliases": ["sleeve"]}
    patterns_map = {"袖子": glossary_refs._patterns_for(entry)}
    counts = glossary_refs.count_references([doc], patterns_map)
    assert counts["袖子"] == 1

    doc2 = tmp_path / "b.md"
    doc2.write_text("净额化与袖子", encoding="utf-8")
    counts = glossary_refs.count_references([doc, doc2], patterns_map)
    assert counts["袖子"] == 2


def test_main_outputs_sorted(tmp_path, monkeypatch, capsys):
    _make_terms(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("袖子 与 netting", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # git 不可用 / 非仓库时 recent 降级为 None，不影响主流程
    rc = glossary_refs.main(["--terms", "terms.yaml"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "袖子" in out and "净额化" in out


def test_csv_output(tmp_path, monkeypatch):
    _make_terms(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = glossary_refs.main(["--terms", "terms.yaml", "--csv", "out.csv"])
    assert rc == 0
    text = (tmp_path / "out.csv").read_text(encoding="utf-8-sig")
    assert "术语" in text and "袖子" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
