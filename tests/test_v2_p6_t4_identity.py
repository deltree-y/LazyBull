# -*- coding: utf-8 -*-
"""P6 T4 run 身份规则（设计 §5.7）：manifest 字节 SHA / 消费文件 canonical 哈希。

覆盖 P6-T4-06：旧 WF 无 manifest 的 source_identity 可复算且 ``manifest_absent``；
以及 runs 有 manifest 时 ``source_identity_sha256 = manifest 字节 SHA``（禁止 mtime/空串）。
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest

from src.lazybull.v2.reporting import generate, trend_csv
from src.lazybull.v2.store.store_identity import canonical_json_sha256
from tests import p6_t4_env


def _identity_row(result: dict, implementation: str) -> dict[str, str]:
    rows = trend_csv.parse_run_index(p6_t4_env.read_csv_text(result, "run_index.csv"))
    return next(row for row in rows if row["implementation"] == implementation)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p6_t4_identity") / "corpus"
    p6_t4_env.build_corpus(root)
    return root


def test_p6_t4_identity_runs_uses_manifest_bytes(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """有 manifest 时身份 = manifest 字节 SHA（三列同值；不含 mtime/空串）。"""
    reports = tmp_path_factory.mktemp("p6_t4_identity_reports")
    result = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="identity_manifest")
    )
    row = _identity_row(result, "runs")
    manifest = corpus / "runs" / row["run_id"] / "manifest.json"
    expected = sha256(manifest.read_bytes()).hexdigest()
    assert row["source_identity_sha256"] == expected
    assert row["source_manifest_sha256"] == expected
    assert row["source_sha256"] == expected
    assert row["source_ref"].startswith("src_")
    assert row["source_ref"].endswith(f":{row['run_id']}")
    # 身份与运行目录 mtime 无关：改 mtime 后同输入仍得到同 report_id
    os.utime(manifest, None)
    again = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="identity_manifest_again")
    )
    assert again["report_id"] == result["report_id"]
    assert again["files"] == result["files"]


def test_p6_t4_identity_legacy_is_recomputable_and_manifest_absent(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """旧 WF：身份 = 排序后 path+sha canonical 哈希；manifest 列空且标 manifest_absent。"""
    reports = tmp_path_factory.mktemp("p6_t4_identity_legacy")
    result = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="identity_legacy")
    )
    row = _identity_row(result, "legacy_wf")
    run_dir = corpus / "legacy" / row["run_id"]
    consumed = [
        {
            "path": f"{row['run_id']}/{name}",
            "sha256": sha256((run_dir / name).read_bytes()).hexdigest(),
        }
        for name in sorted(("chain_nav_wf_corpus.csv", "walk_forward_summary_wf_corpus.csv"))
    ]
    assert row["source_identity_sha256"] == canonical_json_sha256(consumed)
    assert row["source_manifest_sha256"] == ""
    identity = result["projection"]["source_identity"]
    assert identity["flag"] == "manifest_absent"
    assert identity["count"] == 1
    # 再生成一次：同输入同 generated_at ⇒ 字节一致（身份不含 mtime）
    again = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="identity_legacy_again")
    )
    assert again["files"] == result["files"]
    # 改动被消费文件的一个字节 ⇒ 身份可复算地变化（不是 mtime 依赖）
    chain = run_dir / "chain_nav_wf_corpus.csv"
    original = chain.read_bytes()
    try:
        chain.write_bytes(original.replace(b"1.", b"2.", 1))
        changed = generate.generate_trend(
            p6_t4_env.research_request(reports, corpus, attempt="identity_legacy_changed")
        )
        changed_row = next(
            item
            for item in trend_csv.parse_run_index(
                p6_t4_env.read_csv_text(changed, "run_index.csv")
            )
            if item["implementation"] == "legacy_wf"
        )
        assert changed_row["source_identity_sha256"] != row["source_identity_sha256"]
        assert changed_row["run_key"] != row["run_key"]
    finally:
        chain.write_bytes(original)
