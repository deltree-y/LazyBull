"""公司行为输入副本的合成检查，不依赖真实配置、行情或公告。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p7.operating_monthly_corporate import (
    DividendResolutionError,
    file_sha256,
    prepare_research_dividends,
    row_sha256,
)

DAYS = tuple(day.strftime("%Y%m%d") for day in pd.bdate_range("2026-01-05", periods=6))
CODES = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Any]:
    raw = tmp_path / "raw"
    raw.mkdir()
    row = {
        "ts_code": CODES[0],
        "end_date": "20250630",
        "ann_date": DAYS[0],
        "imp_ann_date": DAYS[0],
        "div_proc": "实施",
        "cash_div": 0.4,
        "cash_div_tax": 0.4,
        "stk_div": 0.0,
        "record_date": DAYS[1],
        "ex_date": DAYS[2],
        "pay_date": DAYS[2],
        "div_listdate": None,
        "opaque_vendor_detail": "原始额外字段也须保留",
    }
    rows = [
        row,
        {**row, "cash_div": 1.4, "cash_div_tax": 1.4},
        {**row, "end_date": "20251011", "cash_div": 1.0, "cash_div_tax": 1.0},
        {**row, "ts_code": "999999.SZ", "cash_div": 2.3, "cash_div_tax": 2.3},
    ]
    pd.DataFrame(rows).to_parquet(raw / "2026.parquet", index=False)
    pd.DataFrame([{**row, "div_proc": "预案"}]).to_parquet(raw / "2025.parquet", index=False)
    (raw / "coverage.json").write_bytes(b'{"retained":"byte-for-byte"}\r\n')
    original = pd.read_parquet(raw / "2026.parquet")
    proof = tmp_path / "original.pdf"
    proof.write_bytes(b"synthetic official record: total 14 per 10 shares, 4 + 10")
    resolution = {
        "resolution_id": "synthetic-aggregate-v1",
        "ts_code": CODES[0],
        "ex_date": DAYS[2],
        "known_date": DAYS[0],
        "expected_row_sha256s": [row_sha256(original.iloc[index].to_dict()) for index in range(3)],
        "retained_row_sha256": row_sha256(original.iloc[1].to_dict()),
        "confirmed_fields": {
            "cash_div": 1.4,
            "cash_div_tax": 1.4,
            "stk_div": 0.0,
            "record_date": DAYS[1],
            "pay_date": DAYS[2],
        },
        "evidence": [
            {
                "source_url": "https://example.test/original.pdf",
                "local_path": str(proof),
                "sha256": file_sha256(proof),
                "publication_date": DAYS[0],
            }
        ],
    }
    return {
        "raw": raw,
        "output": tmp_path / "output",
        "original": original,
        "resolution": resolution,
        "proof": proof,
    }


def prepare(fixture: dict[str, Any], resolutions: list | None = None) -> dict[str, Any]:
    return prepare_research_dividends(
        raw_dir=fixture["raw"],
        output_dir=fixture["output"],
        calendar=DAYS,
        codes=CODES,
        resolutions=[fixture["resolution"]] if resolutions is None else resolutions,
    )


def test_exact_mapping_preserves_all_other_stocks_partitions_and_raw(fixture: dict) -> None:
    before = {path.name: file_sha256(path) for path in fixture["raw"].iterdir()}
    result = prepare(fixture)
    copied = result["dividend_dir"]
    assert file_sha256(copied / "2025.parquet") == before["2025.parquet"]
    assert file_sha256(copied / "coverage.json") == before["coverage.json"]
    assert pd.read_parquet(copied / "2026.parquet").equals(
        fixture["original"].iloc[[1, 3]].reset_index(drop=True)
    )
    assert {path.name: file_sha256(path) for path in fixture["raw"].iterdir()} == before
    manifest = result["manifest"]
    assert manifest["removed_rows"] == 2
    assert manifest["audit_before"]["unresolved_problem_count"] == 1
    assert manifest["audit_after"]["unresolved_problem_count"] == 0
    assert manifest["audit_after"]["events"][0]["cash_per_share"] == 1.4
    assert manifest["audit_after"]["event_count"] == 1
    assert manifest["resolutions"][0]["rows"][0]["source_fields"]["opaque_vendor_detail"]


def test_no_resolution_leaves_conflict_visible_without_guessing(fixture: dict) -> None:
    result = prepare(fixture, [])
    assert result["manifest"]["removed_rows"] == 0
    assert result["manifest"]["audit_after"]["unresolved_problem_count"] == 1
    assert result["manifest"]["audit_after"]["event_count"] == 0
    assert all(item["byte_identical"] for item in result["manifest"]["files"])


@pytest.mark.parametrize(
    "mutation", ["incomplete_rows", "raw_changed", "wrong_fact", "proof_changed"]
)
def test_unconfirmed_or_changed_input_is_rejected_before_output(
    fixture: dict, mutation: str
) -> None:
    if mutation == "incomplete_rows":
        fixture["resolution"]["expected_row_sha256s"].pop()
    elif mutation == "raw_changed":
        frame = fixture["original"].copy()
        frame.loc[0, "opaque_vendor_detail"] = "供应商已修订，旧冻结规则不再适用"
        frame.to_parquet(fixture["raw"] / "2026.parquet", index=False)
    elif mutation == "wrong_fact":
        fixture["resolution"]["confirmed_fields"]["cash_div"] = 1.393962
    else:
        fixture["proof"].write_bytes(b"changed evidence")
    with pytest.raises(DividendResolutionError):
        prepare(fixture)
    assert not fixture["output"].exists()


def test_future_publication_cannot_rewrite_historical_event(fixture: dict) -> None:
    fixture["resolution"]["evidence"][0]["publication_date"] = DAYS[3]
    with pytest.raises(DividendResolutionError, match="迟于"):
        prepare(fixture)


def test_overlapping_resolutions_and_reused_output_are_rejected(fixture: dict) -> None:
    duplicate = copy.deepcopy(fixture["resolution"])
    duplicate["resolution_id"] = "another-id-same-rows"
    with pytest.raises(DividendResolutionError, match="重复"):
        prepare(fixture, [fixture["resolution"], duplicate])
    prepare(fixture)
    with pytest.raises(DividendResolutionError, match="已存在"):
        prepare(fixture)


def test_output_cannot_enter_raw_tree(fixture: dict) -> None:
    fixture["output"] = fixture["raw"] / "new"
    with pytest.raises(DividendResolutionError, match="独立"):
        prepare(fixture)
