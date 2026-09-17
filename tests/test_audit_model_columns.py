#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型列集审计（scripts/model_audit）专项测试。"""

import json

import pytest

from scripts.model_audit import (
    STATUS_ALWAYS,
    STATUS_INTERMITTENT,
    STATUS_NEW,
    STATUS_REMOVED,
    build_audit_tables,
    build_markdown,
    cross_source_diff,
    drift_ledger,
    frequency_table,
    parse_source_arg,
    presence_matrix,
    scan_source,
    with_family,
)


def _write_features(root, name, columns):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_features.json"
    path.write_text(json.dumps(list(columns), ensure_ascii=False), encoding="utf-8")
    return path


def test_parse_source_arg_with_and_without_label(tmp_path):
    spec = parse_source_arg(f"deploy={tmp_path}")
    assert spec.label == "deploy"
    assert spec.root == tmp_path

    spec2 = parse_source_arg(str(tmp_path))
    assert spec2.label == tmp_path.name


def test_parse_source_arg_rejects_bad_input():
    with pytest.raises(ValueError, match="不能为空"):
        parse_source_arg("   ")
    with pytest.raises(ValueError, match="格式非法"):
        parse_source_arg("deploy=")


def test_scan_source_orders_by_version_and_limits_tail(tmp_path):
    _write_features(tmp_path, "v10", ["a", "b"])
    _write_features(tmp_path, "v2", ["a"])
    _write_features(tmp_path, "v30", ["a", "b", "c"])

    items = scan_source(parse_source_arg(str(tmp_path)))

    assert [item.name for item in items] == ["v2", "v10", "v30"]

    tail = scan_source(parse_source_arg(str(tmp_path)), last=2)
    assert [item.name for item in tail] == ["v10", "v30"]


def test_scan_source_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="来源目录不存在"):
        scan_source(parse_source_arg(str(tmp_path / "not_exist")))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="未找到匹配"):
        scan_source(parse_source_arg(str(empty)))


def test_scan_source_rejects_empty_or_invalid_feature_file(tmp_path):
    (tmp_path / "v1_features.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="特征清单为空"):
        scan_source(parse_source_arg(str(tmp_path)))

    (tmp_path / "v1_features.json").write_text("{bad-json", encoding="utf-8")
    with pytest.raises(ValueError, match="特征清单解析失败"):
        scan_source(parse_source_arg(str(tmp_path)))


def test_presence_matrix_and_drift_ledger(tmp_path):
    _write_features(tmp_path, "v1", ["a", "b", "c"])
    _write_features(tmp_path, "v2", ["a", "b", "d"])
    items = scan_source(parse_source_arg(str(tmp_path)), last=0)

    matrix = presence_matrix(items)
    assert list(matrix.columns) == ["a", "b", "c", "d"]
    assert bool(matrix.iloc[0]["c"]) is True
    assert bool(matrix.iloc[1]["c"]) is False

    ledger = drift_ledger(items)
    assert list(ledger["版本"]) == ["v1", "v2"]
    first, second = ledger.iloc[0], ledger.iloc[1]
    assert first["相对上一版本新增数"] == 0 and first["相对上一版本移除数"] == 0
    assert second["相对上一版本新增数"] == 1
    assert second["相对上一版本移除数"] == 1
    assert second["新增列"] == "d"
    assert second["移除列"] == "c"


def test_frequency_table_status_rules(tmp_path):
    # c 只在 v1 出现（已移除）；d 只在最新版本出现（新增）；e 间断出现；a 常驻
    _write_features(tmp_path, "v1", ["a", "c", "e"])
    _write_features(tmp_path, "v2", ["a"])
    _write_features(tmp_path, "v3", ["a", "d", "e"])
    items = scan_source(parse_source_arg(str(tmp_path)), last=0)

    table = frequency_table(items).set_index("特征")

    assert table.loc["a", "状态"] == STATUS_ALWAYS
    assert table.loc["a", "出现版本数"] == 3
    assert table.loc["c", "状态"] == STATUS_REMOVED
    assert bool(table.loc["c", "最新版本保留"]) is False
    assert table.loc["d", "状态"] == STATUS_NEW
    assert table.loc["e", "状态"] == STATUS_INTERMITTENT
    assert table.loc["e", "末次出现版本"] == "v3"
    assert table.loc["e", "出现版本数"] == 2


def test_with_family_attaches_availability_marker_family(tmp_path):
    _write_features(tmp_path, "v1", ["has_cons_coverage", "zscore_pe_ttm", "neu_ret_1"])
    items = scan_source(parse_source_arg(str(tmp_path)))
    table = with_family(frequency_table(items)).set_index("特征")

    assert table.loc["has_cons_coverage", "家族"] == "availability_marker"
    assert table.loc["neu_ret_1", "家族"].startswith("base-")


def test_with_family_handles_empty_table():
    import pandas as pd

    empty = pd.DataFrame(columns=["来源", "特征"])
    result = with_family(empty)
    assert "家族" in result.columns
    assert result.empty


def test_cross_source_diff_between_deploy_and_fold(tmp_path):
    deploy = tmp_path / "models"
    fold = tmp_path / "fold_0"
    _write_features(deploy, "v24052", ["a", "b", "c"])
    _write_features(fold, "v24052", ["a", "b"])

    items = scan_source(parse_source_arg(f"deploy={deploy}"), last=0) + scan_source(
        parse_source_arg(f"fold0={fold}"), last=0
    )

    diff = cross_source_diff(items)
    assert len(diff) == 1
    row = diff.iloc[0]
    assert row["A独有列数"] == 1
    assert row["A独有列"] == "c"
    assert row["B独有列数"] == 0
    assert row["共有列数"] == 2


def test_build_audit_tables_and_markdown(tmp_path):
    _write_features(tmp_path, "v1", ["a", "b"])
    _write_features(tmp_path, "v2", ["a", "d"])
    items = scan_source(parse_source_arg(str(tmp_path)), last=0)

    tables = build_audit_tables(items)
    assert "列集漂移台账.csv" in tables
    assert "列集出现频次.csv" in tables
    assert "列集存在矩阵.csv" in tables

    report = build_markdown(tables, items, last=0)
    assert "# 模型列集审计" in report
    assert "列集漂移" in report
    assert "v2" in report
