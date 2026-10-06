# -*- coding: utf-8 -*-
"""v2 列族分组表测试（冻结 §2/§6 代码化的完备性断言）。

覆盖：383+34 无重复、PANEL_GROUPS 并集 == 377（383−labels 6）、labels 不在 panel 族、
group_of_column / columns_of_group 语义、与 manifest VALID_GROUPS 的兼容性、
与冻结 JSON 的逐组一致（防手工改表漂移）。
"""

import json
from pathlib import Path

import pytest

from src.lazybull.v2.store.column_groups import (
    LABEL_TABLES,
    LABEL_VALUE_COLUMNS,
    MATERIALIZED_COLUMNS,
    MATERIALIZED_DERIVED,
    PANEL_GROUPS,
    columns_of_group,
    group_of_column,
    validate_mapping,
)
from src.lazybull.v2.store.manifest import VALID_GROUPS

_ROOT = Path(__file__).resolve().parents[1]


class TestMappingCompleteness:
    def test_validate_mapping_passes(self):
        validate_mapping()  # 不报错即通过（377 并集 / 无重复 / 34 物化 / 标签不混入）

    def test_panel_group_sizes(self):
        expected = {
            "core": 52,
            "fundamental": 69,
            "moneyflow": 44,
            "technical": 43,
            "announcement": 10,
            "risk": 29,
            "market_state": 13,
            "neutralized": 117,
        }
        assert {g: len(c) for g, c in PANEL_GROUPS.items()} == expected
        assert sum(len(c) for c in PANEL_GROUPS.values()) == 377

    def test_materialized_sizes(self):
        assert {k: len(v) for k, v in MATERIALIZED_DERIVED.items()} == {
            "ht": 10,
            "rp": 6,
            "tfh": 8,
            "ti": 6,
            "has": 4,
        }
        assert len(MATERIALIZED_COLUMNS) == 34

    def test_no_duplicates_anywhere(self):
        all_cols = [c for cols in PANEL_GROUPS.values() for c in cols] + list(MATERIALIZED_COLUMNS)
        assert len(all_cols) == len(set(all_cols)) == 411

    def test_labels_not_in_panel_groups(self):
        panel_cols = {c for cols in PANEL_GROUPS.values() for c in cols}
        assert not (set(LABEL_VALUE_COLUMNS) & panel_cols)
        assert set(LABEL_TABLES) == {"y_ret_5", "y_ret_10", "y_ret_20"}
        assert LABEL_TABLES["y_ret_20"] == ("y_ret_20", "neu_y_ret_20")

    def test_frozen_json_consistency(self):
        """与冻结生成物逐项一致（JSON 是生成中间物，本断言防代码表被手工改漂移）。"""
        frozen = json.loads(
            (_ROOT / "tests" / "fixtures" / "p1_column_groups_20261002.json").read_text(encoding="utf-8")
        )
        for group, cols in frozen.items():
            if group == "labels":
                assert sorted(cols) == sorted(LABEL_VALUE_COLUMNS)
            else:
                assert tuple(cols) == PANEL_GROUPS[group]

    def test_groups_compatible_with_manifest(self):
        """8 族 ⊆ manifest VALID_GROUPS；labels 不是 panel 族（走 append_labels 不经该校验）。"""
        assert set(PANEL_GROUPS) <= VALID_GROUPS
        assert "labels" not in VALID_GROUPS


class TestLookupHelpers:
    def test_group_of_column_panel(self):
        assert group_of_column("ret_1") == "core"
        assert group_of_column("zscore_pb") == "neutralized"
        assert group_of_column("mkt_ret_vol_20") == "market_state"

    def test_group_of_column_materialized(self):
        assert group_of_column("ht_net_ratio_30d") == "announcement"
        assert group_of_column("ti_schema_v1") == "announcement"
        assert group_of_column("has_cons_coverage") == "announcement"

    def test_group_of_column_unknown_raises(self):
        with pytest.raises(KeyError, match="未登记"):
            group_of_column("no_such_col")

    def test_columns_of_group_announcement_includes_materialized(self):
        cols = columns_of_group("announcement")
        assert len(cols) == 44  # 10 基础 + 34 物化
        assert "pledge_ratio" in cols and "ti_inst_days_20" in cols

    def test_columns_of_group_unknown_raises(self):
        with pytest.raises(KeyError, match="未知 panel 族"):
            columns_of_group("candidate")  # candidate 在 VALID_GROUPS 但不在 PANEL_GROUPS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
