"""交易侧行业 PIT 接线测试（专家反例：开放区间被排除）。"""

from __future__ import annotations

import pandas as pd

from scripts.v2_p7.defensive_ml import b_native


def _pit(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_open_interval_is_kept() -> None:
    """空 out_date = 仍有效的开放区间，必须参与当日映射。"""
    pit = _pit(
        [{"ts_code": "A", "in_date": "20100101", "out_date": None, "l1_code": "480000"}]
    )
    mapping = b_native.industry_mapping_for_day(pit, {"480000": "银行"}, "20210817")
    assert mapping.get("A") == "银行"


def test_closed_interval_boundary() -> None:
    """out_date 为开区间上界：当天不再生效。"""
    pit = _pit(
        [{"ts_code": "A", "in_date": "20100101", "out_date": "20210817", "l1_code": "480000"}]
    )
    names = {"480000": "银行"}
    assert b_native.industry_mapping_for_day(pit, names, "20210816").get("A") == "银行"
    assert b_native.industry_mapping_for_day(pit, names, "20210817").get("A") is None


def test_in_date_boundary_is_inclusive() -> None:
    pit = _pit(
        [{"ts_code": "A", "in_date": "20210817", "out_date": None, "l1_code": "220000"}]
    )
    names = {"220000": "基础化工"}
    assert b_native.industry_mapping_for_day(pit, names, "20210817").get("A") == "基础化工"
    assert b_native.industry_mapping_for_day(pit, names, "20210816").get("A") is None


def test_unmapped_code_lands_in_unknown_bucket() -> None:
    """无可核一级名称 ⇒ 显式"未知行业"，不静默消失（选择期再据 F01 排除）。"""
    pit = _pit(
        [{"ts_code": "A", "in_date": "20100101", "out_date": None, "l1_code": "470000"}]
    )
    mapping = b_native.industry_mapping_for_day(pit, {}, "20210817")
    assert mapping.get("A") == "未知行业"
