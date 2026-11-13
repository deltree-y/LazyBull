# -*- coding: utf-8 -*-
"""申万行业分类历史（PIT）数据层单测（离线，不依赖真实网络与生产数据）。"""

import pandas as pd
import pytest

from src.lazybull.data.sw_industry_pit import (
    SW_SRC_SW2014,
    SW_SRC_SW2021,
    SW_VERSION_UNMAPPED,
    assign_version,
    attach_code_names,
    attach_pit,
    build_code_table,
    build_exclusion_ledger,
    diff_panels,
    finalize_pit,
    infer_ts_code,
    lookup_industry,
    normalize_history,
    summarize_ledger,
    validate_pit,
)


def _raw(rows):
    """构造原始四列表（与官网 Excel 语义一致）。"""
    frame = pd.DataFrame(rows, columns=["code6", "in_date", "ind_code", "source_updated_at"])
    frame["source_updated_at"] = pd.to_datetime(frame["source_updated_at"])
    return frame


def _code_table():
    rows = [
        (SW_SRC_SW2014, "480000", "L1", "银行", "801780.SI"),
        (SW_SRC_SW2014, "480100", "L2", "银行", "801192.SI"),
        (SW_SRC_SW2014, "480101", "L3", "国有大型银行", "851911.SI"),
        (SW_SRC_SW2021, "480000", "L1", "银行", "801780.SI"),
        (SW_SRC_SW2021, "480300", "L2", "股份制银行Ⅱ", "801783.SI"),
        (SW_SRC_SW2021, "480301", "L3", "股份制银行Ⅲ", "857831.SI"),
    ]
    return pd.DataFrame(
        rows, columns=["src", "industry_code", "level", "industry_name", "index_code"]
    )


def test_normalize_history_builds_intervals_and_dedupes():
    raw = _raw(
        [
            ("000001", "1991-04-03", "440101", "2015-10-27"),
            ("000001", "2014-02-21", "480101", "2024-09-27"),
            ("000001", "2021-07-30", "480301", "2024-09-27"),
            ("000001", "2021-07-30", "480301", "2025-12-15"),
        ]
    )
    panel = normalize_history(raw)
    assert len(panel) == 3
    assert list(panel["in_date"]) == ["19910403", "20140221", "20210730"]
    assert list(panel["out_date"].fillna("")) == ["20140221", "20210730", ""]
    assert panel.iloc[-1]["ind_code"] == "480301"


def test_normalize_history_drops_unparsable_rows():
    raw = _raw(
        [
            ("000001", "not-a-date", "480301", "2024-01-01"),
            ("bad", "20200101", "480301", "2024-01-01"),
        ]
    )
    assert len(normalize_history(raw)) == 0


@pytest.mark.parametrize(
    "code,expected",
    [
        ("920017", "920017.BJ"),
        ("830799", "830799.BJ"),
        ("600030", "600030.SH"),
        ("300059", "300059.SZ"),
        ("000001", "000001.SZ"),
        ("200011", "200011.SZ"),
        ("900901", "900901.SH"),
    ],
)
def test_infer_ts_code(code, expected):
    assert infer_ts_code(code) == expected


def test_infer_ts_code_unknown_prefix():
    assert infer_ts_code("777777") is None


def test_assign_version_uses_cutover_and_code_table():
    panel = normalize_history(
        _raw(
            [
                ("000001", "19910403", "440101", "2015-10-27"),
                ("000001", "20140221", "480101", "2015-10-27"),
                ("000001", "20210730", "480301", "2024-09-27"),
            ]
        )
    )
    out = assign_version(panel, _code_table())
    assert list(out["sw_version"]) == [SW_VERSION_UNMAPPED, SW_SRC_SW2014, SW_SRC_SW2021]


def test_attach_code_names_maps_levels_and_leaves_unmapped_nan():
    panel = assign_version(
        normalize_history(
            _raw(
                [
                    ("000001", "20140221", "480101", "2015-10-27"),
                    ("000002", "19910403", "440101", "2015-10-27"),
                ]
            )
        ),
        _code_table(),
    )
    out = attach_code_names(panel, _code_table())
    known = out[out["code6"] == "000001"].iloc[0]
    assert (known["l1_name"], known["l2_name"], known["l3_name"]) == (
        "银行",
        "银行",
        "国有大型银行",
    )
    assert known["index_code"] == "851911.SI"
    assert pd.isna(out[out["code6"] == "000002"].iloc[0]["l3_name"])


def _pit():
    panel = assign_version(
        normalize_history(
            _raw(
                [
                    ("000001", "20140221", "480101", "2024-09-27"),
                    ("000001", "20210730", "480301", "2024-09-27"),
                    ("920017", "20230530", "480301", "2024-09-27"),
                ]
            )
        ),
        _code_table(),
    )
    panel = attach_code_names(panel, _code_table())
    return finalize_pit(panel)


def test_attach_pit_boundaries_and_row_order():
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "600000.SH"],
            "trade_date": ["20140221", "20210730", "20210729", "20200102"],
        }
    )
    out = attach_pit(_pit(), frame)
    assert len(out) == 4
    assert list(out["ts_code"]) == list(frame["ts_code"])
    assert out.loc[0, "ind_code"] == "480101"  # in_date 当天算已知
    assert out.loc[1, "ind_code"] == "480301"
    assert out.loc[2, "ind_code"] == "480101"  # 前一区间末日仍有效
    assert not bool(out.loc[3, "industry_known"])  # 无记录 → 未知
    assert pd.isna(out.loc[3, "ind_code"])


def test_attach_pit_out_date_exclusive():
    frame = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20210730"]})
    panel = _pit()
    assert panel.loc[panel["code6"] == "000001", "out_date"].iloc[0] == "20210730"
    assert attach_pit(panel, frame).loc[0, "ind_code"] == "480301"


def test_lookup_industry_single_point():
    panel = _pit()
    assert lookup_industry(panel, "000001.SZ", "2015-01-05")["ind_code"] == "480101"
    assert lookup_industry(panel, "600000.SH", "2015-01-05") is None


def test_build_exclusion_ledger_reasons():
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "920017.BJ", "601360.SH"],
            "trade_date": ["20100104", "20200102", "20150105"],
        }
    )
    ledger = build_exclusion_ledger(_pit(), frame)
    assert len(ledger) == 3
    reasons = dict(zip(ledger["ts_code"], ledger["reason"]))
    assert reasons["000001.SZ"] == "not_yet_classified"
    assert reasons["920017.BJ"] == "not_yet_classified"
    assert reasons["601360.SH"] == "relisting_shell"
    summary = summarize_ledger(ledger, total_rows=len(frame))
    assert summary["excluded_share"] == pytest.approx(1.0)
    assert summary["by_reason"]["relisting_shell"] == 1


def test_bse_alias_rows_are_added():
    raw = _raw([("920017", "20230530", "480301", "2024-09-27")])
    panel = normalize_history(raw)
    panel = attach_code_names(assign_version(panel, _code_table()), _code_table())
    pit = finalize_pit(panel, alias={"830799": "920017"})
    assert set(pit.loc[pit["code_alias"], "code6"]) == {"830799"}
    assert pit.loc[pit["code_alias"], "ts_code"].iloc[0] == "830799.BJ"


def test_validate_pit_rejects_duplicates():
    panel = _pit()
    dup = pd.concat([panel, panel.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="重复键"):
        validate_pit(dup)


def _panel(rows):
    frame = pd.DataFrame(
        rows, columns=["code6", "in_date", "out_date", "ind_code", "sw_version", "code_alias"]
    )
    frame["source_updated_at"] = pd.to_datetime("2026-01-01")
    return frame


def test_panel_fingerprint_is_order_independent():
    a = _panel([("000001", "20140221", "20210730", "480101", SW_SRC_SW2014, False)])
    b = a.iloc[::-1].reset_index(drop=True)
    assert diff_panels(a, b)["changed"] is False


def test_diff_panels_detects_no_change():
    frame = _panel([("000001", "20140221", "20210730", "480101", SW_SRC_SW2014, False)])
    diff = diff_panels(frame, frame.copy())
    assert diff["changed"] is False
    assert diff["added_intervals"] == 0
    assert diff["removed_intervals"] == 0
    assert diff["reclassified_intervals"] == 0


def test_diff_panels_detects_changes():
    old = _panel(
        [
            ("000001", "20140221", "20210730", "480101", SW_SRC_SW2014, False),
            ("000002", "20140221", "", "430101", SW_SRC_SW2014, False),
        ]
    )
    new = _panel(
        [
            ("000001", "20140221", "20210730", "480301", SW_SRC_SW2014, False),  # 改类
            ("000001", "20210730", "", "480301", SW_SRC_SW2021, False),  # 新增区间
            ("000003", "20200101", "", "430101", SW_SRC_SW2021, False),  # 新增股票
        ]
    )
    diff = diff_panels(old, new)
    assert diff["changed"] is True
    assert diff["added_intervals"] == 2
    assert diff["removed_intervals"] == 1  # 000002 的区间消失
    assert diff["reclassified_intervals"] == 1
    assert diff["codes_added"] == 1
    assert diff["codes_removed"] == 1
    assert diff["old"]["content_sha256"] != diff["new"]["content_sha256"]


def test_diff_panels_without_baseline_marks_changed():
    frame = _panel([("000001", "20140221", "", "480101", SW_SRC_SW2014, False)])
    diff = diff_panels(None, frame)
    assert diff["changed"] is True
    assert diff["baseline"] == "none"


def test_build_code_table_dedupes_and_requires_rows():
    table = build_code_table([_code_table(), _code_table()])
    assert len(table) == 6
    with pytest.raises(ValueError, match="码表为空"):
        build_code_table([])
