"""公告解析合成探针：不访问网络、配置或真实数据。"""

from scripts.v2_p7.operating_monthly_source import parse_announcement, period_from_title


def test_empty_terminal_page_does_not_erase_total(tmp_path, monkeypatch):
    import json
    import time

    from scripts.v2_p7 import operating_monthly_source as source

    monkeypatch.setattr(source, "OUTPUT", tmp_path)
    (tmp_path / "api").mkdir()
    for page, rows, total in [(1, [meta()], 1), (2, [], 0)]:
        (tmp_path / "api" / f"002352_2025_{page}.json").write_text(
            json.dumps({"response": {"announcements": rows, "totalAnnouncement": total}}),
            encoding="utf-8",
        )
    result = source.query_year("002352", 2025, time.monotonic() + 10)
    assert result["complete"] is True
    assert result["reported_total"] == 1
    assert result["page_counts"] == [1, 0]


def meta(code="002352", title="2025年3月快递物流业务经营简报"):
    return {
        "secCode": code,
        "announcementTitle": title,
        "announcementId": "synthetic",
        "announcementTime": 1744905600000,
        "adjunctUrl": "finalpage/2025-04-18/synthetic.PDF",
    }


def test_three_column_rows_and_publication_date():
    text = (
        "营业收入（人民币亿元） 178.97 161.80 10.61% 业务量（亿票） 12.95 10.33 25.36% "
        "单票收入（人民币元） 13.82 15.66 -11.75% 二〇二五年四月十七日"
    )
    row = parse_announcement(text, meta())
    assert row["publication_date"] == "20250418"
    assert row["revenue"]["value"] == 178.97
    assert row["revenue"]["prior_year_value"] == 161.80
    assert row["volume_yoy"] == 25.36
    assert row["canonical_price_yoy"] == -11.75


def test_2019_yunda_break_keeps_raw_but_blocks_canonical():
    text = "快递服务业务收入（亿元） 25.80 153.19% 完成业务量（亿票） 7.82 35.06% 快递服务单票收入（元） 3.30 87.50%"
    row = parse_announcement(text, meta("002120", "2019年3月快递服务主要经营指标快报"))
    assert row["reported_price_yoy"] == 87.50
    assert row["canonical_price_yoy"] is None
    assert row["volume_yoy"] == 35.06


def test_explicit_adjusted_yoy_used_without_deriving_numbers():
    text = (
        "快递服务业务收入（亿元） 25.80 153.19% 完成业务量（亿票） 7.82 35.06% "
        "快递服务单票收入（元） 3.30 87.50%。菜鸟业务结算调整。"
        "剔除上述影响后，单票收入同比增长10.25%。"
    )
    row = parse_announcement(text, meta("002468", "2022年3月经营简报"))
    assert row["reported_price_yoy"] == 87.50
    assert row["adjusted_price_yoy"] == 10.25
    assert row["canonical_price_yoy"] == 10.25


def test_sto_and_yto_adjustment_word_order():
    text = (
        "快递产品收入（亿元） 35.12 15.68% 业务完成量（亿票） 14.17 5.06% "
        "快递产品单票收入（元） 2.48 10.11%。菜鸟业务结算调整，剔除该等影响后，"
        "公司快递产品单票收入为2.40元，同比增长6.59%。"
    )
    row = parse_announcement(text, meta("600233", "2022年3月快递业务主要经营数据公告"))
    assert row["volume_yoy"] == 5.06
    assert row["adjusted_price_yoy"] == 6.59
    text = text.replace("公司快递产品单票收入", "单票快递服务收入")
    row = parse_announcement(text, meta("002468", "2022年3月经营简报"))
    assert row["canonical_price_yoy"] == 6.59


def test_adjustment_missing_is_not_raw_fallback():
    text = (
        "快递服务业务收入（亿元） 25.80 153.19% 完成业务量（亿票） 7.82 35.06% "
        "快递服务单票收入（元） 3.30 87.50%。菜鸟业务结算调整，增加收入0.20元。"
    )
    row = parse_announcement(text, meta("002468", "2022年3月经营简报"))
    assert row["canonical_price_yoy"] is None
    assert row["adjusted_price_yoy"] is None


def test_missing_price_marks_partial_and_merged_months_preserved():
    row = parse_announcement(
        "完成业务量（亿票） 7.82 35.06%", meta("002468", "2020年1-2月经营简报")
    )
    assert row["period_months"] == [1, 2]
    assert row["month"] == 2
    assert row["parse_status"] == "partial"
    assert row["price_yoy_pct"] is None


def test_revisions_stay_separate_and_late_publications_ineligible():
    value = meta(title="2026年9月快递物流业务经营简报更正公告")
    value["announcementTime"] = 1794787200000
    row = parse_announcement("", value)
    assert row["eligible_by_cutoff"] is False
    assert "更正或修订版本，必须按本公告日生效" in row["semantic_flags"]


def test_nonmonthly_titles_and_out_of_scope_periods():
    assert period_from_title("2022年经营情况讨论") is None
    assert parse_announcement("", meta(title="2018年12月经营简报")) is None
    assert parse_announcement("", meta(title="2026年10月经营简报")) is None


def test_qualitative_flat_adjustment_is_not_invented_zero():
    text = (
        "快递产品收入（亿元） 25.77 87.05% 业务完成量（亿票） 9.67 82.83% "
        "快递产品单票收入（元） 2.66 2.31%。菜鸟业务结算调整，剔除该等影响后，"
        "公司快递产品单票收入为2.60元，同比基本持平。"
    )
    row = parse_announcement(text, meta("600233", "2022年2月快递业务主要经营数据公告"))
    assert row["adjusted_price_yoy_text"] == "同比基本持平"
    assert row["canonical_price_yoy"] is None
    assert row["reported_price_yoy"] == 2.31


def test_preliminary_aggregate_does_not_divide_revenue_into_price():
    text = "实现快递业务完成量22.97亿票，同比增长27.81%；实现营业收入约76.24亿元，同比增长39.28%。"
    row = parse_announcement(text, meta("600233", "2022年1至2月主要经营数据公告"))
    assert row["revenue"]["value"] == 76.24
    assert row["volume_yoy"] == 27.81
    assert row["period_months"] == [1, 2]
    assert row["price"] is None
    assert row["canonical_price_yoy"] is None


def test_sf_disposal_adjusted_column_can_reverse_signal_sign():
    text = (
        "同比变动 同比变动（不含丰网） 营业收入（亿元）161.80 156.56 3.35% 5.05% "
        "业务量（亿票）10.33 10.33 - 10.84% 单票收入（元）15.66 15.16 3.30% -5.26%"
    )
    row = parse_announcement(text, meta("002352", "2024年3月快递物流业务经营简报"))
    assert row["reported_price_yoy"] == 3.30
    assert row["canonical_price_yoy"] == -5.26
    assert row["reported_volume_yoy"] is None
    assert row["volume_yoy"] == 10.84


def test_ratio_validation_uses_published_rounding_not_exact_division():
    from scripts.v2_p7.operating_monthly_source import _ratio_check

    text = (
        "快递产品收入（亿元） 25.77 87.05% 业务完成量（亿票） 9.67 82.83% "
        "快递产品单票收入（元） 2.66 2.31%"
    )
    row = parse_announcement(text, meta("600233", "2022年2月快递业务主要经营数据公告"))
    assert _ratio_check(row)["status"] == "passed"
    row["price"]["value"] = 2.90
    assert _ratio_check(row)["status"] == "failed"


def test_trace_understands_explicit_decline_word(tmp_path, monkeypatch):
    import hashlib

    from scripts.v2_p7 import operating_monthly_source as source

    evidence = "剔除影响后，单票收入同比下降3.19%"
    (tmp_path / "sample.txt").write_text(evidence, encoding="utf-8")
    (tmp_path / "sample.pdf").write_bytes(b"synthetic-pdf")
    monkeypatch.setattr(source, "ROOT", tmp_path)
    row = {
        "announcement_id": "synthetic",
        "local_text": "sample.txt",
        "local_pdf": "sample.pdf",
        "sha256": hashlib.sha256(b"synthetic-pdf").hexdigest(),
        "revenue": None,
        "volume": None,
        "price": None,
        "adjusted_price_yoy": -3.19,
        "adjustment_evidence": evidence,
    }
    assert source._trace_check(row)["adjusted_yoy_token_passed"]
    row["adjusted_price_yoy"] = -3.20
    assert not source._trace_check(row)["adjusted_yoy_token_passed"]
