"""官方周报正文解析的合成检验；只调用parse_article，不访问网络或行情。"""

from html import escape

import pytest

from scripts.v2_p7.build_hog_inputs import parse_article

URL = "https://xmsyj.moa.gov.cn/jcyj/202503/t20250325_1234567.htm"
TITLE = "3月份第3周畜产品和饲料集贸市场价格情况"
PRICES = "全国生猪平均价格为15.25元/公斤。育肥猪配合饲料平均价格为3.50元/公斤。"


def article(
    body: str,
    publication: str = "2025-03-25 10:00:00",
    title: str = TITLE,
) -> str:
    return (
        f'<html><head><meta name="ArticleTitle" content="{escape(title)}">'
        f'<meta name="PubDate" content="{escape(publication)}"></head>'
        f'<body><div class="TRS_Editor">{body}</div></body></html>'
    )


@pytest.mark.parametrize("name", ["活猪", "生猪"])
@pytest.mark.parametrize("slash", ["/", "／"])
def test_old_and_new_hog_names_extract_the_same_per_kg_quote(name: str, slash: str) -> None:
    body = (
        "采集日为3月20日。"
        f"全国{name}平均价格为15.25元{slash}公斤。"
        f"育肥猪配合饲料平均价格为3.50元{slash}公斤。"
    )
    row = parse_article(article(body), URL)
    assert row["hog"] == 15.25 and row["feed"] == 3.5
    assert row["unit"] == "CNY/kg"
    assert row["observation_date"] == "20250320"
    assert row["publication_date"] == "20250325"
    assert row["publication_timestamp_label"] == "2025-03-25 10:00:00"
    assert row["source_id"] == "t20250325_1234567" and row["source_url"] == URL
    assert row["quality_usable"] and not row["first_publication_immutable_proven"]
    assert row["source_text"] == body


def test_optional_average_and_is_words_and_html_spacing() -> None:
    html = article(
        "<p>采集日3月20日。</p><p>全国 活猪 价格15.25元 / 公斤。</p>"
        "<p>育肥猪配合饲料价格3.50元/公斤。</p>"
    )
    row = parse_article(html, URL)
    assert row["hog"] == 15.25 and row["feed"] == 3.5


@pytest.mark.parametrize("date_text", ["12月30日", "2024年12月30日"])
def test_cross_year_observation_uses_prior_year(date_text: str) -> None:
    row = parse_article(article(f"采集日为{date_text}。{PRICES}", "2025-01-03 09:30:00"), URL)
    assert row["observation_date"] == "20241230"
    assert row["publication_date"] == "20250103"


def test_explicit_current_year_is_preserved() -> None:
    row = parse_article(article(f"采集日为2025年3月20日。{PRICES}"), URL)
    assert row["observation_date"] == "20250320"


@pytest.mark.parametrize("date_text", ["3月26日", "2026年3月20日", "2019年3月20日"])
def test_future_or_explicit_old_year_cannot_be_reinterpreted(date_text: str) -> None:
    with pytest.raises(ValueError, match="采集日期"):
        parse_article(article(f"采集日为{date_text}。{PRICES}"), URL)


def test_observation_older_than_35_days_is_not_guessed() -> None:
    with pytest.raises(ValueError, match="35日"):
        parse_article(article(f"采集日为3月20日。{PRICES}", "2025-04-25 00:00:00"), URL)
    row = parse_article(article(f"采集日为3月21日。{PRICES}", "2025-04-25 00:00:00"), URL)
    assert row["observation_date"] == "20250321"


@pytest.mark.parametrize(
    "body",
    [
        PRICES,
        f"采集日为3月19日。采集日为3月20日。{PRICES}",
        f"采集日为2024年3月20日。采集日为2025年3月20日。{PRICES}",
        f"采集日为2月30日。{PRICES}",
    ],
)
def test_missing_invalid_or_conflicting_observation_dates_are_rejected(body: str) -> None:
    with pytest.raises(ValueError, match="采集日期"):
        parse_article(article(body), URL)


@pytest.mark.parametrize("repeat_date", ["3月20日", "2025年3月20日"])
def test_identical_repeated_dates_and_quotes_are_one_observation(repeat_date: str) -> None:
    body = f"采集日为3月20日。{PRICES}采集日为{repeat_date}。{PRICES}全国活猪价格15.25元/公斤。"
    row = parse_article(article(body), URL)
    assert row["hog"] == 15.25 and row["feed"] == 3.5


@pytest.mark.parametrize(
    "extra",
    ["全国活猪平均价格15.26元/公斤。", "育肥猪配合饲料平均价格3.51元/公斤。"],
)
def test_conflicting_repeated_prices_are_not_silently_selected(extra: str) -> None:
    with pytest.raises(ValueError, match="价格不唯一"):
        parse_article(article(f"采集日为3月20日。{PRICES}{extra}"), URL)


@pytest.mark.parametrize(
    "prices",
    [
        "全国猪肉平均价格25元/公斤。育肥猪配合饲料平均价格3.5元/公斤。",
        "河北省活猪平均价格15元/公斤。育肥猪配合饲料平均价格3.5元/公斤。",
        "全国仔猪平均价格40元/公斤。育肥猪配合饲料平均价格3.5元/公斤。",
        "全国生猪平均价格15元/公斤。仔猪配合饲料平均价格3.5元/公斤。",
        "全国生猪平均价格15250元/吨。育肥猪配合饲料平均价格3.5元/公斤。",
        "全国生猪平均价格15元/公斤。育肥猪配合饲料平均价格3500元/吨。",
        "全国生猪平均价格0元/公斤。育肥猪配合饲料平均价格3.5元/公斤。",
    ],
)
def test_missing_national_hog_or_feed_and_wrong_units_are_rejected(prices: str) -> None:
    with pytest.raises(ValueError, match="价格不唯一或非法"):
        parse_article(article(f"采集日为3月20日。{prices}"), URL)


def test_local_hog_and_national_pork_do_not_override_national_hog() -> None:
    body = (
        f"采集日为3月20日。全国猪肉平均价格25元/公斤。{PRICES}"
        "河北活猪平均价格16元/公斤。辽宁生猪平均价格14元/公斤。"
    )
    row = parse_article(article(body), URL)
    assert row["hog"] == 15.25


@pytest.mark.parametrize(
    "html",
    [
        article(f"采集日为3月20日。{PRICES}", publication=""),
        article(f"采集日为3月20日。{PRICES}", title="粮食市场情况"),
        article(f"采集日为3月20日。{PRICES}").replace('class="TRS_Editor"', 'class="other"'),
    ],
)
def test_missing_article_metadata_or_body_is_rejected(html: str) -> None:
    with pytest.raises(ValueError, match="缺正文/公开标签或不是畜产品周报"):
        parse_article(html, URL)


def test_national_slaughter_ready_hog_name_is_accepted() -> None:
    body = (
        "采集日为3月20日。全国出栏生猪平均价格17.45元/公斤。"
        "全国猪肉平均价格25元/公斤。育肥猪配合饲料平均价格3.50元/公斤。"
    )
    assert parse_article(article(body), URL)["hog"] == 17.45


@pytest.mark.parametrize(
    "clause,expected",
    [
        (
            "育肥猪配合饲料、肉鸡配合饲料、蛋鸡配合饲料平均价格分别为"
            "3.61元/公斤、3.63元/公斤、3.35元/公斤",
            3.61,
        ),
        (
            "育肥猪配合饲料、肉鸡配合饲料和蛋鸡配合饲料平均价格分别为"
            "3.07元/公斤、3.16元/公斤和2.89元/公斤",
            3.07,
        ),
        (
            "全国肉鸡配合饲料、育肥猪配合饲料及蛋鸡配合饲料价格分别为"
            "3.63元／公斤、3.61元／公斤及3.35元／公斤",
            3.61,
        ),
        (
            "仔猪配合饲料、蛋鸡配合饲料和育肥猪配合饲料平均价格分别为"
            "4.10元/公斤、3.35元/公斤和3.61元/公斤",
            3.61,
        ),
    ],
)
def test_parallel_feed_quotes_follow_named_positions(clause: str, expected: float) -> None:
    body = f"采集日为3月20日。全国活猪平均价格15.25元/公斤。{clause}，与上周持平。"
    row = parse_article(article(body), URL)
    assert row["feed"] == expected and row["hog"] == 15.25
    assert row["observation_date"] == "20250320" and row["publication_date"] == "20250325"


@pytest.mark.parametrize(
    "clause",
    [
        "育肥猪配合饲料、肉鸡配合饲料平均价格分别为3.61元/公斤",
        "育肥猪配合饲料、肉鸡配合饲料平均价格分别为3.61元/公斤、3.63元/公斤、3.35元/公斤",
        "育肥猪配合饲料和育肥猪配合饲料平均价格分别为3.61元/公斤和3.61元/公斤",
        "母猪配合饲料、育肥猪配合饲料、肉鸡配合饲料平均价格分别为3.61元/公斤、3.63元/公斤",
        "育肥猪配合饲料和肉鸡配合饲料平均价格分别为3.61元/只和3.63元/公斤",
        "育肥猪配合饲料和肉鸡配合饲料平均价格分别为3.61和3.63元/公斤",
        "育肥猪配合饲料和肉鸡配合饲料平均价格分别为0元/公斤和3.63元/公斤",
        "河北育肥猪配合饲料和肉鸡配合饲料平均价格分别为3.61元/公斤和3.63元/公斤",
    ],
)
def test_malformed_parallel_quotes_cannot_be_positionally_guessed(clause: str) -> None:
    with pytest.raises(ValueError, match="并列饲料"):
        parse_article(article(f"采集日为3月20日。全国活猪价格15元/公斤。{clause}。"), URL)


@pytest.mark.parametrize("direct,raises", [(3.61, False), (3.60, True)])
def test_direct_and_parallel_feed_conflicts_are_rejected(direct: float, raises: bool) -> None:
    body = (
        f"采集日为3月20日。全国活猪价格15元/公斤。育肥猪配合饲料平均价格{direct}元/公斤。"
        "肉鸡配合饲料和育肥猪配合饲料平均价格分别为3.63元/公斤和3.61元/公斤。"
    )
    if raises:
        with pytest.raises(ValueError, match="全国feed价格不唯一"):
            parse_article(article(body), URL)
    else:
        assert parse_article(article(body), URL)["feed"] == direct


def test_parallel_chicken_and_piglet_prices_are_not_fattening_feed() -> None:
    body = (
        "采集日为3月20日。全国生猪价格15元/公斤。"
        "仔猪配合饲料和肉鸡配合饲料平均价格分别为4.10元/公斤和3.63元/公斤。"
    )
    with pytest.raises(ValueError, match="全国feed价格不唯一"):
        parse_article(article(body), URL)


def test_conflicting_national_hog_aliases_are_rejected() -> None:
    body = f"采集日为3月20日。{PRICES}全国出栏生猪平均价格17.45元/公斤。"
    with pytest.raises(ValueError, match="全国hog价格不唯一"):
        parse_article(article(body), URL)


def test_ambiguous_table_is_not_a_fallback_for_missing_text_quote() -> None:
    body = (
        "采集日为3月20日。全国生猪价格15元/公斤。"
        "<table><tr><td>单位：元／公斤、元／只</td></tr>"
        "<tr><td>项目</td><td>上周</td><td>本周</td></tr>"
        "<tr><td>育肥猪配合饲料</td><td>3.61</td><td>3.63</td></tr></table>"
    )
    with pytest.raises(ValueError, match="全国feed价格不唯一"):
        parse_article(article(body), URL)
