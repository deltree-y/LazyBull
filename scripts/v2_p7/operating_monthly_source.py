"""P7月度经营研究专用：公开公告取数与保守解析，不访问行情或生产目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/reports/p7_operating_monthly_economic_20261116/input_announcements"
COMPANIES = {
    "002352": "9900010448",
    "002120": "9900002261",
    "002468": "9900014251",
    "600233": "gssh0600233",
}
API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}
NUMBER = r"[+\-−]?\d[\d,]*(?:\.\d+)?"


def write_json(path: Path, data: Any) -> None:
    """保存可追溯的研究证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def period_from_title(title: str) -> tuple[int, list[int]] | None:
    """仅从明确月报标题取报告期，保留合并月份而不拆造月值。"""
    title = re.sub(r"<[^>]+>|\s+", "", title)
    match = re.search(r"(20\d{2})年(\d{1,2})(?:月?[至—–~、和及-](\d{1,2}))?月份?", title)
    if not match:
        return None
    year, first, last = int(match[1]), int(match[2]), int(match[3] or match[2])
    if not 1 <= first <= last <= 12:
        raise ValueError(f"月报标题非法：{title}")
    return year, list(range(first, last + 1))


def _row(text: str, label: str) -> dict[str, Any] | None:
    """提取具有单位和明确同比百分号的表格行，未知格式返回空。"""
    flat = re.sub(r"\s+", " ", text)
    flat = re.sub(r"\s+([%％])", r"\1", flat)
    pattern = rf"(?:{label})\s*[（(]([^）)]+)[）)]\s*([+\-−\d.,%％\s—–]+)"
    found = re.search(pattern, flat)
    if not found:
        return None

    def convert(value: str) -> float:
        return float(value.replace(",", "").replace("−", "-"))

    tokens = re.findall(rf"{NUMBER}[%％]?|[—–-]", found[2])
    if len(tokens) < 2 or tokens[0] in ("-", "—", "–"):
        return None
    has_prior = len(tokens) >= 3 and not tokens[1].endswith(("%", "％"))
    yoy_index = 2 if has_prior else 1

    def percent_at(index: int) -> float | None:
        if len(tokens) <= index or not tokens[index].endswith(("%", "％")):
            return None
        return convert(tokens[index][:-1])

    return {
        "value": convert(tokens[0]),
        "unit": found[1],
        "prior_year_value": convert(tokens[1]) if has_prior else None,
        "yoy_pct": percent_at(yoy_index),
        "additional_yoy_pct": percent_at(yoy_index + 1),
        "evidence": found[0],
    }


def _narrative_row(text: str, label: str, unit: str) -> dict[str, Any] | None:
    """保留额外合并期初步核算公告的明示值，不构造未披露单票价。"""
    flat = re.sub(r"\s+", "", text)
    found = re.search(rf"(?:{label})约?({NUMBER}){unit}，同比(增长|下降)({NUMBER})%", flat)
    if not found:
        return None
    value = float(found[1].replace(",", ""))
    yoy = float(found[3].replace(",", "")) * (-1 if found[2] == "下降" else 1)
    return {
        "value": value,
        "unit": unit,
        "prior_year_value": None,
        "yoy_pct": yoy,
        "evidence": found[0],
    }


def _sf_adjustment(result: dict[str, Any], flat: str, price: dict, volume: dict) -> None:
    """丰网剥离后的公司明示同口径数据。"""
    adjusted = price.get("additional_yoy_pct")
    adjusted_volume = volume.get("additional_yoy_pct")
    if adjusted_volume is None:
        match = re.search(
            r"不含丰网.{0,80}?业务量同比(增长|增加|下降|减少)为?([+\-]?\d+(?:\.\d+)?)%", flat
        )
        if match:
            adjusted_volume = float(match[2])
            if match[1] in ("下降", "减少"):
                adjusted_volume = -abs(adjusted_volume)
    result["semantic_flags"].append("丰网转让；采用公司表内不含丰网同口径价量同比")
    result.update(
        adjusted_price_yoy=adjusted,
        adjusted_volume_yoy=adjusted_volume,
        volume_yoy=adjusted_volume,
        price_comparable=adjusted is not None,
        adjustment_evidence=price.get("evidence"),
    )
    if adjusted is None:
        result["semantic_flags"].append("丰网调整单票同比缺失，禁止回退原表")


def _cainiao_adjustment(result: dict[str, Any], flat: str) -> None:
    """仅接受公司直接披露的剔除后同比；定性持平不补零。"""
    flags = result["semantic_flags"]
    flags.append("菜鸟业务结算调整")
    match = re.search(
        r"(?:剔除|还原|扣除).{0,100}?单票(?:快递服务|快递产品)?收入.{0,45}?"
        r"同比(?:增长|增加|提升|上升|下降|降低|减少)?[为约]?([+\-−]?\d+(?:\.\d+)?)[%％]",
        flat,
    )
    if match:
        adjusted = float(match[1].replace("−", "-"))
        if any(x in match[0].split("同比")[-1] for x in ("下降", "降低", "减少")):
            adjusted = -abs(adjusted)
        result.update(adjusted_price_yoy=adjusted, adjustment_evidence=match[0])
        flags.append("采用公司披露剔除影响单票同比")
    else:
        result["price_comparable"] = False
        flags.append("公司调整单票同比未解析，禁止用原表替代")
        if "同比基本持平" in flat:
            result["adjusted_price_yoy_text"] = "同比基本持平"
            flags.append("调整后同比仅定性披露，不填造为0")


def _semantics(code: str, year: int, text: str, price: dict, volume: dict) -> dict[str, Any]:
    """独立生成原表、调整和策略可用值，不覆盖原始字段。"""
    flat = re.sub(r"\s+", "", text)
    reported = price.get("yoy_pct")
    flags = ["披露日粒度，首次公开时刻未知", "未经审计"]
    result = {
        "price_yoy_pct": reported,
        "reported_price_yoy": reported,
        "adjusted_price_yoy": None,
        "adjusted_price_yoy_text": None,
        "price_comparable": reported is not None,
        "semantic_flags": flags,
        "reported_volume_yoy": volume.get("yoy_pct"),
        "adjusted_volume_yoy": None,
        "volume_yoy": volume.get("yoy_pct"),
        "adjustment_evidence": None,
    }
    if code == "002120" and year == 2019:
        flags.append("2019新增派费，收入和单票同比不可比")
        result["price_comparable"] = False
    if code == "002352" and year == 2019:
        flags.append("供应链另列；不得用总收入除速运件量")
    if "重述" in flat:
        flags.append("本版本重述上年同期，不回写早期版本")
    if "嘉里" in flat:
        flags.append("供应链国际受嘉里并表影响；使用速运行")
    if code == "002352" and "不含丰网" in flat:
        _sf_adjustment(result, flat, price, volume)
    if "菜鸟" in flat and any(x in flat for x in ("调整", "剔除", "还原", "扣除")):
        _cainiao_adjustment(result, flat)
    adjusted = result["adjusted_price_yoy"]
    if code in ("002468", "600233") and year == 2022 and adjusted is None:
        result["price_comparable"] = False
        flags.append("2022业务调整可比性未确认")
    result["canonical_price_yoy"] = None
    if result["price_comparable"]:
        result["canonical_price_yoy"] = adjusted if adjusted is not None else reported
    return result


def parse_announcement(text: str, meta: dict[str, Any]) -> dict[str, Any] | None:
    """解析原文并标记口径；不能证明可比时不输出策略可用单票同比。"""
    title = re.sub(r"<[^>]+>", "", meta["announcementTitle"])
    period = period_from_title(title)
    if period is None:
        return None
    year, months = period
    if year < 2019 or year > 2026 or (year == 2026 and max(months) > 9):
        return None
    date = datetime.fromtimestamp(meta["announcementTime"] / 1000, timezone(timedelta(hours=8)))
    code = meta["secCode"]
    revenue = _row(text, "快递服务业务收入|快递产品收入|快递业务收入|营业收入")
    volume = _row(text, "完成业务量|业务完成量|快递产品业务量|快递业务量|业务量")
    price = _row(text, "快递服务单票收入|快递产品单票收入|快递单票收入|单票收入")
    revenue = revenue or _narrative_row(text, "营业收入", "亿元")
    volume = volume or _narrative_row(text, "快递业务完成量", "亿票")
    semantics = _semantics(code, year, text, price or {}, volume or {})
    if len(months) > 1:
        semantics["semantic_flags"].append("合并月份，不拆造单月值")
    if any(x in title for x in ("更正", "修订", "补充")):
        semantics["semantic_flags"].append("更正或修订版本，必须按本公告日生效")
    status = "parsed" if all(x is not None for x in (revenue, volume, price)) else "partial"
    return {
        "code": code,
        "ts_code": code + (".SH" if code.startswith("6") else ".SZ"),
        "operating_year": year,
        "month": max(months),
        "period_months": months,
        "publication_date": date.strftime("%Y%m%d"),
        "announcement_time_raw_ms": meta["announcementTime"],
        "announcement_id": meta["announcementId"],
        "title": title,
        "source_pdf": "https://static.cninfo.com.cn/" + meta["adjunctUrl"],
        "revenue": revenue,
        "volume": volume,
        "price": price,
        **semantics,
        "parse_status": status,
        "eligible_by_cutoff": date.strftime("%Y%m%d") <= "20261029",
    }


def query_year(code: str, year: int, deadline: float) -> dict[str, Any]:
    """按公司披露年分页到空；保存响应并检验总量，保留所有标题供审计。"""
    rows: dict[str, Any] = {}
    pages = []
    total = None
    for page in range(1, 31):
        if time.monotonic() > deadline:
            return {"code": code, "year": year, "complete": False, "reason": "time_budget"}
        path = OUTPUT / "api" / f"{code}_{year}_{page}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
        else:
            request = {
                "pageNum": page,
                "pageSize": 30,
                "column": "szse",
                "tabName": "fulltext",
                "searchkey": "经营",
                "stock": f"{code},{COMPANIES[code]}",
                "seDate": f"{year}-01-01~{year}-12-31" if year < 2026 else "2026-01-01~2026-11-16",
                "isHLtitle": "false",
            }
            time.sleep(0.3)
            response = requests.post(API, data=request, headers=HEADERS, timeout=20)
            response.raise_for_status()
            saved = {
                "request": request,
                "response": response.json(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(path, saved)
        data = saved["response"]
        batch = data.get("announcements") or []
        # 巨潮的空终页将totalAnnouncement重置为0；总量锚定首个非空页。
        if total is None or (batch and not total):
            total = data.get("totalAnnouncement", total)
        pages.append(len(batch))
        if not batch:
            break
        for row in batch:
            old = rows.get(row["announcementId"])
            if old is not None and old != row:
                raise ValueError(f"同公告ID跨页内容漂移：{code}/{year}/{row['announcementId']}")
            rows[row["announcementId"]] = row
    complete = bool(pages and pages[-1] == 0 and len(rows) == total)
    result = {
        "code": code,
        "year": year,
        "page_counts": pages,
        "reported_total": total,
        "complete": complete,
        "announcements": list(rows.values()),
    }
    write_json(OUTPUT / "catalog" / f"{code}_{year}.json", result)
    return result


def download_one(meta: dict[str, Any], deadline: float) -> dict[str, Any]:
    """下载原件且固定SHA，文本缓存不改变PDF。"""
    code, aid = meta["secCode"], meta["announcementId"]
    path = OUTPUT / "pdf" / f"{code}_{aid}.pdf"
    if time.monotonic() > deadline:
        return {"announcement_id": aid, "download_error": "time_budget"}
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        time.sleep(0.3)
        response = requests.get(
            "https://static.cninfo.com.cn/" + meta["adjunctUrl"], headers=HEADERS, timeout=25
        )
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"公告返回非PDF：{aid}")
        path.write_bytes(response.content)
    text_path = path.with_suffix(".txt")
    if text_path.exists():
        text = text_path.read_text(encoding="utf-8")
    else:
        sys.path.insert(0, str(ROOT / "temp/p7_pdf_tools_20261116"))
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        text_path.write_text(text, encoding="utf-8")
    record = parse_announcement(text, meta) or {
        "announcement_id": aid,
        "parse_status": "outside_monthly_scope",
    }
    record.update(
        {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "local_pdf": str(path.relative_to(ROOT)),
            "local_text": str(text_path.relative_to(ROOT)),
        }
    )
    return record


def _rounding_halfwidth(row: dict[str, Any]) -> float:
    """按原件展示小数位计算半个末位单位。"""
    found = re.search(rf"[）)]\s*({NUMBER})", row["evidence"])
    token = found[1] if found else str(row["value"])
    decimals = len(token.split(".")[1]) if "." in token else 0
    return 0.5 * 10 ** (-decimals)


def _ratio_check(row: dict[str, Any]) -> dict[str, Any]:
    """公告舍入区间内，营收/件量与已报单价应存在共同可行值。"""
    revenue, volume, price = row["revenue"], row["volume"], row["price"]
    if not all((revenue, volume, price)):
        return {"announcement_id": row["announcement_id"], "status": "excluded_missing_price"}

    def scale(unit: str) -> float:
        return 1e8 if "亿" in unit else (1e4 if "万" in unit else 1)

    rs, vs = scale(revenue["unit"]), scale(volume["unit"])
    rh, vh, ph = (_rounding_halfwidth(x) for x in (revenue, volume, price))
    lo = (revenue["value"] - rh) * rs / ((volume["value"] + vh) * vs)
    hi = (revenue["value"] + rh) * rs / ((volume["value"] - vh) * vs)
    valid = lo <= price["value"] + ph + 1e-10 and hi >= price["value"] - ph - 1e-10
    return {
        "announcement_id": row["announcement_id"],
        "status": "passed" if valid else "failed",
        "ratio_rounding_interval": [lo, hi],
        "reported_price": price["value"],
        "price_halfwidth": ph,
    }


def _trace_check(row: dict[str, Any]) -> dict[str, Any]:
    """逐原件核验PDF、表行及调整语句，不使用其他公告代替。"""
    text = (ROOT / row["local_text"]).read_text(encoding="utf-8")
    flat = re.sub(r"\s+", "", text)
    fields = [x for x in (row["revenue"], row["volume"], row["price"]) if x]
    rows_ok = all(re.sub(r"\s+", "", x["evidence"]) in flat for x in fields)
    adjustment = row.get("adjustment_evidence")
    adjustment_ok = not adjustment or re.sub(r"\s+", "", adjustment) in flat
    pdf_ok = hashlib.sha256((ROOT / row["local_pdf"]).read_bytes()).hexdigest() == row["sha256"]
    raw_tokens_ok = True
    for field in fields:
        tokens = {
            float(x.replace("−", "-"))
            for x in re.findall(r"([+\-−]?\d+(?:\.\d+)?)\s*[%％]", field["evidence"])
        }
        value = field["yoy_pct"]
        raw_tokens_ok = raw_tokens_ok and (value is None or value in tokens)
    adjusted = row.get("adjusted_price_yoy")
    adjusted_tokens = {
        float(x.replace("−", "-"))
        for x in re.findall(r"([+\-−]?\d+(?:\.\d+)?)\s*[%％]", adjustment or "")
    }
    adjusted_token_ok = adjusted is None or adjusted in adjusted_tokens
    if adjusted is not None and adjusted < 0:
        downward = any(
            word in (adjustment or "").split("同比")[-1] for word in ("下降", "降低", "减少")
        )
        adjusted_token_ok = adjusted_token_ok or (downward and abs(adjusted) in adjusted_tokens)
    return {
        "announcement_id": row["announcement_id"],
        "pdf_sha_passed": pdf_ok,
        "row_text_passed": rows_ok,
        "raw_yoy_token_passed": raw_tokens_ok,
        "adjustment_text_passed": adjustment_ok,
        "adjusted_yoy_token_passed": adjusted_token_ok,
    }


def _prior_sample_comparison(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对照上一轮24份人工核对数据，只核原始数值，不混同比归一化。"""
    prior_root = ROOT / "data/reports/p7_operating_monthly_feasibility_20261116"
    sf = json.loads((prior_root / "sources_sfyunda/sources.json").read_text(encoding="utf-8"))
    other = json.loads((prior_root / "sources_stoyto/sources.json").read_text(encoding="utf-8"))[
        "records"
    ]
    targets = [(r["announcement_metadata"]["announcementId"], r["metrics"]) for r in sf]
    targets += [(r["matches"][0]["announcementId"], r["numeric_values"]) for r in other]
    lookup = {r["announcement_id"]: r for r in records}
    results = []
    for aid, old in targets:
        current = lookup[aid]
        mismatches = []
        for field, old_name in [
            ("revenue", "revenue"),
            ("volume", "volume"),
            ("price", "revenue_per_parcel"),
        ]:
            previous = old[old_name]
            expected_yoy = previous.get("yoy_pct", previous.get("yoy_percent"))
            if (
                current[field]["value"] != previous["value"]
                or current[field]["yoy_pct"] != expected_yoy
            ):
                mismatches.append(field)
        results.append({"announcement_id": aid, "passed": not mismatches, "mismatches": mismatches})
    return results


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """实施方只读数据自验；所有失败显式列出，不吞并到缺失项。"""
    ratios = [_ratio_check(r) for r in records]
    traces = [_trace_check(r) for r in records]
    prior = _prior_sample_comparison(records)
    regular = [r for r in records if r["parse_status"] == "parsed"]
    expected = {
        (code, year, month)
        for code in COMPANIES
        for year in range(2019, 2027)
        for month in range(1, 13)
        if year < 2026 or month <= 9
    }
    actual = {(r["code"], r["operating_year"], r["month"]) for r in regular}
    result = {
        "regular_records": len(regular),
        "expected_company_months": len(expected),
        "missing_company_months": sorted(expected - actual),
        "extra_preliminary_records": [
            r["announcement_id"] for r in records if r["parse_status"] != "parsed"
        ],
        "ratio_checks": ratios,
        "trace_checks": traces,
        "prior_24_checks": prior,
        "all_passed": (
            actual == expected
            and len(regular) == len(expected)
            and not any(r["status"] == "failed" for r in ratios)
            and all(
                all(value for key, value in r.items() if key != "announcement_id") for r in traces
            )
            and len(prior) == 24
            and all(r["passed"] for r in prior)
        ),
    }
    write_json(OUTPUT / "validation.json", result)
    return result


def run(years: list[int], batch_name: str, seconds: int = 900) -> dict[str, Any]:
    """执行有限预算取数；部分失败落盘，禁止用失败记录填造指标。"""
    start = time.monotonic()
    deadline = start + min(seconds, 900)
    errors: list[dict[str, Any]] = []

    def query_guard(pair: tuple[str, int]) -> dict[str, Any]:
        try:
            return query_year(*pair, deadline)
        except (requests.RequestException, ValueError, OSError) as exc:
            return {"code": pair[0], "year": pair[1], "complete": False, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        catalogs = list(
            pool.map(query_guard, [(code, year) for year in years for code in COMPANIES])
        )
    annotations = {
        a["announcementId"]: a
        for c in catalogs
        for a in c.get("announcements", [])
        if period_from_title(a["announcementTitle"]) is not None
    }
    logger.info(
        "清单完成：{}公司年，{}月报公告，耗时{:.2f}s",
        len(catalogs),
        len(annotations),
        time.monotonic() - start,
    )

    def download_guard(meta: dict[str, Any]) -> dict[str, Any]:
        try:
            return download_one(meta, deadline)
        except (requests.RequestException, ValueError, OSError) as exc:
            return {"announcement_id": meta["announcementId"], "download_error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download_guard, annotations.values()))
    errors.extend(r for r in records if "download_error" in r)
    records = [r for r in records if "operating_year" in r]
    summary = {
        "batch_name": batch_name,
        "years": years,
        "catalogs_complete": sum(c.get("complete", False) for c in catalogs),
        "catalog_count": len(catalogs),
        "announcements": len(annotations),
        "records": len(records),
        "parsed": sum(r["parse_status"] == "parsed" for r in records),
        "elapsed_seconds": time.monotonic() - start,
        "errors": errors,
        "read_returns": False,
        "purchase_cost": 0,
    }
    write_json(OUTPUT / f"records_{batch_name}.json", records)
    write_json(OUTPUT / f"summary_{batch_name}.json", summary)
    write_json(OUTPUT / f"catalogs_{batch_name}.json", catalogs)
    if batch_name.startswith("final"):
        records.sort(
            key=lambda row: (
                row["code"],
                row["operating_year"],
                row["month"],
                row["publication_date"],
                row["announcement_id"],
            )
        )
        write_json(OUTPUT / "records.json", records)
        validate_records(records)
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        evidence_files = [
            {
                "path": str(path.relative_to(OUTPUT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for folder in ("api", "pdf", "catalog")
            for path in sorted((OUTPUT / folder).glob("*"))
            if path.is_file()
        ]
        write_json(
            OUTPUT / "source_manifest.json",
            {
                "source_script_sha256": script_sha,
                "records_sha256": hashlib.sha256(
                    (OUTPUT / "records.json").read_bytes()
                ).hexdigest(),
                "files": evidence_files,
                "query_cutoff": "20261116",
                "signal_cutoff": "20261029",
                "all_catalogs_complete": all(c.get("complete", False) for c in catalogs),
                "known_limitations": [
                    "平台披露日非首次公开时刻",
                    "仅经营标题目录，非全公告更正穷尽证明",
                    "原件为本次抓取版本，不证明历史内容未被源站替换",
                ],
            },
        )
    logger.info("取数完成：{}", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    run(args.years, args.batch_name, args.seconds)
