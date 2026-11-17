"""农业农村部生猪周报的只读研究采集；不访问证券行情或生产写域。

复杂度豁免：阶段专用资料取证，不入生产链路（CLAUDE §7.8）；
如转为长期生产采集，须先拆分目录发现、原件获取、解析与证据发布模块。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://xmsyj.moa.gov.cn/jcyj/"
API_URL = "https://api.so-gov.cn/query/"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _parallel_feed_prices(text: str, url: str) -> set[float]:
    """仅按完整命名顺序配对并列价格；数量、名称或单位不明即拒绝。"""
    allowed = {"育肥猪配合饲料", "肉鸡配合饲料", "蛋鸡配合饲料", "仔猪配合饲料"}
    values = set()
    for clause in re.split(r"[。；;，,]", text):
        if "育肥猪配合饲料" not in clause:
            continue
        parts = re.split(r"(?:平均)?价格分别(?:为)?", clause, maxsplit=1)
        if len(parts) != 2:
            continue
        names = re.split(r"[、和及]", parts[0].removeprefix("全国"))
        prices = re.split(r"[、和及]", parts[1])
        if (
            len(names) < 2
            or len(names) != len(prices)
            or len(names) != len(set(names))
            or not set(names).issubset(allowed)
        ):
            raise ValueError(f"并列饲料名称与价格不能唯一配对：{url}/{clause}")
        parsed = []
        for price in prices:
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)元[／/]公斤", price)
            if match is None:
                raise ValueError(f"并列饲料价格单位或数值不明：{url}/{clause}")
            value = float(match.group(1))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"并列饲料价格必须正且有限：{url}/{clause}")
            parsed.append(value)
        values.add(parsed[names.index("育肥猪配合饲料")])
    return values


def parse_article(html: str, url: str) -> dict:
    """只解析同页全国两价；历史网页时间为公开标签证据，不称不可修订档案。"""
    soup = BeautifulSoup(html, "html.parser")
    metadata = {m.get("name"): m.get("content", "") for m in soup.find_all("meta")}
    title = metadata.get("ArticleTitle", "")
    publication = metadata.get("PubDate", "")
    body = soup.select_one(".TRS_Editor")
    if body is None or not publication or "畜产品" not in title:
        raise ValueError(f"缺正文/公开标签或不是畜产品周报：{url}")
    text = re.sub(r"\s+", "", body.get_text(" ", strip=True))
    pub = datetime.fromisoformat(publication)
    dates = re.findall(r"采集日(?:为)?(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日", text)
    observations = set()
    for explicit_year, month, day in set(dates):
        # 原文给出年份时必须尊重原文；仅省略年份时才推断跨年日期。
        years = (int(explicit_year),) if explicit_year else (pub.year - 1, pub.year)
        possibilities = []
        for year in years:
            try:
                value = datetime(year, int(month), int(day))
            except ValueError:
                continue
            if 0 <= (pub - value).days <= 35:
                possibilities.append(value)
        if len(possibilities) != 1:
            raise ValueError(f"采集日期不能在发布前35日内唯一定位：{url}/{dates}")
        observations.add(possibilities[0])
    if len(observations) != 1:
        raise ValueError(f"采集日期不唯一：{url}/{dates}")
    observation = observations.pop()
    fields = {}
    patterns = {
        "hog": r"全国(?:生猪|活猪|出栏生猪)(?:平均)?价格(?:为)?([0-9]+(?:\.[0-9]+)?)元[／/]公斤",
        "feed": r"育肥猪配合饲料(?:平均)?价格(?:为)?([0-9]+(?:\.[0-9]+)?)元[／/]公斤",
    }
    for name, pattern in patterns.items():
        values = {float(value) for value in re.findall(pattern, text)}
        if name == "feed":
            values.update(_parallel_feed_prices(text, url))
        if len(values) != 1 or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"全国{name}价格不唯一或非法：{url}/{values}")
        fields[name] = values.pop()
    return {
        "source_id": urlparse(url).path.rsplit("/", 1)[-1].replace(".htm", ""),
        "source_url": url,
        "title": title,
        "publication_timestamp_label": publication,
        "publication_date": pub.strftime("%Y%m%d"),
        "observation_date": observation.strftime("%Y%m%d"),
        "unit": "CNY/kg",
        "series_id": "moa_500_counties_hog_complete_feed_v1",
        "quality_usable": True,
        "first_publication_immutable_proven": False,
        "source_text": text,
        **fields,
    }


def collect(output: Path, *, max_seconds: float = 1100, catalog_cache: Path | None = None) -> dict:
    """公开栏目加公开检索同标题展开；每个响应留原件，禁止覆盖旧批次。"""
    allowed = (ROOT / "data/reports/p7_hog_feasibility_20261116").resolve()
    output = output.resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("采集输出必须为本轮资料根下的新子目录")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("catalog", "search", "articles"):
        (output / name).mkdir()
    (output / "methods").mkdir()
    (output / "methods" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    errors, items = [], {}

    def guard() -> None:
        if time.perf_counter() - started > max_seconds:
            raise TimeoutError("资料批达到固定机器预算")

    def get(url: str, path: Path) -> bytes:
        guard()
        if catalog_cache is not None and path.parent.name == "catalog":
            cached = catalog_cache / path.name
            if cached.is_file():
                content = cached.read_bytes()
                path.write_bytes(content)
                return content
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        path.write_bytes(response.content)
        return response.content

    def query(endpoint: str, params: dict, path: Path) -> dict:
        guard()
        response = requests.post(API_URL + endpoint, data=params, timeout=25)
        response.raise_for_status()
        path.write_bytes(response.content)
        value = response.json()
        if value.get("ok") is False:
            raise ValueError(f"公开搜索返回失败：{endpoint}/{value.get('code')}")
        return value

    try:
        # 栏目是持续更新入口，检索是旧文发现入口；不得把检索的标题折叠当完整历史。
        for page in range(25):
            suffix = "" if page == 0 else f"index_{page}.htm"
            url = BASE_URL + suffix
            content = get(url, output / "catalog" / f"{page:02d}.html")
            soup = BeautifulSoup(content.decode("utf-8"), "html.parser")
            for a in soup.find_all("a"):
                title = a.get_text(strip=True)
                if "畜产品和饲料" in title:
                    target = urljoin(url, a.get("href", ""))
                    items[target] = {"url": target, "title": title, "discovered": "catalog"}
        print(f"栏目完成：{len(items)}篇，{time.perf_counter()-started:.1f}s", flush=True)
        search_docs = {}
        for page in range(1, 21):
            params = {
                "siteCode": "xmsyj_moa",
                "qt": "畜产品和饲料",
                "page": page,
                "pageSize": 20,
                "sort": "dateDesc",
            }
            response = query("s", params, output / "search" / f"page_{page:02d}.json")
            docs = response.get("resultDocs", [])
            if not docs:
                break
            before = len(search_docs)
            for doc in docs:
                data = doc["data"]
                search_docs[data["id"]] = data
            if len(search_docs) == before:
                raise ValueError("公开搜索翻页没有新增结果，不能视为完整")
        else:
            raise ValueError("搜索分页超过预算，未到空页")
        if len(search_docs) != response.get("totalHits", len(search_docs)):
            raise ValueError("搜索总量与逐页唯一条数不符")
        print(f"检索标题组：{len(search_docs)}，开始展开历史同名周报", flush=True)
        for i, doc in enumerate(search_docs.values()):
            items.setdefault(
                doc["url"], {"url": doc["url"], "title": doc["titleO"], "discovered": "search"}
            )
            if doc.get("combins", 1) <= 1:
                continue
            seen = set()
            for page in range(1, 5):
                params = {
                    "siteCode": "xmsyj_moa",
                    "tab": "all",
                    "qt": "畜产品和饲料",
                    "reference": doc["id"],
                    "combinField": "MD5TITLE",
                    "combinValue": doc["md5Title"],
                    "page": page,
                    "pageSize": 20,
                }
                data = query("combin", params, output / "search" / f"group_{i:03d}_{page}.json")
                for row in data.get("docs", []):
                    if row["url"] in seen:
                        raise ValueError("同标题展开出现重复页")
                    seen.add(row["url"])
                    if "2018-12-01" <= row["displayTime"] <= "2026-11-16":
                        items.setdefault(
                            row["url"], {**row, "discovered": "search_title_expansion"}
                        )
                if len(seen) >= data.get("total", 0):
                    break
            else:
                raise ValueError("同标题展开未完整结束")
            if i % 20 == 0:
                print(
                    f"展开 {i}/{len(search_docs)}，候选原页 {len(items)}，"
                    f"{time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        selected = [
            row
            for url, row in sorted(items.items())
            if urlparse(url).hostname == "xmsyj.moa.gov.cn" and "畜产品和饲料" in row["title"]
        ]
        save_json(output / "catalog.json", selected)

        def article(item: dict) -> dict:
            url = item["url"]
            path = output / "articles" / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".html")
            try:
                content = get(url, path)
                parsed = parse_article(content.decode("utf-8"), url)
                return {
                    **parsed,
                    "raw_path": path.relative_to(ROOT).as_posix(),
                    "sha256": digest(path),
                    "catalog_record": item,
                }
            except (requests.RequestException, ValueError, UnicodeDecodeError, TimeoutError) as exc:
                return {
                    "source_url": url,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "raw_path": path.relative_to(ROOT).as_posix() if path.exists() else None,
                }

        rows = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, row in enumerate(pool.map(article, selected), 1):
                (errors if "reason" in row else rows).append(row)
                if i % 40 == 0:
                    print(
                        f"原页 {i}/{len(selected)}，可解析 {len(rows)}，异常 {len(errors)}，"
                        f"{time.perf_counter()-started:.1f}s",
                        flush=True,
                    )
        save_json(output / "records_raw.json", rows)
        save_json(output / "exclusions.json", errors)
        summary = {
            "catalog_count": len(selected),
            "parsed_count": len(rows),
            "error_count": len(errors),
            "machine_seconds": time.perf_counter() - started,
            "data_purchase_yuan": 0,
            "stock_returns_read": False,
        }
        save_json(output / "summary.json", summary)
        return summary
    except (requests.RequestException, ValueError, TimeoutError, KeyError) as exc:
        save_json(
            output / "failure.json",
            {
                "type": type(exc).__name__,
                "reason": str(exc),
                "machine_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.output, catalog_cache=args.catalog_cache), ensure_ascii=False))
