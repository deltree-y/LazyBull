"""月经营研究的公司行为输入副本；按已核原件的明确行映射消除重复表达。

只准备独立输入，不改 raw、公司行为桥、交易或信号。未改分区逐字节复制；
被修订分区仅删除明确列举的组件行，保留已存在的合计行，不推算分红金额。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook, _day


class DividendResolutionError(ValueError):
    """原件、修订行集合或副本与明确登记不一致。"""


def file_sha256(path: Path) -> str:
    """计算文件原始字节 SHA。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    raise DividendResolutionError(f"原始字段类型不支持稳定序列化：{type(value).__name__}")


def row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """全部原始字段参与行身份；缺失统一为 JSON null，不混入来源路径。"""
    return {str(key): _scalar(value) for key, value in sorted(row.items())}


def row_sha256(row: Mapping[str, Any]) -> str:
    """按 UTF-8、键排序、紧凑 JSON 计算全字段行 SHA。"""
    encoded = json.dumps(
        row_payload(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence(items: Sequence[Mapping[str, Any]], ex_date: str) -> list[dict[str, Any]]:
    if not items:
        raise DividendResolutionError("明确修订缺少原始公告证据")
    checked = []
    for item in items:
        path = Path(item["local_path"])
        publication = str(item["publication_date"])
        if _day(publication) != publication or publication > ex_date:
            raise DividendResolutionError("公告公开日期无效或迟于除权日")
        if not str(item["source_url"]).startswith(("https://", "http://")):
            raise DividendResolutionError("原件 source_url 不是 HTTP(S) 地址")
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise DividendResolutionError(f"公告原件 SHA 不匹配：{path}")
        checked.append({**dict(item), "local_path": str(path.resolve())})
    return checked


def _matching_rows(
    frames: Mapping[str, pd.DataFrame], resolution: Mapping[str, Any]
) -> list[dict[str, Any]]:
    found = []
    for name, frame in frames.items():
        mask = frame["ts_code"].eq(resolution["ts_code"]) & frame["div_proc"].eq("实施")
        selected = frame.loc[mask]
        for index, row in selected.iterrows():
            payload = row_payload(row.to_dict())
            known = max(_day(payload["ann_date"]), _day(payload["imp_ann_date"]))
            if _day(payload["ex_date"]) != resolution["ex_date"]:
                continue
            if known != resolution["known_date"]:
                continue
            found.append(
                {
                    "partition": name,
                    "row_index": int(index),
                    "row_sha256": row_sha256(payload),
                    "source_fields": payload,
                }
            )
    return found


def _validate_resolution(
    frames: Mapping[str, pd.DataFrame], resolution: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "resolution_id",
        "ts_code",
        "ex_date",
        "known_date",
        "expected_row_sha256s",
        "retained_row_sha256",
        "confirmed_fields",
        "evidence",
    }
    if not required.issubset(resolution):
        raise DividendResolutionError(f"修订缺字段：{sorted(required - set(resolution))}")
    for name in ("ex_date", "known_date"):
        if _day(resolution[name]) != resolution[name]:
            raise DividendResolutionError(f"修订日期必须 YYYYMMDD：{name}")
    if resolution["known_date"] > resolution["ex_date"]:
        raise DividendResolutionError("不得用除权后新公告改写历史可知版本")
    evidence = _evidence(resolution["evidence"], resolution["ex_date"])
    rows = _matching_rows(frames, resolution)
    expected = list(resolution["expected_row_sha256s"])
    actual = [row["row_sha256"] for row in rows]
    if len(expected) < 2 or len(expected) != len(set(expected)):
        raise DividendResolutionError("预期组件/合计行 SHA 必须至少两个且互不重复")
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise DividendResolutionError("原始行集合与明确登记不一致，拒绝自动取 max/sum")
    retained = [row for row in rows if row["row_sha256"] == resolution["retained_row_sha256"]]
    if len(retained) != 1:
        raise DividendResolutionError("必须明确保留唯一一条现有合计行")
    facts = resolution["confirmed_fields"]
    fact_fields = {"cash_div", "cash_div_tax", "stk_div", "record_date", "pay_date"}
    if not fact_fields.issubset(facts):
        raise DividendResolutionError(f"原件确认字段不足：{sorted(fact_fields - set(facts))}")
    row = retained[0]["source_fields"]
    if any(name not in row or row[name] != _scalar(value) for name, value in facts.items()):
        raise DividendResolutionError("保留行与公告确认事实不一致")
    return {
        **dict(resolution),
        "evidence": evidence,
        "rows": rows,
        "removed_row_sha256s": [sha for sha in actual if sha != resolution["retained_row_sha256"]],
    }


def _book_audit(book: NativeCorporateBook, codes: set[str]) -> dict[str, Any]:
    problems = [
        {"effective_day": day, **row}
        for day, rows in book.problems.items()
        for row in rows
        if row["ts_code"] in codes
    ]
    undated = [row for row in book.undated_problems if row["ts_code"] in codes]
    events = [
        {
            "event_id": event.event_id,
            "ts_code": event.ts_code,
            "ex_date": event.ex_date,
            "cash_per_share": event.cash_per_share,
            "share_ratio": event.share_ratio,
            "issues": book.details[event.event_id]["issues"],
        }
        for event in book._events
        if event.ts_code in codes
    ]
    return {
        "problems": problems,
        "undated_problems": undated,
        "events": events,
        "event_count": len(events),
        "unresolved_problem_count": len(problems)
        + len(undated)
        + sum(bool(event["issues"]) for event in events),
    }


def _audit_subset(
    frames: Mapping[str, pd.DataFrame], output: Path, calendar: Sequence[str], codes: set[str]
) -> dict[str, Any]:
    output.mkdir()
    subset = pd.concat(
        [frame.loc[frame["ts_code"].isin(codes)] for frame in frames.values()], ignore_index=True
    )
    subset.to_parquet(output / "subset.parquet", index=False)
    book = NativeCorporateBook(calendar=calendar, dividend_dir=output)
    result = _book_audit(book, codes)
    result["subset_file_sha256"] = file_sha256(output / "subset.parquet")
    result["subset_rows"] = len(subset)
    return result


def _write_copy(
    raw_dir: Path,
    output: Path,
    frames: Mapping[str, pd.DataFrame],
    removed: Mapping[str, set[int]],
    source_sha: Mapping[str, str],
) -> list[dict[str, Any]]:
    output.mkdir()
    files = []
    for name, sha in source_sha.items():
        source, target = raw_dir / name, output / name
        dropped = sorted(removed.get(name, set()))
        if dropped:
            expected = frames[name].drop(index=dropped).reset_index(drop=True)
            expected.to_parquet(target, index=False)
            if not pd.read_parquet(target).equals(expected):
                raise DividendResolutionError(f"研究副本非删除行发生变化：{name}")
        else:
            shutil.copyfile(source, target)
        after = file_sha256(target)
        if not dropped and after != sha:
            raise DividendResolutionError(f"未改分区未能逐字节复制：{name}")
        files.append(
            {
                "source_path": str(source.resolve()),
                "output_path": str(target.resolve()),
                "source_sha256": sha,
                "output_sha256": after,
                "removed_row_indices": dropped,
                "byte_identical": after == sha,
                "unchanged_rows_preserved": True,
            }
        )
    return files


def prepare_research_dividends(
    *,
    raw_dir: Path,
    output_dir: Path,
    calendar: Sequence[str],
    codes: Sequence[str],
    resolutions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """生成全域原件副本与修订审计，返回 dividend_dir/manifest_path/manifest。

    每个 resolution 显式提供原始全字段行 SHA 集合、唯一保留行 SHA、公告确认字段、
    原件 URL/本地路径/SHA/公开日期。本函数不自行推断组件和合计关系。
    codes 只限制审计范围，输出分区保留全部股票。output_dir 必须尚不存在。
    """
    raw_dir, output_dir = raw_dir.resolve(), output_dir.resolve()
    if raw_dir == output_dir or raw_dir in output_dir.parents or output_dir in raw_dir.parents:
        raise DividendResolutionError("研究输出必须独立于原始数据目录及其父子路径")
    if output_dir.exists():
        raise DividendResolutionError(f"研究输出已存在，须使用新的证据目录：{output_dir}")
    paths = sorted(path for path in raw_dir.iterdir() if path.is_file())
    source_sha = {path.name: file_sha256(path) for path in paths}
    frames = {path.name: pd.read_parquet(path) for path in paths if path.suffix == ".parquet"}
    if not frames:
        raise DividendResolutionError("原始公司行为目录没有 Parquet 分区")
    validated = [_validate_resolution(frames, resolution) for resolution in resolutions]
    removed: dict[str, set[int]] = {}
    seen = set()
    for resolution in validated:
        for row in resolution["rows"]:
            key = (row["partition"], row["row_index"])
            if key in seen:
                raise DividendResolutionError("两个修订重复触及同一原始行")
            seen.add(key)
            if row["row_sha256"] != resolution["retained_row_sha256"]:
                removed.setdefault(row["partition"], set()).add(row["row_index"])
    output_dir.mkdir(parents=True)
    dividend_dir = output_dir / "dividend"
    files = _write_copy(raw_dir, dividend_dir, frames, removed, source_sha)
    audit_before = _audit_subset(frames, output_dir / "audit_before", calendar, set(codes))
    after_frames = {
        name: frame.drop(index=sorted(removed.get(name, set()))) for name, frame in frames.items()
    }
    audit_after = _audit_subset(after_frames, output_dir / "audit_after", calendar, set(codes))
    if any(file_sha256(raw_dir / name) != sha for name, sha in source_sha.items()):
        raise DividendResolutionError("准备期间原始分区发生变化，不能冻结此副本")
    manifest = {
        "schema": "p7_operating_monthly_corporate_input/v1",
        "raw_dir": str(raw_dir),
        "dividend_dir": str(dividend_dir),
        "scope": "全部股票、全部原分区；codes 仅限制问题审计",
        "audit_codes": sorted(set(codes)),
        "calendar_start": calendar[0],
        "calendar_end": calendar[-1],
        "calendar_days": len(calendar),
        "files": files,
        "resolutions": validated,
        "removed_rows": sum(len(rows) for rows in removed.values()),
        "audit_before": audit_before,
        "audit_after": audit_after,
        "raw_unchanged": True,
        "amounts_recalculated": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return {"dividend_dir": dividend_dir, "manifest_path": manifest_path, "manifest": manifest}
