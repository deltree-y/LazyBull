"""只读输入审计：文件身份、字段单位、冲突台账、A 基准身份（R0 证据）。

本模块不写任何生产数据；所有哈希都是字节级 SHA256。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import panel, report_events, spec
from scripts.v2_p7.common import sha256_file


class SourceError(RuntimeError):
    """输入身份不符或字段不可核（fail-closed）。"""


def partition_files(directory: Path) -> list[Path]:
    """分区文件（按文件名排序）。"""
    if not Path(directory).exists():
        return []
    return sorted(Path(directory).glob("*.parquet"))


def _digest_paths(paths: Sequence[Path], root: Path) -> str:
    """分区集合内容指纹（相对路径 + 逐文件 SHA 拼接后再 SHA）。"""
    import hashlib

    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def inventory(root: Path, entries: Iterable[dict[str, Any]] = spec.INPUTS) -> list[dict[str, Any]]:
    """逐输入登记存在性、分区数、首末分区、字节数与内容 SHA。"""
    out: list[dict[str, Any]] = []
    for entry in entries:
        target = Path(root) / entry["relative"]
        record: dict[str, Any] = {
            "name": entry["name"],
            "relative": entry["relative"],
            "kind": entry["kind"],
            "exists": target.exists(),
        }
        if not target.exists():
            record.update({"n_partitions": 0, "first": None, "last": None, "bytes": 0,
                           "content_sha256": None})
            out.append(record)
            continue
        if entry["kind"] == "partition":
            files = partition_files(target)
            record.update(
                {
                    "n_partitions": len(files),
                    "first": files[0].stem if files else None,
                    "last": files[-1].stem if files else None,
                    "bytes": int(sum(path.stat().st_size for path in files)),
                    "content_sha256": _digest_paths(files, Path(root)),
                }
            )
        else:
            record.update(
                {
                    "n_partitions": 1,
                    "first": target.name,
                    "last": target.name,
                    "bytes": int(target.stat().st_size),
                    "content_sha256": sha256_file(target),
                }
            )
        out.append(record)
    return out


def verify_a_baseline(root: Path) -> dict[str, Any]:
    """A 基准身份：manifest 字节 SHA + 四臂目录 + 逐臂 baseline_identity。"""
    base = Path(root) / spec.A_BASELINE["relative"]
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return {"status": "fail", "reason": f"缺 A manifest：{manifest_path}"}
    digest = sha256_file(manifest_path)
    expected = spec.A_BASELINE["expected_manifest_sha256"]
    checks: list[dict[str, Any]] = [
        {
            "name": "manifest_sha256",
            "ok": digest == expected,
            "expected": expected,
            "actual": digest,
        }
    ]
    for relative in spec.A_BASELINE["required_arm_dirs"]:
        arm_dir = base / relative
        identity_path = arm_dir / "baseline_identity.json"
        if not identity_path.exists():
            checks.append({"name": f"{relative}/identity", "ok": False, "reason": "缺文件"})
            continue
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        capital = int(relative.split("/")[0])
        # A80 是同一本金的 80% 独立账户（协议 §10.1），不得由 A100 缩放生成
        expected_capital = capital * (0.8 if "A80" in relative else 1.0)
        checks.append(
            {
                "name": f"{relative}/identity",
                "ok": (
                    abs(float(identity.get("initial_capital", -1)) - expected_capital) < 1e-6
                    and str(identity.get("start")) == str(spec.WINDOW["start"])
                    and str(identity.get("end")) == str(spec.WINDOW["end"])
                    and str(identity.get("initial_point")) == str(spec.WINDOW["cash_anchor"])
                    and bool(identity.get("not_virgin_holdout")) is True
                    and bool(identity.get("cross_sleeve_netting", False)) is False
                    and bool(identity.get("capital_scaled_from_other_nav", False)) is False
                ),
                "identity": identity,
            }
        )
    series = [
        "1000000/A100_daily.parquet",
        "1000000/A80_B_cash_daily.parquet",
        "1000000/A80/daily.json",
        "2000000/A100_daily.parquet",
        "2000000/A80_B_cash_daily.parquet",
        "2000000/A80/daily.json",
    ]
    for relative in series:
        checks.append({"name": relative, "ok": (base / relative).exists()})
    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "status": "pass" if not failed else "fail",
        "manifest_sha256": digest,
        "checks": checks,
        "failed": failed,
        "relative": spec.A_BASELINE["relative"],
    }


def calendar_span(root: Path) -> dict[str, Any]:
    """交易日历覆盖与主窗锚点。"""
    days = panel.load_calendar(root)
    anchors = {
        "cash_anchor": str(spec.WINDOW["cash_anchor"]),
        "start": str(spec.WINDOW["start"]),
        "end": str(spec.WINDOW["end"]),
    }
    return {
        "first": days[0],
        "last": days[-1],
        "n_days": len(days),
        "anchors_present": {name: (value in days) for name, value in anchors.items()},
        "window_days": int(sum(anchors["start"] <= day <= anchors["end"] for day in days)),
    }


def industry_coverage(root: Path) -> dict[str, Any]:
    """主窗内 PIT 一级行业可得比例（按日 × 股票）。"""
    calendar = panel.load_calendar(root)
    days = [day for day in calendar if str(spec.WINDOW["start"]) <= day <= str(spec.WINDOW["end"])]
    pit = pd.read_parquet(Path(root) / "data/clean/sw_industry_pit.parquet")
    table_path = Path(root) / "data/clean/sw_industry_code_table.parquet"
    table = pd.read_parquet(table_path) if table_path.exists() else None
    from scripts.v2_p7.analyst_diffusion_ml import universe as uni

    names = uni.industry_l1_names(table, pit)
    frame = pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    frame["known"] = frame["l1_code"].astype(str).map(lambda code: code in names)
    daily = []
    for day in days:
        active = frame[(frame["in_date"] <= day) & (frame["out_date"] > day)]
        if len(active) == 0:
            daily.append(0.0)
            continue
        daily.append(float(active["known"].mean()))
    return {
        "days": len(days),
        "mean_known_ratio": float(np.mean(daily)) if daily else 0.0,
        "min_known_ratio": float(np.min(daily)) if daily else 0.0,
        "l1_name_count": len(names),
        "first": days[0] if days else None,
        "last": days[-1] if days else None,
    }


def report_rc_audit(raw: pd.DataFrame) -> dict[str, Any]:
    """report_rc 字段/单位/冲突/水位审计。"""
    availability = report_events.field_availability(raw)
    frame, conflict = report_events.normalize(raw)
    prepared = report_events.prepare_fields(frame)
    eps = prepared["eps_value"].dropna()
    target = prepared["target_value"].dropna()
    required = [
        "ts_code",
        "report_date",
        "report_title",
        "org_name",
        "author_name",
        "quarter",
        "eps",
        "max_price",
        "min_price",
        "rating",
    ]
    missing = [name for name in required if name not in raw.columns]
    return {
        "missing_columns": missing,
        "columns": list(raw.columns),
        "rows": int(len(raw)),
        "field_availability": availability,
        "conflict": conflict,
        "eps_quantiles": {
            "p01": float(eps.quantile(0.01)) if len(eps) else None,
            "p50": float(eps.median()) if len(eps) else None,
            "p99": float(eps.quantile(0.99)) if len(eps) else None,
        },
        "target_quantiles": {
            "p01": float(target.quantile(0.01)) if len(target) else None,
            "p50": float(target.median()) if len(target) else None,
            "p99": float(target.quantile(0.99)) if len(target) else None,
        },
        "rating_mapped_ratio": availability["rating_available_ratio"],
        "unit_conclusion": (
            "EPS 元/股（p50 量级 %s）、目标价 元/股（p50 量级 %s）"
            % (
                round(float(eps.median()), 3) if len(eps) else "n/a",
                round(float(target.median()), 2) if len(target) else "n/a",
            )
        ),
    }


def delisted_traceability(root: Path, end: str) -> dict[str, Any]:
    """退市/停报价股票是否仍出现在历史合格日（不得因后来退市删除历史行）。"""
    directory = Path(root) / "data/clean/daily"
    frames = []
    for day in ("20190603", "20210601", "20230601", end):
        matches = list(directory.glob(f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"))
        if not matches:
            continue
        frame = pd.read_parquet(matches[0], columns=["ts_code", "close_adj"])
        frame["trade_date"] = day
        frames.append(frame)
    if not frames:
        return {"status": "not_estimable", "reason": "探针分区缺失"}
    last = pd.read_parquet(
        list(directory.glob(f"{end[:4]}-{end[4:6]}-{end[6:]}.parquet"))[0],
        columns=["ts_code"],
    )
    alive = set(last["ts_code"].astype(str))
    early = set(frames[0]["ts_code"].astype(str))
    gone = sorted(early - alive)
    sample = gone[:10]
    present_later = {}
    if gone:
        later = pd.concat(frames[1:], ignore_index=True)
        later_codes = set(later["ts_code"].astype(str))
        present_later = {code: code in later_codes for code in sample}
    return {
        "status": "pass",
        "codes_at_first_probe": len(early),
        "codes_still_quoted_at_end": len(alive),
        "codes_absent_at_end": len(gone),
        "sample_absent": sample,
        "sample_present_in_later_probe": present_later,
        "note": "历史合格行不因后来退市删除；退市不填 -100% 回收",
    }


def daily_watermark(root: Path) -> dict[str, Any]:
    """clean/daily 与 clean/daily_basic 的分区水位。"""
    out: dict[str, Any] = {}
    for name in ("daily", "daily_basic"):
        files = partition_files(Path(root) / "data/clean" / name)
        out[name] = {
            "n": len(files),
            "first": files[0].stem if files else None,
            "last": files[-1].stem if files else None,
        }
    return out


def check_units(root: Path, probe_days: Sequence[str] | None = None) -> dict[str, Any]:
    """clean 层单位断言（市值万元、成交额千元），与研究层换算只允许一次。"""
    days = list(probe_days or ("20190401", "20230601", "20261029"))
    daily_dir = Path(root) / "data/clean/daily"
    basic_dir = Path(root) / "data/clean/daily_basic"
    picks = []
    for day in days:
        name = f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        if (daily_dir / name).exists() and (basic_dir / name).exists():
            picks.append(name)
    if not picks:
        return {"status": "not_estimable", "reason": "探针分区缺失"}
    rows = []
    for name in picks:
        daily = pd.read_parquet(daily_dir / name, columns=["amount", "close"])
        basic = pd.read_parquet(basic_dir / name, columns=["total_mv", "close"])
        rows.append(
            {
                "partition": name,
                "amount_p50": float(daily["amount"].median()),
                "close_p50": float(daily["close"].median()),
                "total_mv_p50": float(basic["total_mv"].median()),
            }
        )
    # amount 千元（p50 ~1e4）；total_mv 万元（p50 ~1e6）
    amount_ok = all(1e2 <= row["amount_p50"] <= 1e8 for row in rows)
    mv_ok = all(1e4 <= row["total_mv_p50"] <= 1e10 for row in rows)
    return {
        "status": "pass" if amount_ok and mv_ok else "fail",
        "rows": rows,
        "amount_unit": "千元",
        "total_mv_unit": "万元",
        "amount_ok": amount_ok,
        "total_mv_ok": mv_ok,
    }
