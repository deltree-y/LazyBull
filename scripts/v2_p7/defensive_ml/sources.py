"""输入审计：数据集清单、内容身份、单位断言、行业 PIT 与退市可追溯性。

只读：本模块不写任何路径，返回结构化事实由上层落盘。
复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.common import sha256_bytes, sha256_file
from scripts.v2_p7.defensive_ml import spec

#: 退市/终止报价抽样（覆盖不同退市年份；000003.SZ 于 2002 退市，超出 clean 起点，预期 0 天）
DELISTED_PROBES: tuple[tuple[str, str], ...] = (
    ("600087.SH", "2014 退市长油"),
    ("601268.SH", "2015 *ST二重"),
    ("300372.SZ", "2016 欣泰电气"),
    ("000033.SZ", "2017 新都退"),
    ("600806.SH", "2018 昆明机床"),
    ("000418.SZ", "2019 小天鹅A 吸收合并"),
    ("000003.SZ", "2002 PT金田A（早于 clean 起点）"),
)

#: 行业 PIT 抽样日（每年一个，覆盖训练早段与 OOS 全段）
INDUSTRY_PROBE_DATES: tuple[str, ...] = (
    "2013-06-03",
    "2014-06-03",
    "2015-06-12",
    "2016-06-01",
    "2017-06-01",
    "2018-06-01",
    "2019-04-01",
    "2020-06-01",
    "2021-06-01",
    "2022-06-01",
    "2023-06-01",
    "2024-06-03",
    "2025-06-03",
    "2026-10-29",
)

#: 单位核对日（覆盖两个时代）
UNIT_PROBE_DATES: tuple[str, ...] = ("2019-04-01", "2026-10-29")


def partition_files(directory: Path) -> list[Path]:
    """分区目录下的 parquet 文件（按文件名升序）。"""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def _digest_paths(paths: Iterable[Path], base: Path) -> str:
    """目录内容身份：按相对路径升序拼接 `<name>\\0<sha>\\n` 后取 SHA256。"""
    lines = [f"{p.relative_to(base).as_posix()}\0{sha256_file(p)}\n" for p in paths]
    return sha256_bytes("".join(lines).encode("utf-8"))


def inventory(root: Path, hash_content: bool = True) -> dict[str, Any]:
    """按 `spec.INPUTS` 盘点输入：存在性、规模、水位与内容身份。"""
    result: dict[str, Any] = {}
    for item in spec.INPUTS:
        name = str(item["name"])
        path = Path(root) / str(item["relative"])
        kind = str(item["kind"])
        if kind == "file":
            entry: dict[str, Any] = {
                "kind": kind,
                "relative": item["relative"],
                "exists": path.exists(),
            }
            if path.exists():
                entry["bytes"] = int(path.stat().st_size)
                if hash_content:
                    entry["sha256"] = sha256_file(path)
            result[name] = entry
            continue
        files = partition_files(path)
        entry = {
            "kind": kind,
            "relative": item["relative"],
            "exists": path.exists(),
            "n_partitions": len(files),
            "first": files[0].stem if files else None,
            "last": files[-1].stem if files else None,
            "bytes": int(sum(p.stat().st_size for p in files)),
        }
        if hash_content and files:
            entry["content_sha256"] = _digest_paths(files, path)
        result[name] = entry
    return result


def verify_a_baseline(root: Path) -> dict[str, Any]:
    """核验 A 基准冻结账户：manifest 字节 SHA + 四臂目录 + 身份字段。"""
    base = Path(root) / str(spec.A_BASELINE["relative"])
    manifest_path = base / "manifest.json"
    out: dict[str, Any] = {
        "relative": spec.A_BASELINE["relative"],
        "exists": base.exists(),
        "manifest_exists": manifest_path.exists(),
    }
    if not manifest_path.exists():
        out["status"] = "fail"
        return out
    actual = sha256_file(manifest_path)
    out["manifest_sha256_actual"] = actual
    out["manifest_sha256_expected"] = spec.A_BASELINE["expected_manifest_sha256"]
    out["manifest_sha256_match"] = actual == spec.A_BASELINE["expected_manifest_sha256"]
    arms: dict[str, Any] = {}
    for name in spec.A_BASELINE["required_arm_dirs"]:
        identity_path = base / name / "baseline_identity.json"
        if not identity_path.exists():
            arms[name] = {"present": False}
            continue
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        arms[name] = {
            "present": True,
            "initial_capital": identity.get("initial_capital"),
            "initial_point": identity.get("initial_point"),
            "start": identity.get("start"),
            "end": identity.get("end"),
            "position_sizing": identity.get("position_sizing"),
            "not_virgin_holdout": identity.get("not_virgin_holdout"),
            "cross_sleeve_netting": identity.get("cross_sleeve_netting"),
            "capital_scaled_from_other_nav": identity.get("capital_scaled_from_other_nav"),
        }
    out["arms"] = arms
    ok = out["manifest_sha256_match"] and all(v.get("present") for v in arms.values())
    window_ok = all(
        v.get("start") == spec.WINDOW["start"]
        and v.get("end") == spec.WINDOW["end"]
        and v.get("initial_point") == spec.WINDOW["cash_anchor"]
        for v in arms.values()
    )
    out["window_matches_protocol"] = window_ok
    out["status"] = "pass" if ok and window_ok else "fail"
    return out


def _read_daily(root: Path, date_dash: str, columns: Sequence[str]) -> pd.DataFrame:
    path = Path(root) / "data/clean/daily" / f"{date_dash}.parquet"
    return pd.read_parquet(path, columns=list(columns))


def _read_daily_basic(root: Path, date_dash: str, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_parquet(
        Path(root) / "data/clean/daily_basic" / f"{date_dash}.parquet", columns=list(columns)
    )


def _ratio_within(series: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    valid = series.notna() & lower.notna() & upper.notna()
    if not bool(valid.any()):
        return 0.0
    inside = (series >= lower) & (series <= upper)
    return float(inside[valid].mean())


def verify_units(root: Path, dates: Sequence[str] = UNIT_PROBE_DATES) -> dict[str, Any]:
    """单位与派生一致性断言：amount(千元) / vol(手) / total_mv(万元) / turnover_rate(%)。"""
    checks: list[dict[str, Any]] = []
    for date_dash in dates:
        day = pd.read_parquet(
            Path(root) / "data/clean/daily" / f"{date_dash}.parquet",
            columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                     "adj_factor", "close_adj", "is_st", "tradable"],
        )
        basic = _read_daily_basic(
            root, date_dash, ["ts_code", "close", "total_mv", "total_share", "turnover_rate",
                              "float_share"]
        )
        # 1) amount(千元) * 1000 / (vol(手) * 100) 应落在 [low, high]
        tradable = day[(day["vol"] > 0) & (day["amount"] > 0)]
        vwap = (tradable["amount"] * spec.UNIT_RULES["daily_amount"]["multiplier"]) / (
            tradable["vol"] * 100.0
        )
        inside = _ratio_within(vwap, tradable["low"] * 0.98, tradable["high"] * 1.02)
        checks.append(
            {
                "date": date_dash,
                "check": "daily_amount_unit_qian_yuan",
                "rows": int(len(tradable)),
                "ratio_inside_low_high": inside,
                "status": "pass" if inside >= 0.95 else "fail",
            }
        )
        # 2) total_mv(万元) * 1e4 / (total_share(万股) * 1e4) 应等于 close
        merged = day.merge(basic, on="ts_code", suffixes=("", "_basic"))
        usable = merged[(merged["total_share"] > 0) & (merged["total_mv"] > 0)]
        implied = (usable["total_mv"] * spec.UNIT_RULES["daily_basic_total_mv"]["multiplier"]) / (
            usable["total_share"] * 1e4
        )
        rel = ((implied - usable["close"]) / usable["close"]).abs()
        share = float((rel <= 0.01).mean()) if len(rel) else 0.0
        checks.append(
            {
                "date": date_dash,
                "check": "daily_basic_total_mv_unit_wan_yuan",
                "rows": int(len(usable)),
                "ratio_rel_error_le_1pct": share,
                "status": "pass" if share >= 0.95 else "fail",
            }
        )
        # 3) turnover_rate 百分数取值域
        tr = basic["turnover_rate"].dropna()
        checks.append(
            {
                "date": date_dash,
                "check": "daily_basic_turnover_rate_percent_scale",
                "rows": int(len(tr)),
                "median": float(np.median(tr)) if len(tr) else None,
                "status": "pass" if len(tr) and 0.0 < float(np.median(tr)) < 20.0 else "warn",
            }
        )
        # 4) close_adj = close * adj_factor
        adj = day[(day["close"] > 0) & (day["adj_factor"] > 0)]
        rel_adj = ((adj["close"] * adj["adj_factor"] - adj["close_adj"]) / adj["close_adj"]).abs()
        share_adj = float((rel_adj <= 1e-6).mean()) if len(rel_adj) else 0.0
        checks.append(
            {
                "date": date_dash,
                "check": "close_adj_equals_close_times_factor",
                "rows": int(len(adj)),
                "ratio_rel_error_le_1e6": share_adj,
                "status": "pass" if share_adj >= 0.999 else "fail",
            }
        )
        # 5) trade_date 形态（YYYYMMDD 字符串）
        values = day["trade_date"].astype(str)
        ok_fmt = bool(values.str.fullmatch(r"\d{8}").all())
        checks.append(
            {
                "date": date_dash,
                "check": "trade_date_yyyymmdd_string",
                "rows": int(len(values)),
                "status": "pass" if ok_fmt else "fail",
                "sample": str(values.iloc[0]),
            }
        )
    status = "fail" if any(c["status"] == "fail" for c in checks) else "pass"
    return {"checks": checks, "status": status}


def industry_coverage(
    root: Path, dates: Sequence[str] = INDUSTRY_PROBE_DATES
) -> dict[str, Any]:
    """行业 PIT 一级映射覆盖率（当日 clean/daily 代码中可命中 L1 的比例）。"""
    pit = pd.read_parquet(Path(root) / "data/clean/sw_industry_pit.parquet")
    pit["in_date"] = pit["in_date"].astype(str)
    pit["out_date"] = pit["out_date"].fillna("99999999").astype(str)
    has_l1 = pit[pit["l1_code"].notna() & pit["l1_name"].notna()]
    rows: list[dict[str, Any]] = []
    for date_dash in dates:
        path = Path(root) / "data/clean/daily" / f"{date_dash}.parquet"
        if not path.exists():
            rows.append({"date": date_dash, "status": "missing_partition"})
            continue
        day8 = date_dash.replace("-", "")
        codes = pd.read_parquet(path, columns=["ts_code"])["ts_code"]
        active = has_l1[(has_l1["in_date"] <= day8) & (has_l1["out_date"] > day8)]
        mapped = codes.isin(set(active["ts_code"]))
        rows.append(
            {
                "date": date_dash,
                "daily_rows": int(len(codes)),
                "mapped_rows": int(mapped.sum()),
                "l1_ratio": float(mapped.mean()) if len(codes) else None,
                "status": "ok",
            }
        )
    ratios = [r["l1_ratio"] for r in rows if r.get("l1_ratio") is not None]
    return {
        "rows": rows,
        "min_l1_ratio": min(ratios) if ratios else None,
        "mean_l1_ratio": float(np.mean(ratios)) if ratios else None,
        "unmapped_pit_rows": int(pit["l1_name"].isna().sum()),
        "open_interval_rows": int((pit["out_date"] == "99999999").sum()),
    }


def delisted_presence(
    root: Path, probes: Sequence[tuple[str, str]] = DELISTED_PROBES
) -> dict[str, Any]:
    """退市可追溯性：已知退市股在 clean/daily 的在场天数与最后报价日。"""
    daily_dir = Path(root) / "data/clean/daily"
    files = partition_files(daily_dir)
    wanted = {code for code, _ in probes}
    seen: dict[str, dict[str, Any]] = {
        code: {"days": 0, "last": None, "note": note} for code, note in probes
    }
    for path in files:
        codes = pd.read_parquet(path, columns=["ts_code"])["ts_code"]
        present = wanted & set(codes.unique())
        for code in present:
            seen[code]["days"] += 1
            seen[code]["last"] = path.stem
    basic = pd.read_parquet(Path(root) / "data/raw/stock_basic.parquet")
    return {
        "has_independent_delist_registry": "delist_date"
        in set(basic.columns),  # stock_basic 当前列集
        "stock_basic_columns": [str(c) for c in basic.columns],
        "stock_basic_rows": int(len(basic)),
        "probes": seen,
    }


def calendar_span(root: Path) -> dict[str, Any]:
    """完整交易日历跨度与关键锚点。"""
    cal = pd.read_parquet(Path(root) / "data/clean/trade_cal.parquet")
    open_days = cal[cal["is_open"] == 1]["cal_date"].astype(str)
    anchors = ["20130401", "20190401", "20261029", spec.SPLITS["sample_anchor"]]
    return {
        "rows": int(len(cal)),
        "open_days": int(len(open_days)),
        "min": str(open_days.min()),
        "max": str(open_days.max()),
        "anchor_present": {a: bool((open_days == a).any()) for a in anchors},
    }


def partition_names(root: Path, relative: str) -> list[str]:
    """分区名列表（用于预热窗口检查）。"""
    return [p.stem for p in partition_files(Path(root) / relative)]


def warmup_availability(root: Path) -> dict[str, Any]:
    """首折特征预热可得性：2013-04-01 之前须有 252 个交易日分区。"""
    names = partition_names(root, "data/clean/daily")
    target = spec.SPLITS["refresh_anchor"]  # 2019-04-01 之前也需覆盖首折起点
    first_fold = spec.WINDOW["first_train_signal_start"]  # 20130401
    target_dash = f"{first_fold[:4]}-{first_fold[4:6]}-{first_fold[6:]}"
    before = [n for n in names if n < target_dash]
    return {
        "first_fold_signal_start": first_fold,
        "partitions_before_first_fold": len(before),
        "warmup_required": int(spec.WINDOW["price_warmup_trading_days"]),
        "warmup_ok": len(before) >= int(spec.WINDOW["price_warmup_trading_days"]),
        "oldest_partition": names[0] if names else None,
        "latest_partition": names[-1] if names else None,
        "anchor_for_refresh": target,
    }
