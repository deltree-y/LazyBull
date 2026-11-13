# -*- coding: utf-8 -*-
"""申万行业分类历史（PIT）数据层。

数据源：申万宏源研究下载中心公开的「股票行业分类」Excel（0 元，无需采购授权）。
数据契约见 ``docs/data/sw_industry_pit_audit.md``；判据见 ``CLAUDE.md`` §5「数据完备性判据契约」。

三条硬口径：
1. 原始文件只有「计入日期」，多段记录构成区间：``out_date`` = 同一股票的下一条 ``in_date``；
2. 文件跨三套行业码表（pre-2014 / SW2014 / SW2021），版本**必须由码表反查**，不能只按日期硬切；
3. 北交所代码段变更（``83xxxx/87xxxx/43xxxx → 920xxx``）由 TuShare ``bse_mapping`` 归一，
   属代码映射问题而非数据缺失。
"""

from __future__ import annotations

import hashlib
import io
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from loguru import logger

SW_HISTORY_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
SW_VERSION_CUTOVER = "20210730"
SW_SRC_SW2014 = "SW2014"
SW_SRC_SW2021 = "SW2021"
SW_VERSION_UNMAPPED = "UNMAPPED"

HISTORY_COLUMNS = ["code6", "in_date", "ind_code", "source_updated_at"]
PIT_COLUMNS = [
    "code6",
    "ts_code",
    "in_date",
    "out_date",
    "ind_code",
    "l1_code",
    "l2_code",
    "l1_name",
    "l2_name",
    "l3_name",
    "index_code",
    "sw_version",
    "source_updated_at",
    "code_alias",
]

# 借壳 / 重新上市导致「旧归属确实未保留」的样本（台账专用；不是代码映射问题）
RELISTING_SHELL_CODES = ("001914", "302132", "601360", "601399", "605055")

_EXCHANGE_RULES = (
    ("920", "BJ"),
    ("43", "BJ"),
    ("83", "BJ"),
    ("87", "BJ"),
    ("900", "SH"),
    ("200", "SZ"),
    ("688", "SH"),
    ("689", "SH"),
    ("600", "SH"),
    ("601", "SH"),
    ("603", "SH"),
    ("605", "SH"),
    ("000", "SZ"),
    ("001", "SZ"),
    ("002", "SZ"),
    ("003", "SZ"),
    ("300", "SZ"),
    ("301", "SZ"),
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def normalize_date(value: object) -> Optional[str]:
    """把任意日期表示归一为 YYYYMMDD；无法解析返回 None。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y%m%d")


def infer_ts_code(code6: str) -> Optional[str]:
    """按代码段推断交易所后缀（6 位代码在 A 股内唯一）。"""
    text = str(code6).zfill(6)
    for prefix, exchange in _EXCHANGE_RULES:
        if text.startswith(prefix):
            return f"{text}.{exchange}"
    return None


def fetch_history_raw(
    url: str = SW_HISTORY_URL,
    timeout: int = 90,
    verify: bool = True,
) -> pd.DataFrame:
    """下载申万行业分类历史 Excel（官方公开文件）。

    TLS 证书链在部分环境不完整：verify=True 失败时降级为 verify=False 并告警，
    降级事实必须由调用方登记（返回 ``DataFrame.attrs["tls_verified"]``）。
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, verify=verify)
        tls_verified = verify
    except requests.exceptions.SSLError as exc:
        if not verify:
            raise RuntimeError(f"申万历史分类下载失败（TLS）：{url}") from exc
        logger.warning(f"申万官网证书链不完整，降级为不校验证书重试（仅限该只读下载）：{exc}")
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, verify=False)
        tls_verified = False

    if resp.status_code != 200 or len(resp.content) < 1024:
        raise RuntimeError(
            f"申万历史分类下载异常：status={resp.status_code} bytes={len(resp.content)} url={url}"
        )
    raw = pd.read_excel(io.BytesIO(resp.content))
    if raw.shape[1] < 4:
        raise ValueError(f"申万历史分类字段数异常：{raw.shape[1]} 列，表头 {list(raw.columns)}")
    raw = raw.iloc[:, :4].copy()
    raw.columns = HISTORY_COLUMNS
    raw.attrs["tls_verified"] = tls_verified
    raw.attrs["source_url"] = url
    return raw


def normalize_history(raw: pd.DataFrame) -> pd.DataFrame:
    """标准化原始四列 → 区间表（含 ``out_date`` / 版本 / 层级前缀）。

    同一 (code6, in_date) 若重复出现，保留 ``source_updated_at`` 最新的一行
    （与「更新日期」字段语义一致：后写覆盖先写）。
    """
    df = raw.copy()
    df["code6"] = df["code6"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["ind_code"] = df["ind_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["in_date"] = df["in_date"].map(normalize_date)
    df["source_updated_at"] = pd.to_datetime(df["source_updated_at"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["in_date"]).copy()
    df = df[df["code6"].str.fullmatch(r"\d{6}") & df["ind_code"].str.fullmatch(r"\d{6}")]
    if len(df) != before:
        logger.warning(f"申万历史分类剔除非法行 {before - len(df)} 条（日期/代码不可解析）")

    df = df.sort_values(["code6", "in_date", "source_updated_at"])
    df = df.drop_duplicates(["code6", "in_date"], keep="last")

    df["out_date"] = df.groupby("code6")["in_date"].shift(-1)
    keep = ["code6", "in_date", "out_date", "ind_code", "source_updated_at"]
    return df[keep].reset_index(drop=True)


def build_code_table(classify_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """合并多版本 ``index_classify`` 结果 → 行业码表（含版本、层级、指数码）。"""
    frames: List[pd.DataFrame] = []
    for frame in classify_frames:
        if frame is None or len(frame) == 0:
            continue
        part = frame.copy()
        part["industry_code"] = (
            part["industry_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
        frames.append(part[["src", "industry_code", "level", "industry_name", "index_code"]])
    if not frames:
        raise ValueError("行业码表为空：index_classify 未返回任何记录")
    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates(["src", "industry_code", "level"], keep="last")
    return table.reset_index(drop=True)


def assign_version(panel: pd.DataFrame, code_table: pd.DataFrame) -> pd.DataFrame:
    """按「日期切点 + 码表反查」确定版本，无法反查的标记 ``UNMAPPED``（pre-2014 码表）。"""
    known = {
        src: set(code_table.loc[code_table["src"] == src, "industry_code"])
        for src in (SW_SRC_SW2014, SW_SRC_SW2021)
    }
    codes = panel["ind_code"].astype(str)
    after_cutover = panel["in_date"].astype(str) >= SW_VERSION_CUTOVER
    version = pd.Series(SW_SRC_SW2021, index=panel.index, dtype=object)
    version = version.mask(~after_cutover, SW_SRC_SW2014)
    unmatched = ~after_cutover & ~codes.isin(known[SW_SRC_SW2014])
    version = version.mask(unmatched, SW_VERSION_UNMAPPED)
    panel = panel.copy()
    panel["sw_version"] = version
    return panel


def attach_code_names(panel: pd.DataFrame, code_table: pd.DataFrame) -> pd.DataFrame:
    """把 6 位行业码翻译为 L1/L2/L3 名称与 L3 指数代码（按版本匹配）。"""
    out = panel.copy()
    out["l1_code"] = out["ind_code"].str[:2] + "0000"
    out["l2_code"] = out["ind_code"].str[:4] + "00"

    for level, code_col, name_col in (
        ("L1", "l1_code", "l1_name"),
        ("L2", "l2_code", "l2_name"),
        ("L3", "ind_code", "l3_name"),
    ):
        right = code_table.loc[
            code_table["level"] == level, ["src", "industry_code", "industry_name"]
        ]
        right = right.rename(
            columns={"src": "sw_version", "industry_code": code_col, "industry_name": name_col}
        )
        out = out.merge(right, on=["sw_version", code_col], how="left")

    right3 = code_table.loc[
        code_table["level"] == "L3", ["src", "industry_code", "index_code"]
    ].rename(columns={"src": "sw_version", "industry_code": "ind_code"})
    out = out.merge(right3, on=["sw_version", "ind_code"], how="left")
    return out


def finalize_pit(panel: pd.DataFrame, alias: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """补齐 ``ts_code`` / 别名行 / 列序，并做结构性校验。"""
    out = panel.copy()
    out["ts_code"] = out["code6"].map(infer_ts_code)
    out["code_alias"] = False

    if alias:
        extra = out[out["code6"].isin(alias.values())].copy()
        if len(extra):
            reverse = {new: old for old, new in alias.items()}
            extra["code6"] = extra["code6"].map(reverse)
            extra["ts_code"] = extra["code6"].map(infer_ts_code)
            extra["code_alias"] = True
            out = pd.concat([out, extra], ignore_index=True)

    out = out[[c for c in PIT_COLUMNS if c in out.columns]]
    out = out.sort_values(["code6", "in_date"]).reset_index(drop=True)
    validate_pit(out)
    return out


def validate_pit(panel: pd.DataFrame) -> None:
    """结构校验：键唯一、区间有序且不重叠。违例直接报错（不做静默修正）。"""
    dup = panel.duplicated(["code6", "in_date"]).sum()
    if dup:
        raise ValueError(f"行业 PIT 面板存在重复键 (code6, in_date) {int(dup)} 条")
    out_date = panel["out_date"].fillna("99999999").astype(str)
    bad_order = (out_date <= panel["in_date"].astype(str)) & panel["out_date"].notna()
    if int(bad_order.sum()):
        raise ValueError(f"行业 PIT 面板存在 out_date <= in_date 的区间 {int(bad_order.sum())} 条")
    if panel["ind_code"].isna().any():
        raise ValueError("行业 PIT 面板存在缺失 ind_code 的行")


def lookup_industry(panel: pd.DataFrame, ts_code: str, day: str) -> Optional[dict]:
    """单点查询：返回该股票在 ``day`` 当日的行业归属（PIT：in_date <= day < out_date）。"""
    code = str(ts_code)[:6]
    stamp = normalize_date(day)
    if stamp is None:
        raise ValueError(f"日期不可解析: {day!r}")
    sub = panel[(panel["code6"] == code) & (panel["in_date"] <= stamp)]
    sub = sub[sub["out_date"].fillna("99999999").astype(str) > stamp]
    if len(sub) == 0:
        return None
    row = sub.iloc[-1]
    return {key: row[key] for key in row.index}


def attach_pit(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """把 PIT 行业归属附加到任意「股票-日期」表上（不改行数、不改顺序）。"""
    left = frame.copy()
    left["_row_id"] = range(len(left))
    left["_code6"] = left[code_col].astype(str).str[:6]
    left["_day"] = left[date_col].map(normalize_date)
    if left["_day"].isna().any():
        raise ValueError(f"{date_col} 存在无法解析的日期")
    left["_day_num"] = left["_day"].astype("int64")

    right = panel[
        [
            "code6",
            "in_date",
            "out_date",
            "ind_code",
            "l1_code",
            "l2_code",
            "l1_name",
            "l2_name",
            "l3_name",
            "index_code",
            "sw_version",
        ]
    ].copy()
    right = right.rename(columns={"code6": "_code6"})
    right["_in_num"] = right["in_date"].astype(str).astype("int64")
    right = right.sort_values("_in_num")

    merged = pd.merge_asof(
        left.sort_values(["_day_num", "_code6"]),
        right,
        left_on="_day_num",
        right_on="_in_num",
        by="_code6",
        direction="backward",
    )
    matched = merged["_in_num"].notna()
    merged["industry_known"] = matched & (
        merged["out_date"].fillna("99999999").astype(str) > merged["_day"]
    )
    payload = [
        "ind_code",
        "l1_code",
        "l2_code",
        "l1_name",
        "l2_name",
        "l3_name",
        "index_code",
        "sw_version",
    ]
    unknown = ~merged["industry_known"]
    for col in payload:
        merged.loc[unknown, col] = pd.NA
    merged = merged.sort_values("_row_id").drop(
        columns=["_row_id", "_code6", "_day", "_day_num", "_in_num", "in_date", "out_date"]
    )
    return merged.reset_index(drop=True)


def build_exclusion_ledger(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """生成排除台账：只保留 ``industry_known=False`` 的行，并给出可审计原因。"""
    merged = attach_pit(panel, frame, code_col=code_col, date_col=date_col)
    unknown = merged[~merged["industry_known"]].copy()
    if len(unknown) == 0:
        return pd.DataFrame(columns=[code_col, date_col, "reason", "fixable"])

    first_in = panel.groupby("code6")["in_date"].min()
    codes = unknown[code_col].astype(str).str[:6]
    first = codes.map(first_in)
    day = unknown[date_col].map(normalize_date)

    reason = pd.Series("no_record", index=unknown.index, dtype=object)
    reason = reason.mask(first.notna(), "not_yet_classified")
    reason = reason.mask(codes.isin(RELISTING_SHELL_CODES), "relisting_shell")
    fixable = reason.eq("not_yet_classified") & first.gt(day)

    unknown["reason"] = reason
    unknown["fixable"] = fixable
    unknown["first_in_date"] = first
    return unknown.reset_index(drop=True)


def panel_fingerprint(panel: pd.DataFrame) -> Dict[str, object]:
    """内容指纹（与 parquet 引擎版本无关）：行数 / 股票数 / 版本分布 / 最大更新日 / 内容 SHA。"""
    key_cols = ["code6", "in_date", "out_date", "ind_code", "sw_version", "code_alias"]
    view = panel[[col for col in key_cols if col in panel.columns]].copy()
    view = view.sort_values(["code6", "in_date"])
    payload = view.astype(str).to_csv(index=False).encode("utf-8")
    updated = panel["source_updated_at"].max() if "source_updated_at" in panel.columns else None
    return {
        "rows": int(len(panel)),
        "codes": int(panel["code6"].nunique()),
        "versions": {str(k): int(v) for k, v in panel["sw_version"].value_counts().items()},
        "max_updated_at": None if updated is None or pd.isna(updated) else str(updated),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def diff_panels(old: Optional[pd.DataFrame], new: pd.DataFrame) -> Dict[str, object]:
    """对比两次抓取的区间表，输出可审计的变更摘要（供刷新告警与数据态登记）。"""
    new_fp = panel_fingerprint(new)
    if old is None or len(old) == 0:
        return {"changed": True, "baseline": "none", "new": new_fp}

    old_fp = panel_fingerprint(old)
    old_key = old.set_index(["code6", "in_date"])["ind_code"]
    new_key = new.set_index(["code6", "in_date"])["ind_code"]
    added = new_key.index.difference(old_key.index)
    removed = old_key.index.difference(new_key.index)
    common = old_key.index.intersection(new_key.index)
    reclassified = int((old_key.loc[common] != new_key.loc[common]).sum())

    return {
        "changed": old_fp["content_sha256"] != new_fp["content_sha256"],
        "baseline": "present",
        "old": old_fp,
        "new": new_fp,
        "added_intervals": int(len(added)),
        "removed_intervals": int(len(removed)),
        "reclassified_intervals": reclassified,
        "codes_added": int(len(set(new["code6"]) - set(old["code6"]))),
        "codes_removed": int(len(set(old["code6"]) - set(new["code6"]))),
    }


def summarize_ledger(ledger: pd.DataFrame, total_rows: int) -> Dict[str, object]:
    """汇总台账（供 prereg 的覆盖门与边界敏感性使用）。"""
    if total_rows <= 0:
        raise ValueError("total_rows 必须为正")
    excluded = len(ledger)
    by_reason = ledger["reason"].value_counts().to_dict() if excluded else {}
    return {
        "total_rows": int(total_rows),
        "excluded_rows": int(excluded),
        "excluded_share": excluded / total_rows,
        "by_reason": by_reason,
    }
