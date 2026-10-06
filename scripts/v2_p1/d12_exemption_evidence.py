# -*- coding: utf-8 -*-
"""v2 P1 单元 4·D-12 豁避免责证据：解锁族严格门残差 = 加载窗内容差（全量证明）。

证明结构（三个恒等式闭合）：
1. panel == 全量历史查询表（修复写入值的独立复算校验，全股票-日）；
2. 参照 == 旧链窗口查询表（load_share_float(20110604, 20260802) + 窗内交易日，
   全股票-日；偏差 = §5 登记的参照修补点，逐点列示）；
3. 由 1、2 推出 strict(panel, 参照) 残差 = 全量 vs 窗口查询表内容差（ann_date≤2010
   年分区 + 末年分区），无第四来源。

产物：data/reports/v2_p1_d12_exemption_evidence_20261006.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.factors.risk.announcement_lookup import (  # noqa: E402
    build_share_float_lookup_by_date,
)
from src.lazybull.v2.common.types import FeatureQuery, TradeDate  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

#: 参照（单调用 20120104~20260702）的 share_float 加载窗 = [起点−7月, 终点+1月]
_REF_LOAD_START = "20110604"
_REF_LOAD_END = "20260802"

_COLS = ["days_to_unlock", "unlock_ratio"]
_ATOL = 1e-9


def _lookup_to_frame(lookup: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """{date: DataFrame} → (trade_date, ts_code) 索引的 Series 帧。"""
    parts = []
    for date, df in lookup.items():
        if df is None or len(df) == 0:
            continue
        part = df[["ts_code", col]].copy()
        part["trade_date"] = date
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["trade_date", "ts_code", col])
    out = pd.concat(parts, ignore_index=True)
    out["ts_code"] = out["ts_code"].astype(str)
    return out.set_index(["trade_date", "ts_code"]).sort_index()


def _match_stats(actual: pd.Series, expected: pd.Series, name: str) -> dict:
    """在 actual 索引上核对 expected：值一致（NaN==NaN 一致、|Δ|≤_ATOL）比例与偏差样本。"""
    aligned = expected.reindex(actual.index)
    both_nan = actual.isna() & aligned.isna()
    diff = (actual.astype(float) - aligned.astype(float)).abs()
    equal = both_nan | (diff <= _ATOL)
    n_mismatch = int((~equal).sum())
    samples = []
    if n_mismatch:
        bad = actual[~equal].head(8)
        for (date, code), val in bad.items():
            samples.append(
                {
                    "trade_date": str(date),
                    "ts_code": str(code),
                    "actual": None if pd.isna(val) else float(val),
                    "expected": (
                        None if pd.isna(aligned.loc[(date, code)]) else float(aligned.loc[(date, code)])
                    ),
                }
            )
    logger.info(f"{name}: 核对 {len(actual)}，不一致 {n_mismatch}")
    return {
        "checked": int(len(actual)),
        "mismatch": n_mismatch,
        "match_share": 1.0 - n_mismatch / max(len(actual), 1),
        "samples": samples,
    }


def main() -> int:
    t0 = time.time()
    loader = DataLoader(Storage())
    store = PanelDataStore(str(ROOT / "data"))

    trade_cal = loader.load_clean_trade_cal()
    dates = trade_cal["cal_date"].astype(str).str.replace("-", "", regex=False)
    open_dates = sorted(dates[trade_cal["is_open"] == 1].tolist())
    window_dates = [d for d in open_dates if "20120104" <= d <= "20260702"]

    # 1) 全量历史查询表（修复同源）
    raw_dir = loader.storage.raw_path / "share_float"
    full_raw = pd.concat(
        [pd.read_parquet(p) for p in sorted(raw_dir.glob("*.parquet"))], ignore_index=True
    )
    logger.info(f"全量 share_float: {len(full_raw)} 条")
    full_lookup = build_share_float_lookup_by_date(full_raw, open_dates)

    # 2) 旧链窗口查询表（参照构建口径）
    win_raw = loader.load_share_float(_REF_LOAD_START, _REF_LOAD_END)
    logger.info(f"窗口 share_float: {len(win_raw)} 条（[{_REF_LOAD_START}, {_REF_LOAD_END}]）")
    win_lookup = build_share_float_lookup_by_date(win_raw, window_dates)

    # 3) panel 实际值（announcement 族两列，全区间）
    panel = store.load_features(
        FeatureQuery(
            columns=_COLS,
            start_date=TradeDate.from_str("20120104"),
            end_date=TradeDate.from_str("20260702"),
        )
    )
    panel_df = panel.df.reset_index()
    panel_df["trade_date"] = panel_df["trade_date"].astype(str)
    panel_df = panel_df.set_index(["trade_date", "ts_code"]).sort_index()
    logger.info(f"panel 股票-日: {len(panel_df)}")

    # 4) 参照实际值（逐日分区投影两列；早期分区缺列按全 NaN 处理——L1 schema 血缘）
    import pyarrow.parquet as pq

    ref_dir = ROOT / "temp" / "p1_frozen_reference" / "features" / "cs_train"
    ref_parts = []
    for path in sorted(ref_dir.glob("*.parquet")):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        read_cols = ["ts_code", *[c for c in _COLS if c in available]]
        df = pd.read_parquet(path, columns=read_cols)
        for col in _COLS:
            if col not in df.columns:
                df[col] = np.nan
        df["trade_date"] = path.stem
        ref_parts.append(df)
    ref_df = pd.concat(ref_parts, ignore_index=True)
    ref_df["ts_code"] = ref_df["ts_code"].astype(str)
    ref_df = ref_df.set_index(["trade_date", "ts_code"]).sort_index()
    logger.info(f"参照股票-日: {len(ref_df)}")

    report: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "columns": {}}
    for col in _COLS:
        full_s = _lookup_to_frame(full_lookup, col)[col]
        win_s = _lookup_to_frame(win_lookup, col)[col]
        col_rep = {
            "panel_vs_full_lookup": _match_stats(panel_df[col], full_s, f"panel==全量表[{col}]"),
            "ref_vs_window_lookup": _match_stats(ref_df[col], win_s, f"参照==窗口表[{col}]"),
        }
        # 5) 内容差规模（全量 vs 窗口）：diff(panel, 参照) 的预测集
        union_idx = full_s.index.union(win_s.index)
        f_al = full_s.reindex(union_idx)
        w_al = win_s.reindex(union_idx)
        both_nan = f_al.isna() & w_al.isna()
        content_diff = ~(both_nan | ((f_al.astype(float) - w_al.astype(float)).abs() <= _ATOL))
        col_rep["content_gap_stock_days"] = int(content_diff.sum())
        # 内容差的年代分布（验证 ann≤2010 / 末年两个来源）
        gap_dates = union_idx[content_diff].get_level_values("trade_date")
        by_year = pd.Series(gap_dates.str[:4]).value_counts().sort_index().to_dict()
        col_rep["content_gap_by_year"] = {str(k): int(v) for k, v in by_year.items()}
        report["columns"][col] = col_rep

    out = ROOT / "data" / "reports" / "v2_p1_d12_exemption_evidence_20261006.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"证据报告: {out}（耗时 {(time.time()-t0)/60:.1f} min）")

    ok = all(
        report["columns"][c]["panel_vs_full_lookup"]["mismatch"] == 0
        and report["columns"][c]["ref_vs_window_lookup"]["mismatch"] == 0
        for c in _COLS
    )
    print("恒等式校验:", "全部成立" if ok else "存在偏差（见报告 samples）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
