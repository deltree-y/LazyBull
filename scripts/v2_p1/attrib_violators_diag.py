# -*- coding: utf-8 -*-
"""归因门 5 个越界列的全量差异日诊断（panel vs cs_train，与对账门同比对口径）。

**一次性脚本，已执行完毕，效果持久化于 manifest.repairs；保留作审计轨迹。**
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.v2.common.types import FeatureQuery, TradeDate  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

COLS = ["downside_corr_20", "kurtosis_20", "skewness_20", "zscore_macd_hist", "zscore_macd_hist_sz"]
ATOL = 1e-6


def main() -> int:
    t0 = time.time()
    store = PanelDataStore(str(ROOT / "data"))
    panel = store.load_features(
        FeatureQuery(
            columns=COLS,
            start_date=TradeDate.from_str("20120104"),
            end_date=TradeDate.from_str("20260702"),
        )
    ).df
    panel = panel.reset_index()
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel = panel.set_index(["trade_date", "ts_code"]).sort_index()
    print(f"panel 股票-日 {len(panel)}（{(time.time()-t0):.0f}s）")

    import pyarrow.parquet as pq

    cs_dir = ROOT / "data" / "features" / "cs_train"
    parts = []
    for path in sorted(cs_dir.glob("*.parquet")):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        read_cols = ["ts_code", *[c for c in COLS if c in available]]
        df = pd.read_parquet(path, columns=read_cols)
        for col in COLS:
            if col not in df.columns:
                df[col] = float("nan")
        df["trade_date"] = path.stem
        parts.append(df)
    cs = pd.concat(parts, ignore_index=True)
    cs["ts_code"] = cs["ts_code"].astype(str)
    cs = cs.set_index(["trade_date", "ts_code"]).sort_index()
    print(f"cs_train 股票-日 {len(cs)}（{(time.time()-t0):.0f}s）")

    out = {}
    for col in COLS:
        joined = panel[[col]].join(cs[[col]], how="inner", lsuffix="_p", rsuffix="_c")
        p, c = joined[f"{col}_p"].astype(float), joined[f"{col}_c"].astype(float)
        both_nan = p.isna() & c.isna()
        over = ((p - c).abs() > ATOL) | (p.isna() != c.isna())
        over &= ~both_nan
        by_day = over.groupby(level="trade_date").sum()
        day_rows = over.groupby(level="trade_date").size()
        days = [
            [d, int(by_day[d]), int(day_rows[d])] for d in by_day.index if by_day[d] > 0
        ]
        l1_end = "20130630"
        late = [d for d in days if d[0] > l1_end]
        out[col] = {
            "total_over": int(over.sum()),
            "n_days": len(days),
            "first": days[0][0] if days else None,
            "last": days[-1][0] if days else None,
            "late_days": late,
            "late_over": sum(d[1] for d in late),
            "days_le_20130630": [d for d in days if d[0] <= l1_end][:10],
            "n_days_le_l1": sum(1 for d in days if d[0] <= l1_end),
        }
        print(f"{col}: over={out[col]['total_over']} days={len(days)} late_over={out[col]['late_over']}")

    out_path = ROOT / "data" / "reports" / "v2_p1_attrib_violators_diag_20261006.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"耗时 {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
