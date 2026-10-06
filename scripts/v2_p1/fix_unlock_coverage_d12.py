# -*- coding: utf-8 -*-
"""v2 P1 单元 4·D-12 修复：解禁前瞻列全量覆盖修正。

缺陷：share_float 按构建区间加载（float_date 年分区），分块回填的块加载窗截断 ⇒
「已公告未解禁」远年记录丢失 ⇒ days_to_unlock/unlock_ratio/unlock_risk_flag 在
分块早期年代值偏缺。修复：以**全量 share_float（全历史起点）+ 全日历**重建解禁
日频查询表（build_share_float_lookup_by_date 同一实现），对 panel 全分区逐日
整列覆盖正确值（announcement 族 days_to_unlock/unlock_ratio + risk 族
unlock_risk_flag），经 store.rewrite_partition（理由登记进 manifest.repairs）。

机器量级：162 冷月 × 2 族 + 244 热日 × 2 族 ≈ 810 文件读写，分钟级。
"""

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
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

_REASON = "D-12 修复：解禁前瞻列全量历史起点+全区间覆盖重建（区间截断缺陷修复）"


def _risk_flag(days_to_unlock: pd.Series) -> pd.Series:
    """unlock_risk_flag = f(days_to_unlock)：<30天=2，30-90天=1，>90天或无=0
    （factors/risk/announcement_factors.py::compute_unlock_risk_flag 同一口径）。"""
    days = days_to_unlock.astype(float)
    result = pd.Series(0.0, index=days_to_unlock.index)
    result[(days > 0) & (days <= 30)] = 2.0
    result[(days > 30) & (days <= 90)] = 1.0
    result[days.isna() | (days <= 0) | (days > 90)] = 0.0
    return result


def _correct_day_maps(lookup: dict, date: str) -> tuple[dict, dict]:
    """该日的正确映射：ts_code -> (days_to_unlock, unlock_ratio)；无记录股缺失。"""
    df = lookup.get(date)
    if df is None or len(df) == 0:
        return {}, {}
    return (
        dict(zip(df["ts_code"].astype(str), df["days_to_unlock"])),
        dict(zip(df["ts_code"].astype(str), df["unlock_ratio"])),
    )


def main() -> int:
    t0 = time.time()
    store = PanelDataStore(str(ROOT / "data"))
    loader = DataLoader(Storage())

    trade_cal = loader.load_clean_trade_cal()
    dates = trade_cal["cal_date"].astype(str).str.replace("-", "", regex=False)
    open_dates = sorted(dates[trade_cal["is_open"] == 1].tolist())
    logger.info(f"全日历交易日 {len(open_dates)} 天")

    raw_dir = loader.storage.raw_path / "share_float"
    frames = [pd.read_parquet(p) for p in sorted(raw_dir.glob("*.parquet"))]
    share_float_raw = pd.concat(frames, ignore_index=True)
    logger.info(f"全量 share_float: {len(share_float_raw)} 条（全历史起点，无区间截断）")
    lookup = build_share_float_lookup_by_date(share_float_raw, open_dates)
    logger.info(f"解禁查询表：{len(lookup)} 个交易日有记录")

    # 逐分区整列覆盖：对每日，以正确映射覆写 days_to_unlock/unlock_ratio（announcement 族）
    # 与 unlock_risk_flag（risk 族）；仅在有变化时 rewrite_partition。
    partitions = sorted(store.manifest.snapshot()["partitions"].keys())
    patched_files = 0
    patched_cells = 0
    for rel_path in partitions:
        is_ann = rel_path.endswith("/announcement.parquet")
        is_risk = rel_path.endswith("/risk.parquet")
        if not (is_ann or is_risk):
            continue
        df = pd.read_parquet(store.panel_dir.parent / rel_path)
        file_changed = 0
        day_keys = df["trade_date"].astype(str)
        for date in day_keys.unique():
            day_pos = df.index[day_keys == date]
            codes = df.loc[day_pos, "ts_code"].astype(str)
            dtu_map, ur_map = _correct_day_maps(lookup, date)
            new_dtu = codes.map(dtu_map).astype(float)  # 无记录 ⇒ NaN
            new_ur = codes.map(ur_map).astype(float)
            new_flag = _risk_flag(new_dtu)
            targets = {}
            if is_ann:
                targets = {"days_to_unlock": new_dtu, "unlock_ratio": new_ur}
            if is_risk:
                targets = {"unlock_risk_flag": new_flag}
            for col, new_series in targets.items():
                cur = df.loc[day_pos, col]
                changed = ((cur - new_series).abs() > 1e-12) | (cur.isna() != new_series.isna())
                n = int(changed.fillna(True).sum())
                if n:
                    df.loc[day_pos, col] = new_series.values
                    file_changed += n
        if file_changed:
            store.rewrite_partition(rel_path, df, reason=_REASON)
            patched_files += 1
            patched_cells += file_changed
            logger.info(f"  修正 {rel_path}：{file_changed} 单元格")

    logger.info(
        f"D-12 修复完成：改写分区 {patched_files} 个，修正单元格 {patched_cells} 个，"
        f"耗时 {(time.time()-t0)/60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
