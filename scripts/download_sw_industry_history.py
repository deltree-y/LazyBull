#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""申万行业分类历史（PIT）落地脚本（含快照归档与变更检测）。

产出（契约见 ``docs/data/sw_industry_pit_audit.md``）：
- ``data/raw/sw_industry_history.parquet``：原始四列**最新**快照
- ``data/raw/sw_industry_history/snapshot_YYYYMMDD.parquet``：按抓取日归档（不覆盖历史）
- ``data/raw/sw_industry_history/meta_YYYYMMDD.json``：抓取元数据 + 变更摘要
- ``data/clean/sw_industry_code_table.parquet``：SW2014 + SW2021 码表
- ``data/clean/sw_industry_pit.parquet``：PIT 区间表**最新**
- ``data/clean/sw_industry_pit/pit_YYYYMMDD.parquet``：按抓取日归档（不覆盖历史）

用法::

    python scripts/download_sw_industry_history.py                    # 抓取 + 变更检测 + 按需落盘
    python scripts/download_sw_industry_history.py --check-only       # 只检测，不写数据文件
    python scripts/download_sw_industry_history.py --force            # 即使无变化也重写 latest
    python scripts/download_sw_industry_history.py --skip-tushare     # 离线降级（无码表/无别名）
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lazybull.common.logger import setup_logger  # noqa: E402
from src.lazybull.data import Storage, TushareClient  # noqa: E402
from src.lazybull.data.sw_industry_pit import (  # noqa: E402
    SW_SRC_SW2014,
    SW_SRC_SW2021,
    SW_VERSION_CUTOVER,
    assign_version,
    attach_code_names,
    attach_pit,
    build_code_table,
    diff_panels,
    fetch_history_raw,
    finalize_pit,
    normalize_history,
)

LEVELS = ("L1", "L2", "L3")
DEFAULT_COVERAGE_DATES = "20120629,20150701,20200701,20240701,20260701"
RAW_ARCHIVE = "sw_industry_history"
CLEAN_ARCHIVE = "sw_industry_pit"


def load_code_table(client: TushareClient) -> pd.DataFrame:
    """拉取 SW2014 / SW2021 的 L1~L3 码表并合并。"""
    frames: List[pd.DataFrame] = []
    for src in (SW_SRC_SW2014, SW_SRC_SW2021):
        for level in LEVELS:
            frame = client.get_index_classify(level=level, src=src)
            size = 0 if frame is None else len(frame)
            logger.info(f"码表 {src}/{level}: {size} 条")
            if size:
                frames.append(frame)
    return build_code_table(frames)


def load_bse_alias(client: TushareClient) -> Dict[str, str]:
    """北交所新旧代码映射（旧 6 位 → 新 6 位）。"""
    frame = client.query("bse_mapping")
    if frame is None or len(frame) == 0:
        logger.warning("bse_mapping 返回空，北交所旧代码将无法归一")
        return {}
    old = frame["o_code"].astype(str).str[:6]
    new = frame["n_code"].astype(str).str[:6]
    alias = {o: n for o, n in zip(old, new) if o != n and o.isdigit() and n.isdigit()}
    logger.info(f"北交所代码映射 {len(alias)} 条")
    return alias


def build_panel(
    raw: pd.DataFrame,
    code_table: Optional[pd.DataFrame] = None,
    alias: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """原始表 → 标准化区间表；无码表时退化为「仅按日期切版本」。"""
    panel = normalize_history(raw)
    if code_table is None:
        panel["sw_version"] = panel["in_date"].map(
            lambda d: SW_SRC_SW2021 if d >= SW_VERSION_CUTOVER else SW_SRC_SW2014
        )
        panel["l1_code"] = panel["ind_code"].str[:2] + "0000"
        panel["l2_code"] = panel["ind_code"].str[:4] + "00"
    else:
        panel = attach_code_names(assign_version(panel, code_table), code_table)
    return finalize_pit(panel, alias=alias)


def archive(archive_dir: Path, prefix: str, stamp: str, frame: pd.DataFrame) -> Path:
    """按抓取日归档一份快照（同日同前缀已存在则不再覆盖）。"""
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{prefix}_{stamp}.parquet"
    if path.exists():
        logger.info(f"归档已存在，跳过：{path}")
        return path
    frame.to_parquet(path, index=False)
    logger.info(f"已归档：{path}（{len(frame)} 行）")
    return path


def coverage_report(pit: pd.DataFrame, storage: Storage, dates: List[str]) -> Dict[str, dict]:
    """对账：指定交易日的行业已知率（分母 = 当日有行情的股票）。"""
    report: Dict[str, dict] = {}
    for stamp in dates:
        path = storage.clean_path / "daily" / f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}.parquet"
        if not path.exists():
            logger.warning(f"缺少干净行情分区，跳过对账: {path}")
            continue
        daily = pd.read_parquet(path, columns=["ts_code"]).copy()
        daily["trade_date"] = stamp
        merged = attach_pit(pit, daily)
        known = float(merged["industry_known"].mean())
        report[stamp] = {"rows": int(len(merged)), "known_share": known}
        logger.info(f"{stamp}: 有行情 {len(merged)}, 行业已知 {known:.2%}")
    return report


def archived_fingerprints(meta_dir: Path) -> set:
    """已归档快照的内容指纹集合（用于「首次归档」与去重）。"""
    if not meta_dir.exists():
        return set()
    shas = set()
    for path in meta_dir.glob("meta_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"归档元数据不可读，跳过：{path}（{exc}）")
            continue
        value = payload.get("content_sha256")
        if isinstance(value, str):
            shas.add(value)
    return shas


def _log_diff(diff: Dict[str, object]) -> None:
    if diff.get("baseline") == "none":
        logger.info("首次落地：无历史基线，全量写入并归档")
        return
    if not diff["changed"]:
        logger.info("行业分类无变化：latest 不重写、不重复归档")
        return
    logger.warning(
        "行业分类发生变化："
        f"新增区间 {diff['added_intervals']}、失效区间 {diff['removed_intervals']}、"
        f"改类 {diff['reclassified_intervals']}；新增股票 {diff['codes_added']}、"
        f"消失股票 {diff['codes_removed']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="申万行业分类历史（PIT）落地")
    parser.add_argument(
        "--skip-tushare", action="store_true", help="不拉码表/北交所映射（离线降级）"
    )
    parser.add_argument(
        "--coverage-dates", default=DEFAULT_COVERAGE_DATES, help="对账交易日，逗号分隔"
    )
    parser.add_argument("--report-out", default=None, help="覆盖报告 JSON 输出路径")
    parser.add_argument("--diff-out", default=None, help="变更摘要 JSON 输出路径")
    parser.add_argument("--check-only", action="store_true", help="只检测变更，不写任何数据文件")
    parser.add_argument("--force", action="store_true", help="无变化时也重写 latest 并归档")
    args = parser.parse_args()

    setup_logger()
    storage = Storage()
    stamp = datetime.now().strftime("%Y%m%d")

    raw = fetch_history_raw()
    tls_verified = bool(raw.attrs.get("tls_verified", True))
    logger.info(f"申万历史分类原始行数 {len(raw)}（TLS 校验={tls_verified}）")

    code_table = alias = None
    if not args.skip_tushare:
        client = TushareClient()
        code_table = load_code_table(client)
        alias = load_bse_alias(client)

    pit = build_panel(raw, code_table=code_table, alias=alias)
    logger.info(
        f"PIT 区间表 {len(pit)} 行 / {pit['code6'].nunique()} 只 / "
        f"版本 {pit['sw_version'].value_counts().to_dict()}"
    )

    latest_path = storage.clean_path / "sw_industry_pit.parquet"
    previous = pd.read_parquet(latest_path) if latest_path.exists() else None
    diff = diff_panels(previous, pit)
    _log_diff(diff)

    if args.diff_out:
        Path(args.diff_out).write_text(
            json.dumps(diff, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    if args.check_only:
        logger.info("--check-only：跳过全部写入")
        print(json.dumps(diff, ensure_ascii=False, indent=2, default=str))
        return 0

    meta_dir = storage.raw_path / RAW_ARCHIVE
    already = archived_fingerprints(meta_dir)
    need_archive = bool(args.force or diff["changed"] or not latest_path.exists())
    need_archive = need_archive or diff["new"]["content_sha256"] not in already

    if diff["changed"] or args.force or not latest_path.exists():
        storage.save_raw(raw, "sw_industry_history", is_force=True)
        storage.save_clean(pit, "sw_industry_pit", is_force=True)
        if code_table is not None:
            storage.save_clean(code_table, "sw_industry_code_table", is_force=True)

    if need_archive:
        archive(storage.raw_path / RAW_ARCHIVE, "snapshot", stamp, raw)
        archive(storage.clean_path / CLEAN_ARCHIVE, "pit", stamp, pit)
        if code_table is not None:
            archive(storage.clean_path / CLEAN_ARCHIVE, "code_table", stamp, code_table)
        meta = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "raw_rows": int(len(raw)),
            "pit_rows": int(len(pit)),
            "codes": int(pit["code6"].nunique()),
            "tls_verified": tls_verified,
            "source_url": raw.attrs.get("source_url"),
            "content_sha256": diff["new"]["content_sha256"],
            "pit_file_sha256": (
                hashlib.sha256(latest_path.read_bytes()).hexdigest()
                if latest_path.exists()
                else None
            ),
            "diff": diff,
        }
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"meta_{stamp}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (storage.raw_path / "sw_industry_history_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    dates = [d.strip() for d in args.coverage_dates.split(",") if d.strip()]
    report = coverage_report(pit, storage, dates)
    if args.report_out:
        Path(args.report_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.info(f"完成：changed={diff['changed']}，latest={latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
