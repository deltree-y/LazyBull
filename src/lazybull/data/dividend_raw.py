# -*- coding: utf-8 -*-
"""TuShare `dividend`（分红送股）原始数据下载与年分区落盘核心。

共享实现：离线下载（scripts/raw_download）与纸面 ensure 自动补齐均复用本模块，
避免同一下载逻辑散落两处。

分区契约：按 `ann_date` 年分区，目录 data/raw/dividend/{YYYY}-12-31.parquet，
与 report_rc/share_float 等公告型事件数据一致；每日增量只重写受影响年份分区。

下载策略（性能优先）：
  - TuShare dividend 接口仅支持单值参数（ts_code/ann_date/ex_date/imp_ann_date），
    不支持日期区间查询；全历史回补采用**按股票查询**（单股历史记录仅数十行，
    全市场约 5000+ 请求，远优于按 ann_date 逐日约 7000+ 请求且覆盖更完整）。
    - 断点续传：逐股覆盖状态持久化为 data/empty/pending/failed；仅前两者跳过，
        pending/failed 自动重试（--force 全量重下）。
  - 去重键 (ts_code, end_date, div_proc, ann_date)：同一分红方案存在
    预案/决案/实施多行，div_proc 必入键；同键冲突按 TuShare `update_flag=1`
    最新语义确定性决胜。
"""

import json
import os
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set

import pandas as pd
from loguru import logger

from .financial_statement_versions import deduplicate_prefer_latest_update_flag
from .storage import Storage
from .tushare_client import TushareClient
from .tushare_client.dividend import DIVIDEND_FIELDS

# 去重键：分红方案身份（div_proc 必入键，预案/决案/实施各行保留）
DIVIDEND_DEDUP_COLS = ["ts_code", "end_date", "div_proc", "ann_date"]

# 全历史回补并发（dividend 单请求响应快，走全局限频 500 次/分钟留余量）
_DIVIDEND_DOWNLOAD_CONCURRENCY = 8

# 逐股全历史覆盖状态：成功空结果也必须持久化，失败/中断状态保持可重试。
_DIVIDEND_COVERAGE_FILE = "_stock_coverage.json"
_DIVIDEND_COVERAGE_VERSION = 1
_DIVIDEND_COVERED_STATUSES = {"data", "empty"}
_DIVIDEND_COVERAGE_STATUSES = _DIVIDEND_COVERED_STATUSES | {"pending", "failed"}
_CONCAT_ALL_NA_WARNING = (
    r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
)


def _norm_date_series(s: pd.Series) -> pd.Series:
    """将日期列统一为 YYYYMMDD 字符串（容错 20240101 / 2024-01-01 / datetime）。"""
    out = s.astype(str).str.strip().str.replace("-", "", regex=False).str[:8]
    return out


def _concat_dividend_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """原样合并 dividend 数据，仅屏蔽 pandas 已知的 dtype 推断告警。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=_CONCAT_ALL_NA_WARNING,
        )
        return pd.concat(frames, ignore_index=True)


def _load_dividend_coverage(storage: Storage) -> Optional[Dict[str, str]]:
    """读取逐股全历史覆盖状态；文件不存在或损坏时返回 None 触发安全迁移。"""
    path = storage.raw_path / "dividend" / _DIVIDEND_COVERAGE_FILE
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("version") != _DIVIDEND_COVERAGE_VERSION:
            raise ValueError(f"不支持的覆盖状态版本: {payload.get('version')!r}")
        stocks = payload.get("stocks")
        if not isinstance(stocks, dict):
            raise ValueError("stocks 必须为字典")
        statuses = {str(code): str(status) for code, status in stocks.items()}
        invalid = sorted(set(statuses.values()) - _DIVIDEND_COVERAGE_STATUSES)
        if invalid:
            raise ValueError(f"包含未知状态: {invalid}")
        return statuses
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"[dividend] 读取逐股覆盖状态失败，将按现有分区迁移: {exc}")
        return None


def _save_dividend_coverage(storage: Storage, statuses: Dict[str, str]) -> None:
    """原子保存逐股全历史覆盖状态。"""
    path = storage.raw_path / "dividend" / _DIVIDEND_COVERAGE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="_stock_coverage.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "version": _DIVIDEND_COVERAGE_VERSION,
                    "stocks": dict(sorted(statuses.items())),
                },
                file,
                ensure_ascii=False,
            )
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def _existing_dividend_df(storage: Storage) -> Optional[pd.DataFrame]:
    """从年分区加载已有 dividend 数据（分区模式必须按分区枚举读取）。

    注意：`storage.load_raw("dividend")` 只读单文件，对分区目录返回 None，
    不能用于分区数据集；此处与 loader 层保持一致按分区枚举加载。
    """
    frames: List[pd.DataFrame] = []
    for partition in storage.list_partitions("raw", "dividend"):
        df = storage.load_raw_by_date("dividend", partition)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    return _concat_dividend_frames(frames)


def _save_dividend_by_year(storage: Storage, df: pd.DataFrame) -> None:
    """按 ann_date 年分区落盘（全量重写各年分区，数据量小成本可控）。"""
    if df is None:
        return
    desired_partitions: Set[str] = set()
    if len(df) > 0:
        if "ann_date" not in df.columns:
            raise ValueError("dividend 数据缺少 ann_date，无法按年分区落盘")
        work = df.copy()
        work["_year"] = _norm_date_series(work["ann_date"]).str[:4]
        for year, grp in work.groupby("_year", sort=True):
            partition = f"{year}-12-31"
            desired_partitions.add(partition)
            storage.save_raw_by_date(grp.drop(columns=["_year"]), "dividend", partition)

    removed = 0
    for partition in storage.list_partitions("raw", "dividend"):
        if partition in desired_partitions:
            continue
        path = storage.raw_path / "dividend" / f"{partition}.parquet"
        if path.exists():
            path.unlink()
            removed += 1
    logger.info(
        f"[dividend] 已按 ann_date 年分区落盘 {len(df)} 条记录"
        + (f"，移除 {removed} 个旧分区" if removed else "")
    )


def _deduplicate_dividend(df: pd.DataFrame) -> pd.DataFrame:
    """去重：键 (ts_code, end_date, div_proc, ann_date)，update_flag=1 最新语义决胜。"""
    if df is None or len(df) == 0:
        return df
    work = df.copy()
    for col in ("ann_date", "ex_date", "imp_ann_date", "end_date"):
        if col in work.columns:
            work[col] = _norm_date_series(work[col])
    if "ann_date" not in work.columns:
        raise ValueError("dividend 数据缺少 ann_date，无法去重或按年分区")
    valid_ann_date = work["ann_date"].str.match(r"^\d{8}$", na=False)
    valid_ann_date &= pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce").notna()
    invalid_count = int((~valid_ann_date).sum())
    if invalid_count > 0:
        invalid_codes = work.loc[~valid_ann_date, "ts_code"].astype(str).unique().tolist()
        logger.warning(
            f"[dividend] 忽略 {invalid_count} 条 ann_date 缺失或非法的记录，"
            f"股票示例: {', '.join(invalid_codes[:10])}"
            + (" ..." if len(invalid_codes) > 10 else "")
        )
        work = work.loc[valid_ann_date].copy()
    if len(work) == 0:
        return work.reset_index(drop=True)
    return deduplicate_prefer_latest_update_flag(
        work,
        list(DIVIDEND_DEDUP_COLS),
        deterministic_ties=True,
    )


def download_dividend_full(
    client: TushareClient,
    storage: Storage,
    stock_basic: pd.DataFrame,
    concurrency: Optional[int] = None,
    force: bool = False,
    existing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """全市场分红送股全历史下载（按股查询 + 年分区落盘）。

    Args:
        client: TuShare 客户端
        storage: Storage 实例
        stock_basic: 股票基础信息（含 ts_code 列表）
        concurrency: 并发线程数（None 使用默认保守值）
        force: 强制重下全部（忽略已下载股票集合）
        existing_df: 调用方已加载的全量数据；提供时避免重复扫描所有年分区

    Returns:
        全量 dividend DataFrame（已去重，含 ts_code/ann_date/ex_date/imp_ann_date 等）
    """
    if stock_basic is None or len(stock_basic) == 0 or "ts_code" not in stock_basic.columns:
        raise ValueError("stock_basic 为空或缺少 ts_code 列，无法下载分红送股数据")

    all_codes = sorted(stock_basic["ts_code"].astype(str).unique().tolist())
    # 旧数据始终加载：成功股票按本次全历史结果整体替换，失败股票保留旧行。
    if existing_df is None:
        existing_df = _existing_dividend_df(storage)
    record_codes = (
        set(existing_df["ts_code"].astype(str))
        if existing_df is not None and len(existing_df) > 0 and "ts_code" in existing_df.columns
        else set()
    )
    # 存量数据缺 base_share 列时（历史下载未显式请求该字段），支付率因子无法计算
    # 现金分红总额，必须把已覆盖股票整体降级为 failed 触发自动重下；不得静默沿用
    # 缺列数据产出全 NaN 因子。
    base_share_missing = (
        existing_df is not None
        and len(existing_df) > 0
        and "base_share" not in existing_df.columns
    )
    if base_share_missing:
        logger.warning(
            "[dividend] 存量数据缺少 base_share 列（历史下载未请求该字段），"
            "将自动重下全部已覆盖股票以补齐基准股本"
        )
    coverage_statuses = _load_dividend_coverage(storage)
    coverage_was_missing = coverage_statuses is None
    coverage_changed = coverage_was_missing
    if coverage_statuses is None:
        coverage_statuses = {code: "data" for code in record_codes}
    else:
        # 分区被手工删除时，data 状态必须退回 failed 以触发重查；手工导入的
        # 新股票则迁移为 data，但 force 失败后保留的旧行仍由 failed 状态主导。
        for code, status in list(coverage_statuses.items()):
            if status == "data" and code not in record_codes:
                coverage_statuses[code] = "failed"
                coverage_changed = True
        for code in record_codes:
            if code not in coverage_statuses or coverage_statuses[code] == "empty":
                coverage_statuses[code] = "data"
                coverage_changed = True
    if base_share_missing:
        # 缺 base_share 列的旧数据必须重下：已覆盖状态整体降级为 failed
        for code, status in list(coverage_statuses.items()):
            if status == "data":
                coverage_statuses[code] = "failed"
                coverage_changed = True
    existing_codes = {
        code for code, status in coverage_statuses.items() if status in _DIVIDEND_COVERED_STATUSES
    }
    if existing_codes:
        logger.info(f"[dividend] 已覆盖 {len(existing_codes)} 只股票，将跳过")

    pending_codes = [c for c in all_codes if c not in existing_codes] if not force else all_codes
    if not pending_codes:
        if coverage_changed:
            _save_dividend_coverage(storage, coverage_statuses)
        logger.info("[dividend] 全部股票已覆盖，无需下载。如需重下加 --force")
        return existing_df if existing_df is not None else pd.DataFrame()

    # 两阶段提交：网络查询及分区写入期间保持 pending；进程中断后下次会安全重试。
    for code in pending_codes:
        coverage_statuses[code] = "pending"
    _save_dividend_coverage(storage, coverage_statuses)

    logger.info(
        f"[dividend] 按股票全历史下载: 共 {len(all_codes)} 只, "
        f"跳过 {len(all_codes) - len(pending_codes)}, 待下 {len(pending_codes)}"
    )

    new_dfs: List[pd.DataFrame] = []
    stats_lock = threading.Lock()
    success = empty = 0
    data_codes: Set[str] = set()
    empty_codes: Set[str] = set()
    failed_codes: List[str] = []

    def _fetch_one(code: str) -> Optional[pd.DataFrame]:
        """查询单只股票全历史分红送股记录（异常时返回 None 并记入失败清单）。

        显式携带 DIVIDEND_FIELDS：TuShare 默认字段不返回 base_share，缺失则
        无法计算现金分红总额（支付率因子会静默全 NaN）。
        """
        try:
            return client.query("dividend", ts_code=code, fields=DIVIDEND_FIELDS)
        except Exception as e:
            logger.warning(f"[dividend] {code} 下载失败: {e}")
            return None

    workers = concurrency or _DIVIDEND_DOWNLOAD_CONCURRENCY
    done = 0
    if workers <= 1 or len(pending_codes) <= 1:
        for code in pending_codes:
            df = _fetch_one(code)
            done += 1
            with stats_lock:
                if df is not None and len(df) > 0:
                    new_dfs.append(df)
                    data_codes.add(code)
                    success += 1
                elif df is None:
                    failed_codes.append(code)
                else:
                    empty_codes.add(code)
                    empty += 1
            if done % 200 == 0 or done == len(pending_codes):
                logger.info(f"[dividend] 进度 {done}/{len(pending_codes)}")
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dividend-dl") as pool:
            futures = {pool.submit(_fetch_one, code): code for code in pending_codes}
            for future in as_completed(futures):
                code = futures[future]
                df = future.result()
                done += 1
                with stats_lock:
                    if df is not None and len(df) > 0:
                        new_dfs.append(df)
                        data_codes.add(code)
                        success += 1
                    elif df is None:
                        failed_codes.append(code)
                    else:
                        empty_codes.add(code)
                        empty += 1
                if done % 200 == 0 or done == len(pending_codes):
                    logger.info(
                        f"[dividend] 进度 {done}/{len(pending_codes)} "
                        f"(有数据={success} 空={empty} 失败={len(failed_codes)})"
                    )

    if failed_codes:
        logger.warning(
            f"[dividend] {len(failed_codes)} 只股票下载失败（本次未写入分区，下次运行重试）:"
            f" {', '.join(failed_codes[:20])}" + (" ..." if len(failed_codes) > 20 else "")
        )
    logger.info(f"[dividend] 下载完成: 有数据={success} 空={empty} 失败={len(failed_codes)}")

    successful_codes = data_codes | empty_codes
    if successful_codes:
        retained = existing_df
        if retained is not None and len(retained) > 0:
            if "ts_code" not in retained.columns:
                raise ValueError("已有 dividend 数据缺少 ts_code，无法按成功股票替换")
            retained = retained[~retained["ts_code"].astype(str).isin(successful_codes)]
        frames = []
        if retained is not None and len(retained) > 0:
            frames.append(retained)
        frames.extend(new_dfs)
        if frames:
            merged = _deduplicate_dividend(_concat_dividend_frames(frames))
        else:
            columns = existing_df.columns if existing_df is not None else []
            merged = pd.DataFrame(columns=columns)
        _save_dividend_by_year(storage, merged)
    else:
        merged = existing_df if existing_df is not None else pd.DataFrame()

    for code in data_codes:
        coverage_statuses[code] = "data"
    for code in empty_codes:
        coverage_statuses[code] = "empty"
    for code in failed_codes:
        coverage_statuses[code] = "failed"
    _save_dividend_coverage(storage, coverage_statuses)
    return merged
