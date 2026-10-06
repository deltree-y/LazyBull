# -*- coding: utf-8 -*-
"""v2 数据面参考实现 ``PanelDataStore``（DataStore Protocol，docs/contracts/protocols.md §1.1）。

路径布局（data_root 默认 ``data/``，测试传 tmp_path）：

- panel 热区：``features/panel/YYYYMMDD/<group>.parquet``（日分区）
- panel 冷区：``features/panel_archive/YYYY-MM/<group>.parquet``（封存月压实）
- labels：``labels/<label_name>/YYYYMMDD.parquet``（日分区，不分冷热）
- manifest：``features/manifest.json``
- 假设台账：``ledger/hypotheses.jsonl``（append-only，本类是唯一写入口）
- 单机写锁：``features/.write.lock``

写入纪律（协议 §8 R4-M8）：
- append_features 两阶段提交：临时文件 → 算指纹 → 与 manifest 已有指纹比对
  （一致放行 / 不一致 RuntimeError）→ 原子 replace + manifest 原子登记；
- append_labels 封存前幂等可重写、封存后拒绝改写；
- append_ledger_entry 防冲突（hypothesis_id 唯一）；
- 所有写路径持单机写锁 + 提交前孤儿 .tmp GC。
"""

from __future__ import annotations

import json
import os
import socket
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import psutil
from loguru import logger

from src.lazybull.v2.common.types import (
    DataState,
    FeatureQuery,
    LabelQuery,
    LedgerEntry,
    PanelFrame,
    TradeDate,
    TSCode,
)
from src.lazybull.v2.store.manifest import (
    VALID_GROUPS,
    Manifest,
    sha256_16_of_file,
)

__all__ = ["PanelDataStore"]

_KEY_COLUMNS = ("trade_date", "ts_code")
_LEDGER_KINDS = frozenset({"prereg", "conclusion"})
_MATURITY_VALUES = frozenset({"forming", "sealed"})


def _date_str(date: TradeDate | str) -> str:
    """TradeDate / YYYYMMDD 字符串统一为字符串口径。"""
    return str(date)


def _ts_code_str(code: TSCode | str) -> str:
    return str(getattr(code, "value", code))


def _empty_panel_df(columns: Sequence[str]) -> pd.DataFrame:
    """空面板（index=(trade_date, ts_code)），load 无命中时返回。"""
    idx = pd.MultiIndex.from_arrays([[], []], names=list(_KEY_COLUMNS))
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}, index=idx)


class _WriteLock:
    """单机写锁（``features/.write.lock``，带 holder 信息 + 获取时间）。

    - 持锁期间第二个实例操作 ⇒ RuntimeError；
    - stale 检测：holder 进程不存在则抢占并告警（loguru warning）；
    - 对同一 PanelDataStore 实例可重入（深度计数）。
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._depth = 0

    def acquire(self) -> None:
        if self._depth > 0:  # 同实例重入
            self._depth += 1
            return
        self._create_lock_file()
        self._depth = 1

    def release(self) -> None:
        if self._depth == 0:
            raise RuntimeError(f"写锁未持有却 release: {self._path}")
        self._depth -= 1
        if self._depth == 0:
            self._path.unlink(missing_ok=True)

    def __enter__(self) -> "_WriteLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def _create_lock_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        holder = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
        }
        for attempt in range(2):
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._handle_existing_lock(attempt)
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(holder, f, ensure_ascii=False)
                return
        raise RuntimeError(f"写锁抢占后仍无法创建: {self._path}")

    def _handle_existing_lock(self, attempt: int) -> None:
        holder = self._read_holder()
        pid = holder.get("pid") if holder else None
        if isinstance(pid, int) and psutil.pid_exists(pid):
            raise RuntimeError(
                f"写锁已被持有: {self._path}（holder={holder}）；持锁期间禁止第二个实例写入"
            )
        logger.warning(
            f"检测到 stale 写锁（holder={holder}），原 holder 进程不存在，抢占: {self._path}"
        )
        self._path.unlink(missing_ok=True)
        if attempt >= 1:
            raise RuntimeError(f"写锁 stale 清理后重试仍失败: {self._path}")

    def _read_holder(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"写锁文件读取失败（按 stale 处理）: {self._path}（{exc}）")
            return {}


class PanelDataStore:
    """DataStore Protocol 的 panel 实现（构造参数 data_root 默认 ``data/``）。"""

    def __init__(self, data_root: Path | str = "data") -> None:
        self._root = Path(data_root)
        self._manifest: Manifest | None = None
        self._lock = _WriteLock(self.features_dir / ".write.lock")
        self._manifest_defer = 0  # defer_manifest_save 计数（批量回填防 manifest O(n²) 落盘）

    # ---------- 路径布局 ----------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def features_dir(self) -> Path:
        return self._root / "features"

    @property
    def panel_dir(self) -> Path:
        return self.features_dir / "panel"

    @property
    def archive_dir(self) -> Path:
        return self.features_dir / "panel_archive"

    @property
    def labels_dir(self) -> Path:
        return self._root / "labels"

    @property
    def ledger_path(self) -> Path:
        return self._root / "ledger" / "hypotheses.jsonl"

    @property
    def manifest(self) -> Manifest:
        """manifest 内存视图（懒加载；文件不存在时按空骨架起步）。"""
        if self._manifest is None:
            self._manifest = Manifest(self.features_dir / "manifest.json")
        return self._manifest

    def _maybe_save_manifest(self) -> None:
        """manifest 落盘（延迟模式外）；延迟模式内跳过（出口统一落盘）。"""
        if self._manifest_defer == 0:
            self.manifest.save()

    def defer_manifest_save(self):
        """批量写入期延迟 manifest 落盘的上下文管理器（可重入）。

        动机：manifest 随分区数线性膨胀，逐分区落盘是全量回填的 O(n²) 瓶颈。
        崩溃语义：延迟期内崩溃 ⇒ 盘上 manifest 不含已写分区登记 ⇒ 重跑按
        「未登记允许重写」覆盖孤儿分区并重新登记，两阶段提交语义不变。
        """
        return self._ManifestDefer(self)

    class _ManifestDefer:
        """defer_manifest_save 的上下文实现（计数可重入）。"""

        def __init__(self, store: "PanelDataStore") -> None:
            self._store = store

        def __enter__(self) -> None:
            self._store._manifest_defer += 1
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            self._store._manifest_defer -= 1
            if self._store._manifest_defer == 0:
                self._store.manifest.save()  # 出口统一落盘（含异常路径——已写分区先落账）

    def write_lock(self) -> _WriteLock:
        """单机写锁上下文管理器（``with store.write_lock():``）。"""
        return self._lock

    # ========== 写入：features ==========

    def append_features(self, date: TradeDate, group: str, df: pd.DataFrame) -> None:
        """追加特征日分区（两阶段提交 + 显式冲突检测，协议 §8 防冲突语义）。"""
        date_str = _date_str(date)
        self._validate_feature_write(date_str, group, df)
        rel_path = f"panel/{date_str}/{group}.parquet"
        with self._lock:
            part_dir = self.panel_dir / date_str
            part_dir.mkdir(parents=True, exist_ok=True)
            self._gc_orphans(part_dir)
            final_path = part_dir / f"{group}.parquet"
            tmp_path = part_dir / f"{group}.parquet.tmp"
            df.to_parquet(tmp_path, index=False)
            fingerprint = sha256_16_of_file(tmp_path)
            existing_fp = self.manifest.partition_fingerprint(rel_path)
            if existing_fp is not None:
                tmp_path.unlink(missing_ok=True)
                if existing_fp == fingerprint:
                    logger.info(f"分区重复写入指纹一致，防冲突放行: {rel_path}")
                    return
                raise RuntimeError(
                    f"分区指纹冲突: {rel_path}（已登记 {existing_fp} ≠ 新内容 {fingerprint}）；"
                    "同日同族重复写仅在内容逐位一致时放行（R4-M8 防冲突语义）"
                )
            os.replace(tmp_path, final_path)
            self.manifest.register_partition(rel_path, fingerprint, rows=len(df), zone="hot")
            self._maybe_save_manifest()
            logger.info(f"特征分区落盘: {rel_path}（{len(df)} 行，指纹 {fingerprint}）")

    def _validate_feature_write(self, date_str: str, group: str, df: pd.DataFrame) -> None:
        """写前校验：族名合法、键列齐全、单日一致、列已登记且族一致、分区未封存。"""
        if group not in VALID_GROUPS:
            raise RuntimeError(f"非法列族 {group!r}（合法值 {sorted(VALID_GROUPS)}）")
        missing_keys = [c for c in _KEY_COLUMNS if c not in df.columns]
        if missing_keys:
            raise RuntimeError(f"特征分区缺键列 {missing_keys}（{date_str}/{group}）")
        dates = df["trade_date"].astype(str).unique()
        if len(dates) != 1 or dates[0] != date_str:
            raise RuntimeError(
                f"特征分区必须单日且与 date 参数一致: 期望 {date_str}，实得 {sorted(dates)[:5]}"
            )
        self._validate_group_columns(group, df, context=f"{date_str}/{group}")
        month = f"{date_str[:4]}-{date_str[4:6]}"
        archive_file = self.archive_dir / month / f"{group}.parquet"
        if archive_file.exists():
            raise RuntimeError(
                f"分区已封存（冷区存在同月同族文件 {archive_file}），拒绝热区写入: {date_str}/{group}"
            )

    def _validate_group_columns(self, group: str, df: pd.DataFrame, context: str) -> None:
        """列级校验：非键列必须已登记 manifest 且登记族与目标族一致。"""
        for col in df.columns:
            if col in _KEY_COLUMNS:
                continue
            try:
                meta = self.manifest.column_meta(col)
            except KeyError as exc:
                raise RuntimeError(f"列未登记 manifest，拒绝写入: {col!r}（{context}）") from exc
            if meta["group"] != group:
                raise RuntimeError(
                    f"列 {col!r} 登记在族 {meta['group']!r}，拒绝写入族 {group!r} 的分区"
                )

    # ========== 写入：features 冷区（封存月分区） ==========

    def repair_archive_partition(
        self, month: str, group: str, df_new_days: pd.DataFrame, reason: str
    ) -> dict[str, Any]:
        """修复性合并冷区月分区：仅存单元格逐值不变、仅追加缺失日（构建缺陷修复通道）。

        与 `append_archive_features` 的「历史分区禁止原地修改」不同：本 API 只允许
        行超集追加（新日的 (trade_date, ts_code) 键不得与存量相交），并在 manifest
        登记修复理由（`repairs` 审计轨迹）。口径修正仍走新列名/版本升级，禁止借用本通道。
        """
        if not reason or not reason.strip():
            raise RuntimeError("修复性合并必须登记理由（reason 非空）")
        rel_path = f"panel_archive/{month}/{group}.parquet"
        with self._lock:
            final_path = self.archive_dir / month / f"{group}.parquet"
            old_fp = self.manifest.partition_fingerprint(rel_path)
            if old_fp is None or not final_path.exists():
                raise RuntimeError(f"修复目标分区不存在或未登记: {rel_path}（新建请走 append）")
            df_old = pd.read_parquet(final_path)
            new_days = sorted(df_new_days["trade_date"].astype(str).unique().tolist())
            overlap = sorted(set(df_old["trade_date"].astype(str)) & set(new_days))
            if overlap:
                raise RuntimeError(f"修复追加日与存量相交: {overlap[:5]}（{rel_path}）")
            keys_old = pd.MultiIndex.from_frame(df_old[["trade_date", "ts_code"]].astype(str))
            keys_new = pd.MultiIndex.from_frame(df_new_days[["trade_date", "ts_code"]].astype(str))
            if not keys_old.is_unique or not keys_new.is_unique:
                raise RuntimeError(f"修复合并键 (trade_date, ts_code) 不唯一（{rel_path}）")
            merged = (
                pd.concat([df_old, df_new_days], ignore_index=True)
                .sort_values(["trade_date", "ts_code"])
                .reset_index(drop=True)
            )
            self._validate_archive_write(month, group, merged)
            part_dir = self.archive_dir / month
            tmp_path = part_dir / f"{group}.parquet.tmp"
            merged.to_parquet(tmp_path, index=False)
            new_fp = sha256_16_of_file(tmp_path)
            os.replace(tmp_path, final_path)
            self.manifest.register_partition(rel_path, new_fp, rows=len(merged), zone="archive")
            self.manifest.register_repair(rel_path, reason, new_days, old_fp, new_fp)
            self._maybe_save_manifest()
            logger.warning(
                f"冷区修复性合并: {rel_path}（追加 {len(new_days)} 日 {new_days}，"
                f"{len(df_old)}→{len(merged)} 行；理由：{reason}）"
            )
            return {"rel_path": rel_path, "added_dates": new_days, "rows": len(merged)}

    def replace_archive_days(
        self, month: str, group: str, df_days: pd.DataFrame, reason: str
    ) -> dict[str, Any]:
        """修复性按日替换冷区月分区：仅替换指定日的行（存量其余日逐位保留）。

        与 `repair_archive_partition`（追加缺失日）互补：本通道服务「已落盘日内容
        错误」的修复（如构建窗口截断/环境污染），要求目标日 ⊆ 存量日集；存量其余日
        的行在新内容中逐位保留（拼接自原文件，不重算）。理由强制登记（manifest.repairs
        记 mode="replace_days"）。
        """
        if not reason or not reason.strip():
            raise RuntimeError("修复性替换必须登记理由（reason 非空）")
        rel_path = f"panel_archive/{month}/{group}.parquet"
        with self._lock:
            final_path = self.archive_dir / month / f"{group}.parquet"
            old_fp = self.manifest.partition_fingerprint(rel_path)
            if old_fp is None or not final_path.exists():
                raise RuntimeError(f"修复目标分区不存在或未登记: {rel_path}（新建请走 append）")
            df_old = pd.read_parquet(final_path)
            days = sorted(df_days["trade_date"].astype(str).unique().tolist())
            if not days:
                raise RuntimeError(f"修复替换内容为空（{rel_path}）")
            existing_days = set(df_old["trade_date"].astype(str))
            missing = sorted(set(days) - existing_days)
            if missing:
                raise RuntimeError(f"修复替换日不在存量分区: {missing}（{rel_path}；缺失日请走 repair 追加）")
            kept = df_old[~df_old["trade_date"].astype(str).isin(days)]
            merged = (
                pd.concat([kept, df_days], ignore_index=True)
                .sort_values(["trade_date", "ts_code"])
                .reset_index(drop=True)
            )
            self._validate_archive_write(month, group, merged)
            part_dir = self.archive_dir / month
            tmp_path = part_dir / f"{group}.parquet.tmp"
            merged.to_parquet(tmp_path, index=False)
            new_fp = sha256_16_of_file(tmp_path)
            os.replace(tmp_path, final_path)
            self.manifest.register_partition(rel_path, new_fp, rows=len(merged), zone="archive")
            self.manifest.register_repair(rel_path, f"[replace_days] {reason}", days, old_fp, new_fp)
            self._maybe_save_manifest()
            logger.warning(
                f"冷区修复性替换: {rel_path}（替换 {len(days)} 日 {days}；理由：{reason}）"
            )
            return {"rel_path": rel_path, "replaced_dates": days, "rows": len(merged)}

    def rewrite_partition(self, rel_path: str, df: pd.DataFrame, reason: str) -> None:
        """修复性整体重写单个分区（列级修正/缺陷修复通道，两阶段提交 + 理由登记）。

        区别于 append（指纹一致才放行）：本通道允许内容变化，但必须登记理由
        （manifest.repairs 记 mode="rewrite_partition"）。仅服务已登记的构建缺陷
        修复（如 D-12 前瞻窗截断的列值修正）；口径演进仍走新列名/版本升级。
        rel_path 形如 ``panel/YYYYMMDD/<group>.parquet`` 或
        ``panel_archive/YYYY-MM/<group>.parquet``。
        """
        if not reason or not reason.strip():
            raise RuntimeError("修复性重写必须登记理由（reason 非空）")
        with self._lock:
            if rel_path.startswith("panel_archive/"):
                final_path = self.archive_dir / rel_path.removeprefix("panel_archive/")
            elif rel_path.startswith("panel/"):
                final_path = self.panel_dir / rel_path.removeprefix("panel/")
            else:
                raise RuntimeError(f"非法分区路径（须 panel/ 或 panel_archive/ 前缀）: {rel_path}")
            old_fp = self.manifest.partition_fingerprint(rel_path)
            if old_fp is None or not final_path.exists():
                raise RuntimeError(f"修复目标分区不存在或未登记: {rel_path}（新建请走 append）")
            for key_col in ("trade_date", "ts_code"):
                if key_col not in df.columns:
                    raise RuntimeError(f"修复重写缺键列 {key_col!r}（{rel_path}）")
            part_dir = final_path.parent
            tmp_path = part_dir / f"{final_path.name}.tmp"
            df.to_parquet(tmp_path, index=False)
            new_fp = sha256_16_of_file(tmp_path)
            os.replace(tmp_path, final_path)
            zone = "archive" if rel_path.startswith("panel_archive/") else "hot"
            self.manifest.register_partition(rel_path, new_fp, rows=len(df), zone=zone)
            self.manifest.register_repair(rel_path, f"[rewrite_partition] {reason}", [], old_fp, new_fp)
            self._maybe_save_manifest()
            logger.warning(f"分区修复性重写: {rel_path}（{len(df)} 行；理由：{reason}）")

    def append_archive_features(self, month: str, group: str, df: pd.DataFrame) -> None:
        """落冷区月分区（``panel_archive/YYYY-MM/<group>.parquet``；两阶段提交 + 写锁）。

        历史分区禁止原地修改：指纹一致 no-op、不一致 raise RuntimeError
        （口径修正 = 新列名 / 版本升级，不是覆盖旧分区）。
        """
        self._validate_archive_write(month, group, df)
        rel_path = f"panel_archive/{month}/{group}.parquet"
        with self._lock:
            part_dir = self.archive_dir / month
            part_dir.mkdir(parents=True, exist_ok=True)
            self._gc_orphans(part_dir)
            final_path = part_dir / f"{group}.parquet"
            tmp_path = part_dir / f"{group}.parquet.tmp"
            df.to_parquet(tmp_path, index=False)
            fingerprint = sha256_16_of_file(tmp_path)
            existing_fp = self.manifest.partition_fingerprint(rel_path)
            if existing_fp is not None:
                tmp_path.unlink(missing_ok=True)
                if existing_fp == fingerprint:
                    logger.info(f"冷区分区指纹一致 no-op: {rel_path}")
                    return
                raise RuntimeError(
                    f"冷区历史分区禁止原地修改: {rel_path}"
                    f"（已登记 {existing_fp} ≠ 新内容 {fingerprint}；口径修正请走新列名/版本升级）"
                )
            os.replace(tmp_path, final_path)
            self.manifest.register_partition(rel_path, fingerprint, rows=len(df), zone="archive")
            self._maybe_save_manifest()
            logger.info(f"冷区分区落盘: {rel_path}（{len(df)} 行，指纹 {fingerprint}）")

    def _validate_archive_write(self, month: str, group: str, df: pd.DataFrame) -> None:
        """冷区写前校验：月份格式、族名、键列、全部日期落在月内、列已登记且族一致。"""
        if len(month) != 7 or month[4] != "-":
            raise RuntimeError(f"冷区月份须为 YYYY-MM 格式，实得 {month!r}")
        if group not in VALID_GROUPS:
            raise RuntimeError(f"非法列族 {group!r}（合法值 {sorted(VALID_GROUPS)}）")
        missing_keys = [c for c in _KEY_COLUMNS if c not in df.columns]
        if missing_keys:
            raise RuntimeError(f"冷区分区缺键列 {missing_keys}（{month}/{group}）")
        dates = df["trade_date"].astype(str)
        out_of_month = dates[(dates.str.slice(0, 4) + "-" + dates.str.slice(4, 6)) != month]
        if len(out_of_month) > 0:
            raise RuntimeError(
                f"冷区月分区含月外日期: {sorted(out_of_month.unique())[:5]}（{month}/{group}）"
            )
        self._validate_group_columns(group, df, context=f"{month}/{group}")

    @staticmethod
    def _gc_orphans(directory: Path) -> None:
        """提交前清理同目录残留 .tmp（两阶段提交崩溃孤儿），逐条登记日志。"""
        for orphan in sorted(directory.glob("*.tmp")):
            orphan.unlink()
            logger.warning(f"孤儿临时文件 GC: {orphan}")

    # ========== 读取：features ==========

    def load_features(self, query: FeatureQuery) -> PanelFrame:
        """加载特征面板（manifest 校验列名合法性与 available_from 越界）。

        core 左表语义在外层；本方法只按请求列所在族读文件，同族 concat、
        跨族 (trade_date, ts_code) outer join 后按请求列裁剪，族缺日整族 NaN 计数登记。
        """
        col_meta = self._check_feature_query(query)
        start_str, end_str = _date_str(query.start_date), _date_str(query.end_date)
        groups: dict[str, list[str]] = {}
        for col, meta in col_meta.items():
            groups.setdefault(str(meta["group"]), []).append(col)
        hot_dates = self._hot_dates_in_range(start_str, end_str)
        frames: list[pd.DataFrame] = []
        for group, cols in groups.items():
            frame = self._load_group_frame(group, cols, start_str, end_str, hot_dates)
            if frame is not None:
                frames.append(frame)
        data_cols = [c for c in query.columns if c not in _KEY_COLUMNS]
        df = self._merge_group_frames(frames, data_cols)
        if query.universe:
            codes = {_ts_code_str(c) for c in query.universe}
            df = df[df.index.get_level_values("ts_code").isin(codes)]
        available_from = {
            col: TradeDate.from_str(str(meta["available_from"]))
            for col, meta in col_meta.items()
            if meta.get("available_from")
        }
        return PanelFrame(
            df=df,
            manifest_version=self.manifest.manifest_version,
            available_from=available_from,
        )

    def _check_feature_query(self, query: FeatureQuery) -> dict[str, dict[str, Any]]:
        """列名登记校验 + available_from PIT 越界校验，返回 {列: 元数据}。"""
        col_meta: dict[str, dict[str, Any]] = {}
        for col in query.columns:
            if col in _KEY_COLUMNS:  # 键列以 index 形态返回，无需读文件
                continue
            try:
                meta = self.manifest.column_meta(col)
            except KeyError as exc:
                raise ValueError(f"特征列未登记 manifest: {col!r}") from exc
            available_from = meta.get("available_from")
            if available_from and query.start_date < TradeDate.from_str(str(available_from)):
                raise ValueError(
                    f"available_from 越界（PIT）: 列 {col!r} 可用起点 {available_from}，"
                    f"查询起点 {_date_str(query.start_date)} 更早"
                )
            col_meta[col] = meta
        return col_meta

    def _hot_dates_in_range(self, start_str: str, end_str: str) -> set[str]:
        """热区在查询区间内的全部日期分区名（缺日计数的并集基准）。"""
        if not self.panel_dir.exists():
            return set()
        return {
            d.name
            for d in self.panel_dir.iterdir()
            if d.is_dir()
            and len(d.name) == 8
            and d.name.isdigit()
            and start_str <= d.name <= end_str
        }

    def _load_group_frame(
        self,
        group: str,
        cols: list[str],
        start_str: str,
        end_str: str,
        hot_dates: set[str],
    ) -> pd.DataFrame | None:
        """读一族在区间内的热区日分区 + 冷区月分区（冷区优先），concat 后裁剪请求列。"""
        parts: list[pd.DataFrame] = []
        archived_months = {
            m
            for m in self._months_in_range(start_str, end_str)
            if (self.archive_dir / m / f"{group}.parquet").exists()
        }
        for month in sorted(archived_months):
            month_df = pd.read_parquet(self.archive_dir / month / f"{group}.parquet")
            dates = month_df["trade_date"].astype(str)
            month_df = month_df[(dates >= start_str) & (dates <= end_str)]
            if not month_df.empty:
                parts.append(month_df)
        present_hot: set[str] = set()
        for date_str in sorted(hot_dates):
            if f"{date_str[:4]}-{date_str[4:6]}" in archived_months:
                continue  # 该月已封存，冷区为准
            hot_file = self.panel_dir / date_str / f"{group}.parquet"
            if hot_file.exists():
                parts.append(pd.read_parquet(hot_file))
                present_hot.add(date_str)
        missing = (
            hot_dates
            - present_hot
            - {d for d in hot_dates if f"{d[:4]}-{d[4:6]}" in archived_months}
        )
        if missing:
            logger.warning(
                f"族 {group} 缺日整族 NaN: {len(missing)} 个热区日期无分区（如 {sorted(missing)[:5]}）"
            )
        if not parts:
            return None
        df = pd.concat(parts, ignore_index=True)
        # 分区早于列登记（available_from 场景）⇒ 整列 NaN；reindex 一次补齐防碎片化
        return df.reindex(columns=[*_KEY_COLUMNS, *cols])

    @staticmethod
    def _months_in_range(start_str: str, end_str: str) -> list[str]:
        """区间内自然月列表（YYYY-MM，闭区间）。"""
        year, month = int(start_str[:4]), int(start_str[4:6])
        end_month = f"{end_str[:4]}-{end_str[4:6]}"
        months = []
        while f"{year:04d}-{month:02d}" <= end_month:
            months.append(f"{year:04d}-{month:02d}")
            month += 1
            if month > 12:
                year, month = year + 1, 1
        return months

    @staticmethod
    def _merge_group_frames(frames: list[pd.DataFrame], data_cols: list[str]) -> pd.DataFrame:
        """跨族 (trade_date, ts_code) outer join，裁剪请求列，index=(trade_date, ts_code)。"""
        if not frames:
            return _empty_panel_df(data_cols)
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=list(_KEY_COLUMNS), how="outer")
        merged = merged.set_index(list(_KEY_COLUMNS)).sort_index()
        # reindex 一次性补齐缺列（NaN）+ 按请求列序裁剪（避免逐列插入碎片化）
        return merged.reindex(columns=data_cols)

    # ========== 写入 / 读取：labels ==========

    def append_labels(self, label_name: str, df: pd.DataFrame) -> None:
        """追加标签（按 trade_date 逐日分区；封存前幂等可重写，封存后拒绝改写）。"""
        self._validate_label_df(label_name, df)
        work = df.copy()
        work["trade_date"] = work["trade_date"].astype(str)
        with self._lock:
            label_dir = self.labels_dir / label_name
            label_dir.mkdir(parents=True, exist_ok=True)
            self._gc_orphans(label_dir)
            for date_str, day_df in work.groupby("trade_date"):
                self._append_label_partition(label_name, label_dir, str(date_str), day_df)
            self._maybe_save_manifest()

    def _validate_label_df(self, label_name: str, df: pd.DataFrame) -> None:
        """标签写前校验：maturity_status 必有且取值合法；键列与 label_value 齐全；单日成熟度一致。"""
        required = [*_KEY_COLUMNS, "label_value", "maturity_status"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"标签 {label_name} 写入缺列 {missing}（须含 {required}）")
        bad = set(df["maturity_status"].astype(str).unique()) - _MATURITY_VALUES
        if bad:
            raise ValueError(f"标签 {label_name} maturity_status 取值非法: {sorted(bad)}")
        per_day = df.groupby(df["trade_date"].astype(str))["maturity_status"].nunique()
        if (per_day > 1).any():
            raise ValueError(f"标签 {label_name} 单日成熟度不一致（封存水位按日推进，禁止混合）")

    def _append_label_partition(
        self, label_name: str, label_dir: Path, date_str: str, day_df: pd.DataFrame
    ) -> None:
        """单标签日分区两阶段提交；维护 labels.<name>.sealed_through（只进不退）。"""
        final_path = label_dir / f"{date_str}.parquet"
        tmp_path = label_dir / f"{date_str}.parquet.tmp"
        day_df.to_parquet(tmp_path, index=False)
        fingerprint = sha256_16_of_file(tmp_path)
        existing_fp = self.manifest.label_partition_fingerprint(label_name, date_str)
        sealed_through = self.manifest.label_sealed_through(label_name)
        is_sealed = sealed_through is not None and date_str <= sealed_through
        is_sealed_data = str(day_df["maturity_status"].iloc[0]) == "sealed"
        if existing_fp is not None and existing_fp == fingerprint:
            tmp_path.unlink(missing_ok=True)
            logger.info(f"标签分区指纹一致 no-op: {label_name}/{date_str}")
        elif existing_fp is not None and is_sealed:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"标签分区已封存（sealed_through={sealed_through}），拒绝改写: "
                f"{label_name}/{date_str}（已登记 {existing_fp} ≠ 新内容 {fingerprint}）"
            )
        else:  # 新分区，或 forming 分区内容变化 ⇒ 幂等重写
            os.replace(tmp_path, final_path)
            self.manifest.register_label_partition(label_name, date_str, fingerprint, len(day_df))
            logger.info(f"标签分区落盘: {label_name}/{date_str}（{len(day_df)} 行）")
        if is_sealed_data:
            self.manifest.set_label_sealed_through(label_name, date_str)

    def load_labels(self, query: LabelQuery) -> pd.DataFrame:
        """按区间读标签日分区，返回 index=(trade_date, ts_code)、columns=[label_value, maturity_status]。"""
        label_dir = self.labels_dir / query.label_name
        start_str, end_str = _date_str(query.start_date), _date_str(query.end_date)
        parts: list[pd.DataFrame] = []
        if label_dir.exists():
            for file in sorted(label_dir.glob("*.parquet")):
                if (
                    len(file.stem) == 8
                    and file.stem.isdigit()
                    and start_str <= file.stem <= end_str
                ):
                    parts.append(pd.read_parquet(file))
        if not parts:
            return _empty_panel_df(["label_value", "maturity_status"])
        df = pd.concat(parts, ignore_index=True)
        missing = [c for c in ("label_value", "maturity_status") if c not in df.columns]
        if missing:
            raise ValueError(
                f"标签 {query.label_name} 分区缺协议列 {missing}（label_value 由 labels_builder 落盘统一）"
            )
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.set_index(list(_KEY_COLUMNS)).sort_index()
        if query.universe:
            codes = {_ts_code_str(c) for c in query.universe}
            df = df[df.index.get_level_values("ts_code").isin(codes)]
        return df[["label_value", "maturity_status"]]

    # ========== 写入：假设台账（唯一入口） ==========

    def append_ledger_entry(self, entry: LedgerEntry) -> None:
        """台账追加（append-only；防冲突 = hypothesis_id 唯一；写锁保护）。"""
        if entry.kind not in _LEDGER_KINDS:
            raise ValueError(f"台账 kind 非法: {entry.kind!r}（合法值 {sorted(_LEDGER_KINDS)}）")
        if not entry.hypothesis_id or not str(entry.hypothesis_id).strip():
            raise ValueError("台账 hypothesis_id 不能为空")
        record = {
            "schema_version": 1,
            "hypothesis_id": entry.hypothesis_id,
            "kind": entry.kind,
            "payload": dict(entry.payload),
            "data_state": self._serialize_data_state(entry.data_state),
            "registered_at": entry.registered_at,
        }
        with self._lock:
            if entry.hypothesis_id in self._existing_ledger_ids():
                raise ValueError(f"台账 hypothesis_id 重复（防冲突）: {entry.hypothesis_id!r}")
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"台账条目登记: {entry.hypothesis_id}（kind={entry.kind}）")

    @staticmethod
    def _serialize_data_state(data_state: DataState) -> dict[str, Any]:
        """DataState 值对象 → JSON 形态（features_partition_max 转 YYYYMMDD 字符串）。"""
        return {
            "raw_partitions": dict(data_state.raw_partitions),
            "features_partition_max": _date_str(data_state.features_partition_max),
            "config_digest": data_state.config_digest,
            "code_digest": data_state.code_digest,
        }

    def _existing_ledger_ids(self) -> set[str]:
        """既有台账条目的 hypothesis_id 集（防冲突查重；坏行跳过并告警）。"""
        ids: set[str] = set()
        if not self.ledger_path.exists():
            return ids
        with open(self.ledger_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"台账第 {line_no} 行 JSON 解析失败（查重跳过）: {exc}")
                    continue
                if isinstance(obj, dict) and obj.get("hypothesis_id"):
                    ids.add(str(obj["hypothesis_id"]))
        return ids

    # ========== manifest 快照 ==========

    def get_manifest(self) -> Mapping[str, Any]:
        """manifest 快照（MappingProxyType 包裹的深拷贝；外层与嵌套改动都不影响内部）。"""
        return types.MappingProxyType(self.manifest.snapshot())

    def resync_partition_fingerprints(self, reason: str) -> dict[str, int]:
        """指纹再同步（管理操作）：重算全部已登记分区的文件指纹并校正登记。

        服务场景：修复性写入在 manifest 落盘阶段被中断（瞬时文件锁等）后，
        盘上分区内容与登记指纹不一致——以盘上内容为准重同步，并对每个被校正的
        分区登记一条 resync 审计（manifest.repairs，mode="resync"）。返回统计。
        """
        if not reason or not reason.strip():
            raise RuntimeError("指纹再同步必须登记理由（reason 非空）")
        with self._lock:
            stats = {"checked": 0, "resynced": 0, "missing_file": 0}
            for rel_path, meta in list(self.manifest.snapshot()["partitions"].items()):
                stats["checked"] += 1
                if rel_path.startswith("panel_archive/"):
                    fp = self.archive_dir / rel_path.removeprefix("panel_archive/")
                elif rel_path.startswith("panel/"):
                    fp = self.panel_dir / rel_path.removeprefix("panel/")
                else:
                    fp = self.labels_dir / rel_path.removeprefix("labels/")
                if not fp.exists():
                    stats["missing_file"] += 1
                    logger.error(f"已登记分区文件缺失: {rel_path}")
                    continue
                current = sha256_16_of_file(fp)
                if current != meta["sha256_16"]:
                    old_fp = meta["sha256_16"]
                    self.manifest.register_partition(
                        rel_path, current, rows=meta.get("rows", 0), zone=meta.get("zone", "hot")
                    )
                    self.manifest.register_repair(
                        rel_path, f"[resync] {reason}", [], old_fp, current
                    )
                    stats["resynced"] += 1
            self.manifest.save()
            logger.warning(
                f"指纹再同步完成：检查 {stats['checked']}，校正 {stats['resynced']}，"
                f"缺文件 {stats['missing_file']}（理由：{reason}）"
            )
            return stats

    # ========== 后续单元留白（签名已按协议冻结） ==========

    def append_events(self, event_type: str, df: pd.DataFrame) -> None:
        """追加事件（后续单元实现：events 库属 P1 后续单元）。"""
        raise NotImplementedError("append_events 属后续单元（events 库未建）")

    def load_events(
        self,
        event_type: str,
        start_date: TradeDate,
        end_date: TradeDate,
        universe: Sequence[TSCode] | None = None,
    ) -> pd.DataFrame:
        """加载事件表（后续单元实现：events 库属 P1 后续单元）。"""
        raise NotImplementedError("load_events 属后续单元（events 库未建）")

    def load_market_state(self, date: TradeDate) -> pd.DataFrame:
        """加载市场状态（后续单元实现：market_state 族读取通路未建）。"""
        raise NotImplementedError("load_market_state 属后续单元（state 库未建）")

    def load_raw(
        self,
        source: str,
        dataset: str,
        start_date: TradeDate,
        end_date: TradeDate,
    ) -> pd.DataFrame:
        """加载原始数据（单元 2 实现：normalized 构建通路）。"""
        raise NotImplementedError("load_raw 属单元 2（normalized 构建通路）")
