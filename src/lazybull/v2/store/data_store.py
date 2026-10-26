# -*- coding: utf-8 -*-
"""v2 数据面参考实现 ``PanelDataStore``（DataStore Protocol，protocols.md §1.1）。
布局：panel/labels 分区、manifest/WAL、台账与单机写锁。
写入口径：分区两阶段提交；labels 封存后拒改；台账唯一；写锁内 sync 与孤儿 tmp GC。
"""

from __future__ import annotations

import json
import os
import re
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
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
from src.lazybull.v2.store.column_groups import LABEL_TABLES
from src.lazybull.v2.store.manifest import VALID_GROUPS, Manifest, sha256_16_of_file
from src.lazybull.v2.store.manifest_wal import ManifestDefer, ManifestWal
from src.lazybull.v2.store.row_domain import row_domain_for_date, row_domain_for_partition
from src.lazybull.v2.store.store_identity import (
    InputIdentityDriftError,
    file_identity,
    verify_file_identities,
)
from src.lazybull.v2.store.write_lock import _WriteLock

__all__ = ["PanelDataStore", "StoreReadReceipt"]

_KEY_COLUMNS = ("trade_date", "ts_code")
_LEDGER_KINDS = frozenset({"prereg", "conclusion"})
_MATURITY_VALUES = frozenset({"forming", "sealed"})


@dataclass(frozen=True)
class StoreReadReceipt:
    """单次 store 读取的只收回执（P3-T3，§13.3.5 方案 P2：读前采集 + 读后核验）。

    ``files`` = 本次查询**实际读取**的分区文件身份清单（``store_identity.file_identity``
    记录，显示路径以 store root 为基准）——清单单一来源 = 读取动作本身，不另造
    冷热区解析规则（R7）。读取完成后已对全部文件重算核验（``verified=True`` 才产出）；
    读取期间任一文件漂移 ⇒ ``InputIdentityDriftError``，不产出 receipt。
    """

    kind: str  # "features" / "labels"
    start_date: str
    end_date: str
    files: tuple[dict[str, Any], ...]
    collected_at: str
    verified: bool = True


def _date_str(date: TradeDate | str) -> str:
    """TradeDate / YYYYMMDD 字符串统一为字符串口径。"""
    return str(date)


def _ts_code_str(code: TSCode | str) -> str:
    return str(getattr(code, "value", code))


def _empty_panel_df(columns: Sequence[str]) -> pd.DataFrame:
    """空面板（index=(trade_date, ts_code)），load 无命中时返回。"""
    idx = pd.MultiIndex.from_arrays([[], []], names=list(_KEY_COLUMNS))
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}, index=idx)


class PanelDataStore:
    """DataStore Protocol 的 panel 实现（构造参数 data_root 默认 ``data/``）。

    读取回执(P3-T3,§13.3.5 P2):``load_features`` / ``load_labels`` 读前逐文件
    采集、读后统一核验(漂移 ⇒ InputIdentityDriftError);会话级累积经
    ``read_receipts`` / ``consumed_files`` 只读属性暴露,``reset_read_receipts``
    由编排器在 run 开始调用。**单线程顺序调用约定**:读取不支持并发/嵌套
    (``_begin_read`` 重入 ⇒ RuntimeError),编排/回放均为单线程顺序读取。
    **会话累积无上限**:非编排器消费者(脚本/长生命周期进程)直接复用 store 时,
    须自行在任务边界调用 ``reset_read_receipts()``(复核处置 reviewer1-P3-6)。
    """

    def __init__(self, data_root: Path | str = "data") -> None:
        self._root = Path(data_root)
        self._manifest: Manifest | None = None
        self._lock = _WriteLock(self.features_dir / ".write.lock")
        self._wal = ManifestWal(self)
        self._wal_stat: tuple[int, int] | None = None  # 已吸收 WAL 的 (mtime_ns, size) 代际
        self._manifest_defer = 0  # defer_manifest_save 计数（批量回填防 manifest O(n²) 落盘）
        self._last_read_receipt: StoreReadReceipt | None = None  # 最近一次读取回执（P3-T3）
        self._read_records: list[dict[str, Any]] | None = None  # 读取中缓冲（None = 非读取态）
        # P3-T3 评审处置（reviewer1-P1-1 ≡ codex-R01）：会话级累积——单槽
        # last_read_receipt 覆盖语义下中间读取（训练窗/labels）会丢失；累积器在
        # run 开始时由编排器 reset_read_receipts() 清空、收尾统一消费。
        self._receipt_log: list[StoreReadReceipt] = []
        self._consumed_files_log: list[dict[str, Any]] = []

    # ---------- 读取回执（P3-T3，§13.3.5 方案 P2；单线程顺序调用约定） ----------

    @property
    def last_read_receipt(self) -> StoreReadReceipt | None:
        """最近一次 load_features / load_labels 的只收回执（每次读取覆盖；None = 未读过）。"""
        return self._last_read_receipt

    @property
    def read_receipts(self) -> tuple[StoreReadReceipt, ...]:
        """会话级回执累积（自上次 ``reset_read_receipts`` 起全部读取）。"""
        return tuple(self._receipt_log)

    @property
    def consumed_files(self) -> tuple[dict[str, Any], ...]:
        """会话级消费文件身份累积（receipt 文件 + ``register_external_read`` 登记）。

        同路径不去重（消费方 ``collect_consumed_file_identities`` 合并去重）；
        返回浅拷贝元组，外部修改不影响内部日志。
        """
        return tuple(self._consumed_files_log)

    def reset_read_receipts(self) -> None:
        """清空会话级累积（编排器 run 开始时调用；单线程顺序调用约定）。"""
        self._receipt_log.clear()
        self._consumed_files_log.clear()

    def register_external_read(self, path: Path | str) -> None:
        """会话级外部读取登记（便捷形态 = begin+finish;供 label_refresh 收敛路径等
        不经 load_features/load_labels 的 store 数据文件读取使用）。

        **口径勘误(复核二 codex):本便捷形态 = begin+finish 紧邻连用,不含读取
        动作,**不防「读取期间替换」**;防洗白须在读取前调 ``begin_external_read``、
        读取后调 ``finish_external_read``(绑定形态,label_refresh 收敛路径即此用法)。
        本便捷形态仅用于「登记当前内容 + 即时一致性核验」。
        """
        self.finish_external_read(self.begin_external_read(path))

    def begin_external_read(self, path: Path | str) -> dict[str, Any]:
        """外部读取**读前**身份采集（P3-T3 复核处置 codex-F01：读前采集与读取
        绑定，读后必须经 ``finish_external_read`` 核验同一身份——仅读后采集会被
        「读取期间替换」洗白为新基线）。"""
        return file_identity(path, base_dir=self._root)

    def finish_external_read(self, record: dict[str, Any]) -> None:
        """读后核验（同文件重算比对）+ 会话清单登记（去重）。"""
        mismatches = verify_file_identities([record], base_dir=self._root)
        if mismatches:
            raise InputIdentityDriftError(f"外部读取期间文件漂移（读后核验失败）: {mismatches[0]}")
        existing = {(r["path"], r["sha256"], r["size"]) for r in self._consumed_files_log}
        key = (record["path"], record["sha256"], record["size"])
        if key not in existing:
            self._consumed_files_log.append(record)

    # ---------- 读取回执（P3-T3，§13.3.5 方案 P2；单线程顺序调用约定） ----------

    def _begin_read(self) -> None:
        """进入读取态（嵌套读取拒绝——训练/回放为单线程顺序调用约定）。"""
        if self._read_records is not None:
            raise RuntimeError("store 读取嵌套调用（回执缓冲非空），违反单线程顺序读取约定")
        self._read_records = []

    def _track_read(self, path: Path) -> None:
        """读取动作点的读前身份采集（清单单一来源 = 实际读取动作）。"""
        if self._read_records is not None:
            self._read_records.append(file_identity(path, base_dir=self._root))

    def _abort_read(self) -> None:
        """读取异常：丢弃缓冲，不产出 receipt。"""
        self._read_records = None

    def _finalize_read(self, kind: str, start_str: str, end_str: str) -> None:
        """读后核验并产出 receipt：全部文件重算，任一漂移 ⇒ InputIdentityDriftError。

        P3-T3 复核处置（codex-F01）增补：panel/labels 分区额外比对 manifest
        登记指纹（sha256 前 16 位）——「manifest 登记值不变而物理文件变更」
        （读取窗口跨越登记后篡改）⇒ 拒绝（§5-1 明文用例）；无登记 ⇒ 跳过
        （孤儿分区由 readiness 校验拦截，非本层职责）。
        """
        records = self._read_records or []
        self._read_records = None
        mismatches = verify_file_identities(records, base_dir=self._root)
        if mismatches:
            raise InputIdentityDriftError(
                f"store {kind} 读取期间文件漂移（读后核验失败）: {mismatches[:5]}"
                f"{'...' if len(mismatches) > 5 else ''}（共 {len(mismatches)} 个）"
            )
        self._verify_manifest_registration(records)
        self._last_read_receipt = StoreReadReceipt(
            kind=kind,
            start_date=start_str,
            end_date=end_str,
            files=tuple(records),
            collected_at=datetime.now().isoformat(timespec="seconds"),
            verified=True,
        )
        # 会话级累积（P3-T3 评审处置 P1-1/R01）
        self._receipt_log.append(self._last_read_receipt)
        self._consumed_files_log.extend(records)

    def _verify_manifest_registration(self, records: list[dict[str, Any]]) -> None:
        """分区文件与 manifest 登记指纹比对（codex-F01；登记缺失跳过）。"""
        snapshot = self.manifest.snapshot()
        partitions = snapshot.get("partitions", {})
        drifted: list[str] = []
        for record in records:
            display = record["path"]
            registered: str | None = None
            if display.startswith("features/"):
                meta = partitions.get(display[len("features/") :])
                registered = (meta or {}).get("sha256_16")
            elif display.startswith("labels/"):
                parts = display.split("/")
                if len(parts) == 3:
                    registered = self.manifest.label_partition_fingerprint(
                        parts[1], parts[2].removesuffix(".parquet")
                    )
            if registered is not None and record["sha256"][:16] != registered:
                drifted.append(display)
        if drifted:
            raise InputIdentityDriftError(
                f"分区物理内容与 manifest 登记指纹不一致（登记不变而文件变更）: "
                f"{drifted[:5]}{'...' if len(drifted) > 5 else ''}（共 {len(drifted)} 个）"
            )

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
    def wal_path(self) -> Path:
        """manifest 登记事件 WAL（``features/manifest_wal.jsonl``，jsonl 逐行追加）。"""
        return self.features_dir / "manifest_wal.jsonl"

    def _can_initialize_manifest(self, manifest_path: Path) -> bool:
        """仅无分区、bak/tmp 或待处理 WAL 的全新数据根允许空骨架初始化。"""
        wal_pending = self.wal_path.exists() and self.wal_path.stat().st_size > 0
        assets = (
            self.panel_dir,
            self.archive_dir,
            self.labels_dir,
            manifest_path.with_suffix(".json.bak"),
            manifest_path.with_suffix(".json.tmp"),
            manifest_path.with_suffix(".json.bak.tmp"),
        )
        return not wal_pending and not any(path.exists() for path in assets)

    @property
    def manifest(self) -> Manifest:
        """manifest 内存视图（写锁内懒加载与 WAL 重放；失败实例不得复用缓存）。"""
        if self._manifest is None:
            with self._lock:
                if self._manifest is None:
                    self._wal.load_manifest()
        return self._manifest

    def defer_manifest_save(self):
        """批量写入期延迟 manifest 落盘（可重入，详见 ManifestDefer）。"""
        return ManifestDefer(self)

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
            self._wal.sync_manifest_under_lock()
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
            self._wal.register_partition(rel_path, fingerprint, len(df), "hot")
            self._wal.maybe_save()
            logger.info(f"特征分区落盘: {rel_path}（{len(df)} 行，指纹 {fingerprint}）")

    def _validate_feature_write(self, date_str: str, group: str, df: pd.DataFrame) -> None:
        """写前校验：族名合法、键列齐全、单日一致、列已登记且族一致、分区未封存。"""
        row_domain_for_date(date_str)
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
        if df.duplicated(subset=list(_KEY_COLUMNS)).any():
            raise RuntimeError(f"特征分区键 (trade_date, ts_code) 重复（{date_str}/{group}）")
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
            self._wal.sync_manifest_under_lock()
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
            self._wal.register_partition(rel_path, new_fp, len(merged), "archive")
            self._wal.register_repair(rel_path, reason, new_days, old_fp, new_fp)
            self._wal.maybe_save()
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
            self._wal.sync_manifest_under_lock()
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
                raise RuntimeError(
                    f"修复替换日不在存量分区: {missing}（{rel_path}；缺失日请走 repair 追加）"
                )
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
            self._wal.register_partition(rel_path, new_fp, len(merged), "archive")
            self._wal.register_repair(rel_path, f"[replace_days] {reason}", days, old_fp, new_fp)
            self._wal.maybe_save()
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
            self._wal.sync_manifest_under_lock()
            if rel_path.startswith("panel_archive/"):
                final_path = self.archive_dir / rel_path.removeprefix("panel_archive/")
            elif rel_path.startswith("panel/"):
                final_path = self.panel_dir / rel_path.removeprefix("panel/")
            else:
                raise RuntimeError(f"非法分区路径（须 panel/ 或 panel_archive/ 前缀）: {rel_path}")
            old_fp = self.manifest.partition_fingerprint(rel_path)
            if old_fp is None or not final_path.exists():
                raise RuntimeError(f"修复目标分区不存在或未登记: {rel_path}（新建请走 append）")
            self._validate_rewrite_payload(rel_path, df)
            part_dir = final_path.parent
            tmp_path = part_dir / f"{final_path.name}.tmp"
            df.to_parquet(tmp_path, index=False)
            new_fp = sha256_16_of_file(tmp_path)
            os.replace(tmp_path, final_path)
            zone = "archive" if rel_path.startswith("panel_archive/") else "hot"
            self._wal.register_partition(rel_path, new_fp, len(df), zone)
            self._wal.register_repair(rel_path, f"[rewrite_partition] {reason}", [], old_fp, new_fp)
            self._wal.maybe_save()
            logger.warning(f"分区修复性重写: {rel_path}（{len(df)} 行；理由：{reason}）")

    def _validate_rewrite_payload(self, rel_path: str, df: pd.DataFrame) -> None:
        """rewrite 结构校验：热区复用 _validate_feature_write、冷区复用 _validate_archive_write。

        rewrite 允许内容变化（不校验指纹不变），但结构口径（键列/单日或月内/
        键唯一/列登记）与 append 完全一致。
        """
        segs = rel_path.split("/")
        if len(segs) != 3:
            raise RuntimeError(f"非法分区路径（须 <zone>/<key>/<group>.parquet 三段）: {rel_path}")
        group = segs[-1].removesuffix(".parquet")
        if rel_path.startswith("panel_archive/"):
            self._validate_archive_write(segs[1], group, df)
        else:
            self._validate_feature_write(segs[1], group, df)

    def append_archive_features(self, month: str, group: str, df: pd.DataFrame) -> None:
        """落冷区月分区（``panel_archive/YYYY-MM/<group>.parquet``；两阶段提交 + 写锁）。

        历史分区禁止原地修改：指纹一致 no-op、不一致 raise RuntimeError
        （口径修正 = 新列名 / 版本升级，不是覆盖旧分区）。
        """
        self._validate_archive_write(month, group, df)
        rel_path = f"panel_archive/{month}/{group}.parquet"
        with self._lock:
            self._wal.sync_manifest_under_lock()
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
            self._wal.register_partition(rel_path, fingerprint, len(df), "archive")
            self._wal.maybe_save()
            logger.info(f"冷区分区落盘: {rel_path}（{len(df)} 行，指纹 {fingerprint}）")

    def _validate_archive_write(self, month: str, group: str, df: pd.DataFrame) -> None:
        """冷区写前校验：月份格式、族名、键列、键唯一、全部日期落在月内、列已登记且族一致。"""
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
            raise RuntimeError(f"冷区月份须为 YYYY-MM 格式，实得 {month!r}")
        if group not in VALID_GROUPS:
            raise RuntimeError(f"非法列族 {group!r}（合法值 {sorted(VALID_GROUPS)}）")
        row_domain_for_partition(f"panel_archive/{month}/{group}.parquet")
        missing_keys = [c for c in _KEY_COLUMNS if c not in df.columns]
        if missing_keys:
            raise RuntimeError(f"冷区分区缺键列 {missing_keys}（{month}/{group}）")
        if df.duplicated(subset=list(_KEY_COLUMNS)).any():
            raise RuntimeError(f"冷区分区键 (trade_date, ts_code) 重复（{month}/{group}）")
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
        """加载特征面板（manifest 校验列名合法性与 available_from 越界；core 左表锚定）。

        core 锚定在本方法内实现（冻结 §4 契约）：拼接键 (trade_date, ts_code)，
        始终加载 core 族键域作左表，其余族 left join 进 core 键域——core 缺的日/键
        不进结果，非 core 族多出的键被裁掉（debug 计数登记），族缺日整族 NaN；
        仅请求键列 ⇒ 返回 core 键域空列帧；core 分区整体缺失（冷/热均无）⇒ 空结果。
        """
        col_meta = self._check_feature_query(query)
        start_str, end_str = _date_str(query.start_date), _date_str(query.end_date)
        groups: dict[str, list[str]] = {}
        for col, meta in col_meta.items():
            groups.setdefault(str(meta["group"]), []).append(col)
        hot_dates = self._hot_dates_in_range(start_str, end_str)
        data_cols = [c for c in query.columns if c not in _KEY_COLUMNS]
        self._begin_read()
        try:
            core_frame = self._load_group_frame(
                "core", groups.pop("core", []), start_str, end_str, hot_dates
            )
            # 逐族读取并立即合并（中间帧随循环释放;core 缺失 ⇒ 不读他族直接空,
            # 与 T1 前语义逐位一致;P3-T3 评审处置 reviewer1-P3-11 内存回归恢复)
            merged: pd.DataFrame | None = None
            if core_frame is not None:
                merged = core_frame
                core_keys = pd.MultiIndex.from_frame(core_frame[list(_KEY_COLUMNS)])
                for group, cols in groups.items():
                    frame = self._load_group_frame(group, cols, start_str, end_str, hot_dates)
                    if frame is None:
                        continue
                    dropped = int(
                        (~pd.MultiIndex.from_frame(frame[list(_KEY_COLUMNS)]).isin(core_keys)).sum()
                    )
                    if dropped:
                        logger.debug(
                            f"族 {group} 多出 core 键域的 {dropped} 键被裁掉（core 左表锚定）"
                        )
                    merged = merged.merge(frame, on=list(_KEY_COLUMNS), how="left")
                merged = merged.set_index(list(_KEY_COLUMNS)).sort_index()
                if not merged.index.is_unique:
                    raise RuntimeError("load_features 合并后键 (trade_date, ts_code) 不唯一")
        except Exception:
            self._abort_read()
            raise
        self._finalize_read("features", start_str, end_str)
        if core_frame is None:
            df = _empty_panel_df(data_cols)
        else:
            # reindex 一次性补齐缺列（NaN）+ 按请求列序裁剪（避免逐列插入碎片化）
            df = merged.reindex(columns=data_cols)
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
            manifest_fingerprint=self.manifest.content_fingerprint(),
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
            month_path = self.archive_dir / month / f"{group}.parquet"
            self._track_read(month_path)
            month_df = pd.read_parquet(month_path)
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
                present_hot.add(date_str)
                self._track_read(hot_file)
                hot_df = pd.read_parquet(hot_file)
                # 热区整文件读入后按分区名日期过滤（与冷区同口径；防御性，正常应为零过滤）
                mask = hot_df["trade_date"].astype(str) == date_str
                if not mask.all():
                    logger.warning(
                        f"热区日分区含分区名外日期（读入已过滤）: {hot_file} "
                        f"实得 {sorted(hot_df.loc[~mask, 'trade_date'].astype(str).unique())[:5]}"
                    )
                    hot_df = hot_df[mask]
                parts.append(hot_df)
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

    # ========== 写入 / 读取：labels ==========

    def append_labels(self, label_name: str, df: pd.DataFrame) -> None:
        """追加标签（按 trade_date 逐日分区；封存前幂等可重写，封存后拒绝改写）。"""
        self._validate_label_df(label_name, df)
        work = df.copy()
        work["trade_date"] = work["trade_date"].astype(str)
        with self._lock:
            self._wal.sync_manifest_under_lock()
            label_dir = self.labels_dir / label_name
            label_dir.mkdir(parents=True, exist_ok=True)
            self._gc_orphans(label_dir)
            for date_str, day_df in work.groupby("trade_date"):
                self._append_label_partition(label_name, label_dir, str(date_str), day_df)
            self._wal.maybe_save()

    def _validate_label_df(self, label_name: str, df: pd.DataFrame) -> None:
        """标签写前校验键列、真实日期与单日成熟度，全部通过后才允许任何物理写入。"""
        required = [*_KEY_COLUMNS, "label_value", "maturity_status"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"标签 {label_name} 写入缺列 {missing}（须含 {required}）")
        dates = df["trade_date"].astype(str)
        if not dates.str.fullmatch(r"\d{8}").all():
            raise ValueError(f"标签 {label_name} trade_date 须为恰 8 位数字日（YYYYMMDD）")
        for date_str in sorted(dates.unique()):
            try:
                row_domain_for_date(str(date_str))
            except ValueError as exc:
                raise ValueError(f"标签 {label_name} trade_date 非法日期 {date_str!r}") from exc
        if df.duplicated(subset=list(_KEY_COLUMNS)).any():
            raise ValueError(f"标签 {label_name} 键 (trade_date, ts_code) 重复")
        bad = set(df["maturity_status"].astype(str).unique()) - _MATURITY_VALUES
        if bad:
            raise ValueError(f"标签 {label_name} maturity_status 取值非法: {sorted(bad)}")
        per_day = df.groupby(dates)["maturity_status"].nunique()
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
        elif is_sealed and (existing_fp is not None or not is_sealed_data):
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"标签分区已封存（sealed_through={sealed_through}），拒绝水位内新增 forming 或改写: "
                f"{label_name}/{date_str}（已登记 {existing_fp}，新内容 {fingerprint}）"
            )
        else:  # 历史新 sealed 分区可回填；forming 仅在水位外新增或未封存时重写
            os.replace(tmp_path, final_path)
            self._wal.register_label_partition(label_name, date_str, fingerprint, len(day_df))
            logger.info(f"标签分区落盘: {label_name}/{date_str}（{len(day_df)} 行）")
        if is_sealed_data:
            self._wal.set_label_sealed_through(label_name, date_str)

    def load_labels(self, query: LabelQuery) -> pd.DataFrame:
        """按区间读标签日分区，返回 index=(trade_date, ts_code)、columns=[label_value, maturity_status]。

        variant=raw ⇒ label_value 取原值列；variant=neu ⇒ 取 neu_label_value
        （输出 shape 不变）。未知标签名（∉ LABEL_TABLES）⇒ ValueError，
        与「合法但无分区 ⇒ 空」区分。
        """
        if query.label_name not in LABEL_TABLES:
            raise ValueError(f"未知标签名: {query.label_name!r}（合法值 {sorted(LABEL_TABLES)}）")
        label_dir = self.labels_dir / query.label_name
        start_str, end_str = _date_str(query.start_date), _date_str(query.end_date)
        parts: list[pd.DataFrame] = []
        self._begin_read()
        try:
            if label_dir.exists():
                for file in sorted(label_dir.glob("*.parquet")):
                    if (
                        len(file.stem) == 8
                        and file.stem.isdigit()
                        and start_str <= file.stem <= end_str
                    ):
                        self._track_read(file)
                        parts.append(pd.read_parquet(file))
        except Exception:
            self._abort_read()
            raise
        self._finalize_read("labels", start_str, end_str)
        if not parts:
            return _empty_panel_df(["label_value", "maturity_status"])
        df = pd.concat(parts, ignore_index=True)
        required = ["label_value", "maturity_status"]
        if query.variant == "neu":
            required.append("neu_label_value")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"标签 {query.label_name} 分区缺协议列 {missing}（label_value 由 labels_builder 落盘统一）"
            )
        if query.variant == "neu":
            df["label_value"] = df["neu_label_value"]
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
            self._wal.sync_manifest_under_lock()
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
        """指纹再同步（管理操作）：重算全部已登记分区（panel 热/冷区 + labels）的文件指纹并校正登记。

        服务场景：修复性写入在 manifest 落盘阶段被中断（瞬时文件锁等）后，
        盘上分区内容与登记指纹不一致——以盘上内容为准重同步，并对每个被校正的
        分区登记一条 resync 审计（manifest.repairs，mode="resync"）。返回统计。
        """
        if not reason or not reason.strip():
            raise RuntimeError("指纹再同步必须登记理由（reason 非空）")
        with self._lock:
            self._wal.sync_manifest_under_lock()
            stats = {"checked": 0, "resynced": 0, "missing_file": 0}
            snapshot = self.manifest.snapshot()
            for rel_path, meta in list(snapshot["partitions"].items()):
                if rel_path.startswith("panel_archive/"):
                    fp = self.archive_dir / rel_path.removeprefix("panel_archive/")
                else:
                    fp = self.panel_dir / rel_path.removeprefix("panel/")
                self._resync_one(rel_path, fp, meta, reason, stats)
            for label_name, entry in snapshot["labels"].items():
                for filename, meta in list(entry["partitions"].items()):
                    rel_path = f"labels/{label_name}/{filename}"
                    self._resync_one(
                        rel_path, self.labels_dir / label_name / filename, meta, reason, stats
                    )
            self._wal.save()
            logger.warning(
                f"指纹再同步完成：检查 {stats['checked']}，校正 {stats['resynced']}，"
                f"缺文件 {stats['missing_file']}（理由：{reason}）"
            )
            return stats

    def _resync_one(
        self, rel_path: str, fp: Path, meta: dict[str, Any], reason: str, stats: dict[str, int]
    ) -> None:
        """单分区指纹核对/校正（labels 分区经 register_label_partition 落登记）。"""
        stats["checked"] += 1
        if not fp.exists():
            stats["missing_file"] += 1
            logger.error(f"已登记分区文件缺失: {rel_path}")
            return
        current = sha256_16_of_file(fp)
        if current == meta["sha256_16"]:
            return
        old_fp = str(meta["sha256_16"])
        rows = int(meta.get("rows", 0))
        if rel_path.startswith("labels/"):
            label_name, filename = rel_path.split("/")[1:]
            self._wal.register_label_partition(
                label_name, filename.removesuffix(".parquet"), current, rows
            )
        else:
            self._wal.register_partition(rel_path, current, rows, str(meta.get("zone", "hot")))
        self._wal.register_repair(rel_path, f"[resync] {reason}", [], old_fp, current)
        stats["resynced"] += 1

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
