"""R1 下载引擎：逐请求 raw 先行、断点续跑、逐日质量裁定与真实对账。

对应复核意见（2026-11-18 第二次）问题 B/C：
- **B 残缺数据不得就绪**：每个 (股, 日) 都要过槽位/OHLC/量额/时间/重复键检查；
  "有一行"不等于完整；预热日同样纳入质量与缺失传播；对账越线单独分桶，
  按预先登记的占比门判定，不把"做过对账"当成"对账可接受"。
- **C 限频可恢复**：每个成功请求的原始行**立即**落 `raw_spool/` 并写 `raw_index/`，
  标准化失败或中断都不丢；续跑优先从 spool 重放，不再请求已保存部分。
  阻塞时**不发布** artifacts（工作态保留），同一 run 可续跑且不覆盖冻结证据。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.common import REPO_ROOT, sha256_file
from scripts.v2_p7.minute_repair_ml import artifacts, quality, source, spec, store

ECONOMIC_COLUMNS: tuple[str, ...] = ("O", "C", "H", "L", "Q", "A")


@dataclass
class Coverage:
    """一个请求覆盖（含预热），`chunk_id` 为其所属块序号。"""

    chunk_id: int
    code_index: int
    start_pos: int
    end_pos: int


@dataclass
class DownloadReport:
    """下载汇总（供 readiness 与报告）。"""

    requests: int = 0
    ok_requests: int = 0
    failed_requests: int = 0
    replayed_requests: int = 0
    blocked: dict[str, Any] | None = None
    years: list[str] = field(default_factory=list)
    standardized_rows: int = 0
    auction_rows: int = 0
    conflict_rows: int = 0
    conflicts: int = 0
    exclusion_rows: int = 0
    recon_rows: int = 0
    recon_counts: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    resumed_batches: int = 0
    raw_blocks: int = 0
    raw_spool_bytes: int = 0
    duration_hours: float = 0.0

    def as_payload(self) -> dict[str, Any]:
        """可落盘摘要。"""
        return dict(self.__dict__)


def build_coverage(eligible: np.ndarray, chunk_days: int, preheat_days: int) -> list[Coverage]:
    """逐股逐块的请求覆盖（R0 计划与 R1 执行共用，禁止两处口径分叉）。"""
    covered: list[Coverage] = []
    for j in range(eligible.shape[1]):
        positions = np.flatnonzero(eligible[:, j])
        if positions.size == 0:
            continue
        first_in_window: dict[int, int] = {}
        for pos in positions.tolist():
            first_in_window.setdefault(int(pos // chunk_days), pos)
        for window_id, first_pos in sorted(first_in_window.items()):
            covered.append(
                Coverage(
                    chunk_id=window_id,
                    code_index=j,
                    start_pos=max(0, first_pos - preheat_days),
                    end_pos=min(eligible.shape[0], (window_id + 1) * chunk_days),
                )
            )
    return covered


def _window(days: Sequence[str], start_pos: int, end_pos: int) -> tuple[str, str]:
    first, last = days[start_pos], days[end_pos - 1]
    return (
        f"{first[:4]}-{first[4:6]}-{first[6:]} 09:00:00",
        f"{last[:4]}-{last[4:6]}-{last[6:]} 15:31:00",
    )


@dataclass
class DailyCache:
    """块内日线缓存：行 + **已成功读取的日期**。

    区分"该日分区存在但没有这只股票（未上市/已摘牌）"与"分区缺失（本地数据缺口）"：
    前者是可披露排除，后者必须算未解决，不能因为读不到日线就放过。
    """

    rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    loaded_days: set[str] = field(default_factory=set)


def _daily_cache(days: Sequence[str], codes: Iterable[str]) -> DailyCache:
    """块内日线缓存（量额高低 + 停牌标记），一次读盘供整块复用。"""
    wanted = set(str(code) for code in codes)
    cache = DailyCache()
    columns = ["ts_code", "close", "high", "low", "vol", "amount", "is_suspended", "tradable"]
    for day in days:
        path = REPO_ROOT / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["ts_code"].astype(str).isin(wanted)]
        for row in frame.to_dict("records"):
            cache.rows[(str(row["ts_code"]), str(day))] = row
        cache.loaded_days.add(str(day))
    return cache


class DownloadEngine:
    """分块分批下载 + 逐请求落盘 + 断点 + 逐日裁定。"""

    def __init__(
        self,
        *,
        campaign: Path,
        work_dir: Path,
        src: "source.MinuteSource",
        days: Sequence[str],
        codes: Sequence[str],
        eligible: np.ndarray,
        log: Callable[[str], None],
        batch_codes: int = 100,
        resume: bool = True,
    ) -> None:
        self.campaign = Path(campaign)
        self.work_dir = Path(work_dir)
        self.src = src
        self.days = list(days)
        self.codes = list(codes)
        self.code_index = {code: index for index, code in enumerate(self.codes)}
        self.eligible = eligible
        self.log = log
        self.batch_codes = int(batch_codes)
        self.resume = bool(resume)
        self.chunk_days = int(spec.MINUTE["chunk_trading_days"])
        self.preheat = int(spec.WINDOW["minute_preheat_trading_days"])
        self.days_index = {day: index for index, day in enumerate(self.days)}
        self.report = DownloadReport()
        self.counts: Counter = Counter()
        self.warmup: Counter = Counter()
        self.unresolved: list[dict[str, Any]] = []
        self.conflict_pairs: set[tuple[int, str]] = set()
        self.request_status: dict[tuple[int, int, int], str] = {}
        self.produced: dict[str, bytearray] = {}
        self.suspended: set[tuple[int, str]] = set()
        self.years: set[str] = set()
        self.chunk_root = self.work_dir / "chunks"
        self.ckpt_root = self.work_dir / "checkpoints"
        self.spool_root = self.work_dir / "raw_spool"
        self.index_root = self.work_dir / "raw_index"
        self.dataset_dir = self.work_dir / "dataset"
        for directory in (
            self.chunk_root,
            self.ckpt_root,
            self.spool_root,
            self.index_root,
            self.dataset_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 公共入口
    def run(self) -> DownloadReport:
        """执行全部覆盖；被限频打断时保留工作态并如实标记。"""
        started = time.perf_counter()
        covered = build_coverage(self.eligible, self.chunk_days, self.preheat)
        self.report.requests = len(covered)
        by_chunk: dict[int, list[Coverage]] = {}
        for item in covered:
            by_chunk.setdefault(item.chunk_id, []).append(item)
        for chunk_id in sorted(by_chunk):
            if self.report.blocked:
                break
            self._run_chunk(chunk_id, by_chunk[chunk_id])
        self._finalize()
        self.report.duration_hours = round((time.perf_counter() - started) / 3600.0, 5)
        return self.report

    # -------------------------------------------------------------- 单块与单批
    def _run_chunk(self, chunk_id: int, items: Sequence[Coverage]) -> None:
        days = self.days[
            chunk_id * self.chunk_days : min(len(self.days), (chunk_id + 1) * self.chunk_days)
        ]
        codes = [self.codes[item.code_index] for item in items]
        cache = _daily_cache(days, codes)
        ordered = sorted(items, key=lambda item: item.code_index)
        batches = [
            ordered[index : index + self.batch_codes]
            for index in range(0, len(ordered), self.batch_codes)
        ]
        progress_path = self.ckpt_root / f"chunk-{chunk_id:03d}.json"
        progress = self._load_progress(progress_path)
        for batch_index, batch in enumerate(batches):
            if self.report.blocked:
                return
            key = f"batch-{batch_index:04d}"
            done = progress.get("done", {}).get(key)
            if self.resume and done:
                self._restore_batch(done)
                self.report.resumed_batches += 1
                continue
            record = self._run_batch(chunk_id, key, batch, cache)
            if record is None:
                return
            progress.setdefault("done", {})[key] = record
            self._write_progress(progress_path, progress)
        self.log(f"chunk {chunk_id}: 完成 {len(batches)} 批")

    def _run_batch(
        self,
        chunk_id: int,
        key: str,
        batch: Sequence[Coverage],
        cache: DailyCache,
    ) -> dict[str, Any] | None:
        std_parts: list[pd.DataFrame] = []
        auction_parts: list[pd.DataFrame] = []
        recon_parts: list[pd.DataFrame] = []
        conflict_parts: list[pd.DataFrame] = []
        raw_frames: list[pd.DataFrame] = []
        records: list[dict[str, Any]] = []
        produced_days: dict[str, list[int]] = {}
        suspended_batch: list[list[Any]] = []
        batch_counts: Counter = Counter()
        batch_warmup: Counter = Counter()
        batch_unresolved: list[dict[str, Any]] = []
        batch_exclusions: list[dict[str, Any]] = []
        batch_recon: Counter = Counter()
        batch_conflicts = 0
        for item in batch:
            code = self.codes[item.code_index]
            start, end = _window(self.days, item.start_pos, item.end_pos)
            request_id = f"{code}_{self.days[item.start_pos]}_{self.days[item.end_pos - 1]}"
            frame, status, conflict_rows = self._obtain_frame(item, request_id, code, start, end)
            if status == "blocked":
                return None
            self.request_status[(item.code_index, item.start_pos, item.end_pos)] = status
            if status == source.OK:
                self.report.ok_requests += 1
            else:
                self.report.failed_requests += 1
            if len(conflict_rows):
                frame_conflict = conflict_rows.copy()
                frame_conflict["request_id"] = request_id
                conflict_parts.append(frame_conflict)
                batch_conflicts += int(len(conflict_rows))
                for row in conflict_rows.to_dict("records"):
                    moment = str(row.get("trade_time"))
                    day = moment[:10].replace("-", "")
                    self.conflict_pairs.add((item.code_index, day))
            by_day: dict[str, pd.DataFrame] = {}
            if frame is not None and len(frame):
                raw_frames.append(frame)
                for day, part in frame.groupby(frame["trade_time"].astype(str).str[:10]):
                    by_day[str(day).replace("-", "")] = part
            for day in self.days[item.start_pos : item.end_pos]:
                eligible = bool(self.eligible[self.days_index[day], item.code_index])
                phase = "eligible" if eligible else "warmup"
                suspended = self._is_suspended(cache, code, day)
                if suspended:
                    suspended_batch.append([item.code_index, day])
                if status != source.OK:
                    bucket, reason = quality.classify_observation(
                        {"status": status, "suspended": suspended}
                    )
                    self._account(
                        bucket,
                        reason,
                        item,
                        day,
                        phase,
                        batch_counts,
                        batch_warmup,
                        batch_unresolved,
                        batch_exclusions,
                    )
                    continue
                part = by_day.get(day)
                if part is None:
                    bucket, reason = quality.classify_observation(
                        {
                            "status": "empty",
                            "suspended": suspended,
                            # 仅在"该日日线已成功读取但无该股"时才判未上市/已摘牌；
                            # 分区缺失属于本地数据缺口，必须算未解决
                            "not_listed": (code, day) not in cache.rows
                            and day in cache.loaded_days,
                        }
                    )
                    self._account(
                        bucket,
                        reason,
                        item,
                        day,
                        phase,
                        batch_counts,
                        batch_warmup,
                        batch_unresolved,
                        batch_exclusions,
                    )
                    continue
                standardized, auction, problems = store.standardize_day(
                    part, day, convention=str(spec.SESSION_MAPPING["vendor_convention"])
                )
                if problems:
                    records.extend(
                        [
                            {"request_id": request_id, "ts_code": code, "kind": "row_problem", **p}
                            for p in problems
                        ]
                    )
                if len(standardized):
                    std_parts.append(standardized)
                    produced_days.setdefault(day, []).append(item.code_index)
                    mask = self.produced.setdefault(day, bytearray(len(self.codes)))
                    mask[item.code_index] = 1
                if len(auction):
                    auction_parts.append(auction)
                verdict = quality.day_verdict(
                    standardized if len(standardized) else None,
                    auction_present=bool(len(auction)),
                )
                recon = None
                daily_row = cache.rows.get((code, day))
                if daily_row is not None:
                    check = quality.reconcile_day(part, daily_row)
                    recon = quality.reconciliation_disposition(check)
                    batch_recon[recon] += 1
                    recon_parts.append(
                        pd.DataFrame(
                            [
                                {
                                    "ts_code": code,
                                    "trade_date": day,
                                    "minute_rows": int(check.get("minute_rows", 0)),
                                    "core_slots": int(verdict["core_slots"]),
                                    "tail_slot_present": bool(verdict["tail_slot_present"]),
                                    # 两端原始总量/总额 + 有符号与绝对误差：
                                    # 只看绝对值会把"偏高"和"偏低"混同（复核 2026-11-19）
                                    "minute_volume": check.get("minute_volume"),
                                    "daily_volume": check.get("daily_volume"),
                                    "minute_amount": check.get("minute_amount"),
                                    "daily_amount": check.get("daily_amount"),
                                    "volume_rel_error": check.get("volume_rel_error"),
                                    "amount_rel_error": check.get("amount_rel_error"),
                                    "signed_volume_rel_error": check.get("signed_volume_rel_error"),
                                    "signed_amount_rel_error": check.get("signed_amount_rel_error"),
                                    "high_gap": check.get("high_gap"),
                                    "low_gap": check.get("low_gap"),
                                    "disposition": recon,
                                    "flag_text": ",".join(check.get("flags", [])) or quality.ROW_OK,
                                }
                            ]
                        )
                    )
                bucket, reason = quality.classify_observation(
                    {
                        "status": source.OK,
                        "suspended": suspended,
                        "quality_status": verdict["status"],
                        "reconciliation": recon,
                    }
                )
                self._account(
                    bucket,
                    reason,
                    item,
                    day,
                    phase,
                    batch_counts,
                    batch_warmup,
                    batch_unresolved,
                    batch_exclusions,
                )
        self._write_batch_files(
            chunk_id,
            key,
            std_parts,
            auction_parts,
            recon_parts,
            conflict_parts,
            batch_exclusions,
        )
        self._commit_batch_raw(raw_frames, chunk_id, key)
        self.counts.update(batch_counts)
        self.warmup.update(batch_warmup)
        self.unresolved.extend(batch_unresolved)
        self.report.conflicts += batch_conflicts
        for name, value in batch_recon.items():
            self.report.recon_counts[name] = self.report.recon_counts.get(name, 0) + int(value)
        return {
            "records": [
                {
                    "ts_code": self.codes[item.code_index],
                    "window": [self.days[item.start_pos], self.days[item.end_pos - 1]],
                    "status": self.request_status[(item.code_index, item.start_pos, item.end_pos)],
                    "rows": 0,
                }
                for item in batch
            ]
            + records,
            "counts": dict(batch_counts),
            "warmup": dict(batch_warmup),
            "unresolved": batch_unresolved,
            "recon": dict(batch_recon),
            "produced": [{"day": d, "indices": i} for d, i in produced_days.items()],
            "suspended": suspended_batch,
            "conflicts": batch_conflicts,
        }

    # ---------------------------------------------------------------- 请求与原始
    def _obtain_frame(
        self, item: Coverage, request_id: str, code: str, start: str, end: str
    ) -> tuple[pd.DataFrame | None, str, pd.DataFrame]:
        spool = self.spool_root / f"{request_id}.parquet"
        index = self.index_root / f"{request_id}.json"
        if self.resume and index.exists() and spool.exists():
            meta = json.loads(index.read_text(encoding="utf-8"))
            if meta.get("status") == source.OK and sha256_file(spool) == meta.get("sha256"):
                self.report.replayed_requests += 1
                return pd.read_parquet(spool), source.OK, pd.DataFrame()
        try:
            fetched = self.src.fetch(code, start, end, str(spec.MINUTE["freq"]))
        except source.RateLimitBlocked as error:
            self.report.blocked = error.as_record()
            self._write_index(index, {"status": "blocked_rate_limit", "window": [start, end]})
            self.log(f"blocked at {code} {start[:10]}：保留工作态，不发布")
            return None, "blocked", pd.DataFrame()
        if fetched.status != source.OK:
            self._write_index(
                index,
                {"status": fetched.status, "window": [start, end], "error": fetched.error},
            )
            return None, fetched.status, pd.DataFrame()
        if fetched.frame.empty:
            self._write_index(
                index, {"status": "empty", "window": [start, end], "sha256": None, "rows": 0}
            )
            return None, source.OK, fetched.conflict_rows
        staging = spool.with_suffix(".partial.parquet")
        fetched.frame.to_parquet(staging, index=False)
        digest = sha256_file(staging)
        staging.replace(spool)
        size = int(spool.stat().st_size)
        self.report.raw_spool_bytes += size
        self._write_index(
            index,
            {
                "status": source.OK,
                "window": [start, end],
                "rows": int(len(fetched.frame)),
                "sha256": digest,
                "bytes": size,
                "splits": int(fetched.splits),
                "retries": int(fetched.retries),
                "conflicts": int(len(fetched.conflicts)),
            },
        )
        return fetched.frame, source.OK, fetched.conflict_rows

    @staticmethod
    def _write_index(path: Path, payload: Mapping[str, Any]) -> None:
        staging = path.with_suffix(".partial.json")
        staging.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(path)

    def _is_suspended(self, cache: DailyCache, code: str, day: str) -> bool:
        row = cache.rows.get((code, day))
        if row is None:
            return False
        return bool(int(row.get("is_suspended", 0) or 0)) or int(row.get("tradable", 1) or 0) == 0

    def _account(
        self,
        bucket: str,
        reason: str | None,
        item: Coverage,
        day: str,
        phase: str,
        counts: Counter,
        warmup: Counter,
        unresolved: list[dict[str, Any]],
        exclusions: list[dict[str, Any]] | None = None,
    ) -> None:
        """计数归属：合格日进覆盖分母，预热日只进预热台账（不混入覆盖）。

        `unresolved` 只留展示用样例；**全量**未解决/异常记录写 `exclusions`，
        落 `exclusions_year=`，不允许用样例代替台账。
        """
        record = {
            "ts_code": self.codes[item.code_index],
            "trade_date": day,
            "phase": phase,
            "bucket": bucket,
            "reason": reason,
            "source": "minutes",
        }
        if exclusions is not None:
            exclusions.append(record)
        if bucket.startswith("unresolved") or bucket == "anomaly":
            if len(unresolved) < 500:
                unresolved.append(record)
        if phase == "warmup":
            warmup[bucket] += 1
            return
        counts[bucket] += 1

    # ---------------------------------------------------------------- 存储辅助
    def _write_batch_files(
        self,
        chunk_id: int,
        key: str,
        std_parts: Sequence[pd.DataFrame],
        auction_parts: Sequence[pd.DataFrame],
        recon_parts: Sequence[pd.DataFrame],
        conflict_parts: Sequence[pd.DataFrame],
        exclusions: Sequence[Mapping[str, Any]],
    ) -> None:
        directory = self.chunk_root / f"chunk-{chunk_id:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.report.standardized_rows += self._write_part(std_parts, directory / f"{key}.parquet")
        self.report.auction_rows += self._write_part(
            auction_parts, directory / f"{key}.auction.parquet"
        )
        self.report.recon_rows += self._write_part(recon_parts, directory / f"{key}.recon.parquet")
        self.report.conflict_rows += self._write_part(
            conflict_parts, directory / f"{key}.conflicts.parquet"
        )
        self.report.exclusion_rows += self._write_part(
            [pd.DataFrame(list(exclusions))] if exclusions else [],
            directory / f"{key}.exclusions.parquet",
        )

    def _commit_batch_raw(self, frames: Sequence[pd.DataFrame], chunk_id: int, key: str) -> None:
        """把本批 spool 合并成内容寻址块（按年分区）；spool 供续跑，发布前清理。"""
        if not frames:
            return
        frame = pd.concat(frames, ignore_index=True)
        for year, part in frame.groupby(frame["trade_time"].astype(str).str[:4]):
            block = artifacts.commit_raw_block(part, self.campaign, str(year))
            self.years.add(str(year))
            self.report.raw_blocks += 1
            self.log(f"raw block {year} {key}: {block['rows']} 行")

    @staticmethod
    def _write_part(parts: Sequence[pd.DataFrame], path: Path) -> int:
        if not parts:
            return 0
        frame = pd.concat(parts, ignore_index=True)
        staging = path.with_suffix(path.suffix + ".partial")
        frame.to_parquet(staging, index=False)
        staging.replace(path)
        return int(len(frame))

    def _load_progress(self, path: Path) -> dict[str, Any]:
        if not self.resume or not path.exists():
            return {"done": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
        staging = path.with_suffix(".partial.json")
        staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(path)

    def _restore_batch(self, record: Mapping[str, Any]) -> None:
        """续跑：恢复计数、覆盖、冲突与请求状态；不重复请求。"""
        self.counts.update(record.get("counts", {}))
        self.warmup.update(record.get("warmup", {}))
        self.unresolved.extend(record.get("unresolved", []))
        for name, value in record.get("recon", {}).items():
            self.report.recon_counts[name] = self.report.recon_counts.get(name, 0) + int(value)
        self.report.conflicts += int(record.get("conflicts", 0))
        for item in record.get("produced", []):
            for index in item.get("indices", []):
                mask = self.produced.setdefault(str(item["day"]), bytearray(len(self.codes)))
                mask[int(index)] = 1
        for code_index, day in record.get("suspended", []):
            self.suspended.add((int(code_index), str(day)))
        for row in record.get("records", []):
            if row.get("kind") == "row_problem" or not row.get("window"):
                continue
            code_index = self.code_index.get(str(row.get("ts_code")))
            start_day, end_day = str(row["window"][0]), str(row["window"][1])
            if code_index is None or start_day not in self.days_index:
                continue
            if end_day not in self.days_index:
                continue
            status = str(row.get("status"))
            self.request_status[
                (code_index, self.days_index[start_day], self.days_index[end_day] + 1)
            ] = status
            if status == source.OK:
                self.report.ok_requests += 1
            elif status:
                self.report.failed_requests += 1

    # ------------------------------------------------------------ 合并与裁定
    def _finalize(self) -> None:
        self._merge_years()
        self._recon_summary()
        self._coverage_summary()

    def _merge_years(self) -> None:
        """按块贡献写年分区 part；跨块重复行按值相同/冲突分别处理。"""
        minute_dir = self.dataset_dir / "minute"
        minute_dir.mkdir(parents=True, exist_ok=True)
        for chunk_dir in sorted(self.chunk_root.glob("chunk-*")):
            files = sorted(
                path
                for path in chunk_dir.glob("batch-*.parquet")
                if not path.name.endswith(
                    (
                        ".auction.parquet",
                        ".recon.parquet",
                        ".conflicts.parquet",
                        ".exclusions.parquet",
                    )
                )
            )
            parts: dict[str, list[pd.DataFrame]] = {}
            for path in files:
                frame = pd.read_parquet(path)
                if not len(frame):
                    continue
                for year, subset in frame.groupby(frame["trade_date"].astype(str).str[:4]):
                    parts.setdefault(str(year), []).append(subset)
            for year, frames in parts.items():
                merged = pd.concat(frames, ignore_index=True)
                merged, conflicts = self._dedupe_without_silent_loss(merged)
                if conflicts:
                    self.report.conflicts += conflicts
                    self.log(f"{chunk_dir.name} {year}: 跨块同键异值 {conflicts} 组")
                merged = merged.sort_values(["trade_date", "ts_code", "slot"]).reset_index(
                    drop=True
                )
                target = minute_dir / f"std_5m_year={year}.{chunk_dir.name}.parquet"
                staging = target.with_suffix(".partial.parquet")
                merged.to_parquet(staging, index=False)
                staging.replace(target)
                self.years.add(year)
        self._merge_aux(minute_dir, ".auction.parquet", "auction_year")
        self._merge_aux(minute_dir, ".conflicts.parquet", "conflicts_year")
        self._merge_aux(minute_dir, ".exclusions.parquet", "exclusions_year")

    @staticmethod
    def _dedupe_without_silent_loss(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """同键重复：值完全相同才去重；值不同登记冲突并整体排除（禁止 keep=first）。"""
        subset = ["trade_date", "ts_code", "slot"]
        duplicated = frame.duplicated(subset=subset, keep=False)
        if not duplicated.any():
            return frame, 0
        rows = frame.loc[duplicated]
        keep: list[Any] = []
        conflicts = 0
        for _, group in rows.groupby(subset, sort=False):
            if group[list(ECONOMIC_COLUMNS)].drop_duplicates().shape[0] > 1:
                conflicts += 1
            else:
                keep.append(group.index[0])
        drop = set(rows.index) - set(keep)
        return frame.loc[~frame.index.isin(drop)], conflicts

    def _merge_aux(self, directory: Path, suffix: str, prefix: str) -> None:
        files = sorted(self.chunk_root.glob(f"chunk-*/batch-*{suffix}"))
        if not files:
            return
        by_year: dict[str, list[pd.DataFrame]] = {}
        for path in files:
            frame = pd.read_parquet(path)
            if not len(frame):
                continue
            key = "trade_date" if "trade_date" in frame.columns else "trade_time"
            years = frame[key].astype(str).str[:4].str.replace("-", "", regex=False)
            for year, part in frame.groupby(years):
                by_year.setdefault(str(year)[:4], []).append(part)
        for year, parts in by_year.items():
            merged = pd.concat(parts, ignore_index=True)
            target = directory / f"{prefix}={year}.parquet"
            staging = target.with_suffix(".partial.parquet")
            merged.to_parquet(staging, index=False)
            staging.replace(target)

    def _recon_summary(self) -> None:
        files = sorted(self.chunk_root.glob("chunk-*/batch-*.recon.parquet"))
        rows = 0
        if files:
            frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
            rows = int(len(frame))
            frame.to_parquet(self.dataset_dir / "reconciliation.parquet", index=False)
        counts = {name: int(value) for name, value in self.report.recon_counts.items()}
        unexplained = sum(value for name, value in counts.items() if name.startswith("unexplained"))
        disclosed = sum(
            value
            for name, value in counts.items()
            if name in {"ohlc_tick_rounding", "volume_rounding", "amount_rounding"}
        )
        within = int(counts.get("within_line", 0))
        self.report.reconciliation = {
            "rows": rows,
            "disposition_counts": counts,
            "within_line": within,
            "disclosed_anomalies": disclosed,
            "unexplained": unexplained,
            "not_reconcilable": int(counts.get("not_reconcilable", 0)),
            "investigation_line": float(spec.UNITS["minute_anomaly_volume_rel_error"]),
            # 0.5% 是调查线，不是准入线：门只要求"越线项都有机械解释并被披露"
            "status": (
                "measured"
                if rows and unexplained == 0
                else "unexplained_anomalies" if rows else "not_attempted"
            ),
        }

    def _coverage_summary(self) -> None:
        """覆盖摘要：计数来自批内逐日裁定；未发出的请求补记为未解决。"""
        for item in build_coverage(self.eligible, self.chunk_days, self.preheat):
            status = self.request_status.get(
                (item.code_index, item.start_pos, item.end_pos), "not_attempted"
            )
            if status == source.OK:
                continue
            for day in self.days[item.start_pos : item.end_pos]:
                phase = (
                    "eligible" if self.eligible[self.days_index[day], item.code_index] else "warmup"
                )
                bucket, reason = quality.classify_observation(
                    {"status": status, "suspended": (item.code_index, day) in self.suspended}
                )
                if phase == "warmup":
                    self.warmup[bucket] += 1
                    if bucket.startswith("unresolved") and len(self.unresolved) < 500:
                        self.unresolved.append(
                            {
                                "ts_code": self.codes[item.code_index],
                                "trade_date": day,
                                "phase": "warmup",
                                "reason": reason,
                            }
                        )
                    continue
                self.counts[bucket] += 1
                if len(self.unresolved) < 500:
                    self.unresolved.append(
                        {
                            "ts_code": self.codes[item.code_index],
                            "trade_date": day,
                            "phase": "eligible",
                            "reason": reason,
                        }
                    )
        expected = int(sum(self.counts.values()))
        accepted = int(
            self.counts.get("ok", 0)
            + self.counts.get("accepted_exclusion", 0)
            + self.counts.get("anomaly", 0)
        )
        unresolved = int(
            sum(value for name, value in self.counts.items() if name.startswith("unresolved"))
        )
        self.report.coverage = {
            "expected_rows": expected,
            "counts": {name: int(value) for name, value in self.counts.items()},
            "accepted_rows": accepted,
            "accepted_share": (accepted / expected) if expected else None,
            "anomaly_count": int(self.counts.get("anomaly", 0)),
            "warmup": {name: int(value) for name, value in self.warmup.items()},
            "warmup_unresolved": int(
                sum(value for name, value in self.warmup.items() if name.startswith("unresolved"))
            ),
            "unresolved_count": unresolved,
            "unresolved_sample": self.unresolved[:50],
            "status": "ready" if unresolved == 0 else "not_ready",
        }
