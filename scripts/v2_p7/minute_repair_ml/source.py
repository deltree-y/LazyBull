"""分钟数据来源：权限探针、时间游标抓取、分块穷尽与限频（协议 §4.3/§5.2）。

实现说明（与"复用现有客户端"的差异，必须如实登记）：
现有 `TushareClient.query` 走 `tushare.pro_api`，其内部对**非 200 响应**返回空表而不抛错
（`bool(Response)` 为 False），会把网关 5xx 误判成"该区间无数据"。分钟下载的完整性证据
要求区分"空页"与"传输失败"，因此本模块用同一 `TS_TOKEN` 注入与同一限频策略直接 POST，
显式记录 HTTP 状态、返回码与重试次数；请求日志不保存 token、认证头或带凭证 URL。

接口未承诺 `offset`：本模块**不使用 offset 翻页**，只做时间窗口二分的穷尽游标。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec

ENDPOINT = "https://api.tushare.pro"
OK = "ok"
EMPTY = "empty"
RATE_LIMIT = "rate_limit"
PERMISSION = "permission_unavailable"
TRANSPORT = "transport_error"
UNCLASSIFIED = "api_rejection_unclassified"
TRUNCATED = "truncated_page"


class SourceError(RuntimeError):
    """来源层错误（权限、传输、无法穷尽的窗口）。"""


class PermissionUnavailable(SourceError):
    """接口权限不可用（明确拒绝或积分不足）。"""


class TransportFailure(SourceError):
    """传输失败（非 200、JSON 不可解析、连接错误）。"""


@dataclass
class FetchResult:
    """单窗口抓取结果与证据。"""

    frame: pd.DataFrame
    requests: int = 0
    status: str = OK
    pages: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    conflict_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    retries: int = 0
    splits: int = 0
    error: str | None = None
    truncated: bool = False


def load_token() -> str:
    """从 `.env` / 环境变量注入 token（与项目同一入口，不打印）。"""
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TS_TOKEN")
    if not token:
        raise SourceError("缺少 TS_TOKEN（.env 或环境变量）")
    return token


WINDOW_SECONDS: dict[str, float] = {"分钟": 60.0, "小时": 3600.0, "天": 86400.0}


def parse_frequency_limit(message: str) -> dict[str, Any] | None:
    """从限频消息里解析"N 次/分钟|小时|天"（服务端配额的直接证据）。

    同时给出该窗口的**长度**与折算每小时速率：多窗口并存时必须同时满足，
    因此调度侧取最严格窗口（最小速率），而不是取最宽松。
    """
    import re

    match = re.search(r"(\d+)\s*次\s*/\s*(分钟|小时|天)", str(message or ""))
    if not match:
        return None
    unit = match.group(2)
    per_hour = {"分钟": 60.0, "小时": 1.0, "天": 1.0 / 24.0}[unit]
    return {
        "count": int(match.group(1)),
        "unit": unit,
        "window_seconds": WINDOW_SECONDS[unit],
        "requests_per_hour": float(match.group(1)) * per_hour,
    }


def classify_http(status: int) -> str:
    """HTTP 状态分类：429 是限频，5xx 是传输，其余 4xx 未归类。"""
    if int(status) == 429:
        return RATE_LIMIT
    if int(status) >= 500:
        return TRANSPORT
    return UNCLASSIFIED


def block_record(
    message: str, limit: dict[str, Any] | None, *, now: float | None = None
) -> dict[str, Any]:
    """**唯一**的阻塞记录结构；probe、fetch、下载器、readiness 共用同一组键。

    历史缺陷：probe 落 `frequency_limit`、fetch 落 `limit`，两侧互不相认，导致配额与
    "下次可尝试时间"在 R0 里丢失。此函数是唯一构造点。
    """
    moment = time.time() if now is None else float(now)
    window = float(limit["window_seconds"]) if limit and limit.get("window_seconds") else None
    next_epoch = moment + window if window else None
    return {
        "reason": "rate_limit",
        "message": str(message)[:200],
        "limit": limit,
        "window_seconds": window,
        "observed_at_epoch": round(moment, 3),
        "next_allowed_at_epoch": round(next_epoch, 3) if next_epoch else None,
        "next_allowed_at_utc": (
            datetime.fromtimestamp(next_epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if next_epoch
            else None
        ),
    }


class RateLimitBlocked(SourceError):
    """命中已知配额：携带下一次可尝试时间，调用方必须停止当前批次而不是短重试。"""

    def __init__(self, message: str, *, limit: dict[str, Any] | None, now: float | None = None):
        super().__init__(message)
        self.record = block_record(message, limit, now=now)
        self.limit = self.record["limit"]
        self.window_seconds = self.record["window_seconds"]
        self.next_allowed_at_epoch = self.record["next_allowed_at_epoch"]
        self.observed_at_epoch = self.record["observed_at_epoch"]

    def as_record(self) -> dict[str, Any]:
        """可落盘的阻塞记录（不落 token）。"""
        return dict(self.record)


def classify_payload(code: Any, message: str) -> str:
    """按返回码与消息分类（不把 4xx 一概归为没权限）。"""
    text = str(message or "")
    if code == 0:
        return OK
    if any(key in text for key in ("频率", "频次", "每分钟", "每天", "每日")):
        return RATE_LIMIT
    if any(key in text for key in ("没有访问该接口的权限", "无权限", "无权", "积分")):
        return PERMISSION
    return UNCLASSIFIED


class MinuteSource:
    """限频 + 显式状态 + 时间二分穷尽的分钟抓取器。"""

    def __init__(
        self,
        *,
        token: str | None = None,
        rate_limit_per_minute: int | None = None,
        max_retries: int | None = None,
        timeout: tuple[float, float] = (8.0, 30.0),
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._token = token or load_token()
        # 显式 0 必须按 0 执行：不能用 `or` 把"禁止重试"变回默认重试
        self.rate_limit = int(
            spec.MINUTE["rate_limit_start_per_minute"]
            if rate_limit_per_minute is None
            else rate_limit_per_minute
        )
        self.max_retries = int(
            spec.MINUTE["max_recoverable_retries"] if max_retries is None else max_retries
        )
        self.timeout = timeout
        self._log = log or (lambda message: None)
        self._last_call = 0.0
        self.requests = 0
        self.retries = 0
        self.last_status: str | None = None

    def _pace(self) -> None:
        interval = 60.0 / max(1, self.rate_limit)
        elapsed = time.monotonic() - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.monotonic()

    def _post(self, params: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        import requests

        body = {
            "api_name": str(spec.MINUTE["api"]),
            "token": self._token,
            "params": dict(params),
            "fields": ",".join(spec.MINUTE["fields"]),
        }
        self._pace()
        self.requests += 1
        try:
            response = requests.post(ENDPOINT, json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TransportFailure(f"{type(exc).__name__}: {exc}") from exc
        status = int(response.status_code)
        try:
            payload = response.json()
        except ValueError:
            # 非 JSON（如网关 HTML 或文本 429）不得在解析层变成"传输错误"：
            # 429 仍必须走限频路径，其余按 HTTP 状态分类。
            payload = {"_raw_text": response.text[:200]}
        if not isinstance(payload, Mapping):
            payload = {"_raw_text": str(payload)[:200]}
        return status, payload

    def probe_once(self, ts_code: str, start: str, end: str, freq: str) -> dict[str, Any]:
        """单次探针（R0 用）：记录状态、行数、时间边界，不做重试放大。"""
        params = {
            "ts_code": ts_code,
            "freq": freq,
            "start_date": start,
            "end_date": end,
            "limit": int(spec.MINUTE["page_limit"]),
        }
        record: dict[str, Any] = {
            "parameters": params,
            "endpoint": ENDPOINT,
            "token_persisted": False,
        }
        try:
            status, payload = self._post(params)
        except RateLimitBlocked as error:
            record.update(http=None, status=RATE_LIMIT, error=str(error)[:300], **error.as_record())
            self.last_status = RATE_LIMIT
            return record
        except TransportFailure as error:
            record.update(http=None, status=TRANSPORT, error=str(error)[:300])
            self.last_status = TRANSPORT
            return record
        record["http"] = status
        code = payload.get("code")
        message = str(payload.get("msg") or payload.get("_raw_text") or "")
        record.update(code=code, message=message[:300])
        if status != 200:
            category = classify_http(status)
            record["status"] = category
            self.last_status = category
            if category == RATE_LIMIT:
                record["blocked"] = block_record(
                    message or "HTTP 429", parse_frequency_limit(message)
                )
            return record
        category = classify_payload(code, message)
        record["status"] = category
        self.last_status = category
        if category == RATE_LIMIT:
            record["blocked"] = block_record(message, parse_frequency_limit(message))
        if category != OK:
            return record
        data = payload.get("data") or {}
        fields = data.get("fields") or []
        items = data.get("items") or []
        frame = pd.DataFrame(items, columns=fields)
        record.update(
            rows=int(len(frame)),
            columns=list(fields),
            first_time=(
                str(frame["trade_time"].iloc[0])
                if "trade_time" in frame.columns and len(frame)
                else None
            ),
            last_time=(
                str(frame["trade_time"].iloc[-1])
                if "trade_time" in frame.columns and len(frame)
                else None
            ),
            unique_times=int(frame["trade_time"].nunique()) if "trade_time" in frame.columns else 0,
            vol_sum=(
                float(pd.to_numeric(frame.get("vol"), errors="coerce").sum()) if len(frame) else 0.0
            ),
            amount_sum=(
                float(pd.to_numeric(frame.get("amount"), errors="coerce").sum())
                if len(frame)
                else 0.0
            ),
        )
        return record

    def capture(
        self, ts_code: str, start: str, end: str, freq: str | None = None
    ) -> dict[str, Any]:
        """单窗口抓取并保留原始行（R0 样例留存；限频时如实返回状态）。"""
        result = self.fetch(ts_code, start, end, freq)
        return {
            "status": result.status,
            "requests": result.requests,
            "rows": int(len(result.frame)),
            "frame": result.frame,
            "conflicts": result.conflicts,
            "conflict_rows": result.conflict_rows,
            "error": result.error,
        }

    def fetch(
        self,
        ts_code: str,
        start: str,
        end: str,
        freq: str | None = None,
        *,
        depth: int = 0,
    ) -> FetchResult:
        """抓取单股时间窗；命中页上限即按时间二分，直到每段都严格小于上限。"""
        freq = freq or str(spec.MINUTE["freq"])
        params = {
            "ts_code": ts_code,
            "freq": freq,
            "start_date": start,
            "end_date": end,
            "limit": int(spec.MINUTE["page_limit"]),
        }
        attempt = 0
        while True:
            try:
                status, payload = self._post(params)
            except TransportFailure as error:
                attempt += 1
                if attempt > self.max_retries:
                    return FetchResult(
                        frame=pd.DataFrame(columns=list(spec.MINUTE["fields"])),
                        requests=self.requests,
                        status=TRANSPORT,
                        error=str(error)[:300],
                    )
                time.sleep(min(60.0, 2.0**attempt))
                self.retries += 1
                continue
            if status != 200:
                category = classify_http(status)
                if category == RATE_LIMIT:
                    text = str(payload.get("msg") or payload.get("_raw_text") or "HTTP 429")
                    raise RateLimitBlocked(
                        f"HTTP 429：{text[:180]}", limit=parse_frequency_limit(text)
                    )
                if category == TRANSPORT:
                    attempt += 1
                    if attempt > self.max_retries:
                        return FetchResult(
                            frame=pd.DataFrame(columns=list(spec.MINUTE["fields"])),
                            requests=self.requests,
                            status=TRANSPORT,
                            error=f"HTTP {status}",
                        )
                    time.sleep(min(60.0, 2.0**attempt))
                    self.retries += 1
                    continue
                return FetchResult(
                    frame=pd.DataFrame(columns=list(spec.MINUTE["fields"])),
                    requests=self.requests,
                    status=UNCLASSIFIED,
                    error=f"HTTP {status}",
                )
            code = payload.get("code")
            message = str(payload.get("msg") or payload.get("_raw_text") or "")
            category = classify_payload(code, message)
            if category == RATE_LIMIT:
                # 已知配额（分钟/小时/天）：同窗口内重试不会成功，交由调用方停止批次
                raise RateLimitBlocked(message[:200], limit=parse_frequency_limit(message))
            if category == PERMISSION:
                raise PermissionUnavailable(message[:200])
            if category != OK:
                return FetchResult(
                    frame=pd.DataFrame(columns=list(spec.MINUTE["fields"])),
                    requests=self.requests,
                    status=category,
                    error=message[:200],
                )
            data = payload.get("data") or {}
            fields = data.get("fields") or []
            items = data.get("items") or []
            frame = pd.DataFrame(items, columns=fields)
            if len(frame) >= int(spec.MINUTE["page_limit"]):
                if depth >= 8:
                    return FetchResult(
                        frame=frame,
                        requests=self.requests,
                        status=TRUNCATED,
                        truncated=True,
                        error="二分深度超过 8 层仍满页",
                    )
                left, right = _split_window(start, end)
                if left is None:
                    return FetchResult(
                        frame=frame,
                        requests=self.requests,
                        status=TRUNCATED,
                        truncated=True,
                        error="单日仍满页且无法再二分",
                    )
                first = self.fetch(ts_code, left[0], left[1], freq, depth=depth + 1)
                second = self.fetch(ts_code, right[0], right[1], freq, depth=depth + 1)
                merged = _merge_pages([first, second])
                merged.splits += first.splits + second.splits + 1
                merged.retries += first.retries + second.retries
                merged.status = _worst_status([first.status, second.status])
                # 子段的冲突必须一路上传：递归分块时只在叶子做检测会丢掉整段冲突
                merged.conflicts = _merge_conflicts([first, second]) + merged.conflicts
                return merged
            # 单页也做全字段去重与同键异值检测：重复行/冲突不得被当成"数据完整"
            single = _merge_pages([FetchResult(frame=frame, status=OK)])
            single.requests = self.requests
            single.pages = [{"rows": len(frame)}]
            return single


def _merge_conflicts(results: Sequence[FetchResult]) -> list[dict[str, Any]]:
    """合并子段冲突（按 (ts_code, trade_time) 去重，保留最大变体数）。"""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in results:
        for conflict in item.conflicts:
            key = (str(conflict.get("ts_code")), str(conflict.get("trade_time")))
            existing = merged.get(key)
            if existing is None or int(conflict.get("variants", 0)) > int(
                existing.get("variants", 0)
            ):
                merged[key] = dict(conflict)
    return list(merged.values())


def _worst_status(statuses: Sequence[str]) -> str:
    """合并多段状态（任何非 ok 都向上传播）。"""
    for status in statuses:
        if status != OK:
            return status
    return OK


def _split_window(start: str, end: str) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """按时间中点二分一个 `YYYY-MM-DD HH:MM:SS` 窗口；无法二分返回 (None, None)。"""
    left = pd.Timestamp(start)
    right = pd.Timestamp(end)
    if right <= left:
        return None, None
    middle = left + (right - left) / 2
    if middle <= left or middle >= right:
        return None, None
    left_end = middle.floor("5min")
    right_start = left_end + pd.Timedelta(minutes=5)
    if left_end <= left or right_start > right:
        middle_exact = left + (right - left) / 2
        left_end = middle_exact - pd.Timedelta(seconds=1)
        right_start = middle_exact
    return (
        (left.strftime("%Y-%m-%d %H:%M:%S"), left_end.strftime("%Y-%m-%d %H:%M:%S")),
        (right_start.strftime("%Y-%m-%d %H:%M:%S"), right.strftime("%Y-%m-%d %H:%M:%S")),
    )


def _merge_pages(results: Sequence[FetchResult]) -> FetchResult:
    """合并分段结果并做全字段去重 + 同键异值冲突登记。

    冲突键**不进入标准化**：原始变体整行保留在 `conflict_rows`（含双方经济值），
    由调用方复核后决定丢弃或重取；不做静默 `keep=last`。
    """
    frames = [item.frame for item in results if len(item.frame)]
    conflicts: list[dict[str, Any]] = []
    conflict_rows: list[pd.DataFrame] = [
        item.conflict_rows for item in results if len(item.conflict_rows)
    ]
    if not frames:
        return FetchResult(
            frame=pd.DataFrame(),
            status=_worst_status([item.status for item in results]) if results else OK,
            conflict_rows=(
                pd.concat(conflict_rows, ignore_index=True) if conflict_rows else pd.DataFrame()
            ),
        )
    merged = pd.concat(frames, ignore_index=True)
    if "ts_code" in merged.columns and "trade_time" in merged.columns:
        subset = ["ts_code", "trade_time"]
        duplicated = merged.duplicated(subset=subset, keep=False)
        if duplicated.any():
            groups = merged.loc[duplicated].groupby(subset, sort=False)
            conflicted_keys: list[tuple[str, str]] = []
            for key, group in groups:
                economic = group.drop(columns=subset).drop_duplicates()
                if len(economic) > 1:
                    conflicts.append(
                        {
                            "ts_code": str(key[0]),
                            "trade_time": str(key[1]),
                            "variants": int(len(economic)),
                        }
                    )
                    conflicted_keys.append((str(key[0]), str(key[1])))
                    conflict_rows.append(group.copy())
            if conflicted_keys:
                drop_mask = pd.Series(False, index=merged.index)
                for code, moment in conflicted_keys:
                    drop_mask |= (merged["ts_code"].astype(str) == code) & (
                        merged["trade_time"].astype(str) == moment
                    )
                merged = merged.loc[~drop_mask]
            merged = merged.drop_duplicates(subset=subset, keep="first")
    merged = merged.sort_values("trade_time").reset_index(drop=True)
    return FetchResult(
        frame=merged,
        requests=sum(item.requests for item in results),
        status=_worst_status([item.status for item in results]),
        conflicts=conflicts,
        conflict_rows=(
            pd.concat(conflict_rows, ignore_index=True) if conflict_rows else pd.DataFrame()
        ),
        retries=sum(item.retries for item in results),
        splits=sum(item.splits for item in results),
    )
