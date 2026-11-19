"""BaoStock → `source.FetchResult` 适配器（让 R1 的 DownloadEngine 原样跑通）。

只做**字段与时间标签**的确定映射，不改经济值：
- 代码 `600000.SH` ↔ `sh.600000`；
- `time`（`YYYYMMDDHHMMSSsss`）→ `trade_time`（`YYYY-MM-DD HH:MM:SS`）；
- `volume` 单位为股、`amount` 为元（已用同源日线逐日核对，误差 ≤1e-9 量级）；
- 5 分钟线为**收盘标签**（09:35…15:00），无 09:30 竞价行。

服务方禁止并发连接：本类只在 `DownloadEngine` 的单线程循环里调用；
外层仍由 `bs_lock` 保证同一 campaign 只有一个会话。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from scripts.v2_p7.minute_repair_ml import source, spec

FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"
OUTPUT_COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]


def to_bs_code(ts_code: str) -> str:
    """`600000.SH` → `sh.600000`。"""
    code, market = str(ts_code).split(".")
    return f"{'sh' if market.upper() == 'SH' else 'sz'}.{code}"


def to_vendor_time(raw: str) -> str:
    """`20250701093500000` → `2025-07-01 09:35:00`。"""
    text = str(raw)
    return f"{text[0:4]}-{text[4:6]}-{text[6:8]} {text[8:10]}:{text[10:12]}:{text[12:14]}"


class BaoStockSource:
    """实现 `MinuteSource` 的最小接口（`fetch` / `requests` / `retries`）。"""

    def __init__(self, *, log=None) -> None:
        self.requests = 0
        self.retries = 0
        self.latencies: list[float] = []
        self.rows: int = 0
        self._log = log or (lambda message: None)
        self._logged_in = False

    # ------------------------------------------------------------------ 会话
    def _login(self) -> None:
        if self._logged_in:
            return
        import baostock as bs

        result = bs.login()
        if result.error_code != "0":
            raise source.PermissionUnavailable(f"BaoStock 登录失败：{result.error_msg}")
        self._logged_in = True

    def close(self) -> None:
        """登出（幂等）。"""
        if not self._logged_in:
            return
        import baostock as bs

        bs.logout()
        self._logged_in = False

    # ------------------------------------------------------------------ 抓取
    def fetch(self, ts_code: str, start: str, end: str, freq: str | None = None) -> Any:
        """单窗口抓取；空数据返回 `status=ok` 的空表（由覆盖裁定处理）。"""
        if str(freq or spec.MINUTE["freq"]) != "5min":
            raise source.SourceError("BaoStock 适配器只支持 5min（1min 未验证）")
        self._login()
        import baostock as bs

        day_start = start[:10]
        day_end = end[:10]
        started = time.perf_counter()
        self.requests += 1
        result = bs.query_history_k_data_plus(
            to_bs_code(ts_code),
            FIELDS,
            start_date=day_start,
            end_date=day_end,
            frequency="5",
            adjustflag="3",
        )
        rows: list[list[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        elapsed = time.perf_counter() - started
        self.latencies.append(elapsed)
        frame = (
            pd.DataFrame(rows, columns=result.fields)
            if rows
            else pd.DataFrame(columns=result.fields)
        )
        if result.error_code != "0":
            message = str(result.error_msg)[:160]
            if "Unbought" in message or result.error_code == "10001015":
                return source.FetchResult(
                    frame=pd.DataFrame(columns=OUTPUT_COLUMNS),
                    requests=self.requests,
                    status=source.PERMISSION,
                    error=message,
                )
            return source.FetchResult(
                frame=pd.DataFrame(columns=OUTPUT_COLUMNS),
                requests=self.requests,
                status=source.TRANSPORT,
                error=f"{result.error_code}: {message}",
            )
        if not len(frame):
            return source.FetchResult(
                frame=pd.DataFrame(columns=OUTPUT_COLUMNS),
                requests=self.requests,
                status=source.OK,
            )
        normalized = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_time": frame["time"].map(to_vendor_time),
                "open": pd.to_numeric(frame["open"], errors="coerce"),
                "high": pd.to_numeric(frame["high"], errors="coerce"),
                "low": pd.to_numeric(frame["low"], errors="coerce"),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "vol": pd.to_numeric(frame["volume"], errors="coerce"),
                "amount": pd.to_numeric(frame["amount"], errors="coerce"),
            }
        )
        self.rows += len(normalized)
        return source.FetchResult(
            frame=normalized,
            requests=self.requests,
            status=source.OK,
            pages=[{"rows": int(len(normalized))}],
        )

    # ------------------------------------------------------------- 统计摘要
    def stats(self) -> dict[str, Any]:
        """请求数、行数、延迟分布（供 pilot 汇总）。"""
        latencies = sorted(self.latencies)
        count = len(latencies)
        return {
            "requests": self.requests,
            "rows": self.rows,
            "latency_median": round(latencies[count // 2], 2) if count else None,
            "latency_max": round(latencies[-1], 2) if count else None,
            "latency_sum": round(sum(latencies), 1) if count else 0.0,
            "queries_per_hour": (
                round(count / sum(latencies) * 3600.0, 1) if count and sum(latencies) else None
            ),
        }
