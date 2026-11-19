"""R0/R1 **实际入口**用例（复核 2026-11-18 第二轮问题 A/B/C）。

每类断言都读最终发布件（`readiness.json` / `state.json` / 工件目录），不只测帮助函数：
- 合格数据完整通过 → 发布 ready；
- 残缺分钟线（每日仅 1 根）→ 发布但 not_ready，且下游拒绝；
- 限频中断 → 不发布、保存已成功请求、同 run 续跑不重复请求、最终一致。

全部离线：换掉 Source / 日历 / 输入面板，使用临时工件根，不触网、不读真实凭据。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.minute_repair_ml import artifacts, download, quality, run, sessions, source, spec

DAYS = ["20260105", "20260106", "20260107"]


class StubSource:
    """可编程分钟源：按股票返回完整 48 槽，或只 1 根，或第 N 次限频。"""

    def __init__(self, mode: str = "full", block_after: int = 0) -> None:
        self.mode = mode
        self.block_after = block_after
        self.requests = 0
        self.retries = 0
        self.calls: list[str] = []

    def fetch(self, code, start, end, freq=None):
        if self.block_after and self.requests >= self.block_after:
            raise source.RateLimitBlocked(
                "频率超限(1次/小时)",
                limit={
                    "count": 1,
                    "unit": "小时",
                    "window_seconds": 3600.0,
                    "requests_per_hour": 1.0,
                },
            )
        self.requests += 1
        self.calls.append(code)
        rows = []
        for day in pd.date_range(start[:10], end[:10], freq="D"):
            slots = range(1, 49) if self.mode == "full" else [1]
            for k in slots:
                rows.append(
                    {
                        "ts_code": code,
                        "trade_time": f"{day.strftime('%Y-%m-%d')} {sessions.slot_label(k)}",
                        "open": 10.0,
                        "close": 10.0,
                        "high": 10.01,
                        "low": 9.99,
                        "vol": 100.0,
                        "amount": 1000.0,
                    }
                )
        return source.FetchResult(frame=pd.DataFrame(rows), status=source.OK)


def _publish_r0(tmp_path: Path, codes=("600000.SH", "000001.SZ")) -> tuple[Path, Path]:
    base = tmp_path / "artifacts"
    campaign = artifacts.campaign_root("c1", base)
    artifacts.ensure_layout(campaign, base)
    existing = campaign / "runs" / "r0_fake"
    if existing.exists():
        return campaign, existing
    run_dir = artifacts.create_run_dir(campaign, "r0_fake", base)
    inputs = run_dir / "work" / "inputs"
    inputs.mkdir(parents=True)
    artifacts.write_json_checked(inputs / "access.json", {"access_status": "ok_partial_sample"})
    artifacts.write_json_checked(
        inputs / "download_plan.json",
        {"requests_estimate": 2, "eligible_stock_days": 6, "sample_codes": list(codes)},
    )
    artifacts.write_json_checked(inputs / "specification.json", {"spec_sha256": "spec"})
    artifacts.write_readiness(
        inputs,
        {"access": {"ok": True, "detail": "stub"}},
        data_status="ready",
    )
    published = artifacts.publish_artifacts(
        run_dir / "work", run_dir, "r0", ("inputs",), {"specification": "x"}
    )
    artifacts.write_json_checked(
        run_dir / "manifest.json", {"stage": "r0", "payload": {"artifacts": published}}
    )
    return campaign, run_dir


def _patch_r1_inputs(monkeypatch, codes) -> None:
    monkeypatch.setattr(run.panel, "load_calendar", lambda _root: list(DAYS))
    monkeypatch.setattr(
        run.dataset,
        "load_inputs",
        lambda *_a, **_k: {
            "prices": {},
            "basic": {},
            "stock_basic": pd.DataFrame(),
            "industry_pit": pd.DataFrame(),
            "l1_names": {},
            "dates": list(DAYS),
            "codes": list(codes),
        },
    )

    class _Domain:
        def __init__(self, eligible):
            self.eligible = eligible

    monkeypatch.setattr(
        run.domain,
        "build_domain",
        lambda *_a, **_k: _Domain(np.ones((len(DAYS), len(codes)), dtype=bool)),
    )

    # 日线缓存：与 48×100 股 / 48×1000 元对齐，使"完整数据"用例能通过对账门
    def _cache(days, codes):
        rows = {
            (str(code), str(day)): {
                "vol": 48.0,
                "amount": 48.0,
                "high": 10.01,
                "low": 9.99,
                "close": 10.0,
                "is_suspended": 0,
                "tradable": 1,
            }
            for day in days
            for code in codes
        }
        return download.DailyCache(rows=rows, loaded_days={str(day) for day in days})

    monkeypatch.setattr(download, "_daily_cache", _cache)


def _run_r1(
    tmp_path: Path,
    monkeypatch,
    *,
    mode="full",
    block_after=0,
    resume=False,
    codes=("600000.SH", "000001.SZ"),
):
    _patch_r1_inputs(monkeypatch, codes)
    fake = StubSource(mode=mode, block_after=block_after)
    monkeypatch.setattr(run.source, "MinuteSource", lambda **_k: fake)
    campaign, _ = _publish_r0(tmp_path, codes=codes)
    if resume:
        run_dir = artifacts.open_run_for_resume(campaign, "r1_fake", tmp_path / "artifacts")
    else:
        run_dir = artifacts.create_run_dir(campaign, "r1_fake", tmp_path / "artifacts")
    result = run.run_r1(campaign, run_dir, upstream="r0_fake", log=lambda _m: None, resume=resume)
    return campaign, run_dir, fake, result


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    base = tmp_path / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(artifacts, "default_artifact_base", lambda: base)
    return base


class TestR1Entry:
    def test_complete_data_publishes_ready(self, tmp_path, monkeypatch) -> None:
        campaign, run_dir, fake, result = _run_r1(tmp_path, monkeypatch, mode="full")
        published = run_dir / "artifacts" / "dataset"
        readiness = artifacts.read_readiness(published)
        assert result["data_status"] == "ready"
        assert readiness["data_status"] == "ready"
        assert readiness["all_checks_ok"] is True
        assert readiness["checks"]["coverage_ready"]["ok"] is True
        assert readiness["checks"]["warmup_quality"]["ok"] is True
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "completed"
        # 下游可以接上
        assert artifacts.require_ready(campaign, "r1_fake", "dataset").exists()

    def test_one_bar_per_day_is_rejected(self, tmp_path, monkeypatch) -> None:
        """B：每日仅 1 根的"看着有数据"必须被判不完整并阻断下游。"""
        campaign, run_dir, fake, result = _run_r1(tmp_path, monkeypatch, mode="sparse")
        published = run_dir / "artifacts" / "dataset"
        readiness = artifacts.read_readiness(published)
        assert result["data_status"] == "data_not_ready"
        assert readiness["data_status"] == "data_not_ready"
        assert readiness["checks"]["coverage_ready"]["ok"] is False
        assert "partial_slots" in readiness["checks"]["coverage_ready"]["detail"] or (
            "未解决" in readiness["checks"]["coverage_ready"]["detail"]
        )
        with pytest.raises(artifacts.ArtifactError):
            artifacts.require_ready(campaign, "r1_fake", "dataset")

    def test_blocked_then_resume_without_refetch(self, tmp_path, monkeypatch) -> None:
        """C：第 2 个请求限频 → 不发布、留工作态 → 续跑不重复第 1 个请求。"""
        campaign, run_dir, fake, result = _run_r1(tmp_path, monkeypatch, mode="full", block_after=1)
        assert result["data_status"] == "blocked"
        assert not (run_dir / "artifacts").exists(), "阻塞态不得发布冻结证据"
        blocked = json.loads((run_dir / "work" / "blocked.json").read_text(encoding="utf-8"))
        assert blocked["blocked"]["next_allowed_at_utc"]
        assert blocked["requests_done"] == 1
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "blocked"

        # 窗口结束后续跑：首个请求应来自 spool 重放，不再联网
        campaign2, run_dir2, fake2, result2 = _run_r1(
            tmp_path, monkeypatch, mode="full", block_after=0, resume=True
        )
        assert run_dir2 == run_dir
        assert result2["data_status"] == "ready"
        assert fake2.calls == ["000001.SZ"], fake2.calls  # 只补缺失的那一个
        published = run_dir / "artifacts" / "dataset"
        readiness = artifacts.read_readiness(published)
        assert readiness["data_status"] == "ready"
        stats = json.loads((published / "standardized_manifest.json").read_text(encoding="utf-8"))
        assert stats["download"]["replayed_requests"] >= 1


class TestRateLimitMetadata:
    """A：200/40203、JSON 429、文本 429 必须走同一套阻塞元数据。"""

    class _Response:
        def __init__(self, status: int, payload, text: str = "") -> None:
            self.status_code = status
            self._payload = payload
            self.text = text

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    def _post(self, monkeypatch, response) -> dict:
        monkeypatch.setattr(
            source.MinuteSource,
            "_post",
            lambda _self, _params: (response.status_code, self._payload(response)),
        )
        fetcher = source.MinuteSource(token="dummy", log=None)
        with pytest.raises(source.RateLimitBlocked) as info:
            fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        return info.value.as_record()

    @staticmethod
    def _payload(response):
        try:
            payload = response.json()
        except ValueError:
            payload = {"_raw_text": response.text[:200]}
        return payload

    def test_api_40203_is_rate_limit(self, monkeypatch) -> None:
        record = self._post(
            monkeypatch, self._Response(200, {"code": 40203, "msg": "频率超限(1次/小时)"})
        )
        assert record["limit"]["count"] == 1
        assert record["limit"]["unit"] == "小时"
        assert record["next_allowed_at_utc"]

    def test_json_429_is_rate_limit(self, monkeypatch) -> None:
        record = self._post(monkeypatch, self._Response(429, {"msg": "频率超限(10次/分钟)"}))
        assert record["limit"]["window_seconds"] == 60.0
        assert record["next_allowed_at_utc"]

    def test_probe_429_keeps_blocked_metadata(self, monkeypatch) -> None:
        """probe 与 fetch 必须写同一套 `blocked` 结构（旧实现写 frequency_limit 会丢配额）。"""
        monkeypatch.setattr(
            source.MinuteSource,
            "_post",
            lambda _self, _params: (429, {"msg": "频率超限(1次/小时)"}),
        )
        fetcher = source.MinuteSource(token="dummy", log=None)
        record = fetcher.probe_once(
            "600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00", "5min"
        )
        assert record["status"] == source.RATE_LIMIT
        assert record["blocked"]["limit"]["unit"] == "小时"
        assert record["blocked"]["next_allowed_at_utc"]

    def test_text_429_is_rate_limit(self, monkeypatch) -> None:
        record = self._post(
            monkeypatch,
            self._Response(429, ValueError("not json"), text="Too Many Requests 1次/分钟"),
        )
        assert record["limit"]["count"] == 1
        assert record["window_seconds"] == 60.0


class TestThroughputEntry:
    def test_measured_sample_unlocks_ready(self, tmp_path, monkeypatch) -> None:
        """A：合格源 + 足够实测速率必须能生成 ready（旧实现永远不可能）。"""
        cfg = dict(spec.MINUTE["throughput_sample"])
        cfg.update(
            max_requests=5,
            min_requests_for_estimate=2,
            min_seconds_for_estimate=0.0,
            trading_days_per_request=3,
        )
        monkeypatch.setitem(spec.MINUTE, "throughput_sample", cfg)
        fake = StubSource(mode="full")
        sample = run.measure_throughput(fake, list(DAYS) * 4, ["600000.SH"], lambda _m: None)
        assert sample["status"] == "measured"
        assert sample["requests"] >= 2
        out = run._throughput_assessment({"requests_estimate": 10}, [], sample)
        assert out["basis"] == "measured"
        assert out["within_cap"] is True

    def test_low_rate_still_rejected(self) -> None:
        sample = {"status": "measured", "success_per_hour": 1.0, "elapsed_seconds": 60.0}
        out = run._throughput_assessment({"requests_estimate": 37829}, [], sample)
        assert out["within_cap"] is False
        assert out["estimated_total_hours"] > 22

    def test_insufficient_sample_is_not_a_pass(self) -> None:
        sample = {"status": "insufficient", "requests": 1, "elapsed_seconds": 1.0}
        out = run._throughput_assessment({"requests_estimate": 100}, [], sample)
        assert out["within_cap"] is None
        assert out["basis"] == "not_measured"


class TestAccessProbe:
    def test_stops_at_first_block_and_records_quota(self, tmp_path, monkeypatch) -> None:
        from scripts.v2_p7.minute_repair_ml import access_probe

        class _Blocked:
            def __init__(self, **_kwargs) -> None:
                self.requests = 0

            def fetch(self, *_args, **_kwargs):
                self.requests += 1
                raise source.RateLimitBlocked(
                    "频率超限(2次/天)",
                    limit={
                        "count": 2,
                        "unit": "天",
                        "window_seconds": 86400.0,
                        "requests_per_hour": 2.0 / 24.0,
                    },
                )

        monkeypatch.setattr(access_probe.source, "MinuteSource", _Blocked)
        monkeypatch.setattr(access_probe.time, "sleep", lambda _s: pytest.fail("不得等待整个窗口"))
        probe_dir = tmp_path / "artifacts" / "probe"
        summary = access_probe.run_probe(tmp_path, probe_dir, max_requests=8, log=lambda _m: None)
        assert summary["status"] == "blocked"
        assert summary["requests_sent"] == 1
        assert summary["quota_observed"]["count"] == 2
        assert summary["by_freq"]["5min"]["skipped"] == 5
        assert (probe_dir / "report.md").exists()

    def test_structure_flags_incomplete_day(self) -> None:
        from scripts.v2_p7.minute_repair_ml import access_probe

        frame = pd.DataFrame(
            [
                {"trade_time": "2026-10-29 09:30:00", "vol": 1.0, "amount": 1.0},
                {"trade_time": "2026-10-29 09:35:00", "vol": 1.0, "amount": 1.0},
            ]
        )
        out = access_probe._structure(frame, "5min")
        assert out["session_ok"] is False
        assert out["core_rows"] == 1


class TestR0SuccessPath:
    """A：合格源 + 完整语义 + 足够实测速率，必须能从入口跑到 ready 并发布。"""

    class _FullSource:
        def __init__(self) -> None:
            self.requests = 0
            self.retries = 0
            self.last_status = None

        @staticmethod
        def _frame(start: str, end: str, bars: int = 48) -> pd.DataFrame:
            rows = []
            for day in pd.date_range(start[:10], end[:10], freq="D"):
                for k in range(1, bars + 1):
                    rows.append(
                        {
                            "ts_code": "600000.SH",
                            "trade_time": f"{day.strftime('%Y-%m-%d')} {sessions.slot_label(k)}",
                            "open": 10.0,
                            "close": 10.0,
                            "high": 10.01,
                            "low": 9.99,
                            "vol": 100.0,
                            "amount": 1000.0,
                        }
                    )
            return pd.DataFrame(rows)

        def probe_once(self, code, start, stop, freq):
            self.requests += 1
            return {
                "parameters": {"ts_code": code, "start_date": start, "freq": freq},
                "status": source.OK,
                "rows": 49 if freq == "5min" else 241,
            }

        def capture(self, code, start, end, freq=None):
            self.requests += 1
            frame = self._frame(start, end, bars=48)
            return {
                "status": source.OK,
                "requests": 1,
                "rows": int(len(frame)),
                "frame": frame,
                "conflicts": [],
                "conflict_rows": pd.DataFrame(),
            }

        def fetch(self, code, start, end, freq=None, **_kwargs):
            self.requests += 1
            return source.FetchResult(frame=self._frame(start, end), status=source.OK)

    def test_full_run_reaches_ready(self, tmp_path, monkeypatch) -> None:
        days = [
            moment.strftime("%Y%m%d")
            for moment in pd.date_range("2020-01-01", periods=1000, freq="B")
        ]
        source_obj = self._FullSource()
        monkeypatch.setattr(run.source, "MinuteSource", lambda **_k: source_obj)
        monkeypatch.setattr(run.panel, "load_calendar", lambda _root: days)
        monkeypatch.setattr(
            run,
            "_download_plan",
            lambda *a, **k: {
                "requests_estimate": 5,
                "eligible_stock_days": 5,
                "active_stocks": 1,
                "requested_stock_days": 5,
                "request_hours_at_100_per_minute": 0.1,
                "sample_codes": ["600000.SH"],
            },
        )
        monkeypatch.setattr(
            run, "_verify_baseline", lambda _root: {"problems": [], "manifest_sha256": "x"}
        )
        monkeypatch.setattr(
            run,
            "_local_sample_evidence",
            lambda: {
                "exists": True,
                "rows": 241,
                "first_time": "09:30:00",
                "last_time": "15:00:00",
                "has_0930_auction": True,
            },
        )
        monkeypatch.setattr(
            run,
            "_daily_rows",
            lambda codes, days_: {
                (str(code), str(day)): {
                    "vol": 48.0,
                    "amount": 48.0,
                    "high": 10.01,
                    "low": 9.99,
                    "close": 10.0,
                    "is_suspended": 0,
                    "tradable": 1,
                }
                for code in codes
                for day in days_
            },
        )
        monkeypatch.setitem(
            spec.MINUTE,
            "throughput_sample",
            {
                "max_requests": 3,
                "max_seconds": 30.0,
                "trading_days_per_request": 2,
                "min_requests_for_estimate": 2,
                "min_seconds_for_estimate": 0.0,
            },
        )
        monkeypatch.setattr(
            run,
            "code_identity",
            lambda: {
                "code_commit": "deadbeef",
                "dirty_patch_sha": "clean",
                "package_files": {},
            },
        )
        monkeypatch.setattr(
            run.spec,
            "INPUTS",
            [{"name": "stub", "relative": "temp/stub.parquet", "kind": "file"}],
        )
        stub = tmp_path / "temp" / "stub.parquet"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(b"stub")
        monkeypatch.setattr(run, "REPO_ROOT", tmp_path)

        base = tmp_path / "artifacts"
        campaign = artifacts.campaign_root("c1", base)
        artifacts.ensure_layout(campaign, base)
        run_dir = artifacts.create_run_dir(campaign, "r0_ok", base)
        result = run.run_r0(campaign, run_dir, log=lambda _m: None, use_network=True)
        assert result["summary"]["data_status"] == "ready"
        assert result["summary"]["engineering_status"] == "valid"
        readiness = artifacts.read_readiness(run_dir / "artifacts" / "inputs")
        assert readiness["data_status"] == "ready"
        assert readiness["all_checks_ok"] is True
        assert readiness["checks"]["throughput"]["ok"] is True
        assert artifacts.require_ready(campaign, "r0_ok", "inputs").exists()


class TestBaoStockSampling:
    """抽样必须覆盖沪深，不能在市场维度被"先到先得"填满（Step 1 全 SH 的教训）。"""

    def test_both_markets_are_sampled(self) -> None:
        from scripts.v2_p7.minute_repair_ml import bs_coverage

        rows = [
            {
                "ts_code": f"{index:06d}.{market}",
                "market": market,
                "industry": "bank",
                "eligible_days": 500,
                "liquidity": 1.0,
                "last_trade_date": "20261029",
                "delisted": False,
            }
            for market in ("SH", "SZ")
            for index in range(50)
        ]
        table = pd.DataFrame(rows)
        picked = bs_coverage._sample(table, 10, 0)
        assert set(picked["market"]) == {"SH", "SZ"}
        assert int((picked["market"] == "SH").sum()) == 5

    def test_delisted_quota_is_reserved(self) -> None:
        from scripts.v2_p7.minute_repair_ml import bs_coverage

        rows = [
            {
                "ts_code": f"9{index:05d}.SH",
                "market": "SH",
                "industry": "bank",
                "eligible_days": 300,
                "liquidity": 1.0,
                "last_trade_date": "20240101",
                "delisted": index < 3,
            }
            for index in range(10)
        ]
        picked = bs_coverage._sample(pd.DataFrame(rows), 6, 3)
        assert int(picked["delisted"].sum()) == 3


class TestBaoStockSingleConnection:
    """服务方禁止并发连接：锁必须拦住第二个会话，且不误伤串行。"""

    def test_second_session_is_refused(self, tmp_path) -> None:
        from scripts.v2_p7.minute_repair_ml import bs_lock

        first = bs_lock.acquire(tmp_path, owner="a")
        try:
            with pytest.raises(bs_lock.ConcurrentSessionError):
                bs_lock.acquire(tmp_path, owner="b")
        finally:
            bs_lock.release(first)
        second = bs_lock.acquire(tmp_path, owner="c")
        bs_lock.release(second)
        assert not (tmp_path / bs_lock.LOCK_NAME).exists()

    def test_stale_lock_can_be_taken_over(self, tmp_path) -> None:
        import json

        from scripts.v2_p7.minute_repair_ml import bs_lock

        (tmp_path / bs_lock.LOCK_NAME).write_text(
            json.dumps({"pid": 999999, "started": 1.0, "owner": "leftover"}),
            encoding="utf-8",
        )
        path = bs_lock.acquire(tmp_path, owner="new", stale_seconds=1.0)
        bs_lock.release(path)
        assert not (tmp_path / bs_lock.LOCK_NAME).exists()


class TestBaoStockProbe:
    """B 方案探查的解析逻辑（离线，不登录、不发请求）。"""

    @staticmethod
    def _frame(days: int = 1, bars: int = 48) -> pd.DataFrame:
        rows = []
        for day in range(days):
            for k in range(bars):
                minute = 935 + k * 5 if k < 24 else 1300 + (k - 24) * 5
                rows.append(
                    {
                        "date": f"2026-10-{26 + day:02d}",
                        "time": f"202610{26 + day:02d}{minute:04d}00000",
                        "open": 10.0,
                        "high": 10.01,
                        "low": 9.99,
                        "close": 10.0,
                        "volume": 100.0,
                        "amount": 1000.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_full_day_is_session_ok_and_end_labelled(self) -> None:
        from scripts.v2_p7.minute_repair_ml import baostock_probe

        out = baostock_probe._bar_structure(self._frame(2, 48))
        assert out["session_ok_days"] == 2
        assert out["has_0930_auction"] is False
        assert out["label_convention"] == "end"

    def test_short_day_is_flagged(self) -> None:
        from scripts.v2_p7.minute_repair_ml import baostock_probe

        out = baostock_probe._bar_structure(self._frame(1, 1))
        assert out["session_ok_days"] == 0

    def test_chunk_consistency_equal_and_unequal(self) -> None:
        from scripts.v2_p7.minute_repair_ml import baostock_probe

        whole = self._frame(3, 48)
        merged = pd.concat([self._frame(3, 48), pd.DataFrame()], ignore_index=True)
        ok = baostock_probe._chunk_consistency(
            {
                "chunk_whole": whole,
                "chunk_a": merged.iloc[:96],
                "chunk_b": merged.iloc[96:],
                "chunk_c": pd.DataFrame(columns=whole.columns),
            }
        )
        assert ok["status"] == "ok"
        bad = baostock_probe._chunk_consistency(
            {
                "chunk_whole": whole,
                "chunk_a": merged.iloc[:-1],
                "chunk_b": pd.DataFrame(columns=whole.columns),
                "chunk_c": pd.DataFrame(columns=whole.columns),
            }
        )
        assert bad["status"] == "unequal"

    def test_daily_reconcile_detects_units(self) -> None:
        from scripts.v2_p7.minute_repair_ml import baostock_probe

        minute = self._frame(1, 48)
        daily = pd.DataFrame(
            [
                {
                    "date": "2026-10-26",
                    "volume": 100.0 * 48,
                    "amount": 1000.0 * 48,
                }
            ]
        )
        out = baostock_probe._daily_reconcile(minute, daily)
        assert out["status"] == "measured"
        assert out["within_line"] is True


class TestSignedReconciliation:
    """对账必须保留方向：绝对值会把"偏高"和"偏低"混成一类（复核 2026-11-19）。"""

    def test_signed_error_keeps_direction(self) -> None:
        minute = pd.DataFrame(
            {
                "ts_code": "A",
                "trade_time": ["2026-01-05 09:35:00"],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.0],
                "vol": [1100.0],
                "amount": [11000.0],
            }
        )
        daily = {"vol": 10.0, "amount": 10.0, "high": 10.1, "low": 9.9}
        check = quality.reconcile_day(minute, daily)
        assert check["signed_volume_rel_error"] == pytest.approx(0.1)
        assert check["volume_rel_error"] == pytest.approx(0.1)
        assert check["signed_amount_rel_error"] == pytest.approx(0.1)

    def test_negative_side_is_not_collapsed(self) -> None:
        minute = pd.DataFrame(
            {
                "ts_code": "A",
                "trade_time": ["2026-01-05 09:35:00"],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.0],
                "vol": [900.0],
                "amount": [9000.0],
            }
        )
        daily = {"vol": 10.0, "amount": 10.0, "high": 10.1, "low": 9.9}
        check = quality.reconcile_day(minute, daily)
        assert check["signed_volume_rel_error"] == pytest.approx(-0.1)
        assert check["volume_rel_error"] == pytest.approx(0.1)


class TestQualityVerdicts:
    def test_one_bar_is_partial_slots(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "trade_date": "20260105",
                    "ts_code": "A",
                    "slot": 1,
                    "O": 10.0,
                    "C": 10.0,
                    "H": 10.01,
                    "L": 9.99,
                    "Q": 100.0,
                    "A": 1000.0,
                }
            ]
        )
        verdict = quality.day_verdict(frame)
        assert verdict["status"] == "partial_slots"
        assert verdict["missing_core"] == 46  # k=1..47 缺 46 个；k=48 只告警
        assert "tail_slot_missing" in verdict["warnings"]

    def test_inconsistent_ohlc_is_invalid(self) -> None:
        rows = [
            {
                "trade_date": "20260105",
                "ts_code": "A",
                "slot": k,
                "O": 10.0,
                "C": 10.5,
                "H": 10.1,
                "L": 9.9,
                "Q": 100.0,
                "A": 1000.0,
            }
            for k in range(1, 49)
        ]
        verdict = quality.day_verdict(pd.DataFrame(rows))
        assert verdict["status"] == "invalid_rows"

    def test_full_day_is_ok(self) -> None:
        rows = [
            {
                "trade_date": "20260105",
                "ts_code": "A",
                "slot": k,
                "O": 10.0,
                "C": 10.0,
                "H": 10.01,
                "L": 9.99,
                "Q": 100.0,
                "A": 1000.0,
            }
            for k in range(1, 49)
        ]
        verdict = quality.day_verdict(pd.DataFrame(rows))
        assert verdict["status"] == "ok"
