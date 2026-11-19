"""独立检视（2026-11-18）整改的隔离控制流测试。

覆盖 D1（缺观测不得崩溃）、D2（阶段门唯一来源 + R1 服从 readiness）、
D3（raw 先行 / 冲突隔离 / 覆盖裁定）、D4（限频阻塞而非秒级重试）、
D5（吞吐只认配额或实测）、D6（预热覆盖与断点）。

全部离线：不触网、不读真实分钟数据、不写正式工件根。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7.minute_repair_ml import (
    artifacts,
    download,
    quality,
    report,
    run,
    source,
    spec,
)


@pytest.fixture(autouse=True)
def _isolated_artifact_base(tmp_path, monkeypatch):
    """隔离输入、工件与凭据；即使阶段拒绝门失效，也不能触及生产或网络。"""
    root = tmp_path / "repo"
    clean = root / "data" / "clean"
    clean.mkdir(parents=True)
    pd.DataFrame(
        {
            "exchange": ["SSE"] * 3,
            "cal_date": ["20260105", "20260106", "20260107"],
            "is_open": [1] * 3,
        }
    ).to_parquet(clean / "trade_cal.parquet", index=False)
    # 此控制流夹具只需输入存在性；下载计划和基准核验另由测试显式替身提供。
    for item in spec.INPUTS:
        path = root / item["relative"]
        if item["kind"] == "partition":
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"fixture_only": [True]}).to_parquet(path, index=False)
    protocol = root / spec.PROTOCOL_RELATIVE
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("# 合成协议：仅供离线控制流测试\n", encoding="utf-8")

    # R0 的输入盘点、日历、旧样例及协议读取全部落在合成根；R1 门失效时亦同。
    for module in (run, download, artifacts, spec):
        monkeypatch.setattr(module, "REPO_ROOT", root)

    base = tmp_path / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(artifacts, "default_artifact_base", lambda: base)

    def forbidden_external_access(*_args, **_kwargs):
        pytest.fail("隔离测试不得读取真实凭据或发起网络请求")

    # 单个 HTTP 分类测试可覆盖 _post 为假响应；底层请求仍有独立拦截。
    import requests

    monkeypatch.delenv("TS_TOKEN", raising=False)
    monkeypatch.setattr(source, "load_token", forbidden_external_access)
    monkeypatch.setattr(source.MinuteSource, "_post", forbidden_external_access)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden_external_access)
    return base


def plan_stub() -> dict:
    """最小下载计划（避免测试读全量日线面板）。"""
    return {
        "requests_estimate": 100,
        "eligible_stock_days": 1000,
        "active_stocks": 10,
        "requested_stock_days": 1200,
        "request_hours_at_100_per_minute": 1.0,
        "preheat_trading_days": 20,
        "chunk_trading_days": 120,
        "trading_days": 240,
    }


def identity_stub() -> dict:
    return {"code_commit": "deadbeef", "dirty_patch_sha": "clean", "package_files": {}}


class TestThroughput:
    def test_no_limit_is_not_measured_not_unlimited(self) -> None:
        out = run._throughput_assessment(plan_stub(), [])
        assert out["basis"] == "not_measured"
        assert out["within_cap"] is None
        assert out["observed_quota_requests_per_hour"] is None

    def test_strictest_window_wins(self) -> None:
        limits = [
            {"count": 60, "unit": "小时", "window_seconds": 3600.0, "requests_per_hour": 60.0},
            {"count": 1, "unit": "小时", "window_seconds": 3600.0, "requests_per_hour": 1.0},
        ]
        out = run._throughput_assessment(plan_stub(), limits)
        assert out["observed_quota_requests_per_hour"] == pytest.approx(1.0)
        assert out["within_cap"] is False
        assert out["estimated_total_hours"] == pytest.approx(100.0)

    def test_none_values_render_as_measured_missing(self) -> None:
        assert run._fmt_optional(None, 0, "请求/小时") == "未测得"
        assert run._fmt_optional(1720.4, 0, "请求/小时") == "1720 请求/小时"


class TestProbeSequence:
    def test_stops_after_block_without_sleeping(self, monkeypatch) -> None:
        monkeypatch.setattr("time.sleep", lambda _s: pytest.fail("不得前台等待"))
        calls: list[tuple[str, str]] = []

        class Fake:
            def probe_once(self, code, start, stop, freq):
                calls.append((code, freq))
                return {
                    "parameters": {"ts_code": code, "start_date": start, "freq": freq},
                    "status": source.RATE_LIMIT,
                    "limit": {"count": 1, "unit": "小时", "requests_per_hour": 1.0},
                    "window_seconds": 3600.0,
                    "next_allowed_at_utc": "2026-11-18T12:00:00Z",
                }

        cases = [
            ("600000.SH", "2026-10-29 09:00:00", "2026-10-29 15:31:00", "5min"),
            ("600000.SH", "2016-01-05 09:00:00", "2016-01-05 15:31:00", "5min"),
            ("000001.SZ", "2019-07-01 09:00:00", "2019-07-01 15:31:00", "5min"),
        ]
        probes, blocked = run._probe_sequence(Fake(), cases, lambda _m: None)
        assert len(calls) == 1
        assert blocked is not None and blocked["limit"]["requests_per_hour"] == 1.0
        assert [item["status"] for item in probes] == [
            source.RATE_LIMIT,
            "skipped_after_block",
            "skipped_after_block",
        ]


class TestHttpClassification:
    def test_429_maps_to_rate_limit_block(self, monkeypatch) -> None:
        monkeypatch.setattr(
            source.MinuteSource,
            "_post",
            lambda _self, _params: (429, {"msg": "频率超限(1次/分钟)"}),
        )
        fetcher = source.MinuteSource(token="dummy", log=None)
        with pytest.raises(source.RateLimitBlocked) as info:
            fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        assert info.value.window_seconds == pytest.approx(60.0)

    def test_max_retries_zero_is_respected(self, monkeypatch) -> None:
        calls: list[int] = []

        def fake_post(_self, _params):
            calls.append(1)
            return 502, {"msg": "bad gateway"}

        monkeypatch.setattr(source.MinuteSource, "_post", fake_post)
        fetcher = source.MinuteSource(token="dummy", log=None, max_retries=0)
        result = fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        assert result.status == source.TRANSPORT
        assert len(calls) == 1
        assert fetcher.retries == 0


class TestChunkConsistency:
    """分块一致性：一次取 与 分块取 的行集合必须相同（D2 的协议必需项）。"""

    class _Src:
        def __init__(self, drop_last: bool = False, miss_one: bool = False) -> None:
            self.drop_last = drop_last
            self.miss_one = miss_one

        def fetch(self, code, start, end, freq=None):
            days = pd.date_range(start[:10], end[:10], freq="D").strftime("%Y-%m-%d").tolist()
            rows = []
            for index, day in enumerate(days):
                if self.miss_one and index == 2:
                    continue
                rows.append(
                    {
                        "ts_code": code,
                        "trade_time": f"{day} 09:35:00",
                        "open": 1.0,
                        "close": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "vol": 1.0,
                        "amount": 1.0,
                    }
                )
            if self.drop_last and len(days) > 5:
                rows = rows[:-1]
            return source.FetchResult(frame=pd.DataFrame(rows), status=source.OK)

    def test_equal_key_sets_pass(self) -> None:
        days = [f"202601{index:02d}" for index in range(5, 15)]
        out = run._chunk_consistency(
            self._Src(), "600000.SH", days, lambda _m: None, chunk_days=5, chunks=2
        )
        assert out["status"] == "ok"
        assert out["missing_in_parts"] == 0 and out["extra_in_parts"] == 0

    def test_unequal_key_sets_fail(self) -> None:
        days = [f"202601{index:02d}" for index in range(5, 15)]
        out = run._chunk_consistency(
            self._Src(miss_one=True), "600000.SH", days, lambda _m: None, chunk_days=5, chunks=2
        )
        assert out["status"] == "unequal"
        assert out["missing_in_parts"] == 1

    def test_short_history_is_not_a_pass(self) -> None:
        out = run._chunk_consistency(
            self._Src(), "600000.SH", ["20260105"], lambda _m: None, chunk_days=5, chunks=2
        )
        assert out["status"] == "insufficient_history"


class TestCoverageAdjudication:
    def test_accepted_exclusion_requires_suspension(self) -> None:
        records = [
            {"ts_code": "A", "trade_date": "20260105", "status": "ok", "rows": 48},
            {
                "ts_code": "B",
                "trade_date": "20260105",
                "status": "empty",
                "rows": 0,
                "suspended": True,
            },
        ]
        out = quality.adjudicate_coverage(records)
        assert out["status"] == "ready"
        assert out["accepted_rows"] == 2
        assert out["unresolved_count"] == 0

    def test_empty_without_suspension_is_unresolved(self) -> None:
        out = quality.adjudicate_coverage(
            [{"ts_code": "B", "trade_date": "20260105", "status": "empty", "rows": 0}]
        )
        assert out["status"] == "not_ready"
        assert out["unresolved_count"] == 1
        assert out["unresolved_sample"][0]["reason"] == "empty_without_suspension"

    def test_failed_and_conflict_are_unresolved(self) -> None:
        out = quality.adjudicate_coverage(
            [
                {"ts_code": "A", "trade_date": "20260105", "status": "transport_error"},
                {"ts_code": "B", "trade_date": "20260105", "status": "conflict", "rows": 0},
            ]
        )
        assert out["status"] == "not_ready"
        assert out["counts"]["unresolved_failed"] == 2


class FakeSource:
    """可编程的分钟源（离线）。"""

    def __init__(self, behaviour: str = "ok", rows: int = 49) -> None:
        self.behaviour = behaviour
        self.rows = rows
        self.calls: list[tuple[str, str, str]] = []
        self.requests = 0
        self.retries = 0
        self.last_status = None

    def fetch(self, code, start, end, freq=None):
        self.calls.append((code, start, end))
        self.requests += 1
        if self.behaviour == "block":
            raise source.RateLimitBlocked(
                "频率超限(1次/小时)",
                limit={
                    "count": 1,
                    "unit": "小时",
                    "window_seconds": 3600.0,
                    "requests_per_hour": 1.0,
                },
            )
        if self.behaviour == "fail":
            return source.FetchResult(
                frame=pd.DataFrame(columns=list(spec.MINUTE["fields"])),
                requests=self.requests,
                status=source.TRANSPORT,
                error="HTTP 502",
            )
        if self.behaviour == "empty":
            return source.FetchResult(
                frame=pd.DataFrame(columns=list(spec.MINUTE["fields"])),
                requests=self.requests,
                status=source.OK,
            )
        frames = []
        for day in pd.date_range(start[:10], end[:10], freq="D"):
            for k in range(1, 49):
                label = f"{day.strftime('%Y-%m-%d')} {_slot_time(k)}"
                frames.append(
                    {
                        "ts_code": code,
                        "trade_time": label,
                        "open": 10.0,
                        "close": 10.0,
                        "high": 10.01,
                        "low": 9.99,
                        "vol": 100.0,
                        "amount": 1000.0,
                    }
                )
        return source.FetchResult(
            frame=pd.DataFrame(frames), requests=self.requests, status=source.OK
        )


def _slot_time(k: int) -> str:
    from scripts.v2_p7.minute_repair_ml import sessions

    return sessions.slot_label(k)


def _cache_stub(days, codes):
    """测试用日线缓存：所有 (股, 日) 都在市、未停牌，量额与 48×100 股/48×1000 元一致。"""
    rows = {
        (str(code), str(day)): {
            "vol": 48.0,
            "amount": 48.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "is_suspended": 0,
            "tradable": 1,
        }
        for day in days
        for code in codes
    }
    return download.DailyCache(rows=rows, loaded_days={str(day) for day in days})


def _engine(
    tmp_path: Path, behaviour: str, *, codes=("600000.SH", "000001.SZ")
) -> download.DownloadEngine:
    campaign = tmp_path / "artifacts" / "campaign"
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    days = ["20260105", "20260106", "20260107"]
    eligible = pd.DataFrame(True, index=range(len(days)), columns=list(codes)).to_numpy()
    return download.DownloadEngine(
        campaign=campaign,
        work_dir=work,
        src=FakeSource(behaviour),  # type: ignore[arg-type]
        days=days,
        codes=list(codes),
        eligible=eligible,
        log=lambda _m: None,
    )


class TestDownloadEngine:
    def test_raw_first_and_ready_on_clean_run(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(download, "_daily_cache", _cache_stub)
        engine = _engine(tmp_path, "ok")
        rep = engine.run()
        assert rep.requests == 2
        assert rep.ok_requests == 2
        assert rep.raw_blocks >= 1  # 每批每年一个内容寻址块
        assert rep.coverage["status"] == "ready"
        assert rep.conflicts == 0
        blocks = list((tmp_path / "artifacts" / "campaign" / "raw").rglob("part-*.parquet"))
        assert blocks, "raw 必须先落盘"

    def test_failed_request_blocks_readiness(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(download, "_daily_cache", _cache_stub)
        rep = _engine(tmp_path, "fail").run()
        assert rep.failed_requests == rep.requests
        assert rep.coverage["status"] == "not_ready"
        assert rep.coverage["unresolved_count"] > 0

    def test_empty_without_suspension_blocks_readiness(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(download, "_daily_cache", _cache_stub)
        rep = _engine(tmp_path, "empty").run()
        assert rep.coverage["status"] == "not_ready"
        assert rep.coverage["unresolved_count"] > 0

    def test_block_stops_batch_and_records_next_attempt(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(download, "_daily_cache", _cache_stub)
        rep = _engine(tmp_path, "block").run()
        assert rep.blocked is not None
        assert rep.blocked["next_allowed_at_utc"]
        assert rep.coverage["status"] == "not_ready"

    def test_resume_skips_completed_batches(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(download, "_daily_cache", _cache_stub)
        first = _engine(tmp_path, "ok")
        first.run()
        second = _engine(tmp_path, "ok")
        second.resume = True
        rep = second.run()
        assert rep.resumed_batches >= 1
        assert second.src.calls == []  # 已完成批次不再请求
        assert rep.coverage["status"] == "ready"


class TestReadinessGate:
    def _publish_r0(self, tmp_path: Path, *, access: str, ready: bool) -> tuple[Path, Path]:
        base = tmp_path / "artifacts"
        campaign = artifacts.campaign_root("c1", base)
        artifacts.ensure_layout(campaign, base)
        run_dir = artifacts.create_run_dir(campaign, "r0_fake", base)
        work_inputs = run_dir / "work" / "inputs"
        work_inputs.mkdir(parents=True)
        artifacts.write_json_checked(work_inputs / "access.json", {"access_status": access})
        artifacts.write_json_checked(work_inputs / "download_plan.json", plan_stub())
        artifacts.write_json_checked(
            work_inputs / "specification.json", {"spec_sha256": "spec-sha"}
        )
        checks = {"access": {"ok": ready, "detail": "stub"}}
        artifacts.write_readiness(
            work_inputs, checks, data_status="ready" if ready else "access_not_ready"
        )
        published = artifacts.publish_artifacts(
            run_dir / "work", run_dir, "r0", ("inputs",), {"specification": "x"}
        )
        artifacts.write_json_checked(
            run_dir / "manifest.json",
            {"stage": "r0", "payload": {"artifacts": published}},
        )
        return campaign, run_dir

    def test_r1_refuses_when_access_ok_but_not_ready(self, tmp_path) -> None:
        campaign, _ = self._publish_r0(tmp_path, access="ok_partial_sample", ready=False)
        run_dir = artifacts.create_run_dir(campaign, "r1_fake", tmp_path / "artifacts")
        with pytest.raises(artifacts.ArtifactError, match="未就绪"):
            run.run_r1(campaign, run_dir, upstream="r0_fake", log=lambda _m: None)

    def test_r1_refuses_when_ready_missing(self, tmp_path) -> None:
        campaign, run_dir = self._publish_r0(tmp_path, access="ok_partial_sample", ready=True)
        (run_dir / "artifacts" / "inputs" / "readiness.json").unlink()
        # 删除后 manifest SHA 不再匹配 -> 先被发布件校验拦住
        with pytest.raises(artifacts.ArtifactError):
            artifacts.require_ready(campaign, "r0_fake", "inputs")


class TestR0ControlFlow:
    def test_offline_r0_never_crashes_on_missing_observations(self, tmp_path, monkeypatch) -> None:
        """无任何限频观测时，报告与摘要不得因 None 格式化崩溃（D1）。"""
        monkeypatch.setattr(run, "_download_plan", lambda *a, **k: plan_stub())
        monkeypatch.setattr(
            run, "_verify_baseline", lambda _root: {"problems": [], "manifest_sha256": "x"}
        )
        monkeypatch.setattr(run, "code_identity", identity_stub)
        # 隔离：不读真实日历、真实输入目录与真实样例文件
        monkeypatch.setattr(
            run.panel, "load_calendar", lambda _root: ["20260105", "20260106", "20260107"]
        )
        stub_input = tmp_path / "temp" / "stub.parquet"
        stub_input.parent.mkdir(parents=True, exist_ok=True)
        stub_input.write_bytes(b"stub")
        monkeypatch.setattr(run, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            run,
            "spec",
            type(
                "Spec",
                (),
                {
                    **{name: getattr(spec, name) for name in dir(spec) if not name.startswith("_")},
                    "INPUTS": [{"name": "stub", "relative": "temp/stub.parquet", "kind": "file"}],
                },
            ),
        )
        monkeypatch.setattr(
            run, "_local_sample_evidence", lambda: {"exists": False, "status": "missing"}
        )
        base = tmp_path / "artifacts"
        campaign = artifacts.campaign_root("c1", base)
        artifacts.ensure_layout(campaign, base)
        run_dir = artifacts.create_run_dir(campaign, "r0_offline", base)
        result = run.run_r0(campaign, run_dir, log=lambda _m: None, use_network=False)
        summary = result["summary"]
        assert summary["data_status"] == "access_not_ready"
        assert summary["engineering_status"] == "partial"
        assert summary["verification_scope"]["r0_throughput"] == "not_measured"
        readiness = artifacts.read_readiness(run_dir / "artifacts" / "inputs")
        assert readiness["data_status"] == "access_not_ready"
        assert readiness["all_checks_ok"] is False
        assert readiness["checks"]["throughput"]["ok"] is False
        assert "未测得" in readiness["checks"]["throughput"]["detail"]

    def test_partial_status_maps_to_data_not_ready(self) -> None:
        decision = report.decide(
            engineering_status="partial",
            data_status="ready",
            targets=None,
            pressure=None,
            recent_flag=None,
        )
        assert decision["research_decision"] == "data_not_ready"
        assert decision["resource_action"].startswith("pause_data_dependency")
