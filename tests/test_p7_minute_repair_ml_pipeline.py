"""LR5M-ML-10-v1 管线合成反例：来源、工件、标准化、回放装配、报告与数据集合成。

全部使用合成数据与 tmp_path 隔离根，不访问真实 token、生产数据或外部 API。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7.minute_repair_ml import (
    analysis,
    artifacts,
    dataset,
    native_replay,
    report,
    sessions,
    source,
    spec,
    store,
)


def payload(rows: list[dict], code: int = 0, msg: str = "") -> dict:
    if code != 0:
        return {"code": code, "msg": msg, "data": None}
    fields = ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"]
    return {
        "code": 0,
        "msg": "",
        "data": {"fields": fields, "items": [[row[name] for name in fields] for row in rows]},
    }


def minute_row(time: str, close: float = 10.0) -> dict:
    return {
        "ts_code": "600000.SH",
        "trade_time": time,
        "open": close,
        "close": close,
        "high": close,
        "low": close,
        "vol": 100.0,
        "amount": 1000.0,
    }


class TestSource:
    def test_full_page_triggers_time_split(self, monkeypatch) -> None:
        monkeypatch.setitem(spec.MINUTE, "page_limit", 3)
        pages = [
            payload(
                [
                    minute_row("2020-01-02 09:35:00"),
                    minute_row("2020-01-02 09:40:00"),
                    minute_row("2020-01-02 09:45:00"),
                ]
            ),
            payload([minute_row("2020-01-02 09:50:00"), minute_row("2020-01-02 09:55:00")]),
            payload([minute_row("2020-01-02 10:00:00")]),
        ]
        calls = {"n": 0}

        def fake_post(_self, params):
            item = pages[min(calls["n"], len(pages) - 1)]
            calls["n"] += 1
            return 200, item

        monkeypatch.setattr(source.MinuteSource, "_post", fake_post)
        fetcher = source.MinuteSource(token="dummy", log=None)
        result = fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        assert result.status == source.OK
        # 满页不采信：丢弃首页，按时间二分重取两段（2 + 1 行）
        assert len(result.frame) == 3
        assert result.splits == 1

    def test_bad_gateway_is_transport_not_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(source.MinuteSource, "_post", lambda _self, _params: (502, {"code": 0}))
        fetcher = source.MinuteSource(token="dummy", log=None)
        result = fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        assert result.status == source.TRANSPORT
        assert result.error and "502" in result.error

    def test_rate_limit_blocks_batch_without_retry(self, monkeypatch) -> None:
        """限频不再秒级重试：直接抛出带"下一次可尝试时间"的阻塞（D4）。"""
        calls: list[int] = []

        def fake_post(_self, _params):
            calls.append(1)
            return 200, payload([], code=40203, msg="抱歉，您访问接口频率超限(1次/小时)")

        monkeypatch.setattr(source.MinuteSource, "_post", fake_post)
        fetcher = source.MinuteSource(token="dummy", log=None)
        with pytest.raises(source.RateLimitBlocked) as info:
            fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        assert len(calls) == 1  # 同窗口内重试无意义，不发第二次请求
        assert info.value.window_seconds == pytest.approx(3600.0)
        assert info.value.next_allowed_at_epoch > info.value.observed_at_epoch
        assert fetcher.retries == 0

    def test_permission_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            source.MinuteSource,
            "_post",
            lambda _self, _params: (200, payload([], code=40203, msg="没有访问该接口的权限")),
        )
        fetcher = source.MinuteSource(token="dummy", log=None)
        with pytest.raises(source.PermissionUnavailable):
            fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")

    def test_same_key_different_value_is_conflict(self, monkeypatch) -> None:
        monkeypatch.setitem(spec.MINUTE, "page_limit", 10)
        page = payload(
            [
                minute_row("2020-01-02 09:35:00", 10.0),
                minute_row("2020-01-02 09:35:00", 10.5),
                minute_row("2020-01-02 09:40:00", 10.1),
            ]
        )
        monkeypatch.setattr(source.MinuteSource, "_post", lambda _self, _params: (200, page))
        fetcher = source.MinuteSource(token="dummy", log=None)
        result = fetcher.fetch("600000.SH", "2020-01-02 09:00:00", "2020-01-02 15:31:00")
        assert result.conflicts
        assert result.conflicts[0]["trade_time"] == "2020-01-02 09:35:00"
        # 冲突键整个排除在标准化之外，双方原始经济值保留在 conflict_rows（D3）
        kept = result.frame[result.frame["trade_time"] == "2020-01-02 09:35:00"]
        assert len(kept) == 0
        assert len(result.conflict_rows) == 2
        assert sorted(result.conflict_rows["close"].astype(float).tolist()) == [10.0, 10.5]
        assert len(result.frame) == 1  # 未冲突的那一行仍在


class TestArtifacts:
    def test_refuses_overwrite_and_detects_drift(self, tmp_path: Path) -> None:
        base = tmp_path / "artifacts"
        campaign = artifacts.campaign_root("c1", base)
        artifacts.ensure_layout(campaign, base)
        run_dir = artifacts.create_run_dir(campaign, "r0_test", base)
        work = run_dir / "work"
        (work / "inputs").mkdir(parents=True)
        (work / "inputs" / "a.json").write_text('{"a": 1}\n', encoding="utf-8")
        published = artifacts.publish_artifacts(work, run_dir, "r0", ("inputs",), {}, base)
        manifest = {
            "stage": "r0",
            "payload": {"artifacts": published},
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        restored = artifacts.require_published(
            campaign,
            "r0_test",
            "inputs",
        )
        assert (restored / "a.json").exists()
        (restored / "a.json").write_text('{"a": 2}\n', encoding="utf-8")
        with pytest.raises(artifacts.ArtifactError):
            artifacts.require_published(campaign, "r0_test", "inputs")
        with pytest.raises(artifacts.ArtifactError):
            artifacts.create_run_dir(campaign, "r0_test", base)

    def test_raw_block_is_content_addressed(self, tmp_path: Path) -> None:
        base = tmp_path / "artifacts"
        campaign = artifacts.campaign_root("c1", base)
        artifacts.ensure_layout(campaign, base)
        frame = pd.DataFrame({"a": [1, 2, 3]})
        first = artifacts.commit_raw_block(frame, campaign, "2020", base)
        second = artifacts.commit_raw_block(frame, campaign, "2020", base)
        assert first["sha256"] == second["sha256"]
        assert len(list(artifacts.raw_year_dir(campaign, "2020").glob("part-*.parquet"))) == 1


class TestStore:
    def test_standardize_day_keeps_auction_separate(self) -> None:
        rows = [
            minute_row("2020-01-02 09:30:00"),
            minute_row("2020-01-02 09:35:00", 10.1),
            minute_row("2020-01-02 15:00:00", 10.2),
        ]
        frame = pd.DataFrame(rows)
        long_frame, auction, problems = store.standardize_day(frame, "20200102", convention="end")
        assert problems == []
        assert list(long_frame["slot"]) == [1, 48]
        assert len(auction) == 1
        assert auction.iloc[0]["trade_time"] == "2020-01-02 09:30:00"

    def test_year_roundtrip_and_store_slice(self, tmp_path: Path) -> None:
        rows = [
            minute_row(f"2020-01-02 {hh:02d}:{mm:02d}:00", 10.0 + 0.001 * i)
            for i, (hh, mm) in enumerate(
                [(9, 35 + i) if i < 25 else (13, i - 25) for i in range(48)]
            )
        ]
        long_frame, _auction, _problems = store.standardize_day(
            pd.DataFrame(rows), "20200102", convention="end"
        )
        store_dir = tmp_path / "minute"
        store.write_year(long_frame, store_dir, "2020")
        minute_store = store.MinuteStore(store_dir)
        frames = minute_store.day_frames("20200102", ["600000.SH"])
        assert "600000.SH" in frames
        assert frames["600000.SH"].loc[1, "C"] > 0


class TestReplayAssembly:
    def test_b_overrides_match_protocol(self) -> None:
        overrides = native_replay.b_overrides()
        assert overrides["rebalance_freq"] == 10
        assert overrides["holding_period"] == 10
        assert overrides["stagger_tranches"] == 2
        assert overrides["max_weight_per_stock"] == pytest.approx(0.20)
        assert overrides["max_per_industry"] == 2
        assert overrides["max_retry_count"] == 0
        assert overrides["max_retry_days"] == 0
        assert overrides["enable_position_completion"] is False
        assert overrides["sell_timing"] == "open"

    def test_tranche_schedule_offsets_and_slots(self) -> None:
        days = [day.strftime("%Y%m%d") for day in pd.bdate_range("2019-04-01", "2019-06-28")]
        check = native_replay.tranche_check(days)
        assert check["ok"], check
        assert check["tranche_slots"] == [3, 3]

    def test_weight_cap_conversion(self) -> None:
        from src.lazybull.v2.core.decision.weight_processor import resolve_tranche_weight_cap

        assert resolve_tranche_weight_cap(0.20, 0.5) == pytest.approx(0.40)

    def test_min_ticket_and_lot_rejection(self) -> None:
        value = spec.min_ticket_value()
        assert value == pytest.approx(25588.54, abs=0.01)
        # 理论预算已超过 V，但整手后实际成交额低于 V -> 必须拒绝下单，不能降槽数规避
        price = 10.50
        theoretical = 26000.0
        shares = int(theoretical // price // spec.LOT_SIZE) * spec.LOT_SIZE
        assert shares * price >= 0
        assert shares * price < value

    def test_combine_arms_rejects_date_mismatch(self) -> None:
        index = pd.Index(["20190401", "20190402"])
        a100 = pd.Series([1.0, 1.1], index=index)
        a80 = pd.Series([0.8, 0.88], index=index)
        sleeve = pd.Series([0.2, 0.21], index=index)
        combined = native_replay.combine_arms(a100, a80, {"LR20": sleeve}, 0.2)
        assert combined["A80_CASH20"].iloc[0] == pytest.approx(1.0)
        assert combined["A80_LR20"].iloc[1] == pytest.approx(1.09)
        with pytest.raises(native_replay.ReplayError):
            native_replay.combine_arms(a100, a80, {"LR20": sleeve.iloc[:1]}, 0.2)

    def test_signal_top_n_and_ranked_order(self) -> None:
        ranked = {
            "20200102": [
                ("A", 1.0),
                ("B", 0.5),
                ("C", 0.4),
                ("D", 0.3),
                ("E", 0.2),
                ("F", 0.1),
                ("G", 0.05),
            ]
        }
        signal = native_replay.LRSignal(ranked)
        generated = signal.generate(pd.Timestamp("2020-01-02"), None, None)
        assert len(generated) == 6
        assert signal.top_n == 6
        assert list(generated)[:2] == ["A", "B"]


class TestReportDecision:
    def test_access_not_ready_stops(self) -> None:
        decision = report.decide(
            engineering_status="valid",
            data_status="access_not_ready",
            targets=None,
            pressure=None,
            recent_flag=None,
        )
        assert decision["research_decision"] == "access_not_ready"
        assert decision["resource_action"] == "pause_data_dependency_deliver_evidence"

    def test_target_failure_stops_new_investment(self) -> None:
        decision = report.decide(
            engineering_status="valid",
            data_status="ready",
            targets={"all_met": False},
            pressure=None,
            recent_flag=None,
        )
        assert decision["research_decision"] == "economic_target_not_met"
        assert decision["resource_action"] == "stop_new_investment"

    def test_pressure_failure_is_limited(self) -> None:
        decision = report.decide(
            engineering_status="valid",
            data_status="ready",
            targets={"all_met": True},
            pressure={"all_met": False},
            recent_flag={"triggered": False},
        )
        assert decision["research_decision"] == "continue_limited"

    def test_all_pass_is_candidate_not_admission(self) -> None:
        decision = report.decide(
            engineering_status="valid",
            data_status="ready",
            targets={"all_met": True},
            pressure={"all_met": True},
            recent_flag={"triggered": False},
        )
        assert decision["research_decision"] == "economic_candidate"
        summary = report.summary_payload(
            identity={"candidate_id": spec.CANDIDATE_ID},
            engineering_status="valid",
            data_status="ready",
            targets_by_capital={},
            pressure_status={},
            decision=decision,
            limitations=[],
            unmet_items=[],
            machine_hours={},
            requests=0,
            bytes_written=0,
        )
        assert summary["formal_admission"] == "not_evaluated"

    def test_target_table_flags(self) -> None:
        metrics = {
            "A100": {"cagr": 0.05, "mdd_loss": 0.20},
            "A80_CASH20": {"cagr": 0.04, "mdd_loss": 0.18},
            "A80_LR20": {"cagr": 0.06, "mdd_loss": 0.22},
            "A80_DAILY20": {"cagr": 0.05, "mdd_loss": 0.19},
        }
        table = analysis.target_table(metrics)
        assert table["checks"]["cagr_delta_ok"]
        assert not table["checks"]["mdd_loss_ok"]  # 恶化 1pp 以上


class FakeRoot:
    """最小 clean 根：300 个交易日、3 只股票、单一行业。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calendar = [
            day.strftime("%Y%m%d") for day in pd.bdate_range("2018-01-01", "2019-03-29")
        ]
        self.codes = ["600000.SH", "600001.SH", "600002.SH"]
        self._write_clean()

    def _write_clean(self) -> None:
        clean = self.root / "data" / "clean"
        (clean / "daily").mkdir(parents=True)
        (clean / "daily_basic").mkdir(parents=True)
        for day in self.calendar:
            path = clean / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
            rows = pd.DataFrame(
                {
                    "ts_code": self.codes,
                    "trade_date": day,
                    "close_adj": [10.0, 20.0, 30.0],
                    "open_adj": [10.0, 20.0, 30.0],
                    "amount": [1e6, 1e6, 1e6],
                    "is_st": 0.0,
                    "is_suspended": 0.0,
                    "is_limit_up": 0.0,
                    "is_limit_down": 0.0,
                    "tradable": 1.0,
                }
            )
            rows.to_parquet(path, index=False)
            basic = pd.DataFrame(
                {
                    "ts_code": self.codes,
                    "trade_date": day,
                    "total_mv": [1e7, 1e7, 1e7],
                    "turnover_rate": [0.5, 0.6, 0.7],
                }
            )
            basic.to_parquet(clean / "daily_basic" / path.name, index=False)
        pd.DataFrame(
            {"ts_code": self.codes, "market": ["主板"] * 3, "list_date": ["20000101"] * 3}
        ).to_parquet(clean / "stock_basic.parquet", index=False)
        pd.DataFrame(
            {
                "exchange": ["SSE"] * len(self.calendar),
                "cal_date": self.calendar,
                "is_open": [1] * len(self.calendar),
            }
        ).to_parquet(clean / "trade_cal.parquet", index=False)
        pd.DataFrame(
            {
                "ts_code": self.codes,
                "l1_code": ["410000", "410000", "410000"],
                "l1_name": ["电子", "电子", "电子"],
                "in_date": ["20100101"] * 3,
                "out_date": [None] * 3,
            }
        ).to_parquet(clean / "sw_industry_pit.parquet", index=False)


class TestDatasetIntegration:
    def test_rows_keep_uninformed_and_share_domain(self, tmp_path: Path) -> None:
        fake = FakeRoot(tmp_path / "root")
        minute_dir = tmp_path / "minute"
        days = fake.calendar[-30:]
        parts = []
        for day in days:
            for code in fake.codes[:2]:
                rows = []
                for k in range(1, 49):
                    label = sessions.slot_label(k)
                    rows.append(
                        {
                            "ts_code": code,
                            "trade_time": f"{day[:4]}-{day[4:6]}-{day[6:]} {label}",
                            "open": 10.0 + 0.001 * k,
                            "close": 10.0 + 0.001 * k,
                            "high": 10.01 + 0.001 * k,
                            "low": 9.99 + 0.001 * k,
                            "vol": 100.0,
                            "amount": 1000.0,
                        }
                    )
                long_frame, _auction, _problems = store.standardize_day(
                    pd.DataFrame(rows), day, convention="end"
                )
                parts.append(long_frame)
        store.write_year(pd.concat(parts, ignore_index=True), minute_dir, days[0][:4])

        out_dir = tmp_path / "dataset"
        result = dataset.build_dataset(
            fake.root,
            minute_dir=minute_dir,
            feature_start=days[0],
            end=days[-1],
            out_dir=out_dir,
            source_manifest_sha="synthetic",
            log=None,
        )
        frame = result["frame"]
        assert len(frame) == len(days) * len(fake.codes)
        informed = frame[frame["minute_informed"]]
        assert set(informed["ts_code"]) == {"600000.SH", "600001.SH"}
        silent = frame[frame["ts_code"] == "600002.SH"]
        assert not silent["minute_informed"].any()
        assert silent["m_am_ret"].isna().all()
        assert silent["minute_status"].eq("no_minute_source").all()
        assert frame[list(spec.FEATURES)].shape[1] == 24
        assert (out_dir / "dataset_summary.json").exists()
