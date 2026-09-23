"""政策层在线暴露系数（`risk/terminal_loss/exposure_online.py`）单元测试。

只使用合成数据（不依赖真实配置与生产数据），锁定：

1. **配置单变量直读**：规范键解析、未知键报错、fixed 口径必须给校准段；
2. **provider 逐日判定**：窗口不足不判定、触发日给 λ、ES 覆盖外不判定、无持仓不判定；
3. **标签成熟机制**：未成熟日仍在面板（阈值可见）、成熟后按 valid 回填/剔除非法行；
4. **引擎接口**：默认关闭逐位一致（无政策源）、provider 现算、非法返回值报错、
   provider 必须实现 `multiplier_for`（拒绝第二套接口）、在线源下减仓单**必须真正执行**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lazybull.backtest.exposure_override import BacktestExposureOverrideMixin  # noqa: E402
from src.lazybull.backtest.exposure_trim import (  # noqa: E402
    TRIM_SELL_TYPE,
    BacktestExposureTrimMixin,
)
from src.lazybull.backtest.sell_execution import BacktestSellExecutionMixin  # noqa: E402
from src.lazybull.common.cost import CostModel  # noqa: E402
from src.lazybull.risk.terminal_loss import exposure_online as online  # noqa: E402
from src.lazybull.risk.terminal_loss.exposure_online import (  # noqa: E402
    DailyExposureProvider,
    OnlinePolicyConfig,
)
from src.lazybull.risk.terminal_loss.policy_sidecar import FoldModel  # noqa: E402

DAYS = [
    "20240102",
    "20240103",
    "20240104",
    "20240105",
    "20240108",
    "20240109",
    "20240110",
    "20240111",
]

D0_TS = pd.Timestamp("2024-01-02")
D1_TS = pd.Timestamp("2024-01-03")
DATE_TO_IDX = {D0_TS: 0, D1_TS: 1}


class _StubModel:
    """固定打分模型：对所有行返回同一 p_loss（便于构造触发/不触发）。"""

    feature_names = ["remaining_intervals", "cvar_95_20"]

    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


def _fold() -> FoldModel:
    return FoldModel(
        fold="2024H1",
        model_path=Path("unused"),
        val_end="20231231",
        es_start="20240102",
        es_end="20240109",
        h_max=1,
        loss_sigma_multiple=1.0,
        feature_names=["remaining_intervals", "cvar_95_20"],
    )


@pytest.fixture()
def provider(monkeypatch):
    """合成 provider：折模型 / 日历 / 面板 / 特征装配 / 标签全部打桩。"""
    monkeypatch.setattr(online, "load_fold_index", lambda *a, **k: [_fold()])
    monkeypatch.setattr(online, "load_trade_calendar", lambda *a, **k: list(DAYS))

    def fake_panels(self, fold, through=None):
        index = pd.Index(DAYS, name="trade_date")
        open_panel = pd.DataFrame({"600000.SH": 10.0, "000001.SZ": 10.0}, index=index)
        sigma_panel = pd.DataFrame({"600000.SH": 0.02, "000001.SZ": 0.02}, index=index)
        # 与真实实现一致：σ 长表随面板一起维护（供 _attach_inputs 使用）
        stacked = sigma_panel.stack().rename("sigma_daily_20").reset_index()
        stacked.columns = ["date", "ts_code", "sigma_daily_20"]
        stacked["date"] = stacked["date"].astype(str)
        stacked["ts_code"] = stacked["ts_code"].astype(str)
        self._sigma_long[fold.fold] = stacked
        return open_panel, sigma_panel, list(DAYS)

    monkeypatch.setattr(DailyExposureProvider, "_panels_for", fake_panels)

    def fake_attach(frame, data_root, feature_root, feature_columns, sigma_table, **kwargs):
        out = frame.copy()
        out["cvar_95_20"] = 0.01
        out["mkt_vol_20"] = [VOL_BY_DAY[d] for d in out["date"]]
        return out

    monkeypatch.setattr(online, "_attach_inputs", fake_attach)

    def fake_labels(open_panel, sigma_panel, config):
        frames = []
        for day in open_panel.index.astype(str):
            for code in ("600000.SH", "000001.SZ"):
                for horizon in range(1, config.h_max + 1):
                    status = INVALID.get((day, code), "valid")
                    frames.append(
                        {
                            "trade_date": day,
                            "ts_code": code,
                            "h": horizon,
                            "terminal_return": 0.01,
                            "loss_label": 0,
                            "label_status": status,
                        }
                    )
        return pd.DataFrame(frames)

    monkeypatch.setattr(online, "build_terminal_loss_labels", fake_labels)

    config = OnlinePolicyConfig.parse(
        "arm=combined,mode=rolling,window=2,min_window=2,regime_q=0.5,score_q=0.5,lambda=0.5"
    )
    instance = DailyExposureProvider(
        config,
        risk_root="unused",
        arm_suffix="_stub",
        data_root="data",
        model_loader=lambda path: _StubModel(0.9),
    )
    return instance


#: 逐日市场波动（当日值）；持仓打分统一 0.9（见 `_StubModel`）
VOL_BY_DAY = {
    "20240102": 0.10,
    "20240103": 0.20,
    "20240104": 0.30,
    "20240105": 0.90,
    "20240108": 0.95,
    "20240109": 0.10,
    "20240110": 0.90,
    "20240111": 0.91,
}
INVALID: dict = {}


def _holdings(shares_weight: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "weight": [shares_weight, shares_weight],
            "remaining_intervals": [1, 1],
        }
    )


# ------------------------------------------------------------------ 配置
def test_config_parse_and_describe_round_trip():
    config = OnlinePolicyConfig.parse(
        "arm=combined,mode=rolling,window=250,regime_q=0.6667,score_q=0.5,lambda=0.5"
    )
    assert config.describe().startswith("arm=combined,mode=rolling,window=250")
    assert config.arm_label == "E2-滚动250日"
    assert OnlinePolicyConfig.parse(config.describe()).describe() == config.describe()


def test_config_rejects_unknown_key_and_bad_values():
    with pytest.raises(ValueError, match="未知策略键"):
        OnlinePolicyConfig.parse("arm=combined,foo=1")
    with pytest.raises(ValueError, match="calib_start"):
        OnlinePolicyConfig.parse("arm=combined,mode=fixed")
    with pytest.raises(ValueError, match="lambda"):
        OnlinePolicyConfig.parse("arm=combined,lambda=1.5")
    with pytest.raises(ValueError, match="未知臂"):
        OnlinePolicyConfig.parse("arm=unknown")


# --------------------------------------------------------------- provider
def test_provider_skips_until_window_ready(provider):
    """窗口不足 ⇒ 不判定（返回 1.0 并计数）。"""
    assert provider.multiplier_for("20240102", _holdings()) == 1.0
    assert provider.stats["skipped_window_short"] == 1
    assert provider.stats["trigger_days"] == 0


def test_provider_triggers_after_window_fills(provider):
    """窗口够且当日高波动 + 高分 ⇒ 触发降暴露；波动低于 regime 阈值则不触发。"""
    assert provider.multiplier_for("20240102", _holdings()) == 1.0  # 窗口不足
    assert provider.multiplier_for("20240103", _holdings()) == 1.0  # 窗口不足
    # 20240104 起窗口可用（[d1, d2]），波动 0.30 ≥ 阈值 0.15 且分数 ≥ 层内阈值 ⇒ 触发
    assert provider.multiplier_for("20240104", _holdings()) == 0.5
    assert provider.multiplier_for("20240105", _holdings()) == 0.5
    # 20240109 波动 0.10 低于窗口（[0.90, 0.95]）阈值 ⇒ 不触发
    provider.multiplier_for("20240108", _holdings())
    assert provider.multiplier_for("20240109", _holdings()) == 1.0
    assert provider.stats["trigger_days"] == 3


def test_provider_skips_outside_fold_es_window(provider):
    """ES 覆盖区间之外不判定（与离线台账覆盖口径一致）。"""
    assert provider.multiplier_for("20231231", _holdings()) == 1.0
    assert provider.stats["skipped_out_of_coverage"] == 1


def test_provider_skips_without_holdings(provider):
    assert provider.multiplier_for("20240102", pd.DataFrame()) == 1.0
    assert provider.stats["skipped_no_holdings"] == 1


def test_provider_carries_forward_on_gap_days(provider):
    """缺口日（无持仓/窗口不足）顺延上一次判定（与离线系数表补缺口语义一致）。"""
    provider.multiplier_for("20240102", _holdings())
    provider.multiplier_for("20240103", _holdings())
    assert provider.multiplier_for("20240104", _holdings()) == 0.5  # 触发
    assert provider.multiplier_for("20240105", pd.DataFrame()) == 0.5  # 无持仓 ⇒ 顺延
    assert provider.stats["carried_forward_days"] == 3  # 前两天窗口不足 + 本次无持仓
    assert provider.stats["skipped_no_holdings"] == 1


def test_provider_matures_labels_and_drops_invalid(monkeypatch, provider):
    """成熟后按 valid 过滤：非法行从面板中剔除（影响后续阈值）。"""
    INVALID.clear()
    INVALID[("20240102", "000001.SZ")] = "endpoint_missing"
    provider.multiplier_for("20240102", _holdings())
    provider.multiplier_for("20240103", _holdings())
    provider.multiplier_for("20240104", _holdings())
    # h_max=1 ⇒ 20240102 的行在 20240104 成熟
    assert provider.stats["dropped_invalid_rows"] == 1
    assert provider.stats["matured_rows"] == 1
    ledger = provider._ledger()  # noqa: SLF001 - 测试内部口径
    matured = ledger[(ledger["date"] == "20240102") & (ledger["label_status"] == "valid")]
    assert set(matured["ts_code"]) == {"600000.SH"}
    # 未成熟日仍在面板（阈值可见）且标签为空
    pending = ledger[ledger["date"] == "20240103"]
    assert pending["label_status"].eq("pending").all()
    INVALID.clear()


def test_provider_coverage_start_suppresses_action_but_keeps_history(provider):
    """覆盖生效日只抑制**动作**，不抑制阈值历史（否则窗口冷启动、开头无法触发）。

    回归背景：早期实现把 ``coverage_start`` 写成“提前返回 1.0” ⇒ 面板无行 ⇒
    滚动窗口从生效日起重算 ⇒ 生效初期数十个交易日永不触发（实测使 2024-01 整月失效）。
    """
    provider.coverage_start = "20240105"  # 20240104 本应触发，但落在生效日之前
    assert provider.multiplier_for("20240102", _holdings()) == 1.0  # 窗口不足
    assert provider.multiplier_for("20240103", _holdings()) == 1.0  # 窗口不足
    # 20240104 判定成立（λ=0.5）但被抑制 ⇒ 返回 1.0
    assert provider.multiplier_for("20240104", _holdings()) == 1.0
    assert provider.stats["pre_coverage_judged_days"] == 1
    assert provider.stats["pre_coverage_trigger_days"] == 1
    assert provider.stats["trigger_days"] == 0
    # 被抑制日必须仍然入面板（阈值历史可见）
    ledger = provider._ledger()  # noqa: SLF001 - 测试内部口径
    assert "20240104" in set(ledger["date"])

    # 生效日当天：阈值窗口含**被抑制日**的历史（0.30 仍在窗口内）⇒ 正常触发
    provider.coverage_start = "20240105"
    assert provider.multiplier_for("20240105", _holdings()) == 0.5
    assert provider.stats["trigger_days"] == 1


def test_provider_requires_holdings_schema(provider):
    with pytest.raises(ValueError, match="持仓行缺少列"):
        provider.multiplier_for("20240102", pd.DataFrame({"ts_code": ["600000.SH"]}))


def test_stats_summary_contains_policy_identity(provider):
    summary = provider.stats_summary()
    assert summary["policy_config"].startswith("arm=combined")
    assert len(summary["policy_fingerprint"]) == 16
    assert summary["policy_book"] == "pre_exec"


def test_lambda_series_is_sorted_and_covers_consumed_days(provider):
    """λ 序列 = 引擎实际消费的全部日期（含缺口日顺延值），按日期升序。"""
    assert provider.lambda_series().empty  # 未评估 ⇒ 空表（列已定义）
    provider.multiplier_for("20240102", _holdings())
    provider.multiplier_for("20240103", _holdings())
    provider.multiplier_for("20240104", _holdings())
    assert provider.multiplier_for("20240105", pd.DataFrame()) == 0.5  # 缺口日：顺延
    series = provider.lambda_series()
    assert series.columns.tolist() == ["date", "multiplier"]
    assert series["date"].tolist() == ["20240102", "20240103", "20240104", "20240105"]
    assert series["multiplier"].tolist() == [1.0, 1.0, 0.5, 0.5]


# -------------------------------------------------------- 跨进程状态 round-trip
def _new_provider(
    arm_suffix: str = "_stub",
    *,
    coverage_mode: str = "es",
    pinned_fold=None,
) -> DailyExposureProvider:
    """按 fixture 相同打桩参数构造新实例（跨进程/实盘模式场景）。"""
    config = OnlinePolicyConfig.parse(
        "arm=combined,mode=rolling,window=2,min_window=2,regime_q=0.5,score_q=0.5,lambda=0.5"
    )
    return DailyExposureProvider(
        config,
        risk_root="unused",
        arm_suffix=arm_suffix,
        data_root="data",
        model_loader=lambda path: _StubModel(0.9),
        coverage_mode=coverage_mode,
        pinned_fold=pinned_fold,
    )


def test_provider_state_export_restore_round_trip(provider):
    """导出→恢复后：已评估日返回缓存值、面板逐值一致、新一日判定与连续运行相同。"""
    seq = ["20240102", "20240103", "20240104", "20240105"]
    first = [provider.multiplier_for(day, _holdings()) for day in seq]
    assert first == [1.0, 1.0, 0.5, 0.5]

    payload = provider.export_state()
    assert payload["fingerprint"] == provider.fingerprint

    clone = _new_provider()
    clone.restore_state(payload)
    # 恢复后重放：全部命中缓存（不重算、不重复污染面板）
    assert [clone.multiplier_for(day, _holdings()) for day in seq] == first
    assert clone.stats["cached_days"] == len(seq)
    # 面板逐值一致（滚动阈值完全由面板决定）
    pd.testing.assert_frame_equal(
        provider._ledger().reset_index(drop=True),  # noqa: SLF001 - 测试内部口径
        clone._ledger().reset_index(drop=True),  # noqa: SLF001 - 测试内部口径
    )
    # 恢复后继续运行：新一日判定必须与连续运行相同（阈值历史未丢失）
    assert clone.multiplier_for("20240108", _holdings()) == provider.multiplier_for(
        "20240108", _holdings()
    )


def test_provider_state_restore_rejects_fingerprint_mismatch(provider):
    """指纹不一致 ⇒ 直接报错（策略/模型源变化禁止静默续用）。"""
    payload = provider.export_state()
    clone = _new_provider(arm_suffix="_other")
    with pytest.raises(ValueError, match="指纹"):
        clone.restore_state(payload)


def test_provider_state_restore_rejects_unknown_version(provider):
    payload = provider.export_state()
    payload["version"] = 99
    clone = _new_provider()
    with pytest.raises(ValueError, match="版本"):
        clone.restore_state(payload)


# ------------------------------------------------------- 实盘模式 / 指定折（P1）
def test_serving_mode_extends_beyond_es_end(provider):
    """serving 模式在 ES 窗口之后继续判定（超出上界计入 post_es_serving_days）。"""
    serving = _new_provider(coverage_mode="serving")
    assert serving.multiplier_for("20240102", _holdings()) == 1.0  # 窗口不足
    assert serving.multiplier_for("20240103", _holdings()) == 1.0  # 窗口不足
    assert serving.multiplier_for("20240104", _holdings()) == 0.5  # 触发
    assert serving.multiplier_for("20240109", _holdings()) == 1.0  # 波动低，不触发
    # 20240110 超出折（es_end=20240109）：serving 继续判定（不再被 ES 上界拦下）
    assert serving.multiplier_for("20240110", _holdings()) == 0.5
    assert serving.stats["post_es_serving_days"] == 1
    assert serving.stats["skipped_out_of_coverage"] == 0


def test_serving_mode_still_requires_past_val_end(provider):
    """serving 仍要求 date > val_end（val_end 当天不可用，防前视）。"""
    serving = _new_provider(coverage_mode="serving")
    assert serving.multiplier_for("20231231", _holdings()) == 1.0  # == val_end
    assert serving.stats["skipped_out_of_coverage"] == 1
    assert serving.multiplier_for("20240102", _holdings()) == 1.0  # 已可用（但窗口不足）
    assert serving.stats["skipped_window_short"] == 1


def test_es_mode_unchanged_beyond_es_end(provider):
    """默认 es 模式行为不变：ES 窗口之外不判定（顺延）。"""
    es = _new_provider()
    assert es.multiplier_for("20240102", _holdings()) == 1.0
    assert es.multiplier_for("20240103", _holdings()) == 1.0
    assert es.multiplier_for("20240104", _holdings()) == 0.5
    assert es.multiplier_for("20240110", _holdings()) == 0.5  # 覆盖外 → 顺延
    assert es.stats["skipped_out_of_coverage"] == 1
    assert es.stats["post_es_serving_days"] == 0


def test_pin_fold_uses_designated_fold(provider):
    """pin 指定折：固定该折打分，配合 serving 可在 ES 窗口外继续判定。"""
    pinned = _new_provider(coverage_mode="serving", pinned_fold="2024H1")
    assert pinned.pinned_fold == "2024H1"
    assert pinned.multiplier_for("20240102", _holdings()) == 1.0  # 窗口不足
    assert pinned.multiplier_for("20240103", _holdings()) == 1.0
    assert pinned.multiplier_for("20240104", _holdings()) == 0.5
    assert pinned.multiplier_for("20240110", _holdings()) == 0.5  # 超出 ES 仍判定
    assert pinned.stats["post_es_serving_days"] == 1


def test_pin_fold_rejects_unknown_name(provider):
    with pytest.raises(ValueError, match="policy_fold"):
        _new_provider(pinned_fold="OOS99_209901")


def test_stats_summary_reports_mode_and_pin(provider):
    summary = provider.stats_summary()
    assert summary["policy_coverage_mode"] == "es"
    assert summary["policy_pinned_fold"] == ""
    pinned = _new_provider(coverage_mode="serving", pinned_fold="2024H1")
    assert pinned.stats_summary()["policy_coverage_mode"] == "serving"
    assert pinned.stats_summary()["policy_pinned_fold"] == "2024H1"


# ------------------------------------------------------------------ 引擎
class _PolicyStub:
    """引擎侧最小桩（只需给政策源用到的接口）。"""

    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        self.calls = 0

    def multiplier_for(self, date, holdings):
        self.calls += 1
        return self.multiplier


def _engine_stub():
    from src.lazybull.backtest.exposure_override import BacktestExposureOverrideMixin

    class _Engine(BacktestExposureOverrideMixin):
        def __init__(self):
            self.exposure_table = None
            self.exposure_policy_provider = None
            self.exposure_stats = None
            self.exposure_replenish_enabled = False
            self.exposure_trim_tolerance = 0.03
            self._exposure_missing_dates = {}
            self.positions = {"600000.SH": {"shares": 100, "buy_date": pd.Timestamp("2024-01-02")}}
            self.holding_period = 20
            self._trade_date_index = {
                pd.Timestamp("2024-01-02"): 0,
                pd.Timestamp("2024-01-03"): 1,
            }

        def _calculate_portfolio_value(self, date):
            return 1000.0

        def _position_market_value(self, date, stock):
            return 500.0

    return _Engine()


def test_engine_default_off_is_identity():
    engine = _engine_stub()
    assert engine._has_exposure_source() is False
    assert engine._get_exposure_multiplier(pd.Timestamp("2024-01-03")) == 1.0


def test_engine_uses_online_provider():
    engine = _engine_stub()
    stub = _PolicyStub(0.5)
    engine.set_exposure_policy(stub, verbose=False, replenish=True, trim_tolerance=0.06)
    assert engine._get_exposure_multiplier(pd.Timestamp("2024-01-03")) == 0.5
    assert stub.calls == 1
    assert engine.exposure_replenish_enabled is True
    assert engine.exposure_trim_tolerance == 0.06
    report = engine.get_exposure_report()
    assert report["enabled"] is True and report["source"] == "online"


def test_engine_rejects_callable_provider_and_bad_multiplier():
    engine = _engine_stub()
    with pytest.raises(TypeError, match="multiplier_for"):
        engine.set_exposure_policy(lambda date, holdings: 1.0, verbose=False)
    engine.set_exposure_policy(_PolicyStub(1.5), verbose=False)
    with pytest.raises(ValueError, match="非法系数"):
        engine._get_exposure_multiplier(pd.Timestamp("2024-01-03"))


def test_write_policy_lambda_series_merges_panel_and_meta(tmp_path):
    """λ 落盘：CSV = 逐日 λ + 日级面板列（左连），JSON = 统计 + λ 计数元数据。"""
    from src.lazybull.ml.walk_forward.runner import write_policy_lambda_series

    class _Stub:
        def lambda_series(self):
            return pd.DataFrame({"date": ["20240102", "20240103"], "multiplier": [0.5, 1.0]})

        def daily_series(self):
            return pd.DataFrame({"date": ["20240102"], "fold": ["fold_0"], "mkt_vol_20": [0.3]})

        def stats_summary(self):
            return {"judged_days": 1, "policy_fingerprint": "deadbeef"}

    summary_csv = tmp_path / "walk_forward_summary_x.csv"
    write_policy_lambda_series(_Stub(), summary_csv, "run_1")
    csv_path = tmp_path / "policy_lambda_run_1.csv"
    json_path = tmp_path / "policy_lambda_run_1.json"
    assert csv_path.exists() and json_path.exists()
    frame = pd.read_csv(csv_path, dtype={"date": str})
    assert frame["date"].tolist() == ["20240102", "20240103"]
    assert frame["multiplier"].tolist() == [0.5, 1.0]
    assert frame["fold"].iloc[0] == "fold_0" and pd.isna(frame["fold"].iloc[1])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["lambda_days"] == 2 and payload["lambda_trigger_days"] == 1
    assert payload["policy_fingerprint"] == "deadbeef"


def test_exported_table_provenance_round_trip(tmp_path):
    """系数表 meta 落盘 + 溯源读取（缺失时返回 None，不阻断）。"""
    from src.lazybull.risk.terminal_loss.exposure_online import (
        log_exported_table_provenance,
        write_exported_table_meta,
    )

    table = tmp_path / "exposure_table.csv"
    table.write_text("date,multiplier\n20240102,0.5\n", encoding="utf-8")
    assert log_exported_table_provenance(table) is None  # 缺 meta ⇒ 告警但仍返回 None
    config = OnlinePolicyConfig.parse(
        "arm=combined,mode=rolling,window=250,regime_q=0.6667,score_q=0.5,lambda=0.5"
    )
    write_exported_table_meta(
        table,
        policy_config=config,
        risk_root="data/walk_forward/terminal_risk_wf",
        arm_suffix="_d5_v6m_fscore",
        coverage_start="20240102",
    )
    meta = log_exported_table_provenance(table)
    assert meta is not None
    assert meta["policy_config"] == config.describe()
    assert meta["arm_suffix"] == "_d5_v6m_fscore"
    assert meta["coverage_start"] == "20240102"
    assert len(meta["policy_fingerprint"]) == 16


def test_engine_policy_takes_precedence_only_without_table():
    """表与 provider 互斥：set_exposure_policy 会清空表；表存在时不再调用 provider。"""
    engine = _engine_stub()
    stub = _PolicyStub(0.5)
    engine.set_exposure_table({"20240103": 0.8}, verbose=False)
    engine.set_exposure_policy(stub, verbose=False)
    assert engine.exposure_table is None
    assert engine._get_exposure_multiplier(pd.Timestamp("2024-01-03")) == 0.5


class _TrimChainEngine(
    BacktestExposureTrimMixin,
    BacktestExposureOverrideMixin,
    BacktestSellExecutionMixin,
):
    """承载「判定 → 减仓执行」最小卖出链的桩（在线政策源专用）。"""

    def __init__(self, positions, cash, prices):
        self.positions = positions
        self.current_capital = cash
        self.prices = prices  # {stock: (open, close)}
        self.trades = []
        self.cost_model = CostModel()
        self.sell_timing = "open"
        self.verbose = False
        self.enable_pending_order = False
        self.pending_condition_sells = {}
        self.pending_stop_loss_sells = {}
        self.pending_order_manager = None
        self.stop_loss_monitor = None
        self.exposure_table = None
        self.exposure_policy_provider = None
        self.exposure_stats = None
        self.exposure_replenish_enabled = False
        self.exposure_trim_tolerance = 0.03
        self._exposure_missing_dates = {}
        self.holding_period = 20
        self._trade_date_index = DATE_TO_IDX
        self._init_exposure_trim_state()

    def _get_trade_price(self, date, stock):
        return self.prices.get(stock, (None, None))[1]

    def _get_trade_price_open(self, date, stock):
        return self.prices.get(stock, (None, None))[0]

    def _get_pnl_price(self, date, stock):
        return self._get_trade_price(date, stock)

    def _get_pnl_price_open(self, date, stock):
        return self._get_trade_price_open(date, stock)

    def _position_market_value(self, date, stock):
        info = self.positions.get(stock)
        if not info:
            return 0.0
        price = self._get_trade_price(date, stock)
        if price is None or not (price > 0):
            return 0.0
        return float(info["shares"]) * float(price)

    def _calculate_portfolio_value(self, date):
        total = self.current_capital
        for stock, info in self.positions.items():
            price = self._get_trade_price(date, stock)
            if price is not None:
                total += float(info["shares"]) * float(price)
        return total


def _trim_position(shares: int, price: float) -> dict:
    return {
        "shares": shares,
        "buy_date": pd.Timestamp("2024-01-02"),
        "signal_date": pd.Timestamp("2024-01-02"),
        "buy_trade_price": price,
        "buy_pnl_price": price,
        "buy_cost_cash": shares * price,
    }


def test_online_provider_trim_orders_do_execute():
    """回归：在线政策源下减仓单必须真正执行。

    历史缺陷：执行侧残留 `exposure_table is None` 判定，在线 provider 路径（表为 None）
    下 pending 卖单被静默丢弃 —— 判定有、成交为零。
    """
    engine = _TrimChainEngine(
        {"600000.SH": _trim_position(1000, 10.0)}, 0.0, {"600000.SH": (10.0, 10.0)}
    )
    engine.set_exposure_policy(_PolicyStub(0.5), verbose=False)
    engine._queue_exposure_trim(D0_TS, [D0_TS, D1_TS], DATE_TO_IDX)
    assert engine.exposure_trim_stats["trim_orders"] == 1
    assert engine.pending_exposure_trims, "判定超配后必须产生 T+1 减仓卖单"
    engine._execute_pending_exposure_trims(D1_TS, [D0_TS, D1_TS], DATE_TO_IDX)
    assert engine.exposure_trim_stats["trim_orders_sold"] == 1
    assert engine.exposure_release_budget > 0.0
    assert any(t.get("sell_type") == TRIM_SELL_TYPE for t in engine.trades)
