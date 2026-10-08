"""P2a-T7 等价测试：v2/hosts/backtest + runs 出口 + 验收门工具。

验证目标（T0 规划 §4 T7 任务卡）：
- A 组：runtime 等价（TradingConfig 映射逐字段一致 / 引擎构造 kwargs 逐项一致 /
  persistent 信号路径 / reporter 等价）；
- B 组：replay 装配等价（R9 对照清单测试化——合成 storage/loader/日历/基础表 +
  引擎工厂/信号工厂捕获 stub，新旧两侧装配行为一致，D6 有意差异锚定）；
- C 组：replay exposure 五参 fail-fast（D6）；
- D 组：runs_writer（合成 2 折 ⇒ load_runs_batch 硬校验通过 + 四类负例 +
  lot_id 重建与 fifo_violations 落 _meta.json + 日期规范化）；
- E 组：input_manifest（3 条反例：代码态不在清单 / 内容变即拒 / 行情日历基础表变即拒）
  + 窗口内容抽查 1 正 1 负；
- F 组：compare_replay_vs_b0（门 3/4/5 各 1 正 1 负）；
- G 组：依赖方向 AST 扫描（旧侧依赖白名单 + 退役符号 0 残留）；
- H 组：禁止真实配置读取永久回归（范式同 T3~T6）。

全部输入为合成 fixture，禁止依赖真实配置与真实生产数据。
"""

import ast
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

import src.lazybull.backtest.reporter as old_reporter
import src.lazybull.common.backtest_runtime as old_runtime
import src.lazybull.common.signal_factory as old_factory
import src.lazybull.ml.walk_forward.backtest as old_replay
import src.lazybull.signals.ml_signal as old_ml_signal
import src.lazybull.v2.core.signal.factory as new_factory
import src.lazybull.v2.core.signal.ml_signal as new_ml_signal
import src.lazybull.v2.hosts.backtest.replay as new_replay
import src.lazybull.v2.hosts.backtest.reporter as new_reporter
import src.lazybull.v2.hosts.backtest.runtime as new_runtime
from scripts.v2_p2a import compare_replay_vs_b0 as compare_tool
from scripts.v2_p2a import input_manifest
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.evidence.runs_schema import (
    CHAIN_NAV_CONTRACT_COLS,
    DAILY_CONTRACT_COLS,
    SNAP_CONTRACT_COLS,
    TRADES_CONTRACT_COLS,
)
from src.lazybull.v2.hosts.backtest import runs_writer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_V2_HOSTS_DIR = _REPO_ROOT / "src" / "lazybull" / "v2" / "hosts"


# ══════════════════ A 组：runtime 等价 ══════════════════


def _args_stub(**overrides) -> SimpleNamespace:
    """walk_forward 参数 stub（只放 build_walk_forward_trading_config 读取的键）。"""
    base = dict(
        bt_rebalance_freq=20,
        label_column="neu_y_ret_20",
        bt_top_n=20,
        stagger_tranches=2,
        bt_max_per_industry=None,
        bt_max_weight_per_stock=0.15,
        enable_early_rebalance_on_empty=True,
        bt_exclude_st=True,
        bt_min_list_days=365,
        bt_stop_loss_enabled=False,
        bt_stop_loss_drawdown_pct=30.0,
        bt_stop_loss_consecutive_limit_down=2,
        position_sizing="half_kelly",
        kelly_vol_window=60,
        kelly_max_leverage=0.2,
        min_buy_value_ratio=0.2,
        bt_initial_capital=1000000.0,
        bt_sell_timing="open",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"bt_rebalance_freq": None, "label_column": "y_ret_10", "bt_top_n": 30},
        {
            "bt_stop_loss_enabled": True,
            "bt_stop_loss_drawdown_pct": 25.0,
            "position_sizing": "equal",
        },
        {"stagger_tranches": 4, "bt_max_per_industry": 3, "bt_sell_timing": "close"},
    ],
    ids=["base", "freq_from_label", "stop_loss_on", "stagger4_close"],
)
def test_build_trading_config_equivalence(overrides):
    """新旧 build_walk_forward_trading_config 对同一 args stub 输出逐字段一致。"""
    args = _args_stub(**overrides)
    cfg_old = old_runtime.build_walk_forward_trading_config(args, model_version=24008)
    cfg_new = new_runtime.build_walk_forward_trading_config(args, model_version=24008)
    assert dataclasses.asdict(cfg_old) == dataclasses.asdict(cfg_new)


class _FakeCostModel:
    """合成成本模型（避免 CostModel() 读取真实配置单例）。"""

    def __init__(self, tag: str = "syn"):
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, _FakeCostModel) and other.tag == self.tag


class _EngineCapture:
    """引擎类捕获 stub：记录构造 kwargs。"""

    instances: List["_EngineCapture"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _EngineCapture.instances.append(self)


def _norm_kwarg(value):
    """kwargs 规范化比对（dataclass → dict，其余原样）。"""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _kwargs_equal(a, b) -> bool:
    """kwargs 值相等判定（DataFrame 用 .equals，dict 递归，其余 ==）。"""
    a, b = _norm_kwarg(a), _norm_kwarg(b)
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        return a.equals(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_kwargs_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def test_create_engine_kwargs_equivalence(monkeypatch):
    """新旧 create_backtest_engine_from_config 的引擎构造 kwargs 逐项一致。

    引擎类身份差异除外：断言各自指向本侧绑定的 BacktestEngineML 捕获 stub。
    """
    monkeypatch.setattr(old_runtime, "BacktestEngineML", _EngineCapture)
    monkeypatch.setattr(new_runtime, "BacktestEngineML", _EngineCapture)
    monkeypatch.setattr(old_runtime, "CostModel", _FakeCostModel)
    monkeypatch.setattr(new_runtime, "CostModel", _FakeCostModel)

    cfg_old = old_runtime.build_walk_forward_trading_config(_args_stub(), model_version=24008)
    cfg_new = new_runtime.build_walk_forward_trading_config(_args_stub(), model_version=24008)
    stock_basic = pd.DataFrame({"ts_code": ["A"]})
    features = {"20240102": pd.DataFrame({"ts_code": ["A"]})}
    sentinel_signal = object()
    sentinel_universe = object()

    _EngineCapture.instances.clear()
    old_runtime.create_backtest_engine_from_config(
        trading_config=cfg_old,
        universe=sentinel_universe,
        signal=sentinel_signal,
        features_by_date=features,
        stock_basic=stock_basic,
        data_storage="storage_stub",
    )
    new_runtime.create_backtest_engine_from_config(
        trading_config=cfg_new,
        universe=sentinel_universe,
        signal=sentinel_signal,
        features_by_date=features,
        stock_basic=stock_basic,
        data_storage="storage_stub",
    )
    assert len(_EngineCapture.instances) == 2
    kw_old, kw_new = (i.kwargs for i in _EngineCapture.instances)
    assert set(kw_old) == set(kw_new)
    for key in kw_old:
        assert _kwargs_equal(kw_old[key], kw_new[key]), f"kwargs[{key}] 不一致"
    # 成本模型缺省时各自实例化本侧 CostModel（此处为同一 stub 类）
    assert isinstance(kw_old["cost_model"], _FakeCostModel)
    assert isinstance(kw_new["cost_model"], _FakeCostModel)


def test_create_or_reuse_signal_persistent_path(monkeypatch):
    """persistent 路径：两侧均更新 top_n + 版本切换且不调用工厂；模型根解析一致。"""
    for module in (old_runtime, new_runtime):
        monkeypatch.setattr(module, "get_stock_selection_models_root", lambda root=None: "syn_root")
        monkeypatch.setattr(
            module,
            "create_signal",
            lambda *a, **k: pytest.fail("persistent 路径不得调用 create_signal"),
        )

    class _PersistentStub:
        def __init__(self):
            self.top_n = None
            self.version_calls = []

        def update_model_version(self, version):
            self.version_calls.append(version)

    cfg_old = old_runtime.build_walk_forward_trading_config(_args_stub(), model_version=24008)
    cfg_new = new_runtime.build_walk_forward_trading_config(_args_stub(), model_version=24008)
    sig_old, sig_new = _PersistentStub(), _PersistentStub()
    out_old = old_runtime.create_or_reuse_signal(
        cfg_old, data_root="syn", persistent_signal=sig_old
    )
    out_new = new_runtime.create_or_reuse_signal(
        cfg_new, data_root="syn", persistent_signal=sig_new
    )
    assert out_old is sig_old and out_new is sig_new
    assert sig_old.top_n == sig_new.top_n == 20
    assert sig_old.version_calls == sig_new.version_calls == [24008]


def test_reporter_equivalence(monkeypatch, tmp_path):
    """Reporter 新旧等价：同一合成输入 ⇒ stats 逐值一致 + 产物文件字节一致。"""
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    monkeypatch.setattr(old_reporter, "get_reports_root", lambda *a, **k: str(old_dir))
    monkeypatch.setattr(new_reporter, "get_reports_root", lambda *a, **k: str(new_dir))

    nav_curve = pd.DataFrame(
        {
            "date": list(pd.bdate_range("2024-01-02", periods=5)),
            "portfolio_value": [100.0, 101.0, 100.5, 102.0, 103.0],
            "capital": [40.0] * 5,
            "market_value": [60.0, 61.0, 60.5, 62.0, 63.0],
            "nav": [1.0, 1.01, 1.005, 1.02, 1.03],
            "return": [0.0, 0.01, 0.005, 0.02, 0.03],
        }
    )
    trades = pd.DataFrame(
        {
            "date": list(pd.bdate_range("2024-01-02", periods=2)),
            "stock": ["A", "A"],
            "action": ["buy", "sell"],
            "price": [10.0, 11.0],
            "shares": [100, 100],
            "amount": [1000.0, 1100.0],
            "cost": [5.0, 5.5],
            "buy_price": [np.nan, 10.0],
            "profit_amount": [np.nan, 94.5],
            "profit_pct": [np.nan, 0.0945],
        }
    )
    stats_old = old_reporter.Reporter().generate_report(nav_curve, trades, "syn")
    stats_new = new_reporter.Reporter().generate_report(nav_curve, trades, "syn")
    assert stats_old == stats_new
    for name in ("syn_nav.csv", "syn_trades.csv", "syn_stats.txt"):
        assert (old_dir / name).read_bytes() == (new_dir / name).read_bytes()


# ══════════════════ B 组：replay 装配等价（R9 测试化） ══════════════════

_TRADE_DATES = ["20240102", "20240103", "20240104"]


def _synth_trade_cal() -> pd.DataFrame:
    return pd.DataFrame({"cal_date": _TRADE_DATES, "is_open": [1, 1, 1]})


def _synth_stock_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "name": ["平安A", "浦发"],
            "list_date": ["19910403", "19991110"],
            "market": ["主板", "主板"],
        }
    )


class _SynthStorage:
    """合成 Storage：20240103 无特征分区（验证过滤路径），其余日返回合成特征。"""

    def __init__(self):
        self.cs_calls: List[str] = []

    def load_cs_train_day(self, trade_date: str) -> Optional[pd.DataFrame]:
        self.cs_calls.append(trade_date)
        if trade_date == "20240103":
            return None
        return pd.DataFrame({"ts_code": ["000001.SZ"], "f1": [1.0]})


class _SynthLoader:
    """合成 DataLoader：返回含全部 desired 列的行情小帧。"""

    def load_clean_daily(self, start: str, end: str) -> pd.DataFrame:
        rows = []
        for date in _TRADE_DATES:
            for code, base in (("000001.SZ", 10.0), ("600000.SH", 20.0)):
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": date,
                        "close": base,
                        "close_adj": base,
                        "open": base,
                        "open_adj": base,
                        "is_suspended": False,
                        "is_limit_up": False,
                        "is_limit_down": False,
                        "vol": 1000.0,
                        "pct_chg": 0.1,
                        "is_st": False,
                        "list_days": 5000,
                        "tradable": True,
                    }
                )
        return pd.DataFrame(rows)


def _canned_nav_curve() -> pd.DataFrame:
    """罐头净值曲线（引擎 stub 的 run 返回；两侧共用同一实例源）。"""
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d in _TRADE_DATES],
            "portfolio_value": [1000000.0, 1005000.0, 1010000.0],
            "capital": [400000.0, 400000.0, 410000.0],
            "market_value": [600000.0, 605000.0, 600000.0],
        }
    )
    df["nav"] = df["portfolio_value"] / 1000000.0
    df["return"] = df["nav"] - 1.0
    return df


class _ReplayEngineStub:
    """replay 用引擎 stub（新旧共用基类；旧侧额外挂 set_exposure_table 捕获）。"""

    instances: List["_ReplayEngineStub"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.record_holdings_snapshot = False
        self.run_kwargs: Dict[str, object] = {}
        _ReplayEngineStub.instances.append(self)

    def run(self, *, start_date, end_date, trading_dates, price_data):
        self.run_kwargs = {
            "start_date": start_date,
            "end_date": end_date,
            "trading_dates": trading_dates,
            "price_data": price_data,
        }
        return _canned_nav_curve()

    def get_trades(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_execution_attribution(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_holdings_snapshot(self) -> pd.DataFrame:
        return pd.DataFrame()


class _OldReplayEngineStub(_ReplayEngineStub):
    """旧侧引擎 stub：多一个 set_exposure_table 捕获（旧装配会调用它）。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposure_table_calls: List[Dict[str, object]] = []

    def set_exposure_table(self, table, **kwargs):
        self.exposure_table_calls.append({"table": table, **kwargs})


def _replay_call_kwargs() -> Dict[str, object]:
    return dict(
        model_version=24008,
        bt_start="20240102",
        bt_end="20240104",
        label_column="neu_y_ret_20",
        bt_top_n=20,
        bt_rebalance_freq=20,
        data_root="syn_root",
        bt_exclude_st=True,
        bt_min_list_days=365,
        bt_sell_timing="open",
        bt_max_weight_per_stock=0.15,
        position_sizing="half_kelly",
        kelly_vol_window=60,
        kelly_max_leverage=0.2,
        stagger_tranches=2,
        enable_early_rebalance_on_empty=True,
        initial_capital=1000000.0,
        split_num=0,
        stock_domain="main",
    )


def _install_replay_stubs(monkeypatch, old_engine_cls, new_engine_cls):
    """新旧 replay 模块命名空间打同一组捕获 stub；返回捕获记录。"""
    captured = {"signal_calls": [], "universe_calls": []}
    sentinel_signal = object()

    def _signal_factory(trading_config, *, data_root=None, persistent_signal=None, verbose=False):
        captured["signal_calls"].append(
            {
                "trading_config": dataclasses.asdict(trading_config),
                "data_root": data_root,
                "persistent_signal": persistent_signal,
                "verbose": verbose,
            }
        )
        return sentinel_signal

    class _UniverseCapture:
        def __init__(self, **kwargs):
            captured["universe_calls"].append(kwargs)

    for module, engine_cls in ((old_replay, old_engine_cls), (new_replay, new_engine_cls)):
        monkeypatch.setattr(module, "get_data_root", lambda: "syn_root")
        monkeypatch.setattr(module, "create_backtest_engine_from_config", engine_cls)
        monkeypatch.setattr(module, "create_or_reuse_signal", _signal_factory)
        monkeypatch.setattr(module, "BasicUniverse", _UniverseCapture)
        monkeypatch.setattr(module, "domain_market_whitelist", lambda domain: ["主板"])
    captured["signal"] = sentinel_signal
    return captured


def test_replay_assembly_equivalence(monkeypatch):
    """新旧 run_oos_backtest 装配等价（R9 对照清单测试化）。

    D6 有意差异锚定：旧侧调 set_exposure_table(None, ...)，新侧无该调用；
    其余（引擎 kwargs / 信号工厂入参 / universe 入参 / features 键集 /
    price_data 列集 / trade_dates / metrics 逐值）两侧一致。
    """
    captured = _install_replay_stubs(monkeypatch, _OldReplayEngineStub, _ReplayEngineStub)
    _ReplayEngineStub.instances.clear()

    storage_old, storage_new = _SynthStorage(), _SynthStorage()
    kwargs = _replay_call_kwargs()
    out_old = old_replay.run_oos_backtest(
        storage=storage_old,
        loader=_SynthLoader(),
        trade_cal=_synth_trade_cal(),
        stock_basic=_synth_stock_basic(),
        **kwargs,
    )
    out_new = new_replay.run_oos_backtest(
        storage=storage_new,
        loader=_SynthLoader(),
        trade_cal=_synth_trade_cal(),
        stock_basic=_synth_stock_basic(),
        **kwargs,
    )

    assert len(_ReplayEngineStub.instances) == 2
    eng_old, eng_new = _ReplayEngineStub.instances
    # 引擎工厂 kwargs 逐项一致（signal/universe 为两侧共享哨兵）
    assert set(eng_old.kwargs) == set(eng_new.kwargs)
    for key in eng_old.kwargs:
        v_old, v_new = eng_old.kwargs[key], eng_new.kwargs[key]
        if key == "trading_config":
            assert dataclasses.asdict(v_old) == dataclasses.asdict(v_new)
        elif key == "signal":
            assert v_old is v_new  # 信号为两侧共享哨兵（create_or_reuse_signal 同一 stub）
        elif key == "universe":
            # universe 由各侧 replay 内部构造（类型同名 + 入参一致即可）
            assert type(v_old).__name__ == type(v_new).__name__
        elif key == "data_storage":
            # 两侧各持合成 storage 实例（调用计数独立锚定），类型一致即可
            assert type(v_old).__name__ == type(v_new).__name__
        else:
            assert _kwargs_equal(v_old, v_new), f"引擎 kwargs[{key}] 不一致"
    # features_by_date 键集一致（20240103 无分区被过滤）
    assert (
        sorted(eng_old.kwargs["features_by_date"])
        == sorted(eng_new.kwargs["features_by_date"])
        == ["20240102", "20240104"]
    )
    assert storage_old.cs_calls == storage_new.cs_calls == _TRADE_DATES
    # price_data 列集与 trade_dates 一致
    assert list(eng_old.run_kwargs["price_data"].columns) == list(
        eng_new.run_kwargs["price_data"].columns
    )
    assert eng_old.run_kwargs["trading_dates"] == eng_new.run_kwargs["trading_dates"]
    # metrics 逐值一致（同一罐头净值曲线）
    assert {k: v for k, v in out_old.items() if not k.startswith("_")} == {
        k: v for k, v in out_new.items() if not k.startswith("_")
    }
    assert out_old["bt_total_return"] == pytest.approx(0.01)
    # D6 有意差异：旧侧 set_exposure_table(None) 被调用一次；新侧 stub 无该方法
    # （新侧若调用将 AttributeError——测试天然失败），此处显式锚定两侧分歧。
    assert eng_old.exposure_table_calls == [
        {
            "table": None,
            "verbose": False,
            "replenish": False,
            "trim_tolerance": None,
            "budget_discount_replenish": False,
        }
    ]
    assert isinstance(eng_new, _ReplayEngineStub) and not isinstance(eng_new, _OldReplayEngineStub)
    # record_holdings_snapshot 两侧都设置（快照是 runs 产物数据源）
    assert eng_old.record_holdings_snapshot is True
    assert eng_new.record_holdings_snapshot is True
    # 信号工厂 / universe 入参两侧一致
    assert captured["signal_calls"][0] == captured["signal_calls"][1]
    assert _kwargs_equal(captured["universe_calls"][0], captured["universe_calls"][1])
    assert captured["universe_calls"][0]["markets"] == ["主板"]


# ══════════════════ C 组：replay exposure fail-fast（D6） ══════════════════

_FAIL_FAST_KWARGS = {
    "exposure_table": {"20240102": 0.5},
    "exposure_policy": object(),
    "exposure_replenish": True,
    "exposure_trim_tolerance": 0.06,
    "exposure_budget_discount_replenish": True,
}


@pytest.mark.parametrize("param", sorted(_FAIL_FAST_KWARGS))
def test_replay_exposure_param_fail_fast(param):
    """5 个 exposure 参数各给非默认值 ⇒ ValueError（消息锚定 D6 退役）。"""
    with pytest.raises(ValueError, match="exposure 政策族已退役摘除（P2a D6）"):
        new_replay.run_oos_backtest(
            storage=None,
            loader=None,
            trade_cal=None,
            stock_basic=None,
            **_replay_call_kwargs(),
            **{param: _FAIL_FAST_KWARGS[param]},
        )


def test_replay_exposure_all_default_no_raise(monkeypatch):
    """全默认值 ⇒ 不抛（走到数据装配；stub 环境下正常返回）。"""
    _install_replay_stubs(monkeypatch, _OldReplayEngineStub, _ReplayEngineStub)
    _ReplayEngineStub.instances.clear()
    out = new_replay.run_oos_backtest(
        storage=_SynthStorage(),
        loader=_SynthLoader(),
        trade_cal=_synth_trade_cal(),
        stock_basic=_synth_stock_basic(),
        **_replay_call_kwargs(),
    )
    assert out and out["bt_trading_days"] == 3


# ══════════════════ D 组：runs_writer ══════════════════


def _nav_curve(dates: List[str], pvs: List[float]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d in dates],
            "portfolio_value": pvs,
            "capital": [p * 0.4 for p in pvs],
            "market_value": [p * 0.6 for p in pvs],
        }
    )
    df["nav"] = df["portfolio_value"] / 1_000_000.0
    df["return"] = df["nav"] - 1.0
    return df


def _bt_metrics(nav_curve: pd.DataFrame, bt_start: str, bt_end: str) -> Dict[str, object]:
    return {
        "bt_total_return": round(float(nav_curve["return"].iloc[-1]), 6),
        "bt_annual_return": 0.02,
        "bt_max_drawdown": -0.01,
        "bt_volatility": 0.1,
        "bt_sharpe": 1.0,
        "bt_calmar": 2.0,
        "bt_trading_days": len(nav_curve),
        "bt_start": bt_start,
        "bt_end": bt_end,
        "bt_top_n": 20,
    }


def _synth_fold0() -> Dict[str, object]:
    """合成折 0：含跨 lot 卖出（FIFO 违规 1 行）+ 止损卖出（sell_reason）+ 加仓（buy_type）。"""
    trades = pd.DataFrame(
        [
            # 主买入路径（无 buy_date 键——真实引擎 _buy_stock 形态，稀疏键）
            {
                "date": pd.Timestamp("2024-01-02"),
                "signal_date": pd.Timestamp("2023-12-29"),
                "stock": "A",
                "action": "buy",
                "price": 10.0,
                "shares": 100,
                "amount": 1000.0,
                "cost": 5.0,
            },
            # 加仓形态（带 buy_date，真实引擎 _add_to_position 形态）
            {
                "date": pd.Timestamp("2024-01-02"),
                "signal_date": pd.Timestamp("2023-12-29"),
                "stock": "B",
                "action": "buy",
                "price": 20.0,
                "shares": 100,
                "amount": 2000.0,
                "cost": 5.0,
                "buy_type": "rebalance",
                "buy_reason": "signal",
                "buy_date": pd.Timestamp("2024-01-02"),
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "signal_date": pd.Timestamp("2024-01-02"),
                "stock": "B",
                "action": "buy",
                "price": 21.0,
                "shares": 100,
                "amount": 2100.0,
                "cost": 5.0,
                "buy_type": "add",
                "buy_reason": "completion",
                "buy_date": pd.Timestamp("2024-01-02"),
            },
            # 止损卖出（sell_reason + trigger_type）
            {
                "date": pd.Timestamp("2024-01-04"),
                "signal_date": pd.Timestamp("2024-01-03"),
                "stock": "A",
                "action": "sell",
                "price": 11.0,
                "shares": 100,
                "amount": 1100.0,
                "cost": 5.0,
                "buy_date": pd.Timestamp("2024-01-02"),
                "buy_price": 10.0,
                "sell_type": "stop_loss",
                "sell_timing": "open",
                "sell_reason": "drawdown",
                "trigger_type": "drawdown",
            },
            # 非 FIFO 卖出：卖出行 buy_date=20240103 但 FIFO 首消耗 lot 为 20240102
            {
                "date": pd.Timestamp("2024-01-04"),
                "signal_date": pd.Timestamp("2024-01-03"),
                "stock": "B",
                "action": "sell",
                "price": 22.0,
                "shares": 150,
                "amount": 3300.0,
                "cost": 5.0,
                "buy_date": pd.Timestamp("2024-01-03"),
                "buy_price": 21.0,
                "sell_type": "rebalance",
                "sell_timing": "open",
            },
        ]
    )
    snapshot = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ts_code": "A",
                "shares": 100,
                "market_value": 1000.0,
                "weight": 0.001,
                "portfolio_value": 1000000.0,
                "buy_date": pd.Timestamp("2024-01-02"),
                "signal_date": "2023-12-29",
                "holding_days": 0,
                "planned_exit_date": pd.Timestamp("2024-01-30"),
                "remaining_intervals": 19,
            },
            {
                "date": "2024-01-02",
                "ts_code": "B",
                "shares": 100,  # ISO 日期形态（规范化断言）
                "market_value": 2000.0,
                "weight": 0.002,
                "portfolio_value": 1000000.0,
                "buy_date": "20240102",
                "signal_date": "2023-12-29",  # YYYYMMDD 形态
                "holding_days": 0,
                "planned_exit_date": "2024-01-30",
                "remaining_intervals": 19,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "ts_code": "A",
                "shares": 100,
                "market_value": 1050.0,
                "weight": 0.001,
                "portfolio_value": 1005000.0,
                "buy_date": pd.Timestamp("2024-01-02"),
                "signal_date": "2023-12-29",
                "holding_days": 1,
                "planned_exit_date": pd.Timestamp("2024-01-30"),
                "remaining_intervals": 18,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "ts_code": "B",
                "shares": 200,
                "market_value": 4200.0,
                "weight": 0.004,
                "portfolio_value": 1005000.0,
                "buy_date": pd.Timestamp("2024-01-02"),
                "signal_date": "2023-12-29",
                "holding_days": 1,
                "planned_exit_date": pd.Timestamp("2024-01-30"),
                "remaining_intervals": 18,
            },
            {
                "date": pd.Timestamp("2024-01-04"),
                "ts_code": "B",
                "shares": 50,
                "market_value": 1100.0,
                "weight": 0.001,
                "portfolio_value": 1010000.0,
                "buy_date": pd.Timestamp("2024-01-02"),
                "signal_date": "2023-12-29",
                "holding_days": 2,
                "planned_exit_date": pd.Timestamp("2024-01-30"),
                "remaining_intervals": 17,
            },
        ]
    )
    attribution = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2023-12-29"),
                "ranking_date": pd.Timestamp("2023-12-29"),
                "execution_date": pd.Timestamp("2024-01-02"),
                "execution_stage": "t1",
                "tranche_idx": 0,
                "planned_stock": "A",
                "actual_stock": "A",
                "planned_rank": 1,
                "actual_rank": 1,
                "pred_score": 0.9,
                "target_weight": 0.05,
                "status": "filled",
                "reason": None,
                "buy_price": 10.0,
                "signal_price": 9.9,
                "signal_to_buy_return": 0.01,
            },
        ]
    )
    nav = _nav_curve(["2024-01-02", "2024-01-03", "2024-01-04"], [1000000.0, 1005000.0, 1010000.0])
    return {
        "split_index": 0,
        "model_version": 24008,
        "train_start": "20180101",
        "train_end": "20231229",
        "test_start": "20240102",
        "test_end": "20240628",
        "metrics": _bt_metrics(nav, "20240102", "20240702"),
        "nav_curve": nav,
        "trades": trades,
        "attribution": attribution,
        "holdings_snapshot": snapshot,
    }


def _synth_fold1() -> Dict[str, object]:
    """合成折 1：跨折 seam 锚定（首行 nav = 折 0 末行）。"""
    trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-07-01"),
                "signal_date": pd.Timestamp("2024-06-28"),
                "stock": "C",
                "action": "buy",
                "price": 30.0,
                "shares": 100,
                "amount": 3000.0,
                "cost": 5.0,
                "buy_type": "rebalance",
                "buy_reason": "signal",
            },
        ]
    )
    snapshot = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-07-01"),
                "ts_code": "C",
                "shares": 100,
                "market_value": 3000.0,
                "weight": 0.003,
                "portfolio_value": 1000000.0,
                "buy_date": pd.Timestamp("2024-07-01"),
                "signal_date": "2024-06-28",
                "holding_days": 0,
                "planned_exit_date": None,
                "remaining_intervals": None,
            },
            {
                "date": pd.Timestamp("2024-07-02"),
                "ts_code": "C",
                "shares": 100,
                "market_value": 3060.0,
                "weight": 0.003,
                "portfolio_value": 1020000.0,
                "buy_date": pd.Timestamp("2024-07-01"),
                "signal_date": "2024-06-28",
                "holding_days": 1,
                "planned_exit_date": None,
                "remaining_intervals": None,
            },
        ]
    )
    nav = _nav_curve(["2024-07-01", "2024-07-02"], [1000000.0, 1020000.0])
    return {
        "split_index": 1,
        "model_version": 24009,
        "train_start": "20180701",
        "train_end": "20240628",
        "test_start": "20240701",
        "test_end": "20241231",
        "metrics": _bt_metrics(nav, "20240701", "20250101"),
        "nav_curve": nav,
        "trades": trades,
        "attribution": None,
        "holdings_snapshot": snapshot,
    }


_SYNTH_CONFIG: Dict[str, object] = {
    "bt_top_n": 20,
    "label_column": "neu_y_ret_20",
    "val_rankic_ir": None,  # 缺值键给 NA（不影响写出；loader 只校验必需列）
    "enable_fundamental": True,
    "stock_domain": "main",
    # 全量窗口键（真实 summary 的指纹键；子集模式跳过值比对的两个键）
    "wf_start_date": "20240102",
    "wf_end_date": "20260630",
}
_SYNTH_CODE_STATE = {"git_commit": "syn0000", "git_dirty": False}
_SYNTH_DATA_STATE = {"data_state_id": "syn00001", "sources": {"daily": "2024-07-02"}}


def _write_synth_batch(tmp_path: Path, batch_id: str = "syn_batch_01") -> Path:
    return runs_writer.write_runs_batch(
        [_synth_fold0(), _synth_fold1()],
        batch_id=batch_id,
        runs_root=tmp_path / "runs",
        config=dict(_SYNTH_CONFIG),
        code_state=dict(_SYNTH_CODE_STATE),
        data_state=dict(_SYNTH_DATA_STATE),
    )


def test_write_runs_batch_roundtrip(tmp_path):
    """合成 2 折 ⇒ 写出 ⇒ load_runs_batch 硬校验通过（writer 内置自校验 + 显式读回）。"""
    out_dir = _write_synth_batch(tmp_path)
    batch = load_runs_batch(out_dir)  # 不抛即过硬校验
    assert batch.batch_id == "syn_batch_01"
    assert sorted(batch.folds) == [0, 1]

    # 目录结构（契约 §1，parquet 形态）
    for name in ("batch_meta.json", "summary.parquet", "chain_nav.parquet"):
        assert (out_dir / name).exists()
    fold0 = out_dir / "folds" / "split00"
    for name in ("trades", "attribution", "holdings_snapshot", "daily"):
        assert (fold0 / f"{name}.parquet").exists()
    assert (fold0 / "_meta.json").exists()

    # 折级 _meta.json：lot_id 重建标注 + FIFO 违规计数 + topk 降级
    meta = json.loads((fold0 / "_meta.json").read_text(encoding="utf-8"))
    assert meta["trades.lot_reconstructed"] is True
    assert meta["lot_id.fifo_violations"] == 1  # B 卖出 buy_date=20240103 vs FIFO 首 lot 20240102
    assert meta["topk_detail.missing"] is True

    # trades：列集 = 契约；血缘头三列；日期规范化 YYYYMMDD str；稀疏键补 NA；lot_id 重建
    trades = batch.folds[0].trades
    assert list(trades.columns) == TRADES_CONTRACT_COLS
    assert set(trades["wf_run_id"]) == {"syn_batch_01"}
    assert set(trades["model_version"]) == {24008}
    row_buy_a = trades[(trades["ts_code"] == "A") & (trades["action"] == "buy")].iloc[0]
    assert row_buy_a["trade_date"] == "20240102" and isinstance(row_buy_a["trade_date"], str)
    assert row_buy_a["signal_date"] == "20231229"
    assert pd.isna(row_buy_a["sell_reason"])  # 缺列补 NA
    assert pd.isna(row_buy_a["tranche_idx"])
    row_sell_a = trades[(trades["ts_code"] == "A") & (trades["action"] == "sell")].iloc[0]
    assert row_sell_a["sell_reason"] == "drawdown" and row_sell_a["trigger_type"] == "drawdown"
    sell_b = trades[(trades["ts_code"] == "B") & (trades["action"] == "sell")].iloc[0]
    assert sell_b["lot_id"] == "20240102_0|20240102_1"  # 跨 2 lot FIFO 消耗
    lot_buy_a = row_buy_a["lot_id"]
    assert (
        row_sell_a["lot_id"] == lot_buy_a
    )  # A 买卖同 lot（buy 行缺 buy_date ⇒ 用 trade_date 命名）

    # holdings_snapshot：内部键 → 契约映射 + 头部三列 + 日期归一（ISO/YYYYMMDD 输入同归）
    snap = batch.folds[0].holdings_snapshot
    assert list(snap.columns) == SNAP_CONTRACT_COLS
    assert set(snap["run_id"]) == {"syn_batch_01"}
    assert set(snap["trade_date"]) == {"20240102", "20240103", "20240104"}
    row_b = snap[(snap["ts_code"] == "B") & (snap["trade_date"] == "20240102")].iloc[0]
    assert row_b["held_days"] == 0 and row_b["due_date"] == "20240130"
    assert row_b["total_value"] == 1000000.0 and row_b["remaining_days"] == 19

    # daily：真实数据源直建；n_buys/n_sells 与 trades 当日计数一致；λ 恒 1.0
    daily = batch.folds[0].daily
    assert list(daily.columns) == DAILY_CONTRACT_COLS
    assert daily["n_buys"].tolist() == [2, 1, 0]
    assert daily["n_sells"].tolist() == [0, 0, 2]
    assert daily["n_positions"].tolist() == [2, 2, 1]
    assert daily["turnover_amount"].tolist() == [3000.0, 2100.0, 4400.0]
    assert daily["nav"].iloc[0] == 1.0 and daily["daily_return"].iloc[0] == 0.0
    assert set(daily["exposure_lambda"]) == {1.0}

    # attribution：改名 + 头部三列（折 0 有、折 1 None ⇒ 不出文件）
    attr = batch.folds[0].attribution
    assert {"planned_ts_code", "actual_ts_code"} <= set(attr.columns)
    assert (out_dir / "folds" / "split01" / "attribution.parquet").exists() is False

    # chain_nav：契约列集 + 折内整数序号 + 跨折 seam 衔接
    chain = batch.chain_nav
    assert list(chain.columns) == CHAIN_NAV_CONTRACT_COLS
    seg0 = chain[chain["split_index"] == 0]
    seg1 = chain[chain["split_index"] == 1]
    assert seg0["date"].tolist() == [0, 1, 2] and seg1["date"].tolist() == [0, 1]
    assert seg0["nav"].tolist() == pytest.approx([1.0, 1.005, 1.01])
    assert seg1["nav"].iloc[0] == pytest.approx(seg0["nav"].iloc[-1])  # 折界重复点

    # summary / batch_meta：配置全键 + 血缘 + 指纹自洽
    summary = batch.summary
    assert len(summary) == 2
    assert summary["bt_top_n"].tolist() == [20, 20]
    assert set(summary["data_state_id"]) == {"syn00001"}
    assert summary["val_rankic_ir"].isna().all()
    meta_doc = batch.batch_meta
    assert meta_doc["host_mode"] == "backtest"
    assert meta_doc["source"] == "v2_p2a_native_replay"
    assert meta_doc["baseline_ref"] is None and meta_doc["arms"] == []
    assert meta_doc["code_state"] == _SYNTH_CODE_STATE
    assert meta_doc["data_state"]["data_state_id"] == "syn00001"
    assert meta_doc["meta_notes"]["topk_detail.missing"] == "true"
    assert fingerprint(meta_doc["config"]) == meta_doc["config_fingerprint"]


def test_norm_date_value_variants():
    """日期规范化：Timestamp / ISO / YYYYMMDD / int 输入归一 YYYYMMDD str；空值归 NA。"""
    norm = runs_writer._norm_date_value
    assert norm(pd.Timestamp("2024-01-02")) == "20240102"
    assert norm("2024-01-02") == "20240102"
    assert norm("20240102") == "20240102"
    assert norm(20240102) == "20240102"
    assert norm(pd.Timestamp("2024-01-02 09:30:00")) == "20240102"
    assert norm(None) is pd.NA and norm(np.nan) is pd.NA and norm(pd.NaT) is pd.NA
    with pytest.raises(ValueError, match="YYYYMMDD"):
        norm("not-a-date")


def test_writer_negative_tampered_config_fingerprint(tmp_path):
    """负例：篡改 batch_meta.config 后 ⇒ 指纹重算不一致报错（契约 §9.3）。"""
    out_dir = _write_synth_batch(tmp_path)
    meta_path = out_dir / "batch_meta.json"
    meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_doc["config"]["bt_top_n"] = 21
    meta_path.write_text(json.dumps(meta_doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="config_fingerprint 重算不一致"):
        load_runs_batch(out_dir)


def test_writer_negative_missing_trades(tmp_path):
    """负例：删 trades.parquet ⇒ 必须报错（契约三态）。"""
    out_dir = _write_synth_batch(tmp_path)
    (out_dir / "folds" / "split00" / "trades.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="trades"):
        load_runs_batch(out_dir)


def test_writer_negative_chinese_column(tmp_path):
    """负例：trades 注入中文列 ⇒ 非 ASCII 报错（契约 §9.5）。"""
    out_dir = _write_synth_batch(tmp_path)
    trades_path = out_dir / "folds" / "split00" / "trades.parquet"
    trades = pd.read_parquet(trades_path)
    trades["中文列"] = 1
    trades.to_parquet(trades_path, index=False)
    with pytest.raises(ValueError, match="非 ASCII"):
        load_runs_batch(out_dir)


def test_writer_negative_chain_seam_broken(tmp_path):
    """负例：chain seam 断裂 ⇒ 跨折衔接报错（契约 §9.8）。"""
    out_dir = _write_synth_batch(tmp_path)
    chain_path = out_dir / "chain_nav.parquet"
    chain = pd.read_parquet(chain_path)
    # 折 1 全部 nav 等比缩放：折内起止比不变（§9.2 不受影响），seam 必断（§9.8）
    chain.loc[chain["split_index"] == 1, "nav"] *= 1.5
    chain.to_parquet(chain_path, index=False)
    with pytest.raises(ValueError, match="跨折衔接断裂"):
        load_runs_batch(out_dir)


def test_writer_empty_trades_fold(tmp_path):
    """空成交折合法：trades.parquet 出契约全列空帧，daily 计数全 0。"""
    fold = _synth_fold1()
    fold["trades"] = pd.DataFrame()
    out_dir = runs_writer.write_runs_batch(
        [fold],
        batch_id="syn_empty",
        runs_root=tmp_path / "runs",
        config=dict(_SYNTH_CONFIG),
        code_state=dict(_SYNTH_CODE_STATE),
        data_state=dict(_SYNTH_DATA_STATE),
    )
    batch = load_runs_batch(out_dir)
    trades = batch.folds[1].trades
    assert list(trades.columns) == TRADES_CONTRACT_COLS and len(trades) == 0
    meta = json.loads((out_dir / "folds" / "split01" / "_meta.json").read_text(encoding="utf-8"))
    assert meta["trades.lot_reconstructed"] is True
    assert "lot_id.fifo_violations" not in meta
    daily = batch.folds[1].daily
    assert daily["n_buys"].tolist() == [0, 0] and daily["n_sells"].tolist() == [0, 0]


def test_writer_rejects_contract_extra_column(tmp_path):
    """契约外列 ⇒ 出口报错（禁止静默丢列，契约 §9.7）。"""
    fold = _synth_fold1()
    fold["trades"] = fold["trades"].assign(rogue_column=1)
    with pytest.raises(ValueError, match="契约外列"):
        runs_writer.write_runs_batch(
            [fold],
            batch_id="syn_rogue",
            runs_root=tmp_path / "runs",
            config=dict(_SYNTH_CONFIG),
            code_state=dict(_SYNTH_CODE_STATE),
            data_state=dict(_SYNTH_DATA_STATE),
        )


def test_writer_requires_data_state_id(tmp_path):
    with pytest.raises(ValueError, match="data_state_id"):
        runs_writer.write_runs_batch(
            [_synth_fold1()],
            batch_id="syn_nods",
            runs_root=tmp_path / "runs",
            config=dict(_SYNTH_CONFIG),
            code_state=dict(_SYNTH_CODE_STATE),
            data_state={},
        )


# ══════════════════ E 组：input_manifest（验收门 1） ══════════════════


def _make_synth_data_root(root: Path) -> Path:
    """合成数据根：cs_train / clean daily / 日历 / 基础表 / suspend / 折模型。"""
    for rel in (
        "features/cs_train/20240102.parquet",
        "clean/daily/2024-01-02.parquet",
        "clean/trade_cal.parquet",
        "clean/stock_basic.parquet",
        "raw/stock_basic.parquet",
        "raw/suspend/2024-01-02.parquet",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic:{rel}".encode("utf-8"))
    models = root / "models" / "stock_selection"
    models.mkdir(parents=True)
    for version in (24008, 24009):
        for suffix in ("model.joblib", "metadata.json", "features.json"):
            (models / f"v{version}_{suffix}").write_bytes(f"model:{version}:{suffix}".encode())
    return root


def test_manifest_collect_validate_roundtrip(tmp_path):
    """采集 ⇒ 校验通过；清单不含代码态字段（反例①：仅代码态变化 ⇒ 通过）。"""
    data_root = _make_synth_data_root(tmp_path / "data")
    out = tmp_path / "manifest.json"
    config = {"driver_config": {"bt_top_n": 20}, "cost_settings": {"commission_rate": 0.0002}}
    manifest = input_manifest.collect_manifest(
        data_root, out, config=config, model_versions=(24008, 24009)
    )
    assert "code_state" not in manifest  # 代码态单独登记，不参与数据身份判定
    # 配置快照内嵌 canonical 内容（校验时逐键定位差异的载体）
    assert "content" in manifest["files"][input_manifest.CONFIG_SNAPSHOT_KEY]
    assert input_manifest.validate_manifest(data_root, out, config=config) == []
    # 配置快照：同 config 通过、改 config 拒绝
    assert input_manifest.validate_config_snapshot(config, out) is None
    changed = {"driver_config": {"bt_top_n": 21}, "cost_settings": {"commission_rate": 0.0002}}
    assert input_manifest.validate_config_snapshot(changed, out) is not None


def test_manifest_requires_config_snapshot(tmp_path):
    """R2-T7-02 负例①：清单缺配置快照键 ⇒ 校验拒绝（必填，禁止放行）。"""
    data_root = _make_synth_data_root(tmp_path / "data")
    out = tmp_path / "manifest.json"
    input_manifest.collect_manifest(data_root, out, model_versions=(24008, 24009))
    diffs = input_manifest.validate_manifest(data_root, out, config={"x": 1})
    assert any("配置快照" in d and "缺失" in d for d in diffs)


def test_manifest_rejects_config_change(tmp_path):
    """R2-T7-02 负例②③：驱动配置变化 / 共享成本配置变化 ⇒ 逐键定位拒绝。"""
    data_root = _make_synth_data_root(tmp_path / "data")
    out = tmp_path / "manifest.json"
    config = {
        "driver_config": {"bt_top_n": 20, "kelly_max_leverage": 0.2},
        "cost_settings": {"commission_rate": 0.0002, "slippage": 0.0005},
    }
    input_manifest.collect_manifest(data_root, out, config=config, model_versions=(24008, 24009))
    # 驱动配置变化
    driver_changed = {
        "driver_config": {"bt_top_n": 30, "kelly_max_leverage": 0.2},
        "cost_settings": {"commission_rate": 0.0002, "slippage": 0.0005},
    }
    diffs = input_manifest.validate_manifest(data_root, out, config=driver_changed)
    assert any("driver_config.bt_top_n" in d for d in diffs), diffs
    # 共享成本配置变化（其余不变——reviewer2 反例形态）
    cost_changed = {
        "driver_config": {"bt_top_n": 20, "kelly_max_leverage": 0.2},
        "cost_settings": {"commission_rate": 0.002, "slippage": 0.0005},
    }
    diffs = input_manifest.validate_manifest(data_root, out, config=cost_changed)
    assert any("cost_settings.commission_rate" in d for d in diffs), diffs
    # 快照存在但调用方未提供同源 config ⇒ 拒绝（禁止仅提示放行）
    diffs = input_manifest.validate_manifest(data_root, out)
    assert any("未提供" in d for d in diffs), diffs


def test_manifest_rejects_content_change_same_watermark(tmp_path):
    """反例②：水位（文件名）不变但特征文件内容改变 ⇒ 拒绝。"""
    data_root = _make_synth_data_root(tmp_path / "data")
    out = tmp_path / "manifest.json"
    config = {"driver_config": {"bt_top_n": 20}, "cost_settings": {}}
    input_manifest.collect_manifest(data_root, out, config=config, model_versions=(24008, 24009))
    target = data_root / "features" / "cs_train" / "20240102.parquet"
    target.write_bytes(b"tampered")
    diffs = input_manifest.validate_manifest(data_root, out, config=config)
    assert any("哈希不符" in d and "cs_train" in d for d in diffs)


def test_manifest_rejects_market_data_change(tmp_path):
    """反例③：特征/模型不变但行情/日历/基础表被改 ⇒ 拒绝（逐一锚定）。"""
    for idx, rel in enumerate(
        (
            "clean/daily/2024-01-02.parquet",
            "clean/trade_cal.parquet",
            "raw/stock_basic.parquet",
            "raw/suspend/2024-01-02.parquet",
        )
    ):
        data_root = _make_synth_data_root(tmp_path / f"data_{idx}")
        out = tmp_path / f"manifest_{idx}.json"
        config = {"driver_config": {"bt_top_n": 20}, "cost_settings": {}}
        input_manifest.collect_manifest(
            data_root, out, config=config, model_versions=(24008, 24009)
        )
        (data_root / rel).write_bytes(b"tampered")
        diffs = input_manifest.validate_manifest(data_root, out, config=config)
        assert any("哈希不符" in d and rel in d for d in diffs), rel


def test_manifest_rejects_missing_and_extra(tmp_path):
    """清单外多出 / 清单内缺失 ⇒ 拒绝。"""
    data_root = _make_synth_data_root(tmp_path / "data")
    out = tmp_path / "manifest.json"
    config = {"driver_config": {"bt_top_n": 20}, "cost_settings": {}}
    input_manifest.collect_manifest(data_root, out, config=config, model_versions=(24008, 24009))
    extra = data_root / "clean" / "daily" / "2024-01-03.parquet"
    extra.write_bytes(b"new partition")
    diffs = input_manifest.validate_manifest(data_root, out, config=config)
    assert any("多出" in d for d in diffs)
    extra.unlink()
    (data_root / "features" / "cs_train" / "20240102.parquet").unlink()
    diffs = input_manifest.validate_manifest(data_root, out, config=config)
    assert any("缺失" in d for d in diffs)


def test_manifest_snapshot_builder_same_source(monkeypatch):
    """快照构建器同源锁定：驱动 config + 成本设置同源于 replay_b0 装配。

    get_cost_settings 打合成 stub（禁止真实配置读取）；成本设置变化 ⇒
    快照 canonical 内容变化（哈希随动）。
    """
    from scripts.v2_p2a import replay_b0

    synthetic_costs = {"commission_rate": 0.0002, "slippage": 0.0005}
    monkeypatch.setattr(replay_b0, "get_cost_settings", lambda: synthetic_costs)
    splits = [
        replay_b0.WalkForwardSplit(
            split_index=0,
            train_start="20120103",
            train_end="20180101",
            test_start="20180102",
            test_end="20180702",
        )
    ]
    snapshot = replay_b0.build_manifest_config_snapshot(splits)
    assert set(snapshot) == {"driver_config", "cost_settings"}
    assert snapshot["cost_settings"] == {"commission_rate": 0.0002, "slippage": 0.0005}
    assert snapshot["driver_config"]["wf_start_date"] == "20120103"
    assert snapshot["driver_config"]["wf_end_date"] == "20180702"
    # 成本设置变化 ⇒ 快照内容变化（门 1 可捕获）
    changed = replay_b0.build_manifest_config_snapshot(splits)
    monkeypatch.setattr(
        replay_b0, "get_cost_settings", lambda: {**synthetic_costs, "commission_rate": 0.002}
    )
    snapshot2 = replay_b0.build_manifest_config_snapshot(splits)
    assert input_manifest._sha256_config(snapshot) != input_manifest._sha256_config(snapshot2)
    assert changed == snapshot  # 同源重复构建确定性


class _SpotCheckLoader:
    """clean/daily 读取 stub（合成行情帧）。"""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def load_clean_daily(self, start: str, end: str) -> pd.DataFrame:
        return self._frame


def _make_frozen_trades_dir(root: Path, prices: List[float]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame(
        {
            "date": ["2024-01-02"] * len(prices),
            "stock": ["A"] * len(prices),
            "action": ["buy"] * len(prices),
            "price": prices,
        }
    )
    trades.to_csv(root / "walk_forward_trades_syn_split00.csv", index=False, encoding="utf-8-sig")
    return root


def test_spot_check_positive_and_negative(tmp_path):
    """窗口内容抽查：成交价 ∈ {open, close, open_adj, close_adj}（容差 1e-6）。"""
    daily = pd.DataFrame(
        {
            "ts_code": ["A"],
            "trade_date": ["20240102"],
            "open": [10.0],
            "close": [10.5],
            "open_adj": [9.9],
            "close_adj": [10.4],
        }
    )
    loader = _SpotCheckLoader(daily)
    ok_dir = _make_frozen_trades_dir(tmp_path / "ok", [10.5, 10.5 + 5e-7])  # 容差内
    assert input_manifest.spot_check_frozen_trades(ok_dir, loader, n=10) == []
    bad_dir = _make_frozen_trades_dir(tmp_path / "bad", [99.0])
    violations = input_manifest.spot_check_frozen_trades(bad_dir, loader, n=10)
    assert len(violations) == 1 and "99.0" in violations[0]


# ══════════════════ F 组：compare_replay_vs_b0（门 3/4/5） ══════════════════


def _make_frozen_dir(new_batch_dir: Path, frozen_dir: Path) -> Path:
    """由合成新批次派生冻结目录（旧命名 CSV 形态：stock/date 列、无 batch_id）。"""
    frozen_dir.mkdir(parents=True, exist_ok=True)
    batch = load_runs_batch(new_batch_dir)
    frozen_summary = batch.summary.drop(columns=["batch_id"])  # 旧 summary 无 batch_id
    frozen_summary.to_csv(
        frozen_dir / "walk_forward_summary_0101_0001.csv", index=False, encoding="utf-8-sig"
    )
    batch.chain_nav.to_csv(frozen_dir / "chain_nav_wf_syn.csv", index=False, encoding="utf-8-sig")
    for split_index, fold in batch.folds.items():
        tag = f"split{split_index:02d}"
        trades = fold.trades.rename(columns={"trade_date": "date", "ts_code": "stock"})
        trades = trades.drop(columns=["lot_id", "tranche_idx"])  # 旧产物无 v2 新增列
        trades.to_csv(
            frozen_dir / f"walk_forward_trades_syn_{tag}.csv", index=False, encoding="utf-8-sig"
        )
        if fold.attribution is not None:
            attr = fold.attribution.rename(
                columns={"planned_ts_code": "planned_stock", "actual_ts_code": "actual_stock"}
            )
            attr.to_csv(
                frozen_dir / f"walk_forward_execution_attribution_syn_{tag}.csv",
                index=False,
                encoding="utf-8-sig",
            )
    return frozen_dir


def test_compare_all_gates_pass(tmp_path):
    """门 3/4/5 各 1 正：自洽批次 vs 派生冻结 ⇒ 总 PASS，batch_id 另列登记放行。"""
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert report.ok, f"{report.gate3_diffs + report.gate4_diffs + report.gate5_diffs}"
    assert any("batch_id" in n for n in report.notes)  # 迁移专用标识登记放行


def test_compare_gate3_value_diff_listed(tmp_path):
    """门 3 负例：冻结侧指纹键错值 ⇒ 列清单（含折号与键名）。"""
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    summary_path = frozen_dir / "walk_forward_summary_0101_0001.csv"
    frozen = pd.read_csv(summary_path, encoding="utf-8-sig")
    frozen.loc[frozen["split_index"] == 0, "bt_top_n"] = 30
    frozen.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert not report.ok
    assert any("bt_top_n" in d and "split00" in d for d in report.gate3_diffs)


def test_compare_gate4_swapped_ties_counted(tmp_path):
    """门 4 负例：并列互换被计数（多重集相等、位次不同 ⇒ 计数 > 0 拒绝）。"""
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    trades_path = frozen_dir / "walk_forward_trades_syn_split00.csv"
    trades = pd.read_csv(trades_path, encoding="utf-8-sig")
    # 折 0 前两笔为同日不同股买入：互换位次（9 列多重集不变、位次变）
    trades = trades.iloc[[1, 0] + list(range(2, len(trades)))].reset_index(drop=True)
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert not report.ok
    assert any("并列互换 2 行" in d for d in report.gate4_diffs)


def test_compare_gate5_nav_tolerance_enforced(tmp_path):
    """门 5 负例：净值超 1e-6 被拒（容差内扰动放行锚定一并验证）。"""
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    chain_path = frozen_dir / "chain_nav_wf_syn.csv"
    # 容差内（5e-7 < 1e-6）⇒ 放行
    chain = pd.read_csv(chain_path, encoding="utf-8-sig")
    chain.loc[chain["split_index"] == 0, "nav"] += 5e-7
    chain.to_csv(chain_path, index=False, encoding="utf-8-sig")
    assert compare_tool.compare_replay_vs_b0(new_dir, frozen_dir).ok
    # 超差（1e-5 > 1e-6）⇒ 拒绝
    chain = pd.read_csv(chain_path, encoding="utf-8-sig")
    chain.loc[chain["split_index"] == 1, "nav"] += 1e-5
    chain.to_csv(chain_path, index=False, encoding="utf-8-sig")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert not report.ok
    assert any("split01" in d and "超差" in d for d in report.gate5_diffs)


def test_compare_frozen_csv_float_roundtrip(tmp_path):
    """回归锁定（T7 预检实测缺陷）：冻结 CSV 17 位浮点 repr 必须 round-trip 解析。

    pandas 默认（含 ``float_precision="high"``）解析器会把 ``48.942251999999996``
    误读为相邻 double（…98 → …99），造成「冻结 vs 重放」的伪 1-ulp 差异；
    比对工具必须用 ``float_precision="round_trip"``。本测试构造的数值在缺
    round_trip 时必然触发门 4 误报（反变异锚定：直接断言两种解析结果不同）。
    """
    tricky = float("48.942251999999996")  # 17 位 repr 才能精确表示的 double
    import struct as _struct

    assert _struct.pack(
        ">d",
        pd.read_csv(  # 反变异锚定：默认解析确实会漂
            __import__("io").StringIO("x\n48.942251999999996")
        )["x"][0],
    ) != _struct.pack(">d", tricky)

    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    # 新侧 parquet 与冻结 CSV 文本写入同一 double（CSV 以 17 位 repr 落盘）
    trades_parquet = new_dir / "folds" / "split00" / "trades.parquet"
    trades = pd.read_parquet(trades_parquet)
    trades.loc[0, "cost"] = tricky
    trades.to_parquet(trades_parquet, index=False)
    frozen_trades = frozen_dir / "walk_forward_trades_syn_split00.csv"
    frame = pd.read_csv(frozen_trades, encoding="utf-8-sig")
    frame.loc[0, "cost"] = tricky
    frame.to_csv(frozen_trades, index=False, encoding="utf-8-sig")
    assert "48.942251999999996" in frozen_trades.read_text(encoding="utf-8-sig")

    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert report.ok, f"round_trip 解析下不应有伪差异: {report.gate4_diffs}"


def test_compare_subset_mode_registers_wf_window_keys(tmp_path):
    """门 3 显式预检子集模式：wf_start_date/wf_end_date 登记跳过值比对。

    预检只跑折子集，旧 runner 同口径从筛选后 splits 端点推导这两个键，
    与冻结全量批必然不同；完整终验模式（默认）不豁免（两键照常比对）。
    """
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    subset_dir = _make_subset_batch(tmp_path / "new_subset", new_dir)

    report = compare_tool.compare_replay_vs_b0(subset_dir, frozen_dir, allow_subset=True)
    assert report.ok, f"{report.gate3_diffs + report.gate4_diffs + report.gate5_diffs}"
    assert any("子集模式" in n and "wf_end_date" in n for n in report.notes)
    # 反向锚定：非子集时 wf_end_date 错值必须报错（跳过逻辑不泄漏到完整模式）
    new_dir2 = _write_synth_batch(tmp_path / "new2")
    frozen_dir2 = _make_frozen_dir(new_dir2, tmp_path / "frozen2")
    summary_path = frozen_dir2 / "walk_forward_summary_0101_0001.csv"
    frozen2 = pd.read_csv(summary_path, encoding="utf-8-sig")
    frozen2["wf_end_date"] = "20991231"
    frozen2.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report2 = compare_tool.compare_replay_vs_b0(new_dir2, frozen_dir2)
    assert not report2.ok
    assert any("wf_end_date" in d for d in report2.gate3_diffs)


def _make_subset_batch(subset_dir: Path, new_dir: Path) -> Path:
    """把合成 2 折批次裁剪为仅 split00 的子集批次（预检形态；summary 窗口键漂移）。"""
    import shutil

    shutil.copytree(new_dir, subset_dir)
    shutil.rmtree(subset_dir / "folds" / "split01")
    summary = pd.read_parquet(subset_dir / "summary.parquet")
    summary = summary[summary["split_index"] == 0].reset_index(drop=True)
    summary["wf_end_date"] = "20991231"  # 子集推导值漂移（显式预检下登记跳过）
    summary.to_parquet(subset_dir / "summary.parquet", index=False)
    chain = pd.read_parquet(subset_dir / "chain_nav.parquet")
    chain = chain[chain["split_index"] == 0].reset_index(drop=True)
    chain.to_parquet(subset_dir / "chain_nav.parquet", index=False)
    # batch_meta 随裁剪重算（config 指纹 + data_state 一致性）
    import json as _json

    from src.lazybull.v2.evidence.fingerprint_keys import (
        config_fingerprint_keys,
        fingerprint,
    )

    meta_path = subset_dir / "batch_meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    fp_keys = config_fingerprint_keys(summary.columns.tolist())
    row = summary.iloc[0]
    meta["config"] = {
        k: (row[k].item() if hasattr(row[k], "item") else row[k])
        for k in fp_keys
        if k in summary.columns and pd.notna(row[k])
    }
    meta["config_fingerprint"] = fingerprint(meta["config"])
    meta_path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2, default=str), "utf-8")
    return subset_dir


def test_compare_strict_mode_rejects_missing_fold(tmp_path):
    """R2-T7-03 负例①：默认完整终验模式下缺折 ⇒ 门 3/门 5 均拒绝（不自动放行）。"""
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    subset_dir = _make_subset_batch(tmp_path / "new_subset", new_dir)

    report = compare_tool.compare_replay_vs_b0(subset_dir, frozen_dir)
    assert not report.ok
    assert any("折集合不一致" in d and "全等" in d for d in report.gate3_diffs)
    assert any("折集合不一致" in d for d in report.gate5_diffs)


def test_compare_subset_mode_rejects_extra_fold(tmp_path):
    """R2-T7-03 负例②：显式预检下新侧出现冻结外折（非真子集）⇒ 拒绝。"""
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    # 冻结侧只留 split00（summary/chain/trades 同步裁剪），新侧全量 2 折
    summary_path = frozen_dir / "walk_forward_summary_0101_0001.csv"
    frozen_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    frozen_summary = frozen_summary[frozen_summary["split_index"] == 0].reset_index(drop=True)
    frozen_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    chain_path = frozen_dir / "chain_nav_wf_syn.csv"
    frozen_chain = pd.read_csv(chain_path, encoding="utf-8-sig")
    frozen_chain = frozen_chain[frozen_chain["split_index"] == 0].reset_index(drop=True)
    frozen_chain.to_csv(chain_path, index=False, encoding="utf-8-sig")

    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir, allow_subset=True)
    assert not report.ok
    assert any("真子集" in d for d in report.gate3_diffs)
    assert any("真子集" in d for d in report.gate5_diffs)


def test_compare_attribution_unconditional_and_strict(tmp_path):
    """R1-T7-01/R2-T7-01 回归锁定：attribution 独立于成交比对无条件执行。

    成交完全一致时：attr 错值 ⇒ 拒绝；attr 文件缺失 ⇒ 拒绝；attr 缺列 ⇒ 拒绝；
    仅 wf_run_id（迁移专用运行标识）差异 ⇒ 规范化后放行。
    """
    new_dir = _write_synth_batch(tmp_path / "new")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen")
    attr_path = frozen_dir / "walk_forward_execution_attribution_syn_split00.csv"

    # 负例 1：成交不变，attr 值错（actual_stock 改写）⇒ 拒绝
    attr = pd.read_csv(attr_path, encoding="utf-8-sig")
    attr.loc[0, "actual_stock"] = "WRONG_STOCK"
    attr.to_csv(attr_path, index=False, encoding="utf-8-sig")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert not report.ok
    assert any("attribution 全列比对不一致" in d for d in report.gate4_diffs)

    # 负例 2：attr 文件删除（冻结侧缺失，新侧有）⇒ 拒绝
    attr_path.unlink()
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert not report.ok
    assert any("attribution 单侧缺失" in d for d in report.gate4_diffs)

    # 恢复 + 负例 3：attr 缺列（冻结侧丢 reason 列）⇒ 拒绝（禁止交集静默）
    new_dir = _write_synth_batch(tmp_path / "new3")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen3")
    attr_path = frozen_dir / "walk_forward_execution_attribution_syn_split00.csv"
    attr = pd.read_csv(attr_path, encoding="utf-8-sig")
    attr = attr.drop(columns=["reason"])
    attr.to_csv(attr_path, index=False, encoding="utf-8-sig")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert not report.ok
    assert any("attribution 列集合不一致" in d for d in report.gate4_diffs)

    # 正例：仅 wf_run_id 不同 ⇒ 规范化放行（防假阳性回归锁定）
    new_dir = _write_synth_batch(tmp_path / "new4")
    frozen_dir = _make_frozen_dir(new_dir, tmp_path / "frozen4")
    attr_path = frozen_dir / "walk_forward_execution_attribution_syn_split00.csv"
    attr = pd.read_csv(attr_path, encoding="utf-8-sig")
    attr["wf_run_id"] = "wf_legacy_other_id"
    attr.to_csv(attr_path, index=False, encoding="utf-8-sig")
    report = compare_tool.compare_replay_vs_b0(new_dir, frozen_dir)
    assert report.ok, f"{report.gate4_diffs}"
    assert any("wf_run_id 规范化" in n for n in report.notes)


# ══════════════════ G 组：依赖方向 AST 扫描 ══════════════════

#: v2/hosts 旧侧依赖白名单（登记口径 = 规划 §3.5 过渡依赖表 + factors/universe 永久沿用）
_ALLOWED_OLD_IMPORT_PREFIXES = (
    "src.lazybull.common.config",  # 只读过渡（全局单例），去除节点 P4
    "src.lazybull.data",  # 只读过渡（loader/storage），去除节点 P2b/P3
    "src.lazybull.factors.holdertrade",  # 运行时派生四族（永久沿用）
    "src.lazybull.factors.repurchase",
    "src.lazybull.factors.top10_floatholders",
    "src.lazybull.factors.top_inst",
    "src.lazybull.universe",  # 永久沿用
    "src.lazybull.ml.walk_forward.utils",  # 只读过渡（切分生成），去除节点 P3
    "src.lazybull.ml.walk_forward.data_state",  # 只读过渡（数据态采集），去除节点 P3
    "src.lazybull.ml.model_registry",  # 只读过渡（若用到），去除节点 P3
)

# 退役 exposure 三 mixin 冻结集（41 项）的扫描口径子集：剔除 T5 文档化排除
# "record"（通用局部变量同名）与 D6 归属的 5 个 fail-fast 签名参数名。
_EXPOSURE_FAIL_FAST_PARAMS = frozenset(
    {
        "exposure_table",
        "exposure_policy",
        "exposure_replenish",
        "exposure_trim_tolerance",
        "exposure_budget_discount_replenish",
    }
)
_RETIRED_EXPOSURE_MODULES = frozenset({"exposure_override", "exposure_trim", "exposure_replenish"})
# 41 项冻结集中进入 v2/hosts 扫描面的符号（= 41 − record − 5 fail-fast 参数名；
# 与 tests/test_v2_p2a_t5_equivalence.py 的 _FROZEN_RETIRED_SYMBOLS 同源性断言见下）
_RETIRED_SYMBOLS_FOR_HOSTS_SCAN = frozenset(
    {
        "BacktestExposureOverrideMixin",
        "BacktestExposureTrimMixin",
        "BacktestExposureReplenishMixin",
        "ExposureOverrideStats",
        "load_exposure_table",
        "_has_exposure_source",
        "_record_budget_discount_release",
        "_date_key",
        "_get_holdings_rows_for_policy",
        "_remaining_intervals",
        "_get_exposure_multiplier",
        "set_exposure_table",
        "set_exposure_policy",
        "get_exposure_report",
        "_init_exposure_trim_state",
        "_queue_exposure_trim",
        "_execute_pending_exposure_trims",
        "get_exposure_trim_report",
        "_init_exposure_replenish_state",
        "_queue_exposure_replenish",
        "_execute_pending_exposure_replenishes",
        "_can_replenish_now",
        "get_exposure_replenish_report",
        "exposure_stats",
        "_exposure_missing_dates",
        "exposure_policy_provider",
        "exposure_policy_summary",
        "exposure_replenish_enabled",
        "exposure_release_budget",
        "pending_exposure_trims",
        "exposure_trim_stats",
        "pending_exposure_replenishes",
        "exposure_replenish_stats",
        "covered_days",
        "missing_days",
        "reduced_days",
        "missing_samples",
    }
)


def _extract_retired_symbols_41() -> frozenset:
    """旧 exposure 三源文件定义侧符号运行时 AST 提取（T5 同款口径，41 项冻结）。"""
    import src.lazybull.backtest.exposure_override as m1
    import src.lazybull.backtest.exposure_replenish as m2
    import src.lazybull.backtest.exposure_trim as m3

    symbols = set()
    for module in (m1, m2, m3):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                symbols.add(node.name)
            elif isinstance(node, ast.ClassDef):
                symbols.add(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name != "__init__":
                            symbols.add(item.name)
                        for sub in ast.walk(item):
                            if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                                targets = (
                                    sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                                )
                                for target in targets:
                                    if (
                                        isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == "self"
                                    ):
                                        symbols.add(target.attr)
                    elif isinstance(item, (ast.AnnAssign, ast.Assign)):
                        targets = [item.target] if isinstance(item, ast.AnnAssign) else item.targets
                        for target in targets:
                            if isinstance(target, ast.Name):
                                symbols.add(target.id)
    return frozenset(symbols)


def _import_candidates_of(path: Path, importer_package: Optional[str] = None) -> List[str]:
    """全量 import 候选模块路径（绝对 + 相对按所属包归一 + 父包成员展开）。

    Args:
        importer_package: 被扫文件所属包（点分路径）；None = 按文件父目录相对
            仓库根推导（仓库外探针文件必须显式传入，T5 R2-T5-R3-01 口径）。
    """
    package = (
        importer_package
        if importer_package is not None
        else ".".join(path.resolve().parent.relative_to(_REPO_ROOT).parts)
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                resolved = node.module or ""
            else:
                parts = package.split(".") if package else []
                keep = len(parts) - (node.level - 1)
                anchor = ".".join(parts[:keep]) if keep > 0 else ""
                resolved = f"{anchor}.{node.module}" if node.module else anchor
            candidates.append(resolved)
            candidates.extend(f"{resolved}.{alias.name}" for alias in node.names if resolved)
    return candidates


def _is_allowed_old_import(candidate: str) -> bool:
    return any(
        candidate == prefix or candidate.startswith(prefix + ".")
        for prefix in _ALLOWED_OLD_IMPORT_PREFIXES
    )


def test_retired_symbol_extraction_consistency():
    """41 项冻结集同源性：提取集 − record − fail-fast 参数名 == 本文件扫描集。"""
    extracted = _extract_retired_symbols_41()
    assert len(extracted) == 41, f"退役符号提取集漂移（应 41 项）: {sorted(extracted)}"
    scan_set = extracted - {"record"} - _EXPOSURE_FAIL_FAST_PARAMS
    assert scan_set == _RETIRED_SYMBOLS_FOR_HOSTS_SCAN, (
        f"扫描集漂移: 新增 {sorted(scan_set - _RETIRED_SYMBOLS_FOR_HOSTS_SCAN)}, "
        f"缺失 {sorted(_RETIRED_SYMBOLS_FOR_HOSTS_SCAN - scan_set)}"
    )


def test_hosts_import_direction_whitelist():
    """v2/hosts 全部文件：v2 以外的 src.lazybull.* import 必须落在旧侧白名单。"""
    sources = sorted(_V2_HOSTS_DIR.rglob("*.py"))
    assert sources, "v2/hosts 扫描面为空（异常）"
    hits: Dict[str, List[str]] = {}
    for path in sources:
        bad = [
            c
            for c in _import_candidates_of(path)
            if c.startswith("src.lazybull")
            and not c.startswith("src.lazybull.v2")
            and not _is_allowed_old_import(c)
        ]
        if bad:
            hits[str(path.relative_to(_REPO_ROOT))] = bad
    assert hits == {}, f"v2/hosts 存在白名单外旧依赖: {hits}"


def _scan_retired_in_hosts(path: Path, importer_package: Optional[str] = None) -> List[str]:
    """单文件退役符号扫描（归属判定：fail-fast 参数名/字符串/文档文本不计）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for candidate in _import_candidates_of(path, importer_package):
        if candidate.split(".")[-1] in _RETIRED_EXPOSURE_MODULES:
            hits.append(f"import {candidate}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _RETIRED_SYMBOLS_FOR_HOSTS_SCAN:
                hits.append(f"def {node.name}")
        elif isinstance(node, ast.Name) and node.id in _RETIRED_SYMBOLS_FOR_HOSTS_SCAN:
            hits.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _RETIRED_SYMBOLS_FOR_HOSTS_SCAN:
            hits.append(f"attr {node.attr}")
    return hits


def test_hosts_no_retired_exposure_symbols():
    """退役 exposure 符号在 v2/hosts 调用面 0 残留（fail-fast 参数名不计）。"""
    hits: Dict[str, List[str]] = {}
    for path in sorted(_V2_HOSTS_DIR.rglob("*.py")):
        found = _scan_retired_in_hosts(path)
        if found:
            hits[path.name] = found
    assert hits == {}, f"v2/hosts 残留退役 exposure 符号: {hits}"


def test_replay_driver_passes_rebalance_freq_to_split_generator():
    """驱动切分生成必须带 rebalance_freq=20（T7 自查缺陷回归锁定）。

    旧 runner 调用 ``generate_walk_forward_splits_by_count`` 传 rebalance_freq=20
    （test_end 对齐调仓边界）；省略该参数实测折窗口整体错位（split0 由
    20181126/20190524 漂为 20181217/20190617），B0 冻结窗口无法复现。
    """
    source = (_REPO_ROOT / "scripts" / "v2_p2a" / "replay_b0.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "generate_walk_forward_splits_by_count"
    ]
    assert len(calls) == 1, f"驱动应恰好调用切分生成器一次，实测 {len(calls)}"
    kw_names = {kw.arg for kw in calls[0].keywords}
    assert (
        "rebalance_freq" in kw_names
    ), "驱动调用切分生成器缺 rebalance_freq 关键字——折窗口将与 B0 冻结批次错位"


def test_retired_hosts_scanner_catches_mutations(tmp_path):
    """扫描器有效性：退役引用注入必捕获；fail-fast 参数名/文档文本零误报。"""
    positives = {
        "attr_call": "engine.set_exposure_table(None)\n",
        "name_ref": "stats = engine.exposure_trim_stats\n",
        "def_method": "def set_exposure_policy(self, p):\n    pass\n",
        "import_module": "from src.lazybull.backtest import exposure_trim\n",
    }
    for case, src in positives.items():
        probe = tmp_path / f"probe_{case}.py"
        probe.write_text(src, encoding="utf-8")
        assert _scan_retired_in_hosts(
            probe, importer_package="src.lazybull.v2.hosts.backtest"
        ), f"变异反例未捕获: {case}"
    negatives = {
        # D6 归属：replay 的 fail-fast 签名参数名（形参/关键字/校验引用）不计
        "failfast_params": (
            "def f(exposure_table=None, exposure_policy=None):\n"
            "    if exposure_table is not None or exposure_policy is not None:\n"
            "        raise ValueError('退役')\n"
        ),
        "doc_text": '"""exposure 政策族已退役摘除（set_exposure_table 调用点删除）。"""\n',
    }
    for case, src in negatives.items():
        probe = tmp_path / f"probe_neg_{case}.py"
        probe.write_text(src, encoding="utf-8")
        assert not _scan_retired_in_hosts(
            probe, importer_package="src.lazybull.v2.hosts.backtest"
        ), f"归属负例误报: {case}"


# ══════════════════ H 组：禁止真实配置读取永久回归 ══════════════════


class _PassedCounter:
    """inner pytest 运行的通过数记录器（复收证据：复跑非空跑）。"""

    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.outcome == "passed":
            self.passed += 1


def test_t7_suite_forbids_real_config_read(monkeypatch):
    """永久回归：真实配置读取替身（抛错）下同进程复跑本文件（排除本项）。

    三个真实配置入口（get_data_root / get_reports_root /
    get_stock_selection_models_root）在新旧两侧全部绑定模块上打抛错替身：
    任一测试绕过合成 stub 读取真实配置 ⇒ 替身抛出 ⇒ 该测试失败。
    """
    real_reads: List[str] = []

    def _reject(name: str):
        def _raiser(*args, **kwargs):
            real_reads.append(name)
            raise RuntimeError(f"reviewer: unexpected configuration read ({name})")

        return _raiser

    bindings = [
        (old_runtime, "get_stock_selection_models_root"),
        (new_runtime, "get_stock_selection_models_root"),
        (old_factory, "get_stock_selection_models_root"),
        (new_factory, "get_stock_selection_models_root"),
        (old_ml_signal, "get_stock_selection_models_root"),
        (new_ml_signal, "get_stock_selection_models_root"),
        (old_replay, "get_data_root"),
        (new_replay, "get_data_root"),
        (old_reporter, "get_reports_root"),
        (new_reporter, "get_reports_root"),
    ]
    for module, name in bindings:
        monkeypatch.setattr(module, name, _reject(f"{module.__name__}.{name}"))

    counter = _PassedCounter()
    exit_code = pytest.main(
        [
            "-q",
            "-k",
            "not test_t7_suite_forbids_real_config_read",
            str(Path(__file__)),
        ],
        plugins=[counter],
    )
    assert exit_code == 0, "禁止真实配置读取下 T7 套件存在失败项"
    assert real_reads == [], f"真实配置读取发生: {real_reads}"
    # 复跑非空跑：当前文件 46 项（含本项），inner 应跑 45 项；下限防未来漂移
    assert counter.passed >= 44, f"inner 复跑通过数异常: {counter.passed}"
