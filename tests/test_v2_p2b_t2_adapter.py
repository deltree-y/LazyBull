# -*- coding: utf-8 -*-
"""P2b-T2 宿主适配器 + 内核接缝 + NAV 入口测试。

验收口径（实施规划 ``docs/plans/v2/p2b_implementation_plan_20261016.md`` §5-T2
任务卡 / §7 合成测试矩阵；方案 F12 §4.10；D3/D4/D5；评审 R3-02）：

- A 组（NAV 账本层唯一入口）：``calculate_nav`` 标量/Series 双形态（保留索引、
  dtype、NaN 与运算次序；不转列表、不逐行强转、不舍入）；``generate_nav_curve``
  委托该入口且默认曲线数值精确等价；非 1 初始资金锚定单位口径。
- B 组（适配器字段与快照，D5）：cash/total_value/NAV/持仓价格单位正确；
  ``last_price`` 只按 ``last_known_price → buy_trade_price``、``buy_pnl_price``
  不混入；聚合 lot 合法性；不可变外层容器；**逐次接缝新建快照**（两次调用
  状态变化后不得复用旧 Account）；权重无损往返（不排序、不 str、不额外
  float 强转——以类型与对象身份双重锁定）；零新增估值（适配器无
  ``_calculate_portfolio_value`` 调用、无 ``last_known_price`` 写回）；
  缺 ``buy_date``/``buy_cost_cash`` 等关键键 fail-fast 含股票与键名；
  ``total_value=None``（``_last_rebalance_nav`` 缺失）fail-fast 且计数=0
  （R3-02：禁止 getattr 回退或静默转 0）；必需持仓字段的 NaN/NaT/NA
  缺失标记同样在真实 merge 前 fail-fast（R4-T2-01/S3-T2-02：
  ``Position`` 聚合校验对 NaN 不自拒，必须在适配器边界拒绝）。
- C 组（接缝与三层零新增，D3/D4 方案 A）：归一化→合并→限权顺序锁定
  （含**非恒等**归一化桩与非恒等端口的分辨力证明——R2-T2-06 处置）；
  默认关闭时接缝严格跳过；``BacktestEngine`` 类级默认 ``merge_port=None``、
  ``BacktestEngineML`` 继承不重定义、实例 ``vars`` 零新增（t4/t7 锚由
  P2a 等价套件在全量中继续锁定）；**补位/T+1 局部计划分类回归锁**
  （R1-T2-01：行为驱动补位方法、两臂输出一致、接缝调用增量=0 +
  源码级零引用锁与注入反变异）。
- D 组（反变异自证）：装配时缓存 Account、半缓存 positions/total_value、
  排序权重、numpy float 强转、无差别/选择性混入 ``buy_pnl_price``、
  宿主私算 NAV（源码注入）七类变异必须被对应断言捕获，且先锚定变异确实
  生效（R2-T2-02/03 处置：快照断言覆盖 cash/total_value/positions 全字段、
  绩效价禁令同时锁缓存价优先路径）。

全部输入为合成值对象/桩，不依赖真实配置与真实数据。
"""

import ast
import inspect
import math
from pathlib import Path
from types import MappingProxyType
from typing import Optional, get_type_hints

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.common.types import TSCode, Weight
from src.lazybull.v2.core.accounting import nav as nav_module
from src.lazybull.v2.core.accounting.nav import calculate_nav, generate_nav_curve
from src.lazybull.v2.core.decision.portfolio_merger import LegacyTargetMergePort
from src.lazybull.v2.core.decision.weight_processor import (
    cap_and_normalize_weights,
    resolve_tranche_weight_cap,
)
from src.lazybull.v2.core.execution import buy_execution, signal_execution
from src.lazybull.v2.core.execution.engine import BacktestEngine
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.core.execution.signal_execution import BacktestSignalExecutionMixin
from src.lazybull.v2.hosts.backtest import merger_adapter as adapter_mod
from src.lazybull.v2.hosts.backtest.merger_adapter import LegacyMergerAdapter

_D0 = pd.Timestamp("2026-01-05")
_D1 = pd.Timestamp("2026-01-06")

_POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
_EMPTY = inspect.Parameter.empty


def _full_signature(func) -> tuple:
    """完整有序签名解析（范式同 test_v2_p2a_t2_protocols.py / P2b-T1）。"""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    params = tuple(
        (name, param.kind, param.default, hints.get(name)) for name, param in sig.parameters.items()
    )
    return params, hints.get("return")


def _positions_fixture() -> dict:
    """合成旧持仓 dict（两只非零持仓 + 一只零股占位 + 缓存价/后复权价区分）。"""
    return {
        "600000.SH": {
            "shares": 1000,
            "buy_date": _D0,
            "signal_date": _D0,
            "buy_trade_price": 10.0,
            "buy_pnl_price": 88.8,  # 后复权绩效价：绝不允许混入 last_price
            "buy_cost_cash": 10013.0,
            "last_known_price": 10.5,
        },
        "000001.SZ": {
            "shares": 2000,
            "buy_date": _D0,
            "signal_date": _D0,
            "buy_trade_price": 20.0,
            "buy_pnl_price": 77.7,
            "buy_cost_cash": 40026.0,
            # 无 last_known_price：走 buy_trade_price 兜底
        },
        "300750.SZ": {  # 零股占位：不进聚合视图
            "shares": 0,
            "buy_date": _D0,
            "buy_trade_price": 30.0,
            "buy_cost_cash": 0.0,
        },
    }


def _make_adapter(**kwargs) -> LegacyMergerAdapter:
    defaults = dict(sleeve_id="value")
    defaults.update(kwargs)
    return LegacyMergerAdapter(**defaults)


def _call_adapter(adapter, weights=None, positions=None, **overrides):
    """以合成默认值驱动端口调用（测试聚焦参数可覆盖）。"""
    kwargs = dict(
        weights=weights if weights is not None else {"600000.SH": 0.6, "000001.SZ": 0.4},
        date=_D1,
        cash=500000.0,
        total_value=1_234_567.89,
        positions=positions if positions is not None else _positions_fixture(),
        initial_capital=777_777.0,  # 非 1 初始资金（锁定单位口径）
    )
    kwargs.update(overrides)
    return adapter.merge_legacy_batch_targets(**kwargs)


class _SpyMerger:
    """记录 merge() 实参的间谍（替换适配器内合并器，观测快照内容）。"""

    def __init__(self) -> None:
        self.calls = []

    def merge(self, targets, current):
        self.calls.append((targets, current))
        return next(iter(targets.values()))


def _make_spy_adapter() -> LegacyMergerAdapter:
    adapter = _make_adapter()
    adapter._merger = _SpyMerger()
    return adapter


# ══════════════════ A. NAV 账本层唯一入口（D5/CX-02） ══════════════════


def test_calculate_nav_scalar_exact():
    """标量：非 1 初始资金下的精确除法（与直接除法逐位一致）。"""
    expected = 1_234_567.89 / 777_777.0
    result = calculate_nav(1_234_567.89, 777_777.0)
    assert result == expected
    assert isinstance(result, float)
    assert result != 1.0  # 锚定：确实非平凡比值（防恒真断言空转）


def test_calculate_nav_series_semantics():
    """Series：保留索引、dtype、NaN 与运算次序；不转列表、不逐行强转、不舍入。"""
    idx = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    values = pd.Series([100.0, float("nan"), 101.5], index=idx, dtype="float64")
    result = calculate_nav(values, 100.0)
    assert isinstance(result, pd.Series)
    assert result.index is idx  # 索引对象保留（未重建）
    assert result.dtype == values.dtype
    assert result.iloc[0] == 1.0 and math.isnan(result.iloc[1])  # NaN 透传，不拒绝不清洗
    assert result.iloc[2] == 1.015
    # 运算次序逐位等价：与直接 Series 除法全列精确一致
    pd.testing.assert_series_equal(result, values / 100.0, check_exact=True)


def test_generate_nav_curve_delegates_and_exact(monkeypatch):
    """generate_nav_curve 委托 calculate_nav，且默认曲线数值精确等价于旧直除。"""
    records = [
        {"date": _D0, "portfolio_value": 1_000_000.0},
        {"date": _D1, "portfolio_value": 1_010_000.5},
    ]
    df = generate_nav_curve(records, 1_000_000.0)
    expected_nav = pd.DataFrame(records)["portfolio_value"] / 1_000_000.0
    pd.testing.assert_series_equal(df["nav"], expected_nav, check_exact=True, check_names=False)
    pd.testing.assert_series_equal(
        df["return"], df["nav"] - 1.0, check_exact=True, check_names=False
    )

    # 委托证据：替换模块内 calculate_nav 引用，generate_nav_curve 必须实际经过它
    calls = []

    def _spy(total_value, initial_capital):
        calls.append((total_value, initial_capital))
        return calculate_nav(total_value, initial_capital)

    monkeypatch.setattr(nav_module, "calculate_nav", _spy)
    generate_nav_curve(records, 1_000_000.0)
    assert len(calls) == 1 and calls[0][1] == 1_000_000.0


# ══════════════════ B. 适配器字段、快照与无损往返（D4/D5） ══════════════════


def test_adapter_account_fields_and_units():
    """现金/总值/NAV/持仓字段与单位：非 1 初始资金锚定；聚合 lot 合法。"""
    adapter = _make_spy_adapter()
    _call_adapter(adapter)
    account = adapter._merger.calls[0][1]
    assert account.cash.value == 500000.0
    assert account.total_value.value == 1_234_567.89
    # NAV 唯一派生口径：total_value / initial_capital（非 1 初始资金锁定单位）
    assert account.nav == 1_234_567.89 / 777_777.0
    assert account.nav == calculate_nav(1_234_567.89, 777_777.0)

    positions = account.positions
    assert set(positions) == {TSCode("600000.SH"), TSCode("000001.SZ")}  # 零股占位被跳过
    pos = positions[TSCode("600000.SH")]
    assert pos.shares.value == 1000
    assert pos.cost_basis.value == 10013.0
    assert len(pos.lots) == 1  # 单聚合视图 lot
    assert pos.lots[0].shares.value == 1000 and pos.lots[0].cost_basis.value == 10013.0
    assert pos.lots[0].buy_date.value == _D0.date()
    assert pos.last_price.value == 10.5  # last_known_price 优先


def test_adapter_last_price_fallback_never_pnl_price():
    """last_price 回退链：last_known_price → buy_trade_price；buy_pnl_price 永不混入。"""
    account = _make_adapter()._build_account(
        cash=1.0,
        total_value=2.0,
        positions=_positions_fixture(),
        initial_capital=777_777.0,
    )
    assert account.positions[TSCode("600000.SH")].last_price.value == 10.5
    # 无 last_known_price → 买入成交价兜底（不是后复权绩效价 77.7）
    assert account.positions[TSCode("000001.SZ")].last_price.value == 20.0


def test_adapter_immutable_outer_containers():
    """weights/positions/lots 不可变外层容器：写入尝试必须抛错。"""
    adapter = _make_spy_adapter()
    _call_adapter(adapter)
    target = adapter._merger.calls[0][0]["value"]
    account = adapter._merger.calls[0][1]
    assert isinstance(target.target_weights, MappingProxyType)
    assert isinstance(account.positions, MappingProxyType)
    assert isinstance(account.positions[TSCode("600000.SH")].lots, tuple)
    with pytest.raises(TypeError):
        target.target_weights[TSCode("000002.SZ")] = Weight(0.1)
    with pytest.raises(TypeError):
        account.positions[TSCode("000002.SZ")] = None


def test_adapter_fresh_snapshot_per_call():
    """逐次接缝新建快照：两次调用间状态变化必须反映到第二个 Account。"""
    adapter = _make_spy_adapter()
    _call_adapter(adapter, cash=500000.0)
    _call_adapter(adapter, cash=300000.0, positions={})  # 状态已变：现金下降、清仓
    first, second = adapter._merger.calls[0][1], adapter._merger.calls[1][1]
    assert first is not second  # 不复用装配时/上次快照
    assert first.cash.value == 500000.0 and second.cash.value == 300000.0
    assert len(first.positions) == 2 and len(second.positions) == 0
    assert adapter.call_count == 2


def test_adapter_weight_roundtrip_lossless():
    """权重无损往返：键序、原值对象身份与类型逐项锁定（不排序/不 str/不强转）。"""
    weights = {"600000.SH": 0.9, "000001.SZ": 0.4, "300750.SZ": 1.0 / 3.0}
    result = _call_adapter(_make_adapter(), weights=weights)
    assert list(result) == list(weights)  # 插入序（非字典序）保持
    for code, weight in weights.items():
        assert result[code] is weight, f"{code} 原浮点对象身份丢失"
        assert type(result[code]) is float, f"{code} 浮点类型被强转"
    # 锚定：插入序确实不是字典序（防顺序断言空转）
    assert list(weights) != sorted(weights)


def test_adapter_missing_critical_fields_fail_fast():
    """缺键/None 值：fail-fast 含股票与键名（阶段=P2b），且在真实 merge 前（计数=0）。"""
    for missing_key in ("buy_date", "buy_cost_cash", "buy_trade_price", "shares"):
        positions = {
            "600000.SH": {
                "shares": 100,
                "buy_date": _D0,
                "buy_trade_price": 10.0,
                "buy_cost_cash": 1000.0,
            }
        }
        del positions["600000.SH"][missing_key]
        adapter = _make_adapter()
        with pytest.raises(ValueError) as exc:
            _call_adapter(adapter, positions=positions)
        msg = str(exc.value)
        assert "600000.SH" in msg and missing_key in msg and "阶段=P2b" in msg
        assert adapter.call_count == 0  # 在真实 merge() 之前失败
    # None 值同样拒绝（禁止静默 NaN/伪造）
    positions = {
        "600000.SH": {
            "shares": 100,
            "buy_date": None,
            "buy_trade_price": 10.0,
            "buy_cost_cash": 1000.0,
        }
    }
    with pytest.raises(ValueError, match="buy_date"):
        _call_adapter(_make_adapter(), positions=positions)


def test_adapter_missing_value_markers_fail_fast():
    """NaN/NaT/NA 缺失标记穿透防护（R4-T2-01 复核条件）：真实 merge 前失败、计数=0。

    ``Position`` 成本聚合校验对 NaN 不自拒（``abs(nan-nan) > 1e-6`` 为 False），
    故必须在适配器字段边界拒绝；禁止补零、伪造日期或转为占位值。
    """
    base = {"shares": 100, "buy_date": _D0, "buy_trade_price": 10.0, "buy_cost_cash": 1000.0}
    for key in ("buy_date", "buy_cost_cash", "buy_trade_price", "shares"):
        for marker in (float("nan"), pd.NaT, pd.NA):
            positions = {"600000.SH": dict(base, **{key: marker})}
            adapter = _make_adapter()
            with pytest.raises(ValueError) as exc:
                _call_adapter(adapter, positions=positions)
            assert key in str(exc.value) and "600000.SH" in str(exc.value)
            assert adapter.call_count == 0
    # 转换后兜底：字符串 "NaT" 经 pd.Timestamp 转换为 NaT，仍须拒绝
    positions = {"600000.SH": dict(base, buy_date="NaT")}
    with pytest.raises(ValueError):
        _call_adapter(_make_adapter(), positions=positions)


def test_adapter_total_value_none_fails_fast():
    """_last_rebalance_nav 缺失（None）→ 报错停止该次启用运行，且 merge 计数=0。"""
    adapter = _make_adapter()
    with pytest.raises(RuntimeError) as exc:
        _call_adapter(adapter, total_value=None)
    assert "_last_rebalance_nav" in str(exc.value)
    assert adapter.call_count == 0  # 到达真实 merge() 之前失败，不计数


def test_adapter_call_count_real_merge_only():
    """D7 计数语义（单元级）：每次成功进入真实 merge() 计 1，从 0 起。"""
    adapter = _make_adapter()
    assert adapter.call_count == 0
    _call_adapter(adapter)
    assert adapter.call_count == 1
    _call_adapter(adapter)
    assert adapter.call_count == 2


def test_adapter_does_not_mutate_legacy_inputs():
    """只读断言：旧持仓 dict 与权重 dict 在调用前后深比对一致（含无写回新增键）。"""
    weights = {"600000.SH": 0.6, "000001.SZ": 0.4}
    positions = _positions_fixture()
    weights_snapshot = dict(weights)
    positions_snapshot = {stock: dict(info) for stock, info in positions.items()}
    _call_adapter(_make_adapter(), weights=weights, positions=positions)
    assert weights == weights_snapshot
    for stock, info in positions.items():
        assert (
            info == positions_snapshot[stock]
        ), f"{stock} 旧持仓被改写（含 last_known_price 写回）"
        assert set(info) == set(positions_snapshot[stock]), f"{stock} 键集被新增/删除"


def test_adapter_port_structural_compatibility():
    """LegacyTargetMergePort 结构兼容 + 完整签名等同（参数名/种类/注解/顺序/返回）。"""
    adapter = _make_adapter()
    assert isinstance(adapter, LegacyTargetMergePort)
    assert _full_signature(LegacyMergerAdapter.merge_legacy_batch_targets) == _full_signature(
        LegacyTargetMergePort.merge_legacy_batch_targets
    )


def test_adapter_never_calls_portfolio_valuation():
    """零新增估值/零写回（D5；R2-T2-07 处置：补公开函数名与 Call 形态逃逸位）。

    字符串禁令覆盖：``calculate_portfolio_value``（含下划线私有形，子串命中）、
    ``position_market_value``、``setdefault(`` / ``.update(`` / ``__setitem__``
    （持仓写回的 Call 形态逃逸位）；AST 禁令锁定 ``info["last_known_price"] =``
    下标写回。残余逃逸面（别名/变量键/属性中转等）归 T3 门 2 AST 工具。
    """
    src = Path(adapter_mod.__file__).read_text(encoding="utf-8")
    banned = (
        "calculate_portfolio_value",
        "position_market_value",
        "setdefault(",
        ".update(",
        "__setitem__",
    )
    for token in banned:
        assert token not in src, f"适配器出现禁止形态: {token}"
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            assert node.slice.value != "last_known_price" or not isinstance(
                node.ctx, ast.Store
            ), "适配器不得写回 last_known_price"
    # 反变异自证：注入各逃逸形态必须被对应禁令捕获
    for token in banned:
        with pytest.raises(AssertionError):
            test_src = src + f"\n{token}\n"
            for banned_token in banned:
                assert banned_token not in test_src, f"适配器出现禁止形态: {banned_token}"
    # 锚定：calculate_nav 调用确实存在于真实源码（防线不是空匹配）
    assert "calculate_nav(" in src


# ══════════════════ C. 接缝与三层零新增（D3/D4 方案 A） ══════════════════


class _SeamStub:
    """``_normalize_and_cap_signals`` 直驱桩（模块级函数以 self 首参调用）。"""

    merge_port = None  # 类级默认（与引擎方案 A 同形态）
    max_weight_per_stock: Optional[float] = None
    verbose = False

    def __init__(self) -> None:
        self.normalize_inputs = []
        self.capital = 250000.0

    @property
    def current_capital(self):
        return self.capital

    @property
    def _last_rebalance_nav(self):
        return 900000.0

    @property
    def positions(self):
        return {}

    @property
    def initial_capital(self):
        return 800000.0

    def _normalize_signals(self, signals, date):
        self.normalize_inputs.append(dict(signals))
        return dict(signals)

    def _get_tranche_capital_fraction(self, tranche_idx):
        return 1.0


class _RecordingPort:
    """记录接缝实参并原样返回权重的端口桩。"""

    def __init__(self) -> None:
        self.calls = []

    def merge_legacy_batch_targets(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["weights"]


def test_seam_closed_state_strictly_skipped():
    """默认关闭：merge_port=None 时接缝严格跳过（无调用、输出与归一化结果同值同序）。"""
    stub = _SeamStub()
    signals = {"600000.SH": 0.6, "000001.SZ": 0.4}
    result = signal_execution._normalize_and_cap_signals(stub, signals, _D0, 0)
    assert stub.normalize_inputs == [signals]  # 归一化确实发生
    assert result == signals and list(result) == list(signals)


def test_seam_open_state_order_normalize_merge_cap():
    """开启：归一化 → 合并 → 限权 的顺序锁定（合并读 pre-cap 权重、限权作用于合并结果）。"""
    stub = _SeamStub()
    stub.max_weight_per_stock = 0.5
    port = _RecordingPort()
    stub.merge_port = port
    signals = {"600000.SH": 0.8, "000001.SZ": 0.2}
    result = signal_execution._normalize_and_cap_signals(stub, signals, _D0, 0)
    # 合并接缝恰调用一次，收到归一化后、限权前的当批权重
    assert len(port.calls) == 1
    call = port.calls[0]
    assert call["weights"] == {"600000.SH": 0.8, "000001.SZ": 0.2}
    assert call["date"] is _D0
    assert call["cash"] == 250000.0
    assert call["total_value"] == 900000.0
    assert call["positions"] == {}
    assert call["initial_capital"] == 800000.0
    # 限权作用于合并返回之后（0.8 超 0.5 上限 ⇒ 被截断重归一）
    assert result != call["weights"]
    assert max(result.values()) <= 0.5 + 1e-12
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_engine_class_level_merge_port_default():
    """方案 A 三层零新增：类级默认 None、ML 继承不重定义、实例 vars 零新增。"""
    assert BacktestEngine.__dict__["merge_port"] is None  # 唯一定义点在 BacktestEngine 类体
    assert "merge_port" not in BacktestEngineML.__dict__  # ML 继承，不重定义
    assert BacktestEngineML.merge_port is None
    engine = object.__new__(BacktestEngine)  # 绕过 __init__ 直接看实例命名空间
    assert "merge_port" not in vars(engine), "实例属性不得新增（t4 锚）"


def test_seam_order_provable_with_non_identity_stubs():
    """接缝顺序分辨力（R2-T2-06 处置）：非恒等归一化桩 + 非恒等端口锁两端次序。

    - 「归一化→合并」：归一化把 {8.0, 2.0} 变为 {0.8, 0.2}，端口必须收到
      **归一化后**的权重（前移变异会投递原始值而被捕获）；
    - 「合并→限权」：端口返回**不同**权重 {0.45, 0.35, 0.2}，结果必须等于
      ``cap(端口返回值)`` 而非 ``cap(归一化结果)``（丢弃返回值变异被捕获）。
    """

    class _ScalingStub(_SeamStub):
        def _normalize_signals(self, signals, date):
            self.normalize_inputs.append(dict(signals))
            total = sum(signals.values())
            return {k: v / total for k, v in signals.items()}  # 非恒等：归一化

    class _NonIdentityPort:
        PORT_OUTPUT = {"600000.SH": 0.45, "000001.SZ": 0.35, "300750.SZ": 0.2}

        def __init__(self) -> None:
            self.calls = []

        def merge_legacy_batch_targets(self, **kwargs):
            self.calls.append(kwargs)
            return dict(self.PORT_OUTPUT)

    stub = _ScalingStub()
    stub.max_weight_per_stock = 0.4
    port = _NonIdentityPort()
    stub.merge_port = port
    result = signal_execution._normalize_and_cap_signals(
        stub, {"600000.SH": 8.0, "000001.SZ": 2.0}, _D0, 0
    )
    # 归一化→合并：端口收到归一化后权重
    assert port.calls[0]["weights"] == {"600000.SH": 0.8, "000001.SZ": 0.2}
    # 合并→限权：结果 == 对端口返回值限权（与对归一化结果限权不同，锚定可区分）
    cap = resolve_tranche_weight_cap(0.4, 1.0)
    expected = cap_and_normalize_weights(
        dict(_NonIdentityPort.PORT_OUTPUT), max_weight_per_stock=cap, verbose=False
    )
    cap_of_normalized = cap_and_normalize_weights(
        {"600000.SH": 0.8, "000001.SZ": 0.2}, max_weight_per_stock=cap, verbose=False
    )
    assert result == expected
    assert result != cap_of_normalized  # 锚定：两条路径确实可区分（防断言空转）


def test_seam_tranche_batches_share_same_seam():
    """分批调仓（tranche_idx>0）同样经过唯一接缝：端口收到该批归一化权重。

    S3-T2-04 处置（任务卡「常规/提前/分批全部锁定」的分批证据）：tranche 语义
    为当批相对权重；分批差异只经既有 ``_get_tranche_capital_fraction`` 进入
    限权分支，接缝输入不变。提前调仓路径由「共用 ``_generate_signal``」结构性
    覆盖 + T4 端到端计数证据兑现（§14 登记）。
    """
    stub = _SeamStub()
    port = _RecordingPort()
    stub.merge_port = port
    for tranche_idx in (0, 1):
        result = signal_execution._normalize_and_cap_signals(
            stub, {"600000.SH": 0.8, "000001.SZ": 0.2}, _D0, tranche_idx
        )
        assert result == {"600000.SH": 0.8, "000001.SZ": 0.2}  # 无限权时直通
    assert len(port.calls) == 2
    assert port.calls[0]["weights"] == {"600000.SH": 0.8, "000001.SZ": 0.2}
    assert port.calls[1]["weights"] == {"600000.SH": 0.8, "000001.SZ": 0.2}


# ── 补位 / T+1 局部计划分类回归锁（R1-T2-01） ──


class _MiniUniverse:
    def __init__(self, stocks) -> None:
        self._stocks = stocks

    def get_stocks(self, date, quote_data=None):
        return list(self._stocks)


class _MiniSignal:
    def generate_ranked(self, date, universe, data):
        return [("000001.SZ", 0.9), ("300750.SZ", 0.8)]


class _RefillStub:
    """``_queue_condition_sell_refill_signal`` 直驱桩（补位场景最小依赖面）。"""

    def __init__(self, port=None) -> None:
        self.merge_port = port
        self.enable_position_completion = True
        self.pending_signals = {}
        self.positions = {"600000.SH": {"shares": 100}}  # 被卖槽位
        self.max_per_industry = None
        self.verbose = False
        self.universe = _MiniUniverse(["600000.SH", "000001.SZ", "300750.SZ"])
        self.signal = _MiniSignal()

    def _build_signal_data(self, date):
        return {}

    def _post_filter_candidates(self, ranked_candidates, date):
        return ranked_candidates

    def _get_target_position_count(self):
        return 2


def test_refill_plan_never_enters_merger_seam():
    """条件卖出补位 = 执行阶段局部计划（D3）：不经合并接缝、两臂输出一致。"""
    price_data = pd.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["20260105"]})
    date_to_idx = {_D0: 0, _D1: 1}
    slot_weights = [{"stock": "600000.SH", "weight": 0.5}]
    drive = BacktestSignalExecutionMixin._queue_condition_sell_refill_signal

    closed = _RefillStub(port=None)
    drive(closed, _D0, slot_weights, price_data, date_to_idx)
    # 锚定：补位计划确实生成（守卫早退会使后续断言空转）
    assert _D0 in closed.pending_signals
    assert closed.pending_signals[_D0]["signals"] == {"000001.SZ": 0.5}

    port = _RecordingPort()
    opened = _RefillStub(port=port)
    drive(opened, _D0, slot_weights, price_data, date_to_idx)
    assert port.calls == [], "补位局部计划不得进入合并接缝（merge 调用增量必须为 0）"
    assert _D0 in opened.pending_signals
    assert opened.pending_signals == closed.pending_signals, "开启/关闭两臂补位计划必须一致"


def _slot_path_offenders(src: str) -> list:
    """局部槽位路径源码中的合并接缝引用（分类锁扫描原语）。"""
    return [token for token in ("merge_port", "_normalize_and_cap_signals") if token in src]


def test_local_slot_plan_paths_never_reference_merger():
    """分类锁（源码级）：补位方法与 T+1 顺位替换所在模块零引用合并接缝。"""
    refill_src = inspect.getsource(BacktestSignalExecutionMixin._queue_condition_sell_refill_signal)
    assert _slot_path_offenders(refill_src) == []
    # T+1 顺位替换（_execute_slot_matched_buys / fill_slots_from_candidates）所在模块
    # （S3-T2-01 处置：与补位共用同一扫描原语，merge_port 与接缝函数双 token 零引用）
    buy_src = Path(inspect.getfile(buy_execution)).read_text(encoding="utf-8")
    assert _slot_path_offenders(buy_src) == []
    # 反变异自证：注入接缝引用后扫描必须报警；扫描原语对已知阳性串非空转
    injected = refill_src + "\n        signals = self.merge_port.merge_legacy_batch_targets()\n"
    assert _slot_path_offenders(injected) != []
    assert _slot_path_offenders("self.merge_port") == ["merge_port"]
    assert _slot_path_offenders("_normalize_and_cap_signals(x)") == ["_normalize_and_cap_signals"]


# ══════════════════ D. 反变异自证 ══════════════════


class _CachedAccountMutant(LegacyMergerAdapter):
    """装配时缓存 Account、逐次复用（违反 D5 逐次新建快照）。"""

    def _build_account(self, **kwargs):
        if getattr(self, "_cached", None) is None:
            self._cached = super()._build_account(**kwargs)
        return self._cached


class _SortingWeightsMutant(LegacyMergerAdapter):
    """包装时按代码排序权重（破坏插入序）。"""

    def merge_legacy_batch_targets(self, *, weights, **kwargs):
        return super().merge_legacy_batch_targets(weights=dict(sorted(weights.items())), **kwargs)


class _NumpyFloatCastMutant(LegacyMergerAdapter):
    """解包时强转 np.float64（改变后续限权输入的浮点类型）。"""

    def merge_legacy_batch_targets(self, **kwargs):
        result = super().merge_legacy_batch_targets(**kwargs)
        return {code: np.float64(weight) for code, weight in result.items()}


class _PnlPriceMutant(LegacyMergerAdapter):
    """last_price 混入后复权 buy_pnl_price（D5 明令禁止）。"""

    def _build_account(self, *, positions, **kwargs):
        patched = {
            stock: {**info, "last_known_price": info.get("buy_pnl_price")}
            for stock, info in positions.items()
        }
        return super()._build_account(positions=patched, **kwargs)


class _CachedPositionsMutant(LegacyMergerAdapter):
    """半缓存变异（R2-T2-02 M-A）：只缓存 positions 映射，cash/total_value 逐次新建。"""

    def _build_account(self, *, positions, **kwargs):
        if getattr(self, "_cached_positions", None) is None:
            self._cached_positions = positions
        return super()._build_account(positions=self._cached_positions, **kwargs)


class _CachedTotalValueMutant(LegacyMergerAdapter):
    """半缓存变异（R2-T2-02 M-B）：只缓存 total_value。"""

    def _build_account(self, *, total_value, **kwargs):
        if getattr(self, "_cached_tv", None) is None:
            self._cached_tv = total_value
        return super()._build_account(total_value=self._cached_tv, **kwargs)


class _SelectivePnlMutant(LegacyMergerAdapter):
    """选择性绩效价混入（R2-T2-03 M-D）：仅当持仓已有 last_known_price 时才覆盖。"""

    def _build_account(self, *, positions, **kwargs):
        patched = {
            stock: (
                {**info, "last_known_price": info["buy_pnl_price"]}
                if "last_known_price" in info
                else dict(info)
            )
            for stock, info in positions.items()
        }
        return super()._build_account(positions=patched, **kwargs)


def _assert_fresh_snapshot(adapter) -> None:
    """逐次新快照断言组（正例与反变异复用）。

    R2-T2-02 处置：两次调用 cash/total_value/positions **全部**不同，
    逐字段断言——半缓存变异（只缓存其中一项）也不得穿透。
    """
    _call_adapter(adapter, cash=500000.0, total_value=1_234_567.89)
    _call_adapter(adapter, cash=300000.0, total_value=900_000.0, positions={})
    first, second = adapter._merger.calls[0][1], adapter._merger.calls[1][1]
    assert first is not second, "不得复用装配时/上次 Account"
    assert first.cash.value == 500000.0 and second.cash.value == 300000.0
    assert first.total_value.value == 1_234_567.89 and second.total_value.value == 900_000.0
    assert len(first.positions) == 2 and len(second.positions) == 0


def _assert_lossless_roundtrip(adapter) -> None:
    weights = {"600000.SH": 0.9, "000001.SZ": 0.4}
    result = _call_adapter(adapter, weights=weights)
    assert list(result) == list(weights), "键序被改变"
    for code, weight in weights.items():
        assert result[code] is weight and type(result[code]) is float, "值对象身份/类型被改写"


def _assert_never_pnl_price(adapter) -> None:
    """buy_pnl_price 永不混入（R2-T2-03 处置：缓存价优先路径与兜底路径双锁）。"""
    account = adapter._build_account(
        cash=1.0, total_value=2.0, positions=_positions_fixture(), initial_capital=777_777.0
    )
    # 缓存价优先路径：last_known_price=10.5 不得被 buy_pnl_price=88.8 覆盖
    assert account.positions[TSCode("600000.SH")].last_price.value == 10.5
    # 兜底路径：无缓存价时取 buy_trade_price=20.0（不是绩效价 77.7）
    assert account.positions[TSCode("000001.SZ")].last_price.value == 20.0


def _assert_no_host_nav_division(src: str) -> None:
    """宿主 NAV 除法禁令（轻量 AST）：不得出现以 initial_capital 为右操作数的除法。"""
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Name)
            and node.right.id == "initial_capital"
        ):
            raise AssertionError("宿主适配器不得自行除以 initial_capital（NAV 归账本层）")


def test_anti_mutation_adapter_variants_detected():
    """适配器七类变异必须被对应断言捕获（含 R2-T2-02/03 半缓存与选择性混入）。"""
    detections = (
        (_CachedAccountMutant, _assert_fresh_snapshot),
        (_CachedPositionsMutant, _assert_fresh_snapshot),
        (_CachedTotalValueMutant, _assert_fresh_snapshot),
        (_SortingWeightsMutant, _assert_lossless_roundtrip),
        (_NumpyFloatCastMutant, _assert_lossless_roundtrip),
        (_PnlPriceMutant, _assert_never_pnl_price),
        (_SelectivePnlMutant, _assert_never_pnl_price),
    )
    for mutant_cls, assertion in detections:
        # 锚定（防空转）：对应断言组对真实适配器必须成立
        assertion(_make_spy_adapter())
        mutant = mutant_cls("value")
        mutant._merger = _SpyMerger()
        with pytest.raises(AssertionError):
            assertion(mutant)


def test_anti_mutation_host_nav_division_detected():
    """宿主私算 NAV 变异：源码注入除法必须被 AST 禁令捕获；真实源码当前通过。"""
    src = Path(adapter_mod.__file__).read_text(encoding="utf-8")
    _assert_no_host_nav_division(src)  # 正例：真实适配器无宿主除法
    injected = src + "\nx = total_value / initial_capital\n"  # 变异：宿主私算 NAV
    with pytest.raises(AssertionError):
        _assert_no_host_nav_division(injected)
    # 锚定：calculate_nav 调用确实存在于真实源码（防线不是空匹配）
    assert "calculate_nav(" in src
