"""P2a-T5 新旧逐位一致等价测试（回测引擎 7 mixin 组装体 + ML 信号喂入组装体）。

T0 实施规划 v4 §4-T5 验收口径（仿 T4 ``test_v2_p2a_t4_equivalence.py`` 范式）：

- A 端到端新旧逐位一致：同一合成场景（静态股票池 stub、确定性分数信号 stub、
  合成 price_data）分别跑旧 ``backtest/engine.py::BacktestEngine`` 与新
  ``v2/core/execution/engine.py::BacktestEngine``，成交记录与净值曲线精确一致；
  场景覆盖 ① 正常买入（T+1 收盘）② 持有期到期卖出 ③ 仓位补齐成功
  ④ 延迟订单（卖出跌停入队重试成交；买入调用级覆盖重试/超次/超期）。
- B 黄金调用点清单（N3）：固化主循环钩子调用序列，新侧 == 旧侧剔除 4 个
  exposure 钩子（旧侧为 no-op 守卫）；含止损启用场景（止损两钩子相对
  条件卖出执行的逐日次序锁定）与「止损/条件卖出换序」变异自证。
- C 退役符号扫描（D6）：清单由三个旧 exposure 源文件**运行时 AST 自动提取**
  （类名 / 类方法名 / 模块级函数名 / 类方法体内 self.<attr>= 与注解赋值字段名 /
  类体注解字段名），与文件内冻结预期集做完整集合相等断言；以提取清单扫描
  ``v2/core``（AST Attribute/Name/Import 口径，ImportFrom 含**父包导入归属**
  判定——绝对与相对形态均归一，R2-T5-R2-01）0 残留；内存源码变异反例
  永久回归（漏项符号与父包导入必识别；docstring/字符串/关键字参数名/
  无关模块导入不误报）。新引擎 MRO = 旧清单摘除 3 个 exposure mixin。
- D 3 条退役隔离测试（不依赖退役组件）。
- E 退役配置非默认值 fail-fast（TypeError / AttributeError）。
- F 接线义务隔离断言（T4 评审 R1-5/R1-6：pending_order_event_sink 接线 +
  min_buy 短路不求值组合市值）。
- G 估值回退余 2 处逐点等价（D4：旧 buy_execution.py:144 嵌套函数 → 新模块级
  helper ``_get_position_weight``；旧 signal_execution.py:73 → 新
  ``_get_position_weight_for_planning``）。
- H 条件卖出 3 触发器 sell_type 一致 + 减仓（risk_trim）不进延迟队列；
  同日止损 + 持有期到期的逐笔有序比对（A/stop_loss 先于 B/holding_period）。
- J reporting 拆分等价性固化：``_build_daily_warning_logs`` /
  ``_build_daily_signal_log`` / ``_format_rebalance_decision_summary`` /
  ``_build_daily_trade_log`` / ``_format_daily_progress_log`` 合成输入直驱，
  新旧逐串相等（R1-T5-01 + R3-T5-03）。
- I 禁止真实配置读取永久回归（同进程 pytest.main 复跑，范式同 T4）。

不依赖真实配置与真实数据：所有输入为合成 fixture；``CostModel`` 两侧的
``get_cost_settings`` 经 ``synthetic_cost_settings`` 打为同一合成读取函数；
停牌日历经合成 ``data_storage``（空 suspend 表）供给，不触真实存储。
"""

import ast
import copy
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# ── 旧模块（生产在用，迁移期不修改） ──
from src.lazybull.backtest import buy_execution as old_buy_mod
from src.lazybull.backtest import exposure_override as old_exposure_override
from src.lazybull.backtest import exposure_replenish as old_exposure_replenish
from src.lazybull.backtest import exposure_trim as old_exposure_trim
from src.lazybull.backtest import reporting as old_reporting
from src.lazybull.backtest.engine import BacktestEngine as OldEngine
from src.lazybull.backtest.engine_ml import BacktestEngineML as OldEngineML
from src.lazybull.common import cost as old_cost
from src.lazybull.common.cost import CostModel as OldCostModel
from src.lazybull.risk.stop_loss import StopLossConfig as OldStopLossConfig
from src.lazybull.signals import base as old_signal_base
from src.lazybull.universe.base import Universe
from src.lazybull.v2.common import cost as new_cost
from src.lazybull.v2.common.cost import CostModel as NewCostModel
from src.lazybull.v2.common.rules.stop_loss import StopLossConfig as NewStopLossConfig

# ── v2 部件（P2a-T5 交付） ──
from src.lazybull.v2.core.execution import buy_execution as new_buy_mod
from src.lazybull.v2.core.execution import reporting as new_reporting
from src.lazybull.v2.core.execution import sell_execution as new_sell_mod
from src.lazybull.v2.core.execution.engine import BacktestEngine as NewEngine
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML as NewEngineML
from src.lazybull.v2.core.signal import base as new_signal_base

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 合成成本参数（两侧 CostModel 显式同参；构造器的无条件配置读取由 fixture 打桩）
_COST_SETTINGS = {
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.001,
}

_TRADING_DATES = list(pd.bdate_range("2026-01-05", periods=25))
_STOCKS = ["A", "B", "C", "D"]
_BASE_PRICES = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}

# 旧侧主循环逐日调用、新侧已摘除的 4 个 exposure 钩子（B0 基线下为 no-op 守卫）
_EXPOSURE_HOOKS = (
    "_execute_pending_exposure_trims",
    "_queue_exposure_trim",
    "_queue_exposure_replenish",
    "_execute_pending_exposure_replenishes",
)

# ── D6 退役符号清单（R2-T5-01/R3-T5-01 处置：运行时 AST 自动提取 + 冻结集合断言） ──
# 提取口径：三个旧 exposure 源文件的
#   ① 模块级函数名（def）；
#   ② 类名（ClassDef）；
#   ③ 类方法名（``__init__`` 构造器名排除，但其体内字段照收）；
#   ④ 类方法体内 ``self.<attr> =`` / ``self.<attr>: T =`` 赋值字段名；
#   ⑤ 类体注解/赋值字段名（dataclass 字段与 mixin 类级属性）。
_RETIRED_SOURCE_MODULES = (old_exposure_override, old_exposure_trim, old_exposure_replenish)

# 退役模块名冻结集（单列，与定义侧符号集语义区分；R2-T5-R2-01）：
# 供 ``ImportFrom`` 父包导入形态（from src.lazybull.backtest import exposure_trim）
# 与完整路径 import 形态判定归属；运行时由 _RETIRED_SOURCE_MODULES 派生比对防漂移
_RETIRED_MODULE_NAMES = frozenset({"exposure_override", "exposure_trim", "exposure_replenish"})

# 退役三模块的父包绝对名（父包导入归属判定的基准）
_RETIRED_PARENT_PACKAGE = "src.lazybull.backtest"


def _extract_retired_symbols() -> frozenset:
    """从三个旧 exposure 源文件 AST 提取定义侧全部符号（见上方口径）。"""
    symbols = set()
    for module in _RETIRED_SOURCE_MODULES:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:  # 仅模块顶层
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


# 冻结预期集（逐字列出全部 41 项；旧文件未来被改动 ⇒ 提取集漂移 ⇒ 测试报警）：
# 类名（4）+ 模块级函数（1）+ 类方法（19，含 ExposureOverrideStats.record）
# + self 赋值/注解状态字段（13）+ ExposureOverrideStats 类体字段（4）
_FROZEN_RETIRED_SYMBOLS = frozenset(
    {
        # 类名（4）
        "BacktestExposureOverrideMixin",
        "BacktestExposureTrimMixin",
        "BacktestExposureReplenishMixin",
        "ExposureOverrideStats",
        # 模块级函数（1）
        "load_exposure_table",
        # 类方法（19）
        "record",
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
        # self.<attr> = / self.<attr>: T = 状态字段（13）
        "exposure_table",
        "exposure_stats",
        "_exposure_missing_dates",
        "exposure_policy_provider",
        "exposure_policy_summary",
        "exposure_budget_discount_replenish",
        "exposure_trim_tolerance",
        "exposure_replenish_enabled",
        "exposure_release_budget",
        "pending_exposure_trims",
        "exposure_trim_stats",
        "pending_exposure_replenishes",
        "exposure_replenish_stats",
        # ExposureOverrideStats 类体字段（4）
        "covered_days",
        "missing_days",
        "reduced_days",
        "missing_samples",
    }
)

# 扫描排除集（文档化 carve-out）：``record`` 是 ExposureOverrideStats 的内部计数
# 方法名，与 v2/core 既有无关代码的通用局部变量同名（run_loop 的 loguru filter
# lambda 参数、reporting._collect_deferred_log 的局部变量），全词扫描必然误报；
# 该名字不承担退役组件接线语义（退役面由类名/字段名/mixin 方法名完整覆盖）。
# 冻结为恰含 "record"：排除集不得悄悄扩大。
_SCAN_EXCLUDED_SYMBOLS = frozenset({"record"})

# 迁移面扫描用清单 = 提取全集 − 文档化排除集
_RETIRED_SYMBOLS = _extract_retired_symbols() - _SCAN_EXCLUDED_SYMBOLS


@pytest.fixture()
def synthetic_cost_settings(monkeypatch):
    """合成成本配置（范式同 T4 ``synthetic_cost_settings``，R2-T4-01 处置）。

    新旧 ``CostModel`` 构造器即使四参全显式传入，也会**无条件**调用
    ``get_cost_settings()`` 读取共享配置单例。本 fixture 把新旧两侧成本模块
    各自绑定的 ``get_cost_settings`` 打为同一合成读取函数，pytest 结束自动
    恢复；返回读取计数供锚定断言。"""
    reads: List[int] = []

    def _stub() -> dict:
        reads.append(1)
        return dict(_COST_SETTINGS)

    monkeypatch.setattr(old_cost, "get_cost_settings", _stub)
    monkeypatch.setattr(new_cost, "get_cost_settings", _stub)
    return {"settings": _COST_SETTINGS, "reads": reads}


# ══════════════════ 合成场景基础设施 ══════════════════


class _FakeStorage:
    """Storage 桩：suspend 数据集恒返回空表（无停牌），免真实数据依赖。"""

    def load_raw_by_date(self, dataset: str, trade_date: str):
        assert dataset == "suspend"
        return pd.DataFrame({"ts_code": [], "suspend_type": []})


class _StaticUniverse(Universe):
    """静态股票池 stub：get_stocks 恒返回固定列表（新旧引擎共用的 Universe 基类）。"""

    def __init__(self, stocks: List[str]):
        super().__init__("static_stub")
        self._stocks = list(stocks)

    def get_stocks(self, date: pd.Timestamp, quote_data: Optional[pd.DataFrame] = None):
        return list(self._stocks)


def _make_det_signal(base, top_n: int, plan: Dict[str, List[Tuple[str, float]]], default):
    """构造确定性分数信号 stub（按日期查表，缺省用 default；过滤到当日股票池）。"""

    class _DeterministicSignal(base):
        def __init__(self):
            super().__init__("deterministic_stub")
            self.top_n = top_n

        def generate(self, date, universe, data):
            return dict(self.generate_ranked(date, universe, data))

        def generate_ranked(self, date, universe, data):
            ranked = plan.get(date.strftime("%Y%m%d"), default)
            universe_set = set(universe)
            return [(stock, score) for stock, score in ranked if stock in universe_set]

    return _DeterministicSignal()


def _flat_closes() -> Dict[str, List[float]]:
    """确定性价格序列：每股温和上行（每日 +0.1%），无随机。"""
    n = len(_TRADING_DATES)
    return {s: [b * (1 + 0.001 * i) for i in range(n)] for s, b in _BASE_PRICES.items()}


def _make_price_data(
    closes: Dict[str, List[float]],
    limit_up: frozenset = frozenset(),
    limit_down: frozenset = frozenset(),
) -> pd.DataFrame:
    """合成 price_data：close/open/close_adj/open_adj + 交易状态列（默认全可交易）。"""
    rows = []
    for i, date in enumerate(_TRADING_DATES):
        date_str = date.strftime("%Y%m%d")
        for stock, series in closes.items():
            close = series[i]
            rows.append(
                {
                    "ts_code": stock,
                    "trade_date": date_str,
                    "close": close,
                    "open": round(close * 0.999, 4),
                    "close_adj": close,
                    "open_adj": round(close * 0.999, 4),
                    "is_suspended": 0,
                    "is_limit_up": 1 if (stock, date_str) in limit_up else 0,
                    "is_limit_down": 1 if (stock, date_str) in limit_down else 0,
                }
            )
    return pd.DataFrame(rows)


def _make_old_engine(signal, cls=None, **overrides) -> OldEngine:
    """构造旧引擎（合成 universe/成本/停牌存储，不触真实配置与数据）。"""
    kwargs = dict(
        universe=_StaticUniverse(_STOCKS),
        signal=signal,
        initial_capital=1_000_000.0,
        rebalance_freq=5,
        verbose=False,
        cost_model=OldCostModel(**_COST_SETTINGS),
        data_storage=_FakeStorage(),
    )
    kwargs.update(overrides)
    return (cls or OldEngine)(**kwargs)


def _make_new_engine(signal, cls=None, **overrides) -> NewEngine:
    """构造新引擎（7 mixin 组装体；与 _make_old_engine 同参数）。"""
    kwargs = dict(
        universe=_StaticUniverse(_STOCKS),
        signal=signal,
        initial_capital=1_000_000.0,
        rebalance_freq=5,
        verbose=False,
        cost_model=NewCostModel(**_COST_SETTINGS),
        data_storage=_FakeStorage(),
    )
    kwargs.update(overrides)
    return (cls or NewEngine)(**kwargs)


def _run_pair(
    plan: Dict[str, List[Tuple[str, float]]],
    default: List[Tuple[str, float]],
    price_data: pd.DataFrame,
    old_overrides: Optional[Dict] = None,
    new_overrides: Optional[Dict] = None,
):
    """同一场景分别跑新旧引擎（top_n=2），返回 (旧引擎, 新引擎, 旧净值, 新净值)。"""
    old_engine = _make_old_engine(
        _make_det_signal(old_signal_base.Signal, 2, plan, default), **(old_overrides or {})
    )
    new_engine = _make_new_engine(
        _make_det_signal(new_signal_base.Signal, 2, plan, default), **(new_overrides or {})
    )
    nav_old = old_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    nav_new = new_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    return old_engine, new_engine, nav_old, nav_new


def _assert_runs_identical(old_engine, new_engine, nav_old, nav_new) -> None:
    """端到端逐位一致：成交记录与净值曲线精确相等（check_exact）。"""
    pd.testing.assert_frame_equal(
        old_engine.get_trades(), new_engine.get_trades(), check_exact=True
    )
    pd.testing.assert_frame_equal(nav_old, nav_new, check_exact=True)


def _scenario_constant_ab():
    """场景①②：候选恒为 [A, B]（正常买入 + 持有期到期卖出循环）。"""
    return _run_pair({}, [("A", 0.9), ("B", 0.8)], _make_price_data(_flat_closes()))


def _scenario_rotation():
    """场景（换出）：每 5 个交易日候选在 AB/CD 间轮换 ⇒ 调仓日换出卖出。"""
    plan = {}
    for i, date in enumerate(_TRADING_DATES):
        plan[date.strftime("%Y%m%d")] = (
            [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
        )
    return _run_pair(
        plan, plan[_TRADING_DATES[0].strftime("%Y%m%d")], _make_price_data(_flat_closes())
    )


def _scenario_completion():
    """场景③：B 在首个 T+1 买入日（idx1）涨停 ⇒ 槽位未成交，窗口期内补齐。"""
    limit_up = frozenset({("B", _TRADING_DATES[1].strftime("%Y%m%d"))})
    return _run_pair(
        {}, [("A", 0.9), ("B", 0.8)], _make_price_data(_flat_closes(), limit_up=limit_up)
    )


def _scenario_pending_sell():
    """场景④（run 级）：A 在首个卖出执行日（idx6）跌停 ⇒ 卖单入队，次日重试成交。"""
    limit_down = frozenset({("A", _TRADING_DATES[6].strftime("%Y%m%d"))})
    return _run_pair(
        {}, [("A", 0.9), ("B", 0.8)], _make_price_data(_flat_closes(), limit_down=limit_down)
    )


def _scenario_stop_loss():
    """场景（止损）：A 买入后下跌 10% ⇒ 回撤止损（drawdown_pct=5）触发。"""
    closes = _flat_closes()
    closes["A"] = [10.0 * (1 + 0.001 * i) if i <= 1 else 9.0 for i in range(len(_TRADING_DATES))]
    return _run_pair(
        {},
        [("A", 0.9), ("B", 0.8)],
        _make_price_data(closes),
        old_overrides={"stop_loss_config": OldStopLossConfig(enabled=True, drawdown_pct=5.0)},
        new_overrides={"stop_loss_config": NewStopLossConfig(enabled=True, drawdown_pct=5.0)},
    )


def _closes_stop_at_expiry_check() -> Dict[str, List[float]]:
    """同日双触发价格：A 恰在到期检查日（idx5）跌破止损，此前价格正常。"""
    n = len(_TRADING_DATES)
    closes = _flat_closes()
    # A：idx1 买入价 10.01；idx2-4 维持正常（不提前触发）；idx5 收盘 9.0（回撤 ~10% > 5%）
    closes["A"] = [
        10.0 * (1 + 0.001 * i) if i < 5 else 9.0 * (1 + 0.0005 * (i - 5)) for i in range(n)
    ]
    return closes


def _scenario_same_day_stop_and_expiry():
    """场景（同日双触发）：idx5 同日 A 触发回撤止损、B 持有期到期。

    执行次序契约：旧主循环先 ``_execute_pending_stop_loss_sells`` 后
    ``_execute_pending_condition_sells`` ⇒ idx6 成交序为 A/stop_loss →
    B/holding_period（R2-T5-02 锁定对象）。
    """
    overrides_old = {"stop_loss_config": OldStopLossConfig(enabled=True, drawdown_pct=5.0)}
    overrides_new = {"stop_loss_config": NewStopLossConfig(enabled=True, drawdown_pct=5.0)}
    return _run_pair(
        {},
        [("A", 0.9), ("B", 0.8)],
        _make_price_data(_closes_stop_at_expiry_check()),
        old_overrides=overrides_old,
        new_overrides=overrides_new,
    )


# ══════════════════ A. 端到端新旧逐位一致 ══════════════════


def test_e2e_normal_buy_and_holding_period_sell(synthetic_cost_settings):
    """场景①②：正常买入（T+1 收盘成交）+ 持有期到期卖出，逐位一致。

    锚定：首个买入成交在 idx1（T+1）收盘价；卖单 sell_type 含 holding_period。
    """
    old_engine, new_engine, nav_old, nav_new = _scenario_constant_ab()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    assert len(synthetic_cost_settings["reads"]) >= 2  # 成本读取确实经由合成 stub

    trades = new_engine.get_trades()
    assert len(trades) > 0
    first_buy = trades[trades["action"] == "buy"].iloc[0]
    assert first_buy["date"] == _TRADING_DATES[1]  # T+1 日
    # T+1 收盘价买入（close 口径，非 open）
    assert first_buy["price"] == pytest.approx(10.0 * 1.001)
    sell_types = set(trades["sell_type"].dropna())
    assert "holding_period" in sell_types


def test_e2e_rebalance_swap_sell(synthetic_cost_settings):
    """场景（换出）：调仓日持仓不在新候选 ⇒ 挂出 T+1 卖出，sell_type=rebalance。"""
    old_engine, new_engine, nav_old, nav_new = _scenario_rotation()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    sell_types = set(new_engine.get_trades()["sell_type"].dropna())
    assert "rebalance" in sell_types


def test_e2e_position_completion(synthetic_cost_settings):
    """场景③：首日部分槽位未成交（B 涨停）⇒ 窗口期内补齐成功，逐位一致。"""
    old_engine, new_engine, nav_old, nav_new = _scenario_completion()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    assert old_engine.completion_stats == new_engine.completion_stats
    assert new_engine.completion_stats["total_unfilled"] > 0
    assert new_engine.completion_stats["total_completed"] > 0


def test_e2e_pending_sell_defer_and_retry(synthetic_cost_settings):
    """场景④（run 级）：卖出日跌停 ⇒ 卖单入延迟队列，次日重试成交，逐位一致。"""
    old_engine, new_engine, nav_old, nav_new = _scenario_pending_sell()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    old_stats = old_engine.pending_order_manager.get_statistics()
    new_stats = new_engine.pending_order_manager.get_statistics()
    assert old_stats == new_stats
    assert new_stats["total_added"] > 0 and new_stats["total_succeeded"] > 0
    assert new_stats["pending"] == 0  # 重试已成交，无残留


def test_pending_buy_retry_success_and_expiry(synthetic_cost_settings):
    """场景④（调用级买入侧）：涨停入队 → 重试成交 / 超次过期 / 超期过期，逐位一致。

    回测主买入路径在执行阶段前置过滤可交易性（不可交易候选走槽位回填/补齐），
    ``_buy_stock_with_status_check`` 的买入延迟入队无生产调用方；此处对新旧
    引擎同状态直接驱动该方法 + ``_process_pending_orders``，覆盖 max_retry_count
    超次（expired_retry）与 max_retry_days 超期（expired_days）两条过期路径。
    """
    d0, d1, d2, d3 = _TRADING_DATES[:4]
    closes = {s: [10.0] * len(_TRADING_DATES) for s in ("X", "Y", "Z")}
    limit_up = frozenset(
        {("Y", d0.strftime("%Y%m%d"))} | {("X", d.strftime("%Y%m%d")) for d in _TRADING_DATES[:5]}
    )
    price_data = _make_price_data(closes, limit_up=limit_up)

    def _drive(engine):
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._buy_stock_with_status_check(d0, "X", 50000.0, signal_date=d0)  # 涨停 → 入队
        engine._buy_stock_with_status_check(d0, "Y", 50000.0, signal_date=d0)  # 涨停 → 入队
        engine._buy_stock_with_status_check(d0, "Z", 50000.0, signal_date=d0)  # 直接成交
        engine._process_pending_orders(d1)  # Y 恢复可交易 → 重试成交；X 重试 1
        engine._process_pending_orders(d2)  # X 重试 2
        engine._process_pending_orders(d3)  # X retry=3 > 2 → 超次过期
        # W 无行情记录 → 视为停牌入队；d0+20 自然日 > max_retry_days=10 → 超期过期
        engine._buy_stock_with_status_check(d0, "W", 1000.0, signal_date=d0)
        engine._process_pending_orders(d0 + pd.Timedelta(days=20))

    overrides = {"max_retry_count": 2, "max_retry_days": 10}
    old_engine = _make_old_engine(
        _make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9)]), **overrides
    )
    new_engine = _make_new_engine(
        _make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9)]), **overrides
    )
    _drive(old_engine)
    _drive(new_engine)

    assert old_engine.trades == new_engine.trades
    assert [t["stock"] for t in new_engine.trades] == ["Z", "Y"]  # 直接成交 + 重试成交
    old_stats = old_engine.pending_order_manager.get_statistics()
    new_stats = new_engine.pending_order_manager.get_statistics()
    assert old_stats == new_stats
    assert new_stats["total_succeeded"] == 1 and new_stats["total_expired"] == 2
    # 事件台账逐值一致（含 added / success / expired_retry / expired_days）
    assert old_engine._daily_warning_items == new_engine._daily_warning_items
    expired = new_engine._daily_warning_items["pending_order_expired"]
    assert {item["expire_type"] for item in expired} == {"expired_retry", "expired_days"}


def test_e2e_stop_loss_sell(synthetic_cost_settings):
    """场景（止损）：回撤止损触发 ⇒ T+1 卖出，sell_type=stop_loss，逐位一致。"""
    old_engine, new_engine, nav_old, nav_new = _scenario_stop_loss()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    trades = new_engine.get_trades()
    stop_trades = trades[trades["sell_type"] == "stop_loss"]
    assert len(stop_trades) > 0
    assert set(stop_trades["stock"]) == {"A"}


def test_e2e_same_day_stop_loss_and_expiry_order(synthetic_cost_settings):
    """同日止损 + 持有期到期：逐笔有序比对（A/stop_loss 先于 B/holding_period）。

    idx5 同日：A 跌破止损（回撤 ~10% > 5%）、B 持有期到期（holding_days=4）。
    执行次序契约：主循环先执行止损卖出、后执行条件卖出 ⇒ idx6 成交序
    前两笔为 A/stop_loss → B/holding_period；只做 sell_type 集合比对无法
    锁定次序，此处逐笔有序断言（R2-T5-02）。
    """
    old_engine, new_engine, nav_old, nav_new = _scenario_same_day_stop_and_expiry()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    trades = new_engine.get_trades()
    sells = trades[trades["action"] == "sell"].head(2)
    assert len(sells) >= 2
    first, second = sells.iloc[0], sells.iloc[1]
    assert first["date"] == second["date"] == _TRADING_DATES[6]  # 同一执行日
    assert (first["stock"], first["sell_type"]) == ("A", "stop_loss")
    assert (second["stock"], second["sell_type"]) == ("B", "holding_period")


def test_ml_engine_e2e_equivalence(synthetic_cost_settings):
    """ML 组装体（旧 engine_ml vs 新 ml_signal_feed）：特征注入下端到端逐位一致。"""
    features_by_date = {
        date.strftime("%Y%m%d"): pd.DataFrame(
            {"ts_code": _STOCKS, "atr_pct_14": [0.02, 0.03, 0.025, 0.035]}
        )
        for date in _TRADING_DATES
    }
    common = dict(
        features_by_date=features_by_date,
        universe=_StaticUniverse(_STOCKS),
        initial_capital=1_000_000.0,
        rebalance_freq=5,
        verbose=False,
        data_storage=_FakeStorage(),
    )
    old_engine = OldEngineML(
        signal=_make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)]),
        cost_model=OldCostModel(**_COST_SETTINGS),
        **common,
    )
    new_engine = NewEngineML(
        signal=_make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)]),
        cost_model=NewCostModel(**_COST_SETTINGS),
        **common,
    )
    price_data = _make_price_data(_flat_closes())
    nav_old = old_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    nav_new = new_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    assert len(new_engine.get_trades()) > 0


# ══════════════════ B. 黄金调用点清单（N3） ══════════════════

# 参与序列比对的主循环钩子（新旧共有；含止损两钩子——R2-T5-02 处置）
_SEQUENCE_HOOKS = (
    "_process_pending_orders",
    "_check_stop_loss",
    "_generate_signal",
    "_queue_rebalance_sells",
    "_execute_pending_stop_loss_sells",
    "_execute_pending_condition_sells",
    "_check_and_sell",
    "_execute_pending_buys",
    "_process_position_completion",
    "_calculate_portfolio_value",
)


def _make_spy(original, name):
    """生成记录调用名的包装器（绑定到动态子类，不触生产类）。"""

    def wrapper(self, *args, **kwargs):
        self._hook_calls.append(name)
        return original(self, *args, **kwargs)

    wrapper.__name__ = name
    return wrapper


def _instrument_engine_class(cls, hook_names):
    """构造带钩子序列记录的引擎子类（生产类不被修改）。"""
    namespace = {}
    for name in hook_names:
        original = getattr(cls, name, None)
        if original is not None:
            namespace[name] = _make_spy(original, name)
    return type(f"Instrumented{cls.__name__}", (cls,), namespace)


def test_golden_hook_sequence_old_minus_exposure_hooks(synthetic_cost_settings):
    """黄金钩子序列：新侧 == 旧侧剔除 4 个 exposure 钩子调用。

    旧侧现状：主循环逐日无条件调用 ``_execute_pending_exposure_trims`` /
    ``_queue_exposure_trim`` / ``_queue_exposure_replenish`` /
    ``_execute_pending_exposure_replenishes``；B0 基线（无 exposure 表/政策源）
    下 4 者均为守卫即返的 no-op（不改变任何共享状态）。
    摘除后差异：新侧主循环不再有这 4 个调用点，其余钩子的调用顺序与次数
    必须与旧侧逐一相同——即旧序列过滤 4 个 exposure 钩子的条目后与新序列
    精确相等。
    """
    old_cls = _instrument_engine_class(OldEngine, _SEQUENCE_HOOKS + _EXPOSURE_HOOKS)
    new_cls = _instrument_engine_class(NewEngine, _SEQUENCE_HOOKS)
    old_engine = _make_old_engine(
        _make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)]), cls=old_cls
    )
    new_engine = _make_new_engine(
        _make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)]), cls=new_cls
    )
    old_engine._hook_calls = []
    new_engine._hook_calls = []

    price_data = _make_price_data(_flat_closes())
    nav_old = old_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    nav_new = new_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)

    old_seq = old_engine._hook_calls
    new_seq = new_engine._hook_calls
    # 锚定（防过滤空转）：旧侧 4 个 exposure 钩子确实逐日被调用
    for hook in _EXPOSURE_HOOKS:
        assert old_seq.count(hook) == len(_TRADING_DATES), f"旧侧 {hook} 未逐日调用"
    # 新侧序列 == 旧侧序列剔除 4 个 exposure 钩子
    filtered = [name for name in old_seq if name not in _EXPOSURE_HOOKS]
    assert new_seq == filtered
    assert len(new_seq) > 0


def test_golden_hook_sequence_with_stop_loss(synthetic_cost_settings):
    """黄金钩子序列（止损启用场景）：止损两钩子逐日调用 + 执行次序锁定。

    R2-T5-02 处置：止损启用后 ``_check_stop_loss`` /
    ``_execute_pending_stop_loss_sells`` 进入主循环逐日调用链；断言
    新序列 == 旧序列剔除 4 个 exposure 钩子，且止损执行（``_execute_
    pending_stop_loss_sells``）相对条件卖出执行（``_execute_pending_
    condition_sells``）的逐日前置次序被锁定（交替子序列严格成对）。
    """
    old_cls = _instrument_engine_class(OldEngine, _SEQUENCE_HOOKS + _EXPOSURE_HOOKS)
    new_cls = _instrument_engine_class(NewEngine, _SEQUENCE_HOOKS)
    old_engine = _make_old_engine(
        _make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)]),
        cls=old_cls,
        stop_loss_config=OldStopLossConfig(enabled=True, drawdown_pct=5.0),
    )
    new_engine = _make_new_engine(
        _make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)]),
        cls=new_cls,
        stop_loss_config=NewStopLossConfig(enabled=True, drawdown_pct=5.0),
    )
    old_engine._hook_calls = []
    new_engine._hook_calls = []

    price_data = _make_price_data(_closes_stop_at_expiry_check())
    nav_old = old_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    nav_new = new_engine.run(_TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, price_data)
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)

    old_seq = old_engine._hook_calls
    new_seq = new_engine._hook_calls
    for hook in _EXPOSURE_HOOKS:
        assert old_seq.count(hook) == len(_TRADING_DATES), f"旧侧 {hook} 未逐日调用"
    # 止损两钩子逐日调用锚定（两侧各 25 次）
    for seq, tag in ((old_seq, "旧"), (new_seq, "新")):
        assert seq.count("_check_stop_loss") == len(_TRADING_DATES), f"{tag}侧止损检查未逐日调用"
        assert seq.count("_execute_pending_stop_loss_sells") == len(
            _TRADING_DATES
        ), f"{tag}侧止损执行未逐日调用"
    filtered = [name for name in old_seq if name not in _EXPOSURE_HOOKS]
    assert new_seq == filtered
    # 次序锁定：止损执行严格先于同日的条件卖出执行（交替子序列逐日成对）
    pair = [
        name
        for name in new_seq
        if name in ("_execute_pending_stop_loss_sells", "_execute_pending_condition_sells")
    ]
    assert pair == [
        "_execute_pending_stop_loss_sells",
        "_execute_pending_condition_sells",
    ] * len(_TRADING_DATES)


def test_stop_loss_execution_order_swap_mutation_detected(monkeypatch, synthetic_cost_settings):
    """换序变异自证（R2-T5-02 复收条件）：错误换序实现下检测器必须报警。

    内存把新侧「止损卖出执行」与「条件卖出执行」两步换序（止损执行延迟到
    条件卖出之后，等价于评审的 ``_run_daily_trade_executions`` 换序变异）：
    同日双触发场景下成交序从 A/stop_loss → B/holding_period 翻转为
    B/holding_period → A/stop_loss——先锚定变异确实生效（翻转发生），再断言
    端到端逐位比对必须抛 AssertionError。原实现的通过性由
    ``test_e2e_same_day_stop_loss_and_expiry_order`` 与
    ``test_golden_hook_sequence_with_stop_loss`` 锁定。
    """
    # 基线：原实现逐位一致且次序正确（防变异未生效的误报）
    old_engine, new_engine, nav_old, nav_new = _scenario_same_day_stop_and_expiry()
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    baseline_sells = new_engine.get_trades()
    baseline_pair = baseline_sells[baseline_sells["action"] == "sell"].head(2)
    assert list(baseline_pair["stock"]) == ["A", "B"]

    # 变异：新侧类方法内存替换（monkeypatch 结束自动恢复；生产文件不动）
    orig_stop = NewEngine._execute_pending_stop_loss_sells
    orig_cond = NewEngine._execute_pending_condition_sells

    def _swapped_condition_sells(self, date, trading_dates, date_to_idx):
        orig_cond(self, date, trading_dates, date_to_idx)
        orig_stop(self, date, trading_dates, date_to_idx)  # 止损延迟到条件卖出之后

    monkeypatch.setattr(NewEngine, "_execute_pending_condition_sells", _swapped_condition_sells)
    monkeypatch.setattr(NewEngine, "_execute_pending_stop_loss_sells", lambda self, d, t, i: None)

    _old2, mutated_engine, _no2, nav_mutated = _scenario_same_day_stop_and_expiry()
    mutated_trades = mutated_engine.get_trades()
    mutated_pair = mutated_trades[mutated_trades["action"] == "sell"].head(2)
    # 变异确实生效：成交序翻转为 B 在前
    assert list(mutated_pair["stock"]) == ["B", "A"]
    assert list(mutated_pair["sell_type"]) == ["holding_period", "stop_loss"]
    # 检测器必须捕获：逐位比对抛 AssertionError
    with pytest.raises(AssertionError):
        _assert_runs_identical(old_engine, mutated_engine, nav_old, nav_mutated)


# ══════════════════ C. 退役符号扫描（D6）+ MRO 锁定 ══════════════════


def _resolve_import_from_parent(node: ast.ImportFrom, importer_package: Optional[str]):
    """解析 from-import 的父包绝对名；相对导入按导入方所属包归一。

    Args:
        node: ImportFrom 节点
        importer_package: 导入方**所属包**的绝对点分路径（如引擎模块为
            src.lazybull.v2.core.execution；__init__.py 即包自身路径）；
            None 且为相对导入时无法解析，返回 None（调用方保守处理）

    Returns:
        父包绝对名；无法解析返回 None
    """
    if node.level == 0:
        return node.module or ""
    if not importer_package:
        return None
    package = importer_package.split(".")
    up = node.level - 1
    if up > len(package):
        return None
    base = package[: len(package) - up] if up else package
    return ".".join(base + ([node.module] if node.module else []))


def _importer_package_of(path: Path) -> str:
    """文件路径 → 导入方所属包的绝对点分路径（供相对导入归属解析）。

    普通模块与包初始化文件统一取**文件的父目录**（R2-T5-R3-01 修正：
    ``__init__.py`` 的所属包是其所在目录本身，``__init__`` 不算包层级）。
    """
    rel = path.resolve().parent.relative_to(_REPO_ROOT)
    return ".".join(rel.parts)


def _scan_retired_symbols_in_source(
    src: str, filename: str, importer_package: Optional[str] = None
) -> List[str]:
    """AST 扫描源码文本中的退役符号（Attribute/Name/Import 节点，不含字符串）。

    ImportFrom 归属判定（R2-T5-R2-01）：
    - ``from ...backtest.exposure_x import ...``：完整模块路径命中（原有口径）；
    - ``from src.lazybull.backtest import exposure_x``（父包导入，含别名与混合
      导入列表）：父包解析归属 == 退役父包 且别名 ∈ 退役模块名集 ⇒ 逐别名命中；
    - 相对导入形态（``from ....backtest import exposure_x``）按导入方所属包
      ``importer_package`` 归一为绝对父包后同口径判定；未提供
      ``importer_package`` 的内存源码遇相对导入时不做父包归属判定
      （保守，不免检其他分支）。
    """
    offenders: List[str] = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _RETIRED_SYMBOLS:
            offenders.append(f"{filename}:{node.lineno} 属性访问 {node.attr}")
        elif isinstance(node, ast.Name) and node.id in _RETIRED_SYMBOLS:
            offenders.append(f"{filename}:{node.lineno} 名称引用 {node.id}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if ".backtest.exposure" in module or module.endswith("backtest.exposure"):
                offenders.append(f"{filename}:{node.lineno} import {module}")
            parent = _resolve_import_from_parent(node, importer_package)
            for alias in node.names:
                if alias.name in _RETIRED_SYMBOLS:
                    offenders.append(f"{filename}:{node.lineno} import 符号 {alias.name}")
                if parent == _RETIRED_PARENT_PACKAGE and alias.name in _RETIRED_MODULE_NAMES:
                    offenders.append(f"{filename}:{node.lineno} 父包导入 {parent}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if ".backtest.exposure" in alias.name:
                    offenders.append(f"{filename}:{node.lineno} import {alias.name}")
    return offenders


def _scan_retired_symbols(path: Path) -> List[str]:
    """AST 扫描单个文件中的退役符号（附所属包路径供相对导入归属解析）。"""
    return _scan_retired_symbols_in_source(
        path.read_text(encoding="utf-8"), path.name, importer_package=_importer_package_of(path)
    )


def test_retired_symbol_extraction_matches_frozen_set():
    """提取完整性锁定：运行时 AST 提取集 == 冻结预期集（41 项逐字冻结）。

    集合相等是双向的：旧 exposure 源文件新增/删除/重命名任何定义侧符号
    （类/方法/self 字段/模块级函数）都会打破相等 ⇒ 测试报警，防清单漂移。
    同时锁定扫描排除集恰为文档化的 {"record"}（不得悄悄扩大），以及退役
    模块名集（R2-T5-R2-01，由 _RETIRED_SOURCE_MODULES 派生比对）。
    """
    extracted = _extract_retired_symbols()
    assert extracted == _FROZEN_RETIRED_SYMBOLS, (
        f"提取集与冻结集漂移: 新增 {sorted(extracted - _FROZEN_RETIRED_SYMBOLS)}, "
        f"缺失 {sorted(_FROZEN_RETIRED_SYMBOLS - extracted)}"
    )
    assert len(extracted) == 41
    assert _SCAN_EXCLUDED_SYMBOLS == {"record"}
    # 迁移面扫描清单 = 提取全集 − 排除集（40 项）
    assert _RETIRED_SYMBOLS == _FROZEN_RETIRED_SYMBOLS - _SCAN_EXCLUDED_SYMBOLS
    # 退役模块名集 = 三个旧源文件模块短名（派生 == 冻结，防漂移）
    derived_modules = frozenset(
        module.__name__.rsplit(".", 1)[-1] for module in _RETIRED_SOURCE_MODULES
    )
    assert derived_modules == _RETIRED_MODULE_NAMES
    # 父包基准与模块来源一致（防父包常量与模块集脱钩）
    assert {module.__name__.rsplit(".", 1)[0] for module in _RETIRED_SOURCE_MODULES} == {
        _RETIRED_PARENT_PACKAGE
    }


def test_retired_symbol_scanner_catches_mutations():
    """变异反例永久回归（R2-T5-01 + R2-T5-R2-01 复收条件）：漏项必识别、负例不误报。

    给扫描器喂内存源码（不写磁盘）：评审实证的漏项（``pending_exposure_trims``
    自赋值 / ``_date_key`` 调用 / ``exposure_release_budget`` 自赋值等）与
    **父包导入形态**（``from src.lazybull.backtest import exposure_trim`` 三模块、
    带别名、混合导入列表、相对导入归一）必须被识别为残留；负例（docstring/
    字符串文本出现符号名、fail-fast 关键字参数名 ``exposure_table=``、无关
    模块导入、归属他处的相对导入）不得算作残留。
    """
    mutants = {
        "self 赋值状态字段": (
            "class X:\n    def m(self):\n        self.pending_exposure_trims = {}\n"
        ),
        "方法调用": "def f(self):\n    return self._date_key(0)\n",
        "释放额自赋值": (
            "class X:\n    def m(self):\n        self.exposure_release_budget = 1.0\n"
        ),
        "统计字段读取": "def f(self):\n    return self.exposure_trim_stats\n",
        "回补队列读取": "def f(self):\n    return self.pending_exposure_replenishes\n",
        "类名 Name 引用": "def f():\n    return ExposureOverrideStats()\n",
        "模块 import": (
            "from src.lazybull.backtest.exposure_trim import BacktestExposureTrimMixin\n"
        ),
        "开关字段注解赋值": (
            "class X:\n    def m(self):\n        self.exposure_replenish_enabled: bool = True\n"
        ),
        # ── R2-T5-R2-01：父包导入形态（评审实测 4 例 + 相对导入归一） ──
        "父包导入 exposure_trim": "from src.lazybull.backtest import exposure_trim\n",
        "父包导入 exposure_override": "from src.lazybull.backtest import exposure_override\n",
        "父包导入 exposure_replenish": "from src.lazybull.backtest import exposure_replenish\n",
        "父包导入带别名": "from src.lazybull.backtest import exposure_trim as et\n",
        "父包混合导入": "from src.lazybull.backtest import engine, exposure_trim\n",
        "父包混合导入（退役项在前）": (
            "from src.lazybull.backtest import exposure_replenish, engine\n"
        ),
        "完整路径 import 带别名": "import src.lazybull.backtest.exposure_trim as et\n",
    }
    for label, src in mutants.items():
        offenders = _scan_retired_symbols_in_source(src, "mutant.py")
        assert offenders, f"变异未被捕获: {label}"

    # 相对导入形态：导入方所属包 = src.lazybull.v2.core.execution，
    # level=4 归一为 src.lazybull.backtest ⇒ 必须命中
    relative_src = "from ....backtest import exposure_trim\n"
    offenders = _scan_retired_symbols_in_source(
        relative_src, "mutant.py", importer_package="src.lazybull.v2.core.execution"
    )
    assert offenders, "相对导入父包形态未被捕获"
    assert any("父包导入" in item for item in offenders)

    # 混合导入的精确性：只命中退役别名，不误伤同行合法别名
    mixed = _scan_retired_symbols_in_source(
        "from src.lazybull.backtest import engine, exposure_trim\n", "mutant.py"
    )
    assert mixed == ["mutant.py:1 父包导入 src.lazybull.backtest.exposure_trim"]

    negatives = {
        "docstring 文本": (
            'def f(self):\n    """调用 self._date_key(date) 并读 pending_exposure_trims。"""'
            "\n    return 1\n"
        ),
        "字符串键": (
            'def f():\n    return {"pending_exposure_trims": 1, "exposure_release_budget": 2}\n'
        ),
        "关键字参数名": (
            "def f():\n    return BacktestEngine(universe, signal, exposure_table={})\n"
        ),
        "模块 docstring": '"""set_exposure_table(None) 装配语义。"""\nX = 1\n',
        # ── R2-T5-R2-01 负例：无关模块导入不得误判 ──
        "父包导入合法模块": "from src.lazybull.backtest import engine\n",
        "导入 backtest 父包自身": "from src.lazybull import backtest\n",
        "父包导入多个合法模块": "from src.lazybull.backtest import engine, reporter\n",
    }
    for label, src in negatives.items():
        offenders = _scan_retired_symbols_in_source(src, "neg.py")
        assert offenders == [], f"负例误报: {label} -> {offenders}"

    # 相对导入精度负例：归属他处（src.lazybull.v2.backtest，非退役父包）不得误报
    misplaced = _scan_retired_symbols_in_source(
        "from ...backtest import exposure_trim\n",
        "neg.py",
        importer_package="src.lazybull.v2.core.execution",
    )
    assert misplaced == [], f"相对导入归属他处误报: {misplaced}"


# 正式零残留扫描面（模块级冻结：27 个文件，其中 5 个包初始化文件）
_V2_CORE_DIR = _REPO_ROOT / "src" / "lazybull" / "v2" / "core"
_V2_CORE_SCAN_FILES = sorted(_V2_CORE_DIR.rglob("*.py"))
_V2_CORE_INIT_FILES = [path for path in _V2_CORE_SCAN_FILES if path.name == "__init__.py"]


def test_v2_core_no_retired_exposure_symbols():
    """D6 退役符号扫描：v2/core/ 全部 .py 对退役符号的 AST 引用为 0。

    扫描清单为运行时自动提取全集减文档化排除集（``record``，见
    ``_SCAN_EXCLUDED_SYMBOLS``）；口径：只查 AST 节点（Attribute/Name/
    Import/ImportFrom），``decision_trace["final_target_exposure"]`` 等
    字符串键与 docstring 文本不计。
    """
    offenders: List[str] = []
    assert _V2_CORE_SCAN_FILES, "扫描目录为空（前置假设失效）"
    for path in _V2_CORE_SCAN_FILES:
        offenders.extend(_scan_retired_symbols(path))
    assert offenders == [], f"退役符号残留: {offenders}"


def test_importer_package_of_scanned_files_exact():
    """文件入口回归（R2-T5-R3-01）：扫描面 30 文件的所属包 == 各自真实父目录。

    普通模块与包初始化文件统一取文件父目录的点分路径；锚定扫描面构成
    （30 个文件、其中恰 5 个 ``__init__.py``），防扫描面漂移后断言空转。
    （T6 注记：core/signal 新增 ml_signal / ensemble_signal / factory 三件，
    扫描面 27→30，属规划内增长；__init__.py 5 个不变。）
    """
    assert len(_V2_CORE_SCAN_FILES) == 30
    assert len(_V2_CORE_INIT_FILES) == 5
    for path in _V2_CORE_SCAN_FILES:
        expected = ".".join(path.resolve().parent.relative_to(_REPO_ROOT).parts)
        assert _importer_package_of(path) == expected, f"{path.name} 所属包计算错误"
    # 锚定（评审复现案例）：execution/__init__.py 的所属包不含 .__init__
    exec_init = _V2_CORE_DIR / "execution" / "__init__.py"
    assert _importer_package_of(exec_init) == "src.lazybull.v2.core.execution"
    assert _importer_package_of(_V2_CORE_DIR / "__init__.py") == "src.lazybull.v2.core"
    assert (
        _importer_package_of(_V2_CORE_DIR / "execution" / "engine.py")
        == "src.lazybull.v2.core.execution"
    )


def _relative_injection_line(path: Path) -> str:
    """生成对指定文件注入后真实归属退役父包的相对导入行（含三模块别名混合）。"""
    package = _importer_package_of(path)
    level = len(package.split(".")) - 1  # 根级 core 包三点、子包四点
    dots = "." * level
    return (
        f"\nfrom {dots}backtest import exposure_override as ro, "
        "exposure_trim as rt, exposure_replenish as rr\n"
    )


def _run_zero_residual_with_injection(monkeypatch, target: Path, inject_line: str):
    """把 inject_line 追加到 target 文件的读取结果（Path.read_text 替身），
    跑正式零残留测试；返回 (是否抛 AssertionError, 替身命中次数)。"""
    original = Path.read_text
    hits: List[Path] = []

    def _patched(self, *args, **kwargs):
        text = original(self, *args, **kwargs)
        try:
            if self.resolve() == target.resolve():
                hits.append(self)
                return text + inject_line
        except OSError:
            pass
        return text

    monkeypatch.setattr(Path, "read_text", _patched)
    failed = False
    try:
        test_v2_core_no_retired_exposure_symbols()
    except AssertionError:
        failed = True
    return failed, len(hits)


@pytest.mark.parametrize(
    "target",
    _V2_CORE_INIT_FILES + [_V2_CORE_DIR / "execution" / "engine.py"],
    ids=lambda p: str(p.relative_to(_V2_CORE_DIR)),
)
def test_retired_relative_import_injection_rejected(monkeypatch, target):
    """内存注入反例（R2-T5-R3-01 复收条件）：真实相对导入退役注入必须被拒绝。

    对 5 个包初始化文件（v2/core 根 + accounting/decision/execution/signal
    子包）与 1 个普通模块（execution/engine.py），经 ``Path.read_text`` 替身
    注入归属退役父包的相对导入行（三模块别名混合，层级按文件实际位置计算：
    根级 core 包三点、子包四点），正式零残留测试必须抛 AssertionError，
    且确认替身确实命中目标文件（防空转）。
    """
    failed, hits = _run_zero_residual_with_injection(
        monkeypatch, target, _relative_injection_line(target)
    )
    assert hits >= 1, f"替身未命中 {target.name}"
    assert failed, f"注入未被正式零残留测试拒绝: {target.relative_to(_REPO_ROOT)}"


def test_legal_relative_import_injection_not_misjudged(monkeypatch):
    """合法相对导入负例：注入合法行后正式零残留测试仍通过（不误报）。"""
    target = _V2_CORE_DIR / "execution" / "__init__.py"
    for legal_line in ("\nfrom . import engine\n", "\nfrom ..common import cost\n"):
        failed, hits = _run_zero_residual_with_injection(monkeypatch, target, legal_line)
        assert hits >= 1, "替身未命中目标文件"
        assert not failed, f"合法相对导入被误报: {legal_line.strip()}"
    # 内存口径负例：合法相对导入的扫描结果为空
    for src, package in (
        ("from . import price_index\n", "src.lazybull.v2.core.execution"),
        ("from ..common import cost\n", "src.lazybull.v2.core.execution"),
        ("from ...v2.common import cost\n", "src.lazybull.v2.core"),
    ):
        offenders = _scan_retired_symbols_in_source(src, "neg.py", importer_package=package)
        assert offenders == [], f"合法相对导入误报: {src.strip()} -> {offenders}"


def test_new_engine_mro_pruned_order():
    """新引擎 MRO 恰为 7 个指定 mixin，且相对顺序 = 旧清单摘除 3 个 exposure mixin。"""
    expected = [
        "BacktestReportingMixin",
        "BacktestBuyExecutionMixin",
        "BacktestSellExecutionMixin",
        "BacktestSignalExecutionMixin",
        "BacktestPendingExecutionMixin",
        "BacktestHoldingsSnapshotMixin",
        "BacktestRunLoopMixin",
    ]
    new_mro = [c.__name__ for c in NewEngine.__mro__[1:-1]]
    assert new_mro == expected

    retired_mixins = {
        "BacktestExposureOverrideMixin",
        "BacktestExposureTrimMixin",
        "BacktestExposureReplenishMixin",
    }
    old_mro = [c.__name__ for c in OldEngine.__mro__[1:-1]]
    # 前置锚定：旧清单确含 3 个 exposure mixin
    assert retired_mixins <= set(old_mro)
    assert [name for name in old_mro if name not in retired_mixins] == expected


# ══════════════════ D. 退役隔离测试（不依赖退役组件） ══════════════════


def test_retired_isolation_normal_buy(synthetic_cost_settings):
    """D-1 正常买入成功：新侧独立跑通且成交记录非空（不 touch 任何 exposure 符号）。"""
    new_engine = _make_new_engine(
        _make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)])
    )
    nav = new_engine.run(
        _TRADING_DATES[0], _TRADING_DATES[-1], _TRADING_DATES, _make_price_data(_flat_closes())
    )
    trades = new_engine.get_trades()
    assert len(trades) > 0
    assert (trades["action"] == "buy").any()
    assert len(nav) == len(_TRADING_DATES)


def test_retired_isolation_position_completion(synthetic_cost_settings):
    """D-2 仓位补齐成功：新侧补齐成交发生（completion_stats 计数 > 0）。"""
    _old, new_engine, _nav_old, _nav_new = _scenario_completion()
    assert new_engine.completion_stats["total_completed"] > 0
    # 补齐买入确实成交（B 在 idx2 以收盘价买入）
    trades = new_engine.get_trades()
    completion_buy = trades[
        (trades["action"] == "buy")
        & (trades["stock"] == "B")
        & (trades["date"] == _TRADING_DATES[2])
    ]
    assert len(completion_buy) == 1


def test_retired_isolation_none_assembly(synthetic_cost_settings):
    """D-3 exposure_table=None 装配启动：新引擎（无 exposure 参数）构造 + 短场景跑通。

    语义等价于旧 replay 装配 ``set_exposure_table(None)`` 后的行为：旧引擎显式
    调用 ``set_exposure_table(None)``（关闭暴露覆盖）后跑同一场景，与新引擎
    端到端逐位一致。
    """
    price_data = _make_price_data(_flat_closes())
    old_engine = _make_old_engine(
        _make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)])
    )
    old_engine.set_exposure_table(None, verbose=False)  # replay 装配：显式关闭
    new_engine = _make_new_engine(
        _make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9), ("B", 0.8)])
    )
    short_dates = _TRADING_DATES[:12]
    nav_old = old_engine.run(short_dates[0], short_dates[-1], short_dates, price_data)
    nav_new = new_engine.run(short_dates[0], short_dates[-1], short_dates, price_data)
    _assert_runs_identical(old_engine, new_engine, nav_old, nav_new)
    assert len(new_engine.get_trades()) > 0


# ══════════════════ E. 退役配置非默认值 fail-fast ══════════════════


def test_retired_config_fail_fast(synthetic_cost_settings):
    """退役配置注入被拒：exposure 关键字 → TypeError；退役方法 → AttributeError。"""
    signal_new = _make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9)])
    for bad_kw in ("exposure_table", "exposure_stats"):
        with pytest.raises(TypeError):
            NewEngine(
                universe=_StaticUniverse(_STOCKS),
                signal=signal_new,
                **{bad_kw: {"20260105": 0.5}},
            )

    new_engine = _make_new_engine(signal_new)
    for name in (
        "set_exposure_table",
        "set_exposure_policy",
        "get_exposure_report",
        "get_exposure_trim_report",
        "get_exposure_replenish_report",
    ):
        assert not hasattr(new_engine, name), f"新引擎不应再有 {name}"
        with pytest.raises(AttributeError):
            getattr(new_engine, name)

    # 锚定（前置假设）：旧引擎仍持有这些退役方法，不对称性是真实存在的
    old_engine = _make_old_engine(_make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9)]))
    for name in ("set_exposure_table", "set_exposure_policy", "get_exposure_report"):
        assert hasattr(old_engine, name)


# ══════════════════ F. 接线义务隔离断言（T4 评审 R1-5/R1-6） ══════════════════


def test_pending_order_event_sink_wiring(synthetic_cost_settings):
    """接线①：pending_order_manager 事件回调 == 引擎的 _record_pending_order_event。

    绑定方法每次访问生成新对象（``is`` 不适用），按绑定方法相等性（同
    __self__ 同 __func__）断言；功能断言：一次入队事件进入引擎台账。
    """
    for engine in (
        _make_old_engine(_make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9)])),
        _make_new_engine(_make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9)])),
    ):
        sink = engine.pending_order_manager.event_sink
        assert sink is not None
        assert sink == engine._record_pending_order_event
        assert sink.__self__ is engine

        # 功能断言：触发一次入队事件 → 进入引擎台账 _daily_warning_items
        engine._daily_warning_items = {}
        d0 = _TRADING_DATES[0]
        engine.pending_order_manager.add_order(
            "X", "buy", d0, d0, target_value=1000.0, reason="涨停"
        )
        assert engine._daily_warning_items["pending_order_added"] == [
            {"stock": "X", "action": "buy", "reason": "涨停"}
        ]


def test_min_buy_threshold_short_circuit(synthetic_cost_settings):
    """接线②：min_buy_value_ratio=0 短路返回 0.0，且不求值组合市值。

    ratio <= 0 时 ``_get_min_buy_value_threshold`` 必须先短路（避免无谓触发
    ``_calculate_portfolio_value`` 的 ``last_known_price`` 缓存写回）；
    对照 ratio > 0 时正常求值一次。
    """
    date = _TRADING_DATES[0]
    for make, base in (
        (_make_old_engine, old_signal_base.Signal),
        (_make_new_engine, new_signal_base.Signal),
    ):
        # ratio=0：短路，零次估值
        engine = make(_make_det_signal(base, 2, {}, [("A", 0.9)]), min_buy_value_ratio=0.0)
        calls: List[pd.Timestamp] = []
        engine._calculate_portfolio_value = lambda d: (calls.append(d), 999999.0)[1]
        assert engine._get_min_buy_value_threshold(date) == 0.0
        assert calls == [], "ratio=0 时不应触发 _calculate_portfolio_value"

        # ratio>0：正常求值一次，阈值 = 总资产 / 目标持仓数 * 比例
        engine = make(_make_det_signal(base, 2, {}, [("A", 0.9)]), min_buy_value_ratio=0.3)
        calls2: List[pd.Timestamp] = []
        engine._calculate_portfolio_value = lambda d: (calls2.append(d), 900000.0)[1]
        assert engine._get_min_buy_value_threshold(date) == pytest.approx(900000.0 / 2 * 0.3)
        assert len(calls2) == 1


# ══════════════════ G. 估值回退余 2 处逐点等价（D4） ══════════════════


def _extract_old_nested_position_weight():
    """从旧 buy_execution.py 源码**逐字**提取 ``_execute_pending_buys`` 内嵌的
    ``_get_position_weight`` 闭包工厂（AST 定位 + 原文 exec，非手抄副本）。"""
    src = Path(old_buy_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    segment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_pending_buys":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_get_position_weight":
                    segment = ast.get_source_segment(src, child)
                    break
    assert segment and segment.startswith("def _get_position_weight"), "旧嵌套函数定位失败"
    wrapper_src = (
        "def _make(self, date, portfolio_value):\n"
        + textwrap.indent(segment, "    ")
        + "\n    return _get_position_weight\n"
    )
    namespace: Dict = {}
    exec(wrapper_src, namespace)
    return namespace["_make"]


def _weight_stub(positions: Dict, price_table: Dict[str, Optional[float]]):
    """估值回退测试桩：positions + 合成价格表驱动的 _get_trade_price。"""
    stub = SimpleNamespace(positions=positions)
    stub._get_trade_price = lambda date, stock: price_table.get(stock)
    return stub


# (说明, 持仓, 价格表, 组合市值) —— 覆盖：当日有价写回 / 缓存价回退 / 买入价兜底 /
# 买入价 0 归零 / 非持仓 / 组合市值非正
_WEIGHT_FALLBACK_CASES = [
    ("当日有价→写回缓存", {"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": 10.0}, 2000.0),
    (
        "无价→回退缓存价",
        {"S": {"shares": 100, "buy_trade_price": 9.0, "last_known_price": 9.5}},
        {"S": None},
        2000.0,
    ),
    ("无价无缓存→买入价兜底", {"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": None}, 2000.0),
    ("无价无缓存且买入价0→0", {"S": {"shares": 100, "buy_trade_price": 0.0}}, {"S": None}, 2000.0),
    ("非持仓→0", {"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": 10.0}, 2000.0),
    ("组合市值为0→0", {"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": 10.0}, 0.0),
    ("组合市值为负→0", {"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": 10.0}, -1.0),
]


def test_buy_weight_fallback_pointwise(synthetic_cost_settings):
    """G-1 旧 buy_execution.py:144 回退链 → 新模块级 helper ``_get_position_weight``。

    旧侧为 ``_execute_pending_buys`` 内嵌闭包（生产路径中 inherited_stocks 恒空，
    不可直接触达），此处以 AST 提取的旧源码原文构造同签名可调用对象，与新侧
    模块级 helper 同输入逐值比对，并锁定 ``last_known_price`` 回写副作用。
    """
    old_factory = _extract_old_nested_position_weight()
    date = _TRADING_DATES[0]
    for label, positions, price_table, portfolio_value in _WEIGHT_FALLBACK_CASES:
        stock = "S" if label != "非持仓→0" else "NOT_HELD"
        old_stub = _weight_stub(copy.deepcopy(positions), price_table)
        new_stub = _weight_stub(copy.deepcopy(positions), price_table)
        old_val = old_factory(old_stub, date, portfolio_value)(stock)
        new_val = new_buy_mod._get_position_weight(new_stub, date, portfolio_value, stock)
        assert old_val == new_val, f"{label} 返回值不一致"
        assert old_stub.positions == new_stub.positions, f"{label} last_known_price 回写不一致"
    # 锚定：写回副作用确实发生（防两侧同错）
    _, positions, _, _ = _WEIGHT_FALLBACK_CASES[0]
    stub = _weight_stub(copy.deepcopy(positions), {"S": 10.0})
    new_buy_mod._get_position_weight(stub, date, 2000.0, "S")
    assert stub.positions["S"]["last_known_price"] == 10.0
    assert new_buy_mod._get_position_weight(
        _weight_stub({"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": 10.0}), date, 2000.0, "S"
    ) == pytest.approx(100 * 10.0 / 2000.0)


def test_planning_weight_fallback_pointwise(synthetic_cost_settings):
    """G-2 旧 signal_execution.py:73 回退链 → 新 ``_get_position_weight_for_planning``。

    两侧方法同签名直接可比；覆盖价格回退链与 portfolio_value 缺失时的
    ``_calculate_portfolio_value`` 兜底分支。
    """
    date = _TRADING_DATES[0]
    for label, positions, price_table, portfolio_value in _WEIGHT_FALLBACK_CASES:
        for pv_override in (portfolio_value, None, 0.0):
            stock = "S" if label != "非持仓→0" else "NOT_HELD"

            def _make_stub(side_positions):
                stub = _weight_stub(side_positions, price_table)
                stub._calculate_portfolio_value = lambda d: 500000.0
                return stub

            old_stub = _make_stub(copy.deepcopy(positions))
            new_stub = _make_stub(copy.deepcopy(positions))
            old_val = OldEngine._get_position_weight_for_planning(
                old_stub, date, stock, portfolio_value=pv_override
            )
            new_val = NewEngine._get_position_weight_for_planning(
                new_stub, date, stock, portfolio_value=pv_override
            )
            assert old_val == new_val, f"{label} pv={pv_override} 返回值不一致"
            assert old_stub.positions == new_stub.positions, f"{label} pv={pv_override} 回写不一致"
    # 锚定：portfolio_value=None 走 _calculate_portfolio_value 兜底
    stub = _weight_stub({"S": {"shares": 100, "buy_trade_price": 9.0}}, {"S": 10.0})
    stub._calculate_portfolio_value = lambda d: 500000.0
    assert NewEngine._get_position_weight_for_planning(stub, date, "S") == pytest.approx(
        100 * 10.0 / 500000.0
    )
    assert stub.positions["S"]["last_known_price"] == 10.0


# ══════════════════ H. 条件卖出 3 触发器 + 减仓不进队列 ══════════════════


def test_condition_sell_three_triggers_parity(synthetic_cost_settings):
    """3 触发器 sell_type 新旧一致：holding_period / rebalance / stop_loss。

    分别由场景①②（持有期）、换出场景（调仓）、止损场景（回撤止损）覆盖；
    各场景端到端逐位一致已由 A 组锁定，此处汇总 sell_type 集合比对并锚定
    三类触发均真实发生。
    """
    sell_types_old: set = set()
    sell_types_new: set = set()
    for old_engine, new_engine, _no, _nw in (
        _scenario_constant_ab(),
        _scenario_rotation(),
        _scenario_stop_loss(),
    ):
        sell_types_old |= set(old_engine.get_trades()["sell_type"].dropna())
        sell_types_new |= set(new_engine.get_trades()["sell_type"].dropna())
    assert sell_types_old == sell_types_new
    assert {"holding_period", "rebalance", "stop_loss"} <= sell_types_new


def test_risk_trim_never_queued(synthetic_cost_settings):
    """减仓（sell_type=risk_trim）直接成交、不进延迟队列。

    源码断言：新侧 sell_execution 中全部 ``pending_order_manager.add_order``
    调用点都收敛在 ``_defer_sell_order`` 内、且被 ``allow_pending`` 守卫
    （减仓路径传 allow_pending=False ⇒ 不可能入队）。
    行为断言：allow_pending=False 下无论可交易与否，add_order 零调用；
    可交易时直接成交（fraction=0.5 部分卖出），新旧逐值一致。
    """
    # ── 源码断言 ──
    src = Path(new_sell_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    add_order_sites: List[Tuple[str, ast.stmt]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "add_order"
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == "pending_order_manager"
                ):
                    add_order_sites.append((node.name, sub))
    assert add_order_sites, "前置假设失效：未找到 add_order 调用点"
    for func_name, call_node in add_order_sites:
        assert func_name == "_defer_sell_order", f"add_order 调用点未收敛: {func_name}"
    # _defer_sell_order 中 add_order 被 allow_pending 守卫
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_defer_sell_order":
            guard_tests = [
                ast.get_source_segment(src, sub.test)
                for sub in ast.walk(node)
                if isinstance(sub, ast.If)
            ]
            assert any(
                test and "pending_order_manager" in test and "allow_pending" in test
                for test in guard_tests
            ), "_defer_sell_order 缺少 allow_pending 守卫"

    # ── 行为断言 ──
    d0 = _TRADING_DATES[0]
    limit_down = frozenset({("T", d0.strftime("%Y%m%d"))})
    closes = {s: [10.0] * len(_TRADING_DATES) for s in ("T",)}
    price_data = _make_price_data(closes, limit_down=limit_down)

    def _prep(engine):
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine.positions["T"] = {
            "shares": 1000,
            "buy_date": d0,
            "signal_date": d0,
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.0,
            "buy_cost_cash": 10000.0,
        }
        add_calls: List[Dict] = []
        original_add = engine.pending_order_manager.add_order
        engine.pending_order_manager.add_order = lambda *a, **k: (
            add_calls.append((a, k)),
            original_add(*a, **k),
        )[1]
        return add_calls

    old_engine = _make_old_engine(_make_det_signal(old_signal_base.Signal, 2, {}, [("A", 0.9)]))
    new_engine = _make_new_engine(_make_det_signal(new_signal_base.Signal, 2, {}, [("A", 0.9)]))
    old_calls = _prep(old_engine)
    new_calls = _prep(new_engine)

    # 不可交易（跌停）+ allow_pending=False：不入队、不成交（减仓丢弃语义）
    for engine in (old_engine, new_engine):
        engine._sell_stock(
            d0,
            "T",
            sell_type="risk_trim",
            sell_reason="t",
            trigger_type="t",
            fraction=0.5,
            allow_pending=False,
        )
    assert old_calls == [] and new_calls == [], "减仓路径不得触碰延迟队列"
    assert old_engine.trades == new_engine.trades == []
    assert old_engine.positions == new_engine.positions
    assert new_engine.positions["T"]["shares"] == 1000

    # 可交易（次日）+ allow_pending=False：直接成交（部分卖出 500 股）
    d1 = _TRADING_DATES[1]
    for engine in (old_engine, new_engine):
        engine._sell_stock(
            d1,
            "T",
            sell_type="risk_trim",
            sell_reason="t",
            trigger_type="t",
            fraction=0.5,
            allow_pending=False,
        )
    assert old_calls == [] and new_calls == []
    assert old_engine.trades == new_engine.trades
    assert len(new_engine.trades) == 1
    trade = new_engine.trades[0]
    assert trade["sell_type"] == "risk_trim" and trade["shares"] == 500
    assert old_engine.positions == new_engine.positions
    assert new_engine.positions["T"]["shares"] == 500


# ══════════════════ J. reporting 拆分等价性固化（R1-T5-01 + R3-T5-03） ══════════════════

_D0 = _TRADING_DATES[0]
_D1 = _TRADING_DATES[1]
_D2 = _TRADING_DATES[2]


def _warning_items_full() -> Dict[str, List[Dict]]:
    """8 类 warning items 全覆盖的合成台账。"""
    return {
        "early_rebalance": [{"label": "空仓触发", "detail": "无持仓, 新信号入队"}],
        "duplicate_buy": [{"stock": "A", "buy_date": _D0}],
        "position_unfilled": [
            {
                "tranche_tag": "",
                "target_n": 2,
                "actually_bought": 1,
                "unfilled_count": 1,
                "unfilled_stocks": ["B"],
            }
        ],
        "completion_skipped": [{"label": "候选不可交易", "detail": "B(涨停)"}],
        "completion_abandoned": [
            {
                "tranche_tag": "",
                "original_signal_date": _D0,
                "attempts": 2,
                "unfilled_stocks": ["B"],
            }
        ],
        "pending_order_added": [{"stock": "C", "action": "buy", "reason": "涨停"}],
        "pending_order_success": [
            {"stock": "C", "action": "buy", "retry_count": 1, "delay_days": 1}
        ],
        "pending_order_expired": [
            {
                "stock": "D",
                "action": "buy",
                "expire_type": "expired_retry",
                "retry_count": 3,
                "max_retry_count": 2,
                "delay_days": 0,
                "max_retry_days": 10,
            },
            {
                "stock": "E",
                "action": "sell",
                "expire_type": "expired_days",
                "retry_count": 0,
                "max_retry_count": 5,
                "delay_days": 20,
                "max_retry_days": 10,
            },
        ],
    }


def _reporting_stub(mixin, **attrs):
    """reporting 直驱测试桩：绑定被测方法内部委托的同 mixin helper（逐字语义）。

    旧侧单体实现经 ``self._format_compact_items`` / ``self._format_trade_cash_wan``
    等实例委托；新侧 D8 拆分后改为模块级 helper。桩按侧绑定对应 mixin 的
    helper，保证两侧各自执行生产实现。
    """
    stub = SimpleNamespace(**attrs)
    for name in (
        "_format_compact_items",
        "_format_trade_cash_wan",
        "_format_pending_order_group",
        "_calculate_current_exposure_pct",
    ):
        fn = getattr(mixin, name, None)
        if fn is None:
            continue
        descriptor = inspect.getattr_static(mixin, name)
        if isinstance(descriptor, staticmethod):
            setattr(stub, name, fn)
        else:
            setattr(stub, name, (lambda *a, _f=fn, **k: _f(stub, *a, **k)))
    return stub


def test_build_daily_warning_logs_equivalence():
    """_build_daily_warning_logs：8 类全覆盖 / 全空 / 部分空 / 未知标签丢弃，逐串相等。"""
    cases = {
        "8类全覆盖": _warning_items_full(),
        "全空": {},
        "部分空": {
            "early_rebalance": [{"label": "拖尾通过", "detail": "残留10.0%+新信号90.0%=100.0%"}],
            "pending_order_expired": [
                {
                    "stock": "D",
                    "action": "buy",
                    "expire_type": "expired_retry",
                    "retry_count": 3,
                    "max_retry_count": 2,
                    "delay_days": 0,
                    "max_retry_days": 10,
                }
            ],
        },
        # 未知标签（不在固定标签序内）应被丢弃；同时混入合法标签
        "未知标签丢弃": {
            "completion_skipped": [
                {"label": "未知原因X", "detail": "不应出现"},
                {"label": "无候选", "detail": "信号日2026-01-05"},
            ]
        },
    }
    for label, items in cases.items():
        old_stub = _reporting_stub(
            old_reporting.BacktestReportingMixin,
            _daily_warning_items=copy.deepcopy(items),
            completion_window_days=3,
        )
        new_stub = _reporting_stub(
            new_reporting.BacktestReportingMixin,
            _daily_warning_items=copy.deepcopy(items),
            completion_window_days=3,
        )
        old_lines = old_reporting.BacktestReportingMixin._build_daily_warning_logs(old_stub)
        new_lines = new_reporting.BacktestReportingMixin._build_daily_warning_logs(new_stub)
        assert old_lines == new_lines, f"{label} 输出不一致"
        if label == "未知标签丢弃":
            assert all("未知原因X" not in line for line in new_lines)
            assert any("无候选" in line for line in new_lines)
    # 锚定：全覆盖输出 8 行（每类一行），且包含各类前缀
    full_lines = new_reporting.BacktestReportingMixin._build_daily_warning_logs(
        _reporting_stub(
            new_reporting.BacktestReportingMixin,
            _daily_warning_items=_warning_items_full(),
            completion_window_days=3,
        )
    )
    assert len(full_lines) == 8


def test_build_daily_signal_log_equivalence():
    """_build_daily_signal_log：多日 × 调仓/补槽/非 signals 形态 / 6 类卖出标签 / 非当日过滤。"""
    sell_label_cases = [
        ("holding_period", None, "持有期"),
        ("rebalance", None, "调仓"),
        ("mystery_type", None, "条件卖出"),
        (None, "drawdown", "回撤止损"),
        (None, "trailing_stop", "移动止损"),
        (None, "consecutive_limit_down", "连续跌停"),
        (None, "unknown", "止损"),
        (None, None, "止损"),
    ]
    for sell_type, trigger_type, expected_label in sell_label_cases:
        condition_sells = {}
        stop_loss_sells = {}
        if sell_type is not None:
            condition_sells["S1"] = {"trigger_date": _D1, "sell_type": sell_type}
        else:
            entry = {"trigger_date": _D1}
            if trigger_type is not None:
                entry["trigger_type"] = trigger_type
            stop_loss_sells["S1"] = entry
        for pending_signals in (
            {_D1: {"signals": {"A": 0.5, "B": 0.5}, "decision_trace": {"queued": True}}},
            {_D1: {"signals": {"C": 1.0}}},  # 无 decision_trace → 补槽
            {_D1: {"A": 0.5}},  # 非 signals 形态（旧格式）→ 调仓
            {},
        ):
            old_stub = _reporting_stub(
                old_reporting.BacktestReportingMixin,
                pending_signals=copy.deepcopy(pending_signals),
                pending_condition_sells=copy.deepcopy(condition_sells),
                pending_stop_loss_sells=copy.deepcopy(stop_loss_sells),
            )
            new_stub = _reporting_stub(
                new_reporting.BacktestReportingMixin,
                pending_signals=copy.deepcopy(pending_signals),
                pending_condition_sells=copy.deepcopy(condition_sells),
                pending_stop_loss_sells=copy.deepcopy(stop_loss_sells),
            )
            old_log = old_reporting.BacktestReportingMixin._build_daily_signal_log(old_stub, _D1)
            new_log = new_reporting.BacktestReportingMixin._build_daily_signal_log(new_stub, _D1)
            assert old_log == new_log, f"{expected_label} 输出不一致"
            if new_log is not None:
                assert expected_label in new_log, f"{expected_label} 未出现在输出"
    # 非当日过滤 + 全空 → None
    stub_kwargs = dict(
        pending_signals={},
        pending_condition_sells={"S1": {"trigger_date": _D2, "sell_type": "holding_period"}},
        pending_stop_loss_sells={"S2": {"trigger_date": _D0, "trigger_type": "drawdown"}},
    )
    for mixin in (
        old_reporting.BacktestReportingMixin,
        new_reporting.BacktestReportingMixin,
    ):
        assert mixin._build_daily_signal_log(_reporting_stub(mixin, **stub_kwargs), _D1) is None
    # 空 pending_signals 值（{} / None）不产生买入分组
    for empty_value in ({}, None):
        for mixin in (
            old_reporting.BacktestReportingMixin,
            new_reporting.BacktestReportingMixin,
        ):
            stub = _reporting_stub(
                mixin,
                pending_signals={_D1: empty_value},
                pending_condition_sells={},
                pending_stop_loss_sells={},
            )
            assert mixin._build_daily_signal_log(stub, _D1) is None


def test_format_rebalance_decision_summary_equivalence():
    """_format_rebalance_decision_summary：正常/半仓/阻断/None/NaN/非法串/缺键/str 日期矩阵。"""
    base_trace = {
        "signal_date": _D0,
        "target_n": 5,
        "candidate_count": 20,
        "queued": True,
        "final_target_exposure": 1.0,
    }
    cases = {
        "正常满仓": (dict(base_trace), _D1, ""),
        "半仓": (dict(base_trace, final_target_exposure=0.5), _D1, ""),
        "阻断": (dict(base_trace, final_target_exposure=0.0, queued=False), None, ""),
        "None暴露": (dict(base_trace, final_target_exposure=None), _D1, ""),
        "NaN暴露": (dict(base_trace, final_target_exposure=float("nan")), _D1, ""),
        "非法字符串暴露": (dict(base_trace, final_target_exposure="abc"), _D1, ""),
        "缺键trace": ({}, _D1, "[批次 1/2] "),
        "缺键且无执行日": ({}, None, ""),
        "str日期": (
            dict(base_trace, signal_date="2026-01-05", final_target_exposure=0.8),
            "2026-01-06",
            "[批次 2/2] ",
        ),
    }
    for label, (trace, execution_date, tranche_tag) in cases.items():
        old_text = old_reporting._format_rebalance_decision_summary(
            dict(trace), execution_date=execution_date, tranche_tag=tranche_tag
        )
        new_text = new_reporting._format_rebalance_decision_summary(
            dict(trace), execution_date=execution_date, tranche_tag=tranche_tag
        )
        assert old_text == new_text, f"{label} 输出不一致"
    # 锚定：阻断与 NaN 分支文案
    blocked = new_reporting._format_rebalance_decision_summary(
        dict(base_trace, final_target_exposure=0.0, queued=False)
    )
    assert "阻断, 不入队" in blocked
    nan_text = new_reporting._format_rebalance_decision_summary(
        dict(base_trace, final_target_exposure=np.nan)
    )
    assert "N/A" in nan_text


def _trade_log_trades() -> List[Dict]:
    """合成成交记录：跨 2 日、买/卖/无 buy_date 与持有天数可算/不可算分支。"""
    return [
        {"date": _D0, "stock": "A", "action": "buy", "amount": 10000.0, "cost": 13.0},
        {"date": _D0, "stock": "B", "action": "buy", "amount": 20000.0, "cost": 26.0},
        {
            "date": _D1,
            "stock": "A",
            "action": "sell",
            "buy_date": _D0,
            "pnl_profit_pct": 0.0123,
        },
        {
            "date": _D1,
            "stock": "C",
            "action": "sell",
            "buy_date": pd.Timestamp("1999-01-04"),  # 不在交易日映射 → 持有 0 天
            "pnl_profit_pct": -0.005,
        },
    ]


def test_build_daily_trade_log_equivalence():
    """_build_daily_trade_log：多日期 × 2 起点（0 / 中间），逐串相等。"""
    date_to_idx = {d: i for i, d in enumerate(_TRADING_DATES)}
    trades = _trade_log_trades()
    for date, start_idx, has_output in (
        (_D0, 0, True),
        (_D1, 0, True),
        (_D1, 2, True),  # 起点切到次日首笔
        (_D2, 0, False),  # 当日无成交
        (_D2, 4, False),  # 起点越过全部记录
    ):
        old_stub = _reporting_stub(
            old_reporting.BacktestReportingMixin, trades=copy.deepcopy(trades)
        )
        new_stub = _reporting_stub(
            new_reporting.BacktestReportingMixin, trades=copy.deepcopy(trades)
        )
        old_result = old_reporting.BacktestReportingMixin._build_daily_trade_log(
            old_stub, date, start_idx, date_to_idx
        )
        new_result = new_reporting.BacktestReportingMixin._build_daily_trade_log(
            new_stub, date, start_idx, date_to_idx
        )
        assert old_result == new_result, f"date={date.date()} start={start_idx} 不一致"
        assert (len(new_result[2]) > 0) == has_output
    # 锚定：_D1 全量起点的买 0 / 卖 2
    result = new_reporting.BacktestReportingMixin._build_daily_trade_log(
        _reporting_stub(new_reporting.BacktestReportingMixin, trades=_trade_log_trades()),
        _D1,
        0,
        date_to_idx,
    )
    assert result[0] == 0 and result[1] == 2


def _progress_stub(mixin, **overrides):
    """_format_daily_progress_log 测试桩（绑定内部委托保持逐字语义）。"""
    attrs = dict(
        initial_capital=1_000_000.0,
        _last_rebalance_nav=1_000_000.0,
        rebalance_freq=5,
        current_capital=400_000.0,
        positions={"A": {"shares": 100}, "B": {"shares": 200}},
    )
    attrs.update(overrides)
    stub = _reporting_stub(mixin, **attrs)
    stub._get_target_position_count = lambda: 2
    return stub


def test_format_daily_progress_log_equivalence():
    """_format_daily_progress_log：净值基准 None/0/正常 × 空仓/持仓，逐串相等。"""
    for rebalance_nav in (None, 0.0, 1_010_000.0):
        for positions, capital in (
            ({"A": {"shares": 100}}, 500_000.0),
            ({}, 1_000_000.0),
        ):
            for mixin in (
                old_reporting.BacktestReportingMixin,
                new_reporting.BacktestReportingMixin,
            ):
                stub = _progress_stub(
                    mixin,
                    _last_rebalance_nav=rebalance_nav,
                    positions=positions,
                    current_capital=capital,
                )
                text = mixin._format_daily_progress_log(
                    stub,
                    date=_D1,
                    trading_days=6,
                    total_days=25,
                    cycle_day=1,
                    portfolio_value=1_005_000.0,
                    buy_count=1,
                    sell_count=2,
                )
                if mixin is old_reporting.BacktestReportingMixin:
                    old_text = text
                else:
                    new_text = text
            assert (
                old_text == new_text
            ), f"nav={rebalance_nav} positions={len(positions)} 输出不一致"
    # 锚定：nav=None → 本调仓收益 N/A；含仓位百分比
    stub = _progress_stub(new_reporting.BacktestReportingMixin, _last_rebalance_nav=None)
    text = new_reporting.BacktestReportingMixin._format_daily_progress_log(
        stub,
        date=_D1,
        trading_days=6,
        total_days=25,
        cycle_day=1,
        portfolio_value=1_005_000.0,
    )
    assert "N/A" in text


# ══════════════════ I. 禁止真实配置读取永久回归 ══════════════════


class _PassedCounter:
    """inner pytest 运行的通过数记录器（复收证据：复跑非空跑）。"""

    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.outcome == "passed":
            self.passed += 1


def test_t5_suite_forbids_real_config_read(monkeypatch):
    """永久回归（范式同 T4 R2-T4-01）：禁止读取替身 + 计数断言。

    把新旧两侧成本模块绑定的 ``get_cost_settings`` 替换为**立即抛
    RuntimeError 的禁止读取替身**，在该替身下复跑整个 T5 测试文件（本项
    除外，防递归）：必须全绿、真实配置读取 0 次。任一测试绕过合成 fixture
    直接读取真实配置 ⇒ 替身抛出 ⇒ 该测试失败。
    """
    real_reads: List[int] = []

    def _reject_configuration_read():
        real_reads.append(1)
        raise RuntimeError("reviewer: unexpected configuration read")

    monkeypatch.setattr(old_cost, "get_cost_settings", _reject_configuration_read)
    monkeypatch.setattr(new_cost, "get_cost_settings", _reject_configuration_read)

    counter = _PassedCounter()
    exit_code = pytest.main(
        [
            "-q",
            "-k",
            "not test_t5_suite_forbids_real_config_read",
            str(Path(__file__)),
        ],
        plugins=[counter],
    )
    assert exit_code == 0, "禁止真实配置读取下 T5 套件存在失败项"
    assert real_reads == [], f"真实配置读取发生 {len(real_reads)} 次"
    # 复跑非空跑：当前文件 39 项（含本项），inner 应跑 38 项；下限防未来漂移
    assert counter.passed >= 37, f"inner 复跑通过数异常: {counter.passed}"
