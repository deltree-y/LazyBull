# -*- coding: utf-8 -*-
"""P4 步进等价性场景套件（**T1 生产实现版**；阶段专用临时开发文件）。

复杂度豁免：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。

T1 验收增项①（评审处置 P2-6）：本套件由 T0.5b spike 原型**改打生产实现**——
逐日步进 = 生产 ``run_step``（``v2/core/execution/run_loop.py``），快照 =
生产 ``engine_state``（fail-fast 恢复），live 排期 = 生产 ``live_schedule``；
spike 原型副本 ``run_one_day``/宽松恢复已删除，无双实现并存。

变异自证机制变更（T1 验收增项②）：spike 的宽松恢复（缺字段回退工厂初值 ⇒
行为发散）已退役；生产 restore 缺字段/schema 版本不符 ⇒ **fail-fast 拒绝**
（``EngineStateError``）——字段剔除类变异（S4b/S6/S9a）以「恢复被拒绝」为
检出形态；值篡改类变异（S9b 配置指纹预改 / S9c last_scheduled 预改）schema
合法 ⇒ 恢复成功 ⇒ 行为发散检出（形态不变）。

P4 规划 §5-T0.5b 覆盖清单与场景映射（评审处置后 v2）：

| 清单项 | 场景 |
|---|---|
| ≥2 个完整调仓周期 | S1（freq=5 × 6 周期，含长假间隙/周末跨越） |
| 分批 tranche 边界 | S2（K=3 分批 × 6 周期） |
| 延迟订单顺延 | S3（卖出执行日连续 2 日跌停 ⇒ 跨日重试） |
| 止损触发 | S4（回撤止损 T0 触发 T+1 执行）+ S4b（连续跌停计数跨日携带） |
| 待执行买卖与补位 | S5（买入日涨停 ⇒ 补齐窗口跨日） |
| signal_dates 运行期改写 | S6（空仓提前调仓删预定调仓日；兼字段剔除变异基底） |
| ranked_candidates 跨日恢复 | 全场景逐日探针比对 |
| 周末/节假日跨越 | S1（bdate 周末 + 显式 6 自然日长假空洞） |
| live 持久化锚定四场景 | S8（旧链生产函数四场景 + 锚定等价 + 隔离根）+ **S9（v2 引擎侧生产实现：anchor < 驱动首日续跑 / 配置变更提案-提交 / 漏批逐日追赶 / K=1 延迟成功按实际成功日判到期；双侧比较 = live 连续参照 vs 步进恢复，语义参照 = 旧链生产函数 oracle 逐日一致，变异自证全检出）**（R2-03 + 复审四子项处置；T1 验收增项③：live 排期黄金测试首项 = S9a 形态） |
| 窗口末日 T0 计划（无未来行情） | S7（完整日历注入 + price_data 截止于窗口末日；末日前全字段比对、末日白名单豁免，R2-05 处置） |
| 生产配置分支 | S10（kelly + 行业上限 + 单股限权 + min_buy_value_ratio + K=2，r1-P2-5 处置） |

判据：成交逐笔一致 + 持仓逐日一致 + 旁路产物（attribution/holdings/completion）
终态逐值一致 + NAV max|Δ| ≤ 1e-6（实测取精确相等）+ export→restore→export
字节一致 + 变异自证（字段剔除 ⇒ 恢复拒绝 / 值篡改 ⇒ 必发散，**变异结果入总判定**）。

用法：``python scripts/v2_p4/t05b_spike_runner.py [--skip-probe] [--with-calibration]``
（纯合成场景秒级；幂等探针只读生产根、禁网禁写守卫；回放标定 ≈10s 写证据目录内隔离 runs 根）。
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lazybull.universe.base import Universe  # noqa: E402
from src.lazybull.v2.common import cost as new_cost  # noqa: E402
from src.lazybull.v2.common.cost import CostModel  # noqa: E402
from src.lazybull.v2.common.rules.stop_loss import StopLossConfig  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import EngineStateError  # noqa: E402
from src.lazybull.v2.core.execution.engine import BacktestEngine  # noqa: E402
from src.lazybull.v2.core.signal.base import Signal  # noqa: E402

from scripts.v2_p4.t05b_gate2b_binding import write_binding_evidence  # noqa: E402
from scripts.v2_p4.t05b_live_anchor import (  # noqa: E402
    drive_paper_oracle,
    run_anchor_scenarios,
    snapshot_protected_root,
    verify_isolated_root_containment,
)
from scripts.v2_p4.t05b_step_driver import (  # noqa: E402
    RUNTIME_STATE_FIELDS,
    SteppedRunResult,
    compare_runs,
    run_continuous_with_probes,
    run_live_continuous,
    run_stepped,
)

# ── 合成基础设施（范式同 tests/test_v2_p2a_t5_equivalence.py，自包含不依赖 tests/） ──

_COST_SETTINGS = {
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.001,
}

# S10 生产配置镜像（data/paper/config.yaml 现役值，r1-P2-5 处置覆盖）
_PRODUCTION_CONFIG_OVERRIDES = {
    "position_sizing": "kelly",
    "kelly_vol_window": 20,  # 合成小区间收窄窗口（生产 60），保证周期性触发真实 Kelly 分支
    "max_per_industry": 5,
    "max_weight_per_stock": 0.15,
    "stagger_tranches": 2,
    "rebalance_freq": 20,
    "min_buy_value_ratio": 0.2,
    "sell_timing": "open",
}

# 合成场景未覆盖的生产开关清单（报告 §9 边界登记用；门 2 真实小区间覆盖）
_UNCOVERED_CONFIGS = ["enable_risk_budget", "sell_timing=close", "停牌日历样本（非空 suspend）"]


@contextmanager
def synthetic_cost_settings():
    """把 v2 成本模块的配置读取打为合成值（构造器无条件读取配置单例）。"""
    original = new_cost.get_cost_settings
    new_cost.get_cost_settings = lambda: dict(_COST_SETTINGS)
    try:
        yield
    finally:
        new_cost.get_cost_settings = original


class _FakeStorage:
    """Storage 桩：suspend 恒空表（无停牌）；shenwan_industry 供合成行业映射。"""

    shenwan_table: Optional[pd.DataFrame] = None  # 场景装配时注入

    def load_raw_by_date(self, dataset: str, trade_date: str):
        assert dataset == "suspend"
        return pd.DataFrame({"ts_code": [], "suspend_type": []})

    def load_raw(self, dataset: str):
        assert dataset == "shenwan_industry"
        assert self.shenwan_table is not None, "行业约束场景须注入合成申万表"
        return self.shenwan_table


class _StaticUniverse(Universe):
    """静态股票池 stub。"""

    def __init__(self, stocks: List[str]):
        super().__init__("static_stub")
        self._stocks = list(stocks)

    def get_stocks(self, date: pd.Timestamp, quote_data: Optional[pd.DataFrame] = None):
        return list(self._stocks)


class _DeterministicSignal(Signal):
    """确定性分数信号 stub：按日期查表，缺省用 default，过滤到当日股票池。"""

    def __init__(self, top_n: int, plan: Dict[str, List[Tuple[str, float]]], default):
        super().__init__("deterministic_stub")
        self.top_n = top_n
        self._plan = plan
        self._default = default

    def generate(self, date, universe, data):
        return dict(self.generate_ranked(date, universe, data))

    def generate_ranked(self, date, universe, data):
        ranked = self._plan.get(date.strftime("%Y%m%d"), self._default)
        universe_set = set(universe)
        return [(stock, score) for stock, score in ranked if stock in universe_set]


def _make_price_data(
    trading_dates: List[pd.Timestamp],
    closes: Dict[str, List[float]],
    limit_up: frozenset = frozenset(),
    limit_down: frozenset = frozenset(),
) -> pd.DataFrame:
    """合成 price_data：close/open/close_adj/open_adj + 交易状态列。"""
    rows = []
    for i, date in enumerate(trading_dates):
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


def _make_engine_factory(stocks, signal, post_set: Optional[Dict] = None, **overrides):
    def _factory() -> BacktestEngine:
        kwargs = dict(
            universe=_StaticUniverse(stocks),
            signal=signal,
            initial_capital=1_000_000.0,
            rebalance_freq=5,
            verbose=False,
            cost_model=CostModel(**_COST_SETTINGS),
            data_storage=_FakeStorage(),
        )
        kwargs.update(overrides)
        engine = BacktestEngine(**kwargs)
        for attr, value in (post_set or {}).items():
            setattr(engine, attr, value)
        return engine

    return _factory


def _flat_closes(stocks: List[str], n: int, base: float = 10.0) -> Dict[str, List[float]]:
    """确定性温和上行价格（每股基准价递增，日 +0.1%）。"""
    return {
        s: [base * (1 + 0.1 * j) * (1 + 0.001 * i) for i in range(n)] for j, s in enumerate(stocks)
    }


def _calendar_with_holiday() -> List[pd.Timestamp]:
    """30 个交易日：两段 bdate 间夹 6 自然日空洞（模拟长假），周末天然跨越。"""
    block1 = list(pd.bdate_range("2026-01-05", periods=15))
    block2 = list(pd.bdate_range("2026-01-30", periods=15))
    return block1 + block2


def _combine_stepped(first: SteppedRunResult, second: SteppedRunResult) -> SteppedRunResult:
    """两段步进的合并视图（trades/旁路经快照携带已在 second 终态，探针拼接）。"""
    return SteppedRunResult(
        engine_state=second.engine_state,
        probes=first.probes + second.probes,
        snapshot_roundtrip_ok=first.snapshot_roundtrip_ok and second.snapshot_roundtrip_ok,
        last_day_pending_plan=second.last_day_pending_plan,
        live_anchor_state=second.live_anchor_state,
        signal_day_trace=first.signal_day_trace + second.signal_day_trace,
    )


# ── 场景执行 ──


def _run_equivalence_pair(
    name: str,
    make_factory,
    trading_dates: List[pd.Timestamp],
    price_data: pd.DataFrame,
    omit_fields: Tuple[str, ...] = (),
    expect_pass: bool = True,
) -> Dict:
    """同一场景跑连续/步进双侧并比对；``expect_pass=False`` 为变异自证（预期发散）。

    T1 机制变更（生产 fail-fast 恢复）：``omit_fields`` 字段剔除 ⇒ 次日恢复
    被 ``EngineStateError`` 拒绝——该检出形态与「行为发散」等价记为变异检出
    （``passed=False, verdict_ok=True``）。
    """
    start, end = trading_dates[0], trading_dates[-1]
    cont_engine, _nav, cont_probes = run_continuous_with_probes(
        make_factory, start, end, trading_dates, price_data
    )
    try:
        stepped = run_stepped(
            make_factory, start, end, trading_dates, price_data, omit_fields=omit_fields
        )
    except EngineStateError as exc:
        if expect_pass:
            raise
        # 生产 fail-fast 检出形态：快照字段剔除 ⇒ 恢复拒绝（D2 生产契约）
        return {
            "name": name,
            "expect_pass": expect_pass,
            "passed": False,
            "verdict_ok": True,
            "detail": f"恢复拒绝（fail-fast 检出）: {exc}",
            "snapshot_roundtrip_ok": None,
            "sidecar_identical": None,
            "probe_mismatches": [],
            "n_trades": len(cont_engine.trades),
            "n_days": len(cont_probes),
        }
    verdict = compare_runs(cont_engine, cont_probes, stepped)
    passed = verdict.passed and stepped.snapshot_roundtrip_ok
    ok = passed if expect_pass else not passed
    return {
        "name": name,
        "expect_pass": expect_pass,
        "passed": passed,
        "verdict_ok": ok,
        "detail": verdict.detail,
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(cont_engine.trades),
        "n_days": len(cont_probes),
    }


def scenario_s1_multi_cycle_with_holiday() -> Dict:
    """S1：≥2 完整调仓周期（6 周期）+ 周末/长假跨越 + 持仓快照开关。"""
    dates = _calendar_with_holiday()
    stocks = ["A", "B", "C", "D"]
    n = len(dates)
    plan = {}
    for i, d in enumerate(dates):
        plan[d.strftime("%Y%m%d")] = (
            [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
        )
    signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
    factory = _make_engine_factory(stocks, signal, post_set={"record_holdings_snapshot": True})
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    result = _run_equivalence_pair("S1_multi_cycle_holiday", factory, dates, price_data)
    result["seams"] = "调仓周期×6 / 轮换换出 / 周末+6日长假 / holdings_snapshot 状态携带"
    return result


def scenario_s2_stagger_boundary() -> Dict:
    """S2：分批 tranche 边界（K=3，freq=6，6 个完整周期）。"""
    dates = list(pd.bdate_range("2026-01-05", periods=36))
    stocks = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    n = len(dates)
    ranked = [(s, 0.9 - 0.05 * j) for j, s in enumerate(stocks)]
    signal = _DeterministicSignal(6, {}, ranked)
    factory = _make_engine_factory(stocks, signal, rebalance_freq=6, stagger_tranches=3)
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    result = _run_equivalence_pair("S2_stagger_boundary", factory, dates, price_data)
    result["seams"] = "K=3 分批排期偏移 0/2/4 / 槽位拆分 2+2+2 / 批次资金比例"
    return result


def scenario_s3_pending_order_postpone() -> Dict:
    """S3：延迟订单顺延（卖出执行日连续 2 日跌停 ⇒ 延迟队列跨日重试）。"""
    dates = list(pd.bdate_range("2026-01-05", periods=20))
    stocks = ["A", "B", "C", "D"]
    n = len(dates)
    plan = {}
    for i, d in enumerate(dates):
        plan[d.strftime("%Y%m%d")] = (
            [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
        )
    signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
    factory = _make_engine_factory(stocks, signal)
    # idx6/7 = 首个轮换卖出执行日及其次日：A 连续跌停 ⇒ 卖单入延迟队列，idx8 成交
    limit_down = frozenset({("A", dates[6].strftime("%Y%m%d")), ("A", dates[7].strftime("%Y%m%d"))})
    price_data = _make_price_data(dates, _flat_closes(stocks, n), limit_down=limit_down)
    result = _run_equivalence_pair("S3_pending_order_postpone", factory, dates, price_data)
    result["seams"] = "pending_order_manager 队列+三计数器 / 跨 2 日重试 / 跌停解除后成交"
    return result


def scenario_s4_stop_loss() -> Dict:
    """S4：回撤止损触发（T0 触发、T+1 执行）+ pending_stop_loss_sells 跨日。"""
    dates = list(pd.bdate_range("2026-01-05", periods=20))
    stocks = ["A", "B"]
    n = len(dates)
    closes = _flat_closes(stocks, n)
    # A 买入（idx1 收盘）后 idx3 起下跌 10%（回撤 > 5% 触发）
    closes["A"] = [10.0 * (1 + 0.001 * i) if i <= 2 else 9.0 for i in range(n)]
    signal = _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)])
    cfg = StopLossConfig(enabled=True, drawdown_pct=5.0, consecutive_limit_down_days=99)
    factory = _make_engine_factory(stocks, signal, stop_loss_config=cfg)
    price_data = _make_price_data(dates, closes)
    result = _run_equivalence_pair("S4_stop_loss_drawdown", factory, dates, price_data)
    result["seams"] = "止损 T0 触发/T+1 执行 + pending_stop_loss_sells 跨日 + 触发后计数清理"
    return result


def scenario_s4b_stop_loss_limit_down_counter() -> Dict:
    """S4b：连续跌停计数跨日携带（stop_loss_monitor 状态）；兼作变异自证基底。"""
    dates = list(pd.bdate_range("2026-01-05", periods=20))
    stocks = ["A", "B"]
    n = len(dates)
    closes = _flat_closes(stocks, n)
    closes["A"] = [10.0 for i in range(n)]
    signal = _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)])
    cfg = StopLossConfig(enabled=True, drawdown_pct=99.0, consecutive_limit_down_days=2)
    factory = _make_engine_factory(stocks, signal, stop_loss_config=cfg)
    # idx3/4 连续两日跌停（计数 1→2 触发），idx5 解除（可成交）
    limit_down = frozenset({("A", dates[3].strftime("%Y%m%d")), ("A", dates[4].strftime("%Y%m%d"))})
    price_data = _make_price_data(dates, closes, limit_down=limit_down)
    result = _run_equivalence_pair("S4b_stop_loss_limit_down_counter", factory, dates, price_data)
    result["seams"] = "consecutive_limit_down_days 逐日演化（非跌停即清零）跨恢复携带"
    # 变异自证：快照漏 stop_loss_monitor ⇒ 生产 fail-fast 恢复拒绝（检出形态）
    mutant = _run_equivalence_pair(
        "S4b_mutant_drop_stop_loss_monitor",
        factory,
        dates,
        price_data,
        omit_fields=("stop_loss_monitor",),
        expect_pass=False,
    )
    result["mutant"] = mutant
    return result


def scenario_s5_position_completion() -> Dict:
    """S5：买入日涨停 ⇒ 槽位未成交 ⇒ 补齐窗口期跨日补齐（unfilled_slots 状态）。"""
    dates = list(pd.bdate_range("2026-01-05", periods=20))
    stocks = ["A", "B"]
    n = len(dates)
    signal = _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)])
    factory = _make_engine_factory(stocks, signal)
    # 首个 T+1 买入日（idx1）与次日（idx2）B 涨停 ⇒ 第 3 日补齐成交
    limit_up = frozenset({("B", dates[1].strftime("%Y%m%d")), ("B", dates[2].strftime("%Y%m%d"))})
    price_data = _make_price_data(dates, _flat_closes(stocks, n), limit_up=limit_up)
    result = _run_equivalence_pair("S5_position_completion", factory, dates, price_data)
    result["seams"] = "unfilled_slots/completion_stats 跨日演化 + 补齐窗口内再尝试"
    return result


def scenario_s6_early_rebalance_signal_dates_mutation() -> Dict:
    """S6：空仓提前调仓改写 signal_dates（隐藏可变状态）；兼作锚漂移变异自证。"""
    dates = list(pd.bdate_range("2026-01-05", periods=25))
    stocks = ["A", "B", "C", "D"]
    n = len(dates)
    plan = {}
    for i, d in enumerate(dates):
        key = d.strftime("%Y%m%d")
        if i == 10:
            plan[key] = []  # 第二个排期信号日门控阻断：无候选 ⇒ 无信号入队
        else:
            plan[key] = (
                [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
            )
    signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
    # 关闭仓位补齐/补位（否则到期卖出会触发补位买入计划，账户永不清空，
    # 无法走到空仓提前调仓分支）
    factory = _make_engine_factory(stocks, signal, enable_position_completion=False)
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    result = _run_equivalence_pair("S6_early_rebalance_signal_dates", factory, dates, price_data)
    result["seams"] = (
        "持仓到期清空 ⇒ 空仓提前调仓触发 ⇒ _accept_early_rebalance 删除预定调仓日"
        "（_signal_dates 运行期改写必须入快照）"
    )
    # 变异自证：快照漏 _signal_dates ⇒ 生产 fail-fast 恢复拒绝（检出形态）
    mutant = _run_equivalence_pair(
        "S6_mutant_drop_signal_dates",
        factory,
        dates,
        price_data,
        omit_fields=("_signal_dates",),
        expect_pass=False,
    )
    result["mutant"] = mutant
    return result


def scenario_s7_live_last_day_t0_plan() -> Dict:
    """S7：live 形态——完整日历注入，最后行情日无需未来行情生成次日 T0 计划。

    R2-05/r1-P2-4 处置：**末日前 20 日全字段比对**（排期键范围归一化到窗口末日），
    末日仅对白名单字段豁免（pending_signal_dates/last_signal_date/
    last_ranked_candidates/signal_dates_remaining 因计划入队合法分叉）；
    末日计划内容单独断言。
    """
    dates = list(pd.bdate_range("2026-01-05", periods=21))  # idx20 为排期信号日（freq=5）
    future_dates = list(pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=3))
    full_calendar = dates + future_dates
    stocks = ["A", "B", "C", "D"]
    n = len(dates)
    plan = {}
    for i, d in enumerate(dates):
        plan[d.strftime("%Y%m%d")] = (
            [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
        )
    signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
    factory = _make_engine_factory(stocks, signal)
    # price_data 严格截止于窗口末日（未来交易日零行情）
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    start, end = dates[0], dates[-1]
    end_str = end.strftime("%Y%m%d")

    cont_engine, _nav, cont_probes = run_continuous_with_probes(
        factory, start, end, dates, price_data
    )
    stepped = run_stepped(factory, start, end, full_calendar, price_data, live_full_calendar=True)

    # 末日前：全字段比对（排期剩余键裁剪到窗口末日——live 侧窗口外未来排期属合法差异）
    pre_view = SteppedRunResult(engine_state=stepped.engine_state, probes=stepped.probes[:-1])
    verdict_pre = compare_runs(cont_engine, cont_probes[:-1], pre_view, signal_dates_max=end_str)
    # 末日：仅白名单豁免，其余字段照比（持仓/估值/资金/笔数 + 排期裁剪后比对）
    last_view = SteppedRunResult(engine_state=stepped.engine_state, probes=stepped.probes[-1:])
    verdict_last = compare_runs(
        cont_engine,
        cont_probes[-1:],
        last_view,
        probe_fields=("last_rebalance_nav", "portfolio_value", "capital", "n_trades"),
        signal_dates_max=end_str,
    )

    # 核心断言：最后行情日（idx20，排期信号日）在零未来行情下生成次日 T0 计划
    plan_info = stepped.last_day_pending_plan
    plan_ok = (
        plan_info is not None
        and plan_info["signal_date"] == end_str
        and plan_info["n_signals"] == 2
    )
    cont_last_day_plan = cont_engine.pending_signals.get(end)
    passed = (
        verdict_pre.passed
        and verdict_last.passed
        and stepped.snapshot_roundtrip_ok
        and plan_ok
        and cont_last_day_plan is None
    )
    return {
        "name": "S7_live_last_day_t0_plan",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed,
        "detail": f"末日前: {verdict_pre.detail}; 末日: {verdict_last.detail}",
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict_pre.sidecar_identical and verdict_last.sidecar_identical,
        "probe_mismatches": (verdict_pre.probe_mismatches + verdict_last.probe_mismatches)[:5],
        "n_trades": len(cont_engine.trades),
        "n_days": len(cont_probes),
        "last_day_t0_plan": plan_info,
        "last_day_t0_plan_ok": plan_ok,
        "continuous_last_day_plan_absent": cont_last_day_plan is None,
        "price_data_max_date": max(price_data["trade_date"]),
        "calendar_extends_to": full_calendar[-1].strftime("%Y%m%d"),
        "seams": (
            "完整日历注入（含 3 个无行情未来日）/ T+1 解析跨窗口末日 / "
            "T0 收单零未来行情依赖（_resolve_buy_date_quote 行情无下游消费）；"
            "末日前 20 日全字段逐日比对（排期键范围归一），末日白名单豁免 + 计划内容断言"
        ),
    }


def _s9_base() -> Tuple[List[pd.Timestamp], List[str], pd.DataFrame, List]:
    """S9 共用：22 交易日日历 + 6 股恒定排序候选 + 行情。

    候选 = 全 6 股恒定排序（引擎经已持仓排除自动轮换 AB→CD→EF），
    保证调仓日永不因「候选全持仓」隐性阻断（oracle 语义参照须排除该噪声）。
    """
    dates = list(pd.bdate_range("2026-01-05", periods=22))
    stocks = ["A", "B", "C", "D", "E", "F"]
    n = len(dates)
    ranked = [(s, 0.9 - 0.05 * j) for j, s in enumerate(stocks)]
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    return dates, stocks, price_data, ranked


def scenario_s9a_live_resume_anchor_before_window(tmp_root: Path) -> Dict:
    """S9a：anchor 早于驱动首日的重启续跑（live 形态引擎接线，R2-03 处置）。

    阶段1 驱动 dates[0..5]（含信号日 0/5）⇒ 落盘（锚定状态随快照同代）；
    阶段2 自快照恢复驱动 dates[6..21]，锚 = dates[0]（< 驱动首日）；
    对照 = 连续 run 全程。变异：落盘 blob 删 ``live_anchor`` 键
    （锚定状态未落盘 ⇒ 恢复日重锚，D1 禁止形态）⇒ 必发散。
    """
    dates, stocks, price_data, ranked = _s9_base()
    signal = _DeterministicSignal(2, {}, ranked)
    factory = _make_engine_factory(stocks, signal)  # freq=5, K=1

    cont_engine, _nav, cont_probes = run_continuous_with_probes(
        factory, dates[0], dates[-1], dates, price_data
    )
    phase1 = run_stepped(
        factory,
        dates[0],
        dates[5],
        dates,
        price_data,
        live_full_calendar=True,
        anchor_date=dates[0],
    )
    phase2 = run_stepped(
        factory,
        dates[6],
        dates[-1],
        dates,
        price_data,
        live_full_calendar=True,
        resume_state=phase1.engine_state,
    )
    combined = _combine_stepped(phase1, phase2)
    verdict = compare_runs(cont_engine, cont_probes, combined)
    passed = verdict.passed and combined.snapshot_roundtrip_ok

    # 变异：锚定状态未随快照落盘（删 blob 键）⇒ 生产 fail-fast 恢复拒绝（检出形态；
    # spike 宽松恢复的「恢复日重锚 ⇒ 发散」形态已被生产契约取代）
    mutant_blob = pickle.loads(pickle.dumps(phase1.engine_state))
    del mutant_blob["fields"]["live_anchor_state"]
    try:
        run_stepped(
            factory,
            dates[6],
            dates[-1],
            dates,
            price_data,
            live_full_calendar=True,
            resume_state=mutant_blob,
        )
        mutant_detected = False
        mutant_detail = "锚定状态缺失未被拒绝（fail-fast 失效）"
    except EngineStateError as exc:
        mutant_detected = True
        mutant_detail = f"恢复拒绝（fail-fast 检出）: {exc}"
    mutant = {
        "name": "S9a_mutant_anchor_state_not_persisted",
        "expect_pass": False,
        "passed": not mutant_detected,
        "verdict_ok": mutant_detected,
        "detail": mutant_detail,
        "probe_mismatches": [],
    }
    return {
        "name": "S9a_live_resume_anchor_before_window",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed and mutant["verdict_ok"],
        "detail": verdict.detail,
        "snapshot_roundtrip_ok": combined.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(cont_engine.trades),
        "n_days": len(cont_probes),
        "signal_day_trace": combined.signal_day_trace,
        "anchor_state_final": combined.live_anchor_state.to_dict(),
        "mutant": mutant,
        "seams": (
            "锚定状态随快照同代落盘/恢复（blob live_anchor 键，含每日往返断言覆盖）；"
            "anchor=dates[0] < 驱动首日 dates[6] 恢复续跑 vs 连续全程逐位一致；"
            "pending_signals 跨进程边界携带；变异=落盘缺失重锚 ⇒ 检出"
        ),
    }


def scenario_s9b_live_config_change_rebuild(tmp_root: Path) -> Dict:
    """S9b：恢复时配置变更（K=1，freq 5→4）⇒ 提案当日生效、成功 T0 后提交（R2 复审处置）。

    双侧比较：步进（阶段1 freq=5 → 落盘 → 阶段2 freq=4 恢复）vs live 连续参照
    （单引擎跨日 + 同政策 + 配置事件当日生效，无快照往返）：成交/持仓/NAV/探针
    逐位一致。语义参照：旧链生产 ``_check_rebalance_day`` 逐日驱动（oracle）
    与引擎侧调仓日轨迹逐日一致。变异：blob 配置指纹预改（变更未检出）
    ⇒ 恢复首日无信号 ⇒ 发散。
    """
    dates, stocks, price_data, ranked = _s9_base()
    day6_str = dates[6].strftime("%Y%m%d")
    factory_p1 = _make_engine_factory(stocks, _DeterministicSignal(2, {}, ranked))
    factory_p2 = _make_engine_factory(
        stocks, _DeterministicSignal(2, {}, ranked), rebalance_freq=4, holding_period=5
    )
    phase1 = run_stepped(
        factory_p1,
        dates[0],
        dates[5],
        dates,
        price_data,
        live_full_calendar=True,
        anchor_date=dates[0],
    )
    phase2 = run_stepped(
        factory_p2,
        dates[6],
        dates[-1],
        dates,
        price_data,
        live_full_calendar=True,
        resume_state=phase1.engine_state,
    )
    stepped = _combine_stepped(phase1, phase2)

    factory_c = _make_engine_factory(stocks, _DeterministicSignal(2, {}, ranked))
    cont_engine, cont_probes, _anchor_c, cont_trace = run_live_continuous(
        factory_c,
        dates,
        dates,
        price_data,
        anchor_date=dates[0],
        config_events={day6_str: {"rebalance_freq": 4}},
    )
    verdict = compare_runs(cont_engine, cont_probes, stepped)

    oracle = drive_paper_oracle(
        tmp_root / "s9b_oracle",
        [d.strftime("%Y%m%d") for d in dates],
        5,
        1,
        config_events={day6_str: (4, 1)},
    )
    oracle_days = [d for d, _t, _p in oracle]
    engine_days = [d for d, _t in stepped.signal_day_trace]
    phase2_days = [d for d, _t in phase2.signal_day_trace]
    oracle_match = oracle_days == engine_days
    n_trades_ok = len(stepped.engine_state["fields"]["trades"]) > 0
    rebuild_ok = bool(phase2_days) and phase2_days[0] == day6_str  # 变更当日即调仓日
    fingerprint_ok = stepped.live_anchor_state.rebalance_freq == 4
    passed = (
        verdict.passed
        and oracle_match
        and n_trades_ok
        and rebuild_ok
        and fingerprint_ok
        and stepped.snapshot_roundtrip_ok
    )

    # 变异：配置指纹被预改（变更未检出）⇒ 旧配置语义续跑（首日无信号）
    # （值篡改 schema 合法 ⇒ 恢复成功 ⇒ 行为发散检出；字段剔除类见 S9a）
    mutant_blob = pickle.loads(pickle.dumps(phase1.engine_state))
    mutant_blob["fields"]["live_anchor_state"]["rebalance_freq"] = 4
    phase2_mutant = run_stepped(
        _make_engine_factory(
            stocks, _DeterministicSignal(2, {}, ranked), rebalance_freq=4, holding_period=5
        ),
        dates[6],
        dates[-1],
        dates,
        price_data,
        live_full_calendar=True,
        resume_state=mutant_blob,
    )
    mutant_days = [d for d, _t in phase2_mutant.signal_day_trace]
    mutant_diverges = mutant_days != engine_days and day6_str not in mutant_days
    mutant = {
        "name": "S9b_mutant_config_change_undetected",
        "expect_pass": False,
        "passed": not mutant_diverges,
        "verdict_ok": mutant_diverges,
        "detail": f"变异信号日={mutant_days}（旧配置语义：恢复首日无信号）",
    }
    return {
        "name": "S9b_live_config_change_rebuild",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed and mutant["verdict_ok"],
        "detail": (
            f"双侧等价={verdict.passed}（{verdict.detail}）；oracle 轨迹一致={oracle_match} "
            f"{engine_days}；成交 {len(stepped.engine_state['fields']['trades'])} 笔"
        ),
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(stepped.engine_state["fields"]["trades"]),
        "signal_day_trace": stepped.signal_day_trace,
        "oracle_trace": oracle,
        "anchor_state_final": stepped.live_anchor_state.to_dict(),
        "mutant": mutant,
        "seams": "配置变更提案-提交（T0 失败不提交）+ 双侧比较（live 连续 vs 步进恢复）+ 旧链生产函数 oracle 逐日一致",
    }


def scenario_s9c_live_missed_batch_catchup(tmp_root: Path) -> Dict:
    """S9c：漏批逐日追赶（K=2 分批，停机跨过 3 个计划日）⇒ 重启后逐日消化（R2 复审处置）。

    双侧比较：步进（阶段1 days 0..4 ⇒ 落盘 ⇒ 阶段2 days 11..25 恢复）vs
    live 连续参照（单引擎、同事件序列 days 0..4 + 11..25、同政策、无快照往返）。
    语义参照：旧链生产函数 oracle（同停机空洞）。变异：blob 的 last_scheduled
    预置到重启日 ⇒ 漏批丢失 ⇒ 发散。
    """
    dates = list(pd.bdate_range("2026-01-05", periods=26))
    stocks = ["A", "B", "C", "D", "E", "F"]
    n = len(dates)
    ranked = [(s, 0.9 - 0.05 * j) for j, s in enumerate(stocks)]
    price_data = _make_price_data(dates, _flat_closes(stocks, n))

    factory1 = _make_engine_factory(
        stocks, _DeterministicSignal(4, {}, ranked), rebalance_freq=6, stagger_tranches=2
    )
    phase1 = run_stepped(
        factory1,
        dates[0],
        dates[4],
        dates,
        price_data,
        live_full_calendar=True,
        anchor_date=dates[0],
    )
    factory2 = _make_engine_factory(
        stocks, _DeterministicSignal(4, {}, ranked), rebalance_freq=6, stagger_tranches=2
    )
    phase2 = run_stepped(
        factory2,
        dates[11],
        dates[-1],
        dates,
        price_data,
        live_full_calendar=True,
        resume_state=phase1.engine_state,
    )
    stepped = _combine_stepped(phase1, phase2)

    drive_days = dates[:5] + dates[11:]
    factory_c = _make_engine_factory(
        stocks, _DeterministicSignal(4, {}, ranked), rebalance_freq=6, stagger_tranches=2
    )
    cont_engine, cont_probes, _anchor_c, cont_trace = run_live_continuous(
        factory_c, drive_days, dates, price_data, anchor_date=dates[0]
    )
    verdict = compare_runs(cont_engine, cont_probes, stepped)

    oracle = drive_paper_oracle(
        tmp_root / "s9c_oracle",
        [d.strftime("%Y%m%d") for d in dates],
        6,
        2,
        drive_days=[d.strftime("%Y%m%d") for d in drive_days],
    )
    oracle_pairs = [(d, t) for d, t, _p in oracle]
    engine_pairs = stepped.signal_day_trace
    oracle_match = oracle_pairs == engine_pairs
    n_trades_ok = len(stepped.engine_state["fields"]["trades"]) > 0
    passed = verdict.passed and oracle_match and n_trades_ok and stepped.snapshot_roundtrip_ok

    # 变异：无追赶（last_scheduled 预置到重启日）⇒ 漏批 6/9 丢失
    mutant_blob = pickle.loads(pickle.dumps(phase1.engine_state))
    mutant_blob["fields"]["live_anchor_state"]["last_scheduled_rebalance_date"] = dates[11]
    factory_m = _make_engine_factory(
        stocks, _DeterministicSignal(4, {}, ranked), rebalance_freq=6, stagger_tranches=2
    )
    phase2_mutant = run_stepped(
        factory_m,
        dates[11],
        dates[-1],
        dates,
        price_data,
        live_full_calendar=True,
        resume_state=mutant_blob,
    )
    mutant_pairs = phase2_mutant.signal_day_trace
    mutant_diverges = mutant_pairs != engine_pairs
    mutant = {
        "name": "S9c_mutant_no_catchup",
        "expect_pass": False,
        "passed": not mutant_diverges,
        "verdict_ok": mutant_diverges,
        "detail": f"变异信号日={[d for d, _t in mutant_pairs]}（漏批 6/9 丢失）",
    }
    return {
        "name": "S9c_live_missed_batch_catchup",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed and mutant["verdict_ok"],
        "detail": (
            f"双侧等价={verdict.passed}（{verdict.detail}）；oracle 轨迹一致={oracle_match}；"
            f"追赶轨迹={engine_pairs}"
        ),
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(stepped.engine_state["fields"]["trades"]),
        "signal_day_trace": engine_pairs,
        "oracle_trace": oracle,
        "anchor_state_final": stepped.live_anchor_state.to_dict(),
        "mutant": mutant,
        "seams": "due=(last_scheduled, today]∩排期 逐日消化最早未完成批次（每日一批）；tranche 身份随批次计划日；双侧比较 + 旧链 oracle",
    }


def scenario_s9d_k1_delayed_success(tmp_root: Path) -> Dict:
    """S9d：K=1 延迟成功——按**实际成功日**计到期（reviewer2 复审实证案例复刻）。

    排期日 idx5（20260112）T0 门控阻断（候选为空）⇒ 不提交；次日 idx6
    （20260113）成功 ⇒ 下一轮到期按实际成功日 +5 交易日 = idx11（20260120）；
    idx10（20260119）距实际成功日仅 4 个交易日 ⇒ **不到期**（旧链语义，
    与按计划日推算的形态分叉）。双侧比较 + 旧链 oracle 逐日一致。
    """
    dates = list(pd.bdate_range("2026-01-05", periods=22))
    # live 形态：日历含未来交易日（生产 trade_cal 超前行情），
    # 避免窗口末日 T+1 守卫吞掉到期日（S7 已证该守卫语义）
    full_calendar = dates + list(pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=5))
    stocks = ["A", "B", "C", "D", "E", "F"]
    n = len(dates)
    ranked = [(s, 0.9 - 0.05 * j) for j, s in enumerate(stocks)]
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    blocked_day = dates[5].strftime("%Y%m%d")  # 20260112
    plan = {blocked_day: []}  # 该日 T0 无候选 ⇒ 未成功
    # 关闭拖尾/空仓提前调仓：隔离 K=1 判到期接缝（本场景不验引擎提前调仓分支）
    factory = _make_engine_factory(
        stocks, _DeterministicSignal(2, plan, ranked), enable_early_rebalance_on_empty=False
    )

    stepped = run_stepped(
        factory,
        dates[0],
        dates[-1],
        full_calendar,
        price_data,
        live_full_calendar=True,
        anchor_date=dates[0],
    )
    cont_engine, cont_probes, _anchor_c, cont_trace = run_live_continuous(
        _make_engine_factory(
            stocks,
            _DeterministicSignal(2, plan, ranked),
            enable_early_rebalance_on_empty=False,
        ),
        dates,
        full_calendar,
        price_data,
        anchor_date=dates[0],
    )
    verdict = compare_runs(cont_engine, cont_probes, stepped)
    oracle = drive_paper_oracle(
        tmp_root / "s9d_oracle",
        [d.strftime("%Y%m%d") for d in full_calendar],
        5,
        1,
        blocked=frozenset({blocked_day}),
        drive_days=[d.strftime("%Y%m%d") for d in dates],
    )
    oracle_days = [d for d, _t, _p in oracle]
    engine_days = [d for d, _t in stepped.signal_day_trace]
    oracle_match = oracle_days == engine_days
    # reviewer2 实证案例断言：0119（距实际成功日 0113 仅 4 交易日）不到期，0120 到期
    case_ok = "20260119" not in engine_days and "20260120" in engine_days
    n_trades_ok = len(stepped.engine_state["fields"]["trades"]) > 0
    passed = (
        verdict.passed
        and oracle_match
        and case_ok
        and n_trades_ok
        and stepped.snapshot_roundtrip_ok
    )
    return {
        "name": "S9d_k1_delayed_success",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed,
        "detail": (
            f"双侧等价={verdict.passed}（{verdict.detail}）；oracle 一致={oracle_match}；"
            f"信号日={engine_days}；0119 不到期/0120 到期断言={case_ok}"
        ),
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(stepped.engine_state["fields"]["trades"]),
        "signal_day_trace": stepped.signal_day_trace,
        "oracle_trace": oracle,
        "anchor_state_final": stepped.live_anchor_state.to_dict(),
        "seams": "K=1 按实际成功日（last_rebalance_date）数交易日判到期；计划日未成功不推进状态；与旧链 _check_single_rebalance_day 逐日一致",
    }


def scenario_s9e_live_early_rebalance_k2(tmp_root: Path) -> Dict:
    """S9e：live K>1 空仓提前调仓（P1-1 处置回归）。

    构造（引擎级真实空仓事件）：day6/day9 两批 T0 均成功提交，但其 T+1 买入日
    候选全部涨停（买入进延迟队列不成交，不补齐）⇒ 存量持仓到期清空 ⇒
    空仓日（0119，非排期日）触发空仓提前调仓。断言：锚/last_scheduled =
    提前调仓当日，后续排期自该日重建（**不产生持久表删除回流**——P1-1 修复）；
    双侧比较（live 连续 vs 步进恢复逐位一致）+ 旧链 oracle（force 当日）逐日一致。
    """
    dates = list(pd.bdate_range("2026-01-05", periods=26))
    full_calendar = dates + list(pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=5))
    stocks = ["A", "B", "C", "D"]
    n = len(dates)
    plan = {}
    for i, d in enumerate(dates):
        plan[d.strftime("%Y%m%d")] = (
            [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
        )
    signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
    factory = _make_engine_factory(
        stocks, signal, rebalance_freq=6, stagger_tranches=2, enable_position_completion=False
    )
    # C/D 在 idx7..idx12（day6/day9 两批的 T+1 买入窗口）涨停 ⇒ 买入延迟不成交
    limit_up = frozenset(
        {(s, dates[i].strftime("%Y%m%d")) for s in ("C", "D") for i in range(7, 13)}
    )
    price_data = _make_price_data(dates, _flat_closes(stocks, n), limit_up=limit_up)

    stepped = run_stepped(
        factory,
        dates[0],
        dates[-1],
        full_calendar,
        price_data,
        live_full_calendar=True,
        anchor_date=dates[0],
    )
    cont_engine, cont_probes, _anchor_c, cont_trace = run_live_continuous(
        factory, dates, full_calendar, price_data, anchor_date=dates[0]
    )
    verdict = compare_runs(cont_engine, cont_probes, stepped)

    trace = stepped.signal_day_trace
    trace_days = [d for d, _t in trace]
    early_day = "20260119"
    # 提前调仓事件断言：0119 不在初始排期表（0105 锚：t0=0105/0113/0121/0129/0206）
    anchor = stepped.live_anchor_state
    early_ok = (
        early_day in trace_days
        and dict(trace)[early_day] == 0
        and anchor is not None
        and anchor.anchor_date.strftime("%Y%m%d")
        >= early_day  # 锚已前移到提前调仓日（或其后批次0）
    )
    # 提前调仓后下一批 = 自该日重建（0119+3 交易日 = 0122 批1；旧持久表形态会给 0121 批0）
    rebuild_ok = "20260122" in trace_days and dict(trace)["20260122"] == 1
    oracle = drive_paper_oracle(
        tmp_root / "s9e_oracle",
        [d.strftime("%Y%m%d") for d in full_calendar],
        6,
        2,
        force_days=frozenset({early_day}),
    )
    oracle_pairs = [(d, t) for d, t, _p in oracle]
    # oracle 覆盖到窗口末日（引擎末日 T0 计划无 T+1 行情不消费——排期轨迹全期对齐）
    oracle_match = oracle_pairs[: len(trace)] == trace
    n_trades_ok = len(stepped.engine_state["fields"]["trades"]) > 0
    passed = (
        verdict.passed
        and early_ok
        and rebuild_ok
        and oracle_match
        and n_trades_ok
        and stepped.snapshot_roundtrip_ok
    )
    return {
        "name": "S9e_live_early_rebalance_k2",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed,
        "detail": (
            f"双侧等价={verdict.passed}（{verdict.detail}）；提前调仓日 {early_day} 在轨；"
            f"oracle 一致={oracle_match}；轨迹={trace}"
        ),
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(stepped.engine_state["fields"]["trades"]),
        "signal_day_trace": trace,
        "oracle_trace": oracle,
        "anchor_state_final": anchor.to_dict() if anchor else None,
        "seams": "空仓提前调仓（T+1 买入涨停×到期清空）⇒ 锚前移当日 + 排期自该日重建（无删除回流）",
    }


def scenario_s9f_first_failure_then_success(tmp_root: Path) -> Dict:
    """S9f：K>1 首次 T0 失败后成功（R2-P4-T1-01 处置回归）。

    day0（排期首日）信号为空 ⇒ T0 失败零提交；day1 成功 ⇒ 锚/last_scheduled =
    **成功日**（而非失败的计划日）；后续批次自成功日重建。双侧比较 + 旧链
    oracle（blocked={day0}）逐日一致 + 后续至少两批日期锁定。
    """
    dates = list(pd.bdate_range("2026-01-05", periods=20))
    full_calendar = dates + list(pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=3))
    stocks = ["A", "B", "C", "D"]
    ranked = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
    plan = {dates[0].strftime("%Y%m%d"): []}  # 首日无候选 ⇒ T0 失败
    signal = _DeterministicSignal(2, plan, ranked)
    factory = _make_engine_factory(stocks, signal, rebalance_freq=6, stagger_tranches=2)
    price_data = _make_price_data(dates, _flat_closes(stocks, len(dates)))

    stepped = run_stepped(
        factory,
        dates[0],
        dates[-1],
        full_calendar,
        price_data,
        live_full_calendar=True,
        anchor_date=dates[0],
    )
    cont_engine, cont_probes, _anchor_c, cont_trace = run_live_continuous(
        factory, dates, full_calendar, price_data, anchor_date=dates[0]
    )
    verdict = compare_runs(cont_engine, cont_probes, stepped)

    anchor = stepped.live_anchor_state
    first_day = dates[1].strftime("%Y%m%d")
    trace_days = [d for d, _t in stepped.signal_day_trace]
    # R2-P4-T1-01 锁定：首次成功 = day1（0106）；第二批 = 0109（自成功日 +3 交易日，
    # 证明排期自成功日 0106 重建而非失败的计划日 0105）；锚定状态在册
    anchor_ok = (
        trace_days[:2] == [first_day, "20260109"]
        and anchor is not None
        and anchor.last_rebalance_date is not None
    )
    oracle = drive_paper_oracle(
        tmp_root / "s9f_oracle",
        [d.strftime("%Y%m%d") for d in full_calendar],
        6,
        2,
        blocked=frozenset({dates[0].strftime("%Y%m%d")}),
    )
    oracle_pairs = [(d, t) for d, t, _p in oracle]
    oracle_match = oracle_pairs[: len(stepped.signal_day_trace)] == stepped.signal_day_trace
    n_trades_ok = len(stepped.engine_state["fields"]["trades"]) > 0
    passed = (
        verdict.passed
        and anchor_ok
        and oracle_match
        and n_trades_ok
        and stepped.snapshot_roundtrip_ok
    )
    return {
        "name": "S9f_first_failure_then_success",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed,
        "detail": (
            f"双侧等价={verdict.passed}（{verdict.detail}）；首成功后锚={first_day}；"
            f"oracle 一致={oracle_match}；轨迹={stepped.signal_day_trace}"
        ),
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict.sidecar_identical,
        "probe_mismatches": verdict.probe_mismatches[:5],
        "n_trades": len(stepped.engine_state["fields"]["trades"]),
        "signal_day_trace": stepped.signal_day_trace,
        "oracle_trace": oracle,
        "anchor_state_final": anchor.to_dict() if anchor else None,
        "seams": "首次 T0 失败零提交；首次成功锚 = 成功日；后续批次自成功日重建",
    }


def scenario_s9g_calendar_extension(tmp_root: Path) -> Dict:
    """S9g：日历水位延长（同前缀）+ 配置不变 ⇒ 排期自动扩展（R2-P4-T1-02 处置回归）。

    阶段1 以 12 日短日历驱动 6 日 ⇒ 落盘；阶段2 恢复后以 26 日延长日历续跑。
    旧形态（持久表不扩展）会在表耗尽后停止调仓；修复后 = 每日自锚以当前日历
    重建 ⇒ 延长段排期继续。oracle 以延长日历全程驱动作语义参照。
    """
    dates_short = list(pd.bdate_range("2026-01-05", periods=12))
    dates_full = list(pd.bdate_range("2026-01-05", periods=26))
    stocks = ["A", "B", "C", "D"]
    ranked = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
    signal = _DeterministicSignal(2, {}, ranked)
    factory = _make_engine_factory(stocks, signal, rebalance_freq=6, stagger_tranches=2)
    price_data = _make_price_data(dates_full, _flat_closes(stocks, len(dates_full)))

    phase1 = run_stepped(
        factory,
        dates_full[0],
        dates_short[5],
        dates_short,
        price_data,
        live_full_calendar=True,
        anchor_date=dates_full[0],
    )
    phase2 = run_stepped(
        factory,
        dates_full[6],
        dates_full[-1],
        dates_full,
        price_data,
        live_full_calendar=True,
        resume_state=phase1.engine_state,
    )
    stepped = _combine_stepped(phase1, phase2)
    cont_engine, cont_probes, _anchor_c, cont_trace = run_live_continuous(
        factory, dates_full, dates_full, price_data, anchor_date=dates_full[0]
    )
    # 日历延长场景的合法分叉面：阶段1 步进侧排期记录基于 12 日短日历，
    # 连续参照侧始终持 26 日全日历——signal_dates_remaining 字段在阶段1
    # 结构性不同（这正是本场景验证的对象），阶段2（延长后）恢复全字段比对
    n_phase1 = len(phase1.probes)
    verdict_p1 = compare_runs(
        cont_engine,
        cont_probes[:n_phase1],
        SteppedRunResult(engine_state=stepped.engine_state, probes=stepped.probes[:n_phase1]),
        probe_fields=(
            "pending_signal_dates",
            "last_signal_date",
            "last_rebalance_nav",
            "last_ranked_candidates",
            "portfolio_value",
            "capital",
            "n_trades",
        ),
    )
    verdict_p2 = compare_runs(
        cont_engine,
        cont_probes[n_phase1:],
        SteppedRunResult(engine_state=stepped.engine_state, probes=stepped.probes[n_phase1:]),
    )
    verdict_passed = verdict_p1.passed and verdict_p2.passed

    oracle = drive_paper_oracle(
        tmp_root / "s9g_oracle",
        [d.strftime("%Y%m%d") for d in dates_full],
        6,
        2,
    )
    trace = stepped.signal_day_trace
    oracle_pairs = [(d, t) for d, t, _p in oracle]
    oracle_match = oracle_pairs[: len(trace)] == trace
    # 延长段排期继续：阶段2（自 dates_full[6] 起）仍有信号日
    extension_ok = any(d >= dates_full[6].strftime("%Y%m%d") for d, _t in trace)
    n_trades_ok = len(stepped.engine_state["fields"]["trades"]) > 0
    passed = (
        verdict_passed
        and oracle_match
        and extension_ok
        and n_trades_ok
        and stepped.snapshot_roundtrip_ok
    )
    return {
        "name": "S9g_calendar_extension",
        "expect_pass": True,
        "passed": passed,
        "verdict_ok": passed,
        "detail": (
            f"双侧等价={verdict_passed}（阶段1: {verdict_p1.detail}；阶段2: {verdict_p2.detail}）；"
            f"延长段排期继续={extension_ok}；oracle 一致={oracle_match}；轨迹={trace}"
        ),
        "snapshot_roundtrip_ok": stepped.snapshot_roundtrip_ok,
        "sidecar_identical": verdict_p1.sidecar_identical and verdict_p2.sidecar_identical,
        "probe_mismatches": (verdict_p1.probe_mismatches + verdict_p2.probe_mismatches)[:5],
        "n_trades": len(stepped.engine_state["fields"]["trades"]),
        "signal_day_trace": trace,
        "oracle_trace": oracle,
        "seams": "日历延长后每日自锚重建排期（保留锚与已履行计划日）；不复活已删除条目",
    }


def scenario_s10_production_config() -> Dict:
    """S10：生产配置镜像分支覆盖（kelly + 行业上限 + 单股限权 + min_buy + K=2，r1-P2-5）。"""
    dates = list(pd.bdate_range("2026-01-05", periods=60))  # 3 个完整周期（freq=20）
    stocks = ["A", "B", "C", "D", "E", "F"]
    industries = {"A": "X", "B": "X", "C": "X", "D": "Y", "E": "Y", "F": "Y"}
    _FakeStorage.shenwan_table = pd.DataFrame(
        {"ts_code": stocks, "sw_l1": [industries[s] for s in stocks]}
    )
    n = len(dates)
    ranked = [(s, 0.9 - 0.05 * j) for j, s in enumerate(stocks)]
    signal = _DeterministicSignal(4, {}, ranked)
    overrides = dict(_PRODUCTION_CONFIG_OVERRIDES)
    overrides["max_per_industry"] = 2  # 合成收紧以实际触发行业约束分支（生产=5）
    overrides["max_weight_per_stock"] = 0.3  # 合成 top_n=4 须满足 0.3*4>=1（生产 0.15×更大 top_n）
    overrides["stock_basic"] = pd.DataFrame(
        {"ts_code": stocks, "industry": [industries[s] for s in stocks]}
    )
    overrides["industry_reader"] = lambda: _FakeStorage.shenwan_table
    factory = _make_engine_factory(stocks, signal, **overrides)
    price_data = _make_price_data(dates, _flat_closes(stocks, n))
    try:
        result = _run_equivalence_pair("S10_production_config", factory, dates, price_data)
    finally:
        _FakeStorage.shenwan_table = None
    result["seams"] = (
        "kelly 权重（历史方差真实分支 + 数据不足回退）/ 行业上限 / 单股限权 cap_and_normalize / "
        "min_buy_value_ratio 阈值 / K=2 分批 / holdings_snapshot"
    )
    result["production_config_mirror"] = dict(_PRODUCTION_CONFIG_OVERRIDES)
    return result


def scenario_s8_live_anchor(tmp_root: Path) -> Dict:
    """S8：live 持久化锚定四场景（旧链生产函数）+ 隔离根 containment（前后快照口径）。

    containment（R2-06/r1-P2-2/r3-P2-2 处置）：运行前对受保护根（生产 data/paper/）
    取文件集+哈希快照，场景全部跑在 tmp 隔离根后复验受保护根零变更；
    「零越界」措辞改为「受保护根零变更 + 隔离根内预期文件在册」。
    """
    trade_dates = [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-01-05", periods=120)]
    protected_root = _REPO_ROOT / "data" / "paper"
    before = snapshot_protected_root(protected_root)
    results = run_anchor_scenarios(tmp_root / "anchor", trade_dates, freq=20, k=3)
    containment_ok, containment_detail = verify_isolated_root_containment(
        tmp_root / "anchor", protected_root, before
    )
    return {
        "name": "S8_live_anchor_scenarios",
        "passed": all(r.passed for r in results) and containment_ok,
        "scenarios": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
        "isolated_root_containment": {"passed": containment_ok, "detail": containment_detail},
        "seams": "持久化锚定（tranche_anchor_date/last_rebalance_date/last_scheduled_rebalance_date）"
        "四场景 + build_tranche_schedule_from_anchor ≡ compute_tranche_schedule（anchor=窗口首日）"
        " + PaperStorage 隔离根（受保护根零变更实证，R14）",
    }


def _aggregate_verdict(step_results: List[Dict], anchor_result: Dict) -> bool:
    """总判定：场景顶层 + **嵌套变异自证**全量纳入（R2-04 处置）。"""
    ok = all(r["verdict_ok"] for r in step_results) and anchor_result["passed"]
    for r in step_results:
        mutant = r.get("mutant")
        if mutant is not None and not mutant["verdict_ok"]:
            ok = False
    return ok


def run_replay_calibration(repo_root: Path, out_dir: Path) -> Dict:
    """回放耗时标定（r1-P2-3 处置：落盘留证）。

    跑 replay_b0 单折（split0，6 个月测试窗）到证据目录内隔离 runs 根，
    登记命令/耗时/批次身份并保留批次产物（可复验）。
    """
    calib_dir = Path(out_dir) / "calibration"
    runs_root = calib_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/v2_p2a/replay_b0.py",
        "--splits",
        "0",
        "--runs-root",
        str(runs_root),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    elapsed = time.time() - t0
    batches = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
    meta_path = runs_root / batches[0] / "batch_meta.json" if batches else None
    result = {
        "command": " ".join(cmd),
        "cwd": str(repo_root),
        "elapsed_seconds": round(elapsed, 2),
        "exit_code": proc.returncode,
        "batch_id": batches[0] if batches else None,
        "runs_root": str(runs_root),
        "batch_meta_present": meta_path is not None and meta_path.exists(),
        "stdout_tail": (proc.stdout or "")[-500:],
    }
    (calib_dir / "calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_all_scenarios(
    out_root: Path,
    repo_root: Optional[Path] = None,
    include_probe: bool = True,
    include_calibration: bool = False,
    run_name: str = "t05b",
) -> Dict:
    """跑全部场景并落证据（单 run 目录自包含）。

    ``run_name`` = 证据目录前缀（T0.5b = ``t05b``；T1 生产实现重跑 = ``t1``）。
    """
    repo_root = Path(repo_root) if repo_root else _REPO_ROOT
    started = time.time()
    with synthetic_cost_settings():
        with tempfile.TemporaryDirectory(prefix="v2_p4_t05b_oracle_") as tmp:
            tmp_root = Path(tmp)
            step_results = [
                scenario_s1_multi_cycle_with_holiday(),
                scenario_s2_stagger_boundary(),
                scenario_s3_pending_order_postpone(),
                scenario_s4_stop_loss(),
                scenario_s4b_stop_loss_limit_down_counter(),
                scenario_s5_position_completion(),
                scenario_s6_early_rebalance_signal_dates_mutation(),
                scenario_s7_live_last_day_t0_plan(),
                scenario_s9a_live_resume_anchor_before_window(tmp_root),
                scenario_s9b_live_config_change_rebuild(tmp_root),
                scenario_s9c_live_missed_batch_catchup(tmp_root),
                scenario_s9d_k1_delayed_success(tmp_root),
                scenario_s9e_live_early_rebalance_k2(tmp_root),
                scenario_s9f_first_failure_then_success(tmp_root),
                scenario_s9g_calendar_extension(tmp_root),
                scenario_s10_production_config(),
            ]
            anchor_result = scenario_s8_live_anchor(tmp_root)

    all_passed = _aggregate_verdict(step_results, anchor_result)
    evidence = {
        "gate": (
            "v2_p4_t1_step_equivalence"
            if run_name != "t05b"
            else "v2_p4_t05b_step_equivalence_spike"
        ),
        "implementation": (
            "production" if run_name != "t05b" else "prototype（已被 T1 生产实现替换）"
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 3),
        "verdict": "PASS" if all_passed else "FAIL",
        "step_equivalence_scenarios": step_results,
        "live_anchor_scenarios": anchor_result,
        "snapshot_fields_frozen": list(RUNTIME_STATE_FIELDS),
        "uncovered_configs_for_gate2": list(_UNCOVERED_CONFIGS),
    }
    out_dir = out_root / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = out_dir / "spike_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    binding_path = write_binding_evidence(repo_root, out_dir)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("verdict") != "PASS":
        all_passed = False

    probe_verdict = None
    if include_probe:
        from scripts.v2_p4.t05b_idempotency_probe import run_idempotency_probe

        probe = run_idempotency_probe(repo_root / "data")
        (out_dir / "idempotency_probe.json").write_text(
            json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        probe_verdict = probe["verdict"]
        if probe_verdict != "PASS":
            all_passed = False

    calibration = None
    if include_calibration:
        calibration = run_replay_calibration(repo_root, out_dir)
        if calibration["exit_code"] != 0 or not calibration["batch_meta_present"]:
            all_passed = False

    # 回填总判定（含绑定/探针/标定）后重写证据文件
    evidence["verdict"] = "PASS" if all_passed else "FAIL"
    evidence["gate2b_binding_verdict"] = binding.get("verdict")
    evidence["idempotency_probe_verdict"] = probe_verdict
    evidence["replay_calibration"] = (
        {k: calibration[k] for k in ("command", "elapsed_seconds", "batch_id", "exit_code")}
        if calibration
        else None
    )
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {
        "verdict": evidence["verdict"],
        "evidence_path": str(evidence_path),
        "evidence_dir": str(out_dir),
        "binding_path": str(binding_path),
        "elapsed_seconds": evidence["elapsed_seconds"],
        "scenarios": evidence,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v2 P4 T0.5b 步进等价性 spike（合成 + 只读绑定采集）"
    )
    parser.add_argument(
        "--out-root",
        default=str(_REPO_ROOT / "data" / "frozen_reference" / "v2_p4"),
        help="证据输出根（默认 data/frozen_reference/v2_p4）",
    )
    parser.add_argument("--skip-probe", action="store_true", help="跳过禁下载幂等探针")
    parser.add_argument(
        "--with-calibration",
        action="store_true",
        help="跑回放标定（≈10s，写证据目录内隔离 runs 根）",
    )
    parser.add_argument(
        "--run-name",
        default="t05b",
        help="证据目录前缀（T0.5b=t05b；T1 生产实现重跑=t1）",
    )
    args = parser.parse_args(argv)
    summary = run_all_scenarios(
        Path(args.out_root),
        include_probe=not args.skip_probe,
        include_calibration=args.with_calibration,
        run_name=args.run_name,
    )
    print(f"步进等价性场景套件 verdict = {summary['verdict']}")
    print(f"证据目录: {summary['evidence_dir']}")
    print(f"耗时: {summary['elapsed_seconds']}s")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
