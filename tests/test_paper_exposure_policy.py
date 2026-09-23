# -*- coding: utf-8 -*-
"""纸面暴露政策接线（``paper/exposure_policy.py``）单元测试。

只使用合成对象（不依赖真实配置与生产数据），锁定：

1. **配置解析**：默认关零副作用、启用需模型源、容差校验、回补无政策源直接报错；
2. **减仓判定**：容差带、按比例整手取整、跳过卖出队列/保护票；
3. **对称回补**：上界三选一、按市值比例、保留买入日、新计划日清零释放额；
4. **结算**：释放额累加、未成交/未花完退回、过日退单、同日幂等；
5. **状态**：round-trip 恢复、指纹不一致清空；
6. **底层改造**：指令 keep_buy_date 持久化（含旧文件向后兼容）、加仓保留原买入日。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lazybull.paper.account import PaperAccount  # noqa: E402
from src.lazybull.paper.exposure_policy import (  # noqa: E402
    DEFAULT_TRIM_TOLERANCE,
    REPLENISH_REASON_PREFIX,
    TRIM_REASON_PREFIX,
    PaperExposurePolicy,
    PaperExposureSettings,
    ensure_paper_exposure_policy,
)
from src.lazybull.paper.models import TradeInstruction  # noqa: E402
from src.lazybull.paper.storage import PaperStorage  # noqa: E402

POLICY_SPEC = "arm=combined,mode=rolling,window=2,min_window=2,regime_q=0.5,score_q=0.5,lambda=0.5"

CAL_DAYS = ["20240102", "20240103", "20240104", "20240105"]


# ------------------------------------------------------------------ 合成对象
class _StubProvider:
    """最小 provider 桩：固定 λ、固定指纹，记录恢复调用。"""

    def __init__(self, multiplier: float = 1.0, fingerprint: str = "fp-test") -> None:
        self.multiplier = float(multiplier)
        self._fp = fingerprint
        self.restored = None
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return self._fp

    def multiplier_for(self, date, holdings) -> float:
        self.calls += 1
        return self.multiplier

    def export_state(self) -> dict:
        return {"version": 1, "fingerprint": self._fp, "frames": {}, "pending": {}}

    def restore_state(self, payload: dict) -> None:
        self.restored = payload


class _RaisingProvider(_StubProvider):
    def multiplier_for(self, date, holdings) -> float:
        raise RuntimeError("打分输入缺失")


class _StubBroker:
    def __init__(self) -> None:
        self.pending_sells: list = []


class _StubLoader:
    def __init__(self, prices) -> None:
        self._prices = dict(prices)

    def load_clean_daily_by_date(self, date):
        return pd.DataFrame(
            {
                "ts_code": list(self._prices),
                "close": list(self._prices.values()),
                "pre_close": list(self._prices.values()),
            }
        )

    def load_clean_trade_cal(self):
        return pd.DataFrame({"cal_date": CAL_DAYS, "is_open": [1] * len(CAL_DAYS)})


class _StubAccount:
    def __init__(self, cash: float, shares: dict) -> None:
        self._cash = float(cash)
        self._positions = {
            code: SimpleNamespace(shares=int(count), buy_price=5.0, buy_date="20240102")
            for code, count in shares.items()
        }

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> dict:
        return self._positions

    def get_total_value(self, prices) -> float:
        total = self._cash
        for code, pos in self._positions.items():
            price = float(prices.get(code, pos.buy_price) or 0.0)
            total += pos.shares * (price if price > 0 else pos.buy_price)
        return total


class _StubRunner:
    def __init__(self, tmp_path, prices, account, t1_instructions=None) -> None:
        self.paper_storage = PaperStorage(root_path=str(tmp_path))
        if t1_instructions:
            self.paper_storage.save_instructions("20240103", t1_instructions)
        self.broker = _StubBroker()
        self.account = account
        self.loader = _StubLoader(prices)
        self.quiet = True


DEFAULT_PRICES = {"600000.SH": 5.0, "000001.SZ": 5.0}


def _make_policy(
    tmp_path,
    *,
    cash: float = 1000.0,
    shares=None,
    prices=None,
    multiplier: float = 1.0,
    replenish: bool = False,
    tolerance: float = DEFAULT_TRIM_TOLERANCE,
    fingerprint: str = "fp-test",
    t1_instructions=None,
    provider=None,
):
    shares = shares if shares is not None else {"600000.SH": 1000, "000001.SZ": 1000}
    prices = prices if prices is not None else dict(DEFAULT_PRICES)
    _touch_feature(Path(tmp_path), "20240102")  # 预置判定日推理特征（跳过 ensure 构建路径）
    account = _StubAccount(cash, shares)
    runner = _StubRunner(tmp_path, prices, account, t1_instructions=t1_instructions)
    settings = PaperExposureSettings(
        policy=POLICY_SPEC,
        model_root="unused",
        arm_suffix="_stub",
        replenish=replenish,
        trim_tolerance=tolerance,
    )
    policy = PaperExposurePolicy(runner, settings, data_root=str(tmp_path))
    policy._provider = provider or _StubProvider(multiplier, fingerprint)  # noqa: SLF001
    return policy, runner


def _touch_feature(tmp_path, day: str) -> None:
    path = Path(tmp_path) / "features" / "cs_infer" / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# ------------------------------------------------------------------ 配置解析
def test_settings_disabled_returns_none():
    assert PaperExposureSettings.from_config({}) is None
    assert PaperExposureSettings.from_config({"exposure_policy": None}) is None
    assert PaperExposureSettings.from_config({"exposure_policy": "   "}) is None


def test_settings_rejects_replenish_without_policy():
    with pytest.raises(ValueError, match="无政策源"):
        PaperExposureSettings.from_config({"exposure_replenish": True})


def test_settings_requires_model_source():
    with pytest.raises(ValueError, match="policy_model_root"):
        PaperExposureSettings.from_config({"exposure_policy": POLICY_SPEC})
    with pytest.raises(ValueError, match="policy_model_root"):
        PaperExposureSettings.from_config(
            {"exposure_policy": POLICY_SPEC, "policy_model_root": "data/x"}
        )


def test_settings_rejects_bad_tolerance():
    with pytest.raises(ValueError, match="trim_tolerance"):
        PaperExposureSettings.from_config(
            {
                "exposure_policy": POLICY_SPEC,
                "policy_model_root": "data/x",
                "policy_arm_suffix": "_v6m",
                "exposure_trim_tolerance": 1.5,
            }
        )


def test_settings_parses_valid_config():
    settings = PaperExposureSettings.from_config(
        {
            "exposure_policy": POLICY_SPEC,
            "policy_model_root": "data/x",
            "policy_arm_suffix": "_v6m",
            "policy_coverage_start": "20190527",
            "exposure_replenish": True,
            "exposure_trim_tolerance": 0.06,
        }
    )
    assert settings is not None
    assert settings.policy == POLICY_SPEC
    assert settings.coverage_start == "20190527"
    assert settings.replenish is True
    assert settings.trim_tolerance == 0.06
    default = PaperExposureSettings.from_config(
        {
            "exposure_policy": POLICY_SPEC,
            "policy_model_root": "data/x",
            "policy_arm_suffix": "_v6m",
        }
    )
    assert default.trim_tolerance == DEFAULT_TRIM_TOLERANCE
    assert default.coverage_start is None
    assert default.replenish is False


# ------------------------------------------------------------------ 减仓判定
def test_plan_trim_scales_by_fraction_with_whole_lots(tmp_path):
    """λ=0.5 且明显超配 => 两只持仓各按比例整手减仓，指令落 T+1 文件。"""
    policy, runner = _make_policy(tmp_path, cash=1000.0, multiplier=0.5)
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=False)

    # 总值 11000、持仓 10000、λ=0.5 => 超配 4500、fraction=0.45 => 每只 400 股
    assert result == {"trim_orders": 2, "replenish_orders": 0}
    instructions = runner.paper_storage.load_instructions("20240103")
    assert instructions is not None and len(instructions) == 2
    for inst in instructions:
        assert inst.action == "sell"
        assert inst.shares == 400
        assert inst.reason.startswith(TRIM_REASON_PREFIX)
        assert inst.source_date == "20240102"
    assert policy.stats["trim_trigger_days"] == 1
    assert policy.stats["trim_orders"] == 2


def test_plan_trim_skips_within_tolerance(tmp_path):
    """超配额落在容差带内 => 不动作（与引擎 EXPOSURE_TRIM_TOLERANCE 同语义）。"""
    # 总值 9800、目标 4900、实际 5000 => 超配 100 ≤ 0.03*9800=294
    policy, runner = _make_policy(
        tmp_path, cash=4800.0, shares={"600000.SH": 1000}, prices={"600000.SH": 5.0}, multiplier=0.5
    )
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=False)
    assert result["trim_orders"] == 0
    assert runner.paper_storage.load_instructions("20240103") is None
    assert policy.stats["trim_skipped_below_tolerance"] == 1


def test_plan_trim_skips_sell_queue_and_protected(tmp_path):
    """已在整仓卖出队列或受保护的持仓不参与政策减仓（整仓卖出优先）。"""
    existing = [
        TradeInstruction(
            ts_code="600000.SH",
            action="sell",
            shares=1000,
            price_type="open",
            reason="持有期到期",
            source_date="20240102",
        )
    ]
    policy, runner = _make_policy(
        tmp_path,
        cash=2000.0,
        shares={"600000.SH": 1000, "000001.SZ": 1000, "000002.SZ": 1000},
        prices={"600000.SH": 5.0, "000001.SZ": 5.0, "000002.SZ": 5.0},
        multiplier=0.5,
        t1_instructions=existing,
    )
    result = policy.plan(
        "20240102", t1_date="20240103", is_plan_day=False, protected_stocks=["000001.SZ"]
    )
    assert result["trim_orders"] == 1
    instructions = runner.paper_storage.load_instructions("20240103")
    merged = {inst.ts_code: inst for inst in instructions}
    assert set(merged) == {"600000.SH", "000002.SZ"}  # 原卖出单 + 仅 C 被减仓
    assert merged["000002.SZ"].action == "sell"
    assert merged["000002.SZ"].shares == 400


def test_plan_trim_no_targets_warns(tmp_path):
    """全部持仓都在卖出队列/保护名单 => 无可减仓标的（计数并告警，不生成指令）。"""
    policy, runner = _make_policy(tmp_path, cash=1000.0, multiplier=0.5)
    result = policy.plan(
        "20240102",
        t1_date="20240103",
        is_plan_day=False,
        protected_stocks=["600000.SH", "000001.SZ"],
    )
    assert result["trim_orders"] == 0
    assert policy.stats["trim_skipped_no_targets"] == 1


# ------------------------------------------------------------------ 对称回补
def test_plan_replenish_scales_by_weight_and_keeps_buy_date(tmp_path):
    """上限放开且存在释放额 => 按持仓市值比例回补、保留买入日、记录待结算计划。"""
    policy, runner = _make_policy(
        tmp_path,
        cash=2000.0,
        shares={"600000.SH": 100, "000001.SZ": 100},
        multiplier=1.0,
        replenish=True,
    )
    policy._release_budget = 1000.0  # noqa: SLF001
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=False)

    # 总值 3000、持仓 1000、λ=1.0 => 上界 2000；min(cash 2000, budget 1000, room 2000)=1000
    # 每只按 50% 市值比例分 500 => 100 股（整手）
    assert result == {"trim_orders": 0, "replenish_orders": 2}
    instructions = runner.paper_storage.load_instructions("20240103")
    assert instructions is not None and len(instructions) == 2
    for inst in instructions:
        assert inst.action == "buy"
        assert inst.shares == 100
        assert inst.keep_buy_date is True
        assert inst.reason.startswith(REPLENISH_REASON_PREFIX)
    assert policy._release_budget == 0.0  # noqa: SLF001 - 额度全部排队
    assert policy._pending_settle["20240103"] == {  # noqa: SLF001
        "600000.SH": 500.0,
        "000001.SZ": 500.0,
    }
    assert policy.stats["replenish_trigger_days"] == 1


def test_plan_replenish_resets_budget_on_plan_day(tmp_path):
    """新调仓计划生成日：不干预并清零释放额（计划会用同一笔现金重新分配）。"""
    policy, runner = _make_policy(
        tmp_path,
        cash=2000.0,
        shares={"600000.SH": 100, "000001.SZ": 100},
        multiplier=1.0,
        replenish=True,
    )
    policy._release_budget = 1000.0  # noqa: SLF001
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=True)
    assert result["replenish_orders"] == 0
    assert policy._release_budget == 0.0  # noqa: SLF001
    assert policy.stats["replenish_budget_reset_days"] == 1
    assert runner.paper_storage.load_instructions("20240103") is None


def test_plan_replenish_skips_below_one_lot(tmp_path):
    """分配额不足一手 => 跳过（比例语义优先），释放额保留待次日重判。"""
    policy, runner = _make_policy(
        tmp_path,
        cash=100.0,
        shares={"600000.SH": 100, "000001.SZ": 100},
        multiplier=1.0,
        replenish=True,
    )
    policy._release_budget = 1000.0  # noqa: SLF001
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=False)
    assert result["replenish_orders"] == 0
    assert policy._release_budget == 1000.0  # noqa: SLF001
    assert runner.paper_storage.load_instructions("20240103") is None


def test_plan_replenish_skips_without_budget(tmp_path):
    policy, runner = _make_policy(tmp_path, replenish=True)
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=False)
    assert result["replenish_orders"] == 0
    assert policy.stats["replenish_skipped_no_budget"] == 1


def test_plan_duplicate_day_is_skipped(tmp_path):
    policy, _ = _make_policy(tmp_path, multiplier=0.5)
    policy.plan("20240102", t1_date="20240103", is_plan_day=False)
    first_orders = policy.stats["trim_orders"]
    result = policy.plan("20240102", t1_date="20240103", is_plan_day=False)
    assert result == {"skipped": "duplicate_day", "trim_orders": 0, "replenish_orders": 0}
    assert policy.stats["trim_orders"] == first_orders
    assert policy.stats["skipped_duplicate_plan_days"] == 1


# ------------------------------------------------------------------ 评估
def test_evaluate_returns_multiplier_and_counts(tmp_path):
    _touch_feature(tmp_path, "20240102")
    policy, _ = _make_policy(tmp_path, multiplier=0.7)
    assert policy.evaluate("20240102") == 0.7
    assert policy.evaluate("20240102") == 0.7  # 同日缓存
    assert policy.stats["evaluated_days"] == 1
    assert policy.stats["trigger_days"] == 1


def test_evaluate_degrades_on_data_error(tmp_path):
    """数据类异常 => 告警 + 顺延上次 λ（非静默、非崩溃）。"""
    _touch_feature(tmp_path, "20240102")
    policy, _ = _make_policy(tmp_path, provider=_RaisingProvider())
    policy._last_multiplier = 0.5  # noqa: SLF001 - 模拟上次判定
    assert policy.evaluate("20240102") == 0.5
    assert policy.stats["degraded_days"] == 1


# ------------------------------------------------------------------ 结算
def _fill(action: str, reason: str, amount: float, ts_code: str = "600000.SH", cost: float = 0.0):
    return SimpleNamespace(
        action=action, reason=reason, amount=amount, ts_code=ts_code, total_cost=cost
    )


def test_settle_accumulates_and_refunds(tmp_path):
    policy, _ = _make_policy(tmp_path)
    policy._pending_settle = {"20240103": {"600000.SH": 500.0, "000001.SZ": 300.0}}  # noqa: SLF001
    fills = [
        _fill("sell", f"{TRIM_REASON_PREFIX}（20240102 判定）", 2000.0),
        _fill("buy", f"{REPLENISH_REASON_PREFIX}（20240102 判定）", 480.0, cost=10.0),
    ]
    policy.settle("20240103", fills)

    # 释放额 = 减仓 2000 + 回补未花完（500-490=10） + 未成交计划 300
    assert policy._release_budget == pytest.approx(2310.0)  # noqa: SLF001
    assert policy.stats["trim_fills"] == 1
    assert policy.stats["replenish_filled"] == 1
    assert policy._pending_settle == {}  # noqa: SLF001
    policy.settle("20240103", fills)  # 同日重复结算无效果
    assert policy._release_budget == pytest.approx(2310.0)  # noqa: SLF001


def test_settle_ignores_foreign_reasons(tmp_path):
    policy, _ = _make_policy(tmp_path)
    policy.settle("20240103", [_fill("sell", "持有期到期", 999.0)])
    assert policy._release_budget == 0.0  # noqa: SLF001


def test_expire_unsettled_refunds_past_days(tmp_path):
    policy, _ = _make_policy(tmp_path)
    policy._pending_settle = {"20240102": {"600000.SH": 400.0}}  # noqa: SLF001
    policy._expire_unsettled("20240103")
    assert policy._release_budget == 400.0  # noqa: SLF001
    assert policy._pending_settle == {}  # noqa: SLF001
    assert policy.stats["replenish_expired_refunds"] == 1


# ------------------------------------------------------------------ 状态
def test_state_round_trip(tmp_path):
    policy, _ = _make_policy(tmp_path)
    policy._release_budget = 123.45  # noqa: SLF001
    policy._pending_settle = {"20240104": {"600000.SH": 66.0}}  # noqa: SLF001
    policy._last_multiplier = 0.5  # noqa: SLF001
    policy._last_judged_date = "20240102"  # noqa: SLF001
    policy.stats["trim_orders"] = 3
    policy._save_state()  # noqa: SLF001

    restored, _ = _make_policy(tmp_path)
    assert restored._state_loaded is False  # noqa: SLF001 - 首次加载
    restored._load_state()  # noqa: SLF001
    assert restored._release_budget == pytest.approx(123.45)  # noqa: SLF001
    assert restored._pending_settle == {"20240104": {"600000.SH": 66.0}}  # noqa: SLF001
    assert restored._last_multiplier == 0.5  # noqa: SLF001
    assert restored._last_judged_date == "20240102"  # noqa: SLF001
    assert restored.stats["trim_orders"] == 3
    assert restored._provider.restored is not None  # noqa: SLF001 - provider 面板快照已回放


def test_state_discarded_on_fingerprint_mismatch(tmp_path):
    policy, _ = _make_policy(tmp_path)
    policy._release_budget = 123.45  # noqa: SLF001
    policy._save_state()  # noqa: SLF001

    other, _ = _make_policy(tmp_path, fingerprint="fp-other")
    other._load_state()  # noqa: SLF001
    assert other._release_budget == 0.0  # noqa: SLF001 - 指纹变化必须清空


def test_stats_summary_contains_policy_identity(tmp_path):
    policy, _ = _make_policy(tmp_path)
    summary = policy.stats_summary()
    assert summary["enabled"] is True
    assert summary["policy"] == POLICY_SPEC
    assert summary["fingerprint"] == "fp-test"
    assert summary["trim_tolerance"] == DEFAULT_TRIM_TOLERANCE


# ------------------------------------------------------------------ 装配
def test_ensure_policy_returns_none_when_disabled(tmp_path, monkeypatch):
    runner = SimpleNamespace()
    runner.paper_storage = PaperStorage(root_path=str(tmp_path))
    assert ensure_paper_exposure_policy(runner) is None
    assert runner.exposure_policy is None
    assert runner._exposure_policy_resolved is True  # noqa: SLF001


# ------------------------------------------------------------------ 底层改造
def test_instructions_keep_buy_date_round_trip(tmp_path):
    storage = PaperStorage(root_path=str(tmp_path))
    storage.save_instructions(
        "20240103",
        [
            TradeInstruction(
                ts_code="600000.SH",
                action="buy",
                shares=100,
                price_type="close",
                reason="暴露门控回补",
                source_date="20240102",
                keep_buy_date=True,
            ),
            TradeInstruction(
                ts_code="000001.SZ",
                action="sell",
                shares=200,
                price_type="open",
                reason="到期",
                source_date="20240102",
            ),
        ],
    )
    loaded = storage.load_instructions("20240103")
    assert loaded[0].keep_buy_date is True
    assert loaded[1].keep_buy_date is False


def test_instructions_legacy_file_without_flag(tmp_path):
    """旧格式指令文件（无 keep_buy_date 列）=> 默认 False（向后兼容）。"""
    storage = PaperStorage(root_path=str(tmp_path))
    legacy = pd.DataFrame(
        {
            "ts_code": ["600000.SH"],
            "action": ["buy"],
            "shares": [100],
            "price_type": ["close"],
            "reason": ["计划买入"],
            "source_date": ["20240102"],
            "target_weight": [0.1],
            "original_signal_date": ["20240102"],
            "desired_position_count": [20],
            "retry_attempt": [0],
            "replacement_slot_code": [""],
        }
    )
    legacy.to_parquet(storage.instructions_path / "20240103.parquet", index=False)
    loaded = storage.load_instructions("20240103")
    assert loaded[0].keep_buy_date is False


def test_account_add_position_keeps_buy_date(tmp_path):
    """暴露门控回补加仓：keep_buy_date=True 保留原买入日/绩效价/ATR 记录。"""
    account = PaperAccount(initial_capital=1000.0, storage=PaperStorage(root_path=str(tmp_path)))
    account.add_position(
        "600000.SH",
        100,
        buy_price=5.0,
        buy_cost=500.0,
        buy_date="20240102",
        buy_pnl_price=5.1,
        buy_atr_pct=0.02,
    )
    account.add_position(
        "600000.SH",
        100,
        buy_price=6.0,
        buy_cost=600.0,
        buy_date="20240105",
        keep_buy_date=True,
    )
    pos = account.get_position("600000.SH")
    assert pos.shares == 200
    assert pos.buy_date == "20240102"
    assert pos.buy_pnl_price == pytest.approx(5.1)
    assert pos.buy_atr_pct == pytest.approx(0.02)

    account.add_position(
        "600000.SH",
        100,
        buy_price=7.0,
        buy_cost=700.0,
        buy_date="20240106",
        keep_buy_date=False,
    )
    assert account.get_position("600000.SH").buy_date == "20240106"
