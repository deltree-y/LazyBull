# -*- coding: utf-8 -*-
"""纸面暴露政策接线（terminal_loss P2-5 / P2-3 / P2-4 的纸面实现）。

定位：把回测引擎已验证的"暴露政策三件套"接到纸面交易链路——

1. **λ_t 在线判定**：复用 ``risk.terminal_loss.exposure_online.DailyExposureProvider``
   （折模型映射 / 打分装配 / 日级聚合 / 滚动阈值**全部复用离线实现，禁止复制公式**）；
2. **按比例减仓**：T0 收盘后判定 → 生成 T+1 部分卖出指令，语义对齐
   ``backtest/exposure_trim.py``（触发带 ``exposure_trim_tolerance`` 容差、整手取整、
   剩余不足一手整仓卖出、跳过已在整仓卖出队列的股票、不可交易不进延迟队列）；
3. **对称回补**：语义对齐 ``backtest/exposure_replenish.py``
   （上界三选一、按市值比例、**不重置买入日**、未花完额度退回、**新调仓计划生成日不干预并清零**）。

硬口径（与引擎一致，禁止偏离）：

- **评估时点**：判定日 T 的"执行前持仓"（纸面即 T1 执行前），对应引擎语义 S1；
- **判定每日进行**（T0 收盘后）；指令写入 T+1 指令文件，复用既有 T0/T1 链路，
  **不新增第二条执行路径**；
- 同日单次评估：本模块对同一日期缓存 λ（provider 自身亦缓存），
  run_t0 买入缩放与减仓/回补判定共用同一取值；
- **默认关闭**：未配置 ``exposure_policy`` 时零副作用（不构造 provider、不读写状态、不触碰指令）。

已知差异（登记，随结论一起报告）：

- provider 特征源 = ``features/cs_infer``（纸面当日推理特征，无标签）；
  与 OOS 回测读 ``cs_train`` 的"数据修订 / 窗口口径"差异已单独登记；
- 补齐（pending_buys）路径的买入缩放取**执行日** λ（回测取信号日），差异 ≤1 个交易日；
- 折集覆盖区间之外（例如 2026 年超出终损折集）λ 顺延为 1.0（不动作），与引擎一致；
- 数据类异常（当日特征构建失败 / 打分输入缺失）按"告警 + 降级顺延"处理（非静默），
  配置/模型源错误（折目录缺失、arm_suffix 不匹配、指纹不一致）在启用时直接报错。

状态持久化：``<paper_root>/state/exposure_policy.json``
（释放额 / 待结算计划 / λ 顺延值 / 统计 / provider 面板快照）。
换策略或换模型源（指纹变化）时旧状态**直接清空并告警**（禁止静默续用）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

from ..features import ensure_features_for_date
from ..risk.terminal_loss.exposure_online import (
    DailyExposureProvider,
    OnlinePolicyConfig,
    folds_model_digest,
    to_date_str,
)
from ..trading.sizing import compute_lot_shares, resolve_trim_shares
from .models import TradeInstruction

#: 触发容差默认值（与引擎 ``backtest/exposure_trim.py::EXPOSURE_TRIM_TOLERANCE`` 同值）
DEFAULT_TRIM_TOLERANCE = 0.03

#: 指令 reason 前缀（结算识别单处定义）
TRIM_REASON_PREFIX = "暴露门控减仓"
REPLENISH_REASON_PREFIX = "暴露门控回补"

_STATE_VERSION = 1

#: 预热面板文件标识（P2a：由 scripts/prepare_paper_exposure_warmup.py 生成）
_WARMUP_KIND = "paper_exposure_warmup"
_WARMUP_VERSION = 1


def _default_state() -> Dict[str, Any]:
    return {
        "evaluated_days": 0,
        "trigger_days": 0,
        "degraded_days": 0,
        "trim_trigger_days": 0,
        "trim_orders": 0,
        "trim_fills": 0,
        "trim_sold_amount": 0.0,
        "trim_skipped_below_tolerance": 0,
        "trim_skipped_no_targets": 0,
        "replenish_trigger_days": 0,
        "replenish_orders": 0,
        "replenish_filled": 0,
        "replenish_bought_amount": 0.0,
        "replenish_skipped_no_budget": 0,
        "replenish_skipped_below_tolerance": 0,
        "replenish_budget_reset_days": 0,
        "replenish_expired_refunds": 0,
        "skipped_duplicate_plan_days": 0,
        "warmup_days": 0,
    }


@dataclass
class PaperExposureSettings:
    """纸面暴露政策配置（单变量直读：全部来自 config.yaml ``exposure_policy`` 区块）。"""

    policy: str
    model_root: str
    arm_suffix: str
    coverage_start: Optional[str] = None
    replenish: bool = False
    trim_tolerance: float = DEFAULT_TRIM_TOLERANCE
    pinned_fold: Optional[str] = None  # 指定折名则固定使用该折；None=自动取最新可用折
    warmup_file: Optional[str] = None  # 预热面板文件；None=默认 <model_root>/paper_warmup/state.json

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["PaperExposureSettings"]:
        """从纸面配置字典解析；未启用返回 None（零副作用）。"""
        raw = config.get("exposure_policy")
        enabled = raw is not None and str(raw).strip() != ""
        if not enabled:
            if config.get("exposure_replenish"):
                raise ValueError(
                    "exposure_replenish=True 但未启用 exposure_policy：回补无政策源，禁止静默忽略"
                )
            return None
        model_root = str(config.get("policy_model_root") or "").strip()
        arm_suffix = str(config.get("policy_arm_suffix") or "").strip()
        if not model_root or not arm_suffix:
            raise ValueError(
                "启用 exposure_policy 时必须同时给出 policy_model_root 与 policy_arm_suffix"
            )
        tolerance_raw = config.get("exposure_trim_tolerance")
        tolerance = DEFAULT_TRIM_TOLERANCE if tolerance_raw is None else float(tolerance_raw)
        if not 0.0 < tolerance < 1.0:
            raise ValueError(f"exposure_trim_tolerance 必须落于 (0, 1)，当前 {tolerance}")
        coverage = str(config.get("policy_coverage_start") or "").strip() or None
        pinned = str(config.get("policy_fold") or "").strip() or None
        warmup = str(config.get("policy_warmup_file") or "").strip() or None
        return cls(
            policy=str(raw).strip(),
            model_root=model_root,
            arm_suffix=arm_suffix,
            coverage_start=coverage,
            replenish=bool(config.get("exposure_replenish", False)),
            trim_tolerance=tolerance,
            pinned_fold=pinned,
            warmup_file=warmup,
        )


class PaperExposurePolicy:
    """纸面暴露政策编排器（runner 级；默认关闭时所有方法零副作用）。"""

    def __init__(self, runner, settings: PaperExposureSettings, *, data_root: str) -> None:
        self.runner = runner
        self.settings = settings
        self.data_root = str(data_root)
        self._feature_root = Path(data_root) / "features" / "cs_infer"
        self._state_path = Path(runner.paper_storage.state_path) / "exposure_policy.json"
        self._holding_period = int(
            (runner.paper_storage.load_config() or {}).get("rebalance_freq", 1) or 1
        )
        config = runner.paper_storage.load_config() or {}
        self._buy_price_type = str(config.get("buy_price", "close"))
        self._sell_price_type = str(config.get("sell_price", "open"))

        self._provider: Optional[DailyExposureProvider] = None
        self._lambda: Dict[str, float] = {}
        self._last_multiplier: float = 1.0
        self._release_budget: float = 0.0
        self._pending_settle: Dict[str, Dict[str, float]] = {}
        self._settled_dates: List[str] = []
        self._last_judged_date: str = ""
        self.stats: Dict[str, Any] = _default_state()
        self._state_loaded = False
        self._warmup_days: int = 0

    # ------------------------------------------------------------------ 生命周期
    def bind(self) -> None:
        """构造 provider 并恢复状态（配置/模型源错误在此直接报错）。"""
        if self._provider is not None:
            return
        model_root = Path(self.settings.model_root)
        if not model_root.exists():
            raise ValueError(f"暴露政策模型根不存在: {model_root}（policy_model_root 配置错误）")
        policy_config = OnlinePolicyConfig.parse(self.settings.policy)
        provider = DailyExposureProvider(
            policy_config,
            risk_root=str(model_root),
            arm_suffix=self.settings.arm_suffix,
            data_root=self.data_root,
            feature_root=str(self._feature_root),
            coverage_start=self.settings.coverage_start,
            book="pre_exec",
            verbose=False,
            coverage_mode="serving",
            pinned_fold=self.settings.pinned_fold,
        )
        self._provider = provider
        self._load_state()
        self._apply_warmup_if_needed()
        logger.info(
            "纸面暴露政策已启用（在线现算/实盘模式）: {}（指纹 {}，模型根 {}，折 {}，覆盖起点 {}，回补 {}，容差 {:.1%}）".format(
                policy_config.describe(),
                provider.fingerprint,
                self.settings.model_root,
                self.settings.pinned_fold or "自动（最新可用）",
                self.settings.coverage_start or "全区间",
                "开" if self.settings.replenish else "关",
                self.settings.trim_tolerance,
            )
        )

    # ------------------------------------------------------------------ 预热面板（P2a）
    def _warmup_path(self) -> Path:
        """预热面板文件路径：默认 ``<policy_model_root>/paper_warmup/state.json``。"""
        if self.settings.warmup_file:
            return Path(self.settings.warmup_file)
        return Path(self.settings.model_root) / "paper_warmup" / "state.json"

    def _apply_warmup_if_needed(self) -> None:
        """首次绑定且面板为空时，用预热文件填充面板历史（阈值窗口预热）。

        - 仅当 provider 面板为空（无纸面状态或已被指纹失效清空）时才应用；
        - 校验：文件标识/版本、折模型内容摘要（重训后失配 → 告警忽略）、
          provider 自带指纹校验（``restore_state`` 内部）；
        - 预热是增强项：任何不满足都只告警，不阻断主流程。
        """
        provider = self._provider
        if provider is None:
            return
        if provider.evaluated_day_count() > 0:
            return
        path = self._warmup_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"暴露政策预热文件读取失败（忽略）: {path}（{exc}）")
            return
        if payload.get("kind") != _WARMUP_KIND or int(payload.get("version", 0)) != _WARMUP_VERSION:
            logger.warning(f"暴露政策预热文件格式不符（忽略）: {path}")
            return
        digest = str(payload.get("folds_digest", ""))
        if digest:
            try:
                current = folds_model_digest(self.settings.model_root, self.settings.arm_suffix)
            except Exception as exc:  # noqa: BLE001 - 摘要失败不阻断（restore 指纹仍兜底）
                logger.warning(f"暴露政策预热折模型摘要计算失败（跳过校验）: {exc}")
                current = ""
            if current and current != digest:
                logger.warning(
                    f"暴露政策预热与当前折模型不一致（digest {digest} != {current}）："
                    "折模型已重训，请重新生成预热文件；本次忽略预热"
                )
                return
        provider_state = payload.get("provider") or {}
        try:
            provider.restore_state(provider_state)
        except ValueError as exc:
            logger.warning(f"暴露政策预热恢复失败（忽略）: {exc}")
            return
        coverage = payload.get("coverage") or {}
        self._warmup_days = int(provider.evaluated_day_count())
        self.stats["warmup_days"] = self._warmup_days
        logger.info(
            "暴露政策预热面板已恢复：{} 个交易日（{}~{}，{} 行）← {}".format(
                self._warmup_days,
                coverage.get("first_day", "-"),
                coverage.get("last_day", "-"),
                coverage.get("rows", "-"),
                path,
            )
        )

    # ------------------------------------------------------------------ 评估（T1 执行前）
    def evaluate(self, date: Any) -> float:
        """评估判定日 λ（fail-soft：数据类异常降级顺延，与引擎缺口日语义一致）。"""
        date_str = to_date_str(date)
        if date_str in self._lambda:
            return self._lambda[date_str]
        self.bind()
        self._expire_unsettled(date_str)
        multiplier: Optional[float] = None
        try:
            self._ensure_feature(date_str)
            rows = self._holdings_rows(date_str)
            multiplier = float(self._provider.multiplier_for(date_str, rows))
            if not 0.0 < multiplier <= 1.0:
                raise ValueError(f"provider 返回非法系数 {multiplier}（政策层只允许降暴露）")
            self.stats["evaluated_days"] += 1
            if multiplier < 1.0:
                self.stats["trigger_days"] += 1
                logger.info(f"暴露政策：{date_str} 触发降暴露 λ={multiplier:.2f}")
        except Exception as exc:  # noqa: BLE001 - 数据类异常降级告警（非静默）
            self.stats["degraded_days"] += 1
            multiplier = self._last_multiplier
            logger.warning(
                f"暴露政策评估降级（{date_str}）: {exc}；本日 λ 顺延为 {multiplier:.2f}"
                "（下次运行重试）"
            )
        self._lambda[date_str] = multiplier
        self._last_multiplier = multiplier
        self._save_state()
        return multiplier

    def multiplier(self, date: Any) -> float:
        """读取当日 λ（未评估则即时评估；供 run_t0 买入预算缩放复用）。"""
        return self.evaluate(date)

    # ------------------------------------------------------------------ 判定（T0 收盘后）
    def plan(
        self,
        date: Any,
        *,
        t1_date: str,
        is_plan_day: bool,
        protected_stocks: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """每日判定：减仓 / 对称回补 → 生成 T+1 指令（追加到既有指令文件）。"""
        date_str = to_date_str(date)
        self.bind()
        self._expire_unsettled(date_str)
        if self._last_judged_date == date_str:
            self.stats["skipped_duplicate_plan_days"] += 1
            return {"skipped": "duplicate_day", "trim_orders": 0, "replenish_orders": 0}

        multiplier = self.evaluate(date_str)
        prices = self._close_prices(date_str)
        instructions = list(self.runner.paper_storage.load_instructions(t1_date) or [])
        existing_keys = {(inst.action, inst.ts_code) for inst in instructions}
        sell_queue = {code for action, code in existing_keys if action == "sell"}
        sell_queue |= {str(item.ts_code) for item in (self.runner.broker.pending_sells or [])}
        protected = set(protected_stocks or ())

        new_instructions: List[TradeInstruction] = []
        trim_stats = self._plan_trim(
            date_str, multiplier, prices, sell_queue, protected, new_instructions
        )
        replenish_stats = self._plan_replenish(
            date_str,
            multiplier,
            prices,
            sell_queue,
            protected,
            is_plan_day,
            t1_date,
            new_instructions,
        )
        appended = 0
        if new_instructions:
            merged = instructions + new_instructions
            self.runner.paper_storage.save_instructions(t1_date, merged)
            appended = len(new_instructions)
            logger.info(
                f"暴露政策：{date_str} 追加 {appended} 条指令到 T+1（{t1_date}）计划"
                f"（减仓 {trim_stats['orders']}，回补 {replenish_stats['orders']}）"
            )
        self._last_judged_date = date_str
        self._save_state()
        return {"trim_orders": trim_stats["orders"], "replenish_orders": replenish_stats["orders"]}

    # ------------------------------------------------------------------ 结算（T1 执行后）
    def settle(self, exec_date: Any, fills: Optional[Sequence[Any]]) -> None:
        """T1 执行后结算：减仓成交额累加释放额；回补计划按实际成交退回差额。"""
        date_str = to_date_str(exec_date)
        self.bind()
        if date_str in self._settled_dates:
            return
        added = 0.0
        refund = 0.0
        filled_codes = set()
        for fill in fills or []:
            action = str(getattr(fill, "action", "") or "")
            reason = str(getattr(fill, "reason", "") or "")
            if action == "sell" and reason.startswith(TRIM_REASON_PREFIX):
                added += float(getattr(fill, "amount", 0.0) or 0.0)
                self.stats["trim_fills"] += 1
                self.stats["trim_sold_amount"] += float(getattr(fill, "amount", 0.0) or 0.0)
            elif action == "buy" and reason.startswith(REPLENISH_REASON_PREFIX):
                code = str(getattr(fill, "ts_code", ""))
                filled_codes.add(code)
                planned = (self._pending_settle.get(date_str) or {}).get(code)
                invested = float(getattr(fill, "amount", 0.0) or 0.0) + float(
                    getattr(fill, "total_cost", 0.0) or 0.0
                )
                self.stats["replenish_filled"] += 1
                self.stats["replenish_bought_amount"] += invested
                if planned is not None:
                    refund += max(0.0, float(planned) - invested)
        # 未成交的回补计划全额退回（不可交易/现金不足 → 次日判定重试）
        pending = self._pending_settle.pop(date_str, {})
        for code, planned in pending.items():
            if code not in filled_codes:
                refund += float(planned)
        if added > 0:
            logger.info(f"暴露政策结算（{date_str}）：减仓释放额 +{added:,.2f}")
        if refund > 0:
            logger.info(f"暴露政策结算（{date_str}）：回补未花完额度退回 +{refund:,.2f}")
        self._release_budget += added + refund
        self._settled_dates.append(date_str)
        self._save_state()

    def stats_summary(self) -> Dict[str, Any]:
        payload = dict(self.stats)
        payload.update(
            {
                "enabled": True,
                "policy": self.settings.policy,
                "model_root": self.settings.model_root,
                "arm_suffix": self.settings.arm_suffix,
                "coverage_start": self.settings.coverage_start or "",
                "replenish": self.settings.replenish,
                "trim_tolerance": self.settings.trim_tolerance,
                "release_budget": float(self._release_budget),
                "pending_settle_days": len(self._pending_settle),
                "fingerprint": self._provider.fingerprint if self._provider else "",
                "pinned_fold": self.settings.pinned_fold or "",
                "warmup_file": str(self._warmup_path()),
            }
        )
        return payload

    # ------------------------------------------------------------------ 判定实现
    def _plan_trim(
        self,
        date_str: str,
        multiplier: float,
        prices: Dict[str, float],
        sell_queue: set,
        protected: set,
        out: List[TradeInstruction],
    ) -> Dict[str, Any]:
        """按比例减仓判定（对齐 ``backtest/exposure_trim.py::_queue_exposure_trim``）。"""
        stats = {"orders": 0}
        if multiplier >= 1.0:
            return stats
        account = self.runner.account
        portfolio_value = float(account.get_total_value(prices))
        if portfolio_value <= 0:
            return stats
        position_value = portfolio_value - float(account.get_cash())
        if position_value <= 0:
            return stats
        excess = position_value - portfolio_value * multiplier
        tolerance = portfolio_value * self.settings.trim_tolerance
        if excess <= tolerance:
            self.stats["trim_skipped_below_tolerance"] += 1
            return stats
        fraction = min(1.0, excess / position_value)
        queued = 0
        for code, pos in account.get_positions().items():
            if pos.shares <= 0:
                continue
            if code in sell_queue or code in protected:
                continue  # 整仓卖出优先；保护持仓不参与政策减仓
            shares = resolve_trim_shares(pos.shares, fraction)
            if shares <= 0:
                continue
            out.append(
                TradeInstruction(
                    ts_code=code,
                    action="sell",
                    shares=shares,
                    price_type=self._sell_price_type,
                    reason=f"{TRIM_REASON_PREFIX}（{date_str} 判定）",
                    source_date=date_str,
                    original_signal_date=date_str,
                )
            )
            queued += 1
        self.stats["trim_trigger_days"] += 1
        self.stats["trim_orders"] += queued
        if queued == 0:
            self.stats["trim_skipped_no_targets"] += 1
            logger.warning(
                f"暴露政策：{date_str} 判定超配 {excess:,.2f}"
                f"（λ={multiplier:.2f}，目标持仓 {portfolio_value * multiplier:,.2f}，"
                f"当前持仓 {position_value:,.2f}），但无可减仓标的"
            )
            return stats
        if not self._quiet():
            logger.info(
                f"暴露政策：{date_str} 判定超配 {excess:,.2f}，{queued} 只按 {fraction:.1%} "
                "比例 T+1 减仓"
            )
        stats["orders"] = queued
        return stats

    def _plan_replenish(
        self,
        date_str: str,
        multiplier: float,
        prices: Dict[str, float],
        sell_queue: set,
        protected: set,
        is_plan_day: bool,
        t1_date: str,
        out: List[TradeInstruction],
    ) -> Dict[str, Any]:
        """对称回补判定（对齐 ``backtest/exposure_replenish.py::_queue_exposure_replenish``）。"""
        stats = {"orders": 0}
        if not self.settings.replenish:
            return stats
        budget = float(self._release_budget)
        if budget <= 0:
            self.stats["replenish_skipped_no_budget"] += 1
            return stats
        if is_plan_day:
            # 新调仓计划日：计划会用同一笔现金重新分配仓位，回补不得抢现金（释放额清零）
            self._release_budget = 0.0
            self.stats["replenish_budget_reset_days"] += 1
            if not self._quiet():
                logger.info(f"暴露政策：{date_str} 新调仓计划生成，释放额 {budget:,.2f} 归零")
            return stats
        account = self.runner.account
        positions = account.get_positions()
        if not positions:
            return stats
        portfolio_value = float(account.get_total_value(prices))
        if portfolio_value <= 0:
            return stats
        position_value = portfolio_value - float(account.get_cash())
        room = portfolio_value * multiplier - position_value
        topup = min(float(account.get_cash()), budget, max(room, 0.0))
        tolerance = portfolio_value * self.settings.trim_tolerance
        if topup <= tolerance:
            self.stats["replenish_skipped_below_tolerance"] += 1
            return stats
        candidates = []
        for code, pos in positions.items():
            if pos.shares <= 0 or code in sell_queue or code in protected:
                continue
            price = prices.get(code)
            value = pos.shares * (float(price) if price and price > 0 else float(pos.buy_price))
            if value > 0:
                candidates.append((code, value))
        total_value = sum(value for _, value in candidates)
        if total_value <= 0:
            logger.warning(f"暴露政策：{date_str} 判定可补 {topup:,.2f}，但无可回补标的")
            return stats
        planned: Dict[str, float] = {}
        for code, value in candidates:
            amount = topup * (value / total_value)
            price = float(prices.get(code) or 0.0)
            if amount <= 0 or price <= 0:
                continue
            shares = compute_lot_shares(amount, price)
            if shares <= 0:
                continue  # 不足一手：比例语义优先，跳过（与引擎一致）
            out.append(
                TradeInstruction(
                    ts_code=code,
                    action="buy",
                    shares=shares,
                    price_type=self._buy_price_type,
                    reason=f"{REPLENISH_REASON_PREFIX}（{date_str} 判定）",
                    source_date=date_str,
                    original_signal_date=date_str,
                    keep_buy_date=True,  # 加仓不重置买入日；失败不进补位队列
                )
            )
            planned[code] = amount
        if not planned:
            logger.warning(
                f"暴露政策：{date_str} 判定已放开 {topup:,.2f}，但无可回补标的（均不足一手）"
            )
            return stats
        self.stats["replenish_trigger_days"] += 1
        self.stats["replenish_orders"] += len(planned)
        self._release_budget = max(0.0, budget - topup)
        self._pending_settle.setdefault(to_date_str(t1_date), {}).update(planned)
        if not self._quiet():
            logger.info(
                f"暴露政策：{date_str} 暴露上限放开，回补 {topup:,.2f}"
                f"（{len(planned)} 只按比例 T+1 买入）"
            )
        stats["orders"] = len(planned)
        return stats

    # ------------------------------------------------------------------ 输入装配
    def _ensure_feature(self, date_str: str) -> None:
        """确保判定日推理特征存在（缺失时走 ensure 构建；失败抛出由调用侧降级）。"""
        path = self._feature_root / f"{date_str}.parquet"
        if path.exists():
            return
        if self._feature_root != Path(self.data_root) / "features" / "cs_infer":
            raise FileNotFoundError(f"缺失推理特征分区: {path}（自定义特征源不自动构建）")
        success, _missing, error_detail = ensure_features_for_date(
            self.runner.storage,
            self.runner.loader,
            self.runner.feature_builder,
            self.runner.cleaner,
            self.runner.client,
            date_str,
            force=False,
        )
        if not success:
            raise RuntimeError(f"当日推理特征构建失败: {error_detail}")
        self.runner.feature_builder.clear_caches()

    def _close_prices(self, date_str: str) -> Dict[str, float]:
        """判定日收盘价字典（缺失回退前收；失败返回空字典由估值层回退买入价）。"""
        try:
            daily = self.runner.loader.load_clean_daily_by_date(date_str)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"暴露政策：加载 {date_str} 行情失败: {exc}")
            return {}
        if daily is None or daily.empty:
            return {}
        prices: Dict[str, float] = {}
        for _, row in daily.iterrows():
            price = row.get("close")
            if pd.isna(price) or float(price or 0) <= 0:
                price = row.get("pre_close")
            if pd.isna(price):
                continue
            value = float(price)
            if value > 0:
                prices[str(row["ts_code"])] = value
        return prices

    def _holdings_rows(self, date_str: str) -> pd.DataFrame:
        """持仓行（ts_code / weight / remaining_intervals；与引擎 ``_get_holdings_rows_for_policy`` 同构）。"""
        account = self.runner.account
        positions = account.get_positions()
        if not positions:
            return pd.DataFrame(columns=["ts_code", "weight", "remaining_intervals"])
        prices = self._close_prices(date_str)
        total_value = float(account.get_total_value(prices))
        if total_value <= 0:
            return pd.DataFrame(columns=["ts_code", "weight", "remaining_intervals"])
        rows = []
        for code, pos in positions.items():
            price = prices.get(code)
            value = pos.shares * (float(price) if price and price > 0 else float(pos.buy_price))
            rows.append(
                {
                    "ts_code": code,
                    "weight": value / total_value,
                    "remaining_intervals": self._remaining_intervals(date_str, pos),
                }
            )
        return pd.DataFrame(rows)

    def _remaining_intervals(self, date_str: str, pos) -> Optional[int]:
        """剩余持有交易日（E = 买入位置 + holding_period；与引擎同口径）。"""
        trade_dates = self._open_trade_dates()
        if not trade_dates:
            return None
        index = {day: idx for idx, day in enumerate(trade_dates)}
        buy_idx = index.get(str(pos.buy_date))
        cur_idx = index.get(date_str)
        if buy_idx is None or cur_idx is None:
            return None
        return int(buy_idx + self._holding_period - cur_idx - 1)

    def _open_trade_dates(self) -> List[str]:
        cached = getattr(self, "_trade_dates_cache", None)
        if cached:
            return cached
        try:
            cal = self.runner.loader.load_clean_trade_cal()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"暴露政策：加载交易日历失败: {exc}")
            return []
        if cal is None or cal.empty:
            return []
        days = [str(day) for day in cal.loc[cal["is_open"] == 1, "cal_date"].tolist()]
        self._trade_dates_cache = days
        return days

    # ------------------------------------------------------------------ 状态
    def _expire_unsettled(self, today: str) -> None:
        """对账：执行日已过的未结算回补计划全额退回（T1 未执行 / 进程中断场景）。"""
        for day in sorted(list(self._pending_settle.keys())):
            if day < today:
                refund = sum(float(value) for value in self._pending_settle.pop(day).values())
                self._release_budget += refund
                self.stats["replenish_expired_refunds"] += 1
                logger.warning(
                    f"暴露政策：{day} 的回补计划未结算（T1 未执行），"
                    f"额度 {refund:,.2f} 退回释放额"
                )

    def _state_payload(self) -> Dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "fingerprint": self._provider.fingerprint if self._provider else "",
            "release_budget": float(self._release_budget),
            "pending_settle": {day: dict(items) for day, items in self._pending_settle.items()},
            "settled_dates": list(self._settled_dates)[-400:],
            "last_judged_date": self._last_judged_date,
            "last_multiplier": float(self._last_multiplier),
            "stats": dict(self.stats),
            "provider": self._provider.export_state() if self._provider else {},
        }

    def _save_state(self) -> None:
        if self._provider is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as handle:
            json.dump(self._state_payload(), handle, ensure_ascii=False)

    def _load_state(self) -> None:
        if self._state_loaded:
            return
        self._state_loaded = True
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"暴露政策状态读取失败（按空状态处理）: {exc}")
            return
        if int(payload.get("version", 0)) != _STATE_VERSION:
            logger.warning("暴露政策状态版本不符，已清空（禁止静默续用旧状态）")
            return
        fingerprint = str(payload.get("fingerprint", ""))
        current = self._provider.fingerprint if self._provider else ""
        if fingerprint and current and fingerprint != current:
            logger.warning(
                f"暴露政策状态指纹不一致（{fingerprint} != {current}）："
                "策略/模型源已变化，旧状态已清空重算"
            )
            return
        self._release_budget = float(payload.get("release_budget", 0.0) or 0.0)
        self._pending_settle = {
            str(day): {str(code): float(amount) for code, amount in (items or {}).items()}
            for day, items in (payload.get("pending_settle") or {}).items()
        }
        self._settled_dates = [str(day) for day in (payload.get("settled_dates") or [])]
        self._last_judged_date = str(payload.get("last_judged_date", "") or "")
        self._last_multiplier = float(payload.get("last_multiplier", 1.0) or 1.0)
        for key, value in (payload.get("stats") or {}).items():
            if key in self.stats:
                self.stats[key] = value
        if self._provider is not None and payload.get("provider"):
            try:
                self._provider.restore_state(payload["provider"])
            except ValueError as exc:
                logger.warning(f"暴露政策 provider 面板状态恢复失败（丢弃并重建）: {exc}")
        logger.info(
            f"暴露政策状态已恢复：释放额 {self._release_budget:,.2f}，"
            f"待结算计划 {len(self._pending_settle)} 日，上次判定 {self._last_judged_date or '-'}"
        )

    # ------------------------------------------------------------------ 其他
    def _quiet(self) -> bool:
        return bool(getattr(self.runner, "quiet", False))


def ensure_paper_exposure_policy(runner) -> Optional[PaperExposurePolicy]:
    """惰性解析并挂载纸面暴露政策（未启用返回 None；配置错误直接报错）。

    - 结果缓存到 ``runner.exposure_policy``（含 None 的"已解析关闭"状态）；
    - ``run_t0`` / T1 结算 / workflow 均通过本函数取政策实例，
      保证手动子命令（t0/t1）与常规 run 行为一致。
    """
    if getattr(runner, "_exposure_policy_resolved", False):
        return getattr(runner, "exposure_policy", None)
    config = runner.paper_storage.load_config() or {}
    settings = PaperExposureSettings.from_config(config)
    runner._exposure_policy_resolved = True
    if settings is None:
        runner.exposure_policy = None
        return None
    from ..common.config import get_data_root

    data_root = str(getattr(runner.storage, "root_path", None) or get_data_root())
    policy = PaperExposurePolicy(runner, settings, data_root=data_root)
    policy.bind()
    runner.exposure_policy = policy
    return policy
