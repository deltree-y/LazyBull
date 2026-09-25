"""回测暴露覆盖（P2-3 shadow 模式）：把"目标总仓位 ≤ λ"接到买入预算上。

设计边界（**最小侵入**）：

- 本 mixin **只做一件事**：把调仓/补齐时的买入预算基数乘上当日暴露系数 λ。
  不新增卖出指令类型、不改 T0/T1 链路、不动持仓与现金账务。
- 触发状态由**外部产物**（`risk.terminal_loss.exposure_gate` 的逐日判定表导出的
  两列 CSV：`日期, 暴露系数`）提供；默认 `None` 表示**不启用**（此时行为与改动前逐位一致）。
- 查表日期用**信号日**（`signal_date` = 执行日的上一交易日）：与门控契约
  "T 日判定 → T+1 生效"一致；缺失日期按 1.0 处理并计数，不静默外推。
- 系数必须落于 (0, 1]；> 1 表示加杠杆，直接报错（政策层只允许降暴露）。

语义说明（写进契约，避免误读）：本 mixin 是 **"缩减/暂停加仓"** 的一半——处于降暴露
状态时，新买入只投 λ 比例的资金；另一半是 **每日主动减仓**
（`backtest/exposure_trim.py`：每日判定 `持仓市值 > λ×组合总值 + 3% 容差` 则 T0 生成
按比例部分卖出单、T+1 开盘执行）。两条路径配合才构成完整降暴露：减仓把**存量**暴露
在 T+1 开盘降到 λ 附近，本 mixin 保证**增量**不买回。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

#: 暴露系数表的内部列名（最小两列；缺列硬报错）
EXPOSURE_TABLE_KEYS = ("date", "exposure_multiplier")


@dataclass
class ExposureOverrideStats:
    """按日查表统计（用于报告与测试）。"""

    covered_days: int = 0
    missing_days: int = 0
    reduced_days: int = 0
    missing_samples: list = field(default_factory=list)

    def record(self, date: str, multiplier: float) -> None:
        self.covered_days += 1
        if multiplier < 1.0:
            self.reduced_days += 1
        if not self.missing_samples:
            self.missing_samples = []


class BacktestExposureOverrideMixin:
    """提供"目标总仓位 ≤ λ"的买入预算覆盖（默认关闭）。"""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - 由具体引擎决定 MRO
        super().__init__(*args, **kwargs)

    #: 日期(YYYYMMDD) → 暴露系数；None 表示不启用
    exposure_table: Optional[Dict[str, float]] = None
    #: 在线政策 provider（terminal_loss P2-5）：需实现 ``multiplier_for(date, holdings)``
    exposure_policy_provider: Any = None
    #: 在线 provider 统计（由 provider 自身持有，此处仅记录摘要）
    exposure_policy_summary: Optional[Dict[str, object]] = None
    #: 查表统计
    exposure_stats: Optional[ExposureOverrideStats] = None
    #: 未在表中出现的日期集合（用于告警样例）
    _exposure_missing_dates: Dict[str, int] = {}
    #: 建仓折扣回补（A3 v2 / P2-4 镜像缺口）：λ<1 信号日的买入预算折扣按实际成交额
    #: 记入回补释放额（`exposure_release_budget`），λ 恢复后由既有回补机制买回。
    #: 默认 False：不产生任何副作用，与既有回测逐位一致。
    exposure_budget_discount_replenish: bool = False

    def _has_exposure_source(self) -> bool:
        """是否启用任一种暴露政策源（系数表 或 在线 provider）。"""
        return bool(self.exposure_table) or self.exposure_policy_provider is not None

    def _record_budget_discount_release(self, signal_date, amount: float) -> None:
        """建仓折扣入回补释放额（A3 v2 / P2-4 镜像缺口，2026-09-25）。

        背景：λ<1 信号日的买入预算被乘以 λ（P2-3），若建仓恰逢政策期，该批次只建了
        λ 比例的仓位；政策恢复后折扣部分**不在减仓释放额记账内**（回补只记减仓动作），
        半额仓位会挂到下一次批次轮换（实测 OOS12：17 只 0.47 权重持续 23 个交易日）。

        口径：实际成交额 A 对应"未打折时应买 A/λ"，折扣额 = A × (1/λ − 1)；
        与减仓释放额共用同一余额（`exposure_release_budget`）与同一回补规则
        （上界三选一 / T+1 执行 / 新调仓计划日清零——全部语义不变）。

        副作用边界：开关关闭或 λ=1 时**不产生任何副作用**（逐位一致）。
        回补买入本身不走本记账（其执行路径不调用本方法），不会循环记账。

        Args:
            signal_date: 买入计划的信号日（λ 取信号日口径，与预算缩放一致）
            amount: 实际成交金额（元）
        """
        if not self.exposure_budget_discount_replenish:
            return
        if amount is None or amount <= 0:
            return
        multiplier = self._get_exposure_multiplier(signal_date)
        if multiplier >= 1.0:
            return
        discount = float(amount) * (1.0 / multiplier - 1.0)
        self.exposure_release_budget = float(
            getattr(self, "exposure_release_budget", 0.0) or 0.0
        ) + discount
        stats = getattr(self, "exposure_replenish_stats", None)
        if stats is not None:
            stats["budget_discount_release_amount"] = (
                float(stats.get("budget_discount_release_amount", 0.0)) + discount
            )
        logger.debug(
            f"建仓折扣入释放额: 信号日 {self._date_key(signal_date)} λ={multiplier:.3f} "
            f"成交额 {amount:.2f} → 折扣 {discount:.2f}（累计释放额 "
            f"{self.exposure_release_budget:.2f}）"
        )

    def _date_key(self, date) -> str:
        """统一日期键格式（YYYYMMDD）。"""
        return str(date)[:10].replace("-", "")

    def _get_holdings_rows_for_policy(self, date) -> pd.DataFrame:
        """当前持仓行（供在线 provider 打分；与持仓快照同列名）。"""
        portfolio_value = self._calculate_portfolio_value(date)
        rows = []
        for stock in self.positions:
            value = self._position_market_value(date, stock)
            rows.append(
                {
                    "ts_code": stock,
                    "weight": (value / portfolio_value) if portfolio_value else None,
                    "remaining_intervals": self._remaining_intervals(date, stock),
                }
            )
        return pd.DataFrame(rows, columns=["ts_code", "weight", "remaining_intervals"])

    def _remaining_intervals(self, date, stock: str) -> Optional[int]:
        """剩余持有交易日（与持仓快照同一口径：E = 买入位置 + holding_period）。"""
        info = self.positions.get(stock) or {}
        buy_date = info.get("buy_date")
        if buy_date is None or not hasattr(self, "_trade_date_index"):
            return None
        buy_idx = self._trade_date_index.get(buy_date)
        idx = self._trade_date_index.get(date)
        if buy_idx is None or idx is None:
            return None
        return int(buy_idx + self.holding_period - idx - 1)

    def _get_exposure_multiplier(self, date) -> float:
        """取当日暴露系数；未启用或日期缺失均返回 1.0。

        优先级：在线 provider（现算，P2-5）> 系数表；两者均为空时直接返回 1.0
        （默认关闭必须与改动前逐位一致）。
        """
        if self.exposure_policy_provider is not None and not self.exposure_table:
            holdings = self._get_holdings_rows_for_policy(date)
            multiplier = float(self.exposure_policy_provider.multiplier_for(date, holdings))
            if not 0.0 < multiplier <= 1.0:
                raise ValueError(
                    f"在线政策 provider 返回非法系数 {multiplier}（日期 {self._date_key(date)}）；"
                    f"政策层只允许降暴露，禁止加杠杆"
                )
            if self.exposure_stats is not None:
                self.exposure_stats.covered_days += 1
                if multiplier < 1.0:
                    self.exposure_stats.reduced_days += 1
            return multiplier
        if not self.exposure_table:
            return 1.0
        key = self._date_key(date)
        if key not in self.exposure_table:
            if not self._exposure_missing_dates:
                logger.warning(
                    f"暴露系数表缺少日期 {key}（首个缺口），该日按 1.0 处理；"
                    f"请确认导出时已按交易日历补齐台账缺口"
                )
            self._exposure_missing_dates[key] = self._exposure_missing_dates.get(key, 0) + 1
            if self.exposure_stats is not None:
                self.exposure_stats.missing_days += 1
            return 1.0
        multiplier = float(self.exposure_table[key])
        if not 0.0 < multiplier <= 1.0:
            raise ValueError(
                f"暴露系数必须落于 (0, 1]，日期 {key} 取到 {multiplier}；"
                f"政策层只允许降暴露，禁止加杠杆"
            )
        if self.exposure_stats is not None:
            self.exposure_stats.covered_days += 1
            if multiplier < 1.0:
                self.exposure_stats.reduced_days += 1
        return multiplier

    def set_exposure_table(
        self,
        table: Optional[Dict[str, float]],
        verbose: bool = True,
        replenish: bool = False,
        trim_tolerance: Optional[float] = None,
        budget_discount_replenish: bool = False,
    ) -> None:
        """装载暴露系数表（重复调用覆盖旧表并重置统计）。

        Args:
            table: {YYYYMMDD: 系数}；None / 空 = 关闭暴露覆盖（逐位一致）
            verbose: 是否打印装载日志
            replenish: 是否启用**对称回补**（P2-4；仅在 table 非空时生效）
            trim_tolerance: 减仓/回补**共用容差**（组合总值比例）；None = 保留引擎默认（3%）
        """
        if trim_tolerance is not None:
            if not 0.0 < float(trim_tolerance) < 1.0:
                raise ValueError(f"trim_tolerance 必须落于 (0, 1)，当前 {trim_tolerance}")
            self.exposure_trim_tolerance = float(trim_tolerance)
        if budget_discount_replenish and not replenish:
            raise ValueError(
                "budget_discount_replenish 必须与 replenish=True 同用"
                "（折扣记账服务于回补，单开会记了没人回补）"
            )
        self.exposure_budget_discount_replenish = bool(budget_discount_replenish) and bool(table)
        self.exposure_replenish_enabled = bool(replenish) and bool(table)
        if table is not None:
            cleaned: Dict[str, float] = {}
            for key, value in table.items():
                multiplier = float(value)
                if not 0.0 < multiplier <= 1.0:
                    raise ValueError(f"暴露系数必须落于 (0, 1]，日期 {key} 取到 {multiplier}")
                cleaned[str(key)[:10].replace("-", "")] = multiplier
            self.exposure_table = cleaned
            self.exposure_stats = ExposureOverrideStats()
            self._exposure_missing_dates = {}
            if verbose:
                reduced = sum(1 for value in cleaned.values() if value < 1.0)
                logger.info(
                    f"暴露覆盖已启用: {len(cleaned)} 个交易日（其中降暴露 {reduced} 日，"
                    f"系数取值集合={sorted(set(cleaned.values()))}）"
                )
        else:
            self.exposure_table = None
            self.exposure_stats = None

    def set_exposure_policy(
        self,
        provider: Any,
        verbose: bool = True,
        replenish: bool = False,
        trim_tolerance: Optional[float] = None,
        budget_discount_replenish: bool = False,
    ) -> None:
        """装载**在线政策 provider**（terminal_loss P2-5：λ_t 随持仓现算）。

        Args:
            provider: 需实现 ``multiplier_for(date, holdings) -> λ``；None = 关闭（逐位一致）
            verbose: 是否打印装载日志
            replenish: 是否启用对称回补（P2-4）
            trim_tolerance: 减仓/回补共用容差；None = 保留引擎默认
            budget_discount_replenish: 是否启用**建仓折扣回补**（A3 v2：λ<1 信号日的买入
                预算折扣按实际成交额入回补释放额；必须与 replenish=True 同用）
        """
        if trim_tolerance is not None:
            if not 0.0 < float(trim_tolerance) < 1.0:
                raise ValueError(f"trim_tolerance 必须落于 (0, 1)，当前 {trim_tolerance}")
            self.exposure_trim_tolerance = float(trim_tolerance)
        if budget_discount_replenish and not replenish:
            raise ValueError(
                "budget_discount_replenish 必须与 replenish=True 同用"
                "（折扣记账服务于回补，单开会记了没人回补）"
            )
        if provider is not None and not hasattr(provider, "multiplier_for"):
            raise TypeError(
                "在线政策 provider 必须实现 multiplier_for(date, holdings)；"
                "传入函数/其他对象一律拒绝（避免第二套接口）"
            )
        self.exposure_policy_provider = provider
        self.exposure_budget_discount_replenish = (
            bool(budget_discount_replenish) and replenish and provider is not None
        )
        self.exposure_replenish_enabled = bool(replenish) and provider is not None
        if provider is None:
            self.exposure_policy_summary = None
            return
        self.exposure_table = None
        self.exposure_stats = ExposureOverrideStats()
        self._exposure_missing_dates = {}
        if verbose:
            summary = getattr(provider, "stats_summary", None)
            detail = summary() if callable(summary) else {}
            self.exposure_policy_summary = detail
            logger.info(
                "暴露政策已启用（在线现算）: "
                f"{detail.get('policy_config', 'provider')}"
                f"（指纹 {detail.get('policy_fingerprint', 'n/a')}）"
            )

    def get_exposure_report(self) -> Dict[str, object]:
        """返回政策源统计（未启用时为 enabled=False）。"""
        if self.exposure_policy_provider is not None and not self.exposure_table:
            summary = getattr(self.exposure_policy_provider, "stats_summary", None)
            detail = dict(summary() if callable(summary) else {})
            detail["enabled"] = True
            detail["source"] = "online"
            stats = self.exposure_stats or ExposureOverrideStats()
            detail["covered_days"] = stats.covered_days
            detail["reduced_days"] = stats.reduced_days
            return detail
        if not self.exposure_table:
            return {"enabled": False}
        stats = self.exposure_stats or ExposureOverrideStats()
        return {
            "enabled": True,
            "table_days": len(self.exposure_table),
            "covered_days": stats.covered_days,
            "reduced_days": stats.reduced_days,
            "missing_days": stats.missing_days,
            "missing_samples": sorted(self._exposure_missing_dates)[:5],
        }


def load_exposure_table(path: Path) -> Dict[str, float]:
    """读取两列 CSV（`日期, 暴露系数`）并返回 {YYYYMMDD: 系数}。

    Raises:
        ValueError: 缺列、空表、非法系数或重复日期
    """
    frame = pd.read_csv(Path(path), encoding="utf-8-sig")
    missing = [column for column in ("日期", "暴露系数") if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{Path(path).name} 缺少列 {missing}；暴露系数表必须为 `日期, 暴露系数` 两列"
        )
    frame = frame.rename(
        columns={"日期": EXPOSURE_TABLE_KEYS[0], "暴露系数": EXPOSURE_TABLE_KEYS[1]}
    )
    if frame.empty:
        raise ValueError(f"{Path(path).name} 为空，无法构造暴露系数表")
    frame["date"] = frame["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    frame["exposure_multiplier"] = frame["exposure_multiplier"].astype(float)
    bad = frame[(frame["exposure_multiplier"] <= 0.0) | (frame["exposure_multiplier"] > 1.0)]
    if not bad.empty:
        raise ValueError(
            f"{Path(path).name} 存在非法暴露系数（必须落于 (0, 1]）："
            f"{bad.head(3).to_dict('records')}"
        )
    duplicated = frame["date"][frame["date"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{Path(path).name} 日期重复：{duplicated[:3]}；系数表必须一日一行")
    return dict(zip(frame["date"], frame["exposure_multiplier"]))
