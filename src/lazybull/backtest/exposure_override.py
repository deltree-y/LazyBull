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
from typing import Dict, Optional

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
    #: 查表统计
    exposure_stats: Optional[ExposureOverrideStats] = None
    #: 未在表中出现的日期集合（用于告警样例）
    _exposure_missing_dates: Dict[str, int] = {}

    def _date_key(self, date) -> str:
        """统一日期键格式（YYYYMMDD）。"""
        return str(date)[:10].replace("-", "")

    def _get_exposure_multiplier(self, date) -> float:
        """查当日暴露系数；未启用或日期缺失均返回 1.0。"""
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

    def get_exposure_report(self) -> Dict[str, object]:
        """返回查表统计（未启用时为 enabled=False）。"""
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
