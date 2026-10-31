# -*- coding: utf-8 -*-
"""P4-T5 纸面影子窗口逐日证据累积与 ``paper_shadow`` runs 正式出口。

live 快照会裁剪历史成交/持仓，故每个真实影子日提交后立即把当日原始记录追加到
窗口累积器；窗口末统一写新批次，不回改既有 runs，也不把历史回放计入 SLA。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from src.lazybull.v2.common.table_schema import SNAPSHOT_KEYS
from src.lazybull.v2.core.accounting.engine_state import loads_blob
from src.lazybull.v2.core.accounting.exports import get_execution_attribution, get_trades
from src.lazybull.v2.core.accounting.nav import generate_nav_curve
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.hosts.backtest.replay import _compute_replay_metrics
from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.replay import execution_config_from_engine


def _day_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().replace("-", "")[:8]


def _record_day(record: Mapping[str, Any]) -> str:
    for key in ("trade_date", "date", "execution_date", "signal_date", "ranking_date"):
        if key in record:
            day = _day_text(record[key])
            if day:
                return day
    return ""


class PaperShadowAccumulator:
    """逐日保存未裁剪的原始记录；重复追加同一记录保持幂等。"""

    def __init__(self) -> None:
        self.portfolio_values: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.attribution: List[Dict[str, Any]] = []
        self.holdings: List[Dict[str, Any]] = []
        self.days: List[str] = []
        self._seen: set[str] = set()

    def _append(self, target: List[Dict[str, Any]], record: Mapping[str, Any]) -> None:
        identity = fingerprint(dict(record))
        if identity in self._seen:
            return
        self._seen.add(identity)
        target.append(dict(record))

    def append_from_host(self, host: PaperHost, day: str) -> Dict[str, int]:
        """从当日权威快照追加指定日记录；快照缺失或代际不符即失败。"""
        data = host.store.load_engine_state_blob()
        if data is None:
            raise RuntimeError(f"影子日 {day} 缺权威引擎快照")
        blob = loads_blob(data)
        generation = blob.get("paper_generation") or {}
        if generation.get("trade_date") != day:
            raise RuntimeError(f"影子日 {day} 快照代际不符: {generation!r}")
        fields = blob["fields"]
        for record in fields["portfolio_values"]:
            if _record_day(record) == day:
                self._append(self.portfolio_values, record)
        counts = {"trades": 0, "attribution": 0, "holdings": 0}
        for key, target, output in (
            ("trades", self.trades, "trades"),
            ("execution_attribution_records", self.attribution, "attribution"),
            ("holdings_snapshots", self.holdings, "holdings"),
        ):
            for record in fields[key]:
                if _record_day(record) == day:
                    before = len(target)
                    self._append(target, record)
                    counts[output] += len(target) - before
        if day not in self.days:
            self.days.append(day)
            self.days.sort()
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """导出跨日跑进程可恢复的累积台账（不含主机路径与运行 ID）。"""
        return {
            "portfolio_values": self.portfolio_values,
            "trades": self.trades,
            "attribution": self.attribution,
            "holdings": self.holdings,
            "days": self.days,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PaperShadowAccumulator":
        """恢复逐日台账并按内容重建去重键；缺字段或多日重复标记即拒绝。"""
        required = ("portfolio_values", "trades", "attribution", "holdings", "days")
        if any(key not in payload or not isinstance(payload[key], list) for key in required):
            raise ValueError("影子累积台账缺字段或字段不是列表")
        accumulator = cls()
        for key in required[:-1]:
            for record in payload[key]:
                if not isinstance(record, Mapping):
                    raise ValueError(f"影子累积台账 {key} 含非对象记录")
                accumulator._append(getattr(accumulator, key), record)
        accumulator.days = sorted(set(payload["days"]))
        if len(accumulator.days) != len(payload["days"]):
            raise ValueError("影子累积台账 days 含重复交易日")
        return accumulator

    def result(self, *, host: PaperHost, start_date: str, end_date: str) -> Dict[str, Any]:
        """生成 runs writer 所需原始帧；净值来自完整逐日组合值而非跳空 NAV 投影。"""
        if not self.portfolio_values:
            raise RuntimeError("影子窗口无 portfolio_values，拒绝生成空 runs 批次")
        nav_curve = generate_nav_curve(
            self.portfolio_values, float(host.engine_kwargs["initial_capital"])
        )
        result = _compute_replay_metrics(
            nav_curve, start_date, end_date, host.trading_config.top_n, 0
        )
        # runs loader 契约 §9.2 以折内 nav 首尾比为准；return 列是相对初始
        # 本金的累计收益，不能直接当作本窗口收益。
        nav0 = float(nav_curve["nav"].iloc[0])
        navn = float(nav_curve["nav"].iloc[-1])
        result["bt_total_return"] = round(navn / nav0 - 1.0, 6)
        result["_nav_curve"] = nav_curve
        result["_trades"] = get_trades(self.trades)
        result["_execution_attribution"] = get_execution_attribution(self.attribution)
        result["_holdings_snapshot"] = pd.DataFrame(self.holdings, columns=SNAPSHOT_KEYS)
        engine = host._new_engine({})
        result["_execution_config"] = execution_config_from_engine(
            engine,
            model_version=host.trading_config.model_version,
            buy_price=host.trading_config.buy_price,
        )
        result["_shadow_days"] = list(self.days)
        return result


def write_paper_shadow_batch(
    result: Mapping[str, Any],
    *,
    batch_id: str,
    runs_root: Path,
    config: Dict[str, object],
    code_state: Dict[str, object],
    data_state: Dict[str, object],
    model_version: Any,
    train_start: Optional[str],
    train_end: Optional[str],
    baseline_ref: Optional[str] = None,
    arm_label: str = "v2_shadow",
) -> Path:
    """按单 fold 拓扑写 ``paper_shadow`` 并立即经 runs loader 硬校验。"""
    metrics = {key: value for key, value in result.items() if not key.startswith("_")}
    fold = {
        "split_index": 0,
        "model_version": model_version,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": result["bt_start"],
        "test_end": result["bt_end"],
        "metrics": metrics,
        "nav_curve": result["_nav_curve"],
        "trades": result["_trades"],
        "attribution": result["_execution_attribution"],
        "holdings_snapshot": result["_holdings_snapshot"],
    }
    notes = {
        "shadow_window": f"{result['bt_start']}~{result['bt_end']}",
        "shadow_days": str(len(result.get("_shadow_days", []))),
        "shadow_sla_source": "real_calendar_only",
        "shadow_arm": arm_label,
    }
    return write_runs_batch(
        [fold],
        batch_id=batch_id,
        runs_root=runs_root,
        config=config,
        code_state=code_state,
        data_state=data_state,
        baseline_ref=baseline_ref,
        arms=(arm_label,),
        source="v2_p4_paper_shadow",
        extra_meta_notes=notes,
        host_mode="paper_shadow",
    )
