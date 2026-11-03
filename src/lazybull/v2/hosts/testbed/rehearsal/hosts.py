# -*- coding: utf-8 -*-
"""P5b T6 演练的宿主层效应注入（只改 NAV 读数，不改成交/现金/持仓/日期）。

注入点（唯一）
==============
``host_factory`` 包装默认宿主：内层宿主照常 ``run_t1`` 驱动真实纸面链路；外层在驱动后读取
当日的**权威完整快照**，按冻结效应算出该臂的注入 NAV，并在**读侧**
（``load_engine_state_blob``）把当日 ``portfolio_values`` 行的 ``portfolio_value`` /
``market_value`` 替换为注入值。

纯函数口径（为什么不用「记住前一天」的状态）
==========================================
注入 NAV 定义为**基准路径前缀的纯函数**::

    V'_0 = V_0,   V'_j = V_0 * exp( Σ_{i<j} f(x_i) ),   x_i = ln(V_i / V_{i-1})

其中 ``f`` 是冻结效应（收益臂：``x - 5δ``；DD 臂：窗口内负收益的 ``(1-λ)/(1+ζ)`` 缩放）。
每日从**已持久化的完整 ``portfolio_values`` 历史**重算一次即可 —— 既不需要跨日内存状态，
也与「同 base + 同参数 ⇒ 同结果」的纯函数要求逐字一致（并有 ``transform_log_returns``
的路径级等价用例）。

为什么在读侧而不是写回状态库
============================
1. 臂根状态与投影保持与 ``base`` **逐字节一致** ⇒ 「只改 NAV」在结构上成立
   （成交、现金、持仓、日期、指令、pending 全不受影响）；
2. 后续虚拟日的决策只依赖现金/持仓/价格（``_calculate_portfolio_value`` 实时计算），
   不依赖历史 ``portfolio_values`` 的数值 ⇒ 注入不会通过状态泄漏进下一个交易决策；
3. 注入仍是宿主层动作：先于账本分片成型，**不是**「事后改分片或改报告」。

fail-closed：当日权威估值行必须恰一条且为最后处理日、NAV 有限为正；注入后非有限/非正即报错，
禁止静默降级为 0/NaN。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.lazybull.v2.core.accounting.engine_state import dumps_blob, loads_blob
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore
from src.lazybull.v2.hosts.testbed.arm_runtime import ArmSpec, HostFactory
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
    RehearsalEffect,
    apply_single_return,
)

__all__ = [
    "InjectedHost",
    "InjectedValuationStore",
    "RehearsalHostError",
    "injected_nav_series",
    "make_injected_host_factory",
]


class RehearsalHostError(TestbedError):
    """注入失败（估值行缺失/重复/非最后处理日、NAV 非有限、路径断裂）。"""

    __test__ = False


def _day_text(value: Any) -> str:
    """日期归一（``Timestamp``/``YYYYMMDD``/``YYYY-MM-DD`` → ``YYYYMMDD``）。"""
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")
    text = str(value).strip().replace("-", "").replace("/", "")
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else text


def injected_nav_series(
    base_nav: List[float], effect: RehearsalEffect, *, mask: Optional[Any] = None
) -> List[float]:
    """按冻结效应把基准 NAV 序列映射为注入 NAV 序列（纯函数；含 D0 首点）。

    Raises:
        RehearsalHostError: 序列非正/非有限，或注入后出现非有限/非正值。
    """
    values = np.asarray(list(base_nav), dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise RehearsalHostError("基准 NAV 序列含非有限或非正值（注入路径不成立）")
    if not effect.active or values.size == 1:
        return [float(value) for value in values]
    log_returns = np.diff(np.log(values))
    transformed = np.asarray(
        [
            apply_single_return(value, index, effect, mask=mask)
            for index, value in enumerate(log_returns)
        ],
        dtype=float,
    )
    path = values[0] * np.exp(np.concatenate(([0.0], np.cumsum(transformed))))
    if not np.isfinite(path).all() or (path <= 0.0).any():
        raise RehearsalHostError("注入后的 NAV 路径出现非有限或非正值（fail-closed）")
    return [float(value) for value in path]


class InjectedValuationStore:
    """读侧估值覆写代理（只覆写 ``load_engine_state_blob``，其余全部透传）。"""

    def __init__(
        self,
        inner: PaperStateStore,
        *,
        effect: RehearsalEffect,
        mask: Optional[Any] = None,
    ) -> None:
        self._inner = inner
        self._effect = effect
        self._mask = mask
        self._date: Optional[str] = None
        self._patch: Optional[Dict[str, float]] = None
        self.injected_days: int = 0

    # -- 透传 ---------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def inner(self) -> PaperStateStore:
        """内层真实状态库（审计用；注入路径不写它）。"""
        return self._inner

    # -- 注入 ---------------------------------------------------------------
    def observe(self, virtual_date: str) -> None:
        """驱动后按持久化的基准路径重算当日注入 NAV（纯函数；不写任何状态）。"""
        date = _day_text(virtual_date)
        self._date = date
        self._patch = None
        if not self._effect.active:
            return
        rows = self._valuation_rows()
        dates = [_day_text(row.get("date")) for row in rows]
        if date not in dates:
            raise RehearsalHostError(
                f"臂 {self._effect.arm_id!r} 在 {date} 缺权威估值行（注入无法定位当日）"
            )
        if dates.count(date) != 1:
            raise RehearsalHostError(
                f"臂 {self._effect.arm_id!r} 在 {date} 的权威估值行重复（{dates.count(date)} 条）"
            )
        if dates[-1] != date:
            raise RehearsalHostError(
                f"臂 {self._effect.arm_id!r} 的 {date} 不是最后处理日（{dates[-1]}）——禁止跳日"
            )
        base_nav = [float(row["portfolio_value"]) for row in rows]
        injected = injected_nav_series(base_nav, self._effect, mask=self._mask)
        value = float(injected[-1])
        if not math.isfinite(value) or value <= 0.0:
            raise RehearsalHostError(
                f"臂 {self._effect.arm_id!r} 在 {date} 的注入 NAV 非法: {value!r}"
            )
        capital = float(rows[-1].get("capital") or 0.0)
        self._patch = {"portfolio_value": value, "market_value": value - capital}
        self.injected_days += 1

    def _valuation_rows(self) -> List[Dict[str, Any]]:
        raw = self._inner.load_engine_state_blob()
        if raw is None:
            raise RehearsalHostError(
                f"臂 {self._effect.arm_id!r} 缺完整状态快照（无法计算注入 NAV）"
            )
        blob = loads_blob(raw)
        rows = [dict(row) for row in ((blob.get("fields") or {}).get("portfolio_values") or [])]
        if not rows:
            raise RehearsalHostError(f"臂 {self._effect.arm_id!r} 的 portfolio_values 为空")
        return rows

    def load_engine_state_blob(self) -> Optional[bytes]:
        """读取引擎状态字节（注入时替换当日估值行，其余字段原样）。"""
        raw = self._inner.load_engine_state_blob()
        if raw is None or self._patch is None or self._date is None:
            return raw
        blob = loads_blob(raw)
        fields = blob.get("fields") or {}
        matched = 0
        for row in fields.get("portfolio_values") or []:
            if _day_text((row or {}).get("date")) != self._date:
                continue
            row["portfolio_value"] = self._patch["portfolio_value"]
            row["market_value"] = self._patch["market_value"]
            matched += 1
        if matched != 1:
            raise RehearsalHostError(
                f"臂 {self._effect.arm_id!r} 在 {self._date} 覆写估值行失败（匹配 {matched} 条）"
            )
        return dumps_blob(blob)


class InjectedHost:
    """宿主包装：内层真实驱动 + 外层读侧估值覆写（其余属性全部透传）。"""

    def __init__(self, inner: PaperHost, store: InjectedValuationStore) -> None:
        self._inner = inner
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def store(self) -> InjectedValuationStore:  # type: ignore[override]
        """读侧估值覆写代理（``drive_virtual_day`` 的权威估值入口）。"""
        return self._store

    @property
    def inner_host(self) -> PaperHost:
        """内层真实宿主（审计用）。"""
        return self._inner

    def run_t1(self, date: Any, fills: Any = None) -> None:
        """驱动真实 T+1 链路后立刻观测当日权威估值（不做任何写回）。

        **不抢占 T2 的失败语义**：只有宿主当日**真的驱动**（``last_day_result.driven``）
        才观测；``not_ready``/``already_run`` 等情形一律不覆写、不抛错，交由
        ``arm_runtime.drive_virtual_day`` 按 §4.3 分类（整族停留 / 幂等恢复）。
        """
        self._inner.run_t1(date, fills)
        result = getattr(self._inner, "last_day_result", None)
        if result is None or not bool(getattr(result, "driven", False)):
            return
        self._store.observe(str(date).strip())


def make_injected_host_factory(
    base_factory: HostFactory,
    *,
    effects: Dict[str, RehearsalEffect],
    masks: Optional[Dict[str, Any]] = None,
) -> HostFactory:
    """把默认宿主工厂包装成「按臂施加冻结效应」的工厂。

    Args:
        base_factory: 内层工厂（``ArmSpec, Path -> PaperHost``）。
        effects: ``{arm_id: RehearsalEffect}``（来自 manifest 冻结值）。
        masks: 可选 ``{arm_id: 窗口掩码}``（DD 臂用；缺省视为无掩码）。

    Returns:
        新工厂（同一臂规格仍落到同一暂存根，隔离语义不变）。
    """
    mask_table = dict(masks or {})

    def _factory(spec: ArmSpec, store_root: Path) -> InjectedHost:
        effect = effects.get(spec.arm_id)
        if effect is None:
            raise RehearsalHostError(
                f"臂 {spec.arm_id!r} 未在演练 manifest 中登记效应（禁止未登记臂注入）"
            )
        inner = base_factory(spec, store_root)
        store = InjectedValuationStore(inner.store, effect=effect, mask=mask_table.get(spec.arm_id))
        return InjectedHost(inner, store)

    return _factory
