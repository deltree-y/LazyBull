# -*- coding: utf-8 -*-
"""数据面协议（契约 docs/contracts/protocols.md §1.1 / §1.2 的代码化）。

签名与语义以协议文本为唯一依据；docstring 保留协议的行为语义注释
（幂等 / 防冲突 / 副作用 / 失败语义，对应 §8 行为语义矩阵 R4-M8：
幂等 = 重复执行无害且成功；防冲突 = 重复执行校验一致性、不一致显式报错）。

本文件只承载 §1 数据面协议；signal / core / hosts / evidence 协议占位见文末 TODO。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

from src.lazybull.v2.common.types import (
    FeatureQuery,
    LabelQuery,
    LedgerEntry,
    PanelFrame,
    TradeDate,
    TSCode,
)

__all__ = ["DataStore", "FeatureBuilder"]


@runtime_checkable
class DataStore(Protocol):
    """数据面唯一入口（raw / normalized / features / labels / events / state 的读写）。

    写入者唯一原则：features 仅构建器写、raw 仅下载器写、台账仅证据层经本接口写（R4-H5）。
    """

    # ========== 读取（无副作用） ==========

    def load_features(self, query: FeatureQuery) -> PanelFrame:
        """加载特征面板（唯一读取入口，manifest 校验列名合法性与 available_from）。

        幂等：是
        副作用：无
        失败：列名未登记或 available_from 越界 ⇒ ValueError
        """
        ...

    def load_labels(self, query: LabelQuery) -> pd.DataFrame:
        """加载标签表。

        返回：index = (trade_date, ts_code)，columns = [label_value, maturity_status]
        """
        ...

    def load_events(
        self,
        event_type: str,
        start_date: TradeDate,
        end_date: TradeDate,
        universe: Sequence[TSCode] | None = None,
    ) -> pd.DataFrame:
        """加载事件表（去重对齐后）。

        返回：columns = [event_type, ts_code, event_date, knowledge_date, revision_key, ...]
        """
        ...

    def load_market_state(self, date: TradeDate) -> pd.DataFrame:
        """加载市场状态（波动 / 流动性 / 宽度 / 风格动量）。

        返回：单行 DataFrame（index = date）
        """
        ...

    def load_raw(
        self,
        source: str,
        dataset: str,
        start_date: TradeDate,
        end_date: TradeDate,
    ) -> pd.DataFrame:
        """加载原始数据（只读，normalized 构建用）。"""
        ...

    # ========== 写入（仅授权写入方调用；防冲突语义，见 §8 矩阵） ==========

    def append_features(self, date: TradeDate, group: str, df: pd.DataFrame) -> None:
        """追加特征分区（两阶段提交：临时文件 → 原子纳入 manifest）。

        防冲突：同日同 group 重复写 ⇒ 内容指纹一致才放行，不一致报错（非幂等语义——
        幂等的定义是重复执行无害且成功，此处是显式冲突检测，见 R4-M8）
        副作用：写磁盘 + 更新 manifest
        失败：列名未登记 / 分区已封存 ⇒ RuntimeError
        """
        ...

    def append_labels(self, label_name: str, df: pd.DataFrame) -> None:
        """追加标签（封存前幂等可重写；封存后指纹一致 no-op、不一致拒绝改写）。"""
        ...

    def append_events(self, event_type: str, df: pd.DataFrame) -> None:
        """追加事件（按 event_date 年分区）。"""
        ...

    def append_ledger_entry(self, entry: LedgerEntry) -> None:
        """假设台账写入（R4-H5：台账是 append-only 登记资产，写入统一走数据面；
        evidence 层只消费本 API，不直接写文件——依赖方向由此干净。
        R5：参数类型化为 LedgerEntry，禁止裸 Mapping。）
        """
        ...

    def get_manifest(self) -> Mapping[str, Any]:
        """获取 manifest 快照（只读；返回 MappingProxyType）。"""
        ...


@runtime_checkable
class FeatureBuilder(Protocol):
    """特征构建器（调度 factors/ 物化列族）。

    注意：本 Protocol 仅在 v2 命名空间生效，与旧 features/builder 的同名实现类区分。
    """

    def build_daily(
        self,
        date: TradeDate,
        groups: Sequence[str],
        store: DataStore,
    ) -> None:
        """构建指定日的特征分区。

        防冲突：重复调用 ⇒ 内容指纹一致才放行（同 append_features 语义）
        副作用：调用 store.append_features
        依赖：normalized 数据 + factors/ 计算函数
        """
        ...

    def backfill(
        self,
        start_date: TradeDate,
        end_date: TradeDate,
        groups: Sequence[str],
        store: DataStore,
    ) -> None:
        """历史回填（冷热分层后写冷区；按日分区幂等可重入，重跑即续传）。

        副作用：批量写分区
        """
        ...


# TODO(后续单元)：signal.py（Signal / DecisionMaker 等 §2~§3 协议）、
# core.py（Executor / Ledger / PortfolioMerger §3~§4）、hosts.py（§5）、
# evidence.py（HypothesisLedger / RegimeResampler §6）协议定义，本单元不实现。
