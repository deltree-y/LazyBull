# -*- coding: utf-8 -*-
"""信号层协议（契约 docs/contracts/protocols.md §2 的代码化）。

§2.1 Signal（信号生成，袖子核心接口，零互 import 保证）；
§2.2 Trainer / WalkForwardOrchestrator（训练与 WF 编排分离，R4-H2）。

签名与语义以协议文本为唯一依据；docstring 保留协议的行为语义注释
（幂等 / 防冲突 / 副作用 / 失败语义，对应 §8 行为语义矩阵）。
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from src.lazybull.v2.common.protocols.store import DataStore
from src.lazybull.v2.common.types import (
    ExecutionContext,
    ModelConfig,
    PanelFrame,
    RankedCandidates,
    TradeDate,
)

__all__ = ["Signal", "Trainer", "WalkForwardOrchestrator"]


@runtime_checkable
class Signal(Protocol):
    """信号生成（袖子核心接口，零互 import 保证）。"""

    @property
    def sleeve_id(self) -> str:
        """袖子唯一 ID（注册表登记）。"""
        ...

    @property
    def holding_period_days(self) -> int:
        """信号持有期（决策层用于到期判定）。

        MVP 语义：固定值（袖子 A = 20）。
        扩展位（R4-M5）：方案 §3.4「持有期由信号衰减速度决定」的动态语义
        （信号生命周期）留给未来 holding_period_for(ts_code, date) -> int 方法，
        MVP 不实现。
        """
        ...

    def generate_ranked(
        self,
        date: TradeDate,
        features: PanelFrame,
        context: ExecutionContext,
    ) -> RankedCandidates:
        """生成当日候选排序。

        幂等：是（同输入 ⇒ 同输出，模型推理确定性）
        副作用：无
        失败：特征缺失 ⇒ ValueError
        """
        ...


@runtime_checkable
class Trainer(Protocol):
    """训练引擎（只管训练与注册；WF 编排不在此协议）。

    集成语义（写死）：train_fold 的注册产物可以是集成——
    注册表把集成视为单一版本（metadata 记子模型列表），
    禁止「一个版本号 secret 指向集成」的类型撒谎，也禁止退化为单模型
    （改变生产行为）。
    """

    def train_fold(
        self,
        config: ModelConfig,  # ensemble_seeds 非空即集成
        train_start: TradeDate,
        train_end: TradeDate,
        val_start: TradeDate,
        val_end: TradeDate,
        store: DataStore,
    ) -> str:
        """训练单折（可为集成）。

        返回：模型版本号（ModelRegistry 注册；集成 = 单一注册版本 + 子模型 metadata）
        副作用：写模型文件 + 注册表

        F10 注记（P3）：四显式日期 ↔ 旧字段映射——``train_start/train_end`` = 实际
        训练窗，``val_start/val_end`` = val-raw 窗（embargo/ES/calibration 尾部拆分前）；
        ES/calibration/embargo 尾部拆分留在 train 侧内部执行，不经协议参数；
        **显式日期换算职责在编排器，train 侧不再二次比例切分**；训练统计
        （train_samples / best_iteration / val_*）从注册表 metadata 读取（既有旁路），
        本方法仍只返回版本号、不改签名语义（S3）。
        """
        ...


@runtime_checkable
class WalkForwardOrchestrator(Protocol):
    """WF 编排器（训练 → OOS 评估 → OOS 回测 → data_state 快照 → summary）。

    独立于 Trainer 的理由（R4-H2）：现实 walk_forward 是编排体
    （ml/walk_forward/runner.py），P3 验收对象（分级验收同 P2a）需要
    评估 / 回测联动才有意义；Trainer 保持纯训练语义，编排归本协议
    （或宿主实现）。
    """

    def run(
        self,
        config: ModelConfig,
        split_count: int,
        final_date: TradeDate,
        trainer: Trainer,
        store: DataStore,
        *,
        test_window_months: int = 6,
        selected_split_indices: Sequence[int] | None = None,
        skip_training: bool = False,
        start_model_version: int | None = None,
        skip_training_eval: bool = False,
        deploy_train: bool = False,
    ) -> str:
        """运行滚动训练全流程。

        返回：wf_run_id（批次标识，产物落 runs/<wf_run_id>/）

        F10 编排参数（P3 增补，宿主编排模式开关，不进 ModelConfig）：
        ``test_window_months``（折窗宽，生产 = 6）/ ``selected_split_indices`` /
        ``skip_training`` + ``start_model_version``（skip 必须带起始版本，实现内校验）/
        ``skip_training_eval`` / ``deploy_train``（**P3 回放默认 False**——Q10-B 已裁决：
        显式关闭 deploy 训练并登记理由，生产部署训练留 P4 验证）。

        Q3-C 产物边界注记：summary/reporting 沿用现格式与路径、回测批次落 runs；
        四键关联 = wf_run_id ↔ 回测批 ID ↔ 模型版本 ↔ data_state 文件，
        实现侧必须支持「从 wf_run_id 定位完整批次证据」。
        """
        ...
