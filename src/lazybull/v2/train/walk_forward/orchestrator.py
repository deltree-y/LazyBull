# -*- coding: utf-8 -*-
"""Walk-forward 编排器（v2 侧，``WalkForwardOrchestrator`` 协议实现）。

P3-T3 搬运改写自 src.lazybull.ml.walk_forward.runner（2026-10-22），改动点（§17.1-B）：

1. **协议形态**:`run(config, split_count, final_date, trainer, store, **F10 编排参数)
   -> wf_run_id`;旧 ``run_walk_forward(args)`` 的 args namespace 拆分为
   ``OrchestratorHostConfig``(宿主配置)+ ``ModelConfig``(训练协议)+ 编排参数;
2. **数据根与输出根分离**(旧 :303-305/:511/:686 三处同根派生 ⇒ ``data_root`` 只读 +
   ``output_root`` 写侧;§7.3 隔离运行根);
3. **域解析剥离**(M4):``domain_market_whitelist`` / ``resolve_stock_domain`` /
   四族查询表构建 / MLSignal 构造全部上移宿主(``OrchestratorHostConfig`` 注入),
   train 包不 import universe / core / hosts / factors;
4. **OOS 回测依赖倒置**:``host.backtest_runner`` 注入(= v2/hosts/backtest/replay.
   run_oos_backtest;kwargs 经本包 ``backtest.build_oos_backtest_kwargs`` 薄组装);
5. **退役开关显式报错**:``reject_retired_switches``——exposure_* / policy_* /
   downside_penalty* 任一非 None ⇒ ValueError(禁止静默忽略,§2.2 cli 行);
   旧 runner 的 exposure 段(:280-299)与 ``write_policy_lambda_series`` /
   ``build_online_policy_provider`` **删除不迁**(退役家族,§13.3.1 ④⑤);
6. **异常形态**:库内不再 ``sys.exit``(旧 :365/:388/:477/:485/:720)——改
   ValueError/RuntimeError 上抛(差异登记;入口退出码归薄入口职责);
   split 级失败 continue / 回测失败记 error 置 {} 的行为语义保持;
7. **数据态增强**(§17.5-B):panel 侧逐读取点累积 ``store.last_read_receipt``
   (读后核验已在 receipt 内闭环)+ 非 store 输入(cs_train_verify 分区 /
   trade_cal / stock_basic / clean/daily 回测窗)运行前按日域枚举现算、
   运行后 ``verify_consumed_file_identities`` 复核(漂移 ⇒ 失败);
   产物 = 增强口径 ``data_state_{wf_run_id}.json``(summary 列 ``data_state_id``
   同步增强口径,§5-4 新旧不可比);
8. **runs 契约批次**(Q3-C):``host.runs_batch_writer`` 注入(宿主适配
   ``write_runs_batch``),收尾携带 wf_run_id + results + data_state——
   四键关联(wf_run_id ↔ 批次 ID ↔ 模型版本 ↔ data_state 文件)可定位;
9. ``run_walk_forward``(旧 497 行,在册 func_lines/mccabe 57/nesting 三条)按
   8 阶段段边界拆分;``_load_skip_training_metadata``(旧 mccabe 15 在册)拆分。
"""

import re
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.types import ModelConfig, ReplayContractError
from src.lazybull.v2.store.store_identity import InputIdentityDriftError
from src.lazybull.v2.train.data_adapter import (
    ACCEPTANCE_PURPOSES,
    SOURCE_CS_TRAIN_VERIFY,
    SOURCE_PANEL,
)
from src.lazybull.v2.train.trainer import V2Trainer, params_from_config, validate_sleeve_identity

from .backtest import BacktestHostParams, build_oos_backtest_kwargs, merge_backtest_metrics
from .data_state import (
    collect_data_state,
    collect_environment_identity,
    compute_enhanced_data_state_id,
    verify_consumed_file_identities,
)
from .deploy_training import execute_deploy_training
from .identity_collect import (
    acquire_non_store_identities,
    collect_identity_extra_sections,
    dedup_identity_records,
)
from .reporting import (
    chain_nav_splits,
    write_walk_forward_holdings_snapshot,
    write_walk_forward_topk_details,
    write_walk_forward_trade_details,
)
from .skip_meta import _load_skip_training_metadata
from .split_training import execute_skip_training_evaluation, execute_split_training
from .summary import build_summary_args_view, write_walk_forward_summary
from .training_core import TrainFoldParams, _build_main_board_codes
from .utils import (
    WalkForwardSplit,
    generate_walk_forward_splits_by_count,
    print_splits_summary,
    resolve_deploy_train_window,
)

__all__ = [
    "OrchestratorHostConfig",
    "V2WalkForwardOrchestrator",
    "reject_retired_switches",
    "_filter_splits_by_selected_indices",
    "_normalize_selected_split_indices",
]

#: 退役开关清单(§13.3.1 ④⑤:exposure 政策层 + downside_penalty 家族,已终结)
_RETIRED_SWITCH_KEYS = (
    "exposure_table",
    "exposure_policy",
    "exposure_replenish",
    "exposure_trim_tolerance",
    "exposure_budget_discount_replenish",
    "policy_model_root",
    "policy_arm_suffix",
    "policy_coverage_start",
    "downside_penalty",
    "downside_penalty_column",
)


def reject_retired_switches(**kwargs: Any) -> None:
    """退役开关显式拒绝(§2.2 cli 行口径:显式报错拒绝启动,禁止静默忽略)。

    薄入口/宿主组装配置时强制调用;任一退役键为非空/非零值 ⇒ ValueError。
    """
    supplied = {key: value for key, value in kwargs.items() if value not in (None, 0, 0.0, False)}
    if supplied:
        raise ValueError(
            "以下开关所属家族已退役(exposure 政策层 / downside_penalty),"
            f"v2 训练链路显式拒绝: {sorted(supplied)}"
            "(如需复核请走旧生产链路;P3 验收链路不提供这些机制)"
        )


@dataclass(frozen=True)
class OrchestratorHostConfig:
    """宿主配置(§13.3.1 ④:回测/输出追踪参数不进训练协议,由宿主承载)。

    - ``data_root`` 只读输入根 / ``output_root`` 写侧输出根(隔离运行根,§7.3);
    - ``domain_markets``:域市场白名单(宿主经 ``universe/domains.py`` 单一来源
      解析注入;默认主板与生产一致);
    - ``backtest_runner``:v2/hosts/backtest/replay.run_oos_backtest(宿主注入,
      依赖倒置);``signal_factory``:跨折持久化信号构造(宿主构造,含域市值边界);
    - ``runs_batch_writer``:runs 契约批次写入适配器(宿主注入;None ⇒ 不写 runs);
    - ``sleeve_identity``:``(sleeve_id, spec_version)`` 袖子身份(P3-T5,§19.6-①;
      非 None 时经 ``runs_batch_writer`` 的 ``extra_meta_notes`` kwarg additive 进
      批次 meta,适配器须支持该可选 kwarg;默认 None ⇒ 调用形态与产物逐位不变);
    """

    data_root: str
    output_root: str
    backtest: BacktestHostParams = field(default_factory=BacktestHostParams)
    domain_markets: tuple = ("主板",)
    run_log_csv: Optional[str] = None  # None ⇒ output_root/models/stock_selection/ml_train_runs.csv
    wf_summary_csv: Optional[str] = None  # None ⇒ output_root/walk_forward/raw/<模板>
    export_topk_details: bool = True
    batch_run_id: Optional[str] = None
    batch_period_label: Optional[str] = None
    oos_detail_metrics: bool = False
    backtest_runner: Optional[Callable[..., Dict[str, Any]]] = None
    signal_factory: Optional[Callable[[int], Any]] = None
    runs_batch_writer: Optional[Callable[..., Optional[Path]]] = None
    # P2-7:域标识由宿主同时承载(与 domain_markets 同源解析自 universe/domains.py);
    # 编排器 run 期断言 params.stock_domain == host.domain,防「训练域≠交易域」
    domain: str = "main"
    # P2-3/R08:评估侧族 DI 注入(宿主从 factors 组装;五族全关 ⇒ None)
    family_output_cols: Optional[Dict[str, List[str]]] = None
    availability_ensurer: Optional[Callable[..., List[str]]] = None
    # P3-T5(§19.6-①):袖子身份(与 trainer.sleeve_identity 同源双注入;
    # run 期一致性守卫——同时非 None 且不等 ⇒ ValueError;R2-T5A-10:run 期解析值
    # 经 trainer.bind_sleeve_identity 同步,host-only 携带 ⇒ 模型 metadata 同值)
    sleeve_identity: Optional[tuple[str, int]] = None

    def __post_init__(self) -> None:
        # P3-T5:袖子身份校验(与 trainer 同一实现,宿主配置构造期 fail-fast)
        validate_sleeve_identity(self.sleeve_identity)

    @property
    def resolved_run_log_csv(self) -> str:
        """台账路径(隔离根派生;显式提供时以显式为准)。"""
        if self.run_log_csv:
            return self.run_log_csv
        return str(Path(self.output_root) / "models" / "stock_selection" / "ml_train_runs.csv")


def _normalize_selected_split_indices(raw_indices: Optional[Sequence[int]]) -> List[int]:
    """规范化 split 下标列表：去重保序，且要求非负整数。

    P3-T3(§17.2/P2-8):归一化逻辑搬自旧 ``ml/walk_forward/cli.py``(归 train
    config 层);旧文件不动(生产在用),薄入口经本函数解析。
    """
    if not raw_indices:
        return []

    normalized: List[int] = []
    seen = set()
    for raw_index in raw_indices:
        split_index = int(raw_index)
        if split_index < 0:
            raise ValueError(f"selected_split_indices 仅支持非负整数，收到: {split_index}")
        if split_index in seen:
            continue
        seen.add(split_index)
        normalized.append(split_index)
    return normalized


def _filter_splits_by_selected_indices(
    splits: List[WalkForwardSplit],
    selected_split_indices: Optional[Sequence[int]],
) -> List[WalkForwardSplit]:
    """按指定 split 下标过滤切分;为空时返回原列表。"""
    if not selected_split_indices:
        return splits

    existing_indices = {split.split_index for split in splits}
    missing_indices = [index for index in selected_split_indices if index not in existing_indices]
    if missing_indices:
        raise ValueError(f"selected_split_indices 包含不存在的 split 下标: {missing_indices}")

    selected_index_set = set(selected_split_indices)
    return [split for split in splits if split.split_index in selected_index_set]


class V2WalkForwardOrchestrator:
    """WalkForwardOrchestrator 协议实现(训练 → OOS 评估 → OOS 回测 → data_state → summary)。

    构造注入(宿主/薄入口职责):
    - ``host``:OrchestratorHostConfig(宿主配置);
    - ``registry``:v2.common.model_registry.ModelRegistry(与 trainer 同一实例,
      宿主保证);``source_mode`` = panel(默认)/ cs_train_verify(验收例外);
    - cs_train_verify 模式必须传 ``storage`` / ``loader`` / ``acceptance_purpose``;
      panel 模式 ``storage``/``loader`` 仍用于行情/日历/基础表读取(接线表 #4,
      F14 同口径改期 P4);
    - ``runtime_lookups``:cs_train_verify 四族数据查询表(宿主经 factors 构建后
      注入;panel 模式必须 None——物化直读)。
    """

    def __init__(
        self,
        host: OrchestratorHostConfig,
        *,
        registry: ModelRegistry,
        source_mode: str = SOURCE_PANEL,
        acceptance_purpose: Optional[str] = None,
        storage: Any = None,
        loader: Any = None,
        runtime_derivers: Any = None,
        runtime_lookups: Optional[Dict[str, Any]] = None,
        frozen_through: Optional[str] = None,
    ) -> None:
        if source_mode not in (SOURCE_PANEL, SOURCE_CS_TRAIN_VERIFY):
            raise ValueError(f"未知数据源模式: {source_mode!r}")
        if source_mode == SOURCE_CS_TRAIN_VERIFY and acceptance_purpose not in ACCEPTANCE_PURPOSES:
            raise ValueError(
                f"cs_train 验证源模式必须声明验收用途({sorted(ACCEPTANCE_PURPOSES)}): "
                f"{acceptance_purpose!r}"
            )
        if source_mode == SOURCE_PANEL and runtime_lookups:
            raise ValueError("panel 模式四族物化直读,runtime_lookups 必须为空(防双写)")
        self._host = host
        self._registry = registry
        self._source_mode = source_mode
        self._acceptance_purpose = acceptance_purpose
        self._storage = storage
        self._loader = loader
        self._runtime_derivers = runtime_derivers
        self._runtime_lookups = runtime_lookups
        # P2-4:OOS 回测窗冻结上界(回放纪律;None ⇒ LABEL_FILTERED_THROUGH)
        from src.lazybull.v2.store.row_domain import LABEL_FILTERED_THROUGH

        self._frozen_through = frozen_through or LABEL_FILTERED_THROUGH
        self._store_receipts: List[Any] = []  # 废弃占位(P1-1 后由 store.consumed_files 取代)
        self._last_store: Any = None  # 本次 run 的 store(panel 模式;数据态收尾消费)
        self._consumed_extra_records: List[Dict[str, Any]] = []  # 非 store 输入运行前枚举
        self._wf_start_date: Optional[str] = None  # run 内推导(_resolve_splits 写入)
        self._wf_end_date: Optional[str] = None
        self._sleeve_identity: Optional[tuple[str, int]] = None  # run 守卫解析(P3-T5)

    # ------------------------------------------------------------------
    # 协议入口
    # ------------------------------------------------------------------

    def run(
        self,
        config: ModelConfig,
        split_count: int,
        final_date: str,
        trainer: V2Trainer,
        store: Any,
        *,
        test_window_months: int = 6,
        selected_split_indices: Optional[Sequence[int]] = None,
        skip_training: bool = False,
        start_model_version: Optional[int] = None,
        skip_training_eval: bool = False,
        deploy_train: bool = False,
    ) -> str:
        """运行滚动训练全流程(镜像旧 runner 8 阶段),返回 wf_run_id。

        Q10-B:``deploy_train`` 默认 False;P3 验收入口显式关闭并登记理由,
        生产部署训练留 P4 验证。
        """
        params = params_from_config(config)
        oos_backtest_months = self._resolve_oos_months(test_window_months)

        self._validate_run_prerequisites(params, trainer, store)

        stock_basic, main_board_codes, oos_backtest = self._open_run(
            config, params, split_count, final_date, test_window_months, oos_backtest_months
        )
        (
            wf_run_id,
            trade_cal,
            splits,
            deploy_train_start,
            deploy_train_end,
        ) = self._prepare_run_context(
            config,
            params,
            split_count,
            final_date,
            test_window_months,
            selected_split_indices,
            skip_training,
            start_model_version,
            deploy_train,
            oos_backtest_months,
        )

        results = self._run_splits(
            splits,
            wf_run_id,
            config,
            params,
            trainer,
            store,
            trade_cal,
            stock_basic,
            main_board_codes,
            skip_training,
            start_model_version,
            skip_training_eval,
            oos_backtest_months,
            oos_backtest,
        )

        self._maybe_deploy(
            deploy_train,
            skip_training,
            results,
            deploy_train_end,
            wf_run_id,
            config,
            params,
            trade_cal,
            main_board_codes,
            store,
        )

        trainer.frame_cache.clear()  # run 收尾兜底释放(P1-2/R02)

        self._write_outputs(
            results,
            wf_run_id,
            config,
            params,
            split_count,
            final_date,
            test_window_months,
            selected_split_indices,
            skip_training,
            skip_training_eval,
            start_model_version,
            deploy_train,
            oos_backtest_months,
        )
        return wf_run_id

    def _validate_run_prerequisites(
        self,
        params: TrainFoldParams,
        trainer: V2Trainer,
        store: Any,
    ) -> None:
        """run 前置校验与 store 会话复位（域/袖子身份/panel 族/信号工厂）。

        纯提取式拆分（复杂度门：run 255 行 > 硬上限 150，行为逐字不变）：
        ① 训练域与宿主域一致；② 袖子身份守卫并同步 trainer；
        ③ panel 模式五族开关拒绝；④ OOS 回测必须注入信号工厂；
        ⑤ store 会话级消费累积清零。
        """
        # P2-7:训练域与交易域一致性(宿主双注入同源,run 期断言)
        if params.stock_domain != self._host.domain:
            raise ValueError(
                f"训练域(params.stock_domain={params.stock_domain!r})与宿主域"
                f"(host.domain={self._host.domain!r})不一致——domain_markets 必须由宿主"
                "自同一 domain 字符串解析(universe/domains.py 单一来源)"
            )
        # P3-T5(§19.6-①):袖子身份守卫 + R2-T5A-10 贯通(解析值训练前同步 trainer;
        # host-only ⇒ 模型 metadata 与 runs batch_meta 两出口同值;异值 trainer 侧仍拒)
        self._sleeve_identity = self._resolve_sleeve_identity(trainer)
        if self._sleeve_identity is not None:
            bind = getattr(trainer, "bind_sleeve_identity", None)
            if bind is None:
                raise TypeError(
                    "宿主携带 sleeve_identity，但 trainer 未实现 bind_sleeve_identity"
                    "——袖子身份无法同步到模型 metadata（§19.6-①/R2-T5A-10）"
                )
            bind(self._sleeve_identity)
        # P2-2:panel 模式五族开关本期拒绝(物化直读接线未完成;P3 回放五族全关;
        # cs_train_verify 模式走同源 lookup 派生兼容)
        if self._source_mode == SOURCE_PANEL:
            panel_family_switches = (
                "enable_holdertrade_features",
                "enable_repurchase_features",
                "enable_top10fh_features",
                "enable_top_inst_features",
                "enable_availability_markers",
            )
            enabled = [name for name in panel_family_switches if getattr(params, name)]
            if enabled:
                raise ValueError(
                    f"panel 模式本期不支持运行时派生族开关(物化直读接线未完成,"
                    f"P3 回放五族全关;如需启用另行裁决): {enabled}"
                )

        # P3-T3 评审处置(P1-3 ≡ R03):缺省信号工厂会从只读 data_root 解析模型根,
        # 与隔离 output_root 注册错位 ⇒ 启用 OOS 回测时必须注入经校验的信号工厂
        if self._host.backtest.oos_backtest and self._host.signal_factory is None:
            raise ValueError(
                "oos_backtest 启用必须注入 signal_factory(宿主构造,绑定隔离输出根模型目录;"
                "缺省路径会从只读 data_root 解析同号模型,P3-T3 评审 P1-3/R03 阻断)"
            )

        # 数据态会话复位(P3-T3 评审处置 P1-1/P3-4):store 会话级消费累积清零,
        # 同实例复跑不混入上一 run 回执
        if store is not None and hasattr(store, "reset_read_receipts"):
            store.reset_read_receipts()
        self._store_receipts = []
        self._last_store = store

    def _prepare_run_context(
        self,
        config: ModelConfig,
        params: TrainFoldParams,
        split_count: int,
        final_date: str,
        test_window_months: int,
        selected_split_indices: Optional[Sequence[int]],
        skip_training: bool,
        start_model_version: Optional[int],
        deploy_train: bool,
        oos_backtest_months: int,
    ) -> tuple[str, pd.DataFrame, List[WalkForwardSplit], Optional[str], Optional[str]]:
        """运行上下文准备（运行 ID/日历/切分/部署窗/输入身份采集）。

        纯提取式拆分（复杂度门：run 255 行 > 硬上限 150，行为逐字不变）。
        """
        wf_run_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Walk-forward 运行ID: {wf_run_id}")

        trade_cal = self._load_trade_cal()
        splits, _ = self._resolve_splits(
            config,
            params,
            trade_cal,
            split_count,
            final_date,
            test_window_months,
            selected_split_indices,
        )

        if skip_training and start_model_version is None:
            raise ValueError("--skip-training 模式必须指定 --start-model-version(F10 校验)")
        self._assert_runtime_lookups(params)

        deploy_train_start, deploy_train_end = self._resolve_deploy_window(
            deploy_train, skip_training, config, trade_cal, splits, final_date
        )
        print_splits_summary(
            splits,
            deploy_train_start=deploy_train_start,
            deploy_train_end=deploy_train_end,
        )

        # §17.5-B:非 store 输入运行前枚举采集(cs_train_verify 分区按日域并集 +
        # trade_cal/stock_basic + clean/daily 回测窗);panel 分区由 receipt 机制闭环
        self._acquire_non_store_identities(splits, trade_cal, oos_backtest_months)
        return wf_run_id, trade_cal, splits, deploy_train_start, deploy_train_end

    def _resolve_sleeve_identity(self, trainer: V2Trainer) -> Optional[tuple[str, int]]:
        """袖子身份一致性守卫 + 有效值解析(P3-T5,§19.6-①)。

        host 与 trainer 同时非 None 且不等 ⇒ ValueError(构造/组装错位 fail-fast);
        仅一侧携带时以携带侧为准;双侧 None ⇒ None(现状零变化)。
        R2-T5A-10:返回值由 run() 经 ``trainer.bind_sleeve_identity`` 同步给 trainer,
        host-only 单侧携带时模型注册 metadata 同样携带袖子键(两出口一致)。
        """
        host_identity = self._host.sleeve_identity
        # 非 spec 驱动调用（测试桩/泛协议 trainer）未实现袖子身份属性 ⇒ 视为未携带，
        # 保持 T5 前的现状行为（§19.6-①「默认 None = 现状行为」）。
        trainer_identity = getattr(trainer, "sleeve_identity", None)
        if (
            host_identity is not None
            and trainer_identity is not None
            and host_identity != trainer_identity
        ):
            raise ValueError(
                f"宿主袖子身份(host.sleeve_identity={host_identity!r})与 trainer 袖子身份"
                f"(trainer.sleeve_identity={trainer_identity!r})不一致——两者必须由宿主"
                "自同一袖子规格解析注入(§19.6-①)"
            )
        return host_identity if host_identity is not None else trainer_identity

    # ------------------------------------------------------------------
    # 阶段 0~1:参数日志 + 组件与域股票池(旧 :226-334)
    # ------------------------------------------------------------------

    def _resolve_oos_months(self, test_window_months: int) -> int:
        """oos_backtest_months=0 ⇒ 对齐 test_window_months(旧 :254-255 语义,局部变量)。"""
        months = self._host.backtest.oos_backtest_months
        return test_window_months if months <= 0 else months

    def _open_run(
        self,
        config: ModelConfig,
        params: TrainFoldParams,
        split_count: int,
        final_date: str,
        test_window_months: int,
        oos_backtest_months: int,
    ) -> tuple:
        """参数日志 + stock_basic 加载 + 域股票池构建(域解析已在宿主完成,M4)。

        返回 (stock_basic, main_board_codes, oos_backtest_effective)——旧「stock_basic
        缺失 ⇒ args.oos_backtest 就地改写」语义改为运行态有效值返回(差异登记)。
        """
        bt = self._host.backtest
        logger.info("=" * 80)
        logger.info("Walk-forward 滚动训练(v2 编排)")
        logger.info("=" * 80)
        logger.info(f"切分数量: {split_count}")
        logger.info(f"最终日期: {final_date}")
        logger.info(f"训练窗口: {config.train_window_months} 个月")
        logger.info(f"测试窗口: {test_window_months} 个月")
        logger.info(f"标签列: {params.label_column}")
        logger.info(f"任务类型: {params.task}")
        logger.info(
            f"早停: rounds={params.early_stopping_rounds if params.early_stopping_rounds else '禁用'}, "
            f"metric={params.early_stopping_metric}, "
            f"min_best_iteration={params.min_best_iteration or '禁用'}"
        )
        logger.info(
            f"多种子筛选: top_ratio={params.ensemble_seed_keep_top_ratio:.0%}, "
            f"min_models={params.ensemble_seed_keep_min_models}"
        )
        logger.info(
            f"OOS 回测: {'启用' if bt.oos_backtest else '禁用'}(时长 {oos_backtest_months} 个月)"
        )
        logger.info(f"数据目录(只读): {self._host.data_root}")
        logger.info(f"输出目录(写侧): {self._host.output_root}")

        if self._loader is None:
            raise ValueError("编排器需要 loader(行情/日历/基础表读取,接线表 #4 口径)")

        oos_backtest = bt.oos_backtest
        # 加载股票基本信息(OOS 回测需要)
        stock_basic = None
        if oos_backtest:
            stock_basic = self._loader.load_clean_stock_basic()
            if stock_basic is None:
                stock_basic = self._loader.load_stock_basic()
            if stock_basic is None:
                logger.warning("无法加载股票基本信息,OOS 回测将被禁用")
                oos_backtest = False
        # 训练/评估统一使用域股票池(默认主板 = 生产现状),保证与交易口径一致
        if stock_basic is None:
            stock_basic = self._loader.load_clean_stock_basic()
        if stock_basic is None:
            stock_basic = self._loader.load_stock_basic()
        if stock_basic is None:
            raise ValueError("无法加载股票基本信息,无法执行主板过滤训练")
        main_board_codes = _build_main_board_codes(
            stock_basic, markets=tuple(self._host.domain_markets)
        )
        logger.info(
            f"域股票池加载完成(markets={list(self._host.domain_markets)}): "
            f"{len(main_board_codes)} 只"
        )
        return stock_basic, main_board_codes, oos_backtest

    # ------------------------------------------------------------------
    # 阶段 2:切分生成与筛选(旧 :336-381)
    # ------------------------------------------------------------------

    def _load_trade_cal(self) -> pd.DataFrame:
        trade_cal = self._loader.load_clean_trade_cal()
        if trade_cal is None:
            trade_cal = self._loader.load_trade_cal()
        if trade_cal is None:
            raise ValueError("无法加载交易日历(clean/raw 均缺失)")
        return trade_cal

    def _resolve_splits(
        self,
        config: ModelConfig,
        params: TrainFoldParams,
        trade_cal: pd.DataFrame,
        split_count: int,
        final_date: str,
        test_window_months: int,
        selected_split_indices: Optional[Sequence[int]],
    ) -> tuple:
        """切分生成 + 调仓频率推断 + 下标筛选(逐字语义;空切分 ⇒ ValueError)。"""
        # 推断调仓频率(与 run_oos_backtest 内逻辑保持一致)
        if self._host.backtest.bt_rebalance_freq is not None:
            rebalance_freq = self._host.backtest.bt_rebalance_freq
        else:
            match = re.search(r"(\d+)", params.label_column)
            rebalance_freq = int(match.group(1)) if match else 20

        splits = generate_walk_forward_splits_by_count(
            trade_cal=trade_cal,
            split_count=split_count,
            final_date=final_date,
            train_window_years=config.train_window_months // 12,
            test_window_months=test_window_months,
            rebalance_freq=rebalance_freq,
        )
        generated_split_count = len(splits)

        splits = _filter_splits_by_selected_indices(
            splits=splits,
            selected_split_indices=_normalize_selected_split_indices(selected_split_indices),
        )

        if len(splits) == 0:
            raise ValueError("未生成任何切分,请检查参数设置(旧 = sys.exit(1),差异登记)")

        if selected_split_indices:
            logger.info(
                f"按下标筛选 split: {list(selected_split_indices)},"
                f"保留 {len(splits)} / {generated_split_count} 个"
            )
        else:
            logger.info(f"未指定 split 下标,默认训练全部 {len(splits)} 个 split")

        # 推导出的 WF 覆盖区间(供汇总视图;旧为 args 就地回写 :376-377)
        self._wf_start_date = splits[0].train_start
        self._wf_end_date = splits[-1].test_end
        logger.info(
            f"推导区间: {self._wf_start_date} 至 {self._wf_end_date} "
            f"(由 split_count={split_count}, final_date={final_date} 反推)"
        )
        return splits, rebalance_freq

    # ------------------------------------------------------------------
    # 阶段 3:四族查询表一致性校验(宿主已构建注入;旧 :383-450 的构建职责上移)
    # ------------------------------------------------------------------

    def _assert_runtime_lookups(self, params: TrainFoldParams) -> None:
        """族启用 ⇒ 派生注入齐备性校验(**训练注册前**,复核处置 codex-F04)。

        cs_train_verify 模式:族启用需同源 lookup + deriver + 评估侧出列清单;
        availability 需 derivers.availability + availability_ensurer;缺一 ⇒ ValueError。
        panel 模式:构造期已拒绝非空 lookups(物化直读,五族开关 run 期另拒)。
        """
        if self._source_mode != SOURCE_CS_TRAIN_VERIFY:
            return
        required = {
            "enable_holdertrade_features": ("holdertrade_lookup", "holdertrade"),
            "enable_repurchase_features": ("repurchase_lookup", "repurchase"),
            "enable_top10fh_features": ("top10fh_panel", "top10fh"),
            "enable_top_inst_features": ("top_inst_lookup", "top_inst"),
        }
        lookups = self._runtime_lookups or {}
        output_cols = self._host.family_output_cols or {}
        for switch, (key, family) in required.items():
            if not getattr(params, switch):
                continue
            missing: List[str] = []
            if lookups.get(key) is None:
                missing.append(f"runtime_lookups[{key!r}]")
            if getattr(self._runtime_derivers, family, None) is None:
                missing.append(f"runtime_derivers.{family}")
            if not output_cols.get(family):
                missing.append(f"host.family_output_cols[{family!r}]")
            if missing:
                raise ValueError(
                    f"{switch}=True 的派生注入不齐备(训练注册前拒绝,codex-F04): "
                    f"缺 {missing}(v2/train 不 import factors,宿主从 factors 组装注入)"
                )
        if params.enable_availability_markers:
            missing = []
            if getattr(self._runtime_derivers, "availability", None) is None:
                missing.append("runtime_derivers.availability")
            if self._host.availability_ensurer is None:
                missing.append("host.availability_ensurer")
            if missing:
                raise ValueError(f"enable_availability_markers=True 的派生注入不齐备: 缺 {missing}")

    # ------------------------------------------------------------------
    # 阶段 4:deploy 窗口解析与冲突校验(旧 :452-491)
    # ------------------------------------------------------------------

    def _resolve_deploy_window(
        self,
        deploy_train: bool,
        skip_training: bool,
        config: ModelConfig,
        trade_cal: pd.DataFrame,
        splits: List[WalkForwardSplit],
        final_date: str,
    ) -> tuple:
        deploy_train_start = None
        deploy_train_end = None
        if not deploy_train or skip_training:
            return None, None
        train_window_years = config.train_window_months // 12
        deploy_train_start, deploy_train_end = resolve_deploy_train_window(
            trade_cal=trade_cal,
            deploy_train_end=final_date,
            train_window_years=train_window_years,
        )

        if deploy_train_start is None or deploy_train_end is None:
            logger.warning(f"部署训练区间解析失败,无法在切分汇总中展示(目标train_end={final_date})")
            return None, None

        last_split = splits[-1]
        if (
            deploy_train_start == last_split.train_start
            and deploy_train_end == last_split.train_end
        ):
            raise ValueError(
                "部署训练区间与最后一个 split 的训练区间完全重叠,"
                "请调整 split_count 或 final_date 后重试(旧 = sys.exit(1),差异登记)"
            )

        if deploy_train_end <= last_split.train_end:
            raise ValueError(
                f"部署训练结束日({deploy_train_end}) 不晚于"
                f"最后一个 split 训练结束日({last_split.train_end}),"
                "会造成训练区间冲突,请调整 final_date(旧 = sys.exit(1),差异登记)"
            )
        return deploy_train_start, deploy_train_end

    # ------------------------------------------------------------------
    # 数据态增强(§17.5-B):运行前枚举非 store 输入,运行后复核
    # ------------------------------------------------------------------

    def _acquire_non_store_identities(
        self,
        splits: List[WalkForwardSplit],
        trade_cal: pd.DataFrame,
        oos_backtest_months: int,
    ) -> None:
        """运行前枚举采集非 store 输入身份(薄委托 identity_collect,§17.8 拆分)。"""
        self._consumed_extra_records = acquire_non_store_identities(
            self._host.data_root,
            self._source_mode,
            splits,
            trade_cal,
            oos_backtest_months,
            SOURCE_CS_TRAIN_VERIFY,
        )

    def _finalize_data_state_identity(self) -> Dict[str, Any]:
        """运行后:合并 store 会话消费(receipt 累积 + readiness 收敛登记)与
        非 store 记录,复核漂移,返回增强段。

        消费清单口径(P3-T3 评审处置 P1-1/R01):panel 模式 =
        ``store.consumed_files``(load_features/load_labels 回执 + label_refresh
        收敛路径 ``register_external_read`` 登记,含训练窗/评估窗/回测窗的
        panel 分区与**全部 labels 分区**)+ 非 store 枚举(cs_train_verify 分区 /
        trade_cal / stock_basic / clean/daily / **raw/suspend** 回测窗)。
        """
        store_files: List[Dict[str, Any]] = []
        if self._source_mode == SOURCE_PANEL and self._last_store is not None:
            store_files = list(self._last_store.consumed_files)
        consumed = dedup_identity_records(store_files + list(self._consumed_extra_records))
        verify_consumed_file_identities(consumed, base_dir=self._host.data_root)
        environment = collect_environment_identity()
        return {
            "input_files": consumed,
            "environment_identity": environment,
            "extra_sections": collect_identity_extra_sections(
                self._host.data_root, self._last_store
            ),
        }

    # ------------------------------------------------------------------
    # 阶段 5:逐 split 循环(旧 :493-648)
    # ------------------------------------------------------------------

    def _run_splits(
        self,
        splits: List[WalkForwardSplit],
        wf_run_id: str,
        config: ModelConfig,
        params: TrainFoldParams,
        trainer: V2Trainer,
        store: Any,
        trade_cal: pd.DataFrame,
        stock_basic: pd.DataFrame,
        main_board_codes: set,
        skip_training: bool,
        start_model_version: Optional[int],
        skip_training_eval: bool,
        oos_backtest_months: int,
        oos_backtest: bool,
    ) -> List[Dict[str, Any]]:
        """逐折训练/评估/回测(split 级失败 continue、回测失败置 {} 语义保持)。"""
        results: List[Dict[str, Any]] = []
        topk_values = sorted({20, 30, 100, 300})

        # 跨 split 持久化 signal(宿主构造;门控历史缓冲跨 split 累积预热)
        persistent_signal = None
        if oos_backtest and self._host.signal_factory is not None:
            persistent_signal = self._host.signal_factory(self._host.backtest.bt_top_n)
            logger.info(f"持久化信号已创建(宿主注入),将跨 {len(splits)} 个 split 复用")

        for split in splits:
            try:
                result = self._run_one_split(
                    split,
                    wf_run_id,
                    config,
                    params,
                    trainer,
                    store,
                    trade_cal,
                    stock_basic,
                    main_board_codes,
                    topk_values,
                    skip_training,
                    start_model_version,
                    skip_training_eval,
                    oos_backtest,
                    oos_backtest_months,
                    persistent_signal,
                )
                results.append(result)
            except (InputIdentityDriftError, ReplayContractError):
                raise  # 评审处置(P2-1 + 复核 P1-1):输入漂移与守卫契约失败必须穿透
            except Exception as exc:
                logger.error(f"Split {split.split_index} 训练失败: {exc}")
                logger.error(traceback.format_exc())
                logger.warning("继续执行下一个 split...")
                continue
            finally:
                # P3-T3 评审处置(P1-2/R02):按折释放帧缓存(一折一清,
                # 覆盖正常/失败/skip-eval 全路径;同折 3 种子共享不受影响)
                trainer.frame_cache.clear()
        return results

    def _run_one_split(
        self,
        split: WalkForwardSplit,
        wf_run_id: str,
        config: ModelConfig,
        params: TrainFoldParams,
        trainer: V2Trainer,
        store: Any,
        trade_cal: pd.DataFrame,
        stock_basic: pd.DataFrame,
        main_board_codes: set,
        topk_values: List[int],
        skip_training: bool,
        start_model_version: Optional[int],
        skip_training_eval: bool,
        oos_backtest: bool,
        oos_backtest_months: int,
        persistent_signal: Any,
    ) -> Dict[str, Any]:
        """单折执行(训练/评估 + OOS 回测;旧 :527-648 循环体段)。"""
        if skip_training:
            model_version = start_model_version + split.split_index
            skip_metadata = _load_skip_training_metadata(self._registry, model_version, params)
            logger.info(
                f"[跳过训练] Split {split.split_index}: "
                f"使用已有模型 v{model_version},"
                f"测试区间 {split.test_start} ~ {split.test_end}"
            )
            if skip_training_eval:
                result = execute_skip_training_evaluation(
                    split=split,
                    wf_run_id=wf_run_id,
                    registry=self._registry,
                    eval_frame_provider=trainer.make_frame_provider(config, params, store),
                    params=params,
                    main_board_codes=main_board_codes,
                    topk_values=topk_values,
                    model_version=model_version,
                    runtime_derivers=self._runtime_derivers,
                    family_output_cols=self._host.family_output_cols,
                    availability_ensurer=self._host.availability_ensurer,
                )
            else:
                result = {
                    "split_index": split.split_index,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    "model_version": model_version,
                    "feature_columns": (skip_metadata or {}).get("feature_columns") or [],
                    "bt_metrics": {},
                }
        else:
            result = execute_split_training(
                split=split,
                wf_run_id=wf_run_id,
                trainer=trainer,
                config=config,
                store=store,
                params=params,
                main_board_codes=main_board_codes,
                topk_values=topk_values,
                run_log_csv=self._host.resolved_run_log_csv,
                eval_frame_provider=trainer.make_frame_provider(config, params, store),
                trade_cal=trade_cal,
                oos_detail_metrics=self._host.oos_detail_metrics,
                runtime_derivers=self._runtime_derivers,
                family_output_cols=self._host.family_output_cols,
                availability_ensurer=self._host.availability_ensurer,
            )
        # OOS 回测(每个 split 训练后运行真实回测)
        if oos_backtest and result.get("model_version"):
            result = self._run_split_backtest(
                split,
                result,
                trade_cal,
                stock_basic,
                params,
                oos_backtest_months,
                persistent_signal,
                store,
            )
        return result

    def _run_split_backtest(
        self,
        split: WalkForwardSplit,
        result: Dict[str, Any],
        trade_cal: pd.DataFrame,
        stock_basic: pd.DataFrame,
        params: TrainFoldParams,
        oos_backtest_months: int,
        persistent_signal: Any,
        store: Any,
    ) -> Dict[str, Any]:
        """单折 OOS 回测(旧 :576-641 段;失败记 error 置 {} 语义保持)。"""
        if self._host.backtest_runner is None:
            raise ValueError("oos_backtest 启用但宿主未注入 backtest_runner")
        try:
            bt_start = split.test_start
            bt_end_dt = datetime.strptime(bt_start, "%Y%m%d") + relativedelta(
                months=oos_backtest_months
            )
            bt_end = bt_end_dt.strftime("%Y%m%d")
            # P2-4:OOS 回测窗冻结上界夹紧(回放纪律;跨缝另行预登记)
            if bt_end > self._frozen_through:
                raise ReplayContractError(
                    f"OOS 回测窗 {bt_start}~{bt_end} 超出冻结上界 {self._frozen_through}"
                    "(回放限冻结段;跨缝须另行预登记,§4 表第 6 行)"
                )
            kwargs = build_oos_backtest_kwargs(
                self._host.backtest,
                model_version=result["model_version"],
                bt_start=bt_start,
                bt_end=bt_end,
                storage=self._storage,
                loader=self._loader,
                trade_cal=trade_cal,
                stock_basic=stock_basic,
                label_column=params.label_column,
                data_root=self._host.data_root,
                persistent_signal=persistent_signal,
                split_index=split.split_index,
                stock_domain=params.stock_domain,
                runtime_lookups=self._runtime_lookups,
                feature_source=(
                    "cs_train_verify" if self._source_mode == SOURCE_CS_TRAIN_VERIFY else "panel"
                ),
                store=store,
                acceptance_purpose=self._acceptance_purpose,
                # P2-4:panel 列集收窄 = 本折模型实际特征列(skip 模式取 metadata 列)
                feature_columns=result.get("feature_columns") or None,
            )
            bt_metrics = self._host.backtest_runner(**kwargs)
            merge_backtest_metrics(result, bt_metrics)
        except (InputIdentityDriftError, ReplayContractError):
            raise  # 复核处置(reviewer1 复核 P1-1):守卫契约失败穿透,不得吞成 bt_metrics={}
        except Exception as exc:
            logger.error(f"Split {split.split_index} OOS回测失败: {exc}")
            logger.error(traceback.format_exc())
            result["bt_metrics"] = {}
        return result

    # ------------------------------------------------------------------
    # 阶段 6:deploy 训练(旧 :650-678;Q10-B 默认关闭)
    # ------------------------------------------------------------------

    def _maybe_deploy(
        self,
        deploy_train: bool,
        skip_training: bool,
        results: List[Dict[str, Any]],
        deploy_train_end: Optional[str],
        wf_run_id: str,
        config: ModelConfig,
        params: TrainFoldParams,
        trade_cal: pd.DataFrame,
        main_board_codes: set,
        store: Any,
    ) -> None:
        if not deploy_train or skip_training or len(results) == 0:
            return
        if deploy_train_end is None:
            logger.error("部署训练区间未成功解析,跳过部署模型训练")
            return
        logger.info("=" * 80)
        logger.info("开始部署模型训练(使用最新可用数据)")
        logger.info(f"  部署模型 train_end: {deploy_train_end}(由 final_date 对齐)")
        logger.info("=" * 80)
        try:
            deploy_result = execute_deploy_training(
                deploy_train_end=deploy_train_end,
                wf_run_id=wf_run_id,
                frame_provider=self._deploy_frame_provider(config, params, store),
                registry=self._registry,
                params=params,
                train_window_years=config.train_window_months // 12,
                main_board_codes=main_board_codes,
                topk_values=sorted({20, 30, 100, 300}),
                trade_cal=trade_cal,
                run_log_csv=self._host.resolved_run_log_csv,
                runtime_derivers=self._runtime_derivers,
            )
            if deploy_result:
                logger.info(f"部署模型已注册: v{deploy_result['model_version']}")
        except Exception as exc:
            logger.error(f"部署模型训练失败: {exc}")
            logger.error(traceback.format_exc())

    def _deploy_frame_provider(
        self, config: ModelConfig, params: TrainFoldParams, store: Any
    ) -> Callable[[str, str], Any]:
        """deploy 帧供给(与训练同适配路径;比例路径无显式缝)。"""
        from src.lazybull.v2.train.data_adapter import AdapterConfig, load_training_frame

        runtime_families = frozenset(
            name
            for name, switch_attr in (
                ("holdertrade", "enable_holdertrade_features"),
                ("repurchase", "enable_repurchase_features"),
                ("top10fh", "enable_top10fh_features"),
                ("availability_markers", "enable_availability_markers"),
            )
            if getattr(params, switch_attr)
        )

        is_panel = self._source_mode == SOURCE_PANEL

        def provider(win_start: str, win_end: str):
            return load_training_frame(
                AdapterConfig(
                    start_date=win_start,
                    end_date=win_end,
                    # P3-5 处置:cs_train_verify 模式传空(该模式返回全 schema,
                    # 适配层禁止传列收窄);panel 模式传候选超集
                    feature_columns=tuple(config.feature_columns) if is_panel else (),
                    label_column=config.label_name,
                    source_mode=self._source_mode,
                    runtime_families=runtime_families,
                    runtime_lookups=self._runtime_lookups,
                ),
                store=store,
                storage=self._storage,
                loader=self._loader,
                acceptance_purpose=self._acceptance_purpose,
            )

        return provider

    # ------------------------------------------------------------------
    # 阶段 7~8:汇总落盘 + 数据态增强 + runs 批次(旧 :680-715)
    # ------------------------------------------------------------------

    def _write_outputs(
        self,
        results: List[Dict[str, Any]],
        wf_run_id: str,
        config: ModelConfig,
        params: TrainFoldParams,
        split_count: int,
        final_date: str,
        test_window_months: int,
        selected_split_indices: Optional[Sequence[int]],
        skip_training: bool,
        skip_training_eval: bool,
        start_model_version: Optional[int],
        deploy_train: bool,
        oos_backtest_months: int,
    ) -> None:
        """summary/reporting 落盘(现格式现路径、根隔离,Q3-C)+ 数据态增强 + runs 批次。"""
        if len(results) == 0:
            logger.warning("没有成功完成的训练,跳过生成汇总文件")
            return

        if self._host.wf_summary_csv:
            summary_csv_path = self._host.wf_summary_csv
        else:
            summary_csv_path = str(
                Path(self._host.output_root)
                / "walk_forward"
                / "raw"
                / f"walk_forward_summary_{wf_run_id}.csv"
            )

        # 数据态增强(§5):运行后复核 + extras 经视图通道进 summary 一次落盘
        # (增强 id 列 + 增强 JSON 同口径;旧键集合不变,additive 段附加)
        extras = self._finalize_data_state_identity()

        summary_args = build_summary_args_view(
            config,
            params,
            self._host,
            split_count=split_count,
            final_date=final_date,
            test_window_months=test_window_months,
            selected_split_indices=selected_split_indices,
            skip_training=skip_training,
            skip_training_eval=skip_training_eval,
            start_model_version=start_model_version,
            deploy_train=deploy_train,
            oos_backtest_months=oos_backtest_months,
            wf_start_date=self._wf_start_date,
            wf_end_date=self._wf_end_date,
            data_state_extras={
                "consumed_files": extras["input_files"],
                "environment": extras["environment_identity"],
                "extra_sections": extras.get("extra_sections"),
            },
        )
        write_walk_forward_summary(results, summary_csv_path, summary_args, wf_run_id)

        if self._host.export_topk_details:
            write_walk_forward_topk_details(results, summary_csv_path, wf_run_id)

        write_walk_forward_trade_details(results, summary_csv_path, wf_run_id)

        # 政策旁路 P2-1:逐日持仓快照(中文表头,供 terminal_loss 政策层离线打分)
        write_walk_forward_holdings_snapshot(results, summary_csv_path, wf_run_id)

        # ── 串联各 split 的 OOS 回测净值曲线 ──────────────────
        chain_nav_splits(results, summary_csv_path, wf_run_id)

        # runs 契约批次(宿主注入 writer;四键关联 Q3-C)
        if self._host.runs_batch_writer is not None:
            state = collect_data_state(
                data_root=self._host.data_root,
                wf_run_id=wf_run_id,
                batch_run_id=self._host.batch_run_id,
            )
            # P3-T5(§19.6-①):袖子身份经 extra_meta_notes additive 进批次 meta;
            # None ⇒ 不传该 kwarg,调用形态与产物与现状逐位一致
            writer_extra_kwargs: Dict[str, Any] = {}
            if self._sleeve_identity is not None:
                writer_extra_kwargs["extra_meta_notes"] = {
                    "sleeve_id": self._sleeve_identity[0],
                    "spec_version": str(self._sleeve_identity[1]),
                }
            self._host.runs_batch_writer(
                results=results,
                wf_run_id=wf_run_id,
                data_state_payload={
                    **state,
                    "data_state_id": compute_enhanced_data_state_id(
                        state,
                        extras["input_files"],
                        extras["environment_identity"],
                        extras.get("extra_sections"),
                    ),
                },
                **writer_extra_kwargs,
            )

        logger.info("=" * 80)
        logger.info("Walk-forward 滚动训练完成!")
        logger.info(f"  成功完成: {len(results)} 个切分")
        logger.info(f"  运行ID: {wf_run_id}")
        logger.info("=" * 80)
