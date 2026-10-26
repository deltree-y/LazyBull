# -*- coding: utf-8 -*-
"""v2 Trainer 协议实现（P3-T3，``v2/common/protocols/signal.py`` §2.2）。

职责 = 训练 + 注册（协议：副作用 = 写模型文件 + 注册表；集成 = 单一注册版本 +
子模型 metadata）。WF 编排（评估 / 回测 / 台账）不在此层（R4-H2）。

设计要点（§17.6）：

- ``ModelConfig`` → ``TrainFoldParams`` 映射（``params_from_config``，逐字段口径
  按 §13.3.1 映射表；缺省字段取 TrainFoldParams 默认 = 旧 cli 默认）；
- 训练帧经适配层 + ``TrainingFrameCache`` 加载（同折 3 种子共享一帧）；
  域过滤执行点保持旧 ``_filter_to_main_board``（帧加载后立即执行；适配层
  不承担训练侧主板过滤，``universe`` 语义归 T4 袖子规格装配）；
- 显式日期缝（F10 D1）：``train_fold`` 四日期由编排器换算（
  ``resolve_explicit_split_dates``），透传 prepare，train 侧不再二次比例切分；
- 显式缝 × 多偏移窗口（ensemble_offsets > 0）组合本期不支持 ⇒ ValueError
  （生产 offsets=0；登记备审）；
- **差异登记**（§17.9-13）：注册 ``performance_metrics.test_daily`` 置空 dict——
  OOS 评估在 ``train_fold`` 返回后由编排器执行（协议职责分离），评估结果经
  summary / 台账落盘（Q3-C 产物等价），不再回写模型 metadata；
- ``last_fold_result`` 只读属性：最近一折的内存候选（模型对象 / 特征列 /
  val_daily_metrics / ensemble_meta 等），供编排器 OOS 评估复用内存对象
  （与旧流程逐位一致，避免注册表回读的序列化往返）。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.types import ModelConfig, to_plain_data
from src.lazybull.v2.store.row_domain import LABEL_FILTERED_THROUGH
from src.lazybull.v2.store.store_identity import canonical_json_sha256
from src.lazybull.v2.train.core import (
    attach_cashflow_quality_train_params,
    attach_cons_revision_schema_version,
    attach_dividend_train_params,
    evaluate_validation_daily,
)
from src.lazybull.v2.train.core.runtime_derive import RuntimeDerivers
from src.lazybull.v2.train.data_adapter import (
    ACCEPTANCE_PURPOSES,
    SOURCE_CS_TRAIN_VERIFY,
    SOURCE_PANEL,
    AdapterConfig,
    TrainingFrame,
)
from src.lazybull.v2.train.ensemble import EnsembleModel
from src.lazybull.v2.train.frame_cache import TrainingFrameCache
from src.lazybull.v2.train.walk_forward.training_core import (
    TrainFoldParams,
    _build_ensemble_sub_models,
    _build_feature_flag_train_params,
    _resolve_ensemble_seeds,
    _train_model_on_window,
)

__all__ = ["V2Trainer"]

#: 适配层运行时派生四族开关 →（族名, params 开关属性, lookups 键）（§13.3.1 M6-b）
_RUNTIME_FAMILY_SWITCHES = (
    ("holdertrade", "enable_holdertrade_features", "holdertrade_lookup"),
    ("repurchase", "enable_repurchase_features", "repurchase_lookup"),
    ("top10fh", "enable_top10fh_features", "top10fh_panel"),
)


def params_from_config(config: ModelConfig) -> TrainFoldParams:
    """ModelConfig → TrainFoldParams 逐字段映射（§13.3.1 映射表）。

    缺省语义：label_spec / feature_recipe / split_spec / hyperparams 缺键 ⇒
    ``TrainFoldParams`` 字段默认值（= 旧 cli.py 默认值逐项一致）。
    """
    hp = dict(config.hyperparams)
    ls = dict(config.label_spec or {})
    fr = dict(config.feature_recipe or {})
    ss = dict(config.split_spec or {})
    return TrainFoldParams(
        label_column=config.label_name,
        neutral_label_blend_weight=float(ls.get("neutral_label_blend_weight", 0.0)),
        task=str(ls.get("task", "regression")),
        label_transform=str(ls.get("label_transform", "raw")),
        winsorize_p=float(ls.get("winsorize_p", 0.01)),
        pos_quantile=ls.get("pos_quantile"),
        pos_topk=ls.get("pos_topk"),
        scale_pos_weight=ls.get("scale_pos_weight"),
        val_ratio=float(ss.get("val_ratio", 0.1)),
        enable_fundamental_features=bool(fr.get("enable_fundamental_features", False)),
        enable_alt_features=bool(fr.get("enable_alt_features", False)),
        enable_margin_features=bool(fr.get("enable_margin_features", False)),
        enable_cyq_features=bool(fr.get("enable_cyq_features", False)),
        enable_fund_features=bool(fr.get("enable_fund_features", False)),
        enable_express_features=bool(fr.get("enable_express_features", False)),
        enable_enhanced_features=bool(fr.get("enable_enhanced_features", False)),
        enable_north_features=bool(fr.get("enable_north_features", False)),
        enable_lhb_features=bool(fr.get("enable_lhb_features", False)),
        enable_consensus_features=bool(fr.get("enable_consensus_features", False)),
        enable_cashflow_quality_features=bool(fr.get("enable_cashflow_quality_features", False)),
        enable_consensus_revision_features=bool(
            fr.get("enable_consensus_revision_features", False)
        ),
        enable_dividend_policy_features=bool(fr.get("enable_dividend_policy_features", False)),
        enable_availability_markers=bool(fr.get("enable_availability_markers", False)),
        enable_holdertrade_features=bool(fr.get("enable_holdertrade_features", False)),
        holdertrade_feature_set=str(fr.get("holdertrade_feature_set", "full")),
        enable_repurchase_features=bool(fr.get("enable_repurchase_features", False)),
        repurchase_feature_set=str(fr.get("repurchase_feature_set", "full")),
        enable_top10fh_features=bool(fr.get("enable_top10fh_features", False)),
        top10fh_feature_set=str(fr.get("top10fh_feature_set", "full")),
        enable_top_inst_features=bool(fr.get("enable_top_inst_features", False)),
        feature_stability_filter=bool(fr.get("feature_stability_filter", False)),
        factor_prune=bool(fr.get("factor_prune", False)),
        factor_exclude_file=fr.get("factor_exclude_file"),
        max_feature_missing_ratio=float(fr.get("max_feature_missing_ratio", 0.6)),
        freshness_strategy=str(hp.get("freshness_strategy", "state_keep_event_decay")),
        event_freshness_half_life_days=float(hp.get("event_freshness_half_life_days", 45.0)),
        rank_weight_enabled=bool(hp.get("rank_weight_enabled", True)),
        rank_weight_topk=int(hp.get("rank_weight_topk", 30)),
        rank_weight=float(hp.get("rank_weight", 5.0)),
        rank_weight_topk_weight_mode=str(hp.get("rank_weight_topk_weight_mode", "linear_decay")),
        time_decay_half_life=float(hp.get("time_decay_half_life", 0)),
        algorithm=str(hp.get("algorithm", "xgboost")),
        n_estimators=int(hp.get("n_estimators", 200)),
        max_depth=int(hp.get("max_depth", 5)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        subsample=float(hp.get("subsample", 0.8)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
        min_child_weight=hp.get("min_child_weight", 100),
        reg_alpha=float(hp.get("reg_alpha", 0.05)),
        reg_lambda=float(hp.get("reg_lambda", 1.0)),
        gamma=float(hp.get("gamma", 0.1)),
        random_state=int(hp.get("random_state", 42)),
        early_stopping_rounds=int(hp.get("early_stopping_rounds", 200)),
        early_stopping_metric=str(hp.get("early_stopping_metric", "rank_ic")),
        min_best_iteration=int(hp.get("min_best_iteration", 0)),
        objective=str(hp.get("objective", "mse")),
        ensemble_seeds=tuple(int(seed) for seed in config.ensemble_seeds) or None,
        ensemble_offsets=int(hp.get("ensemble_offsets", 0)),
        ensemble_seed_keep_top_ratio=float(hp.get("ensemble_seed_keep_top_ratio", 0.3)),
        ensemble_seed_keep_min_models=int(hp.get("ensemble_seed_keep_min_models", 3)),
        stock_domain=str(config.domain or "main"),
    )


def validate_sleeve_identity(
    value: Optional[tuple[str, int]],
) -> Optional[tuple[str, int]]:
    """``sleeve_identity`` 校验（P3-T5，§19.6-①；口径与 sleeves/spec.py 标识字段一致）。

    None（默认，现状零变化）或 ``(sleeve_id, spec_version)`` 二元组：
    ``sleeve_id`` 为非空且前后无空白的字符串；``spec_version`` 为 >= 1 的
    严格整数（拒绝 bool——bool 是 int 子类但语义上不是版本号）。
    """
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"sleeve_identity 必须为 (sleeve_id, spec_version) 二元组: {value!r}")
    sleeve_id, spec_version = value
    if not isinstance(sleeve_id, str) or not sleeve_id or sleeve_id.strip() != sleeve_id:
        raise ValueError(
            f"sleeve_identity 的 sleeve_id 必须为非空且前后无空白的字符串: {sleeve_id!r}"
        )
    if isinstance(spec_version, bool) or not isinstance(spec_version, int) or spec_version < 1:
        raise ValueError(f"sleeve_identity 的 spec_version 必须为 >= 1 的整数: {spec_version!r}")
    return value


@dataclass(frozen=True)
class _FoldArtifacts:
    """最近一折的内存候选（旧 _build_split_training_candidate 返回键 + 版本号）。"""

    model: Any
    feature_columns: List[str]
    train_params: Dict[str, Any]
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    val_daily_metrics: Dict[str, Any]
    df_val_split_original: pd.DataFrame
    data_stats: Dict[str, Any]
    train_days_count: int
    total_train_samples: int
    X_train_len: int
    X_val_len: int
    label_column: str
    ensemble_meta: Dict[str, Any]
    full_train_params: Dict[str, Any]  # 注册同款完整参数(台账用,与注册 metadata 同源)
    model_version: int


class V2Trainer:
    """Trainer 协议实现（训练 + 注册；WF 编排在 orchestrator）。

    构造注入（宿主/薄入口职责，§17.6）：
    - ``registry``:v2.common.model_registry.ModelRegistry（模型输出根隔离）；
    - ``main_board_codes``:显式域股票池（编排器经 universe/domains.py 单一来源
      解析注入；train 不 import universe，M4）；
    - ``source_mode`` = ``panel``（默认）/ ``cs_train_verify``（验收专用例外，
      须传 ``storage`` / ``loader`` / ``acceptance_purpose``，适配层白名单守卫）；
    - ``runtime_derivers`` / ``runtime_lookups``:四族派生函数集合与数据查询表
      （调用方从 factors 组装注入；生产/验收臂五族全关 ⇒ None）；
    - ``frame_cache``:None ⇒ 内置新实例（同折 3 种子共享；编排按折 clear）；
    - ``sleeve_identity``:``(sleeve_id, spec_version)`` 袖子身份（P3-T5，§19.6-①；
      宿主注入，注册 metadata additive 写入 ``sleeve_id`` / ``sleeve_spec_version``
      两键；默认 None ⇒ 不加键，现状逐位不变）；run 期编排器把解析后的有效身份
      经 ``bind_sleeve_identity`` 同步给 trainer（R2-T5A-10 贯通：host-only 单侧
      携带时模型 metadata 与 runs batch_meta 两出口同值）。
    """

    # P3-T5 兼容默认：单元测试/部分初始化对象经 ``__new__`` 构造时未跑 ``__init__``，
    # 仍应读到"未注入"语义（生产构造路径在 ``__init__`` 里显式赋值覆盖）。
    _sleeve_identity: Optional[tuple[str, int]] = None

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        main_board_codes: set,
        source_mode: str = SOURCE_PANEL,
        acceptance_purpose: Optional[str] = None,
        storage: Any = None,
        loader: Any = None,
        frame_cache: Optional[TrainingFrameCache] = None,
        runtime_derivers: Optional[RuntimeDerivers] = None,
        runtime_lookups: Optional[Dict[str, Any]] = None,
        frozen_through: str = LABEL_FILTERED_THROUGH,
        trade_cal: Optional[pd.DataFrame] = None,
        topk_values: Sequence[int] = (20, 30, 100, 300),
        sleeve_identity: Optional[tuple[str, int]] = None,
    ) -> None:
        if source_mode not in (SOURCE_PANEL, SOURCE_CS_TRAIN_VERIFY):
            raise ValueError(
                f"未知训练数据源模式: {source_mode!r}"
                f"（合法值 {SOURCE_PANEL!r} / {SOURCE_CS_TRAIN_VERIFY!r}）"
            )
        if source_mode == SOURCE_CS_TRAIN_VERIFY and acceptance_purpose not in ACCEPTANCE_PURPOSES:
            raise ValueError(
                f"cs_train 验证源模式必须声明验收用途（{sorted(ACCEPTANCE_PURPOSES)}）: "
                f"{acceptance_purpose!r}"
            )
        self._registry = registry
        self._main_board_codes = main_board_codes
        self._source_mode = source_mode
        self._acceptance_purpose = acceptance_purpose
        self._storage = storage
        self._loader = loader
        self._frame_cache = frame_cache if frame_cache is not None else TrainingFrameCache()
        self._runtime_derivers = runtime_derivers
        self._runtime_lookups = runtime_lookups
        self._frozen_through = frozen_through
        self._trade_cal = trade_cal
        self._topk_values = list(topk_values)
        self._sleeve_identity = validate_sleeve_identity(sleeve_identity)
        self._last_fold_result: Optional[_FoldArtifacts] = None

    @property
    def last_fold_result(self) -> Optional[_FoldArtifacts]:
        """最近一折内存候选（OOS 评估复用，与旧流程逐位一致）。"""
        return self._last_fold_result

    @property
    def frame_cache(self) -> TrainingFrameCache:
        return self._frame_cache

    @property
    def sleeve_identity(self) -> Optional[tuple[str, int]]:
        """袖子身份（P3-T5，§19.6-①；None = 未注入，注册 metadata 不写袖子键）。"""
        return self._sleeve_identity

    def bind_sleeve_identity(self, value: Optional[tuple[str, int]]) -> None:
        """编排器 run 期解析后的有效袖子身份写入点（R2-T5A-10 贯通修复）。

        校验同 ``validate_sleeve_identity``；None ⇒ 不写（现状零变化）；同值重复
        注入 ⇒ 幂等无操作；本 trainer 已携带身份且与注入值不等 ⇒ ValueError
        （双侧不等仍拒绝，与编排器 run 期守卫同口径）。用途：host-only 单侧携带
        时编排器在训练前把解析值同步进来，使模型 metadata（train_params 两键）
        与 runs ``batch_meta``（extra_meta_notes）两出口一致。
        """
        validated = validate_sleeve_identity(value)
        if validated is None:
            return
        if self._sleeve_identity is not None and self._sleeve_identity != validated:
            raise ValueError(
                f"trainer 已携带袖子身份 {self._sleeve_identity!r}，与编排器 run 期解析值 "
                f"{validated!r} 不一致——禁止覆盖（§19.6-① 双侧同源纪律）"
            )
        self._sleeve_identity = validated

    def make_frame_provider(
        self, config: ModelConfig, params: TrainFoldParams, store: Any
    ) -> Callable[[str, str], TrainingFrame]:
        """窗口 → 训练帧（适配层 + 帧缓存；四族开关 → runtime_families 集合推导）。

        cs_train_verify 模式 ``feature_columns`` 传空（该模式返回分区原始全
        schema,适配层禁止传列收窄,T1 口径);panel 模式传候选超集。
        """
        runtime_families = frozenset(
            name
            for name, switch_attr, _lookup_key in _RUNTIME_FAMILY_SWITCHES
            if getattr(params, switch_attr)
        ) | (
            frozenset({"availability_markers"})
            if params.enable_availability_markers
            else frozenset()
        )
        is_panel = self._source_mode == SOURCE_PANEL

        def provider(win_start: str, win_end: str) -> TrainingFrame:
            adapter_config = AdapterConfig(
                start_date=win_start,
                end_date=win_end,
                feature_columns=tuple(config.feature_columns) if is_panel else (),
                label_column=config.label_name,
                universe=None,
                source_mode=self._source_mode,
                frozen_through=self._frozen_through,
                runtime_families=runtime_families,
                runtime_lookups=self._runtime_lookups,
            )
            return self._frame_cache.get_or_load(
                adapter_config,
                store=store,
                storage=self._storage,
                loader=self._loader,
                acceptance_purpose=self._acceptance_purpose,
            )

        return provider

    def _train_fold_candidates(
        self,
        params: TrainFoldParams,
        provider: Callable[[str, str], TrainingFrame],
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        ensemble_seeds: List[int],
    ) -> tuple:
        """候选构建段(P3-T3 提取自 train_fold;集成/单模型分支逐字保持)。"""
        explicit_dates = (train_end, val_start)
        if len(ensemble_seeds) > 1:
            windows = [(train_start, val_end)]
            logger.info(
                f"base 集成训练: {len(windows)}个窗口 × {len(ensemble_seeds)}个种子 "
                f"= {len(windows) * len(ensemble_seeds)}个子模型"
                f"(偏移±{params.ensemble_offsets}个月, seeds={ensemble_seeds})"
            )
            (
                sub_models,
                base_result,
                ensemble_meta,
            ) = _build_ensemble_sub_models(
                windows,
                provider,
                params,
                self._main_board_codes,
                ensemble_seeds,
                runtime_derivers=self._runtime_derivers,
                topk_values=self._topk_values,
                explicit_dates=explicit_dates,
            )

            model = EnsembleModel(sub_models)
            validation_result = (
                ensemble_meta.pop("_ensemble_validation_result", None) or base_result
            )
            logger.info(f"base 集成模型创建完成: {model}")
            return (
                model,
                base_result["feature_columns"],
                base_result["train_params"],
                base_result["train_metrics"],
                base_result["val_metrics"],
                validation_result["df_val_split_original"],
                validation_result["feature_columns"],
                validation_result["label_column"],
                base_result["data_stats"],
                base_result["train_days_count"],
                base_result["total_train_samples"],
                base_result["X_train_len"],
                base_result["X_val_len"],
                ensemble_meta,
            )

        tr = _train_model_on_window(
            provider(train_start, val_end),
            params,
            self._main_board_codes,
            runtime_derivers=self._runtime_derivers,
            random_state_override=ensemble_seeds[0],
            explicit_train_end=explicit_dates[0],
            explicit_val_start=explicit_dates[1],
        )
        ensemble_meta = {
            "sub_model_best_iterations": [
                (ensemble_seeds[0], tr["train_params"].get("best_iteration"))
            ]
        }
        return (
            tr["model"],
            tr["feature_columns"],
            tr["train_params"],
            tr["train_metrics"],
            tr["val_metrics"],
            tr["df_val_split_original"],
            tr["feature_columns"],
            tr["label_column"],
            tr["data_stats"],
            tr["train_days_count"],
            tr["total_train_samples"],
            tr["X_train_len"],
            tr["X_val_len"],
            ensemble_meta,
        )

    def _register_fold(
        self,
        *,
        config: ModelConfig,
        model: Any,
        params: TrainFoldParams,
        feature_columns: List[str],
        train_params: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        train_start: str,
        val_end: str,
        X_train_len: int,
        X_val_len: int,
        ensemble_offsets: int,
        ensemble_seeds: List[int],
    ) -> tuple:
        """注册参数组装 + 注册段(P3-T3 提取自 train_fold,逐字;返回 (version, full_params))。"""
        full_train_params = train_params.copy()
        attach_cons_revision_schema_version(
            full_train_params,
            params.enable_consensus_revision_features,
        )
        attach_cashflow_quality_train_params(
            full_train_params,
            params.enable_cashflow_quality_features,
            feature_columns=feature_columns,
        )
        attach_dividend_train_params(
            full_train_params,
            params.enable_dividend_policy_features,
        )
        full_train_params.update(
            {
                "algorithm": params.algorithm,
                "task": params.task,
                "label_transform": (
                    params.label_transform if params.task == "regression" else None
                ),
                "winsorize_p": (
                    params.winsorize_p if params.label_transform == "cs_zscore" else None
                ),
                "pos_quantile": params.pos_quantile if params.task == "classification" else None,
                "pos_topk": params.pos_topk if params.task == "classification" else None,
                "scale_pos_weight_manual": params.scale_pos_weight is not None,
                **_build_feature_flag_train_params(params),
                "freshness_strategy": params.freshness_strategy,
                "event_freshness_half_life_days": params.event_freshness_half_life_days,
            }
        )
        # F10 接线(T3 评审处置 reviewer1-P1-4 ≡ codex-R05):三 spec 原文(dict 化,
        # MappingProxyType 快照转回可 JSON 序列化形态)+ canonical digest + domain +
        # 配方字段补齐(旧注册缺失,additive 增强,差异登记 §17.9-21)
        spec_sources = {
            "label_spec": config.label_spec,
            "feature_recipe": config.feature_recipe,
            "split_spec": config.split_spec,
        }
        for name, spec in spec_sources.items():
            # F10 版本化(复核处置 codex-F02):非空 spec 必须带 version 键;
            # 空映射 = F10 合法默认(复核二修正:空映射不拒)
            if spec and "version" not in spec:
                raise ValueError(
                    f"ModelConfig.{name} 缺版本号(F10:规格须带版本号 + canonical 序列化,"
                    "「同列集不同配方」须可审计;薄入口组装时给 version=1)"
                )
            # 递归转纯 dict 树(嵌套 MappingProxy ⇒ 可 JSON 序列化;F02)
            spec_dict = to_plain_data(spec) if spec is not None else None
            full_train_params[name] = spec_dict
            full_train_params[f"{name}_digest"] = (
                canonical_json_sha256(spec_dict) if spec_dict is not None else None
            )
        full_train_params["domain"] = params.stock_domain
        full_train_params.update(
            {
                "factor_prune": params.factor_prune,
                "factor_exclude_file": params.factor_exclude_file,
                "feature_stability_filter": params.feature_stability_filter,
                "max_feature_missing_ratio": params.max_feature_missing_ratio,
                "rank_weight_enabled": params.rank_weight_enabled,
                "rank_weight_topk": params.rank_weight_topk,
                "rank_weight": params.rank_weight,
                "rank_weight_topk_weight_mode": params.rank_weight_topk_weight_mode,
                "time_decay_half_life": params.time_decay_half_life,
            }
        )
        if isinstance(model, EnsembleModel):
            full_train_params["ensemble_offsets"] = ensemble_offsets
            full_train_params["ensemble_seeds"] = ensemble_seeds
            full_train_params["ensemble_seed_keep_top_ratio"] = params.ensemble_seed_keep_top_ratio
            full_train_params["ensemble_seed_keep_min_models"] = (
                params.ensemble_seed_keep_min_models
            )
            full_train_params["ensemble_n_models"] = model.n_models

        # P3-T5(§19.6-①):袖子身份 additive 键(None ⇒ 不加键,注册 metadata 与现状逐位一致)
        if self._sleeve_identity is not None:
            full_train_params["sleeve_id"] = self._sleeve_identity[0]
            full_train_params["sleeve_spec_version"] = self._sleeve_identity[1]

        version = self._registry.register_model(
            model=model,
            model_type=f"{params.algorithm}_{params.task}_wf",
            train_start_date=train_start,
            train_end_date=val_end,
            feature_columns=feature_columns,
            label_column=params.label_column,
            n_samples=X_train_len + X_val_len,
            train_params=full_train_params,
            performance_metrics=performance_metrics,
        )

        logger.info(f"模型已注册: v{version}")
        return version, full_train_params

    def train_fold(
        self,
        config: ModelConfig,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        store: Any,
    ) -> str:
        """训练单折（可为集成）并注册，返回模型版本号（字符串形态，协议签名）。

        四显式日期（F10）：加载窗 = [``train_start``, ``val_end``]（外层训练窗）；
        显式缝 = (``train_end``, ``val_start``) 透传 prepare（不再二次比例切分）。
        """
        params = params_from_config(config)
        ensemble_seeds = _resolve_ensemble_seeds(params)
        ensemble_offsets = params.ensemble_offsets
        if ensemble_offsets > 0:
            raise ValueError(
                "显式日期缝 × 多偏移窗口(ensemble_offsets>0)组合本期不支持"
                "(生产 offsets=0;如需启用须另行裁决,§17.6 登记)"
            )

        provider = self.make_frame_provider(config, params, store)

        (
            model,
            feature_columns,
            train_params,
            _train_metrics,
            _val_metrics,
            df_val_split_original,
            validation_feature_columns,
            validation_label_column,
            data_stats,
            train_days_count,
            total_train_samples,
            X_train_len,
            X_val_len,
            ensemble_meta,
        ) = self._train_fold_candidates(
            params, provider, train_start, train_end, val_start, val_end, ensemble_seeds
        )

        val_daily_metrics: Dict[str, Any] = {}
        if len(df_val_split_original) > 0:
            val_daily_metrics = evaluate_validation_daily(
                model=model,
                df_val=df_val_split_original,
                feature_columns=validation_feature_columns,
                original_return_col=validation_label_column,
                task=params.task,
                topk_values=self._topk_values,
            )

        performance_metrics = {
            "train": _train_metrics,
            "validation": _val_metrics,
            "validation_daily": val_daily_metrics,
            "test": {},
            # 差异登记(§17.9-13):OOS 评估在 train_fold 返回后由编排器执行,
            # test_daily 不再回写注册 metadata(旧 = 含值;评估证据经 summary/台账落盘)
            "test_daily": {},
        }

        version, full_train_params = self._register_fold(
            config=config,
            model=model,
            params=params,
            feature_columns=feature_columns,
            train_params=train_params,
            performance_metrics=performance_metrics,
            train_start=train_start,
            val_end=val_end,
            X_train_len=X_train_len,
            X_val_len=X_val_len,
            ensemble_offsets=ensemble_offsets,
            ensemble_seeds=ensemble_seeds,
        )

        self._last_fold_result = _FoldArtifacts(
            model=model,
            feature_columns=feature_columns,
            train_params=train_params,
            train_metrics=_train_metrics,
            val_metrics=_val_metrics,
            val_daily_metrics=val_daily_metrics,
            df_val_split_original=df_val_split_original,
            data_stats=data_stats,
            train_days_count=train_days_count,
            total_train_samples=total_train_samples,
            X_train_len=X_train_len,
            X_val_len=X_val_len,
            label_column=validation_label_column,
            ensemble_meta=ensemble_meta,
            full_train_params=full_train_params,
            model_version=version,
        )
        return str(version)
