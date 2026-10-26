# -*- coding: utf-8 -*-
"""袖子规格值对象（P3-T4，规划 §18.2；T4 评审处置 v2.18 修订）。

``SleeveSpec`` = 袖子规格的代码化形态：列集声明（候选超集语义）+ 标签定义 +
模型配置 + 排序与后处理配置 + 信号实现引用。规格为**纯数据 + 纯函数**，
不承载因子计算（架构 §4.3：因子唯一计算归属地 = ``factors/``）。

依赖方向（protocols §7）：sleeves → common.types / common.protocols / store（只读）；
**sleeves 不 import core / train / hosts / universe / factors**——域解析
（``universe/domains.py`` 单一来源）与信号类解析（``signal_impl`` 字符串）
均为调用方（薄入口 / 宿主）职责。

**规格完整性纪律（T4 评审 reviewer1-P2-1）**：规格是「生产配置代码化单一来源」，
四映射（``label_spec`` / ``feature_recipe`` / ``split_spec`` / ``hyperparams``）
键集必须与 ``v2/train/trainer.py::params_from_config`` 读取面**精确一致**
（缺一 / 多一即 ``ValueError``——缺键静默回落旧 CLI 默认值会产出与声明不一致的
训练配置且不可发现；默认值与生产值差异巨大，见规划 §18.11 处置表）。
注意分层：协议层 ``ModelConfig`` 的 None / 空映射三态合法（F10/F02 可选字段语义）
与本层完整性要求不矛盾——规格层是生产声明，协议层是通用载体。

**``spec_version`` 递增规则（reviewer1-P2-2/P3-10）**：规格语义任何变更（列集 /
信号参数 / 配方取值 / 键集表面）必须递增 ``spec_version``（同 dividend/consensus
哨兵版本纪律）；开启任一运行时派生族须同步扩展候选超集声明并递增版本；
``column_groups.PANEL_GROUPS`` 变更须同步重取 parquet 序并递增版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.lazybull.v2.common.types import ModelConfig, _freeze_plain_mapping
from src.lazybull.v2.store.column_groups import LABEL_VALUE_COLUMNS

__all__ = [
    "SleeveSpec",
    "LABEL_SPEC_KEYS",
    "FEATURE_RECIPE_KEYS",
    "SPLIT_SPEC_KEYS",
    "HYPERPARAM_KEYS",
]

#: 分区键列（与 v2/train/data_adapter._KEY_COLUMNS 同值；train 不可引用，本地声明，
#: 同值由测试跨模块锁定）
_KEY_COLUMNS = ("trade_date", "ts_code")

#: 规格四映射键集（= params_from_config 读取面 + 三 spec 的 version；
#: 精确匹配：缺一 / 多一即拒。测试经 AST 提取 trainer.py 读取键锁定，防双侧漂移）
LABEL_SPEC_KEYS: frozenset = frozenset(
    {
        "version",
        "task",
        "label_transform",
        "neutral_label_blend_weight",
        "winsorize_p",
        "pos_quantile",
        "pos_topk",
        "scale_pos_weight",
    }
)
SPLIT_SPEC_KEYS: frozenset = frozenset({"version", "val_ratio"})
FEATURE_RECIPE_KEYS: frozenset = frozenset(
    {
        "version",
        "enable_fundamental_features",
        "enable_alt_features",
        "enable_margin_features",
        "enable_cyq_features",
        "enable_fund_features",
        "enable_express_features",
        "enable_enhanced_features",
        "enable_north_features",
        "enable_lhb_features",
        "enable_consensus_features",
        "enable_cashflow_quality_features",
        "enable_consensus_revision_features",
        "enable_dividend_policy_features",
        "enable_availability_markers",
        "enable_holdertrade_features",
        "holdertrade_feature_set",
        "enable_repurchase_features",
        "repurchase_feature_set",
        "enable_top10fh_features",
        "top10fh_feature_set",
        "enable_top_inst_features",
        "feature_stability_filter",
        "factor_prune",
        "factor_exclude_file",
        "max_feature_missing_ratio",
    }
)
HYPERPARAM_KEYS: frozenset = frozenset(
    {
        "algorithm",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "colsample_bytree",
        "subsample",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "objective",
        "early_stopping_rounds",
        "early_stopping_metric",
        "min_best_iteration",
        "random_state",
        "rank_weight_enabled",
        "rank_weight_topk",
        "rank_weight",
        "rank_weight_topk_weight_mode",
        "time_decay_half_life",
        "freshness_strategy",
        "event_freshness_half_life_days",
        "ensemble_offsets",
        "ensemble_seed_keep_top_ratio",
        "ensemble_seed_keep_min_models",
    }
)

#: 键集校验遍历表（名称 → 期望键集）
_SPEC_KEY_SETS = (
    ("label_spec", LABEL_SPEC_KEYS),
    ("feature_recipe", FEATURE_RECIPE_KEYS),
    ("split_spec", SPLIT_SPEC_KEYS),
    ("hyperparams", HYPERPARAM_KEYS),
)


def _is_strict_int(value: Any) -> bool:
    """严格整数（拒绝 bool——bool 是 int 子类但语义上不是版本号/计数）。"""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SleeveSpec:
    """袖子规格（不可变；映射字段构造时递归冻结，与 ModelConfig F10/F02 同款纪律）。

    字段语义（P3 规划 §18.2 逐字段来源表）：

    - ``label_spec`` / ``feature_recipe`` / ``split_spec``：版本化规格映射
      （``version`` 键必须为非零正整数；canonical JSON 由注册表 metadata /
      配置指纹侧消费）；
    - ``feature_columns``：候选超集声明（**数据面（panel / cs_train 冻结段）
      可提供列的超集**，panel 模式 ``FeatureQuery.columns``；recipe 决定训练
      实际列集，不裁剪候选超集；逐折实际列集只写模型 metadata）；
    - ``domain``：域标识（解析为显式股票池 / 市值边界的职责在宿主，经
      ``universe/domains.py`` 单一来源；本对象不复制域市值，防双源漂移）；
    - ``signal_impl``：信号实现类的限定名字符串（sleeves 不 import core，
      由调用方 importlib 解析）；
    - ``holding_period_days``：协议 §2.1 MVP 语义（袖子 A = 20，决策层到期判定用）；
      现役到期判定唯一来源仍为 ``trading/sell_rules.py``（不动）。
    """

    sleeve_id: str
    spec_version: int
    model_family: str
    label_name: str
    label_spec: Mapping[str, Any]
    feature_recipe: Mapping[str, Any]
    split_spec: Mapping[str, Any]
    hyperparams: Mapping[str, Any]
    ensemble_seeds: tuple[int, ...]
    train_window_months: int
    domain: str
    feature_columns: tuple[str, ...]
    top_n: int
    holding_period_days: int
    min_amount_ma20: float
    exclude_financial: bool
    signal_impl: str

    def __post_init__(self) -> None:
        for name in ("label_spec", "feature_recipe", "split_spec", "hyperparams"):
            object.__setattr__(self, name, _freeze_plain_mapping(getattr(self, name)))
        _normalize_sequences(self)
        _validate_identity_fields(self)
        _validate_naming_fields(self)
        _validate_specs_versioned(self)
        _validate_spec_key_sets(self)
        _validate_feature_columns(self.feature_columns)
        _validate_training_fields(self)
        _validate_signal_fields(self)

    def build_model_config(self) -> ModelConfig:
        """规格 → 训练协议 ``ModelConfig``（§13.3.1 映射表；``val_window_months``
        比例模式置 None，日期换算职责在编排器）。

        注（reviewer1-P2-2）：``sleeve_id`` / ``spec_version`` 不映射进
        ``ModelConfig``（协议无此字段）；身份入产物（模型 metadata / runs）归
        T5 薄入口 / 宿主组装承接（规划 §18.11 处置表登记）。
        """
        return ModelConfig(
            model_family=self.model_family,
            feature_columns=tuple(self.feature_columns),
            label_name=self.label_name,
            hyperparams=self.hyperparams,
            train_window_months=self.train_window_months,
            val_window_months=None,
            ensemble_seeds=tuple(self.ensemble_seeds),
            label_spec=self.label_spec,
            feature_recipe=self.feature_recipe,
            split_spec=self.split_spec,
            domain=self.domain,
        )


def _normalize_sequences(spec: SleeveSpec) -> None:
    """序列字段归一化为 tuple（P2-4：list 入参不得保留可变容器；str 显式拒绝，
    防逐字符拆分为单字符序列）。"""
    for name in ("ensemble_seeds", "feature_columns"):
        value = getattr(spec, name)
        if value is None:
            raise ValueError(f"{name} 不得为 None")
        if isinstance(value, str):
            raise ValueError(f"{name} 必须为序列（拒绝 str，防逐字符拆分）: {value!r}")
        object.__setattr__(spec, name, tuple(value))


def _validate_identity_fields(spec: SleeveSpec) -> None:
    """标识字段校验（``__post_init__`` 提取件）。"""
    if not isinstance(spec.sleeve_id, str) or not spec.sleeve_id.strip():
        raise ValueError(f"sleeve_id 必须为非空字符串: {spec.sleeve_id!r}")
    if spec.sleeve_id != spec.sleeve_id.strip():
        raise ValueError(
            f"sleeve_id 不得含前后空白（注册键与查询口径一致，P2-4）: {spec.sleeve_id!r}"
        )
    if not _is_strict_int(spec.spec_version) or spec.spec_version < 1:
        raise ValueError(f"spec_version 必须为 >= 1 的整数: {spec.spec_version!r}")
    if not isinstance(spec.domain, str) or not spec.domain.strip():
        raise ValueError(f"domain 必须为非空字符串: {spec.domain!r}")


def _validate_naming_fields(spec: SleeveSpec) -> None:
    """命名字段校验（标签名 / 家族目录名）。"""
    if not isinstance(spec.label_name, str) or not spec.label_name.strip():
        raise ValueError(f"label_name 必须为非空字符串: {spec.label_name!r}")
    if not isinstance(spec.model_family, str) or not spec.model_family.strip():
        raise ValueError(f"model_family 必须为非空字符串: {spec.model_family!r}")


def _validate_specs_versioned(spec: SleeveSpec) -> None:
    """三 spec 版本门（F02 版本化前移至规格层；version 必须为 >= 1 的严格整数）。"""
    for name in ("label_spec", "feature_recipe", "split_spec"):
        mapping = getattr(spec, name) or {}
        version = mapping.get("version")
        if not _is_strict_int(version) or version < 1:
            raise ValueError(f"{name} 的 version 必须为 >= 1 的整数（F10 版本化纪律）: {version!r}")


def _validate_spec_key_sets(spec: SleeveSpec) -> None:
    """键集完整性（reviewer1-P2-1）：与 params_from_config 读取面精确一致，
    缺一 / 多一即拒（缺键静默回落旧 CLI 默认值 = 不可发现的配置漂移）。"""
    for name, expected in _SPEC_KEY_SETS:
        keys = set(getattr(spec, name) or {})
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            raise ValueError(
                f"{name} 键集与规格表面不一致: 缺 {missing} / 多 {extra}"
                "（规格 = 生产配置代码化单一来源，表面变动须显式修订并递增 spec_version）"
            )


def _validate_feature_columns(feature_columns: tuple[str, ...]) -> None:
    """候选超集声明校验：非空、元素为 str、唯一、不含键列/标签列。"""
    if not feature_columns:
        raise ValueError("feature_columns 候选超集为空（至少一列）")
    if not all(isinstance(c, str) for c in feature_columns):
        raise ValueError("feature_columns 元素必须全部为字符串列名")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns 存在重复列")
    bad = [c for c in feature_columns if c in _KEY_COLUMNS or c in LABEL_VALUE_COLUMNS]
    if bad:
        raise ValueError(f"feature_columns 不得含键列 {sorted(_KEY_COLUMNS)} 或标签列 {bad}")


def _validate_training_fields(spec: SleeveSpec) -> None:
    """训练侧字段校验（集成种子 / 窗口月数）。"""
    if not spec.ensemble_seeds or not all(_is_strict_int(seed) for seed in spec.ensemble_seeds):
        raise ValueError(f"ensemble_seeds 必须为非空整数序列: {spec.ensemble_seeds!r}")
    if not _is_strict_int(spec.train_window_months) or spec.train_window_months < 1:
        raise ValueError(f"train_window_months 必须为 >= 1 的整数: {spec.train_window_months!r}")


def _validate_signal_fields(spec: SleeveSpec) -> None:
    """信号侧字段校验（实现引用 / 排序与后处理参数）。"""
    if not isinstance(spec.signal_impl, str) or "." not in spec.signal_impl:
        raise ValueError(f"signal_impl 必须为限定名字符串（含模块路径）: {spec.signal_impl!r}")
    if not _is_strict_int(spec.top_n) or spec.top_n < 1:
        raise ValueError(f"top_n 必须为 >= 1 的整数: {spec.top_n!r}")
    if not _is_strict_int(spec.holding_period_days) or spec.holding_period_days < 1:
        raise ValueError(f"holding_period_days 必须为 >= 1 的整数: {spec.holding_period_days!r}")
    if isinstance(spec.min_amount_ma20, bool) or not isinstance(spec.min_amount_ma20, (int, float)):
        raise ValueError(
            f"min_amount_ma20 必须为数值类型（拒绝 bool/None）: {spec.min_amount_ma20!r}"
        )
    if float(spec.min_amount_ma20) < 0:
        raise ValueError(f"min_amount_ma20 必须 >= 0（千元）: {spec.min_amount_ma20!r}")
    if not isinstance(spec.exclude_financial, bool):
        raise ValueError(f"exclude_financial 必须为 bool: {spec.exclude_financial!r}")
