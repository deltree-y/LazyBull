# -*- coding: utf-8 -*-
"""P3-T4：`v2/sleeves/value/` 规格化 + `_registry.py` 登记测试。

映射 T4 卡独立验收（规划 §18.5）：规格加载 / 未知袖子 fail-fast / 信号接线核验，
外加候选超集声明、spec → params_from_config 接线、依赖方向静态守卫、
注册文件唯一静态守卫。
T4 评审处置（v2.18）：键集完整性（reviewer1-P2-1）/ 构造缺口（P2-4）/
守卫候选展开（reviewer2-R2-T4-02，沿 test_v2_p2a_t6_equivalence.py `_import_candidates`
先例）/ 全序指纹锁定（R2-T4-S01）/ 注册表代理封闭（P3-1）等。
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Tuple

import pytest

from src.lazybull.v2.sleeves import get_sleeve_spec, list_sleeve_ids
from src.lazybull.v2.sleeves import spec as sleeves_spec_module
from src.lazybull.v2.sleeves._registry import get_sleeve_spec as registry_get
from src.lazybull.v2.sleeves.spec import (
    FEATURE_RECIPE_KEYS,
    HYPERPARAM_KEYS,
    LABEL_SPEC_KEYS,
    SPLIT_SPEC_KEYS,
    SleeveSpec,
)
from src.lazybull.v2.sleeves.value import VALUE_SLEEVE_SPEC
from src.lazybull.v2.sleeves.value.feature_columns import VALUE_FEATURE_COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SLEEVES_DIR = _REPO_ROOT / "src" / "lazybull" / "v2" / "sleeves"


# ---------------------------------------------------------------------------
# 1. 规格加载
# ---------------------------------------------------------------------------


class TestSpecLoading:
    def test_registry_returns_value_spec(self):
        spec = get_sleeve_spec("value")
        assert spec is VALUE_SLEEVE_SPEC
        assert registry_get("value") is VALUE_SLEEVE_SPEC
        assert spec.sleeve_id == "value"
        assert spec.spec_version == 1

    def test_build_model_config_fields(self):
        config = VALUE_SLEEVE_SPEC.build_model_config()
        assert config.model_family == "stock_selection"
        assert config.label_name == "neu_y_ret_20"
        assert config.feature_columns == VALUE_FEATURE_COLUMNS
        assert config.train_window_months == 72
        assert config.val_window_months is None  # 比例模式置 None（F10）
        assert config.ensemble_seeds == (42, 61, 82)
        assert config.domain == "main"
        # 三 spec 版本键（F10/F02 版本化纪律）
        assert config.label_spec["version"] == 1
        assert config.feature_recipe["version"] == 1
        assert config.split_spec["version"] == 1
        # 映射同值（经双重冻结，内容逐项一致）
        assert dict(config.hyperparams) == dict(VALUE_SLEEVE_SPEC.hyperparams)
        assert dict(config.label_spec) == dict(VALUE_SLEEVE_SPEC.label_spec)

    def test_mappings_immutable(self):
        """spec 与产物的映射均为 MappingProxyType 且拒绝写。"""
        for mapping in (
            VALUE_SLEEVE_SPEC.hyperparams,
            VALUE_SLEEVE_SPEC.label_spec,
            VALUE_SLEEVE_SPEC.feature_recipe,
            VALUE_SLEEVE_SPEC.split_spec,
        ):
            assert isinstance(mapping, MappingProxyType)
            with pytest.raises(TypeError):
                mapping["algorithm"] = "lgb"  # type: ignore[index]
        config = VALUE_SLEEVE_SPEC.build_model_config()
        assert isinstance(config.hyperparams, MappingProxyType)
        with pytest.raises(TypeError):
            config.split_spec["val_ratio"] = 0.1  # type: ignore[index]

    def test_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            VALUE_SLEEVE_SPEC.top_n = 5  # type: ignore[misc]

    def test_freeze_helper_identity_lock(self):
        """私有引用签名锁定（沿 N-3 先例）：冻结语义唯一来源 = v2.common.types。"""
        from src.lazybull.v2.common import types as v2_types

        assert sleeves_spec_module._freeze_plain_mapping is v2_types._freeze_plain_mapping


# ---------------------------------------------------------------------------
# 2. 未知袖子 fail-fast 与注册表封闭性
# ---------------------------------------------------------------------------


class TestUnknownSleeveFailFast:
    @pytest.mark.parametrize("bad", ["nope", "", None, "VALUE", " value ", 42])
    def test_unknown_sleeve_raises(self, bad):
        """含 P3-R4：非 str（None/42）与前后空白（" value "）查询侧同样拒绝。"""
        with pytest.raises(ValueError, match="未知袖子"):
            get_sleeve_spec(bad)  # type: ignore[arg-type]

    def test_error_lists_registered(self):
        with pytest.raises(ValueError, match="value"):
            get_sleeve_spec("events")

    def test_list_sleeve_ids(self):
        assert list_sleeve_ids() == ("value",)


class TestRegistryClosure:
    def test_no_public_register_surface(self):
        """封闭性：注册表模块无公开 register / 可写映射面。"""
        from src.lazybull.v2.sleeves import _registry as registry_module

        public = [n for n in dir(registry_module) if not n.startswith("_")]
        assert "register" not in public
        assert "REGISTRY" not in public
        assert set(registry_module.__all__) == {"get_sleeve_spec", "list_sleeve_ids"}

    def test_registry_mapping_write_rejected(self):
        """P3-1：_REGISTRY 经 MappingProxyType 封闭，语言层只读。"""
        from src.lazybull.v2.sleeves import _registry as registry_module

        assert isinstance(registry_module._REGISTRY, MappingProxyType)
        with pytest.raises(TypeError):
            registry_module._REGISTRY["events"] = VALUE_SLEEVE_SPEC  # type: ignore[index]
        assert list_sleeve_ids() == ("value",)

    def test_registry_key_matches_sleeve_id(self):
        """注册键 == spec.sleeve_id 自洽（P2-2 处置轮补：身份一致性断言）。"""
        from src.lazybull.v2.sleeves import _registry as registry_module

        for key, spec in registry_module._REGISTRY.items():
            assert key == spec.sleeve_id
            assert key == key.strip()  # 注册键无前后空白（P2-4 口径一致）


# ---------------------------------------------------------------------------
# 3. 候选超集声明
# ---------------------------------------------------------------------------


class TestFeatureColumnsDeclaration:
    def test_count_unique_and_clean(self):
        cols = VALUE_FEATURE_COLUMNS
        assert len(cols) == 375  # 冻结段末分区 schema：383 = 2 键 + 375 特征 + 6 标签
        assert len(set(cols)) == len(cols)
        assert "trade_date" not in cols and "ts_code" not in cols

    def test_no_label_columns(self):
        from src.lazybull.v2.store.column_groups import LABEL_VALUE_COLUMNS

        assert set(VALUE_FEATURE_COLUMNS) & set(LABEL_VALUE_COLUMNS) == set()

    def test_set_matches_panel_groups_single_source(self):
        """集合一致性：声明 == column_groups.PANEL_GROUPS 派生 375 集（冻结单一来源）。"""
        from src.lazybull.v2.store.column_groups import PANEL_GROUPS

        derived = {c for cols in PANEL_GROUPS.values() for c in cols} - {
            "trade_date",
            "ts_code",
        }
        assert set(VALUE_FEATURE_COLUMNS) == derived

    def test_order_sentinels_parquet_order(self):
        """序 = cs_train parquet schema 序（parity，§18.0-6）：哨兵断言首/尾。"""
        assert VALUE_FEATURE_COLUMNS[0] == "vol"
        assert VALUE_FEATURE_COLUMNS[-1] == "mkt_atr_pct_ma250"
        assert VALUE_FEATURE_COLUMNS[:4] == ("vol", "amount", "is_st", "is_suspended")

    def test_full_order_fingerprint_lock(self):
        """R2-T4-S01：完整有序声明指纹锁定（内部换位可检出；不读生产分区）。

        口径（与 feature_columns.py 注释一致）：
        ``sha256("\\n".join(VALUE_FEATURE_COLUMNS).encode()).hexdigest()``
        """
        digest = hashlib.sha256("\n".join(VALUE_FEATURE_COLUMNS).encode()).hexdigest()
        assert digest == "5827da6dd899b693eb62c384c40f92114ab8d783f34fed8eecf20c598e8807fd"

    def test_spec_carries_declaration(self):
        assert VALUE_SLEEVE_SPEC.feature_columns == VALUE_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 4. 信号接线核验
# ---------------------------------------------------------------------------


class TestSignalWiring:
    def test_signal_impl_resolves_to_ml_signal(self):
        """调用方（薄入口/宿主）视角：importlib 解析 signal_impl ⇒ v2 MLSignal。"""
        module_name, class_name = VALUE_SLEEVE_SPEC.signal_impl.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        from src.lazybull.v2.core.signal.ml_signal import MLSignal

        assert cls is MLSignal

    def test_construct_signal_from_spec(self, tmp_path):
        """规格信号参数 + 域解析市值（domains.py 单一来源）→ MLSignal 逐项相等。

        P3-5：``models_dir`` 显式传 tmp_path 隔离（缺省会读真实配置根）。
        """
        from src.lazybull.universe.domains import resolve_stock_domain
        from src.lazybull.v2.core.signal.ml_signal import MLSignal

        domain_cfg = resolve_stock_domain(VALUE_SLEEVE_SPEC.domain)
        signal = MLSignal(
            top_n=VALUE_SLEEVE_SPEC.top_n,
            models_dir=str(tmp_path),
            min_amount_ma20=VALUE_SLEEVE_SPEC.min_amount_ma20,
            min_total_mv=float(domain_cfg["min_total_mv_wan"]),
            max_total_mv=float(domain_cfg["max_total_mv_wan"]),
            exclude_financial=VALUE_SLEEVE_SPEC.exclude_financial,
            verbose=False,
        )
        assert signal.top_n == 20
        assert signal.min_amount_ma20 == 50000.0
        assert signal.min_total_mv == 500000.0  # main 域 50 亿（万元口径）
        assert signal.max_total_mv == 15000000.0  # 1500 亿
        assert signal.exclude_financial is True
        assert signal.downside_penalty == 0.0  # λ=0 不触发退役守卫（A5）

    def test_holding_period_days(self):
        """协议 §2.1 MVP 语义：袖子 A 持有期 = 20（到期判定唯一来源 sell_rules 不动）。"""
        assert VALUE_SLEEVE_SPEC.holding_period_days == 20

    def test_holding_period_matches_trading_config_default(self):
        """P3-8：spec.holding_period_days 与宿主回测默认 rebalance_freq 同源一致性
        （T5 宿主组装须做一致性断言的先行锚点）。"""
        from src.lazybull.v2.common.trading_config import TradingConfig

        assert TradingConfig().rebalance_freq == 20
        assert VALUE_SLEEVE_SPEC.holding_period_days == TradingConfig().rebalance_freq

    def test_spec_has_no_domain_market_value_fields(self):
        """域市值边界不入 spec（domains.py 单一来源，防双源漂移）。"""
        field_names = {f.name for f in dataclasses.fields(SleeveSpec)}
        assert not any("total_mv" in name for name in field_names)


# ---------------------------------------------------------------------------
# 5. spec → params_from_config 接线（生产值逐项对照）+ 键集完整性（P2-1）
# ---------------------------------------------------------------------------


class TestParamsFromConfigWiring:
    def test_train_fold_params_production_values(self):
        from src.lazybull.v2.train.trainer import params_from_config

        params = params_from_config(VALUE_SLEEVE_SPEC.build_model_config())
        # 标签系
        assert params.label_column == "neu_y_ret_20"
        assert params.task == "regression"
        assert params.label_transform == "cs_zscore"
        assert params.neutral_label_blend_weight == 0.0
        assert params.winsorize_p == 0.01
        assert params.pos_quantile is None
        assert params.pos_topk is None
        assert params.scale_pos_weight is None
        # 切分
        assert params.val_ratio == 0.2
        # 13 族开关（生产值）+ 运行时派生五族全关
        for name in (
            "fundamental",
            "alt",
            "margin",
            "cyq",
            "fund",
            "express",
            "enhanced",
            "north",
            "lhb",
            "consensus",
            "cashflow_quality",
        ):
            assert getattr(params, f"enable_{name}_features") is True, name
        assert params.enable_consensus_revision_features is False
        assert params.enable_dividend_policy_features is False
        assert params.enable_availability_markers is False
        assert params.enable_holdertrade_features is False
        assert params.enable_repurchase_features is False
        assert params.enable_top10fh_features is False
        assert params.enable_top_inst_features is False
        # 五族 feature_set 显式声明（P2-1-③，防未来开关打开时静默默认）
        assert params.holdertrade_feature_set == "full"
        assert params.repurchase_feature_set == "full"
        assert params.top10fh_feature_set == "full"
        # 门禁与裁剪
        assert params.feature_stability_filter is False
        assert params.factor_prune is False
        assert params.factor_exclude_file is None
        assert params.max_feature_missing_ratio == 0.6
        # 超参 16 项
        assert params.algorithm == "xgboost"
        assert params.n_estimators == 500
        assert params.max_depth == 5
        assert params.learning_rate == 0.03
        assert params.min_child_weight == 200
        assert params.colsample_bytree == 0.3
        assert params.subsample == 0.8
        assert params.reg_alpha == 0.03
        assert params.reg_lambda == 7.0
        assert params.gamma == 0.5
        assert params.objective == "mse"
        assert params.early_stopping_rounds == 50
        assert params.early_stopping_metric == "rank_ic_daily"
        assert params.min_best_iteration == 30
        assert params.random_state == 42
        # 样本权重 5 项 + freshness 2 项
        assert params.rank_weight_enabled is True
        assert params.rank_weight_topk == 120
        assert params.rank_weight == 100.0
        assert params.rank_weight_topk_weight_mode == "linear_decay"
        assert params.time_decay_half_life == 0.0
        assert params.freshness_strategy == "state_keep_event_decay"
        assert params.event_freshness_half_life_days == 120.0
        # 集成
        assert params.ensemble_seeds == (42, 61, 82)
        assert params.ensemble_offsets == 0
        assert params.ensemble_seed_keep_top_ratio == 1.0
        assert params.ensemble_seed_keep_min_models == 3
        # 域
        assert params.stock_domain == "main"


class TestSpecKeyCompleteness:
    """P2-1：规格键集 = params_from_config 读取面，缺一 / 多一即拒。"""

    def test_key_sets_match_params_from_config_surface(self):
        """AST 提取 params_from_config 实际读取键，与规格键集精确一致（防双侧漂移）。

        提取假设登记（reviewer1-P3-R5）：仅识别 ``hp/ls/fr/ss.get("字面量")`` 形态；
        若 params_from_config 未来改用 ``hp["key"]`` / ``hp.get(变量)`` / 解包读取，
        本锁会静默漏检（提取结果变空 ⇒ 下方非空断言先报）——新增读取方式时必须
        同步升级本提取（或改为 TrainFoldParams 字段反射比对）。
        """
        trainer_path = _REPO_ROOT / "src" / "lazybull" / "v2" / "train" / "trainer.py"
        tree = ast.parse(trainer_path.read_text(encoding="utf-8"), filename=str(trainer_path))
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "params_from_config"
        )
        groups: dict[str, set] = {"ls": set(), "fr": set(), "ss": set(), "hp": set()}
        for node in ast.walk(func):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
            ):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id in groups
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                groups[owner.id].add(node.args[0].value)
        assert groups["ls"], "未提取到 label_spec 读取键（AST 假设失效）"
        assert groups["ls"] == LABEL_SPEC_KEYS - {"version"}
        assert groups["fr"] == FEATURE_RECIPE_KEYS - {"version"}
        assert groups["ss"] == SPLIT_SPEC_KEYS - {"version"}
        assert groups["hp"] == HYPERPARAM_KEYS

    def test_missing_key_rejected(self):
        """漏写键（如 rank_weight）⇒ ValueError 且消息含缺失键名。"""
        hyper = dict(VALUE_SLEEVE_SPEC.hyperparams)
        del hyper["rank_weight"]
        with pytest.raises(ValueError, match="rank_weight"):
            _make_spec(hyperparams=hyper)

    def test_unknown_key_rejected(self):
        """拼写错误键（rank_weight_top_k）⇒ ValueError 且消息含多余键名。"""
        recipe = dict(VALUE_SLEEVE_SPEC.feature_recipe)
        recipe["enable_alt_feature"] = recipe.pop("enable_alt_features")
        with pytest.raises(ValueError, match="enable_alt_feature"):
            _make_spec(feature_recipe=recipe)

    def test_none_mapping_rejected(self):
        """规格层完整性：None 映射（键集全缺）⇒ ValueError（协议层 None 合法与
        本层无关——规格层是生产声明）。"""
        with pytest.raises(ValueError, match="label_spec"):
            _make_spec(label_spec=None)

    def test_value_spec_key_sets_exact(self):
        assert set(VALUE_SLEEVE_SPEC.label_spec) == LABEL_SPEC_KEYS
        assert set(VALUE_SLEEVE_SPEC.feature_recipe) == FEATURE_RECIPE_KEYS
        assert set(VALUE_SLEEVE_SPEC.split_spec) == SPLIT_SPEC_KEYS
        assert set(VALUE_SLEEVE_SPEC.hyperparams) == HYPERPARAM_KEYS


# ---------------------------------------------------------------------------
# 6. 依赖方向静态守卫（AST 扫描；候选展开沿 test_v2_p2a_t6_equivalence.py
#    `_import_candidates` 先例：from <pkg> import <name> 同时展开父包与
#    父包.名称 两个候选，覆盖父包成员导入形态，R2-T4-02）
# ---------------------------------------------------------------------------


def _matches_prefix(module: str, prefix: str) -> bool:
    """按段边界匹配 dotted 模块名前缀。"""
    return module == prefix or module.startswith(prefix + ".")


def _import_candidates_from_source(
    source: str, package_parts: Tuple[str, ...]
) -> Iterator[Tuple[int, str]]:
    """从源码文本产出 (行号, import 候选模块)——绝对/相对（按包层级归一）/
    父包成员导入全形态覆盖。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and node.level > 0:
                keep = len(package_parts) - (node.level - 1)
                base = ".".join(package_parts[:keep])
                module = f"{base}.{module}" if module else base
            yield node.lineno, module
            for alias in node.names:
                if module:
                    yield node.lineno, f"{module}.{alias.name}"


def _iter_imported_modules(py_file: Path) -> Iterator[Tuple[int, str]]:
    package_parts = tuple(py_file.parent.relative_to(_REPO_ROOT).parts)
    yield from _import_candidates_from_source(py_file.read_text(encoding="utf-8"), package_parts)


def _collect_offenders(root: Path, forbidden_prefixes: Tuple[str, ...]) -> list[str]:
    offenders: list[str] = []
    for py_file in sorted(root.rglob("*.py")):
        for lineno, module in _iter_imported_modules(py_file):
            for prefix in forbidden_prefixes:
                if _matches_prefix(module, prefix):
                    rel = py_file.relative_to(_REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}: {module}")
    return offenders


class TestDependencyBans:
    #: sleeves 允许依赖 = common.types / common.protocols / store（只读）；
    #: 禁 core/train/hosts/evidence/factors/universe/scripts（§7 / F10）
    _FORBIDDEN_SLEEVES_PREFIXES = (
        "src.lazybull.v2.core",
        "src.lazybull.v2.train",
        "src.lazybull.v2.hosts",
        "src.lazybull.v2.evidence",
        "src.lazybull.factors",
        "src.lazybull.universe",
        "scripts",
    )

    def test_sleeves_no_forbidden_imports(self):
        assert _SLEEVES_DIR.is_dir()
        offenders = _collect_offenders(_SLEEVES_DIR, self._FORBIDDEN_SLEEVES_PREFIXES)
        assert offenders == [], f"v2/sleeves 依赖禁令违例: {offenders}"

    def test_train_no_sleeves_universe_imports(self):
        """F10/§3.1：train 不得 import sleeves / universe（补缺静态守卫，§18.0-3）。"""
        train_dir = _REPO_ROOT / "src" / "lazybull" / "v2" / "train"
        assert train_dir.is_dir()
        offenders = _collect_offenders(
            train_dir, ("src.lazybull.v2.sleeves", "src.lazybull.universe")
        )
        assert offenders == [], f"v2/train sleeves/universe 禁令违例: {offenders}"

    def test_value_no_sibling_sleeve_imports(self):
        """袖子互不 import：value 包不得 import 兄弟袖子包与 _registry（注册方向反向）。"""
        value_dir = _SLEEVES_DIR / "value"
        siblings = [
            d.name
            for d in _SLEEVES_DIR.iterdir()
            if d.is_dir() and d.name not in ("value", "__pycache__")
        ]
        forbidden = tuple(f"src.lazybull.v2.sleeves.{name}" for name in siblings) + (
            "src.lazybull.v2.sleeves._registry",
        )
        offenders = _collect_offenders(value_dir, forbidden)
        assert offenders == [], f"value 袖子互 import 禁令违例: {offenders}"


class TestGuardEffectiveness:
    """R2-T4-02：守卫候选展开有效性（合成正反例 + 真实入口注入，沿 P2a-T6 先例）。"""

    #: (注入源码, 扫描根相对路径, 被扫文件包路径, 对应禁令前缀)
    _INJECTION_CASES = (
        (
            "from src.lazybull.v2 import core",
            "src/lazybull/v2/sleeves",
            "src.lazybull.v2.sleeves",
            "src.lazybull.v2.core",
        ),
        (
            "from src.lazybull import universe",
            "src/lazybull/v2/train",
            "src.lazybull.v2.train",
            "src.lazybull.universe",
        ),
        (
            "from .. import _registry",
            "src/lazybull/v2/sleeves/value",
            "src.lazybull.v2.sleeves.value",
            "src.lazybull.v2.sleeves._registry",
        ),
        (
            "from scripts.v2_p3 import t1_reconcile_adapter",
            "src/lazybull/v2/sleeves",
            "src.lazybull.v2.sleeves",
            "scripts",
        ),
    )

    def test_synthetic_injections_caught(self):
        """四个注入形态（父包成员 / 相对父包 / 顶层 scripts）均被候选展开捕获。"""
        for source, _root, package, prefix in self._INJECTION_CASES:
            candidates = [
                module
                for _lineno, module in _import_candidates_from_source(
                    source, tuple(package.split("."))
                )
            ]
            assert any(
                _matches_prefix(candidate, prefix) for candidate in candidates
            ), f"注入形态漏检: {source!r}（候选 {candidates}）"

    def test_legit_imports_not_flagged(self):
        """合法导入（common.types / store.column_groups / 本袖 spec）零误报。"""
        legit = (
            ("from src.lazybull.v2.common.types import ModelConfig", "src.lazybull.v2.sleeves"),
            (
                "from src.lazybull.v2.store.column_groups import LABEL_VALUE_COLUMNS",
                "src.lazybull.v2.sleeves",
            ),
            ("from ..spec import SleeveSpec", "src.lazybull.v2.sleeves.value"),
            ("from . import spec", "src.lazybull.v2.sleeves.value"),
        )
        forbidden = TestDependencyBans._FORBIDDEN_SLEEVES_PREFIXES
        for source, package in legit:
            candidates = [
                module
                for _lineno, module in _import_candidates_from_source(
                    source, tuple(package.split("."))
                )
            ]
            assert not any(
                _matches_prefix(candidate, prefix)
                for candidate in candidates
                for prefix in forbidden
            ), f"合法导入误报: {source!r}"

    @pytest.mark.parametrize(
        "inject_line,root_rel,prefix",
        [
            (
                "from src.lazybull.v2 import core",
                "src/lazybull/v2/sleeves",
                "src.lazybull.v2.core",
            ),
            (
                "from src.lazybull import universe",
                "src/lazybull/v2/train",
                "src.lazybull.universe",
            ),
        ],
    )
    def test_injection_via_real_entry_caught(self, monkeypatch, inject_line, root_rel, prefix):
        """注入真实被扫文件后经**正式扫描入口**（真实路径 → 真实所属包归一）捕获。"""
        target = _REPO_ROOT / root_rel / ("spec.py" if "sleeves" in root_rel else "trainer.py")
        original = target.read_text(encoding="utf-8")
        real_read_text = Path.read_text

        def _patched(self, *args, **kwargs):
            if self == target:
                return original + inject_line + "\n"
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _patched)
        offenders = _collect_offenders(_REPO_ROOT / root_rel, (prefix,))
        assert any(
            str(target.relative_to(_REPO_ROOT)).replace("\\", "/") in item for item in offenders
        ), f"真实入口注入未捕获: {inject_line!r}"


# ---------------------------------------------------------------------------
# 7. 注册文件唯一静态守卫
# ---------------------------------------------------------------------------


class TestRegistrationUniqueness:
    def test_spec_construction_sites_unique(self):
        """``SleeveSpec(`` 出现点仅许 `sleeves/value/spec.py`（构造点，文件级）、
        `sleeves/_registry.py`（注册文件自身）与 `tests/`；排除 temp/ logs/
        __pycache__（P3-2：临时脚本字样不得误报）。"""
        allowed_exact = {
            "src/lazybull/v2/sleeves/value/spec.py",
            "src/lazybull/v2/sleeves/_registry.py",
        }
        excluded_dirs = {"temp", "logs", "__pycache__"}
        offenders: list[str] = []
        for py_file in sorted(_REPO_ROOT.rglob("*.py")):
            rel = py_file.relative_to(_REPO_ROOT).as_posix()
            parts = py_file.relative_to(_REPO_ROOT).parts
            if any(part.startswith(".") or part in excluded_dirs for part in parts):
                continue  # 跳过隐藏目录（.git 等）与临时/日志/缓存目录
            if rel in allowed_exact or rel.startswith("tests/"):
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "SleeveSpec(" in line:
                    offenders.append(f"{rel}:{lineno}")
        assert offenders == [], f"SleeveSpec 构造点越界（注册文件唯一违例）: {offenders}"


# ---------------------------------------------------------------------------
# 8. 规格构造 fail-fast（__post_init__ 正反例；P2-4 归一化与类型缺口）
# ---------------------------------------------------------------------------


def _make_spec(**over):
    """合法完整键集 spec（测试构造，不进注册表；键集 = 生产表面，P2-1）。"""
    base = dict(
        sleeve_id="t",
        spec_version=1,
        model_family="stock_selection",
        label_name="neu_y_ret_20",
        label_spec={
            "version": 1,
            "task": "regression",
            "label_transform": "cs_zscore",
            "neutral_label_blend_weight": 0.0,
            "winsorize_p": 0.01,
            "pos_quantile": None,
            "pos_topk": None,
            "scale_pos_weight": None,
        },
        feature_recipe={
            "version": 1,
            **{key: False for key in FEATURE_RECIPE_KEYS if key.startswith("enable_")},
            "holdertrade_feature_set": "full",
            "repurchase_feature_set": "full",
            "top10fh_feature_set": "full",
            "feature_stability_filter": False,
            "factor_prune": False,
            "factor_exclude_file": None,
            "max_feature_missing_ratio": 0.6,
        },
        split_spec={"version": 1, "val_ratio": 0.2},
        hyperparams={
            "algorithm": "xgboost",
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 200,
            "colsample_bytree": 0.3,
            "subsample": 0.8,
            "reg_alpha": 0.03,
            "reg_lambda": 7.0,
            "gamma": 0.5,
            "objective": "mse",
            "early_stopping_rounds": 50,
            "early_stopping_metric": "rank_ic_daily",
            "min_best_iteration": 30,
            "random_state": 42,
            "rank_weight_enabled": True,
            "rank_weight_topk": 120,
            "rank_weight": 100.0,
            "rank_weight_topk_weight_mode": "linear_decay",
            "time_decay_half_life": 0.0,
            "freshness_strategy": "state_keep_event_decay",
            "event_freshness_half_life_days": 120.0,
            "ensemble_offsets": 0,
            "ensemble_seed_keep_top_ratio": 1.0,
            "ensemble_seed_keep_min_models": 3,
        },
        ensemble_seeds=(42,),
        train_window_months=72,
        domain="main",
        feature_columns=("vol",),
        top_n=20,
        holding_period_days=20,
        min_amount_ma20=50000.0,
        exclude_financial=True,
        signal_impl="m.C",
    )
    base.update(over)
    return SleeveSpec(**base)


class TestSpecValidation:
    @pytest.mark.parametrize(
        "over",
        [
            {"sleeve_id": ""},
            {"sleeve_id": "  "},
            {"sleeve_id": " value"},  # P2-4：前后空白（注册键与查询口径一致）
            {"spec_version": 0},
            {"spec_version": True},  # P2-4：bool 是 int 子类，拒绝
            {"label_name": ""},
            {"model_family": ""},  # P2-4：家族名空串拒绝
            {"signal_impl": "NoDot"},
            {"ensemble_seeds": ()},
            {"ensemble_seeds": ("42",)},
            {"ensemble_seeds": (True,)},  # bool 种子拒绝
            {"train_window_months": 0},
            {"domain": " "},
            {"feature_columns": ()},
            {"feature_columns": ("vol", "vol")},
            {"feature_columns": ("vol", "ts_code")},
            {"feature_columns": ("vol", "neu_y_ret_20")},
            {"feature_columns": ("vol", 42)},  # 非 str 元素拒绝
            {"top_n": 0},
            {"holding_period_days": 0},
            {"min_amount_ma20": -1.0},
            {"min_amount_ma20": None},  # P3-R3：None 必须 ValueError（非 TypeError）
            {"min_amount_ma20": True},  # P3-R3：bool 拒绝
            {"exclude_financial": "yes"},  # P3-R3：非 bool 拒绝
            {"exclude_financial": 1},
        ],
    )
    def test_invalid_specs_rejected(self, over):
        with pytest.raises(ValueError):
            _make_spec(**over)

    def test_spec_missing_version_rejected(self):
        """F02 版本门前移规格层：spec 缺 version ⇒ ValueError。"""
        label_spec = {
            "task": "regression",
            "label_transform": "cs_zscore",
            "neutral_label_blend_weight": 0.0,
            "winsorize_p": 0.01,
            "pos_quantile": None,
            "pos_topk": None,
            "scale_pos_weight": None,
        }
        with pytest.raises(ValueError, match="version"):
            _make_spec(label_spec=label_spec)

    @pytest.mark.parametrize("bad_version", ["1", True, 0, None])
    def test_spec_version_type_gate(self, bad_version):
        """P2-4：三 spec 的 version 必须为 >= 1 严格整数（字符串 / bool / 0 / None 拒绝）。"""
        split_spec = {"version": bad_version, "val_ratio": 0.2}
        with pytest.raises(ValueError, match="version"):
            _make_spec(split_spec=split_spec)

    def test_sequence_fields_normalized_to_tuple(self):
        """P2-4：list 入参归一化为 tuple（不留可变容器）。"""
        spec = _make_spec(ensemble_seeds=[42, 61], feature_columns=["vol", "amount"])
        assert spec.ensemble_seeds == (42, 61) and isinstance(spec.ensemble_seeds, tuple)
        assert spec.feature_columns == ("vol", "amount")
        assert isinstance(spec.feature_columns, tuple)

    @pytest.mark.parametrize("field", ["ensemble_seeds", "feature_columns"])
    def test_sequence_str_rejected(self, field):
        """P2-4：str 入参显式拒绝（防 tuple('vol') 逐字符拆分）。"""
        with pytest.raises(ValueError, match=field):
            _make_spec(**{field: "vol"})

    def test_caller_mutation_does_not_leak(self):
        """调用方改原映射/序列不穿透已构造规格（脱钩 + 冻结 + 归一化）。"""
        recipe = dict(_make_spec().feature_recipe)  # 完整键集（enable_* 全 False）
        seeds = [42, 61, 82]
        spec = _make_spec(feature_recipe=recipe, ensemble_seeds=seeds)
        recipe["enable_alt_features"] = True
        seeds.append(7)
        assert spec.feature_recipe["enable_alt_features"] is False
        assert spec.ensemble_seeds == (42, 61, 82)


# ---------------------------------------------------------------------------
# 9. 键列同值锁定（P3-4：_KEY_COLUMNS 多处声明防漂移）
# ---------------------------------------------------------------------------


class TestKeyColumnsLock:
    def test_key_columns_same_value_across_modules(self):
        """五处 ``_KEY_COLUMNS`` 声明同值锁定（沿 holdertrade 列集双处锁定先例）。"""
        from src.lazybull.v2.store import data_store, panel_builder, replay_compare
        from src.lazybull.v2.train import data_adapter

        expected = ("trade_date", "ts_code")
        assert sleeves_spec_module._KEY_COLUMNS == expected
        assert data_adapter._KEY_COLUMNS == expected
        assert data_store._KEY_COLUMNS == expected
        assert panel_builder._KEY_COLUMNS == expected
        assert replay_compare._KEY_COLUMNS == expected
