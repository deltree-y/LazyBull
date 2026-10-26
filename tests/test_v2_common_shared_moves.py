# -*- coding: utf-8 -*-
"""v2 P3-T2 共享件上移与 shim 等价测试（单一来源身份断言 + 数值语义 + AST 依赖禁令）。

覆盖：
- 身份断言（is）：ml.eval_utils / ml.walk_forward.chain_metrics / ml.train_core.freshness /
  ml.train_core.constants / factors 三因子模块的 shim re-export 与 v2.common 唯一来源
  为同一对象；
- freshness 数值语义：指数衰减权重手算比对、NaN/负值/非法半衰期边界、
  apply_serving_event_decay 仅 state_keep_event_decay 衰减；
- chain_metrics 已知值：total_return / max_drawdown / trading_days 手算比对、
  空输入全 None、annual_trading_days<=0 拒绝；
- read_*_schema_version 边界：None/空 dict/非法串 ⇒ -1，正确版本回读，两条导入路径一致；
- ml_signal 改指冒烟：ModelRegistry / freshness 符号 is v2.common 对应件；
- 依赖禁令静态守卫（AST）：v2/common 不得 import 业务包；v2/train 不得 import factors。
"""

import ast
import math
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import pytest

import src.lazybull.factors.cashflow_quality as factors_cashflow_quality
import src.lazybull.factors.consensus_revision as factors_consensus_revision
import src.lazybull.factors.dividend as factors_dividend
import src.lazybull.ml.eval_utils as legacy_eval_utils
import src.lazybull.ml.train_core.constants as legacy_constants
import src.lazybull.ml.train_core.freshness as legacy_freshness
import src.lazybull.ml.walk_forward.chain_metrics as legacy_chain_metrics
import src.lazybull.v2.common.chain_metrics as v2_chain_metrics
import src.lazybull.v2.common.eval_utils as v2_eval_utils
import src.lazybull.v2.common.freshness as v2_freshness
import src.lazybull.v2.common.model_schema as v2_model_schema

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. 单一来源身份断言（is）
# ---------------------------------------------------------------------------


class TestSingleSourceIdentity:
    @pytest.mark.parametrize(
        "name",
        [
            "compute_daily_rankic",
            "compute_daily_topk_returns",
            "compute_diagnostic_statistics",
            "evaluate_predictions_by_date",
            "print_diagnostic_report",
            "summarize_daily_metrics",
        ],
    )
    def test_eval_utils_identity(self, name):
        """ml.eval_utils 六函数 is v2.common.eval_utils 同名函数。"""
        assert getattr(legacy_eval_utils, name) is getattr(v2_eval_utils, name)

    def test_chain_metrics_identity(self):
        """ml.walk_forward.chain_metrics.calculate_chain_metrics is v2.common 同名函数。"""
        assert (
            legacy_chain_metrics.calculate_chain_metrics is v2_chain_metrics.calculate_chain_metrics
        )

    @pytest.mark.parametrize("name", ["apply_event_freshness_decay", "apply_serving_event_decay"])
    def test_freshness_function_identity(self, name):
        """ml.train_core.freshness 两函数 is v2.common.freshness 同名函数。"""
        assert getattr(legacy_freshness, name) is getattr(v2_freshness, name)

    @pytest.mark.parametrize(
        "name",
        [
            "FRESHNESS_STRATEGY_DROP_ALL",
            "FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY",
            "FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY",
            "DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS",
            "EVENT_FRESHNESS_TO_VALUE_COLUMNS",
        ],
    )
    def test_constants_freshness_symbols_identity(self, name):
        """ml.train_core.constants 的 5 个 freshness 共享符号 is v2.common.freshness 同名对象。"""
        assert getattr(legacy_constants, name) is getattr(v2_freshness, name)

    @pytest.mark.parametrize(
        "name",
        [
            "read_cashflow_quality_schema_version",
            "read_cons_revision_schema_version",
            "read_dividend_policy_schema_version",
        ],
    )
    def test_constants_read_functions_identity(self, name):
        """ml.train_core.constants 的 read_*×3 is v2.common.model_schema 同名函数。"""
        assert getattr(legacy_constants, name) is getattr(v2_model_schema, name)

    def test_factors_schema_symbols_identity(self):
        """factors 三因子模块 schema 符号 is v2.common.model_schema 对应件。"""
        # cashflow_quality 三符号
        assert (
            factors_cashflow_quality.CASHFLOW_QUALITY_SCHEMA_VERSION
            is v2_model_schema.CASHFLOW_QUALITY_SCHEMA_VERSION
        )
        assert (
            factors_cashflow_quality.CASHFLOW_QUALITY_VERSION_COL
            is v2_model_schema.CASHFLOW_QUALITY_VERSION_COL
        )
        assert (
            factors_cashflow_quality.cashflow_quality_live_columns
            is v2_model_schema.cashflow_quality_live_columns
        )
        # dividend 两符号
        assert (
            factors_dividend.DIVIDEND_POLICY_SCHEMA_VERSION
            is v2_model_schema.DIVIDEND_POLICY_SCHEMA_VERSION
        )
        assert (
            factors_dividend.DIVIDEND_POLICY_VERSION_COL
            is v2_model_schema.DIVIDEND_POLICY_VERSION_COL
        )
        # consensus_revision 两符号
        assert (
            factors_consensus_revision.CONSENSUS_REVISION_SCHEMA_VERSION
            is v2_model_schema.CONSENSUS_REVISION_SCHEMA_VERSION
        )
        assert (
            factors_consensus_revision.CONSENSUS_REVISION_VERSION_COL
            is v2_model_schema.CONSENSUS_REVISION_VERSION_COL
        )


# ---------------------------------------------------------------------------
# 2. freshness 数值语义
# ---------------------------------------------------------------------------


class TestFreshnessSemantics:
    def test_event_decay_weights_hand_computed(self):
        """权重 = exp(-ln2*days/45)：手算期望值比对；NaN ⇒ 1.0，负值按 0。"""
        df = pd.DataFrame(
            {
                "forecast_freshness_days": [0.0, 45.0, 90.0, np.nan, -5.0],
                "forecast_type_score": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        out_df, decay_stats = v2_freshness.apply_event_freshness_decay(
            df, event_freshness_cols=["forecast_freshness_days"], half_life_days=45.0
        )
        ln2 = math.log(2.0)
        expected = np.array(
            [
                1.0 * math.exp(-ln2 * 0.0 / 45.0),
                2.0 * math.exp(-ln2 * 45.0 / 45.0),
                3.0 * math.exp(-ln2 * 90.0 / 45.0),
                4.0,  # freshness NaN ⇒ 权重 1.0
                5.0,  # freshness 负值按 0 ⇒ 权重 1.0
            ]
        )
        np.testing.assert_allclose(out_df["forecast_type_score"].to_numpy(), expected, rtol=1e-12)
        assert decay_stats == {"forecast_freshness_days": 5}

    def test_event_decay_half_life_nonpositive_rejected(self):
        """half_life_days <= 0 ⇒ ValueError。"""
        df = pd.DataFrame({"forecast_freshness_days": [1.0], "forecast_type_score": [1.0]})
        with pytest.raises(ValueError, match="必须 > 0"):
            v2_freshness.apply_event_freshness_decay(
                df, event_freshness_cols=["forecast_freshness_days"], half_life_days=0.0
            )
        with pytest.raises(ValueError, match="必须 > 0"):
            v2_freshness.apply_event_freshness_decay(
                df, event_freshness_cols=["forecast_freshness_days"], half_life_days=-1.0
            )

    def test_serving_decay_only_for_decay_strategy(self):
        """apply_serving_event_decay：仅 state_keep_event_decay 衰减，其他策略原样。"""
        base = pd.DataFrame({"forecast_freshness_days": [45.0], "forecast_type_score": [2.0]})
        decayed = v2_freshness.apply_serving_event_decay(
            base.copy(),
            v2_freshness.FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
            45.0,
        )
        assert decayed["forecast_type_score"].iloc[0] == pytest.approx(1.0)

        for strategy in (
            v2_freshness.FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
            v2_freshness.FRESHNESS_STRATEGY_DROP_ALL,
        ):
            untouched = v2_freshness.apply_serving_event_decay(base.copy(), strategy, 45.0)
            assert untouched["forecast_type_score"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. chain_metrics 已知值
# ---------------------------------------------------------------------------


class TestChainMetrics:
    def test_known_values_with_split_index(self):
        """含 split_index 两折的小 nav 序列：手算 total_return / max_drawdown / trading_days。"""
        chain_df = pd.DataFrame(
            {
                "nav": [1.0, 1.1, 1.21, 1.0, 1.05],
                "split_index": [0, 0, 0, 1, 1],
            }
        )
        metrics = v2_chain_metrics.calculate_chain_metrics(chain_df)
        # total_return = 末值/首值 - 1
        assert metrics["total_return"] == pytest.approx(1.05 / 1.0 - 1.0)
        # max_drawdown：cummax=[1,1.1,1.21,1.21,1.21]，最深为 1.0/1.21-1
        assert metrics["max_drawdown"] == pytest.approx(1.0 / 1.21 - 1.0)
        # 跨折收益被置 NaN：有效日收益 = [0.1, 0.1, 0.05] ⇒ 3 个
        assert metrics["trading_days"] == 3
        # cagr = nav_ratio ** (annual_trading_days / trading_days) - 1
        assert metrics["cagr"] == pytest.approx(1.05 ** (252 / 3) - 1.0)

    def test_empty_inputs_return_all_none(self):
        """空 df / 无 nav 列 / None ⇒ 全 None 字典。"""
        expected = {
            "total_return": None,
            "cagr": None,
            "max_drawdown": None,
            "volatility": None,
            "sharpe": None,
            "trading_days": None,
        }
        assert v2_chain_metrics.calculate_chain_metrics(pd.DataFrame()) == expected
        assert v2_chain_metrics.calculate_chain_metrics(pd.DataFrame({"x": [1.0]})) == expected
        assert v2_chain_metrics.calculate_chain_metrics(None) == expected

    def test_annual_trading_days_nonpositive_rejected(self):
        """annual_trading_days <= 0 ⇒ ValueError。"""
        chain_df = pd.DataFrame({"nav": [1.0, 1.1, 1.21]})
        with pytest.raises(ValueError, match="annual_trading_days"):
            v2_chain_metrics.calculate_chain_metrics(chain_df, annual_trading_days=0)
        with pytest.raises(ValueError, match="annual_trading_days"):
            v2_chain_metrics.calculate_chain_metrics(chain_df, annual_trading_days=-1)


# ---------------------------------------------------------------------------
# 4. read_*_schema_version 边界
# ---------------------------------------------------------------------------


class TestReadSchemaVersion:
    @pytest.mark.parametrize(
        "fn_name, key, expected_version",
        [
            ("read_cashflow_quality_schema_version", "cashflow_quality_schema_version", 3),
            ("read_cons_revision_schema_version", "cons_revision_schema_version", 2),
            ("read_dividend_policy_schema_version", "dividend_policy_schema_version", 3),
        ],
    )
    def test_boundaries_and_dual_import_paths(self, fn_name, key, expected_version):
        """None/{} ⇒ -1；非法内容（"v2" 字符串）⇒ -1；正确版本回读；两条导入路径一致。"""
        for module in (legacy_constants, v2_model_schema):
            fn = getattr(module, fn_name)
            assert fn(None) == -1
            assert fn({}) == -1
            assert fn({key: "v2"}) == -1
            assert fn({key: expected_version}) == expected_version


# ---------------------------------------------------------------------------
# 5. ml_signal 改指冒烟
# ---------------------------------------------------------------------------


class TestMlSignalRewiring:
    def test_ml_signal_points_to_v2_common(self):
        """v2.core.signal.ml_signal 的 ModelRegistry / freshness 符号 is v2.common 对应件。"""
        from src.lazybull.v2.common.model_registry import ModelRegistry
        from src.lazybull.v2.core.signal import ml_signal

        assert ml_signal.ModelRegistry is ModelRegistry
        assert ml_signal.apply_event_freshness_decay is v2_freshness.apply_event_freshness_decay
        assert (
            ml_signal.EVENT_FRESHNESS_TO_VALUE_COLUMNS
            is v2_freshness.EVENT_FRESHNESS_TO_VALUE_COLUMNS
        )


# ---------------------------------------------------------------------------
# 6. 依赖禁令静态守卫（AST 扫描）
# ---------------------------------------------------------------------------

#: F14：v2/common 不得依赖业务包（允许 src.lazybull.common.* 等配置基础设施过渡依赖）
_FORBIDDEN_COMMON_PREFIXES = (
    "src.lazybull.factors",
    "src.lazybull.v2.train",
    "src.lazybull.v2.store",
    "src.lazybull.v2.core",
    "src.lazybull.v2.sleeves",
    "src.lazybull.v2.universe",
    "src.lazybull.v2.hosts",
    "src.lazybull.v2.evidence",
)

#: v2/train 包级契约：不得 import factors
_FORBIDDEN_TRAIN_PREFIXES = ("src.lazybull.factors",)


def _matches_prefix(module: str, prefix: str) -> bool:
    """按段边界匹配 dotted 模块名前缀（避免 protocols.core 误伤 v2.core 禁令）。"""
    return module == prefix or module.startswith(prefix + ".")


def _iter_imported_modules(py_file: Path) -> Iterator[Tuple[int, str]]:
    """解析 py 文件 AST，产出 (行号, import 候选模块)（相对导入按文件所在包解析）。

    P3-T4 评审处置（reviewer2-R2-T4-02 同源缺陷一并修复）：``from <pkg> import <name>``
    同时展开父包与「父包.名称」两个候选（父包成员导入形态全捕获；零真实命中已实测，
    沿 test_v2_p2a_t6_equivalence.py ``_import_candidates`` 先例）。
    """
    package_parts = list(py_file.parent.relative_to(_REPO_ROOT).parts)
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
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


def _collect_offenders(root: Path, forbidden_prefixes: Tuple[str, ...]) -> List[str]:
    """扫描 root 下全部 .py，返回触犯禁令的 '文件:行号: 模块' 清单。"""
    offenders: List[str] = []
    for py_file in sorted(root.rglob("*.py")):
        for lineno, module in _iter_imported_modules(py_file):
            for prefix in forbidden_prefixes:
                if _matches_prefix(module, prefix):
                    rel = py_file.relative_to(_REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}: {module}")
    return offenders


class TestDependencyBans:
    def test_v2_common_no_business_package_imports(self):
        """F14：v2/common（含子包）不得 import factors/train/store/core/sleeves 等业务包。"""
        common_dir = _REPO_ROOT / "src" / "lazybull" / "v2" / "common"
        assert common_dir.is_dir()
        offenders = _collect_offenders(common_dir, _FORBIDDEN_COMMON_PREFIXES)
        assert offenders == [], f"v2/common 依赖禁令违例: {offenders}"

    def test_v2_train_no_factors_imports(self):
        """v2/train 包级契约：不得 import src.lazybull.factors。"""
        train_dir = _REPO_ROOT / "src" / "lazybull" / "v2" / "train"
        assert train_dir.is_dir()
        offenders = _collect_offenders(train_dir, _FORBIDDEN_TRAIN_PREFIXES)
        assert offenders == [], f"v2/train factors 禁令违例: {offenders}"
