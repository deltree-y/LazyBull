# -*- coding: utf-8 -*-
"""P3-T3 辅助测试:v2 训练链路参数映射、默认值对账与双源常量锁定。

全合成数据,不依赖真实 data/。覆盖:

1. ``params_from_config``(ModelConfig → TrainFoldParams,§13.3.1)全字段映射 /
   缺省语义 / 空 ensemble_seeds;
2. ``TrainFoldParams`` 默认值 vs 旧 ``ml/walk_forward/cli.py`` parser 默认值逐项一致;
3. 旧新 ``_build_feature_flag_train_params`` 同输入同输出(13 键 dict 全等);
4. ``resolve_explicit_split_dates`` 与 ``split_train_val_by_date`` 比例切分同数学;
5. ``hosts/backtest/replay.py`` 与 ``train/data_adapter.py`` 特征源常量双源等价;
6. ``reject_retired_switches`` 退役开关显式拒绝(错误消息含「已退役」);
7. 旧模块零改动守卫(git 工作区扫描 + 旧文件存在性)。
"""

import dataclasses
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from src.lazybull.ml.walk_forward.cli import build_walk_forward_parser
from src.lazybull.ml.walk_forward.training_core import (
    _build_feature_flag_train_params as legacy_build_feature_flag_train_params,
)
from src.lazybull.v2.common.types import ModelConfig
from src.lazybull.v2.hosts.backtest import replay
from src.lazybull.v2.train import data_adapter
from src.lazybull.v2.train.core import split_train_val_by_date
from src.lazybull.v2.train.trainer import params_from_config
from src.lazybull.v2.train.walk_forward.orchestrator import reject_retired_switches
from src.lazybull.v2.train.walk_forward.training_core import (
    TrainFoldParams,
    _build_feature_flag_train_params as v2_build_feature_flag_train_params,
    resolve_explicit_split_dates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _full_mapping_case() -> Tuple[ModelConfig, Dict[str, Any]]:
    """构造全量键 ModelConfig 与 61 字段期望值。

    label_spec 七键 / feature_recipe 全量键 / hyperparams 全量键的键名与
    TrainFoldParams 字段名 1:1 同名,直接并入期望字典;config 侧仅
    label_name→label_column、domain→stock_domain、ensemble_seeds→tuple 三个改名/转型键。
    """
    label_spec = {
        "task": "classification",
        "label_transform": "cs_zscore",
        "neutral_label_blend_weight": 0.25,
        "winsorize_p": 0.02,
        "pos_quantile": 0.2,
        "pos_topk": 300,
        "scale_pos_weight": 2.5,
    }
    split_spec = {"val_ratio": 0.2}
    feature_recipe = {
        # 13 族 enable_* 开关
        "enable_fundamental_features": True,
        "enable_alt_features": True,
        "enable_margin_features": True,
        "enable_cyq_features": True,
        "enable_fund_features": True,
        "enable_express_features": True,
        "enable_enhanced_features": True,
        "enable_north_features": True,
        "enable_lhb_features": True,
        "enable_consensus_features": True,
        "enable_cashflow_quality_features": True,
        "enable_consensus_revision_features": True,
        "enable_dividend_policy_features": True,
        # 可用性标记 + 运行时派生四族开关与列集
        "enable_availability_markers": True,
        "enable_holdertrade_features": True,
        "holdertrade_feature_set": "core",
        "enable_repurchase_features": True,
        "repurchase_feature_set": "headroom",
        "enable_top10fh_features": True,
        "top10fh_feature_set": "concentration",
        "enable_top_inst_features": True,
        # 筛选/裁剪
        "feature_stability_filter": True,
        "factor_prune": True,
        "factor_exclude_file": "configs/factor_exclude_test.json",
        "max_feature_missing_ratio": 0.5,
    }
    hyperparams = {
        # freshness 2 键
        "freshness_strategy": "drop_all",
        "event_freshness_half_life_days": 30.0,
        # 样本权重 5 键
        "rank_weight_enabled": False,
        "rank_weight_topk": 50,
        "rank_weight": 3.0,
        "rank_weight_topk_weight_mode": "flat",
        "time_decay_half_life": 1.5,
        # algorithm + 训练超参
        "algorithm": "xgboost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "gamma": 0.2,
        "random_state": 7,
        "early_stopping_rounds": 100,
        "early_stopping_metric": "rank_ic_daily",
        "min_best_iteration": 5,
        "objective": "lambdarank",
        # 集成 3 键(ensemble_seeds 由 config 顶层承载)
        "ensemble_offsets": 1,
        "ensemble_seed_keep_top_ratio": 0.5,
        "ensemble_seed_keep_min_models": 4,
    }
    config = ModelConfig(
        model_family="stock_selection",
        feature_columns=("a",),
        label_name="neu_y_ret_20",
        hyperparams=hyperparams,
        train_window_months=72,
        ensemble_seeds=(42, 61, 82),
        label_spec=label_spec,
        feature_recipe=feature_recipe,
        split_spec=split_spec,
        domain="main",
    )
    expected: Dict[str, Any] = {
        "label_column": "neu_y_ret_20",
        "stock_domain": "main",
        "ensemble_seeds": (42, 61, 82),
        **label_spec,
        **split_spec,
        **feature_recipe,
        **hyperparams,
    }
    return config, expected


class TestParamsFromConfig:
    """``params_from_config``(ModelConfig → TrainFoldParams)映射口径(§13.3.1)。"""

    def test_full_field_mapping(self):
        """全量键输入:61 字段逐键值断言,且期望字典与字段集合完全相等(无漏断言)。"""
        config, expected = _full_mapping_case()
        params = params_from_config(config)
        field_names = {f.name for f in dataclasses.fields(TrainFoldParams)}
        assert len(field_names) == 61, "TrainFoldParams 字段数变化,需同步本测试"
        assert set(expected) == field_names, "期望字典未覆盖全部 TrainFoldParams 字段"
        mismatches = {
            name: (expected[name], getattr(params, name))
            for name in sorted(field_names)
            if getattr(params, name) != expected[name]
        }
        assert mismatches == {}
        assert isinstance(params.ensemble_seeds, tuple)

    def test_defaults_when_specs_missing(self):
        """label_spec/feature_recipe/split_spec=None + hyperparams={} ⇒ 全字段 = 默认值。"""
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=("a",),
            label_name="y_ret_5",
            hyperparams={},
            train_window_months=72,
            label_spec=None,
            feature_recipe=None,
            split_spec=None,
        )
        params = params_from_config(config)
        defaults = TrainFoldParams()
        mismatches = {
            f.name: (getattr(defaults, f.name), getattr(params, f.name))
            for f in dataclasses.fields(TrainFoldParams)
            if getattr(params, f.name) != getattr(defaults, f.name)
        }
        assert mismatches == {}

    def test_empty_ensemble_seeds_becomes_none(self):
        """config.ensemble_seeds=() ⇒ params.ensemble_seeds is None(单模型语义)。"""
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=("a",),
            label_name="y_ret_5",
            hyperparams={},
            train_window_months=72,
            ensemble_seeds=(),
        )
        assert params_from_config(config).ensemble_seeds is None

    def test_domain_mapping(self):
        """config.domain → stock_domain 透传(主用例 domain="main" 之外的非默认锚点)。"""
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=("a",),
            label_name="y_ret_5",
            hyperparams={},
            train_window_months=72,
            domain="main_gem",
        )
        assert params_from_config(config).stock_domain == "main_gem"


class TestDefaultsMatchCli:
    """``TrainFoldParams`` 默认值 = 旧 cli.py parser 默认值逐项一致(迁移对账门)。"""

    def test_defaults_match_parser(self):
        parser = build_walk_forward_parser()
        cli_defaults = {
            action.dest: action.default
            for action in parser._actions
            if action.dest != "help"
        }
        params = TrainFoldParams()
        field_names = {f.name for f in dataclasses.fields(TrainFoldParams)}
        shared = sorted(field_names & set(cli_defaults))
        # 交集下限断言:防两侧改名漂移导致本测试空转
        assert len(shared) >= 55, f"同名字段交集过小({len(shared)}),疑似改名漂移"
        mismatches = {
            name: (cli_defaults[name], getattr(params, name))
            for name in shared
            if cli_defaults[name] != getattr(params, name)
        }
        assert mismatches == {}


class TestFeatureFlagParamsParity:
    """旧(SimpleNamespace + getattr 缺省) vs 新(TrainFoldParams)特征开关 13 键全等。"""

    #: 13 键清单(顺序 = 两侧实现返回 dict 的插入序,逐字一致)
    EXPECTED_KEYS = [
        "stock_domain",
        "enable_consensus_features",
        "enable_cashflow_quality_features",
        "enable_consensus_revision_features",
        "enable_dividend_policy_features",
        "enable_holdertrade_features",
        "holdertrade_feature_set",
        "enable_repurchase_features",
        "repurchase_feature_set",
        "enable_top10fh_features",
        "top10fh_feature_set",
        "enable_top_inst_features",
        "enable_availability_markers",
    ]

    def test_default_state(self):
        """缺省态:旧侧 getattr 默认值 vs 新侧 dataclass 默认值。"""
        legacy = legacy_build_feature_flag_train_params(SimpleNamespace())
        v2 = v2_build_feature_flag_train_params(TrainFoldParams())
        assert legacy == v2
        assert list(legacy) == self.EXPECTED_KEYS
        assert not any(v for k, v in legacy.items() if k.startswith("enable_"))

    def test_all_true_state(self):
        """全开态:9 个 enable_* 全 True + 列集/域非默认取值。"""
        flags = {
            "stock_domain": "main_gem",
            "enable_consensus_features": True,
            "enable_cashflow_quality_features": True,
            "enable_consensus_revision_features": True,
            "enable_dividend_policy_features": True,
            "enable_holdertrade_features": True,
            "holdertrade_feature_set": "core",
            "enable_repurchase_features": True,
            "repurchase_feature_set": "headroom",
            "enable_top10fh_features": True,
            "top10fh_feature_set": "concentration",
            "enable_top_inst_features": True,
            "enable_availability_markers": True,
        }
        legacy = legacy_build_feature_flag_train_params(SimpleNamespace(**flags))
        v2 = v2_build_feature_flag_train_params(TrainFoldParams(**flags))
        assert legacy == v2 == flags


class TestExplicitSplitDatesMath:
    """``resolve_explicit_split_dates`` 显式换算与 ``split_train_val_by_date`` 同数学。"""

    @staticmethod
    def _synthetic_trade_dates(n: int = 120) -> List[str]:
        """120 个合成交易日(工作日序列,YYYYMMDD 字符串,序即字典序)。"""
        days = pd.date_range("2024-01-02", periods=n, freq="B")
        return [d.strftime("%Y%m%d") for d in days]

    def test_matches_ratio_split(self):
        """合成 120 交易日 × 每日 3 行:显式缝 == 比例切分 stats 缝,并锚定手算下标。"""
        trade_dates = self._synthetic_trade_dates()
        df = pd.DataFrame(
            {
                "trade_date": [d for d in trade_dates for _ in range(3)],
                "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"] * len(trade_dates),
                "feature_a": 1.0,
            }
        )
        _, _, stats = split_train_val_by_date(df, val_ratio=0.2, delta=21)
        train_start, train_end, val_start, val_end = resolve_explicit_split_dates(
            trade_dates[0], trade_dates[-1], 0.2, trade_dates, 21
        )
        # 主断言:显式换算缝 == 比例切分统计缝
        assert train_end == stats["train_end_date"]
        assert val_start == stats["val_start_date"]
        assert train_start == stats["train_start_date"] == trade_dates[0]
        assert val_end == stats["val_end_date"] == trade_dates[-1]
        # 数学锚点:n=120 ⇒ n_val=ceil(120*0.2)=24 ⇒ n_train=96;
        # train_end = D[96-21-1] = D[74],val_start = D[96+21] = D[117]
        assert train_end == trade_dates[74]
        assert val_start == trade_dates[117]


class TestDualSourceConstants:
    """replay(hosts) 与 data_adapter(train) 双源常量等价锁定(hosts 不依赖 train,防漂移)。"""

    def test_feature_source_constants(self):
        assert replay.FEATURE_SOURCE_PANEL == data_adapter.SOURCE_PANEL == "panel"
        assert (
            replay.FEATURE_SOURCE_CS_TRAIN_VERIFY
            == data_adapter.SOURCE_CS_TRAIN_VERIFY
            == "cs_train_verify"
        )

    def test_acceptance_purposes(self):
        assert replay.REPLAY_ACCEPTANCE_PURPOSES == data_adapter.ACCEPTANCE_PURPOSES
        assert replay.REPLAY_ACCEPTANCE_PURPOSES == frozenset(
            {"g_code_parity", "regression_edge"}
        )


class TestRetiredSwitches:
    """``reject_retired_switches``:退役开关(exposure 政策层 / downside_penalty)显式拒绝。"""

    RETIRED_CASES = {
        "exposure_table": {"a": 1.0},
        "exposure_policy": "x",
        "exposure_replenish": True,
        "exposure_trim_tolerance": 0.1,
        "exposure_budget_discount_replenish": True,
        "policy_model_root": "p",
        "policy_arm_suffix": "s",
        "policy_coverage_start": "20240101",
        "downside_penalty": 0.5,
        "downside_penalty_column": "x",
    }

    def test_empty_and_default_call_passes(self):
        """全空 / 全默认值(None/0/False)调用放行。"""
        reject_retired_switches()
        reject_retired_switches(
            exposure_table=None,
            exposure_policy=None,
            exposure_replenish=False,
            exposure_trim_tolerance=None,
            exposure_budget_discount_replenish=False,
            policy_model_root=None,
            policy_arm_suffix=None,
            policy_coverage_start=None,
            downside_penalty=0.0,
            downside_penalty_column=None,
        )

    @pytest.mark.parametrize("key", sorted(RETIRED_CASES))
    def test_each_retired_key_rejected(self, key: str):
        """逐键非空 ⇒ ValueError 且消息含「已退役」。"""
        with pytest.raises(ValueError, match="已退役"):
            reject_retired_switches(**{key: self.RETIRED_CASES[key]})


class TestLegacyModulesUntouched:
    """旧模块零改动守卫:git 工作区不得出现旧训练链路改动,旧文件须存在(防误删)。

    豁免:``chain_metrics.py``(T2 已 shim 化,既有提交内容);git diff / status
    只看工作区未提交改动。
    """

    FORBIDDEN_PREFIX = "src/lazybull/ml/walk_forward/"
    ALLOWED_SHIM = "src/lazybull/ml/walk_forward/chain_metrics.py"
    LEGACY_FILES = (
        "src/lazybull/ml/walk_forward/cli.py",
        "src/lazybull/ml/walk_forward/training_core.py",
        "src/lazybull/ml/walk_forward/chain_metrics.py",
        "src/lazybull/ml/run_logger.py",
    )

    @staticmethod
    def _git_changed_paths() -> List[str]:
        """工作区未提交改动路径(git diff --name-only + git status --porcelain)。"""
        paths: List[str] = []
        for argv in (["diff", "--name-only"], ["status", "--porcelain"]):
            result = subprocess.run(
                ["git", *argv],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                # porcelain 行格式 = 两字符状态 + 空格 + 路径(可能含 "old -> new")
                path = line if argv[0] == "diff" else line[3:]
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                paths.append(path.strip().strip('"'))
        return paths

    def test_legacy_files_exist(self):
        """旧文件在工作区存在(防误删)。"""
        missing = [rel for rel in self.LEGACY_FILES if not (PROJECT_ROOT / rel).is_file()]
        assert missing == [], f"旧文件缺失(防误删): {missing}"

    def test_no_uncommitted_changes_in_legacy_modules(self):
        """工作区未提交改动不得触碰旧训练链路(chain_metrics.py shim 豁免)。"""
        changed = self._git_changed_paths()
        hits = [
            path
            for path in changed
            if (path.startswith(self.FORBIDDEN_PREFIX) and path != self.ALLOWED_SHIM)
            or path == "src/lazybull/ml/run_logger.py"
        ]
        assert hits == [], f"旧训练链路存在工作区未提交改动(零改动守卫): {hits}"


class TestSplitFunctionsParity:
    """切分函数拆分等价永久 parity(P3-T3 评审处置 reviewer1-P3-3:

    §17.8 提取式拆分(闭包提升模块级 + 循环体提取)的行为等价固化;
    评审专家独立对跑 297 组零差异,本测试以参数化网格固化为回归门)。
    """

    @staticmethod
    def _cal() -> "pd.DataFrame":
        import pandas as pd

        days = pd.bdate_range("20180101", "20261231").strftime("%Y%m%d").tolist()
        return pd.DataFrame({"cal_date": days, "is_open": 1})

    @staticmethod
    def _norm(splits) -> list:
        return [
            (s.split_index, s.train_start, s.train_end, s.test_start, s.test_end)
            for s in splits
        ]

    @pytest.mark.parametrize("freq", ["monthly", "quarterly", "semiannual", "yearly"])
    @pytest.mark.parametrize("rebalance_freq", [None, 20])
    def test_forward_splits_parity(self, freq, rebalance_freq):
        from src.lazybull.ml.walk_forward import utils as old_u
        from src.lazybull.v2.train.walk_forward import utils as new_u

        cal = self._cal()
        old = old_u.generate_walk_forward_splits(
            cal, "20190101", "20250101", step_frequency=freq,
            train_window_years=5, rebalance_freq=rebalance_freq,
        )
        new = new_u.generate_walk_forward_splits(
            cal, "20190101", "20250101", step_frequency=freq,
            train_window_years=5, rebalance_freq=rebalance_freq,
        )
        assert self._norm(old) == self._norm(new)

    @pytest.mark.parametrize("split_count", [1, 3, 8, 14])
    @pytest.mark.parametrize("rebalance_freq", [None, 20])
    @pytest.mark.parametrize("train_window_years", [5, 6])
    def test_by_count_splits_parity(self, split_count, rebalance_freq, train_window_years):
        from src.lazybull.ml.walk_forward import utils as old_u
        from src.lazybull.v2.train.walk_forward import utils as new_u

        cal = self._cal()
        old = old_u.generate_walk_forward_splits_by_count(
            cal, split_count, "20260105", train_window_years=train_window_years,
            rebalance_freq=rebalance_freq,
        )
        new = new_u.generate_walk_forward_splits_by_count(
            cal, split_count, "20260105", train_window_years=train_window_years,
            rebalance_freq=rebalance_freq,
        )
        assert self._norm(old) == self._norm(new)

    def test_error_paths_parity(self):
        from src.lazybull.ml.walk_forward import utils as old_u
        from src.lazybull.v2.train.walk_forward import utils as new_u

        cal = self._cal()
        for caller in (old_u, new_u):
            with pytest.raises(ValueError):
                caller.generate_walk_forward_splits_by_count(cal, 0, "20260105")
            with pytest.raises(ValueError):
                caller.generate_walk_forward_splits(
                    cal, "20190101", "20250101", step_frequency="bad"
                )
        assert old_u.resolve_deploy_train_window(cal, "20260105", 6) == (
            new_u.resolve_deploy_train_window(cal, "20260105", 6)
        )
