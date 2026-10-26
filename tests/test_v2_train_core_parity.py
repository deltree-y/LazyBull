# -*- coding: utf-8 -*-
"""v2 P3-T2 train_core 同码重放 parity 测试（T2 主验收；全合成数据，不触碰真实 data/）。

核心断言：旧 ``src.lazybull.ml.train_core.prepare.prepare_training_data`` 与新
``src.lazybull.v2.train.core.prepare.prepare_training_data`` 同参对跑，返回九元组
``(X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split,
data_stats, df_val_split_original)`` 逐位一致（DataFrame/Series 用
``check_exact=True``，data_stats 深比较）。

覆盖：默认参数 / 全声明族+三哨兵族 / 标签变换 / 稳定性筛选 / 因子精简联动剔除 /
特征列 override 补 NaN / freshness 三策略 / 运行时派生 DI parity 与负例 /
显式日期缝 / xgb 训练逐位一致。

评审处置增补：真实 factors 组装 RuntimeDerivers 的 DI parity（reviewer1-P1-1）、
label_transform_fn 复制型变换前帧 weakref 释放（R-CODEX-03 / reviewer1-P2-1）、
拆分助手旧新冒烟（reviewer1-P3-8：ensemble / weights / labels / eval / xgb lambdarank）。
"""

import functools
import gc
import json
import weakref
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml.train_core.labels import transform_labels_cs_zscore
from src.lazybull.ml.train_core.prepare import prepare_training_data as legacy_prepare_training_data
from src.lazybull.v2.common.freshness import (
    FRESHNESS_STRATEGY_DROP_ALL,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
)
from src.lazybull.v2.common.model_schema import (
    CASHFLOW_QUALITY_SCHEMA_VERSION,
    CASHFLOW_QUALITY_VERSION_COL,
    CONSENSUS_REVISION_SCHEMA_VERSION,
    CONSENSUS_REVISION_VERSION_COL,
    DIVIDEND_POLICY_SCHEMA_VERSION,
    DIVIDEND_POLICY_VERSION_COL,
)
from src.lazybull.v2.train.core import constants as v2_constants
from src.lazybull.v2.train.core.prepare import prepare_training_data as v2_prepare_training_data
from src.lazybull.v2.train.core.runtime_derive import (
    RuntimeDerivers,
    RuntimeFamilyDeriver,
)

LABEL_COLUMN = "neu_y_ret_20"

#: 基础特征列（与 prepare._base_feature_columns 字面量一致；缺失会在矩阵化时 KeyError）
BASE_FEATURE_COLUMNS = [
    "neu_ret_1",
    "neu_ret_20",
    "neu_ret_5",
    "alpha_industry_20",
    "alpha_industry_5",
    "ind_ret_avg",
    "ind_momentum_rank",
    "zscore_ma_deviation_20",
    "zscore_acceleration",
    "zscore_macd_hist",
    "bb_pct",
    "zscore_turnover_rate",
    "vol_ratio_20",
    "vol_burst_20",
    "zscore_amount_ma20",
    "zscore_net_mf_amount",
    "zscore_elg_net_amount_sum_20",
    "lg_net_amount_sum_5",
    "zscore_volatility_20",
    "zscore_volatility_5",
    "amplitude",
    "zscore_bb_width",
    "upper_shadow",
    "lower_shadow",
    "spec_score",
    "rsi_14",
    "kdj_j",
    "zscore_size",
    "zscore_bp",
    "zscore_dv_ttm",
    "zscore_pe_ttm",
    "is_loss",
    "list_days",
    "mkt_adv_dec_ratio",
    "mkt_ret_avg_20",
    "mkt_turnover_std",
    "mkt_vol_20",
]

#: 九个声明式族列清单（v2/train/core/constants 为唯一来源）
_DECLARED_FAMILY_COLUMNS = (
    v2_constants.FUNDAMENTAL_FEATURE_COLUMNS
    + v2_constants.ALT_FEATURE_COLUMNS
    + v2_constants.MARGIN_FEATURE_COLUMNS
    + v2_constants.CYQ_FEATURE_COLUMNS
    + v2_constants.FUND_FEATURE_COLUMNS
    + v2_constants.EXPRESS_FEATURE_COLUMNS
    + v2_constants.ENHANCED_FEATURE_COLUMNS
    + v2_constants.NORTH_FEATURE_COLUMNS
    + v2_constants.LHB_FEATURE_COLUMNS
)

#: 用例 2 的开关集（九声明族 + 一致预期 + 三哨兵族）
_ALL_FAMILY_KWARGS: Dict[str, Any] = {
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
}


# ---------------------------------------------------------------------------
# 合成数据构建器
# ---------------------------------------------------------------------------


def _base_frame(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """基础合成帧：300 工作日 × 8 代码，37 基础列 + 标签（5% NaN）+ 过滤列全 False。

    300 日使 val-raw 段（60 日）在扣除 21 日 embargo 后仍有非空 ES/calibration 子集
    （150 日时 ES 子集为空，label_transform 只调一次、xgb 早停路径无法真实触发）。
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y%m%d").tolist()
    codes = [f"00000{i}.SZ" for i in range(1, 9)]
    n_rows = len(days) * len(codes)
    df = pd.DataFrame(
        {
            "trade_date": np.repeat(days, len(codes)),
            "ts_code": codes * len(days),
        }
    )
    for col in BASE_FEATURE_COLUMNS:
        df[col] = rng.standard_normal(n_rows)
    label = rng.standard_normal(n_rows) * 0.05
    label[rng.random(n_rows) < 0.05] = np.nan
    df[LABEL_COLUMN] = label
    df["is_st"] = False
    df["is_suspended"] = False
    df["is_limit_up"] = False
    return df


def _add_random_columns(df: pd.DataFrame, columns: List[str], rng: np.random.Generator) -> None:
    """按列名批量填列：freshness 列填正值浮点，其余填标准正态。"""
    n_rows = len(df)
    for col in dict.fromkeys(columns):
        if "freshness" in col:
            df[col] = rng.uniform(1.0, 200.0, n_rows)
        else:
            df[col] = rng.standard_normal(n_rows)


def _full_frame(seed: int = 42) -> pd.DataFrame:
    """全族合成帧：基础列 + 九声明族 + 一致预期 + 三哨兵族（含哨兵列）+ 缺失标记 + _sz。"""
    df = _base_frame(seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_rows = len(df)
    _add_random_columns(df, _DECLARED_FAMILY_COLUMNS, rng)
    _add_random_columns(df, v2_constants.CONSENSUS_FEATURE_COLUMNS, rng)
    _add_random_columns(df, v2_constants.CASHFLOW_QUALITY_FEATURE_COLUMNS, rng)
    _add_random_columns(df, v2_constants.CONSENSUS_REVISION_FEATURE_COLUMNS, rng)
    _add_random_columns(df, v2_constants.DIVIDEND_POLICY_FEATURE_COLUMNS, rng)
    # 估值缺失标记（按列存在性自动入模）
    df["dv_ttm_missing"] = (rng.random(n_rows) < 0.1).astype(float)
    df["pe_ttm_missing"] = (rng.random(n_rows) < 0.1).astype(float)
    # 三哨兵列（版本取自 v2.common.model_schema 真值）
    df[CASHFLOW_QUALITY_VERSION_COL] = CASHFLOW_QUALITY_SCHEMA_VERSION
    df[CONSENSUS_REVISION_VERSION_COL] = CONSENSUS_REVISION_SCHEMA_VERSION
    df[DIVIDEND_POLICY_VERSION_COL] = DIVIDEND_POLICY_SCHEMA_VERSION
    # 市值中性化派生列（zscore_*_sz，基础列在列时自动发现入模）
    df["zscore_ocf_to_profit_sz"] = rng.standard_normal(n_rows)
    df["zscore_bp_sz"] = rng.standard_normal(n_rows)
    df["zscore_size_sz"] = rng.standard_normal(n_rows)
    return df


def _freshness_frame(seed: int = 42) -> pd.DataFrame:
    """freshness 用例帧：基础列 + fundamental/alt/express/consensus 四族（含 freshness 列）。"""
    df = _base_frame(seed=seed)
    rng = np.random.default_rng(seed + 2)
    _add_random_columns(df, v2_constants.FUNDAMENTAL_FEATURE_COLUMNS, rng)
    _add_random_columns(df, v2_constants.ALT_FEATURE_COLUMNS, rng)
    _add_random_columns(df, v2_constants.EXPRESS_FEATURE_COLUMNS, rng)
    _add_random_columns(df, v2_constants.CONSENSUS_FEATURE_COLUMNS, rng)
    return df


# ---------------------------------------------------------------------------
# 九元组 parity 比较器
# ---------------------------------------------------------------------------


def _assert_stats_value_equal(old_val: Any, new_val: Any, path: str) -> None:
    """data_stats 深比较：dict/list/tuple 递归；float 处理 NaN；bool 与数值分流。"""
    if isinstance(old_val, dict):
        assert isinstance(new_val, dict), f"{path}: 类型不一致 {type(old_val)} vs {type(new_val)}"
        assert set(old_val) == set(
            new_val
        ), f"{path}: 键集不一致 {sorted(set(old_val) ^ set(new_val))}"
        for key in old_val:
            _assert_stats_value_equal(old_val[key], new_val[key], f"{path}.{key}")
        return
    if isinstance(old_val, (list, tuple)):
        assert isinstance(new_val, (list, tuple)), f"{path}: 类型不一致"
        assert len(old_val) == len(new_val), f"{path}: 长度不一致 {len(old_val)} vs {len(new_val)}"
        for idx, (left, right) in enumerate(zip(old_val, new_val)):
            _assert_stats_value_equal(left, right, f"{path}[{idx}]")
        return
    if old_val is None or new_val is None:
        assert old_val is None and new_val is None, f"{path}: {old_val!r} vs {new_val!r}"
        return
    if isinstance(old_val, bool) or isinstance(new_val, bool):
        assert isinstance(old_val, bool) and isinstance(new_val, bool), f"{path}: bool 类型不一致"
        assert old_val == new_val, f"{path}: {old_val!r} != {new_val!r}"
        return
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(old_val, numeric_types) and isinstance(new_val, numeric_types):
        if pd.isna(old_val) and pd.isna(new_val):
            return
        assert float(old_val) == float(new_val), f"{path}: {old_val!r} != {new_val!r}"
        return
    assert old_val == new_val, f"{path}: {old_val!r} != {new_val!r}"


def assert_prepare_parity(old_result: Tuple, new_result: Tuple) -> None:
    """九元组逐位一致断言（DataFrame/Series check_exact=True，data_stats 深比较）。"""
    assert len(old_result) == 9, f"旧实现返回元数异常: {len(old_result)}"
    assert len(new_result) == 9, f"新实现返回元数异常: {len(new_result)}"
    (
        old_X_train,
        old_y_train,
        old_X_val,
        old_y_val,
        old_feature_columns,
        old_df_train,
        old_df_val,
        old_stats,
        old_df_val_original,
    ) = old_result
    (
        new_X_train,
        new_y_train,
        new_X_val,
        new_y_val,
        new_feature_columns,
        new_df_train,
        new_df_val,
        new_stats,
        new_df_val_original,
    ) = new_result

    pd.testing.assert_frame_equal(old_X_train, new_X_train, check_exact=True)
    pd.testing.assert_series_equal(old_y_train, new_y_train, check_exact=True)
    pd.testing.assert_frame_equal(old_X_val, new_X_val, check_exact=True)
    pd.testing.assert_series_equal(old_y_val, new_y_val, check_exact=True)
    assert old_feature_columns == new_feature_columns, "feature_columns 不一致"
    pd.testing.assert_frame_equal(old_df_train, new_df_train, check_exact=True)
    pd.testing.assert_frame_equal(old_df_val, new_df_val, check_exact=True)
    pd.testing.assert_frame_equal(old_df_val_original, new_df_val_original, check_exact=True)
    assert set(old_stats) == set(
        new_stats
    ), f"data_stats 键集不一致: {sorted(set(old_stats) ^ set(new_stats))}"
    for key in old_stats:
        _assert_stats_value_equal(old_stats[key], new_stats[key], f"data_stats.{key}")


def _run_both(df: pd.DataFrame, **kwargs) -> Tuple[Tuple, Tuple]:
    """旧新同参对跑（各自吃一份深拷贝，防派生写列串扰）。"""
    old_result = legacy_prepare_training_data(df.copy(), **kwargs)
    new_result = v2_prepare_training_data(df.copy(), **kwargs)
    return old_result, new_result


# ---------------------------------------------------------------------------
# prepare parity 用例
# ---------------------------------------------------------------------------


class TestPrepareParity:
    def test_default_params(self):
        """用例 1：默认参数（仅 label_column/val_ratio）。"""
        df = _base_frame()
        old_result, new_result = _run_both(df, label_column=LABEL_COLUMN, val_ratio=0.2)
        assert_prepare_parity(old_result, new_result)

    def test_all_families_with_sentinels(self):
        """用例 2：全声明族 + 一致预期 + 三哨兵族 + 缺失标记。"""
        df = _full_frame()
        old_result, new_result = _run_both(
            df, label_column=LABEL_COLUMN, val_ratio=0.2, **_ALL_FAMILY_KWARGS
        )
        assert_prepare_parity(old_result, new_result)
        feature_columns = new_result[4]
        # 哨兵族入模与别名去重语义顺带锁定（新侧）
        assert "zscore_ocf_to_profit" in feature_columns
        assert "zscore_dividend_payout_ratio" in feature_columns
        assert "zscore_cons_eps_revision_accel" in feature_columns
        assert "zscore_cf_nm" not in feature_columns  # zscore_ocf_to_profit 的构建别名
        assert "zscore_bp_sz" in feature_columns  # _sz 自动发现

    def test_label_transform_fn(self):
        """用例 3：label_transform_fn（两侧传同一函数对象，partial 绑定列名与 winsorize_p）。"""
        df = _base_frame()
        transform_fn = functools.partial(
            transform_labels_cs_zscore, label_column=LABEL_COLUMN, winsorize_p=0.01
        )
        old_result, new_result = _run_both(
            df,
            label_column=LABEL_COLUMN,
            val_ratio=0.2,
            label_transform_fn=transform_fn,
        )
        assert_prepare_parity(old_result, new_result)

    def test_feature_stability_filter(self):
        """用例 4：feature_stability_filter=True。"""
        df = _base_frame()
        old_result, new_result = _run_both(
            df,
            label_column=LABEL_COLUMN,
            val_ratio=0.2,
            feature_stability_filter=True,
        )
        assert_prepare_parity(old_result, new_result)
        # 8 代码/日 < 30 样本下限 ⇒ 各段 IC 为空记 0 ⇒ 全部特征被筛除（两侧一致）
        assert new_result[7]["feature_filter_info"]["skipped"] is False

    def test_factor_prune(self, tmp_path):
        """用例 5：factor_prune + 显式排除清单 ⇒ zscore_bp 与 zscore_bp_sz 联动剔除。"""
        df = _full_frame()
        exclude_file = tmp_path / "factor_exclude_list.json"
        exclude_file.write_text(json.dumps({"exclude_factors": ["zscore_bp"]}), encoding="utf-8")
        old_result, new_result = _run_both(
            df,
            label_column=LABEL_COLUMN,
            val_ratio=0.2,
            factor_prune=True,
            factor_exclude_file=str(exclude_file),
        )
        assert_prepare_parity(old_result, new_result)
        feature_columns = new_result[4]
        assert "zscore_bp" not in feature_columns
        assert "zscore_bp_sz" not in feature_columns  # 联动剔除
        assert "zscore_size" in feature_columns  # 未排除者保留

    def test_feature_columns_override(self):
        """用例 6：feature_columns_override 含 2 个数据中不存在的列 ⇒ 补 NaN 路径。"""
        df = _base_frame()
        override = ["neu_ret_1", "neu_ret_5", "ghost_col_alpha", "ghost_col_beta"]
        old_result, new_result = _run_both(
            df,
            label_column=LABEL_COLUMN,
            val_ratio=0.2,
            feature_columns_override=override,
        )
        assert_prepare_parity(old_result, new_result)
        assert new_result[4] == override
        assert new_result[0]["ghost_col_alpha"].isna().all()
        assert new_result[0]["ghost_col_beta"].isna().all()

    @pytest.mark.parametrize(
        "strategy",
        [
            FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
            FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY,
            FRESHNESS_STRATEGY_DROP_ALL,
        ],
    )
    def test_freshness_strategies(self, strategy):
        """用例 7：freshness 三策略（df 含状态型 + 事件型 freshness 列）。"""
        df = _freshness_frame()
        old_result, new_result = _run_both(
            df,
            label_column=LABEL_COLUMN,
            val_ratio=0.2,
            enable_fundamental_features=True,
            enable_alt_features=True,
            enable_express_features=True,
            enable_consensus_features=True,
            freshness_strategy=strategy,
        )
        assert_prepare_parity(old_result, new_result)
        feature_columns = new_result[4]
        if strategy == FRESHNESS_STRATEGY_DROP_ALL:
            assert not any("freshness" in col for col in feature_columns)
        else:
            # 状态型保留；事件型 freshness 列移除（decay 策略另用于值列衰减）
            assert "fundamental_freshness_days" in feature_columns
            assert "forecast_freshness_days" not in feature_columns
            assert "express_freshness_days" not in feature_columns
            assert "consensus_freshness_days" not in feature_columns


# ---------------------------------------------------------------------------
# 运行时派生 DI（用例 8/9）
# ---------------------------------------------------------------------------

#: 四族 stub 列集（哨兵列不入列集，由 derive 单独写入；取真实家族列名子集）
_STUB_FAMILY_COLUMNS = {
    "holdertrade": ["ht_net_ratio_30d", "ht_buy_count_30d"],
    "repurchase": ["rp_amount_to_mv_90d", "rp_price_headroom"],
    "top10fh": ["tfh_top10_ratio", "tfh_inst_ratio"],
    "top_inst": ["ti_inst_net_ratio"],
}


def _family_stub(version_col: str, schema_version: int, columns: List[str], no_arg: bool = False):
    """构造单族 stub：derive_fn 把 wanted 列写成 0.01*(i+1)、写哨兵列、返回 wanted。

    no_arg=True 时 feature_columns_fn 为无参（对齐真实 factors.top_inst 的无参签名；
    其余三族带 feature_set 单参）。
    """
    calls = {"derive": 0}

    def derive_fn(df, source, wanted, log_prefix=""):
        calls["derive"] += 1
        for idx, col in enumerate(wanted):
            df[col] = 0.01 * (idx + 1)
        df[version_col] = schema_version
        return list(wanted)

    if no_arg:

        def feature_columns_fn() -> List[str]:
            return list(columns)

    else:

        def feature_columns_fn(feature_set: str = "full") -> List[str]:
            return list(columns)

    return calls, derive_fn, feature_columns_fn


def _availability_stub():
    """可用性标记 stub：写 avail_m1 列（0/1 交替，避免被常数门禁移除）并返回列名。"""
    calls = {"derive": 0}

    def stub(df) -> List[str]:
        calls["derive"] += 1
        df["avail_m1"] = (np.arange(len(df)) % 2).astype(float)
        return ["avail_m1"]

    return calls, stub


def _build_runtime_stubs():
    """构建四族 + availability 的双份 stub（旧路径 monkeypatch 用与新路径 DI 用各一份）。

    哨兵 version_col/schema_version 取 factors 侧真值常量，保证与旧实现的
    哨兵校验（直接读 factors 常量）语义一致。
    """
    from src.lazybull.factors.holdertrade import (
        HOLDERTRADE_SCHEMA_VERSION,
        HOLDERTRADE_VERSION_COL,
    )
    from src.lazybull.factors.repurchase import (
        REPURCHASE_SCHEMA_VERSION,
        REPURCHASE_VERSION_COL,
    )
    from src.lazybull.factors.top10_floatholders import (
        TOP10FH_SCHEMA_VERSION,
        TOP10FH_VERSION_COL,
    )
    from src.lazybull.factors.top_inst import (
        TOP_INST_SCHEMA_VERSION,
        TOP_INST_VERSION_COL,
    )

    sentinel = {
        "holdertrade": (HOLDERTRADE_VERSION_COL, HOLDERTRADE_SCHEMA_VERSION),
        "repurchase": (REPURCHASE_VERSION_COL, REPURCHASE_SCHEMA_VERSION),
        "top10fh": (TOP10FH_VERSION_COL, TOP10FH_SCHEMA_VERSION),
        "top_inst": (TOP_INST_VERSION_COL, TOP_INST_SCHEMA_VERSION),
    }
    stubs: Dict[str, Dict[str, Any]] = {}
    for family, (version_col, schema_version) in sentinel.items():
        for side in ("old", "new"):
            calls, derive_fn, columns_fn = _family_stub(
                version_col,
                schema_version,
                _STUB_FAMILY_COLUMNS[family],
                # top_inst 真实 feature_columns_fn 为无参签名（prepare_families 对齐）
                no_arg=(family == "top_inst"),
            )
            stubs.setdefault(family, {})[side] = {
                "calls": calls,
                "derive_fn": derive_fn,
                "columns_fn": columns_fn,
                "version_col": version_col,
                "schema_version": schema_version,
            }
    avail_old_calls, avail_old = _availability_stub()
    avail_new_calls, avail_new = _availability_stub()
    stubs["availability"] = {
        "old": {"calls": avail_old_calls, "derive_fn": avail_old},
        "new": {"calls": avail_new_calls, "derive_fn": avail_new},
    }
    return stubs


def _make_derivers(stubs: Dict[str, Dict[str, Any]]) -> RuntimeDerivers:
    """由 new 侧 stub 组装 RuntimeDerivers 注入容器。"""

    def family_deriver(family: str) -> RuntimeFamilyDeriver:
        stub = stubs[family]["new"]
        return RuntimeFamilyDeriver(
            version_col=stub["version_col"],
            schema_version=stub["schema_version"],
            feature_columns_fn=stub["columns_fn"],
            derive_fn=stub["derive_fn"],
        )

    return RuntimeDerivers(
        holdertrade=family_deriver("holdertrade"),
        repurchase=family_deriver("repurchase"),
        top10fh=family_deriver("top10fh"),
        top_inst=family_deriver("top_inst"),
        availability=stubs["availability"]["new"]["derive_fn"],
    )


class TestRuntimeDeriveDI:
    def test_runtime_derive_parity(self, monkeypatch):
        """用例 8：运行时派生五族 DI parity（旧 monkeypatch factors vs 新注入 derivers）。"""
        df = _base_frame()
        stubs = _build_runtime_stubs()
        derivers = _make_derivers(stubs)

        # 旧路径：monkeypatch factors 四族 derive/列集函数 + prepare 内 availability 引用
        monkeypatch.setattr(
            "src.lazybull.factors.holdertrade.derive_holdertrade_columns",
            stubs["holdertrade"]["old"]["derive_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.holdertrade.holdertrade_feature_columns",
            stubs["holdertrade"]["old"]["columns_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.repurchase.derive_repurchase_columns",
            stubs["repurchase"]["old"]["derive_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.repurchase.repurchase_feature_columns",
            stubs["repurchase"]["old"]["columns_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.top10_floatholders.derive_top10fh_columns",
            stubs["top10fh"]["old"]["derive_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.top10_floatholders.top10fh_feature_columns",
            stubs["top10fh"]["old"]["columns_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.top_inst.derive_top_inst_columns",
            stubs["top_inst"]["old"]["derive_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.factors.top_inst.top_inst_feature_columns",
            stubs["top_inst"]["old"]["columns_fn"],
        )
        monkeypatch.setattr(
            "src.lazybull.ml.train_core.prepare.derive_availability_markers",
            stubs["availability"]["old"]["derive_fn"],
        )

        common_kwargs: Dict[str, Any] = {
            "label_column": LABEL_COLUMN,
            "val_ratio": 0.2,
            "enable_holdertrade_features": True,
            "holdertrade_lookup": {},
            "enable_repurchase_features": True,
            "repurchase_lookup": {},
            "enable_top10fh_features": True,
            "top10fh_panel": pd.DataFrame(),
            "enable_top_inst_features": True,
            "top_inst_lookup": {},
            "enable_availability_markers": True,
        }
        old_result = legacy_prepare_training_data(df.copy(), **common_kwargs)
        new_result = v2_prepare_training_data(df.copy(), runtime_derivers=derivers, **common_kwargs)
        assert_prepare_parity(old_result, new_result)

        # 双侧派生均被真实调用一次（证明 parity 不是「两边都没派生」的空转）
        for family in ("holdertrade", "repurchase", "top10fh", "top_inst", "availability"):
            assert stubs[family]["old"]["calls"]["derive"] == 1, f"{family} 旧侧未派生"
            assert stubs[family]["new"]["calls"]["derive"] == 1, f"{family} 新侧未派生"
        # availability 标记非定值 ⇒ 存活入模
        assert "avail_m1" in new_result[4]

    def test_missing_derivers_rejected(self):
        """用例 9a：新路径启用 holdertrade + 传 lookup 但不传 runtime_derivers ⇒ ValueError。"""
        df = _base_frame()
        with pytest.raises(ValueError) as exc_info:
            v2_prepare_training_data(
                df.copy(),
                label_column=LABEL_COLUMN,
                enable_holdertrade_features=True,
                holdertrade_lookup={},
            )
        message = str(exc_info.value)
        assert "holdertrade" in message
        assert "runtime_derivers" in message

    def test_missing_lookup_rejected(self):
        """用例 9b：新路径传 derivers 但不传 holdertrade_lookup ⇒ ValueError。"""
        df = _base_frame()
        stubs = _build_runtime_stubs()
        derivers = _make_derivers(stubs)
        with pytest.raises(ValueError) as exc_info:
            v2_prepare_training_data(
                df.copy(),
                label_column=LABEL_COLUMN,
                enable_holdertrade_features=True,
                runtime_derivers=derivers,
            )
        assert "holdertrade_lookup" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 显式日期缝（用例 10）
# ---------------------------------------------------------------------------


class TestExplicitDateSeam:
    def test_explicit_split_reproduces_ratio_split(self):
        """显式日期缝：比例路径取边界 ⇒ 显式重跑九元组全等。"""
        df = _base_frame()
        ratio_result = v2_prepare_training_data(df.copy(), label_column=LABEL_COLUMN, val_ratio=0.2)
        stats = ratio_result[7]
        train_end = stats["train_end_date"]
        val_start = stats["val_raw_start_date"]
        assert train_end != "N/A" and val_start != "N/A"
        explicit_result = v2_prepare_training_data(
            df.copy(),
            label_column=LABEL_COLUMN,
            val_ratio=0.2,
            explicit_train_end=train_end,
            explicit_val_start=val_start,
        )
        assert_prepare_parity(ratio_result, explicit_result)

    def test_single_explicit_param_rejected(self):
        """只给一个显式参数 ⇒ ValueError。"""
        df = _base_frame()
        with pytest.raises(ValueError, match="必须同时给出"):
            v2_prepare_training_data(
                df.copy(), label_column=LABEL_COLUMN, explicit_train_end="20240430"
            )
        with pytest.raises(ValueError, match="必须同时给出"):
            v2_prepare_training_data(
                df.copy(), label_column=LABEL_COLUMN, explicit_val_start="20240502"
            )

    def test_inverted_explicit_bounds_rejected(self):
        """explicit_train_end >= explicit_val_start ⇒ ValueError。"""
        df = _base_frame()
        with pytest.raises(ValueError, match="倒置"):
            v2_prepare_training_data(
                df.copy(),
                label_column=LABEL_COLUMN,
                explicit_train_end="20240502",
                explicit_val_start="20240430",
            )
        with pytest.raises(ValueError, match="倒置"):
            v2_prepare_training_data(
                df.copy(),
                label_column=LABEL_COLUMN,
                explicit_train_end="20240502",
                explicit_val_start="20240502",
            )

    def test_non_yyyymmdd_explicit_bounds_rejected(self):
        """非 8 位数字边界 ⇒ ValueError。"""
        df = _base_frame()
        with pytest.raises(ValueError, match="八位数字"):
            v2_prepare_training_data(
                df.copy(),
                label_column=LABEL_COLUMN,
                explicit_train_end="2024-04-30",
                explicit_val_start="20240502",
            )
        with pytest.raises(ValueError, match="八位数字"):
            v2_prepare_training_data(
                df.copy(),
                label_column=LABEL_COLUMN,
                explicit_train_end="20240430",
                explicit_val_start="N/A",
            )


# ---------------------------------------------------------------------------
# xgb 训练 parity（用例 11）
# ---------------------------------------------------------------------------


def _cuda_training_available() -> bool:
    """探测本机 XGBoost cuda 训练可用性（train_xgboost_model 硬编码 device="cuda"）。"""
    import xgboost as xgb

    try:
        probe = xgb.XGBRegressor(n_estimators=2, tree_method="hist", device="cuda")
        probe.fit(
            pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([0.01, -0.01, 0.02, -0.02]),
            verbose=False,
        )
    except Exception:
        return False
    return True


class TestXgbParity:
    def test_train_xgboost_model_parity(self):
        """旧/新 train_xgboost_model 相同确定性超参训练 ⇒ predict 逐位一致。"""
        if not _cuda_training_available():
            pytest.skip("本机 XGBoost cuda 训练不可用，跳过 xgb parity")
        from src.lazybull.ml.train_core.xgb import train_xgboost_model as legacy_train_xgboost_model
        from src.lazybull.v2.train.core.xgb import train_xgboost_model as v2_train_xgboost_model

        df = _full_frame()
        result = v2_prepare_training_data(
            df.copy(), label_column=LABEL_COLUMN, val_ratio=0.2, **_ALL_FAMILY_KWARGS
        )
        X_train, y_train, X_val, y_val = result[0], result[1], result[2], result[3]
        X_train = X_train.iloc[:2000]
        y_train = y_train.iloc[:2000]
        # 合成数据为纯噪声 + 默认 gamma=0.1 ⇒ 树不分裂、预测恒为常数，np.array_equal
        # 判别力不足；注入确定性信号（neu_ret_1 线性项）使树真实分裂
        y_train = y_train + X_train["neu_ret_1"] * 0.3
        y_val = y_val + X_val["neu_ret_1"] * 0.3

        train_kwargs: Dict[str, Any] = {
            "task": "regression",
            "n_estimators": 20,
            "random_state": 42,
            "min_child_weight": 10,
        }
        old_model, old_train_params, _, _ = legacy_train_xgboost_model(
            X_train, y_train, X_val, y_val, **train_kwargs
        )
        new_model, new_train_params, _, _ = v2_train_xgboost_model(
            X_train, y_train, X_val, y_val, **train_kwargs
        )
        old_pred = old_model.predict(X_val)
        new_pred = new_model.predict(X_val)
        assert np.ptp(old_pred) > 0, "预测为常数（树未分裂），parity 断言判别力不足"
        assert np.array_equal(old_pred, new_pred), "旧/新 xgb 训练预测不逐位一致"
        # 关键训练超参一致（callable eval_metric 已函数名化，可直接比）
        assert old_train_params == new_train_params


# ---------------------------------------------------------------------------
# 真实 factors 组装 DI parity（reviewer1-P1-1）
# ---------------------------------------------------------------------------


def _real_derive_frame(seed: int = 42) -> pd.DataFrame:
    """真实派生 parity 用帧：基础列 + circ_mv/amount/vol（repurchase/top_inst 缺列硬报错）
    + availability 来源列（fund/cons 两列撒 NaN 使标记非常数，免被常数门禁移除）。"""
    df = _base_frame(seed=seed)
    rng = np.random.default_rng(seed + 5)
    n_rows = len(df)
    df["circ_mv"] = rng.uniform(5e4, 5e5, n_rows)  # 流通市值（万元）
    df["amount"] = rng.uniform(1e4, 1e6, n_rows)  # 成交额（千元）
    df["vol"] = rng.uniform(1e3, 1e5, n_rows)  # 成交量（手）
    fund_hold = rng.uniform(0.0, 10.0, n_rows)
    fund_hold[rng.random(n_rows) < 0.3] = np.nan
    df["fund_hold_ratio"] = fund_hold
    cons_count = rng.uniform(0.0, 5.0, n_rows)
    cons_count[rng.random(n_rows) < 0.3] = np.nan
    df["cons_analyst_count_30d"] = cons_count
    df["rzye_chg_5"] = rng.standard_normal(n_rows)
    df["express_roe"] = rng.standard_normal(n_rows)
    return df


def _real_holdertrade_lookup(dates: List[str]):
    """微型 holdertrade raw ⇒ 真实 lookup（镜像 tests/test_holdertrade_runtime.py 的最小构造）。

    每代码 3 个公告（0110/0315/0520），保证 ht_freshness_days 在窗口内非空。
    """
    from src.lazybull.factors.holdertrade import build_holdertrade_lookup_by_date

    rows = []
    for idx in range(1, 9):
        code = f"00000{idx}.SZ"
        rows.append((code, "20240110", "G", "IN", 0.5 + idx * 0.1))
        rows.append((code, "20240315", "C", "DE", 0.2 + idx * 0.05))
        rows.append((code, "20240520", "P", "IN", 0.3))
    raw = pd.DataFrame(
        rows, columns=["ts_code", "ann_date", "holder_type", "in_de", "change_ratio"]
    )
    return build_holdertrade_lookup_by_date(raw, dates)


def _real_repurchase_lookup(dates: List[str]):
    """微型 repurchase raw ⇒ 真实 lookup（镜像 tests/test_repurchase_factors.py 的最小构造）。"""
    from src.lazybull.factors.repurchase import build_repurchase_lookup_by_date

    rows = []
    for idx in range(1, 9):
        code = f"00000{idx}.SZ"
        rows.append((code, "20240110", "实施", 1.0e8 + idx * 1.0e7, 1.0e6, 10.0 + idx))
        rows.append((code, "20240520", "实施", 5.0e7, 5.0e5, 12.0 + idx))
    raw = pd.DataFrame(rows, columns=["ts_code", "ann_date", "proc", "amount", "vol", "high_limit"])
    return build_repurchase_lookup_by_date(raw, dates)


def _real_top10fh_panel():
    """微型 top10fh raw ⇒ 真实报告期面板（镜像 tests/test_top10fh_wiring.py 的最小构造）。

    每代码两报告期（20230930/20231231，ann 均在训练窗口前披露），集中度环比非 NaN。
    """
    from src.lazybull.factors.top10_floatholders import build_top10fh_panel

    raw_columns = [
        "ts_code",
        "end_date",
        "ann_date",
        "holder_name",
        "hold_amount",
        "hold_ratio",
        "hold_float_ratio",
        "hold_change",
        "holder_type",
    ]
    rows = []
    for idx in range(1, 9):
        code = f"00000{idx}.SZ"
        rows.append(
            (
                code,
                "20230930",
                "20231025",
                f"社保{idx}",
                40.0 + idx,
                4.0,
                30.0 + idx,
                np.nan,
                "社保基金、社保机构",
            )
        )
        rows.append(
            (code, "20230930", "20231025", f"自然人{idx}", 20.0, 2.0, 20.0, np.nan, "自然人")
        )
        rows.append(
            (
                code,
                "20231231",
                "20240120",
                f"社保{idx}",
                45.0 + idx,
                4.5,
                32.0 + idx,
                np.nan,
                "社保基金、社保机构",
            )
        )
        rows.append(
            (code, "20231231", "20240120", f"自然人{idx}", 18.0, 1.8, 18.0, np.nan, "自然人")
        )
    return build_top10fh_panel(pd.DataFrame(rows, columns=raw_columns))


def _real_top_inst_lookup(dates: List[str]):
    """微型 top_inst 逐日聚合 ⇒ 真实 lookup（镜像 tests/test_top_inst_factors.py 的构造）。"""
    from src.lazybull.factors.top_inst import build_top_inst_lookup_by_date

    rows = []
    for idx in range(1, 9):
        code = f"00000{idx}.SZ"
        rows.append((code, "20240110", 1.0e6 + idx * 1.0e5, 2.0e5, 8.0e5 + idx * 1.0e5, 1))
        rows.append((code, "20240315", 5.0e5, 1.0e5, 4.0e5, 1))
    daily = pd.DataFrame(
        rows,
        columns=["ts_code", "trade_date", "inst_buy", "inst_sell", "inst_net", "inst_rows"],
    )
    return build_top_inst_lookup_by_date(daily, dates)


def _assemble_real_derivers() -> RuntimeDerivers:
    """从 factors 真实函数组装 RuntimeDerivers（生产组装形态的测试镜像）。"""
    from src.lazybull.factors import availability as factors_availability
    from src.lazybull.factors import holdertrade as factors_holdertrade
    from src.lazybull.factors import repurchase as factors_repurchase
    from src.lazybull.factors import top10_floatholders as factors_top10fh
    from src.lazybull.factors import top_inst as factors_top_inst

    return RuntimeDerivers(
        holdertrade=RuntimeFamilyDeriver(
            version_col=factors_holdertrade.HOLDERTRADE_VERSION_COL,
            schema_version=factors_holdertrade.HOLDERTRADE_SCHEMA_VERSION,
            feature_columns_fn=factors_holdertrade.holdertrade_feature_columns,
            derive_fn=factors_holdertrade.derive_holdertrade_columns,
        ),
        repurchase=RuntimeFamilyDeriver(
            version_col=factors_repurchase.REPURCHASE_VERSION_COL,
            schema_version=factors_repurchase.REPURCHASE_SCHEMA_VERSION,
            feature_columns_fn=factors_repurchase.repurchase_feature_columns,
            derive_fn=factors_repurchase.derive_repurchase_columns,
        ),
        top10fh=RuntimeFamilyDeriver(
            version_col=factors_top10fh.TOP10FH_VERSION_COL,
            schema_version=factors_top10fh.TOP10FH_SCHEMA_VERSION,
            feature_columns_fn=factors_top10fh.top10fh_feature_columns,
            derive_fn=factors_top10fh.derive_top10fh_columns,
        ),
        top_inst=RuntimeFamilyDeriver(
            version_col=factors_top_inst.TOP_INST_VERSION_COL,
            schema_version=factors_top_inst.TOP_INST_SCHEMA_VERSION,
            # 真实 top_inst_feature_columns 为无参签名（prepare_families 对齐）
            feature_columns_fn=factors_top_inst.top_inst_feature_columns,
            derive_fn=factors_top_inst.derive_top_inst_columns,
        ),
        availability=factors_availability.derive_availability_markers,
    )


class TestRealFactorsDeriveParity:
    def test_real_factors_derive_parity(self):
        """真实 factors 组装 derivers：四族 + availability 全开，旧新九元组逐位一致。"""
        df = _real_derive_frame()
        dates = sorted(df["trade_date"].unique())
        common_kwargs: Dict[str, Any] = {
            "label_column": LABEL_COLUMN,
            "val_ratio": 0.2,
            "enable_holdertrade_features": True,
            "holdertrade_lookup": _real_holdertrade_lookup(dates),
            "enable_repurchase_features": True,
            "repurchase_lookup": _real_repurchase_lookup(dates),
            "enable_top10fh_features": True,
            "top10fh_panel": _real_top10fh_panel(),
            "enable_top_inst_features": True,
            "top_inst_lookup": _real_top_inst_lookup(dates),
            "enable_availability_markers": True,
        }
        # 旧路径：不 patch，走真实 factors 顶层/惰性 import
        old_result = legacy_prepare_training_data(df.copy(), **common_kwargs)
        # 新路径：注入真实 derivers
        new_result = v2_prepare_training_data(
            df.copy(), runtime_derivers=_assemble_real_derivers(), **common_kwargs
        )
        assert_prepare_parity(old_result, new_result)

        # 真实派生语义顺带锁定（新侧）：四族代表列与可用性标记存活入模
        feature_columns = new_result[4]
        assert "ht_net_ratio_30d" in feature_columns
        assert "rp_amount_to_mv_90d" in feature_columns
        assert "tfh_top10_ratio" in feature_columns
        assert "ti_inst_net_ratio" in feature_columns
        assert "has_fund_holding" in feature_columns


# ---------------------------------------------------------------------------
# 生命周期：变换前帧释放（R-CODEX-03 / reviewer1-P2-1）
# ---------------------------------------------------------------------------


class TestLifecycleRelease:
    def test_pre_transform_frames_released(self):
        """label_transform_fn 复制型：变换前训练/验证帧在新旧实现返回后 weakref 均已死。"""
        results: List[Tuple] = []
        pre_transform_refs: List[weakref.ref] = []
        for impl in (legacy_prepare_training_data, v2_prepare_training_data):
            seen: List[weakref.ref] = []

            def copying_transform(frame, _seen=seen):
                """复制型变换：记录传入帧 weakref 后返回深拷贝（不原地修改）。"""
                _seen.append(weakref.ref(frame))
                return frame.copy(deep=True)

            result = impl(
                _base_frame(),
                label_column=LABEL_COLUMN,
                val_ratio=0.2,
                label_transform_fn=copying_transform,
            )
            results.append(result)
            # 每侧恰调用两次（训练集 + 非空验证集）
            assert len(seen) == 2
            pre_transform_refs.extend(seen)

        gc.collect()
        dead = [ref() is None for ref in pre_transform_refs]
        assert all(dead), (
            f"变换前帧存在驻留引用（存活 {dead.count(False)}/{len(dead)}）："
            "内存生命周期修复未生效"
        )
        # 九元组仍逐位一致
        assert_prepare_parity(results[0], results[1])


# ---------------------------------------------------------------------------
# 拆分助手旧新冒烟（reviewer1-P3-8）
# ---------------------------------------------------------------------------


def _labels_frame(n_days: int = 20, seed: int = 7) -> pd.DataFrame:
    """标签/评估冒烟用小帧：trade_date/ts_code + neu_y_ret_20/y_ret_20（无 NaN）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y%m%d").tolist()
    codes = [f"00000{i}.SZ" for i in range(1, 9)]
    n_rows = len(days) * len(codes)
    return pd.DataFrame(
        {
            "trade_date": np.repeat(days, len(codes)),
            "ts_code": codes * len(days),
            LABEL_COLUMN: rng.standard_normal(n_rows) * 0.05,
            "y_ret_20": rng.standard_normal(n_rows) * 0.05,
        }
    )


class TestSplitHelpersParity:
    def test_ensemble_wrappers_parity(self):
        """ensemble：旧新 EnsembleModel / TreeLimitedModel 包装同一模型 ⇒ 预测逐位一致。"""
        import xgboost as xgb

        from src.lazybull.ml import ensemble as legacy_ensemble
        from src.lazybull.v2.train import ensemble as v2_ensemble

        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.standard_normal((60, 3)), columns=["f1", "f2", "f3"])
        y = pd.Series(rng.standard_normal(60) * 0.05)
        model = xgb.XGBRegressor(n_estimators=10, tree_method="hist", device="cpu", random_state=42)
        model.fit(X, y)

        old_ens = legacy_ensemble.EnsembleModel([model, model])
        new_ens = v2_ensemble.EnsembleModel([model, model])
        assert np.array_equal(old_ens.predict(X), new_ens.predict(X))

        old_limited = legacy_ensemble.TreeLimitedModel(model, tree_limit=5)
        new_limited = v2_ensemble.TreeLimitedModel(model, tree_limit=5)
        assert np.array_equal(old_limited.predict(X), new_limited.predict(X))

    def test_build_rank_sample_weights_parity(self):
        """weights：旧新 build_rank_sample_weights 同参 ⇒ 权重逐位一致。"""
        from src.lazybull.ml.train_core.weights import (
            build_rank_sample_weights as legacy_build_weights,
        )
        from src.lazybull.v2.train.core.weights import build_rank_sample_weights as v2_build_weights

        df = _labels_frame()
        old_weights = legacy_build_weights(df, LABEL_COLUMN, topk=3, top_weight=5.0)
        new_weights = v2_build_weights(df, LABEL_COLUMN, topk=3, top_weight=5.0)
        assert np.array_equal(old_weights, new_weights)

    def test_labels_helpers_parity(self):
        """labels：旧新 transform_labels_cs_zscore / add_blended_return_label /
        generate_classification_labels 同输入 ⇒ 输出逐位一致。"""
        from src.lazybull.ml.train_core import labels as legacy_labels
        from src.lazybull.v2.train.core import labels as v2_labels

        df = _labels_frame()

        old_out = legacy_labels.transform_labels_cs_zscore(
            df.copy(), label_column=LABEL_COLUMN, winsorize_p=0.01
        )
        new_out = v2_labels.transform_labels_cs_zscore(
            df.copy(), label_column=LABEL_COLUMN, winsorize_p=0.01
        )
        pd.testing.assert_frame_equal(old_out, new_out, check_exact=True)

        old_blended_df, new_blended_df = df.copy(), df.copy()
        old_name = legacy_labels.add_blended_return_label(old_blended_df, LABEL_COLUMN, 0.3)
        new_name = v2_labels.add_blended_return_label(new_blended_df, LABEL_COLUMN, 0.3)
        assert old_name == new_name == "y_blend_ret_20"
        pd.testing.assert_frame_equal(old_blended_df, new_blended_df, check_exact=True)

        old_cls = legacy_labels.generate_classification_labels(
            df.copy(), label_column="y_ret_20", pos_quantile=0.2
        )
        new_cls = v2_labels.generate_classification_labels(
            df.copy(), label_column="y_ret_20", pos_quantile=0.2
        )
        pd.testing.assert_frame_equal(old_cls, new_cls, check_exact=True)

    def test_eval_helpers_parity(self):
        """eval：旧新 neg_rank_ic / evaluate_validation_daily 同输入 ⇒ 输出一致。"""
        from src.lazybull.ml.train_core import eval as legacy_eval
        from src.lazybull.v2.train.core import eval as v2_eval

        rng = np.random.default_rng(13)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        assert legacy_eval.neg_rank_ic(y_true, y_pred) == v2_eval.neg_rank_ic(y_true, y_pred)

        # df_val 自带 pred_score ⇒ model 不被消费（传 None），纯评估口径对比
        df_val = _labels_frame(n_days=10, seed=17)
        df_val["pred_score"] = rng.standard_normal(len(df_val))
        old_result = legacy_eval.evaluate_validation_daily(
            None,
            df_val.copy(),
            [LABEL_COLUMN],
            "y_ret_20",
            "regression",
            topk_values=[2, 3],
            emit_logs=False,
        )
        new_result = v2_eval.evaluate_validation_daily(
            None,
            df_val.copy(),
            [LABEL_COLUMN],
            "y_ret_20",
            "regression",
            topk_values=[2, 3],
            emit_logs=False,
        )
        assert set(old_result) == set(new_result), "evaluate_validation_daily 键集不一致"
        for key in old_result:
            _assert_stats_value_equal(old_result[key], new_result[key], f"eval.{key}")

    def test_xgb_lambdarank_parity(self):
        """xgb lambdarank 分支：旧新 train_xgboost_model 同小数据 ⇒ 预测逐位一致。"""
        if not _cuda_training_available():
            pytest.skip("本机 XGBoost cuda 训练不可用，跳过 lambdarank parity")
        from src.lazybull.ml.train_core.xgb import train_xgboost_model as legacy_train_xgboost_model
        from src.lazybull.v2.train.core.xgb import train_xgboost_model as v2_train_xgboost_model

        df = _base_frame()
        result = v2_prepare_training_data(df.copy(), label_column=LABEL_COLUMN, val_ratio=0.2)
        X_train, y_train, X_val, y_val = result[0], result[1], result[2], result[3]
        df_train_split, df_val_split = result[5], result[6]
        # 同 regression parity：纯噪声 + gamma=0.1 ⇒ 不分裂；注入确定性信号
        y_train = y_train + X_train["neu_ret_1"] * 0.3
        y_val = y_val + X_val["neu_ret_1"] * 0.3
        train_kwargs: Dict[str, Any] = {
            "task": "regression",
            "objective_type": "lambdarank",
            "n_estimators": 10,
            "random_state": 42,
            "early_stopping_rounds": None,
            "min_child_weight": 10,  # 同 regression parity：让树真实分裂，避免常数预测
            "df_train_for_group": df_train_split,
            "df_val_for_group": df_val_split,
        }
        old_model, _, _, _ = legacy_train_xgboost_model(
            X_train, y_train, X_val, y_val, **train_kwargs
        )
        new_model, _, _, _ = v2_train_xgboost_model(X_train, y_train, X_val, y_val, **train_kwargs)
        old_pred = old_model.predict(X_val)
        new_pred = new_model.predict(X_val)
        assert np.ptp(old_pred) > 0, "预测为常数（树未分裂），parity 断言判别力不足"
        assert np.array_equal(old_pred, new_pred), "旧/新 xgb lambdarank 训练预测不逐位一致"
