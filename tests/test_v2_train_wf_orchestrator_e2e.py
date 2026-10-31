# -*- coding: utf-8 -*-
"""P3-T3 单折端到端合成验收（T3 卡独立验收主件）。

证据面：
1. **微缩 G-code**（同合成 cs_train 数据面）：旧 ``ml.walk_forward.training_core``
   args 路径 vs 新 ``V2Trainer.train_fold``（cs_train_verify）——特征列全等 +
   同一测试帧预测逐位一致（np.array_equal）+ 逐折切分边界对账（R2-06 微缩：
   显式缝换算边界 == 旧比例路径实际边界）；
2. **panel 端到端**：合成 store 上 ``V2WalkForwardOrchestrator.run`` 单折 ×
   3 种子全流程（训练注册 → OOS 评估 → stub 回测接线 → summary/台账/
   增强 data_state 产物齐备；增强 id 与旧口径不同，§5-4）；
3. **双源一致性微缩**（G-data 机制）：同值合成 panel / cs_train 双侧各跑
   train_fold ⇒ 预测逐位一致（行域对齐前提下的机制验证）；
4. **区分力**（§3.3）：panel 默认路径不触旧 ``load_cs_train_day``（拦截计数 0）+
   合成异值 panel 被实际消费 + panel 模式传 lookup 报错 + 四族 derive 零调用；
5. **真实内核集成**（codex-R07，2026-10-22 评审处置）：真实 ``replay.run_oos_backtest``
   + 真实 MLSignal 在合成全列行情上跑通 panel 特征源，异值 panel（salt）进入
   信号/成交/净值产生可观测差异 + panel 读取列集窄化证据 + 双根同号异模型的
   信号根绑定守卫证据（reviewer1-P1-3）；
6. **runs 契约批次宿主适配**（codex-R07 后半 + reviewer1-P2-5）：测试内适配器接
   ``write_runs_batch``，orchestrator 真实内核跑批后断言批次结构 + 四键关联
   （wf_run_id ↔ 批次 ↔ 模型版本 ↔ data_state 增强 id）；
7. **清单补齐**（reviewer1-P2-5）：域过滤顺序（§17.10-7，创业板股在帧加载后即剔）、
   deploy 输出根隔离（§17.10-9）、R2-06 全段对账（train/val-raw/ES/calibration/
   embargo 日期 + 样本键 + rank 样本权重逐项相等）。

全部 tmp_path 合成数据；真实 xgboost 小样本训练（生产 hist+cuda 配置的
确定性由 T0.5b spike 覆盖，本测试小样本 CPU 路径）。
"""

import functools
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.types import FeatureQuery, ModelConfig, TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.train.data_adapter import SOURCE_CS_TRAIN_VERIFY, AdapterConfig
from src.lazybull.v2.train.frame_cache import TrainingFrameCache
from src.lazybull.v2.train.trainer import V2Trainer
from src.lazybull.v2.train.walk_forward.backtest import BacktestHostParams
from src.lazybull.v2.train.walk_forward.orchestrator import (
    OrchestratorHostConfig,
    V2WalkForwardOrchestrator,
    reject_retired_switches,
)
from src.lazybull.v2.train.walk_forward.training_core import (
    TrainFoldParams,
    resolve_explicit_split_dates,
)

# ---------------------------------------------------------------------------
# 合成数据夹具
# ---------------------------------------------------------------------------

_CODES = [f"{i:06d}.SH" for i in range(1, 25)]  # 24 只主板合成股
_BASE_COLUMNS = [
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
_LABEL_COLS = ["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"]
_FILTER_COLS = ["is_st", "is_suspended", "is_limit_up"]
_TRAIN_DAYS = 120  # 外层训练窗交易日数
_DELTA = 21


def _days() -> List[str]:
    return pd.bdate_range("20240102", periods=_TRAIN_DAYS + 40).strftime("%Y%m%d").tolist()


def _cell_value(day_idx: int, code_idx: int, col_idx: int, *, salt: float = 0.0) -> float:
    """确定性特征值（双侧同值 / 区分力臂异值经 salt）。"""
    return round(((day_idx + 1) * 0.01 + (code_idx + 1) * 0.001 + (col_idx + 1) * 0.0001) + salt, 6)


def _frame_rows(day: str, day_idx: int, *, salt: float = 0.0) -> Dict[str, list]:
    payload: Dict[str, list] = {
        "trade_date": [day] * len(_CODES),
        "ts_code": list(_CODES),
    }
    for col_idx, col in enumerate(_BASE_COLUMNS):
        payload[col] = [
            _cell_value(day_idx, code_idx, col_idx, salt=salt) for code_idx in range(len(_CODES))
        ]
    for col in _FILTER_COLS:
        payload[col] = [0] * len(_CODES)
    for label_idx, col in enumerate(_LABEL_COLS):
        payload[col] = [
            round(0.02 * (label_idx + 1) + day_idx * 0.001 + code_idx * 0.0001, 6)
            for code_idx in range(len(_CODES))
        ]
    return payload


def _write_base_inputs(root: Path) -> None:
    calendar = _days()
    (root / "clean").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cal_date": calendar, "is_open": [1] * len(calendar)}).to_parquet(
        root / "clean" / "trade_cal.parquet", index=False
    )
    pd.DataFrame({"ts_code": _CODES, "market": ["主板"] * len(_CODES)}).to_parquet(
        root / "clean" / "stock_basic.parquet", index=False
    )
    daily = root / "clean" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for day in calendar:
        pd.DataFrame({"trade_date": [day], "ts_code": [_CODES[0]], "close": [1.0]}).to_parquet(
            daily / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
        )


def _build_cs_train_root(root: Path, *, salt: float = 0.0) -> None:
    _write_base_inputs(root)
    cs_dir = root / "features" / "cs_train"
    cs_dir.mkdir(parents=True, exist_ok=True)
    for day_idx, day in enumerate(_days()):
        pd.DataFrame(_frame_rows(day, day_idx, salt=salt)).to_parquet(
            cs_dir / f"{day}.parquet", index=False
        )


def _populate_store(
    store: PanelDataStore, rows_fn: Callable[[str, int], Dict[str, list]]
) -> PanelDataStore:
    """panel store 填充(全部族 + 六标签表;``rows_fn(day, day_idx)`` 供给当日行 payload)。"""
    store.manifest.register_columns(
        {
            **{col: {"group": "core", "source": "test"} for col in _BASE_COLUMNS},
            **{col: {"group": "core", "source": "test"} for col in _FILTER_COLS},
        }
    )
    store.manifest.save()
    from src.lazybull.v2.store.column_groups import PANEL_GROUPS

    # 批量回填延迟 manifest 落盘(160 天 × 11 次写入;非延迟模式逐次全量落盘为 O(n²))
    with store.defer_manifest_save():
        for day_idx, day in enumerate(_days()):
            rows = rows_fn(day, day_idx)
            for group in PANEL_GROUPS:
                if group == "core":
                    core_cols = (
                        ("trade_date", "ts_code") + tuple(_BASE_COLUMNS) + tuple(_FILTER_COLS)
                    )
                    payload = {col: rows[col] for col in core_cols}
                else:
                    # readiness 要求八族当日齐备(T1 夹具口径:非 core 族仅键列)
                    payload = {"trade_date": rows["trade_date"], "ts_code": rows["ts_code"]}
                store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))
            for table_idx, table in enumerate(LABEL_TABLES):
                raw_col = _LABEL_COLS[table_idx]  # y_ret_5/10/20
                neu_col = _LABEL_COLS[table_idx + 3]  # neu_y_ret_5/10/20
                store.append_labels(
                    table,
                    pd.DataFrame(
                        {
                            "ts_code": rows["ts_code"],
                            "trade_date": rows["trade_date"],
                            "label_value": rows[raw_col],
                            "neu_label_value": rows[neu_col],
                            "maturity_status": ["sealed"] * len(rows["ts_code"]),
                        },
                        columns=[
                            "ts_code",
                            "trade_date",
                            "label_value",
                            "neu_label_value",
                            "maturity_status",
                        ],  # readiness 协议列序强校验
                    ),
                )
    return store


def _build_panel_root(root: Path, *, salt: float = 0.0) -> PanelDataStore:
    _write_base_inputs(root)
    return _populate_store(
        PanelDataStore(root), lambda day, day_idx: _frame_rows(day, day_idx, salt=salt)
    )


def _old_path_objects(root: Path):
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(root))
    return storage, DataLoader(storage)


def _main_codes() -> set:
    return set(_CODES)


def _params() -> TrainFoldParams:
    return TrainFoldParams(
        label_column="neu_y_ret_20",
        task="regression",
        label_transform="raw",
        val_ratio=0.2,
        rank_weight_enabled=False,
        n_estimators=20,
        max_depth=3,
        early_stopping_rounds=10,
        early_stopping_metric="rank_ic",
        ensemble_seeds=(42, 61, 82),
        ensemble_seed_keep_top_ratio=1.0,
        ensemble_seed_keep_min_models=3,
    )


def _config() -> ModelConfig:
    params = _params()
    return ModelConfig(
        model_family="stock_selection",
        feature_columns=tuple(_BASE_COLUMNS + _FILTER_COLS),
        label_name="neu_y_ret_20",
        hyperparams={
            "algorithm": "xgboost",
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "early_stopping_rounds": params.early_stopping_rounds,
            "early_stopping_metric": params.early_stopping_metric,
            "rank_weight_enabled": False,
            "random_state": 42,
        },
        train_window_months=72,  # 生产口径 6 年;合成日历 160 日 ⇒ 起点对齐日历首日
        ensemble_seeds=(42, 61, 82),
        label_spec={"task": "regression", "label_transform": "raw", "version": 1},
        split_spec={"val_ratio": 0.2, "version": 1},
        # 全 False 配方(合成数据仅基础列;开族 ⇒ prepare 缺列 fail-fast,会破 parity)
        feature_recipe={
            "enable_fundamental_features": False,
            "max_feature_missing_ratio": 0.6,
            "version": 1,
        },
        domain="main",
    )


def _new_trainer(
    root: Path, *, source_mode: str, store=None, models_root: Optional[Path] = None
) -> V2Trainer:
    # models_root 独立传参:共享只读数据根时注册表仍落各测试私有输出位(防跨测试版本累加)
    models_base = models_root if models_root is not None else root / "out"
    registry = ModelRegistry(models_dir=str(models_base / "models" / "stock_selection"))
    storage, loader = _old_path_objects(root)
    return V2Trainer(
        registry=registry,
        main_board_codes=_main_codes(),
        source_mode=source_mode,
        acceptance_purpose="g_code_parity",
        storage=storage,
        loader=loader,
        frame_cache=TrainingFrameCache(),
    )


# ---------------------------------------------------------------------------
# 真实内核集成 / runs 批次 / 清单补齐共用夹具(2026-10-22 评审处置)
# ---------------------------------------------------------------------------

_GEM_CODE = "300001.SZ"  # 创业板合成股(域过滤顺序证据,§17.10-7)


def _gem_frame_rows(day: str, day_idx: int) -> Dict[str, list]:
    """24 主板 + 1 创业板合成行(创业板股值域与主板股同构)。"""
    rows = _frame_rows(day, day_idx)
    gem_idx = len(_CODES)
    rows["trade_date"].append(day)
    rows["ts_code"].append(_GEM_CODE)
    for col_idx, col in enumerate(_BASE_COLUMNS):
        rows[col].append(_cell_value(day_idx, gem_idx, col_idx))
    for col in _FILTER_COLS:
        rows[col].append(0)
    for label_idx, col in enumerate(_LABEL_COLS):
        rows[col].append(round(0.02 * (label_idx + 1) + day_idx * 0.001 + gem_idx * 0.0001, 6))
    return rows


def _build_gem_panel_root(root: Path) -> PanelDataStore:
    """含一只创业板股的 panel 根(stock_basic 同步登记 market='创业板')。"""
    _write_base_inputs(root)
    pd.DataFrame(
        {"ts_code": _CODES + [_GEM_CODE], "market": ["主板"] * len(_CODES) + ["创业板"]}
    ).to_parquet(root / "clean" / "stock_basic.parquet", index=False)
    return _populate_store(PanelDataStore(root), _gem_frame_rows)


def _kernel_cell_value(day_idx: int, code_idx: int, col_idx: int, *, salt: float = 0.0) -> float:
    """真实内核集成专用特征值:日域平稳的确定性伪随机截面(区分力前提)。

    共享 ``_cell_value`` 随 day_idx 单调递增 ⇒ OOS 测试日特征值整体落在训练值域外,
    树模型路径饱和、分数退化为常数(区分力臂无从谈起);本生成器值域平稳([-0.5,0.5]
    邻域),测试日截面内逐股可区分,模型预测逐股异。
    """
    hash_val = (day_idx * 31 + code_idx * 7 + col_idx * 13) % 101
    return round((hash_val - 50) * 0.01 + salt, 6)


def _kernel_label_value(day_idx: int, code_idx: int, label_idx: int, sign: float = 1.0) -> float:
    """与前两个特征列可学习的标签(树分裂增益足以越过 gamma=0.1 门槛)。

    ``sign=-1.0`` 产出反向排序标签(双根同号异模型守卫臂:同 version 不同内容)。
    """
    f0 = _kernel_cell_value(day_idx, code_idx, 0)
    f1 = _kernel_cell_value(day_idx, code_idx, 1)
    return round(sign * (0.05 + 0.08 * f0 + 0.05 * f1 + 0.005 * label_idx), 6)


def _kernel_frame_rows(
    day: str, day_idx: int, *, salt: float = 0.0, label_sign: float = 1.0
) -> Dict[str, list]:
    """同 ``_frame_rows`` 结构,特征/标签改用 kernel 生成器(过滤列不变)。"""
    payload: Dict[str, list] = {
        "trade_date": [day] * len(_CODES),
        "ts_code": list(_CODES),
    }
    for col_idx, col in enumerate(_BASE_COLUMNS):
        payload[col] = [
            _kernel_cell_value(day_idx, code_idx, col_idx, salt=salt)
            for code_idx in range(len(_CODES))
        ]
    for col in _FILTER_COLS:
        payload[col] = [0] * len(_CODES)
    for label_idx, col in enumerate(_LABEL_COLS):
        payload[col] = [
            _kernel_label_value(day_idx, code_idx, label_idx, sign=label_sign)
            for code_idx in range(len(_CODES))
        ]
    return payload


def _build_kernel_panel_root(
    root: Path, *, salt: float = 0.0, label_sign: float = 1.0
) -> PanelDataStore:
    """真实内核集成专用 panel 根(平稳伪随机特征;供 replay/MLSignal 真实消费)。"""
    _write_base_inputs(root)
    return _populate_store(
        PanelDataStore(root),
        lambda day, day_idx: _kernel_frame_rows(day, day_idx, salt=salt, label_sign=label_sign),
    )


def _write_engine_daily(root: Path) -> None:
    """覆写 clean/daily 为回测引擎全列分区(真实 replay 内核消费)。

    列清单对齐 ``replay._load_replay_price_data``;价格为确定性正值且逐股/逐日异
    (净值随持仓真实变动,is_st/is_suspended/涨跌停全 0、tradable=1、list_days=1000)。
    """
    daily = root / "clean" / "daily"
    for day_idx, day in enumerate(_days()):
        n = len(_CODES)
        close = [round(10.0 + day_idx * 0.01 + code_idx * 0.05, 4) for code_idx in range(n)]
        pd.DataFrame(
            {
                "trade_date": [day] * n,
                "ts_code": list(_CODES),
                "close": close,
                "close_adj": close,
                "open": close,
                "open_adj": close,
                "is_suspended": [0] * n,
                "is_limit_up": [0] * n,
                "is_limit_down": [0] * n,
                "vol": [100000.0] * n,
                "pct_chg": [0.1] * n,
                "is_st": [0] * n,
                "list_days": [1000] * n,
                "tradable": [1] * n,
            }
        ).to_parquet(daily / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False)


# ---------------------------------------------------------------------------
# 模块级共享只读数据根(同值夹具建一次复用;测试对数据根只读,产物一律写各自
# tmp_path 输出根;test_data_state_covers_train_labels_and_suspend 仅做增量补文件)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_panel_root(tmp_path_factory) -> Path:
    """普通 panel 根(salt=0;11 处复用)。"""
    root = tmp_path_factory.mktemp("shared_panel") / "root"
    _build_panel_root(root)
    return root


@pytest.fixture(scope="module")
def shared_cs_root(tmp_path_factory) -> Path:
    """cs_train 根(6 处复用)。"""
    root = tmp_path_factory.mktemp("shared_cs") / "root"
    _build_cs_train_root(root)
    return root


@pytest.fixture(scope="module")
def shared_kernel_root(tmp_path_factory) -> Path:
    """kernel panel 根 + 引擎全列行情(3 处复用;真实 replay 消费)。"""
    root = tmp_path_factory.mktemp("shared_kernel") / "root"
    _build_kernel_panel_root(root)
    _write_engine_daily(root)
    return root


def _make_ml_signal(models_dir: Path, *, top_n: int = 2):
    """真实 MLSignal(绑定隔离 registry;市值边界取域定义单一来源,replay_b0 先例)。"""
    from src.lazybull.universe.domains import resolve_stock_domain
    from src.lazybull.v2.core.signal.ml_signal import MLSignal

    bt_domain = resolve_stock_domain("main")
    return MLSignal(
        top_n=top_n,
        model_version=None,
        models_dir=str(models_dir),
        min_total_mv=float(bt_domain["min_total_mv_wan"]),
        max_total_mv=float(bt_domain["max_total_mv_wan"]),
        verbose=False,
        downside_penalty=0.0,
        downside_penalty_column="downside_vol_20",
    )


def _config_with(
    *,
    hyperparams_over: Optional[Dict[str, Any]] = None,
    ensemble_seeds: Optional[tuple] = None,
) -> ModelConfig:
    """``_config()`` 变体构造(frozen 配置不可就地改,整棵重建)。"""
    base = _config()
    hyperparams = dict(base.hyperparams)
    hyperparams.update(hyperparams_over or {})
    return ModelConfig(
        model_family=base.model_family,
        feature_columns=tuple(base.feature_columns),
        label_name=base.label_name,
        hyperparams=hyperparams,
        train_window_months=base.train_window_months,
        ensemble_seeds=tuple(ensemble_seeds) if ensemble_seeds else tuple(base.ensemble_seeds),
        label_spec=dict(base.label_spec),
        feature_recipe=dict(base.feature_recipe),
        split_spec=dict(base.split_spec),
        domain=base.domain,
    )


def _legacy_args_ns(**overrides: Any) -> SimpleNamespace:
    """旧 args 命名空间(默认值逐项对齐现役 cli 默认;overrides 覆盖差异键)。"""
    base: Dict[str, Any] = dict(
        label_column="neu_y_ret_20",
        neutral_label_blend_weight=0.0,
        task="regression",
        pos_quantile=None,
        pos_topk=None,
        label_transform="raw",
        winsorize_p=0.01,
        val_ratio=0.2,
        enable_fundamental_features=False,
        enable_alt_features=False,
        enable_margin_features=False,
        enable_cyq_features=False,
        enable_fund_features=False,
        enable_express_features=False,
        enable_enhanced_features=False,
        enable_north_features=False,
        enable_lhb_features=False,
        enable_consensus_features=False,
        enable_cashflow_quality_features=False,
        enable_consensus_revision_features=False,
        enable_dividend_policy_features=False,
        enable_holdertrade_features=False,
        holdertrade_lookup=None,
        holdertrade_feature_set="full",
        enable_repurchase_features=False,
        repurchase_lookup=None,
        repurchase_feature_set="full",
        enable_top10fh_features=False,
        top10fh_panel=None,
        top10fh_feature_set="full",
        enable_top_inst_features=False,
        top_inst_lookup=None,
        enable_availability_markers=False,
        feature_stability_filter=False,
        factor_prune=False,
        factor_exclude_file=None,
        max_feature_missing_ratio=0.6,
        freshness_strategy="state_keep_event_decay",
        event_freshness_half_life_days=45.0,
        rank_weight_enabled=False,
        rank_weight_topk=30,
        rank_weight=5.0,
        rank_weight_topk_weight_mode="linear_decay",
        time_decay_half_life=0,
        scale_pos_weight=None,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        min_child_weight=100,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.1,
        early_stopping_rounds=10,
        early_stopping_metric="rank_ic",
        min_best_iteration=0,
        objective="mse",
        algorithm="xgboost",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _train_panel_fold_v1(
    root: Path,
    store: PanelDataStore,
    models_dir: Path,
    *,
    config: Optional[ModelConfig] = None,
) -> List[str]:
    """panel 模式训练外层首折(显式缝)并注册 v1 到 ``models_dir``,返回折模型实际特征列。"""
    trainer = V2Trainer(
        registry=ModelRegistry(models_dir=str(models_dir)),
        main_board_codes=_main_codes(),
        source_mode="panel",
        frame_cache=TrainingFrameCache(),
    )
    days = _days()
    t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
        days[0], days[_TRAIN_DAYS - 1], 0.2, days[:_TRAIN_DAYS], _DELTA
    )
    version = trainer.train_fold(config or _config(), t_start, t_end, v_start, v_end, store)
    assert version == "1"
    result = trainer.last_fold_result
    assert result is not None
    return list(result.feature_columns)


def _run_real_replay(
    root: Path, store: PanelDataStore, models_dir: Path, feature_columns: List[str]
) -> Dict[str, Any]:
    """真实 v2 内核 OOS 回测(panel 特征源 + 真实 MLSignal + 合成全列行情)。"""
    from src.lazybull.v2.hosts.backtest.replay import run_oos_backtest

    storage, loader = _old_path_objects(root)
    days = _days()
    trade_cal = loader.load_clean_trade_cal()
    stock_basic = loader.load_clean_stock_basic()
    assert trade_cal is not None and stock_basic is not None
    return run_oos_backtest(
        model_version=1,
        bt_start=days[_TRAIN_DAYS],
        bt_end=days[-1],
        storage=storage,
        loader=loader,
        trade_cal=trade_cal,
        stock_basic=stock_basic,
        label_column="neu_y_ret_20",
        bt_top_n=2,
        bt_rebalance_freq=20,
        data_root=str(root),
        persistent_signal=_make_ml_signal(models_dir),
        bt_exclude_st=True,
        bt_min_list_days=365,
        bt_sell_timing="open",
        position_sizing="equal",
        stagger_tranches=1,
        enable_early_rebalance_on_empty=True,
        initial_capital=1000000.0,
        split_num=0,
        stock_domain="main",
        feature_source="panel",
        store=store,
        feature_columns=feature_columns,
    )


def _bt_outputs_identical(m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
    """两笔回测的公开 metrics + 成交 + 净值是否完全一致(区分力判据)。"""
    public1 = {k: v for k, v in m1.items() if not k.startswith("_")}
    public2 = {k: v for k, v in m2.items() if not k.startswith("_")}
    nav_equal = m1["_nav_curve"]["nav"].round(12).equals(m2["_nav_curve"]["nav"].round(12))
    trades_equal = m1["_trades"].equals(m2["_trades"])
    return public1 == public2 and nav_equal and trades_equal


def _runs_writer_adapter(
    results: List[Dict[str, Any]],
    wf_run_id: str,
    data_state_payload: Dict[str, Any],
    *,
    runs_root: Path,
) -> Path:
    """orchestrator results → runs 契约批次的宿主适配(测试内)。

    键映射对齐 ``runs_writer._RESULT_REQUIRED_KEYS``:orchestrator 下划线键
    (``_nav_curve`` / ``_trades`` / ``_execution_attribution`` / ``_holdings_snapshot``)
    → runs 契约键;config 给必要键(127 键从简,缺键不入指纹);data_state 原样透传
    (编排器已注入增强 ``data_state_id``)。
    """
    from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch

    mapped = []
    for result in results:
        mapped.append(
            {
                "split_index": result["split_index"],
                "model_version": result["model_version"],
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "test_start": result["test_start"],
                "test_end": result["test_end"],
                "metrics": dict(result.get("bt_metrics") or {}),
                "nav_curve": result["_nav_curve"],
                "trades": result["_trades"],
                "attribution": result["_execution_attribution"],
                "holdings_snapshot": result["_holdings_snapshot"],
            }
        )
    return write_runs_batch(
        mapped,
        batch_id=wf_run_id,
        runs_root=Path(runs_root),
        config={
            "label_column": "neu_y_ret_20",
            "algorithm": "xgboost",
            "bt_top_n": 2,
            "bt_rebalance_freq": 20,
            "stock_domain": "main",
            "feature_source": "panel",
        },
        code_state={"git_commit": "test", "git_dirty": False},
        data_state=data_state_payload,
        arms=("p3_t3_real_kernel",),
        source="p3_t3_orchestrator_acceptance",
    )


# ---------------------------------------------------------------------------
# 1. 微缩 G-code:同合成 cs_train 下旧新对跑
# ---------------------------------------------------------------------------


class TestCsTrainParityWithLegacy:
    def test_single_fold_ensemble_parity(self, tmp_path, shared_cs_root):
        cs_root = shared_cs_root
        days = _days()
        outer_start, outer_end = days[0], days[_TRAIN_DAYS - 1]
        params = _params()

        # 旧臂:args namespace + 旧 _train_model_on_window(读旧 cs_train 通道)
        from src.lazybull.ml.walk_forward.training_core import (
            _build_ensemble_sub_models as old_build,
        )

        storage, loader = _old_path_objects(cs_root)
        args_ns = SimpleNamespace(
            label_column=params.label_column,
            neutral_label_blend_weight=0.0,
            task="regression",
            pos_quantile=None,
            pos_topk=None,
            label_transform="raw",
            winsorize_p=0.01,
            val_ratio=0.2,
            enable_fundamental_features=False,
            enable_alt_features=False,
            enable_margin_features=False,
            enable_cyq_features=False,
            enable_fund_features=False,
            enable_express_features=False,
            enable_enhanced_features=False,
            enable_north_features=False,
            enable_lhb_features=False,
            enable_consensus_features=False,
            enable_cashflow_quality_features=False,
            enable_consensus_revision_features=False,
            enable_dividend_policy_features=False,
            enable_holdertrade_features=False,
            holdertrade_lookup=None,
            holdertrade_feature_set="full",
            enable_repurchase_features=False,
            repurchase_lookup=None,
            repurchase_feature_set="full",
            enable_top10fh_features=False,
            top10fh_panel=None,
            top10fh_feature_set="full",
            enable_top_inst_features=False,
            top_inst_lookup=None,
            enable_availability_markers=False,
            feature_stability_filter=False,
            factor_prune=False,
            factor_exclude_file=None,
            max_feature_missing_ratio=0.6,
            freshness_strategy="state_keep_event_decay",
            event_freshness_half_life_days=45.0,
            rank_weight_enabled=False,
            rank_weight_topk=30,
            rank_weight=5.0,
            rank_weight_topk_weight_mode="linear_decay",
            time_decay_half_life=0,
            scale_pos_weight=None,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            min_child_weight=100,
            reg_alpha=0.05,
            reg_lambda=1.0,
            gamma=0.1,
            early_stopping_rounds=10,
            early_stopping_metric="rank_ic",
            min_best_iteration=0,
            objective="mse",
            algorithm="xgboost",
            ensemble_seeds="42,61,82",
            ensemble_seed_keep_top_ratio=1.0,
            ensemble_seed_keep_min_models=3,
            stock_domain="main",
        )
        old_sub_models, old_base, old_meta = old_build(
            [(outer_start, outer_end)],
            storage,
            loader,
            args_ns,
            _main_codes(),
            [42, 61, 82],
            topk_values=[20, 30],
        )

        # 新臂:cs_train_verify + 显式缝(编排器换算)
        trainer = _new_trainer(cs_root, source_mode=SOURCE_CS_TRAIN_VERIFY, models_root=tmp_path)
        config = _config()
        trade_dates = days[:_TRAIN_DAYS]
        t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
            outer_start, outer_end, 0.2, trade_dates, _DELTA
        )
        version = trainer.train_fold(config, t_start, t_end, v_start, v_end, None)
        assert version == "1"
        new_result = trainer.last_fold_result
        assert new_result is not None

        # 特征列全等 + data_stats 切分边界对账(R2-06 微缩)
        assert list(new_result.feature_columns) == list(old_base["feature_columns"])
        new_stats = new_result.data_stats
        old_stats = old_base["data_stats"]
        for key in ("train_start_date", "train_end_date", "val_start_date", "val_end_date"):
            assert str(new_stats.get(key)) == str(old_stats.get(key)), key

        # 同一测试帧预测逐位一致(集成 3 种子)
        from src.lazybull.v2.train.data_adapter import load_training_frame

        test_frame = load_training_frame(
            AdapterConfig(
                start_date=days[_TRAIN_DAYS],
                end_date=days[_TRAIN_DAYS + 10],
                source_mode=SOURCE_CS_TRAIN_VERIFY,
            ),
            storage=storage,
            loader=loader,
            acceptance_purpose="g_code_parity",
        )
        X = test_frame.df[list(new_result.feature_columns)]
        from src.lazybull.ml.ensemble import EnsembleModel as OldEnsemble
        from src.lazybull.v2.train.ensemble import EnsembleModel as NewEnsemble

        old_ensemble = OldEnsemble(old_sub_models)
        assert isinstance(new_result.model, NewEnsemble)
        np.testing.assert_array_equal(old_ensemble.predict(X), new_result.model.predict(X))

    def test_explicit_split_dates_matches_ratio_math(self):
        """显式缝换算 = 现役比例切分同数学(日历日域前提,§13.3.2 D1)。"""
        days = _days()[:_TRAIN_DAYS]
        n = len(days)
        import math

        n_val = max(1, math.ceil(n * 0.2))
        n_train = n - n_val
        expected_train_end = days[n_train - _DELTA - 1]
        expected_val_start = days[n_train + _DELTA]
        result = resolve_explicit_split_dates(days[0], days[-1], 0.2, days, _DELTA)
        assert result == (days[0], expected_train_end, expected_val_start, days[-1])

    def test_explicit_split_dates_rejections(self):
        days = _days()
        with pytest.raises(ValueError, match="无交易日"):
            resolve_explicit_split_dates("20300101", "20310101", 0.2, days, _DELTA)
        with pytest.raises(ValueError, match="隔离带"):
            resolve_explicit_split_dates(days[0], days[10], 0.2, days[:11], _DELTA)

    def test_r206_full_segment_reconciliation(self, tmp_path, shared_cs_root, monkeypatch):
        """R2-06 全段对账:旧 ``_train_model_on_window``(args 路径)vs 新 ``train_fold``
        (cs_train_verify + 显式缝)——逐折 train/val-raw/ES/calibration/embargo 日期 +
        样本数逐项相等;``df_train_split`` (trade_date, ts_code) 键集全等;rank_weight
        样本权重逐项相等(两侧真实调用点 monkeypatch 捕获,非重导实现;两臂同开
        rank_weight_enabled=True 且 topk=10 < 24 股截面,权重非平凡)。
        """
        cs_root = shared_cs_root
        days = _days()
        storage, loader = _old_path_objects(cs_root)

        from src.lazybull.ml.walk_forward import training_core as legacy_tc
        from src.lazybull.v2.train.walk_forward import training_core as v2_tc

        captured: Dict[str, Dict[str, Any]] = {}

        def _spy(side: str, fn: Callable) -> Callable:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                weights = fn(*args, **kwargs)
                df_train = kwargs.get("df_train", args[0] if args else None)
                captured[side] = {"df": df_train, "weights": weights}
                return weights

            return wrapped

        monkeypatch.setattr(
            legacy_tc,
            "build_rank_sample_weights",
            _spy("legacy", legacy_tc.build_rank_sample_weights),
        )
        monkeypatch.setattr(
            v2_tc,
            "build_rank_sample_weights",
            _spy("v2", v2_tc.build_rank_sample_weights),
        )

        # 旧臂:args 路径(rank_weight 开)
        old_tr = legacy_tc._train_model_on_window(
            days[0],
            days[_TRAIN_DAYS - 1],
            storage,
            loader,
            _legacy_args_ns(rank_weight_enabled=True, rank_weight_topk=10),
            _main_codes(),
        )

        # 新臂:cs_train_verify + 显式缝(单种子 ⇒ train_fold 直走同一 _train_model_on_window)
        trainer = _new_trainer(cs_root, source_mode=SOURCE_CS_TRAIN_VERIFY, models_root=tmp_path)
        config = _config_with(
            hyperparams_over={"rank_weight_enabled": True, "rank_weight_topk": 10},
            ensemble_seeds=(42,),
        )
        t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
            days[0], days[_TRAIN_DAYS - 1], 0.2, days[:_TRAIN_DAYS], _DELTA
        )
        assert trainer.train_fold(config, t_start, t_end, v_start, v_end, None) == "1"
        new_result = trainer.last_fold_result
        assert new_result is not None

        # ① 全段日期 + 样本数逐项相等(train/val-raw/ES/calibration/embargo)
        new_stats, old_stats = new_result.data_stats, old_tr["data_stats"]
        for key in (
            "train_start_date",
            "train_end_date",
            "val_raw_start_date",
            "val_raw_end_date",
            "val_es_start_date",
            "val_es_end_date",
            "val_calib_start_date",
            "val_calib_end_date",
            "val_embargo_start_date",
            "val_embargo_end_date",
        ):
            assert str(new_stats.get(key)) == str(old_stats.get(key)), key
        for key in (
            "samples_after_filter",
            "val_raw_samples",
            "val_es_samples",
            "val_calib_samples",
            "val_embargo_samples",
        ):
            assert int(new_stats.get(key)) == int(old_stats.get(key)), key
        assert new_result.X_train_len == old_tr["X_train_len"]
        assert new_result.X_val_len == old_tr["X_val_len"]

        # ② df_train_split (trade_date, ts_code) 键集全等(真实调用点捕获)
        key_cols = ["trade_date", "ts_code"]
        old_df = captured["legacy"]["df"]
        new_df = captured["v2"]["df"]
        old_keys = set(map(tuple, old_df[key_cols].astype(str).to_numpy()))
        new_keys = set(map(tuple, new_df[key_cols].astype(str).to_numpy()))
        assert old_keys == new_keys

        # ③ rank 样本权重逐项相等(按键对齐消除行序差异;标签列同臂一致)
        assert new_result.label_column == old_tr["label_column"]
        old_w = pd.Series(
            captured["legacy"]["weights"],
            index=pd.MultiIndex.from_frame(old_df[key_cols].astype(str)),
        ).sort_index()
        new_w = pd.Series(
            captured["v2"]["weights"],
            index=pd.MultiIndex.from_frame(new_df[key_cols].astype(str)),
        ).sort_index()
        np.testing.assert_array_equal(new_w.values, old_w.values)


# ---------------------------------------------------------------------------
# 2. panel 端到端(orchestrator.run 单折全流程)
# ---------------------------------------------------------------------------


def _stub_backtest_runner(**kwargs) -> Dict[str, Any]:
    """宿主注入的回测 stub:接线证据(记录 kwargs)+ 合成净值/成交产物。"""
    bt_start = kwargs["bt_start"]
    dates = pd.bdate_range(bt_start, periods=3).strftime("%Y%m%d").tolist()
    nav = pd.DataFrame(
        {"nav": [1.0, 1.005, 1.011]},
        index=pd.to_datetime(dates),
    )
    return {
        "bt_total_return": 0.011,
        "bt_annual_return": 0.011,
        "bt_max_drawdown": -0.001,
        "bt_volatility": 0.01,
        "bt_sharpe": 1.0,
        "bt_calmar": 11.0,
        "bt_trading_days": 3,
        "bt_start": bt_start,
        "bt_end": kwargs["bt_end"],
        "bt_top_n": kwargs["bt_top_n"],
        "_nav_curve": nav,
        "_trades": pd.DataFrame(),
        "_execution_attribution": pd.DataFrame(),
        "_holdings_snapshot": pd.DataFrame(),
    }


class TestPanelEndToEnd:
    def test_orchestrator_single_fold_run(self, tmp_path, shared_panel_root):
        root = shared_panel_root
        store = PanelDataStore(root)
        output_root = tmp_path / "out"
        days = _days()
        registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = _old_path_objects(root)
        captured: Dict[str, Any] = {}

        def runner_stub(**kwargs):
            captured.update(kwargs)
            return _stub_backtest_runner(**kwargs)

        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=runner_stub,
            signal_factory=lambda top_n: None,  # P1-3 守卫:回测启用必须注入工厂
        )
        orch = V2WalkForwardOrchestrator(
            host,
            registry=registry,
            source_mode="panel",
            storage=storage,
            loader=loader,
        )
        config = _config()
        wf_run_id = orch.run(
            config,
            1,
            days[-1],
            trainer,
            store,
            test_window_months=1,
        )
        assert wf_run_id.startswith("wf_")

        # 回测接线:panel 特征源 + store 透传
        assert captured["feature_source"] == "panel"
        assert captured["store"] is store
        assert captured["persistent_signal"] is None

        # 产物齐备:summary CSV + 台账 CSV + 增强 data_state JSON + chain_nav
        raw_dir = output_root / "walk_forward" / "raw"
        summary_csv = raw_dir / f"walk_forward_summary_{wf_run_id}.csv"
        assert summary_csv.is_file()
        assert (output_root / "models" / "stock_selection" / "ml_train_runs.csv").is_file()
        assert (raw_dir / f"chain_nav_{wf_run_id}.csv").is_file()
        import json

        state_payload = json.loads(
            (raw_dir / f"data_state_{wf_run_id}.json").read_text(encoding="utf-8")
        )
        assert "input_files" in state_payload and "environment_identity" in state_payload
        from src.lazybull.v2.train.walk_forward.data_state import compute_data_state_id

        legacy_id = compute_data_state_id(
            {k: v for k, v in state_payload.items() if k != "data_state_id"}
        )
        assert state_payload["data_state_id"] != legacy_id  # §5-4 新旧不可比
        # 消费清单覆盖 panel 分区与 clean/daily
        paths = {record["path"] for record in state_payload["input_files"]}
        assert any(p.startswith("features/panel/") for p in paths)
        assert any(p.startswith("clean/daily/") for p in paths)
        assert "clean/trade_cal.parquet" in paths

        # summary 的 data_state_id 列 = 增强口径(与 JSON 同值)。
        # data_state_id 是标识符而非数值：纯数字+科学计数形态（如 ``631630e6``）
        # 会被 read_csv 自动推断成 float（631630000000.0），必须显式按字符串读取。
        summary_df = pd.read_csv(summary_csv, encoding="utf-8-sig", dtype={"data_state_id": str})
        assert summary_df["data_state_id"].iloc[0] == state_payload["data_state_id"]


# ---------------------------------------------------------------------------
# 3. 双源一致性微缩(G-data 机制:同值 panel vs cs_train_verify)
# ---------------------------------------------------------------------------


class TestDualSourceMechanism:
    def test_same_values_panel_vs_cs_train_parity(
        self, tmp_path, shared_panel_root, shared_cs_root
    ):
        panel_store = PanelDataStore(shared_panel_root)
        cs_root = shared_cs_root
        days = _days()
        config = _config()
        trade_dates = days[:_TRAIN_DAYS]
        t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
            days[0], days[_TRAIN_DAYS - 1], 0.2, trade_dates, _DELTA
        )

        trainer_panel = V2Trainer(
            registry=ModelRegistry(models_dir=str(tmp_path / "m1")),
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        trainer_cs = _new_trainer(
            cs_root, source_mode=SOURCE_CS_TRAIN_VERIFY, models_root=tmp_path / "m2"
        )
        # 合成日历超出生产冻结边界:经构造参数显式放宽(验收夹具;生产回放默认 20260702)

        v1 = trainer_panel.train_fold(config, t_start, t_end, v_start, v_end, panel_store)
        v2 = trainer_cs.train_fold(config, t_start, t_end, v_start, v_end, None)
        assert v1 == "1" and v2 == "1"

        # 同值双侧 ⇒ 特征列全等 + 同一测试帧预测逐位一致(机制验证)
        rp = trainer_panel.last_fold_result
        rc = trainer_cs.last_fold_result
        assert list(rp.feature_columns) == list(rc.feature_columns)
        frame = panel_store.load_features(
            FeatureQuery(
                columns=list(rp.feature_columns),
                start_date=days[_TRAIN_DAYS],
                end_date=days[_TRAIN_DAYS + 5],
            )
        )
        X = frame.df.reset_index()[list(rp.feature_columns)]
        np.testing.assert_array_equal(rp.model.predict(X), rc.model.predict(X))


# ---------------------------------------------------------------------------
# 4. 区分力验证(§3.3)
# ---------------------------------------------------------------------------


class TestDiscrimination:
    def test_panel_path_never_calls_load_cs_train_day(self, shared_panel_root):
        """v2 OOS 默认(panel)路径禁止旧 load_cs_train_day(拦截计数 = 0)。"""
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        store = PanelDataStore(shared_panel_root)
        calls: List[str] = []

        class _StorageSpy:
            def load_cs_train_day(self, day):
                calls.append(day)
                raise AssertionError("panel 路径不得调用 load_cs_train_day")

        days = _days()
        out = replay_mod._select_replay_features(
            "panel",
            store,
            None,
            _StorageSpy(),
            days[_TRAIN_DAYS : _TRAIN_DAYS + 5],
            days[_TRAIN_DAYS],
            days[_TRAIN_DAYS + 4],
            None,
            None,
            None,
            None,
        )
        assert calls == []
        assert len(out) == 5

    def test_synthetic_panel_values_actually_consumed(self, tmp_path):
        """合成异值 panel ⇒ OOS 特征输出反映合成值(证明实际消费 panel)。"""
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        store = _build_panel_root(tmp_path / "root", salt=9.0)
        days = _days()
        out = replay_mod._select_replay_features(
            "panel",
            store,
            None,
            None,
            days[_TRAIN_DAYS : _TRAIN_DAYS + 3],
            days[_TRAIN_DAYS],
            days[_TRAIN_DAYS + 2],
            None,
            None,
            None,
            None,
        )
        day_frame = out[days[_TRAIN_DAYS]]
        expected = _cell_value(_TRAIN_DAYS, 0, 0, salt=9.0)
        assert day_frame["neu_ret_1"].iloc[0] == pytest.approx(expected)

    def test_panel_mode_rejects_lookups_and_skips_derive(self, monkeypatch, shared_panel_root):
        """panel 模式传 lookup ⇒ 报错;derive 函数零调用(防双写,§3.3-③)。"""
        from src.lazybull.factors import holdertrade as ht_mod
        from src.lazybull.v2.common.types import ReplayContractError
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        store = PanelDataStore(shared_panel_root)
        derive_calls: List[str] = []
        monkeypatch.setattr(
            ht_mod, "derive_holdertrade_columns", lambda *a, **k: derive_calls.append("ht")
        )
        monkeypatch.setattr(
            replay_mod, "derive_holdertrade_columns", lambda *a, **k: derive_calls.append("ht")
        )
        days = _days()
        with pytest.raises(ReplayContractError, match="防双写"):
            replay_mod._select_replay_features(
                "panel",
                store,
                None,
                None,
                days[_TRAIN_DAYS : _TRAIN_DAYS + 2],
                days[_TRAIN_DAYS],
                days[_TRAIN_DAYS + 1],
                {"x": pd.DataFrame()},  # holdertrade_lookup 非空
                None,
                None,
                None,
            )
        # 正常 panel 调用不触 derive
        replay_mod._select_replay_features(
            "panel",
            store,
            None,
            None,
            days[_TRAIN_DAYS : _TRAIN_DAYS + 2],
            days[_TRAIN_DAYS],
            days[_TRAIN_DAYS + 1],
            None,
            None,
            None,
            None,
        )
        assert derive_calls == []

    def test_panel_missing_data_fails_no_fallback(self, shared_panel_root):
        """panel 模式缺数据 ⇒ RuntimeError(旧特征读取回退已显式关闭)。"""
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        store = PanelDataStore(shared_panel_root)
        with pytest.raises(RuntimeError, match="旧特征读取回退已显式关闭"):
            replay_mod._select_replay_features(
                "panel",
                store,
                None,
                None,
                ["20300102"],
                "20300102",
                "20300103",
                None,
                None,
                None,
                None,
            )

    def test_cs_train_verify_requires_purpose(self, tmp_path):
        """cs_train_verify 缺 purpose ⇒ 拒绝(验收例外守卫)。"""
        from src.lazybull.v2.common.types import ReplayContractError
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        with pytest.raises(ReplayContractError, match="验收用途"):
            replay_mod._select_replay_features(
                "cs_train_verify",
                None,
                None,
                None,
                [],
                "20240102",
                "20240103",
                None,
                None,
                None,
                None,
            )


# ---------------------------------------------------------------------------
# 5. 协议符合性 + 退役开关(§17.10-2/8 核心项)
# ---------------------------------------------------------------------------


class TestProtocolAndRetired:
    def test_trainer_and_orchestrator_satisfy_protocols(self):
        from src.lazybull.v2.common.protocols import Trainer, WalkForwardOrchestrator

        assert isinstance(V2Trainer.__new__(V2Trainer), Trainer)
        assert isinstance(
            V2WalkForwardOrchestrator.__new__(V2WalkForwardOrchestrator),
            WalkForwardOrchestrator,
        )
        # 协议签名可绑定合法参数集(不得只验方法存在,§7.6)
        inspect.signature(V2Trainer.train_fold).bind(
            None, _config(), "20240102", "20240601", "20240603", "20241231", None
        )
        inspect.signature(V2WalkForwardOrchestrator.run).bind(
            None, _config(), 1, "20241231", None, None
        )

    def test_model_config_f10_fields_defaults(self):
        config = _config()
        assert config.val_window_months is None  # 比例模式
        assert config.domain == "main"
        frozen_err = False
        try:
            config.domain = "other"
        except Exception:
            frozen_err = True
        assert frozen_err  # frozen 契约

    def test_reject_retired_switches(self):
        reject_retired_switches()  # 全空放行
        reject_retired_switches(exposure_table=None, downside_penalty=0.0)
        with pytest.raises(ValueError, match="已退役"):
            reject_retired_switches(downside_penalty=0.5)
        with pytest.raises(ValueError, match="已退役"):
            reject_retired_switches(exposure_replenish=True)

    def test_orchestrator_rejects_unknown_mode_and_panel_lookups(self, tmp_path):
        registry = ModelRegistry(models_dir=str(tmp_path / "m"))
        host = OrchestratorHostConfig(data_root=str(tmp_path), output_root=str(tmp_path))
        with pytest.raises(ValueError, match="未知数据源模式"):
            V2WalkForwardOrchestrator(host, registry=registry, source_mode="bogus")
        with pytest.raises(ValueError, match="必须声明验收用途"):
            V2WalkForwardOrchestrator(host, registry=registry, source_mode="cs_train_verify")
        with pytest.raises(ValueError, match="必须为空"):
            V2WalkForwardOrchestrator(
                host,
                registry=registry,
                source_mode="panel",
                runtime_lookups={"holdertrade_lookup": object()},
            )


# ---------------------------------------------------------------------------
# 5. 评估路径 parity(evaluate_test_window 旧新对跑,R2-06 逐折对账微缩)
# ---------------------------------------------------------------------------


class TestEvalParityWithLegacy:
    def test_evaluate_test_window_parity(self, shared_cs_root):
        cs_root = shared_cs_root
        days = _days()
        params = _params()
        storage, loader = _old_path_objects(cs_root)

        # 旧臂训练微型模型(共享同一内存模型 ⇒ 评估两侧输入模型同一)
        from src.lazybull.ml.walk_forward.training_core import _train_model_on_window as old_train

        args_ns = SimpleNamespace(
            label_column="neu_y_ret_20",
            neutral_label_blend_weight=0.0,
            task="regression",
            pos_quantile=None,
            pos_topk=None,
            label_transform="raw",
            winsorize_p=0.01,
            val_ratio=0.2,
            enable_fundamental_features=False,
            enable_alt_features=False,
            enable_margin_features=False,
            enable_cyq_features=False,
            enable_fund_features=False,
            enable_express_features=False,
            enable_enhanced_features=False,
            enable_north_features=False,
            enable_lhb_features=False,
            enable_consensus_features=False,
            enable_cashflow_quality_features=False,
            enable_consensus_revision_features=False,
            enable_dividend_policy_features=False,
            enable_holdertrade_features=False,
            holdertrade_lookup=None,
            holdertrade_feature_set="full",
            enable_repurchase_features=False,
            repurchase_lookup=None,
            repurchase_feature_set="full",
            enable_top10fh_features=False,
            top10fh_panel=None,
            top10fh_feature_set="full",
            enable_top_inst_features=False,
            top_inst_lookup=None,
            enable_availability_markers=False,
            feature_stability_filter=False,
            factor_prune=False,
            factor_exclude_file=None,
            max_feature_missing_ratio=0.6,
            freshness_strategy="state_keep_event_decay",
            event_freshness_half_life_days=45.0,
            rank_weight_enabled=False,
            rank_weight_topk=30,
            rank_weight=5.0,
            rank_weight_topk_weight_mode="linear_decay",
            time_decay_half_life=0,
            scale_pos_weight=None,
            n_estimators=20,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            min_child_weight=100,
            reg_alpha=0.05,
            reg_lambda=1.0,
            gamma=0.1,
            early_stopping_rounds=10,
            early_stopping_metric="rank_ic",
            min_best_iteration=0,
            objective="mse",
            algorithm="xgboost",
            downside_penalty=0.0,
        )
        tr = old_train(days[0], days[_TRAIN_DAYS - 1], storage, loader, args_ns, _main_codes())
        model, fcols = tr["model"], tr["feature_columns"]

        from src.lazybull.ml.walk_forward.split_training import evaluate_test_window as old_eval
        from src.lazybull.ml.walk_forward.utils import WalkForwardSplit as OldSplit

        split_old = OldSplit(
            0, days[0], days[_TRAIN_DAYS - 1], days[_TRAIN_DAYS], days[_TRAIN_DAYS + 10]
        )
        old_out = old_eval(
            split_old, model, fcols, storage, loader, args_ns, _main_codes(), [20, 30]
        )

        from src.lazybull.v2.train.data_adapter import SOURCE_CS_TRAIN_VERIFY as _CS
        from src.lazybull.v2.train.data_adapter import load_training_frame
        from src.lazybull.v2.train.walk_forward.split_training import (
            evaluate_test_window as new_eval,
        )
        from src.lazybull.v2.train.walk_forward.utils import WalkForwardSplit as NewSplit

        frame = load_training_frame(
            AdapterConfig(
                start_date=days[_TRAIN_DAYS],
                end_date=days[_TRAIN_DAYS + 10],
                source_mode=_CS,
            ),
            storage=storage,
            loader=loader,
            acceptance_purpose="g_code_parity",
        )
        new_out = new_eval(
            split=NewSplit(
                0, days[0], days[_TRAIN_DAYS - 1], days[_TRAIN_DAYS], days[_TRAIN_DAYS + 10]
            ),
            model=model,
            feature_columns=list(fcols),
            frame=frame,
            params=params,
            main_board_codes=_main_codes(),
            topk_values=[20, 30],
        )

        # 逐日 TopK 明细逐位一致 + 测试指标全等 + 样本数一致
        pd.testing.assert_frame_equal(
            old_out["topk_detail_df"], new_out["topk_detail_df"], check_exact=True
        )
        assert old_out["test_daily_metrics"] == new_out["test_daily_metrics"]
        assert old_out["test_samples"] == new_out["test_samples"]


# ---------------------------------------------------------------------------
# 6. 评审处置回归(reviewer1 P1-x / codex R0x;2026-10-22)
# ---------------------------------------------------------------------------


class TestReviewFixesP1:
    def test_classification_path_frame_returned(self, tmp_path, shared_cs_root):
        """codex-R04:分类任务 generate_classification_labels 新帧必须传回训练链路。"""
        cs_root = shared_cs_root
        days = _days()
        trainer = _new_trainer(cs_root, source_mode=SOURCE_CS_TRAIN_VERIFY, models_root=tmp_path)
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=(),
            label_name="neu_y_ret_20",
            hyperparams={
                "algorithm": "xgboost",
                "n_estimators": 10,
                "max_depth": 3,
                "early_stopping_rounds": 5,
                "early_stopping_metric": "rank_ic",
                "rank_weight_enabled": False,
                "random_state": 42,
            },
            train_window_months=72,
            ensemble_seeds=(42,),
            label_spec={"task": "classification", "pos_topk": 3, "version": 1},
            split_spec={"val_ratio": 0.2, "version": 1},
        )
        trade_dates = days[:_TRAIN_DAYS]
        t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
            days[0], days[_TRAIN_DAYS - 1], 0.2, trade_dates, _DELTA
        )
        version = trainer.train_fold(config, t_start, t_end, v_start, v_end, None)
        assert version == "1"  # 修复前:prepare 缺 _binary 列必然报错
        result = trainer.last_fold_result
        assert result is not None and result.label_column == "neu_y_ret_20_binary"

    def test_frame_cache_cleared_per_fold(self, tmp_path, shared_panel_root):
        """P1-2/R02:两折运行,每折后帧缓存清空(失败路径同覆盖)。"""
        root = shared_panel_root
        store = PanelDataStore(root)
        output_root = tmp_path / "out"
        days = _days()
        registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
        cache = TrainingFrameCache()
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=cache,
        )
        storage, loader = _old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=_stub_backtest_runner,
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        observed: List[int] = []
        original_clear = cache.clear

        def spy_clear():
            original_clear()
            observed.append(len(cache))  # clear 后应为空

        cache.clear = spy_clear
        orch.run(
            _config(),
            2,
            days[-1],
            trainer,
            store,
            test_window_months=1,
            selected_split_indices=[0, 1],
        )
        assert len(cache) == 0  # run 后缓存空
        # 每次 clear(2 折 finally + run 收尾兜底)后均为空,且次数覆盖逐折
        assert observed and all(count == 0 for count in observed)
        assert len(observed) >= 3

    def test_signal_factory_required_when_backtest(self, tmp_path, shared_panel_root):
        """P1-3/R03:oos_backtest=True 且未注入 signal_factory ⇒ fail-fast。"""
        root = shared_panel_root
        store = PanelDataStore(root)
        registry = ModelRegistry(models_dir=str(tmp_path / "m"))
        trainer = V2Trainer(registry=registry, main_board_codes=_main_codes(), source_mode="panel")
        storage, loader = _old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(tmp_path / "out"),
            backtest_runner=_stub_backtest_runner,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        with pytest.raises(ValueError, match="signal_factory"):
            orch.run(_config(), 1, _days()[-1], trainer, store, test_window_months=1)

    def test_data_state_covers_train_labels_and_suspend(self, tmp_path, shared_panel_root):
        """P1-1/R01:数据态清单覆盖训练窗 panel 分区 + 全部 labels 分区 + raw/suspend。"""
        root = shared_panel_root
        store = PanelDataStore(root)
        # raw/suspend 逐日分区(内核停牌日历实际读取)
        suspend_dir = root / "raw" / "suspend"
        suspend_dir.mkdir(parents=True, exist_ok=True)
        for day in _days():
            pd.DataFrame({"trade_date": [day], "ts_code": [_CODES[0]]}).to_parquet(
                suspend_dir / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
            )
        output_root = tmp_path / "out"
        registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = _old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=_stub_backtest_runner,
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        wf_run_id = orch.run(_config(), 1, _days()[-1], trainer, store, test_window_months=1)
        import json

        payload = json.loads(
            (output_root / "walk_forward" / "raw" / f"data_state_{wf_run_id}.json").read_text(
                encoding="utf-8"
            )
        )
        paths = {r["path"] for r in payload["input_files"]}
        assert any(p.startswith("labels/y_ret_20/") for p in paths), "labels 分区缺失"
        assert any(p.startswith("features/panel/") for p in paths), "panel 分区缺失"
        assert any(p.startswith("raw/suspend/") for p in paths), "raw/suspend 缺失"
        # 训练窗首日(20240102)的标签与 panel 分区均在清单内
        assert "labels/y_ret_20/20240102.parquet" in paths
        assert "features/panel/20240102/core.parquet" in paths

        # reviewer1 §5 P3-4:附加段齐备(水位/manifest 指纹/代码态)且
        # summary 摘要列 data_state_id 与 JSON 内 id 同口径
        for key in ("watermarks", "manifest_fingerprint", "code_state"):
            assert key in payload, f"data_state 附加段缺失: {key}"
        summary_csv = output_root / "walk_forward" / "raw" / f"walk_forward_summary_{wf_run_id}.csv"
        summary_df = pd.read_csv(summary_csv, encoding="utf-8-sig")
        assert summary_df["data_state_id"].iloc[0] == payload["data_state_id"]

    def test_drift_passthrough_not_swallowed(self, tmp_path, monkeypatch, shared_panel_root):
        """P2-1:输入漂移异常穿透 split 级 continue(run 必须失败)。"""
        from src.lazybull.v2.store.store_identity import InputIdentityDriftError
        from src.lazybull.v2.train.walk_forward import orchestrator as orch_mod
        from src.lazybull.v2.train.walk_forward import split_training as st_mod

        root = shared_panel_root
        store = PanelDataStore(root)
        registry = ModelRegistry(models_dir=str(tmp_path / "m"))
        trainer = V2Trainer(registry=registry, main_board_codes=_main_codes(), source_mode="panel")
        storage, loader = _old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(tmp_path / "out"),
            backtest_runner=_stub_backtest_runner,
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )

        def boom(*a, **k):
            raise InputIdentityDriftError("注入漂移")

        monkeypatch.setattr(st_mod, "execute_split_training", boom)
        monkeypatch.setattr(orch_mod, "execute_split_training", boom)
        with pytest.raises(InputIdentityDriftError):
            orch.run(_config(), 1, _days()[-1], trainer, store, test_window_months=1)

    def test_f10_spec_wired_into_metadata(self, tmp_path, shared_cs_root):
        """P1-4/R05:三 spec 原文 + canonical digest + domain + 配方字段入注册 metadata。"""
        from src.lazybull.v2.store.store_identity import canonical_json_sha256

        cs_root = shared_cs_root
        days = _days()
        trainer = _new_trainer(cs_root, source_mode=SOURCE_CS_TRAIN_VERIFY, models_root=tmp_path)
        config = _config()
        trade_dates = days[:_TRAIN_DAYS]
        t_start, t_end, v_start, v_end = resolve_explicit_split_dates(
            days[0], days[_TRAIN_DAYS - 1], 0.2, trade_dates, _DELTA
        )
        version = trainer.train_fold(config, t_start, t_end, v_start, v_end, None)
        registry = trainer._registry
        metadata = registry._load_metadata(int(version))
        tp = metadata["train_params"]
        assert tp["label_spec"] == {"task": "regression", "label_transform": "raw", "version": 1}
        assert tp["split_spec"] == {"val_ratio": 0.2, "version": 1}
        assert tp["feature_recipe"] == {
            "enable_fundamental_features": False,
            "max_feature_missing_ratio": 0.6,
            "version": 1,
        }
        assert tp["label_spec_digest"] == canonical_json_sha256(tp["label_spec"])
        assert tp["split_spec_digest"] == canonical_json_sha256(tp["split_spec"])
        assert tp["domain"] == "main"
        for key in (
            "factor_prune",
            "max_feature_missing_ratio",
            "rank_weight_topk",
            "rank_weight",
            "time_decay_half_life",
        ):
            assert key in tp

    def test_model_config_mapping_immutable(self):
        """R05:原映射修改不影响已构造配置(不可变快照)。"""
        spec = {"val_ratio": 0.2}
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=("a",),
            label_name="y",
            hyperparams={},
            train_window_months=72,
            split_spec=spec,
        )
        spec["val_ratio"] = 0.9  # 原映射篡改
        assert config.split_spec["val_ratio"] == 0.2
        with pytest.raises(TypeError):
            config.split_spec["val_ratio"] = 0.9  # type: ignore[index]


# ---------------------------------------------------------------------------
# 7. 真实 replay 内核集成 + 区分力(codex-R07 核心;reviewer1-P1-3 守卫证据)
# ---------------------------------------------------------------------------


class TestRealReplayKernelIntegration:
    """真实 v2 内核回测实际消费 panel 特征(非 stub),差异进入信号/成交/净值。"""

    def test_real_replay_consumes_panel_with_discrimination(
        self, tmp_path, monkeypatch, shared_kernel_root
    ):
        root = shared_kernel_root
        store = PanelDataStore(root)
        models_dir = tmp_path / "registry" / "stock_selection"
        feature_columns = _train_panel_fold_v1(root, store, models_dir)

        # panel 读取窄化探针:仅挂回测段(训练段查询已完成,不混入)
        queries: List[List[str]] = []
        original_load = store.load_features

        def spy_load(query: FeatureQuery) -> Any:
            queries.append(list(query.columns))
            return original_load(query)

        monkeypatch.setattr(store, "load_features", spy_load)
        metrics_base = _run_real_replay(root, store, models_dir, feature_columns)

        # 真实内核跑通:metrics 非空 + 回测交易日 > 0 + 成交帧为 DataFrame
        assert metrics_base, "真实内核回测返回空 metrics"
        assert metrics_base["bt_trading_days"] > 0
        assert isinstance(metrics_base["_trades"], pd.DataFrame)

        # 列集窄化:panel 读取列集 == 折模型实际特征列(reviewer1-P2-4 口径)
        assert queries, "真实内核未触发 panel load_features"
        for cols in queries:
            assert set(cols) == set(feature_columns)

        # 区分力:异值 panel(salt=9.0)+ 同一注册模型 ⇒ 决策产物不完全相同
        salt_root = tmp_path / "salt"
        salt_store = _build_kernel_panel_root(salt_root, salt=9.0)
        _write_engine_daily(salt_root)
        metrics_salt = _run_real_replay(salt_root, salt_store, models_dir, feature_columns)
        assert metrics_salt, "异值 panel 臂真实内核回测返回空 metrics"
        assert not _bt_outputs_identical(
            metrics_base, metrics_salt
        ), "异值 panel 未进入决策——两臂 metrics/成交/净值完全一致"

    def test_same_version_different_model_root_binding(self, tmp_path, shared_kernel_root):
        """双根同号异模型守卫:信号 models_dir 绑定根决定实际消费的模型内容。

        根 A / 根 B 各注册同版本号 v1 的不同内容模型(异种子 + 异超参);
        同一 OOS 窗「根 A 数据面 + 根 B 信号」与「根 A 数据面 + 根 A 信号」对跑,
        成交/净值必须不同(证明信号根绑定生效,而非 data_root 缺省解析)。
        缺省信号工厂在 oos_backtest=True 时的 ValueError 已由
        test_signal_factory_required_when_backtest 覆盖,不重复。
        """
        root_a = shared_kernel_root
        store_a = PanelDataStore(root_a)
        models_a = tmp_path / "reg_a" / "stock_selection"
        feature_columns = _train_panel_fold_v1(root_a, store_a, models_a)

        root_b = tmp_path / "b"
        # 根 B:同号 v1 不同内容(反向排序标签 ⇒ 选股序与根 A 模型相反;另加异种子/超参)
        store_b = _build_kernel_panel_root(root_b, label_sign=-1.0)
        models_b = tmp_path / "reg_b" / "stock_selection"
        config_b = _config_with(hyperparams_over={"n_estimators": 30}, ensemble_seeds=(7, 9, 11))
        feature_columns_b = _train_panel_fold_v1(root_b, store_b, models_b, config=config_b)
        assert set(feature_columns_b) == set(feature_columns)  # 同数据 ⇒ 同折模型列集

        metrics_b = _run_real_replay(root_a, store_a, models_b, feature_columns)
        metrics_a = _run_real_replay(root_a, store_a, models_a, feature_columns)
        assert metrics_a and metrics_b
        assert not _bt_outputs_identical(
            metrics_a, metrics_b
        ), "同号异内容模型未产生差异——信号根绑定未生效"


# ---------------------------------------------------------------------------
# 8. runs 契约批次宿主适配 + 四键证据(codex-R07 后半 + reviewer1-P2-5)
# ---------------------------------------------------------------------------


class TestRunsBatchHostAdapter:
    """orchestrator.run 经 ``runs_batch_writer`` 适配器产出 runs 契约批次。"""

    def test_orchestrator_runs_batch_with_real_kernel(self, tmp_path, shared_kernel_root):
        from src.lazybull.v2.hosts.backtest.replay import run_oos_backtest

        root = shared_kernel_root
        store = PanelDataStore(root)
        output_root = tmp_path / "out"
        runs_root = tmp_path / "runs"
        days = _days()
        models_dir = output_root / "models" / "stock_selection"
        registry = ModelRegistry(models_dir=str(models_dir))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = _old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest=BacktestHostParams(bt_top_n=2, bt_rebalance_freq=20),
            # 真实 v2 内核(非 stub)+ 绑定隔离输出根的真实信号工厂
            backtest_runner=run_oos_backtest,
            signal_factory=lambda top_n: _make_ml_signal(models_dir, top_n=top_n),
            runs_batch_writer=functools.partial(_runs_writer_adapter, runs_root=runs_root),
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        wf_run_id = orch.run(_config(), 1, days[-1], trainer, store, test_window_months=1)

        # runs 批次结构(契约 §1 parquet 形态;写后已经 load_runs_batch 自校验)
        batch_dir = runs_root / wf_run_id
        assert (batch_dir / "batch_meta.json").is_file()
        assert (batch_dir / "summary.parquet").is_file()
        assert (batch_dir / "chain_nav.parquet").is_file()
        fold_dir = batch_dir / "folds" / "split00"
        assert (fold_dir / "trades.parquet").is_file()
        assert (fold_dir / "daily.parquet").is_file()
        assert (fold_dir / "_meta.json").is_file()

        # batch_meta 血缘:batch_id=wf_run_id 且 data_state_id = 增强口径
        raw_dir = output_root / "walk_forward" / "raw"
        state_payload = json.loads(
            (raw_dir / f"data_state_{wf_run_id}.json").read_text(encoding="utf-8")
        )
        batch_meta = json.loads((batch_dir / "batch_meta.json").read_text(encoding="utf-8"))
        assert batch_meta["batch_id"] == wf_run_id
        assert batch_meta["data_state"]["data_state_id"] == state_payload["data_state_id"]

        # 四键关联定位:wf_run_id → summary CSV + data_state JSON + runs 批次 + 模型版本
        summary_csv = raw_dir / f"walk_forward_summary_{wf_run_id}.csv"
        assert summary_csv.is_file()
        runs_summary = pd.read_parquet(batch_dir / "summary.parquet")
        assert runs_summary["wf_run_id"].iloc[0] == wf_run_id
        assert int(runs_summary["model_version"].iloc[0]) == 1
        summary_df = pd.read_csv(
            summary_csv, encoding="utf-8-sig", dtype={"data_state_id": str}
        )
        assert summary_df["data_state_id"].iloc[0] == state_payload["data_state_id"]
        assert int(summary_df["model_version"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# 9. 清单补齐(reviewer1-P2-5):域过滤顺序 / deploy 输出根隔离
# ---------------------------------------------------------------------------


class TestDomainFilterOrder:
    """§17.10-7:域过滤(``_filter_to_main_board``)在帧加载后立即执行,训练/评估同口径。"""

    def test_gem_stock_filtered_at_frame_entry(self, tmp_path, monkeypatch):
        root = tmp_path / "iso"
        store = _build_gem_panel_root(root)
        output_root = tmp_path / "out"
        days = _days()
        registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = _old_path_objects(root)

        # 训练侧(v2 training_core)与评估侧(split_training)过滤点同步探针
        from src.lazybull.v2.train.walk_forward import split_training as st_mod
        from src.lazybull.v2.train.walk_forward import training_core as tc_mod

        seen: Dict[str, Dict[str, set]] = {}
        for mod in (tc_mod, st_mod):
            original = mod._filter_to_main_board

            def spy(df: pd.DataFrame, codes: set, stage: str, _orig: Any = original):
                out = _orig(df, codes, stage)
                seen[stage] = {
                    "input": set(df["ts_code"].astype(str)),
                    "output": set(out["ts_code"].astype(str)),
                }
                return out

            monkeypatch.setattr(mod, "_filter_to_main_board", spy)

        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=_stub_backtest_runner,
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        orch.run(_config(), 1, days[-1], trainer, store, test_window_months=1)

        # 训练侧:适配层不承担域过滤(输入含创业板股),过滤后立即剔除
        assert _GEM_CODE in seen["训练窗口"]["input"]
        assert _GEM_CODE not in seen["训练窗口"]["output"]
        # 评估侧:测试窗帧行集同口径剔除
        assert _GEM_CODE in seen["测试窗口"]["input"]
        assert _GEM_CODE not in seen["测试窗口"]["output"]

        # data_stats 过滤语义:样本量 = 域内 24 股 × 外层窗交易日数(prepare 无额外行损)
        result = trainer.last_fold_result
        assert result is not None
        assert result.total_train_samples == result.train_days_count * len(_CODES)
        assert int(result.data_stats["samples_after_filter"]) == result.total_train_samples


class TestDeployOutputIsolation:
    """§17.10-9:deploy 训练只写隔离 output_root,只读 data_root 不触碰。"""

    def test_deploy_model_registers_to_output_root_only(self, tmp_path, shared_panel_root):
        root = shared_panel_root
        store = PanelDataStore(root)
        output_root = tmp_path / "out"
        days = _days()
        models_dir = output_root / "models" / "stock_selection"
        registry = ModelRegistry(models_dir=str(models_dir))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = _old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=_stub_backtest_runner,
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        # 合成日历 160 日:deploy 窗 = [days[0], days[-1]](72 月对齐回首日),
        # 与最后折训练窗(days[0] ~ 折 train_end < days[-1])不完全重叠 ⇒ 合法
        orch.run(
            _config(),
            1,
            days[-1],
            trainer,
            store,
            test_window_months=1,
            deploy_train=True,
        )

        # deploy 模型注册在 output_root(新版本 v2 存在;is_deploy 标记;窗尾=final_date)
        deploy_meta = registry._load_metadata(2)
        assert deploy_meta["train_params"]["is_deploy"] is True
        assert str(deploy_meta["train_end_date"]) == days[-1]

        # 台账:deploy 行(split_index='deploy')落 output_root 台账;与最后折不重叠
        ledger = pd.read_csv(models_dir / "ml_train_runs.csv", dtype=str)
        assert set(ledger["split_index"]) == {"0", "deploy"}
        deploy_row = ledger[ledger["split_index"] == "deploy"].iloc[0]
        fold_row = ledger[ledger["split_index"] == "0"].iloc[0]
        assert deploy_row["end_date"] == days[-1]
        assert fold_row["end_date"] < deploy_row["end_date"]

        # 只读输入根不触碰:无 models/ 目录(或为空)
        data_models = root / "models"
        assert not data_models.exists() or not any(data_models.rglob("*"))
