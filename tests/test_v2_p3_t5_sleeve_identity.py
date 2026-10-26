# -*- coding: utf-8 -*-
"""P3-T5 袖子身份（sleeve_identity）贯通验收（§19.6-①；T4 评审承接项 P2-2-①）。

证据面：
1. trainer 注入 ``sleeve_identity`` ⇒ 注册 metadata 的 train_params additive 含
   ``sleeve_id`` / ``sleeve_spec_version`` 两键且值正确（MagicMock registry 截获
   ``register_model`` 调用）；
2. 默认 None ⇒ metadata 无两键，且 orchestrator 调 ``runs_batch_writer`` 不带
   ``extra_meta_notes`` kwarg（「None 时零行为变化」硬契约的接缝证据）；
3. runs 批次 ``batch_meta.json`` 的 ``meta_notes`` 含袖子身份（合成 writer 截获
   并透传真实 ``write_runs_batch`` 落盘后读回断言）；
4. host 与 trainer 身份同时非 None 且不等 ⇒ run() 开头 ValueError（fail-fast）；
5. 构造期校验负例参数化：空串 / 前后空白 / 0 / -1 / True / "1" / 非二元组。

全部 tmp_path 合成数据 + 真实 xgboost 小样本训练。夹具口径复制自
``tests/test_v2_train_wf_orchestrator_e2e.py``（本仓测试文件自包含约定，
不跨测试模块导入）。
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.types import ModelConfig, TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.train.frame_cache import TrainingFrameCache
from src.lazybull.v2.train.trainer import V2Trainer, params_from_config
from src.lazybull.v2.train.walk_forward.orchestrator import (
    OrchestratorHostConfig,
    V2WalkForwardOrchestrator,
)

# ---------------------------------------------------------------------------
# 合成数据夹具（与 test_v2_train_wf_orchestrator_e2e.py 同口径）
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


def _days() -> List[str]:
    return pd.bdate_range("20240102", periods=_TRAIN_DAYS + 40).strftime("%Y%m%d").tolist()


def _cell_value(day_idx: int, code_idx: int, col_idx: int) -> float:
    """确定性特征值。"""
    return round((day_idx + 1) * 0.01 + (code_idx + 1) * 0.001 + (col_idx + 1) * 0.0001, 6)


def _frame_rows(day: str, day_idx: int) -> Dict[str, list]:
    payload: Dict[str, list] = {
        "trade_date": [day] * len(_CODES),
        "ts_code": list(_CODES),
    }
    for col_idx, col in enumerate(_BASE_COLUMNS):
        payload[col] = [_cell_value(day_idx, code_idx, col_idx) for code_idx in range(len(_CODES))]
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


def _populate_store(
    store: PanelDataStore, rows_fn: Callable[[str, int], Dict[str, list]]
) -> PanelDataStore:
    """panel store 填充（全部族 + 六标签表；``rows_fn(day, day_idx)`` 供给当日行 payload）。"""
    store.manifest.register_columns(
        {
            **{col: {"group": "core", "source": "test"} for col in _BASE_COLUMNS},
            **{col: {"group": "core", "source": "test"} for col in _FILTER_COLS},
        }
    )
    store.manifest.save()
    from src.lazybull.v2.store.column_groups import PANEL_GROUPS

    for day_idx, day in enumerate(_days()):
        rows = rows_fn(day, day_idx)
        for group in PANEL_GROUPS:
            if group == "core":
                core_cols = ("trade_date", "ts_code") + tuple(_BASE_COLUMNS) + tuple(_FILTER_COLS)
                payload = {col: rows[col] for col in core_cols}
            else:
                # readiness 要求八族当日齐备（非 core 族仅键列）
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


def _build_panel_root(root: Path) -> PanelDataStore:
    _write_base_inputs(root)
    return _populate_store(PanelDataStore(root), _frame_rows)


def _old_path_objects(root: Path):
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(root))
    return storage, DataLoader(storage)


def _main_codes() -> set:
    return set(_CODES)


def _config() -> ModelConfig:
    return ModelConfig(
        model_family="stock_selection",
        feature_columns=tuple(_BASE_COLUMNS + _FILTER_COLS),
        label_name="neu_y_ret_20",
        hyperparams={
            "algorithm": "xgboost",
            "n_estimators": 20,
            "max_depth": 3,
            "early_stopping_rounds": 10,
            "early_stopping_metric": "rank_ic",
            "rank_weight_enabled": False,
            "random_state": 42,
        },
        train_window_months=72,  # 生产口径 6 年;合成日历 160 日 ⇒ 起点对齐日历首日
        ensemble_seeds=(42, 61, 82),
        label_spec={"task": "regression", "label_transform": "raw", "version": 1},
        split_spec={"val_ratio": 0.2, "version": 1},
        # 全 False 配方（合成数据仅基础列;开族 ⇒ prepare 缺列 fail-fast）
        feature_recipe={
            "enable_fundamental_features": False,
            "max_feature_missing_ratio": 0.6,
            "version": 1,
        },
        domain="main",
    )


def _minimal_config() -> ModelConfig:
    """轻量配置（守卫/注册接缝测试用,三 spec 带 version 满足 F10 版本门）。"""
    return ModelConfig(
        model_family="stock_selection",
        feature_columns=("f1", "f2"),
        label_name="neu_y_ret_20",
        hyperparams={"algorithm": "xgboost", "random_state": 42},
        train_window_months=72,
        label_spec={"task": "regression", "label_transform": "raw", "version": 1},
        split_spec={"val_ratio": 0.2, "version": 1},
        feature_recipe={"max_feature_missing_ratio": 0.6, "version": 1},
        domain="main",
    )


def _stub_backtest_runner(**kwargs) -> Dict[str, Any]:
    """宿主注入的回测 stub：合成净值/空成交产物（空折账本契约合法）。

    净值曲线列形态对齐真实 v2 内核（``date`` 列 + RangeIndex，
    见 ``v2/core/accounting/nav.py::generate_nav_curve``）——``write_runs_batch``
    的 ``_build_daily`` 依赖 ``date`` 列重建逐日明细。
    """
    bt_start = kwargs["bt_start"]
    dates = pd.bdate_range(bt_start, periods=3)
    nav = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": [1.0, 1.005, 1.011],
            "capital": [1.0, 1.005, 1.011],
            "market_value": [0.0, 0.0, 0.0],
            "nav": [1.0, 1.005, 1.011],
            "return": [0.0, 0.005, 0.011],
        }
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


def _run_single_fold_wf(
    tmp_path: Path,
    *,
    sleeve_identity: Optional[tuple[str, int]],
    runs_writer: Callable[..., Any],
) -> str:
    """单折全流程（训练注册 → OOS 评估 → stub 回测 → 落盘 + runs writer 接缝）。"""
    root = tmp_path / "iso"
    store = _build_panel_root(root)
    output_root = tmp_path / "out"
    days = _days()
    registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
    trainer = V2Trainer(
        registry=registry,
        main_board_codes=_main_codes(),
        source_mode="panel",
        frame_cache=TrainingFrameCache(),
        sleeve_identity=sleeve_identity,
    )
    storage, loader = _old_path_objects(root)
    host = OrchestratorHostConfig(
        data_root=str(root),
        output_root=str(output_root),
        backtest_runner=_stub_backtest_runner,
        signal_factory=lambda top_n: None,  # 回测启用必须注入工厂（P1-3 守卫口径）
        runs_batch_writer=runs_writer,
        sleeve_identity=sleeve_identity,
    )
    orch = V2WalkForwardOrchestrator(
        host,
        registry=registry,
        source_mode="panel",
        storage=storage,
        loader=loader,
    )
    return orch.run(_config(), 1, days[-1], trainer, store, test_window_months=1)


# ---------------------------------------------------------------------------
# 1. trainer 注册 metadata 接缝（MagicMock registry 截获 register_model）
# ---------------------------------------------------------------------------


class TestTrainerSleeveIdentityMetadata:
    """trainer 注入 ⇒ 注册 metadata additive 两键；默认 None ⇒ 无两键。"""

    def _capture_train_params(self, sleeve_identity: Optional[tuple[str, int]]) -> Dict[str, Any]:
        registry = MagicMock()
        registry.register_model.return_value = 1
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=_main_codes(),
            sleeve_identity=sleeve_identity,
        )
        config = _minimal_config()
        trainer._register_fold(
            config=config,
            model=object(),  # 非 EnsembleModel ⇒ 走单模型分支
            params=params_from_config(config),
            feature_columns=["f1", "f2"],
            train_params={"learning_rate": 0.05},
            performance_metrics={},
            train_start="20240101",
            val_end="20240630",
            X_train_len=100,
            X_val_len=20,
            ensemble_offsets=0,
            ensemble_seeds=[42],
        )
        return registry.register_model.call_args.kwargs["train_params"]

    def test_register_metadata_contains_sleeve_identity(self):
        train_params = self._capture_train_params(("core_value", 3))
        assert train_params["sleeve_id"] == "core_value"
        assert train_params["sleeve_spec_version"] == 3

    def test_register_metadata_default_none_without_sleeve_keys(self):
        train_params = self._capture_train_params(None)
        assert "sleeve_id" not in train_params
        assert "sleeve_spec_version" not in train_params


# ---------------------------------------------------------------------------
# 2. 构造期校验（trainer 与 host 共用同一 validate_sleeve_identity 实现）
# ---------------------------------------------------------------------------


class TestSleeveIdentityValidation:
    def test_identity_accepted_and_default_none(self):
        trainer = V2Trainer(
            registry=MagicMock(),
            main_board_codes=set(),
            sleeve_identity=("core_value", 3),
        )
        assert trainer.sleeve_identity == ("core_value", 3)
        host = OrchestratorHostConfig(
            data_root="a",
            output_root="b",
            sleeve_identity=("core_value", 3),
        )
        assert host.sleeve_identity == ("core_value", 3)
        # 默认 None（现状零变化）
        assert V2Trainer(registry=MagicMock(), main_board_codes=set()).sleeve_identity is None
        assert OrchestratorHostConfig(data_root="a", output_root="b").sleeve_identity is None

    @pytest.mark.parametrize(
        "bad_identity",
        [
            ("", 1),  # 空串 sleeve_id
            (" value ", 1),  # 前后空白
            ("a", 0),  # spec_version < 1
            ("a", -1),
            ("a", True),  # bool 不是版本号
            ("a", "1"),  # str 版本号
            ["a", 1],  # 非 tuple
            ("a", 1, "x"),  # 非二元组
        ],
    )
    def test_invalid_identity_rejected(self, bad_identity):
        with pytest.raises(ValueError, match="sleeve_identity"):
            V2Trainer(
                registry=MagicMock(),
                main_board_codes=set(),
                sleeve_identity=bad_identity,
            )
        with pytest.raises(ValueError, match="sleeve_identity"):
            OrchestratorHostConfig(
                data_root="a",
                output_root="b",
                sleeve_identity=bad_identity,
            )


# ---------------------------------------------------------------------------
# 3. host × trainer 一致性守卫（run 期 fail-fast）+ 有效值解析
# ---------------------------------------------------------------------------


class TestSleeveIdentityGuard:
    def test_host_trainer_mismatch_raises(self, tmp_path):
        host = OrchestratorHostConfig(
            data_root=str(tmp_path),
            output_root=str(tmp_path / "out"),
            sleeve_identity=("sleeve_b", 1),
        )
        orch = V2WalkForwardOrchestrator(host, registry=MagicMock(), source_mode="panel")
        trainer = V2Trainer(
            registry=MagicMock(),
            main_board_codes=set(),
            sleeve_identity=("sleeve_a", 1),
        )
        with pytest.raises(ValueError, match="袖子身份"):
            orch.run(_minimal_config(), 1, "20240630", trainer, None, test_window_months=1)

    def test_resolve_single_side_and_both_none(self):
        trainer_plain = V2Trainer(registry=MagicMock(), main_board_codes=set())
        trainer_with = V2Trainer(
            registry=MagicMock(),
            main_board_codes=set(),
            sleeve_identity=("t_side", 4),
        )
        orch_host_only = V2WalkForwardOrchestrator(
            OrchestratorHostConfig(
                data_root="a",
                output_root="b",
                sleeve_identity=("h_side", 2),
            ),
            registry=MagicMock(),
        )
        orch_plain = V2WalkForwardOrchestrator(
            OrchestratorHostConfig(data_root="a", output_root="b"),
            registry=MagicMock(),
        )
        # 仅宿主携带 ⇒ 以宿主为准；仅 trainer 携带 ⇒ 以 trainer 为准；双 None ⇒ None
        assert orch_host_only._resolve_sleeve_identity(trainer_plain) == ("h_side", 2)
        assert orch_plain._resolve_sleeve_identity(trainer_with) == ("t_side", 4)
        assert orch_plain._resolve_sleeve_identity(trainer_plain) is None


# ---------------------------------------------------------------------------
# 4. runs 批次接缝（合成 writer 截获 + 真实 write_runs_batch 落盘读回）
# ---------------------------------------------------------------------------


class TestRunsBatchSleeveIdentity:
    def test_batch_meta_contains_sleeve_identity(self, tmp_path):
        """host 与 trainer 同值注入 ⇒ batch_meta.meta_notes 含袖子身份。"""
        from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch

        runs_root = tmp_path / "runs"
        captured: Dict[str, Any] = {}

        def writer(results, wf_run_id, data_state_payload, *, extra_meta_notes=None):
            captured["extra_meta_notes"] = extra_meta_notes
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
                source="p3_t5_sleeve_identity_acceptance",
                extra_meta_notes=extra_meta_notes,
            )

        wf_run_id = _run_single_fold_wf(
            tmp_path,
            sleeve_identity=("core_value", 3),
            runs_writer=writer,
        )

        # 接缝截获:orchestrator 经 extra_meta_notes 传入袖子身份(spec_version 字符串化)
        assert captured["extra_meta_notes"] == {
            "sleeve_id": "core_value",
            "spec_version": "3",
        }
        # 落盘读回:batch_meta.meta_notes 含袖子身份
        batch_meta = json.loads(
            (runs_root / wf_run_id / "batch_meta.json").read_text(encoding="utf-8")
        )
        assert batch_meta["meta_notes"]["sleeve_id"] == "core_value"
        assert batch_meta["meta_notes"]["spec_version"] == "3"

    def test_batch_writer_default_none_call_unchanged(self, tmp_path):
        """默认 None ⇒ writer 调用不带任何附加 kwarg(零行为变化接缝证据)。"""
        captured: Dict[str, Any] = {}

        def writer(results, wf_run_id, data_state_payload, **kwargs):
            captured["extra_kwargs"] = kwargs
            return None

        _run_single_fold_wf(tmp_path, sleeve_identity=None, runs_writer=writer)
        assert captured["extra_kwargs"] == {}
