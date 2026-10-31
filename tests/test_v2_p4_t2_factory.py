# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主装配工厂（factory）测试（tests/ 全量豁免复杂度）。

覆盖（T2 评审处置轮）：
- F7 展示同源：``format_host_model_info`` 消费 ``host.resolved_model_versions``
  （装配期 ``_resolve_model_versions`` 产出）——两个隔离纸面根配置不同模型版本 ⇒
  各自展示版本 == 各自信号实际加载版本且不读另一个根（同版本号跨根元数据不同，
  串根即检出）；``model_version=None``（latest 解析）⇒ 展示版本 == 实际加载版本，
  且装配后注册新版本不影响展示（不再二次解析 latest）；ensemble 双臂渲染；
  直构造宿主回退 trading_config；已解析版本在注册表缺失 ⇒ 报告可用版本。
- F8 派生族并集：EnsembleSignal 的 train_params 派生族开关键取 A/B 两臂并集
  （B 启用 A 未启用 ⇒ 并集启用；反向同；双未启用 ⇒ 缺键），非开关键仍以 A 为基底；
  工厂派生路径流向 ``host.train_params``（就绪门第 5 条判定基准，门侧归另一代理）。
- F21 注入接缝：``cost_model`` 注入透传 / 缺省自配置构建；自定义 DataLoader 子类
  实例经 ``data_loader`` 透传宿主（``host.loader is`` 注入实例，不被重建掉子类语义）。

测试隔离硬契约：全部 tmp_path 隔离根 + 合成数据 + 桩模型注册表，禁止读真实 data/。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_p4.t05b_spike_runner import _DeterministicSignal
from src.lazybull.data import DataLoader, Storage
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.core.signal.ensemble_signal import EnsembleSignal
from src.lazybull.v2.core.signal.ml_signal import MLSignal
from src.lazybull.v2.hosts.paper.daily_assembly import DERIVED_FAMILY_SWITCHES
from src.lazybull.v2.hosts.paper.factory import (
    _derive_signal_metadata,
    create_paper_host,
    format_host_model_info,
)
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.store.data_store import PanelDataStore

# ---------------------------------------------------------------------------
# 合成数据基建（镜像 tests/test_v2_p4_t2_host.py 的工厂段构造方式，自包含）
# ---------------------------------------------------------------------------

_CODES = [f"0000{i:02d}.SZ" for i in range(6)]
_CALENDAR_STRS = pd.bdate_range("2026-07-06", periods=10).strftime("%Y%m%d").tolist()


class _MockModel:
    """最小模型桩（无 save_model ⇒ registry 回退 joblib 落盘/加载）。"""

    def predict(self, X):  # pragma: no cover - 测试不需要真实推理
        return [0.0] * len(X)


def _register_mock_model(
    models_dir: Path,
    *,
    train_params: dict | None = None,
    feature_columns: tuple = ("ret_1", "bp"),
    train_start_date: str = "20230101",
    train_end_date: str = "20231231",
    n_samples: int = 100,
) -> int:
    """tmp 注册表注册一个 MockModel，返回版本号。"""
    registry = ModelRegistry(models_dir=str(models_dir))
    return registry.register_model(
        model=_MockModel(),
        model_type="xgboost",
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        feature_columns=list(feature_columns),
        label_column="y_ret_5",
        n_samples=n_samples,
        train_params=dict(train_params or {"max_depth": 6}),
    )


def _write_storage(root: Path) -> Storage:
    """指定根 Storage：clean/daily 逐日分区 + raw stock_basic/trade_cal（含未来日）。"""
    storage = Storage(root_path=str(root))
    daily_dir = storage.clean_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    n = len(_CODES)
    closes = [10.0 * (1 + 0.1 * j) for j in range(n)]
    for day in _CALENDAR_STRS:
        frame = pd.DataFrame(
            {
                "trade_date": [day] * n,
                "ts_code": _CODES,
                "close": closes,
                "close_adj": closes,
                "open": [round(c * 0.999, 4) for c in closes],
                "open_adj": [round(c * 0.999, 4) for c in closes],
                "is_suspended": [0] * n,
                "is_limit_up": [0] * n,
                "is_limit_down": [0] * n,
                "vol": np.full(n, 1000.0),
                "pct_chg": np.full(n, 0.5),
                "is_st": [0] * n,
                "list_days": [1000] * n,
                "tradable": [1] * n,
            }
        )
        frame.to_parquet(daily_dir / f"{day[:4]}-{day[4:6]}-{day[6:8]}.parquet", index=False)
    stock_basic = pd.DataFrame(
        {
            "ts_code": _CODES[:3] + ["300001.SZ"],
            "name": ["甲", "乙", "丙", "创业板丁"],
            "market": ["主板", "主板", "主板", "创业板"],
            "list_date": ["20200101"] * 4,
        }
    )
    stock_basic.to_parquet(storage.raw_path / "stock_basic.parquet", index=False)
    calendar = pd.DataFrame({"cal_date": _CALENDAR_STRS, "is_open": [1] * len(_CALENDAR_STRS)})
    calendar.to_parquet(storage.raw_path / "trade_cal.parquet", index=False)
    return storage


def _flat_config(**overrides) -> dict:
    """纸面扁平配置（合成；top_n=2 / freq=5 / K=1 / 等权 / close 买 open 卖 / v2 引擎）。"""
    config = {
        "model_version": 1,
        "top_n": 2,
        "rebalance_freq": 5,
        "stagger_tranches": 1,
        "exclude_st": True,
        "min_list_days": 365,
        "position_sizing": "equal",
        "buy_price": "close",
        "sell_price": "open",
        "initial_capital": 1_000_000.0,
        "min_buy_value_ratio": 0.0,
        "horizon": 20,
        "universe": "all",
        "engine": "v2",
    }
    config.update(overrides)
    return config


def _build_host(root: Path, flat: dict, **extra) -> PaperHost:
    """经 create_paper_host 全接缝装配（真实信号路径：模型注册表在 root/models 下）。"""
    storage = _write_storage(root / "io")
    kwargs = dict(
        store=str(root / "paper"),
        config_flat=flat,
        data_root=str(root),
        loader=DataLoader(storage=storage),
        data_storage=storage,
        panel_store=PanelDataStore(root / "store"),
        verbose=False,
    )
    kwargs.update(extra)
    return create_paper_host(**kwargs)


# ---------------------------------------------------------------------------
# F7：展示与信号加载同源（format_host_model_info）
# ---------------------------------------------------------------------------


class TestFormatHostModelInfo:
    def test_isolated_roots_display_own_signal_version(self, tmp_path):
        # 根 A：仅 v1（标记 20220101/111）；根 B：v1（20230101/100）+ v2（20240101/999）
        root_a = tmp_path / "a"
        _register_mock_model(
            root_a / "models" / "stock_selection", train_start_date="20220101", n_samples=111
        )
        host_a = _build_host(root_a, _flat_config(model_version=1))
        root_b = tmp_path / "b"
        _register_mock_model(
            root_b / "models" / "stock_selection", train_start_date="20230101", n_samples=100
        )
        _register_mock_model(
            root_b / "models" / "stock_selection", train_start_date="20240101", n_samples=999
        )
        host_b = _build_host(root_b, _flat_config(model_version=2))

        text_a = format_host_model_info(host_a)
        text_b = format_host_model_info(host_b)
        # 各自展示版本 == 各自信号实际加载版本
        assert host_a.signal.model_version == 1
        assert host_a.resolved_model_versions["model_version"] == host_a.signal.model_version
        assert f"当前模型: v{host_a.signal.model_version}" in text_a
        assert host_b.signal.model_version == 2
        assert host_b.resolved_model_versions["model_version"] == host_b.signal.model_version
        assert f"当前模型: v{host_b.signal.model_version}" in text_b
        # 不读另一个根：同版本号 v1 跨根元数据不同，串根即检出
        assert "20220101" in text_a and "训练样本: 111" in text_a
        assert "20230101" not in text_a and "20240101" not in text_a
        assert "20240101" in text_b and "训练样本: 999" in text_b
        assert "20220101" not in text_b and "20230101" not in text_b
        # 解析产出登记的注册表目录各自指向本根
        assert (
            host_a.resolved_model_versions["models_dir"]
            != host_b.resolved_model_versions["models_dir"]
        )

    def test_latest_resolution_display_matches_loaded_and_no_reresolution(self, tmp_path):
        root = tmp_path / "r"
        models_dir = root / "models" / "stock_selection"
        _register_mock_model(models_dir, train_start_date="20230101")
        _register_mock_model(models_dir, train_start_date="20230102")
        host = _build_host(root, _flat_config(model_version=None))
        # latest 解析结果 == 信号实际加载版本（共享同一解析产出）
        assert host.signal.model_version == 2
        assert host.resolved_model_versions["model_version"] == 2
        assert host.resolved_model_versions["model_version_is_latest"] is True
        text = format_host_model_info(host)
        assert "当前模型: v2 (最新)" in text
        # 装配后注册 v3（latest 指针推进）：展示仍 == 已解析/已加载 v2，不再二次解析
        _register_mock_model(models_dir, train_start_date="20230103")
        text_after = format_host_model_info(host)
        assert "当前模型: v2 (最新)" in text_after
        assert "20230103" not in text_after

    def test_ensemble_display_renders_both_resolved_arms(self, tmp_path):
        root = tmp_path / "r"
        models_dir = root / "models" / "stock_selection"
        _register_mock_model(models_dir)
        _register_mock_model(models_dir)
        host = _build_host(
            root, _flat_config(model_version=1, model_version_b=2, ensemble_weight_a=0.25)
        )
        assert isinstance(host.signal, EnsembleSignal)
        assert host.signal.signal_a.model_version == 1
        assert host.signal.signal_b.model_version == 2
        assert host.resolved_model_versions["model_version_b"] == 2
        text = format_host_model_info(host)
        assert "当前模型: v1" in text
        assert "集成模式: 双模型 Ensemble" in text
        assert "模型A: v1 (权重 0.25)" in text
        assert "模型B: v2 (权重 0.75)" in text

    def test_direct_constructed_host_falls_back_to_trading_config(self, tmp_path):
        root = tmp_path / "r"
        models_dir = root / "models" / "stock_selection"
        _register_mock_model(models_dir)
        host = _build_host(root, _flat_config(model_version=1))
        del host.resolved_model_versions  # 直构造形态（无工厂挂点）
        text = format_host_model_info(host, models_dir=str(models_dir))
        assert "当前模型: v1" in text
        assert "(最新)" not in text

    def test_resolved_version_missing_reports_available(self, tmp_path):
        root = tmp_path / "r"
        _register_mock_model(root / "models" / "stock_selection")  # 仅 v1
        signal = _DeterministicSignal(2, {}, [])
        host = _build_host(
            root, _flat_config(model_version=7), signal=signal, feature_columns=["ret_1"]
        )
        text = format_host_model_info(host)
        assert "未找到版本 7 的模型" in text
        assert "可用版本: [1]" in text

    def test_unresolved_none_version_reports_unavailable(self, tmp_path):
        root = tmp_path / "r"
        _register_mock_model(root / "models" / "stock_selection")
        host = _build_host(root, _flat_config(model_version=1))
        host.trading_config.model_version = None  # 直构造未解析形态
        del host.resolved_model_versions
        assert "未解析" in format_host_model_info(host)


# ---------------------------------------------------------------------------
# F8：EnsembleSignal 派生族开关并集（_derive_signal_metadata）
# ---------------------------------------------------------------------------


class TestEnsembleDerivedFamilyUnion:
    def _ensemble(self, root: Path, params_a: dict, params_b: dict) -> EnsembleSignal:
        dir_a = root / "models_a"
        dir_b = root / "models_b"
        v_a = _register_mock_model(dir_a, train_params=params_a, feature_columns=("ret_1", "bp"))
        v_b = _register_mock_model(
            dir_b, train_params=params_b, feature_columns=("ret_1", "turnover_rate")
        )
        return EnsembleSignal(
            MLSignal(top_n=2, model_version=v_a, models_dir=str(dir_a), verbose=False),
            MLSignal(top_n=2, model_version=v_b, models_dir=str(dir_b), verbose=False),
        )

    def test_arm_b_enabled_union_reflected(self, tmp_path):
        signal = self._ensemble(
            tmp_path, {"max_depth": 6}, {"max_depth": 9, "enable_holdertrade_features": True}
        )
        columns, params = _derive_signal_metadata(signal)
        assert columns == ["ret_1", "bp", "turnover_rate"]  # 列集并集（保序，A 先）
        assert params["enable_holdertrade_features"] is True  # B 启用 ⇒ 并集启用
        assert params["max_depth"] == 6  # 非开关键仍以主模型 A 为基底

    def test_arm_a_enabled_union_reflected(self, tmp_path):
        signal = self._ensemble(tmp_path, {"enable_repurchase_features": True}, {"max_depth": 9})
        _, params = _derive_signal_metadata(signal)
        assert params["enable_repurchase_features"] is True

    def test_neither_arm_enabled_no_switch_keys(self, tmp_path):
        signal = self._ensemble(tmp_path, {"max_depth": 6}, {"max_depth": 9})
        _, params = _derive_signal_metadata(signal)
        for key in DERIVED_FAMILY_SWITCHES.values():
            assert not params.get(key), f"双未启用时 {key} 不得为真"

    def test_factory_derived_metadata_flows_to_host_train_params(self, tmp_path):
        signal = self._ensemble(
            tmp_path / "ens", {"max_depth": 6}, {"max_depth": 9, "enable_top_inst_features": 1}
        )
        host = _build_host(tmp_path / "root", _flat_config(model_version=7), signal=signal)
        # 工厂派生（未传 feature_columns）⇒ 并集流向宿主就绪门判定基准
        assert host.feature_columns == ["ret_1", "bp", "turnover_rate"]
        assert host.train_params.get("enable_top_inst_features") == 1

    def test_factory_preserves_per_arm_freshness_metadata(self, tmp_path):
        signal = self._ensemble(
            tmp_path / "ens",
            {"freshness_strategy": "state_keep_event_no_decay"},
            {"freshness_strategy": "state_keep_event_decay"},
        )
        host = _build_host(tmp_path / "root", _flat_config(model_version=7), signal=signal)
        assert host.freshness_members is not None
        assert len(host.freshness_members) == 2
        assert host.freshness_members[0][0] == ["ret_1", "bp"]
        assert host.freshness_members[1][0] == ["ret_1", "turnover_rate"]
        assert host.freshness_members[0][1]["freshness_strategy"] == "state_keep_event_no_decay"
        assert host.freshness_members[1][1]["freshness_strategy"] == "state_keep_event_decay"


# ---------------------------------------------------------------------------
# F21：cost_model 可注入 + data_loader 透传
# ---------------------------------------------------------------------------


class _CustomLoader(DataLoader):
    """自定义 DataLoader 子类（透传语义验证用）。"""


class TestFactoryInjectionSeams:
    def test_cost_model_injected_passthrough(self, tmp_path):
        root = tmp_path / "r"
        _register_mock_model(root / "models" / "stock_selection")
        custom = CostModel(commission_rate=0.01, min_commission=1.0, stamp_tax=0.0, slippage=0.0)
        host = _build_host(root, _flat_config(model_version=1), cost_model=custom)
        assert host.cost_model is custom

    def test_cost_model_default_constructed_from_config(self, tmp_path):
        root = tmp_path / "r"
        _register_mock_model(root / "models" / "stock_selection")
        host = _build_host(root, _flat_config(model_version=1))
        assert isinstance(host.cost_model, CostModel)

    def test_data_loader_passthrough_preserves_subclass(self, tmp_path):
        root = tmp_path / "r"
        _register_mock_model(root / "models" / "stock_selection")
        storage = _write_storage(root / "io")
        loader = _CustomLoader(storage=storage)
        host = create_paper_host(
            store=str(root / "paper"),
            config_flat=_flat_config(model_version=1),
            data_root=str(root),
            loader=loader,  # 只传 loader（不传 data_storage）：工厂自 loader 派生 storage
            panel_store=PanelDataStore(root / "store"),
            verbose=False,
        )
        assert host.loader is loader, "注入的自定义 loader 实例必须原样透传宿主"
        assert host.data_storage is loader.storage

    def test_default_loader_reaches_host(self, tmp_path):
        root = tmp_path / "r"
        _register_mock_model(root / "models" / "stock_selection")
        host = _build_host(root, _flat_config(model_version=1))
        assert isinstance(host.loader, DataLoader)
        assert host.loader.storage is host.data_storage
