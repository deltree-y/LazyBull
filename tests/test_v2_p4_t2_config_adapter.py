# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主配置适配（config_adapter）测试（tests/ 全量豁免复杂度）。

覆盖（对应任务卡测试矩阵 config 行）：
- 分组模板 → 扁平 dict（含 extra 块并入、未知块原样保留）；
- 退役键剔除（精确键 + 前缀两形态；区块内嵌退役键同样剔除）；
- engine 切换键三态（缺省=legacy / v2 / 非法 fail-fast；extra 块内嵌同样解析）；
- resolve_model_version（指定版本转 int / 配置对象 / 缺省走 registry latest /
  空注册表失败 / 非法版本值失败）；
- 执行价格冻结指纹（buy_price 非 close ⇒ 拒绝；sell_price 非法 ⇒ 拒绝）；
- to_engine_kwargs（enable_trailing_early_rebalance=False / sell_timing 映射 /
  completion_window_days=5 / 模板值 / rebalance_freq=None 拒绝）；
- weight_method 旧别名；exposure 家族透传不消费；downside_penalty 透传；
- F14：load_paper_config 只读形态（create_layout=False，读配置不建目录副作用）；
- F19：capital_retention_ratio 漂移防御（非 0/非 None ⇒ warning 且不 fail-fast；
  0/缺省 ⇒ 静默；适配输出不受影响）。

测试隔离硬契约：全部 tmp_path 隔离根 + 合成数据，禁止读真实 data/。
"""

from __future__ import annotations

import json

import pytest
import yaml
from loguru import logger

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.hosts.paper.config_adapter import (
    PAPER_COMPLETION_WINDOW_DAYS,
    PAPER_CONFIG_TO_V2,
    PaperHostError,
    load_paper_config,
    resolve_engine,
    resolve_model_version,
    to_engine_kwargs,
    to_trading_config,
)
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore


class _MockModel:
    """最小模型桩（joblib 落盘路径；无 save_model ⇒ registry 回退 joblib）。"""

    def predict(self, X):  # pragma: no cover - 测试不需要真实推理
        return [0.0] * len(X)


def _write_config(root, config: dict) -> PaperStateStore:
    """隔离根写入 config.yaml 并返回该根的 PaperStateStore。"""
    store = PaperStateStore(root_path=str(root))
    (root / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return store


def _grouped_config(**overrides) -> dict:
    """分组模板形态的合成配置（镜像 data/paper/config.yaml 结构）。"""
    config = {
        "model": {"model_version": 24022, "model_version_b": None, "ensemble_weight_a": 0.5},
        "portfolio": {
            "top_n": 20,
            "rebalance_freq": 20,
            "stagger_tranches": 2,
            "max_per_industry": 5,
            "max_weight_per_stock": 0.15,
            "enable_early_rebalance_on_empty": True,
            "exclude_st": True,
            "min_list_days": 365,
        },
        "stop_loss": {
            "stop_loss_enabled": False,
            "stop_loss_drawdown_pct": 30.0,
            "stop_loss_consecutive_limit_down": 2,
        },
        "position_management": {
            "position_sizing": "kelly",
            "kelly_vol_window": 60,
            "kelly_max_leverage": 0.2,
        },
        "paper_trade": {
            "buy_price": "close",
            "sell_price": "open",
            "initial_capital": 650000.0,
            "min_buy_value_ratio": 0.2,
            "horizon": 20,
            "universe": "mainboard",
        },
        "exposure": {"exposure_policy": None, "exposure_replenish": False},
        "signal_penalty": {"downside_penalty": 0.0, "downside_penalty_column": "downside_vol_20"},
    }
    config.update(overrides)
    return config


def _register_mock_model(models_dir, train_params=None) -> int:
    """tmp 注册表注册一个 MockModel，返回版本号。"""
    registry = ModelRegistry(models_dir=str(models_dir))
    return registry.register_model(
        model=_MockModel(),
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["ret_1", "bp"],
        label_column="y_ret_5",
        n_samples=100,
        train_params=train_params or {"max_depth": 6},
    )


# ---------------------------------------------------------------------------
# 分组 → 扁平
# ---------------------------------------------------------------------------


class TestLoadPaperConfigFlatten:
    def test_grouped_config_flattens(self, tmp_path):
        store = _write_config(tmp_path, _grouped_config())
        flat = load_paper_config(store)
        assert flat["model_version"] == 24022
        assert flat["top_n"] == 20
        assert flat["buy_price"] == "close"
        assert flat["universe"] == "mainboard"
        # 分区块名本身不残留在扁平 dict
        for section in ("model", "portfolio", "stop_loss", "position_management", "paper_trade"):
            assert section not in flat

    def test_extra_section_flattens_and_unknown_block_kept(self, tmp_path):
        raw = _grouped_config()
        raw["extra"] = {"my_custom_key": 1}
        raw["unknown_block"] = {"nested": True}  # 未知块：值原样保留（不展开）
        flat = load_paper_config(_write_config(tmp_path, raw))
        assert flat["my_custom_key"] == 1
        assert flat["unknown_block"] == {"nested": True}

    def test_root_path_and_none_forms(self, tmp_path):
        _write_config(tmp_path, _grouped_config())
        flat = load_paper_config(str(tmp_path))
        assert flat["top_n"] == 20

    def test_missing_config_fails(self, tmp_path):
        store = PaperStateStore(root_path=str(tmp_path / "paper"))
        with pytest.raises(PaperHostError, match="未找到配置文件"):
            load_paper_config(store)


class TestRetiredKeysStripped:
    def test_retired_exact_and_prefix_keys_stripped(self, tmp_path):
        raw = _grouped_config()
        raw["extra"] = {
            "signal_gate_mode": "hard",  # 精确退役键
            "market_regime_enabled": True,  # 前缀退役键
            "time_stop_loss_days": 3,
            "my_custom_key": 1,
        }
        flat = load_paper_config(_write_config(tmp_path, raw))
        assert "signal_gate_mode" not in flat
        assert "market_regime_enabled" not in flat
        assert "time_stop_loss_days" not in flat
        assert flat["my_custom_key"] == 1  # 未消费但非退役键：保留透传

    def test_retired_key_inside_section_stripped(self, tmp_path):
        raw = _grouped_config()
        raw["portfolio"]["signal_gate"] = {"mode": "x"}  # 区块内嵌退役整块键
        flat = load_paper_config(_write_config(tmp_path, raw))
        assert "signal_gate" not in flat


# ---------------------------------------------------------------------------
# engine 切换键（Q9）
# ---------------------------------------------------------------------------


class TestResolveEngine:
    def test_default_is_legacy(self):
        assert resolve_engine({}) == "legacy"
        assert resolve_engine({"engine": None}) == "legacy"
        assert resolve_engine({"engine": ""}) == "legacy"

    def test_v2_accepted_case_and_whitespace_insensitive(self):
        assert resolve_engine({"engine": "v2"}) == "v2"
        assert resolve_engine({"engine": " V2 "}) == "v2"

    def test_invalid_engine_fails(self):
        with pytest.raises(PaperHostError, match="engine 切换键非法"):
            resolve_engine({"engine": "v3"})

    def test_engine_key_inside_extra_section_resolves(self, tmp_path):
        raw = _grouped_config()
        raw["extra"] = {"engine": "v2"}
        flat = load_paper_config(_write_config(tmp_path, raw))
        assert resolve_engine(flat) == "v2"


# ---------------------------------------------------------------------------
# 模型版本收口（F19）
# ---------------------------------------------------------------------------


class TestResolveModelVersion:
    def test_explicit_version_int_and_str(self, tmp_path):
        assert resolve_model_version(24022, models_dir=str(tmp_path)) == 24022
        assert resolve_model_version("24022", models_dir=str(tmp_path)) == 24022

    def test_config_object_reads_model_version(self, tmp_path):
        config = TradingConfig(model_version=24022)
        assert resolve_model_version(config, models_dir=str(tmp_path)) == 24022

    def test_blank_falls_back_to_registry_latest(self, tmp_path):
        models_dir = tmp_path / "models"
        v1 = _register_mock_model(models_dir)
        v2 = _register_mock_model(models_dir)
        assert resolve_model_version(None, models_dir=str(models_dir)) == v2
        assert v2 == v1 + 1
        # 空字符串同样走 latest
        assert resolve_model_version("", models_dir=str(models_dir)) == v2

    def test_empty_registry_fails(self, tmp_path):
        with pytest.raises(PaperHostError, match="无可用主模型版本"):
            resolve_model_version(None, models_dir=str(tmp_path / "empty"))

    def test_invalid_version_value_fails(self, tmp_path):
        with pytest.raises(PaperHostError, match="model_version 非法"):
            resolve_model_version("abc", models_dir=str(tmp_path))
        with pytest.raises(PaperHostError, match="model_version 非法"):
            resolve_model_version(0, models_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# to_trading_config（含 D7 执行价格指纹）
# ---------------------------------------------------------------------------


class TestToTradingConfig:
    def test_full_mapping(self):
        flat = load_paper_config  # noqa: F841（防误用；本测试直接构造 flat）
        flat = {
            "model_version": 24022,
            "top_n": 20,
            "rebalance_freq": 20,
            "stagger_tranches": 2,
            "max_per_industry": 5,
            "max_weight_per_stock": 0.15,
            "position_sizing": "kelly",
            "kelly_vol_window": 60,
            "kelly_max_leverage": 0.2,
            "buy_price": "close",
            "sell_price": "open",
            "initial_capital": 650000.0,
            "min_buy_value_ratio": 0.2,
            "horizon": 20,
            "universe": "mainboard",
            "exposure_policy": "arm=combined",  # 退役家族：不消费
            "engine": "v2",  # 切换键：不进 TradingConfig
        }
        config = to_trading_config(flat)
        assert config.model_version == 24022
        assert config.top_n == 20
        assert config.stagger_tranches == 2
        assert config.position_sizing == "kelly"
        assert config.buy_price == "close"
        assert config.sell_price == "open"
        assert config.universe == "mainboard"
        assert not hasattr(config, "exposure_policy")
        assert not hasattr(config, "engine")

    def test_weight_method_alias(self):
        config = to_trading_config({"weight_method": "score"})
        assert config.position_sizing == "score"

    def test_downside_penalty_passthrough(self):
        # λ 落于 [0,1) 由 TradingConfig 接收；λ>0 的 fail-fast 归 MLSignal 构造（A5 退役）
        config = to_trading_config({"downside_penalty": 0.25})
        assert config.downside_penalty == 0.25

    def test_buy_price_open_rejected(self):
        with pytest.raises(PaperHostError, match="buy_price"):
            to_trading_config({"buy_price": "open"})

    def test_buy_price_default_close_accepted(self):
        config = to_trading_config({})
        assert config.buy_price == "close"

    def test_sell_price_invalid_rejected(self):
        with pytest.raises(PaperHostError, match="sell_price"):
            to_trading_config({"sell_price": "mid"})
        assert to_trading_config({"sell_price": "close"}).sell_price == "close"

    def test_invalid_combo_fails_with_context(self):
        with pytest.raises(PaperHostError, match="TradingConfig"):
            to_trading_config({"top_n": 2, "stagger_tranches": 5})  # stagger > top_n

    def test_non_mapping_rejected(self):
        with pytest.raises(PaperHostError, match="Mapping"):
            to_trading_config(["not", "a", "mapping"])


# ---------------------------------------------------------------------------
# to_engine_kwargs
# ---------------------------------------------------------------------------


class TestToEngineKwargs:
    def _config(self, **overrides) -> TradingConfig:
        base = {
            "top_n": 20,
            "rebalance_freq": 20,
            "stagger_tranches": 2,
            "buy_price": "close",
            "sell_price": "open",
            "initial_capital": 650000.0,
            "position_sizing": "kelly",
            "kelly_vol_window": 60,
            "kelly_max_leverage": 0.2,
            "min_buy_value_ratio": 0.2,
            "max_per_industry": 5,
            "max_weight_per_stock": 0.15,
        }
        base.update(overrides)
        return to_trading_config(base)

    def test_hard_constraints_and_template_values(self):
        kwargs = to_engine_kwargs({}, self._config())
        # 硬项①：live 复刻旧链仅空仓语义
        assert kwargs["enable_trailing_early_rebalance"] is False
        # D7：sell_timing 映射
        assert kwargs["sell_timing"] == "open"
        # 与回测宿主同值（T4 门 3 前提）
        assert kwargs["completion_window_days"] == 5 == PAPER_COMPLETION_WINDOW_DAYS
        # 模板值
        assert kwargs["enable_pending_order"] is True
        assert kwargs["max_retry_count"] == 5
        assert kwargs["max_retry_days"] == 10
        assert kwargs["enable_position_completion"] is True
        # 配置透传
        assert kwargs["initial_capital"] == 650000.0
        assert kwargs["rebalance_freq"] == 20
        assert kwargs["stagger_tranches"] == 2
        assert kwargs["position_sizing"] == "kelly"
        assert kwargs["kelly_vol_window"] == 60
        assert kwargs["kelly_max_leverage"] == 0.2
        assert kwargs["min_buy_value_ratio"] == 0.2
        assert kwargs["max_per_industry"] == 5
        assert kwargs["max_weight_per_stock"] == 0.15
        # 止损未启用 ⇒ None
        assert kwargs["stop_loss_config"] is None
        # 非配置来源参数不在映射内（宿主装配注入）
        for key in (
            "universe",
            "signal",
            "cost_model",
            "stock_basic",
            "data_storage",
            "features_by_date",
        ):
            assert key not in kwargs

    def test_sell_timing_close_mapping(self):
        kwargs = to_engine_kwargs({}, self._config(sell_price="close"))
        assert kwargs["sell_timing"] == "close"

    def test_stop_loss_enabled_builds_config(self):
        kwargs = to_engine_kwargs({}, self._config(stop_loss_enabled=True))
        assert kwargs["stop_loss_config"] is not None
        assert kwargs["stop_loss_config"].enabled is True
        assert kwargs["stop_loss_config"].drawdown_pct == 30.0

    def test_none_rebalance_freq_rejected(self):
        config = self._config()
        config.rebalance_freq = None
        with pytest.raises(PaperHostError, match="rebalance_freq"):
            to_engine_kwargs({}, config)

    def test_mapping_table_is_identity_for_known_keys(self):
        # 消费白名单 = TradingConfig 字段同名集合（展平后同名）
        assert set(PAPER_CONFIG_TO_V2.values()) == set(PAPER_CONFIG_TO_V2)
        for key in PAPER_CONFIG_TO_V2:
            assert key in TradingConfig.__dataclass_fields__
        # 切换键与退役家族不在消费面
        assert "engine" not in PAPER_CONFIG_TO_V2
        assert "exposure_policy" not in PAPER_CONFIG_TO_V2


# ---------------------------------------------------------------------------
# 端到端：config.yaml（分组）→ load → TradingConfig → engine_kwargs
# ---------------------------------------------------------------------------


class TestEndToEndConfigChain:
    def test_production_shaped_config_roundtrip(self, tmp_path):
        store = _write_config(tmp_path, _grouped_config())
        flat = load_paper_config(store)
        assert resolve_engine(flat) == "legacy"  # 缺省
        config = to_trading_config(flat)
        assert config.model_version == 24022
        assert config.top_n == 20
        kwargs = to_engine_kwargs(flat, config)
        assert kwargs["enable_trailing_early_rebalance"] is False
        assert kwargs["sell_timing"] == "open"

    def test_engine_v2_in_config_yaml(self, tmp_path):
        raw = _grouped_config()
        raw["extra"] = {"engine": "v2"}
        store = _write_config(tmp_path, raw)
        assert resolve_engine(load_paper_config(store)) == "v2"

    def test_registry_latest_resolution_independent_of_model_file(self, tmp_path):
        # 版本解析只读注册表（latest 旁路语义），不加载模型文件
        models_dir = tmp_path / "models"
        _register_mock_model(models_dir, train_params={"max_depth": 3})
        version = resolve_model_version(TradingConfig(), models_dir=str(models_dir))
        registry = ModelRegistry(models_dir=str(models_dir))
        assert (
            version
            == json.loads((models_dir / "latest_model_version.txt").read_text(encoding="utf-8"))
            or version == registry.get_latest_version()
        )


# ---------------------------------------------------------------------------
# F14：load_paper_config 只读形态（读配置不得改变磁盘布局）
# ---------------------------------------------------------------------------


class TestLoadPaperConfigNoLayoutSideEffect:
    def test_existing_root_with_only_config_creates_no_dirs(self, tmp_path):
        # 根目录仅含 config.yaml（不经 PaperStateStore 默认形态写文件）
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(_grouped_config(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        flat = load_paper_config(str(tmp_path))
        assert flat["top_n"] == 20  # 返回形态正确（展平照常）
        assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"], "读配置不得建子目录"

    def test_nonexistent_root_raises_without_creating_dirs(self, tmp_path):
        missing = tmp_path / "ghost"
        with pytest.raises(PaperHostError, match="未找到配置文件"):
            load_paper_config(str(missing))
        assert not missing.exists(), "失败路径同样不得落任何目录"


# ---------------------------------------------------------------------------
# F19：capital_retention_ratio 配置漂移防御（告警不 fail-fast）
# ---------------------------------------------------------------------------


@pytest.fixture
def warning_messages():
    """loguru WARNING+ 捕获（loguru 不经标准 logging，caplog 捕不到）。"""
    messages: list = []
    sink_id = logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class TestCapitalRetentionDriftWarning:
    _BASE = {
        "top_n": 20,
        "rebalance_freq": 20,
        "stagger_tranches": 2,
        "buy_price": "close",
        "sell_price": "open",
    }

    def _config(self) -> TradingConfig:
        return to_trading_config(dict(self._BASE))

    def test_nonzero_ratio_warns_and_output_unaffected(self, warning_messages):
        config = self._config()
        kwargs = to_engine_kwargs({"capital_retention_ratio": 0.1}, config)
        assert any(
            "capital_retention_ratio" in message and "不生效" in message
            for message in warning_messages
        ), "非 0 留存比必须触发漂移告警"
        # 告警不 fail-fast：适配输出与无该键形态逐键一致
        assert kwargs == to_engine_kwargs({}, config)

    def test_non_numeric_ratio_warns(self, warning_messages):
        to_engine_kwargs({"capital_retention_ratio": "abc"}, self._config())
        assert any("capital_retention_ratio" in message for message in warning_messages)

    def test_zero_none_and_absent_ratio_silent(self, warning_messages):
        to_engine_kwargs({}, self._config())
        to_engine_kwargs({"capital_retention_ratio": 0}, self._config())
        to_engine_kwargs({"capital_retention_ratio": 0.0}, self._config())
        to_engine_kwargs({"capital_retention_ratio": None}, self._config())
        assert not any("capital_retention_ratio" in m for m in warning_messages)

    def test_end_to_end_via_config_yaml(self, tmp_path, warning_messages):
        raw = _grouped_config()
        raw["extra"] = {"capital_retention_ratio": 0.1}
        store = _write_config(tmp_path, raw)
        flat = load_paper_config(store)
        assert flat["capital_retention_ratio"] == 0.1  # 透传不消费（不进 TradingConfig）
        config = to_trading_config(flat)
        assert not hasattr(config, "capital_retention_ratio")
        kwargs = to_engine_kwargs(flat, config)
        assert kwargs["initial_capital"] == 650000.0  # 指令预算参数不受留存比影响
        assert any("capital_retention_ratio" in m for m in warning_messages)
