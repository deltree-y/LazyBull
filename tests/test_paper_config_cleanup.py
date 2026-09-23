# -*- coding: utf-8 -*-
"""纸面配置生成/清理（``paper/storage/config.py``）单元测试。

只用临时目录与合成 YAML（不依赖真实配置），锁定 v0.127.11 两个修复：

1. **生成兜底**：从零生成（或历史文件刷新）时 ``exposure`` 区块必须出现
   （非 TradingConfig 字段缺键时补默认，null / false 即关闭语义，运行时零副作用）；
2. **已下线键剔除**：``signal_gate``（整块嵌套形态）与 ``time_stop_loss_*``
   （v0.90.2 已删除的死接口）与既有一批家族键一样，在任何保存/生成路径被剔除，
   且不会以任何形式（含 ``extra:`` 兼容区块）回到文件里。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lazybull.paper.exposure_policy import PaperExposureSettings  # noqa: E402
from src.lazybull.paper.storage import PaperStorage  # noqa: E402
from src.lazybull.paper.storage.config import is_retired_config_field  # noqa: E402

RETIRED_SAMPLES = [
    "holding_management",
    "enable_profit_based_holding",
    "weakness_exit",
    "equity_curve",
    "market_regime",
    "industry",
    "signal_gate",  # 整块（旧嵌套形态）
    "signal_gate_mode",
    "signal_gate_dynamic_topn",
    "signal_confidence_gate_top_k",  # 旧扁平形态
    "time_stop_loss_enabled",
    "time_stop_loss_days",
    "time_stop_loss_profit_ratio",
    "stop_loss_trailing_enabled",
    "profit_extension_mode",
    "early_exit_mode",
    "holding_bonus_sigma",
]

LIVE_SAMPLES = [
    "top_n",
    "stop_loss_enabled",
    "stop_loss_consecutive_limit_down",
    "downside_penalty",
    "downside_penalty_column",
    "exposure_policy",
    "policy_model_root",
    "policy_arm_suffix",
    "exposure_replenish",
    "position_sizing",
    "buy_price",
]


def test_retired_field_detection():
    for key in RETIRED_SAMPLES:
        assert is_retired_config_field(key), f"{key} 应判为已下线"
    for key in LIVE_SAMPLES:
        assert not is_retired_config_field(key), f"{key} 不该被判为已下线"


def test_fresh_generation_includes_exposure_section(tmp_path):
    """从零生成（仅给出 model_version）时：exposure 区块可见、extra 不出现。"""
    storage = PaperStorage(root_path=str(tmp_path))
    storage.save_config({"model_version": 24022})
    raw = (tmp_path / "config.yaml").read_text(encoding="utf-8")

    assert "exposure:" in raw
    assert "exposure_policy: null" in raw
    assert "policy_model_root: null" in raw
    assert "policy_fold: null" in raw
    assert "policy_warmup_file: null" in raw
    assert "exposure_replenish: false" in raw
    assert "signal_penalty:" in raw
    assert "extra:" not in raw

    # 运行时语义：默认关闭（零副作用）
    loaded = storage.load_config()
    assert PaperExposureSettings.from_config(loaded) is None


def test_legacy_retired_keys_purged_on_save(tmp_path):
    """历史样式文件（含 signal_gate 嵌套块 / time_stop_loss / 旧家族键）刷新后彻底净化。"""
    storage = PaperStorage(root_path=str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  model_version: 24022\n"
        "exposure:\n"
        "  exposure_policy: arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5\n"
        "  policy_model_root: data/walk_forward/terminal_risk_wf_oos14\n"
        "  policy_arm_suffix: _v6m_fscore\n"
        "  exposure_replenish: true\n"
        "signal_gate:\n"
        "  signal_gate_mode: disabled\n"
        "  signal_confidence_gate_top_k: 20\n"
        "  signal_gate_dynamic_topn: false\n"
        "extra:\n"
        "  time_stop_loss_enabled: false\n"
        "  time_stop_loss_days: 15\n"
        "  time_stop_loss_profit_ratio: -0.02\n"
        "  market_regime:\n"
        "    market_regime_enabled: false\n"
        "  signal_confidence_gate_enabled: false\n",
        encoding="utf-8",
    )

    loaded = storage.load_config()
    for key in (
        "signal_gate",
        "signal_gate_mode",
        "signal_gate_dynamic_topn",
        "signal_confidence_gate_top_k",
        "signal_confidence_gate_enabled",
        "time_stop_loss_enabled",
        "time_stop_loss_days",
        "time_stop_loss_profit_ratio",
        "market_regime",
    ):
        assert key not in loaded, f"{key} 应在读取视图被剔除"

    storage.save_config(loaded)
    raw = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    for token in (
        "signal_gate",
        "signal_confidence_gate",
        "time_stop_loss",
        "market_regime",
        "extra:",
    ):
        assert token not in raw, f"{token} 不应残留在刷新后的文件里"
    # 既有 exposure 配置必须保留（不被默认兜底覆盖）
    assert "arm=combined" in raw
    settings = PaperExposureSettings.from_config(storage.load_config())
    assert settings is not None
    assert settings.replenish is True
    assert settings.arm_suffix == "_v6m_fscore"


def test_explicit_exposure_config_round_trip(tmp_path):
    """显式给出的 exposure 全字段在保存/读取间逐值保留。"""
    storage = PaperStorage(root_path=str(tmp_path))
    storage.save_config(
        {
            "model_version": 24022,
            "exposure_policy": "arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5",
            "policy_model_root": "data/walk_forward/terminal_risk_wf_oos14",
            "policy_arm_suffix": "_v6m_fscore",
            "policy_coverage_start": "20190527",
            "exposure_replenish": True,
            "exposure_trim_tolerance": 0.06,
        }
    )
    loaded = storage.load_config()
    assert loaded["exposure_policy"].startswith("arm=combined")
    assert loaded["policy_model_root"].endswith("terminal_risk_wf_oos14")
    assert loaded["policy_arm_suffix"] == "_v6m_fscore"
    assert loaded["policy_coverage_start"] == "20190527"
    assert loaded["exposure_replenish"] is True
    assert loaded["exposure_trim_tolerance"] == pytest.approx(0.06)
    settings = PaperExposureSettings.from_config(loaded)
    assert settings is not None and settings.trim_tolerance == pytest.approx(0.06)


def test_save_is_idempotent_after_normalize(tmp_path):
    """保存 -> 读取 -> 再保存：文本稳定（默认兜底不会引入额外差异）。"""
    storage = PaperStorage(root_path=str(tmp_path))
    storage.save_config({"model_version": 24022})
    first = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    storage.save_config(storage.load_config())
    second = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert first == second
