"""期末异常亏损训练脚本端到端冒烟测试（合成数据目录，不依赖真实数据）"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_terminal_risk_model import main as train_main
from src.lazybull.risk.terminal_loss import BASE_FEATURES, TerminalLossModel

N_DAYS = 70
WARMUP = 22  # sigma 窗口 20 + pct_change 首行 + 余量


@pytest.fixture
def synthetic_env(tmp_path):
    """构造 clean/daily、trade_cal、cs_train 合成数据目录。"""
    dates = pd.bdate_range("2024-01-02", periods=N_DAYS)
    cal_str = [d.strftime("%Y%m%d") for d in dates]
    rng = np.random.default_rng(42)
    codes = [f"00000{i}.SZ" for i in range(1, 7)]

    clean_dir = tmp_path / "clean"
    daily_dir = clean_dir / "daily"
    daily_dir.mkdir(parents=True)
    cs_dir = tmp_path / "features" / "cs_train"
    cs_dir.mkdir(parents=True)

    pd.DataFrame({"exchange": "SSE", "cal_date": cal_str, "is_open": 1}).to_parquet(
        clean_dir / "trade_cal.parquet"
    )

    for i, d in enumerate(cal_str):
        day = pd.DataFrame(
            {
                "ts_code": codes,
                "open_adj": 100 + rng.normal(0, 1.5, len(codes)).cumsum() * 0 + 100 * 0,
                "close_adj": [100.0] * len(codes),
                "is_limit_down": 0,
            }
        )
        # 生成有波动的价格路径（每股独立随机游走）
        if i == 0:
            day["open_adj"] = 100.0
            day["close_adj"] = 100.0
        else:
            prev = 100.0
            opens, closes = [], []
            for c in codes:
                r = rng.normal(0, 0.02)
                o = prev * (1 + rng.normal(0, 0.005))
                cl = o * (1 + r)
                opens.append(o)
                closes.append(cl)
            day["open_adj"] = opens
            day["close_adj"] = closes
        day.to_parquet(daily_dir / f"{d[:4]}-{d[4:6]}-{d[6:8]}.parquet")

        # cs_train 仅 train/es 区间生成（特征列全量随机）
        if i >= WARMUP:
            feat = {"ts_code": codes}
            for col in BASE_FEATURES:
                feat[col] = rng.normal(0, 1, len(codes))
            pd.DataFrame(feat).to_parquet(cs_dir / f"{d}.parquet")

    out_dir = tmp_path / "models" / "terminal_loss"
    return {
        "data_root": str(tmp_path),
        "out_dir": str(out_dir),
        "cal": cal_str,
        "train": (cal_str[WARMUP], cal_str[WARMUP + 23]),
        "es": (cal_str[WARMUP + 24], cal_str[WARMUP + 38]),
        "end": cal_str[-1],
    }


def _run_train(env, monkeypatch, extra_argv):
    """以给定 argv 运行训练脚本主入口，返回 0 表示成功。"""
    argv = [
        "train_terminal_risk_model.py",
        "--data-root",
        env["data_root"],
        "--output-dir",
        env["out_dir"],
        "--start-date",
        env["train"][0],
        "--end-date",
        env["end"],
        "--train-start",
        env["train"][0],
        "--train-end",
        env["train"][1],
        "--es-start",
        env["es"][0],
        "--es-end",
        env["es"][1],
        "--n-estimators",
        "40",
        "--device",
        "cpu",
    ] + extra_argv
    monkeypatch.setattr(sys, "argv", argv)
    return train_main()


def test_script_flat_mode_end_to_end(synthetic_env, monkeypatch):
    """--fixed-name 模式回归：固定名四件套覆盖落盘（WF 折目录行为）。"""
    env = synthetic_env
    assert _run_train(env, monkeypatch, ["--fixed-name"]) == 0

    out = Path(env["out_dir"])
    assert (out / "terminal_loss_model.joblib").exists()
    assert (out / "terminal_loss_model.json").exists()
    assert (out / "terminal_loss_report.json").exists()
    assert (out / "calibration_by_h_sigma.csv").exists()

    with open(out / "terminal_loss_report.json", encoding="utf-8") as f:
        report = json.load(f)
    assert report["es"]["n"] > 0
    assert 0.0 <= report["es"]["event_rate"] <= 1.0
    assert report["es"]["logloss"] > 0
    # 模型元数据落盘了已知限制登记（pct 母截面口径）
    with open(out / "terminal_loss_model.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["task_id"] == "terminal_vol_scaled_loss"
    assert len(meta["feature_names"]) == 33
    assert meta["metadata"]["known_limitations"]


def test_script_versioned_mode_end_to_end(synthetic_env, monkeypatch):
    """默认版本化模式：同目录连续两次训练产生 v1/v2 两套产物且互不覆盖。"""
    env = synthetic_env
    assert _run_train(env, monkeypatch, []) == 0
    assert _run_train(env, monkeypatch, []) == 0

    out = Path(env["out_dir"])
    # 不存在 flat 固定名产物
    assert not (out / "terminal_loss_model.joblib").exists()

    for v in ("v1", "v2"):
        assert (out / f"{v}_model.joblib").exists()
        assert (out / f"{v}_features.json").exists()
        assert (out / f"{v}_metadata.json").exists()
        assert (out / f"{v}_report.json").exists()
        assert (out / f"{v}_calibration_by_h_sigma.csv").exists()
        assert (out / f"{v}_label_coverage.csv").exists()

    assert (out / "latest_model_version.txt").read_text(encoding="utf-8").strip() == "2"
    with open(out / "model_registry.json", encoding="utf-8") as f:
        registry = json.load(f)
    assert registry["next_version"] == 3
    assert [m["version"] for m in registry["models"]] == [1, 2]
    assert all(m["model_type"] == "xgboost_terminal_loss" for m in registry["models"])
    assert all(m["label_column"] == "loss_label" for m in registry["models"])

    with open(out / "v1_features.json", encoding="utf-8") as f:
        features = json.load(f)
    assert len(features) == 33

    # 元数据包含训练/标签配置快照与 ES 概率质量指标
    with open(out / "v1_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["train_params"]["train_config"]["max_depth"] == 3
    assert meta["train_params"]["label_config"]["task_id"] == "terminal_vol_scaled_loss"
    assert meta["performance_metrics"]["es_logloss"] >= 0
    # 合成数据事件率可能为 0，此时 lift 必须为 None（与 summarize 口径一致）
    if meta["performance_metrics"]["es_event_rate"] > 0:
        assert meta["performance_metrics"]["lift"] is not None
    else:
        assert meta["performance_metrics"]["lift"] is None

    # 注册表可加载模型实例并直接预测（joblib 分支 = TerminalLossModel 实例）
    from src.lazybull.ml.model_registry import ModelRegistry
    from src.lazybull.risk.terminal_loss import TERMINAL_LOSS_FEATURES

    model, loaded_meta = ModelRegistry(models_dir=str(out)).load_model(version=1)
    assert isinstance(model, TerminalLossModel)
    assert loaded_meta["version"] == 1
    infer_df = pd.DataFrame(
        {c: np.random.default_rng(0).normal(0, 1, 3) for c in TERMINAL_LOSS_FEATURES}
    )
    proba = model.predict_proba(infer_df)
    assert proba.shape == (3,)
    assert ((proba >= 0) & (proba <= 1)).all()
