"""期末异常亏损训练脚本端到端冒烟测试（合成数据目录，不依赖真实数据）"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_terminal_risk_model import main as train_main
from src.lazybull.risk.terminal_loss import BASE_FEATURES

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

    pd.DataFrame(
        {"exchange": "SSE", "cal_date": cal_str, "is_open": 1}
    ).to_parquet(clean_dir / "trade_cal.parquet")

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


def test_script_end_to_end(synthetic_env, monkeypatch):
    """主链路：面板加载→sigma→标签→矩阵→分割→训练→报告与模型落盘。"""
    env = synthetic_env
    argv = [
        "train_terminal_risk_model.py",
        "--data-root", env["data_root"],
        "--output-dir", env["out_dir"],
        "--start-date", env["train"][0],
        "--end-date", env["end"],
        "--train-start", env["train"][0],
        "--train-end", env["train"][1],
        "--es-start", env["es"][0],
        "--es-end", env["es"][1],
        "--n-estimators", "40",
        "--device", "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train_main() == 0

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
