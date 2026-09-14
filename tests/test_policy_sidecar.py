"""terminal_loss 政策旁路测试（合成数据，不依赖真实配置与真实数据）。

覆盖：
1. 折索引与前视规则（按 val_end 选择、早于最早折的日期跳过）；
2. 台账打分链路（h=0 剔除、h 超网格报错、截面分位在完整同日截面内计算、
   标签经规范实现关联）；
3. 触发清单与阈值扫描的口径与**中文表头**；
4. 输入缺失（缺 cs_train 分区 / 缺特征列）明确报错，不静默降级。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lazybull.common.sidecar_schema import (
    LEDGER_COLUMNS_ZH,
    SCAN_COLUMNS_ZH,
    SNAPSHOT_COLUMNS_ZH,
    TRIGGER_COLUMNS_ZH,
)
from src.lazybull.risk.terminal_loss import policy_sidecar as ps

N_DAYS = 40
WARMUP = 21
CODES = ["000001.SZ", "000002.SZ", "000003.SZ"]


class _StubModel:
    """桩模型：分数与 cvar_95_20 单调，便于断言分位与触发。"""

    feature_names = [
        "remaining_intervals",
        "sigma_daily_20",
        "expected_vol_over_horizon",
        "cvar_95_20",
    ]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame["cvar_95_20"].to_numpy(dtype=float)
        scaled = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw) + 1e-12)
        return 0.05 + 0.5 * scaled


def _stub_loader(path: str) -> _StubModel:
    assert Path(path).exists(), "桩加载器仍要求模型文件存在（折索引契约）"
    return _StubModel()


@pytest.fixture
def env(tmp_path):
    """合成 clean/daily + cs_train + 一个折目录 + 中文表头持仓快照。"""
    days = pd.bdate_range("2024-01-02", periods=N_DAYS)
    cal = [day.strftime("%Y%m%d") for day in days]
    rng = np.random.default_rng(7)

    clean_dir = tmp_path / "clean"
    daily_dir = clean_dir / "daily"
    daily_dir.mkdir(parents=True)
    cs_dir = tmp_path / "features" / "cs_train"
    cs_dir.mkdir(parents=True)
    pd.DataFrame({"exchange": "SSE", "cal_date": cal, "is_open": 1}).to_parquet(
        clean_dir / "trade_cal.parquet"
    )

    closes = {code: 100.0 for code in CODES}
    for day in days:
        rows = []
        for code in CODES:
            ret = rng.normal(0, 0.02)
            prev_close = closes[code]
            open_adj = prev_close * (1 + rng.normal(0, 0.005))
            close_adj = open_adj * (1 + ret)
            closes[code] = close_adj
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": day.strftime("%Y-%m-%d"),
                    "open_adj": open_adj,
                    "close_adj": close_adj,
                    "is_limit_down": 0,
                }
            )
        pd.DataFrame(rows).to_parquet(daily_dir / f"{day:%Y-%m-%d}.parquet")

    for idx, date in enumerate(cal):
        frame = pd.DataFrame(
            {
                "ts_code": CODES,
                "cvar_95_20": rng.normal(-0.03, 0.01, len(CODES)),
                "mkt_vol_20": 0.02 + 0.001 * idx,
            }
        )
        frame.to_parquet(cs_dir / f"{date}.parquet")

    risk_root = tmp_path / "risk"
    fold_dir = risk_root / "2024H1_arm"
    fold_dir.mkdir(parents=True)
    (fold_dir / "terminal_loss_model.joblib").write_bytes(b"stub")
    with open(fold_dir / "terminal_loss_model.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_names": _StubModel.feature_names,
                "metadata": {
                    "stage_dates": {
                        "train": ["20230101", "20230630"],
                        "val": ["20230701", "20231231"],
                        "es": ["20240101", "20240630"],
                    },
                    "label_config": {"h_max": 5, "loss_sigma_multiple": 1.0},
                },
            },
            handle,
        )

    snapshot_rows = []
    for offset, code in enumerate(CODES):
        snapshot_rows.append(
            {
                "运行标识": "wf_test",
                "折序号": 0,
                "模型版本": 1,
                "日期": cal[WARMUP + offset],
                "股票代码": code,
                "持仓股数": 100 * (offset + 1),
                "持仓市值": 10000.0,
                "持仓权重": 0.1 * (offset + 1),
                "买入日": cal[WARMUP - 3 + offset],
                "信号日": cal[WARMUP - 4 + offset],
                "持有交易日数": 3,
                "到期执行日": cal[WARMUP + offset + 2],
                "剩余持有交易日": 2,
                "组合总值": 100000.0,
            }
        )
    snapshot = tmp_path / "walk_forward_持仓快照_wf_test_split00.csv"
    pd.DataFrame(snapshot_rows).to_csv(snapshot, index=False, encoding="utf-8-sig")

    return {
        "data_root": str(tmp_path),
        "risk_root": str(risk_root),
        "snapshot": snapshot,
        "cal": cal,
        "cs_dir": cs_dir,
    }


# ── 折索引与选择规则 ────────────────────────────────────────────────


def test_load_fold_index_sorted_and_validated(env):
    folds = ps.load_fold_index(env["risk_root"], "_arm")
    assert [f.fold for f in folds] == ["2024H1"]
    assert folds[0].val_end == "20231231"
    assert folds[0].feature_names == tuple(_StubModel.feature_names)


def test_select_fold_is_strictly_past_only(env):
    folds = ps.load_fold_index(env["risk_root"], "_arm")
    assert ps.select_fold_for_date("20230101", folds) is None  # 早于 val_end → 无可用折
    assert ps.select_fold_for_date("20240102", folds) is folds[0]
    assert ps.select_fold_for_date("20250101", folds) is folds[0]  # 晚于 ES 段仍用最近折


def test_load_fold_index_missing_meta_raises(tmp_path):
    fold_dir = tmp_path / "risk" / "x_arm"
    fold_dir.mkdir(parents=True)
    with open(fold_dir / "terminal_loss_model.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": {}}, handle)
    with pytest.raises(ValueError, match="缺少 stage_dates"):
        ps.load_fold_index(str(tmp_path / "risk"), "_arm")
    with pytest.raises(ValueError, match="未找到匹配"):
        ps.load_fold_index(str(tmp_path / "risk"), "_missing")


# ── 打分链路 ────────────────────────────────────────────────────────


def test_score_holdings_builds_ledger_with_cross_section_percentile(env):
    ledger = ps.score_holdings(
        snapshot_files=[env["snapshot"]],
        risk_root=env["risk_root"],
        arm_suffix="_arm",
        data_root=env["data_root"],
        model_loader=_stub_loader,
    )
    assert len(ledger) == len(CODES)
    assert ledger["fold"].unique().tolist() == ["2024H1"]
    assert ledger["p_loss"].between(0.0, 1.0).all()
    # 截面分位分母是当日完整截面（3 只股票 × 该 h），不是持仓子集
    assert ledger["day_pool_size"].eq(len(CODES)).all()
    assert ledger["p_loss_daypct"].between(0.0, 1.0).all()
    # 标签经规范实现关联（h=2 且端点齐全 → valid）
    assert ledger["label_status"].eq("valid").all()
    assert ledger["loss_label"].isin([0.0, 1.0]).all()
    assert ledger["avoidable_loss"].notna().all()
    assert ledger["last_holding_date"].notna().all()


def test_score_holdings_drops_zero_horizon_and_requires_grid(env, tmp_path):
    frame = pd.read_csv(env["snapshot"], dtype={"股票代码": str}, encoding="utf-8-sig")
    frame.loc[0, "剩余持有交易日"] = 0
    zero_path = tmp_path / "zero.csv"
    frame.to_csv(zero_path, index=False, encoding="utf-8-sig")
    ledger = ps.score_holdings(
        snapshot_files=[zero_path],
        risk_root=env["risk_root"],
        arm_suffix="_arm",
        data_root=env["data_root"],
        model_loader=_stub_loader,
    )
    assert len(ledger) == len(CODES) - 1  # h=0 的行被剔除

    frame.loc[1, "剩余持有交易日"] = ps.HORIZON_MAX + 1
    bad_path = tmp_path / "bad.csv"
    frame.to_csv(bad_path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="超出模型网格"):
        ps.score_holdings(
            snapshot_files=[bad_path],
            risk_root=env["risk_root"],
            arm_suffix="_arm",
            data_root=env["data_root"],
            model_loader=_stub_loader,
        )


def test_score_holdings_errors_are_explicit(env, tmp_path):
    # 缺 cs_train 分区：显式报错，不静默跳过
    frame = pd.read_csv(env["snapshot"], dtype={"股票代码": str}, encoding="utf-8-sig")
    frame["日期"] = "20240102"
    early = tmp_path / "early.csv"
    frame.to_csv(early, index=False, encoding="utf-8-sig")
    missing_dir = tmp_path / "empty_cs"
    missing_dir.mkdir()
    with pytest.raises(ValueError, match="缺少 cs_train 分区"):
        ps.score_holdings(
            snapshot_files=[early],
            risk_root=env["risk_root"],
            arm_suffix="_arm",
            data_root=env["data_root"],
            model_loader=_stub_loader,
            feature_root=str(missing_dir),
        )


# ── 触发清单与阈值扫描（纯单元）────────────────────────────────────


def _ledger_rows() -> pd.DataFrame:
    """构造完整台账行（含触发清单/扫描所需全部内部列）。"""
    base = {
        "date": ["20240110", "20240111", "20240112", "20240115"],
        "ts_code": CODES + ["000004.SZ"],
        "weight": [0.25, 0.25, 0.25, 0.25],
        "sigma_daily_20": [0.02, 0.03, 0.02, 0.02],
        "expected_vol_over_horizon": [0.028, 0.042, 0.028, 0.028],
        "mkt_vol_20": [0.02, 0.02, 0.02, 0.02],
        # 命中（高风险 + 高概率 + 事后异常亏损）
        "p_loss": [0.30, 0.25, 0.04, 0.02],
        "p_loss_daypct": [0.99, 0.96, 0.30, 0.10],
        "loss_label": [1.0, 0.0, 1.0, 0.0],
        "terminal_return": [-0.20, 0.08, -0.15, 0.01],
    }
    frame = pd.DataFrame(base)
    frame["remaining_intervals"] = 2
    frame["avoidable_loss"] = -frame["terminal_return"]
    return frame


def test_build_trigger_list_classifies_events():
    ledger = _ledger_rows()
    triggers = ps.build_trigger_list(ledger, p_hi=0.95, p_abs=0.20)
    assert triggers["event_type"].tolist() == ["正确拦截", "误杀", "漏报"]
    assert len(triggers) == 3
    only_hits = ps.build_trigger_list(ledger, p_hi=0.95, p_abs=0.20, include_missed=False)
    assert set(only_hits["event_type"]) == {"正确拦截", "误杀"}
    # 阈值放宽到不触发任何行 → 全部为漏报，且命中集为空
    relaxed = ps.build_trigger_list(ledger, p_hi=1.0, p_abs=1.0)
    assert set(relaxed["event_type"]) == {"漏报"}


def test_scan_thresholds_reports_intercept_and_cost():
    ledger = _ledger_rows()
    scan = ps.scan_thresholds(ledger, p_hi_list=[0.95], p_abs_list=[0.20])
    assert list(scan.columns) == SCAN_COLUMNS_ZH
    row = scan.iloc[0]
    assert row["触发笔数"] == 2
    assert row["事件拦截率"] == pytest.approx(0.5)  # 2 个事件里拦到 1 个
    assert row["误杀率"] == pytest.approx(0.5)
    assert row["漏报率"] == pytest.approx(0.5)
    # 触发样本事后收益 = (-0.20 + 0.08) / 2
    assert row["触发样本平均事后收益"] == pytest.approx(-0.06)
    assert row["事后可避免损失合计"] == pytest.approx(0.12)


# ── 中文表头落盘 ────────────────────────────────────────────────────


def test_writers_use_chinese_headers(env, tmp_path):
    ledger = ps.score_holdings(
        snapshot_files=[env["snapshot"]],
        risk_root=env["risk_root"],
        arm_suffix="_arm",
        data_root=env["data_root"],
        model_loader=_stub_loader,
    )
    ledger_path = tmp_path / "风险台账.csv"
    ps.write_ledger(ledger, ledger_path)
    written = pd.read_csv(ledger_path, encoding="utf-8-sig")
    assert list(written.columns) == LEDGER_COLUMNS_ZH

    triggers = ps.build_trigger_list(ledger, p_hi=0.9, p_abs=0.1)
    trigger_path = tmp_path / "触发清单.csv"
    ps.write_trigger_list(triggers, trigger_path)
    if not triggers.empty:
        assert list(pd.read_csv(trigger_path, encoding="utf-8-sig").columns) == TRIGGER_COLUMNS_ZH

    scan = ps.scan_thresholds(ledger, p_hi_list=[0.9], p_abs_list=[0.1])
    scan_path = tmp_path / "阈值扫描.csv"
    ps.write_threshold_scan(scan, scan_path)
    assert list(pd.read_csv(scan_path, encoding="utf-8-sig").columns) == SCAN_COLUMNS_ZH


def test_snapshot_schema_is_frozen():
    """快照中文表头必须与 sidecar_schema 冻结清单一致（生产者/消费者同源）。"""
    assert len(SNAPSHOT_COLUMNS_ZH) == 11
    assert SNAPSHOT_COLUMNS_ZH["remaining_intervals"] == "剩余持有交易日"
