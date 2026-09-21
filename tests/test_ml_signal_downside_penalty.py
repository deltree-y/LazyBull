"""A5 路由型下行风险惩罚（downside penalty）测试。

覆盖：
- 惩罚公式（单截面 / 按日分组 / cvar 负值列方向 / NaN 按中位填充）；
- 非法参数与缺列必须明确报错（禁止静默跳过）；
- MLSignal 集成：λ=0 逐位 no-op（且不要求风险列存在）、λ>0 按构造改序、缺列报错；
- TradingConfig / argparse / signal_factory 的参数透传与校验。
"""

import argparse

import pandas as pd
import pytest

from src.lazybull.common.signal_factory import create_signal
from src.lazybull.common.trading_config import TradingConfig, add_trading_args
from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal
from src.lazybull.signals.downside_penalty import (
    RISK_NAN_FILL,
    apply_downside_penalty,
)


class MockMLModel:
    """模拟 ML 模型：预测值 = 第一列 × 0.1（与既有 signals 测试同构）。"""

    def predict(self, X):
        return X.iloc[:, 0].values * 0.1


def _register_mock_model(models_dir, feature_columns):
    registry = ModelRegistry(models_dir=models_dir)
    return registry.register_model(
        model=MockMLModel(),
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=feature_columns,
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100},
    )


# ─────────────────────── 纯函数级 ───────────────────────


def test_apply_penalty_formula_single_cross_section():
    """单截面：score' = 分位(score) − λ×风险分位（精确值）。"""
    df = _make_df(scores=[4.0, 3.0, 2.0, 1.0], downside=[4.0, 1.0, 2.0, 3.0])
    stats = apply_downside_penalty(
        df,
        score_column="score",
        risk_column="downside_vol_20",
        penalty=0.5,
    )
    # score 分位: A=1.0, B=0.75, C=0.5, D=0.25
    # 风险分位: A=1.0, B=0.25, C=0.5, D=0.75
    # adj = 分位 − 0.5×风险分位
    assert stats == {"rows": 4, "days": 1, "risk_nan_rows": 0}
    adj = dict(zip(df["ts_code"], df["score"]))
    assert adj["A"] == pytest.approx(1.0 - 0.5 * 1.0)
    assert adj["B"] == pytest.approx(0.75 - 0.5 * 0.25)
    assert adj["C"] == pytest.approx(0.5 - 0.5 * 0.5)
    assert adj["D"] == pytest.approx(0.25 - 0.5 * 0.75)
    # 构造下 B > A（高风险顶格分被惩罚压过）
    assert adj["B"] > adj["A"]


def test_apply_penalty_cvar_negative_direction():
    """cvar_95_20 为负值列：越负越危险，必须获得更高惩罚。"""
    df = _make_df(
        scores=[4.0, 3.0, 2.0, 1.0],
        cvar_95_20=[-0.20, -0.01, -0.05, -0.10],
    )
    apply_downside_penalty(
        df,
        score_column="score",
        risk_column="cvar_95_20",
        penalty=0.5,
    )
    adj = dict(zip(df["ts_code"], df["score"]))
    # A 的 cvar 最负（最危险）⇒ 风险分位 1.0 ⇒ 被罚到 1.0−0.5=0.5；
    # B 的 cvar 最接近 0（最安全）⇒ 风险分位 0.25 ⇒ 0.75−0.125=0.625
    assert adj["A"] == pytest.approx(0.5)
    assert adj["B"] == pytest.approx(0.625)
    assert adj["B"] > adj["A"]


def test_apply_penalty_nan_risk_filled_with_median():
    """风险缺失按截面中位 (0.5) 处理并计数。"""
    df = _make_df(scores=[3.0, 2.0, 1.0], downside=[1.0, None, 2.0])
    stats = apply_downside_penalty(
        df,
        score_column="score",
        risk_column="downside_vol_20",
        penalty=0.5,
    )
    assert stats["risk_nan_rows"] == 1
    adj = dict(zip(df["ts_code"], df["score"]))
    # A: 1.0 − 0.5×0.5；B: 2/3 − 0.5×RISK_NAN_FILL；C: 1/3 − 0.5×1.0
    assert adj["A"] == pytest.approx(1.0 - 0.5 * 0.5)
    assert adj["B"] == pytest.approx(2.0 / 3.0 - 0.5 * RISK_NAN_FILL)
    assert adj["C"] == pytest.approx(1.0 / 3.0 - 0.5 * 1.0)


def test_apply_penalty_by_day_grouping():
    """按日分组：分位只在同日截面内计算（不得跨日泄漏）。"""
    df = _make_df(scores=[1.0, 2.0], downside=[1.0, 2.0], dates=["20240102", "20240102"])
    df2 = _make_df(scores=[10.0, 20.0], downside=[1.0, 2.0], dates=["20240103", "20240103"])
    df = pd.concat([df, df2], ignore_index=True)
    stats = apply_downside_penalty(
        df,
        score_column="score",
        risk_column="downside_vol_20",
        penalty=0.25,
        date_column="trade_date",
    )
    assert stats["days"] == 2
    # 两日内部结构相同 ⇒ 每日调整值结构一致（0.5 − 0.25×0.5 与 1.0 − 0.25×1.0）
    day1 = df[df["trade_date"] == "20240102"]["score"].tolist()
    day2 = df[df["trade_date"] == "20240103"]["score"].tolist()
    assert day1 == pytest.approx([0.5 - 0.25 * 0.5, 1.0 - 0.25 * 1.0])
    assert day2 == pytest.approx(day1)


def test_apply_penalty_errors():
    """非法参数 / 缺列必须明确报错。"""
    df = _make_df(scores=[1.0, 2.0], downside=[1.0, 2.0])
    with pytest.raises(ValueError, match="λ 必须为正数"):
        apply_downside_penalty(df, score_column="score", risk_column="downside_vol_20", penalty=0)
    with pytest.raises(ValueError, match="未登记的下行风险列"):
        apply_downside_penalty(df, score_column="score", risk_column="not_registered", penalty=0.25)
    with pytest.raises(ValueError, match="缺少列"):
        apply_downside_penalty(df, score_column="score", risk_column="cvar_95_20", penalty=0.25)
    with pytest.raises(ValueError, match="缺少列"):
        apply_downside_penalty(
            df, score_column="missing_score", risk_column="downside_vol_20", penalty=0.25
        )
    with pytest.raises(ValueError, match="缺少日期列"):
        apply_downside_penalty(
            df,
            score_column="score",
            risk_column="downside_vol_20",
            penalty=0.25,
            date_column="trade_date",
        )


# ─────────────────────── MLSignal 集成 ───────────────────────


def test_ml_signal_penalty_off_is_noop_and_needs_no_risk_column(tmp_path):
    """λ=0：不要求风险列存在，排序与分数与原始预测逐位一致。"""
    version = _register_mock_model(str(tmp_path), ["f1"])
    signal = MLSignal(top_n=3, model_version=version, models_dir=str(tmp_path), verbose=False)
    df = _make_ml_df(f1=[10.0, 30.0, 20.0])  # 预测 = f1×0.1 → s0=1.0, s1=3.0, s2=2.0
    ranked = signal.generate_ranked(
        pd.Timestamp("2024-01-02"), df["ts_code"].tolist(), {"features": df}
    )
    assert [code for code, _ in ranked] == ["s1", "s2", "s0"]
    assert [score for _, score in ranked] == pytest.approx([3.0, 2.0, 1.0])


def test_ml_signal_penalty_reorders_by_construction(tmp_path):
    """λ>0：构造下最高分但最高风险的股票被挤出前列。"""
    version = _register_mock_model(str(tmp_path), ["f1"])
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=str(tmp_path),
        verbose=False,
        downside_penalty=0.5,
    )
    # s0 分数最高但风险顶格；其余按 i+1 递增 ⇒ s1 最低风险
    df = _make_ml_df(f1=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    df["downside_vol_20"] = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ranked = signal.generate_ranked(
        pd.Timestamp("2024-01-02"), df["ts_code"].tolist(), {"features": df}
    )
    assert [code for code, _ in ranked][:3] == ["s1", "s2", "s3"]
    assert ranked[0][1] == pytest.approx(0.85)  # 0.9 − 0.5×0.1
    assert dict(ranked)["s0"] == pytest.approx(0.5)  # 1.0 − 0.5×1.0


def test_ml_signal_penalty_missing_column_raises(tmp_path):
    """λ>0 且风险列缺失：明确报错，禁止静默跳过。"""
    version = _register_mock_model(str(tmp_path), ["f1"])
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=str(tmp_path),
        verbose=False,
        downside_penalty=0.25,
    )
    df = _make_ml_df(f1=[10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="缺少列"):
        signal.generate_ranked(
            pd.Timestamp("2024-01-02"), df["ts_code"].tolist(), {"features": df}
        )


def test_ml_signal_constructor_validation():
    """构造校验：λ 范围与风险列白名单。"""
    with pytest.raises(ValueError, match="downside_penalty"):
        MLSignal(downside_penalty=-0.1)
    with pytest.raises(ValueError, match="downside_penalty"):
        MLSignal(downside_penalty=1.0)
    with pytest.raises(ValueError, match="未登记的下行风险惩罚列"):
        MLSignal(downside_penalty=0.25, downside_penalty_column="zscore_volatility_20")
    # λ=0 时风险列取值不受限（不生效）
    MLSignal(downside_penalty=0.0, downside_penalty_column="zscore_volatility_20")


# ─────────────────────── 配置 / 工厂透传 ───────────────────────


def test_trading_config_fields_roundtrip_and_validation():
    cfg = TradingConfig(downside_penalty=0.25, downside_penalty_column="cvar_95_20")
    assert cfg.to_dict()["downside_penalty"] == 0.25
    restored = TradingConfig.from_dict(cfg.to_dict())
    assert restored.downside_penalty == 0.25
    assert restored.downside_penalty_column == "cvar_95_20"
    with pytest.raises(ValueError, match="downside_penalty"):
        TradingConfig(downside_penalty=1.5)
    with pytest.raises(ValueError, match="downside_penalty"):
        TradingConfig(downside_penalty=-0.1)


def test_trading_args_parser_choices():
    parser = argparse.ArgumentParser()
    add_trading_args(parser)
    args = parser.parse_args(
        ["--downside-penalty", "0.25", "--downside-penalty-column", "cvar_95_20"]
    )
    cfg = TradingConfig.from_args(args)
    assert cfg.downside_penalty == 0.25
    assert cfg.downside_penalty_column == "cvar_95_20"
    # 冻结网格：第三档必须被 argparse 拒绝
    with pytest.raises(SystemExit):
        parser.parse_args(["--downside-penalty", "0.3"])


def test_signal_factory_passes_penalty(tmp_path):
    cfg = TradingConfig(
        model_version=1, downside_penalty=0.25, downside_penalty_column="downside_vol_20"
    )
    signal = create_signal(cfg, models_dir=str(tmp_path), verbose=False)
    assert isinstance(signal, MLSignal)
    assert signal.downside_penalty == 0.25
    assert signal.downside_penalty_column == "downside_vol_20"

    cfg_ensemble = TradingConfig(
        model_version=1,
        model_version_b=2,
        downside_penalty=0.5,
        downside_penalty_column="cvar_95_20",
    )
    ensemble = create_signal(cfg_ensemble, models_dir=str(tmp_path), verbose=False)
    for sub_signal in (ensemble.signal_a, ensemble.signal_b):
        assert sub_signal.downside_penalty == 0.5
        assert sub_signal.downside_penalty_column == "cvar_95_20"


# ─────────────────────── 构造辅助 ───────────────────────


def _make_df(scores, downside=None, cvar_95_20=None, dates=None):
    payload = {
        "ts_code": ["A", "B", "C", "D"][: len(scores)],
        "score": list(scores),
    }
    if downside is not None:
        payload["downside_vol_20"] = list(downside)
    if cvar_95_20 is not None:
        payload["cvar_95_20"] = list(cvar_95_20)
    if dates is not None:
        payload["trade_date"] = list(dates)
    return pd.DataFrame(payload)


def _make_ml_df(f1):
    return pd.DataFrame(
        {
            "ts_code": [f"s{index}" for index in range(len(f1))],
            "f1": list(f1),
        }
    )
