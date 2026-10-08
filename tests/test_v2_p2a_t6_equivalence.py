"""P2a-T6 等价测试：v2/core/signal（ml_signal / ensemble_signal / factory）

验证目标（T0 规划 §4 T6 任务卡）：
- 同模型同特征 ⇒ 新旧两侧排序/信号输出**逐位一致**（A 组，合成 stub 模型 +
  合成注册表，不依赖真实模型文件与真实配置）；
- A5 退役摘除（D6）：λ>0 显式 fail-fast（B 组），参数签名保留（B3）；
- ensemble / factory 行为等价（C / D 组）；
- 改指与退役符号扫描（E 组）：v2/core/signal 不引用旧 ``signals.*`` 模块、
  不残留退役 downside_penalty 符号（归属判定：保留签名的参数名 /
  ``self.downside_penalty`` 状态属性 / 文档文本不计）；只读过渡依赖与旧侧
  为同一模块对象（B5 口径）；
- 禁止真实配置读取永久回归（F 组，范式同 T4 R2-T4-01 / T5）。

真实注册模型冒烟按评审 2 注记为**手工证据**，测试套件不依赖生产模型文件。
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from loguru import logger

import src.lazybull.common.signal_factory as old_factory
import src.lazybull.signals.ensemble_signal as old_ens
import src.lazybull.signals.ml_signal as old_ml
import src.lazybull.v2.core.signal.ensemble_signal as new_ens
import src.lazybull.v2.core.signal.factory as new_factory
import src.lazybull.v2.core.signal.ml_signal as new_ml
from src.lazybull.common.trading_config import TradingConfig as OldTradingConfig
from src.lazybull.v2.common.trading_config import TradingConfig as NewTradingConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_V2_SIGNAL_DIR = _REPO_ROOT / "src" / "lazybull" / "v2" / "core" / "signal"
_OLD_DOWNSIDE_MODULE = _REPO_ROOT / "src" / "lazybull" / "signals" / "downside_penalty.py"

_DATE = pd.Timestamp("2026-01-05")
_FEATURE_COLUMNS = ["f1", "f2", "f3"]

# 退役 downside_penalty 模块的模块级符号冻结集（D6 扫描清单，语义同 T5 冻结集：
# 旧文件未来被改动即报警）
_FROZEN_RETIRED_DOWNSIDE_SYMBOLS = frozenset(
    {
        "DOWNSIDE_PENALTY_COLUMNS",
        "DOWNSIDE_PENALTY_GRID",
        "RISK_NAN_FILL",
        "apply_downside_penalty",
    }
)
# 退役挂载在 MLSignal 上的成员名（非模块级符号，单列冻结）
_FROZEN_RETIRED_MEMBER_NAMES = frozenset({"_apply_downside_penalty"})


# ---------------------------------------------------------------------------
# 合成 stub：模型 / 注册表 / 特征
# ---------------------------------------------------------------------------


class _FakeModel:
    """确定性合成模型：默认 predict = f1 + 10*f2 - f3。"""

    def __init__(self, score_fn=None):
        self._score_fn = score_fn or (lambda X: (X["f1"] + 10.0 * X["f2"] - X["f3"]).values)
        self.predict_calls = 0
        self.proba_calls = 0
        self.predict_inputs = []

    def predict(self, X):
        self.predict_calls += 1
        self.predict_inputs.append(X)
        return self._score_fn(X)

    def predict_proba(self, X):
        self.proba_calls += 1
        pos = np.asarray(self._score_fn(X), dtype=float)
        return np.column_stack([1.0 - pos, pos])


class _FakeModelNoProba:
    """无 predict_proba 的合成模型（classification 回退分支用）。"""

    def __init__(self, score_fn=None):
        self._score_fn = score_fn or (lambda X: X["f1"].values)
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return self._score_fn(X)


def _metadata(
    feature_columns: Optional[List[str]] = None,
    task: str = "regression",
    extra_train_params: Optional[Dict] = None,
) -> Dict:
    cols = list(feature_columns) if feature_columns is not None else list(_FEATURE_COLUMNS)
    train_params: Dict = {"task": task}
    if extra_train_params:
        train_params.update(extra_train_params)
    return {
        "feature_columns": cols,
        "version_str": "vSyn",
        "feature_count": len(cols),
        "train_params": train_params,
    }


def _install_fake_registries(monkeypatch, models, metas, consistency_error=None):
    """把新旧两侧 ``ModelRegistry`` 绑定打为同一合成注册表类。

    Args:
        models: {version: 模型对象}
        metas: {version: 元数据字典}
        consistency_error: 非 None 时 check_feature_consistency 抛该异常

    Returns:
        已创建的注册表实例列表（按创建顺序，供调用计数比对）。
    """
    instances = []

    class FakeRegistry:
        def __init__(self, models_dir=None):
            self.models_dir = models_dir
            self.load_calls = []
            instances.append(self)

        def load_model(self, version, strict_version_check=True):
            self.load_calls.append((version, strict_version_check))
            return models[version], metas[version]

        def check_feature_consistency(self, metadata, available_features):
            if consistency_error is not None:
                raise consistency_error

    monkeypatch.setattr(old_ml, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(new_ml, "ModelRegistry", FakeRegistry)
    return instances


def _make_features(n: int = 6) -> pd.DataFrame:
    codes = [f"{i:06d}.SZ" for i in range(1, n + 1)]
    rng = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame(
        {
            "ts_code": codes,
            "amount_ma20": 60000.0 + rng,
            "total_mv": 600000.0 + rng,
            "sw_l1_code": ["801110"] * n,
            "f1": rng,
            "f2": rng / 10.0,
            "f3": rng / 100.0,
        }
    )


def _universe(features: pd.DataFrame) -> List[str]:
    return features["ts_code"].tolist()


def _make_signal_pair(monkeypatch, model=None, meta=None, version: int = 1, **kwargs):
    """构造共享同一合成模型/元数据的新旧 MLSignal 对。"""
    model = model if model is not None else _FakeModel()
    meta = meta if meta is not None else _metadata()
    instances = _install_fake_registries(monkeypatch, {version: model}, {version: meta})
    kwargs.setdefault("models_dir", "syn")
    kwargs.setdefault("model_version", version)
    kwargs.setdefault("verbose", False)
    old = old_ml.MLSignal(**kwargs)
    new = new_ml.MLSignal(**kwargs)
    return old, new, instances


def _generate_both(old, new, features: pd.DataFrame, universe: Optional[List[str]] = None):
    uni = universe if universe is not None else _universe(features)
    out_old = old.generate(_DATE, uni, {"features": features.copy()})
    out_new = new.generate(_DATE, uni, {"features": features.copy()})
    return out_old, out_new


def _ranked_both(old, new, features: pd.DataFrame, universe: Optional[List[str]] = None):
    uni = universe if universe is not None else _universe(features)
    out_old = old.generate_ranked(_DATE, uni, {"features": features.copy()})
    out_new = new.generate_ranked(_DATE, uni, {"features": features.copy()})
    return out_old, out_new


# ---------------------------------------------------------------------------
# A 组：同模型同特征 ⇒ 输出逐位一致
# ---------------------------------------------------------------------------


def test_generate_bit_identical(monkeypatch):
    old, new, instances = _make_signal_pair(monkeypatch, top_n=3)
    features = _make_features()
    features.loc[0, "sw_l1_code"] = "801780"  # 金融股剔除
    out_old, out_new = _generate_both(old, new, features)
    assert out_old == out_new
    assert list(out_old.keys()) == list(out_new.keys())
    assert len(out_old) == 3
    assert "000001.SZ" not in out_old  # 金融股已被剔除
    # 两侧均走了 predict（regression）且各加载一次模型
    model = old.model
    assert model.predict_calls == 2 and model.proba_calls == 0
    assert instances[0].load_calls == instances[1].load_calls == [(1, True)]


def test_generate_ranked_bit_identical_and_cache(monkeypatch):
    old, new, _ = _make_signal_pair(monkeypatch, top_n=2)
    features = _make_features()
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    # ranked 返回全部候选（不受 top_n 截断）
    assert len(out_old) == len(features)
    assert old._last_ranked_candidates == new._last_ranked_candidates == out_old


def test_generate_equal_weight_fallback_when_all_nonpositive(monkeypatch):
    model = _FakeModel(score_fn=lambda X: np.full(len(X), -1.0))
    old, new, _ = _make_signal_pair(monkeypatch, model=model, top_n=4)
    features = _make_features()
    out_old, out_new = _generate_both(old, new, features)
    assert out_old == out_new
    assert set(out_old.values()) == {0.25}  # 全负 ⇒ 等权回退


def test_classification_uses_predict_proba(monkeypatch):
    model = _FakeModel()
    meta = _metadata(task="classification")
    old, new, _ = _make_signal_pair(monkeypatch, model=model, meta=meta, top_n=3)
    features = _make_features()
    out_old, out_new = _generate_both(old, new, features)
    assert out_old == out_new
    assert model.proba_calls == 2 and model.predict_calls == 0


def test_classification_without_proba_falls_back_to_predict(monkeypatch):
    model = _FakeModelNoProba()
    meta = _metadata(task="classification")
    old, new, _ = _make_signal_pair(monkeypatch, model=model, meta=meta, top_n=3)
    features = _make_features()
    out_old, out_new = _generate_both(old, new, features)
    assert out_old == out_new
    assert model.predict_calls == 2


def _early_exit_cases():
    features = _make_features()
    return {
        "no_features_key": ({}, None),
        "features_none": ({"features": None}, None),
        "features_empty": ({"features": features.iloc[0:0]}, None),
        "universe_no_match": ({"features": features}, ["999999.SZ"]),
        "all_filtered_out": (
            {"features": features.assign(amount_ma20=1.0)},
            None,
        ),
    }


@pytest.mark.parametrize(
    "case",
    ["no_features_key", "features_none", "features_empty", "universe_no_match", "all_filtered_out"],
)
def test_early_exit_paths_identical(monkeypatch, case):
    old, new, _ = _make_signal_pair(monkeypatch, top_n=3)
    data, universe = _early_exit_cases()[case]
    features = data.get("features")
    if universe is not None:
        uni = universe
    elif features is not None and len(features):
        uni = _universe(features)
    else:
        uni = ["000001.SZ"]
    assert old.generate(_DATE, uni, data) == new.generate(_DATE, uni, data) == {}
    assert old.generate_ranked(_DATE, uni, data) == new.generate_ranked(_DATE, uni, data) == []


def test_missing_feature_column_nan_fill_identical(monkeypatch):
    old, new, _ = _make_signal_pair(monkeypatch, top_n=5)
    features = _make_features().drop(columns=["f3"])
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    gen_old, gen_new = _generate_both(old, new, features)
    assert gen_old == gen_new


def test_feature_consistency_error_propagates_identically(monkeypatch):
    err = ValueError("合成一致性失败")
    _install_fake_registries(
        monkeypatch, {1: _FakeModel()}, {1: _metadata()}, consistency_error=err
    )
    old = old_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", verbose=False)
    new = new_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", verbose=False)
    features = _make_features()
    with pytest.raises(ValueError) as exc_old:
        old.generate(_DATE, _universe(features), {"features": features.copy()})
    with pytest.raises(ValueError) as exc_new:
        new.generate(_DATE, _universe(features), {"features": features.copy()})
    assert str(exc_old.value) == str(exc_new.value) == "合成一致性失败"


def test_all_nan_feature_column_rejects_identically(monkeypatch):
    model = _FakeModel()
    old, new, _ = _make_signal_pair(monkeypatch, model=model, top_n=3)
    features = _make_features()
    features["f2"] = np.nan  # 全空列 ⇒ 质量门禁拒绝预测
    assert _generate_both(old, new, features) == ({}, {})
    assert _ranked_both(old, new, features) == ([], [])
    assert model.predict_calls == 0


def test_serving_event_decay_positive_path(monkeypatch):
    """R2-T6-03/R1-T6-01 复收：事件 freshness 衰减正路径合成锁定。

    train_params 显式半衰期 30 天 + 特征注入 ``forecast_freshness_days=30``
    ⇒ 衰减权重 exp(-ln2·30/30) = 0.5，``forecast_type_score`` 2.0 → 1.0；
    断言模型实收值为衰减后数值（「删除/改弱衰减调用」变异必失败），
    且新旧输出逐位一致。
    """
    meta = _metadata(
        feature_columns=["f1", "forecast_type_score"],
        extra_train_params={"event_freshness_half_life_days": 30.0},
    )
    model = _FakeModel(score_fn=lambda X: X["forecast_type_score"].values)
    old, new, _ = _make_signal_pair(monkeypatch, model=model, meta=meta, top_n=10)
    features = _make_features()
    features["forecast_type_score"] = 2.0
    features["forecast_freshness_days"] = 30.0

    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    assert len(model.predict_inputs) == 2  # 新旧各预测一次
    for X_seen in model.predict_inputs:
        assert X_seen["forecast_type_score"].tolist() == [1.0] * len(features)

    gen_old, gen_new = _generate_both(old, new, features)
    assert gen_old == gen_new


def test_update_model_version_switch_identical(monkeypatch):
    model_v1 = _FakeModel(score_fn=lambda X: X["f1"].values)
    model_v2 = _FakeModel(score_fn=lambda X: (-X["f1"]).values)
    metas = {1: _metadata(), 2: _metadata()}
    instances = _install_fake_registries(monkeypatch, {1: model_v1, 2: model_v2}, metas)
    old = old_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", verbose=False)
    new = new_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", verbose=False)
    features = _make_features()

    ranked_old_v1, ranked_new_v1 = _ranked_both(old, new, features)
    assert ranked_old_v1 == ranked_new_v1

    # 同版本且模型已加载 ⇒ 不切换不重载
    old.update_model_version(1)
    new.update_model_version(1)
    assert [len(i.load_calls) for i in instances] == [1, 1]

    # 跨版本切换 ⇒ 缓存重置并按新版本重载，两侧输出逐位一致且不同于 v1
    old.update_model_version(2)
    new.update_model_version(2)
    ranked_old_v2, ranked_new_v2 = _ranked_both(old, new, features)
    assert ranked_old_v2 == ranked_new_v2
    assert ranked_old_v2 != ranked_old_v1
    # 切换后 _load_model 重新构造注册表：instances[0/1] 载 v1、instances[2/3] 载 v2
    assert instances[0].load_calls == instances[1].load_calls == [(1, True)]
    assert instances[2].load_calls == instances[3].load_calls == [(2, True)]


def test_selection_filters_boundary_identical(monkeypatch):
    old, new, _ = _make_signal_pair(monkeypatch, top_n=10)
    features = _make_features()
    features.loc[0, "amount_ma20"] = 50000.0  # 恰好下限 ⇒ 保留
    features.loc[1, "amount_ma20"] = 49999.9  # 低于下限 ⇒ 剔除
    features.loc[2, "total_mv"] = 500000.0  # 市值下界 ⇒ 保留
    features.loc[3, "total_mv"] = 15000000.0  # 市值上界 ⇒ 保留
    features.loc[4, "sw_l1_code"] = "801790"  # 非银金融 ⇒ 剔除
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    kept = {code for code, _ in out_old}
    assert kept == {"000001.SZ", "000003.SZ", "000004.SZ", "000006.SZ"}


def test_selection_filters_missing_columns_identical(monkeypatch):
    old, new, _ = _make_signal_pair(monkeypatch, top_n=10)
    features = _make_features().drop(columns=["amount_ma20", "total_mv", "sw_l1_code"])
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    assert len(out_old) == len(features)  # 缺列 ⇒ 跳过规则不过滤


def test_exclude_financial_off_identical(monkeypatch):
    old, new, _ = _make_signal_pair(monkeypatch, top_n=10, exclude_financial=False)
    features = _make_features()
    features.loc[0, "sw_l1_code"] = "801780"
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    assert "000001.SZ" in {code for code, _ in out_old}


@pytest.fixture()
def captured_logs():
    """loguru 捕获（新旧模块共用 loguru 全局 logger ⇒ 逐侧运行逐侧比对）。

    R1-T6-02 处置：诊断日志路径（警告分支 / 汇总日志）的级别+文本对账。
    """
    records = []

    def _sink(message) -> None:
        rec = message.record
        records.append((rec["level"].name, rec["message"]))

    handler_id = logger.add(_sink, level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(handler_id)


def test_check_feature_quality_log_text_identical(captured_logs):
    """质量门禁全分支日志（级别+文本）新旧逐串一致；广播列豁免不误报。"""
    old = old_ml.MLSignal(top_n=1, model_version=1, models_dir="syn", verbose=False)
    new = new_ml.MLSignal(top_n=1, model_version=1, models_dir="syn", verbose=False)
    captured_logs.clear()  # 丢弃构造期日志

    X = pd.DataFrame(
        {
            "all_nan": [np.nan] * 4,  # 全空 ⇒ ERROR 拒绝
            "high_missing": [1.0, np.nan, np.nan, np.nan],  # 75% > 50% ⇒ WARNING
            "all_zero": [0.0] * 4,  # 全零 ⇒ WARNING
            "constant": [7.0] * 4,  # 截面常量 ⇒ WARNING
            "mkt_const": [1.0] * 4,  # 广播列常量 ⇒ 不告警
            "north_zero": [0.0] * 4,  # 广播列全零 ⇒ 不告警
        }
    )
    result_old = old._check_feature_quality(X)
    logs_old = list(captured_logs)
    captured_logs.clear()
    result_new = new._check_feature_quality(X)
    logs_new = list(captured_logs)

    assert result_old == result_new is False
    assert logs_old == logs_new
    # 全分支命中锚定（防断言空转）：1 条拒绝 ERROR + 1 条聚合 WARNING，
    # 两条广播列不出现在任何日志中
    assert [level for level, _ in logs_new].count("ERROR") == 1
    assert [level for level, _ in logs_new].count("WARNING") == 1
    warn_text = next(text for level, text in logs_new if level == "WARNING")
    assert "high_missing" in warn_text and "all_zero" in warn_text and "constant" in warn_text
    assert all("mkt_const" not in text and "north_zero" not in text for _, text in logs_new)


def test_check_feature_quality_missing_rate_boundary_logs(captured_logs):
    """缺失率 50% 不告警 / 51% 告警（> 严格语义），两侧日志逐串一致。"""
    old = old_ml.MLSignal(top_n=1, model_version=1, models_dir="syn", verbose=False)
    new = new_ml.MLSignal(top_n=1, model_version=1, models_dir="syn", verbose=False)
    captured_logs.clear()

    def _frame(n_nan: int) -> pd.DataFrame:
        values = np.arange(100, dtype=float)
        values[:n_nan] = np.nan
        return pd.DataFrame({"col": values})

    for n_nan, expect_warn in ((50, False), (51, True)):
        assert old._check_feature_quality(_frame(n_nan)) is True
        logs_old = list(captured_logs)
        captured_logs.clear()
        assert new._check_feature_quality(_frame(n_nan)) is True
        logs_new = list(captured_logs)
        captured_logs.clear()
        assert logs_old == logs_new
        assert bool(logs_new) is expect_warn


def test_prediction_summary_log_identical(captured_logs):
    """汇总日志：ranked 早退不输出 / 过滤前后计数两形态，两侧逐串一致。"""
    old = old_ml.MLSignal(top_n=1, model_version=1, models_dir="syn", verbose=False)
    new = new_ml.MLSignal(top_n=1, model_version=1, models_dir="syn", verbose=False)
    for sig in (old, new):
        sig.feature_columns = ["a", "b"]
    captured_logs.clear()

    for sig in (old, new):
        sig._log_prediction_pipeline_summary(5, 3, ranked=False)  # ⇒ "5→3"
        sig._log_prediction_pipeline_summary(5, 5, ranked=False)  # ⇒ "5"
        sig._log_prediction_pipeline_summary(5, 3, ranked=True)  # ⇒ 不输出
    logs = list(captured_logs)
    # 两侧各 2 条 INFO（ranked=True 早退不输出），交替顺序旧新各半且逐串一致
    assert [level for level, _ in logs] == ["INFO"] * 4
    assert logs[0] == logs[2] and logs[1] == logs[3]
    assert logs[0][1] == "选股/预测: 5→3, 特征2"
    assert logs[1][1] == "选股/预测: 5, 特征2"


# ---------------------------------------------------------------------------
# B 组：A5 退役摘除（D6）——λ>0 fail-fast、签名保留（B3）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.25, 0.5])
def test_lambda_positive_fail_fast(lam):
    with pytest.raises(ValueError, match="A5 退役"):
        new_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", downside_penalty=lam)
    # 对照：旧侧 λ>0 合法构造（fail-fast 为新侧有意的退役分歧，B3/D6）
    old = old_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", downside_penalty=lam)
    assert old.downside_penalty == lam


@pytest.mark.parametrize("lam", [-0.1, 1.0])
def test_lambda_range_validation_message_identical(lam):
    with pytest.raises(ValueError) as exc_old:
        old_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", downside_penalty=lam)
    with pytest.raises(ValueError) as exc_new:
        new_ml.MLSignal(top_n=3, model_version=1, models_dir="syn", downside_penalty=lam)
    assert str(exc_old.value) == str(exc_new.value)
    assert "必须落于 [0, 1)" in str(exc_new.value)


def test_lambda_zero_signature_preserved():
    sig = new_ml.MLSignal(
        top_n=3,
        model_version=1,
        models_dir="syn",
        downside_penalty=0.0,
        downside_penalty_column="custom_col",
    )
    assert sig.downside_penalty == 0.0
    assert sig.downside_penalty_column == "custom_col"


# ---------------------------------------------------------------------------
# C 组：EnsembleSignal 等价
# ---------------------------------------------------------------------------


def _make_ensemble_pair(monkeypatch, weight_a: float = 0.6, top_n: int = 3):
    # 模型分数方向不同且非共线（R2-T6-02：f1 线性递增 vs 顶点 3.5 的倒抛物线），
    # 任一权重下融合排序都同时依赖两侧——「忽略 B」变异必改变输出
    model_a = _FakeModel(score_fn=lambda X: X["f1"].values)
    model_b = _FakeModel(score_fn=lambda X: (-((X["f1"] - 3.5) ** 2)).values)
    metas = {1: _metadata(), 2: _metadata()}
    _install_fake_registries(monkeypatch, {1: model_a, 2: model_b}, metas)
    old_a = old_ml.MLSignal(top_n=top_n, model_version=1, models_dir="syn", verbose=False)
    old_b = old_ml.MLSignal(top_n=top_n, model_version=2, models_dir="syn", verbose=False)
    new_a = new_ml.MLSignal(top_n=top_n, model_version=1, models_dir="syn", verbose=False)
    new_b = new_ml.MLSignal(top_n=top_n, model_version=2, models_dir="syn", verbose=False)
    old = old_ens.EnsembleSignal(old_a, old_b, weight_a=weight_a)
    new = new_ens.EnsembleSignal(new_a, new_b, weight_a=weight_a)
    return old, new


def test_ensemble_generate_and_ranked_bit_identical(monkeypatch):
    old, new = _make_ensemble_pair(monkeypatch)
    features = _make_features()
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new
    assert old._last_ranked_candidates == new._last_ranked_candidates == out_old
    gen_old, gen_new = _generate_both(old, new, features)
    assert gen_old == gen_new
    assert list(gen_old.keys()) == list(gen_new.keys())


def test_ensemble_fusion_weight_effectiveness(monkeypatch):
    """R2-T6-02 复收：融合权重实效锁定（「忽略 B」/权重漂移变异必失败）。

    样本 f1 = 1..6；A 分 = f1，B 分 = -(f1-3.5)^2；weight_a=0.6 时
    融合分 = 0.6*f1 - 0.4*(f1-3.5)^2 ⇒ 排序 4 > 5 > 3 > 6 > 2 > 1，
    既不同于 A 单排（6>5>…>1）也不同于 B 单排（3≈4 > 2≈5 > 1≈6）。
    """
    old, new = _make_ensemble_pair(monkeypatch, weight_a=0.6, top_n=6)
    features = _make_features()
    out_old, out_new = _ranked_both(old, new, features)
    assert out_old == out_new

    expected_scores = {f"{i:06d}.SZ": 0.6 * i - 0.4 * (i - 3.5) ** 2 for i in range(1, 7)}
    expected_order = [
        "000004.SZ",
        "000005.SZ",
        "000003.SZ",
        "000006.SZ",
        "000002.SZ",
        "000001.SZ",
    ]
    assert [code for code, _ in out_new] == expected_order
    for code, score in out_new:
        assert score == pytest.approx(expected_scores[code])

    # 反变异锚定：「忽略 B」替身（仅 ranked_a 排序）必须与真实融合结果不同
    a_only_order = [
        code
        for code, _ in sorted(
            ((f"{i:06d}.SZ", float(i)) for i in range(1, 7)), key=lambda kv: kv[1], reverse=True
        )
    ]
    assert [code for code, _ in out_new] != a_only_order


def test_ensemble_top_n_setter_and_version_updates(monkeypatch):
    old, new = _make_ensemble_pair(monkeypatch)
    for ens in (old, new):
        ens.top_n = 7
        assert ens.top_n == ens.signal_a.top_n == ens.signal_b.top_n == 7
    assert old.top_n == new.top_n == 7

    old.update_versions(3, 4)
    new.update_versions(3, 4)
    assert (old.model_version, old.model_version_b) == (new.model_version, new.model_version_b)
    assert (old.model_version, old.model_version_b) == (3, 4)

    old.update_model_version(5)
    new.update_model_version(5)
    assert old.model_version == new.model_version == 5
    assert old.model_version_b == new.model_version_b == 4


def test_ensemble_equal_weight_fallback(monkeypatch):
    model_neg = _FakeModel(score_fn=lambda X: np.full(len(X), -2.0))
    metas = {1: _metadata(), 2: _metadata()}
    _install_fake_registries(monkeypatch, {1: model_neg, 2: model_neg}, metas)
    old = old_ens.EnsembleSignal(
        old_ml.MLSignal(top_n=4, model_version=1, models_dir="syn", verbose=False),
        old_ml.MLSignal(top_n=4, model_version=2, models_dir="syn", verbose=False),
        weight_a=0.5,
    )
    new = new_ens.EnsembleSignal(
        new_ml.MLSignal(top_n=4, model_version=1, models_dir="syn", verbose=False),
        new_ml.MLSignal(top_n=4, model_version=2, models_dir="syn", verbose=False),
        weight_a=0.5,
    )
    features = _make_features()
    gen_old, gen_new = _generate_both(old, new, features)
    assert gen_old == gen_new
    assert set(gen_old.values()) == {0.25}  # 合成分全负 ⇒ 等权回退


# ---------------------------------------------------------------------------
# D 组：factory 等价
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_models_root(monkeypatch):
    """新旧两侧工厂绑定的 get_stock_selection_models_root 打为同一合成 stub。"""
    calls = []

    def _stub() -> str:
        calls.append(1)
        return "syn_root"

    monkeypatch.setattr(old_factory, "get_stock_selection_models_root", _stub)
    monkeypatch.setattr(new_factory, "get_stock_selection_models_root", _stub)
    return calls


def _config_pair(**overrides):
    return OldTradingConfig(**overrides), NewTradingConfig(**overrides)


def test_factory_single_model_parity_explicit_dir(synthetic_models_root):
    cfg_old, cfg_new = _config_pair(top_n=5, model_version=1)
    sig_old = old_factory.create_signal(cfg_old, models_dir="explicit", verbose=False)
    sig_new = new_factory.create_signal(cfg_new, models_dir="explicit", verbose=False)
    assert type(sig_old).__name__ == type(sig_new).__name__ == "MLSignal"
    assert sig_old.top_n == sig_new.top_n == 5
    assert sig_old.model_version == sig_new.model_version == 1
    assert sig_old.models_dir == sig_new.models_dir == "explicit"
    assert sig_old.downside_penalty == sig_new.downside_penalty == 0.0
    assert synthetic_models_root == []  # 显式 models_dir ⇒ 不读模型根


def test_factory_single_model_parity_root_resolution(synthetic_models_root):
    cfg_old, cfg_new = _config_pair(top_n=5, model_version=1)
    sig_old = old_factory.create_signal(cfg_old, verbose=False)
    sig_new = new_factory.create_signal(cfg_new, verbose=False)
    assert sig_old.models_dir == sig_new.models_dir == "syn_root"
    assert len(synthetic_models_root) == 2  # 新旧各解析一次


def test_factory_ensemble_parity(synthetic_models_root):
    cfg_old, cfg_new = _config_pair(
        top_n=5, model_version=1, model_version_b=2, ensemble_weight_a=0.7
    )
    sig_old = old_factory.create_signal(cfg_old, models_dir="explicit", verbose=False)
    sig_new = new_factory.create_signal(cfg_new, models_dir="explicit", verbose=False)
    assert type(sig_old).__name__ == type(sig_new).__name__ == "EnsembleSignal"
    assert sig_old.weight_a == sig_new.weight_a == 0.7
    assert sig_old.weight_b == sig_new.weight_b
    assert (
        (sig_old.model_version, sig_old.model_version_b)
        == (
            sig_new.model_version,
            sig_new.model_version_b,
        )
        == (1, 2)
    )


def test_factory_lambda_positive_fail_fast_divergence(synthetic_models_root):
    cfg_old, cfg_new = _config_pair(top_n=5, model_version=1, downside_penalty=0.25)
    # 新侧：λ>0 经工厂透传至 MLSignal 构造 ⇒ fail-fast（D6/B3）
    with pytest.raises(ValueError, match="A5 退役"):
        new_factory.create_signal(cfg_new, models_dir="explicit", verbose=False)
    # 旧侧：行为冻结不动，λ>0 照常构造（对照证明分歧为有意摘除）
    sig_old = old_factory.create_signal(cfg_old, models_dir="explicit", verbose=False)
    assert sig_old.downside_penalty == 0.25


# ---------------------------------------------------------------------------
# E 组：改指锁定 + 退役符号扫描（归属判定，范式同 T5 D6 扫描制）
# ---------------------------------------------------------------------------


def _v2_signal_sources() -> List[Path]:
    return sorted(_V2_SIGNAL_DIR.glob("*.py"))


def _extract_downside_module_symbols() -> frozenset:
    """旧 downside_penalty 模块的模块级符号（函数 + 赋值目标）运行时 AST 提取。"""
    tree = ast.parse(_OLD_DOWNSIDE_MODULE.read_text(encoding="utf-8"))
    symbols = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return frozenset(symbols)


_OLD_SIGNAL_PACKAGE = "src.lazybull.signals"
_OLD_SIGNAL_FACTORY_MODULE = "src.lazybull.common.signal_factory"
_V2_SIGNAL_PACKAGE = "src.lazybull.v2.core.signal"


def _importer_package_of(path: Path) -> str:
    """被扫文件的所属包 = 文件父目录相对仓库根的点分路径。

    普通模块与包初始化文件统一取文件父目录（T5 R2-T5-R3-01 口径：
    不把 ``__init__`` 当包名后缀）。
    """
    return ".".join(path.resolve().parent.relative_to(_REPO_ROOT).parts)


def _resolve_import_from_module(node: ast.ImportFrom, importer_package: str) -> str:
    """ImportFrom 目标模块解析：绝对导入原样；相对导入按所属包向上 level-1 级归一。"""
    if node.level == 0:
        return node.module or ""
    parts = importer_package.split(".") if importer_package else []
    keep = len(parts) - (node.level - 1)
    anchor = ".".join(parts[:keep]) if keep > 0 else ""
    return f"{anchor}.{node.module}" if node.module else anchor


def _import_candidates(tree: ast.AST, importer_package: str) -> List[str]:
    """全量 import 候选模块路径（R2-T6-01 处置：统一导入归属解析）。

    ``from <pkg> import <name>`` 既可能引用 pkg 本身，也可能经父包引用
    <pkg>.<name> 子模块——两种归属都展开为候选（含相对导入归一），
    由调用方按目标集合判定。
    """
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from_module(node, importer_package)
            candidates.append(resolved)
            candidates.extend(f"{resolved}.{alias.name}" for alias in node.names if resolved)
    return candidates


def _scan_source_for_retired_downside(
    path: Path, importer_package: Optional[str] = None
) -> List[str]:
    """扫描单个源文件中的退役 downside_penalty 引用（归属判定）。

    命中口径：退役模块 import（绝对/相对/父包成员导入全形态，经
    ``_import_candidates`` 统一解析）/ 退役模块级符号的 Name 读取 /
    退役成员名（``_apply_downside_penalty``）的定义、调用与属性访问。
    不计：保留签名的参数名与关键字参数（B3）、``self.downside_penalty``
    状态属性、字符串与文档文本。
    """
    package = importer_package or _importer_package_of(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    retired_symbols = _FROZEN_RETIRED_DOWNSIDE_SYMBOLS | _FROZEN_RETIRED_MEMBER_NAMES
    for candidate in _import_candidates(tree, package):
        if candidate.split(".")[-1] == "downside_penalty":
            hits.append(f"import {candidate}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _FROZEN_RETIRED_MEMBER_NAMES:
                hits.append(f"def {node.name}")
        elif isinstance(node, ast.Name) and node.id in retired_symbols:
            hits.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in retired_symbols:
            hits.append(f"attr {node.attr}")
    return hits


def _scan_source_for_old_signal_imports(
    path: Path, importer_package: Optional[str] = None
) -> List[str]:
    """扫描对旧信号层模块（``src.lazybull.signals.*`` / 旧工厂）的 import 引用。

    绝对导入、相对导入（按被扫文件真实所属包归一）与父包成员导入全形态
    覆盖（R2-T6-01 处置）。
    """
    package = importer_package or _importer_package_of(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for candidate in _import_candidates(tree, package):
        if candidate == _OLD_SIGNAL_PACKAGE or candidate.startswith(_OLD_SIGNAL_PACKAGE + "."):
            hits.append(candidate)
        elif candidate == _OLD_SIGNAL_FACTORY_MODULE:
            hits.append(candidate)
    return hits


def test_retired_downside_symbol_extraction_matches_frozen_set():
    assert _extract_downside_module_symbols() == _FROZEN_RETIRED_DOWNSIDE_SYMBOLS


def test_v2_signal_no_retired_downside_symbols():
    sources = _v2_signal_sources()
    assert len(sources) == 5  # __init__ / base / ml_signal / ensemble_signal / factory
    hits = {p.name: _scan_source_for_retired_downside(p) for p in sources}
    hits = {name: h for name, h in hits.items() if h}
    assert hits == {}, f"v2/core/signal 残留退役 downside_penalty 引用: {hits}"


def test_v2_signal_no_old_signal_module_imports():
    hits = {p.name: _scan_source_for_old_signal_imports(p) for p in _v2_signal_sources()}
    hits = {name: h for name, h in hits.items() if h}
    assert hits == {}, f"v2/core/signal 仍引用旧信号层模块: {hits}"


def test_retired_downside_scanner_catches_mutations(tmp_path):
    """扫描器有效性永久回归：变异反例全捕获、归属负例零误报。"""
    positive_cases = {
        "abs_import": "from src.lazybull.signals.downside_penalty import apply_downside_penalty\n",
        "rel_import": "from .downside_penalty import RISK_NAN_FILL\n",
        "rel_parent_pkg_import": "from ....signals import downside_penalty as legacy\n",
        "abs_parent_pkg_import": "from src.lazybull.signals import downside_penalty\n",
        "plain_import": "import src.lazybull.signals.downside_penalty as dp\n",
        "name_call": "def f(df):\n    return apply_downside_penalty(df)\n",
        "name_read": "x = DOWNSIDE_PENALTY_COLUMNS\n",
        "method_def": "class A:\n    def _apply_downside_penalty(self, df, col):\n        pass\n",
        "method_call": "self._apply_downside_penalty(features_df, score_column)\n",
    }
    for case, src in positive_cases.items():
        probe = tmp_path / f"probe_{case}.py"
        probe.write_text(src, encoding="utf-8")
        hits = _scan_source_for_retired_downside(probe, importer_package=_V2_SIGNAL_PACKAGE)
        assert hits, f"变异反例未捕获: {case}"

    negative_cases = {
        "state_attr": "self.downside_penalty = float(downside_penalty or 0.0)\n",
        "signature_param": (
            "def f(downside_penalty=0.0, downside_penalty_column='x'):\n"
            "    return downside_penalty\n"
        ),
        "keyword_arg": "MLSignal(downside_penalty=0.25, downside_penalty_column='x')\n",
        "doc_text": '"""A5 路由型下行风险惩罚（downside_penalty）已摘除。"""\n',
    }
    for case, src in negative_cases.items():
        probe = tmp_path / f"probe_neg_{case}.py"
        probe.write_text(src, encoding="utf-8")
        hits = _scan_source_for_retired_downside(probe, importer_package=_V2_SIGNAL_PACKAGE)
        assert not hits, f"归属负例误报: {case}"


def test_old_signal_import_scanner_catches_mutations(tmp_path):
    """旧信号层 import 扫描：绝对/相对/父包成员导入全捕获，合法 v2 导入零误报。"""
    positive_cases = {
        "abs_module": "from src.lazybull.signals.base import Signal\n",
        "abs_pkg_member": "from src.lazybull import signals\n",
        "rel_module": "from ....signals.ml_signal import MLSignal as legacy\n",
        "rel_pkg_member": "from ....signals import ensemble_signal as legacy\n",
        "rel_factory": "from ....common.signal_factory import create_signal as legacy\n",
        "abs_factory": "from src.lazybull.common.signal_factory import create_signal\n",
    }
    for case, src in positive_cases.items():
        probe = tmp_path / f"probe_old_{case}.py"
        probe.write_text(src, encoding="utf-8")
        hits = _scan_source_for_old_signal_imports(probe, importer_package=_V2_SIGNAL_PACKAGE)
        assert hits, f"变异反例未捕获: {case}"

    negative_cases = {
        "v2_abs": "from src.lazybull.v2.core.signal.base import Signal\n",
        "v2_rel_base": "from .base import Signal\n",
        "v2_rel_ml": "from .ml_signal import MLSignal\n",
        "v2_rel_pkg_member": "from . import ensemble_signal\n",
        "v2_rel_common": "from ...common.trading_config import TradingConfig\n",
        "v2_rel_up3": "from ....v2.core.signal.base import Signal\n",
    }
    for case, src in negative_cases.items():
        probe = tmp_path / f"probe_old_neg_{case}.py"
        probe.write_text(src, encoding="utf-8")
        hits = _scan_source_for_old_signal_imports(probe, importer_package=_V2_SIGNAL_PACKAGE)
        assert not hits, f"合法 v2 导入误报: {case}"


@pytest.mark.parametrize(
    "inject_line",
    [
        "from ....signals import downside_penalty as legacy",
        "from ....signals.ml_signal import MLSignal as legacy",
        "from ....signals import ensemble_signal as legacy",
        "from ....common.signal_factory import create_signal as legacy",
    ],
)
def test_old_import_injection_via_real_entry_rejected(monkeypatch, inject_line):
    """R2-T6-01 复收：四例合法相对/父包导入注入真实被扫文件后，

    经**正式扫描文件入口**（真实路径 → 真实所属包归一）必须被捕获。
    """
    target = _V2_SIGNAL_DIR / "ml_signal.py"
    original = target.read_text(encoding="utf-8")
    reads = {"n": 0}
    real_read_text = Path.read_text

    def _patched(self, *args, **kwargs):
        if self == target:
            reads["n"] += 1
            return original + "\n" + inject_line + "\n"
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _patched)
    retired_hits = _scan_source_for_retired_downside(target)
    old_signal_hits = _scan_source_for_old_signal_imports(target)
    assert reads["n"] >= 2, "替身未命中，注入空转"
    assert retired_hits or old_signal_hits, f"注入未被任何扫描器捕获: {inject_line}"


def test_importer_package_of_v2_signal_files_exact():
    """所属包锚定：扫描面 5 文件的所属包 == 真实父目录（含 __init__.py 统一口径）。"""
    sources = _v2_signal_sources()
    assert len(sources) == 5
    for path in sources:
        assert _importer_package_of(path) == _V2_SIGNAL_PACKAGE, f"{path.name} 所属包计算错误"


def test_shared_dependency_object_identity():
    """只读过渡依赖与旧侧为同一模块对象（B5 口径：无第二份配置/注册表状态）。"""
    assert old_ml.get_stock_selection_models_root is new_ml.get_stock_selection_models_root
    assert old_ml.ModelRegistry is new_ml.ModelRegistry
    assert old_ml.ensure_availability_markers is new_ml.ensure_availability_markers
    assert old_ml.apply_event_freshness_decay is new_ml.apply_event_freshness_decay
    assert (
        old_factory.get_stock_selection_models_root is new_factory.get_stock_selection_models_root
    )
    # MRO 名称序列一致（基类为 v2 复制件，对象不同但结构等价）
    assert [c.__name__ for c in old_ml.MLSignal.__mro__] == [
        c.__name__ for c in new_ml.MLSignal.__mro__
    ]


# ---------------------------------------------------------------------------
# F 组：禁止真实配置读取永久回归（范式同 T4 R2-T4-01 / T5）
# ---------------------------------------------------------------------------


class _PassedCounter:
    """inner pytest 运行的通过数记录器（复收证据：复跑非空跑）。"""

    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.outcome == "passed":
            self.passed += 1


def test_t6_suite_forbids_real_config_read(monkeypatch):
    """永久回归：禁止读取替身 + 计数断言。

    把新旧两侧信号/工厂模块绑定的 ``get_stock_selection_models_root`` 替换为
    **立即抛 RuntimeError 的禁止读取替身**，在该替身下复跑整个 T6 测试文件
    （本项除外，防递归）：必须全绿、真实配置读取 0 次。任一测试绕过合成
    stub 直接读取真实配置 ⇒ 替身抛出 ⇒ 该测试失败。
    """
    real_reads: List[int] = []

    def _reject_configuration_read():
        real_reads.append(1)
        raise RuntimeError("reviewer: unexpected configuration read")

    for module in (old_ml, new_ml, old_factory, new_factory):
        monkeypatch.setattr(module, "get_stock_selection_models_root", _reject_configuration_read)

    counter = _PassedCounter()
    exit_code = pytest.main(
        [
            "-q",
            "-k",
            "not test_t6_suite_forbids_real_config_read",
            str(Path(__file__)),
        ],
        plugins=[counter],
    )
    assert exit_code == 0, "禁止真实配置读取下 T6 套件存在失败项"
    assert real_reads == [], f"真实配置读取发生 {len(real_reads)} 次"
    # 复跑非空跑：当前文件 46 项（含本项），inner 应跑 45 项；下限防未来漂移
    assert counter.passed >= 44, f"inner 复跑通过数异常: {counter.passed}"
