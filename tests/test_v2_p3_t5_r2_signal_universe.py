"""R2-T5A-R2-01：真实回放股票池与信号辅证同域，主门冻结矩阵不受影响。"""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pandas as pd
import pytest

from scripts.v2_p3.t5_arm_assembly import build_backtest_host_params
from scripts.v2_p3.t5_arm_executors import _signal_universe_config_mapping
from scripts.v2_p3.t05b_fingerprint import score_model_fingerprint
from src.lazybull.data.storage import Storage
from src.lazybull.universe import BasicUniverse
from src.lazybull.universe.domains import domain_market_whitelist
from src.lazybull.v2.core.signal.ml_signal import MLSignal
from src.lazybull.v2.sleeves._registry import get_sleeve_spec

_DAY = "20240102"
_CODES = [f"60000{i}.SH" for i in range(1, 7)]
_SELECTION = {
    "top_n": 20,
    "min_amount_ma20": 50000.0,
    "min_total_mv": 500000.0,
    "max_total_mv": 15000000.0,
    "exclude_financial": True,
}
_UNIVERSE = {"exclude_st": True, "min_list_days": 365, "markets": ("主板",)}


class _Signal(MLSignal):
    """保留真实信号过滤与 generate_ranked；用明确特征值代替训练模型的预测。"""

    def _load_model(self) -> None:
        self.model = SimpleNamespace(models=[])
        self.feature_columns = ["f"]
        self.metadata = {"train_params": {}}

    def _build_model_input(self, date, frame):
        return frame.copy(), frame[["f"]]

    def _predict_scores(self, X, before, after, ranked=False):
        return X["f"].to_numpy()


@pytest.fixture
def signal_data(tmp_path, monkeypatch):
    """215 日新股/ST/停牌/无行情/涨停都具备标签与特征，只有候选池语义不同。"""
    storage = Storage(root_path=str(tmp_path))
    basic = pd.DataFrame(
        {
            "ts_code": _CODES,
            "name": ["NORMAL", "NEW", "STBAD", "SUSPEND", "NOQUOTE", "LIMIT"],
            "market": ["主板"] * 6,
            "list_date": ["20100101", "20230601"] + ["20100101"] * 4,
        }
    )
    features = pd.DataFrame(
        {
            "ts_code": _CODES,
            "trade_date": [_DAY] * 6,
            "f": [0.1, 0.9, 0.8, 0.7, 0.6, 0.5],
            "is_st": [False, False, True, False, False, False],
            "is_suspended": [False, False, False, True, False, False],
            "is_limit_up": [False, False, False, False, False, True],
            "amount_ma20": [100000.0] * 6,
            "total_mv": [1000000.0] * 6,
            "sw_l1_code": ["801120"] * 6,
            "y_ret_20": [0.05] * 6,
            "neu_y_ret_20": [0.01] * 6,
        }
    )
    quote = features.loc[
        features["ts_code"] != _CODES[4], ["ts_code", "trade_date", "is_suspended"]
    ].copy()
    quote["quote_origin"] = "clean_daily"
    storage.save_clean(basic, "stock_basic")
    storage.save_clean_by_date(quote, "daily", _DAY)
    storage.save_cs_train_day(features, _DAY)
    monkeypatch.setattr("src.lazybull.v2.core.signal.ml_signal.MLSignal", _Signal)
    return storage, basic, features, quote


def _score(root: Path, **overrides) -> Dict[str, object]:
    return score_model_fingerprint(
        models_dir=root,
        model_version=1,
        data_root=root,
        score_days=[_DAY],
        main_board_codes=set(_CODES),
        **overrides,
    )


def test_signal_artifact_matches_real_universe_and_main_matrix_is_unchanged(
    tmp_path, monkeypatch, signal_data
):
    _, basic, features, quote = signal_data
    assert (pd.Timestamp(_DAY) - pd.Timestamp("20230601")).days == 215
    universe = BasicUniverse(stock_basic=basic, verbose=False, **_UNIVERSE)
    actual_codes = universe.get_stocks(pd.Timestamp(_DAY), quote_data=quote)
    actual_signal = _Signal(**_SELECTION, verbose=False)
    expected = actual_signal.generate_ranked(
        pd.Timestamp(_DAY), actual_codes, {"features": features}
    )
    assert actual_codes == [_CODES[0], _CODES[5]]  # 涨停股仍可进入 T 日信号候选。
    calls = []
    original = BasicUniverse.get_stocks

    def capture(self, date, quote_data=None):
        assert quote_data is not None
        assert set(quote_data["quote_origin"]) == {"clean_daily"}
        calls.append((self.exclude_st, self.min_list_days, self.filter_suspended))
        return original(self, date, quote_data=quote_data)

    monkeypatch.setattr(BasicUniverse, "get_stocks", capture)
    baseline = _score(tmp_path)
    result = _score(
        tmp_path,
        persist_predictions=True,
        predictions_path=tmp_path / "artifacts/predictions.parquet",
        signal_selection=_SELECTION,
        signal_universe_config=_UNIVERSE,
    )
    assert calls == [(True, 365, True)]
    actual = pd.read_parquet(tmp_path / "artifacts/signal_topk.parquet")
    assert list(zip(actual["ts_code"], actual["score"])) == expected
    assert result["fingerprint"] == baseline["fingerprint"]
    assert result["per_seed_diagnostic"] == baseline["per_seed_diagnostic"]
    assert result["n_rows"] == baseline["n_rows"] == 3  # NORMAL、NEW、NOQUOTE：主门母行集保持原样。


@pytest.mark.parametrize("missing", ["stock_basic", "daily", "status_column"])
def test_missing_universe_metadata_rejected_before_artifact_write(tmp_path, signal_data, missing):
    _, _, _, quote = signal_data
    if missing == "stock_basic":
        (tmp_path / "clean/stock_basic.parquet").unlink()
    elif missing == "daily":
        (tmp_path / "clean/daily/2024-01-02.parquet").unlink()
    else:
        quote.drop(columns="is_suspended").to_parquet(
            tmp_path / "clean/daily/2024-01-02.parquet", index=False
        )
    with pytest.raises(ValueError, match="信号候选股票池"):
        _score(
            tmp_path,
            persist_predictions=True,
            predictions_path=tmp_path / "out/predictions.parquet",
            signal_selection=_SELECTION,
            signal_universe_config=_UNIVERSE,
        )
    assert not (tmp_path / "out").exists()


def test_default_signal_disabled_does_not_read_universe(tmp_path, monkeypatch, signal_data):
    def forbidden(*args, **kwargs):
        pytest.fail("spike 默认关闭信号辅证，不得新建 DataLoader 或读取候选池输入")

    monkeypatch.setattr("src.lazybull.data.loader.DataLoader", forbidden)
    assert _score(tmp_path)["n_rows"] == 3


def test_t5_universe_config_uses_backtest_host_parameters():
    spec = get_sleeve_spec("value")
    params = build_backtest_host_params(spec)
    assert _signal_universe_config_mapping() == {
        "exclude_st": params.bt_exclude_st,
        "min_list_days": params.bt_min_list_days,
        "markets": tuple(domain_market_whitelist(spec.domain)),
    }


@pytest.mark.parametrize(
    "selection,config,message",
    [(None, _UNIVERSE, "关闭时禁止"), (_SELECTION, None, "必须提供 signal_universe_config")],
)
def test_universe_config_none_guard_before_io(tmp_path, selection, config, message):
    with pytest.raises(ValueError, match=message):
        _score(
            tmp_path,
            persist_predictions=True,
            predictions_path=tmp_path / "out/predictions.parquet",
            signal_selection=selection,
            signal_universe_config=config,
        )
    assert not (tmp_path / "out").exists()
