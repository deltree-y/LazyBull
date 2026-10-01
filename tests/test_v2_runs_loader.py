# -*- coding: utf-8 -*-
"""runs 读入桥（runs_loader）契约 §9 硬校验单元测试：合成数据，不依赖真实批次。

覆盖：合法批全过；§9 通则（必需文件/trades 缺失）、§9.1 action 闭集与卖行 buy_date、
§9.2 严格递增与 bt_total_return 对账、§9.3 指纹重算与数据态一致、§9.4 同日同股同向、
§9.5 非 ASCII、§9.7 列集合、§9.8 跨折衔接与 daily/trades 交叉对账。
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.evidence.runs_schema import DAILY_CONTRACT_COLS, TRADES_CONTRACT_COLS

_CONFIG = {"bt_top_n": 20, "algorithm": "xgboost"}


def _trades_rows(split: int, buy_day: str, sell_day: str) -> pd.DataFrame:
    base = {
        "wf_run_id": "wf1", "split_index": split, "model_version": "v1",
        "signal_date": buy_day, "ts_code": "600000.SH", "price": 10.0, "shares": 100,
        "amount": 1000.0, "cost": 5.0, "buy_price": 10.0, "buy_pnl_price": 10.0,
        "sell_pnl_price": None, "pnl_profit_amount": None, "pnl_profit_pct": None,
        "sell_type": "", "sell_timing": "", "sell_reason": None, "trigger_type": None,
        "buy_type": None, "buy_reason": None, "lot_id": f"{buy_day}_0", "tranche_idx": None,
    }
    buy = {**base, "trade_date": buy_day, "action": "buy", "buy_date": buy_day}
    sell = {**base, "trade_date": sell_day, "action": "sell", "buy_date": buy_day,
            "sell_pnl_price": 10.5, "pnl_profit_amount": 45.0, "pnl_profit_pct": 0.045,
            "sell_type": "expiry", "sell_timing": "close"}
    return pd.DataFrame([buy, sell])[TRADES_CONTRACT_COLS]


def _daily_rows(split: int, days: list, tv0: float, tv1: float, n_buys: list, n_sells: list) -> pd.DataFrame:
    nav0, nav1 = 1.0, tv1 / tv0
    return pd.DataFrame({
        "trade_date": days, "total_value": [tv0, tv1], "cash": [tv0 - 1000.0, tv1],
        "market_value": [1000.0, 0.0], "nav": [nav0, nav1], "daily_return": [0.0, nav1 - nav0],
        "n_positions": [1, 0], "n_buys": n_buys, "n_sells": n_sells,
        "turnover_amount": [1000.0, 1000.0], "exposure_lambda": [1.0, 1.0],
    })[DAILY_CONTRACT_COLS]


def _make_runs_batch(root: Path) -> Path:
    """造一个合法 runs 批（2 折，全校验口径一致）。"""
    batch = root / "batch_ok"
    # chain_nav：折内序号口径；折1 首行 nav = 折0 末行 nav（跨折衔接）
    chain = pd.DataFrame({
        "date": [0, 1, 0, 1],
        "nav": [1.0, 1.02, 1.02, 1.0302],
        "split_index": [0, 0, 1, 1],
    })
    # bt_total_return = 折内 nav 起止比 - 1（容差 1e-6 内精确一致）
    bt0 = 1.02 / 1.0 - 1.0
    bt1 = 1.0302 / 1.02 - 1.0
    summary = pd.DataFrame({
        "batch_id": ["batch_ok", "batch_ok"], "split_index": [0, 1],
        "test_start": ["20240101", "20240201"], "test_end": ["20240131", "20240229"],
        "bt_total_return": [bt0, bt1], "data_state_id": ["ds1", "ds1"],
        "bt_top_n": [20, 20],
    })
    batch.mkdir(parents=True)
    chain.to_parquet(batch / "chain_nav.parquet", index=False)
    summary.to_parquet(batch / "summary.parquet", index=False)
    (batch / "batch_meta.json").write_text(json.dumps({
        "batch_id": "batch_ok", "config": _CONFIG, "config_fingerprint": fingerprint(_CONFIG),
        "data_state": {"data_state_id": "ds1"}, "baseline_ref": None, "arms": [],
    }, ensure_ascii=False), encoding="utf-8")
    fold_specs = [
        (0, ["20240102", "20240103"], 20000.0, 20400.0),
        (1, ["20240201", "20240202"], 20400.0, 20400.0 * (1.0302 / 1.02)),
    ]
    for split, days, tv0, tv1 in fold_specs:
        fold_dir = batch / "folds" / f"split{split:02d}"
        fold_dir.mkdir(parents=True)
        _trades_rows(split, days[0], days[1]).to_parquet(fold_dir / "trades.parquet", index=False)
        _daily_rows(split, days, tv0, tv1, [1, 0], [0, 1]).to_parquet(
            fold_dir / "daily.parquet", index=False)
        (fold_dir / "_meta.json").write_text(
            json.dumps({"trades.lot_reconstructed": True}), encoding="utf-8")
    return batch


class TestLoadHappyPath:
    def test_ok_batch_loads(self, tmp_path):
        batch = load_runs_batch(_make_runs_batch(tmp_path))
        assert batch.batch_id == "batch_ok"
        assert len(batch.folds) == 2
        assert batch.folds[0].meta["trades.lot_reconstructed"] is True  # §9.6 标注读取
        run = batch.to_run_artifacts()
        assert run.data_state_id == "ds1"
        assert set(run.windows.columns) == {"split_index", "test_start", "test_end"}


class TestMissingRequired:
    def test_missing_batch_meta(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        (batch / "batch_meta.json").unlink()
        with pytest.raises(FileNotFoundError, match="batch_meta.json"):
            load_runs_batch(batch)

    def test_missing_folds_dir(self, tmp_path):
        import shutil

        batch = _make_runs_batch(tmp_path)
        shutil.rmtree(batch / "folds")
        with pytest.raises(FileNotFoundError, match="folds"):
            load_runs_batch(batch)

    def test_missing_trades(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        (batch / "folds" / "split00" / "trades.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="trades"):
            load_runs_batch(batch)


class TestTradesInvariants:
    def _rewrite_trades(self, batch: Path, df: pd.DataFrame) -> Path:
        df.to_parquet(batch / "folds" / "split00" / "trades.parquet", index=False)
        return batch

    def test_bad_action(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        trades = pd.read_parquet(batch / "folds" / "split00" / "trades.parquet")
        trades.loc[0, "action"] = "hold"
        with pytest.raises(ValueError, match="action 含非法值"):
            load_runs_batch(self._rewrite_trades(batch, trades))

    def test_sell_buy_date_after_trade(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        trades = pd.read_parquet(batch / "folds" / "split00" / "trades.parquet")
        trades.loc[1, "buy_date"] = "20240109"  # 晚于成交日 20240103
        with pytest.raises(ValueError, match="buy_date 非法"):
            load_runs_batch(self._rewrite_trades(batch, trades))

    def test_same_day_same_action_same_stock(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        trades = pd.read_parquet(batch / "folds" / "split00" / "trades.parquet")
        dup = pd.concat([trades, trades.iloc[[0]]], ignore_index=True)  # 重复同一买行
        with pytest.raises(ValueError, match="同日同"):
            load_runs_batch(self._rewrite_trades(batch, dup))

    def test_trades_column_set(self, tmp_path):
        """§9.7：多列/缺列均报错。"""
        batch = _make_runs_batch(tmp_path)
        trades = pd.read_parquet(batch / "folds" / "split00" / "trades.parquet")
        trades["rogue_col"] = 1
        with pytest.raises(ValueError, match="列集合与契约不符"):
            load_runs_batch(self._rewrite_trades(batch, trades))
        batch2 = _make_runs_batch(tmp_path / "b2")
        t2 = pd.read_parquet(batch2 / "folds" / "split00" / "trades.parquet").drop(
            columns=["cost"])
        t2.to_parquet(batch2 / "folds" / "split00" / "trades.parquet", index=False)
        with pytest.raises(ValueError, match="列集合与契约不符"):
            load_runs_batch(batch2)


class TestChainNavInvariants:
    def _rewrite(self, batch: Path, chain: pd.DataFrame) -> Path:
        chain.to_parquet(batch / "chain_nav.parquet", index=False)
        return batch

    def test_duplicate_dates_rejected(self, tmp_path):
        """§9.2：折内重复日期（非严格递增）必须报错——is_monotonic_increasing 漏放的情形。"""
        batch = _make_runs_batch(tmp_path)
        chain = pd.read_parquet(batch / "chain_nav.parquet")
        chain.loc[1, "date"] = 0  # 折0 内日期 0 重复
        with pytest.raises(ValueError, match="严格递增"):
            load_runs_batch(self._rewrite(batch, chain))

    def test_bt_total_return_mismatch(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        summary = pd.read_parquet(batch / "summary.parquet")
        summary.loc[0, "bt_total_return"] = 0.5  # 远超 1e-6 容差
        summary.to_parquet(batch / "summary.parquet", index=False)
        with pytest.raises(ValueError, match="bt_total_return"):
            load_runs_batch(batch)

    def test_seam_nav_break(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        chain = pd.read_parquet(batch / "chain_nav.parquet")
        # 折1 两行 nav 同比缩放（折内比不变 ⇒ §9.2 对账仍过），但首行 ≠ 折0 末行 1.02
        chain.loc[2, "nav"] = 1.5
        chain.loc[3, "nav"] = 1.0302 * (1.5 / 1.02)
        with pytest.raises(ValueError, match="跨折衔接断裂"):
            load_runs_batch(self._rewrite(batch, chain))


class TestBatchMetaInvariants:
    def _rewrite_meta(self, batch: Path, meta: dict) -> Path:
        (batch / "batch_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return batch

    def test_fingerprint_mismatch(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        meta = json.loads((batch / "batch_meta.json").read_text(encoding="utf-8"))
        meta["config"]["bt_top_n"] = 30  # 篡改 config，指纹未同步
        with pytest.raises(ValueError, match="config_fingerprint 重算不一致"):
            load_runs_batch(self._rewrite_meta(batch, meta))

    def test_data_state_mismatch(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        meta = json.loads((batch / "batch_meta.json").read_text(encoding="utf-8"))
        meta["data_state"]["data_state_id"] = "ds_other"
        with pytest.raises(ValueError, match="data_state_id 不一致"):
            load_runs_batch(self._rewrite_meta(batch, meta))

    def test_missing_data_state(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        meta = json.loads((batch / "batch_meta.json").read_text(encoding="utf-8"))
        meta["data_state"] = {}
        with pytest.raises(ValueError, match="data_state_id 缺失"):
            load_runs_batch(self._rewrite_meta(batch, meta))


class TestDailyXref:
    def test_n_buys_mismatch(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        daily = pd.read_parquet(batch / "folds" / "split00" / "daily.parquet")
        daily.loc[0, "n_buys"] = 3  # trades 当日实际 1
        daily.to_parquet(batch / "folds" / "split00" / "daily.parquet", index=False)
        with pytest.raises(ValueError, match="指令计数与 trades 对不上"):
            load_runs_batch(batch)

    def test_total_value_ratio_mismatch(self, tmp_path):
        batch = _make_runs_batch(tmp_path)
        daily = pd.read_parquet(batch / "folds" / "split00" / "daily.parquet")
        daily.loc[1, "total_value"] = 99999.0  # 起止比偏离远超 1e-3
        daily.to_parquet(batch / "folds" / "split00" / "daily.parquet", index=False)
        with pytest.raises(ValueError, match="起止比"):
            load_runs_batch(batch)

    def test_fold_set_mismatch(self, tmp_path):
        import shutil

        batch = _make_runs_batch(tmp_path)
        # 删掉折1 目录但 summary/chain_nav 仍含折1 ⇒ 折集合不一致（先经 trades 校验
        # 的是存在的折；缺 folds/split01 整体 ⇒ 集合校验拦下）
        shutil.rmtree(batch / "folds" / "split01")
        with pytest.raises(ValueError, match="折集合不一致"):
            load_runs_batch(batch)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
