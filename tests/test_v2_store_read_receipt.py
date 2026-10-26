# -*- coding: utf-8 -*-
"""store 读取回执（StoreReadReceipt，P3-T3 §13.3.5 方案 P2）测试。

验收口径：读前采集 + 读后核验 + receipt 只读属性返回（不改 load_features /
load_labels 签名）；清单单一来源 = 实际读取动作；读取期间漂移 ⇒
InputIdentityDriftError 且不产出 receipt。全部 tmp_path 合成数据。
"""

import inspect

import pandas as pd
import pytest

from src.lazybull.v2.common.types import FeatureQuery, LabelQuery, TradeDate
from src.lazybull.v2.store.data_store import PanelDataStore, StoreReadReceipt
from src.lazybull.v2.store.store_identity import InputIdentityDriftError

_D1 = TradeDate.from_str("20260105")
_D2 = TradeDate.from_str("20260106")
_CODES = ["600000.SH", "000001.SZ"]


def _make_store(tmp_path) -> PanelDataStore:
    store = PanelDataStore(tmp_path)
    store.manifest.register_columns(
        {
            "ret_1": {"group": "core", "source": "clean/daily"},
            "net_mf_amount": {"group": "moneyflow", "source": "raw/moneyflow"},
        }
    )
    store.manifest.save()
    return store


def _feature_df(date: str, group_col: str = "ret_1", values=(0.01, -0.02)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date] * len(_CODES),
            "ts_code": list(_CODES),
            group_col: list(values),
        }
    )


def _label_df(date: str, values=(0.03, -0.01)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date] * len(_CODES),
            "ts_code": list(_CODES),
            "label_value": list(values),
            "maturity_status": ["forming"] * len(_CODES),
        }
    )


def _feature_query(cols=("ret_1",), start=_D1, end=_D2) -> FeatureQuery:
    return FeatureQuery(columns=list(cols), start_date=start, end_date=end)


class TestReceiptContent:
    def test_features_receipt_lists_actual_files(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.append_features(_D2, "core", _feature_df("20260106"))
        store.append_features(_D1, "moneyflow", _feature_df("20260105", "net_mf_amount"))
        store.load_features(_feature_query(("ret_1", "net_mf_amount")))
        receipt = store.last_read_receipt
        assert isinstance(receipt, StoreReadReceipt)
        assert receipt.kind == "features"
        assert (receipt.start_date, receipt.end_date) == ("20260105", "20260106")
        assert receipt.verified is True
        paths = {record["path"] for record in receipt.files}
        assert paths == {
            "features/panel/20260105/core.parquet",
            "features/panel/20260106/core.parquet",
            "features/panel/20260105/moneyflow.parquet",
        }
        for record in receipt.files:
            assert len(record["sha256"]) == 64 and record["size"] > 0

    def test_labels_receipt_lists_actual_files(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_5", _label_df("20260105"))
        store.append_labels("y_ret_5", _label_df("20260106"))
        store.load_labels(LabelQuery(label_name="y_ret_5", start_date=_D1, end_date=_D2))
        receipt = store.last_read_receipt
        assert receipt.kind == "labels"
        assert {record["path"] for record in receipt.files} == {
            "labels/y_ret_5/20260105.parquet",
            "labels/y_ret_5/20260106.parquet",
        }

    def test_empty_result_still_produces_receipt(self, tmp_path):
        store = _make_store(tmp_path)
        store.load_features(_feature_query())
        receipt = store.last_read_receipt
        assert receipt.kind == "features" and receipt.files == () and receipt.verified
        store.load_labels(LabelQuery(label_name="y_ret_5", start_date=_D1, end_date=_D2))
        receipt = store.last_read_receipt
        assert receipt.kind == "labels" and receipt.files == ()

    def test_receipt_overwritten_per_read(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.load_features(_feature_query(("ret_1",), _D1, _D1))
        first = store.last_read_receipt
        store.load_labels(LabelQuery(label_name="y_ret_5", start_date=_D1, end_date=_D1))
        second = store.last_read_receipt
        assert first.kind == "features" and second.kind == "labels"
        assert second.files == ()


class TestReceiptDriftAndFailure:
    def test_drift_during_read_rejected(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.append_features(_D2, "core", _feature_df("20260106"))
        original_read = pd.read_parquet
        mutated: list[str] = []

        def tampering_read(path, *args, **kwargs):
            result = original_read(path, *args, **kwargs)
            if not mutated:  # 首个文件读取成功后、读后核验前篡改 ⇒ 核验必须拒绝
                with open(path, "ab") as handle:
                    handle.write(b"drift")
                mutated.append(str(path))
            return result

        monkeypatch.setattr("pandas.read_parquet", tampering_read)
        with pytest.raises(InputIdentityDriftError, match="读取期间文件漂移"):
            store.load_features(_feature_query())
        assert store.last_read_receipt is None  # 漂移不产出 receipt

    def test_read_failure_produces_no_receipt(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.load_features(_feature_query(("ret_1",), _D1, _D1))
        before = store.last_read_receipt
        bad_file = tmp_path / "features" / "panel" / "20260106" / "core.parquet"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_bytes(b"not-a-parquet")
        with pytest.raises(Exception):  # noqa: B017（损坏 parquet 的具体异常类型依引擎而定）
            store.load_features(_feature_query())
        assert store.last_read_receipt is before  # 读取异常不覆盖上一回执

    def test_nested_read_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        store._begin_read()
        with pytest.raises(RuntimeError, match="嵌套"):
            store._begin_read()
        store._abort_read()  # 复位，避免影响同实例后续读取


class TestSignatureLock:
    """消费方兼容（§13.3.5 S6）：receipt 增强不得改变既有读取签名。"""

    def test_load_signatures_unchanged(self):
        features_sig = inspect.signature(PanelDataStore.load_features)
        assert list(features_sig.parameters) == ["self", "query"]
        labels_sig = inspect.signature(PanelDataStore.load_labels)
        assert list(labels_sig.parameters) == ["self", "query"]

    def test_receipt_is_read_only_property(self, tmp_path):
        assert isinstance(inspect.getattr_static(PanelDataStore, "last_read_receipt"), property)
        store = _make_store(tmp_path)  # 属性无 setter：实例赋值应失败
        with pytest.raises(AttributeError):
            store.last_read_receipt = None  # type: ignore[misc]
