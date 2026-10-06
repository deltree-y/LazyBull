# -*- coding: utf-8 -*-
"""v2 labels 构建器测试（合成数据，不触碰真实 data/）。

覆盖：extract_labels 三表结构/列名适配（y_ret_N→label_value、neu_y_ret_N→neu_label_value）、
sealed/forming 边界（22 交易日合成日历：idx=0 sealed、idx≥1 forming）、
异常路径（缺列/多日/日期不在日历）、append_labels_for_day 与 store 生命周期联动
（forming 可重写 / sealed 拒改写 / 指纹一致 no-op）。
"""

import pandas as pd
import pytest

from datetime import date, timedelta

from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.labels_builder import (
    append_labels_for_day,
    extract_labels,
    maturity_status_for,
)

_CODES = ["600000.SH", "000001.SZ", "000002.SZ"]
#: 22 个交易日的合成日历（idx+1+20 < 22 ⟺ idx==0 才 sealed；恰 8 位数字日契约）
_CALENDAR_22 = [(date(2024, 1, 2) + timedelta(days=i)).strftime("%Y%m%d") for i in range(22)]


def _big_day(date: str, labels: tuple = (0.01, 0.02, 0.03)) -> pd.DataFrame:
    """合成单日大表（含 6 标签列 + 噪声特征列）。"""
    n = len(_CODES)
    return pd.DataFrame(
        {
            "trade_date": [date] * n,
            "ts_code": _CODES,
            "ret_1": [0.01, -0.02, 0.0],
            "y_ret_5": [labels[0]] * n,
            "y_ret_10": [labels[1]] * n,
            "y_ret_20": [labels[2]] * n,
            "neu_y_ret_5": [labels[0] / 2] * n,
            "neu_y_ret_10": [labels[1] / 2] * n,
            "neu_y_ret_20": [labels[2] / 2] * n,
        }
    )


class TestMaturityStatus:
    def test_sealed_boundary(self):
        assert maturity_status_for(_CALENDAR_22[0], _CALENDAR_22) == "sealed"  # idx+21=21 < 22

    def test_forming_boundary(self):
        assert maturity_status_for(_CALENDAR_22[1], _CALENDAR_22) == "forming"  # idx+21=22 越界
        assert maturity_status_for(_CALENDAR_22[-1], _CALENDAR_22) == "forming"

    def test_date_not_in_calendar_raises(self):
        with pytest.raises(ValueError, match="不在交易日历"):
            maturity_status_for("20990101", _CALENDAR_22)

    def test_data_end_gates_sealed(self):
        """R3-06：日历存在 ≠ 数据可算——端点超数据水位（data_end）⇒ forming。"""
        day0 = _CALENDAR_22[0]  # 日历端点 _CALENDAR_22[21] 存在
        endpoint = _CALENDAR_22[21]
        assert maturity_status_for(day0, _CALENDAR_22, data_end=None) == "sealed"  # 不约束
        assert maturity_status_for(day0, _CALENDAR_22, data_end=endpoint) == "sealed"  # 端点 ≤ 水位
        assert maturity_status_for(day0, _CALENDAR_22, data_end=_CALENDAR_22[20]) == "forming"

    def test_extract_labels_data_end_passthrough(self):
        tables = extract_labels(_big_day(_CALENDAR_22[0]), _CALENDAR_22, data_end=_CALENDAR_22[20])
        assert tables["y_ret_5"]["maturity_status"].iloc[0] == "forming"
        tables = extract_labels(_big_day(_CALENDAR_22[0]), _CALENDAR_22, data_end="20991231")
        assert tables["y_ret_5"]["maturity_status"].iloc[0] == "sealed"


class TestExtractLabels:
    def test_three_tables_schema(self):
        tables = extract_labels(_big_day(_CALENDAR_22[0]), _CALENDAR_22)
        assert set(tables) == {"y_ret_5", "y_ret_10", "y_ret_20"}
        for name, table in tables.items():
            assert list(table.columns) == [
                "ts_code",
                "trade_date",
                "label_value",
                "neu_label_value",
                "maturity_status",
            ]
            assert table["maturity_status"].unique().tolist() == ["sealed"]
        assert tables["y_ret_5"]["label_value"].tolist() == [0.01] * 3
        assert tables["y_ret_5"]["neu_label_value"].tolist() == [0.005] * 3
        assert tables["y_ret_20"]["label_value"].tolist() == [0.03] * 3

    def test_forming_status_propagates(self):
        tables = extract_labels(_big_day(_CALENDAR_22[5]), _CALENDAR_22)
        assert tables["y_ret_10"]["maturity_status"].iloc[0] == "forming"

    def test_missing_label_column_raises(self):
        df = _big_day(_CALENDAR_22[0]).drop(columns=["neu_y_ret_10"])
        with pytest.raises(ValueError, match="缺列"):
            extract_labels(df, _CALENDAR_22)

    def test_multi_date_raises(self):
        df = pd.concat([_big_day(_CALENDAR_22[0]), _big_day(_CALENDAR_22[1])])
        with pytest.raises(ValueError, match="单日"):
            extract_labels(df, _CALENDAR_22)


class TestAppendLabelsForDay:
    def test_forming_rewrite_and_sealed_reject(self, tmp_path):
        store = PanelDataStore(tmp_path)
        forming_day = _CALENDAR_22[1]  # forming
        append_labels_for_day(store, _big_day(forming_day), _CALENDAR_22)
        # forming 可重写（内容变化覆盖）
        append_labels_for_day(store, _big_day(forming_day, labels=(0.09, 0.08, 0.07)), _CALENDAR_22)
        df = pd.read_parquet(tmp_path / "labels" / "y_ret_5" / f"{forming_day}.parquet")
        assert df["label_value"].tolist() == [0.09] * 3
        assert store.manifest.label_sealed_through("y_ret_5") is None

        sealed_day = _CALENDAR_22[0]  # sealed
        append_labels_for_day(store, _big_day(sealed_day), _CALENDAR_22)
        assert store.manifest.label_sealed_through("y_ret_20") == sealed_day
        append_labels_for_day(store, _big_day(sealed_day), _CALENDAR_22)  # 指纹一致 no-op
        with pytest.raises(RuntimeError, match="已封存"):
            append_labels_for_day(store, _big_day(sealed_day, labels=(0.9, 0.9, 0.9)), _CALENDAR_22)

    def test_all_three_tables_written(self, tmp_path):
        store = PanelDataStore(tmp_path)
        append_labels_for_day(store, _big_day(_CALENDAR_22[0]), _CALENDAR_22)
        for name in ("y_ret_5", "y_ret_10", "y_ret_20"):
            assert (tmp_path / "labels" / name / f"{_CALENDAR_22[0]}.parquet").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
