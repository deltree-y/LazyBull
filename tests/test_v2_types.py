# -*- coding: utf-8 -*-
"""v2 公共值对象测试（合成数据，不依赖真实数据）。

覆盖 P0 冻结口径：lot 聚合不变量（含空 lots 禁令，P0 评审 C1）、
VirtualAccount.borrowed_credit 默认值、PanelFrame.available_columns_at（协议 v0.4）。
"""

from datetime import date

import pandas as pd
import pytest

from src.lazybull.v2.common.types import (
    LabelQuery,
    Lot,
    Money,
    PanelFrame,
    Position,
    Price,
    Shares,
    TradeDate,
    TSCode,
    VirtualAccount,
)

_D1 = TradeDate(date(2026, 1, 5))
_D2 = TradeDate(date(2026, 2, 2))
_CODE = TSCode("600000.SH")


def _lot(shares: int, buy_date: TradeDate, cost: float) -> Lot:
    return Lot(shares=Shares(shares), buy_date=buy_date, cost_basis=Money(cost))


class TestPositionInvariants:
    def test_lots_consistent_ok(self):
        pos = Position(
            ts_code=_CODE,
            shares=Shares(1500),
            cost_basis=Money(15500.0),
            last_price=Price(10.5),
            lots=[_lot(1000, _D1, 10000.0), _lot(500, _D2, 5500.0)],
        )
        assert pos.first_buy_date == _D1

    def test_empty_lots_with_nonzero_shares_rejected(self):
        """P0 评审 C1：空 lots + 非零 shares 属矛盾态，构造即失败"""
        with pytest.raises(ValueError, match="空 lots"):
            Position(
                ts_code=_CODE,
                shares=Shares(100),
                cost_basis=Money(0.0),
                last_price=Price(10.0),
                lots=[],
            )

    def test_empty_lots_zero_shares_ok_but_no_first_buy_date(self):
        pos = Position(
            ts_code=_CODE,
            shares=Shares(0),
            cost_basis=Money(0.0),
            last_price=Price(10.0),
            lots=[],
        )
        with pytest.raises(ValueError, match="无 first_buy_date"):
            _ = pos.first_buy_date

    def test_shares_mismatch_rejected(self):
        with pytest.raises(ValueError, match="!= Σ lots.shares"):
            Position(
                ts_code=_CODE,
                shares=Shares(999),
                cost_basis=Money(10000.0),
                last_price=Price(10.0),
                lots=[_lot(1000, _D1, 10000.0)],
            )

    def test_cost_mismatch_rejected(self):
        with pytest.raises(ValueError, match="!= Σ lots.cost_basis"):
            Position(
                ts_code=_CODE,
                shares=Shares(1000),
                cost_basis=Money(1.0),
                last_price=Price(10.0),
                lots=[_lot(1000, _D1, 10000.0)],
            )


class TestVirtualAccount:
    def test_borrowed_credit_default_zero(self):
        acc = VirtualAccount(
            sleeve_id="A",
            cash=Money(100.0),
            positions={},
            total_value=Money(100.0),
        )
        assert acc.borrowed_credit == Money(0.0)

    def test_borrowed_credit_explicit(self):
        acc = VirtualAccount(
            sleeve_id="B",
            cash=Money(120.0),
            positions={},
            total_value=Money(120.0),
            borrowed_credit=Money(20.0),
        )
        assert acc.borrowed_credit.value == 20.0


class TestPanelFrameAvailableColumns:
    def _frame(self) -> PanelFrame:
        df = pd.DataFrame({"old_col": [1.0], "new_col": [2.0]})
        return PanelFrame(
            df=df,
            manifest_version="test",
            available_from={"old_col": TradeDate(date(2005, 1, 4)), "new_col": TradeDate(date(2020, 1, 1))},
        )

    def test_filters_by_available_from(self):
        frame = self._frame()
        assert list(frame.available_columns_at(TradeDate(date(2010, 1, 1)))) == ["old_col"]
        assert list(frame.available_columns_at(TradeDate(date(2026, 1, 1)))) == ["old_col", "new_col"]

    def test_unregistered_column_treated_as_always_available(self):
        """未登记起点的列按宽松默认视为恒可用（严格校验归 store 层）"""
        df = pd.DataFrame({"mystery": [1.0]})
        frame = PanelFrame(df=df, manifest_version="test")
        assert list(frame.available_columns_at(TradeDate(date(2005, 1, 4)))) == ["mystery"]

    def test_validate_schema_missing_raises(self):
        frame = self._frame()
        with pytest.raises(ValueError, match="缺列"):
            frame.validate_schema(["old_col", "not_exist"])
        frame.validate_schema(["old_col", "new_col"])  # 不报错


class TestLabelQueryVariant:
    """LabelQuery.variant（R3-05）：raw/neu 合法值 + 非法值构造即拒。"""

    def test_default_raw(self):
        q = LabelQuery(label_name="y_ret_20", start_date=_D1, end_date=_D2)
        assert q.variant == "raw"

    def test_invalid_variant_rejected(self):
        with pytest.raises(ValueError, match="变体非法"):
            LabelQuery(label_name="y_ret_20", start_date=_D1, end_date=_D2, variant="bad")


class TestPanelFrameManifestFingerprint:
    """PanelFrame.manifest_fingerprint（R3-10）：默认空串，由 load_features 填充。"""

    def test_default_empty(self):
        frame = PanelFrame(df=pd.DataFrame({"x": [1.0]}), manifest_version="1")
        assert frame.manifest_fingerprint == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
