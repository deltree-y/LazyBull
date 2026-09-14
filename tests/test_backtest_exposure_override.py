"""暴露覆盖（P2-3 shadow）单元测试：查表语义、契约校验与导出工具。

覆盖两个硬契约：
1. **默认关闭时行为必须与改动前逐位一致**（表为 None 时系数恒 1.0）；
2. 系数必须落于 (0, 1]，缺失日期按 1.0 处理并计数（禁止静默外推或加杠杆）。
"""

import pandas as pd
import pytest

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.exposure_override import (
    BacktestExposureOverrideMixin,
    load_exposure_table,
)
from src.lazybull.risk.terminal_loss.exposure_gate import export_exposure_table


class _StubEngine(BacktestExposureOverrideMixin):
    """只承载查表逻辑的最小桩（不依赖真实引擎）。"""

    def _calculate_portfolio_value(self, date):  # pragma: no cover - 桩用
        return 1_000_000.0


def test_mixin_is_mounted_on_engine():
    """契约：暴露覆盖必须由专用 mixin 提供，且引擎已挂载。"""
    assert BacktestExposureOverrideMixin in BacktestEngine.__mro__


def test_disabled_returns_one_for_every_date():
    """未启用（表为 None）时系数恒为 1.0（逐位一致的开关）。"""
    engine = _StubEngine()
    assert engine.exposure_table is None
    for date in ("20240102", "20240103", pd.Timestamp("2024-01-04")):
        assert engine._get_exposure_multiplier(date) == 1.0
    assert engine.get_exposure_report() == {"enabled": False}


def test_missing_date_falls_back_to_one_and_counts():
    """表中缺失的日期按 1.0 处理并计数（不静默外推）。"""
    engine = _StubEngine()
    engine.set_exposure_table({"20240102": 0.5}, verbose=False)
    assert engine._get_exposure_multiplier("20240102") == 0.5
    assert engine._get_exposure_multiplier("20240103") == 1.0
    report = engine.get_exposure_report()
    assert report["enabled"] is True
    assert report["covered_days"] == 1
    assert report["reduced_days"] == 1
    assert report["missing_days"] == 1
    assert report["missing_samples"] == ["20240103"]


def test_date_keys_normalize_dash_and_timestamp():
    """日期键统一为 YYYYMMDD（接受 Timestamp 与带横杠字符串）。"""
    engine = _StubEngine()
    engine.set_exposure_table({"2024-01-02": 0.7}, verbose=False)
    assert engine._get_exposure_multiplier(pd.Timestamp("2024-01-02")) == 0.7
    assert engine._get_exposure_multiplier("2024-01-02") == 0.7


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, 2.0])
def test_invalid_multiplier_rejected_on_load(bad):
    """装载非法系数必须报错（只允许降暴露，禁止加杠杆）。"""
    engine = _StubEngine()
    with pytest.raises(ValueError, match="暴露系数必须落于"):
        engine.set_exposure_table({"20240102": bad}, verbose=False)


def test_invalid_multiplier_rejected_at_lookup():
    """绕过装载直接改表也要在查表时报错。"""
    engine = _StubEngine()
    engine.set_exposure_table({"20240102": 1.0}, verbose=False)
    engine.exposure_table["20240102"] = 1.8
    with pytest.raises(ValueError, match="暴露系数必须落于"):
        engine._get_exposure_multiplier("20240102")


def test_reload_resets_stats():
    """重复装载必须重置统计，避免跨实验串味。"""
    engine = _StubEngine()
    engine.set_exposure_table({"20240102": 0.5}, verbose=False)
    engine._get_exposure_multiplier("20240109")
    assert engine.get_exposure_report()["missing_days"] == 1
    engine.set_exposure_table({"20240102": 0.5, "20240103": 1.0}, verbose=False)
    report = engine.get_exposure_report()
    assert report["missing_days"] == 0
    assert report["table_days"] == 2


def test_load_exposure_table_roundtrip(tmp_path):
    """两列 CSV 读取 + 日期规范化。"""
    path = tmp_path / "暴露系数表.csv"
    pd.DataFrame({"日期": ["2024-01-02", "20240103"], "暴露系数": [0.5, 1.0]}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    table = load_exposure_table(path)
    assert table == {"20240102": 0.5, "20240103": 1.0}


def test_load_exposure_table_rejects_bad_input(tmp_path):
    """缺列 / 空表 / 非法系数 / 日期重复都必须报错。"""
    missing = tmp_path / "缺列.csv"
    pd.DataFrame({"日期": ["20240102"]}).to_csv(missing, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="缺少列"):
        load_exposure_table(missing)

    empty = tmp_path / "空表.csv"
    pd.DataFrame({"日期": [], "暴露系数": []}).to_csv(empty, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="为空"):
        load_exposure_table(empty)

    bad = tmp_path / "非法.csv"
    pd.DataFrame({"日期": ["20240102"], "暴露系数": [1.2]}).to_csv(
        bad, index=False, encoding="utf-8-sig"
    )
    with pytest.raises(ValueError, match="非法暴露系数"):
        load_exposure_table(bad)

    dup = tmp_path / "重复.csv"
    pd.DataFrame({"日期": ["20240102", "2024-01-02"], "暴露系数": [0.5, 0.6]}).to_csv(
        dup, index=False, encoding="utf-8-sig"
    )
    with pytest.raises(ValueError, match="日期重复"):
        load_exposure_table(dup)


def test_export_exposure_table_requires_explicit_arm(tmp_path):
    """逐日判定表含多臂时必须显式指定臂，否则报错（禁止默默取错臂）。"""
    import pandas as pd

    judged = pd.DataFrame(
        {
            "arm": ["E2", "E2", "对照B", "对照B"],
            "date": ["20240102", "20240103", "20240102", "20240103"],
            "exposure_multiplier": [0.5, 1.0, 0.7, 0.7],
        }
    )
    with pytest.raises(ValueError, match="必须显式指定臂"):
        export_exposure_table(judged, tmp_path / "x.csv")

    path = export_exposure_table(judged, tmp_path / "暴露系数表.csv", arm_label="E2")
    table = load_exposure_table(path)
    assert table == {"20240102": 0.5, "20240103": 1.0}
    assert list(pd.read_csv(path, encoding="utf-8-sig").columns) == ["日期", "暴露系数"]


def test_export_exposure_table_fills_ledger_gaps(tmp_path):
    """台账缺口（无持仓日）按交易日历**顺延前一状态**，不得留下让引擎按 1.0 兜底的空位。"""
    judged = pd.DataFrame(
        {
            "arm": ["E2", "E2", "E2"],
            "date": ["20240102", "20240105", "20240108"],
            "exposure_multiplier": [0.5, 0.5, 1.0],
        }
    )
    trading_days = ["20231229", "20240102", "20240103", "20240104", "20240105", "20240108",
                    "20240109"]
    path = export_exposure_table(
        judged, tmp_path / "gap.csv", arm_label="E2", trading_days=trading_days
    )
    table = load_exposure_table(path)
    # 首末日期之外不补（不臆造区间外状态）；区间内缺口 0103/0104/0109 顺延前值
    assert table == {
        "20240102": 0.5,
        "20240103": 0.5,
        "20240104": 0.5,
        "20240105": 0.5,
        "20240108": 1.0,
    }


def test_export_exposure_table_without_calendar_keeps_raw_dates(tmp_path):
    """不传交易日历时保持原样（与历史产物一致，缺口留给引擎兜底）。"""
    judged = pd.DataFrame(
        {"arm": ["E2", "E2"], "date": ["20240102", "20240105"],
         "exposure_multiplier": [0.5, 1.0]}
    )
    path = export_exposure_table(judged, tmp_path / "raw.csv", arm_label="E2")
    assert load_exposure_table(path) == {"20240102": 0.5, "20240105": 1.0}


def test_export_exposure_table_rejects_duplicate_dates(tmp_path):
    """单臂内日期重复必须报错（系数表必须一日一行）。"""
    judged = pd.DataFrame(
        {"arm": ["E2", "E2"], "date": ["20240102", "20240102"], "exposure_multiplier": [0.5, 0.6]}
    )
    with pytest.raises(ValueError, match="日期重复"):
        export_exposure_table(judged, tmp_path / "x.csv", arm_label="E2")
