"""测试数据确保和 T0 打印增强功能"""

import importlib
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.lazybull.data import DataCleaner, DataLoader, Storage, TushareClient
from src.lazybull.data.ensure import (
    _daily_basic_confirms_daily,
    _is_daily_coverage_low,
    ensure_basic_data,
    ensure_clean_data_for_date,
    ensure_raw_data_for_date,
)
import src.lazybull.features.ensure as ensure_module
import src.lazybull.features.ensure.downloads as ensure_downloads
import src.lazybull.features.ensure.entry as ensure_entry
import src.lazybull.features.ensure.factor_load as ensure_factor_load
import src.lazybull.features.ensure.historical_assets as ensure_hist_assets
from src.lazybull.features import FeatureBuilder, ensure_features_for_date
from src.lazybull.paper import PaperAccount, PaperStorage, TargetWeight
from src.lazybull.paper.reporting import load_position_snapshot
from src.lazybull.paper.runner import PaperTradingRunner


@pytest.fixture
def mock_client():
    """模拟 TushareClient"""
    client = Mock(spec=TushareClient)

    # 模拟交易日历
    trade_cal = pd.DataFrame(
        {
            "exchange": ["SSE"] * 5,
            "cal_date": ["20250120", "20250121", "20250122", "20250123", "20250124"],
            "is_open": [1, 1, 1, 1, 1],
        }
    )
    client.get_trade_cal.return_value = trade_cal

    # 模拟股票基本信息
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "name": ["测试股票1", "测试股票2"],
            "list_date": ["20200101", "20200101"],
            "market": ["主板", "主板"],
        }
    )
    client.get_stock_basic.return_value = stock_basic

    # 模拟日线数据
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20250121", "20250121"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "vol": [1000000, 2000000],
            "amount": [10000000, 40000000],
            "pct_chg": [5.0, 2.5],
            "pre_close": [10.0, 20.0],
        }
    )
    client.get_daily.return_value = daily

    # 模拟复权因子
    adj_factor = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20250121", "20250121"],
            "adj_factor": [1.0, 1.0],
        }
    )
    client.get_adj_factor.return_value = adj_factor

    # 模拟停复牌和涨跌停（空数据）
    client.get_suspend_d.return_value = pd.DataFrame()
    client.get_stk_limit.return_value = pd.DataFrame()
    client.get_stock_st.return_value = pd.DataFrame()

    return client


@pytest.fixture
def temp_storage():
    """临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        yield storage


def test_ensure_raw_data_for_date(mock_client, temp_storage):
    """测试确保 raw 数据存在"""
    trade_date = "20250121"

    # 首次调用应该下载数据
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date)
    assert result is True

    # 再次调用应该跳过下载（数据已存在）
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True


def test_ensure_raw_data_for_date_adj_factor_independent_fill(mock_client, temp_storage):
    """daily 已落盘但 adj_factor 缺失时，仍独立补齐 adj_factor（审计问题1）"""
    trade_date = "20250121"
    # 预置 daily 已存在（模拟历史某次只落盘了 daily，adj_factor 因抖动缺失）
    temp_storage.save_raw_by_date(mock_client.get_daily(trade_date=trade_date), "daily", trade_date)
    mock_client.get_daily.reset_mock()

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True
    # daily 已存在，不应再次下载
    mock_client.get_daily.assert_not_called()
    # adj_factor 被独立补齐落盘
    adj = temp_storage.load_raw_by_date("adj_factor", trade_date)
    assert adj is not None
    assert len(adj) == 2


def test_ensure_raw_data_for_date_adj_factor_empty_returns_false(mock_client, temp_storage):
    """daily 已存在但 adj_factor 返回空时补齐失败，不得报告成功。"""
    trade_date = "20250121"
    temp_storage.save_raw_by_date(mock_client.get_daily(trade_date=trade_date), "daily", trade_date)
    mock_client.get_adj_factor.return_value = pd.DataFrame()

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)

    assert result is False
    assert not temp_storage.is_data_exists("raw", "adj_factor", trade_date)


def test_ensure_raw_data_for_date_non_trade_day_short_circuit(mock_client, temp_storage):
    """日线为空且已确认非交易日时提前返回，不下载其余数据（审计问题3/4）"""
    trade_date = "20250119"  # 周日，非交易日
    temp_storage.save_raw(
        pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20250119", "20250121"],
                "is_open": [0, 1],
            }
        ),
        "trade_cal",
    )
    mock_client.get_daily.return_value = pd.DataFrame()

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True
    # 不落盘任何数据
    assert not temp_storage.is_data_exists("raw", "daily", trade_date)
    assert not temp_storage.is_data_exists("raw", "adj_factor", trade_date)
    # 不再调用其余数据下载接口
    mock_client.get_adj_factor.assert_not_called()
    mock_client.get_moneyflow.assert_not_called()
    mock_client.get_daily_basic.assert_not_called()


def test_ensure_raw_data_for_date_trade_day_empty_returns_false(mock_client, temp_storage):
    """交易日日线接口返回空（接口故障）时返回 False，不被误报为成功（审计问题3）"""
    trade_date = "20250121"  # 交易日
    temp_storage.save_raw(
        pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20250119", "20250121"],
                "is_open": [0, 1],
            }
        ),
        "trade_cal",
    )
    mock_client.get_daily.return_value = pd.DataFrame()

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is False
    assert not temp_storage.is_data_exists("raw", "daily", trade_date)


def test_ensure_raw_data_for_date_suspend_empty_placeholder(mock_client, temp_storage):
    """suspend 为空（当日无停牌）时写占位空文件，避免下次 ensure 重复请求（审计问题4）"""
    trade_date = "20250121"
    # 首次调用：suspend 返回空 → 写占位
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date)
    assert result is True
    assert temp_storage.is_data_exists("raw", "suspend", trade_date)
    # 再次调用：占位存在 → 不再请求 get_suspend_d
    mock_client.get_suspend_d.reset_mock()
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True
    mock_client.get_suspend_d.assert_not_called()


def test_ensure_raw_data_for_date_coverage_gate(mock_client, temp_storage):
    """daily_basic 文件存在但行数不足时，视为未补齐并重新下载（审计问题2，评审恢复门控）"""
    trade_date = "20250121"
    daily = mock_client.get_daily(trade_date=trade_date)  # 2 行
    temp_storage.save_raw_by_date(daily, "daily", trade_date)
    # 模拟历史截断落盘：daily_basic 只有 1 行（daily 有 2 行）
    temp_storage.save_raw_by_date(daily.iloc[:1], "daily_basic", trade_date)
    # 让 daily_basic 下载返回完整数据
    mock_client.get_daily_basic.return_value = daily

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True
    db = temp_storage.load_raw_by_date("daily_basic", trade_date)
    assert db is not None
    assert len(db) == 2  # 覆盖度门控触发重新补齐


def test_ensure_raw_data_for_date_daily_low_coverage_refetch(mock_client, temp_storage):
    """历史 daily 覆盖度偏低（截断/部分返回）时触发强制重下并成功修复（评审意见1）"""
    trade_date = "20250121"
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    temp_storage.save_raw_by_date(_small_daily(30, trade_date), "daily", trade_date)
    # 完整重下 100 行
    mock_client.get_daily.return_value = _small_daily(100, trade_date)

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True
    assert temp_storage.count_rows("raw", "daily", trade_date) == 100


def test_ensure_raw_data_for_date_fresh_low_coverage_not_saved(mock_client, temp_storage):
    """首次下载 daily 覆盖率偏低（截断/部分返回）时验证后不落盘并返回 False（评审意见1）"""
    trade_date = "20250121"
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    mock_client.get_daily.return_value = _small_daily(30, trade_date)  # 服务端返回 30/100

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is False
    assert not temp_storage.is_data_exists("raw", "daily", trade_date)  # 缺陷数据未落盘


def test_ensure_raw_data_for_date_daily_refetch_empty_fails(mock_client, temp_storage):
    """历史 daily 低覆盖重下返回空（接口故障）时返回 False（评审意见1）"""
    trade_date = "20250121"
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    temp_storage.save_raw_by_date(_small_daily(30, trade_date), "daily", trade_date)
    mock_client.get_daily.return_value = pd.DataFrame()  # 重下返回空

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is False
    # 旧分区保留（不删除），但明确失败
    assert temp_storage.is_data_exists("raw", "daily", trade_date)


def test_ensure_raw_data_for_date_daily_refetch_still_low_fails(mock_client, temp_storage):
    """历史 daily 低覆盖重下后仍低时返回 False（评审意见1）"""
    trade_date = "20250121"
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    temp_storage.save_raw_by_date(_small_daily(30, trade_date), "daily", trade_date)
    mock_client.get_daily.return_value = _small_daily(30, trade_date)  # 重下仍 30/100

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is False


def test_ensure_raw_data_for_date_duplicate_daily_not_saved(mock_client, temp_storage):
    """重复行不得虚增覆盖率：唯一代码严重不足时新 daily 不落盘。"""
    trade_date = "20250121"
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    duplicated = pd.concat([_small_daily(30, trade_date)] * 4, ignore_index=True)
    mock_client.get_daily.return_value = duplicated

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)

    assert result is False
    assert not temp_storage.is_data_exists("raw", "daily", trade_date)


def test_ensure_raw_data_for_date_historical_low_coverage_peer_confirmed(mock_client, temp_storage):
    """历史停牌导致覆盖率偏低时，daily_basic 代码域一致即可确认数据完整。"""
    trade_date = "20060104"
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    historical_daily = _small_daily(33, trade_date)  # 33/40=82.5%，低于粗筛阈值
    mock_client.get_daily.return_value = historical_daily
    mock_client.get_daily_basic.return_value = historical_daily[["ts_code", "trade_date"]]

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)

    assert result is True
    stored = temp_storage.load_raw_by_date("daily", trade_date)
    assert stored is not None
    assert stored["ts_code"].nunique() == 33


def test_daily_basic_peer_confirmation_uses_historical_tolerance():
    """历史接口约 2% 的代码域差异可接受，明显部分返回仍拒绝。"""
    trade_date = "20150710"
    daily = _small_daily(1425, trade_date)
    historical_basic = daily.iloc[:-28][["ts_code", "trade_date"]]
    partial_basic = daily.iloc[:-100][["ts_code", "trade_date"]]

    assert _daily_basic_confirms_daily(daily, historical_basic) is True
    assert _daily_basic_confirms_daily(daily, partial_basic) is False


def test_ensure_raw_data_for_date_moneyflow_not_daily_gated(mock_client, temp_storage):
    """moneyflow 天然不覆盖全部 daily 股票（如不含北交所），不应受 daily 行数门控（评审意见1）"""
    trade_date = "20250121"
    daily = mock_client.get_daily(trade_date=trade_date)  # 2 行
    temp_storage.save_raw_by_date(daily, "daily", trade_date)
    # moneyflow 只有 1 行（少于 daily，属正常代码域差异）
    temp_storage.save_raw_by_date(daily.iloc[:1], "moneyflow", trade_date)
    mock_client.get_moneyflow.reset_mock()

    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True
    # 已有 moneyflow 存在即视为已补齐，不重复下载
    mock_client.get_moneyflow.assert_not_called()
    mf = temp_storage.load_raw_by_date("moneyflow", trade_date)
    assert len(mf) == 1


def _stock_basic_100() -> pd.DataFrame:
    """100 只股票：40 只 2004 年上市、60 只 2020 年上市。"""
    return pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(100)],
            "name": ["测试"] * 100,
            "list_date": (["20040101"] * 40) + (["20200101"] * 60),
        }
    )


def _small_daily(n: int, trade_date: str) -> pd.DataFrame:
    """构造 n 行单日 daily 数据。"""
    return pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "trade_date": [trade_date] * n,
            "close": [10.0] * n,
        }
    )


def test_is_daily_coverage_low(mock_client, temp_storage):
    """daily 行数显著低于"当日已上市股票"时判定为低覆盖度（评审意见1/2）"""
    temp_storage.save_raw(_stock_basic_100(), "stock_basic")
    # 2005-01-04：当时已上市 40 只，35 只交易（87.5% >= 85%）→ 正常（不再用当前全集误伤历史）
    assert _is_daily_coverage_low(temp_storage, "20050104", 35) is False
    # 2005-01-04：仅 20 只交易（50% < 85%）→ 低
    assert _is_daily_coverage_low(temp_storage, "20050104", 20) is True
    # 2025-01-21：当时已上市 100 只，90 只交易（90% >= 85%）→ 正常
    assert _is_daily_coverage_low(temp_storage, "20250121", 90) is False
    # 无 stock_basic 基准时跳过（返回 False）
    import tempfile

    from src.lazybull.data.storage import Storage

    with tempfile.TemporaryDirectory() as tmpdir:
        empty_storage = Storage(root_path=tmpdir)
        assert _is_daily_coverage_low(empty_storage, "20250121", 30) is False


def test_ensure_basic_data(mock_client, temp_storage):
    """测试确保基础数据存在"""
    end_date = "20250121"

    # 首次调用应该下载数据
    result = ensure_basic_data(mock_client, temp_storage, end_date)
    assert result is True

    # 验证数据已保存
    trade_cal = temp_storage.load_raw("trade_cal")
    assert trade_cal is not None
    assert len(trade_cal) > 0

    stock_basic = temp_storage.load_raw("stock_basic")
    assert stock_basic is not None
    assert len(stock_basic) > 0


def test_ensure_clean_data_for_date(mock_client, temp_storage):
    """测试确保 clean 数据存在"""
    trade_date = "20250121"
    loader = DataLoader(temp_storage)
    cleaner = DataCleaner()

    # 确保基础数据存在
    ensure_basic_data(mock_client, temp_storage, trade_date)

    # 确保 clean 数据
    result = ensure_clean_data_for_date(temp_storage, loader, cleaner, mock_client, trade_date)
    assert result is True

    # 验证 clean 数据已保存
    assert temp_storage.is_data_exists("clean", "daily", trade_date)


def test_ensure_clean_rebuilds_cached_daily_with_invalid_adjusted_prices(mock_client, temp_storage):
    """旧 clean/daily 的复权价全空时自动失效并覆盖重建。"""
    trade_date = "20250121"
    loader = DataLoader(temp_storage)
    cleaner = DataCleaner()
    ensure_basic_data(mock_client, temp_storage, trade_date)
    assert ensure_clean_data_for_date(temp_storage, loader, cleaner, mock_client, trade_date)

    stale = temp_storage.load_clean_by_date("daily", trade_date)
    stale["close_adj"] = float("nan")
    temp_storage.save_clean_by_date(stale, "daily", trade_date)

    result = ensure_clean_data_for_date(
        temp_storage, loader, cleaner, mock_client, trade_date, force=False
    )

    rebuilt = temp_storage.load_clean_by_date("daily", trade_date)
    assert result is True
    assert rebuilt["close_adj"].notna().all()


def test_print_t0_targets():
    """测试 T0 打印信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 runner
        runner = PaperTradingRunner(initial_capital=500000.0, data_root=tmpdir, paper_root=tmpdir)

        # 创建测试数据
        targets = [
            TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="测试信号1"),
            TargetWeight(ts_code="000002.SZ", target_weight=0.3, reason="测试信号2"),
        ]

        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["测试股票1", "测试股票2"],
            }
        )

        daily_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "close": [10.5, 20.5],
            }
        )

        # 调用打印方法（不应抛出异常）
        try:
            runner._print_t0_targets(targets, stock_basic, daily_data)
            success = True
        except Exception as e:
            print(f"打印失败: {e}")
            success = False

        assert success is True


def test_enhance_target_info():
    """测试增强目标信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 runner
        runner = PaperTradingRunner(initial_capital=500000.0, data_root=tmpdir, paper_root=tmpdir)

        # 创建测试数据
        signal_dict = {
            "000001.SZ": 0.2,
            "000002.SZ": 0.3,
        }

        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["测试股票1", "测试股票2"],
            }
        )

        daily_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "close": [10.5, 20.5],
            }
        )

        # 调用增强方法
        targets = runner._enhance_target_info(signal_dict, stock_basic, daily_data, "20250121")

        # 验证结果
        assert len(targets) == 2
        assert targets[0].ts_code == "000001.SZ"
        assert targets[0].target_weight == 0.2
        assert "权重=0.2000" in targets[0].reason


def test_generate_instructions_missing_capital_retention_ratio_uses_default(monkeypatch):
    """测试缺少 capital_retention_ratio 时仍可生成 T0 指令。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.lazybull.paper.runner.TushareClient"):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr("src.lazybull.paper.runner.get_cost_settings", lambda: {})

        instructions = runner._generate_instructions(
            targets=[TargetWeight(ts_code="000001.SZ", target_weight=0.5, reason="测试信号")],
            buy_price_type="close",
            sell_price_type="open",
            current_prices={"000001.SZ": 10.0},
            source_date="20260325",
        )

        assert len(instructions) == 1
        assert instructions[0].action == "buy"
        assert instructions[0].shares == 5000


def test_generate_instructions_keeps_target_order_for_buys(monkeypatch):
    """测试 _generate_instructions 生成买单顺序与 targets 顺序一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.lazybull.paper.runner.TushareClient"):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr("src.lazybull.paper.runner.get_cost_settings", lambda: {})

        targets = [
            TargetWeight(ts_code="000003.SZ", target_weight=0.34, reason="r3"),
            TargetWeight(ts_code="000001.SZ", target_weight=0.33, reason="r1"),
            TargetWeight(ts_code="000002.SZ", target_weight=0.33, reason="r2"),
        ]
        current_prices = {
            "000001.SZ": 10.0,
            "000002.SZ": 10.0,
            "000003.SZ": 10.0,
        }

        instructions = runner._generate_instructions(
            targets=targets,
            buy_price_type="close",
            sell_price_type="open",
            current_prices=current_prices,
            source_date="20260325",
        )

        buy_codes = [inst.ts_code for inst in instructions if inst.action == "buy"]
        assert buy_codes == ["000003.SZ", "000001.SZ", "000002.SZ"]


def test_correct_trade_date_supports_next_with_last_trade_date(monkeypatch):
    """测试 next 会解析为上次执行日后的下一个交易日。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.lazybull.paper.runner.TushareClient"):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr(
            runner.loader,
            "load_clean_trade_cal",
            lambda: pd.DataFrame(
                {
                    "cal_date": ["20260325", "20260326", "20260327"],
                    "is_open": [1, 1, 1],
                }
            ),
        )
        monkeypatch.setattr(runner.paper_storage, "load_last_trade_date", lambda: "20260325")

        assert runner._correct_trade_date("next") == "20260326"


def test_correct_trade_date_refreshes_trade_calendar_when_clean_is_stale(monkeypatch):
    """测试非交易日输入会先刷新交易日历，再顺延到下一交易日。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.lazybull.paper.runner.TushareClient"):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        stale_clean = pd.DataFrame(
            {
                "cal_date": ["20260501", "20260502"],
                "is_open": [1, 1],
            }
        )
        stale_raw = pd.DataFrame(
            {
                "cal_date": ["20260501", "20260502"],
                "is_open": [1, 1],
            }
        )
        refreshed_raw = pd.DataFrame(
            {
                "cal_date": ["20260501", "20260502", "20260506"],
                "is_open": [1, 1, 1],
            }
        )

        clean_calls = {"count": 0}
        refresh_state = {"done": False}

        def fake_load_clean_trade_cal():
            clean_calls["count"] += 1
            return stale_clean

        monkeypatch.setattr(runner.loader, "load_clean_trade_cal", fake_load_clean_trade_cal)

        def fake_load_trade_cal():
            if refresh_state["done"]:
                return refreshed_raw
            return stale_raw

        monkeypatch.setattr(runner.loader, "load_trade_cal", fake_load_trade_cal)

        refresh_calls = []

        def fake_ensure_basic_data(client, storage, end_date, force=False):
            refresh_calls.append((end_date, force))
            refresh_state["done"] = True
            return True

        # 拆分后 _correct_trade_date 位于 runner_calendar mixin，ensure_basic_data 由其模块级引用提供
        monkeypatch.setattr(
            "src.lazybull.paper.runner.calendar.ensure_basic_data", fake_ensure_basic_data
        )

        assert runner._correct_trade_date("20260505") == "20260506"
        assert refresh_calls == [("20260505", False)]
        assert clean_calls["count"] >= 1


def test_get_next_trade_date_no_next_only_logs_debug(monkeypatch):
    """测试末尾交易日无下一交易日时不再打印 warning。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.lazybull.paper.runner.TushareClient"):
            runner = PaperTradingRunner(
                initial_capital=100000.0,
                data_root=tmpdir,
                paper_root=tmpdir,
                verbose=False,
            )

        monkeypatch.setattr(
            runner.loader,
            "load_clean_trade_cal",
            lambda: pd.DataFrame(
                {
                    "cal_date": ["20260506"],
                    "is_open": [1],
                }
            ),
        )
        monkeypatch.setattr(runner.loader, "load_trade_cal", lambda: None)

        warning_calls = []
        debug_calls = []
        monkeypatch.setattr(
            "src.lazybull.paper.runner.logger.warning",
            lambda message: warning_calls.append(message),
        )
        monkeypatch.setattr(
            "src.lazybull.paper.runner.logger.debug", lambda message: debug_calls.append(message)
        )

        assert runner._get_next_trade_date("20260506") is None
        assert warning_calls == []
        assert debug_calls == ["未找到 20260506 的下一个交易日"]


def test_load_factor_data_only_builds_trade_date_output(monkeypatch):
    """测试 _load_factor_data 只为目标交易日构建因子查询表输出。"""
    trade_date = "20260422"
    trading_dates = ["20260418", "20260421", trade_date]
    # 门控基于“最新公告日是否覆盖目标交易日”，stub 数据需带日期列且已覆盖目标日
    stub_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": [trade_date],
            "report_date": [trade_date],
        }
    )
    captured_dates = {}

    monkeypatch.setattr(
        ensure_factor_load,
        "_try_ensure_historical_margin",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ensure_factor_load,
        "_try_ensure_historical_cyq_perf",
        lambda *args, **kwargs: stub_df,
    )
    monkeypatch.setattr(
        ensure_factor_load,
        "_try_ensure_historical_fund_portfolio",
        lambda *args, **kwargs: stub_df,
    )
    monkeypatch.setattr(
        ensure_factor_load,
        "_try_ensure_historical_moneyflow_hsgt",
        lambda *args, **kwargs: stub_df,
    )
    monkeypatch.setattr(
        ensure_factor_load,
        "_try_ensure_historical_top_list",
        lambda *args, **kwargs: stub_df,
    )

    def _record_builder(name):
        def _builder(_df, output_dates, *args, **kwargs):
            captured_dates[name] = list(output_dates)
            return {trade_date: stub_df}

        return _builder

    builder_targets = [
        ("src.lazybull.factors.fundamental", "build_fundamental_lookup_by_date", "fundamental"),
        ("src.lazybull.factors.margin", "build_margin_lookup_by_date", "margin"),
        ("src.lazybull.factors.holder", "build_holder_lookup_by_date", "holder"),
        ("src.lazybull.factors.earnings", "build_earnings_lookup_by_date", "earnings"),
        ("src.lazybull.factors.cyq_perf", "build_cyq_perf_lookup_by_date", "cyq_perf"),
        ("src.lazybull.factors.express", "build_express_lookup_by_date", "express"),
        (
            "src.lazybull.factors.fund_portfolio",
            "build_fund_portfolio_lookup_by_date",
            "fund_portfolio",
        ),
        ("src.lazybull.factors.north_flow", "build_north_flow_lookup_by_date", "north_flow"),
        ("src.lazybull.factors.lhb", "build_lhb_lookup_by_date", "lhb"),
        ("src.lazybull.factors.consensus", "build_consensus_lookup_by_date", "consensus"),
        (
            "src.lazybull.factors.cashflow_quality",
            "build_cashflow_quality_lookup_by_date",
            "cashflow",
        ),
        (
            "src.lazybull.factors.consensus_revision",
            "build_consensus_revision_lookup_by_date",
            "consensus_revision",
        ),
        ("src.lazybull.factors.risk.announcement_lookup", "build_pledge_lookup_by_date", "pledge"),
        (
            "src.lazybull.factors.risk.announcement_lookup",
            "build_share_float_lookup_by_date",
            "share_float",
        ),
        (
            "src.lazybull.factors.risk.announcement_lookup",
            "build_block_trade_lookup_by_date",
            "block_trade",
        ),
    ]
    expected_names = {item[2] for item in builder_targets}

    for module_path, attr_name, name in builder_targets:
        monkeypatch.setattr(
            importlib.import_module(module_path),
            attr_name,
            _record_builder(name),
        )

    class StubLoader:
        def load_fina_indicator(self, start_date=None, end_date=None):
            return stub_df

        def load_margin_detail(self, start_date, end_date):
            return stub_df

        def load_stk_holdernumber(self):
            return stub_df

        def load_forecast(self):
            return stub_df

        def load_express(self):
            return stub_df

        def load_report_rc(self):
            return stub_df

        def load_cashflow(self, start_date=None, end_date=None):
            return stub_df

        def load_pledge_stat(self, start_date=None, end_date=None):
            return stub_df

        def load_share_float(self, start_date=None, end_date=None):
            return stub_df

        def load_block_trade(self, start_date=None, end_date=None):
            return stub_df

    storage = Mock()
    storage.load_sync_watermark.return_value = None

    result = ensure_module._load_factor_data(
        loader=StubLoader(),
        client=Mock(),
        storage=storage,
        trade_date=trade_date,
        trading_dates_str=trading_dates,
        start_date="20260401",
        end_date=trade_date,
    )

    assert set(captured_dates.keys()) == expected_names
    assert result[-1] == []
    for output_dates in captured_dates.values():
        assert output_dates == [trade_date]


def test_try_ensure_historical_fund_portfolio_builds_and_reuses_agg_cache(
    monkeypatch, temp_storage
):
    """测试基金持仓历史补齐会写入并复用季度聚合缓存。"""
    period = "20260331"
    raw_df = pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "symbol": "000001",
                "ann_date": "20260420",
                "end_date": period,
                "stk_float_ratio": 1.2,
                "mkv": 1000,
                "amount": 100,
            },
            {
                "ts_code": "000002.OF",
                "symbol": "000001",
                "ann_date": "20260421",
                "end_date": period,
                "stk_float_ratio": 0.8,
                "mkv": 900,
                "amount": 90,
            },
        ]
    )
    temp_storage.save_raw_by_date(raw_df, "fund_portfolio", period)

    monkeypatch.setattr(ensure_hist_assets, "_generate_quarter_periods", lambda *_args: [period])
    monkeypatch.setattr(
        ensure_hist_assets, "_query_with_pagination", lambda *args, **kwargs: pd.DataFrame()
    )

    result = ensure_module._try_ensure_historical_fund_portfolio(
        client=Mock(),
        storage=temp_storage,
        trading_dates_str=["20260422"],
    )

    assert result is not None
    assert temp_storage.is_data_exists("raw", "fund_portfolio_agg", period)

    original_load = temp_storage.load_raw_by_date

    def guarded_load(name, trade_date, format="parquet", columns=None):
        if name == "fund_portfolio":
            raise AssertionError("存在聚合缓存时不应再回读原始季度明细")
        return original_load(name, trade_date, format=format, columns=columns)

    monkeypatch.setattr(temp_storage, "load_raw_by_date", guarded_load)

    cached = ensure_module._try_ensure_historical_fund_portfolio(
        client=Mock(),
        storage=temp_storage,
        trading_dates_str=["20260422"],
    )

    assert cached is not None
    assert len(cached) == 1


def test_ensure_features_aligns_build_window_and_precompute(monkeypatch):
    """测试 ensure_features_for_date 对齐离线构建窗口并调用 precompute。"""
    trade_date = "20250121"
    trade_dt = pd.Timestamp(trade_date)

    expected_start = (
        trade_dt - pd.DateOffset(months=ensure_module.FEATURE_DATA_HISTORY_MONTHS)
    ).strftime("%Y%m%d")
    expected_end = (
        trade_dt + pd.DateOffset(months=ensure_module.FEATURE_DATA_FUTURE_MONTHS)
    ).strftime("%Y%m%d")

    trade_cal = pd.DataFrame(
        {
            "cal_date": pd.date_range("2024-05-01", "2025-03-31", freq="B"),
            "is_open": 1,
        }
    )
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["测试股票"],
            "list_date": ["20200101"],
            "market": ["主板"],
        }
    )
    daily_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [trade_date],
            "close": [10.0],
            "close_adj": [10.0],
            "open": [10.0],
            "open_adj": [10.0],
            "vol": [10000],
            "amount": [100000],
            "pct_chg": [0.0],
            "pre_close": [10.0],
        }
    )

    storage = Mock(spec=Storage)
    storage.is_feature_exists.return_value = False

    loader = Mock(spec=DataLoader)
    loader.load_clean_trade_cal.return_value = trade_cal
    loader.load_clean_stock_basic.return_value = stock_basic
    loader.load_clean_daily.return_value = daily_df
    loader.load_clean_daily_basic.return_value = pd.DataFrame()
    # 覆盖“moneyflow 缺失仅告警”路径
    loader.load_clean_moneyflow.return_value = None
    loader.load_shenwan_industry.return_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "sw_l1": ["测试行业"],
            "sw_l1_code": ["801000"],
        }
    )

    builder = Mock(spec=FeatureBuilder)
    builder.build_features_for_day.return_value = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
        }
    )

    monkeypatch.setattr(ensure_entry, "ensure_basic_data", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        ensure_entry,
        "ensure_clean_data_for_date",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        ensure_entry,
        "_ensure_historical_clean_data",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        ensure_entry,
        "_load_factor_data",
        lambda *args, **kwargs: (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [],
        ),
    )

    success, missing, error_detail = ensure_features_for_date(
        storage=storage,
        loader=loader,
        builder=builder,
        cleaner=Mock(spec=DataCleaner),
        client=Mock(spec=TushareClient),
        trade_date=trade_date,
        force=False,
    )

    assert success is True
    assert missing == []
    loader.load_clean_daily.assert_called_once_with(expected_start, expected_end)
    loader.load_clean_daily_basic.assert_called_once_with(expected_start, expected_end)
    loader.load_clean_moneyflow.assert_called_once_with(expected_start, expected_end)
    builder.precompute_daily_adj.assert_called_once()
    storage.save_cs_train_day.assert_called_once()


def test_incremental_catchup_by_calendar_date_covers_weekend_announcements(temp_storage):
    """测试公告类增量会按自然日补齐（含周末），而非只查交易日单点。"""
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20260331",
                "ann_date": "20260410",
                "revenue": 1.0,
            }
        ]
    )
    temp_storage.save_raw(existing, "express", is_force=True)

    queried_dates = []

    def _fetch_by_date(ann_date: str) -> pd.DataFrame:
        queried_dates.append(ann_date)
        if ann_date == "20260411":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "end_date": "20260331",
                        "ann_date": "20260411",
                        "revenue": 2.0,
                    }
                ]
            )
        if ann_date == "20260413":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000003.SZ",
                        "end_date": "20260331",
                        "ann_date": "20260413",
                        "revenue": 3.0,
                    }
                ]
            )
        return pd.DataFrame()

    result = ensure_module._incremental_catchup_by_calendar_date(
        storage=temp_storage,
        dataset_name="express",
        existing_df=temp_storage.load_raw("express"),
        trade_date="20260413",
        date_col="ann_date",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fetch_by_date=_fetch_by_date,
    )

    assert queried_dates == ["20260411", "20260412", "20260413"]
    assert result is not None
    ann_dates = result["ann_date"].astype(str).str.replace("-", "").str[:8]
    assert ann_dates.max() == "20260413"
    assert (ann_dates == "20260411").any()


@pytest.mark.parametrize(
    "func_name,dataset_name,threshold_attr,date_col",
    [
        ("_try_download_fina_indicator", "fina_indicator", "_MIN_FINA_RECORDS", "ann_date"),
        (
            "_try_download_stk_holdernumber",
            "stk_holdernumber",
            "_MIN_HOLDER_RECORDS",
            "ann_date",
        ),
        ("_try_download_forecast", "forecast", "_MIN_FORECAST_RECORDS", "ann_date"),
        ("_try_download_express", "express", "_MIN_EXPRESS_RECORDS", "ann_date"),
        ("_try_download_report_rc", "report_rc", "_MIN_REPORT_RC_RECORDS", "report_date"),
    ],
)
def test_incremental_download_functions_delegate_to_range_catchup(
    monkeypatch,
    temp_storage,
    func_name,
    dataset_name,
    threshold_attr,
    date_col,
):
    """测试公告/研报类增量下载统一走“日期区间补齐”入口。"""
    if date_col == "report_date":
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20260410",
                    "org_name": "测试机构",
                    "quarter": "2026Q1",
                }
            ]
        )
    else:
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20260410",
                    "end_date": "20260331",
                }
            ]
        )

    # 分区存储数据集按分区保存；其余公告类数据集仍为单文件
    if dataset_name in ("forecast", "fina_indicator"):
        temp_storage.save_raw_by_date(existing, dataset_name, "20260331")
    elif dataset_name == "report_rc":
        temp_storage.save_raw_by_date(existing, dataset_name, "2026-12-31")
    else:
        temp_storage.save_raw(existing, dataset_name, is_force=True)
    monkeypatch.setattr(ensure_downloads, threshold_attr, 1)

    captured = {}

    def _fake_catchup(**kwargs):
        captured.update(kwargs)
        return kwargs["existing_df"]

    monkeypatch.setattr(ensure_downloads, "_incremental_catchup_by_calendar_date", _fake_catchup)

    client = Mock(spec=TushareClient)
    func = getattr(ensure_module, func_name)
    result = func(client=client, storage=temp_storage, trade_date="20260430")

    assert result is not None
    assert captured["dataset_name"] == dataset_name
    assert captured["date_col"] == date_col
    assert captured["trade_date"] == "20260430"


def test_load_position_snapshot_ensures_trade_date_clean_data(monkeypatch):
    """查看/打印持仓前应自动补齐当日 clean 数据（缺数据自动下载）。"""
    runner = MagicMock()
    runner._correct_trade_date.return_value = "20260120"
    runner.storage = MagicMock()
    runner.loader = MagicMock()
    runner.cleaner = MagicMock()
    runner.client = MagicMock()
    runner.account.get_cash.return_value = 500000.0
    runner.account.initial_capital = 500000.0
    runner.broker.get_positions_detail.return_value = pd.DataFrame()
    runner.broker.calculate_round_pnl_metrics.return_value = (0.0, 0.0, 0.0)

    ensure_calls = []

    def _fake_ensure(storage_, loader_, cleaner_, client_, trade_date_, force=False):
        ensure_calls.append(
            {
                "storage": storage_,
                "loader": loader_,
                "cleaner": cleaner_,
                "client": client_,
                "trade_date": trade_date_,
            }
        )
        return True

    monkeypatch.setattr(
        "src.lazybull.paper.reporting.ensure_clean_data_for_date",
        _fake_ensure,
    )
    monkeypatch.setattr(
        "src.lazybull.paper.reporting._get_rebalance_status",
        lambda *_args, **_kwargs: "",
    )

    daily_data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "close": [10.5],
            "pre_close": [10.0],
        }
    )
    loader = MagicMock()
    loader.load_clean_daily_by_date.return_value = daily_data
    loader.build_stock_names_dict.return_value = {"000001.SZ": "测试股票1"}
    monkeypatch.setattr(
        "src.lazybull.paper.reporting.DataLoader",
        lambda *_args, **_kwargs: loader,
    )
    monkeypatch.setattr(
        "src.lazybull.paper.reporting.PaperStorage",
        lambda *_args, **_kwargs: MagicMock(load_config=lambda: None),
    )

    snapshot = load_position_snapshot("20260120", runner=runner)

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["storage"] is runner.storage
    assert ensure_calls[0]["loader"] is runner.loader
    assert ensure_calls[0]["cleaner"] is runner.cleaner
    assert ensure_calls[0]["client"] is runner.client
    assert ensure_calls[0]["trade_date"] == "20260120"
    assert snapshot.trade_date == "20260120"
    assert snapshot.total_assets == 500000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
