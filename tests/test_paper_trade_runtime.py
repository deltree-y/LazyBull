"""测试纸面交易共享运行时与结果展示。"""

from unittest.mock import MagicMock

import pandas as pd

from src.lazybull.paper.reporting import format_trade_result
from src.lazybull.paper.runtime import (
    PaperTradeExecutionResult,
    PaperTradeRuntimeContext,
    _handle_failed_buys,
    execute_trade_workflow,
)
from src.lazybull.paper.models import TargetWeight


def test_execute_trade_workflow_runs_full_shared_sequence(monkeypatch):
    """共享运行时应完整串起止损、提前换出、止盈、T1、T0。"""
    runner = MagicMock()
    runner._correct_trade_date.return_value = "20260120"
    runner._get_next_trade_date.return_value = "20260121"
    runner.missing_factors = ["moneyflow_hsgt"]
    runner.paper_storage.load_instructions.return_value = []

    storage = MagicMock()
    storage.load_account_state.return_value = None
    storage.load_stop_loss_state.return_value = None

    trading_config = MagicMock()
    trading_config.create_stop_loss_config.return_value = None

    context = PaperTradeRuntimeContext(
        storage=storage,
        config={
            "stop_loss_enabled": True,
            "enable_profit_based_holding": True,
            "early_exit_mode": "strength_veto",
            "buy_price": "close",
            "sell_price": "close",
        },
        trading_config=trading_config,
        runner=runner,
    )

    monitor = MagicMock()
    call_order = []

    monkeypatch.setattr(
        "src.lazybull.paper.runtime.StopLossMonitor",
        lambda *_args, **_kwargs: monitor,
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._check_stop_loss",
        lambda *_args, **_kwargs: call_order.append("stop_loss") or [{"ts_code": "000001.SZ"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._check_early_exit",
        lambda *_args, **_kwargs: call_order.append("early_exit") or [{"ts_code": "000002.SZ"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._check_take_profit",
        lambda *_args, **_kwargs: call_order.append("take_profit") or [{"ts_code": "000003.SZ"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._process_pending_sells",
        lambda *_args, **_kwargs: call_order.append("pending_sells") or [{"ts_code": "000004.SZ"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t1_if_pending",
        lambda *_args, **_kwargs: call_order.append("t1")
        or [{"ts_code": "000005.SZ", "action": "buy"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t0_if_rebalance_day",
        lambda *_args, **_kwargs: call_order.append("t0")
        or (
            [{"ts_code": "000006.SZ", "target_weight": 0.1, "reason": "补位", "score": None}],
            0.8,
            "ECT测试",
            "success",
            ["000007.SZ"],
        ),
    )

    loader = MagicMock()
    loader.build_stock_names_dict.return_value = {"000001.SZ": "平安银行"}
    monkeypatch.setattr(
        "src.lazybull.paper.runtime.DataLoader",
        lambda *_args, **_kwargs: loader,
    )

    result = execute_trade_workflow("20260120", runtime=context)

    assert call_order == ["stop_loss", "early_exit", "take_profit", "pending_sells", "t1", "t0"]
    assert result.corrected_date == "20260120"
    assert result.stop_loss_actions == [{"ts_code": "000001.SZ"}]
    assert result.early_exit_actions == [{"ts_code": "000002.SZ"}]
    assert result.take_profit_actions == [{"ts_code": "000003.SZ"}]
    assert result.pending_sell_actions == [{"ts_code": "000004.SZ"}]
    assert result.t1_actions == [{"ts_code": "000005.SZ", "action": "buy"}]
    assert result.t0_status == "success"
    assert result.protected_stocks == ["000007.SZ"]
    assert result.stock_names == {"000001.SZ": "平安银行"}
    assert result.missing_factors == ["moneyflow_hsgt"]
    storage.save_stop_loss_state.assert_called_once()
    storage.save_last_trade_date.assert_called_once_with("20260120")


def test_execute_trade_workflow_still_checks_early_exit_in_disabled_mode(monkeypatch):
    """early_exit_mode=disabled 仍应执行基础亏损提前换出（原硬卖）。"""
    runner = MagicMock()
    runner._correct_trade_date.return_value = "20260120"
    runner._get_next_trade_date.return_value = None
    runner.missing_factors = []
    runner.paper_storage.load_instructions.return_value = []

    storage = MagicMock()
    storage.load_account_state.return_value = None
    storage.load_stop_loss_state.return_value = None

    trading_config = MagicMock()
    trading_config.create_stop_loss_config.return_value = None

    context = PaperTradeRuntimeContext(
        storage=storage,
        config={
            "stop_loss_enabled": False,
            "enable_profit_based_holding": True,
            "early_exit_mode": "disabled",
            "take_profit_threshold": None,
            "buy_price": "close",
            "sell_price": "close",
        },
        trading_config=trading_config,
        runner=runner,
    )

    call_order = []

    monkeypatch.setattr(
        "src.lazybull.paper.runtime.StopLossMonitor",
        lambda *_args, **_kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._check_early_exit",
        lambda *_args, **_kwargs: call_order.append("early_exit")
        or [{"ts_code": "000002.SZ"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._check_take_profit",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._process_pending_sells",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t1_if_pending",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t0_if_rebalance_day",
        lambda *_args, **_kwargs: ([], 1.0, "ECT 未启用", "not_rebalance_day", []),
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime.DataLoader",
        lambda *_args, **_kwargs: MagicMock(build_stock_names_dict=lambda: {}),
    )

    result = execute_trade_workflow("20260120", runtime=context)

    assert call_order == ["early_exit"]
    assert result.early_exit_actions == [{"ts_code": "000002.SZ"}]


def test_format_trade_result_includes_profit_management_sections():
    """交易结果 Markdown 应包含新增的提前换出与整体止盈摘要。"""
    runner = MagicMock()
    runner.account.get_positions.return_value = {}
    runner.account.get_cash.return_value = 12345.0

    result = PaperTradeExecutionResult(
        requested_date="20260120",
        corrected_date="20260120",
        storage=MagicMock(),
        config={},
        trading_config=MagicMock(),
        runner=runner,
        stop_loss_actions=[
            {"ts_code": "000001.SZ", "shares": 100, "reason": "回撤止损", "can_execute": True}
        ],
        early_exit_actions=[
            {"ts_code": "000002.SZ", "shares": 100, "reason": "亏损提前换出", "can_execute": True}
        ],
        take_profit_actions=[
            {"ts_code": "000003.SZ", "shares": 100, "reason": "整体止盈", "can_execute": True}
        ],
        pending_sell_actions=[],
        t1_actions=[],
        t0_targets=[],
        t0_instructions=[],
        ect_exposure=0.7,
        ect_reason="ECT 回撤保护",
        t0_status="not_rebalance_day",
        protected_stocks=["000004.SZ", "000005.SZ"],
        stock_names={
            "000001.SZ": "平安银行",
            "000002.SZ": "万科A",
            "000003.SZ": "招商银行",
            "000004.SZ": "豫园股份",
            "000005.SZ": "中国软件",
        },
    )

    text = format_trade_result(result)

    assert "提前换出: 1笔" in text
    assert "整体止盈: 1笔" in text
    assert "盈利延续保护: 2只" in text
    assert "--- 亏损提前换出 ---" in text
    assert "--- 整体止盈 ---" in text
    assert "--- 盈利延续保护 ---" in text
    assert "1. 豫园股份(000004.SZ)" in text
    assert "ECT系数: 0.70 (ECT 回撤保护)" in text


def test_execute_trade_workflow_skip_save_ranked_candidates_when_no_t0(monkeypatch):
    """非调仓日不应覆盖持久化 ranked_candidates。"""
    runner = MagicMock()
    runner._correct_trade_date.return_value = "20260120"
    runner._get_next_trade_date.return_value = None
    runner.missing_factors = []
    runner.paper_storage.load_instructions.return_value = []
    runner.signal = MagicMock()
    runner.signal._last_ranked_candidates = [("000001.SZ", 0.1)]
    runner.signal._last_signal_date = pd.Timestamp("20260119")

    storage = MagicMock()
    storage.load_account_state.return_value = None
    storage.load_stop_loss_state.return_value = None
    storage.load_ranked_candidates.return_value = None

    trading_config = MagicMock()
    trading_config.create_stop_loss_config.return_value = None

    context = PaperTradeRuntimeContext(
        storage=storage,
        config={
            "stop_loss_enabled": False,
            "enable_profit_based_holding": True,
            "buy_price": "close",
            "sell_price": "close",
        },
        trading_config=trading_config,
        runner=runner,
    )

    monkeypatch.setattr("src.lazybull.paper.runtime.StopLossMonitor", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("src.lazybull.paper.runtime._check_early_exit", lambda *_a, **_k: [])
    monkeypatch.setattr("src.lazybull.paper.runtime._check_take_profit", lambda *_a, **_k: [])
    monkeypatch.setattr("src.lazybull.paper.runtime._process_pending_sells", lambda *_a, **_k: [])
    monkeypatch.setattr("src.lazybull.paper.runtime._execute_t1_if_pending", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t0_if_rebalance_day",
        lambda *_a, **_k: ([], 1.0, "ECT 未启用", "not_rebalance_day", []),
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime.DataLoader",
        lambda *_a, **_k: MagicMock(build_stock_names_dict=lambda: {}),
    )

    execute_trade_workflow("20260120", runtime=context)
    storage.save_ranked_candidates.assert_not_called()


def test_execute_trade_workflow_save_ranked_candidates_on_t0_success(monkeypatch):
    """调仓日 T0 成功且信号日期匹配时应保存 ranked_candidates。"""
    runner = MagicMock()
    runner._correct_trade_date.return_value = "20260120"
    runner._get_next_trade_date.return_value = None
    runner.missing_factors = []
    runner.paper_storage.load_instructions.return_value = []
    runner.signal = MagicMock()
    runner.signal._last_ranked_candidates = [("000001.SZ", 0.1), ("000002.SZ", 0.2)]
    runner.signal._last_signal_date = pd.Timestamp("20260120")

    storage = MagicMock()
    storage.load_account_state.return_value = None
    storage.load_stop_loss_state.return_value = None
    storage.load_ranked_candidates.return_value = None

    trading_config = MagicMock()
    trading_config.create_stop_loss_config.return_value = None

    context = PaperTradeRuntimeContext(
        storage=storage,
        config={
            "stop_loss_enabled": False,
            "enable_profit_based_holding": True,
            "buy_price": "close",
            "sell_price": "close",
        },
        trading_config=trading_config,
        runner=runner,
    )

    monkeypatch.setattr("src.lazybull.paper.runtime.StopLossMonitor", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("src.lazybull.paper.runtime._check_early_exit", lambda *_a, **_k: [])
    monkeypatch.setattr("src.lazybull.paper.runtime._check_take_profit", lambda *_a, **_k: [])
    monkeypatch.setattr("src.lazybull.paper.runtime._process_pending_sells", lambda *_a, **_k: [])
    monkeypatch.setattr("src.lazybull.paper.runtime._execute_t1_if_pending", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t0_if_rebalance_day",
        lambda *_a, **_k: ([{"ts_code": "000001.SZ", "target_weight": 0.1, "reason": "r", "score": None}], 1.0, "ECT 未启用", "success", []),
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime.DataLoader",
        lambda *_a, **_k: MagicMock(build_stock_names_dict=lambda: {}),
    )

    execute_trade_workflow("20260120", runtime=context)
    storage.save_ranked_candidates.assert_called_once_with(
        [("000001.SZ", 0.1), ("000002.SZ", 0.2)],
        "20260120",
    )


def test_handle_failed_buys_preserves_original_slot_weight():
    """买入失败转补位时应保留原始槽位权重，供下一交易日按槽位补齐。"""
    runner = MagicMock()
    runner._get_next_trade_date.return_value = "20260122"

    failed = [
        TargetWeight(ts_code="000001.SZ", target_weight=0.07, reason="涨停"),
        TargetWeight(ts_code="000002.SZ", target_weight=0.03, reason="停牌"),
    ]

    _handle_failed_buys(
        runner=runner,
        trade_date="20260121",
        config={
            "buy_price": "close",
            "universe": "mainboard",
        },
        failed_buy_targets=failed,
        attempt_count=0,
    )

    runner.paper_storage.save_pending_buys.assert_called_once()
    pending_buys = runner.paper_storage.save_pending_buys.call_args[0][0]
    assert len(pending_buys) == 2
    assert pending_buys[0].ts_code == "000001.SZ"
    assert pending_buys[1].ts_code == "000002.SZ"
    assert pending_buys[0].target_weight == 0.07
    assert pending_buys[1].target_weight == 0.03
    runner.broker.clear_failed_buy_targets.assert_called_once()
