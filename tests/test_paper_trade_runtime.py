"""测试纸面交易共享运行时与结果展示。"""

from unittest.mock import MagicMock

import pandas as pd

from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.paper.reporting import format_trade_result
from src.lazybull.paper.runtime import (
    PaperTradeExecutionResult,
    PaperTradeRuntimeContext,
    _execute_t0_if_rebalance_day,
    _plan_pending_buy_retry_instructions,
    _execute_t1_if_pending,
    _handle_failed_buys,
    execute_trade_workflow,
)
from src.lazybull.paper.models import Fill, PendingBuy, TargetWeight, TradeInstruction


def test_execute_trade_workflow_runs_full_shared_sequence(monkeypatch):
    """共享运行时应先执行 T1，再做当日 T0 规划。"""
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
        "src.lazybull.paper.runtime._execute_t1_if_pending",
        lambda *_args, **_kwargs: call_order.append("t1")
        or [{"ts_code": "000005.SZ", "action": "buy"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._plan_next_day_retry_and_sell_instructions",
        lambda *_args, **_kwargs: call_order.append("plan") or [{"ts_code": "000004.SZ"}],
    )
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._execute_t0_if_rebalance_day",
        lambda *_args, **_kwargs: call_order.append("t0")
        or (
            [{"ts_code": "000006.SZ", "target_weight": 0.1, "reason": "补位", "score": None}],
            1.0,
            "已移除",
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

    assert call_order == ["t1", "stop_loss", "plan", "t0"]
    assert result.corrected_date == "20260120"
    assert result.stop_loss_actions == [{"ts_code": "000001.SZ"}]
    assert result.pending_sell_actions == [{"ts_code": "000004.SZ"}]
    assert result.t1_actions == [{"ts_code": "000005.SZ", "action": "buy"}]
    assert result.t0_status == "success"
    assert result.protected_stocks == ["000007.SZ"]
    assert result.stock_names == {"000001.SZ": "平安银行"}
    assert result.missing_factors == ["moneyflow_hsgt"]
    storage.save_stop_loss_state.assert_called_once()
    storage.save_last_trade_date.assert_called_once_with("20260120")


def test_format_trade_result_includes_stop_loss_and_protected_sections():
    """交易结果 Markdown 应包含止损与盈利延续保护摘要。"""
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
        pending_sell_actions=[],
        t1_actions=[],
        t0_targets=[],
        t0_instructions=[],
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

    assert "止损: 1笔" in text
    assert "盈利延续保护: 2只" in text
    assert "--- 盈利延续保护 ---" in text
    assert "1. 豫园股份(000004.SZ)" in text


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
            "buy_price": "close",
            "sell_price": "close",
        },
        trading_config=trading_config,
        runner=runner,
    )

    monkeypatch.setattr("src.lazybull.paper.runtime.StopLossMonitor", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("src.lazybull.paper.runtime._execute_t1_if_pending", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._plan_next_day_retry_and_sell_instructions",
        lambda *_a, **_k: [],
    )
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
            "buy_price": "close",
            "sell_price": "close",
        },
        trading_config=trading_config,
        runner=runner,
    )

    monkeypatch.setattr("src.lazybull.paper.runtime.StopLossMonitor", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("src.lazybull.paper.runtime._execute_t1_if_pending", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "src.lazybull.paper.runtime._plan_next_day_retry_and_sell_instructions",
        lambda *_a, **_k: [],
    )
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
    runner._build_pending_buys_from_failed_targets = (
        PaperTradingRunner._build_pending_buys_from_failed_targets.__get__(runner, PaperTradingRunner)
    )

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


def test_execute_t1_if_pending_should_defer_failed_buys_to_next_t0(monkeypatch):
    """T1 主路径的失败买单应转入下一日 T0 规划，而非同日补位执行。"""

    runner = MagicMock()
    runner.paper_storage.check_run_exists.return_value = False
    runner.paper_storage.find_pending_instructions.return_value = (
        '20260121',
        [
            TradeInstruction(
                ts_code='000001.SZ',
                action='buy',
                shares=1000,
                price_type='close',
                reason='信号生成',
                source_date='20260120',
                target_weight=0.1,
                original_signal_date='20260120',
                desired_position_count=1,
            )
        ],
    )
    runner._load_prices.return_value = ({'000002.SZ': 10.0}, {'000001.SZ': 10.0})
    runner.broker.execute_instructions.return_value = []

    failed_targets = [
        TargetWeight(
            ts_code='000001.SZ',
            target_weight=0.1,
            reason='信号生成（涨停）',
            original_signal_date='20260120',
        )
    ]
    runner.broker.get_failed_buy_targets.return_value = failed_targets

    handle_calls = []

    monkeypatch.setattr(
        'src.lazybull.paper.runtime._handle_failed_buys',
        lambda *args, **kwargs: handle_calls.append((args, kwargs)),
    )

    actions = _execute_t1_if_pending(
        runner=runner,
        trade_date='20260121',
        config={
            'buy_price': 'close',
            'sell_price': 'close',
            'universe': 'mainboard',
            'exclude_st': True,
            'min_list_days': 365,
        },
    )

    assert len(handle_calls) == 1
    assert handle_calls[0][0][1] == '20260121'
    assert handle_calls[0][0][3] == failed_targets
    assert handle_calls[0][1]['attempt_count'] == 0
    runner.broker.clear_failed_buy_targets.assert_not_called()
    runner._build_pending_buys_from_failed_targets.assert_not_called()
    runner._execute_pending_buys.assert_not_called()


def test_plan_pending_buy_retry_instructions_uses_portfolio_topn_as_slot_limit():
    """补位重试生成次日买单时，目标持仓数应沿用组合 top_n，而不是补位批次数。"""
    runner = MagicMock()
    runner.paper_storage.load_pending_buys.return_value = [
        PendingBuy(
            ts_code='000001.SZ',
            target_weight=0.05,
            reason='补位槽位-信号生成',
            create_date='20260120',
            attempts=1,
            original_signal_date='20260120',
        )
    ]
    runner.generate_replacement_targets.return_value = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.05, reason='信号生成')
    ]
    runner.loader.load_clean_daily_by_date.return_value = pd.DataFrame(
        [{'ts_code': '000001.SZ', 'close': 10.0}]
    )

    captured = {}

    def _fake_generate_instructions(**kwargs):
        captured.update(kwargs)
        return [
            TradeInstruction(
                ts_code='000001.SZ',
                action='buy',
                shares=1000,
                price_type='close',
                reason='补位槽位-信号生成',
                source_date='20260121',
                target_weight=0.05,
                desired_position_count=kwargs['desired_position_count'],
            )
        ]

    runner._generate_instructions.side_effect = _fake_generate_instructions

    instructions = _plan_pending_buy_retry_instructions(
        runner=runner,
        trade_date='20260121',
        config={
            'buy_price': 'close',
            'sell_price': 'close',
            'top_n': 20,
            'universe': 'mainboard',
        },
    )

    assert captured['desired_position_count'] == 20
    assert instructions[0].desired_position_count == 20
    assert instructions[0].retry_attempt == 1


def test_execute_t0_skips_early_rebalance_when_positions_exist():
    """非调仓日且仍有持仓时，不应提前执行 T0（仅空仓可提前调仓）。"""
    runner = MagicMock()
    runner.paper_storage.check_run_exists.return_value = False
    runner.paper_storage.load_strategy_state.return_value = {}
    runner.paper_storage.find_pending_instructions.return_value = None
    runner.paper_storage.load_pending_buys.return_value = []
    runner.account.get_positions.return_value = {"600925.SH": MagicMock()}
    runner.broker.pending_sells = []
    runner._check_rebalance_day.side_effect = RuntimeError("当前不是调仓日")

    _, _, _, status, protected = _execute_t0_if_rebalance_day(
        runner=runner,
        trade_date="20260120",
        config={
            "enable_early_rebalance_on_empty": True,
            "buy_price": "close",
            "sell_price": "open",
            "universe": "mainboard",
            "top_n": 15,
            "rebalance_freq": 5,
            "exclude_st": True,
            "min_list_days": 365,
        },
    )

    runner.run_t0.assert_not_called()
    assert protected == []
    assert status == "not_rebalance_day"


def test_execute_t0_allows_early_rebalance_when_empty():
    """非调仓日但空仓时，应允许提前执行 T0。"""
    runner = MagicMock()
    runner.paper_storage.check_run_exists.return_value = False
    runner.paper_storage.load_strategy_state.return_value = {}
    runner.paper_storage.find_pending_instructions.return_value = None
    runner.paper_storage.load_pending_buys.return_value = []
    runner.paper_storage.load_instructions.return_value = []
    runner.account.get_positions.return_value = {}
    runner.broker.pending_sells = []
    runner._check_rebalance_day.side_effect = RuntimeError("当前不是调仓日")
    runner._get_next_trade_date.return_value = "20260121"

    _, _, _, status, protected = _execute_t0_if_rebalance_day(
        runner=runner,
        trade_date="20260120",
        config={
            "enable_early_rebalance_on_empty": True,
            "buy_price": "close",
            "sell_price": "open",
            "universe": "mainboard",
            "top_n": 15,
            "rebalance_freq": 5,
            "exclude_st": True,
            "min_list_days": 365,
        },
    )

    runner.run_t0.assert_called_once()
    assert protected == []
    assert status == "no_targets"


def test_rebalance_day_executes_run_t0():
    """真实调仓日应正常执行 run_t0。"""
    runner = MagicMock()
    runner.paper_storage.check_run_exists.return_value = False
    runner.paper_storage.load_strategy_state.return_value = {}
    runner.paper_storage.find_pending_instructions.return_value = None
    runner.paper_storage.load_pending_buys.return_value = []
    runner.paper_storage.load_instructions.return_value = []
    runner._check_rebalance_day.return_value = (True, 0)  # 真实调仓日
    runner._get_next_trade_date.return_value = "20260121"

    runner.account.get_positions.return_value = {"600925.SH": MagicMock(shares=1000)}
    runner.broker.pending_sells = []

    _, _, _, status, _ = _execute_t0_if_rebalance_day(
        runner=runner,
        trade_date="20260120",
        config={
            "enable_early_rebalance_on_empty": True,
            "buy_price": "close",
            "sell_price": "open",
            "universe": "mainboard",
            "top_n": 20,
            "rebalance_freq": 20,
            "exclude_st": True,
            "min_list_days": 365,
        },
    )

    runner.run_t0.assert_called_once()
    assert status == "no_targets"
