# -*- coding: utf-8 -*-
"""paper_trade 与 bot_service 的 legacy/v2 路由测试（全 mock，无生产读写）。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bot_service, paper_trade  # noqa: E402


def test_run_main_v2_never_creates_legacy_runtime(monkeypatch) -> None:
    result = SimpleNamespace(push_text="v2完成", corrected_date="20261030")
    called = []
    monkeypatch.setattr(
        paper_trade,
        "run_v2_day",
        lambda day, model_version: called.append((day, model_version)) or result,
    )
    monkeypatch.setattr(paper_trade, "format_v2_positions", lambda *args, **kwargs: "positions")
    monkeypatch.setattr(
        paper_trade,
        "create_paper_trade_runtime",
        lambda *args: (_ for _ in ()).throw(AssertionError("不得创建 legacy runtime")),
    )
    paper_trade.run_main(
        SimpleNamespace(trade_date="20261030", engine_mode="v2", model_version=24022)
    )
    assert called == [("20261030", 24022)]


def test_run_main_default_legacy_calls_guard_and_runtime(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(paper_trade, "_guard_legacy", lambda args: called.append("guard"))
    runtime = SimpleNamespace(config={
        "buy_price": "close", "sell_price": "open", "top_n": 20,
        "rebalance_freq": 20, "horizon": 20, "stop_loss_enabled": False,
    })
    monkeypatch.setattr(paper_trade, "create_paper_trade_runtime", lambda version: runtime)
    result = SimpleNamespace(
        corrected_date="20261030",
        runner=SimpleNamespace(_get_next_trade_date=lambda day: "20261102"),
    )
    monkeypatch.setattr(paper_trade, "execute_trade_workflow", lambda day, runtime: result)
    monkeypatch.setattr(paper_trade, "print_positions", lambda day, runner: None)
    paper_trade.run_main(SimpleNamespace(trade_date="20261030", model_version=None))
    assert called == ["guard"]


def test_model_info_v2_uses_host_resolution(monkeypatch) -> None:
    monkeypatch.setattr(paper_trade, "format_v2_model_info", lambda: "v24022")
    monkeypatch.setattr(
        paper_trade,
        "format_model_info",
        lambda: (_ for _ in ()).throw(AssertionError("不得走旧展示")),
    )
    paper_trade.run_model_info(SimpleNamespace(engine_mode="v2"))


def test_adjust_cash_v2_uses_authoritative_service(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        paper_trade,
        "adjust_v2_account",
        lambda store, **kwargs: called.append(kwargs) or {"cash": 123.0},
    )
    args = SimpleNamespace(engine_mode="v2", trade_date="20261030", set=123.0)
    paper_trade.run_adjust_cash(args)
    assert called[0]["command"] == "cash"
    assert called[0]["values"] == {"set": 123.0}


def test_reset_t0_v2_preserves_account(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(paper_trade, "reset_v2_t0", lambda store, **kwargs: called.append(kwargs))
    paper_trade.run_reset_t0(SimpleNamespace(engine_mode="v2"))
    assert len(called) == 1
    assert "trade_date" in called[0]


def test_bot_execute_trade_v2_uses_paper_host(monkeypatch) -> None:
    result = SimpleNamespace(push_text="v2完成", corrected_date="20261030")
    monkeypatch.setattr(bot_service, "resolve_entry_engine", lambda: "v2")
    monkeypatch.setattr(bot_service, "run_v2_day", lambda day: result)
    monkeypatch.setattr(
        bot_service,
        "execute_trade_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不得走 legacy")),
    )
    assert bot_service.execute_trade("20261030") == ("v2完成", "20261030")


def test_bot_execute_trade_default_legacy_keeps_formatter(monkeypatch) -> None:
    monkeypatch.setattr(bot_service, "resolve_entry_engine", lambda: "legacy")
    monkeypatch.setattr(bot_service, "assert_legacy_rollback_allowed", lambda store: None)
    result = SimpleNamespace(corrected_date="20261030")
    monkeypatch.setattr(bot_service, "execute_trade_workflow", lambda day, progress_callback: result)
    monkeypatch.setattr(bot_service, "format_trade_result", lambda value: "legacy完成")
    assert bot_service.execute_trade("20261030") == ("legacy完成", "20261030")


def test_cli_v2_override_cannot_bypass_production_switch(monkeypatch) -> None:
    monkeypatch.setattr(paper_trade, "resolve_entry_engine", lambda cli_override: "v2")
    monkeypatch.setattr(
        paper_trade,
        "PaperStorage",
        lambda: type("Storage", (), {"load_config": lambda self: {"engine": "legacy"}})(),
    )
    args = SimpleNamespace(command="run", engine="v2")
    with pytest.raises(RuntimeError, match="绕过门禁"):
        paper_trade._resolve_cli_engine(args)


def test_cli_v2_override_allowed_after_formal_switch(monkeypatch) -> None:
    monkeypatch.setattr(paper_trade, "resolve_entry_engine", lambda cli_override: "v2")
    monkeypatch.setattr(
        paper_trade,
        "PaperStorage",
        lambda: type("Storage", (), {"load_config": lambda self: {"engine": "v2"}})(),
    )
    args = SimpleNamespace(command="run", engine="v2")
    assert paper_trade._resolve_cli_engine(args) == "v2"
