# -*- coding: utf-8 -*-
"""CLI/Bot 共用的 engine 路由、v2 只读展示与 legacy 回退防呆。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.lazybull.paper.models import AccountState, Position
from src.lazybull.v2.core.accounting.engine_state import loads_blob
from src.lazybull.v2.hosts.paper.config_adapter import (
    ENGINE_LEGACY,
    ENGINE_V2,
    PaperHostError,
    load_paper_config,
    resolve_engine,
)
from src.lazybull.v2.hosts.paper.factory import create_paper_host, format_host_model_info
from src.lazybull.v2.hosts.paper.host import PaperDayResult
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore


def resolve_entry_engine(
    paper_root: Optional[Path] = None, *, cli_override: Optional[str] = None
) -> str:
    """所有入口同一解析规则：配置缺省 legacy，CLI 覆盖后仍经闭集校验。"""
    flat = load_paper_config(paper_root)
    if cli_override is not None:
        flat["engine"] = cli_override
    return resolve_engine(flat)


def assert_production_switch_allowed(evidence_dir: Path, *, authorized: bool) -> None:
    """真实 legacy→v2 切换门：门 1/5/6/7/8 全过、技术就绪且用户当时显式授权。"""
    if not authorized:
        raise PaperHostError("真实切换需要用户当时显式授权；隔离影子请使用 T5 驱动")
    status_path = evidence_dir / "closeout_status.json"
    if not status_path.is_file():
        raise PaperHostError(f"真实切换缺收口证据: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    gates = status.get("gates") or {}
    failed = [
        name
        for name in ("gate1", "gate5", "gate6", "gate7", "gate8")
        if gates.get(name) != "PASS"
    ]
    if failed:
        raise PaperHostError(f"真实切换门禁未过: {failed}")
    if status.get("verdict") != "READY_FOR_FINAL_AUTHORIZATION":
        raise PaperHostError(f"真实切换收口结论未就绪: {status.get('verdict')!r}")
    statuses = status.get("statuses") or {}
    if statuses.get("technical_conditions_ready") is not True:
        raise PaperHostError("真实切换技术条件未就绪")
    if status.get("blocked_reasons"):
        raise PaperHostError(f"真实切换仍有阻塞项: {status['blocked_reasons']}")


def run_v2_day(
    trade_date: str,
    *,
    paper_root: Optional[Path] = None,
    data_root: Optional[str] = None,
    model_version: Optional[int] = None,
) -> PaperDayResult:
    """run/bot trade 的 v2 入口；未就绪/未真实驱动不得伪装成功。"""
    flat = load_paper_config(paper_root) if model_version is not None else None
    if flat is not None:
        flat["model_version"] = model_version
    host = create_paper_host(store=paper_root, config_flat=flat, data_root=data_root)
    host.run_t1(trade_date)
    result = host.last_day_result
    if result is None or not result.driven:
        gaps = [] if result is None or result.readiness is None else result.readiness.gaps
        raise PaperHostError(f"v2 纸面日未真实驱动: {trade_date} readiness={gaps}")
    return result


def format_v2_model_info(
    *, paper_root: Optional[Path] = None, data_root: Optional[str] = None
) -> str:
    """model/bot model 与信号实际加载同源的展示路径。"""
    host = create_paper_host(store=paper_root, data_root=data_root)
    return format_host_model_info(host)


def load_v2_account_state(store: PaperStateStore) -> AccountState:
    """v2 account.json 投影 → legacy 展示值对象；只读，不触发旧 runner/ensure。"""
    raw = store.load_account()
    if raw is None:
        raise PaperHostError(f"v2 账户投影缺失: {store.root_path}")
    positions = {
        code: Position(
            ts_code=code,
            shares=int(position["shares"]),
            buy_price=float(position["buy_price"]),
            buy_cost=float(position["buy_cost"]),
            buy_date=str(position["buy_date"]),
            buy_pnl_price=float(position.get("buy_pnl_price") or 0.0),
            buy_atr_pct=float(position.get("buy_atr_pct") or 0.0),
            status=str(position.get("status") or "持有"),
            notes=str(position.get("notes") or ""),
        )
        for code, position in (raw.get("positions") or {}).items()
    }
    return AccountState(
        cash=float(raw["cash"]), positions=positions, last_update=str(raw["last_update"])
    )


def _load_mark_prices(data_root: Path, trade_date: str) -> Dict[str, float]:
    day = trade_date.replace("-", "")
    path = data_root / "clean" / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
    if not path.is_file():
        raise PaperHostError(f"v2 持仓展示行情缺失: {path}")
    frame = pd.read_parquet(path, columns=["ts_code", "close", "pre_close"])
    prices: Dict[str, float] = {}
    for record in frame.to_dict("records"):
        for column in ("close", "pre_close"):
            value = pd.to_numeric(record.get(column), errors="coerce")
            if pd.notna(value) and float(value) > 0:
                prices[str(record["ts_code"])] = float(value)
                break
    return prices


def format_v2_positions(
    trade_date: str, *, store: PaperStateStore, data_root: Optional[Path] = None
) -> str:
    """v2 positions/bot positions 只读展示；不使用 legacy runner，不触发下载。"""
    account = load_v2_account_state(store)
    root = data_root or store.root_path.parent
    prices = _load_mark_prices(root, trade_date)
    total = account.get_total_value(prices)
    lines = [
        f"持仓概览（{trade_date}，v2 只读）",
        f"现金: {account.cash:,.2f}  总资产: {total:,.2f}  持仓数: {len(account.positions)}",
    ]
    for code, position in sorted(account.positions.items()):
        price = prices.get(code, position.buy_price)
        value = position.shares * price
        weight = value / total if total > 0 else 0.0
        lines.append(
            f"{code}  {position.shares}股  现价{price:.2f}  市值{value:,.2f}  权重{weight:.2%}"
        )
    return "\n".join(lines)


def assert_legacy_rollback_allowed(store: PaperStateStore) -> None:
    """存在 v2 快照时，legacy 只接受经双向校验的回滚清单，禁止陈旧账户直接接回。"""
    raw = store.load_engine_state_blob()
    if raw is None:
        return
    path = store.state_path / "rollback_manifest.json"
    if not path.is_file():
        raise PaperHostError("检测到 v2 权威快照但无 D11 回滚清单，拒绝 legacy 直接接回")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current_hash = hashlib.sha256(raw).hexdigest()
    account_path = store.state_path / "account.json"
    account_hash = hashlib.sha256(account_path.read_bytes()).hexdigest() if account_path.is_file() else None
    if manifest.get("validated") is not True or manifest.get("engine_state_sha256") != current_hash:
        raise PaperHostError("D11 回滚清单缺失/身份不符，拒绝 legacy 接回陈旧状态")
    account = json.loads(account_path.read_text(encoding="utf-8")) if account_path.is_file() else {}
    rollback_date = str(manifest.get("rollback_date") or "")
    current_date = str(account.get("last_update") or "")
    if current_date == rollback_date and manifest.get("account_sha256") == account_hash:
        return
    continued = current_date > rollback_date and any(
        store.check_run_exists(kind, current_date) for kind in ("t0", "t1")
    )
    if not continued:
        raise PaperHostError("D11 回滚账户哈希不符，且后续日期缺 legacy 运行标记")


def assert_complete_generation(store: PaperStateStore) -> None:
    """远端/树莓派只读完整代守卫；legacy 无 v2 快照时保持原读取行为。"""
    raw = store.load_engine_state_blob()
    if raw is None:
        return
    generation = loads_blob(raw).get("paper_generation") or {}
    day = generation.get("trade_date")
    kind = generation.get("run_kind")
    if kind not in ("t0", "t1") or not day or not store.check_run_exists(kind, str(day)):
        raise PaperHostError(
            f"v2 半提交代禁止只读发布: trade_date={day!r} run_kind={kind!r}"
        )


def assert_complete_generation_at(paper_root: Path, smb_reader: Any = None) -> None:
    """树莓派调用便利层：只读装配，不创建目录。"""
    store = PaperStateStore(
        root_path=str(paper_root), smb_reader=smb_reader, create_layout=False
    )
    assert_complete_generation(store)
