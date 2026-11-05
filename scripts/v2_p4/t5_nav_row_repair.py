# -*- coding: utf-8 -*-
"""v2 P4 T5 权威 NAV 末日行估值口径修复（默认 dry-run，阶段专用）。

背景（v0.209.9 登记）：0.209.9 之前 ``t5_migration`` 的默认估值来源是「最新 clean
截面」，迁移把**行情日**（如 20261029）的收盘价写进了**业务日**（如 20260603）的
``portfolio_values`` 行；``nav.parquet`` 同日行仍是业务日收盘价，于是两个载体永久
相差一整段价格变动（现网 20260603：98,754.00 = 20 只持仓 6/3→10/29 的市值缩水）。

本工具只做一件事：把权威快照中**业务日那一行**的三个估值数值改回该业务日收盘复算值
（= ``nav.parquet`` 同日行），快照其余字节全部不动。三重独立校验全部通过才允许落盘：

1. **当日分区复算**：``clean/daily/<业务日>.parquet`` 的 close（→ pre_close → 买入价
   回退链）逐股算市值；
2. **nav.parquet 交叉**：同日行 cash / position_value / total_value 与复算逐字段
   1e-6 内一致（nav 是独立载体，不是同一计算的重算）；
3. **操作者显式期望值**：``--expect-total-value`` / ``--expect-market-value`` 必须与
   复算一致（防手滑；授权文本由命令行给出即视为显式授权）。

只修末日行（业务日必须 = 权威末日），不改日期、不改持仓/现金/账本，不碰
``nav.parquet``。``--apply`` 前先把原始字节备份进证据目录（可整字节回滚）。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.core.accounting.engine_state import (  # noqa: E402
    dumps_blob,
    loads_blob,
    state_from_blob,
)
from src.lazybull.v2.hosts.paper.nav_valuation import (  # noqa: E402
    NAV_VALUE_TOLERANCE,
    nav_projection_divergence,
    same_value,
)
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402

BACKUP_FILENAME = "engine_state.before.bin"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _daily_section(data_root: Path, day: str) -> Path:
    return data_root / "clean" / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"


def _mark_price(current: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """单只收盘价（close → pre_close；与门 1 D10 回退链同序）。"""
    for column in ("close", "pre_close"):
        value = pd.to_numeric(current.get(column), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value), f"clean.daily.{column}"
    return None, "unavailable"


def recompute_business_day(
    positions: Mapping[str, Mapping[str, Any]],
    data_root: Path,
    business_day: str,
) -> Dict[str, Any]:
    """按业务日收盘价逐股复算持仓市值（分区缺失 / 价格不可得 ⇒ 就地记录，不猜）。"""
    path = _daily_section(data_root, business_day)
    if not path.is_file():
        raise FileNotFoundError(
            f"业务日 clean/daily 分区缺失: {business_day}（{path}）；"
            "无该日截面即无法复算业务日收盘估值，禁止用其他日期替代"
        )
    frame = pd.read_parquet(path)
    market = frame.set_index("ts_code", drop=False)
    rows: List[Dict[str, Any]] = []
    market_value = 0.0
    missing: List[str] = []
    for code, position in sorted(positions.items()):
        shares = int(position["shares"])
        price: Optional[float] = None
        source = "unavailable"
        if code in market.index:
            current = market.loc[code]
            if isinstance(current, pd.DataFrame):
                raise ValueError(f"clean/daily 同日股票键重复: {business_day} {code}")
            price, source = _mark_price(current)
        if price is None:
            missing.append(code)
            continue
        value = shares * price
        market_value += value
        rows.append(
            {
                "ts_code": code,
                "shares": shares,
                "mark_price": price,
                "mark_source": source,
                "market_value": value,
            }
        )
    if missing:
        raise ValueError(
            f"业务日 {business_day} 有持仓现价不可得，拒绝复算: {missing}；"
            "缺价时一律不得以买入价/最新价兜底写入权威净值行"
        )
    return {
        "business_day": business_day,
        "market_date": business_day,
        "market_value": market_value,
        "positions": rows,
    }


def _nav_row_of(nav: pd.DataFrame, business_day: str) -> Optional[Dict[str, Any]]:
    days = nav["trade_date"].astype(str).str.replace("-", "", regex=False)
    hit = nav.loc[days == business_day]
    if hit.empty:
        return None
    if len(hit) > 1:
        raise ValueError(f"nav.parquet 业务日行重复: {business_day}（{len(hit)} 行）")
    return hit.iloc[0].to_dict()


def build_repair_plan(
    *,
    paper_root: Path,
    data_root: Path,
    expect_total_value: Optional[float] = None,
    expect_market_value: Optional[float] = None,
) -> Dict[str, Any]:
    """只读规划：复算业务日估值并与权威行 / nav 行双向核对（不写盘）。"""
    store = PaperStateStore(root_path=str(paper_root), create_layout=False)
    raw = store.load_engine_state_blob()
    if raw is None:
        raise RuntimeError(f"无权威快照，无法修复: {store.state_path / 'engine_state.bin'}")
    state = state_from_blob(loads_blob(raw))
    if not state.portfolio_values:
        raise RuntimeError("权威快照 portfolio_values 为空，无末日行可修")
    business_day = pd.Timestamp(state.portfolio_values[-1]["date"]).strftime("%Y%m%d")
    recomputed = recompute_business_day(state.positions, data_root, business_day)
    cash = float(state.current_capital)
    market_value = float(recomputed["market_value"])
    total_value = cash + market_value

    nav = store.load_all_nav()
    if nav is None or nav.empty:
        raise RuntimeError("nav.parquet 缺失/为空，缺独立参照载体（禁止单侧自证），拒绝修复")
    nav_row = _nav_row_of(nav, business_day)
    if nav_row is None:
        raise RuntimeError(
            f"nav.parquet 无业务日 {business_day} 行，缺独立参照载体（禁止单侧自证），拒绝修复"
        )
    for label, left, right in (
        ("cash", nav_row.get("cash"), cash),
        ("position_value", nav_row.get("position_value"), market_value),
        ("total_value", nav_row.get("total_value"), total_value),
    ):
        if not same_value(left, right):
            raise RuntimeError(
                f"业务日 {business_day} 复算与 nav.parquet 同日行不一致（{label}）: "
                f"复算={right!r} nav={left!r}；两载体都不自洽时禁止择一覆盖，请先人工核查"
            )
    for name, expected, actual in (
        ("total_value", expect_total_value, total_value),
        ("market_value", expect_market_value, market_value),
    ):
        if expected is not None and abs(float(expected) - actual) > NAV_VALUE_TOLERANCE:
            raise RuntimeError(
                f"操作者声明的期望 {name}={expected} 与当日分区复算 {actual} 不一致；"
                "拒绝执行（期望值须来自独立复算，不得凭记忆填写）"
            )

    current = dict(state.portfolio_values[-1])
    return {
        "paper_root": str(paper_root),
        "business_day": business_day,
        "blob_sha256_before": _sha256(raw),
        "row_index": len(state.portfolio_values) - 1,
        "row_before": current,
        "row_after": {
            **{
                k: v
                for k, v in current.items()
                if k not in ("portfolio_value", "capital", "market_value")
            },
            "portfolio_value": total_value,
            "capital": cash,
            "market_value": market_value,
        },
        "recomputed": recomputed,
        "nav_row": nav_row,
        "already_ok": (
            same_value(current.get("portfolio_value"), total_value)
            and same_value(current.get("capital"), cash)
            and same_value(current.get("market_value"), market_value)
        ),
        "nav_divergence_before": nav_projection_divergence(state.portfolio_values, nav_row),
        "_raw": raw,
    }


def apply_repair(plan: Mapping[str, Any], *, evidence_dir: Path) -> Dict[str, Any]:
    """落盘修复：只改末日行三数值，备份原字节，事后跑双状态 + 第三腿校验。"""
    paper_root = Path(plan["paper_root"])
    store = PaperStateStore(root_path=str(paper_root), create_layout=False)
    raw = plan["_raw"]
    blob = loads_blob(raw)
    state_from_blob(blob)
    portfolio_values = list(blob["fields"]["portfolio_values"])
    index = int(plan["row_index"])
    if index != len(portfolio_values) - 1:
        raise RuntimeError(
            f"权威末日行下标漂移: 规划={index} 当前={len(portfolio_values) - 1}"
            "（规划后快照被改写），拒绝落盘"
        )
    portfolio_values[index] = dict(plan["row_after"])
    patched = dict(blob)
    patched["fields"] = {**blob["fields"], "portfolio_values": portfolio_values}
    new_raw = dumps_blob(patched)
    # 往返自检：除该行三数值外整 blob 必须逐键相等
    if loads_blob(new_raw) != patched:
        raise RuntimeError("修复后快照往返不一致，拒绝落盘")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    backup = evidence_dir / BACKUP_FILENAME
    backup.write_bytes(raw)
    written: List[str] = []
    with store.write_lock():
        current_raw = store.load_engine_state_blob()
        if current_raw is None or _sha256(current_raw) != plan["blob_sha256_before"]:
            raise RuntimeError(
                "快照在规划与落盘之间被改写（sha256 比较交换失败），拒绝落盘；请重新规划"
            )
        store._atomic_write_bytes(store.state_path / "engine_state.bin", new_raw)
        written.append("state/engine_state.bin")

    consistency = store.check_consistency()
    state_after = store.load_engine_state()
    nav = store.load_all_nav()
    divergence_after = (
        nav_projection_divergence(state_after.portfolio_values, nav.iloc[-1].to_dict())
        if state_after is not None and nav is not None and not nav.empty
        else None
    )
    # ``repaired`` = 投影缺失/滞后，已按修复后的权威重建（受控且幂等），同样合格；
    # 冲突形态会直接抛 PaperStateError，不再以返回码表达。
    if consistency not in ("ok", "repaired") or divergence_after is not None:
        raise RuntimeError(
            f"修复后校验未通过: check_consistency={consistency!r} "
            f"nav_divergence={divergence_after!r}；原始字节已备份于 {backup}，可整字节回滚"
        )
    return {
        "written": written,
        "blob_sha256_before": plan["blob_sha256_before"],
        "blob_sha256_after": _sha256(new_raw),
        "backup": str(backup),
        "backup_sha256": _sha256(raw),
        "check_consistency_after": consistency,
        "nav_divergence_after": divergence_after,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", default=str(ROOT / "data" / "paper"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--expect-total-value", type=float, default=None)
    parser.add_argument("--expect-market-value", type=float, default=None)
    parser.add_argument("--apply", action="store_true", help="落盘（缺省只做只读规划）")
    args = parser.parse_args(argv)

    plan = build_repair_plan(
        paper_root=Path(args.paper_root),
        data_root=Path(args.data_root),
        expect_total_value=args.expect_total_value,
        expect_market_value=args.expect_market_value,
    )
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    public_plan = {k: v for k, v in plan.items() if not k.startswith("_")}
    if plan["already_ok"]:
        _dump(evidence / "nav_row_repair.json", {"verdict": "ALREADY_OK", **public_plan})
        print(json.dumps({"verdict": "ALREADY_OK", "business_day": plan["business_day"]}))
        return 0
    if not args.apply:
        _dump(evidence / "nav_row_repair_dryrun.json", {"verdict": "DRY_RUN", **public_plan})
        print(
            json.dumps(
                {
                    "verdict": "DRY_RUN",
                    "business_day": plan["business_day"],
                    "row_before": plan["row_before"],
                    "row_after": plan["row_after"],
                    "nav_divergence_before": plan["nav_divergence_before"],
                },
                ensure_ascii=False,
                default=str,
            )
        )
        return 0
    applied = apply_repair(plan, evidence_dir=evidence)
    _dump(
        evidence / "nav_row_repair.json",
        {"verdict": "PASS", "captured_at": _now(), **public_plan, "applied": applied},
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "business_day": plan["business_day"],
                "row_after": plan["row_after"],
                "applied": applied,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
