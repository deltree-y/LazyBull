# -*- coding: utf-8 -*-
"""v2 P4 T5 门 1 身份采集与 Q11-A 连续恢复校验（阶段专用）。

只读生产数据、模型与纸面账户；证据仅写显式 ``--evidence-dir``。没有连续 10 个
真实交易日的恢复台账时只产出 ``BLOCKED_Q11A`` 预检，绝不把陈旧账户或历史回放
冻结成三臂生产轨迹。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t1_worktree_digest import compute_worktree_digest  # noqa: E402
from scripts.v2_p4.t4_gate_runner import _code_state, capture_data_state  # noqa: E402
from scripts.v2_p4.t5_daily_evidence import load_cumulative_ledger  # noqa: E402
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import (  # noqa: E402
    PAPER_CONFIG_TO_V2,
    load_paper_config,
    resolve_engine,
)

LABELS = ("y_ret_5", "y_ret_10", "y_ret_20")
PACKAGES = ("numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "tushare")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是对象: {path}")
    return payload


def collect_model_identity(models_root: Path, version: int) -> Dict[str, Any]:
    """现役模型内容与旁路身份；只读，不触发 registry latest 写副作用。"""
    artifacts: Dict[str, Any] = {}
    for suffix in ("model.joblib", "metadata.json", "features.json"):
        path = models_root / f"v{version}_{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"现役模型产物缺失: {path}")
        artifacts[suffix] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    metadata = _read_json(models_root / f"v{version}_metadata.json")
    return {
        "version": int(version),
        "artifacts": artifacts,
        "content_fingerprint": fingerprint(artifacts),
        "model_type": metadata.get("model_type"),
        "train_start_date": metadata.get("train_start_date"),
        "train_end_date": metadata.get("train_end_date"),
        "feature_count": len(json.loads((models_root / f"v{version}_features.json").read_text())),
    }


def collect_execution_identity(flat: Mapping[str, Any]) -> Dict[str, Any]:
    """冻结有效执行配置；显式保留 buy/sell price 与 engine 解析结果。"""
    config = {key: flat[key] for key in PAPER_CONFIG_TO_V2 if key in flat}
    config["engine"] = resolve_engine(flat)
    config["exposure_policy"] = flat.get("exposure_policy")
    config["exposure_replenish"] = flat.get("exposure_replenish")
    return {"config": config, "fingerprint": fingerprint(config)}


def _environment() -> Dict[str, Any]:
    versions: Dict[str, Optional[str]] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _latest_open_day(data_root: Path) -> str:
    path = data_root / "clean" / "daily"
    days = sorted(p.stem.replace("-", "") for p in path.glob("*.parquet"))
    if not days:
        raise RuntimeError(f"clean/daily 无分区: {path}")
    return days[-1]


def assess_holdings(account: Mapping[str, Any], data_root: Path) -> Dict[str, Any]:
    """按最新 clean 截面复核持仓有效性与 D10 原位估值回退，不改账户。"""
    day = _latest_open_day(data_root)
    path = data_root / "clean" / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
    frame = pd.read_parquet(path)
    if "ts_code" not in frame.columns:
        raise ValueError(f"持仓有效性截面缺 ts_code: {path}")
    market = frame.set_index("ts_code", drop=False)
    rows: List[Dict[str, Any]] = []
    total_market_value = 0.0
    for ts_code, raw in sorted((account.get("positions") or {}).items()):
        position = dict(raw)
        shares = int(position["shares"])
        record: Dict[str, Any] = {
            "ts_code": ts_code,
            "shares": shares,
            "buy_date": position.get("buy_date"),
            "market_date": day,
            "in_current_section": ts_code in market.index,
        }
        price: Optional[float] = None
        source = "unavailable"
        if record["in_current_section"]:
            current = market.loc[ts_code]
            if isinstance(current, pd.DataFrame):
                raise ValueError(f"clean/daily 同日股票键重复: {day} {ts_code}")
            for column in ("close", "pre_close"):
                value = pd.to_numeric(current.get(column), errors="coerce")
                if pd.notna(value) and float(value) > 0:
                    price, source = float(value), f"clean.daily.{column}"
                    break
            for column in ("tradable", "is_st", "is_suspended"):
                record[column] = current.get(column)
        if price is None:
            fallback = pd.to_numeric(position.get("buy_pnl_price"), errors="coerce")
            if pd.notna(fallback) and float(fallback) > 0:
                price, source = float(fallback), "account.buy_pnl_price"
        record["mark_price"] = price
        record["mark_source"] = source
        record["market_value"] = None if price is None else shares * price
        record["valid"] = bool(
            record["in_current_section"]
            and price is not None
            and bool(record.get("tradable", 0))
            and not bool(record.get("is_st", 0))
            and not bool(record.get("is_suspended", 0))
        )
        if price is not None:
            total_market_value += shares * price
        rows.append(record)
    return {
        "market_date": day,
        "mark_to_market": "clean.daily.close -> clean.daily.pre_close -> account.buy_pnl_price（D10 原位，只读预检）",
        "positions": rows,
        "valid_count": sum(bool(row["valid"]) for row in rows),
        "position_count": len(rows),
        "cash": float(account["cash"]),
        "total_value": float(account["cash"]) + total_market_value,
        "content_fingerprint": fingerprint(rows),
    }


def _select_restart_segment(
    days: Sequence[str], required: int, segment_end: Optional[str], errors: List[str]
) -> List[str]:
    """从台账中挑出被验证的连续段；超过 10 条时必须显式指定，禁止隐式取最后十条。"""
    if len(days) < required:
        errors.append(f"连续恢复日数={len(days)}，要求={required}，不足不得冻结")
        return []
    if len(days) == required:
        return list(days)
    if not segment_end:
        errors.append(
            f"恢复台账共 {len(days)} 条 > 要求 {required}；须显式指定被验证的连续段末日"
            "（--restart-segment-end），不得隐式取最后十条或删除失败记录"
        )
        return []
    if segment_end not in days:
        errors.append(f"指定恢复段末日不在台账: {segment_end}")
        return []
    end_index = days.index(segment_end)
    if end_index + 1 < required:
        errors.append(f"指定恢复段末日 {segment_end} 之前不足 {required} 个恢复日")
        return []
    return list(days[end_index + 1 - required : end_index + 1])


def validate_restart_records(
    records: Sequence[Mapping[str, Any]],
    calendar: Sequence[str],
    required: int = 10,
    segment_end: Optional[str] = None,
    account_last_update: Optional[str] = None,
    mode: str = "live",
) -> Dict[str, Any]:
    """Q11-A：只认连续、真实推进、无降级的旧链日跑台账；同日重跑不累计。

    ``mode="sim"`` 仅用于用户 2026-10-30 批准的 Q11-SIM 等效口径：允许历史交易日，
    但每笔记录必须显式标记 historical_sim/restart_verified 并带授权；不覆盖 live 网络
    与实时钟证据。

    超过 ``required`` 条时须由 ``segment_end`` 显式指定被验证段（原台账保留不改）；
    给出 ``account_last_update`` 时，被验证段末日必须与之对齐，禁止用旧恢复段证明
    一个不相干的当前账户。
    """
    if mode not in {"live", "sim"}:
        raise ValueError(f"Q11 模式非法: {mode!r}")
    errors: List[str] = []
    by_day: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        day = str(record.get("corrected_date") or record.get("trade_date") or "")
        if not day:
            errors.append(f"records[{index}] 缺 corrected_date")
            continue
        if day in by_day:
            errors.append(f"同日重复台账不得累计: {day}")
            continue
        by_day[day] = record
    days = sorted(by_day)
    selected = _select_restart_segment(days, required, segment_end, errors)
    if selected:
        positions: List[int] = []
        for day in selected:
            if day not in calendar:
                errors.append(f"恢复日不在交易日历: {day}")
            else:
                positions.append(calendar.index(day))
        if len(positions) == required and positions != list(
            range(positions[0], positions[0] + required)
        ):
            errors.append(f"恢复日不是连续交易日: {selected}")
    for day in selected:
        record = by_day[day]
        status = str(record.get("t0_status") or "")
        if not record.get("run_completed", False):
            errors.append(f"{day} 未标记 run_completed")
        if mode == "live":
            if record.get("account_last_update") != day:
                errors.append(f"{day} 账户未推进: {record.get('account_last_update')}")
        else:
            if not record.get("daily_run_record"):
                errors.append(f"{day} 缺当日 t0/t1 运行记录")
            if record.get("fill_count", 0) and not record.get("account_advanced", False):
                errors.append(f"{day} 有成交但账户未推进")
            before_update = str(record.get("before_account_last_update") or "")
            after_update = str(record.get("account_last_update") or "")
            if before_update and after_update and after_update < before_update:
                errors.append(f"{day} 账户日期回退: {before_update} -> {after_update}")
        if not status:
            errors.append(f"{day} 缺 t0_status（无运行记录不得计入连续恢复）")
        elif status.startswith("error") or status == "not_ready":
            errors.append(f"{day} t0_status 非成功形态: {status}")
        if record.get("ensure_degraded", False):
            errors.append(f"{day} ensure 发生降级，不得计入连续恢复")
        if mode == "live":
            if not record.get("actual_calendar_day", False):
                errors.append(f"{day} 非真实日历日跑（历史回放不计入）")
        else:
            if record.get("simulation") is not True:
                errors.append(f"{day} Q11-SIM 记录未标记 simulation=true")
            if record.get("source_kind") != "historical_sim":
                errors.append(f"{day} Q11-SIM 来源非法: {record.get('source_kind')!r}")
            if record.get("restart_verified") is not True:
                errors.append(f"{day} Q11-SIM 未证明独立进程重启")
            if record.get("actual_calendar_day") is not False:
                errors.append(f"{day} Q11-SIM 不得标记为真实日历日")
            if not str(record.get("authorization") or "").strip():
                errors.append(f"{day} Q11-SIM 缺用户授权文本")
            raw_evidence = str(record.get("raw_evidence") or "")
            if not raw_evidence or not (ROOT / raw_evidence).is_file():
                errors.append(f"{day} Q11-SIM 缺原始证据文件: {raw_evidence!r}")
    aligned: Optional[bool] = None
    if mode == "live" and account_last_update is not None and selected:
        aligned = selected[-1] == str(account_last_update)
        if not aligned:
            errors.append(
                f"冻结账户末日 {account_last_update} 与被验证恢复段末日 {selected[-1]} 不对齐"
                "（不得用旧恢复段证明当前账户）"
            )
    result = {
        "required_days": required,
        "observed_days": len(days),
        "ledger_days": len(days),
        "days": selected or days,
        "selected_days": selected,
        "segment_end": segment_end,
        "discarded_days": {
            "before": len(days) - len(selected) if selected else 0,
            "after": 0 if not selected else len(days) - len(days[: days.index(selected[-1]) + 1]),
        },
        "last_update": selected[-1] if selected else None,
        "account_last_update": account_last_update,
        "account_aligned": aligned,
        "errors": errors,
        "passed": not errors,
        "disposition": (
            "Q11-A：旧链连续 10 个真实交易日重启；同日幂等、历史回放、ensure 降级均不计数"
            if mode == "live"
            else "Q11-SIM（用户 2026-10-30 批准等效）：10 个连续历史交易日逐日独立进程重启；"
            "不覆盖真实 live 网络/时钟证据"
        ),
    }
    if mode == "sim":
        result["evidence_mode"] = "Q11_SIM"
        result["simulation_not_live_evidence"] = True
        result["account_alignment_rule"] = (
            "每日必须有 t0/t1 运行记录；有成交时必须推进账户；无成交日允许账户不变，"
            "但禁止日期回退"
        )
        result["limitations"] = [
            "historical_data_only",
            "not_live_network_evidence",
            "not_real_clock_evidence",
            "not_production_account_advance",
        ]
    return result


def load_calendar(data_root: Path) -> List[str]:
    frame = pd.read_parquet(data_root / "clean" / "trade_cal.parquet")
    days = frame["cal_date"].astype(str).str.replace("-", "", regex=False)
    return sorted(days[frame["is_open"] == 1].tolist())


def collect_gate1(
    *,
    repo_root: Path,
    data_root: Path,
    paper_root: Path,
    model_version: int,
    restart_records: Optional[Sequence[Mapping[str, Any]]] = None,
    restart_source: Optional[Mapping[str, Any]] = None,
    require_restart: bool = False,
    segment_end: Optional[str] = None,
    q11_mode: str = "live",
    q11_sim_authorization: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    account_path = paper_root / "state" / "account.json"
    account = _read_json(account_path)
    if require_restart and restart_records is None:
        raise FileNotFoundError(
            f"申请门 1 冻结但未提供 Q11-A 恢复台账（paper_root={paper_root}）；"
            "该项不得静默当作已检查的 0 日恢复"
        )
    flat = load_paper_config(paper_root)
    data_state = capture_data_state(data_root)
    labels = {
        name: (data_state.get("labels", {}).get(name, {}) or {}).get("sealed_through")
        for name in LABELS
    }
    calendar = load_calendar(data_root)
    if q11_mode == "sim":
        if not isinstance(q11_sim_authorization, Mapping):
            raise PermissionError("Q11-SIM 冻结必须提供用户授权证据")
        if q11_sim_authorization.get("approved") is not True:
            raise PermissionError("Q11-SIM 授权未标记 approved=true")
    restart = validate_restart_records(
        restart_records or [],
        calendar,
        segment_end=segment_end,
        account_last_update=account.get("last_update"),
        mode=q11_mode,
    )
    restart["q11_mode"] = q11_mode
    if q11_mode == "sim":
        restart["authorization"] = dict(q11_sim_authorization or {})
    restart["precheck_only"] = restart_records is None
    restart["source"] = dict(restart_source or {"kind": "none", "path": None})
    holdings = assess_holdings(account, data_root)
    frozen = bool(restart["passed"] and holdings["valid_count"] == holdings["position_count"])
    note = (
        "未提供 Q11-A 恢复台账，本次仅为预检，禁止作为三臂冻结起点。"
        if restart["precheck_only"]
        else "已读入 Q11-A 恢复台账并按连续段校验；未通过前禁止作为三臂冻结起点。"
    )
    if q11_mode == "sim":
        note += " 本次为用户 2026-10-30 批准的 Q11-SIM 等效：不覆盖 live 网络/实时钟证据。"
    result = {
        "stage": "v2_p4_t5_gate1",
        "captured_at": _now(),
        "verdict": "PASS" if frozen else "BLOCKED_Q11A",
        "gate1_frozen": frozen,
        "gate1_mode": "Q11_SIM" if q11_mode == "sim" else "Q11_LIVE",
        "frozen_at": _now() if frozen else None,
        "data_state": data_state,
        "labels_sealed_through": labels,
        "model": collect_model_identity(data_root / "models" / "stock_selection", model_version),
        "execution": collect_execution_identity(flat),
        "worktree": compute_worktree_digest(repo_root),
        "code_state": _code_state(),
        "environment": _environment(),
        "account": {
            "path": str(account_path),
            "sha256": _sha256(account_path),
            "last_update": account.get("last_update"),
            "cash": account.get("cash"),
            "position_count": len(account.get("positions") or {}),
            "content": account,
        },
        "holdings_validity": holdings,
        "q11a_restart": restart,
        "production_switch_changed": False,
        "note": note,
    }
    return result


def _frozen_identity_keys(result: Mapping[str, Any]) -> Dict[str, Any]:
    """冻结身份的比较键：账户、数据态与模型内容（不含随时可变的采集时间）。"""
    return {
        "account_sha256": (result.get("account") or {}).get("sha256"),
        "account_last_update": (result.get("account") or {}).get("last_update"),
        "data_state_id": (result.get("data_state") or {}).get("data_state_id"),
        "model_fingerprint": (result.get("model") or {}).get("content_fingerprint"),
        "execution_fingerprint": (result.get("execution") or {}).get("fingerprint"),
    }


def _read_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--paper-root", default=str(ROOT / "data" / "paper"))
    parser.add_argument("--model-version", type=int, default=24022)
    parser.add_argument(
        "--restart-ledger",
        help="Q11-A 日跑台账 JSON（list 或 {records:[...]}；本证据根的 q11a_restart_ledger.json）",
    )
    parser.add_argument(
        "--restart-segment-end",
        help="累计台账超过 10 条时，显式指定被验证连续段的末日（YYYYMMDD）",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="申请冻结门 1：必须同时提供已存在的 --restart-ledger，缺失即报错",
    )
    parser.add_argument(
        "--q11-mode",
        choices=("live", "sim"),
        default="live",
        help="Q11 判据模式；sim 仅用于用户批准的 Q11-SIM 等效",
    )
    parser.add_argument(
        "--q11-sim-authorization",
        help="Q11-SIM 用户授权 JSON（approved=true，含限制声明）",
    )
    return parser.parse_args(argv)


def _restart_ledger_input(
    path_text: Optional[str],
) -> Tuple[Optional[List[Mapping[str, Any]]], Optional[Dict[str, Any]]]:
    if not path_text:
        return None, None
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"指定的 Q11-A 恢复台账不存在: {path}（不得当作已检查的 0 日恢复）")
    records = load_cumulative_ledger(path)
    source = {
        "kind": "restart_ledger",
        "path": str(path),
        "sha256": _sha256(path),
        "records": len(records),
    }
    return records, source


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.freeze and not args.restart_ledger:
        raise ValueError(
            "申请冻结门 1 必须显式提供 --restart-ledger（本证据根 q11a_restart_ledger.json）"
        )
    records, source = _restart_ledger_input(args.restart_ledger)
    authorization = (
        _read_optional(Path(args.q11_sim_authorization))
        if args.q11_sim_authorization
        else None
    )
    if args.q11_mode == "sim" and authorization is None:
        raise PermissionError("--q11-mode sim 必须提供 --q11-sim-authorization")
    result = collect_gate1(
        repo_root=Path(args.repo_root),
        data_root=Path(args.data_root),
        paper_root=Path(args.paper_root),
        model_version=args.model_version,
        restart_records=records,
        restart_source=source,
        require_restart=bool(args.freeze or args.restart_ledger),
        segment_end=args.restart_segment_end,
        q11_mode=args.q11_mode,
        q11_sim_authorization=authorization,
    )
    evidence = Path(args.evidence_dir)
    frozen_path = evidence / "verification_summary.json"
    existing = _read_optional(frozen_path)
    if existing is not None and existing.get("gate1_frozen") is True:
        # 冻结证据按不可变输入使用：身份一致只写进度预检，身份漂移直接拒绝覆盖。
        before = _frozen_identity_keys(existing)
        after = _frozen_identity_keys(result)
        if before != after:
            raise RuntimeError(
                "门 1 冻结身份不可变：本次身份与已冻结身份不一致"
                f"（已冻结={before}，本次={after}）；如需新起点请使用新证据根"
            )
        precheck = {
            "captured_at": result["captured_at"],
            "reason": "门 1 已冻结，本次仅写进度预检，不改写已绑定三臂的冻结身份",
            "frozen_at": existing.get("frozen_at"),
            "identity": before,
            "q11a_restart": result["q11a_restart"],
            "verdict": result["verdict"],
        }
        _dump(evidence / "gate1_progress_precheck.json", precheck)
        print(
            json.dumps(
                {"verdict": "FROZEN_UNCHANGED", "q11a": result["q11a_restart"]},
                ensure_ascii=False,
            )
        )
        return 0
    _dump(evidence / "gate1_identity.json", result)
    _dump(evidence / "worktree_digest.json", result["worktree"])
    _dump(frozen_path, result)
    pd.DataFrame(result["holdings_validity"]["positions"]).to_csv(
        evidence / "holdings_validity.csv", index=False
    )
    print(
        json.dumps(
            {"verdict": result["verdict"], "q11a": result["q11a_restart"]}, ensure_ascii=False
        )
    )
    return 0 if result["gate1_frozen"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
