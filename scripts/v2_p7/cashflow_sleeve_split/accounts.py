# -*- coding: utf-8 -*-
"""run-a 四账户 × 两本金回放（协议 §6「大腿规格」；审点二 A 批）。

口径（冻结）
============
- 账户引擎 = :func:`scripts.v2_p7.operating_monthly_baseline.run_baseline_account`
  原样复用（half-Kelly、Top20、20 日持有/调仓、分两批、挂单/补齐/提前调仓、
  ``current_config=False`` 硬性要求）。
- 公司行为簿 = run_r3 冻结 ``corporate_input``（只读；逐件 SHA 复核）。
- 价格 = ``load_baseline_prices``（2018-01-01 起预热，每股实际行取尾 60）。
- 压力臂 = 唯一差异为滑点 5bp→10bp（``CostModel(0.0001954, 5.0, 0.0010, 0.0010)``，
  协议 §7.6）；回放函数与主臂逐行同源，差异仅 ``cost_model`` 覆写。
- 现金腿 / 主组合（R80+B20）属 run-b 范围，本轮 ``not_run``。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT

#: 价格预热起点（与 operating_monthly_baseline.WARMUP_START 同值，独立声明防漂移）
WARMUP_START = "20180101"

#: run_r3 corporate_input 冻结清单（协议 §6「旧 A 的 corporate_input 只读」）
CORPORATE_MANIFEST_RELATIVE = f"{spec.A_BASELINE['relative']}/corporate_input/manifest.json"
CORPORATE_DIVIDEND_RELATIVE = f"{spec.A_BASELINE['relative']}/corporate_input/dividend"

#: 主账户角色（协议 §8 审点二：F100/F80/R100/R80 两本金）
PRIMARY_ROLES: Tuple[str, ...] = ("F100", "F80", "R100", "R80")
STRESS_ROLES: Tuple[str, ...] = ("F100", "F80", "R100", "R80")


class AccountError(RuntimeError):
    """账户输入/身份/守恒失败（fail-fast）。"""


def build_calendar(root: Path) -> Tuple[List[str], str]:
    """账户执行日历 + 现金锚（锚 = 窗前第一个交易日）。"""
    from scripts.v2_p7.cashflow_sleeve_split.training import load_trade_calendar

    full = load_trade_calendar(root)
    start = str(spec.WINDOW["start"])
    end = str(spec.WINDOW["end"])
    if start not in full or full.index(start) == 0:
        raise AccountError("账户起点或前一交易日（现金锚）缺失")
    days = [day for day in full if start <= day <= end]
    if len(days) != int(spec.WINDOW["return_intervals"]):
        raise AccountError(f"执行日历长度不符：{len(days)} != {spec.WINDOW['return_intervals']}")
    anchor = full[full.index(start) - 1]
    return days, anchor


def build_baseline_inputs(
    ranked: Mapping[str, Sequence[Tuple[str, float]]],
    calendar: Sequence[str],
    anchor: str,
    identity: Mapping[str, Any],
    *,
    allowed_empty: Sequence[str] = (),
) -> Any:
    """构造 ``BaselineInputs``（与旧 A 三段拼接产物同 schema；ranked 全域校验）。

    ``allowed_empty``：score 宽度台账登记的空排名日（北向等源数据缺口 → 内核
    fail-closed 拒评）。缺件日必须 ⊆ 该集合，并按生产语义落 ``[]``（该日无买入）；
    其余缺口仍 fail-closed。
    """
    from scripts.v2_p7.operating_monthly_baseline import BaselineInputs, _days, _validate_ranked

    days = _days(calendar, "run-a 执行")
    if set(ranked) - set(days):
        raise AccountError("ranked 日期超出执行窗口")
    empty_allowed = {str(day) for day in allowed_empty}
    missing = [day for day in days if day not in ranked]
    unresolved = [day for day in missing if day not in empty_allowed]
    if unresolved:
        raise AccountError(f"ranked 存在缺口：{unresolved[:5]}…（{len(unresolved)} 日）")
    unexpected_empty = sorted(empty_allowed & set(ranked))
    if unexpected_empty:
        raise AccountError(
            f"空排名日 {unexpected_empty[:5]}… 在 ranked 中出现但登记为空（口径冲突）"
        )
    _validate_ranked(ranked)
    full = _days(load_full_calendar(), "run-a 全日历")
    price_days = tuple(day for day in full if WARMUP_START <= day <= days[-1])
    if not price_days or price_days[0] >= days[0]:
        raise AccountError("价格预热起点必须早于现金起步")
    return BaselineInputs(
        ranked={
            day: [(str(code), float(score)) for code, score in ranked.get(day, ())] for day in days
        },
        calendar=days,
        anchor=str(anchor),
        price_days=price_days,
        identity=dict(identity),
    )


def load_full_calendar() -> List[str]:
    from scripts.v2_p7.cashflow_sleeve_split.training import load_trade_calendar

    return load_trade_calendar(REPO_ROOT)


def verify_corporate_input(root: Path) -> Dict[str, Any]:
    """复核 run_r3 冻结公司行为清单与逐件 SHA（协议 §6「只读 + 校验」）。"""
    base = Path(root)
    manifest_path = base / CORPORATE_MANIFEST_RELATIVE
    if not manifest_path.exists():
        raise AccountError(f"冻结公司行为清单缺件：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dividend_dir = base / CORPORATE_DIVIDEND_RELATIVE
    files = sorted(dividend_dir.glob("*.parquet"))
    checked = 0
    mismatched: List[str] = []
    entries = {Path(item["output_path"]).name: item for item in manifest.get("files") or []}
    for path in files:
        entry = entries.get(path.name)
        if entry is None:
            mismatched.append(f"{path.name}: 无清单登记")
            continue
        from scripts.v2_p7.buffer_native_replay import sha256

        if sha256(path) != entry["output_sha256"]:
            mismatched.append(f"{path.name}: SHA 不符")
        checked += 1
    if mismatched:
        raise AccountError(f"冻结公司行为输入被改写：{mismatched[:5]}")
    if manifest.get("calendar_days") != int(spec.WINDOW["nav_points"]):
        raise AccountError(
            f"冻结公司行为日历点数不符：{manifest.get('calendar_days')} != {spec.WINDOW['nav_points']}"
        )
    return {
        "manifest_relative": CORPORATE_MANIFEST_RELATIVE,
        "dividend_files": checked,
        "calendar_start": manifest.get("calendar_start"),
        "calendar_end": manifest.get("calendar_end"),
        "verified": True,
    }


def _dedupe_economic_duplicates(directory: Path, audit_path: Path) -> Dict[str, Any]:
    """经济全同的重复实施登记去重（同 (ts_code, ex_date, known_date) 数值组合一致）。

    背景：TuShare dividend 对同一分红方案存在「双报告期重复登记」——两条
    ``div_proc=实施`` 行经济字段（cash/stk/tax）完全相同、仅 ``end_date``
    （报告期）不同。``buffer_native_corporate`` 把 ``end_date`` 计入经济字段
    ⇒ 必触发「冲突实施版本」fail-closed，而既定消解件（裁决 B）只处理数值
    组合冲突。此处按确定性规则去重：保留 ``end_date`` 最晚者（并列依次比
    ``pay_date``/``record_date``，再按元组字符串序），经济结果不受影响；
    冻结输入不动（仅研究副本），逐组留痕审计。
    """
    from scripts.v2_p7.minute_repair_ml.corporate_resolve import NUMERIC_FIELDS, _numeric, _text

    frames = {path.name: pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))}
    total = pd.concat(
        [frame.assign(_src=name) for name, frame in frames.items()], ignore_index=True
    )
    if total.empty:
        audit: Dict[str, Any] = {"groups": 0, "rows_dropped": 0, "records": []}
    else:
        total = total.copy()
        keep_mask = pd.Series(True, index=total.index)
        for name in NUMERIC_FIELDS:
            total[name] = _numeric(total, name)
        for name in (
            "ex_date",
            "ann_date",
            "imp_ann_date",
            "end_date",
            "pay_date",
            "record_date",
            "div_proc",
        ):
            total[f"_{name}"] = _text(total, name)
        total["_known"] = total[["_ann_date", "_imp_ann_date"]].max(axis=1)
        # 镜像引擎预过滤：去重只作用于「实施」且除权日已知的行（引擎不读其余行，
        # 也不得因本步骤改变副本中其他行的存在性）
        scope = total.loc[total["_div_proc"].eq("实施") & total["_ex_date"].ne("")]
        combo_cols = list(NUMERIC_FIELDS)
        records: List[Dict[str, Any]] = []
        for key, group in scope.groupby(["ts_code", "_ex_date", "_known"], sort=True):
            if len(group) < 2 or group[combo_cols].drop_duplicates().shape[0] != 1:
                continue
            ordered = group.sort_values(["_end_date", "_pay_date", "_record_date"], kind="stable")
            winner = ordered.iloc[-1]
            for index in group.index:
                if index != winner.name:
                    keep_mask.loc[index] = False
            records.append(
                {
                    "ts_code": str(key[0]),
                    "ex_date": str(key[1]),
                    "known_date": str(key[2]),
                    "kept": {
                        "end_date": str(winner["_end_date"]),
                        "source": str(winner["_src"]),
                    },
                    "rows_dropped": int(len(group) - 1),
                }
            )
        kept = total.loc[keep_mask]
        helper_cols = [
            "_src",
            "_known",
            "_ex_date",
            "_ann_date",
            "_imp_ann_date",
            "_end_date",
            "_pay_date",
            "_record_date",
            "_div_proc",
        ]
        for name, frame in frames.items():
            part = kept[kept["_src"] == name].drop(
                columns=[c for c in helper_cols if c in kept.columns],
                errors="ignore",
            )
            frames[name] = part[list(frame.columns)].reset_index(drop=True)
        for name, frame in frames.items():
            frame.to_parquet(directory / name, index=False)
        audit = {
            "rule": "economic_identical_duplicate_dedupe",
            "groups": len(records),
            "rows_dropped": int(sum(item["rows_dropped"] for item in records)),
            "records": records,
        }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _isolate_pre_window_undated(directory: Path, audit_path: Path) -> Dict[str, Any]:
    """隔离「除权日未知 ∧ 登记日未知 ∧ known_date < 价格预热起点」的实施行。

    ``buffer_native_corporate._check_undated`` 对无登记日的未定日事件没有时间
    界限：任何时点持仓即 fail-closed（checkpoint-2 Codex 第 4 项：前裁决保留
    严格检查）。本隔离**只为让回放可完成**——被隔离事件的权益状态引擎无法
    判定 ⇒ 按协议登记 ``unbounded``，a_review 逐受影响账户给出两端测算
    （上界 = 税后每股 × 持有股数）；冻结原件不动。
    结清凭证尝试：本地 adj_factor 原始分区自 2005 年起（1992 年事件无法从
    复权因子取证）⇒ 凭证不可得，登记于 audit。
    """
    frames = {path.name: pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))}
    total = pd.concat(
        [frame.assign(_src=name) for name, frame in frames.items()], ignore_index=True
    )
    records: List[Dict[str, Any]] = []
    if not total.empty:
        drop_idx: List[int] = []
        for _, row in total.iterrows():
            if str(row.get("div_proc")) != "实施":
                continue
            ex_date = str(row.get("ex_date")) if pd.notna(row.get("ex_date")) else ""
            record_date = str(row.get("record_date")) if pd.notna(row.get("record_date")) else ""
            known = max(str(row.get("ann_date") or ""), str(row.get("imp_ann_date") or ""))
            if not ex_date and not record_date and known and known < WARMUP_START:
                drop_idx.append(row.name)
                records.append(
                    {
                        "ts_code": str(row["ts_code"]),
                        "end_date": str(row.get("end_date")),
                        "ann_date": str(row.get("ann_date")),
                        "imp_ann_date": str(row.get("imp_ann_date")),
                        "known_date": known,
                        "cash_div_tax": str(row.get("cash_div_tax")),
                        "status": "unbounded",
                        "note": (
                            "除权/登记/支付日全缺；结清凭证不可得（本地 adj_factor 原始分区自"
                            " 2005 年起）；按前裁决保留严格检查，仅隔离以完成回放；实际持仓下"
                            "两端测算见 a_review"
                        ),
                    }
                )
        if drop_idx:
            kept = total.drop(index=drop_idx)
            for name, frame in frames.items():
                part = kept[kept["_src"] == name].drop(columns=["_src"], errors="ignore")
                frames[name] = part[list(frame.columns)].reset_index(drop=True)
            for name, frame in frames.items():
                frame.to_parquet(directory / name, index=False)
    audit = {
        "rule": "pre_window_undated_isolation",
        "bound_known_date_lt": WARMUP_START,
        "strict_check_ruling": "p7_buffer_native_result_20261116（保留严格检查）",
        "settlement_evidence": "not_obtainable（adj_factor 原始分区 2005 起）",
        "groups": len(records),
        "records": records,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def build_corporate_book(
    root: Path,
    calendar: Sequence[str],
    work: Optional[Path] = None,
) -> Any:
    """冻结公司行为事件簿（日历 = [锚, *执行日]，逐件输入已复核）。

    生产范式（minute_repair native_replay 同源）：冻结输入只读；复制到
    ``work/corporate_resolved/dividend`` 后两步研究副本清理：①经济全同的
    重复实施登记确定性去重（ TuShare 双报告期登记伪影，引擎因 end_date 计入
    经济字段必拒判；本步骤经济零影响、逐组留痕）；②按用户裁决 B（2026-11-18）
    窄口径消解数值组合冲突。审计分别落 ``dedupe_audit.json`` 与
    ``resolutions.json``，再从清理副本建簿。
    """
    import shutil

    from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook

    source = Path(root) / CORPORATE_DIVIDEND_RELATIVE
    dividend_dir = source
    resolve_audit: Dict[str, Any] = {"resolved": False}
    if work is not None:
        target = Path(work) / "corporate_resolved" / "dividend"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for path in sorted(source.glob("*.parquet")):
            shutil.copy2(path, target / path.name)
            copied += 1
        dedupe_audit = _dedupe_economic_duplicates(
            target, Path(work) / "corporate_resolved" / "dedupe_audit.json"
        )
        undated_audit = _isolate_pre_window_undated(
            target, Path(work) / "corporate_resolved" / "undated_audit.json"
        )
        from scripts.v2_p7.minute_repair_ml.corporate_resolve import resolve_in_place

        resolutions_path = Path(work) / "corporate_resolved" / "resolutions.json"
        audit = resolve_in_place(target, resolutions_path)
        resolve_audit = {
            "copied": copied,
            "resolved": True,
            "dedupe": {
                "groups": dedupe_audit["groups"],
                "rows_dropped": dedupe_audit["rows_dropped"],
            },
            "undated_isolated": {"groups": undated_audit["groups"]},
            "conflicts": {
                "groups": int(audit.get("n_unresolved") or 0),
                "rows_dropped": int(audit.get("rows_dropped") or 0),
            },
        }
        dividend_dir = target
    full_calendar = [str(day) for day in calendar]
    book = NativeCorporateBook(
        calendar=full_calendar,
        dividend_dir=dividend_dir,
    )
    book.resolve_audit = resolve_audit
    return book


def load_prices_with_identity(root: Path, inputs: Any) -> Tuple[Any, Dict[str, Any]]:
    """价格与预热身份（``load_baseline_prices`` 原样复用）。"""
    from scripts.v2_p7.operating_monthly_baseline import load_baseline_prices

    return load_baseline_prices(Path(root), inputs)


def run_account(
    root: Path,
    *,
    inputs: Any,
    prices: Any,
    indexes: Any,
    capital: int,
    corporate_book: Any,
    output: Path,
    cost_override: Any = None,
    label: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> Dict[str, Any]:
    """单账户原生回放，所有主账户和压力账户共用实际槽位检查。

    ``cost_override`` 非 None 时为压力臂：唯一差异 = ``cost_model`` 覆写
    （滑点双边 10bp）；其余校验、驱动、锚日、守恒门与输出形态逐行一致。
    """
    from scripts.v2_p7.buffer_native_replay import initialize_driver, write_json
    from scripts.v2_p7.cashflow_sleeve_split.account_engine import CashflowAccountEngine
    from scripts.v2_p7.operating_monthly_baseline import (
        baseline_result_payload,
        cash_anchor,
        engine_kwargs,
        validate_account,
    )

    if isinstance(capital, bool) or int(capital) != capital or capital <= 0:
        raise AccountError("账户初始资金必须为正整数元")
    days = list(inputs.calendar)
    if inputs.anchor >= days[0] or set(inputs.ranked) - set(days):
        raise AccountError("锚点/排名日超出统一执行窗口")
    expected = tuple([inputs.anchor, *days])
    if tuple(corporate_book._days) != expected:
        raise AccountError("公司行为事件簿与现金锚点/执行日历不一致")
    mark = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    kwargs = engine_kwargs(Path(root), inputs.ranked, capital, False)
    if cost_override is not None:
        kwargs["cost_model"] = cost_override
    engine = CashflowAccountEngine(**kwargs, corporate_book=corporate_book)
    dates = [pd_ts(day) for day in days]
    initialize_driver(engine, dates, prices, indexes)
    daily, state = cash_anchor(inputs.anchor, capital)
    engine.native_corporate.before_day(engine, pd_ts(inputs.anchor))
    engine.native_corporate.after_day(engine, pd_ts(inputs.anchor))
    engine.research_daily.append(daily)
    engine.research_states.append(state)
    try:
        for count, date in enumerate(dates, 1):
            engine.run_step(date, dates, prices)
            if progress is not None and (count % 200 == 0 or count == len(dates)):
                progress(count, len(dates))
        result = baseline_result_payload(engine)
        result["position_limit_rejections"] = engine.position_limit_rejections
        validation = validate_account(result, initial_capital=capital)
        if not validation["valid"]:
            raise AccountError(f"账户守恒失败：{validation['violations'][:3]}")
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_days": len(engine.research_daily),
                "economic_success_claimed": False,
            },
        )
        raise
    result["validation"] = validation
    result["baseline_identity"] = {
        "initial_capital": capital,
        "current_config": False,
        "position_sizing": "half_kelly",
        "max_per_industry": None,
        "initial_point": inputs.anchor,
        "start": days[0],
        "end": days[-1],
        "capital_scaled_from_other_nav": False,
        "cross_sleeve_borrowing": False,
        "cross_sleeve_netting": False,
        "locked_sell_policy": "defer_entire_order_until_all_shares_unlocked",
        "not_virgin_holdout": True,
        "stress_cost_override": (
            {"slippage_per_side": 0.0010} if cost_override is not None else None
        ),
        "label": label,
        "seconds": time.perf_counter() - mark,
    }
    for key, value in result.items():
        write_json(output / f"{key}.json", value)
    return result


def pd_ts(day: str) -> Any:
    import pandas as pd

    return pd.Timestamp(str(day))


def capital_for_role(role: str, scenario: int) -> int:
    """角色 → 实跑本金（F100/R100 = K，F80/R80 = 0.8K；只分配一次）。"""
    share = spec.ACCOUNT_ROLES[role]["capital_share"]
    value = int(round(scenario * float(share)))
    if scenario * float(share) != value:
        raise AccountError(f"角色本金非整数：{role}@{scenario}")
    return value


__all__ = [
    "AccountError",
    "PRIMARY_ROLES",
    "STRESS_ROLES",
    "WARMUP_START",
    "build_baseline_inputs",
    "build_calendar",
    "build_corporate_book",
    "capital_for_role",
    "load_prices_with_identity",
    "run_account",
    "verify_corporate_input",
]
