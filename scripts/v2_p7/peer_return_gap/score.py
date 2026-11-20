# -*- coding: utf-8 -*-
"""月分数、三个研究臂与匹配对照（协议 §4）。

# 复杂度豁免：P7 候选一阶段专用研究工具，不入生产链路（AGENTS §7.5）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import spec
from .common import ResearchError, round12, sorted_by_score


def month_return_frame(
    month_panel: pd.DataFrame,
    *,
    formation: str,
    signal: str,
    median: pd.Series,
    days: Sequence[str],
) -> pd.DataFrame:
    """当月超额对数涨幅 ``L`` 与逐股原因。

    ``month_panel`` 覆盖 ``F<d<=T`` 的日行情；市场日中位收益须逐日有效。
    """
    signal_days = [day for day in days if formation < day <= signal]
    if not signal_days:
        raise ResearchError("信号月没有开市日")
    med = median.reindex(signal_days)
    med_ok = bool(med.notna().all()) and len(med) == len(signal_days)

    panel = month_panel.copy()
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel["trade_date"] = panel["trade_date"].astype(str)
    suspended = pd.to_numeric(panel["is_suspended"], errors="coerce").fillna(1) > 0
    real = (pd.to_numeric(panel["close"], errors="coerce") > 0) & ~suspended
    real_days = real.groupby(panel["ts_code"]).sum()
    rows = panel["ts_code"].nunique()
    coverage = real_days / float(len(signal_days))
    start = panel[panel["trade_date"] == formation].set_index("ts_code")
    end = panel[panel["trade_date"] == signal].set_index("ts_code")
    records: list[dict[str, Any]] = []
    for code in start.index:
        begin = start.loc[code]
        finish = end.loc[code] if code in end.index else None
        reason = ""
        value = np.nan
        if begin is None or finish is None:
            reason = "missing_endpoint"
        elif not (float(begin["close"]) > 0 and float(finish["close"]) > 0):
            reason = "invalid_endpoint_price"
        elif bool(pd.isna(begin["close_adj"]) or pd.isna(finish["close_adj"])):
            reason = "invalid_endpoint_adj"
        elif float(begin["close_adj"]) <= 0 or float(finish["close_adj"]) <= 0:
            reason = "invalid_endpoint_adj"
        elif float(coverage.get(code, 0.0)) < spec.MONTH_MIN_COVERAGE:
            reason = f"month_coverage<{spec.MONTH_MIN_COVERAGE}"
        elif not med_ok:
            reason = "market_reference_missing"
        else:
            total = float(np.log(float(finish["close_adj"]) / float(begin["close_adj"])))
            value = total - float(med.sum())
        records.append(
            {
                "ts_code": code,
                "L": round12(value) if np.isfinite(value) else np.nan,
                "valid": not reason,
                "reason": reason,
                "coverage": round12(float(coverage.get(code, np.nan))),
                "close_start": float(begin["close"]) if begin is not None else np.nan,
                "close_end": float(finish["close"]) if finish is not None else np.nan,
            }
        )
    frame = pd.DataFrame(records)
    frame["market_reference_valid"] = med_ok
    frame.attrs["signal_days"] = len(signal_days)
    frame.attrs["rows"] = int(rows)
    return frame


def peer_month_value(
    excess_month: pd.Series, peers: Sequence[str]
) -> tuple[float, dict[str, Any]]:
    """冻结同伴的等权均值；少于 8 个有效即不可得（不补 0、不换邻居）。"""
    values = excess_month.reindex(list(peers))
    valid = int(values.notna().sum())
    meta = {"valid": valid, "total": len(peers)}
    if valid < spec.PEER_MIN_VALID:
        return float("nan"), meta
    return round12(float(values.mean(skipna=True))), meta


def score_centers(
    *,
    beta_frame: pd.DataFrame,
    month_frame: pd.DataFrame,
    excess_month_by_code: Mapping[str, float],
    peers_by_center: Mapping[str, Sequence[str]],
    base_codes: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """主分数 ``beta*peer_month - L``；中心须属 T 基本域且有有效 beta。"""
    beta_map = (
        beta_frame.dropna(subset=["beta"]).set_index("center")["beta"].to_dict()
        if len(beta_frame)
        else {}
    )
    month_map = month_frame.set_index("ts_code").to_dict("index")
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for code in list(dict.fromkeys(base_codes)):
        record = month_map.get(code)
        beta = beta_map.get(code)
        if record is None or not record.get("valid"):
            reasons["month_invalid"] = reasons.get("month_invalid", 0) + 1
            continue
        if beta is None:
            reasons["no_beta"] = reasons.get("no_beta", 0) + 1
            continue
        peers = peers_by_center.get(code, ())
        peer_value, peer_meta = peer_month_value(
            pd.Series(excess_month_by_code), peers
        )
        if not np.isfinite(peer_value):
            reasons["peer_invalid"] = reasons.get("peer_invalid", 0) + 1
            continue
        value = round12(float(beta) * float(peer_value) - float(record["L"]))
        rows.append(
            {
                "ts_code": code,
                "beta": round12(float(beta)),
                "peer_month": peer_value,
                "L": float(record["L"]),
                "score": value,
                "peer_valid": peer_meta["valid"],
                "peer_total": peer_meta["total"],
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {"scored": int(len(frame)), "reasons": reasons}


def select_arm(
    scored: pd.DataFrame,
    *,
    key: str,
    industry: Mapping[str, str],
    slots: int = spec.SLOTS,
    sector_cap: int = spec.SECTOR_CAP,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """按 ``key`` 排序（默认降序）、行业上限挑选；返回有序目标。"""
    if ascending:
        material = [dict(row) for row in scored.to_dict("records")]
        ordered = sorted(
            material, key=lambda row: (round12(float(row[key])), str(row["ts_code"]))
        )
    else:
        ordered = sorted_by_score(scored.to_dict("records"), key)
    picked: list[dict[str, Any]] = []
    per_sector: dict[str, int] = {}
    for row in ordered:
        sector = industry.get(str(row["ts_code"]))
        if sector is None:
            continue
        if per_sector.get(sector, 0) >= sector_cap:
            continue
        picked.append(dict(row))
        per_sector[sector] = per_sector.get(sector, 0) + 1
        if len(picked) >= slots:
            break
    return picked


def midrank_frame(base_frame: pd.DataFrame) -> pd.DataFrame:
    """``u=(midrank-0.5)/n``，分母 = T 基本域中 L 与市值均有效的对象。"""
    frame = base_frame.dropna(subset=["L", "total_mv_yuan"]).copy()
    if frame.empty:
        return frame
    n = len(frame)
    frame["u_mv"] = (frame["total_mv_yuan"].rank(method="average") - 0.5) / n
    frame["u_L"] = (frame["L"].rank(method="average") - 0.5) / n
    frame["n_rank"] = n
    return frame


def match_targets(
    *,
    peer_targets: Sequence[Mapping[str, Any]],
    scored: pd.DataFrame,
    base_frame: pd.DataFrame,
    industry: Mapping[str, str],
) -> pd.DataFrame:
    """逐名义槽匹配同行业、近市值、近当月涨幅的另一只股票。

    缺匹配的槽保持现金；不替代、不回补。账上持仓不参与匹配。
    """
    ranks = midrank_frame(base_frame).set_index("ts_code")
    pool = scored.copy()
    pool["ts_code"] = pool["ts_code"].astype(str)
    pool = pool.set_index("ts_code")
    peer_codes = {str(row["ts_code"]) for row in peer_targets}
    used: set[str] = set()
    records: list[dict[str, Any]] = []
    for order, row in enumerate(peer_targets, start=1):
        source = str(row["ts_code"])
        sector = industry.get(source)
        record: dict[str, Any] = {
            "slot": order,
            "source": source,
            "sector": sector,
            "match": None,
            "distance": None,
            "reason": "",
        }
        if sector is None:
            record["reason"] = "source_industry_unknown"
            records.append(record)
            continue
        if source not in ranks.index:
            record["reason"] = "source_rank_missing"
            records.append(record)
            continue
        candidates = [
            code
            for code in pool.index
            if code not in peer_codes
            and code not in used
            and code != source
            and industry.get(code) == sector
            and code in ranks.index
        ]
        target_mv = float(ranks.loc[source, "u_mv"])
        target_l = float(ranks.loc[source, "u_L"])
        best = None
        for code in sorted(candidates):
            diff_mv = abs(target_mv - float(ranks.loc[code, "u_mv"]))
            diff_l = abs(target_l - float(ranks.loc[code, "u_L"]))
            if diff_mv > spec.MATCH_RANK_TOL or diff_l > spec.MATCH_RANK_TOL:
                continue
            distance = round12(diff_mv + diff_l)
            key = (distance, code)
            if best is None or key < best[0]:
                best = (key, code, diff_mv, diff_l)
        if best is None:
            record["reason"] = "no_candidate_within_tolerance"
            records.append(record)
            continue
        _key, code, diff_mv, diff_l = best
        used.add(code)
        record.update(
            {
                "match": code,
                "distance": round12(diff_mv + diff_l),
                "diff_mv": round12(diff_mv),
                "diff_L": round12(diff_l),
                "reason": "matched",
            }
        )
        records.append(record)
    return pd.DataFrame(records)
