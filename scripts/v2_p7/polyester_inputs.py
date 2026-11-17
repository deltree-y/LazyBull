"""三段报价的固定归一化；只处理商品输入，不读取证券行情。

复杂度豁免：阶段专用研究数据工具，不入生产（CLAUDE §7.8）。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from scripts.v2_p7.polyester_policy import validate_records


def normalize_htfc(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """华泰原件主段；三价消费方向固定，不按价差或股票结果选报价。"""
    records, excluded = [], []
    for row in rows:
        day = row.get("observation_date")
        if not day or not "20181201" <= day <= "20250228":
            excluded.append(
                {"source_id": row.get("article_id"), "reason": "outside_htfc_period_or_no_date"}
            )
            continue
        if not row.get("quality_usable"):
            excluded.append(
                {
                    "source_id": row.get("article_id"),
                    "reason": row.get("parse_status"),
                    "observation_date": day,
                    "fixable": True,
                }
            )
            continue
        # 早期同报告与后来双报告口径分别成序列，跨断点不滚动拼窗。
        paired = row.get("parse_status", "").startswith("same_day_TA_EG")
        record = {
            "source_id": f"htfc_{row['article_id']}_{day}",
            "observation_date": day,
            "publication_date": row["publication_date"],
            "available_date": row["available_date"],
            "poy": float(row["price_poy"]),
            "pta": float(row["price_pta"]),
            "meg": float(row["price_meg"]),
            "unit": "CNY/t",
            "series_id": "htfc_ta_eg_pair_v1" if paired else "htfc_ta_single_v1",
            "quality_usable": True,
            "strict_usable": True,
            "source_sha256": row["sha256"],
            "source_url": row["source_url"],
            "source_pdf": row["source_pdf"],
            "source_evidence": row.get("source_evidence", []),
            "semantic_flags": row.get("semantic_flags", []),
            "publication_basis": row.get("availability_rule"),
            "source_kind": "htfc",
        }
        records.append(record)
    return records, excluded


def normalize_nanhua(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """南华接2025年3–11月，明确把规格承接作为H级代理假设。"""
    records, excluded = [], []
    for row in rows:
        day = row["observation_date"]
        if not "20250301" <= day <= "20251130":
            excluded.append({"source_id": row["source_id"], "reason": "outside_nanhua_period"})
            continue
        if not row.get("numeric_complete"):
            excluded.append(
                {"source_id": row["source_id"], "reason": "incomplete_three_legs", "fixable": True}
            )
            continue
        records.append(
            {
                "source_id": f"{row['source_id']}_{day}",
                "observation_date": day,
                "publication_date": row["publication_date"],
                "available_date": row["available_date"],
                "poy": float(row["poy"]),
                "pta": float(row["pta"]),
                "meg": float(row["meg"]),
                "unit": "CNY/t",
                "series_id": "nanhua_poy_reference_inferred_v1",
                "quality_usable": True,
                "strict_usable": False,
                "source_sha256": row["sha256"],
                "source_url": row["source_url"],
                "source_pdf": row["pdf_path"],
                "spec_evidence": row["spec_evidence"],
                "publication_basis": row.get("publication_basis"),
                "source_kind": "nanhua",
                "semantic_flags": [
                    "poy_spec_inferred_not_explicit",
                    "tax_and_freight_not_explicit",
                ],
            }
        )
    return records, excluded


def dated_tbs_rows(
    payload: dict, start: str, end: str, *, duplicate_ledger: list[dict] | None = None
) -> dict[str, float]:
    """由固定区间唯一还原MM-DD；冲突不得通过字典覆盖消失。"""
    grouped: dict[str, list[dict]] = {}
    for index, (short_day, raw_price) in enumerate(payload["data"][1:], start=1):
        candidates = [
            datetime.strptime(f"{year}-{short_day}", "%Y-%m-%d").strftime("%Y%m%d")
            for year in range(int(start[:4]), int(end[:4]) + 1)
        ]
        candidates = [day for day in candidates if start <= day <= end]
        if len(candidates) != 1:
            raise ValueError(f"TBS日期无法由区间唯一还原：{short_day}")
        grouped.setdefault(candidates[0], []).append(
            {"raw_index": index, "price": float(raw_price)}
        )
    result = {}
    for day, rows in grouped.items():
        values = {row["price"] for row in rows}
        if len(rows) > 1:
            if duplicate_ledger is None:
                raise ValueError(f"TBS重复日期需要明确审计处置：{day}")
            duplicate_ledger.append(
                {
                    "observation_date": day,
                    "original_rows": rows,
                    "reason": (
                        "duplicate_same_value_deduplicated"
                        if len(values) == 1
                        else "conflicting_prices_excluded"
                    ),
                    "fixable": False,
                }
            )
        if len(values) == 1:
            result[day] = next(iter(values))
    return result


def normalize_tbs(
    payloads: dict[str, dict], sources: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """近端当前历史版本，公开日为显式假设；不冒充逐日首发证据。"""
    tables, duplicate_rows = {}, []
    for key, value in payloads.items():
        ledger: list[dict] = []
        tables[key] = dated_tbs_rows(value, "20251201", "20261031", duplicate_ledger=ledger)
        duplicate_rows.extend({**row, "leg": key} for row in ledger)
    all_days = sorted(set().union(*(set(table) for table in tables.values())))
    records, excluded = [], duplicate_rows
    for day in all_days:
        missing = [key for key, table in tables.items() if day not in table]
        if missing:
            excluded.append(
                {
                    "observation_date": day,
                    "reason": "tbs_missing_same_date_leg",
                    "missing_legs": missing,
                    "fixable": False,
                }
            )
            continue
        available = (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        records.append(
            {
                "source_id": f"tbs_{day}",
                "observation_date": day,
                "publication_date": day,
                "available_date": available,
                **{key: table[day] for key, table in tables.items()},
                "unit": "CNY/t",
                "series_id": "tbs_current_vintage_daily_v1",
                "quality_usable": True,
                "strict_usable": False,
                "publication_date_assumed": True,
                "historical_publication_verified": False,
                "publication_basis": "研究假设：观测日标记为当日发布，次日可用；不是已核验的首发事实",
                "source_evidence": sources,
                "source_kind": "tbs",
                "semantic_flags": [
                    "current_historical_vintage",
                    "first_publication_not_verified",
                    "three_leg_unit_assumption",
                    "tax_freight_method_unknown",
                ],
            }
        )
    return records, excluded


def assemble_records(
    htfc: list[dict],
    nanhua: list[dict],
    tbs_payloads: dict[str, dict],
    tbs_sources: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """固定时间段合并；在股票收益之前冻结，跨来源保留独立序列身份。"""
    records, excluded = [], []
    for values, omissions in (
        normalize_htfc(htfc),
        normalize_nanhua(nanhua),
        normalize_tbs(tbs_payloads, tbs_sources),
    ):
        records.extend(values)
        excluded.extend(omissions)
    validate_records(records)
    records.sort(
        key=lambda row: (row["observation_date"], row["publication_date"], row["source_id"])
    )
    return records, excluded
