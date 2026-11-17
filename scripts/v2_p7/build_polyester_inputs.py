"""冻结涤纶报价输入及覆盖台账，不读取股票收益。

复杂度豁免：阶段专用证据工具，不入生产（CLAUDE §7.8）。从仓库根运行。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.v2_p7.polyester_inputs import assemble_records  # noqa: E402
from scripts.v2_p7.polyester_policy import build_signals, signal_summary  # noqa: E402

BASE = ROOT / "data/reports/p7_polyester_economic_20261116"
SPEC_SHA = "0fc4445c8d707a3fd1564b6a889a5af918b2a7718377c2df64d46d2f2beecff4"
SELECTION = ROOT / "docs/experiments/p7_polyester_source_selection_20261116.md"
SEALED = {
    "prices_htfc": "d7a018b34fbdb24896c9eaed77dc1589682e6ff50eba7abd85fbf443abc28efa",
    "prices_alternative": "214ac283f9661e96f8ef05371e608210e19e4a5991bbb8f1c1f03f9be4eff368",
    "prices_tbs_audit": "6a8981a0dae33a8a135113a9247a7de85a7d98545e048013c65da36525e0ce3c",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    spec = read(BASE / "prereg/spec.json")
    assert spec["sha256"] == SPEC_SHA == sha(ROOT / spec["spec_path"])
    for folder, digest in SEALED.items():
        assert sha(BASE / folder / "manifest.json") == digest, folder
    htfc = read(BASE / "prices_htfc/quote_rows.json") + read(
        BASE / "prices_htfc_late/quote_rows_final.json"
    )
    nanhua = read(BASE / "prices_alternative/quotes.json")["quotes"]
    tbs = BASE / "prices_tbs"
    payloads = {
        key: read(tbs / f"price_{pid}.json")
        for key, pid in (("poy", 161), ("pta", 156), ("meg", 157))
    }
    sources = {str(row["pid"]): row for row in read(tbs / "sources.json")}
    records, exclusions = assemble_records(htfc, nanhua, payloads, sources)
    table = pd.read_parquet(ROOT / "data/clean/trade_cal.parquet")
    dates = table.loc[
        table.exchange.eq("SSE") & table.is_open.astype(int).eq(1), "cal_date"
    ].astype(str)
    clock = sorted(set(dates.loc[dates.ge("20180101") & dates.le("20261029")]))
    calendar = [day for day in clock if day >= "20190401"]
    assert calendar[0] == "20190401" and calendar[-1] == "20261029"
    output = BASE / "input_prices"
    output.mkdir(exist_ok=False)
    save(output / "records.json", records)
    save(output / "exclusion_ledger.json", exclusions)
    shutil.copyfile(SELECTION, output / "source_selection.md")
    signals = build_signals(records, calendar, availability_calendar=clock)
    delayed = build_signals(records, calendar, delay_trading_days=5, availability_calendar=clock)
    strict_rows = [
        {**row, "quality_usable": row["quality_usable"] and row["strict_usable"]} for row in records
    ]
    strict = build_signals(strict_rows, calendar, availability_calendar=clock)
    for name, frame in (("main", signals), ("delay5", delayed), ("strict", strict)):
        frame.to_parquet(output / f"{name}_signals_before_returns.parquet", index=False)
    signal_exclusions = signals.loc[
        ~signals.input_known, ["signal_date", "ts_code", "reason", "months", "source_ids"]
    ]
    signal_exclusions.to_parquet(output / "signal_exclusion_ledger.parquet", index=False)
    assert len(signal_exclusions) == int((~signals.input_known).sum())
    # 原件差异的输入端检验先于任何股票收益：两种预先指定值均计算，不挑赢家。
    conflict_sensitivity = {}
    for use in ("eg", "ta"):
        alternate = copy.deepcopy(htfc)
        for row in alternate:
            if row.get("parse_status") == "poy_conflict_between_same_date_reports":
                row.update(row["candidate_primary_inputs"])
                if use == "ta":
                    row["price_poy"] = row["poy_ta_report"]
                row.update(quality_usable=True, parse_status="same_day_TA_EG_conflict_sensitivity")
        alt_records, _ = assemble_records(alternate, nanhua, payloads, sources)
        readings = {}
        for delay, reference in ((0, signals), (5, delayed)):
            alt = build_signals(
                alt_records, calendar, delay_trading_days=delay, availability_calendar=clock
            )
            different = alt.eligible.ne(reference.eligible) | alt.unknown_on.ne(
                reference.unknown_on
            )
            readings[str(delay)] = sorted(set(alt.loc[different, "signal_date"]))
        conflict_sensitivity[use] = readings
    save(output / "conflict_signal_sensitivity.json", conflict_sensitivity)
    coverage = {
        "source_rows": len(records),
        "commodity_months": len({row["observation_date"][:6] for row in records}),
        "main": signal_summary(signals),
        "delay5": signal_summary(delayed),
        "source_strict": signal_summary(strict),
        "records_by_source": {
            key: sum(row["source_kind"] == key for row in records)
            for key in ("htfc", "nanhua", "tbs")
        },
        "exclusion_ledger_rows": len(exclusions),
        "unknown_stock_month_slots": len(signal_exclusions),
        "stock_returns_read": False,
    }
    save(output / "coverage.json", coverage)
    methods = output / "methods"
    methods.mkdir()
    for name in ("build_polyester_inputs.py", "polyester_inputs.py", "polyester_policy.py"):
        shutil.copyfile(ROOT / "scripts/v2_p7" / name, methods / name)
    files = {SELECTION, ROOT / "data/clean/trade_cal.parquet"}
    files.update(output.rglob("*"))
    for folder in (*SEALED, "prices_htfc_late", "prices_tbs"):
        files.update((BASE / folder).rglob("*"))
    files = {path for path in files if path.is_file() and "__pycache__" not in path.parts}
    record_path = (output / "records.json").relative_to(ROOT).as_posix()
    save(
        output / "source_manifest.json",
        {
            "schema": "p7_polyester_price_inputs_v1",
            "spec_sha256": SPEC_SHA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_stock_returns": True,
            "records_file": record_path,
            "records_sha256": sha(output / "records.json"),
            "source_selection_file": SELECTION.relative_to(ROOT).as_posix(),
            "files": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(files)},
            "coverage": coverage,
            "data_purchase_yuan": 0,
        },
    )
    print(
        json.dumps(
            {
                "source_manifest_sha256": sha(output / "source_manifest.json"),
                "coverage": coverage,
                "conflict_sensitivity": conflict_sensitivity,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
