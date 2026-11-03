# -*- coding: utf-8 -*-
"""Minimal P5b family-transaction/recovery spike.

This is a deterministic file-level spike, not the production PaperHost. It validates
the transaction state machine and timing only; formal H1/H2 still use the real host.
All writes are under an explicit temp root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARMS = ("A", "B", "C")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class FamilyCampaign:
    root: Path
    generation: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)
    journal: dict[str, Any] = field(default_factory=dict)
    last_txn: str = ""

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.state = self.root / "state"
        self.backups = self.root / "backups"
        self.failed = self.root / "failed"
        self.ledger_path = self.root / "ledger.jsonl"
        self.journal_path = self.root / "journal.json"
        self.state.mkdir(parents=True, exist_ok=True)
        self.backups.mkdir(parents=True, exist_ok=True)
        if not self.journal_path.exists():
            self._write_initial()

    def _write_initial(self) -> None:
        for arm in ARMS:
            p = self.state / f"{arm}.json"
            if not p.exists():
                _dump(p, {"arm": arm, "generation": 0, "nav": 1.0})
            backup = self.backups / f"{arm}.json"
            if not backup.exists():
                shutil.copy2(p, backup)
        self.journal = {"status": "IDLE", "generation": 0, "txn": "", "promoted": []}
        _dump(self.journal_path, self.journal)

    def _ledger_has(self, txn: str) -> bool:
        return any(x.get("txn") == txn for x in self.ledger)

    def _append_ledger(self, txn: str, day: str) -> None:
        if self._ledger_has(txn):
            return
        entry = {"txn": txn, "day": day, "generation": self.generation,
                 "arms": {a: _sha(self.state / f"{a}.json") for a in ARMS}}
        self.ledger.append(entry)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def attempt(self, day: str, crash_after: int = 99, crash_recovery: bool = False) -> str:
        if self.journal.get("status") not in ("IDLE", "COMMITTED", "ABORTED"):
            raise RuntimeError(f"campaign not idle: {self.journal.get('status')}")
        txn = f"day-{day}-gen-{self.generation + 1}"
        self.last_txn = txn
        staged = self.root / "staging" / txn
        staged.mkdir(parents=True, exist_ok=True)
        old = {a: _sha(self.state / f"{a}.json") for a in ARMS}
        for arm in ARMS:
            payload = _load(self.state / f"{arm}.json")
            payload["generation"] += 1
            payload["nav"] = float(payload.get("nav", 1.0)) + 0.001
            _dump(staged / f"{arm}.json", payload)
        self.journal = {"status": "STAGED", "generation": self.generation + 1,
                        "txn": txn, "day": day, "old": old,
                        "staged": str(staged), "promoted": [],
                        "crash_recovery": bool(crash_recovery)}
        _dump(self.journal_path, self.journal)
        for i, arm in enumerate(ARMS):
            if i >= crash_after:
                raise InjectedCrash(f"crash after {i} roots")
            shutil.copy2(staged / f"{arm}.json", self.state / f"{arm}.json")
            self.journal["promoted"].append(arm)
            _dump(self.journal_path, self.journal)
        if crash_after == len(ARMS):
            raise InjectedCrash("crash after all roots, before publish")
        self.journal["status"] = "COMMITTED"
        self.generation += 1
        self.journal["generation"] = self.generation
        _dump(self.journal_path, self.journal)
        self._append_ledger(txn, day)
        return txn

    def recover(self, crash_during_recovery: bool = False) -> dict[str, Any]:
        status = self.journal.get("status")
        if status == "COMMITTED":
            self._append_ledger(self.journal["txn"], self.journal.get("day", ""))
            return {"status": "COMMITTED", "txn": self.journal["txn"], "recomputed": False}
        if status not in ("STAGED", "ABORTING"):
            raise RuntimeError(f"nothing to recover: {status}")
        self.journal["status"] = "ABORTING"
        _dump(self.journal_path, self.journal)
        promoted = list(self.journal.get("promoted", []))
        if crash_during_recovery and promoted:
            raise InjectedCrash("crash during abort")
        for arm in promoted:
            shutil.copy2(self.backups / f"{arm}.json", self.state / f"{arm}.json")
        self.journal["status"] = "ABORTED"
        _dump(self.journal_path, self.journal)
        return {"status": "ABORTED", "txn": self.journal["txn"], "recomputed": False}

    def retry(self, day: str) -> str:
        txn = self.attempt(day, crash_after=99)
        return txn

    def read_visible(self) -> dict[str, Any]:
        if self.journal.get("status") == "COMMITTED":
            return {"generation": self.generation,
                    "roots": {a: _sha(self.state / f"{a}.json") for a in ARMS}}
        return {"generation": self.generation, "roots": "not_published"}


class InjectedCrash(RuntimeError):
    pass


def run_spike(root: Path, days: int = 10) -> dict[str, Any]:
    campaign = FamilyCampaign(root)
    starts: list[float] = []
    crashes: list[dict[str, Any]] = []
    for i in range(days):
        day = f"2026{7 + i // 28:02d}{17 + i % 28:02d}"
        t0 = time.perf_counter()
        crash_after = 1 if i == 2 else (2 if i == 5 else (3 if i == 7 else 99))
        try:
            campaign.attempt(day, crash_after=crash_after)
        except InjectedCrash as exc:
            crashes.append({"day": day, "crash": str(exc), "promoted": len(campaign.journal.get("promoted", []))})
            campaign.recover()
            campaign.retry(day)
        starts.append(time.perf_counter() - t0)
    return {
        "status": "PASS",
        "days": days,
        "ledger_entries": len(campaign.ledger),
        "generation": campaign.generation,
        "crashes": crashes,
        "elapsed_seconds": sum(starts),
        "per_day_seconds": starts,
        "visible": campaign.read_visible(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(f"refusing existing spike root: {root}")
    result = run_spike(root, args.days)
    if args.out:
        _dump(Path(args.out), result)
    print(json.dumps({"status": result["status"], "days": result["days"],
                      "crashes": len(result["crashes"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
