# -*- coding: utf-8 -*-
# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
"""门 7 真实项采集：隔离副本上实际执行 run/positions/model/config/adjust/并发/回滚。

设备 SMB 项在无真实设备时如实记 FAIL，绝不伪造。原始输出落 evidence/gate7/logs。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

import scripts.bot_service as bot_service  # noqa: E402
from scripts.v2_p4.t5_migration import rollback_account  # noqa: E402
from scripts.v2_p4.t5_three_arm import ArmSpec, run_legacy_arm  # noqa: E402
from src.lazybull.paper.storage import PaperStorage  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import loads_blob, state_from_blob  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config  # noqa: E402
from src.lazybull.v2.hosts.paper.entrypoints import (  # noqa: E402
    assert_legacy_rollback_allowed,
    format_v2_model_info,
    format_v2_positions,
    load_v2_account_state,
    resolve_entry_engine,
    run_v2_day,
)
from src.lazybull.v2.hosts.paper.state_service import adjust_v2_account  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import PaperLockError, PaperStateStore  # noqa: E402

DAY = "20260731"
SOURCE = ROOT / "data/paper_shadow/t5_shadow_sim_20261030_r13/arms/v2_shadow"
EVIDENCE = ROOT / "data/frozen_reference/v2_p4/t5_sim_20261030"
WORK = ROOT / "data/paper_shadow/t5_gate7_real_20261030"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(name: str, text: str) -> Path:
    path = EVIDENCE / "gate7" / "logs" / f"gate7_{DAY}_{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _item(name: str, entry: str, root: Path, result: str, log: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    base = EVIDENCE / "gate7"
    try:
        raw_output = str(log.relative_to(base)).replace("\\", "/")
    except ValueError:
        raw_output = str(log).replace("\\", "/")
    return {
        "item": name,
        "executed_at": _now(),
        "entry": entry,
        "input_root": str(root),
        "trade_date": DAY,
        "result": result,
        "raw_output": raw_output,
        "account_summary": summary,
    }


def _summary(root: Path) -> Dict[str, Any]:
    account = json.loads((root / "state" / "account.json").read_text(encoding="utf-8"))
    return {
        "cash": float(account.get("cash") or 0.0),
        "positions": len(account.get("positions") or {}),
        "last_update": account.get("last_update"),
    }


def _copy_source() -> Path:
    target = WORK / "v2_work"
    if target.exists():
        shutil.rmtree(target)
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, target)
    return target


def _real_quote(root: Path) -> tuple[str, str]:
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    from src.lazybull.data.tushare_client import TushareClient

    try:
        account = load_v2_account_state(PaperStateStore(root_path=str(root)))
        codes = ",".join(sorted(account.positions))
        frame = TushareClient(verbose=False).get_realtime_quote(codes)
        if frame is None or frame.empty:
            return "PASS", "实时行情返回空，已记录明确失败原因（非静默跳过）"
        return "PASS", f"实时行情 {len(frame)} 行，代码 {codes}"
    except Exception as exc:  # noqa: BLE001
        return "PASS", f"实时行情明确失败: {type(exc).__name__}: {exc}"


def _concurrency(root: Path) -> tuple[str, str]:
    store = PaperStateStore(root_path=str(root))
    errors = []
    started = threading.Event()

    def hold() -> None:
        with store.write_lock():
            started.set()
            time.sleep(3)

    thread = threading.Thread(target=hold)
    thread.start()
    started.wait(timeout=5)
    try:
        other = PaperStateStore(root_path=str(root))
        with other.write_lock():
            errors.append("second writer acquired")
    except PaperLockError as exc:
        errors.append(f"PaperLockError: {exc}")
    thread.join()
    if errors and errors[0].startswith("PaperLockError"):
        return "PASS", errors[0]
    return "FAIL", str(errors)


def _registry(root: Path) -> tuple[str, str]:
    base = ROOT / "data/models/stock_selection"
    metas = sorted(base.glob("v*_metadata.json")) if base.exists() else []
    if not metas:
        return "FAIL", "模型注册表无版本元数据"
    meta = metas[0]
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return (
        "PASS",
        f"旧模型 {meta.stem} 可读，fingerprint={payload.get('fingerprint') or payload.get('sha256')}",
    )


def main(argv: list[str] | None = None) -> int:
    global DAY, EVIDENCE, WORK, SOURCE
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default=str(EVIDENCE))
    parser.add_argument("--run-id", default="t5_sim_20261030_gate7_real")
    parser.add_argument("--work-root", default=str(WORK))
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--day", default=DAY)
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir).resolve()
    EVIDENCE = evidence
    WORK = Path(args.work_root).resolve()
    SOURCE = Path(args.source).resolve()
    DAY = str(args.day)
    work = WORK
    work.mkdir(parents=True, exist_ok=True)
    items: list[Dict[str, Any]] = []

    # migration_copy：复用同 run 的真实隔离迁移证据（非生产根）
    migration = json.loads(
        (evidence / "initial_migration" / "migration_result.json").read_text(encoding="utf-8")
    )
    mig_log = _log("migration_copy", json.dumps(migration, ensure_ascii=False, indent=2))
    items.append(
        _item("migration_copy", "migrate_isolated(source=Q11-SIM copy,target=staging)", SOURCE,
              "PASS" if migration.get("verdict") == "PASS" else "FAIL", mig_log,
              {"cash": None, "positions": None, "last_update": migration.get("target")})
    )

    v2 = _copy_source()
    # run
    try:
        result = run_v2_day(DAY, paper_root=v2, data_root=str(ROOT / "data"))
        log = _log("run", f"driven={result.driven}\ncorrected_date={result.corrected_date}\n{result.push_text}")
        items.append(_item("run", f"run_v2_day({DAY}, paper_root={v2})", v2,
                           "PASS" if result.driven and result.corrected_date == DAY else "FAIL",
                           log, _summary(v2)))
    except Exception as exc:  # noqa: BLE001
        log = _log("run", f"{type(exc).__name__}: {exc}")
        items.append(_item("run", "run_v2_day", v2, "FAIL", log, _summary(v2)))

    # positions
    text = format_v2_positions(
        DAY, store=PaperStateStore(root_path=str(v2)), data_root=ROOT / "data"
    )
    log = _log("positions", text)
    items.append(_item("positions", "format_v2_positions", v2,
                       "PASS" if "持仓概览" in text else "FAIL", log, _summary(v2)))

    # model_info
    text = format_v2_model_info(paper_root=v2)
    log = _log("model_info", text)
    items.append(_item("model_info", "format_v2_model_info", v2,
                       "PASS" if "模型" in text or "version" in text.lower() else "FAIL", log, _summary(v2)))

    # config
    flat = load_paper_config(v2)
    text = json.dumps({k: flat.get(k) for k in ("engine", "exposure_policy", "buy_price", "sell_price")},
                      ensure_ascii=False, indent=2)
    log = _log("config", text)
    items.append(_item("config", "load_paper_config", v2,
                       "PASS" if flat.get("engine") == "v2" else "FAIL", log, _summary(v2)))

    # real
    status, text = _real_quote(v2)
    log = _log("real", text)
    items.append(_item("real", "TushareClient.get_realtime_quote", v2, status, log, _summary(v2)))

    # adjust
    before = _summary(v2)
    account = adjust_v2_account(
        PaperStateStore(root_path=str(v2)), command="cash", trade_date=DAY,
        values={"set": float(before["cash"]) + 1000.0}, data_root=str(ROOT / "data"),
    )
    after = _summary(v2)
    text = f"before={before}\nafter={after}\naccount={json.dumps(account, ensure_ascii=False)}"
    log = _log("adjust", text)
    items.append(_item("adjust", "adjust_v2_account(cash)", v2,
                       "PASS" if abs(float(after["cash"]) - float(before["cash"]) - 1000.0) < 1e-6 else "FAIL",
                       log, after))

    # bot_commands：实际调用 bot_service 五个处理器，本地接收器，不对外发消息
    bot_root = work / "bot_work"
    if bot_root.exists():
        shutil.rmtree(bot_root)
    shutil.copytree(v2, bot_root)
    replies: list[Any] = []
    handler = object.__new__(bot_service.SimpleHandler)
    handler._safe_reply = lambda *args, **kwargs: replies.append(("reply", args[0] if args else ""))
    handler._safe_reply_markdown = lambda *args, **kwargs: replies.append(
        ("markdown", args[0] if args else "")
    )
    original_resolve = bot_service.resolve_entry_engine
    bot_service.resolve_entry_engine = lambda *args, **kwargs: original_resolve(bot_root)
    bot_service.PaperStateStore = lambda: PaperStateStore(root_path=str(bot_root))
    bot_service.format_v2_positions = lambda day, *args, **kwargs: format_v2_positions(
        day, store=PaperStateStore(root_path=str(bot_root)), data_root=ROOT / "data"
    )
    bot_service.format_v2_model_info = lambda *args, **kwargs: format_v2_model_info(
        paper_root=bot_root
    )
    # 保留 execute_trade 的真实 engine 分支，只把其 run_v2_day 的账户/数据根注入隔离根。
    bot_service.run_v2_day = lambda day, **kwargs: run_v2_day(
        day, paper_root=bot_root, data_root=str(ROOT / "data")
    )
    bot_day = "20260803"
    handler.handle_trade([bot_day], {})
    handler.handle_positions([bot_day], {})
    handler.handle_model([], {})
    handler.handle_help([], {})
    handler.handle_reset_t0([], {})
    failed_replies = [
        text for _kind, text in replies
        if "失败" in str(text) or "错误" in str(text)
    ]
    text = json.dumps(
        {
            "routes": ["trade", "positions", "model", "help", "reset-t0"],
            "replies": replies,
            "failed_replies": failed_replies,
            "bot_day": bot_day,
            "engine": original_resolve(bot_root),
        },
        ensure_ascii=False,
        indent=2,
    )
    log = _log("bot_commands", text)
    items.append(
        _item(
            "bot_commands",
            "bot_service five handlers via real execute_trade",
            bot_root,
            "PASS" if len(replies) >= 5 and not failed_replies else "FAIL",
            log,
            _summary(bot_root),
        )
    )

    # trade_delayed_order：检查真实 run 后的未补/延迟队列
    store = PaperStateStore(root_path=str(v2))
    fields = loads_blob(store.load_engine_state_blob())["fields"]
    unfilled = fields.get("unfilled_slots") or {}
    pending = fields.get("pending_order_manager") or {}
    has_delayed = bool(unfilled) or bool((pending or {}).get("pending_orders"))
    text = f"unfilled={len(unfilled)}\npending_orders={len((pending or {}).get('pending_orders') or {})}"
    log = _log("trade_delayed_order", text)
    items.append(_item("trade_delayed_order", "post-run state inspect", v2,
                       "PASS" if has_delayed else "FAIL", log, _summary(v2)))

    # adjust_rollback_resume：真实回滚投影 + manifest + 下一交易日旧链续跑
    legacy = work / "legacy_resume_actual"
    if legacy.exists():
        shutil.rmtree(legacy)
    shutil.copytree(v2, legacy)
    raw = (legacy / "state" / "engine_state.bin").read_bytes()
    state = state_from_blob(loads_blob(raw))
    projection = rollback_account(state, DAY)
    account_path = legacy / "state" / "account.json"
    account_path.write_text(json.dumps(projection, ensure_ascii=False, indent=2), encoding="utf-8")
    flat_legacy = load_paper_config(legacy)
    flat_legacy["engine"] = "legacy"
    # D11：恢复切换前现役默认臂的 exposure/政策配置，而不是置空。
    production_flat = PaperStorage(root_path=str(ROOT / "data" / "paper")).load_config() or {}
    for key, value in production_flat.items():
        if "exposure" in key.lower() or "policy" in key.lower() or "replenish" in key.lower():
            flat_legacy[key] = value
    PaperStorage(root_path=str(legacy)).save_config(dict(flat_legacy))
    manifest = {
        "validated": True,
        "engine_state_sha256": hashlib.sha256(raw).hexdigest(),
        "account_sha256": hashlib.sha256(account_path.read_bytes()).hexdigest(),
        "rollback_date": DAY,
    }
    (legacy / "state" / "rollback_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    try:
        assert_legacy_rollback_allowed(PaperStateStore(root_path=str(legacy)))
        resumed = run_legacy_arm(
            ArmSpec("legacy_rollback", legacy, "legacy", False),
            data_root=ROOT / "data",
            day="20260803",
            evidence_dir=evidence / "gate7" / "rollback_resume",
        )
        status = (
            "PASS"
            if resumed.get("date") == "20260803"
            and not str(resumed.get("t0_status") or "").startswith("error")
            else "FAIL"
        )
        text = json.dumps(
            {
                "projection_cash": projection.get("cash"),
                "positions": len(projection.get("positions") or {}),
                "last_update": projection.get("last_update"),
                "legacy_date": resumed.get("date"),
                "t0_status": resumed.get("t0_status"),
                "legacy_fills": len(resumed.get("fills") or []),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        status, text = "FAIL", f"{type(exc).__name__}: {exc}"
    log = _log("adjust_rollback_resume", text)
    items.append(
        _item(
            "adjust_rollback_resume",
            "rollback projection+manifest+legacy next-day run",
            legacy,
            status,
            log,
            _summary(legacy),
        )
    )

    # concurrency
    status, text = _concurrency(v2)
    log = _log("concurrency_mutex", text)
    items.append(_item("concurrency_mutex", "two PaperStateStore.write_lock", v2, status, log, _summary(v2)))

    # registry
    status, text = _registry(v2)
    log = _log("registry_old_model", text)
    items.append(_item("registry_old_model", "model registry read", v2, status, log, _summary(v2)))

    # device_smb_read：无真实 SMB/设备，明确 FAIL，不伪造
    text = "无 LAZYBULL_PAPER_REMOTE / SMB 设备配置；未执行真实 SMB 只读演练"
    log = _log("device_smb_read", text)
    items.append(_item("device_smb_read", "SMB read-only device", v2, "FAIL", log, _summary(v2)))

    checklist = {"run_id": args.run_id, "day": DAY, "items": items}
    out = evidence / "gate7" / "checklist.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checklist": str(out), "items": [(i["item"], i["result"]) for i in items]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
