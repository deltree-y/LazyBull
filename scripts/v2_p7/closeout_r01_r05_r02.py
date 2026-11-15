# -*- coding: utf-8 -*-
"""P7 整改收口：候选状态新根（拒绝）+ 台账追加撤回/降级条目（append-only）。

只做两件事：
1. 在新根写 ``candidate_status.json``（旧候选状态根只读保留，不覆盖）；
2. 通过唯一入口 ``PanelDataStore.append_ledger_entry`` 追加一条 ``conclusion``，
   ``supersedes`` 指向被撤回的旧结论，把「P7a 正式通过 + 候选准入」改成
   ``fail``（正式 T2/T3 判据失败）。

不修改任何历史台账行、不覆盖任何冻结根、不产生新的经济结论。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.common.types import DataState, LedgerEntry, TradeDate  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.store_identity import workspace_identity  # noqa: E402

OLD_STATUS = (
    ROOT
    / "data/frozen_reference/v2_p7/p7_idx_in300_p7_candidate_20261114_r1/candidate_status.json"
)
REPORT = ROOT / "docs/reports/p7_r01_r05_r02_fix_20261115.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return Path(path).relative_to(ROOT).as_posix()


def build_status(
    *, f2_root: Path, p4_root: Path, p5_root: Path, t6_root: Path, supersedes: str
) -> Dict[str, Any]:
    verdict = json.loads((f2_root / "verdict.json").read_text(encoding="utf-8"))
    e4 = json.loads((f2_root / "e4.json").read_text(encoding="utf-8"))
    e1 = json.loads((f2_root / "e1.json").read_text(encoding="utf-8"))
    return {
        "schema": "p7_multi_sleeve_candidate_status/v2",
        "candidate_id": "P7-IDX-IN300-CANDIDATE-20261114",
        "status": "rejected_formal_t2t3_fail",
        "status_note": (
            "正式 T2/T3（现行契约 E1~E4 + 独立夏普区间 + 三口径相关上界 + 完整复合门）"
            "判定 fail：候选不获准入，也不是 pending（判据经济失败，不是证据不足）。"
        ),
        "p7a_formal_status": "fail",
        "p7a_multi_sleeve_candidate": "withdrawn",
        "p7b_mechanical": "inconclusive_retained",
        "formal_verdict_ref": rel(f2_root / "verdict.json"),
        "formal_prereg_ref": rel(f2_root / "prereg.json"),
        "supersedes_status_ref": supersedes,
        "reasons": {
            "E1_signal_layer": {
                "statistic": e1["statistic"],
                "threshold": e1["threshold"],
                "p_value": e1["null"]["share_above_statistic"],
                "reading": "组内按 F 取前 6 未显著优于随机 6 只（功效足够，δ_S=2 可检出）",
            },
            "E4_4_utilization": {
                "median": e4["lines"]["E4_4_utilization"]["median"],
                "min_required": e4["lines"]["E4_4_utilization"]["min_required"],
                "reading": "30 万元独立袖在 179 个月中过半月份完全空仓 ⇒ 触发利用率否决线",
            },
            "Q1_status": "fail_economic",
            "Q1_note": (
                "零假设流模拟器 A/A 校准逐位通过（Δ总收益 0.00pp < 2pp、ΔSharpe 0.029 < 0.05）；"
                "L_B=−0.345 低于 Q_0.95(S^0)=0.377，且 MDE=1.0（δ=1 检出率 100%）功效足够 "
                "⇒ 经济失败，不是证据不足（v6 更正：v5 的'冻结'源于 A/A 分母错位一天）"
            ),
        },
        "passing_components": {
            "E2_regime": "pass",
            "T2_combo_ruler": "pass",
            "three_caliber_correlation": "pass",
            "E4_1_crash_drawdown": "pass",
            "E4_2_correlation_upper": "pass",
            "E4_3_capacity": "pass",
            "E4_5_pit_negative_side": "pass",
        },
        "evidence": {
            "formal_t2t3": rel(f2_root / "manifest.json"),
            "p4": rel(p4_root / "summary.json"),
            "p5_accounting": rel(p5_root / "audit.json"),
            "a_reference": rel(t6_root / "manifest.json"),
            "report": rel(REPORT),
        },
        "not_production": True,
        "p6_5_required_for_production": False,
        "boundaries": [
            "本状态只对 IDX-IN300 在冻结历史窗口与现行契约判据成立；不构成生产/实盘/前向/日历验证。",
            "未新增任何收益/相关性/容量/数据门槛；E1~E4 与相关上界均沿用现行契约。",
            "不重开 B0/C1/C3；B0 停投资源决定不受本状态影响。",
            "重开需新候选定义 + 新预登记 + 新 α 决策，不得在本条上就地调阈值。",
        ],
        "verdict_snapshot": {
            "verdict": verdict["verdict"],
            "failed_components": verdict["failed_components"],
            "frozen_components": verdict["frozen_components"],
            "p7a_formal_status": verdict["p7a_formal_status"],
        },
    }


def append_ledger(
    status: Dict[str, Any],
    *,
    f2_root: Path,
    p4_root: Path,
    p5_root: Path,
    t6_root: Path,
    status_root: Path,
    ledger_id: str,
    supersedes: Sequence[str],
) -> str:
    identity = workspace_identity(ROOT)
    data_state = DataState(
        raw_partitions={"daily": "20261029"},
        features_partition_max=TradeDate.from_str("20261029"),
        config_digest="76925839ea3025ff",
        code_digest=f"{str(identity.get('git_commit') or '')[:8]}+wt",
    )
    verdict = json.loads((f2_root / "verdict.json").read_text(encoding="utf-8"))
    e1 = json.loads((f2_root / "e1.json").read_text(encoding="utf-8"))
    e4 = json.loads((f2_root / "e4.json").read_text(encoding="utf-8"))
    e2 = json.loads((f2_root / "e2.json").read_text(encoding="utf-8"))
    t2 = json.loads((f2_root / "t2.json").read_text(encoding="utf-8"))
    q1 = json.loads((f2_root / "q1.json").read_text(encoding="utf-8"))
    corr = json.loads((f2_root / "correlation.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "title": (
            "IDX-IN300 正式 T2/T3 判定 fail（判定口径更正后）："
            "E1/E4/Q1 三项经济失败，撤回旧 P5「P7a 正式通过 + 候选准入」语义"
        ),
        "hypothesis_id": "H-P7-IDX-IN300-P5-20261114",
        "verdict": "fail",
        "result_metrics": {
            "formal_verdict": verdict["verdict"],
            "p7a_formal_status": verdict["p7a_formal_status"],
            "failed_components": verdict["failed_components"],
            "frozen_components": verdict["frozen_components"],
            "E1_statistic": e1["statistic"],
            "E1_threshold": e1["threshold"],
            "E1_p_value": e1["null"]["share_above_statistic"],
            "E2_annualized_pp": e2["annualized_pp"],
            "E2_lower_pp": e2["annualized_lower_pp"],
            "T2_delta_maxdd_pp": t2["delta_maxdd_pp"],
            "T2_delta_maxdd_lower_pp": t2["delta_maxdd_lower_pp"],
            "T2_delta_cagr_pp": t2["delta_cagr_pp"],
            "corr_upper_max": corr["upper_max"],
            "E4_4_utilization_median": e4["lines"]["E4_4_utilization"]["median"],
            "Q1_L_B": q1.get("L_B"),
            "Q1_null_q95": q1.get("Q_095_null"),
            "Q1_aa_delta_total_return": (q1.get("aa_calibration") or {}).get(
                "delta_total_return"
            ),
            "Q1_aa_delta_sharpe": (q1.get("aa_calibration") or {}).get("delta_sharpe"),
            "Q1_mde": (q1.get("mde") or {}).get("mde"),
            "Q1_power_ok": (q1.get("mde") or {}).get("power_ok"),
            "candidate_status": status["status"],
        },
        "evidence_refs": [
            rel(f2_root / "prereg.json"),
            rel(f2_root / "verdict.json"),
            rel(f2_root / "e1.json"),
            rel(f2_root / "e4.json"),
            rel(p4_root / "summary.json"),
            rel(p5_root / "audit.json"),
            rel(t6_root / "manifest.json"),
            rel(status_root / "candidate_status.json"),
            rel(REPORT),
        ],
        "supersedes": list(supersedes),
        "notes": (
            "正式 T2/T3 按现行契约（E1~E4、独立夏普区间、三口径相关上界、完整复合门）"
            "在预登记（f2 根 prereg.json）下执行：E1 显著性失败、E4-4 利用率否决、"
            "Q1 夏普下界低于零假设流 p95 且功效足够，三项均属经济失败，verdict=fail。"
            "本行同时更正 v5 的两个判定口径实现错误：①A/A 校准分母错位一天（模拟器首行为"
            "首日收盘 NAV，误用 nav[0] 作分母使 Δ总收益虚增 4.54pp，Q1 被误判为冻结）；"
            "②e4 组件缺 estimable 字段被误判为不可估计（E4 实际是 veto）。更正后 "
            "frozen_components 由 [E4, Q1_aa_calibration] 变为 []，failed_components 由 "
            "[E1] 变为 [E1, E4, Q1]。结论方向不变（仍 fail、候选仍 rejected），"
            "但定性由「1 失败 + 2 证据不足」变为「3 失败 + 0 冻结」。"
            "本条撤回旧结论的「P7a 正式通过 + 候选准入」语义并降级候选为 rejected；"
            "预登记为机器件（registered_at 早于全部组件产物），台账行 append 于判定之后，"
            "属诚实补记，不声称台账预登记先行；α=0.05 已由该预登记锁定。"
            "不重开 B0/C1/C3，不改任何收益/相关/容量阈值。"
        ),
        "alpha_consumed": True,
    }
    store = PanelDataStore(ROOT / "data")
    store.append_ledger_entry(
        LedgerEntry(
            hypothesis_id=ledger_id,
            kind="conclusion",
            payload=payload,
            data_state=data_state,
            registered_at=datetime.now().isoformat(timespec="seconds"),
        )
    )
    return ledger_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P7 整改收口（候选状态 + 台账撤回）")
    parser.add_argument("--f2-root", required=True)
    parser.add_argument("--p4-root", required=True)
    parser.add_argument("--p5-root", required=True)
    parser.add_argument("--t6-root", required=True)
    parser.add_argument("--status-root", required=True)
    parser.add_argument("--ledger-id", required=True)
    parser.add_argument("--supersedes-ledger", action="append", default=[])
    parser.add_argument("--supersedes-status", default=rel(OLD_STATUS))
    parser.add_argument("--dry-run", action="store_true", help="只写状态根，不追加台账")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    f2_root = Path(args.f2_root) if Path(args.f2_root).is_absolute() else ROOT / args.f2_root
    p4_root = Path(args.p4_root) if Path(args.p4_root).is_absolute() else ROOT / args.p4_root
    p5_root = Path(args.p5_root) if Path(args.p5_root).is_absolute() else ROOT / args.p5_root
    t6_root = Path(args.t6_root) if Path(args.t6_root).is_absolute() else ROOT / args.t6_root
    status_root = (
        Path(args.status_root) if Path(args.status_root).is_absolute() else ROOT / args.status_root
    )
    if status_root.exists():
        raise FileExistsError(f"状态根已存在（禁止覆盖）：{status_root}")
    if not (f2_root / "verdict.json").exists():
        raise FileNotFoundError("正式 T2/T3 verdict 缺件：先运行 f2/run_f2.py")
    status = build_status(
        f2_root=f2_root,
        p4_root=p4_root,
        p5_root=p5_root,
        t6_root=t6_root,
        supersedes=args.supersedes_status,
    )
    digest = write_json(status_root / "candidate_status.json", status)
    manifest = {
        "schema": "p7_multi_sleeve_candidate_status_manifest/v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": status["candidate_id"],
        "status": status["status"],
        "written_files_sha256": {"candidate_status.json": digest},
        "inputs_sha256": {
            rel(f2_root / "verdict.json"): sha256(f2_root / "verdict.json"),
            rel(f2_root / "prereg.json"): sha256(f2_root / "prereg.json"),
            rel(p4_root / "summary.json"): sha256(p4_root / "summary.json"),
            rel(p5_root / "audit.json"): sha256(p5_root / "audit.json"),
            rel(t6_root / "manifest.json"): sha256(t6_root / "manifest.json"),
        },
        "not_production": True,
        "not_forward_evidence": True,
        "not_real_clock_evidence": True,
        "not_production_authorization": True,
    }
    write_json(status_root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps({"status": status["status"], "root": rel(status_root), "dry_run": True}))
        return 0
    ledger_id = append_ledger(
        status,
        f2_root=f2_root,
        p4_root=p4_root,
        p5_root=p5_root,
        t6_root=t6_root,
        status_root=status_root,
        ledger_id=args.ledger_id,
        supersedes=args.supersedes_ledger,
    )
    print(
        json.dumps({"status": status["status"], "ledger": ledger_id, "root": rel(status_root)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
