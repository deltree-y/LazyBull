# -*- coding: utf-8 -*-
"""P7 正式 T2/T3（IDX-IN300）：预登记 → 组件 → 复合门 → 证据。

用法::

    python scripts/v2_p7/f2/run_f2.py \
        --p4-root data/frozen_reference/v2_p7/p7_idx_in300_p4_XXXX \
        --a-nav <T6 根>/arms/vvh/daily_vvh.parquet --out <新根>

F29 分型（草案 v5）：``--sleeve-type enhancement``（型 I，默认，已接通）；
``risk_adjusted``（型 II）**入口尚未接通**，当前显式拒绝（先完成校准计划与独立验证再做接线）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p7.f2 import ALPHA_TOTAL, CORE_CELLS, LAMBDA_FROZEN  # noqa: E402

DEFAULT_A_NAV = str(
    ROOT / "data/paper_shadow/p7_testbed/p7_t6_20261115_r3/arms/vvh/daily_vvh.parquet"
)
CRASH_SPEC = str(
    ROOT / "data/paper_shadow/p7_testbed/p7_t05_20261111_r6/specs/crash_windows_v6.json"
)
#: 判据文本模板（数据驱动；改判据只改 JSON，不在代码里硬编码）
GATES_TEMPLATE = Path(__file__).resolve().parent / "prereg_gates_template.json"
CODE_FILES = (
    "scripts/v2_p7/f2/run_f2.py",
    "scripts/v2_p7/f2/formal.py",
    "scripts/v2_p7/f2/panel.py",
    "scripts/v2_p7/f2/__init__.py",
    "scripts/v2_p7/p4/data.py",
    "scripts/v2_p7/p4/engine.py",
    "scripts/v2_p7/p4/metrics.py",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return Path(path).relative_to(ROOT).as_posix()


def build_prereg(
    *,
    p4_root: Path,
    a_nav: Path,
    crash_spec: Path,
    run_id: str,
    sleeve_type: str = "enhancement",
    pool: int = 1,
) -> Dict[str, Any]:
    """预登记（判据常量 + 输入身份；**不含任何结果读数**）。

    ``pool`` 为候选池分母，参与 Bonferroni：``alpha_c = 0.05 / pool``（F29 选项 A）。
    """
    gates = json.loads(GATES_TEMPLATE.read_text(encoding="utf-8"))
    alpha_c = ALPHA_TOTAL / int(pool)
    quantile = 1.0 - alpha_c / (CORE_CELLS * LAMBDA_FROZEN)
    e1 = dict(gates.get("E1_signal_layer") or {})
    e1.update(
        {
            "alpha_component": alpha_c,
            "quantile": quantile,
            "candidate_pool_denominator": int(pool),
        }
    )
    gates["E1_signal_layer"] = e1
    return {
        "schema": "p7_f2_prereg/v1",
        "prereg_id": f"p7-idx-in300-f2-{run_id}",
        "run_id": run_id,
        "candidate": "IDX-IN300",
        "sleeve_type": sleeve_type,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "alpha_accounting": {
            "candidate_count_P": 1,
            "candidate_pool_denominator": int(pool),
            "alpha_total": ALPHA_TOTAL,
            "alpha_c": alpha_c,
            "alpha_consumed": True,
            "consumed_by": "本条预登记（登记即锁定；P-1 不消费 α）",
            "sequential_reads": ["P-1", "T2", "T3"],
            "note": "P-1/T2/T3 是同一历史的顺序读取，不重置 α、不称独立验证",
        },
        "candidate_definition": {
            "source": "P7-IDX-IN300（P3 预登记冻结算法，未改选股/规模/费用/持有期）",
            "account": "30 万元、6 槽、T+1 开盘买、flow_date 收盘卖、实际现金/整手/费用",
            "corporate_actions": "除权日应收 + 到账日现金 + 送转并入原 lot（P3 §2.4 税后口径）",
            "excluded": "临时调整、剔除侧、中证500/1000、其他持有期（P3 §1.1）",
        },
        "gates": gates,
        "verdict_domain": ["pass", "fail", "inconclusive_frozen"],
        "pass_extra_requirement": (
            "用户组织的独立存活评审通过（本程序不替代；未完成则不得标记 P7a 达成）"
        ),
        "inputs": _prereg_inputs(p4_root=p4_root, a_nav=a_nav, crash_spec=crash_spec),
        "boundaries": {
            "historical_evidence_level": "H",
            "historical_virtual_clock": True,
            "not_forward_evidence": True,
            "not_real_clock_evidence": True,
            "not_production_authorization": True,
            "does_not_reopen_b0_c1_c3": True,
            "p65_not_prerequisite": True,
            "note": (
                "本预登记不构成生产多袖、实盘、前向/日历验证或 B 区转正；"
                "只回答 IDX-IN300 是否满足现行 T2/T3 经济准入。"
            ),
        },
    }


def _prereg_inputs(*, p4_root: Path, a_nav: Path, crash_spec: Path) -> Dict[str, Any]:
    """输入身份（逐件 SHA；不含任何读数）。"""
    return {
        "p4_root": p4_root.relative_to(ROOT).as_posix(),
        "p4_summary_sha256": sha256(p4_root / "summary.json"),
        "p4_fills_sha256": (
            sha256(p4_root / "fills.csv") if (p4_root / "fills.csv").exists() else None
        ),
        "p4_utilization_sha256": (
            sha256(p4_root / "daily_utilization.parquet")
            if (p4_root / "daily_utilization.parquet").exists()
            else None
        ),
        "a_nav": a_nav.relative_to(ROOT).as_posix(),
        "a_nav_sha256": sha256(a_nav),
        "crash_spec": crash_spec.relative_to(ROOT).as_posix(),
        "crash_spec_sha256": sha256(crash_spec),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
    }


def _load_inputs(*, p4_root: Path, a_nav: Path, crash_spec: Path) -> Dict[str, Any]:
    """读取冻结输入（只读；缺件即抛）。"""
    import pandas as pd

    summary = json.loads((p4_root / "summary.json").read_text(encoding="utf-8"))
    candidate_levels = pd.read_parquet(p4_root / "candidate_daily_nav.parquet")["candidate_nav"]
    benchmark_levels = pd.read_parquet(p4_root / "benchmark_daily_nav.parquet")["benchmark_nav"]
    table3 = pd.read_parquet(p4_root / "table3_daily_nav.parquet")
    a_frame = pd.read_parquet(a_nav, columns=["trade_date", "nav"])
    a_frame["trade_date"] = a_frame["trade_date"].astype(str).str.replace("-", "")
    a_nav_series = (
        a_frame.drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")["nav"]
        .astype(float)
    )
    return {
        "summary": summary,
        "candidate_levels": candidate_levels,
        "benchmark_levels": benchmark_levels,
        "table3": table3,
        "a_nav_series": a_nav_series,
        "crash_windows": json.loads(crash_spec.read_text(encoding="utf-8"))["windows"],
        "candidate_returns": candidate_levels.pct_change().dropna().to_numpy(dtype=float),
        "start_date": str(candidate_levels.index[0]),
        "common_start": max(str(candidate_levels.index[0]), str(a_nav_series.index[0])),
        "common_end": min(str(candidate_levels.index[-1]), str(a_nav_series.index[-1])),
    }


def _compute_components(
    inputs: Mapping[str, Any],
    *,
    p4_root: Path,
    skip_null: bool,
    quantile: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    """逐组件计算（E1 → Q1 → E2 → 相关 → T2 → E4）。"""
    from scripts.v2_p7.f2 import formal
    from scripts.v2_p7.f2.panel import build_panels
    from scripts.v2_p7.p4.data import P4Data
    from scripts.v2_p7.p4.engine import prepare_plans

    candidate_levels = inputs["candidate_levels"]
    data = P4Data()
    plans, _cache = prepare_plans(data)
    panels = build_panels(data, plans)
    components: Dict[str, Dict[str, Any]] = {
        "E1": formal.e1_component(panels, quantile=quantile)
    }
    if skip_null:
        components["Q1"] = {"estimable": False, "reason": "skip_null_smoke"}
    else:
        components["Q1"] = formal.q1_sharpe(
            panels,
            real_returns=inputs["candidate_returns"],
            principal=float(inputs["summary"]["table1"]["metrics"]["start_nav"]),
            calendar=[str(day) for day in candidate_levels.index],
        )
    components["E2"] = formal.e2_regime(
        candidate_levels, inputs["benchmark_levels"], inputs["start_date"]
    )
    components["correlation"] = formal.three_correlations(
        inputs["a_nav_series"],
        candidate_levels,
        inputs["crash_windows"],
        common_start=inputs["common_start"],
        common_end=inputs["common_end"],
    )
    components["T2"] = formal.t2_combo_ruler(inputs["table3"])
    components["E4"] = formal.e4_vetoes(
        p4_root=p4_root,
        table3_levels=inputs["table3"],
        crash_windows=inputs["crash_windows"],
        correlations=components["correlation"],
        negative_side_boundary=(
            "仅使用沪深300定期调整的调入方向；无坏消息侧排序，不做个股退出；"
            "PIT 可得性由公告日 + 事件表 SHA 归因（P3 §1.1）"
        ),
    )
    return components


def _build_verdict(
    *,
    gate: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    prereg: Mapping[str, Any],
    prereg_sha: str,
    run_id: str,
    alpha_c: float,
    pool: int,
) -> Dict[str, Any]:
    """组装 verdict 文件（型 I 口径；型 II 接线后另行扩展）。"""
    return {
        "schema": "p7_f2_verdict/v1",
        "run_id": run_id,
        "prereg_id": prereg["prereg_id"],
        "prereg_sha256": prereg_sha,
        "candidate": "IDX-IN300",
        "sleeve_type": gate.get("sleeve_type"),
        "verdict": gate["verdict"],
        "verdict_domain": ["pass", "fail", "inconclusive_frozen"],
        "frozen_components": gate["frozen_components"],
        "failed_components": gate["failed_components"],
        "machine_pass": bool(gate["verdict"] == "pass"),
        "pending_independent_review": bool(gate["pending_independent_review"]),
        "p7a_formal_status": (
            "pending_independent_review" if gate["verdict"] == "pass" else gate["verdict"]
        ),
        "alpha_consumed": True,
        "alpha_c": alpha_c,
        "candidate_pool_denominator": int(pool),
        "gate_rule": gate["rule"],
        "components": {
            name: {
                "estimable": bool(item.get("estimable", False)),
                "pass": item.get("pass"),
                "state": item.get("state"),
                "calibrated": item.get("calibrated"),
            }
            for name, item in components.items()
        },
        "boundaries": {
            "historical_evidence_level": "H",
            "not_forward_evidence": True,
            "not_real_clock_evidence": True,
            "not_production_authorization": True,
            "does_not_reopen_b0_c1_c3": True,
            "not_p65_prerequisite": True,
            "note": (
                "本 verdict 只对 IDX-IN300 在冻结历史窗口与现行契约判据成立；"
                "pass 也需用户组织的独立存活评审，未完成前不得标记 P7a 正式达成。"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P7 正式 T2/T3（IDX-IN300）")
    parser.add_argument("--p4-root", required=True)
    parser.add_argument("--a-nav", default=DEFAULT_A_NAV)
    parser.add_argument("--crash-spec", default=CRASH_SPEC)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-null", action="store_true", help="冒烟：跳过 Q1 零假设流")
    parser.add_argument(
        "--sleeve-type",
        default="enhancement",
        choices=["enhancement", "risk_adjusted"],
        help="袖型（F29）：enhancement=型 I，risk_adjusted=型 II（未接通）",
    )
    parser.add_argument("--pool", type=int, default=1, help="候选池分母（参与 α/pool）")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if int(args.pool) < 1:
        raise SystemExit("--pool 必须 ≥1（禁止把非法分母静默转成最宽松预算）")
    if args.sleeve_type != "enhancement":
        raise SystemExit(
            "型 II（risk_adjusted）入口尚未接通：需先完成两套门的校准计划、独立验证与落盘接线；"
            "当前仅支持 --sleeve-type enhancement"
        )

    p4_root = _resolve(args.p4_root)
    a_nav = _resolve(args.a_nav)
    crash_spec = _resolve(args.crash_spec)
    out = _resolve(args.out)
    if out.exists():
        raise FileExistsError(f"证据根已存在（禁止覆盖）：{out}")
    run_id = args.run_id or f"f2-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    pool = int(args.pool)
    alpha_c = ALPHA_TOTAL / pool
    quantile = 1.0 - alpha_c / (CORE_CELLS * LAMBDA_FROZEN)

    out.mkdir(parents=True, exist_ok=False)
    prereg = build_prereg(
        p4_root=p4_root,
        a_nav=a_nav,
        crash_spec=crash_spec,
        run_id=run_id,
        sleeve_type=args.sleeve_type,
        pool=pool,
    )
    prereg_sha = write_json(out / "prereg.json", prereg)
    written: Dict[str, Any] = {
        "prereg.json": {"sha256": prereg_sha, "bytes": (out / "prereg.json").stat().st_size}
    }

    os.environ.setdefault("P7_A_NAV", str(a_nav))
    from scripts.v2_p7.f2 import formal

    inputs = _load_inputs(p4_root=p4_root, a_nav=a_nav, crash_spec=crash_spec)
    components = _compute_components(
        inputs, p4_root=p4_root, skip_null=bool(args.skip_null), quantile=quantile
    )
    gate = formal.composite_gate(components)
    verdict = _build_verdict(
        gate=gate,
        components=components,
        prereg=prereg,
        prereg_sha=prereg_sha,
        run_id=run_id,
        alpha_c=alpha_c,
        pool=pool,
    )
    for name in ("E1", "E2", "T2", "Q1", "correlation", "E4"):
        path = out / f"{name.lower()}.json"
        digest = write_json(path, {"schema": f"p7_f2_{name.lower()}/v1", **components[name]})
        written[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    verdict_sha = write_json(out / "verdict.json", verdict)
    verdict_bytes = (out / "verdict.json").stat().st_size
    written["verdict.json"] = {"sha256": verdict_sha, "bytes": verdict_bytes}
    manifest = {
        "schema": "p7_f2_manifest/v1",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sleeve_type": args.sleeve_type,
        "prereg_sha256": prereg_sha,
        "verdict_sha256": verdict_sha,
        "alpha_consumed": True,
        "written_files_sha256": written,
        "coverage_gaps": [],
        "not_production": True,
        "not_forward_evidence": True,
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "verdict": verdict["verdict"],
                "frozen": verdict["frozen_components"],
                "failed": verdict["failed_components"],
                "p7a": verdict["p7a_formal_status"],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
