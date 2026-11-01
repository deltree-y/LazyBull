# -*- coding: utf-8 -*-
"""76 日就绪门影子对照（T5 证据工具，B′ 部署判据 B1 的前置证据）。

复杂度豁免：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。

定位
====
在**同一份真实组装结果**（`assemble_daily_data` 产出的 `day_frame` 与
`ReadinessReport`）上，并行施加两套判据并逐日比对：

- **B′（现行代码）**：逐列分布分位 p95/p99 分级 + 聚合升级阻断 + 主板域分母；
- **C（冻结复刻）**：旧生产判据 = 全局 60% + `cons_target_upside` 白名单 +
  全空硬拒，**分母为全行**（旧实现未做域限定）。

为什么必须冻结复刻：切换后生产常量（`COLUMN_MISSING_RATE_MAX` /
`SPARSE_TOLERATED_COLUMNS`）已删除，事后无法再复现旧基线；本工具内嵌旧常量
副本，作为不可回退的对照锚点。

输出
====
- `<out>/shadow_76d.json`：逐日逐列证据（B′/C 判定、双口径率、滚动分位敏感性）；
- `<out>/shadow_76d.md`：可复核表格（逐日表 + 列级频次 + B1 三条核对）；
- 控制台摘要 + 返回码（0 = B1 三条全部满足；1 = 存在未归因/超频列）。

B1 判据（用户裁决 2026-10-31）
==============================
① 阻断日 = 0，或全部阻断日可归因「真实源退化且已修复」；
② error 级告警 ≤ 每列 20% 的交易日（结构性漂移列单列豁免，须逐列登记）；
③ 全量回归 + 复杂度门通过（由调用方另行核对，本工具只输出 ① ② 证据）。

用法
====
``python scripts/v2_p4/t5_coverage_shadow_compare.py``（默认 20260703~20261026）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as stat
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from src.lazybull.common.config import get_data_root, get_stock_selection_models_root  # noqa: E402
from src.lazybull.data import DataLoader, Storage  # noqa: E402
from src.lazybull.v2.common.model_schema import (  # noqa: E402
    FEATURE_MISSING_RATE_ALERT_QUANTILE,
    FEATURE_MISSING_RATE_ERROR_QUANTILE,
    read_drift_exempt,
    read_feature_missing_rate,
)
from src.lazybull.v2.hosts.paper.coverage_gate import (  # noqa: E402
    AGG_ERROR_FRACTION,
    AGG_MIN_ERROR_COLUMNS,
    COVERAGE_RATE_FLOOR,
    classify_severity,
    main_board_codes,
)
from src.lazybull.v2.hosts.paper.daily_assembly import assemble_daily_data  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

# ---------------------------------------------------------------------------
# 旧判据冻结副本（C；生产常量已删除，不得回改生产代码）
# ---------------------------------------------------------------------------
_C_COLUMN_MISSING_RATE_MAX = 0.6
_C_SPARSE_TOLERATED_COLUMNS = frozenset({"cons_target_upside"})

#: B1② 闸值：单列 error 交易日占比上限
_ERROR_DAY_RATIO_MAX = 0.20
#: 滚动分位窗口（与生产常量一致；本工具独立持值避免耦合内部实现）
_ROLLING_WINDOW = 60
_ROLLING_MIN_DAYS = 20


@dataclass
class DayRow:
    date: str
    ok: bool
    blocked: bool
    errors: int
    warnings: int
    rates_domain: Dict[str, float]
    rates_all: Dict[str, float]
    error_columns: List[dict]
    gaps: List[str]
    c_blocked: bool
    c_reasons: List[str]
    error: Optional[str] = None
    n_rows_used: int = 0
    n_rows_total: int = 0
    static_blocked: bool = False
    rolling_blocked: bool = False
    rolling_errors: int = 0
    rolling_source_days: int = 0


def _c_judge(rates_all: Dict[str, float]) -> Tuple[bool, List[str]]:
    """旧 C 判据冻结复刻（全行分母；全局 60% + 白名单 + 全空硬拒）。"""
    exceeded: List[str] = []
    tolerated: List[str] = []
    all_nan: List[str] = []
    for col, rate in rates_all.items():
        if rate >= 1.0:
            all_nan.append(col)
        elif rate > _C_COLUMN_MISSING_RATE_MAX:
            if col in _C_SPARSE_TOLERATED_COLUMNS:
                tolerated.append(f"{col}={rate:.1%}")
            else:
                exceeded.append(f"{col}={rate:.1%}")
    reasons: List[str] = []
    if all_nan:
        reasons.append(f"当日列全空: {sorted(all_nan)}")
    if exceeded:
        reasons.append(f"缺失率超 {_C_COLUMN_MISSING_RATE_MAX:.0%}: {sorted(exceeded)}")
    if tolerated:
        reasons.append(f"白名单放行: {tolerated}")
    return bool(all_nan or exceeded), reasons


def _rolling_severity(
    history: Dict[str, List[float]],
    column: str,
    rate: float,
    exempt: frozenset = frozenset(),
) -> str:
    """滚动分位分级（**与生产同算法**：np.percentile + classify_severity + 豁免封顶）。

    样本 < 最小日数 ⇒ 返回 'static'（调用方按静态基线理解）。
    """
    values = history.get(column) or []
    if len(values) < _ROLLING_MIN_DAYS:
        return "static"
    window = np.asarray(values[-_ROLLING_WINDOW:], dtype=float)
    p95 = float(np.percentile(window, 95.0))
    p99 = float(np.percentile(window, 99.0))
    severity = classify_severity(rate, p95, p99)
    if column in exempt and severity == "error":
        return "warning"
    return severity


def _resolve_days(
    panel_store: PanelDataStore, loader: DataLoader, calendar: List[str], start: str, end: str
) -> List[str]:
    """panel ∩ clean/daily ∩ 日历（区间内，升序）。"""
    panel_dir = panel_store.panel_dir
    panel_dates = (
        {p.name for p in panel_dir.iterdir() if p.is_dir()} if panel_dir.exists() else set()
    )
    daily_dir = Path(loader.storage.clean_path) / "daily"
    clean_dates = (
        {p.stem.replace("-", "") for p in daily_dir.glob("*.parquet")}
        if daily_dir.exists()
        else set()
    )
    return sorted(
        d for d in calendar if start <= d <= end and d in panel_dates and d in clean_dates
    )


def _load_model_refs(models_dir: Path, version: int) -> Tuple[Sequence[str], dict, dict]:
    metadata = json.loads((models_dir / f"v{version}_metadata.json").read_text(encoding="utf-8"))
    features = json.loads((models_dir / f"v{version}_features.json").read_text(encoding="utf-8"))
    train_params = dict(metadata.get("train_params") or {})
    _, error = read_feature_missing_rate(train_params)
    if error is not None:
        raise ValueError(f"v{version} metadata 未通过分布校验（{error}）——先补录再对照")
    return list(features), train_params, metadata


def _all_row_rates(
    feature_columns: Sequence[str], day_frame, target_columns: Sequence[str]
) -> Dict[str, float]:
    """全行口径率（旧 C 判据的分母：所有行，含非主板）。"""
    rates: Dict[str, float] = {}
    for col in target_columns:
        if col in day_frame.columns:
            rates[col] = float(day_frame[col].isna().mean())
    return rates


def run(
    start: str,
    end: str,
    version: int,
    out_dir: Path,
    *,
    fixed_evidence: Optional[Dict[str, str]] = None,
) -> Tuple[List[DayRow], dict]:
    data_root = get_data_root()
    models_dir = Path(get_stock_selection_models_root())
    feature_columns, train_params, metadata = _load_model_refs(models_dir, version)
    exempt = read_drift_exempt(train_params)
    storage = Storage(root_path=data_root)
    loader = DataLoader(storage)
    panel_store = PanelDataStore(data_root)
    stock_basic = loader.load_clean_stock_basic()
    domain = main_board_codes(stock_basic)
    if domain is None:
        raise ValueError("stock_basic 缺失 market 列，无法构建主板域（拒绝跨域对照）")
    trade_cal = loader.load_clean_trade_cal()
    calendar = [str(day) for day in trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"].tolist()]
    days = _resolve_days(panel_store, loader, calendar, start, end)
    logger.info(f"对照窗口: {start}~{end}，可用交易日 {len(days)}（panel∩clean∩日历）")

    rows: List[DayRow] = []
    history: Dict[str, List[float]] = {}
    for index, day in enumerate(days, start=1):
        try:
            assembly = assemble_daily_data(
                day,
                panel_store,
                loader,
                calendar,
                feature_columns,
                train_params=train_params,
                coverage_domain_codes=domain,
                coverage_rate_history=None,  # 静态基线；滚动敏感性离线复算
            )
            report = assembly.readiness
            day_frame = assembly.features_by_date.get(day)
            rates_domain = dict(report.column_missing_rates)
            rates_all = (
                _all_row_rates(feature_columns, day_frame, list(rates_domain))
                if day_frame is not None
                else {}
            )
            c_blocked, c_reasons = _c_judge(rates_all or rates_domain)
            error_alerts = [a for a in report.coverage_alerts if a.get("severity") == "error"]
            # reviewer3 P1-6/R2-P1-2：完整保存（不截断）；uncovered 等无 rate 的
            # 列级 error 也必须保留（rate=None），否则列级频次漏计
            error_columns = [
                {
                    "column": a.get("column"),
                    "rate": a.get("rate"),
                    "code": a.get("code"),
                    "member": a.get("member"),
                }
                for a in error_alerts
                if a.get("column")
            ]
            row = DayRow(
                date=day,
                ok=bool(report.ok),
                blocked=bool(report.coverage_blocked),
                errors=len(error_alerts),
                warnings=len([a for a in report.coverage_alerts if a.get("severity") == "warning"]),
                rates_domain=rates_domain,
                rates_all=rates_all,
                error_columns=error_columns,
                gaps=list(report.gaps),
                c_blocked=c_blocked,
                c_reasons=c_reasons,
                n_rows_used=int(report.n_rows_used),
                n_rows_total=int(report.n_rows_total),
            )
            # 滚动分位敏感性：用**前序**日率（不含当日）模拟在线自校准
            rolling_errors = sum(
                1
                for col, rate in rates_domain.items()
                if _rolling_severity(history, col, rate, exempt) == "error"
            )
            row.rolling_errors = rolling_errors
            row.rolling_source_days = len(next(iter(history.values()))) if history else 0
            threshold = max(15, int(-(-len(rates_domain) * 0.20 // 1)))
            row.static_blocked = row.blocked
            row.rolling_blocked = rolling_errors >= threshold and row.n_rows_used >= 1000
            rows.append(row)
            for col, rate in rates_domain.items():
                history.setdefault(col, []).append(rate)
        except Exception as exc:  # 单日组装失败如实登记，不伪造
            rows.append(
                DayRow(
                    date=day,
                    ok=False,
                    blocked=False,
                    errors=0,
                    warnings=0,
                    rates_domain={},
                    rates_all={},
                    error_columns=[],
                    gaps=[],
                    c_blocked=False,
                    c_reasons=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        if index % 10 == 0:
            logger.info(f"对照进度 {index}/{len(days)}")

    meta_version = metadata.get("version")
    if meta_version is not None and int(meta_version) != int(version):
        raise ValueError(f"证据身份不一致：--version={version} 但 metadata 版本={meta_version}")
    identity = {
        "model_version": int(version),
        "rate_floor": COVERAGE_RATE_FLOOR,
        "agg_min_error_columns": AGG_MIN_ERROR_COLUMNS,
        "agg_error_fraction": AGG_ERROR_FRACTION,
        "quantiles": [FEATURE_MISSING_RATE_ALERT_QUANTILE, FEATURE_MISSING_RATE_ERROR_QUANTILE],
        "feature_columns_sha256": hashlib.sha256(
            json.dumps(list(feature_columns)).encode("utf-8")
        ).hexdigest(),
        "distribution_sha256": hashlib.sha256(
            json.dumps(train_params["feature_missing_rate"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "drift_exempt": sorted(exempt),
        "fixed_evidence": dict(fixed_evidence or {}),
    }
    summary = _summarize(
        rows,
        metadata,
        start,
        end,
        expected_days=len(days),
        fixed_evidence=dict(fixed_evidence or {}),
        identity=identity,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shadow_76d.json").write_text(
        json.dumps(
            {
                "window": [start, end],
                "model_version": version,
                "identity": identity,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows": [row.__dict__ for row in rows],
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "shadow_76d.md").write_text(_render_markdown(rows, summary), encoding="utf-8")
    return rows, summary


def _summarize(
    rows: List[DayRow],
    metadata: dict,
    start: str,
    end: str,
    *,
    expected_days: int,
    fixed_evidence: Dict[str, str],
    identity: Dict[str, object],
) -> dict:
    valid = [row for row in rows if row.error is None]
    c_blocked_days = [row.date for row in valid if row.c_blocked]
    b_blocked_days = [row.date for row in valid if row.blocked]
    b_only = sorted(set(b_blocked_days) - set(c_blocked_days))
    c_only = sorted(set(c_blocked_days) - set(b_blocked_days))
    # 归因（R2-P1-3）：必须由**该日实际 error 列**支撑（全行率 >60% 或全空），
    # 且阻断日必须附结构化修复证据，否则不算"已归因"
    attribution: Dict[str, str] = {}
    attributed_ok = True
    for row in valid:
        if not row.blocked:
            continue
        error_cols = {item["column"] for item in row.error_columns}
        real = sorted(
            col
            for col in error_cols
            if (row.rates_all.get(col) or 0.0) > _C_COLUMN_MISSING_RATE_MAX
            or (row.rates_all.get(col) or 0.0) >= 1.0
        )
        evidence = fixed_evidence.get(row.date)
        if real and evidence:
            attribution[row.date] = (
                f"真实源退化（实际 error 列 {real[:3]}；C 同样拦截；证据 {evidence}）"
            )
        elif real:
            attribution[row.date] = f"真实源退化但**缺修复证据**（实际 error 列 {real[:3]}）"
            attributed_ok = False
        else:
            attribution[row.date] = "仅 B′ 拦截（误差列均非真实退化）"
            attributed_ok = False
    # 列级频次（R2-P1-2）：有列名即计（含 uncovered 的 rate=None）；max_rate 仅有限率参与
    freq: Dict[str, dict] = {}
    for row in valid:
        seen = set()
        for alert in getattr(row, "error_columns", []):
            col = alert["column"]
            if col in seen:
                continue
            seen.add(col)
            item = freq.setdefault(col, {"error_days": 0, "max_rate": 0.0})
            item["error_days"] += 1
            rate = alert.get("rate")
            if isinstance(rate, (int, float)) and rate == rate and 0.0 <= float(rate) <= 1.0:
                item["max_rate"] = max(item["max_rate"], float(rate))
    n_days = len(valid)
    freq_rows = [
        {
            "column": col,
            "error_days": item["error_days"],
            "ratio": item["error_days"] / max(1, n_days),
            "max_rate": item["max_rate"],
            "over_20pct": n_days > 0 and item["error_days"] / n_days > _ERROR_DAY_RATIO_MAX,
        }
        for col, item in sorted(freq.items(), key=lambda kv: -kv[1]["error_days"])
    ]
    over_limit = [item for item in freq_rows if item["over_20pct"]]
    b1_1 = len(b_blocked_days) == 0 or (attributed_ok and len(attribution) == len(b_blocked_days))
    b1_2 = n_days > 0 and len(over_limit) == 0
    exit_ok = (
        len([row for row in rows if row.error]) == 0
        and n_days == expected_days
        and n_days > 0
        and b1_1
        and b1_2
    )
    return {
        "n_days": n_days,
        "n_days_total": len(rows),
        "expected_days": expected_days,
        "n_errors": len([row for row in rows if row.error]),
        "c_blocked_days": c_blocked_days,
        "b_blocked_days": b_blocked_days,
        "b_only_blocked": b_only,
        "c_only_blocked": c_only,
        "attribution": attribution,
        "freq": freq_rows,
        "over_20pct_columns": over_limit,
        "b1_1_blocked_zero_or_attributed": b1_1,
        "b1_2_no_column_over_20pct": b1_2,
        "b1_exit_ok": exit_ok,
        "rolling_blocked_days": [row.date for row in valid if row.rolling_blocked],
        "identity": identity,
        "meta": {
            "model_version": metadata.get("version"),
            "model_type": metadata.get("model_type"),
            "window": [start, end],
        },
    }


def _render_markdown(rows: List[DayRow], summary: dict) -> str:
    lines: List[str] = []
    lines.append("# 76 日就绪门影子对照表（B′ vs 旧 C）\n")
    lines.append(f"- 窗口：{summary['meta']['window'][0]} ~ {summary['meta']['window'][1]}")
    lines.append(f"- 模型：v{summary['meta']['model_version']}（{summary['meta']['model_type']}）")
    lines.append(f"- 对照日数：{summary['n_days']}（无效日 {summary['n_errors']}）")
    lines.append(
        f"- 旧 C 阻断：{len(summary['c_blocked_days'])} 日；B′ 阻断：{len(summary['b_blocked_days'])} 日；"
        f"仅 B′ 拦：{len(summary['b_only_blocked'])} 日；仅 C 拦：{len(summary['c_only_blocked'])} 日\n"
    )
    lines.append("## B1 判据核对\n")
    lines.append(
        f"- ① 阻断日 = 0 或全部归因真实源退化：**{'PASS' if summary['b1_1_blocked_zero_or_attributed'] else 'FAIL'}**"
        f"（阻断 {len(summary['b_blocked_days'])} 日；归因 {summary['attribution']}）"
    )
    lines.append(
        f"- ② 单列 error ≤ 20% 交易日：**{'PASS' if summary['b1_2_no_column_over_20pct'] else 'FAIL'}**"
        f"（超限列 {len(summary['over_20pct_columns'])}）"
    )
    lines.append("- ③ 全量回归 + 复杂度门：由发布检查另行核对（见 CHANGELOG）\n")
    lines.append("## 逐日表\n")
    lines.append(
        "| # | 交易日 | 旧C | B′ | error/warn | 域内/全行 | 主要 error 列（域内率） | cons 全行→域内 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for index, row in enumerate(rows, start=1):
        if row.error is not None:
            lines.append(f"| {index} | {row.date} | - | 组装失败 | - | - | {row.error[:48]} | - |")
            continue
        c = "阻断" if row.c_blocked else "通过"
        b = "阻断" if row.blocked else ("通过" if row.ok else "未就绪")
        top = ", ".join(
            (
                f"{item['column']}={float(item['rate']):.1%}"
                if item.get("rate") is not None
                else f"{item['column']}(无率)"
            )
            for item in row.error_columns[:3]
        )
        cons_all = row.rates_all.get("cons_target_upside")
        cons_dom = row.rates_domain.get("cons_target_upside")
        cons_text = (
            f"{cons_all:.1%}→{cons_dom:.1%}"
            if cons_all is not None and cons_dom is not None
            else "-"
        )
        lines.append(
            f"| {index} | {row.date} | {c} | {b} | {row.errors}/{row.warnings} | "
            f"{row.n_rows_used}/{row.n_rows_total} | {top or '-'} | {cons_text} |"
        )
    lines.append("\n## 列级 error 频次（Top 20）\n")
    lines.append("| 列 | error 日数 | 占比 | 最大当日率 | >20% |")
    lines.append("|---|---|---|---|---|")
    for item in summary["freq"][:20]:
        lines.append(
            f"| {item['column']} | {item['error_days']} | {item['ratio']:.1%} | "
            f"{item['max_rate']:.1%} | {'是' if item['over_20pct'] else '否'} |"
        )
    lines.append("\n## 滚动分位敏感性（近 60 日、最少 20 日，前序样本）\n")
    lines.append(f"- 静态基线阻断日：{len(summary['b_blocked_days'])}")
    lines.append(
        f"- 滚动基线阻断日：{len(summary['rolling_blocked_days'])}（{summary['rolling_blocked_days']}）"
    )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="76 日就绪门影子对照（B′ vs 旧 C）")
    parser.add_argument("--start", default="20260703")
    parser.add_argument("--end", default="20261026")
    parser.add_argument("--version", type=int, default=24022)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--fixed-evidence",
        default=None,
        help="阻断日修复证据 JSON（{date: 证据说明}；B1① 归因为真实退化时必需）",
    )
    args = parser.parse_args(argv)
    fixed: Dict[str, str] = {}
    if args.fixed_evidence:
        payload = json.loads(Path(args.fixed_evidence).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("--fixed-evidence 必须是 {date: evidence} JSON 对象")
        fixed = {str(k): str(v) for k, v in payload.items()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir
        or (_REPO_ROOT / "data" / "frozen_reference" / "v2_p4" / f"coverage_shadow_76d_{stamp}")
    )
    rows, summary = run(args.start, args.end, args.version, out_dir, fixed_evidence=fixed)
    print("=" * 78)
    print(
        f"对照完成：{summary['n_days']} 日 / 旧C 阻断 {len(summary['c_blocked_days'])} / "
        f"B′ 阻断 {len(summary['b_blocked_days'])} / 仅 B′ 拦 {len(summary['b_only_blocked'])} / "
        f"仅 C 拦 {len(summary['c_only_blocked'])}"
    )
    print(f"B1① 阻断日归因: {summary['attribution']}")
    print(
        f"B1① {'PASS' if summary['b1_1_blocked_zero_or_attributed'] else 'FAIL'}；"
        f"B1② {'PASS' if summary['b1_2_no_column_over_20pct'] else 'FAIL'}（超 20% 列 {len(summary['over_20pct_columns'])}）"
    )
    if summary["over_20pct_columns"]:
        print("超 20% 列（需单列登记豁免或修复）:")
        for item in summary["over_20pct_columns"][:10]:
            print(
                f"  {item['column']}: {item['error_days']}/{summary['n_days']} 日 ({item['ratio']:.0%})"
            )
    print(f"滚动基线阻断日: {summary['rolling_blocked_days']}")
    print(
        f"有效日 {summary['n_days']}/{summary['expected_days']}，"
        f"无效日 {summary['n_errors']}；出口 {'PASS' if summary['b1_exit_ok'] else 'FAIL'}"
    )
    print(f"证据: {out_dir}")
    return 0 if summary["b1_exit_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
