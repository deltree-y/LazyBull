# -*- coding: utf-8 -*-
"""P7 T1 证据发布入口：P-1 同域静态可执行性（原子发布、禁覆盖、写域守卫）。

用法
====
```bash
python scripts/v2_p7/t1/run_p1.py --run-id p7_t1_20261111_r1
```

小样本冒烟（**不得**作为证据根发布）：

```bash
python scripts/v2_p7/t1/run_p1.py --run-id tmp_smoke \
  --smoke-panel-start 20200101 --limit-signal-days 40 --implementations 200 \\
  --out-root temp/p7_t1_smoke
```

退出码：`0` 已发布（pass/fail/inconclusive_frozen 三种三向结论均视为已发布）、
`2` 身份或输入不可用（未发布、零副作用）、`4` 写域拒绝、`5` 硬错误、
`6` 计数不守恒（拒绝生成经济 verdict，仍发布审计证据）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.v2_p7 import common  # noqa: E402
from scripts.v2_p7.event_builder import (  # noqa: E402
    EVENT_SOURCES,
    build_announcement_index,
    build_events,
    event_builder_identity,
    scan_fina_watermark,
)
from scripts.v2_p7.t1 import (  # noqa: E402
    dataplane,
    execution,
    matching,
    randomization,
    report,
    verdict,
)
from scripts.v2_p7.t1.params import (  # noqa: E402
    DEFAULT_RUN_ID,
    EVIDENCE_ROOT_REL,
    P1Params,
    ParamsError,
    build_params,
    cost_model_from_params,
    sha256_file,
)

EXIT_OK = 0
EXIT_INPUT_FAIL = 2
EXIT_WRITE_SCOPE = 4
EXIT_HARD_ERROR = 5
EXIT_COUNT_REFUSED = 6

EVIDENCE_WRITE_ROOT = EVIDENCE_ROOT_REL

FORBIDDEN_WATCH_ROOTS: Tuple[str, ...] = (
    "data/paper",
    "data/runs",
    "data/models",
    "data/frozen_reference/v2_p1",
    "data/frozen_reference/v2_p3",
    "data/frozen_reference/v2_p4",
    "data/frozen_reference/v2_p5b",
    "data/frozen_reference/v2_p65",
)

INPUT_WATCH_ROOTS: Tuple[str, ...] = (
    "data/raw",
    "data/clean",
    "configs",
)


# --------------------------------------------------------------------------------------
# 身份
# --------------------------------------------------------------------------------------
def git_state(repo_root: Path) -> Dict[str, Any]:
    """代码态：HEAD + 工作树脏标记 + 脏文件清单（只读）。"""

    def run(args: Sequence[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "--no-pager", *args],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    head = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    dirty_files = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {
        "git_head": (head or "").strip() or None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:200],
        "dirty_file_count": len(dirty_files),
    }


def script_identity(repo_root: Path) -> Dict[str, str]:
    """T1 全部脚本与共用件的字节 SHA（含 T0.5 公共件）。"""
    files = sorted((repo_root / "scripts" / "v2_p7").rglob("*.py"))
    return {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in files
        if path.is_file()
    }


def input_identity(repo_root: Path, params: P1Params) -> Dict[str, Any]:
    """输入根身份：大目录元数据摘要 + 事件源/日历/行业/成本逐文件内容 SHA。"""
    roots = {name: dataplane.metadata_digest(repo_root / name) for name in INPUT_WATCH_ROOTS}
    files: Dict[str, str] = {}
    for rel in (
        "data/raw/trade_cal.parquet",
        "data/raw/shenwan_industry.parquet",
        "data/clean/sw_industry_pit.parquet",
        "data/clean/sw_industry_code_table.parquet",
        "data/raw/stock_basic.parquet",
        "configs/base.yaml",
        params.prereg_path.relative_to(repo_root).as_posix(),
        params.crash_spec_path.relative_to(repo_root).as_posix(),
    ):
        path = repo_root / rel
        if path.exists():
            files[rel] = sha256_file(path)
    return {"roots": roots, "files": files}


def source_family_identity(repo_root: Path) -> Dict[str, Any]:
    """三源 raw 分区逐文件 SHA（事件源身份，供水位与复算使用）。"""
    payload: Dict[str, Any] = {}
    for name in ("forecast", "express", "fina_indicator"):
        entries = {}
        for path in sorted((repo_root / "data" / "raw" / name).glob("*.parquet")):
            entries[path.name] = sha256_file(path)
        payload[name] = {
            "files": entries,
            "file_count": len(entries),
            "combined_sha256": hashlib.sha256(
                "".join(f"{key}:{value}\n" for key, value in sorted(entries.items())).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
    return payload


def forbidden_watch(repo_root: Path) -> Dict[str, Any]:
    return {name: dataplane.metadata_digest(repo_root / name) for name in FORBIDDEN_WATCH_ROOTS}


# --------------------------------------------------------------------------------------
# 证据写出
# --------------------------------------------------------------------------------------
def write_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return sha256_file(path)


def write_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def collect_files(root: Path) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload[path.relative_to(root).as_posix()] = sha256_file(path)
    return payload


# --------------------------------------------------------------------------------------
# 单格计算
# --------------------------------------------------------------------------------------
def cell_payload(
    params: P1Params,
    source: str,
    horizon: int,
    table: Any,
    result: Any,
    panel: Any,
    timeline_start: int,
    *,
    implementations: int,
    quality: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """返回 (单元报告, 随机化报告)。"""
    counts = result.counts()
    resolved = result.status == execution.STATUS_RESOLVED
    unfilled = result.status == execution.STATUS_UNFILLED
    unknown = result.status == execution.STATUS_UNRESOLVED
    if counts["intent_total"] != int(resolved.sum() + unfilled.sum() + unknown.sum()):
        raise ValueError(f"状态分类覆盖不完整：{counts}")
    assignment = verdict.assign_blocks(
        table.signal_pos + 1,
        timeline_start=timeline_start,
        timeline_end=panel.n_days - 1,
        block_days=params.block_days,
        guard_days=params.guard_days,
    )
    blocks = verdict.effective_blocks(assignment, resolved, result.exit_pos)

    known = ~unknown
    member_residuals = np.where(resolved[:, None], result.net, 0.0)[known]
    group_date = table.signal_pos[known]
    intent_residual = np.where(resolved, result.residual, np.where(unfilled, 0.0, np.nan))
    rand = randomization.analyze_cell(
        member_residuals,
        group_date,
        intent_residual,
        resolved,
        unfilled,
        block_h=int(horizon),
        implementations=int(implementations),
        seeds=dict(params.seeds),
        quantile=params.component_quantile,
        delta_grid=params.delta_r_grid,
        min_power=params.mde_min_power,
        day_span=(int(timeline_start), int(panel.n_days - 1)),
    )
    stress = verdict.stress_statistics(result.status, result.event_net, result.control_net)
    state = verdict.cell_state(
        counts,
        effective_blocks=blocks["effective_history_block"],
        min_blocks=params.min_effective_blocks,
        statistic_intent=rand["statistic_intent"],
        critical_value=rand["critical_value"],
        power=rand["power"],
    )
    execution_info = report.execution_report(
        result,
        table,
        panel,
        matching_quality=quality,
        matching_diagnostics=diagnostics,
        min_ticket=params.min_ticket,
        horizon=int(horizon),
        timeline_start=int(timeline_start),
    )
    missing_status = state["missing_identification_status"]
    payload = {
        "source": source,
        "horizon": int(horizon),
        "counts": counts,
        "counts_conserved": state["counts_conserved"],
        "gate_state": state["state"],
        "missing_identification_status": missing_status,
        "blocks": blocks,
        "randomization": {
            key: rand[key]
            for key in (
                "statistic_intent",
                "statistic_filled_only",
                "critical_value",
                "quantile",
                "quantile_method",
                "sigma_D0",
                "false_pass",
                "power",
            )
        },
        "randomization_detail": rand,
        "stress": stress,
        "execution": execution_info,
        "disclosure_seasons": report.disclosure_seasons(table),
        "domain": dict(quality.get("domain", {})),
        "state": state,
    }
    return payload, rand


# --------------------------------------------------------------------------------------
# runbook
# --------------------------------------------------------------------------------------
RUNBOOK_TEMPLATE = """# P7 T1 证据运行手册（自动生成）

- run_id：`{run_id}`
- prereg：`{prereg}`（SHA `{prereg_sha}`）
- 事件构建器：`{builder}`（SHA `{builder_sha}`）
- 共同数据截止：`{cutoff}`；评价窗：`{start}`~`{end}`
- 核心格：三源 × 10 日；敏感性：三源 × 5 日
- 内层实现：{implementations} 次/格/流；种子：{seeds}
- 三向结论：`{verdict}`（{verdict_state}）

## 复算命令

```bash
python scripts/v2_p7/t1/run_p1.py --run-id {run_id}
```

（同一输入、同一代码态下逐字节可复现；证据根原子发布、禁覆盖，重跑请换新 run_id。）

## 目录

| 路径 | 内容 |
|---|---|
| `manifest.json` | prereg/规格/事件构建器/输入根/代码态/参数 SHA 与水位 |
| `events/` | 三源事件流、fina 分区扫描与水位证明 |
| `matched_groups/` | 六股匹配组账本与匹配质量摘要 |
| `randomization/` | 每格零假设分布、单侧临界值、误放行与功效 |
| `power/summary.json` | 每格有效块、披露季、MDE 汇总 |
| `cells/` | 每格执行/匹配/构成诊断 |
| `verdict.json` | 三向 verdict + 计数守恒 + 每格有效块 + 组件门结果 |
| `readonly_audit.json` | 输入根前后 SHA、写入面清单、未触碰禁改面 |

## 口径摘要

- 计划入场 = T+1；六股共同入场、共同顺延最长 10 日；到期 = 入场 + h；离场共同顺延最长 10 日。
- 无共同入场日 ⇒ `unfilled_group_count`（收益/费用真实 0）；10 日内无共同离场或价格链缺失
  ⇒ `unresolved_count`（不补 0）。
- 任一核心格 `unresolved_count>0` ⇒ `inconclusive_frozen`；5% 仅告警分级。
- 压力情景（事件 −100%/对照 +100% 及反向）只写 `stress_scenario_results`，`diagnostic_only=true`。
- 完整复合门（CP 上界、λ 选择、独立验证流）**未在 T1 检验**，固定 `not_tested_in_T1`。
- 证据等级 H：`historical_virtual_clock=true / not_forward_evidence=true /
  not_production_authorization=true`。
"""


def build_runbook(
    params: P1Params,
    run_id: str,
    builder: Mapping[str, Any],
    implementations: int,
    verdict_payload: Mapping[str, Any],
) -> str:
    return RUNBOOK_TEMPLATE.format(
        run_id=run_id,
        prereg=params.prereg_path.as_posix(),
        prereg_sha=params.prereg_sha256,
        builder=builder["path"],
        builder_sha=builder["sha256"],
        cutoff=params.common_data_cutoff,
        start=params.evaluation_start,
        end=params.evaluation_end,
        implementations=implementations,
        seeds=json.dumps(dict(params.seeds), ensure_ascii=False, sort_keys=True),
        verdict=verdict_payload.get("verdict"),
        verdict_state=verdict_payload.get("verdict_state"),
    )


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 T1 P-1 证据发布")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-root", default=EVIDENCE_WRITE_ROOT)
    parser.add_argument(
        "--industry-source",
        choices=("snapshot", "pit"),
        default="snapshot",
        help="行业输入来源：snapshot=r2 口径（默认）；pit=区间表驱动（r3）",
    )
    parser.add_argument("--implementations", type=int, default=0, help="0 = 用 prereg 的 10000")
    parser.add_argument(
        "--limit-signal-days", type=int, default=0, help="冒烟用：每源最多 N 个信号日"
    )
    parser.add_argument("--smoke-panel-start", default="", help="冒烟用：面板分区起始日")
    parser.add_argument("--smoke-event-start", default="", help="冒烟用：事件 knowledge_date 下界")
    parser.add_argument(
        "--intent-domain",
        default="listed",
        choices=("listed", "strict"),
        help="意图域口径：listed=事件股须在信号日已上市（默认，停牌仍计入）；strict=全部计入并记 unresolved",
    )
    parser.add_argument("--allow-root", action="append", default=[])
    parser.add_argument("--skip-input-audit", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def build_cell_tables(
    params: P1Params,
    panel: Any,
    events: Mapping[str, pd.DataFrame],
    evaluation_start_pos: int,
    *,
    limit_signal_days: int,
    event_start: str = "",
    intent_domain: str = "listed",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """逐 (源, 持期) 构造六股匹配组（观察窗随 h 变化，尾端右删失按格登记）。

    ``limit_signal_days`` / ``event_start`` 仅供冒烟降配，正式运行必须保持默认（0 / 空），
    否则 run_id 的产物一律标 ``diagnostic_only``。
    """
    tables: Dict[Any, Any] = {}
    qualities: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    for source, horizon in params.cell_keys():
        key = f"{source}_{int(horizon)}d"
        frame = events[source]
        if event_start:
            frame = frame[frame["knowledge_date"] >= str(event_start)]
        if limit_signal_days:
            mapped, _ = matching.map_events_to_days(
                frame,
                panel,
                evaluation_start_pos=evaluation_start_pos,
                horizon=int(horizon),
            )
            keep_days = sorted(mapped["signal_pos"].unique())[: int(limit_signal_days)]
            frame = mapped[mapped["signal_pos"].isin(keep_days)].loc[
                :, [column for column in frame.columns]
            ]
        table, source_diagnostics, domain = matching.build_groups(
            frame,
            panel,
            source=source,
            evaluation_start_pos=evaluation_start_pos,
            intent_domain=intent_domain,
            horizon=int(horizon),
        )
        tables[(source, int(horizon))] = table
        qualities[key] = matching.matching_quality(table)
        qualities[key]["domain"] = domain
        diagnostics[key] = source_diagnostics.as_dict()
    return tables, qualities, diagnostics


class _Abort(Exception):
    """带退出码与 partial 目录的中止（保证拒绝路径零副作用）。"""

    def __init__(self, code: int, message: str, partial: Optional[Path] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.partial = partial


@dataclass
class RunContext:
    """一次 T1 运行的可变上下文（避免长参数列表）。"""

    args: argparse.Namespace
    repo_root: Path
    params: P1Params
    run_id: str
    target: Path
    roots: List[Path]
    partial: Path
    implementations: int
    diagnostic_only: bool
    started: float
    written: Dict[str, str] = field(default_factory=dict)
    cells: List[Dict[str, Any]] = field(default_factory=list)
    group_summaries: Dict[str, Any] = field(default_factory=dict)
    qualities: Dict[str, Any] = field(default_factory=dict)
    matching_diagnostics: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[Any, Any] = field(default_factory=dict)
    panel: Any = None
    events: Dict[str, Any] = field(default_factory=dict)
    event_stats: Dict[str, Any] = field(default_factory=dict)
    announcements: Any = None
    fina_watermark: Dict[str, Any] = field(default_factory=dict)
    source_families: Dict[str, Any] = field(default_factory=dict)
    identity_before: Dict[str, Any] = field(default_factory=dict)
    forbidden_before: Dict[str, Any] = field(default_factory=dict)
    code_state: Dict[str, Any] = field(default_factory=dict)
    scripts: Dict[str, str] = field(default_factory=dict)
    builder: Dict[str, Any] = field(default_factory=dict)
    panel_start: str = ""
    cost: Dict[str, float] = field(default_factory=dict)
    verdict_payload: Dict[str, Any] = field(default_factory=dict)


def _bootstrap(args: argparse.Namespace) -> RunContext:
    """写域预检 → 参数派生 → partial 目录 → 身份快照。"""
    repo_root = _REPO_ROOT
    out_root = (
        Path(args.out_root) if Path(args.out_root).is_absolute() else repo_root / args.out_root
    )
    run_id = str(args.run_id)
    target = (out_root / run_id).resolve()
    roots = list(common.resolve_write_roots(args.allow_root))
    roots.append((repo_root / EVIDENCE_WRITE_ROOT).resolve())
    planned = [target / name for name in ("manifest.json", "verdict.json", "readonly_audit.json")]
    try:
        common.assert_targets_within_write_roots(planned, roots)
    except common.WriteScopeError as error:
        raise _Abort(EXIT_WRITE_SCOPE, f"写域拒绝：{error}") from error
    if target.exists():
        raise _Abort(EXIT_WRITE_SCOPE, f"目标证据根已存在（禁止覆盖）：{target}")
    try:
        params = build_params(repo_root)
    except (ParamsError, FileNotFoundError, KeyError, ValueError) as error:
        raise _Abort(EXIT_INPUT_FAIL, f"身份/输入不可用：{error}") from error

    implementations = int(args.implementations) if args.implementations else params.implementations
    diagnostic_only = (
        implementations != params.implementations
        or int(args.limit_signal_days) > 0
        or bool(args.smoke_panel_start)
        or bool(args.smoke_event_start)
    )
    partial = target.with_name(f"{run_id}.partial-{os.getpid()}")
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True, exist_ok=False)
    context = RunContext(
        args=args,
        repo_root=repo_root,
        params=params,
        run_id=run_id,
        target=target,
        roots=roots,
        partial=partial,
        implementations=implementations,
        diagnostic_only=diagnostic_only,
        started=time.time(),
        panel_start=str(args.smoke_panel_start),
        cost=params.cost.as_dict(),
    )
    context.identity_before = input_identity(repo_root, params)
    context.forbidden_before = forbidden_watch(repo_root)
    context.code_state = git_state(repo_root)
    context.scripts = script_identity(repo_root)
    context.builder = event_builder_identity().as_dict()
    return context


def _progress(message: str) -> None:
    """运行进度（写到 stdout 并立即 flush，便于后台日志轮询）。"""
    print(f"[t1] {message}", flush=True)


def _prepare_cells(context: RunContext) -> None:
    """事件流、面板、匹配组与执行（只读计算段）。"""
    params = context.params
    repo_root = context.repo_root
    raw_root = repo_root / "data" / "raw"
    context.fina_watermark = scan_fina_watermark(raw_root)
    context.source_families = source_family_identity(repo_root)
    _progress("身份快照完成，开始物化事件流")
    context.events, context.event_stats = build_events(raw_root, cutoff=params.common_data_cutoff)
    context.announcements = build_announcement_index(raw_root, cutoff=params.common_data_cutoff)
    _progress("事件流完成：" + str({k: len(v) for k, v in context.events.items()}))
    days = dataplane.load_trading_days(repo_root / "data", params.common_data_cutoff)
    first_clean = dataplane.first_clean_daily_date(repo_root / "data")
    days = [day for day in days if day >= first_clean]
    if params.evaluation_start not in set(days):
        raise ValueError(f"评价窗起点 {params.evaluation_start} 不在面板日历内")
    _progress(f"加载面板：{len(days)} 个交易日（自 {days[0]}）")
    context.panel = dataplane.load_panel(
        repo_root / "data",
        days,
        cutoff=params.common_data_cutoff,
        announcements=context.announcements,
        lookback=20,
        panel_start=context.panel_start,
        industry_source=str(context.args.industry_source),
    )
    _progress(f"面板就绪：{context.panel.n_days} 日 × {context.panel.n_stocks} 股")
    evaluation_start_pos = context.panel.day_position(params.evaluation_start)
    context.tables, context.qualities, context.matching_diagnostics = build_cell_tables(
        params,
        context.panel,
        context.events,
        evaluation_start_pos,
        limit_signal_days=int(context.args.limit_signal_days),
        event_start=str(context.args.smoke_event_start),
        intent_domain=str(context.args.intent_domain),
    )
    _progress("匹配组完成：" + str({k: v.status.size for k, v in context.tables.items()}))
    assert cost_model_from_params(params) is not None
    context.results = execution.execute_all_cells(
        context.panel, context.tables, params, cost=context.cost
    )
    context.cost["min_ticket"] = params.min_ticket
    _progress("执行模拟完成")


def _write_cells(context: RunContext) -> None:
    """逐格写出：cells / randomization / matched_groups 账本。"""
    params = context.params
    evaluation_start_pos = context.panel.day_position(params.evaluation_start)
    for source, horizon in params.cell_keys():
        key = f"{source}_{int(horizon)}d"
        table = context.tables[(source, int(horizon))]
        result = context.results[(source, int(horizon))]
        payload, rand = cell_payload(
            params,
            source,
            int(horizon),
            table,
            result,
            context.panel,
            evaluation_start_pos,
            implementations=context.implementations,
            quality=context.qualities[key],
            diagnostics=context.matching_diagnostics[key],
        )
        context.cells.append(payload)
        _progress(f"格 {key} 完成：{payload['counts']}")
        context.written[f"cells/{key}.json"] = common.write_json(
            context.partial / "cells" / f"{key}.json", payload
        )
        context.written[f"randomization/{key}.json"] = common.write_json(
            context.partial / "randomization" / f"{key}.json", rand
        )
        groups = report.group_frame(
            table, result, context.panel, missing_status=payload["missing_identification_status"]
        )
        stocks = report.stock_frame(table, result, context.panel)
        groups_sha = write_parquet(
            context.partial / "matched_groups" / f"{key}_groups.parquet", groups
        )
        stocks_sha = write_parquet(
            context.partial / "matched_groups" / f"{key}_stocks.parquet", stocks
        )
        context.written[f"matched_groups/{key}_groups.parquet"] = groups_sha
        context.written[f"matched_groups/{key}_stocks.parquet"] = stocks_sha
        context.group_summaries[key] = {
            "rows_groups": int(len(groups)),
            "rows_stocks": int(len(stocks)),
            "counts": payload["counts"],
            "files_sha256": {"groups": groups_sha, "stocks": stocks_sha},
        }
        del groups, stocks


def _write_shared_evidence(context: RunContext) -> None:
    """事件流与水位、匹配摘要、功效汇总、verdict、诊断。"""
    params = context.params
    partial = context.partial
    for source in EVENT_SOURCES:
        context.written[f"events/{source}.parquet"] = write_parquet(
            partial / "events" / f"{source}.parquet", context.events[source]
        )
    context.written["events/watermarks.json"] = common.write_json(
        partial / "events" / "watermarks.json",
        {
            "fina_indicator": context.fina_watermark,
            "forecast": {
                "watermark_file": "data/raw/forecast/_sync_watermark.json",
                "synced_to": params.common_data_cutoff,
            },
            "express": {
                "watermark_file": "data/raw/express/_sync_watermark.json",
                "synced_to": params.common_data_cutoff,
            },
            "common_data_cutoff": params.common_data_cutoff,
            "source_family_sha256": context.source_families,
            "event_stats": context.event_stats,
            "announcement_rows": int(len(context.announcements)),
        },
    )
    context.written["matched_groups/summary.json"] = common.write_json(
        partial / "matched_groups" / "summary.json",
        {
            "run_id": context.run_id,
            "matching_quality": context.qualities,
            "matching_diagnostics": context.matching_diagnostics,
            "cells": context.group_summaries,
            "control_pool_rule": (
                "完整基础行情（开高低收/前收/复权收盘有限为正、总市值>0、20 日协变量可算）"
                "且当日无三源公告"
            ),
        },
    )
    context.written["power/summary.json"] = common.write_json(
        partial / "power" / "summary.json",
        {
            "run_id": context.run_id,
            "cells": {
                payload["source"]
                + f"_{payload['horizon']}d": {
                    "effective_history_block": payload["blocks"]["effective_history_block"],
                    "block_years": payload["blocks"]["block_years"],
                    "min_required": params.min_effective_blocks,
                    "disclosure_seasons": payload["disclosure_seasons"],
                    "sigma_D0": payload["randomization"]["sigma_D0"],
                    "critical_value": payload["randomization"]["critical_value"],
                    "power": payload["randomization"]["power"],
                    "false_pass": payload["randomization"]["false_pass"],
                    "state": payload["state"]["state"],
                }
                for payload in context.cells
            },
        },
    )
    _write_verdict(context)


def _write_verdict(context: RunContext) -> None:
    params = context.params
    payload = verdict.build_verdict(
        context.cells,
        core_cells=params.core_cells,
        min_blocks=params.min_effective_blocks,
        run_id=context.run_id,
        prereg_id=params.prereg_id,
        prereg_sha256=params.prereg_sha256,
    )
    payload["implementation"] = {
        "implementations_per_cell": context.implementations,
        "diagnostic_only": context.diagnostic_only,
        "intent_domain": str(context.args.intent_domain),
        "intent_domain_rule": (
            "事件股须在信号日**已上市**（list_date ∪ 首次可观测日），未上市记录不计入 P-1 意图域；"
            "**停牌仍计入**并进入执行/不可解析分类。尾端不足以覆盖「10 日入场 + h 日持有 + 10 日离场」"
            "完整观察窗的事件按右删失单独登记，不记真实 0"
        ),
        "intent_domain_requires_review": False,
        "smoke_limits": {
            "limit_signal_days": int(context.args.limit_signal_days),
            "smoke_panel_start": context.panel_start or None,
            "smoke_event_start": context.args.smoke_event_start or None,
        },
        "cost": context.cost,
        "min_ticket": params.min_ticket,
        "seeds": dict(params.seeds),
        "core_cells": [f"{source}_{int(horizon)}d" for source, horizon in params.core_cells],
        "common_data_cutoff": params.common_data_cutoff,
        "evaluation_window": [params.evaluation_start, params.evaluation_end],
    }
    if context.diagnostic_only:
        payload["verdict"] = None
        payload["verdict_state"] = "diagnostic_only"
        payload["verdict_note"] = "冒烟/降配运行，不构成任何三向结论"
    context.verdict_payload = payload
    context.written["verdict.json"] = common.write_json(context.partial / "verdict.json", payload)
    context.written["diagnostics.json"] = common.write_json(
        context.partial / "diagnostics.json",
        {
            "watermark": dataplane.watermark_payload(context.panel, params.common_data_cutoff),
            "panel_load": context.panel.load_stats,
            "industry_source": context.panel.industry_source,
            "event_stats": context.event_stats,
            "matching": context.matching_diagnostics,
            "quality": context.qualities,
        },
    )


def _publish(context: RunContext) -> int:
    """只读审计 → runbook → manifest → 原子发布。"""
    params = context.params
    inputs_after = input_identity(context.repo_root, params)
    forbidden_after = forbidden_watch(context.repo_root)
    input_unchanged = context.identity_before == inputs_after
    forbidden_unchanged = context.forbidden_before == forbidden_after
    if not input_unchanged or not forbidden_unchanged:
        raise _Abort(EXIT_INPUT_FAIL, "输入根或禁改面前后 SHA 变化：拒绝发布", context.partial)
    runbook = build_runbook(
        params, context.run_id, context.builder, context.implementations, context.verdict_payload
    )
    context.written["runbook.md"] = write_text(context.partial / "runbook.md", runbook)
    context.written["readonly_audit.json"] = common.write_json(
        context.partial / "readonly_audit.json",
        _audit_payload(context, inputs_after, forbidden_after),
    )
    manifest = _manifest_payload(context)
    context.written["manifest.json"] = common.write_json(
        context.partial / "manifest.json", manifest
    )
    if context.target.exists():
        raise _Abort(
            EXIT_WRITE_SCOPE, f"目标证据根已存在（禁止覆盖）：{context.target}", context.partial
        )
    context.partial.rename(context.target)
    _print_summary(context)
    state = context.verdict_payload["verdict_state"]
    return EXIT_COUNT_REFUSED if state == verdict.STATE_REFUSED else EXIT_OK


def _audit_payload(
    context: RunContext, inputs_after: Dict[str, Any], forbidden_after: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "run_id": context.run_id,
        "write_scope": {
            "allowed_roots": [path.as_posix() for path in context.roots],
            "forbidden_prefixes": list(common.FORBIDDEN_WRITE_PREFIXES),
            "target": context.target.as_posix(),
        },
        "input_roots_before": context.identity_before["roots"],
        "input_roots_after": inputs_after["roots"],
        "input_files_before": context.identity_before["files"],
        "input_files_after": inputs_after["files"],
        "input_unchanged": True,
        "forbidden_roots": FORBIDDEN_WATCH_ROOTS,
        "forbidden_before": context.forbidden_before,
        "forbidden_after": forbidden_after,
        "forbidden_unchanged": True,
        "untouched": [
            "data/paper",
            "data/runs",
            "data/models",
            "data/frozen_reference/v2_p1",
            "data/frozen_reference/v2_p3",
            "data/frozen_reference/v2_p4",
            "data/frozen_reference/v2_p5b",
            "data/frozen_reference/v2_p65",
            "src/lazybull",
            "configs",
        ],
        "written_files": sorted(context.written),
    }


def _manifest_payload(context: RunContext) -> Dict[str, Any]:
    params = context.params
    return {
        "schema": "p7_t1_manifest/v1",
        "run_id": context.run_id,
        "task": "P7-T1 P-1 同域静态可执行性",
        "status": "diagnostic_only" if context.diagnostic_only else "final",
        "prereg": {
            "path": params.prereg_path.as_posix(),
            "sha256": params.prereg_sha256,
            "prereg_id": params.prereg_id,
        },
        "crash_spec": {
            "path": params.crash_spec_path.as_posix(),
            "sha256": params.crash_spec_sha256,
            "note": "仅登记引用；T1 不使用其窗口，也不读取窗口结果",
        },
        "event_builder": context.builder,
        "code_state": {**context.code_state, "script_sha256": context.scripts},
        "data_state": {
            "common_data_cutoff": params.common_data_cutoff,
            "evaluation_window": [params.evaluation_start, params.evaluation_end],
            "precheck_attachments": dict(params.precheck_attachments),
            "watermarks": {
                "forecast": params.common_data_cutoff,
                "express": params.common_data_cutoff,
                "fina_report": context.fina_watermark["max_ann_date"],
                "clean_daily": params.common_data_cutoff,
            },
            "input_roots": context.identity_before["roots"],
            "input_files": context.identity_before["files"],
            "source_family_sha256": context.source_families,
        },
        "params": params.params_payload(),
        "implementations_per_cell": context.implementations,
        "industry_source": str(context.args.industry_source),
        "intent_domain": str(context.args.intent_domain),
        "smoke_limits": {
            "limit_signal_days": int(context.args.limit_signal_days),
            "smoke_panel_start": context.panel_start or None,
            "smoke_event_start": context.args.smoke_event_start or None,
        },
        "cells": [f"{payload['source']}_{payload['horizon']}d" for payload in context.cells],
        "written_files_sha256": context.written,
        "elapsed_seconds": round(time.time() - context.started, 1),
    }


def _print_summary(context: RunContext) -> None:
    payload = context.verdict_payload
    print(f"published={context.target}")
    print(f"verdict={payload['verdict']} state={payload['verdict_state']}")
    for cell in context.cells:
        counts = cell["counts"]
        print(
            f"{cell['source']}/{cell['horizon']}d "
            f"intent={counts['intent_total']} resolved={counts['resolved_filled_count']} "
            f"unfilled={counts['unfilled_group_count']} unresolved={counts['unresolved_count']} "
            f"blocks={cell['blocks']['effective_history_block']} "
            f"mde={cell['state']['mde']} state={cell['state']['state']}"
        )


def run(args: argparse.Namespace) -> int:
    """执行一次 T1 运行并原子发布证据根（禁覆盖）。"""
    context: Optional[RunContext] = None
    try:
        context = _bootstrap(args)
        _prepare_cells(context)
        _write_cells(context)
        _write_shared_evidence(context)
        return _publish(context)
    except _Abort as abort:
        if abort.partial is not None and abort.partial.exists():
            shutil.rmtree(abort.partial, ignore_errors=True)
        print(abort.message, file=sys.stderr)
        return abort.code
    except Exception:  # noqa: BLE001 - 异常路径必须删除未完成的 partial 目录
        if context is not None and context.partial.exists():
            shutil.rmtree(context.partial, ignore_errors=True)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except common.WriteScopeError as error:
        print(f"写域拒绝：{error}", file=sys.stderr)
        return EXIT_WRITE_SCOPE
    except (FileNotFoundError, ParamsError, KeyError, ValueError) as error:
        print(f"硬错误，未发布证据：{error}", file=sys.stderr)
        return EXIT_HARD_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
