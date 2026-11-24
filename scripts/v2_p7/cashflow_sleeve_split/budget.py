# -*- coding: utf-8 -*-
"""探针实测成本与主批次预算（协议 §8「预算」、§7.6）。

口径
====
- **实测**只包含本阶段真正跑过的只读动作：探针日读取、哈希吞吐、分区 I/O 基线；
- **预算**分两层：``reference``（协议 §8 已给的历史量级）与 ``projection``（用实测
  I/O 下限推的上/下界）。两者必须并列，禁止用协议区间冒充实测；
- **结构重建**单列：本阶段**没有**跑真实 ``_run_capture``，因此只给 I/O 地板值并
  标 ``status = lower_bound_only``；申请 run-a 前必须先做「固定一个月分块」实测
  （耗时 / 峰值内存 / 磁盘），再用同一分块外推全窗。
- 超过 ``approval_required_over_minutes``（20 分钟）的机器批次一律须用户按命名
  批次批准；本阶段**未启动**任何此类批次。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT

#: 结构重建所需的自然月区间（账户窗 + 特征预热 120 交易日 + 价格 sizing 60 条）
REBUILD_MONTH_START = "2018-10"
REBUILD_MONTH_END = "2026-10"

#: I/O 基线采样（只读、固定、可复算）
IO_SAMPLE_CS_TRAIN_DAYS = 20
IO_SAMPLE_PANEL_MONTH = "2024-06"


class BudgetError(RuntimeError):
    """预算测算输入缺件或口径不符。"""


def _timed(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - started


def measure_io_baseline(root: Path | None = None) -> Dict[str, Any]:
    """实测 I/O 地板：month 面板、日特征、clean 日分区的读取成本与峰值内存。"""
    import pandas as pd

    base = Path(root) if root is not None else REPO_ROOT
    result: Dict[str, Any] = {"samples": {}}
    peak_rss = _rss_mb()

    month_dir = base / "data/features/panel_archive" / IO_SAMPLE_PANEL_MONTH
    if month_dir.is_dir():
        files = sorted(month_dir.glob("*.parquet"))
        bytes_total = sum(path.stat().st_size for path in files)

        def _read_month() -> int:
            rows = 0
            for path in files:
                rows += len(pd.read_parquet(path))
            return rows

        rows, seconds = _timed(_read_month)
        peak_rss = max(peak_rss, _rss_mb())
        result["samples"]["panel_month"] = {
            "month": IO_SAMPLE_PANEL_MONTH,
            "files": len(files),
            "bytes": bytes_total,
            "rows": int(rows),
            "seconds": round(seconds, 3),
            "mb_per_second": round(bytes_total / 1e6 / max(seconds, 1e-9), 1),
        }
    else:
        result["samples"]["panel_month"] = {"status": "missing", "path": str(month_dir)}

    cs_train_dir = base / "data/features/cs_train"
    days = sorted(cs_train_dir.glob("2024*.parquet"))[:IO_SAMPLE_CS_TRAIN_DAYS]
    if days:
        bytes_total = sum(path.stat().st_size for path in days)

        def _read_days() -> int:
            rows = 0
            for path in days:
                rows += len(pd.read_parquet(path))
            return rows

        rows, seconds = _timed(_read_days)
        peak_rss = max(peak_rss, _rss_mb())
        result["samples"]["cs_train_days"] = {
            "days": len(days),
            "first": days[0].name,
            "last": days[-1].name,
            "bytes": bytes_total,
            "rows": int(rows),
            "seconds": round(seconds, 3),
            "seconds_per_day": round(seconds / len(days), 4),
            "mb_per_second": round(bytes_total / 1e6 / max(seconds, 1e-9), 1),
        }
    else:
        result["samples"]["cs_train_days"] = {"status": "missing"}

    clean_day = base / "data/clean/daily" / "2024-06-03.parquet"
    if clean_day.exists():
        _, seconds = _timed(lambda: pd.read_parquet(clean_day))
        result["samples"]["clean_daily_day"] = {
            "path": str(clean_day.relative_to(base)).replace("\\", "/"),
            "bytes": clean_day.stat().st_size,
            "seconds": round(seconds, 4),
        }
    else:
        result["samples"]["clean_daily_day"] = {"status": "missing"}

    result["peak_rss_mb"] = round(peak_rss, 1)
    result["note"] = (
        "I/O 地板 ≠ 构建成本：结构重建还要跑因子计算与清洗链，"
        "本表只用于给 run-a 申请建立下限，不得当作批次预算"
    )
    return result


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 1e6
    except Exception:  # noqa: BLE001 - 缺少 psutil 只降级为 0，不阻断测算
        return 0.0


def structural_rebuild_floor(io_baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """结构重建的 I/O 地板与待测项（**不是**可用预算）。"""
    sample = (io_baseline.get("samples") or {}).get("panel_month") or {}
    months = _month_span(REBUILD_MONTH_START, REBUILD_MONTH_END)
    if sample.get("status") == "missing" or not sample.get("seconds"):
        return {
            "status": "not_measured",
            "months_needed": months,
            "reason": "缺面板月分区样本，无法给地板值",
        }
    floor_seconds = float(sample["seconds"]) * months
    return {
        "status": "lower_bound_only",
        "months_needed": months,
        "month_sample": sample,
        "io_floor_seconds": round(floor_seconds, 2),
        "io_floor_minutes": round(floor_seconds / 60.0, 3),
        "io_floor_hours": round(floor_seconds / 3600.0, 4),
        "must_measure_before_request": [
            "固定一个月分块的 _run_capture(structural=True) 耗时",
            "该分块峰值内存（RSS）",
            "该分块磁盘占用（8 个列组 parquet）",
            "串行/复用方案：按月分块 + 同代价格索引复用",
        ],
        "note": (
            "构建成本 = I/O × 未知倍数（因子计算、清洗、行业 PIT 注入）；"
            "未做分块实测前不得把地板值当批次预算（协议 §8）"
        ),
    }


def _month_span(start: str, end: str) -> int:
    start_year, start_month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    return (end_year - start_year) * 12 + (end_month - start_month) + 1


def structural_measurement(measured: Mapping[str, Any] | None) -> Dict[str, Any]:
    """结构重建固定分块实测与修正报价（复审第 7 项 + r0_003 复核第 3 项）。

    无实测时不得给预算；有实测时按「拟实际使用的块数 ×（每块重复预热 +
    输出日）」报价，另给单大块摊薄下界；墙钟不含写盘、RSS 非峰值的
    限制必须显式登记，未测区间写假设。
    """
    if not measured:
        return {
            "status": "not_measured",
            "requirement": "固定分块（含 7 个月预热）实测：墙钟、峰值 RSS、输出规模、磁盘",
            "blocked": "未实测前不得把 I/O 地板当批次预算（协议 §8）",
        }
    # 字段映射必须显式：缺墙钟就是「未知」，禁止用 0 兜底（复审 r0_002 反例）
    wall = measured.get("wall_seconds", measured.get("elapsed_seconds"))
    kept = int(measured.get("kept_written_days") or measured.get("computed_days") or 0)
    if wall is None or kept <= 0:
        return {
            "status": "unknown",
            "reason": "缺 wall_seconds 或 kept_written_days，禁止按 0 报价",
            "raw_keys": sorted(measured),
        }
    elapsed = float(wall)
    window_days = int(measured.get("window_trading_days") or 0)
    warmup = int(measured.get("warmup_trading_days") or 0)
    # 交易日拆分以修正值为准（145 预热 + 57 输出 = 202；r0_003 附件写 148 预热）
    if window_days <= 0:
        window_days = kept + warmup
    per_kept_day = elapsed / max(kept, 1)
    per_computed_day = elapsed / max(window_days, 1)
    output_days = int(spec.WINDOW["return_intervals"])
    # 拟实际使用的分块方案：run-a 按月分块重建（2018-10~2026-10 = 97 个月），
    # 每块重复预热（预热交易日数取本块实测）
    blocks = int(measured.get("planned_blocks") or 0)
    warmup_per_block = warmup if warmup > 0 else int(spec.WINDOW["feature_warmup_trading_days"])
    if blocks <= 0:
        blocks = _month_span(REBUILD_MONTH_START, REBUILD_MONTH_END)
    avg_output_per_block = output_days / blocks
    upper_seconds = blocks * (warmup_per_block + avg_output_per_block) * per_computed_day
    lower_seconds = output_days * per_kept_day
    disk = measured.get("disk_bytes")
    write = measured.get("write") or {}
    return {
        "status": "measured",
        "schema": "p7_cashflow_split_structural_block/v3",
        "block": measured.get("block"),
        "accounting": {
            "computed_days": window_days,
            "kept_written_days": kept,
            "warmup_trading_days": warmup,
            "window_split_check": measured.get("window_split_check"),
            "seconds_per_kept_day": round(per_kept_day, 4),
            "seconds_per_computed_day": round(per_computed_day, 4),
            "note": "按输出块计费：预热日、计算日、保留并写盘的交易日分列",
        },
        "wall_seconds": round(elapsed, 2),
        "wall_note": "计到捕获结束；写 Parquet 在停表后发生（见 write）",
        "write_measurement": {
            "seconds": write.get("elapsed_seconds"),
            "bytes": write.get("bytes") or disk,
            "note": "单日验证（20190401）实测；分块写盘按字节比例折算，未整块实测",
        },
        "rss_mb": measured.get("rss_peak_sampled_mb") or measured.get("rss_after_mb"),
        "rss_is_true_peak": bool(measured.get("rss_peak_sampled_mb")),
        "rss_note": (
            "r0_003 分块件的 1,554.4 MB 是捕获前后 RSS 读数，不能充当峰值；"
            "采样峰值以单日验证件为准"
        ),
        "rows": measured.get("rows"),
        "columns_n": measured.get("columns_n"),
        "disk_bytes": disk if disk is not None else "not_measured",
        "quotes": {
            "planned_blocks": blocks,
            "warmup_trading_days_per_block": warmup_per_block,
            "avg_output_days_per_block": round(avg_output_per_block, 1),
            "upper_hours": round(upper_seconds / 3600.0, 2),
            "upper_basis": "按月分块（每块重复预热）× 实测每计算日秒数；串行、无块间复用",
            "lower_hours": round(lower_seconds / 3600.0, 2),
            "lower_basis": "单大块一次预热摊薄（1838 输出日 × 实测每保留日秒数）",
            "reuse_options": [
                "按月分块串行，块间复用价格索引与训练帧（复用比例未实测）",
                "同 identity 缓存命中即跳过（命中数未实测）",
            ],
            "assumptions": [
                "墙钟 495.6s 不含写盘；写盘按单日验证字节比例折算",
                "F/R 另报的 4–9.5 小时仍以结构输入已就绪为前提",
                "后期股票数增长与序列化开销未计入（r0_003 复核同口径）",
            ],
        },
        "provenance": measured.get("provenance"),
        "content_digest": measured.get("content_digest"),
    }


def build_pilot_costs(
    probe_costs: Mapping[str, Any],
    io_baseline: Mapping[str, Any],
    hash_throughput: Mapping[str, Any],
    extra_measured: Mapping[str, Any] | None = None,
    structural: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """组装 ``pilot_costs.json``（实测 + 协议参照 + 投影 + 授权状态）。"""
    measured = {
        "probe": dict(probe_costs),
        "io_baseline": dict(io_baseline),
        "hash_throughput": dict(hash_throughput),
    }
    if extra_measured:
        measured.update(dict(extra_measured))

    probe_seconds = float(probe_costs.get("total_seconds") or 0.0)
    reference = dict(spec.BUDGET_REFERENCE)
    projections: List[Dict[str, Any]] = [
        {
            "batch": "run-a",
            "command": (
                "python -m scripts.v2_p7.cashflow_sleeve_split.run run-a "
                "--campaign cf_split_v1 --parent <prepare_run> --run <a_run>"
            ),
            "reference_hours": reference["run_a_hours"],
            "what_it_does": [
                "复用/批建全窗结构特征缓存（含推理端 freshness 与衍生列）",
                "F 与 R 两臂 × 14 折 + 部署重训与隔离注册（**不含 B**）",
                "全域逐日打分与两档 F100/F80/R100/R80 账户",
                "A 审核材料（a_review.md / a_metrics.csv / b_required.csv）",
            ],
            "drivers": [
                "历史结构补建（单列，见 structural_rebuild）",
                "两臂重训（历史 14 折单臂 5h09m，按臂线性拆分）",
                "逐日打分 1,838 日 × 2 臂",
                "四账户回放（历史两档 A 账户为分钟级）",
            ],
            "saving_options": [
                "同身份训练帧复用（两臂共用同一帧，只换列投影）",
                "价格索引 / 排名缓存复用",
                "按折恢复（每折独立完成标记，不整批重跑）",
            ],
            "status": "not_authorized",
        },
        {
            "batch": "run-b",
            "command": (
                "python -m scripts.v2_p7.cashflow_sleeve_split.run run-b "
                "--campaign cf_split_v1 --parent <a_run> --run <b_run>"
            ),
            "reference_hours": reference["run_b_hours"],
            "gate": "审点二通过后单独立项；A 批不预算、不训练 B",
            "what_it_does": [
                "只训练已冻结 G（B 的大腿 + 小腿）",
                "主组合 R80+B20 与四个对照组（现金 / F20 同槽 / 换主仓 / 旧 A80+B20）",
                "压力项与逐日账务对账",
            ],
            "drivers": ["B 单臂训练（独立预算）", "六类账户回放", "公司行为入账"],
            "saving_options": ["复用 run-a 结构缓存与价格索引", "小账户仅在必要日重算"],
            "status": "not_authorized",
        },
        {
            "batch": "verify",
            "command": (
                "python -m scripts.v2_p7.cashflow_sleeve_split.run verify "
                "--campaign cf_split_v1 --parent <b_run> --run <verify_run>"
            ),
            "reference_hours": reference["verify_hours"],
            "what_it_does": [
                "另一条读盘路径从逐日账户/逐笔成交复算（不取 summary）",
                "产权恒等式 1e-6、指标 1e-10 容差核验",
                "压力重放与 20 日块自举",
            ],
            "drivers": ["全量账本复算", "压力重放", "2,000 次自举"],
            "saving_options": ["自举只在主结果有效时执行；资源停止须标 not_run_resource_stop"],
            "status": "not_authorized",
        },
    ]

    return {
        "schema": "p7_cashflow_split_pilot_costs/v1",
        "candidate_id": spec.CANDIDATE_ID,
        "protocol_sha256": spec.protocol_sha256(),
        "spec_sha256": spec.spec_sha256(),
        "measured": measured,
        "measured_totals": {
            "probe_seconds": round(probe_seconds, 3),
            "probe_minutes": round(probe_seconds / 60.0, 3),
            "hash_mb_per_second": hash_throughput.get("mb_per_second"),
            "within_pilot_reference": bool(
                probe_seconds / 60.0 <= float(reference["pilot"]["high_minutes"])
            ),
            "pilot_reference_minutes": reference["pilot"],
        },
        "reference": reference,
        "projections": projections,
        "structural_rebuild": structural_measurement(structural),
        "structural_io_floor": structural_rebuild_floor(io_baseline),
        "data_purchase_yuan": spec.DATA_PURCHASE_YUAN,
        "approval": {
            "threshold_minutes": reference["approval_required_over_minutes"],
            "statement": (
                "run-a / run-b / verify 均超过 20 分钟，须用户按命名批次批准后启动"
                "（协议 §8 末段）；本阶段未启动任何此类批次"
            ),
            "authorized_stages": list(spec.AUTHORIZED_STAGES),
            "machine_hours_consumed_by_prepare": round(probe_seconds / 3600.0, 4),
        },
        "evidence_paths": {
            "readiness": "readiness.json",
            "input_manifest": "input_manifest.json",
            "feature_routes": "feature_routes.json",
            "fold_clock": "fold_clock.csv",
            "spec": "spec.json",
            "tests": "tests.txt",
        },
    }


def load_structural_measurement(campaign_dir: Path) -> Dict[str, Any] | None:
    """读取 campaign 下的结构分块实测（读入前校验附件自摘要；无则返回 None）。

    r0_003 复核：父 run 消费测量附件前必须校验其自校验摘要
    （``content_digest``）；校验失败即拒绝（``BudgetError``）。
    """
    import json

    measurements = Path(campaign_dir) / "measurements"
    # 优先用修正后的历史块实测（v3）；回退到早期 7 个月块
    for name in ("structural_block_hist_2019.json", "structural_block_7m.json"):
        path = measurements / name
        if path.exists():
            break
    else:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    from scripts.v2_p7.cashflow_sleeve_split.structbuild import verify_measurement

    if not verify_measurement(payload):
        raise BudgetError(f"测量附件自校验失败（content_digest 不符）：{path}")
    payload["provenance"] = payload.get("provenance") or {}
    payload["provenance"]["source_file"] = str(path.name)
    return payload


def calibration_hint(probe_result: Mapping[str, Any]) -> Dict[str, Any]:
    """外推提示：把探针实测折算成「全窗打分」的量级下限（只作下限）。"""
    summary = dict(probe_result.get("summary") or {})
    per_day = summary.get("per_day_elapsed_seconds") or {}
    values = [float(value) for value in per_day.values() if value]
    if not values:
        return {"status": "not_measured"}
    avg = sum(values) / len(values)
    days = int(spec.WINDOW["return_intervals"]) + 1
    return {
        "status": "lower_bound_only",
        "avg_seconds_per_probe_day": round(avg, 4),
        "full_window_days": days,
        "read_only_floor_hours": round(avg * days / 3600.0, 2),
        "note": "只含两分区读取；评分、模型预测与账户逻辑均未计入",
    }


__all__ = [
    "REBUILD_MONTH_END",
    "REBUILD_MONTH_START",
    "build_pilot_costs",
    "calibration_hint",
    "load_structural_measurement",
    "measure_io_baseline",
    "structural_measurement",
    "structural_rebuild_floor",
]
