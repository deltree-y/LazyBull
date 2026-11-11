# -*- coding: utf-8 -*-
"""P7 崩盘窗规格冻结（R1-T05-03 处置；用户 2026-11-11 裁决取市场指数基座）。

冻结内容
========
- **基座**：宽基指数 ``000300.SH`` 日收盘（价格指数），2015 等历史崩盘因此进入覆盖；
  指数快照落 P7 证据根并由 SHA256 引用，不写生产 raw/clean。
- **算法**：复用 P5b 冻结的峰到谷 episode 选择（对数累计曲线；深度降序、最多 3 个
  互不重叠、同深取较早 peak；窗口 = 收益索引 ``[peak, trough)``）。
- **容差与判定**：``tolerance_pp = 0``，``ΔMaxDD = base − arm``（改善为正、单位 pp），
  制度重排配对 ΔMaxDD 的 95% 区间下界 > 0 才算改善，否则触发 E4-1 否决。
- **身份**：规格路径 + SHA + 算法模块 SHA + 快照 SHA 全部进入 prereg，改任一即改身份。

用法
====
``python scripts/v2_p7/crash_spec.py --out <governance_dir>/specs/crash_windows_v2.json``
（首次运行会联网取指数并落快照；快照或规格已存在即拒绝覆盖。）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WriteScopeError,
    assert_within_write_roots,
    resolve_write_roots,
    sha256_file,
    write_json,
)
from index_source import (  # noqa: E402
    INDEX_CODE,
    INDEX_START,
    IndexSourceError,
    coverage_report,
    fetch_index_daily,
    log_returns,
    normalize_index_frame,
    read_snapshot,
    resolve_end_date,
    verify_small_window,
    write_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA = REPO_ROOT / "data"
SPEC_SCHEMA = "p7_crash_windows_spec/v2"
CONTRACT_DATE = "2026-11-11"
BASIS_INDEX = "market_index"
BASIS_A0 = "A0_strategy_chain_nav"
ALLOWED_BASES = (BASIS_INDEX, BASIS_A0)
MAX_WINDOWS = 3
MIN_WINDOWS = 2
RULE_YEAR = "year_peak_trough"
RULE_P5B = "p5b_episode"
ALLOWED_RULES = (RULE_YEAR, RULE_P5B)
RULE_MODULES = {
    RULE_YEAR: "scripts/v2_p7/crash_spec.py",
    RULE_P5B: "src/lazybull/v2/hosts/testbed/referee/drawdown/crash_window.py",
}
CRASH_MODULE_REL = RULE_MODULES[RULE_P5B]
DEFAULT_EVAL_START = "20111107"
ANCHOR_RUN_REL = "data/runs/p2a_b0replay_20261008_144307_d79f3f"
DEFAULT_SPEC_REL = "data/paper_shadow/p7_testbed/p7_t05_20261111_r6/specs/crash_windows_v6.json"
DEFAULT_SNAPSHOT_REL = (
    "data/paper_shadow/p7_testbed/p7_t05_20261111_r6/specs/sources/index_daily_000300SH.csv"
)

DECISION_RULE = (
    "ΔMaxDD = arm − base（两臂 MaxDD 均为共享实现 chain_metrics 的 signed 非正数；"
    "改善为正、单位 pp），容差 0；配对 ΔMaxDD 的 95% 区间下界 > 0 才算改善，否则 E4-1 否决。"
    "例：base=-0.30、arm=-0.20 ⇒ Δ=+0.10 通过；base=-0.20、arm=-0.30 ⇒ Δ=-0.10 否决"
)

#: E3/E4-1 唯一裁决口径 = 现行配对制度重排（R2R-01）；日块自举降级为诊断
E3_IMPLEMENTATION = "scripts/compare/fold_subset.py"
E3_WRAPPER = "src/lazybull/v2/evidence/regime_resample.py"
CORRELATION_FIELDS = {
    "full_sample": 10,
    "rebalance_10d": 1,
    "crash_segment": 5,
}


def build_interval_spec() -> Dict[str, Any]:
    """区间与日期对齐规格（R2R-01/R2R-03）：E3 沿用配对制度重排，日块仅诊断。"""
    return {
        "e3_paired_delta_mdd": {
            "field": "delta_mdd_e3",
            "sample_unit": "fold",
            "implementation": f"{E3_IMPLEMENTATION}::bootstrap_delta",
            "implementation_sha256": sha256_file(REPO_ROOT / E3_IMPLEMENTATION),
            "wrapper": f"{E3_WRAPPER}::paired_regime_bootstrap",
            "wrapper_sha256": sha256_file(REPO_ROOT / E3_WRAPPER),
            "sampling": (
                "对**完整折**有放回重采样（14 折 = 14 个制度积木）；两臂共用同一折排列集，"
                "各自按原口径重建链式净值后重算 MaxDD"
            ),
            "fold_set": "锚批次 14 折 test 域（20181126~20260105），两侧必须同集合",
            "n_boot": 1000,
            "seed": 42,
            "interval": "95% percentile [2.5%, 97.5%]",
            "pass_rule": "下界 > 0（ΔMaxDD = arm − base，signed 非正数之差，改善为正）",
            "min_folds": 10,
            "sign_examples": {
                "improve": {"base_mdd": -0.30, "arm_mdd": -0.20, "delta": 0.10, "verdict": "pass"},
                "worsen": {"base_mdd": -0.20, "arm_mdd": -0.30, "delta": -0.10, "verdict": "veto"},
            },
        },
        "diagnostics": {
            "daily_block_bootstrap": {
                "field": "delta_mdd_daily_block_diag",
                "decision_role": "diagnostic_only",
                "sample_unit": "day",
                "block_length": 10,
                "resamples": 10000,
                "seed": 42,
                "interval": "95% percentile [2.5%, 97.5%]",
                "reason": (
                    "崩盘窗稳健性读数；样本单位=日，与制度重排的折单位不同，"
                    "不得替代 E3、不得进入 FULL_GATE"
                ),
                "forbidden": "禁止作为 E4-1 判据或与 E3 结果混报",
            }
        },
        "correlation": {
            "definitions": {
                name: {
                    "field": f"corr_upper_{name}",
                    "series": series,
                    "block_length": block,
                    "resamples": 10000,
                    "seed": 42,
                    "upper_bound": "97.5% 分位",
                    "min_observations": minimum,
                }
                for name, (series, block, minimum) in {
                    "full_sample": ("两侧日收益（同日对齐）", 10, 250),
                    "rebalance_10d": ("10 日非重叠收益（按 B0 调仓日对齐）", 1, 40),
                    "crash_segment": ("窗口并集内可配对日收益（条件相关）", 5, 60),
                }.items()
            },
            "pass_rule": "三口径区间上界均 < 0.3",
            "insufficient_sample": "not_estimable ⇒ inconclusive_frozen（不补 0、不删窗口）",
            "note": (
                "相关区间是 P7 为本门新采用的预登记程序（非替换任何既有尺子）；"
                "与 E3 折重排的单位不同，字段名独立"
            ),
        },
        "alignment": {
            "b0_full_history": "B0 首个净值日起至共同数据截止",
            "common_oos": "与锚批次 A0 test 域 [20181126, 20260105] 的交集",
            "e3_fold_alignment": "E3 使用 14 折 test 域；两臂都须在该折内有收益，缺折即拒绝",
            "windows": "窗口按指数冻结、不裁剪；只在双方都有收益的日期上配对",
            "partial_coverage": "逐窗登记可配对日数；0 可配对日的窗口记 not_pairable，不补 0",
            "min_pairable_windows": 2,
            "min_union_pairable_days": 60,
            "insufficient": "低于上述下限 ⇒ inconclusive_frozen",
            "expected_gap": "2015 窗早于 A0 test 域起点，预期可配对日为 0，须在报告中显式列出",
        },
    }


class CrashSpecError(RuntimeError):
    """崩盘规格不可构造/不合规（一律 fail-closed）。"""


def _load_trading_days() -> List[str]:
    """SSE 开市日（YYYYMMDD 升序）。"""
    cal = pd.read_parquet(DATA / "raw" / "trade_cal.parquet")
    cal = cal[(cal["exchange"] == "SSE") & (cal["is_open"] == 1)]
    return sorted(str(int(value)) for value in cal["cal_date"])


def _load_clean_cutoff() -> str:
    """clean/daily 水位 = 指数基座的截止日（禁止使用截止后数据选窗）。"""
    files = sorted((DATA / "clean" / "daily").glob("*.parquet"))
    if not files:
        raise CrashSpecError("缺少 clean/daily 分区，无法确定指数基座截止日")
    return files[-1].stem.replace("-", "")


def load_anchor_series(run_rel: str = ANCHOR_RUN_REL) -> Dict[str, Any]:
    """A 基座：锚批次逐折收益串接，复利须与 chain_nav 末值一致。"""
    run = REPO_ROOT / run_rel
    fold_dirs = sorted(path for path in (run / "folds").iterdir() if path.is_dir())
    if not fold_dirs:
        raise CrashSpecError(f"锚批次缺少逐折目录：{run_rel}/folds")
    returns: List[float] = []
    dates: List[str] = []
    for fold in fold_dirs:
        frame = pd.read_parquet(fold / "daily.parquet", columns=["trade_date", "daily_return"])
        frame = frame.sort_values("trade_date", kind="mergesort")
        values = pd.to_numeric(frame["daily_return"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise CrashSpecError(f"{fold.name} 的 daily_return 存在非有限值")
        returns.extend(values.tolist())
        dates.extend(frame["trade_date"].astype(str).tolist())
    nav = pd.read_parquet(run / "chain_nav.parquet", columns=["nav"])
    if len(nav) != len(returns):
        raise CrashSpecError(f"chain_nav 行数 {len(nav)} 与逐折合计 {len(returns)} 不一致")
    terminal = float(nav["nav"].iloc[-1])
    compounded = float(np.prod(1.0 + np.asarray(returns, dtype=float)))
    if abs(compounded - terminal) > 1e-9 * max(1.0, abs(terminal)):
        raise CrashSpecError("串接复利与 chain_nav 末值不一致（拒绝推导）")
    return {
        "basis": BASIS_A0,
        "log_returns": np.log1p(np.asarray(returns, dtype=float)),
        "dates": dates,
        "source": {
            "run_rel": run_rel,
            "chain_nav_rel": f"{run_rel}/chain_nav.parquet",
            "chain_nav_sha256": sha256_file(run / "chain_nav.parquet"),
            "rows": len(returns),
            "date_range": [dates[0], dates[-1]],
            "terminal_nav": terminal,
            "note": "逐折 daily.parquet 的 daily_return 按 split 顺序串接",
        },
    }


def load_index_series(
    snapshot_path: Path,
    spec_dir: Path,
    index_code: str = INDEX_CODE,
    index_start: str = INDEX_START,
    index_end: Optional[str] = None,
) -> Dict[str, Any]:
    """市场指数基座：优先读冻结快照；快照不存在则联网取数并落快照。"""
    calendar = _load_trading_days()
    cutoff = _load_clean_cutoff()
    end_date = resolve_end_date(index_end, calendar) if index_end else min(calendar[-1], cutoff)
    if end_date > cutoff:
        raise CrashSpecError(f"指数基座截止 {end_date} 晚于 clean 水位 {cutoff}（禁止截止后数据）")
    if snapshot_path.exists():
        frame = read_snapshot(snapshot_path)
        snapshot_sha = sha256_file(snapshot_path)
        fetched = False
    else:
        import os  # noqa: PLC0415

        from dotenv import load_dotenv  # noqa: PLC0415
        from src.lazybull.data.tushare_client import TushareClient  # noqa: PLC0415

        if not os.getenv("TS_TOKEN"):
            load_dotenv(REPO_ROOT / ".env")
        client = TushareClient()
        raw = fetch_index_daily(client, index_code, index_start, end_date)
        frame = normalize_index_frame(raw)
        probe_start = str(frame["trade_date"].iloc[-20])
        probe_end = str(frame["trade_date"].iloc[-1])
        small_check = verify_small_window(client, frame, index_code, probe_start, probe_end)
        snapshot_sha = write_snapshot(frame, snapshot_path)
        fetched = True
    coverage = coverage_report(frame, calendar)
    if not coverage["date_range"][0].startswith("2005"):
        raise CrashSpecError(f"指数起点异常：{coverage['date_range'][0]}")
    dates = list(frame["trade_date"])
    return {
        "basis": BASIS_INDEX,
        "log_returns": np.asarray(log_returns(frame), dtype=float),
        "dates": dates[1:],
        "closes": frame["close"].to_numpy(dtype=float)[1:],
        "source": {
            "provider": "tushare",
            "api": "index_daily",
            "index_code": index_code,
            "snapshot_rel": str(snapshot_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "snapshot_sha256": snapshot_sha,
            "snapshot_fetched_by_this_run": fetched,
            "rows": coverage["rows"],
            "date_range": coverage["date_range"],
            "coverage": coverage,
            "price_kind": "price_index（不含分红再投）",
            "note": "翻页读到空页 + 全字段去重/冲突检测 + 有限且严格为正断言 + 小窗口对账",
            "small_window_check": small_check if fetched else {"skipped": "使用既有冻结快照"},
        },
    }


def derive_year_windows(
    dates: Sequence[str],
    closes: Sequence[float],
    eval_start: str,
    top_n: int = MAX_WINDOWS,
) -> List[Dict[str, Any]]:
    """自然年内峰到谷最大回撤取窗（无自由参数）。

    规则：按自然年切段，年内取 running max 为峰、峰后最深点为谷；只保留**峰与谷都落在
    评价窗内**（``>= eval_start``）的年份；按深度降序取前 ``top_n`` 年，窗口 = [峰, 谷]。

    限制（登记）：跨年崩盘按自然年切分，窗口可能短于真实峰到谷（如 2015-06→2016-01）。
    """
    values = np.asarray(closes, dtype=float)
    labels = list(dates)
    rows: List[Dict[str, Any]] = []
    for year in sorted({label[:4] for label in labels}):
        positions = [index for index, label in enumerate(labels) if label.startswith(year)]
        if len(positions) < 2:
            continue
        segment = values[positions]
        running = np.maximum.accumulate(segment)
        drawdown = segment / running - 1.0
        trough_local = int(np.argmin(drawdown))
        peak_local = int(np.argmax(segment[: trough_local + 1]))
        peak_date = labels[positions[peak_local]]
        trough_date = labels[positions[trough_local]]
        if peak_date < eval_start or trough_date < eval_start:
            continue
        depth_log = float(np.log(values[positions[trough_local]] / values[positions[peak_local]]))
        rows.append(
            {
                "year": year,
                "start_date": peak_date,
                "end_date": trough_date,
                "trading_days": trough_local - peak_local + 1,
                "depth_log": depth_log,
            }
        )
    if len(rows) < MIN_WINDOWS:
        raise CrashSpecError(
            f"评价窗内可按年取窗的崩盘段仅 {len(rows)} 个 < 下限 {MIN_WINDOWS}"
            "⇒ 覆盖不足，按 inconclusive_frozen 处理"
        )
    ordered = sorted(rows, key=lambda item: (item["depth_log"], item["start_date"]))
    selected = ordered[: int(top_n)]
    windows = []
    for rank, item in enumerate(selected):
        windows.append(
            {
                "rank": rank,
                "year": item["year"],
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "trading_days": int(item["trading_days"]),
                "depth_pp": round(float(log_to_percent_value(item["depth_log"])), 6),
            }
        )
    return sorted(windows, key=lambda item: item["start_date"])


def log_to_percent_value(log_value: float) -> float:
    """对数收益转百分点（与 P5b ``log_to_percent`` 同口径：×100）。"""
    return float(log_value) * 100.0


def derive_windows(series: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按 P5b 冻结算法选窗（保留作审计对照，不再作为默认规则）。"""
    from src.lazybull.v2.hosts.testbed.referee.drawdown.crash_window import (  # noqa: PLC0415
        find_drawdown_episodes,
        select_crash_windows,
    )
    from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import log_to_percent  # noqa: PLC0415

    episodes = find_drawdown_episodes(series["log_returns"])
    if len(episodes) < MIN_WINDOWS:
        raise CrashSpecError(
            f"崩盘 episode 仅 {len(episodes)} 个 < 下限 {MIN_WINDOWS} ⇒ 覆盖不足，"
            "按 inconclusive_frozen 处理，禁止编造窗口"
        )
    selected = select_crash_windows(episodes, max_windows=MAX_WINDOWS)
    dates = series["dates"]
    windows: List[Dict[str, Any]] = []
    for window in selected:
        start = int(window.peak)
        stop = int(window.trough)
        windows.append(
            {
                "rank": int(window.rank),
                "return_index_start": start,
                "return_index_stop": stop,
                "start_date": dates[start],
                "end_date": dates[stop - 1],
                "trading_days": stop - start,
                "depth_pp": round(float(log_to_percent(window.depth_log)), 6),
            }
        )
    return sorted(windows, key=lambda item: item["start_date"])


def build_spec(
    series: Dict[str, Any],
    windows: Sequence[Dict[str, Any]],
    basis: str,
    rule: str,
    eval_start: str,
) -> Dict[str, Any]:
    """组装规格 JSON（含基座、规则、窗口、容差与覆盖限制）。"""
    if basis not in ALLOWED_BASES:
        raise CrashSpecError(f"未知基座：{basis}")
    if rule not in ALLOWED_RULES:
        raise CrashSpecError(f"未知选窗规则：{rule}")
    if basis == BASIS_INDEX and rule != RULE_YEAR:
        raise CrashSpecError("指数基座只允许年内峰谷规则（P5b episode 规则会跨年合并崩盘）")
    module_rel = RULE_MODULES[rule]
    union_days = sum(item["trading_days"] for item in windows)
    return {
        "schema": SPEC_SCHEMA,
        "contract_date": CONTRACT_DATE,
        "purpose": "P7 崩盘窗与回撤否决口径（E4-1 / 三口径相关第三口径）",
        "basis": basis,
        "basis_resolution": (
            "用户 2026-11-11 裁决：能取得指数 raw 则用市场基准指数，否则用 A0 策略链式净值；"
            "2026-11-11 实测 index_daily(000300.SH) 可取（2005-04 起）⇒ 冻结为市场指数基座。"
            "换基座须在读取任何 P-1/经济结果前以新规格 + 新 prereg（supersede）替换"
        ),
        "algorithm": {
            "rule": rule,
            "module": module_rel,
            "module_sha256": sha256_file(REPO_ROOT / module_rel),
            "selection_rule": (
                "按自然年切段；年内取 running max 为峰、峰后最深点为谷；只保留峰与谷都落在"
                "评价窗内的年份；按深度降序取前 3 年；窗口 = [峰, 谷]（自然日内峰到谷）"
            ),
            "parameters": {"top_n": MAX_WINDOWS, "min_windows": MIN_WINDOWS},
            "eval_start": eval_start,
            "scope_note": "覆盖指数基座全历史，但只保留评价窗内的窗口",
            "limitation": "跨年崩盘按自然年切分，窗口可能短于真实峰到谷（如 2015-06→2016-01）",
        },
        "algorithm_history": {
            "rejected": [
                {
                    "rule": "p5b_episode（running-max 峰到谷 + 深度前 3）",
                    "reason": (
                        "P5b 规则为「最后 500 个收益样本」设计；指数全历史上 2007-10 高点至 "
                        "2021-02 才收复，规则把 2008/2015/2018 合并为单一 episode，"
                        "选出 2005/2007-07/2007-10~2008-11，评价窗内几乎无可用窗口"
                    ),
                    "observed": ["20050309~20050603", "20070620~20070705", "20071017~20081104"],
                },
                {
                    "rule": "zigzag 反转分段（阈值 15%/20%/25%）",
                    "reason": (
                        "20% 阈值下 2021-02→2024-09 合成 871 个交易日单段，25% 阈值下"
                        "2011→2012 合成 506 日；阈值本身是自由参数，需额外预登记与敏感性"
                    ),
                    "observed": ["20080505~20081104", "20210210~20240913", "20150608~20150826"],
                },
            ],
            "decision_note": (
                "两次尝试均在读取任何 B0/VV 经济结果之前，仅基于指数基座本身；"
                "改用无自由参数的年内峰谷规则，并保留上述记录供审计"
            ),
        },
        "source": dict(series["source"]),
        "windows": list(windows),
        "min_windows": MIN_WINDOWS,
        "window_union_trading_days": int(union_days),
        "coverage": {
            "basis_date_range": list(series["source"]["date_range"]),
            "evaluation_start": eval_start,
            "independent_crash_events": len(windows),
            "e2_min_events": 2,
            "limitation": ("指数为价格指数，不含分红再投；仅用于识别市场崩盘窗，不参与收益计量"),
        },
        "tolerance": {"tolerance_pp": 0.0, "decision_rule": DECISION_RULE},
        "interval": build_interval_spec(),
        "reselect_forbidden": True,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrashSpecError(message)


INTERVAL_KEYS = ("e3_paired_delta_mdd", "diagnostics", "correlation", "alignment")


def _require_int(value: Any, name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CrashSpecError(f"`{name}` 必须为 ≥{minimum} 的整数，实际 {value!r}")


def _validate_e3_block(block: Any) -> None:
    if not isinstance(block, dict):
        raise CrashSpecError("interval.e3_paired_delta_mdd 缺失或类型非法")
    required = (
        "field",
        "sample_unit",
        "implementation",
        "implementation_sha256",
        "sampling",
        "fold_set",
        "n_boot",
        "seed",
        "interval",
        "pass_rule",
        "min_folds",
        "sign_examples",
    )
    missing = [key for key in required if key not in block]
    if missing:
        raise CrashSpecError(f"E3 区间规格缺项：{missing}")
    if block["sample_unit"] != "fold":
        raise CrashSpecError("E3 样本单位必须是 fold（制度重排），不得用日块")
    module_rel = str(block["implementation"]).split("::")[0]
    module_path = REPO_ROOT / module_rel
    if not module_path.exists():
        raise CrashSpecError(f"E3 实现不存在：{module_rel}")
    if sha256_file(module_path) != str(block["implementation_sha256"]):
        raise CrashSpecError("E3 实现字节与规格登记不符")
    _require_int(block["n_boot"], "e3.n_boot", 100)
    _require_int(block["seed"], "e3.seed", 0)
    _require_int(block["min_folds"], "e3.min_folds", 2)
    if "95%" not in str(block["interval"]):
        raise CrashSpecError("E3 区间置信水平必须写明 95%")
    if "下界 > 0" not in str(block["pass_rule"]):
        raise CrashSpecError("E3 判定必须是区间下界 > 0")
    examples = block["sign_examples"]
    improve = examples.get("improve") if isinstance(examples, dict) else None
    worsen = examples.get("worsen") if isinstance(examples, dict) else None
    if not isinstance(improve, dict) or not isinstance(worsen, dict):
        raise CrashSpecError("E3 必须写死正反例")
    if round(float(improve.get("delta", 0.0)), 6) != 0.10 or improve.get("verdict") != "pass":
        raise CrashSpecError("E3 正例必须为 +0.10 通过")
    if round(float(worsen.get("delta", 0.0)), 6) != -0.10 or worsen.get("verdict") != "veto":
        raise CrashSpecError("E3 反例必须为 −0.10 否决")


def _validate_diagnostics(block: Any) -> None:
    diagnostics = block.get("daily_block_bootstrap") if isinstance(block, dict) else None
    if not isinstance(diagnostics, dict):
        raise CrashSpecError("interval.diagnostics.daily_block_bootstrap 缺失")
    if diagnostics.get("decision_role") != "diagnostic_only":
        raise CrashSpecError("日块自举必须标 diagnostic_only")
    _require_int(diagnostics.get("block_length"), "diagnostics.block_length", 1)
    _require_int(diagnostics.get("resamples"), "diagnostics.resamples", 100)


def _validate_correlation(block: Any) -> None:
    if not isinstance(block, dict):
        raise CrashSpecError("interval.correlation 缺失或类型非法")
    definitions = block.get("definitions")
    if not isinstance(definitions, dict):
        raise CrashSpecError("interval.correlation.definitions 缺失")
    if sorted(definitions) != sorted(CORRELATION_FIELDS):
        raise CrashSpecError(f"相关口径必须恰为 {sorted(CORRELATION_FIELDS)}")
    for name, definition in definitions.items():
        if not isinstance(definition, dict):
            raise CrashSpecError(f"相关口径 {name} 类型非法")
        _require_int(definition.get("block_length"), f"correlation.{name}.block_length", 1)
        _require_int(definition.get("resamples"), f"correlation.{name}.resamples", 100)
        _require_int(definition.get("seed"), f"correlation.{name}.seed", 0)
        _require_int(definition.get("min_observations"), f"correlation.{name}.min_obs", 10)
        if "97.5%" not in str(definition.get("upper_bound")):
            raise CrashSpecError(f"相关口径 {name} 上界必须写明 97.5% 分位")


def _validate_alignment(block: Any) -> None:
    if not isinstance(block, dict):
        raise CrashSpecError("interval.alignment 缺失或类型非法")
    required = (
        "b0_full_history",
        "common_oos",
        "e3_fold_alignment",
        "windows",
        "partial_coverage",
    )
    missing = [key for key in required if key not in block]
    if missing:
        raise CrashSpecError(f"日期对齐规格缺项：{missing}")
    _require_int(block.get("min_pairable_windows"), "alignment.min_pairable_windows", 1)
    _require_int(block.get("min_union_pairable_days"), "alignment.min_union_pairable_days", 1)


def _validate_interval(interval: Any) -> Dict[str, Any]:
    """统计程序必须齐备（R2R-03）：缺项、类型非法、参数越界一律拒绝。"""
    if not isinstance(interval, dict):
        raise CrashSpecError("规格缺少 interval（统计程序与日期对齐）")
    missing = [key for key in INTERVAL_KEYS if key not in interval]
    extra = [key for key in interval if key not in INTERVAL_KEYS]
    if missing or extra:
        raise CrashSpecError(f"interval 键集不符：缺 {missing} / 多 {extra}")
    _validate_e3_block(interval["e3_paired_delta_mdd"])
    _validate_diagnostics(interval["diagnostics"])
    _validate_correlation(interval["correlation"])
    _validate_alignment(interval["alignment"])
    return interval


def validate_spec(payload: Dict[str, Any], spec_sha256: Optional[str] = None) -> Dict[str, Any]:
    """校验规格结构与冻结口径；不含业务取值重估。"""
    _require(payload.get("schema") == SPEC_SCHEMA, f"规格 schema 不受支持：{payload.get('schema')}")
    basis = payload.get("basis")
    _require(basis in ALLOWED_BASES, f"未知基座：{basis}")
    algorithm = payload.get("algorithm") or {}
    rule = algorithm.get("rule")
    _require(rule in ALLOWED_RULES, f"未知选窗规则：{rule}")
    _require(
        algorithm.get("module") == RULE_MODULES[rule],
        "算法模块不是所选规则的冻结实现",
    )
    _require(
        len(str(algorithm.get("module_sha256") or "")) == 64,
        "算法模块 SHA 非法",
    )
    windows = payload.get("windows") or []
    _require(len(windows) >= MIN_WINDOWS, f"冻结窗口 {len(windows)} 个 < 下限 {MIN_WINDOWS}")
    for item in windows:
        _require(str(item.get("start_date", "")).isdigit(), "窗口 start_date 非法")
        _require(str(item.get("end_date", "")).isdigit(), "窗口 end_date 非法")
        _require(str(item["start_date"]) <= str(item["end_date"]), "窗口起止颠倒")
    tolerance = payload.get("tolerance") or {}
    _require(
        float(tolerance.get("tolerance_pp", 1.0)) == 0.0,
        "容差必须为 0（区间下界规则）",
    )
    _require("95%" in str(tolerance.get("decision_rule") or ""), "决策规则未写明区间下界口径")
    _require(
        "arm − base" in str(tolerance.get("decision_rule") or ""),
        "决策规则必须写明 signed arm − base（禁止 base − arm）",
    )
    source = payload.get("source") or {}
    if basis == BASIS_INDEX:
        _require(source.get("api") == "index_daily", "指数基座必须来自 index_daily")
        _require(len(str(source.get("snapshot_sha256") or "")) == 64, "缺少指数快照 SHA")
        _require(bool(source.get("index_code")), "缺少指数代码")
    else:
        _require(len(str(source.get("chain_nav_sha256") or "")) == 64, "缺少 chain_nav SHA")
    return {
        "spec_path": payload.get("_spec_path"),
        "spec_sha256": spec_sha256,
        "basis": basis,
        "algorithm_module": algorithm.get("module"),
        "algorithm_module_sha256": algorithm.get("module_sha256"),
        "source_sha256": source.get("snapshot_sha256") or source.get("chain_nav_sha256"),
        "source": source,
        "windows": windows,
        "window_union_trading_days": int(payload.get("window_union_trading_days", 0)),
        "tolerance_pp": float(tolerance.get("tolerance_pp")),
        "decision_rule": tolerance.get("decision_rule"),
        "coverage": payload.get("coverage") or {},
        "interval": _validate_interval(payload.get("interval")),
    }


def load_spec(path: Path) -> Dict[str, Any]:
    """读取并校验已冻结规格（治理入口使用）。"""
    if not path.exists():
        raise CrashSpecError(f"崩盘规格缺失：{path}（先运行 crash_spec.py 冻结）")
    data = path.read_bytes()
    import hashlib

    payload = json.loads(data.decode("utf-8"))
    payload["_spec_path"] = str(path)
    validated = validate_spec(payload, spec_sha256=hashlib.sha256(data).hexdigest())
    module = REPO_ROOT / str(validated["algorithm_module"])
    _require(module.exists(), f"算法模块缺失：{validated['algorithm_module']}")
    _require(
        sha256_file(module) == str(validated["algorithm_module_sha256"]),
        "算法模块字节与规格登记不符（实现已被改动，须新规格 + 新 prereg）",
    )
    if validated["basis"] == BASIS_INDEX:
        source = validated["source"]
        snapshot = REPO_ROOT / str(source["snapshot_rel"])
        _require(snapshot.exists(), f"指数快照缺失：{source['snapshot_rel']}")
        _require(
            sha256_file(snapshot) == str(source["snapshot_sha256"]),
            "指数快照字节与规格登记不符",
        )
    return validated


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P7 崩盘窗规格冻结（确定性、只读外部源）")
    parser.add_argument("--out", default=DEFAULT_SPEC_REL, help="规格输出路径")
    parser.add_argument("--basis", default=BASIS_INDEX, choices=list(ALLOWED_BASES))
    parser.add_argument("--rule", default=RULE_YEAR, choices=list(ALLOWED_RULES))
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START, help="评价窗起点")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT_REL, help="指数快照路径")
    parser.add_argument("--index-code", default=INDEX_CODE)
    parser.add_argument("--index-start", default=INDEX_START)
    parser.add_argument("--index-end", default=None, help="结束日（默认取交易日历末日）")
    parser.add_argument("--run", default=ANCHOR_RUN_REL, help="A 基座锚批次目录")
    parser.add_argument("--allow-root", action="append", default=[], help="追加允许写根（测试用）")
    args = parser.parse_args(argv)

    spec_path = Path(args.out)
    spec_path = spec_path if spec_path.is_absolute() else REPO_ROOT / spec_path
    snapshot_path = Path(args.snapshot)
    snapshot_path = snapshot_path if snapshot_path.is_absolute() else REPO_ROOT / snapshot_path
    try:
        roots = resolve_write_roots(args.allow_root)
        spec_path = assert_within_write_roots(spec_path, roots)
        snapshot_path = assert_within_write_roots(snapshot_path, roots)
    except WriteScopeError as error:
        print(f"写域拒绝：{error}", file=sys.stderr)
        return 4
    if spec_path.exists():
        print(f"规格已存在，拒绝覆盖：{spec_path}", file=sys.stderr)
        return 5

    try:
        if args.basis == BASIS_INDEX:
            series = load_index_series(
                snapshot_path, spec_path.parent, args.index_code, args.index_start, args.index_end
            )
        else:
            series = load_anchor_series(args.run)
        if args.rule == RULE_YEAR:
            if "closes" not in series:
                raise CrashSpecError("年内峰谷规则需要价格序列（指数基座）")
            windows = derive_year_windows(
                series["dates"], series["closes"], args.eval_start, MAX_WINDOWS
            )
        else:
            windows = derive_windows(series)
        spec = build_spec(series, windows, args.basis, args.rule, args.eval_start)
    except (CrashSpecError, IndexSourceError, FileNotFoundError) as error:
        print(f"崩盘规格构造失败：{error}", file=sys.stderr)
        return 2

    spec_sha = write_json(spec_path, spec)
    print(f"spec={spec_path}")
    print(f"spec_sha256={spec_sha}")
    print(f"basis={spec['basis']} windows={len(windows)}")
    for item in windows:
        print(
            f"  rank={item['rank']} {item['start_date']}~{item['end_date']} "
            f"days={item['trading_days']} depth={item['depth_pp']}pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
