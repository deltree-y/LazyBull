# -*- coding: utf-8 -*-
"""折表、阶段时钟、接缝与部署时钟（协议 §4、§5.2-2、§5.2-3）。

口径来源（不得另立）
====================
- **阶段边界 = 原训练日志实际记录值**（复审整改第 1 项）：
  ``data/models/stock_selection/ml_train_runs.csv`` 逐版本给出
  ``val_start_date/val_end_date``（早停段）、``val_calib_start_date/val_calib_end_date``
  （校准段）、``end_date``（末尾隔离段终点）与 ``test_start_date/test_end_date``（测试段）。
  **不再**用日历比例近似产出 val/es 日期；比例规则只作参照并逐版登记差值。
- **标签成熟日按 T+21 计算**：``features/labels.py::compute_forward_returns`` 的
  卖出价取 ``T+1+horizon`` 开盘 ⇒ 成熟偏移 = 1 + 20 = 21 个交易日。
- 折表 test 域与模型版本取锚批次 ``data/runs/p2a_b0replay_20261008_144307_d79f3f``
  的 ``summary.parquet``，并与 T0.5 ``precheck/fold_check.json`` 逐值对账。
- 接缝与部署段：协议 §4（2026-01-06~02-02 共 20 日沿用末折；02-03~10-29 共 176 日
  固定各臂部署模型）。
- 账户日历：2019-03-29 收盘全现金锚 + 2019-04-01~2026-10-29 开市日。

``fold_clock.csv`` 同时承载折行与接缝/部署段行（``kind`` 列区分）。
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT

#: 锚批次（折表 test 域与模型版本来源；只读）
ANCHOR_RUN_RELATIVE = "data/runs/p2a_b0replay_20261008_144307_d79f3f"
#: T0.5 已核验折表（存在即逐值对账）
T25_PRECHECK_RELATIVE = spec.T25_PRECHECK_RELATIVE

CSV_COLUMNS: Tuple[str, ...] = (
    "kind",
    "split_index",
    "model_version",
    "log_row",
    "log_timestamp",
    "outer_train_start",
    "outer_train_end",
    "es_start",
    "es_end",
    "es_last_usable_info",
    "es_last_usable_sample",
    "es_purged_samples",
    "calib_start",
    "calib_end",
    "calib_last_usable_info",
    "calib_last_usable_sample",
    "calib_purged_samples",
    "tail_isolation_start",
    "tail_isolation_end",
    "tail_last_usable_info",
    "test_start",
    "test_end",
    "test_last_usable_info",
    "maturity_offset",
    "derived_train_end",
    "derived_train_last_label_maturity",
    "derived_val_start",
    "derived_val_start_delta_days",
    "maturity_breach",
    "maturity_breach_purified",
    "account_first_day",
    "account_last_day",
    "account_days",
    "derivation",
    "val_date_status",
    "note",
)


class ClockError(RuntimeError):
    """折表/日历与冻结口径不符（fail-fast）。"""


def load_trade_days(root: Path, start: str, end: str) -> List[str]:
    """交易日历 ∩ [start, end]（``clean/trade_cal`` 开市日）。"""
    import pandas as pd

    path = Path(root) / "data/clean/trade_cal.parquet"
    if not path.exists():
        raise ClockError(f"交易日历缺件：{path}")
    cal = pd.read_parquet(path, columns=["cal_date", "is_open"])
    opened = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str)
    days = sorted(day for day in opened if str(start) <= day <= str(end))
    if not days:
        raise ClockError(f"交易日历在 [{start}, {end}] 内无开市日")
    return days


def full_calendar(root: Path) -> List[str]:
    """全量开市日（升序），用于 T+21 成熟日定位。"""
    import pandas as pd

    path = Path(root) / "data/clean/trade_cal.parquet"
    if not path.exists():
        raise ClockError(f"交易日历缺件：{path}")
    cal = pd.read_parquet(path, columns=["cal_date", "is_open"])
    return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())


def account_calendar(root: Path) -> List[str]:
    """账户日历 = 锚日 + 窗口内开市日（1,839 点）。"""
    window = load_trade_days(root, spec.WINDOW["start"], spec.WINDOW["end"])
    calendar = [spec.WINDOW["cash_anchor"], *window]
    if len(calendar) != int(spec.WINDOW["nav_points"]):
        raise ClockError(
            f"账户日历点数不符（协议 §4 = {spec.WINDOW['nav_points']}）：{len(calendar)}"
        )
    return calendar


def maturity_day(day: str, calendar: Sequence[str], offset: int | None = None) -> str:
    """``day`` 之后第 ``offset`` 个交易日的日期（标签成熟日，T+21）。"""
    span = int(offset if offset is not None else spec.SPLIT_CLOCK["label_maturity_offset"])
    position = {item: index for index, item in enumerate(calendar)}
    if day not in position:
        raise ClockError(f"日期不在交易日历内：{day}")
    target = position[day] + span
    if target >= len(calendar):
        raise ClockError(f"{day} 之后不足 {span} 个交易日，无法确定标签成熟日")
    return calendar[target]


def _resolve_split_dates(
    outer_start: str, outer_end: str, val_ratio: float, label_delta: int, root: Path
) -> Dict[str, Any]:
    """按冻结规则重算切分（**参照值**，不产出时钟；复审第 1 项）。"""
    days = load_trade_days(root, outer_start, outer_end)
    n = len(days)
    if n <= 2 * label_delta + 2:
        raise ClockError(f"训练窗过短，无法容纳隔离带：{outer_start}~{outer_end}（n={n}）")
    n_val = max(1, -(-int(n * val_ratio) // 1))  # ceil
    n_train = n - n_val
    train_end_index = max(n_train - label_delta - 1, 0)
    val_start_index = n_train + label_delta
    if val_start_index >= n:
        raise ClockError(f"验证段起点越界：{outer_start}~{outer_end}")
    return {
        "n_days": n,
        "n_val": n_val,
        "n_train": n_train,
        "train_end": days[train_end_index],
        "val_start": days[val_start_index],
        "val_end": days[-1],
        "gap_train_to_val_days": val_start_index - train_end_index - 1,
    }


def load_training_log(root: Path) -> Dict[int, Dict[str, Any]]:
    """逐版本读取原训练日志的阶段边界（含 CSV 物理行号）。

    物理行号约定：CSV 表头占第 1 行，数据第 k 行 ⇒ 物理行 = k + 1。
    """
    import pandas as pd

    base = Path(root)
    path = base / spec.TRAINING_LOG["relative"]
    if not path.exists():
        raise ClockError(f"原训练日志缺件：{path}")
    frame = pd.read_csv(path, low_memory=False)
    columns = spec.TRAINING_LOG["columns"]
    missing = sorted({value for value in columns.values()} - set(frame.columns))
    if missing:
        raise ClockError(f"原训练日志缺列：{missing}")

    versions = [*spec.FOLD_VERSIONS, spec.DEPLOY_VERSION]
    table: Dict[int, Dict[str, Any]] = {}
    for version in versions:
        rows = frame[frame[columns["split_index"]].notna()]
        matched = frame.index[frame["model_version"].astype(str) == str(version)].tolist()
        if len(matched) != 1:
            raise ClockError(f"v{version} 在训练日志中的行数不为 1：{len(matched)}")
        if len(rows) == 0:  # pragma: no cover - 防御性分支
            raise ClockError("原训练日志为空")
        index = int(matched[0])
        record = frame.loc[index]
        table[version] = {
            "physical_line": index + 2,
            "timestamp": str(record[spec.TRAINING_LOG["timestamp_column"]]),
            "outer_train_start": str(record[columns["outer_train_start"]]),
            "outer_train_end": str(record[columns["outer_train_end"]]),
            "es_start": str(record[columns["es_start"]]),
            "es_end": str(record[columns["es_end"]]),
            "calib_start": _maybe_day(record[columns["calib_start"]]),
            "calib_end": _maybe_day(record[columns["calib_end"]]),
            "test_start": _maybe_day(record[columns["test_start"]]),
            "test_end": _maybe_day(record[columns["test_end"]]),
            "split_index": _maybe_int(record[columns["split_index"]]),
            "best_iteration": _maybe_int(record[columns["best_iteration"]]),
        }
    return table


def _maybe_day(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in ("", "nan", "None", "NaT"):
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _maybe_int(value: Any) -> int | None:
    text = _maybe_day(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _next_trading_day(day: str, calendar: Sequence[str]) -> str:
    position = {item: index for index, item in enumerate(calendar)}
    if day not in position:
        raise ClockError(f"日期不在交易日历内：{day}")
    index = position[day] + 1
    if index >= len(calendar):
        raise ClockError(f"{day} 无后续交易日，无法确定末尾隔离段起点")
    return calendar[index]


def build_fold_row(
    summary_row: Mapping[str, Any],
    log_row: Mapping[str, Any],
    calendar: Sequence[str],
    root: Path,
) -> Dict[str, Any]:
    """单折的阶段时钟（日志为准 + 比例参照 + 成熟日与跨段检查）。"""
    offset = int(spec.SPLIT_CLOCK["label_maturity_offset"])
    version = int(summary_row["model_version"])
    es_end = log_row["es_end"]
    calib_end = log_row["calib_end"]
    train_end = log_row["outer_train_end"]
    test_start = log_row["test_start"] or str(summary_row["test_start"])

    tail_start = _next_trading_day(calib_end, calendar) if calib_end else ""
    derived = _resolve_split_dates(
        log_row["outer_train_start"],
        log_row["outer_train_end"],
        float(summary_row.get("val_ratio", spec.SPLIT_CLOCK["val_ratio"])),
        int(spec.SPLIT_CLOCK["label_isolation_days"]),
        root,
    )
    derived_maturity = maturity_day(derived["train_end"], calendar, offset)
    # 比例参照与日志是两个来源，直接比较会得出无意义结论：
    # 只核「日志段的标签成熟日不得跨进测试域」这两条，参照值仅登记差值。
    calib_start = log_row.get("calib_start") or ""
    breaches: List[str] = []
    if maturity_day(es_end, calendar, offset) >= test_start:
        breaches.append("es_maturity>=test_start")
    if calib_end and maturity_day(calib_end, calendar, offset) >= test_start:
        breaches.append("calib_maturity>=test_start")
    # 早停 → 校准 的边界同样要查：早停段末端标签不得跨进校准段
    if calib_start and maturity_day(es_end, calendar, offset) >= calib_start:
        breaches.append("es_maturity>=calib_start")
    if train_end != log_row["outer_train_end"]:
        breaches.append("tail_end_mismatch")

    es_purified = last_sample_before(es_end, calib_start or test_start, calendar, offset)
    calib_purified = (
        last_sample_before(calib_end, test_start, calendar, offset)
        if calib_end
        else {"last_sample": "", "purged_n": 0}
    )
    purified_breaches: List[str] = []
    if es_purified["last_sample"] and maturity_day(
        es_purified["last_sample"], calendar, offset
    ) >= (calib_start or test_start):
        purified_breaches.append("es_purified_maturity>=calib_start")
    if (
        calib_purified["last_sample"]
        and maturity_day(calib_purified["last_sample"], calendar, offset) >= test_start
    ):
        purified_breaches.append("calib_purified_maturity>=test_start")

    return {
        "split_index": int(summary_row["split_index"]),
        "model_version": version,
        "log_row": int(log_row["physical_line"]),
        "log_timestamp": log_row["timestamp"],
        "outer_train_start": log_row["outer_train_start"],
        "outer_train_end": train_end,
        "es_start": log_row["es_start"],
        "es_end": es_end,
        "es_last_usable_info": maturity_day(es_end, calendar, offset),
        "calib_start": log_row["calib_start"],
        "calib_end": calib_end,
        "calib_last_usable_info": (maturity_day(calib_end, calendar, offset) if calib_end else ""),
        "tail_isolation_start": tail_start,
        "tail_isolation_end": train_end,
        "tail_last_usable_info": maturity_day(train_end, calendar, offset),
        "test_start": test_start,
        "test_end": str(summary_row["test_end"]),
        "test_last_usable_info": maturity_day(str(summary_row["test_end"]), calendar, offset),
        "maturity_offset": offset,
        "derived_train_end": derived["train_end"],
        "derived_train_last_label_maturity": derived_maturity,
        "derived_val_start": derived["val_start"],
        "derived_val_start_delta_days": _day_delta(
            derived["val_start"], log_row["es_start"], calendar
        ),
        "es_last_usable_sample": es_purified["last_sample"],
        "es_purged_samples": int(es_purified["purged_n"]),
        "calib_last_usable_sample": calib_purified["last_sample"],
        "calib_purged_samples": int(calib_purified["purged_n"]),
        "maturity_breach": ";".join(breaches),
        "maturity_breach_purified": ";".join(purified_breaches),
        "val_ratio": float(summary_row.get("val_ratio", spec.SPLIT_CLOCK["val_ratio"])),
    }


def last_sample_before(
    stage_end: str, next_start: str, calendar: Sequence[str], offset: int
) -> Dict[str, Any]:
    """阶段允许的最后样本日：其标签成熟日**严格早于**下一阶段首日。

    返回 ``{last_sample, purged_n}``；``purged_n`` = 该阶段被净化的样本数。
    """
    if not stage_end or not next_start:
        return {"last_sample": "", "purged_n": 0}
    days = [day for day in calendar if day <= str(stage_end)]
    for day in reversed(days):
        try:
            matured = maturity_day(day, calendar, offset)
        except ClockError:
            continue
        if matured < str(next_start):
            return {"last_sample": day, "purged_n": len([d for d in days if d > day])}
    return {"last_sample": "", "purged_n": len(days)}


def _day_delta(left: str, right: str, calendar: Sequence[str]) -> int | None:
    position = {item: index for index, item in enumerate(calendar)}
    if left not in position or right not in position:
        return None
    return position[right] - position[left]


def load_folds(root: Path) -> List[Dict[str, Any]]:
    """读取锚批次折表 + 原训练日志，并在 T0.5 precheck 存在时逐值对账。"""
    import pandas as pd

    base = Path(root)
    summary_path = base / spec.ANCHOR_SUMMARY_RELATIVE
    if not summary_path.exists():
        raise ClockError(f"锚批次 summary 缺件：{summary_path}")
    summary = pd.read_parquet(summary_path)
    required = {
        "split_index",
        "model_version",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ClockError(f"锚批次 summary 缺列：{missing}")
    if len(summary) != len(spec.FOLD_VERSIONS):
        raise ClockError(f"折数不符冻结值 {len(spec.FOLD_VERSIONS)}：{len(summary)}")

    log = load_training_log(base)
    calendar = full_calendar(base)
    precheck = _load_precheck(base)

    folds: List[Dict[str, Any]] = []
    for _, record in summary.sort_values("split_index").iterrows():
        split_index = int(record["split_index"])
        version = int(record["model_version"])
        if version != spec.FOLD_VERSIONS[split_index]:
            raise ClockError(
                f"split{split_index:02d} 模型版本不符冻结序列：{version} != "
                f"{spec.FOLD_VERSIONS[split_index]}"
            )
        row = {
            "split_index": split_index,
            "model_version": version,
            "outer_train_start": str(record["train_start"]),
            "outer_train_end": str(record["train_end"]),
            "test_start": str(record["test_start"]),
            "test_end": str(record["test_end"]),
            "val_ratio": float(record.get("val_ratio", spec.SPLIT_CLOCK["val_ratio"])),
        }
        logged = log[version]
        for key in ("outer_train_start", "outer_train_end", "test_start", "test_end"):
            if str(logged[key]) != str(row[key]):
                raise ClockError(f"v{version} {key} 日志与锚表不符：{logged[key]} != {row[key]}")
        row.update(build_fold_row(row, logged, calendar, base))
        if precheck is not None:
            _assert_precheck(precheck, row)
        folds.append(row)

    _assert_fold_contiguity(folds, base)
    return folds


def deploy_row(root: Path) -> Dict[str, Any]:
    """部署模型（v24022）的阶段时钟（日志记录，无 test 段）。

    r0_003 复核第 2 项：部署行与历史折同口径给出**早停净化**边界——
    原早停段截至 2025-08-28、校准始于 08-29 ⇒ 早停最后可用信号日 =
    2025-07-30（T+21 = 08-28），剔除其后 21 个早停信号日。部署训练入口
    必须同时收到原始段与净化段两套边界，run-a 实际训练键逐键核对。
    """
    base = Path(root)
    log = load_training_log(base)[spec.DEPLOY_VERSION]
    calendar = full_calendar(base)
    offset = int(spec.SPLIT_CLOCK["label_maturity_offset"])
    tail_start = _next_trading_day(log["calib_end"], calendar)
    es_purified = last_sample_before(
        log["es_end"], log["calib_start"] or log["test_start"], calendar, offset
    )
    breaches: List[str] = []
    if es_purified["last_sample"] and maturity_day(
        es_purified["last_sample"], calendar, offset
    ) >= (log["calib_start"] or log["test_start"]):
        breaches.append("deploy_es_purified_maturity>=calib_start")
    return {
        "model_version": int(spec.DEPLOY_VERSION),
        "log_row": int(log["physical_line"]),
        "log_timestamp": log["timestamp"],
        "outer_train_start": log["outer_train_start"],
        "outer_train_end": log["outer_train_end"],
        "es_start": log["es_start"],
        "es_end": log["es_end"],
        "es_last_usable_info": maturity_day(log["es_end"], calendar, offset),
        "es_last_usable_sample": es_purified["last_sample"],
        "es_purged_samples": int(es_purified["purged_n"]),
        "es_purification_status": "purified" if es_purified["last_sample"] else "purge_failed",
        "es_purification_breach": ";".join(breaches),
        "calib_start": log["calib_start"],
        "calib_end": log["calib_end"],
        "calib_last_usable_info": maturity_day(log["calib_end"], calendar, offset),
        "tail_isolation_start": tail_start,
        "tail_isolation_end": log["outer_train_end"],
        "tail_last_usable_info": maturity_day(log["outer_train_end"], calendar, offset),
        "test_start": log["test_start"],
        "test_end": log["test_end"],
        "test_last_usable_info": (
            maturity_day(log["test_end"], calendar, offset) if log["test_end"] else ""
        ),
        "maturity_offset": offset,
        "best_iteration": log["best_iteration"],
    }


def _load_precheck(root: Path) -> Dict[str, Any] | None:
    path = root / T25_PRECHECK_RELATIVE / "fold_check.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen_folds_match", True):
        raise ClockError("T0.5 fold_check 未通过（frozen_folds_match=false）")
    return payload


def _assert_precheck(precheck: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    rows = {int(item["split_index"]): item for item in precheck.get("rows") or []}
    check = rows.get(int(row["split_index"]))
    if check is None:
        return
    pairs = (
        ("model_version", "model_version"),
        ("test_start", "test_start"),
        ("test_end", "test_end"),
        ("train_start", "outer_train_start"),
        ("train_end", "outer_train_end"),
    )
    for precheck_key, row_key in pairs:
        if precheck_key in check and str(check[precheck_key]) != str(row.get(row_key)):
            raise ClockError(
                f"split{int(row['split_index']):02d} {row_key} 与 T0.5 不符："
                f"{check[precheck_key]} != {row.get(row_key)}"
            )


def _assert_fold_contiguity(folds: Sequence[Mapping[str, Any]], root: Path) -> None:
    """测试域必须连续、无重复、不得早于训练末端；成熟日不得跨段。"""
    previous_end = ""
    for fold in folds:
        # 原日志边界允许跨段（实测 14/14 折的早停末端标签都跨进校准），
        # 只登记；致命条件是「净化后」仍跨段。
        if fold["maturity_breach_purified"]:
            raise ClockError(
                f"split{int(fold['split_index']):02d} 净化后仍跨段："
                f"{fold['maturity_breach_purified']}"
            )
        if fold["test_start"] <= fold["outer_train_end"]:
            raise ClockError(f"split{int(fold['split_index']):02d} 测试起点不晚于训练末端")
        if previous_end and fold["test_start"] <= previous_end:
            raise ClockError(f"split{int(fold['split_index']):02d} 测试域与前折重叠")
        previous_end = str(fold["test_end"])
    last = folds[-1]
    if int(last["split_index"]) != len(spec.FOLD_VERSIONS) - 1:
        raise ClockError("末折序号不符")
    if str(last["test_end"]) != "20260105":
        raise ClockError(f"末折测试末端不符协议 §4（20260105）：{last['test_end']}")
    days = load_trade_days(root, folds[0]["test_start"], folds[-1]["test_end"])
    covered: List[str] = []
    for fold in folds:
        covered.extend(load_trade_days(root, fold["test_start"], fold["test_end"]))
    if covered != days:
        raise ClockError("14 折测试域并集与连续交易日历不一致（存在缺口或插入日）")


def build_timeline(folds: Sequence[Mapping[str, Any]], root: Path) -> List[Dict[str, Any]]:
    """账户窗口内的模型时间线（折 + 接缝 + 部署），必须无缝覆盖全窗。"""
    calendar = account_calendar(root)
    start, end = spec.WINDOW["start"], spec.WINDOW["end"]
    last_version = int(folds[-1]["model_version"])

    entries: List[Dict[str, Any]] = []
    for fold in folds:
        first = max(str(fold["test_start"]), start)
        last = min(str(fold["test_end"]), end)
        if first > last:
            continue
        entries.append(
            {
                "kind": "fold",
                "split_index": int(fold["split_index"]),
                "model_version": int(fold["model_version"]),
                "first": first,
                "last": last,
                "source": "anchor summary test 域",
            }
        )
    seam = spec.WINDOW["seam"]
    entries.append(
        {
            "kind": "seam",
            "split_index": int(folds[-1]["split_index"]),
            "model_version": last_version,
            "first": seam[0],
            "last": seam[1],
            "source": "协议 §4 接缝（沿用末折）",
        }
    )
    deploy = spec.WINDOW["deploy"]
    entries.append(
        {
            "kind": "deploy",
            "split_index": -1,
            "model_version": spec.DEPLOY_VERSION,
            "first": deploy[0],
            "last": deploy[1],
            "source": "协议 §4 部署段（固定各臂部署模型）",
        }
    )

    covered: List[str] = []
    previous = ""
    for entry in entries:
        days = load_trade_days(root, entry["first"], entry["last"])
        entry["account_days"] = len(days)
        if previous and entry["first"] <= previous:
            raise ClockError(f"模型时间线重叠：{entry['kind']} 起于 {entry['first']}")
        previous = entry["last"]
        covered.extend(days)
    expected = [day for day in calendar if day != spec.WINDOW["cash_anchor"]]
    if covered != expected:
        raise ClockError(
            "模型时间线未无缝覆盖账户窗口：" f"covered={len(covered)} expected={len(expected)}"
        )
    return entries


def _assert_deploy_purification(deploy: Mapping[str, Any]) -> None:
    """部署早停净化对照冻结期望（r0_003 复核第 2 项；不符即 fail-fast）。"""
    expected_last = str(spec.SPLIT_CLOCK.get("deploy_es_last_usable_expected") or "")
    if expected_last and str(deploy["es_last_usable_sample"]) != expected_last:
        raise ClockError(
            f"部署早停最后可用信号日不符冻结期望：{deploy['es_last_usable_sample']} != "
            f"{expected_last}（早停段截至 {deploy['es_end']}、校准始于 {deploy['calib_start']}）"
        )
    expected_purged = int(spec.SPLIT_CLOCK.get("deploy_es_purged_expected") or 0)
    if expected_purged and int(deploy["es_purged_samples"]) != expected_purged:
        raise ClockError(
            f"部署早停净化剔除数不符冻结期望：{deploy['es_purged_samples']} != {expected_purged}"
        )
    if deploy.get("es_purification_breach"):
        raise ClockError(f"部署早停净化后仍越界：{deploy['es_purification_breach']}")


def build_fold_clock(root: Path | None = None) -> Dict[str, Any]:
    """构建折时钟文档（提交物 ``fold_clock.csv`` 的正文与校验结果）。"""
    base = Path(root) if root is not None else REPO_ROOT
    folds = load_folds(base)
    timeline = build_timeline(folds, base)
    deploy = deploy_row(base)
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for entry in timeline:
        by_kind.setdefault(str(entry["kind"]), []).append(entry)

    seam_days = sum(item["account_days"] for item in by_kind.get("seam") or [])
    deploy_days = sum(item["account_days"] for item in by_kind.get("deploy") or [])
    if seam_days != int(spec.WINDOW["seam_days"]):
        raise ClockError(f"接缝交易日数不符（协议 §4 = 20）：{seam_days}")
    if deploy_days != int(spec.WINDOW["deploy_days"]):
        raise ClockError(f"部署段交易日数不符（协议 §4 = 176）：{deploy_days}")
    # 协议 §4 的 20260105 = 校准段末端 + T+21；末尾隔离段的标签被整段弃用
    if deploy["calib_last_usable_info"] != "20260105":
        raise ClockError(
            "部署模型校准段末端的标签成熟日不符协议 §4（20260105）："
            f"{deploy['calib_last_usable_info']}"
        )
    if str(spec.WINDOW["deploy"][0]) <= deploy["tail_isolation_end"]:
        raise ClockError("部署段首日未晚于部署模型训练末端（时序倒置）")
    # r0_003 复核第 2 项：部署早停净化必须与历史折同口径（期望值按日志+日历冻结）
    _assert_deploy_purification(deploy)

    rows: List[Dict[str, Any]] = []
    windows = {int(item["split_index"]): item for item in by_kind.get("fold") or []}
    for fold in folds:
        window = windows.get(int(fold["split_index"]))
        rows.append(
            {
                "kind": "fold",
                "account_first_day": window["first"] if window else "",
                "account_last_day": window["last"] if window else "",
                "account_days": window["account_days"] if window else 0,
                "derivation": "training_log + anchor summary",
                "val_date_status": spec.SPLIT_CLOCK["segment_date_status"],
                "note": _fold_note(fold, window),
                **{key: fold[key] for key in _FOLD_FIELDS},
            }
        )
    for kind in ("seam", "deploy"):
        for entry in by_kind.get(kind) or []:
            row: Dict[str, Any] = {
                "kind": kind,
                "split_index": entry["split_index"],
                "model_version": entry["model_version"],
                "test_start": entry["first"],
                "test_end": entry["last"],
                "account_first_day": entry["first"],
                "account_last_day": entry["last"],
                "account_days": entry["account_days"],
                "derivation": entry["source"],
                "val_date_status": "documented_by_protocol",
                "note": _segment_note(kind),
            }
            if kind == "deploy":
                # r0_003 复核第 2 项：部署行与折行同口径登记早停净化边界
                # （原始段 es_start/es_end 与净化段 last_sample/purged 同表并列）
                row.update(
                    {
                        "es_start": deploy["es_start"],
                        "es_end": deploy["es_end"],
                        "es_last_usable_info": deploy["es_last_usable_info"],
                        "es_last_usable_sample": deploy["es_last_usable_sample"],
                        "es_purged_samples": deploy["es_purged_samples"],
                        "calib_start": deploy["calib_start"],
                        "calib_end": deploy["calib_end"],
                        "calib_last_usable_info": deploy["calib_last_usable_info"],
                        "tail_isolation_start": deploy["tail_isolation_start"],
                        "tail_isolation_end": deploy["tail_isolation_end"],
                        "tail_last_usable_info": deploy["tail_last_usable_info"],
                    }
                )
            rows.append(row)

    deltas = {str(fold["model_version"]): fold["derived_val_start_delta_days"] for fold in folds}
    return {
        "schema": "p7_cashflow_split_fold_clock/v2",
        "candidate_id": spec.CANDIDATE_ID,
        "protocol_sha256": spec.protocol_sha256(base),
        "spec_sha256": spec.spec_sha256(),
        "rows": rows,
        "deploy": deploy,
        "checks": {
            "folds": len(folds),
            "fold_versions": [int(fold["model_version"]) for fold in folds],
            "test_domain_contiguous": True,
            "account_calendar_points": len(account_calendar(base)),
            "account_window_days": int(spec.WINDOW["return_intervals"]),
            "seam_days": seam_days,
            "deploy_days": deploy_days,
            "timeline_covers_window": True,
            "segment_dates_source": spec.TRAINING_LOG["relative"],
            "segment_date_status": spec.SPLIT_CLOCK["segment_date_status"],
            "label_maturity_offset": int(spec.SPLIT_CLOCK["label_maturity_offset"]),
            "label_maturity_rule": "卖出价取 T+1+horizon 开盘 ⇒ 成熟日 = T+21 交易日",
            "maturity_breach_free": True,
            "derived_reference_status": spec.SPLIT_CLOCK["derived_reference_status"],
            "derived_val_start_delta_days": deltas,
            "derived_val_start_mismatches": {
                version: delta for version, delta in deltas.items() if delta not in (0, None)
            },
            "derived_reference_note": spec.SPLIT_CLOCK["derived_reference_note"],
            "log_rows": {str(fold["model_version"]): fold["log_row"] for fold in folds},
            "deploy_log_row": deploy["log_row"],
            "deploy_training_history": {
                "train_start": deploy["outer_train_start"],
                "train_end": deploy["outer_train_end"],
                "early_stop_end": deploy["es_end"],
                "calibration_end": deploy["calib_end"],
                "tail_isolation": [deploy["tail_isolation_start"], deploy["tail_isolation_end"]],
                "last_matured_label": deploy["calib_last_usable_info"],
                "tail_purged_maturity": deploy["tail_last_usable_info"],
                "es_last_usable_sample": deploy["es_last_usable_sample"],
                "es_purged_samples": deploy["es_purged_samples"],
                "es_purification_status": deploy["es_purification_status"],
                "expected": {
                    "es_last_usable_sample": spec.SPLIT_CLOCK.get("deploy_es_last_usable_expected"),
                    "es_purged_samples": spec.SPLIT_CLOCK.get("deploy_es_purged_expected"),
                },
                "status": "logged",
                "note": (
                    "校准段末端 20251203 的 T+21 成熟日 = 20260105（协议 §4 的读数）；"
                    "末尾隔离段 20251204~20260105 的标签整体弃用。部署早停按历史折同口径"
                    "剔除最后 21 个早停信号日（最后可用信号日 20250730，T+21=20250828）；"
                    "部署训练入口收到原始段与净化段两套边界，run-a 实际训练键逐键核对"
                ),
            },
        },
    }


#: 折行写进 CSV 的字段（其余列走统一空值）
_FOLD_FIELDS: Tuple[str, ...] = (
    "split_index",
    "model_version",
    "log_row",
    "log_timestamp",
    "outer_train_start",
    "outer_train_end",
    "es_start",
    "es_end",
    "es_last_usable_info",
    "es_last_usable_sample",
    "es_purged_samples",
    "calib_start",
    "calib_end",
    "calib_last_usable_info",
    "calib_last_usable_sample",
    "calib_purged_samples",
    "tail_isolation_start",
    "tail_isolation_end",
    "tail_last_usable_info",
    "test_start",
    "test_end",
    "test_last_usable_info",
    "maturity_offset",
    "derived_train_end",
    "derived_train_last_label_maturity",
    "derived_val_start",
    "derived_val_start_delta_days",
    "maturity_breach",
)


def _fold_note(fold: Mapping[str, Any], window: Mapping[str, Any] | None) -> str:
    if window is None:
        return "测试域完全早于账户窗口（仅作技术对照）"
    if str(fold["test_start"]) < spec.WINDOW["start"]:
        return (
            f"测试域起于 {fold['test_start']}，账户窗口自 {spec.WINDOW['start']} 起"
            f"⇒ 账户内仅用 {window['first']}~{window['last']}"
        )
    return ""


def _segment_note(kind: str) -> str:
    if kind == "seam":
        return "沿用末折模型，不训练新模型；折与折之间连续持仓与挂单"
    return (
        "固定各臂部署模型（v24022）；训练历史 2020-01-06~2026-01-05，"
        "该文件 2026-09-07 才生成，此处复原的是历史决策"
    )


def write_fold_clock_csv(clock: Mapping[str, Any], path: Path) -> str:
    """写 ``fold_clock.csv``（稳定机器键、UTF-8、LF）。"""
    from scripts.v2_p7.common import sha256_bytes

    rows = list(clock.get("rows") or [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    data = buffer.getvalue().encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return sha256_bytes(data)


__all__ = [
    "ANCHOR_RUN_RELATIVE",
    "CSV_COLUMNS",
    "ClockError",
    "T25_PRECHECK_RELATIVE",
    "account_calendar",
    "build_fold_clock",
    "build_fold_row",
    "build_timeline",
    "deploy_row",
    "full_calendar",
    "load_folds",
    "load_trade_days",
    "load_training_log",
    "maturity_day",
    "write_fold_clock_csv",
]
