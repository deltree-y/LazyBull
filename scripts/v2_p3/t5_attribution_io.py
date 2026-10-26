# -*- coding: utf-8 -*-
"""P3 T5 F11 数据影响归因——臂产物与生产数据读取层（P3 规划 §7.5 / §13.3.7）。

职责边界：只按 T5 臂产物布局契约读入输入，不做任何判定/分类——
- 臂目录：``folds/splitNN/{model_version.txt, boundary.json, fingerprint.json,
  predictions.parquet}``（预测行持久化跨路契约：列 trade_date/ts_code/score，
  fingerprint.json 增键 ``predictions_path/predictions_sha256/predictions_rows``，
  缺失 ⇒ 该折产物不完整）、``runs/<batch_id>/`` 批次 parquet、``chain_metrics.json``、
  ``arm_meta.json``、``consumption_snapshot.json``；
- B0' 冻结目录：``freeze_manifest.json``（代码态/数据态身份唯一来源）；
- 生产数据根：panel 侧经 ``PanelDataStore.load_features`` **按消费列投影**读取
  （月粒度），cs_train 侧经 ``Storage.load_cs_train_day`` 逐日读取；交易日历经
  DataLoader（clean 优先、raw 兜底），语义同 ``t05b_spike_runner.list_expected_oos_days``。

判定（F11 词表分类/影响对比/判定表接线）在 ``t5_attribution_sections.py`` 与
``run_data_impact_attribution.py``；本模块全部函数 fail-fast 带上下文。
"""

from __future__ import annotations

import calendar as calendar_module
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lazybull.v2.common.types import FeatureQuery, TradeDate  # noqa: E402

_FOLD_DIR_RE = re.compile(r"^split(\d+)$")
_DATE8_RE = re.compile(r"\d{8}")

#: 折 boundary.json 必需键（窗口 = 外层训练窗 ∪ test 窗，§13.3.7 归因范围）
_BOUNDARY_REQUIRED_KEYS = ("outer_train_start", "outer_train_end", "test_start", "test_end")


@dataclass(frozen=True)
class FoldWindow:
    """单折归因窗口：外层训练窗 ∪ test 窗（YYYYMMDD 闭区间对）。"""

    outer_train_start: str
    outer_train_end: str
    test_start: str
    test_end: str

    def contains(self, day: str) -> bool:
        """日期是否落在窗口内（任一段）。"""
        in_outer = self.outer_train_start <= day <= self.outer_train_end
        in_test = self.test_start <= day <= self.test_end
        return in_outer or in_test

    def span(self) -> tuple[str, str]:
        """覆盖两段的最小连续区间（月迭代外框）。"""
        return (
            min(self.outer_train_start, self.test_start),
            max(self.outer_train_end, self.test_end),
        )


@dataclass(frozen=True)
class FoldInputs:
    """单折归因输入：两臂模型版本 + 各自模型登记消费列 + 归因窗口。"""

    fold: str
    split_index: int
    model_version_arm2: int
    model_version_arm3: int
    columns_arm2: tuple[str, ...]
    columns_arm3: tuple[str, ...]
    window: FoldWindow


def load_json_object(path: Path) -> dict[str, Any] | None:
    """读取 JSON object；文件缺失返回 None，顶层非 object 抛错（带路径上下文）。"""
    path = Path(path)
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return doc


def discover_fold_names(arm2_dir: Path, arm3_dir: Path) -> tuple[list[str], list[str]]:
    """发现两臂折目录名（splitNN）：返回 (排序并集, 对称差错误清单)。"""
    folds2 = _list_fold_dirs(Path(arm2_dir))
    folds3 = _list_fold_dirs(Path(arm3_dir))
    errors: list[str] = []
    for name in sorted(set(folds2) - set(folds3)):
        errors.append(f"折 {name} 仅臂②存在（臂③缺 folds/{name}）")
    for name in sorted(set(folds3) - set(folds2)):
        errors.append(f"折 {name} 仅臂③存在（臂②缺 folds/{name}）")
    return sorted(set(folds2) | set(folds3), key=_fold_sort_key), errors


def _list_fold_dirs(arm_dir: Path) -> list[str]:
    folds_root = Path(arm_dir) / "folds"
    if not folds_root.is_dir():
        return []
    return sorted(
        (d.name for d in folds_root.iterdir() if d.is_dir() and _FOLD_DIR_RE.match(d.name)),
        key=_fold_sort_key,
    )


def _fold_sort_key(name: str) -> int:
    match = _FOLD_DIR_RE.match(name)
    return int(match.group(1)) if match else -1


def load_model_version(arm_dir: Path, fold: str) -> int:
    """读取 ``folds/<fold>/model_version.txt``（该折注册模型版本号）。"""
    path = Path(arm_dir) / "folds" / fold / "model_version.txt"
    if not path.exists():
        raise FileNotFoundError(f"缺折模型版本文件: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"模型版本文件内容非法（须为整数）: {path} 实得 {raw!r}") from exc


def load_boundary_window(arm_dir: Path, fold: str) -> FoldWindow:
    """读取 ``folds/<fold>/boundary.json`` 并抽出归因窗口（必需键缺失即拒绝）。"""
    path = Path(arm_dir) / "folds" / fold / "boundary.json"
    doc = load_json_object(path)
    if doc is None:
        raise FileNotFoundError(f"缺折边界登记: {path}")
    missing = [key for key in _BOUNDARY_REQUIRED_KEYS if doc.get(key) is None]
    if missing:
        raise ValueError(f"折边界缺必需键 {missing}: {path}")
    values = {
        key: _as_date8(str(doc[key]), context=f"{path}:{key}") for key in _BOUNDARY_REQUIRED_KEYS
    }
    return FoldWindow(**values)


def _as_date8(value: str, *, context: str) -> str:
    if not _DATE8_RE.fullmatch(value):
        raise ValueError(f"{context} 须为 YYYYMMDD 八位数字日，实得 {value!r}")
    TradeDate.from_str(value)  # 非法日历日在此抛 ValueError
    return value


def resolve_models_dir(arm_dir: Path) -> Path | None:
    """解析臂内模型注册表目录（隔离根惯例优先；均不存在返回 None 由调用方定成败）。"""
    arm_dir = Path(arm_dir)
    candidates = (
        arm_dir / "root" / "models" / "stock_selection",
        arm_dir / "models" / "stock_selection",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def load_feature_columns(arm_dir: Path, model_version: int) -> list[str]:
    """读取该折模型实际消费列：``v{N}_features.json``（JSON 列名清单）优先，
    旁路 ``v{N}_metadata.json`` 的 ``feature_columns`` 兜底；两者皆缺 ⇒ 报错。"""
    models_dir = resolve_models_dir(arm_dir)
    if models_dir is None:
        raise FileNotFoundError(f"臂目录内无法解析模型注册表目录: {arm_dir}")
    features_path = models_dir / f"v{model_version}_features.json"
    if features_path.exists():
        columns = json.loads(features_path.read_text(encoding="utf-8"))
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise ValueError(f"特征列文件须为字符串清单: {features_path}")
        return list(columns)
    metadata_path = models_dir / f"v{model_version}_metadata.json"
    metadata = load_json_object(metadata_path)
    if metadata is not None and isinstance(metadata.get("feature_columns"), list):
        return [str(c) for c in metadata["feature_columns"]]
    raise FileNotFoundError(
        f"模型 v{model_version} 消费列不可得（{features_path.name} 与 "
        f"{metadata_path.name}:feature_columns 均缺失）: {models_dir}"
    )


def load_fold_inputs(arm2_dir: Path, arm3_dir: Path, fold: str) -> FoldInputs:
    """装配单折归因输入（两臂模型版本/消费列 + 窗口；窗口以臂③为准并核臂②一致）。"""
    version2 = load_model_version(arm2_dir, fold)
    version3 = load_model_version(arm3_dir, fold)
    columns2 = tuple(load_feature_columns(arm2_dir, version2))
    columns3 = tuple(load_feature_columns(arm3_dir, version3))
    window3 = load_boundary_window(arm3_dir, fold)
    window2 = load_boundary_window(arm2_dir, fold)
    if window2 != window3:
        raise ValueError(f"折 {fold} 两臂 boundary.json 窗口不一致: 臂②={window2} vs 臂③={window3}")
    return FoldInputs(
        fold=fold,
        split_index=_fold_sort_key(fold),
        model_version_arm2=version2,
        model_version_arm3=version3,
        columns_arm2=columns2,
        columns_arm3=columns3,
        window=window3,
    )


def load_fingerprint_doc(arm_dir: Path, fold: str) -> dict[str, Any] | None:
    """读取 ``folds/<fold>/fingerprint.json``（缺文件返回 None，由调用方记拒绝项）。"""
    return load_json_object(Path(arm_dir) / "folds" / fold / "fingerprint.json")


#: 预测行持久化跨路契约：fingerprint.json 必需增键（缺失 ⇒ 该折产物不完整 ⇒ FAIL）
PREDICTIONS_FINGERPRINT_KEYS = ("predictions_path", "predictions_sha256", "predictions_rows")

#: 预测行产物必需列（跨路契约冻结：trade_date str / ts_code str / score float64）
PREDICTIONS_REQUIRED_COLUMNS = ("trade_date", "ts_code", "score")


def missing_predictions_keys(doc: Mapping[str, Any]) -> list[str]:
    """fingerprint 文档缺失的 predictions_* 必需键清单（空列表 = 齐全）。"""
    return [key for key in PREDICTIONS_FINGERPRINT_KEYS if doc.get(key) is None]


def load_fold_predictions(
    arm_dir: Path, fold: str, doc: Mapping[str, Any], *, boundary: FoldWindow | None = None
) -> pd.DataFrame:
    """读取 ``folds/<fold>/predictions.parquet`` 并做证据自洽核验（R2-T5A-06 + R2-T5A-R1-05）。

    校验链（任一不符 ⇒ 抛错 FAIL，不得 PASS）：
    ① 回读行经 ``t05b_fingerprint.serialize_prediction_rows`` 重算文本指纹 == 同折
      ``fingerprint.json["fingerprint"]``（证明 parquet 内容 == 被指纹化的行集）；
    ② ``predictions_rows`` == fingerprint ``n_rows``；
    ③ parquet 日集 == fingerprint ``score_days`` 集合，且 ⊆ 折 boundary test 窗
      （``boundary`` 传入时校验）；
    ④ 非空（rows > 0 且 days > 0）；
    ⑤ 既有 sha256 / 列集 / 键唯一校验保留。
    路径取 fingerprint 的 ``predictions_path``（相对臂目录；拒绝绝对路径/越界 ``..``）。
    """
    missing = missing_predictions_keys(doc)
    if missing:
        raise ValueError(
            f"折 {fold} fingerprint.json 缺 predictions_* 必需键 {missing}（产物不完整）"
        )
    path = _resolve_predictions_path(arm_dir, fold, str(doc["predictions_path"]))
    actual_sha = file_sha256(path)
    if actual_sha != str(doc["predictions_sha256"]):
        raise ValueError(
            f"折 {fold} 预测行产物与指纹登记 sha256 不一致（证据漂移）: {path} "
            f"实得 {actual_sha} vs 登记 {doc['predictions_sha256']}"
        )
    frame = pd.read_parquet(path)
    if len(frame) != int(doc["predictions_rows"]):
        raise ValueError(
            f"折 {fold} 预测行行数与指纹登记不符: {len(frame)} vs {doc['predictions_rows']}"
        )
    missing_cols = [c for c in PREDICTIONS_REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"折 {fold} 预测行缺必需列 {missing_cols}: {path}")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError(f"折 {fold} 预测行 (trade_date, ts_code) 键不唯一: {path}")
    _assert_predictions_self_consistent(frame, fold, doc, boundary)
    return frame


def _resolve_predictions_path(arm_dir: Path, fold: str, rel: str) -> Path:
    """解析 predictions_path（臂目录相对；拒绝绝对路径/越界）并断言存在。"""
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"折 {fold} predictions_path 须为臂目录相对路径: {rel!r}")
    path = Path(arm_dir) / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"折 {fold} 预测行产物缺失: {path}")
    return path


def _assert_predictions_self_consistent(
    frame: pd.DataFrame, fold: str, doc: Mapping[str, Any], boundary: FoldWindow | None
) -> None:
    """证据自洽核验（R2-T5A-R1-05 ①~④）：行集文本指纹/行数/日集/非空。"""
    if len(frame) == 0:
        raise ValueError(f"折 {fold} 预测行为空产物（rows=0，R2-T5A-R1-05④ 拒绝）")
    n_rows = doc.get("n_rows")
    if n_rows is None or int(doc["predictions_rows"]) != int(n_rows):
        raise ValueError(
            f"折 {fold} predictions_rows({doc['predictions_rows']}) != fingerprint "
            f"n_rows({n_rows})（② 行数自洽失败）"
        )
    days = sorted({str(d) for d in frame["trade_date"]})
    score_days = sorted(str(d) for d in (doc.get("score_days") or []))
    if days != score_days:
        raise ValueError(
            f"折 {fold} 预测行日集与 fingerprint score_days 不一致（③ 日集自洽失败）: "
            f"产物 {days[:3]}…({len(days)} 日) vs 登记 {score_days[:3]}…({len(score_days)} 日)"
        )
    if boundary is not None:
        outside = [d for d in days if not (boundary.test_start <= d <= boundary.test_end)]
        if outside:
            raise ValueError(
                f"折 {fold} 预测行日集越出 test 窗 [{boundary.test_start}, "
                f"{boundary.test_end}]（③ 越窗拒绝）: {outside[:5]}"
            )
    _assert_predictions_fingerprint(frame, fold, doc)


def _assert_predictions_fingerprint(frame: pd.DataFrame, fold: str, doc: Mapping[str, Any]) -> None:
    """① 行集文本指纹自洽：回读行重算 == fingerprint.json["fingerprint"]。"""
    from scripts.v2_p3.t05b_fingerprint import serialize_prediction_rows, sha256_text

    rows = [
        (str(trade_date), str(ts_code), float(score))
        for trade_date, ts_code, score in frame[["trade_date", "ts_code", "score"]].itertuples(
            index=False
        )
    ]
    recomputed = sha256_text(serialize_prediction_rows(rows))
    registered = doc.get("fingerprint")
    if recomputed != str(registered):
        raise ValueError(
            f"折 {fold} 预测行回读重算指纹与 fingerprint.json 不一致（① 行集自洽失败，"
            f"parquet 内容 != 被指纹化的行集）: 重算 {recomputed[:16]}… vs 登记 {registered}"
        )


#: 信号 topk 产物指纹键（跨路契约，t05b_fingerprint docstring 冻结；可选产物）
SIGNAL_TOPK_FINGERPRINT_KEYS = (
    "signal_topk_path",
    "signal_topk_sha256",
    "signal_topk_rows",
    "signal_topk_days",
    "signal_topk_empty_days",
)


def load_fold_signal_topk(arm_dir: Path, fold: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    """可选读取 ``signal_topk.parquet`` 并登记展示字段（R2-T5A-R1-05；判定归 gates）。

    fingerprint.json 不含 ``signal_topk_path`` ⇒ 未产出（available=False，不 FAIL）；
    含键但文件缺失/读取失败 ⇒ available=False + note 登记（展示语义，不阻断）；
    正常 ⇒ rows/days（产物实读）+ empty_days（指纹键透出，合法零选股日口径）+
    sha256_match（登记一致性展示）。
    """
    rel = doc.get("signal_topk_path")
    if rel is None:
        return {"available": False, "note": "fingerprint 未登记 signal_topk_path（未产出）"}
    path = Path(arm_dir) / str(rel)
    if not path.is_file():
        return {"available": False, "note": f"已登记但文件缺失: {path}"}
    frame = pd.read_parquet(path)
    registered_sha = doc.get("signal_topk_sha256")
    return {
        "available": True,
        "rows": int(len(frame)),
        "days": (
            int(frame["trade_date"].astype(str).nunique()) if "trade_date" in frame.columns else 0
        ),
        "empty_days": doc.get("signal_topk_empty_days"),
        "sha256_match": (
            (file_sha256(path) == str(registered_sha)) if registered_sha is not None else None
        ),
    }


def resolve_runs_batch_dir(arm_dir: Path) -> Path:
    """解析臂内 runs 批次目录（复用 t5_compare_gates 同一解析语义，禁止另写）。"""
    from scripts.v2_p3.t5_compare_gates import _resolve_runs_batch_dir

    return _resolve_runs_batch_dir(Path(arm_dir))


def load_fold_artifact(batch_dir: Path, fold: str, name: str) -> pd.DataFrame | None:
    """读取折级产物 parquet（trades/daily/holdings_snapshot）；缺文件返回 None。"""
    path = Path(batch_dir) / "folds" / fold / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_chain_nav(batch_dir: Path) -> pd.DataFrame:
    """读取批次 chain_nav.parquet（契约必需文件，缺失即报错）。"""
    path = Path(batch_dir) / "chain_nav.parquet"
    if not path.exists():
        raise FileNotFoundError(f"runs 批次缺 chain_nav.parquet: {path}")
    return pd.read_parquet(path)


def load_batch_meta(batch_dir: Path) -> dict[str, Any]:
    """读取批次 batch_meta.json（契约必需文件，缺失即报错）。"""
    doc = load_json_object(Path(batch_dir) / "batch_meta.json")
    if doc is None:
        raise FileNotFoundError(f"runs 批次缺 batch_meta.json: {batch_dir}")
    return doc


def list_cs_train_days(data_root: Path, window: FoldWindow) -> list[str]:
    """窗口内 cs_train 实际分区日（glob 日分区名，YYYYMMDD 升序）。"""
    cs_train_dir = Path(data_root) / "features" / "cs_train"
    if not cs_train_dir.is_dir():
        return []
    days = []
    for path in cs_train_dir.glob("*.parquet"):
        day = path.stem
        if _DATE8_RE.fullmatch(day) and window.contains(day):
            days.append(day)
    return sorted(days)


def load_trade_calendar(data_root: Path, *, ledger: ReadLedger | None = None) -> pd.DataFrame:
    """交易日历帧（cal_date str YYYYMMDD + is_open；clean 优先、raw 兜底）。

    语义与 ``t05b_spike_runner.list_expected_oos_days`` 的日历来源一致，差别仅在
    本函数把整本日历返回给调用方（多折复用一次读取，并提供非开市日剔除计数输入）。
    每个实际读取的候选文件均做读前/读后身份核验并登记台账；clean 空表回退 raw
    时保留两次读取证据，禁止用未绑定的另一份日历缩小覆盖域（R2-T5A-R2-02）。
    """
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    loader = DataLoader(Storage(root_path=str(data_root)))
    read_ledger = ledger if ledger is not None else ReadLedger()
    cal = None
    source_path: Path | None = None
    source_digest: str | None = None
    for layer, load in (
        ("clean", loader.load_clean_trade_cal),
        ("raw", loader.load_trade_cal),
    ):
        path = Path(data_root) / layer / "trade_cal.parquet"
        if not path.is_file():
            continue
        digest = read_ledger.record(path)
        candidate = load()
        read_ledger.assert_unchanged(path, digest)
        if candidate is not None and len(candidate):
            cal, source_path, source_digest = candidate, path.resolve(), digest
            break
    if cal is None or len(cal) == 0:
        raise ValueError(f"无法加载交易日历: {data_root}")
    missing = [c for c in ("cal_date", "is_open") if c not in cal.columns]
    if missing:
        raise ValueError(f"交易日历缺必需列 {missing}: {data_root}")
    result = cal[["cal_date", "is_open"]].copy()
    result["cal_date"] = result["cal_date"].astype(str)
    result["_is_open"] = pd.to_numeric(result["is_open"], errors="coerce") == 1
    result.attrs["source_path"] = str(source_path)
    result.attrs["source_sha256"] = source_digest
    return result


def _window_mask(dates: pd.Series, window: FoldWindow) -> pd.Series:
    in_outer = (dates >= window.outer_train_start) & (dates <= window.outer_train_end)
    in_test = (dates >= window.test_start) & (dates <= window.test_end)
    return in_outer | in_test


def expected_open_days(calendar: pd.DataFrame, window: FoldWindow) -> list[str]:
    """独立预期日域 = trade_cal ``is_open=1`` ∩ boundary 窗口（外层训练窗 ∪ test 窗）。

    预期日来源独立于已存在分区清单（R2-T5A-04 完整覆盖前置）；窗口内无开市日
    即抛错（预期域不可得 ⇒ 覆盖不可核验）。
    """
    dates = calendar["cal_date"]
    days = sorted(str(d) for d in dates[calendar["_is_open"] & _window_mask(dates, window)])
    if not days:
        raise ValueError(f"交易日历在归因窗口 {window.span()} 内无开市日")
    return days


def count_non_open_days(calendar: pd.DataFrame, window: FoldWindow) -> int:
    """窗口内非开市日计数（合法剔除登记：双源同缺的非交易日不进入预期日域）。"""
    dates = calendar["cal_date"]
    return int((~calendar["_is_open"] & _window_mask(dates, window)).sum())


def window_months(window: FoldWindow) -> list[str]:
    """覆盖窗口跨度的自然月清单（YYYY-MM 升序；月迭代粒度，沿 panel_reconcile 口径）。"""
    start, end = window.span()
    year, month = int(start[:4]), int(start[4:6])
    end_month = f"{end[:4]}-{end[4:6]}"
    months = []
    while f"{year:04d}-{month:02d}" <= end_month:
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def load_panel_month(store: Any, columns: Sequence[str], month: str) -> pd.DataFrame:
    """按月读 panel（**只读消费列**，``FeatureQuery`` 列投影；core 左表锚定由 store 实现）。

    返回 (trade_date, ts_code) 索引帧；core 分区整体缺失时为空帧（store 契约语义）。
    """
    start = f"{month}01".replace("-", "")
    last_day = calendar_module.monthrange(int(month[:4]), int(month[5:7]))[1]
    end = f"{month}{last_day:02d}".replace("-", "")
    frame = store.load_features(
        FeatureQuery(
            columns=list(columns),
            start_date=TradeDate.from_str(start),
            end_date=TradeDate.from_str(end),
        )
    )
    return frame.df


def load_cs_train_day(storage: Any, day: str) -> pd.DataFrame | None:
    """逐日读 cs_train 分区（``Storage.load_cs_train_day``；缺失/空帧返回 None）。"""
    frame = storage.load_cs_train_day(day)
    if frame is None or len(frame) == 0:
        return None
    return frame


def cs_train_day_path(data_root: Path, day: str) -> Path:
    """cs_train 日分区路径推导（``<root>/features/cs_train/<day>.parquet`` 固定口径）。"""
    return Path(data_root) / "features" / "cs_train" / f"{day}.parquet"


class ReadLedger:
    """a 节实际读取文件台账（R2-T5A-R1-03）：逐文件绝对路径 + 读取时点内容 sha256。

    判定前由 runner 核验：实际读取集合 ⊆ 臂 ``consumption_snapshot.json`` 登记集合
    （双方 ``os.path.normcase(resolve())`` 同基规范化后按绝对路径比对）且逐文件
    当前内容哈希 == 快照登记哈希。panel 月分区侧身份来自 ``PanelDataStore`` 只读
    receipt（``consumed_files``，§13.3.5 P2 机制）；cs_train 与日历先登记读前身份，
    读后即时核验，报告全部分节结束后再按唯一实际读取集合复核，缓存命中亦适用。
    """

    def __init__(self) -> None:
        self._records: dict[str, str] = {}  # 规范化绝对路径 -> sha256

    @staticmethod
    def normalize(path: Path) -> str:
        """路径同基规范化（resolve + normcase；跨根/跨形态可比）。"""
        return os.path.normcase(str(Path(path).resolve()))

    def record(self, path: Path, *, digest: str | None = None) -> str:
        """登记一个实读文件并返回当前内容 sha256（``digest`` 缺省时读取时点现算）。"""
        if digest is None:
            digest = file_sha256(path)
            if digest is None:
                raise FileNotFoundError(f"实读登记失败（文件不存在）: {path}")
        normalized = self.normalize(path)
        previous = self._records.get(normalized)
        if previous is not None and previous != digest:
            raise ValueError(f"实读内容漂移（同一会话重复读取身份改变）: {path}")
        self._records[normalized] = digest
        return digest

    def assert_unchanged(self, path: Path, digest: str) -> None:
        """读后（含比较缓存命中后）即时核验同一文件身份，末次读取也不得漏检。"""
        current = file_sha256(path)
        if current != digest:
            raise ValueError(f"实读期间文件漂移（读后核验失败）: {path}")
        self.record(path, digest=digest)

    def verify_current(self) -> list[str]:
        """全部分节结束后按唯一实读集合重算，检测末次读后及缓存复用期间的漂移。"""
        return [
            f"运行后实读内容漂移（文件缺失或当前哈希 != 读取身份）: {path}"
            for path, digest in sorted(self._records.items())
            if file_sha256(Path(path)) != digest
        ]

    @property
    def records(self) -> dict[str, str]:
        """实读清单（规范化绝对路径 → sha256）浅拷贝。"""
        return dict(self._records)


def extract_input_snapshot_digest(arm_meta: Mapping[str, Any]) -> str | None:
    """臂 arm_meta 的消费快照摘要 = ``producer_identity.input_snapshot_digest``
    （t5_arm_runner 落盘口径；兜底顶层同名键）。"""
    producer = arm_meta.get("producer_identity")
    if isinstance(producer, Mapping) and producer.get("input_snapshot_digest"):
        return str(producer["input_snapshot_digest"])
    digest = arm_meta.get("input_snapshot_digest")
    return str(digest) if digest else None


def load_arm_meta(arm_dir: Path) -> dict[str, Any] | None:
    """读取臂 ``arm_meta.json``（缺失返回 None；§13.3.8 身份前置校验的输入）。"""
    return load_json_object(Path(arm_dir) / "arm_meta.json")


def load_freeze_manifest(b0p_dir: Path) -> dict[str, Any] | None:
    """读取 B0' ``freeze_manifest.json``（缺失返回 None；g 节 B0' 身份唯一来源，P3-2）。"""
    return load_json_object(Path(b0p_dir) / "freeze_manifest.json")


def extract_source_digest(arm_meta: Mapping[str, Any]) -> str | None:
    """从 arm_meta 提取代码态内容摘要。

    键位优先级 = ``producer_identity.source_digest``（t5_arm_runner 落盘口径）→
    ``workspace_identity.source_digest`` → 顶层 ``source_digest``。
    """
    for key in ("producer_identity", "workspace_identity"):
        nested = arm_meta.get(key)
        if isinstance(nested, Mapping) and nested.get("source_digest"):
            return str(nested["source_digest"])
    digest = arm_meta.get("source_digest")
    return str(digest) if digest else None


def file_sha256(path: Path) -> str | None:
    """文件内容 sha256（缺失返回 None；消费快照指针的证据摘要）。"""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
