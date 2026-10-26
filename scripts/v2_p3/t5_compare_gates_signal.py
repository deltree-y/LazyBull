# -*- coding: utf-8 -*-
"""P3 T5 比较边·topk 辅证子模块（自 ``t5_compare_gates`` 拆出，R2-T5A-R1 拆分消化）。

承载 G-code ③ 辅证两路（判据冻结，本模块只实现判定）：

- **评估路径**：两臂逐折 topk 明细 CSV 逐行对比（旧 walk_forward 出口
  ``write_walk_forward_topk_details`` 冻结形态）。**任一折缺失 ⇒ FAIL**
  （R2-T5A-R1-01 收紧，取代 R1 前的 skip 登记语义——「辅证缺失 = FAIL」）。
- **信号路径（R2-T5A-R1-01）**：读两臂 ``signal_topk.parquet``（真实信号过滤通路
  产物，跨路冻结契约：列 trade_date/rank/ts_code/score，按 (trade_date,rank) 升序），
  经 fingerprint.json 增键（``signal_topk_path``/``_sha256``/``_rows``/``_days``/
  ``_empty_days``）登记核验——缺文件/缺增键/哈希漂移/rows 交叉核对不符 ⇒ FAIL；
  **完整性门**：逐折 ``signal_topk_days`` ∪ ``empty_days`` 必须 == 该折主证据
  （predictions.parquet）日集且 == fingerprint.json ``score_days``（缺日 FAIL）；
  全折 ``signal_topk_rows`` 合计 == 0 ⇒ FAIL（零辅证不得过总门）。
- **身份配对互换计数（R2-T5A-09 + R2-T5A-R1-06）**：按 ``group_cols``（评估路径默认
  ``(trade_date, topk)``——正式出口 ``build_daily_topk_detail_df`` 同帧输出
  Top20/Top30，同股跨 K 合法重复）分组、组内按 ``key_col``（ts_code）配对；
  **组内 rank 升序 score 序列单调不增核验**——异分反转（排名与分数矛盾）⇒
  selection_mismatch；逐键分数跨臂相等而 rank 不同（同分并列段内位次差异）⇒
  互换计数，>入选数 5% 登记（信息性，不作硬失败，§7.1-③）。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p2a.compare_replay_vs_b0 import _frame_equal_exact  # noqa: E402

#: 并列互换登记阈值（>入选数 5% 时登记，信息性，§7.1-③；单一来源）
TIE_SWAP_REGISTER_RATIO = 0.05
#: topk 明细 CSV 命名（旧 walk_forward 出口 ``write_walk_forward_topk_details`` 冻结形态）
TOPK_CSV_GLOB = "walk_forward_topk_details_*_{fold}.csv"
#: 信号 topk 产物 fingerprint.json 增键五件套（跨路冻结契约）
SIGNAL_TOPK_KEYS: Tuple[str, ...] = (
    "signal_topk_path",
    "signal_topk_sha256",
    "signal_topk_rows",
    "signal_topk_days",
    "signal_topk_empty_days",
)
#: 信号 topk 产物契约列（跨路冻结契约）
SIGNAL_TOPK_COLUMNS: Tuple[str, ...] = ("trade_date", "rank", "ts_code", "score")


def _fold_name(split: int) -> str:
    """折名（函数内延迟 import 主文件唯一实现，避免模块循环依赖）。"""
    from scripts.v2_p3.t5_compare_gates import _fold_name as fold_name

    return fold_name(split)


def _expected_splits() -> Tuple[int, ...]:
    """预期折集（函数内延迟 import 主文件唯一来源，避免模块循环依赖）。"""
    from scripts.v2_p3.t5_compare_gates import EXPECTED_SPLITS

    return EXPECTED_SPLITS


# ---------------------------------------------------------------------------
# 身份配对互换计数核心（R2-T5A-09 + R2-T5A-R1-06）
# ---------------------------------------------------------------------------


def _values_equal(left: object, right: object) -> bool:
    """逐值相等口径（None/NaN 互等视等，与 ``_frame_equal_exact`` 同口径）。"""
    left_none = left is None or (not isinstance(left, str) and pd.isna(left))
    right_none = right is None or (not isinstance(right, str) and pd.isna(right))
    if left_none or right_none:
        return left_none and right_none
    return bool(left == right)


def _key_topk_rows(
    frame: pd.DataFrame, key_cols: Sequence[str], label: str
) -> Dict[Tuple[object, ...], pd.Series]:
    """按配对键（组列 + ts_code）建配对映射；键重复 ⇒ 配对前提违反抛错。

    正式多 K 出口下同股跨 K 合法重复——键含 ``topk`` 组列故不冲突；同股同组
    重复才是真正的契约违反。
    """
    keyed: Dict[Tuple[object, ...], pd.Series] = {}
    for _, row in frame.iterrows():
        key = tuple(row[column] for column in key_cols)
        if key in keyed:
            raise ValueError(f"{label} topk 明细配对键 {key} 重复（身份配对前提违反）")
        keyed[key] = row
    return keyed


def _monotonic_rank_score_errors(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    rank_col: str,
    score_col: str,
    label: str,
) -> List[str]:
    """组内单调性核验（R2-T5A-R1-06）：按 rank 升序的 score 序列必须单调不增。

    异分反转（rank 与 score 降序矛盾）⇒ selection_mismatch 级错误；同分并列段内
    任意位次均满足单调 ⇒ 合法互换前提。score 缺失/非有限 ⇒ 同组即违反。
    """
    errors: List[str] = []
    for group_key, group in frame.groupby(list(group_cols), sort=True):
        scores = pd.to_numeric(group.sort_values(rank_col)[score_col], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(scores).all():
            errors.append(f"{label} 组 {group_key} 分数含缺失/非有限值")
            continue
        if bool((np.diff(scores) > 0).any()):
            errors.append(f"{label} 组 {group_key} 排名与分数矛盾（异分反转 ⇒ selection_mismatch）")
    return errors


def _compare_topk_fold(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    register_ratio: float = TIE_SWAP_REGISTER_RATIO,
    group_cols: Tuple[str, ...] = ("trade_date", "topk"),
    key_col: str = "ts_code",
    rank_col: str = "rank",
    score_col: str = "pred_score",
) -> Dict[str, object]:
    """单折 topk 明细对比：逐行精确 ⇒ 分组身份配对互换计数 ⇒ 选股差异。

    互换计数口径（R2-T5A-09 + R2-T5A-R1-06）：按 ``group_cols``（默认
    ``(trade_date, topk)``，多 K 出口同股跨 K 合法重复）分组、组内按 ``key_col``
    配对——配对行**非分组非 rank 列全等（含分数跨臂相等）而 rank 不同**，且两臂
    组内 rank-score 单调不增（同分并列段内位次差异）⇒ 计一次位次互换；
    配对键集不等（跨 K 边界/跨日选股差异）、配对行分数等列不等、任一组排名与
    分数矛盾（异分反转）⇒ selection_mismatch（真实差异，硬失败）。
    互换计数 > 入选数 × register_ratio 时登记（信息性，不作硬失败，§7.1-③）。
    """
    result: Dict[str, object] = {
        "status": "match",
        "swap_count": 0,
        "n_selected": int(len(frame_a)),
        "registered": False,
        "errors": [],
    }
    if set(frame_a.columns) != set(frame_b.columns):
        only_a = sorted(set(frame_a.columns) - set(frame_b.columns))
        only_b = sorted(set(frame_b.columns) - set(frame_a.columns))
        result["status"] = "column_set_mismatch"
        message = f"topk 明细列集不一致: 臂①多 {only_a} / 臂②多 {only_b}"
        result["errors"] = [message]  # type: ignore[assignment]
        return result
    cols = sorted(frame_a.columns)
    left = frame_a[cols].reset_index(drop=True)
    right = frame_b[cols].reset_index(drop=True)
    if _frame_equal_exact(left, right):
        return result
    required_cols = [c for c in (*group_cols, key_col, rank_col, score_col)]
    missing_cols = [c for c in required_cols if c not in frame_a.columns]
    if missing_cols:
        raise ValueError(f"topk 明细缺分组/身份/排名/分数列 {missing_cols}（身份配对前提）")
    key_cols = tuple(group_cols) + (key_col,)
    keyed_a = _key_topk_rows(left, key_cols, "臂①")
    keyed_b = _key_topk_rows(right, key_cols, "臂②")
    if set(keyed_a) != set(keyed_b):
        only_a = sorted(set(keyed_a) - set(keyed_b))[:5]
        only_b = sorted(set(keyed_b) - set(keyed_a))[:5]
        result["status"] = "selection_mismatch"
        result["errors"] = [  # type: ignore[assignment]
            f"topk 明细多重集不一致（选股集合差异：臂①独有 {only_a} / 臂②独有 {only_b}）"
        ]
        return result
    monotone_errors = _monotonic_rank_score_errors(
        left, group_cols, rank_col, score_col, "臂①"
    ) + _monotonic_rank_score_errors(right, group_cols, rank_col, score_col, "臂②")
    if monotone_errors:
        result["status"] = "selection_mismatch"
        result["errors"] = monotone_errors  # type: ignore[assignment]
        return result
    compare_cols = [c for c in cols if c != rank_col and c not in group_cols]
    swaps = 0
    value_diffs: List[str] = []
    for key in sorted(keyed_a):
        row_a, row_b = keyed_a[key], keyed_b[key]
        unequal = [c for c in compare_cols if not _values_equal(row_a[c], row_b[c])]
        if unequal:
            value_diffs.append(f"{key} 列 {unequal[:3]}")
        elif not _values_equal(row_a[rank_col], row_b[rank_col]):
            swaps += 1
    if value_diffs:
        suffix = f" 等 {len(value_diffs)} 键" if len(value_diffs) > 5 else ""
        result["status"] = "selection_mismatch"
        errors_msg = f"topk 明细多重集不一致（分数/内容真实差异：{value_diffs[:5]}{suffix}）"
        result["errors"] = [errors_msg]  # type: ignore[assignment]
        return result
    if swaps == 0:
        return result  # 行序差异但逐键整行全等（含 rank）⇒ 内容一致
    result["status"] = "tie_swaps"
    result["swap_count"] = swaps
    result["registered"] = swaps > register_ratio * len(frame_a)
    return result


def _collect_topk_fold_result(
    name: str,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    registrations: List[str],
    label: str,
    *,
    group_cols: Tuple[str, ...] = ("trade_date", "topk"),
    score_col: str = "pred_score",
) -> Tuple[Dict[str, object], List[str]]:
    """单折 topk 对比 + 结果归集（评估/信号两路共用）：错误带折号前缀，互换 >5% 登记。"""
    fold_result = {
        "fold": name,
        **_compare_topk_fold(frame_a, frame_b, group_cols=group_cols, score_col=score_col),
    }
    if fold_result["registered"]:
        registrations.append(
            f"{name}: {label}并列分数互换 {fold_result['swap_count']} 行 > 入选数 "
            f"{fold_result['n_selected']} 的 5%（信息性登记，不作硬失败）"
        )
    errors = [f"{name}: {message}" for message in fold_result["errors"]]  # type: ignore[union-attr]
    return fold_result, errors


# ---------------------------------------------------------------------------
# 评估路径：topk 明细 CSV 逐行对比（双臂全折同缺 ⇒ FAIL）
# ---------------------------------------------------------------------------


def _find_topk_csv(arm_dir: Path, fold: str) -> Optional[Path]:
    """定位一折 topk 明细 CSV（``root/walk_forward/`` 下递归 glob；多命中拒绝取其一）。"""
    base = Path(arm_dir) / "root" / "walk_forward"
    if not base.is_dir():
        return None
    hits = sorted(base.rglob(TOPK_CSV_GLOB.format(fold=fold)))
    if len(hits) > 1:
        raise ValueError(f"{fold} topk 明细多命中（禁止多命中取其一）: {[str(h) for h in hits]}")
    return hits[0] if hits else None


def compare_topk_details(arm1_dir: Path, arm2_dir: Path) -> Dict[str, object]:
    """G-code ③ 辅证·评估路径：两臂逐折 topk 明细 CSV 逐行对比 + 并列互换计数。

    逐行一致或仅并列互换 ⇒ PASS（互换 >入选数 5% 时登记，信息性）；多重集/列集
    不一致、任一折单侧或双侧缺文件 ⇒ FAIL，完整 14 折辅证不得以少数折替代。
    """
    from scripts.v2_p3.t5_compare_gates import (  # 延迟 import（避免模块循环依赖）
        PAIR_PROJECTION_KEYS,
        _project_frame,
    )

    arm1_dir, arm2_dir = Path(arm1_dir), Path(arm2_dir)
    folds: List[Dict[str, object]] = []
    errors: List[str] = []
    notes: List[str] = []
    registrations: List[str] = []
    compared = 0
    for split in _expected_splits():
        name = _fold_name(split)
        path_a = _find_topk_csv(arm1_dir, name)
        path_b = _find_topk_csv(arm2_dir, name)
        if path_a is None and path_b is None:
            errors.append(f"{name}: 双臂均无 topk 明细 CSV ⇒ FAIL（每折辅证必需）")
            continue
        if path_a is None or path_b is None:
            errors.append(
                f"{name}: topk 明细单侧缺失（臂①={'有' if path_a else '无'} / 臂②={'有' if path_b else '无'}）"
            )
            continue
        compared += 1
        frame_a = _project_frame(
            pd.read_csv(path_a, encoding="utf-8-sig", float_precision="round_trip"),
            PAIR_PROJECTION_KEYS,
        )
        frame_b = _project_frame(
            pd.read_csv(path_b, encoding="utf-8-sig", float_precision="round_trip"),
            PAIR_PROJECTION_KEYS,
        )
        fold_result, fold_errors = _collect_topk_fold_result(
            name, frame_a, frame_b, registrations, ""
        )
        folds.append(fold_result)
        errors += fold_errors
    if compared == 0:
        errors.append("评估路径 topk 明细 CSV 全折双臂同缺 ⇒ FAIL（辅证缺失 = FAIL，R2-T5A-R1-01）")
    return {
        "verdict": "FAIL" if errors else "PASS",
        "folds": folds,
        "tie_swap_registrations": registrations,
        "register_rule": "并列互换计数 > 入选数 5% 时登记（信息性，不作硬失败，§7.1-③）；"
        "任一折辅证缺失 = FAIL（R2-T5A-R2-01）",
        "errors": errors,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 信号路径：signal_topk.parquet 产物核验 + 逐折对比（R2-T5A-R1-01）
# ---------------------------------------------------------------------------


def _load_signal_topk_artifact(
    arm_dir: Path, fold: str
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]], List[str]]:
    """加载一折信号 topk 产物：fingerprint.json 增键 + 登记路径 + sha256 + rows 核验。

    缺 fingerprint.json / 缺增键 / 路径非法（非相对臂目录 POSIX）/ 缺文件 /
    哈希漂移 / rows 交叉核对不符 / score 非有限 ⇒ 登记错误且返回 ``frame=None``
    （产物身份可疑即不进入对比）。返回 (frame, fingerprint_doc, errors)。
    """
    from scripts.v2_p3.t5_compare_gates import _load_json  # 延迟 import（避免循环依赖）

    errors: List[str] = []
    fp_doc = _load_json(Path(arm_dir) / "folds" / fold / "fingerprint.json")
    if fp_doc is None:
        return None, None, [f"{fold}: 缺 fingerprint.json（信号 topk 登记核验不可行）"]
    missing_keys = [key for key in SIGNAL_TOPK_KEYS if key not in fp_doc]
    if missing_keys:
        return None, fp_doc, [f"{fold}: fingerprint.json 缺信号 topk 增键 {missing_keys}"]
    rel_path = str(fp_doc["signal_topk_path"])
    rel = PurePosixPath(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        return None, fp_doc, [f"{fold}: signal_topk_path 非相对臂目录 POSIX 路径: {rel_path!r}"]
    path = Path(arm_dir).joinpath(*rel.parts)
    if not path.exists():
        return None, fp_doc, [f"{fold}: 缺 signal_topk 产物文件 {rel_path}"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != fp_doc["signal_topk_sha256"]:
        errors.append(
            f"{fold}: signal_topk 哈希漂移（登记 {str(fp_doc['signal_topk_sha256'])[:16]}… "
            f"vs 重算 {digest[:16]}…）"
        )
    frame = pd.read_parquet(path)
    missing_cols = sorted(set(SIGNAL_TOPK_COLUMNS) - set(frame.columns))
    if missing_cols:
        raise ValueError(f"{path} 缺契约列 {missing_cols}（signal_topk 跨路契约）")
    if len(frame) != int(fp_doc["signal_topk_rows"]):  # type: ignore[arg-type]
        errors.append(
            f"{fold}: signal_topk_rows 交叉核对不符（登记 {fp_doc['signal_topk_rows']} "
            f"vs 实际 {len(frame)}）"
        )
    scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        errors.append(f"{fold}: signal_topk score 含 NaN/Inf（契约违反）")
    if errors:
        return None, fp_doc, errors
    return frame, fp_doc, errors


def _signal_topk_integrity_errors(
    arm_dir: Path, fold: str, fp_doc: Dict[str, object], frame: pd.DataFrame
) -> List[str]:
    """完整性门（R2-T5A-R1-01）：日集关联 + score_days 一致 + 登记与产物一致。

    逐折 ``signal_topk_days`` ∪ ``signal_topk_empty_days`` 必须 == 该折主证据
    （predictions.parquet）日集（关联性核验）且 == fingerprint.json ``score_days``
    （预期日域兑现点，缺日 FAIL）；登记 days 与产物实际日集须一致；
    days 与 empty_days 交集须为空。
    """
    errors: List[str] = []
    days = fp_doc["signal_topk_days"]
    empty_days = fp_doc["signal_topk_empty_days"]
    if not isinstance(days, list) or not all(isinstance(d, str) for d in days):
        return [f"{fold}: signal_topk_days 非字符串日清单: {days!r}"]
    if not isinstance(empty_days, list) or not all(isinstance(d, str) for d in empty_days):
        return [f"{fold}: signal_topk_empty_days 非字符串日清单: {empty_days!r}"]
    overlap = set(days) & set(empty_days)
    if overlap:
        errors.append(f"{fold}: signal_topk_days 与 empty_days 交集非空 {sorted(overlap)[:5]}")
    covered = set(days) | set(empty_days)
    pred_path = Path(arm_dir) / "folds" / fold / "predictions.parquet"
    if not pred_path.exists():
        errors.append(f"{fold}: 缺 predictions.parquet（主证据日集关联性核验不可行）")
    else:
        pred_frame = pd.read_parquet(pred_path, columns=["trade_date"])
        pred_days = {str(d) for d in pred_frame["trade_date"]}
        if covered != pred_days:
            missing = sorted(pred_days - covered)[:5]
            extra = sorted(covered - pred_days)[:5]
            errors.append(
                f"{fold}: 信号 topk 覆盖日集与主证据日集不符（缺 {missing} / 多 {extra}）"
            )
    score_days = fp_doc.get("score_days")
    if not isinstance(score_days, list) or not all(isinstance(d, str) for d in score_days):
        errors.append(f"{fold}: fingerprint.json 缺合法 score_days 清单（预期日域锚）")
    elif covered != set(score_days):
        missing = sorted(set(score_days) - covered)[:5]
        errors.append(f"{fold}: 信号 topk 覆盖日集缺 score_days 日 {missing}（缺日 FAIL）")
    frame_days = {str(d) for d in frame["trade_date"]}
    if frame_days != set(days):
        errors.append(f"{fold}: signal_topk_days 登记与产物实际日集不符")
    return errors


def compare_signal_topk(arm1_dir: Path, arm2_dir: Path) -> Dict[str, object]:
    """G-code ③ 辅证·信号路径 top-k 对比（§13.3.3 辅证后半，R2-T5A-R1-01 收紧）。

    读两臂 ``signal_topk.parquet``（真实信号过滤通路产物；**不再从主门矩阵
    predictions 派生**）：fingerprint.json 增键 + sha256 + rows 交叉核验，缺一即
    FAIL（不得静默跳过）；逐折完整性门（日集关联 + score_days 一致，缺日 FAIL）；
    两臂逐折身份配对对比（组内同分位次互换计数、>入选数 5% 登记信息性）；
    **全折 signal_topk_rows 合计 == 0 ⇒ FAIL（零辅证不得过总门）**。
    """
    arm1_dir, arm2_dir = Path(arm1_dir), Path(arm2_dir)
    folds: List[Dict[str, object]] = []
    errors: List[str] = []
    registrations: List[str] = []
    rows_total = [0, 0]
    for split in _expected_splits():
        name = _fold_name(split)
        frame_a, doc_a, errors_a = _load_signal_topk_artifact(arm1_dir, name)
        frame_b, doc_b, errors_b = _load_signal_topk_artifact(arm2_dir, name)
        errors += errors_a + errors_b
        for index, (frame, doc, arm) in enumerate(
            ((frame_a, doc_a, arm1_dir), (frame_b, doc_b, arm2_dir))
        ):
            if frame is None or doc is None:
                continue
            errors += _signal_topk_integrity_errors(arm, name, doc, frame)
            rows_total[index] += int(doc["signal_topk_rows"])  # type: ignore[arg-type]
        if frame_a is None or frame_b is None:
            continue  # 产物核验/完整性错误已登记，本折不进入对比
        fold_result, fold_errors = _collect_topk_fold_result(
            name,
            frame_a,
            frame_b,
            registrations,
            "信号路径",
            group_cols=("trade_date",),
            score_col="score",
        )
        folds.append(fold_result)
        errors += fold_errors
    for label, total in zip(("臂①", "臂②"), rows_total):
        if total == 0:
            errors.append(
                f"{label} 全折 signal_topk_rows 合计为 0（零辅证不得过总门，R2-T5A-R1-01）"
            )
    return {
        "verdict": "FAIL" if errors else "PASS",
        "source": "两臂 folds/splitNN/signal_topk.parquet（真实信号过滤通路产物，"
        "fingerprint.json 增键 + sha256 + rows 登记核验；R2-T5A-R1-01）",
        "folds": folds,
        "tie_swap_registrations": registrations,
        "errors": errors,
    }
