# -*- coding: utf-8 -*-
"""v2 P4 T5 门 5 完整决策输入指纹、逐笔比较与因果归因（结果前冻结）。

验收边仅为 ``v2_shadow vs legacy_neutral``。本模块不根据“预期差异”放行：
每个差异键都必须有独立因果证据，未覆盖差异一律计入 ``unexplained``。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATTRIBUTION_CATEGORIES = frozenset(
    {
        "data_source",
        "normalize_semantics",
        "model_resolution",
        "readiness_timing",
        "holding_path",
        "execution_backfill",
        "policy_append",
        "unexplained",
    }
)
ACCEPTANCE_CATEGORIES = ATTRIBUTION_CATEGORIES - {"policy_append", "unexplained"}
DECISION_GROUPS = (
    "date_phase",
    "merged_target",
    "account",
    "pending",
    "schedule_anchor",
    "ranked_candidates",
    "stop_loss_state",
    "market_data",
    "effective_identity",
)
ORDER_FIELDS = (
    "date",
    "phase",
    "action",
    "ts_code",
    "shares",
    "price_basis",
    "reason",
    "execution_date",
)


class ShadowCompareError(RuntimeError):
    """门 5 输入或证据形态非法。"""


def _normalize(value: Any, *, sort_sets: bool = False) -> Any:
    """冻结规范化：浮点 repr、映射键排序、集合排序、DataFrame 稳定行列序。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y%m%d")
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = [str(column) for column in frame.columns]
        frame = frame[sorted(frame.columns)]
        sort_columns = [column for column in ("date", "trade_date", "ts_code") if column in frame]
        if sort_columns:
            frame = frame.sort_values(sort_columns, kind="mergesort")
        return {
            "columns": list(frame.columns),
            "rows": [_normalize(row, sort_sets=sort_sets) for row in frame.to_dict("records")],
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, sort_sets=sort_sets)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)) or sort_sets:
        return sorted((_normalize(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, (list, tuple)):
        return [_normalize(item, sort_sets=sort_sets) for item in value]
    if hasattr(value, "item"):
        try:
            return _normalize(value.item(), sort_sets=sort_sets)
        except (TypeError, ValueError):
            pass
    raise ShadowCompareError(f"决策输入含不可规范化类型: {type(value).__name__}")


def _digest(payload: Any) -> str:
    normalized = _normalize(payload)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def decision_fingerprints(record: Mapping[str, Any]) -> Dict[str, str]:
    """完整指纹与模型输入指纹分离；缺任一九组即 fail-fast。"""
    missing = [group for group in DECISION_GROUPS if group not in record]
    if missing:
        raise ShadowCompareError(f"完整决策输入缺字段组: {missing}")
    complete = {group: record[group] for group in DECISION_GROUPS}
    model_input = {group: record[group] for group in ("market_data", "effective_identity")}
    return {
        "full_decision_input": _digest(complete),
        "model_input": _digest(model_input),
    }


def canonical_orders(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """指令按 (date, action, ts_code) 稳定排序，字段口径与门 5 冻结清单一致。"""
    output: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        missing = [field for field in ORDER_FIELDS if field not in record]
        if missing:
            raise ShadowCompareError(f"orders[{index}] 缺字段: {missing}")
        row = {field: _normalize(record[field]) for field in ORDER_FIELDS}
        row["reason"] = _normalize_order_reason(row.get("reason"))
        row["shares"] = int(record["shares"])
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row["date"], row["action"], row["ts_code"], row["phase"], row["execution_date"]
        ),
    )


def _normalize_order_reason(value: Any) -> Any:
    """去除仅展示用的“(权重=...)”后缀；不动信号生成/调仓/持有期等语义主体。"""
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"\s*[（(]权重=[^)）]*[)）]", "", text)
    # legacy 补位失败会每轮重复追加“（买入后市值过小）”；重复次数是展示噪声，
    # 真实股数/执行日/失败语义仍由订单字段和 causal evidence 校验。
    text = re.sub(r"(?:（买入后市值过小）)+", "", text)
    return text.strip()


def _order_key(row: Mapping[str, Any]) -> str:
    return f"{row['date']}|{row['action']}|{row['ts_code']}"


def diff_orders(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """逐键比较；同键字段差与左右独有指令均显式列出。"""
    left_map = {_order_key(row): row for row in canonical_orders(left)}
    right_map = {_order_key(row): row for row in canonical_orders(right)}
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise ShadowCompareError("同日同向同股指令未唯一合并，拒绝门 5 比较")
    differences: List[Dict[str, Any]] = []
    for key in sorted(set(left_map) | set(right_map)):
        lrow, rrow = left_map.get(key), right_map.get(key)
        if lrow != rrow:
            differences.append({"difference_key": key, "left": lrow, "right": rrow})
    return differences


def _validate_evidence(
    evidence: Mapping[str, Mapping[str, Any]], *, acceptance_edge: bool
) -> Tuple[List[str], Dict[str, str]]:
    errors: List[str] = []
    categories: Dict[str, str] = {}
    required = ("category", "input_difference", "affected_instruction", "validation")
    for key, raw in evidence.items():
        if not isinstance(raw, Mapping):
            errors.append(f"{key}: 证据不是对象")
            continue
        missing = [field for field in required if field not in raw]
        if missing:
            errors.append(f"{key}: 因果证据缺字段 {missing}")
            continue
        category = str(raw["category"])
        categories[key] = category
        if category not in ATTRIBUTION_CATEGORIES:
            errors.append(f"{key}: 未知归因类别 {category}")
        if acceptance_edge and category not in ACCEPTANCE_CATEGORIES:
            errors.append(f"{key}: {category} 不得用于验收边放行")
        if not raw["input_difference"] or not raw["affected_instruction"]:
            errors.append(f"{key}: 输入差异→受影响指令链为空")
        validation = raw["validation"]
        if not isinstance(validation, Mapping) or validation.get("passed") is not True:
            errors.append(f"{key}: 因果验证未通过或缺 passed=true")
        if category == "data_source":
            if not raw.get("feature_difference") or not raw.get("score_difference"):
                errors.append(f"{key}: data_source 缺特征/分数差异输入级证据")
    return errors, categories


def robust_z_anomalies(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """门 5 冻结异常检出：最少 5 日，稳健 z 绝对值 >3.5。"""
    valid = [(index, float(value)) for index, value in enumerate(values) if value is not None]
    result: Dict[str, Any] = {"minimum_samples": 5, "threshold": 3.5, "anomalies": []}
    if len(valid) < 5:
        result["status"] = "insufficient_samples"
        return result
    series = pd.Series([value for _, value in valid], dtype=float)
    median = float(series.median())
    mad = float((series - median).abs().median())
    anomalies: List[Dict[str, Any]] = []
    for (index, _), value in zip(valid, series):
        z_score = 0.0 if mad == 0 and value == median else (
            math.copysign(math.inf, value - median) if mad == 0 else (value - median) / (1.4826 * mad)
        )
        if abs(z_score) > 3.5:
            anomalies.append({"index": index, "value": value, "robust_z": z_score})
    result.update(
        {
            "status": "evaluated",
            "median": median,
            "mad": mad,
            "anomalies": anomalies,
            "special_attribution_required": bool(anomalies),
        }
    )
    return result


def compare_shadow_day(
    *,
    left_input: Mapping[str, Any],
    right_input: Mapping[str, Any],
    left_orders: Sequence[Mapping[str, Any]],
    right_orders: Sequence[Mapping[str, Any]],
    evidence: Optional[Mapping[str, Mapping[str, Any]]] = None,
    acceptance_edge: bool = True,
) -> Dict[str, Any]:
    """门 5 三项硬判定；证据只覆盖其声明差异键，禁止桶级兜底。"""
    left_fp = decision_fingerprints(left_input)
    right_fp = decision_fingerprints(right_input)
    differences = diff_orders(left_orders, right_orders)
    difference_keys = {row["difference_key"] for row in differences}
    supplied = dict(evidence or {})
    unknown_evidence = sorted(set(supplied) - difference_keys)
    errors, categories = _validate_evidence(supplied, acceptance_edge=acceptance_edge)
    if unknown_evidence:
        errors.append(f"证据指向不存在的差异键: {unknown_evidence}")
    uncovered = sorted(key for key in difference_keys if key not in supplied)
    unexplained = list(uncovered)
    if left_fp["full_decision_input"] == right_fp["full_decision_input"] and differences:
        # 完整输入相同但输出不同永远是实现错误，不能用归因桶豁免。
        unexplained = sorted(difference_keys)
        errors.append("完整决策输入相同但输出逐笔不一致（硬判定 3）")
    evidence_rate = (
        1.0
        if not differences
        else len(difference_keys & set(supplied)) / len(difference_keys)
    )
    bucket_counts: Dict[str, int] = {category: 0 for category in sorted(ATTRIBUTION_CATEGORIES)}
    for key in uncovered:
        bucket_counts["unexplained"] += 1
    for key, category in categories.items():
        if key in difference_keys and key not in uncovered:
            bucket_counts[category] += 1
    hard1 = not unexplained
    hard2 = not errors and evidence_rate == 1.0
    hard3 = not (
        left_fp["full_decision_input"] == right_fp["full_decision_input"] and differences
    )
    return {
        "verdict": "PASS" if hard1 and hard2 and hard3 else "FAIL",
        "hard_gates": {
            "unexplained_zero": hard1,
            "causal_evidence_complete": hard2,
            "same_full_input_same_output": hard3,
        },
        "left_fingerprints": left_fp,
        "right_fingerprints": right_fp,
        "full_input_equal": left_fp["full_decision_input"] == right_fp["full_decision_input"],
        "model_input_equal": left_fp["model_input"] == right_fp["model_input"],
        "difference_count": len(differences),
        "differences": differences,
        "causal_evidence_rate": evidence_rate,
        "bucket_counts": bucket_counts,
        "unexplained": unexplained,
        "errors": errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-input", required=True)
    parser.add_argument("--right-input", required=True)
    parser.add_argument("--left-orders", required=True)
    parser.add_argument("--right-orders", required=True)
    parser.add_argument("--evidence", help="按 difference_key 索引的因果证据 JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    def load(path: str) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    result = compare_shadow_day(
        left_input=load(args.left_input),
        right_input=load(args.right_input),
        left_orders=load(args.left_orders),
        right_orders=load(args.right_orders),
        evidence=load(args.evidence) if args.evidence else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "errors": result["errors"]}, ensure_ascii=False))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
