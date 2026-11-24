# -*- coding: utf-8 -*-
"""逐折 F / R / B 列路由与逐列校验（协议 §3、§5.2-4、§5.2-6）。

冻结规则
========
对原版第 k 折的**有序**已注册特征列表 ``F_k``：

- ``F = F_k``（逐字继承，不重排）
- ``R = [c for c in F_k if c not in G]``
- ``B = [c for c in F_k if c in G]``

不排序、不做重要性筛选、不做逐列消融。校验项（任一不符即 fail-fast）：

1. ``len(F)`` 等于协议 §3 冻结列数；
2. ``F`` 无重复列；
3. ``B`` 恰为 11 列且集合等于 ``G``；
4. ``R ∩ B = ∅``、``R ∪ B = F``（集合相等）；
5. ``R`` / ``B`` 都是 ``F`` 的**保序子序列**；
6. 标签 / 目标 / 哨兵 / 已删除别名不得进入任何一臂矩阵；
7. 旧 metadata 的现金流 live 列（9 列）⊂ ``G`` 且 ``≠ G``——
   说明注册器必须按**本臂最终列**重算元数据，不能把 9 列当成 B 的业务组。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT, sha256_file


class RouteError(RuntimeError):
    """列集不符冻结口径（fail-fast，禁止"删到刚好相等"）。"""


def load_feature_list(root: Path, version: int) -> List[str]:
    """读取 ``v{version}_features.json``（有序列表）。"""
    path = Path(root) / spec.MODEL_ROOT_RELATIVE / f"v{version}_features.json"
    if not path.exists():
        raise RouteError(f"特征清单缺件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise RouteError(f"特征清单不是字符串列表：{path}")
    return [str(item) for item in payload]


def load_metadata_live_columns(root: Path, version: int) -> List[str]:
    """读取旧 metadata 的现金流 live 列（9 列口径）。"""
    path = Path(root) / spec.MODEL_ROOT_RELATIVE / f"v{version}_metadata.json"
    if not path.exists():
        raise RouteError(f"模型元数据缺件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("train_params") or {}
    return [str(item) for item in (params.get("cashflow_quality_feature_columns") or [])]


def split_route(feature_list: Sequence[str]) -> Tuple[List[str], List[str]]:
    """按冻结规则切分 ``(R, B)``；两段均为输入顺序的保序子序列。"""
    group = set(spec.CASHFLOW_GROUP)
    r = [column for column in feature_list if column not in group]
    b = [column for column in feature_list if column in group]
    return r, b


def _assert_subsequence(short: Sequence[str], long: Sequence[str], label: str) -> None:
    """``short`` 必须是 ``long`` 的保序子序列（逐位扫描，不重排）。"""
    index = 0
    for column in long:
        if index < len(short) and short[index] == column:
            index += 1
    if index != len(short):
        raise RouteError(f"{label} 不是 F 的保序子序列（禁止重排）")


def verify_route(
    version: int,
    feature_list: Sequence[str],
    metadata_live: Sequence[str],
    role: str,
) -> Dict[str, Any]:
    """单版本路由校验（协议 §3 全部硬条件）。"""
    expected = spec.EXPECTED_FEATURE_COUNTS.get(int(version))
    if expected is None:
        raise RouteError(f"未登记的模型版本：{version}")
    if len(feature_list) != expected:
        raise RouteError(f"v{version} 列数不符冻结值：{len(feature_list)} != {expected}（协议 §3）")
    duplicates = sorted({c for c in feature_list if feature_list.count(c) > 1})
    if duplicates:
        raise RouteError(f"v{version} F 存在重复列：{duplicates}")

    r, b = split_route(feature_list)
    if len(b) != len(spec.CASHFLOW_GROUP):
        missing_group = sorted(set(spec.CASHFLOW_GROUP) - set(b))
        raise RouteError(
            f"v{version} 现金流组列数不符：{len(b)} != {len(spec.CASHFLOW_GROUP)}；"
            f"缺 {missing_group}"
        )
    if set(b) != set(spec.CASHFLOW_GROUP):
        raise RouteError(f"v{version} B 集合与 G 不等：{sorted(set(b) ^ set(spec.CASHFLOW_GROUP))}")
    if set(r) & set(b):
        raise RouteError(f"v{version} R 与 B 交集非空：{sorted(set(r) & set(b))}")
    if set(r) | set(b) != set(feature_list):
        raise RouteError(f"v{version} R∪B 不等于 F")
    if len(r) + len(b) != len(feature_list):
        raise RouteError(f"v{version} 列数不守恒：{len(r)}+{len(b)} != {len(feature_list)}")
    _assert_subsequence(r, feature_list, f"v{version} R")
    _assert_subsequence(b, feature_list, f"v{version} B")

    forbidden = sorted(set(feature_list) & set(spec.FORBIDDEN_MATRIX_COLUMNS))
    if forbidden:
        raise RouteError(f"v{version} 矩阵含禁用列（标签/哨兵/别名）：{forbidden}")

    live_check = _check_metadata_live(version, feature_list, metadata_live)
    return {
        "version": int(version),
        "role": role,
        "f_count": len(feature_list),
        "r_count": len(r),
        "b_count": len(b),
        "f_sha256": _list_digest(feature_list),
        "r_sha256": _list_digest(r),
        "b_sha256": _list_digest(b),
        "b_equals_group": sorted(b) == sorted(spec.CASHFLOW_GROUP),
        "b_order_inherits_f": True,
        "checks": {
            "count_matches_protocol": True,
            "no_duplicates": True,
            "intersection_empty": True,
            "union_equals_f": True,
            "order_preserved": True,
            "no_forbidden_columns": True,
        },
        "metadata_live_columns": live_check,
        "f": list(feature_list),
        "r": r,
        "b": b,
    }


def _check_metadata_live(
    version: int, feature_list: Sequence[str], metadata_live: Sequence[str]
) -> Dict[str, Any]:
    """旧 metadata 的 9 列 live 口径校验（不得当成 B 的完整业务组）。"""
    from src.lazybull.v2.common.model_schema import cashflow_quality_live_columns

    recomputed = cashflow_quality_live_columns(list(feature_list))
    declared = list(metadata_live)
    return {
        "metadata_live_columns": declared,
        "metadata_live_count": len(declared),
        "recomputed_from_f": recomputed,
        "recomputed_count": len(recomputed),
        "metadata_matches_recompute": sorted(declared) == sorted(recomputed),
        "equals_business_group": sorted(declared) == sorted(spec.CASHFLOW_GROUP),
        "note": (
            "旧 metadata 只含 9 列（缺 cf_sales 两列）；B 的业务组固定 11 列，"
            "注册器必须按本臂最终列重算 live 元数据（协议 §5.2-6）"
        ),
    }


def _list_digest(columns: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def build_routes(root: Path | None = None) -> Dict[str, Any]:
    """构建 14 折 + 部署的完整路由文档（提交物 ``feature_routes.json``）。"""
    base = Path(root) if root is not None else REPO_ROOT
    routes: List[Dict[str, Any]] = []
    for version in spec.FOLD_VERSIONS:
        features = load_feature_list(base, version)
        live = load_metadata_live_columns(base, version)
        routes.append(verify_route(version, features, live, "fold"))
    deploy_features = load_feature_list(base, spec.DEPLOY_VERSION)
    deploy_live = load_metadata_live_columns(base, spec.DEPLOY_VERSION)
    deploy = verify_route(spec.DEPLOY_VERSION, deploy_features, deploy_live, "deploy")

    union_ok = all(item["b_equals_group"] for item in routes) and deploy["b_equals_group"]
    if not union_ok:
        raise RouteError("存在 B 集合不等于冻结 G 的版本")

    return {
        "schema": "p7_cashflow_split_feature_routes/v1",
        "candidate_id": spec.CANDIDATE_ID,
        "protocol_sha256": spec.protocol_sha256(base),
        "spec_sha256": spec.spec_sha256(),
        "group": spec.feature_group_payload(),
        "fold_versions": list(spec.FOLD_VERSIONS),
        "deploy_version": spec.DEPLOY_VERSION,
        "folds": routes,
        "deploy": deploy,
        "summary": {
            "versions": len(routes) + 1,
            "all_b_equal_group": union_ok,
            "f_counts": {str(item["version"]): item["f_count"] for item in routes + [deploy]},
            "b_count": len(spec.CASHFLOW_GROUP),
            "r_counts": {str(item["version"]): item["r_count"] for item in routes + [deploy]},
            "metadata_live_all_nine": all(
                item["metadata_live_columns"]["metadata_live_count"] == 9
                for item in routes + [deploy]
            ),
            "checklist": [
                "140/154 两种清单逐版本对账（协议 §3 冻结列数）",
                "两对回填/别名：cf_sales 两列整体归 B；cf_nm 别名不得回流",
                "哨兵与目标/标签未进入任何一臂矩阵",
                "R/B 均为 F 的保序子序列，交集空、并集等于 F",
            ],
        },
    }


def route_table(routes: Mapping[str, Any]) -> Dict[int, Dict[str, List[str]]]:
    """``{version: {"F": [...], "R": [...], "B": [...]}}``（训练/评分消费入口）。"""
    table: Dict[int, Dict[str, List[str]]] = {}
    for item in list(routes.get("folds") or []) + [routes.get("deploy")]:
        if not item:
            continue
        table[int(item["version"])] = {
            "F": list(item["f"]),
            "R": list(item["r"]),
            "B": list(item["b"]),
        }
    return table


def feature_file_digest(root: Path, version: int) -> str:
    """特征清单整字节 SHA（登记血缘用）。"""
    return sha256_file(Path(root) / spec.MODEL_ROOT_RELATIVE / f"v{version}_features.json")


__all__ = [
    "RouteError",
    "build_routes",
    "feature_file_digest",
    "load_feature_list",
    "load_metadata_live_columns",
    "route_table",
    "split_route",
    "verify_route",
]
