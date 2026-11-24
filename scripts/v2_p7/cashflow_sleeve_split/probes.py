# -*- coding: utf-8 -*-
"""七日只读资格探针（协议 §5.1、§8；复审 r0_002 整改）。

母表与域
========
母表 = 当日 ``clean/daily`` **全部行**，键 ``(trade_date, ts_code)``。分两层：

1. **合格域 universe**：只用**当时可知**的时点输入（板块、上市天数、当日 ``is_st``、
   当日停牌/可交易、T 日行业区间、金融剔除），**不使用** ``stock_basic.name``
   （那是当前名称，含未来退市/ST 信息）也不使用特征是否可算；
2. **可评分域 scorable**：universe ∩ 当日特征行（历史期为 label_filtered 域），
   再叠成交额与市值门槛。

两层的差集 = ``structural_gap_n``，即旧行域按未来标签删掉的股票数（恢复动因）。

行业
====
金融与中性化必须用**同一份 T 日区间映射**（``sw_industry_pit``），代码系统用旧 6 位
码（银行 ``480000``、非银 ``490000``）。特征里 ``sw_l1_code`` 是**旧截面**（如
``801890.SI``），只作对照，并登记冲突。

来源取证
========
B 必须命中**原始比率**（``ocf``/``ocf_to_revenue``/``ocf_to_profit``/``fcf``/
``fcf_yield``/``capex_to_ocf``）且该列可追溯；仅公告新鲜度 ``cashflow_freshness_days``
**不算**来源（协议 §5.1）。只用标准化 ``zscore``/``_sz`` 记「默认值」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT

#: 申万**旧 6 位码**的银行 / 非银金融（与 sw_industry_pit.l1_code 同一系统）
FINANCIAL_PIT_CODES = tuple(spec.PROBE_SELECTION["financial_pit_codes"])
#: 旧截面（特征 sw_l1_code）里的金融码，只用于冲突对照
LEGACY_FINANCIAL_CODES = tuple(spec.PROBE_SELECTION["financial_l1_codes"])
MOTHER_COLUMNS: Sequence[str] = ("ts_code", "list_days", "is_st", "is_suspended", "tradable")


class ProbeError(RuntimeError):
    """探针输入缺件或口径不符。"""


def feature_partition(root: Path, day: str) -> Path:
    """当日特征分区（尾段 cs_infer，历史 cs_train）。"""
    day = str(day)
    infer = root / "data/features/cs_infer" / f"{day}.parquet"
    if infer.exists():
        return infer
    return root / "data/features/cs_train" / f"{day}.parquet"


def daily_partition(root: Path, day: str) -> Path:
    return root / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"


def read_mother(root: Path, day: str) -> Dict[str, Any]:
    """当日母表（clean/daily 全部行）+ 缺失列登记。"""
    import pandas as pd

    path = daily_partition(root, day)
    if not path.exists():
        return {"status": "source_not_ready", "reason": f"母表缺件：{path}"}
    frame = pd.read_parquet(path)
    missing = [column for column in MOTHER_COLUMNS if column not in frame.columns]
    if missing:
        return {"status": "source_not_ready", "reason": f"母表缺列：{missing}", "missing": missing}
    keep = frame[list(MOTHER_COLUMNS)].copy()
    keep["trade_date"] = str(day)
    return {"status": "ok", "frame": keep, "missing": []}


def read_stock_basic(root: Path) -> Dict[str, Any]:
    """只取 ``market``（板块）；``name`` 已知含未来信息，不参与资格。"""
    import pandas as pd

    path = root / "data/clean/stock_basic.parquet"
    if not path.exists():
        return {"status": "source_not_ready", "reason": f"stock_basic 缺件：{path}"}
    frame = pd.read_parquet(path)
    missing = [column for column in ("ts_code", "market") if column not in frame.columns]
    if missing:
        return {"status": "source_not_ready", "reason": f"stock_basic 缺列：{missing}"}
    return {"status": "ok", "frame": frame[["ts_code", "market"]]}


def read_industry_pit(root: Path) -> Dict[str, Any]:
    import pandas as pd

    path = root / "data/clean/sw_industry_pit.parquet"
    if not path.exists():
        return {"status": "source_not_ready", "reason": f"行业 PIT 缺件：{path}"}
    frame = pd.read_parquet(path)
    missing = sorted({"ts_code", "in_date", "out_date", "l1_code"} - set(frame.columns))
    if missing:
        return {"status": "source_not_ready", "reason": f"行业 PIT 缺列：{missing}"}
    return {"status": "ok", "frame": frame}


def industry_for_day(pit: Any, day: str) -> Any:
    """区间命中（``in_date ≤ T < out_date``；``out_date`` 空 = 仍有效）。"""
    frame = pit.copy()
    day = str(day)
    has_in = frame["in_date"].notna()
    started = frame["in_date"] <= day
    active = frame["out_date"].isna() | (frame["out_date"] > day)
    mask = has_in & started & active
    hit = frame.loc[mask, ["ts_code", "l1_code"]].dropna(subset=["l1_code"])
    return hit.drop_duplicates(subset=["ts_code"], keep="last")


def _step(name: str, before: int, after: int, unknown: int = 0, status: str = "ok", note: str = ""):
    return {
        "id": name,
        "input_n": int(before),
        "excluded_n": int(before - after - unknown),
        "unknown_n": int(unknown),
        "output_n": int(after),
        "status": status,
        "note": note,
    }


def _universe(mother: Any, basic: Any, industry: Any, steps: List[Dict[str, Any]]) -> Any:
    """只用当时可知输入建合格域（不含特征与名称）。"""
    import pandas as pd

    frame = mother.copy()
    steps.append(_step("mother", len(frame), len(frame)))
    merged = frame.merge(basic, on="ts_code", how="left")

    whitelist = list(spec.PROBE_SELECTION["market_whitelist"])
    before = len(merged)
    unknown = int(merged["market"].isna().sum())
    merged = merged[merged["market"].isin(whitelist)]
    steps.append(_step("main_board", before, len(merged), unknown=unknown))

    min_days = int(spec.PROBE_SELECTION["min_list_days"])
    before = len(merged)
    listed = pd.to_numeric(merged["list_days"], errors="coerce")
    unknown = int(listed.isna().sum())
    merged = merged[listed.fillna(-1) >= min_days]
    steps.append(_step("listed_days", before, len(merged), unknown=unknown))

    # ST 只用当日 is_st；当前名称不参与（复审：未来名称会改写历史候选池）
    before = len(merged)
    st_flag = pd.to_numeric(merged["is_st"], errors="coerce")
    unknown = int(st_flag.isna().sum())
    merged = merged[st_flag.fillna(0) <= 0]
    steps.append(
        _step(
            "not_st_pit",
            before,
            len(merged),
            unknown=unknown,
            note="仅 clean/daily.is_st；不使用 stock_basic.name（含未来信息）",
        )
    )

    before = len(merged)
    suspended = pd.to_numeric(merged["is_suspended"], errors="coerce").fillna(1) > 0
    not_tradable = pd.to_numeric(merged["tradable"], errors="coerce").fillna(0) <= 0
    merged = merged[~(suspended | not_tradable)]
    steps.append(_step("not_suspended", before, len(merged)))

    before = len(merged)
    if industry is None:
        steps.append(
            _step("industry_known", before, before, unknown=before, status="source_not_ready")
        )
        return merged.iloc[0:0]
    merged = merged.merge(industry, on="ts_code", how="left")
    unknown = int(merged["l1_code"].isna().sum())
    merged = merged[merged["l1_code"].notna()]
    steps.append(_step("industry_known", before, len(merged), unknown=unknown))

    before = len(merged)
    codes = merged["l1_code"].astype(str).str.replace(".SI", "", regex=False)
    financial = codes.isin(FINANCIAL_PIT_CODES)
    merged = merged[~financial]
    steps.append(
        _step(
            "not_financial_pit",
            before,
            len(merged),
            note=f"T 日 PIT 旧码金融剔除 {int(financial.sum())} 只（480000/490000）",
        )
    )
    return merged


def apply_selection(
    mother: Any,
    features: Any,
    basic: Any,
    industry: Any,
    day: str,
    feature_source: str = "cs_train",
) -> Dict[str, Any]:
    """两层资格：universe（时点可知）→ scorable（有特征行 + 成交额/市值）。"""
    import pandas as pd

    steps: List[Dict[str, Any]] = []
    universe = _universe(mother, basic, industry, steps)
    universe_n = len(universe)
    if not steps or steps[-1]["status"] == "source_not_ready":
        return {"eligible": universe.iloc[0:0], "steps": steps, "status": "source_not_ready"}

    if features is None:
        steps.append(
            _step("has_feature_row", universe_n, 0, unknown=universe_n, status="source_not_ready")
        )
        return {"eligible": universe.iloc[0:0], "steps": steps, "status": "source_not_ready"}

    selection_columns = [
        column
        for column in ("ts_code", "amount_ma20", "total_mv", "sw_l1_code")
        if column in features.columns
    ]
    merged = universe.merge(
        features[selection_columns].drop_duplicates(subset=["ts_code"], keep="last"),
        on="ts_code",
        how="left",
        suffixes=("", "_feature"),
    )
    has_feature = merged["ts_code"].astype(str).isin(set(features["ts_code"].astype(str)))
    structural_gap = int((~has_feature).sum())
    steps.append(
        _step(
            "has_feature_row",
            universe_n,
            int(has_feature.sum()),
            unknown=structural_gap,
            note=(
                f"特征来源 {feature_source}；历史期属 label_filtered 行域，"
                "缺口即按未来标签被删掉的股票（不可在审点一恢复）"
            ),
        )
    )
    merged = merged[has_feature]

    legacy = None
    if "sw_l1_code" in merged.columns:
        legacy = (
            merged["sw_l1_code"]
            .astype(str)
            .str.replace(".SI", "", regex=False)
            .isin(LEGACY_FINANCIAL_CODES)
        )
    bridge = {
        "pit_financial_n": int(
            merged["l1_code"]
            .astype(str)
            .str.replace(".SI", "", regex=False)
            .isin(FINANCIAL_PIT_CODES)
            .sum()
        ),
        "legacy_snapshot_financial_n": int(legacy.sum()) if legacy is not None else None,
        "note": "PIT 旧码 vs 特征旧截面；冲突行随七日逐日登记",
    }

    before = len(merged)
    amount = pd.to_numeric(merged.get("amount_ma20"), errors="coerce")
    floor = float(spec.PROBE_SELECTION["amount_floor_kilo_yuan"])
    unknown = int(amount.isna().sum())
    merged = merged[amount.fillna(-1.0) >= floor]
    steps.append(_step("amount_floor", before, len(merged), unknown=unknown))

    before = len(merged)
    cap = pd.to_numeric(merged.get("total_mv"), errors="coerce")
    low, high = (float(value) for value in spec.PROBE_SELECTION["market_cap_band_wan"])
    unknown = int(cap.isna().sum())
    merged = merged[cap.between(low, high)]
    steps.append(_step("market_cap_band", before, len(merged), unknown=unknown))

    return {
        "eligible": merged,
        "steps": steps,
        "status": "ok",
        "universe_n": int(universe_n),
        "structural_gap_n": structural_gap,
        "feature_source": feature_source,
        "industry_bridge": bridge,
    }


def source_evidence(
    eligible: Any,
    version: int,
    routes: Mapping[int, Mapping[str, Any]],
    columns: Sequence[str],
) -> Dict[str, Any]:
    """按当折白名单分别证明 F / R / B 有**可追溯原始值**。

    r0_003 复核修正：① 掩码用**有限值**（inf/-inf 不算有值）；② B 的来源
    白名单只认经济比率与销售现金代理（``ocf``/``fcf`` 绝对额单独存在不算，
    单独计数 ``B_absolute_amount_only``）。
    """
    route = routes.get(int(version))
    if route is None:
        return {"status": "source_not_ready", "reason": f"缺 v{version} 路由"}
    available = set(columns)
    forbidden = set(spec.FORBIDDEN_MATRIX_COLUMNS)
    f_columns = [column for column in route["F"] if column in available]
    r_columns = [column for column in route["R"] if column in available]
    b_columns = [column for column in route["B"] if column in available]
    leaks = sorted((set(route["F"]) | set(route["R"]) | set(route["B"])) & forbidden)

    raw_columns = [c for c in spec.SOURCE_EVIDENCE["B"]["raw_columns"] if c in available]
    absolute_columns = [
        c for c in spec.SOURCE_EVIDENCE["B"].get("absolute_amount_columns", []) if c in available
    ]
    freshness = spec.SOURCE_EVIDENCE["B"]["announcement_proxy"]
    b_standardized = [column for column in b_columns if column not in set(raw_columns)]

    result = {
        "status": "ok" if not leaks else "leak",
        "version": int(version),
        "forbidden_in_route": leaks,
        "F": _mask_counts(eligible, f_columns),
        "R": _mask_counts(eligible, r_columns),
        "B": _mask_counts(eligible, raw_columns),
        "B_absolute_amount_only": _absolute_only(eligible, raw_columns, absolute_columns),
        "B_freshness_only": _freshness_only(eligible, raw_columns, freshness),
        "B_standardized_only": _standardized_only(eligible, b_standardized, raw_columns),
        "B_raw_columns_present": raw_columns,
        "B_absolute_columns_present": absolute_columns,
        "B_freshness_present": freshness in available,
        "B_policy": (
            "B 只认经济比率与销售现金代理（有限值）；ocf/fcf 绝对额单独存在不算来源；"
            "公告新鲜度单独计数、不构成来源（协议 §5.1 + r0_003 复核）"
        ),
    }
    result["R_minus_F_n"] = int(result["R"]["known_n"] - result["F"]["known_n"])
    return result


def _finite_mask(frame: Any, columns: Sequence[str]) -> Any:
    """数值列**有限值**掩码（NaN / inf / -inf / 非数值均不算有值）。

    r0_003 复核：``.notna()`` 会让单个无穷值计入 F/R，来源判定必须用有限值。
    """
    import numpy as np
    import pandas as pd

    if frame is None or len(frame) == 0 or not columns:
        return None
    present = [column for column in columns if column in frame.columns]
    if not present:
        return None
    numeric = frame[list(present)].apply(lambda series: pd.to_numeric(series, errors="coerce"))
    return np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan)).any(axis=1)


def _mask_counts(frame: Any, columns: Sequence[str]) -> Dict[str, Any]:
    mask = _finite_mask(frame, columns)
    if mask is None:
        return {
            "known_n": 0,
            "columns_n": len(list(columns)),
            "ratio": 0.0,
            "rule": "有限值（np.isfinite）；空列集或空域记 0",
        }
    return {
        "known_n": int(mask.sum()),
        "columns_n": len(list(columns)),
        "ratio": round(float(mask.mean()), 4),
        "rule": "有限值（np.isfinite）；NaN/inf/非数值均不算",
    }


def _absolute_only(
    frame: Any, ratio_columns: Sequence[str], absolute_columns: Sequence[str]
) -> Dict[str, Any]:
    """只有 ocf/fcf 绝对额、没有可用比率的行数（负例，不算 B 来源）。"""
    if frame is None or len(frame) == 0 or not absolute_columns:
        return {"n": 0, "absolute_present": bool(absolute_columns)}
    has_absolute = _finite_mask(frame, absolute_columns)
    has_ratio = _finite_mask(frame, ratio_columns)
    if has_absolute is None:
        return {"n": 0, "absolute_present": False}
    if has_ratio is None:
        return {
            "n": int(has_absolute.sum()),
            "absolute_present": True,
            "note": "特征帧无比率列（比率全部缺失），有绝对额的行全部记为仅绝对额",
        }
    return {
        "n": int((has_absolute & ~has_ratio).sum()),
        "absolute_present": True,
        "note": "有 ocf/fcf 绝对额（有限值）但无任何可用比率 ⇒ 不算 B 来源",
    }


def _freshness_only(frame: Any, raw_columns: Sequence[str], freshness: str) -> Dict[str, Any]:
    """只有新鲜度、没有原始比率的行数（负例，不算来源）。"""
    if frame is None or len(frame) == 0 or freshness not in getattr(frame, "columns", []):
        return {"n": 0, "freshness_present": False}
    has_fresh = frame[freshness].notna()
    has_raw = _finite_mask(frame, raw_columns)
    if has_raw is None:
        return {"n": int(has_fresh.sum()), "freshness_present": True, "note": "无比率列"}
    return {"n": int((has_fresh & ~has_raw).sum()), "freshness_present": True}


def _standardized_only(
    frame: Any, standardized: Sequence[str], evidence: Sequence[str]
) -> Dict[str, Any]:
    """只有标准化列、没有原始值的行数（默认填值）。"""
    if frame is None or len(frame) == 0 or not standardized:
        return {"n": 0, "note": "无标准化列或空域"}
    has_z = frame[list(standardized)].notna().any(axis=1)
    has_raw = _finite_mask(frame, evidence)
    if has_raw is None:
        return {"n": int(has_z.sum()), "note": "只有 zscore/副本（无比率对照列）⇒ 默认填值"}
    return {"n": int((has_z & ~has_raw).sum()), "note": "只有 zscore/副本 ⇒ 默认填值"}


def probe_day(
    root: Path | None = None,
    day: str = "",
    extra_roots: Sequence[str] = (),
    routes: Mapping[int, Mapping[str, Any]] | None = None,
    version: int | None = None,
    features_frame: Any = None,
    features_label: str = "structural_injected",
) -> Dict[str, Any]:
    """单日探针：母表 → 两层资格 → 三臂来源（只读）。

    ``features_frame`` 非 None 时消费**结构域真实输出**（r0_003 复核第 1 项：
    探针必须接到与资格过滤同一的注入输入），并保留旧训练域行数对照；
    三层宽度分开登记：旧域可观察数 / 结构域可观察数 / 实际评分数
    （评分数须待模型训练，本阶段恒 not_estimable）。
    """
    import time

    import pandas as pd

    base = Path(root) if root is not None else REPO_ROOT
    started = time.perf_counter()
    record: Dict[str, Any] = {"date": str(day), "status": "ok"}

    if features_frame is not None:
        features = features_frame.copy().drop_duplicates(subset=["ts_code"], keep="last")
        feature_source = features_label
        record["feature_source"] = feature_source
        legacy_path = feature_partition(base, day)
        if legacy_path.exists():
            record["old_domain_observed_n"] = int(
                len(pd.read_parquet(legacy_path, columns=["ts_code"]))
            )
        else:
            record["old_domain_observed_n"] = None
        record["structural_domain_observed_n"] = int(len(features))
    else:
        feature_path = feature_partition(base, day)
        feature_source = "cs_infer" if "cs_infer" in feature_path.as_posix() else "cs_train"
        record["feature_source"] = feature_source
        if not feature_path.exists():
            record.update({"status": "source_not_ready", "reason": f"特征分区缺件：{feature_path}"})
            return record
        features = pd.read_parquet(feature_path).drop_duplicates(subset=["ts_code"], keep="last")

    record["scored_n"] = None
    record["scored_n_status"] = "not_estimable_at_checkpoint_1"
    record["width_layers"] = {
        "old_domain_observed_n": record.get("old_domain_observed_n"),
        "structural_domain_observed_n": record.get("structural_domain_observed_n"),
        "scored_n": None,
        "note": (
            "旧域可观察数 = cs_train 当日行数；结构域可观察数 = 注入 PIT 行业的"
            "结构构建当日行数；实际评分数须待模型训练后逐日统计（协议 §5.1）"
        ),
    }

    mother = read_mother(base, day)
    basic = read_stock_basic(base)
    for name, payload in (("母表", mother), ("stock_basic", basic)):
        if payload["status"] != "ok":
            record.update({"status": payload["status"], "reason": payload["reason"]})
            return record
    pit = read_industry_pit(base)
    industry = industry_for_day(pit["frame"], day) if pit["status"] == "ok" else None
    record["industry_status"] = pit["status"]

    record["feature_rows_n"] = int(len(features))
    record["feature_columns_n"] = int(len(features.columns))
    record["group_columns_missing"] = [c for c in spec.CASHFLOW_GROUP if c not in features.columns]
    record["mother_n"] = int(len(mother["frame"]))
    record["mother_tradable_n"] = int(
        (pd.to_numeric(mother["frame"]["tradable"], errors="coerce").fillna(0) > 0).sum()
    )

    if features_frame is not None:
        record["industry_pit_alignment"] = _industry_alignment(features, pit, day, base)

    selection = apply_selection(
        mother["frame"], features, basic["frame"], industry, day, feature_source
    )
    record["selection_steps"] = selection["steps"]
    record["selection_status"] = selection["status"]
    record["universe_n"] = selection.get("universe_n")
    record["structural_gap_n"] = selection.get("structural_gap_n")
    record["industry_bridge"] = selection.get("industry_bridge")
    if selection["status"] != "ok":
        record["status"] = "source_not_ready"

    keys = set(selection["eligible"]["ts_code"].astype(str))
    eligible = features[features["ts_code"].astype(str).isin(keys)].copy()
    record["eligible_n"] = int(len(eligible))
    record["width_floor_500"] = bool(record["eligible_n"] >= 500)

    evidence = source_evidence(eligible, int(version or 0), routes or {}, list(features.columns))
    record["source_evidence"] = evidence
    record["source_known_n"] = int(evidence.get("B", {}).get("known_n", 0))
    record["source_known_f_r_n"] = int(evidence.get("R", {}).get("known_n", 0))
    record["source_known_f_n"] = int(evidence.get("F", {}).get("known_n", 0))
    record["b_freshness_only_n"] = int(evidence.get("B_freshness_only", {}).get("n", 0))
    record["b_source_floor_500"] = bool(record["source_known_n"] >= 500)

    record["cashflow_column_coverage"] = _coverage(eligible, spec.CASHFLOW_GROUP)
    record["sentinel"] = _sentinel(eligible)
    record["row_domain"] = _row_domain(features, day)
    record["feature_to_mother_ratio"] = round(len(features) / max(len(mother["frame"]), 1), 4)
    record["kernel_scored_n"] = None
    record["effective_scored_n"] = None
    record["kernel_status"] = {
        "status": "not_estimable_at_checkpoint_1",
        "reason": "需日期绑定模型与全窗结构缓存（run-a）",
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def _coverage(frame: Any, columns: Sequence[str]) -> Dict[str, Any]:
    present = [column for column in columns if column in frame.columns]
    if not present or len(frame) == 0:
        return {"columns": {}, "missing": list(columns)}
    return {
        "columns": {column: round(float(frame[column].notna().mean()), 4) for column in present},
        "missing": [column for column in columns if column not in present],
    }


def _sentinel(frame: Any) -> Dict[str, Any]:
    column = spec.CASHFLOW_SENTINEL_COLUMN
    if column not in frame.columns or len(frame) == 0:
        return {"present": False, "matches_expected": None}
    values = frame[column].dropna().unique().tolist()
    return {
        "present": True,
        "values": [float(values[0])] if len(values) == 1 else values,
        "matches_expected": bool(
            len(values) == 1 and float(values[0]) == spec.CASHFLOW_SENTINEL_VALUE
        ),
    }


def _row_domain(frame: Any, day: str) -> Dict[str, Any]:
    """行域**声明值**取单一来源模块；标签非空率只作事实。"""
    from src.lazybull.v2.store.row_domain import row_domain_for_date

    declared = row_domain_for_date(str(day))
    base = {
        "declared": declared,
        "source": "v2/store/row_domain.py::row_domain_for_date",
        "note": "标签非空率不是行域判别依据",
    }
    if "y_ret_20" not in frame.columns:
        return {**base, "label_column_present": False}
    return {
        **base,
        "label_column_present": True,
        "label_non_null_ratio": round(float(frame["y_ret_20"].notna().mean()), 4),
    }


def _industry_alignment(
    features: Any, pit: Any, day: str, root: Path | None = None
) -> Dict[str, Any]:
    """结构域特征帧 vs T 日 PIT 行业的对齐核对（同一 ``sw_industry_pit`` 数据源）。

    用与构建注入相同的映射（PIT 旧 6 位码 → code_table 指数码）逐只比对特征
    ``sw_l1_code``；注册身份股票 600783.SH 单独断言。任何不一致都会使后续
    「资格 / 中性化 / 标签共用同一输入」失效。
    """
    import pandas as pd

    from scripts.v2_p7.cashflow_sleeve_split.structbuild import industry_pit_snapshot

    snapshot = industry_pit_snapshot(day, root)
    expected = snapshot["frame"].set_index("ts_code")["sw_l1_code"]
    feature_codes = features["ts_code"].astype(str)
    feature_l1 = features.get("sw_l1_code")
    if feature_l1 is None:
        return {"status": "not_checked", "reason": "特征帧无 sw_l1_code 列"}
    compared = 0
    aligned = 0
    mismatch_examples: List[Dict[str, str]] = []
    for ts_code, value in zip(feature_codes, feature_l1):
        want = expected.get(str(ts_code))
        if want is None or pd.isna(want):
            continue  # PIT 无当日命中：资格过滤会剔除，不参与对齐
        compared += 1
        got = "" if pd.isna(value) else str(value)
        if got == str(want):
            aligned += 1
        elif len(mismatch_examples) < 5:
            mismatch_examples.append(
                {"ts_code": str(ts_code), "expected": str(want), "actual": got}
            )
    identity = snapshot["stats"]["identity_probe"]
    feature_identity = ""
    if identity.get("checked"):
        row = features[feature_codes == identity["ts_code"]]
        if len(row) > 0 and "sw_l1_code" in row.columns:
            value = row.iloc[0]["sw_l1_code"]
            feature_identity = "" if pd.isna(value) else str(value)
    return {
        "status": "ok" if compared and aligned == compared else "mismatch",
        "compared_n": compared,
        "aligned_n": aligned,
        "aligned_ratio": round(aligned / compared, 4) if compared else 0.0,
        "mismatch_examples": mismatch_examples,
        "identity_probe": {
            **identity,
            "feature_sw_l1_code": feature_identity,
            "feature_match": bool(
                feature_identity == identity.get("actual_l1_code") and bool(identity.get("match"))
            ),
        },
        "snapshot_stats": snapshot["stats"],
    }


def run_probes(
    root: Path | None = None,
    extra_roots: Sequence[str] = (),
    routes: Mapping[int, Mapping[str, Any]] | None = None,
    version_for_date: Mapping[str, int] | None = None,
    structural_frames: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """跑预登记的七个探针日（只读、无收益）。

    ``structural_frames``：{day: DataFrame}——该日探针消费**注入 PIT 行业的
    结构构建真实输出**（r0_003 复核第 1 项），其余日仍走旧域分区对照。
    """
    base = Path(root) if root is not None else REPO_ROOT
    records: List[Dict[str, Any]] = []
    for item in spec.PROBE_DATES:
        day = str(item["date"])
        frame = (structural_frames or {}).get(day)
        record = probe_day(
            base,
            day,
            extra_roots,
            routes,
            (version_for_date or {}).get(day),
            features_frame=frame,
        )
        record["segment"] = item["segment"]
        record["note"] = item.get("note")
        records.append(record)

    ok = [record for record in records if record.get("status") == "ok"]
    return {
        "records": records,
        "summary": {
            "probe_days": len(records),
            "ok_days": len(ok),
            "not_ready_days": [r["date"] for r in records if r.get("status") != "ok"],
            "width_failures": [r["date"] for r in ok if not r["width_floor_500"]],
            "source_failures": [r["date"] for r in ok if not r["b_source_floor_500"]],
            "total_elapsed_seconds": round(
                sum(float(r.get("elapsed_seconds") or 0.0) for r in records), 3
            ),
            "kernel_scored_status": "not_estimable_at_checkpoint_1",
            "scope": "只读七日；不产生收益、不训练、不写生产",
            "notes": [
                "universe 只用时点可知输入（板块/上市/当日 is_st/停牌/T 日行业 PIT）",
                "scorable = universe ∩ 当日特征行；差集为旧行域按未来标签删掉的股票",
                "B 来源只认经济比率与销售现金代理（有限值）；ocf/fcf 绝对额、"
                "只有 freshness 或 zscore 的记为负例（r0_003 复核）",
                "宽度三层分开：旧域可观察数 / 结构域可观察数 / 实际评分数（须待模型训练）",
                "kernel/effective 宽度需 run-a 解封；1,326 类读数只是当日结构文件验证读数",
            ],
        },
    }


def probe_costs(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(result.get("summary") or {})
    return {
        "probe_days": summary.get("probe_days"),
        "total_seconds": summary.get("total_elapsed_seconds"),
        "per_day_elapsed_seconds": {
            record["date"]: record.get("elapsed_seconds") for record in result.get("records") or []
        },
        "scope": summary.get("scope"),
    }


__all__ = [
    "FINANCIAL_PIT_CODES",
    "LEGACY_FINANCIAL_CODES",
    "ProbeError",
    "apply_selection",
    "daily_partition",
    "feature_partition",
    "industry_for_day",
    "probe_costs",
    "probe_day",
    "read_industry_pit",
    "read_mother",
    "read_stock_basic",
    "run_probes",
    "source_evidence",
]
