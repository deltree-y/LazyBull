# -*- coding: utf-8 -*-
"""prepare：v2/train/core 训练模块（P3-T2 搬运改写，§16.1-B / §16.2 / §16.2-bis）。

与旧模块 ``ml/train_core/prepare.py`` 双源并存至 v2 切换。登记改写点（其余逐字保持）：

1. ``prepare_training_data``（旧 976 行硬超限）按 P3 规划 §16.2 / §16.2-bis 边界做
   **最小提取式拆分**——语句顺序 / 数值操作顺序 / 日志调用顺序逐段保持，不做泛化重构；
   特征族扩展助手群按复杂度契约 §4.4 文件二分出移至 ``prepare_families.py``；
2. import 改指：``.constants`` / ``.features`` / ``.split`` / ``.runtime_derive`` +
   ``v2.common.freshness``（事件衰减唯一实现）+ ``v2.common.model_schema``
   （哨兵版本/列清单唯一来源，R2-02）；
3. 显式日期切分缝：新增可选参数 ``explicit_train_end`` / ``explicit_val_start``
   （默认 None = 旧比例路径；F10「显式四日期由编排器换算、train 侧不再二次比例
   切分」，§13.3.2 D1）；
4. 运行时派生五族（holdertrade / repurchase / top10fh / top_inst / availability）
   改 DI 注入（``runtime_derivers``，§16.4）——v2/train 不 import factors
   （包级依赖契约），派生计算本体（factors 侧）由调用方组装注入；族启用而未注入
   ⇒ 报错（与「启用而未传派生表必须报错」契约同级）。验收臂五族全关（生产口径），
   DI 路径不触发；
5. 复杂度再拆（§16.2-bis）：``_FamilyOptions`` 打包族/策略参数；列编排收敛为
   ``_resolve_feature_columns``，质量门禁流水线收敛为 ``_run_quality_pipeline``，
   ``_build_data_stats`` 改命名空间入参并更名 ``_assemble_data_stats``；
   ``_apply_feature_quality_gates`` 明细日志块提取为 ``_log_gate_removal_details``。

参数详表（主函数 docstring 五族 lookup 长文档移存于此，文字不丢）：
- holdertrade_lookup: 股东增减持因子运行时查询表（`{trade_date: DataFrame}`，
  由 `factors.holdertrade.load_holdertrade_lookup` 构建）。启用
  `enable_holdertrade_features` 时**必须提供**：本族列一律**运行时派生**，
  不写入 cs_train（与可用性标记同一先例）。
- holdertrade_feature_set: 股东增减持列集取值（`full` / `core`）。**超参签名维度**，
  禁止跨取值并组比较；`core` = Phase 3 证据支持的 4 列（见
  `factors.holdertrade.HOLDERTRADE_CORE_COLS`）加哨兵列。
- repurchase_lookup: 股票回购因子运行时查询表（`{trade_date: DataFrame}`，
  由 `factors.repurchase.load_repurchase_lookup` 构建）。启用
  `enable_repurchase_features` 时**必须提供**：本族列一律**运行时派生**，
  不写入 cs_train（与股东增减持同一先例）。
- repurchase_feature_set: 股票回购列集取值（`full` / `headroom`）。**超参签名维度**，
  禁止跨取值并组比较；`headroom` = **单列** `rp_price_headroom` 加哨兵列
  （Phase 3 体检显示家族内仅该列有跨期稳定信息）。
- top10fh_panel: 十大流通股东**报告期面板**（由 `factors.top10_floatholders.build_top10fh_panel`
  构建；每股每报告期一行）。启用 `enable_top10fh_features` 时**必须提供**：
  本族列一律**运行时派生**，不写入 cs_train（与股东增减持/股票回购同一先例）。
  本族逐日全市场稠密 ⇒ 用面板容器而非逐日字典（内存原因，见审计 §6.5）。
- top10fh_feature_set: 十大流通股东列集取值（`full` / `concentration`）。**超参签名维度**，
  禁止跨取值并组比较；`concentration` = **单列** `tfh_concentration_chg` + 哨兵
  （Phase 3 体检：家族内仅该列有强信息且正交；不含 freshness——与
  `fundamental_freshness_days` ρ=0.999 重复）。
- top_inst_lookup: 龙虎榜机构席位因子运行时查询表（`{trade_date: DataFrame}`，
  由 `factors.top_inst.build_top_inst_runtime_lookup` 构建）。启用
  `enable_top_inst_features` 时**必须提供**：本族列一律**运行时派生**，
  不写入 cs_train（与股东增减持/股票回购同一先例）。
"""

import gc
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.v2.common.freshness import apply_event_freshness_decay
from src.lazybull.v2.common.model_schema import (
    CASHFLOW_QUALITY_SCHEMA_VERSION,
    cashflow_quality_live_columns,
)

from .constants import (
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
)
from .features import filter_stable_features
from .prepare_families import (
    _append_sz_columns,
    _apply_factor_prune,
    _base_feature_columns,
    _dedup_cashflow_alias,
    _extend_availability_markers,
    _extend_cashflow_quality_columns,
    _extend_consensus_columns,
    _extend_consensus_revision_columns,
    _extend_declared_family_columns_a,
    _extend_declared_family_columns_b,
    _extend_dividend_policy_columns,
    _extend_holdertrade_columns,
    _extend_missing_markers,
    _extend_repurchase_columns,
    _extend_top10fh_columns,
    _extend_top_inst_columns,
    _resolve_freshness_columns,
)
from .runtime_derive import RuntimeDerivers
from .split import (
    split_train_val_by_date,
    split_train_val_by_explicit_dates,
    split_val_for_selection_protocol_by_date,
)

#: 过滤标记列（使用统一的列名，与 clean 层一致）：
#: 收盘买入策略：涨停无法买入需过滤；跌停可以买入（有人卖出），保留参与训练
_FILTER_COLUMNS = ["is_st", "is_suspended", "is_limit_up"]


def _legacy_exclude_columns(df: pd.DataFrame) -> list:
    """旧排除列定义（旧 :163-209 原样保留；计算结果在当前实现中未被消费）。"""
    # 定义需要排除的列（非特征列）
    # 标识列
    id_columns = ["ts_code", "trade_date", "name"]
    # 标签列
    label_columns = [col for col in df.columns if col.startswith("y_")]
    # 过滤标记列（使用统一的列名，与clean层一致）
    # 收盘买入策略：涨停无法买入需过滤；跌停可以买入（有人卖出），保留参与训练
    filter_columns = ["is_st", "is_suspended", "is_limit_up"]
    # 其他非特征列
    other_exclude_columns = [
        "tradable",
        "list_date",
        "list_days",
        "is_limit_up",
        "is_limit_down",
        "industry",
    ]
    # 临时过滤掉的列
    temp_test_exclude_columns = (
        ["total_mv", "circ_mv", "log_circ_mv"]
        + ["kdj_k", "kdj_d"]
        + ["bb_upper", "bb_lower"]
        + ["macd_dif", "macd_dea"]
        + ["ps_ttm", "ep_ttm"]
        + [
            "amount_ma10",
            "amount_ma20",
            "volume_ratio",
            "log_circ_mv",
            "net_mf_amount_mean_5",
            "net_mf_amount_mean_20",
            "vol_burst_10",
            "vol_burst_20",
            "kdj_d",
            "macd_dea",
            "bb_upper",
            "bb_lower",
        ]
    )

    exclude_columns = (
        id_columns
        + label_columns
        + filter_columns
        + other_exclude_columns
        + temp_test_exclude_columns
    )
    return exclude_columns


def _log_gate_removal_details(
    max_feature_missing_ratio: float,
    removed_high_missing: List[str],
    removed_all_nan: List[str],
    removed_constant: List[str],
    linked_removed,
    base_removed,
) -> None:
    """训练入口特征清洗明细日志（旧 :888-907 原样提取）。"""
    # 打印各类移除列的详细名称，便于定位数据链路问题（全空是高缺失的子集，计数独立）
    detail_lines: List[str] = []
    if removed_high_missing:
        detail_lines.append(
            f"高缺失>{max_feature_missing_ratio}（{len(removed_high_missing)}列）: "
            f"{', '.join(removed_high_missing)}"
        )
    if removed_all_nan:
        detail_lines.append(f"全空（{len(removed_all_nan)}列）: {', '.join(removed_all_nan)}")
    if removed_constant:
        detail_lines.append(f"常数（{len(removed_constant)}列）: {', '.join(removed_constant)}")
    linked_extra = sorted(linked_removed - base_removed)
    if linked_extra:
        detail_lines.append(f"联动移除（{len(linked_extra)}列）: {', '.join(linked_extra)}")
    if detail_lines:
        logger.info("训练入口特征清洗明细:\n" + "\n".join(detail_lines))


def _apply_feature_quality_gates(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    max_feature_missing_ratio: float,
) -> tuple:
    """训练入口特征质量门禁（旧 :810-866 原样）：高缺失/全空/常数 + 联动移除。"""
    # 训练入口特征质量门禁：删除高缺失、全空、常数列
    available_feature_cols = [c for c in feature_columns if c in df_train.columns]
    removed_high_missing: List[str] = []
    removed_all_nan: List[str] = []
    removed_constant: List[str] = []
    if available_feature_cols:
        missing_ratio = df_train[available_feature_cols].isna().mean()
        removed_high_missing = sorted(
            missing_ratio[missing_ratio > float(max_feature_missing_ratio)].index.tolist()
        )
        removed_all_nan = sorted(missing_ratio[missing_ratio >= 1.0].index.tolist())

        candidate_cols = [c for c in available_feature_cols if c not in removed_high_missing]
        if candidate_cols:
            nunique = df_train[candidate_cols].nunique(dropna=True)
            removed_constant = sorted(nunique[nunique <= 1].index.tolist())

        base_removed = set(removed_high_missing) | set(removed_all_nan) | set(removed_constant)
        linked_removed = set(base_removed)
        for col in list(base_removed):
            if col.startswith("zscore_") and col.endswith("_sz"):
                linked_removed.add(col[:-3])
            if col.startswith("zscore_") and not col.endswith("_sz"):
                linked_removed.add(f"{col}_sz")

        if linked_removed:
            before = len(feature_columns)
            feature_columns = [c for c in feature_columns if c not in linked_removed]
            removed_count = before - len(feature_columns)
            logger.info(
                "训练入口特征清洗: "
                f"移除 {removed_count} 列（高缺失>{max_feature_missing_ratio}: {len(removed_high_missing)}，"
                f"全空: {len(removed_all_nan)}，常数: {len(removed_constant)}）"
            )
            _log_gate_removal_details(
                max_feature_missing_ratio,
                removed_high_missing,
                removed_all_nan,
                removed_constant,
                linked_removed,
                base_removed,
            )

    if not feature_columns:
        raise ValueError("特征列为空（在缺失率/常数列过滤后）")

    return feature_columns, removed_high_missing, removed_all_nan, removed_constant


def _trim_to_needed_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    decay_helper_cols: List[str],
) -> pd.DataFrame:
    """内存优化列裁剪（旧 :761-772 原样）。"""
    # ── 内存优化：只保留训练必需的列，避免 copy 时 OOM ──
    needed_cols = set(
        feature_columns
        + [label_column, "trade_date", "ts_code"]
        + _FILTER_COLUMNS
        + decay_helper_cols
    )
    needed_cols &= set(df.columns)  # 仅保留实际存在的列
    kept = list(needed_cols)
    n_before = len(df.columns)
    df = df[kept]
    logger.debug(f"内存优化: {n_before} → {len(df.columns)} 列（仅保留训练必需列）")
    return df


def _filter_and_decay(
    df: pd.DataFrame,
    label_column: str,
    freshness_strategy: str,
    event_freshness_cols_used: List[str],
    event_freshness_half_life_days: float,
) -> tuple:
    """可训练样本过滤 + 标签 dropna + 事件衰减（旧 :774-808 原样）。"""
    # 过滤可训练样本（移除含有过滤标记的样本）
    mask = pd.Series([True] * len(df), index=df.index)
    for col in _FILTER_COLUMNS:
        if col in df.columns:
            mask = mask & (~df[col].astype(bool))

    # df[mask]（布尔索引）已复制独立数据，无需再 deep copy；后续
    # df_train[col] = np.nan 等写操作作用于独立副本，不会写穿 df。
    # 大训练集上 .copy() 会触发 BlockManager 合并导致额外全量内存分配（OOM）。
    df_train = df[mask]
    logger.info(f"过滤后样本数: {len(df_train)} / {len(df)}")
    samples_after_filter = len(df_train)

    # 移除标签为 NaN 的样本
    df_train = df_train.dropna(subset=[label_column])
    logger.info(f"移除标签 NaN 后样本数: {len(df_train)}")

    if len(df_train) == 0:
        raise ValueError("没有可用的训练样本")

    decay_applied_stats: Dict[str, int] = {}
    if (
        freshness_strategy == FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY
        and event_freshness_cols_used
    ):
        df_train, decay_applied_stats = apply_event_freshness_decay(
            df_train,
            event_freshness_cols=event_freshness_cols_used,
            half_life_days=float(event_freshness_half_life_days),
        )
        if decay_applied_stats:
            logger.info(
                "事件型 freshness 衰减已应用: "
                + ", ".join([f"{k}->{v}" for k, v in sorted(decay_applied_stats.items())])
            )

    return df_train, samples_after_filter, decay_applied_stats


def _apply_feature_columns_override(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    feature_columns_override: Optional[List[str]],
) -> List[str]:
    """多窗口集成特征列统一（旧 :868-890 原样）：门禁后强制覆盖。"""
    # 多窗口集成特征列统一：以基础窗口特征列为准，保证集成子模型特征 schema 一致。
    # 不同窗口稀疏因子缺失率不同，高缺失门禁可能产生不同特征列（如 express_revenue_yoy），
    # 若不统一会导致集成预测时 XGBoost feature_names mismatch。
    if feature_columns_override is not None:
        override_missing_in_data = [
            c for c in feature_columns_override if c not in df_train.columns
        ]
        for col in override_missing_in_data:
            df_train[col] = np.nan
        if len(feature_columns) != len(feature_columns_override):
            logger.info(
                "训练入口特征列统一(override): "
                f"{len(feature_columns)} -> {len(feature_columns_override)} 列（以基础窗口为准）"
            )
        if override_missing_in_data:
            logger.warning(
                "训练入口特征列统一(override): "
                f"当前窗口数据缺失列补 NaN {len(override_missing_in_data)} 个: "
                f"{override_missing_in_data}"
            )
        feature_columns = list(feature_columns_override)
        if not feature_columns:
            raise ValueError("特征列为空（在 override 统一后）")
    return feature_columns


def _split_train_val(
    df_train: pd.DataFrame, label_column: str, opts: "_FamilyOptions"
) -> SimpleNamespace:
    """label_delta 推导 + 比例/显式双路径切分 + 协议三段拆分（旧 :892-916 原样 + 显式缝）。

    比例/显式日期取自 ``opts``（val_ratio / explicit_train_end / explicit_val_start）；
    返回 SimpleNamespace，字段沿用原 unpack 名（§16.2-bis）。
    """
    # 从标签列名自动推断 delta（例如 neu_y_ret_20 -> horizon=20，y_ret_5 -> horizon=5）
    # delta 是训练集末尾与验证集开头之间的交易日间隔，需 >= 标签实际跨越的交易日数以防止标签泄露
    # 当前标签语义为 T+1 收盘买入 / T+1+N 开盘卖出，实际跨越 N+1 个交易日，故 +1
    try:
        inferred_horizon = int(label_column.rstrip("d").split("_")[-1])
    except (ValueError, IndexError):
        inferred_horizon = 20
    label_delta = max(inferred_horizon + 1, 5)  # 最少 5 个交易日间隔

    # 按 trade_date 粒度切分训练集和验证集（确保同日样本不被拆分到两侧）
    if opts.explicit_train_end is not None or opts.explicit_val_start is not None:
        # F10 显式日期缝：四日期由编排器换算，train 侧不再二次比例切分（§13.3.2 D1）
        if opts.explicit_train_end is None or opts.explicit_val_start is None:
            raise ValueError(
                "explicit_train_end 与 explicit_val_start 必须同时给出"
                "（显式四日期由编排器按现役规则换算后传入）"
            )
        df_train_split, df_val_split_raw, split_stats = split_train_val_by_explicit_dates(
            df_train, train_end=opts.explicit_train_end, val_start=opts.explicit_val_start
        )
    else:
        df_train_split, df_val_split_raw, split_stats = split_train_val_by_date(
            df_train, val_ratio=opts.val_ratio, delta=label_delta
        )

    # 从验证集尾部自动隔离与测试期可能重叠的标签窗口样本，
    # 仅隔离后的子集参与 early stopping / best_iteration 选择。
    (
        df_val_split,
        df_val_split_calib,
        df_val_split_embargo,
        val_protocol_stats,
    ) = split_val_for_selection_protocol_by_date(
        df_val_split_raw,
        embargo_days=label_delta,
    )

    return SimpleNamespace(
        df_train_split=df_train_split,
        df_val_split_raw=df_val_split_raw,
        split_stats=split_stats,
        df_val_split=df_val_split,
        df_val_split_calib=df_val_split_calib,
        df_val_split_embargo=df_val_split_embargo,
        val_protocol_stats=val_protocol_stats,
        label_delta=label_delta,
    )


def _log_val_protocol(val_protocol_stats: Dict) -> None:
    """验证集协议拆分日志（旧 :917-928 原样）。"""
    logger.info(
        "验证集协议拆分（按标签自动推导）: "
        f"raw={val_protocol_stats['val_raw_start_date']}~{val_protocol_stats['val_raw_end_date']} "
        f"({val_protocol_stats['val_raw_n_dates']}日/{val_protocol_stats['val_raw_samples']}样本), "
        f"es={val_protocol_stats['val_es_start_date']}~{val_protocol_stats['val_es_end_date']} "
        f"({val_protocol_stats['val_es_n_dates']}日/{val_protocol_stats['val_es_samples']}样本), "
        f"calib={val_protocol_stats['val_calib_start_date']}~{val_protocol_stats['val_calib_end_date']} "
        f"({val_protocol_stats['val_calib_n_dates']}日/{val_protocol_stats['val_calib_samples']}样本), "
        f"embargo={val_protocol_stats['val_embargo_start_date']}~{val_protocol_stats['val_embargo_end_date']} "
        f"({val_protocol_stats['val_embargo_n_dates']}日/{val_protocol_stats['val_embargo_samples']}样本), "
        f"embargo_days={val_protocol_stats['val_embargo_days_requested']}"
    )


def _finalize_splits(
    splits_ns: SimpleNamespace, qual_ns: SimpleNamespace, label_column: str, opts: "_FamilyOptions"
) -> SimpleNamespace:
    """标签变换 / 稳定性筛选 / X-y 矩阵化（旧 :940-975 原样）。

    入参取命名空间形态（§16.2-bis）；返回 SimpleNamespace，字段沿用原 unpack 名。
    """
    df_train_split = splits_ns.df_train_split
    df_val_split = splits_ns.df_val_split
    feature_columns = qual_ns.feature_columns
    label_transform_fn = opts.label_transform_fn
    feature_stability_filter = opts.feature_stability_filter
    # P2-1/R-CODEX-03 处置：命名空间持有的切分表引用读入局部变量后即删除——
    # 恢复旧实现局部变量覆盖的释放语义（复制型 label_transform_fn 返回后，
    # 变换前训练表/ES 表不再驻留；calibration 原始快照由调用方在 splits_ns
    # 的 df_val_split_calib / df_val_split 选取后持有，不受影响）。
    del splits_ns.df_train_split, splits_ns.df_val_split

    # 如果提供了标签变换函数，切分后各自独立变换（避免跨集合统计量污染）
    if label_transform_fn is not None:
        logger.info("切分后分别对训练集与验证集独立进行标签变换...")
        df_train_split = label_transform_fn(df_train_split)
        if len(df_val_split) > 0:
            df_val_split = label_transform_fn(df_val_split)

    # 特征稳定性筛选：移除跨时期IC方向不一致的特征
    feature_filter_info = None
    if feature_stability_filter:
        feature_columns, feature_filter_info = filter_stable_features(
            df_train=df_train_split,
            feature_columns=feature_columns,
            label_column=label_column,
        )

    # 获取用于 early stopping 的验证集时间范围（标签变换后可能进一步收缩）
    val_es_dates = sorted(df_val_split["trade_date"].unique()) if len(df_val_split) > 0 else []
    val_start_date = str(val_es_dates[0]) if val_es_dates else "N/A"
    val_end_date = str(val_es_dates[-1]) if val_es_dates else "N/A"

    # 准备训练集 X 和 y（float32 内存减半，XGBoost/LightGBM 原生支持）
    X_train = df_train_split[feature_columns].astype(np.float32)
    y_train = df_train_split[label_column].astype(np.float32)

    # 准备验证集 X 和 y
    X_val = df_val_split[feature_columns].astype(np.float32)
    y_val = df_val_split[label_column].astype(np.float32)

    # 显式释放中间 DataFrame 引用，帮助 GC 回收内存
    gc.collect()

    # NaN 处理：XGBoost / LightGBM 原生支持 NaN（自动学习缺失值的最优分裂方向），
    # 不再 fillna(0)，保留 NaN 让模型区分"无数据"与"值为0"。

    logger.info(f"训练数据准备完成: X_train shape={X_train.shape}, X_val shape={X_val.shape}")

    return SimpleNamespace(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_columns=feature_columns,
        df_train_split=df_train_split,
        df_val_split=df_val_split,
        val_start_date=val_start_date,
        val_end_date=val_end_date,
        val_es_n_dates=len(val_es_dates),
        feature_filter_info=feature_filter_info,
    )


def _assemble_data_stats(
    opts: "_FamilyOptions",
    cols_ns: SimpleNamespace,
    qual_ns: SimpleNamespace,
    splits_ns: SimpleNamespace,
    fin_ns: SimpleNamespace,
    val_raw_samples: int,
    df_val_split_original: pd.DataFrame,
) -> Dict:
    """data_stats 字典组装（旧 :977-1024 原样；命名空间入参形态，§16.2-bis）。

    字典内容逐键不变（原 ``_build_data_stats`` 更名）。
    """
    # 数据统计
    return {
        "samples_after_filter": qual_ns.samples_after_filter,
        "train_start_date": splits_ns.split_stats["train_start_date"],
        "train_end_date": splits_ns.split_stats["train_end_date"],
        "val_start_date": str(fin_ns.val_start_date),
        "val_end_date": str(fin_ns.val_end_date),
        "val_raw_start_date": splits_ns.split_stats["val_start_date"],
        "val_raw_end_date": splits_ns.split_stats["val_end_date"],
        "val_raw_n_dates": splits_ns.split_stats["val_n_dates"],
        "val_raw_samples": val_raw_samples,
        "val_es_start_date": str(fin_ns.val_start_date),
        "val_es_end_date": str(fin_ns.val_end_date),
        "val_es_n_dates": fin_ns.val_es_n_dates,
        "val_es_samples": len(fin_ns.df_val_split),
        "val_calib_start_date": splits_ns.val_protocol_stats["val_calib_start_date"],
        "val_calib_end_date": splits_ns.val_protocol_stats["val_calib_end_date"],
        "val_calib_n_dates": splits_ns.val_protocol_stats["val_calib_n_dates"],
        "val_calib_samples": len(df_val_split_original),
        "val_embargo_days": splits_ns.label_delta,
        "val_embargo_days_applied": splits_ns.val_protocol_stats["val_embargo_days_applied"],
        "val_embargo_n_dates": splits_ns.val_protocol_stats["val_embargo_n_dates"],
        "val_embargo_samples": len(splits_ns.df_val_split_embargo),
        "val_embargo_start_date": splits_ns.val_protocol_stats["val_embargo_start_date"],
        "val_embargo_end_date": splits_ns.val_protocol_stats["val_embargo_end_date"],
        "val_calibration_ratio": splits_ns.val_protocol_stats["val_calibration_ratio"],
        "feature_filter_info": fin_ns.feature_filter_info,
        "max_feature_missing_ratio": float(opts.max_feature_missing_ratio),
        "removed_high_missing_features": qual_ns.removed_high_missing,
        "removed_all_nan_features": qual_ns.removed_all_nan,
        "removed_constant_features": qual_ns.removed_constant,
        "removed_duplicate_features": cols_ns.removed_duplicate_features,
        "freshness_strategy": opts.freshness_strategy,
        "event_freshness_half_life_days": float(opts.event_freshness_half_life_days),
        "removed_freshness_features": cols_ns.removed_freshness_features,
        "kept_state_freshness_features": cols_ns.state_freshness_cols_kept,
        "event_freshness_columns_used": cols_ns.event_freshness_cols_used,
        "event_decay_applied_stats": qual_ns.decay_applied_stats,
        # 现金流质量因子：记录哨兵语义版本与门禁后实际入模列，避免汇总只记名义开关
        "cashflow_quality_schema_version": (
            CASHFLOW_QUALITY_SCHEMA_VERSION if opts.enable_cashflow_quality_features else None
        ),
        "cashflow_quality_feature_columns_live": (
            cashflow_quality_live_columns(fin_ns.feature_columns)
            if opts.enable_cashflow_quality_features
            else []
        ),
    }


@dataclass
class _FamilyOptions:
    """族/策略参数打包（§16.2-bis）：主函数开头一次构造，供各编排助手消费。"""

    val_ratio: float
    label_transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]]
    enable_fundamental_features: bool
    enable_alt_features: bool
    enable_margin_features: bool
    enable_cyq_features: bool
    enable_fund_features: bool
    enable_express_features: bool
    enable_enhanced_features: bool
    enable_north_features: bool
    enable_lhb_features: bool
    enable_consensus_features: bool
    enable_cashflow_quality_features: bool
    enable_consensus_revision_features: bool
    enable_dividend_policy_features: bool
    enable_holdertrade_features: bool
    holdertrade_lookup: Optional[dict]
    holdertrade_feature_set: str
    enable_repurchase_features: bool
    repurchase_lookup: Optional[dict]
    repurchase_feature_set: str
    enable_top10fh_features: bool
    top10fh_panel: Optional[pd.DataFrame]
    top10fh_feature_set: str
    enable_top_inst_features: bool
    top_inst_lookup: Optional[dict]
    enable_availability_markers: bool
    feature_stability_filter: bool
    factor_prune: bool
    factor_exclude_file: Optional[str]
    max_feature_missing_ratio: float
    feature_columns_override: Optional[List[str]]
    freshness_strategy: str
    event_freshness_half_life_days: float
    explicit_train_end: Optional[str]
    explicit_val_start: Optional[str]


def _resolve_feature_columns(
    df: pd.DataFrame, opts: _FamilyOptions, runtime_derivers: Optional[RuntimeDerivers]
) -> SimpleNamespace:
    """特征列编排（§16.2-bis）：族扩展助手调用序 = 原主函数调用序。

    返回 SimpleNamespace（feature_columns、removed_duplicate_features、
    removed_freshness_features、event_freshness_cols_used、state_freshness_cols_kept、
    decay_helper_cols）。
    """
    _legacy_exclude_columns(df)  # 旧排除列定义保留（计算结果未被消费，旧 :163-209）

    feature_columns = _base_feature_columns()
    _extend_missing_markers(df, feature_columns)
    _extend_availability_markers(
        df, feature_columns, opts.enable_availability_markers, runtime_derivers
    )
    _extend_declared_family_columns_a(
        df,
        feature_columns,
        enable_fundamental_features=opts.enable_fundamental_features,
        enable_alt_features=opts.enable_alt_features,
        enable_margin_features=opts.enable_margin_features,
        enable_cyq_features=opts.enable_cyq_features,
    )
    _extend_declared_family_columns_b(
        df,
        feature_columns,
        enable_fund_features=opts.enable_fund_features,
        enable_express_features=opts.enable_express_features,
        enable_enhanced_features=opts.enable_enhanced_features,
        enable_north_features=opts.enable_north_features,
        enable_lhb_features=opts.enable_lhb_features,
    )
    _extend_consensus_columns(df, feature_columns, opts.enable_consensus_features)
    _extend_cashflow_quality_columns(df, feature_columns, opts.enable_cashflow_quality_features)
    feature_columns, removed_duplicate_features = _dedup_cashflow_alias(feature_columns)
    _extend_consensus_revision_columns(df, feature_columns, opts.enable_consensus_revision_features)
    _extend_dividend_policy_columns(df, feature_columns, opts.enable_dividend_policy_features)
    _extend_holdertrade_columns(
        df,
        feature_columns,
        opts.enable_holdertrade_features,
        opts.holdertrade_lookup,
        opts.holdertrade_feature_set,
        runtime_derivers,
    )
    _extend_repurchase_columns(
        df,
        feature_columns,
        opts.enable_repurchase_features,
        opts.repurchase_lookup,
        opts.repurchase_feature_set,
        runtime_derivers,
    )
    _extend_top10fh_columns(
        df,
        feature_columns,
        opts.enable_top10fh_features,
        opts.top10fh_panel,
        opts.top10fh_feature_set,
        runtime_derivers,
    )
    _extend_top_inst_columns(
        df, feature_columns, opts.enable_top_inst_features, opts.top_inst_lookup, runtime_derivers
    )
    _append_sz_columns(df, feature_columns)
    feature_columns = _apply_factor_prune(
        feature_columns, opts.factor_prune, opts.factor_exclude_file
    )
    (
        feature_columns,
        removed_freshness_features,
        event_freshness_cols_used,
        state_freshness_cols_kept,
        decay_helper_cols,
    ) = _resolve_freshness_columns(feature_columns, opts.freshness_strategy)

    # 去重，保持顺序稳定
    feature_columns = list(dict.fromkeys(feature_columns))

    if not feature_columns:
        raise ValueError("特征列为空（在因子精简/freshness 策略后）")

    logger.info(f"特征列数量: {len(feature_columns)}")
    logger.debug(f"特征列: {feature_columns[:10]}...")  # 只显示前10个

    return SimpleNamespace(
        feature_columns=feature_columns,
        removed_duplicate_features=removed_duplicate_features,
        removed_freshness_features=removed_freshness_features,
        event_freshness_cols_used=event_freshness_cols_used,
        state_freshness_cols_kept=state_freshness_cols_kept,
        decay_helper_cols=decay_helper_cols,
    )


def _run_quality_pipeline(
    df: pd.DataFrame, cols_ns: SimpleNamespace, label_column: str, opts: _FamilyOptions
) -> SimpleNamespace:
    """质量门禁流水线（§16.2-bis）：列裁剪 → 过滤/衰减 → 门禁 → override。

    返回 SimpleNamespace（df_train、samples_after_filter、decay_applied_stats、
    feature_columns、removed_high_missing、removed_all_nan、removed_constant）。
    P2-1/R-CODEX-03 处置：裁剪后帧不再随命名空间返回（无消费方），其引用在
    本函数返回时即释放（恢复旧实现「释放 ~4-5 GiB 内存」的释放语义）。
    """
    df = _trim_to_needed_columns(
        df, cols_ns.feature_columns, label_column, cols_ns.decay_helper_cols
    )
    df_train, samples_after_filter, decay_applied_stats = _filter_and_decay(
        df,
        label_column,
        opts.freshness_strategy,
        cols_ns.event_freshness_cols_used,
        opts.event_freshness_half_life_days,
    )
    (
        feature_columns,
        removed_high_missing,
        removed_all_nan,
        removed_constant,
    ) = _apply_feature_quality_gates(
        df_train, cols_ns.feature_columns, opts.max_feature_missing_ratio
    )
    feature_columns = _apply_feature_columns_override(
        df_train, feature_columns, opts.feature_columns_override
    )
    return SimpleNamespace(
        df_train=df_train,
        samples_after_filter=samples_after_filter,
        decay_applied_stats=decay_applied_stats,
        feature_columns=feature_columns,
        removed_high_missing=removed_high_missing,
        removed_all_nan=removed_all_nan,
        removed_constant=removed_constant,
    )


def prepare_training_data(
    df: pd.DataFrame,
    label_column: str = "neu_y_ret_20",
    val_ratio: float = 0.2,
    label_transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    enable_fundamental_features: bool = False,
    enable_alt_features: bool = False,
    enable_margin_features: bool = False,
    enable_cyq_features: bool = False,
    enable_fund_features: bool = False,
    enable_express_features: bool = False,
    enable_enhanced_features: bool = False,
    enable_north_features: bool = False,
    enable_lhb_features: bool = False,
    enable_consensus_features: bool = False,
    enable_cashflow_quality_features: bool = False,
    enable_consensus_revision_features: bool = False,
    enable_dividend_policy_features: bool = False,
    enable_holdertrade_features: bool = False,
    holdertrade_lookup: Optional[dict] = None,
    holdertrade_feature_set: str = "full",
    enable_repurchase_features: bool = False,
    repurchase_lookup: Optional[dict] = None,
    repurchase_feature_set: str = "full",
    enable_top10fh_features: bool = False,
    top10fh_panel: Optional[pd.DataFrame] = None,
    top10fh_feature_set: str = "full",
    enable_top_inst_features: bool = False,
    top_inst_lookup: Optional[dict] = None,
    enable_availability_markers: bool = False,
    feature_stability_filter: bool = False,
    factor_prune: bool = False,
    factor_exclude_file: Optional[str] = None,
    max_feature_missing_ratio: float = 0.6,
    feature_columns_override: Optional[List[str]] = None,
    freshness_strategy: str = FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    event_freshness_half_life_days: float = 45.0,
    explicit_train_end: Optional[str] = None,
    explicit_val_start: Optional[str] = None,
    runtime_derivers: Optional[RuntimeDerivers] = None,
) -> tuple:
    """准备训练数据，并按 trade_date 粒度切分训练集和验证集

    Args:
        df: 特征 DataFrame
        label_column: 标签列名
        val_ratio: 验证集比例，默认 0.2（最后 20% 的交易日作为验证集）
        label_transform_fn: 可选的标签变换函数；切分后分别对训练集与验证集独立调用。
        enable_*_features: 各族因子开关（声明式族 + 哨兵校验族 + 运行时派生五族）。
        *_lookup / top10fh_panel: 运行时派生族查询表/面板（详表见模块 docstring）。
        *_feature_set: 运行时派生族列集取值（超参签名维度，详表见模块 docstring）。
        enable_availability_markers: 可用性标记派生开关；feature_stability_filter: 特征稳定性筛选开关。
        factor_prune: 因子精简开关（默认 False）；factor_exclude_file: 显式清单路径。
        max_feature_missing_ratio: 训练入口特征缺失率上限，默认 0.6。
        feature_columns_override: 若提供，质量门禁后强制对齐特征列（缺失列补 NaN）。
        freshness_strategy: freshness 处理策略（state_keep_event_decay 默认 /
            state_keep_event_no_decay / drop_all）。
        event_freshness_half_life_days: 事件型特征衰减半衰期（天），默认 45。
        explicit_train_end / explicit_val_start: 显式切分缝日期（YYYYMMDD，须同时给出；
            默认 None = 旧比例路径）。
        runtime_derivers: 运行时派生五族派生能力注入容器（§16.4）。

    Returns:
                (X_train, y_train, X_val, y_val, feature_columns, df_train_split, df_val_split, data_stats,
                 df_val_split_original) 元组
                - df_val_split: early stopping 子集（仅供模型训练/early stopping 使用）
                - df_val_split_original: calibration 子集的原始快照；若 calibration 为空则回退到
                    early stopping 子集。供逐日收益评估、候选比较与验证评估使用。
                data_stats 包含：samples_after_filter, val_start_date, val_end_date 及
                    val_es_* / val_calib_* / val_embargo_* 统计
    """
    logger.info("准备训练数据...")

    # 确认标签列存在
    if label_column not in df.columns:
        raise ValueError(f"标签列 {label_column} 不存在")

    opts = _FamilyOptions(
        val_ratio=val_ratio,
        label_transform_fn=label_transform_fn,
        enable_fundamental_features=enable_fundamental_features,
        enable_alt_features=enable_alt_features,
        enable_margin_features=enable_margin_features,
        enable_cyq_features=enable_cyq_features,
        enable_fund_features=enable_fund_features,
        enable_express_features=enable_express_features,
        enable_enhanced_features=enable_enhanced_features,
        enable_north_features=enable_north_features,
        enable_lhb_features=enable_lhb_features,
        enable_consensus_features=enable_consensus_features,
        enable_cashflow_quality_features=enable_cashflow_quality_features,
        enable_consensus_revision_features=enable_consensus_revision_features,
        enable_dividend_policy_features=enable_dividend_policy_features,
        enable_holdertrade_features=enable_holdertrade_features,
        holdertrade_lookup=holdertrade_lookup,
        holdertrade_feature_set=holdertrade_feature_set,
        enable_repurchase_features=enable_repurchase_features,
        repurchase_lookup=repurchase_lookup,
        repurchase_feature_set=repurchase_feature_set,
        enable_top10fh_features=enable_top10fh_features,
        top10fh_panel=top10fh_panel,
        top10fh_feature_set=top10fh_feature_set,
        enable_top_inst_features=enable_top_inst_features,
        top_inst_lookup=top_inst_lookup,
        enable_availability_markers=enable_availability_markers,
        feature_stability_filter=feature_stability_filter,
        factor_prune=factor_prune,
        factor_exclude_file=factor_exclude_file,
        max_feature_missing_ratio=max_feature_missing_ratio,
        feature_columns_override=feature_columns_override,
        freshness_strategy=freshness_strategy,
        event_freshness_half_life_days=event_freshness_half_life_days,
        explicit_train_end=explicit_train_end,
        explicit_val_start=explicit_val_start,
    )

    cols_ns = _resolve_feature_columns(df, opts, runtime_derivers)
    qual_ns = _run_quality_pipeline(df, cols_ns, label_column, opts)
    splits_ns = _split_train_val(qual_ns.df_train, label_column, opts)
    _log_val_protocol(splits_ns.val_protocol_stats)
    # 在标签变换前保存 calibration 原始 df 引用；若 calibration 为空则回退到 es 子集。
    # label_transform_fn 内部会创建新 df，原始 df 不会被修改，无需深拷贝。
    df_val_split_original = (
        splits_ns.df_val_split_calib
        if len(splits_ns.df_val_split_calib) > 0
        else splits_ns.df_val_split
    )

    # 释放已不再需要的原始/中间 DataFrame，回收 ~4-5 GiB 内存
    # （P2-1/R-CODEX-03 处置后口径：裁剪后帧随 `_run_quality_pipeline` 返回即释放；
    # 此处释放原始输入帧、过滤后训练帧与 val-raw 帧）
    # 提前保存后续 data_stats 需要的值
    _val_raw_samples = len(splits_ns.df_val_split_raw)
    del df, qual_ns.df_train, splits_ns.df_val_split_raw
    gc.collect()
    fin_ns = _finalize_splits(splits_ns, qual_ns, label_column, opts)
    data_stats = _assemble_data_stats(
        opts, cols_ns, qual_ns, splits_ns, fin_ns, _val_raw_samples, df_val_split_original
    )
    return (
        fin_ns.X_train,
        fin_ns.y_train,
        fin_ns.X_val,
        fin_ns.y_val,
        fin_ns.feature_columns,
        fin_ns.df_train_split,
        fin_ns.df_val_split,
        data_stats,
        df_val_split_original,
    )
