"""十大流通股东因子模块（top10_floatholders）。

数据来源：TuShare `top10_floatholders`（Phase 0 审计与 **Phase 2 口径定稿**见
`docs/top10_floatholders_pit_audit.md` §5.1 / §6），raw 按 `end_date`（报告期）年分区落盘
（`data/raw/top10_floatholders/YYYY-12-31.parquet`）。

**PIT 契约**：`ann_date` 是唯一可用时间（缺失 0%，且实测 `ann_date ≥ end_date`）；T 日可见 =
`ann_date ≤ T` 的**最新一次披露**（同 `ann_ord` 多行取 `end_ord` 最大者，即同日披露多个报告期时取更近的
报告期）。本族为**状态型**（报告期状态保留 + freshness），沿 consensus 契约**不做硬断崖**；
已知代价：极少数长期停牌/退市股票携带极旧报告（freshness p99 > 2,000 自然日），由
`tfh_freshness_days` 显式暴露。

**列集**（6 值列 + freshness + 哨兵 `tfh_schema_v1`）：
`tfh_top10_ratio`（前 10 合计占流通比）、`tfh_top1_ratio`（第一大占流通比）、
`tfh_inst_ratio`（长线机构合计占流通比）、`tfh_inst_count`（长线机构户数）、
`tfh_social_security_flag`（社保类出现）、`tfh_concentration_chg`（集中度环比变化，对齐上一已存报告期）、
`tfh_freshness_days`（距该报告期 `ann_date` 的自然日数）。

**聚合口径**（口径定稿 §6.3）：
① 同 `(ts_code, end_date, ann_date, holder_name)` 多行（全库 0.06%）取 `hold_amount` 最大行；
② 组内按 `hold_amount` 降序取前 10 行后再求和/取首；
③ `hold_float_ratio` 组级缺失 6.99% ⇒ **缺失行按 0 贡献跳过**（不整组置 NaN、不用 `hold_ratio` 冒充），
已知代价：缺失行所在组会低估集中度；
④ `tfh_top1_ratio` 用组内 `max`：同一 (股, 报告期) 流通股本一致 ⇒ 金额最大行的流通比即最大值，
且 `max` 可跳过缺失。

**填充语义（与事件族相反，必须严格遵守）**：有已披露报告的股票 ⇒ 值列为真实值
（**0 = 前 10 中无此类持有人**，是有效观测）；**无任何已披露报告**（新股尚未披露）⇒ 值列与 freshness
**全 NaN**（语义 = 未披露），哨兵列仍写当前版本（覆盖全市场，供训练入口校验）。
**禁止对未披露股票填 0**。

**运行时容器（偏离既有 `{trade_date: DataFrame}` 模式，原因登记见口径定稿 §6.5）**：本族逐日全市场稠密
（~5,500 只/日）⇒ 训练 / OOS 侧一律用**面板**（每股每报告期一行，全历史 ~27 万行）
经 `derive_top10fh_columns`（按 `ts_code` 分组 + `ann_ord` 向后匹配）一次性对齐，**不物化逐日表**；
`build_top10fh_lookup_by_date` 仅用于单日/短区间（纸面 ensure、`features/pipeline.py` 按日路径）。

**不做公共事件衰减**：本族为状态型，freshness 直接入模（沿 consensus 契约）。
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd

#: 因子值列（真实值，未披露股票为 NaN）
TOP10FH_TOP10_RATIO_COL = "tfh_top10_ratio"
TOP10FH_TOP1_RATIO_COL = "tfh_top1_ratio"
TOP10FH_INST_RATIO_COL = "tfh_inst_ratio"
TOP10FH_INST_COUNT_COL = "tfh_inst_count"
TOP10FH_SOCIAL_FLAG_COL = "tfh_social_security_flag"
TOP10FH_CONCENTRATION_CHG_COL = "tfh_concentration_chg"

TOP10FH_COLS = [
    TOP10FH_TOP10_RATIO_COL,  # 前 10 大流通股东合计占流通比（%）
    TOP10FH_TOP1_RATIO_COL,  # 第一大流通股东占流通比（%）
    TOP10FH_INST_RATIO_COL,  # 长线机构合计占流通比（%）
    TOP10FH_INST_COUNT_COL,  # 长线机构户数（0~10）
    TOP10FH_SOCIAL_FLAG_COL,  # 前 10 中是否出现社保类持有人（0/1）
    TOP10FH_CONCENTRATION_CHG_COL,  # 集中度环比变化（百分点；对齐上一已存报告期）
]

#: 报告新鲜度（自然日；未披露股票为 NaN）
TOP10FH_FRESHNESS_COL = "tfh_freshness_days"

#: schema 哨兵列与当前版本（语义重做时递增；训练入口校验）
TOP10FH_VERSION_COL = "tfh_schema_v1"
TOP10FH_SCHEMA_VERSION = 1

#: 列集开关取值（**超参签名维度**，禁止跨取值并组比较）
#: - full：6 个值列 + freshness + 哨兵；
#: - concentration：**单列** `tfh_concentration_chg` + 哨兵
#:   （依据 Phase 3 体检：家族内仅该列有强信息，且正交；见 `docs/top10_floatholders_factor_health.md`）。
#:   注意：`concentration` **不含 freshness**——`tfh_freshness_days` 与 `fundamental_freshness_days`
#:   实测 ρ=0.999（同一“报告新鲜度”轴），且其为全市场覆盖而**不会**被缺失率门禁自动删除，
#:   纳入会把测试变成“1 值列 + 1 重复列”，不得作为单列对照（口径定稿 §6.6）。
TOP10FH_FEATURE_SET_FULL = "full"
TOP10FH_FEATURE_SET_CONCENTRATION = "concentration"
TOP10FH_FEATURE_SETS = (TOP10FH_FEATURE_SET_FULL, TOP10FH_FEATURE_SET_CONCENTRATION)

#: 面板内部列（PIT 对齐用；**不进特征列**）
TOP10FH_ANN_ORD_COL = "tfh_ann_ord"
TOP10FH_END_ORD_COL = "tfh_end_ord"

#: 组内取前 N 大持有人（口径定稿 §6.3 第 2 条）
TOP10FH_TOP_N = 10

#: 逐日表路径允许的最大交易日数（超过请改用面板 + `derive_top10fh_columns`，见 §6.5）
TOP10FH_MAX_DAILY_DATES = 260

#: 长线机构白名单（`holder_type` **整串相等**匹配；实测 38 类中的 6 类，合计 4.66% 行占比）
#: 归一化规则：去首尾空白 + 去内部空白（含全角空格）后比较，禁止模糊子串匹配
#: （否则 `金融机构—证券公司`、`公益基金`、`职工工会` 会被误纳）。
LONG_TERM_INSTITUTIONAL_TYPES: Tuple[str, ...] = (
    "社保基金、社保机构",
    "基本养老保险基金",
    "企业年金",
    "保险投资组合",
    "保险资管产品",
    "金融机构—保险公司",
)

#: 社保类持有人（用于 `tfh_social_security_flag`）
SOCIAL_SECURITY_TYPES: Tuple[str, ...] = ("社保基金、社保机构",)

#: 已排除的相近取值（登记在案：体量可忽略或语义含混，日后可评估纳入）
EXCLUDED_HOLDER_TYPES: Tuple[str, ...] = (
    "保险公司",  # 仅 69 行（保险资金通常记在「保险投资组合」/「保险资管产品」）
    "基金管理公司",  # 仅 164 行，且与 fund_portfolio 家族重叠
)


def _normalize_holder_type(series: pd.Series) -> pd.Series:
    """`holder_type` 归一化：去首尾与内部空白（含全角空格）后用于整串相等比较。"""
    text = series.astype(str).str.strip()
    for blank in ("\u3000", " ", "\t", "\xa0"):
        text = text.str.replace(blank, "", regex=False)
    return text


def _to_ordinal(dates: pd.Series) -> np.ndarray:
    """YYYYMMDD 字符串 → 自然日序号（int64；非法日期为 -1）。"""
    parsed = pd.to_datetime(dates.astype(str), format="%Y%m%d", errors="coerce")
    ordinals = parsed.to_numpy(dtype="datetime64[D]").astype("int64")
    ordinals[pd.isna(parsed).to_numpy()] = -1
    return ordinals.astype("int64")


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ts_code",
            "end_date",
            TOP10FH_ANN_ORD_COL,
            TOP10FH_END_ORD_COL,
            *TOP10FH_COLS,
        ]
    )


def aggregate_top10fh_periods(df: pd.DataFrame) -> pd.DataFrame:
    """把逐持有人明细聚合为 (ts_code, end_date, ann_date) 报告期行（PIT 事件单位）。

    步骤（口径定稿 §6.3）：
    ① 剔除 `ann_date` / `end_date` 缺失或非法的行（禁止伪报告期）；
    ② 同 `(ts_code, end_date, ann_date, holder_name)` 多行 ⇒ 取 `hold_amount` 最大行；
    ③ 组内按 `hold_amount` 降序取前 `TOP10FH_TOP_N` 行；
    ④ 值列：前 10 合计占流通比、第一大占流通比（组内 max）、长线机构合计占流通比与户数、
       社保标志（0/1）。

    **前置条件**：输入应已由 raw 层（`top10_floatholders_raw.deduplicate_top10fh`）按全字段去重；
    本函数检测到整行重复时告警（不静默）。

    Returns:
        DataFrame（`ts_code` + `end_date` + `ann_ord` + `end_ord` + 5 个值列，**不含**环比变化列）。
    """
    if df is None or len(df) == 0:
        return _empty_panel().drop(columns=[TOP10FH_CONCENTRATION_CHG_COL])

    work = df.copy()
    for col in ("ts_code", "end_date", "ann_date", "holder_name", "hold_amount"):
        if col not in work.columns:
            raise ValueError(f"top10_floatholders 缺少必要列: {col}")
    if "hold_float_ratio" not in work.columns:
        raise ValueError(
            "top10_floatholders 缺少 hold_float_ratio 列（禁止静默零因子，请重下数据："
            "python scripts/download_raw.py --download top10_floatholders）"
        )
    if "holder_type" not in work.columns:
        raise ValueError("top10_floatholders 缺少 holder_type 列（长线机构归类依赖它，禁止兜底）")

    dup_rows = int(work.duplicated().sum())
    if dup_rows:
        logger.warning(
            f"[top10_floatholders] 输入含 {dup_rows} 行整行重复（未去重？）——"
            "持仓与计数可能被放大，请先经 raw 层去重"
        )

    work["ann_date"] = normalize_series_to_yyyymmdd(work["ann_date"])
    work["end_date"] = normalize_series_to_yyyymmdd(work["end_date"])
    work[TOP10FH_ANN_ORD_COL] = _to_ordinal(work["ann_date"])
    work[TOP10FH_END_ORD_COL] = _to_ordinal(work["end_date"])
    bad = (work[TOP10FH_ANN_ORD_COL] < 0) | (work[TOP10FH_END_ORD_COL] < 0)
    if int(bad.sum()):
        logger.warning(
            f"[top10_floatholders] 剔除 {int(bad.sum())} 条 ann_date/end_date 非法记录（禁止伪报告期）"
        )
        work = work.loc[~bad].copy()
    if len(work) == 0:
        return _empty_panel().drop(columns=[TOP10FH_CONCENTRATION_CHG_COL])

    amount = pd.to_numeric(work["hold_amount"], errors="coerce")
    if amount.notna().sum() == 0:
        raise ValueError(
            "top10_floatholders hold_amount 整列为空（禁止静默零因子）：请重下数据"
            "（python scripts/download_raw.py --download top10_floatholders）"
        )
    if int(amount.isna().sum()):
        logger.warning(
            f"[top10_floatholders] {int(amount.isna().sum())} 行缺 hold_amount"
            "（排序时排在最后，不参与前 N 选择）"
        )
    work["_amt"] = amount

    ratio = pd.to_numeric(work["hold_float_ratio"], errors="coerce")
    missing_ratio = int(ratio.isna().sum())
    if missing_ratio:
        # 口径定稿 §6.3 第 3 条：缺失行按 0 贡献跳过（已知代价：低估集中度）
        ratio_share = missing_ratio / len(work)
        logger.warning(
            f"[top10_floatholders] {missing_ratio} 行缺 hold_float_ratio（{ratio_share:.2%}），"
            "按登记口径跳过其贡献（该组集中度可能被低估）"
        )
    work["_fr"] = ratio

    holder_type = _normalize_holder_type(work["holder_type"])
    work["_inst"] = holder_type.isin(LONG_TERM_INSTITUTIONAL_TYPES)
    work["_social"] = holder_type.isin(SOCIAL_SECURITY_TYPES)

    # ② 同键同名多行取金额最大行（§4）
    work = work.sort_values("_amt", ascending=False, na_position="last", kind="mergesort")
    work = work.drop_duplicates(
        subset=["ts_code", "end_date", "ann_date", "holder_name"], keep="first"
    )

    # ③ 组内按金额降序取前 N
    work = work.sort_values(
        ["ts_code", "end_date", "ann_date", "_amt"],
        ascending=[True, True, True, False],
        na_position="last",
        kind="mergesort",
    )
    work["_rk"] = work.groupby(["ts_code", "end_date", "ann_date"], sort=False).cumcount() + 1
    if int((work["_rk"] > 1).sum()):
        over = work.loc[work["_rk"] > TOP10FH_TOP_N, ["ts_code", "end_date", "ann_date"]]
        if len(over):
            logger.debug(
                f"[top10_floatholders] {over.drop_duplicates().shape[0]} 个报告期的行数超过 "
                f"{TOP10FH_TOP_N}（已按金额取前 {TOP10FH_TOP_N}）"
            )
    top = work.loc[work["_rk"] <= TOP10FH_TOP_N]

    keys = ["ts_code", "end_date", TOP10FH_ANN_ORD_COL, TOP10FH_END_ORD_COL]
    grouped = top.groupby(keys, sort=False)
    inst = top.loc[top["_inst"]].groupby(keys, sort=False)

    agg = pd.DataFrame(
        {
            # NaN 跳过求和；整组全 NaN ⇒ NaN（未披露流通比，而非 0）
            TOP10FH_TOP10_RATIO_COL: grouped["_fr"].sum(min_count=1),
            # 同一 (股, 报告期) 流通股本一致 ⇒ 金额最大行的流通比即组内最大值（且可跳过缺失）
            TOP10FH_TOP1_RATIO_COL: grouped["_fr"].max(),
        }
    )
    agg[TOP10FH_INST_RATIO_COL] = inst["_fr"].sum(min_count=1).reindex(agg.index).fillna(0.0)
    agg[TOP10FH_INST_COUNT_COL] = inst.size().reindex(agg.index).fillna(0.0)
    agg[TOP10FH_SOCIAL_FLAG_COL] = (
        top.loc[top["_social"]].groupby(keys, sort=False).size().reindex(agg.index).fillna(0.0) > 0
    ).astype("float64")

    panel = agg.reset_index()
    panel = panel.sort_values(["ts_code", TOP10FH_END_ORD_COL], kind="mergesort").reset_index(
        drop=True
    )
    return panel


def build_top10fh_panel(df: pd.DataFrame) -> pd.DataFrame:
    """构建运行时**面板**：每股每报告期一行（含环比变化列），PIT 对齐用。

    在 `aggregate_top10fh_periods` 基础上：
    ① 计算 `tfh_concentration_chg`（对齐该股票**上一已存报告期**，按 `end_ord` 排序；首期为 NaN）——
    同一报告期多个 `ann_date` 版本（0.02%）先按 `(ts_code, end_ord)` 去重取最新公告版本，
    避免把"同期间修订"误算成环比；
    ② 按 `(ts_code, ann_ord)` 去重，保留 `end_ord` 最大者
    （同日披露多个报告期时取更近的报告期；同报告期多次披露时取更晚版本）。

    Returns:
        DataFrame（`ts_code` + `ann_ord` + `end_ord` + `TOP10FH_COLS`），按 `(ts_code, ann_ord)` 升序。
    """
    agg = aggregate_top10fh_periods(df)
    if len(agg) == 0:
        return _empty_panel()

    # ① 环比变化：同期间多版本先取最新公告版本，再按 end_ord 排序取上一期
    per_period = agg.sort_values(
        ["ts_code", TOP10FH_END_ORD_COL, TOP10FH_ANN_ORD_COL], kind="mergesort"
    ).drop_duplicates(subset=["ts_code", TOP10FH_END_ORD_COL], keep="last")
    prev = per_period.groupby("ts_code", sort=False)[TOP10FH_TOP10_RATIO_COL].shift(1)
    per_period = per_period.assign(
        **{TOP10FH_CONCENTRATION_CHG_COL: per_period[TOP10FH_TOP10_RATIO_COL] - prev}
    )
    panel = agg.merge(
        per_period[["ts_code", TOP10FH_END_ORD_COL, TOP10FH_CONCENTRATION_CHG_COL]],
        on=["ts_code", TOP10FH_END_ORD_COL],
        how="left",
    )

    # ② 按 (ts_code, ann_ord) 去重：同日披露多个报告期 ⇒ 取 end_ord 更大者
    panel = panel.sort_values(
        ["ts_code", TOP10FH_ANN_ORD_COL, TOP10FH_END_ORD_COL], kind="mergesort"
    )
    panel = panel.drop_duplicates(subset=["ts_code", TOP10FH_ANN_ORD_COL], keep="last")

    columns = ["ts_code", "end_date", TOP10FH_ANN_ORD_COL, TOP10FH_END_ORD_COL, *TOP10FH_COLS]
    panel = panel[columns].reset_index(drop=True)
    for col in TOP10FH_COLS:
        panel[col] = panel[col].astype("float64")
    panel[TOP10FH_ANN_ORD_COL] = panel[TOP10FH_ANN_ORD_COL].astype("int64")
    panel[TOP10FH_END_ORD_COL] = panel[TOP10FH_END_ORD_COL].astype("int64")
    logger.info(
        f"[top10_floatholders] 报告期面板构建完成: {len(panel):,} 行 / "
        f"{panel['ts_code'].nunique():,} 只股票 / {panel['end_date'].nunique()} 个报告期"
    )
    return panel


def build_top10fh_day_frame(panel: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """由面板取单日截面（**全部有已披露报告的股票**，未披露股票缺席 ⇒ 消费侧 NaN）。

    Returns:
        DataFrame（`ts_code` + `TOP10FH_COLS` + `tfh_ann_ord`）；无数据时返回空表（含列）。
    """
    columns = ["ts_code", *TOP10FH_COLS, TOP10FH_ANN_ORD_COL]
    if panel is None or len(panel) == 0:
        return pd.DataFrame(columns=columns)
    day_ord = int(_to_ordinal(pd.Series([str(trade_date)]))[0])
    if day_ord < 0:
        raise ValueError(f"非法交易日: {trade_date!r}（top10_floatholders 单日截面）")
    sub = panel.loc[panel[TOP10FH_ANN_ORD_COL] <= day_ord]
    if len(sub) == 0:
        return pd.DataFrame(columns=columns)
    # 按 (ann_ord, end_ord) 升序后取每股最后一行 = 最近一次披露（同日多报告期取更近报告期）
    day = sub.sort_values(
        [TOP10FH_ANN_ORD_COL, TOP10FH_END_ORD_COL], kind="mergesort"
    ).drop_duplicates(subset=["ts_code"], keep="last")
    return day[columns].reset_index(drop=True)


def build_top10fh_lookup_by_date(
    df: pd.DataFrame,
    trading_dates: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """构建逐交易日的截面表（**仅用于单日/短区间**，见口径定稿 §6.5）。

    本族逐日全市场稠密（~5,500 只/日），逐日表在长区间会膨胀到 GB 级 ⇒
    交易日数超过 `TOP10FH_MAX_DAILY_DATES` 直接报错；训练/OOS 侧请改用
    `build_top10fh_panel` + `derive_top10fh_columns`。
    """
    dates = [str(d) for d in trading_dates]
    if not dates:
        return {}
    if len(dates) > TOP10FH_MAX_DAILY_DATES:
        raise ValueError(
            f"top10_floatholders 逐日查询表仅支持 ≤ {TOP10FH_MAX_DAILY_DATES} 个交易日"
            f"（收到 {len(dates)} 个）：本族逐日全市场稠密，长区间请改用"
            " build_top10fh_panel + derive_top10fh_columns（口径定稿 §6.5）"
        )
    panel = build_top10fh_panel(df)
    lookup: Dict[str, pd.DataFrame] = {}
    for trade_date in dates:
        lookup[trade_date] = build_top10fh_day_frame(panel, trade_date)
    return lookup


def available_top10fh_columns() -> List[str]:
    """因子模块输出的全部列（含哨兵列），用于 schema 与 handler 默认列。"""
    return list(TOP10FH_COLS) + [TOP10FH_FRESHNESS_COL, TOP10FH_VERSION_COL]


def top10fh_feature_columns(
    feature_set: str = TOP10FH_FEATURE_SET_FULL,
) -> List[str]:
    """按列集取值返回训练/派生使用的列清单（含哨兵列，单一取值判定，无回退）。"""
    if feature_set == TOP10FH_FEATURE_SET_FULL:
        return available_top10fh_columns()
    if feature_set == TOP10FH_FEATURE_SET_CONCENTRATION:
        return [TOP10FH_CONCENTRATION_CHG_COL, TOP10FH_VERSION_COL]
    raise ValueError(
        f"未知 top10fh 列集: {feature_set!r}（可选 {list(TOP10FH_FEATURE_SETS)}）"
    )


def load_top10fh_panel(loader: Any) -> pd.DataFrame:
    """从 raw 加载并构建报告期面板（运行时派生入口；PIT 对齐由派生函数完成）。

    Args:
        loader: `DataLoader`（或具备 `load_top10_floatholders()` 的等价对象）

    Returns:
        面板 DataFrame；raw 为空时返回空面板（调用方按"未披露"处理 ⇒ 值列 NaN）。
    """
    raw = loader.load_top10_floatholders()
    if raw is None or len(raw) == 0:
        logger.warning("[top10_floatholders] raw 为空，运行时派生将全部按未披露（NaN）处理")
        return _empty_panel()
    return build_top10fh_panel(raw)


def _source(merged: pd.DataFrame, features: pd.DataFrame, col: str) -> pd.Series:
    """从合并结果取列；缺列 ⇒ 全 NaN（未披露语义，**不是 0**）。"""
    if col in merged.columns:
        return pd.to_numeric(merged[col], errors="coerce")
    return pd.Series(np.nan, index=features.index, dtype="float64")


def build_top10fh_feature_frame(
    features: pd.DataFrame,
    merged: Optional[pd.DataFrame],
    trade_date: str,
) -> pd.DataFrame:
    """由"当日特征帧 + 当日截面合并结果"算出因子列（**未披露一律 NaN，禁止 0 填充**）。

    单一实现来源：handler（构建/推理侧）与 `derive_top10fh_columns`（训练/OOS 侧）共用，
    保证四侧逐值一致。`merged is None` 或空表 ⇒ 值列与 freshness 全 NaN
    （语义 = 当日无已披露报告），哨兵列由 handler 恒写当前版本。

    Returns:
        DataFrame（index 与 `features` 一致），列 = `TOP10FH_COLS` + `tfh_freshness_days`。
    """
    day_ord = int(_to_ordinal(pd.Series([str(trade_date)]))[0])
    if day_ord < 0:
        raise ValueError(f"非法交易日: {trade_date!r}（top10_floatholders 因子构建）")
    result = pd.DataFrame(index=features.index)
    if merged is None or len(merged) == 0:
        for col in TOP10FH_COLS:
            result[col] = pd.Series(np.nan, index=features.index, dtype="float64")
        result[TOP10FH_FRESHNESS_COL] = pd.Series(np.nan, index=features.index, dtype="float64")
        return result

    for col in TOP10FH_COLS:
        result[col] = _source(merged, features, col)
    ann_ord = _source(merged, features, TOP10FH_ANN_ORD_COL)
    result[TOP10FH_FRESHNESS_COL] = day_ord - ann_ord
    return result


def derive_top10fh_columns(
    frame: pd.DataFrame,
    panel: Optional[pd.DataFrame],
    wanted: Optional[Iterable[str]] = None,
    log_prefix: str = "",
) -> List[str]:
    """训练/OOS 评估侧就地派生十大流通股东列（**不写回 cs_train / cs_infer 分区**）。

    与 `available_top10fh_columns()` 同语义来源；按 `ts_code` 分组、以 `ann_ord` 向后匹配
    （`searchsorted`）一次性对齐，**不物化逐日表**（面板容器，见口径定稿 §6.5）。

    规则：
    - 需要 `trade_date` 与 `ts_code` 列，缺任一列直接报错（不得静默跳过）；`trade_date` 非法直接报错；
    - **已存在的列不覆盖**（特征分区若已含本族列，以分区为准）；
    - `wanted` 用于只派生模型实际使用的列（None = 全部）；
    - 截止日之前**无任何已披露报告**（或 panel 为空）⇒ 值列与 freshness 为 NaN；
      哨兵列对**全部行**写当前版本（语义标记，与是否有数据无关）。

    Returns:
        实际新增的列名列表（按 `available_top10fh_columns()` 顺序）。
    """
    for col in ("trade_date", "ts_code"):
        if col not in frame.columns:
            raise ValueError(f"十大流通股东运行时派生缺少必要列: {col}")
    if not frame.index.is_unique:
        raise ValueError("十大流通股东运行时派生要求输入帧索引唯一（按位置回填）")

    candidates = available_top10fh_columns()
    if wanted is not None:
        wanted_set = set(wanted)
        candidates = [col for col in candidates if col in wanted_set]
    targets = [col for col in candidates if col not in frame.columns]
    if not targets:
        return []

    day_ords = _to_ordinal(frame["trade_date"])
    if (day_ords < 0).any():
        bad = frame.loc[day_ords < 0, "trade_date"].unique()[:5]
        raise ValueError(f"Top10_floatholders 派生遇到非法 trade_date: {list(bad)}")

    value_targets = [
        col for col in targets if col not in (TOP10FH_FRESHNESS_COL, TOP10FH_VERSION_COL)
    ]
    out = {col: np.full(len(frame), np.nan, dtype="float64") for col in value_targets}
    if TOP10FH_FRESHNESS_COL in targets:
        out[TOP10FH_FRESHNESS_COL] = np.full(len(frame), np.nan, dtype="float64")

    if panel is not None and len(panel) > 0:
        panel_sorted = panel.sort_values(["ts_code", TOP10FH_ANN_ORD_COL], kind="mergesort")
        per_code: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
        for code, grp in panel_sorted.groupby("ts_code", sort=False):
            per_code[str(code)] = (
                grp[TOP10FH_ANN_ORD_COL].to_numpy(dtype="int64"),
                {col: grp[col].to_numpy(dtype="float64") for col in value_targets},
            )

        for code, sub in frame.groupby("ts_code", sort=False):
            entry = per_code.get(str(code))
            if entry is None:
                continue
            ann_ord, values = entry
            positions = frame.index.get_indexer(sub.index)
            sub_days = day_ords[positions]
            matched = np.searchsorted(ann_ord, sub_days, side="right") - 1
            valid = matched >= 0
            if not valid.any():
                continue
            target_pos = positions[valid]
            selected = matched[valid]
            for col in value_targets:
                out[col][target_pos] = values[col][selected]
            if TOP10FH_FRESHNESS_COL in targets:
                out[TOP10FH_FRESHNESS_COL][target_pos] = sub_days[valid] - ann_ord[selected]

    derived = pd.DataFrame({col: out[col] for col in out}, index=frame.index)
    frame[list(out)] = derived
    if TOP10FH_VERSION_COL in targets:
        # 哨兵恒写当前版本（含未披露股票），供训练入口校验语义版本
        frame[TOP10FH_VERSION_COL] = np.int8(TOP10FH_SCHEMA_VERSION)
    if log_prefix:
        derived_cols = list(out)
        if TOP10FH_VERSION_COL in targets:
            derived_cols.append(TOP10FH_VERSION_COL)
        logger.info(f"{log_prefix}十大流通股东运行时派生列: {derived_cols}")
    return targets


def top10fh_coverage_report(panel: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """粗略覆盖率自检：返回 (报告期行数, 唯一股票数)。"""
    if panel is None or len(panel) == 0:
        return None
    return int(len(panel)), int(panel["ts_code"].nunique())
