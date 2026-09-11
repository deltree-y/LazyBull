"""期末异常亏损数据集构建：多期限瘦表、特征矩阵、时间分割与隔离

实现 docs/plans/terminal_loss_risk_model_plan.md 第 4/5/6 节契约：

- pct_* 百分位必须基于标签过滤前的完整当日母截面生成（方案 4.4）：
  调用方传入的母截面来自 ``mother_section.build_mother_section``（clean/daily
  全量化重建、与特征流水线同一实现的四个基列），先全截面排名、再筛有效
  标签行；禁止用 cs_train 过滤后的行重建分母（分母窄 5%~10%），也禁止
  先筛持仓/有效行再排名；
- mkt_* 广播列不做同日百分位（同日所有股票相同，排名无意义）；
- 特征 manifest 冻结：缺整列必须失败，不逐日静默缩减列数（方案 3.2）；
- 样本权重 = 1/期限网格大小（方案第 6 节规则 1：每个 (ts_code,
  trade_date) 组总权重为 1，未成熟/缺失期限不机械重归一）；
- 阶段分割必须满足 label_end_date < 下一阶段起点（方案 5.3 多期限
  标签隔离），同一股票同日的不同 h 随 T 整组进入同一阶段。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .labels import LABEL_STATUS_VALID, TerminalLossLabelConfig

# ── 特征 manifest（方案 3.2 首轮 33 列，冻结清单）────────────────────

#: 由 cs_train 特征母截面直接提供的基础因子列
BASE_FEATURES: List[str] = [
    "ret_5",
    "ret_20",
    "momentum_decay",
    "rsi_14",
    "ma_deviation_20",
    "acceleration",
    "downside_vol_20",
    "downside_corr_20",
    "cvar_95_20",
    "max_drawdown_20",
    "drawdown_duration",
    "skewness_20",
    "gap_risk",
    "parkinson_vol_20",
    "vol_of_vol_20",
    "amihud_illiq_20",
    "amount_cv_20",
    "vol_ratio_5_20",
    "up_down_vol_ratio",
    "volume_price_divergence",
    "mkt_ret_avg_20",
    "mkt_ret_avg_60",
    "mkt_ret_vol_20",
    "mkt_vol_20",
    "mkt_drawdown_20",
    "mkt_adv_dec_ratio",
]

#: pct_* 派生列 → 百分位基列（基列由完整同日母截面提供，见 mother_section）
PCT_FEATURE_BASES: Dict[str, str] = {
    "pct_ret_20": "ret_20",
    "pct_cvar_95_20": "cvar_95_20",
    "pct_max_drawdown_20": "max_drawdown_20",
    "pct_amihud_illiq_20": "amihud_illiq_20",
}

#: dataset 派生列（remaining_intervals 即 h；sigma_daily_20 来自独立面板）
DERIVED_FEATURES: List[str] = [
    "remaining_intervals",
    "sigma_daily_20",
    "expected_vol_over_horizon",
]

#: 完整冻结 manifest（33 列）
TERMINAL_LOSS_FEATURES: List[str] = (
    DERIVED_FEATURES + BASE_FEATURES + list(PCT_FEATURE_BASES.keys())
)

#: 训练矩阵元数据列（非特征）
META_COLUMNS: List[str] = [
    "ts_code",
    "trade_date",
    "h",
    "label_end_date",
    "terminal_return",
    "loss_label",
    "sample_weight",
]


@dataclass(frozen=True)
class DatasetConfig:
    """数据集配置。

    Attributes:
        horizon_grid_size: 权重归一的期限网格大小（每行权重 = 1/该值，
            与方案第 6 节规则 1 对应；未成熟期限不重归一）
    """

    horizon_grid_size: int = 20


def empty_training_matrix() -> pd.DataFrame:
    """构造与正常输出同列同 dtype 的空训练矩阵（schema 契约）。

    分块构建时，数据末端的块可能整块无 valid 标签（如 ES 终点日全部
    h 端点超出数据末端，均 immature）。``pd.DataFrame(columns=...)``
    产生的空帧全列为 object dtype，与其他块 ``pd.concat`` 后整列被
    提升为 object，XGBoost 将拒绝输入（dtypes must be int/float/bool）。
    空矩阵必须保持与非空路径一致的列与 dtype。
    """
    dtypes: Dict[str, str] = {
        "ts_code": "object",
        "trade_date": "object",
        "label_end_date": "object",
        "h": "int64",
        "remaining_intervals": "int64",
    }
    return pd.DataFrame(
        {
            c: pd.Series(dtype=dtypes.get(c, "float64"))
            for c in META_COLUMNS + TERMINAL_LOSS_FEATURES
        }
    )


def validate_feature_manifest(available_columns: Sequence[str]) -> None:
    """校验冻结 manifest 完整性：缺列必须失败，不静默缩减。"""
    available = set(available_columns)
    missing = [c for c in BASE_FEATURES if c not in available]
    missing += [b for b in PCT_FEATURE_BASES.values() if b not in available]
    if missing:
        raise ValueError(
            f"特征母截面缺少 manifest 基础列: {sorted(set(missing))}；"
            f"特征清单冻结（方案 3.2），禁止静默缩减列数，请检查特征构建"
        )


def add_pct_features(day_df: pd.DataFrame, mother_df: pd.DataFrame) -> pd.DataFrame:
    """基于完整同日母截面生成 pct_* 百分位列（方案 4.2/4.3/4.4）。

    百分位分母必须是**标签过滤前的完整当日截面**（``mother_section``
    的产物，证券域为该日 clean/daily 全部有行股票）。传入 cs_train 当日
    子集会收窄分母（实测约 88%），因此本函数强制要求显式传入母截面，
    不做隐式回退。

    Args:
        day_df: 当日训练行（含 manifest 基础列与 ts_code）
        mother_df: 完整同日母截面（ts_code + PCT_FEATURE_BASES 的四个基列）

    Returns:
        追加 pct_* 列后的副本（行序与 day_df 一致）

    Raises:
        ValueError: day_df 缺 manifest 基础列、母截面缺基列，或 day_df 的
            股票不在母截面证券域内（数据链路不一致，禁止静默给 NaN）
    """
    missing_manifest = [c for c in BASE_FEATURES if c not in day_df.columns]
    if missing_manifest:
        raise ValueError(
            f"特征母截面缺少 manifest 基础列: {sorted(set(missing_manifest))}；"
            f"特征清单冻结（方案 3.2），禁止静默缩减列数，请检查特征构建"
        )
    if mother_df is None or mother_df.empty:
        raise ValueError(
            "pct_* 必须基于完整同日母截面生成（方案 4.4），母截面为空；"
            "请检查 mother_section.build_mother_section 是否覆盖该交易日"
        )
    base_cols = list(PCT_FEATURE_BASES.values())
    missing_mother = [c for c in base_cols if c not in mother_df.columns]
    if missing_mother:
        raise ValueError(
            f"完整母截面缺少 pct 基列: {sorted(set(missing_mother))}；"
            f"pct_* 分母不可缺列，拒绝继续"
        )
    mother_codes = set(mother_df["ts_code"])
    absent = ~day_df["ts_code"].isin(mother_codes)
    if bool(absent.any()):
        sample = day_df.loc[absent, "ts_code"].head(5).tolist()
        raise ValueError(
            f"{int(absent.sum())} 行股票不在完整母截面中（示例 {sample}）："
            f"母截面证券域与当日行情不一致，拒绝以 NaN 冒充百分位"
        )

    mother_pct = mother_df[["ts_code"] + base_cols].copy()
    for pct_col, base_col in PCT_FEATURE_BASES.items():
        mother_pct[pct_col] = mother_pct[base_col].rank(pct=True)
    # pct_* 是本函数的派生输出：输入若已带同名列（旧口径产物），必须先丢弃
    # 再按母截面重算，避免 merge 产生 _x/_y 后缀把旧分母悄悄带进矩阵
    day = day_df.drop(columns=list(PCT_FEATURE_BASES), errors="ignore")
    out = day.merge(
        mother_pct[["ts_code"] + list(PCT_FEATURE_BASES)],
        on="ts_code",
        how="left",
        validate="many_to_one",
    )
    return out


def attach_horizon_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """追加 remaining_intervals 与 expected_vol_over_horizon（σ√h）。

    Args:
        matrix: 已含 h 与 sigma_daily_20 列的关联矩阵

    Returns:
        追加派生列后的副本
    """
    out = matrix.copy()
    out["remaining_intervals"] = out["h"].astype(int)
    out["expected_vol_over_horizon"] = out["sigma_daily_20"].astype(float) * np.sqrt(
        out["h"].astype(int)
    )
    return out


def build_training_matrix(
    labels_df: pd.DataFrame,
    features_by_date: Dict[str, pd.DataFrame],
    sigma_panel: pd.DataFrame,
    mother_by_date: Dict[str, pd.DataFrame],
    config: Optional[DatasetConfig] = None,
    label_config: Optional[TerminalLossLabelConfig] = None,
) -> pd.DataFrame:
    """关联标签瘦表与特征母截面，产出训练矩阵。

    约定：features_by_date 的键为 trade_date，值为该日训练行
    （不含 pct_*，由本函数统一生成，保证"先全截面排名、再筛行"）；
    mother_by_date 为同日**标签过滤前的完整母截面**，是 pct_* 的分母。

    Args:
        labels_df: build_terminal_loss_labels 输出的瘦标签表
        features_by_date: {trade_date: 当日训练行特征}
        sigma_panel: sigma_daily_20 面板（index=trade_date, columns=ts_code）
        mother_by_date: {trade_date: 完整当日母截面（pct 基列）}
        config: 数据集配置
        label_config: 标签配置（用于日志与 manifest 校验的 h 网格）

    Returns:
        训练矩阵：META_COLUMNS + TERMINAL_LOSS_FEATURES，
        仅含 label_status == valid 的行，sample_weight = 1/网格大小
    """
    cfg = config or DatasetConfig()
    if labels_df.empty:
        return empty_training_matrix()

    valid = labels_df[labels_df["label_status"] == LABEL_STATUS_VALID].copy()
    if valid.empty:
        logger.warning("标签表无 valid 行，训练矩阵为空")
        return empty_training_matrix()

    sigma_stack = sigma_panel.stack()
    sigma_stack.index = sigma_stack.index.set_names(["trade_date", "ts_code"])
    valid["sigma_daily_20"] = valid.set_index(["trade_date", "ts_code"]).index.map(sigma_stack)

    pieces: List[pd.DataFrame] = []
    skipped_days = 0
    for trade_date, day_labels in valid.groupby("trade_date"):
        day_features = features_by_date.get(trade_date)
        if day_features is None or day_features.empty:
            skipped_days += 1
            continue
        mother = mother_by_date.get(trade_date)
        if mother is None or mother.empty:
            raise ValueError(
                f"{trade_date} 有训练行但缺少完整母截面：pct_* 分母不可用，"
                f"禁止退化为过滤后子集排名（方案 4.4）"
            )
        day_features = add_pct_features(day_features, mother)
        merged = day_labels.merge(
            day_features[["ts_code"] + BASE_FEATURES + list(PCT_FEATURE_BASES)],
            on="ts_code",
            how="inner",
        )
        pieces.append(merged)
        _ = trade_date  # 日志占位，避免长循环内大量 IO 日志
    if skipped_days:
        logger.warning(f"特征母截面缺失 {skipped_days} 个交易日，相关标签行被跳过")

    if not pieces:
        return empty_training_matrix()
    matrix = pd.concat(pieces, ignore_index=True)

    matrix = attach_horizon_features(matrix)
    matrix["sample_weight"] = 1.0 / cfg.horizon_grid_size
    dropped = int((matrix["sigma_daily_20"].isna()).sum())
    if dropped:
        # sigma 来自标签构造同一面板，valid 行不应缺失；出现即数据链路异常
        raise ValueError(f"关联后 {dropped} 行 sigma_daily_20 缺失，sigma 面板与标签股票域不一致")
    return matrix[META_COLUMNS + TERMINAL_LOSS_FEATURES]


@dataclass
class StageSpec:
    """单个时间阶段定义（方案 5.2：Train/ES/校准 C/政策验证 V/OOS）。

    Attributes:
        name: 阶段名
        start / end: 该阶段 trade_date 区间（闭区间，YYYYMMDD）
    """

    name: str
    start: str
    end: str


@dataclass
class StageSplitResult:
    """阶段分割结果：样本随 T 整组进入同一阶段，隔离剔除单独统计。"""

    stages: Dict[str, pd.DataFrame] = field(default_factory=dict)
    isolation_dropped: Dict[str, int] = field(default_factory=dict)


def split_stages_with_label_isolation(
    matrix: pd.DataFrame, stages: Sequence[StageSpec]
) -> StageSplitResult:
    """按阶段切分训练矩阵并执行多期限标签隔离（方案 5.3）。

    阶段 i 的样本条件：trade_date ∈ [start_i, end_i] 且（若存在下一阶段）
    label_end_date < start_{i+1}——短 h 已成熟的样本可保留，未成熟跨界
    的行必须剔除；同一 (ts_code, trade_date) 的不同 h 天然随 T 同阶段。

    Args:
        matrix: 训练矩阵（须含 trade_date, label_end_date）
        stages: 有序阶段列表

    Returns:
        StageSplitResult：各阶段样本 + 各阶段因隔离剔除的行数
    """
    result = StageSplitResult()
    if matrix.empty:
        result.stages = {s.name: matrix for s in stages}
        return result

    for i, stage in enumerate(stages):
        in_range = (matrix["trade_date"] >= stage.start) & (matrix["trade_date"] <= stage.end)
        if i + 1 < len(stages):
            next_start = stages[i + 1].start
            isolated = matrix["label_end_date"].notna() & (matrix["label_end_date"] >= next_start)
            keep = in_range & ~isolated
            result.isolation_dropped[stage.name] = int((in_range & isolated).sum())
        else:
            keep = in_range
            result.isolation_dropped[stage.name] = 0
        result.stages[stage.name] = matrix[keep].reset_index(drop=True)
        logger.info(
            f"阶段 {stage.name} [{stage.start},{stage.end}]: "
            f"{int(keep.sum())} 行"
            + (
                f"，标签隔离剔除 {result.isolation_dropped[stage.name]} 行"
                if i + 1 < len(stages)
                else ""
            )
        )
    return result


# ── 数据加载（clean/daily 与 cs_train 逐日分区的薄 IO 封装）──────────


def _to_dash_date(yyyymmdd: str) -> str:
    """YYYYMMDD → YYYY-MM-DD（clean/daily 分区名格式）。"""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def load_trade_calendar(data_root: str, start_date: str, end_date: str) -> List[str]:
    """从 clean/trade_cal.parquet 读取区间内的开市日历（YYYYMMDD 升序）。"""
    cal = pd.read_parquet(f"{data_root}/clean/trade_cal.parquet")
    cal = cal[cal["is_open"] == 1].copy()
    cal["cal_date"] = cal["cal_date"].astype(str)
    in_range = (cal["cal_date"] >= start_date) & (cal["cal_date"] <= end_date)
    return sorted(cal.loc[in_range, "cal_date"].tolist())


def load_clean_daily_panels(
    data_root: str, start_date: str, end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """加载区间 clean/daily，产出复权开盘/收盘与跌停面板（reindex 完整日历）。

    clean/daily 停牌日无行，pivot + reindex 后为 NaN 槽位（契约：不可前填）。
    分区文件缺失按数据缺口明确报错，不静默跳过。

    Args:
        data_root: 数据根目录（如 data）
        start_date / end_date: YYYYMMDD 闭区间

    Returns:
        (open_adj_panel, close_adj_panel, limit_down_panel, calendar)
    """
    calendar = load_trade_calendar(data_root, start_date, end_date)
    if not calendar:
        raise ValueError(f"日历区间 [{start_date},{end_date}] 内无开市日")
    long_rows: List[pd.DataFrame] = []
    for d in calendar:
        path = f"{data_root}/clean/daily/{_to_dash_date(d)}.parquet"
        day = pd.read_parquet(path, columns=["ts_code", "open_adj", "close_adj", "is_limit_down"])
        long_rows.append(day.assign(trade_date=d))
    long_df = pd.concat(long_rows, ignore_index=True)

    def _pivot(value_col: str) -> pd.DataFrame:
        return long_df.pivot(index="trade_date", columns="ts_code", values=value_col).reindex(
            calendar
        )

    open_panel = _pivot("open_adj")
    close_panel = _pivot("close_adj")
    limit_panel = _pivot("is_limit_down").fillna(0)
    return open_panel, close_panel, limit_panel, calendar


def load_cs_train_days(
    data_root: str, dates: Sequence[str], columns: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    """逐日加载 cs_train 特征母截面（ts_code + 指定列）。

    Args:
        data_root: 数据根目录
        dates: YYYYMMDD 日期列表（分区名同格式）
        columns: 除 ts_code 外需要的列

    Returns:
        {trade_date: 当日完整母截面}；分区缺失明确报错
    """
    needed = ["ts_code"] + list(columns)
    out: Dict[str, pd.DataFrame] = {}
    for d in dates:
        path = f"{data_root}/features/cs_train/{d}.parquet"
        day = pd.read_parquet(path, columns=needed)
        out[d] = day
    return out


# ── 预登记抽样（方案第 6 节：日期/股票/h 分层抽样，保留自然事件率）────


def subsample_h_per_group(
    matrix: pd.DataFrame, n_h: int = 2, h_values: Optional[Sequence[int]] = None
) -> pd.DataFrame:
    """每个 (ts_code, trade_date) 组确定性抽取 n_h 个 h（跨组覆盖全部期限）。

    抽样规则（预登记）：按组键的稳定哈希（pandas hash_pandas_object，
    指定固定分类顺序）决定起始偏移，从升序 h 网格中等距取 n_h 个值——
    不同组落在不同偏移上，跨组整体覆盖全部 h；同组的多 h 行随 T 同
    阶段（方案 5.3），不随机拆散。

    Args:
        matrix: 训练矩阵
        n_h: 每组保留的 h 数量
        h_values: 可用期限网格（默认取矩阵内全部 h）

    Returns:
        抽样后的矩阵副本
    """
    if n_h <= 0:
        raise ValueError(f"n_h 必须为正: {n_h}")
    grid = sorted(h_values) if h_values else sorted(matrix["h"].unique().tolist())
    if len(grid) <= n_h:
        return matrix.copy()
    picks = np.linspace(0, len(grid) - 1, n_h).round().astype(int)
    n_g = len(grid)
    # keep_table[offset, h]：该偏移的组保留哪些 h
    keep_table = np.zeros((n_g, int(max(grid)) + 1), dtype=bool)
    for off in range(n_g):
        for p in picks:
            keep_table[off, grid[(p + off) % n_g]] = True

    group_keys = matrix["ts_code"].str.cat(matrix["trade_date"], sep="|")
    offsets = pd.util.hash_pandas_object(group_keys, index=False).to_numpy() % n_g
    h_arr = matrix["h"].to_numpy(dtype=int)
    keep = keep_table[offsets, h_arr]
    return matrix[keep].reset_index(drop=True)


def subsample_dates(matrix: pd.DataFrame, every_n: int = 1) -> pd.DataFrame:
    """按交易日序列位置等距抽样（every_n=3 即每 3 个交易日取 1）。

    抽样按日期位置而非日期字符串哈希，保证同日全截面整组保留或剔除
    （截面完整性优先，不截半截面破坏 pct 语义之外的组结构）。
    """
    if every_n <= 1:
        return matrix.copy()
    unique_dates = sorted(matrix["trade_date"].unique())
    kept_dates = set(unique_dates[::every_n])
    return matrix[matrix["trade_date"].isin(kept_dates)].reset_index(drop=True)
