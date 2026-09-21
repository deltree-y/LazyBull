"""A5 路由型下行风险惩罚（downside penalty）

在不新增训练列、不触碰模型列集的前提下，对当日候选排序做一次后处理：

    score' = pct(score) − λ × risk_pct

- 两项均为**同一截面内**的百分位（`rank(pct=True)`，并列取平均，值域 (0, 1]）；
- 风险方向**单处定义**（`DOWNSIDE_PENALTY_COLUMNS`）：`downside_vol_20` 越大越危险；
  `cvar_95_20` 为负值列，越负越危险（risk = −列值）；
- 风险分位缺失按截面中位 `RISK_NAN_FILL` 处理并计数（禁止静默丢弃）；
- λ=0 时调用方必须走 **no-op 分支**（本模块拒绝非正 λ），保证与基线逐位一致；
- 列缺失 / 未登记列 / 非法参数一律 `ValueError` 报错，禁止静默降级。

母截面口径（两处，禁止混用结论）：
- 执行侧（`MLSignal.generate_ranked` / `generate`）：当日**候选域**（universe ∩ 选股过滤后）；
- OOS 评估明细侧（`split_training.evaluate_test_window`）：**评估域**（与既有信号层尺子同域）。

预登记（口径/网格/判据）：`docs/plans/downside_penalty_prereg.md`；
契约条目：CLAUDE.md「低波惩罚契约（A5）」。
"""

from typing import Dict, Optional, Tuple

import pandas as pd

# 允许列 → 风险方向符号（risk = 符号 × 列值；风险越大惩罚越高）
DOWNSIDE_PENALTY_COLUMNS: Dict[str, int] = {
    "downside_vol_20": 1,
    "cvar_95_20": -1,
}

# 冻结 λ 网格（超参签名维度；CLI choices 强制，禁止事后扩展）
DOWNSIDE_PENALTY_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5)

# 风险/分数缺失时的分位填充值（截面中位语义）
RISK_NAN_FILL = 0.5


def apply_downside_penalty(
    features_df: pd.DataFrame,
    *,
    score_column: str,
    risk_column: str,
    penalty: float,
    date_column: Optional[str] = None,
) -> Dict[str, int]:
    """在原地把 ``score_column`` 调整为「分位(score) − λ × 风险分位」。

    Args:
        features_df: 待调整的截面数据（原地修改 ``score_column``）。
        score_column: 模型分数列；调整后为分位分数（值域约 [−λ, 1−λ]）。
        risk_column: 风险列；必须在 ``DOWNSIDE_PENALTY_COLUMNS`` 登记。
        penalty: λ（必须 > 0；λ=0 由调用方 no-op，不走本函数）。
        date_column: 为 None 时按单一截面计算（引擎侧单日）；
            否则按该列分组逐日计算（OOS 评估明细侧）。

    Returns:
        {"rows": 行数, "days": 截面数, "risk_nan_rows": 风险缺失行数}

    Raises:
        ValueError: λ 非法、列未登记或列缺失（禁止静默跳过）。
    """
    if penalty is None or penalty <= 0:
        raise ValueError(f"下行风险惩罚 λ 必须为正数（λ=0 是调用方 no-op 分支），当前值: {penalty}")
    if risk_column not in DOWNSIDE_PENALTY_COLUMNS:
        raise ValueError(
            f"未登记的下行风险列: {risk_column}；允许值: {sorted(DOWNSIDE_PENALTY_COLUMNS)}"
        )
    for col in (score_column, risk_column):
        if col not in features_df.columns:
            raise ValueError(
                f"下行风险惩罚缺少列 `{col}`（{len(features_df)} 行）；"
                "不得静默跳过，请检查特征构建/运行时派生的完整性"
            )
    if date_column is not None and date_column not in features_df.columns:
        raise ValueError(f"下行风险惩罚缺少日期列 `{date_column}`（按日分组口径）")

    sign = float(DOWNSIDE_PENALTY_COLUMNS[risk_column])
    risk = features_df[risk_column].astype(float) * sign
    score = features_df[score_column].astype(float)

    if date_column is None:
        risk_pct = risk.rank(pct=True, method="average")
        score_pct = score.rank(pct=True, method="average")
        days = 1
    else:
        groups = features_df[date_column]
        risk_pct = risk.groupby(groups, sort=False).rank(pct=True, method="average")
        score_pct = score.groupby(groups, sort=False).rank(pct=True, method="average")
        days = int(groups.nunique())

    risk_nan_rows = int(risk_pct.isna().sum())
    risk_pct = risk_pct.fillna(RISK_NAN_FILL)
    score_pct = score_pct.fillna(RISK_NAN_FILL)

    features_df[score_column] = score_pct - float(penalty) * risk_pct
    return {"rows": int(len(features_df)), "days": days, "risk_nan_rows": risk_nan_rows}
