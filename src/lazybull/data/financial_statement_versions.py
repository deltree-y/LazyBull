"""财务报表版本行去重工具。"""

from typing import Sequence

import pandas as pd

CASHFLOW_VERSION_DEDUP_COLS = ("ts_code", "end_date", "f_ann_date")
CASHFLOW_VERSIONED_RAW_WATERMARK = "cashflow_revision_full_refresh"


def deduplicate_prefer_latest_update_flag(
    df: pd.DataFrame,
    dedup_cols: Sequence[str],
    *,
    deterministic_ties: bool = False,
) -> pd.DataFrame:
    """按键去重，同键冲突时优先保留 TuShare 标记的最新行。

    TuShare 财务报表接口约定 ``update_flag=1`` 表示当前最新记录。同一实际
    公告日没有更细的修订时间可用于 PIT 排序。现金流版本键或显式启用确定性
    决胜时，官方标志相同或缺失则按全行内容哈希选择，避免结果随行序变化；
    其他调用保留原有 keep-last 语义。
    """
    if df is None or len(df) == 0:
        return df

    deterministic_ties = deterministic_ties or tuple(dedup_cols) == CASHFLOW_VERSION_DEDUP_COLS
    dedup_cols = list(dedup_cols)
    missing_cols = [col for col in dedup_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"财务报表版本去重缺少键列: {', '.join(missing_cols)}")

    work = df.copy()
    content_cols = sorted(work.columns)
    sort_cols = list(dedup_cols)
    helper_cols = []
    if "update_flag" in work.columns:
        helper_suffix = ""
        while f"_update_flag_latest{helper_suffix}" in work.columns:
            helper_suffix += "_"
        update_flag_col = f"_update_flag_latest{helper_suffix}"
        work[update_flag_col] = pd.to_numeric(work["update_flag"], errors="coerce").eq(1)
        sort_cols.append(update_flag_col)
        helper_cols.append(update_flag_col)

    if deterministic_ties:
        helper_suffix = ""
        while f"_row_content_hash{helper_suffix}" in work.columns:
            helper_suffix += "_"
        row_hash_col = f"_row_content_hash{helper_suffix}"
        work[row_hash_col] = pd.util.hash_pandas_object(work[content_cols], index=False).to_numpy()
        sort_cols.append(row_hash_col)
        helper_cols.append(row_hash_col)

    work = work.sort_values(sort_cols, kind="mergesort")
    work = work.drop_duplicates(subset=dedup_cols, keep="last")
    return work.drop(columns=helper_cols).reset_index(drop=True)
