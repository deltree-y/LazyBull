"""申万行业合并 —— 从 builder.py 拆出的 _merge_shenwan_industry()。

将申万行业分类信息合并到特征 DataFrame。
支持 L3 新格式（优先）和 L2 旧格式（兼容）。
"""

from typing import Optional

import pandas as pd
from loguru import logger


def merge_shenwan_industry(
    features: pd.DataFrame,
    shenwan_industry: pd.DataFrame,
    shenwan_level: str = "l2",
    verbose: bool = False,
) -> pd.DataFrame:
    """合并申万行业分类信息（主行业口径由项目配置决定）。

    新式 L3 格式（优先）：产出 sw_industry/sw_industry_code/sw_industry_id
    以及 sw_l1/sw_l2/sw_l3 各级字段。
    旧式 L2 格式（兼容）：产出 sw_industry/sw_industry_code/sw_industry_id。
    """
    from ..factors.industry import generate_industry_encoding

    if shenwan_industry is None or len(shenwan_industry) == 0:
        logger.warning("申万行业分类数据为空，跳过合并")
        return features

    sw_cols = shenwan_industry.columns.tolist()
    is_l3_format = "sw_l3_code" in sw_cols or "sw_l3" in sw_cols

    if is_l3_format:
        return _merge_l3_format(features, shenwan_industry, sw_cols, shenwan_level, verbose)
    return _merge_l2_format(features, shenwan_industry, sw_cols)


def _merge_l3_format(
    features: pd.DataFrame,
    shenwan_industry: pd.DataFrame,
    sw_cols: list,
    shenwan_level: str,
    verbose: bool,
) -> pd.DataFrame:
    from ..factors.industry import generate_industry_encoding

    l3_cols = [
        col
        for col in ["sw_l3_code", "sw_l3", "sw_l2_code", "sw_l2", "sw_l1_code", "sw_l1", "in_date"]
        if col in sw_cols
    ]
    merge_cols = ["ts_code"] + l3_cols
    existing_merge_cols = [col for col in merge_cols if col in sw_cols]

    if len(existing_merge_cols) < 2:
        logger.warning(f"申万 L3 数据缺少必要字段，现有列：{sw_cols}")
        return features

    result = features.merge(shenwan_industry[existing_merge_cols], on="ts_code", how="left")

    level_to_name_col = {"l1": "sw_l1", "l2": "sw_l2", "l3": "sw_l3"}
    level_to_code_col = {"l1": "sw_l1_code", "l2": "sw_l2_code", "l3": "sw_l3_code"}
    fallback_levels = {"l1": ["l2", "l3"], "l2": ["l3", "l1"], "l3": ["l2", "l1"]}

    selected_name_col = level_to_name_col[shenwan_level]
    selected_code_col = level_to_code_col[shenwan_level]

    if (
        selected_name_col not in result.columns
        or not result[selected_name_col].notna().any()
        or selected_code_col not in result.columns
        or not result[selected_code_col].notna().any()
    ):
        for fallback_level in fallback_levels[shenwan_level]:
            f_name_col = level_to_name_col[fallback_level]
            f_code_col = level_to_code_col[fallback_level]
            if (
                f_name_col in result.columns
                and result[f_name_col].notna().any()
                and f_code_col in result.columns
                and result[f_code_col].notna().any()
            ):
                logger.warning(
                    f"申万行业数据缺少 {selected_name_col}/{selected_code_col}，"
                    f"sw_industry 临时回退到 {fallback_level}"
                )
                selected_name_col = f_name_col
                selected_code_col = f_code_col
                break

    if selected_code_col in result.columns and result[selected_code_col].notna().any():
        result["sw_industry_code"] = result[selected_code_col]
    if selected_name_col in result.columns and result[selected_name_col].notna().any():
        result["sw_industry"] = result[selected_name_col]

    if "sw_industry" in result.columns:
        id_dict = generate_industry_encoding(result["sw_industry"])
        result["sw_industry_id"] = result["sw_industry"].map(id_dict)
    if "sw_l2" in result.columns:
        id_dict_l2 = generate_industry_encoding(result["sw_l2"])
        result["sw_l2_id"] = result["sw_l2"].map(id_dict_l2)
    if "sw_l1" in result.columns:
        id_dict_l1 = generate_industry_encoding(result["sw_l1"])
        result["sw_l1_id"] = result["sw_l1"].map(id_dict_l1)

    if verbose:
        industry_counts = result.get("sw_industry", pd.Series()).value_counts()
        logger.info(f"申万{shenwan_level.upper()}主行业分布（前5）：\n{industry_counts.head()}")

    return result


def _merge_l2_format(
    features: pd.DataFrame,
    shenwan_industry: pd.DataFrame,
    sw_cols: list,
) -> pd.DataFrame:
    from ..factors.industry import generate_industry_encoding

    industry_cols = ["ts_code", "sw_code", "sw_name"]
    existing_cols = [col for col in industry_cols if col in sw_cols]

    if len(existing_cols) < 2:
        logger.warning(f"申万行业数据缺少必要字段，现有列：{sw_cols}")
        return features

    result = features.merge(shenwan_industry[existing_cols], on="ts_code", how="left")

    rename_map = {}
    if "sw_name" in result.columns:
        rename_map["sw_name"] = "sw_industry"
    if "sw_code" in result.columns:
        rename_map["sw_code"] = "sw_industry_code"
    if rename_map:
        result = result.rename(columns=rename_map)

    if "sw_industry" in result.columns:
        id_dict = generate_industry_encoding(result["sw_industry"])
        result["sw_industry_id"] = result["sw_industry"].map(id_dict)

    return result
