# -*- coding: utf-8 -*-
"""ensure 子包：申万行业分类自动下载。"""

from typing import Dict, Optional

import pandas as pd
from loguru import logger

from ...data import DataCleaner, Storage, TushareClient


def _ensure_shenwan_industry(
    client: TushareClient,
    storage: Storage,
    cleaner: DataCleaner,
) -> Optional[pd.DataFrame]:
    """自动下载申万三级行业分类数据

    逻辑与 scripts/update_basic_data.py 中 update_shenwan_industry() 一致，
    但集成到 ensure 链路中，纸面交易可自动触发。

    Returns:
        申万行业分类 DataFrame，或 None（失败时）
    """
    try:
        # 1. 获取申万三级行业指数列表
        logger.info("获取申万三级行业指数列表...")
        index_classify = client.get_index_classify(level="L3", src="SW2021")
        if index_classify is None or len(index_classify) == 0:
            logger.warning("未获取到申万三级行业指数")
            return None

        if "index_code" not in index_classify.columns:
            logger.warning("index_classify 缺少 index_code 字段")
            return None

        sw_l3_indices = index_classify
        logger.info(f"获取到 {len(sw_l3_indices)} 个申万三级指数")

        # 2. 逐个获取成分股
        logger.info("获取各三级行业成分股...")
        index_members: Dict[str, pd.DataFrame] = {}
        success_count = 0

        for _, row in sw_l3_indices.iterrows():
            index_code = row["index_code"]
            try:
                members = client.get_index_member(l3_code=index_code)
                if len(members) > 0:
                    index_members[index_code] = members
                    success_count += 1
            except Exception as e:
                logger.debug(f"获取 {index_code} 成分股失败: {e}")

        logger.info(f"成功获取 {success_count}/{len(sw_l3_indices)} 个三级行业成分股")

        if success_count == 0:
            logger.warning("未获取到任何行业成分股数据")
            return None

        # 3. 清洗并保存
        clean_data = cleaner.clean_shenwan_industry(
            sw_l3_indices, index_members, level_str="l3"
        )
        if len(clean_data) == 0:
            logger.warning("申万行业清洗后无有效数据")
            return None

        storage.save_raw(clean_data, "shenwan_industry", is_force=True)
        logger.info(f"申万行业分类已自动下载: {len(clean_data)} 条映射")
        return clean_data

    except Exception as e:
        logger.warning(f"自动下载申万行业分类失败: {e}")
        return None