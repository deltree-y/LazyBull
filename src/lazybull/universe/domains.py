# -*- coding: utf-8 -*-
"""选股域定义（单一来源）——训练侧 / OOS 评估侧 / OOS 回测侧 / 推理侧共用。

C 臂（域扩展 A/B，预登记 ``docs/plans/domain_expansion_ab_prereg.md``）的域开关。
依据 Phase 0 D5 迁移性探针（2026-09-24）：主板 25-50 亿段迁移 RankIC 0.1369 高于
训练域对照 0.1021、创业板 ≥50 亿段 0.1142，两段 8/8 年全正。

硬契约：
- **域是超参签名维度**（``--stock-domain``，入 ``train_params`` 与 summary 列），
  禁止跨取值并组比较；
- **markets 同时作用于训练股票池（``training_core`` 域过滤）与回测 universe**；
  **市值上下限只作用于 MLSignal 推理过滤**（训练侧维持"宽域学排序"的架构语义——
  与现状一致：现状训练 = 纯主板全市值，推理 = 50~1500 亿）；
- 由上推论：``main_small`` 的训练池与 ``main`` 相同（主板）⇒ 小市值臂可 skip-training
  复用生产模型；``main_gem`` 的训练池扩展（+创业板）⇒ 创业板臂**必须全折重训**；
- 成交额下限（5000 万）与剔金融**不随域变化**（MLSignal 既有过滤，所有域一致）；
- 新增域必须在此登记（含 label 与依据），禁止在调用侧散落硬编码。
"""

from typing import Dict, List

#: 域注册表。市值单位 = 万元（与 MLSignal ``min_total_mv/max_total_mv`` 同口径）。
STOCK_DOMAINS: Dict[str, Dict[str, object]] = {
    "main": {
        "markets": ["主板"],
        "min_total_mv_wan": 500000.0,   # 50 亿
        "max_total_mv_wan": 15000000.0,  # 1500 亿
        "label": "主板50-1500亿（生产现状）",
    },
    "main_small": {
        "markets": ["主板"],
        "min_total_mv_wan": 250000.0,   # 25 亿（C1：小市值段放开，D5 RankIC 0.1369）
        "max_total_mv_wan": 15000000.0,
        "label": "主板25-1500亿（C1 小市值臂）",
    },
    "main_gem": {
        "markets": ["主板", "创业板"],
        "min_total_mv_wan": 500000.0,
        "max_total_mv_wan": 15000000.0,
        "label": "主板+创业板≥50亿（C2 创业板臂，D5 RankIC 0.1142）",
    },
}


def resolve_stock_domain(name: str) -> Dict[str, object]:
    """解析域名为域配置；未知取值直接报错（fail-fast，禁止静默回退 main）。

    Args:
        name: 域名（``main`` / ``main_small`` / ``main_gem``）

    Returns:
        域配置字典（``markets`` / ``min_total_mv_wan`` / ``max_total_mv_wan`` / ``label``）
    """
    key = str(name or "").strip()
    if key not in STOCK_DOMAINS:
        raise ValueError(
            f"未知选股域: {name!r}；已登记域: {sorted(STOCK_DOMAINS)}（域是超参签名维度，"
            f"新增域必须先在 universe/domains.py 登记并预登记实验）"
        )
    return STOCK_DOMAINS[key]


def domain_market_whitelist(name: str) -> List[str]:
    """域的市场白名单（训练股票池与回测 universe 共用）。"""
    return list(resolve_stock_domain(name)["markets"])  # type: ignore[arg-type]


def domain_training_pool_unchanged(name: str) -> bool:
    """该域的训练股票池是否与 ``main``（生产现状）相同。

    ``main_small`` 只放开推理侧市值下限（训练本就含 25-50 亿主板股）⇒ True；
    ``main_gem`` 训练池扩展（+创业板）⇒ False，必须全折重训。
    """
    return domain_market_whitelist(name) == domain_market_whitelist("main")
