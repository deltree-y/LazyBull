# -*- coding: utf-8 -*-
"""v2 panel 对账归因登记唯一来源（无第三方依赖，供对账器与诊断脚本共用）。"""

from __future__ import annotations

COMPARE_ATOL = 1e-6
GATE_OVER_SHARE = 1e-4
GATE_MAX_ABS_DIFF = 0.05

#: D-04①：share_float/block_trade/top_list 水位晚于基线数据态（直接列）
_D04_1_EXACT = {"days_to_unlock", "unlock_ratio", "unlock_risk_flag"}
_D04_1_PREFIXES = ("block_discount_", "lhb_")
#: D-04②：raw daily 2024-01-16/17 丢停牌股行（等权市场序列微移）
_D04_2_EXACT = {"downside_corr_20"}
#: D-04③：季频财务/基金持仓 raw 修订（NaN 型差异，单元 2 标定实证集）
_D04_3_EXACT = {"dividend_payout_ratio", "cf_nm", "ocf_to_profit"}
_D04_3_PREFIXES = ("fund_",)
#: D-04④：margin_detail raw 分区修订（丢行；单元 3 探测实证 2026-04-17 分区缺行）
_D04_4_EXACT = {"rzye_chg_5", "rzye_chg_20", "short_balance_change_5", "short_sell_vol_change_5"}
#: D-05：pandas 3.0 rolling.skew/kurt 回归（环境漂移，已修复；残留差异 = 异常待查）
_D05_EXACT = {"skewness_20", "kurtosis_20"}

_D04_1_TEXT = "D-04① share_float/block_trade/top_list 水位晚于基线数据态"
_D04_2_TEXT = "D-04② raw daily 2024-01-16/17 分区丢行（等权市场序列微移）"
_D04_3_TEXT = "D-04③ 季频财务/基金持仓 raw 修订"
_D04_4_TEXT = "D-04④ margin_detail raw 分区修订（丢行）"
_D05_TEXT = "D-05 pandas 3.0 rolling.skew/kurt 回归（已修复，残留=异常）"
PENDING_ATTRIBUTION = "待调查"

#: L1/L2 血缘差异日边界（单元 3 报告 §4a；单元 4 复核更正：L2 实为参照侧污染，见 D-09）
L1_BOUNDARY_END = "20121231"
L2_RANGE = ("20130101", "20151231")
#: D-04② 登记事件日窗口（2024-01-16/17 丢行 + 20 日窗口余波）
D04_2_RANGE = ("20240101", "20240331")
#: L1 长尾末（与 L1_BOUNDARY_END 同值，语义别名：长窗列 EMA/250 日长尾自然延伸至 2012 年底）
_L1_TAIL_END = L1_BOUNDARY_END
#: 长窗/EMA/风控族的 L1 长尾延伸边界（250 日窗口与 EMA 链自 2012 冷启动成熟需 ~1.5 年）
L1_LONG_TAIL_END = "20130630"
_LONG_WINDOW_PREFIXES = (
    "mkt_",
    "macd_",
    "vol_regime",
    "turnover_percentile",
    "skewness",
    "kurtosis",
    "downside_",
    "var_95",
    "cvar_95",
    "zscore_macd",
    "volatility_",
    "parkinson",
    "garch_",
    "vol_of_vol",
    "vol_ratio_5_20",
    "amihud",
    "atr_",
    "bb_",
)
#: 数据态孤立单点的判定阈值：单日超门行占比 ≤1% 且期外累计 ≤0.1%
_SCATTER_DAY_SHARE = 0.01
_SCATTER_TOTAL_SHARE = 0.001

#: D-12 裁决豁免列族（2026-10-06 用户裁决，冻结文档 §8 D-13）：panel 解锁前瞻列
#: 经全量 share_float（2005~2026 全分区）+ 全日历重建查询表重写，语义 = PIT 完整；
#: 参照/cs_train 保持旧链区间截断行为（行为冻结不动）⇒ 严格门/归因门对该族从
#: 「逐值一致」改为「豁免 + 差异全归因登记」（参照缺 ann≤2010 分区与末年分区的内容差）。
D12_EXEMPT_COLUMNS = frozenset({"days_to_unlock", "unlock_ratio", "unlock_risk_flag"})
_D12_TEXT = "D-12 公告前瞻截断豁免（裁决 2026-10-06：panel 全量 PIT vs 旧链区间截断）"

#: D-11 分块锚定影响窗（严格门专用）：panel 回填分块覆盖段起点 20170103/20220104，
#: 批量缓存锚定 120 交易日长尾 ≈ 6 个自然月；复牌股伴随事件（实证 002260.SZ 2022-05
#: 复牌 ⇒ EMA 链几何收敛 0.8/日，持续 ~4 个月）再留 3 个月余量 ⇒ 边界 +9 个月；
#: 参照为单调用锚定 20120104。窗口内长记忆列（EMA/MA250/滚动族及其 zscore/neu 派生）
#: 及稀疏伴随单点差异属预期锚定效应。
_D11_WINDOWS = (("20170103", "20170831"), ("20220104", "20220930"))
_D11_TEXT = "D-11 分块锚定效应（panel 分块锚 20170103/20220104 vs 参照单调用锚 20120104）"
#: D-11 稀疏伴随单点分支文本（仅此分支允许归因门 fallback；主分支不得由 fallback 放行）
D11_SPARSE_COMPANION_TEXT = _D11_TEXT + "（伴随稀疏单点）"


def is_d11_sparse_companion_class(klass: str) -> bool:
    """判定 strict 分类结果是否为 D-11 稀疏伴随单点分支。

    归因门 fallback 只允许该结构化分支；禁止用文本前缀把 D-11 主分支（长记忆
    易感列窗内任意密度）一并放行（R2-T5AE-01）。
    """
    return klass == D11_SPARSE_COMPANION_TEXT


#: 纯数据态孤立单点判定阈值（严格门专用）：全日稀疏（每日 ≤1%）+ 总量 ≤20 行
#: + 日数 ≤10 + max|Δ| ≤ 0.05（超幅单点一律调查，不登记）
_SCATTER_PURE_MAX_DAYS = 10
_SCATTER_PURE_MAX_ROWS = 20

#: 结构合法性：CYQ5 物化列白名单（参照侧 cyq_perf 数据态差异；实证恰这 5 列 ×
#: 20120104~20171229 落在 cols_only_panel）
CYQ5_COLUMNS = frozenset(
    {
        "cost_concentration",
        "weight_avg_bias",
        "winner_rate",
        "winner_rate_chg_5",
        "winner_rate_chg_20",
    }
)
#: CYQ5 登记差异末日上界（实证 last=20171229）
CYQ5_LAST_DAY = "20171229"
#: 2012 冷启动 schema 演进边界：cols_only_panel / cols_only_ref_days 登记日 ≤ 此 ⇒ 合法
SCHEMA_EVOLUTION_END = "20121231"

#: D-04③/④ 登记差异窗（实证 first=20161228；上界 20260702 = 终版参照构建数据态水位）
D04_3_4_RANGE = ("20161201", "20260702")
#: D-04① 差异日下界（水位类差异只会出现在接近参照构建数据态的近期窗口；
#: lhb/block_discount 终版零差异——本规则对存量无约束、对未来漂移 fail-closed）
D04_1_DIFF_START = "20260101"
#: D-11 锚定分支超幅上限（实证窗内最大 kurtosis_20=2.326；窗内任意密度放行、超幅拒）
D11_ANCHOR_MAX_ABS = 5.0

#: D-14 专项三上限（天数 / 累计 over / max|Δ|，越任一 ⇒ 越界；括注终版产物实证值）
_D14_SKEW_MAX_LATE_DAYS = 250  # 实证 194
_D14_SKEW_MAX_LATE_OVER = 40000  # 实证总 33355
_D14_SKEW_MAX_ABS = 5.0  # 实证 3.121
_D14_KURT_MAX_LATE_DAYS = 750  # 实证 658
_D14_KURT_MAX_LATE_OVER = 40000  # 实证总 34492
_D14_KURT_MAX_ABS = 40.0  # 实证 35.13（2015 极端行情簇）
_D14_ZSCORE_MACD_MAX_DAYS = 500  # 实证 410 / 413
_D14_ZSCORE_MACD_MAX_OVER = 600000  # 实证 348151 / 471326
_D14_ZSCORE_MACD_MAX_ABS = 20.0  # 实证 14.28 / 15.29
_D14_DOWNSIDE_CORR_MAX_DAYS = 200  # 实证 147
_D14_DOWNSIDE_CORR_MAX_OVER = 300000  # 实证 255697
_D14_DOWNSIDE_CORR_MAX_ABS = 1.5  # 实证 1.116

#: D-14 日级密度与专项日期边界（与三上限共同 fail-closed）
_D14_SKEW_KURT_MAX_DAY_ROWS = 25
_D14_SKEW_KURT_MAX_DAY_SHARE = 0.015
_D14_SPARSE_MAX_DAY_ROWS = 5
_D14_ZSCORE_MACD_EMA_TAIL_END = "20130930"
_D14_DOWNSIDE_CORR_LINEAGE_END = "20131031"

D14_SKEW_KURT_COLUMNS = frozenset({"skewness_20", "kurtosis_20"})
D14_ZSCORE_MACD_COLUMNS = frozenset({"zscore_macd_hist", "zscore_macd_hist_sz"})
D14_DOWNSIDE_CORR_COLUMNS = frozenset({"downside_corr_20"})
ATTRIB_DIAG_COLUMNS = (
    "downside_corr_20",
    "kurtosis_20",
    "skewness_20",
    "zscore_macd_hist",
    "zscore_macd_hist_sz",
)

D14_SKEW_KURT_TEXT = (
    "L1 + D-05 实现对偶数值尾（rolling vs 中心矩实现的边界股翻转，"
    "逐日 ≤25 股且 ≤1.5%，聚 2015 停牌簇，受天数/累计/超幅三上限约束；现役模型不含）"
)
D14_ZSCORE_MACD_TEXT = (
    "L1 macd-EMA 冷启动长尾（至 2013-09）+ D-11 分块锚定（2022-05 复牌簇）"
    "+ 稀疏孤立单点（受天数/累计/超幅三上限约束）"
)
D14_DOWNSIDE_CORR_TEXT = (
    "L1 + 2013-09/10 cs_train 构建环境血缘块（pandas 2.x vs 3.0.5 滚动聚合"
    "行为差推断）+ D-04② + 稀疏孤立单点（受天数/累计/超幅三上限约束）"
)
L1_TEXT = "L1 2012 构建血缘（冷启动零预热/schema 演进）"
L1_SCATTER_TEXT = "L1 血缘 + 数据态孤立单点（期外每日 ≤1% 且累计 ≤0.1%）"
ATTRIB_VIOLATION_TEXT = "越界（第六类/日期越出登记边界，须调查）"
STRICT_SCATTER_TEXT = "数据态孤立单点（纯：全日稀疏微量，单日 ≤1% 且总量 ≤20 行）"
STRICT_UNATTRIBUTED_TEXT = "未归因（差异模式越出登记类边界，实现漂移嫌疑，须调查）"


def derived_parent(col: str) -> str | None:
    """zscore_X / zscore_X_sz / neu_X 派生列的母列名（非派生列返回 None）。"""
    for prefix in ("zscore_", "neu_"):
        if col.startswith(prefix):
            plen = len(prefix)
            parent = col[plen:]
            if parent.endswith("_sz"):
                parent = parent[: -len("_sz")]
            return parent
    return None


def _attribute_base(col: str) -> str | None:
    """基础列归因（未命中返回 None）。"""
    if col in _D04_1_EXACT or col.startswith(_D04_1_PREFIXES):
        return _D04_1_TEXT
    if col in _D04_2_EXACT:
        return _D04_2_TEXT
    if col in _D04_3_EXACT or col.startswith(_D04_3_PREFIXES):
        return _D04_3_TEXT
    if col in _D04_4_EXACT:
        return _D04_4_TEXT
    if col in _D05_EXACT:
        return _D05_TEXT
    return None


def attribute_column(col: str) -> str:
    """列归因：基础列查 D-04 清单；zscore/neu 子列随母列归因。"""
    base = _attribute_base(col)
    if base is not None:
        return base
    parent = derived_parent(col)
    if parent is not None:
        parent_attr = _attribute_base(parent)
        if parent_attr is not None:
            return f"{parent_attr}（zscore/neu 派生连锁）"
    return PENDING_ATTRIBUTION
