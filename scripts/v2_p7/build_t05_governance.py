# -*- coding: utf-8 -*-
"""P7 T0.5 治理工件生成器（确定性、可复算、不联网、不触生产）。

产出（默认 `data/paper_shadow/p7_testbed/p7_t05_20261111/`）：

- `prereg/b0_composite_prereg_v1.json`：B0 单一候选的复合预登记（P=1、总 α=0.05）；
- `governance/user_authorization.json`：用户 R3 授权快照；
- `governance/budget_ledger.json`：逐 task 上限与累计上限；
- `governance/write_whitelist.json`：精确写白名单与禁改面；
- `governance/h_report_schema.json`：H 报告 schema 与 kill criteria；
- `governance/governance_snapshot.json`：上述文件与预检证据、契约草案、起点文本的 SHA 汇总。

约束
====
- 只写 `data/paper_shadow/p7_testbed/<governance_id>/`，不写 frozen H 根、不写生产根；
- 规范化 JSON（sort_keys + UTF-8 + 缩进 2 + allow_nan=False），无墙钟时间，重复运行逐字节一致；
- 生成物状态一律为 `draft_pending_review`：T0.5 生效须经规定评审与用户确认。

用法
====
``python scripts/v2_p7/build_t05_governance.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WriteScopeError,
    assert_targets_within_write_roots,
    resolve_write_roots,
    sha256_bytes,
    sha256_file as _sha256_file,
    write_json as _write_json,
)
from crash_spec import CrashSpecError, load_spec  # noqa: E402
from prereg_checks import assert_prereg_consistency  # noqa: E402,F401

_REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE_ID = "p7_t05_20261111_r6"
DEFAULT_ROOT_REL = f"data/paper_shadow/p7_testbed/{GOVERNANCE_ID}"
CONTRACT_DATE = "2026-11-11"

CONTRACT_DRAFT_REL = "docs/plans/v2/p7_t05_contract_revisions_20261111.md"
RUNBOOK_REL = "docs/plans/v2/p7_t05_runbook_20261111.md"
DESIGN_REL = "docs/plans/v2/p7_design_20261111.md"
PLAN_REL = "docs/plans/v2/p7_implementation_plan_20261111.md"
ADJUDICATION_REL = "docs/plans/v2/p7_adjudication_record_20261111.md"
PRECHECK_SUMMARY_NAME = "precheck_summary.json"
CRASH_SPEC_REL = f"{DEFAULT_ROOT_REL}/specs/crash_windows_v6.json"

#: 预检附件集合（缺一不可，且必须与 summary 登记 SHA 逐件一致）
PRECHECK_ATTACHMENTS = (
    "fold_check.json",
    "model_identity.json",
    "panel_coverage.json",
    "phase_calendar.json",
    "source_watermarks.json",
    "event_blocks.json",
)

EFFECTIVE_TEXTS = [
    "docs/contracts/v2_architecture_plan.md",
    "docs/contracts/netting_freeze.md",
    "docs/contracts/protocols.md",
    "docs/contracts/events_state_schema.md",
    "docs/contracts/hypothesis_ledger_schema.md",
]

#: 与 `docs/plans/v2/p7_implementation_plan_20261111.md` §2.1 冻结表同源
FROZEN_FOLDS = [
    (0, 24008, "20121123", "20181123", "20181126", "20190524"),
    (1, 24009, "20130524", "20190524", "20190527", "20191125"),
    (2, 24010, "20131125", "20191125", "20191126", "20200526"),
    (3, 24011, "20140526", "20200526", "20200527", "20201127"),
    (4, 24012, "20141127", "20201127", "20201130", "20210528"),
    (5, 24013, "20150528", "20210528", "20210531", "20211130"),
    (6, 24014, "20151130", "20211130", "20211201", "20220601"),
    (7, 24015, "20160601", "20220601", "20220602", "20221202"),
    (8, 24016, "20161202", "20221202", "20221205", "20230605"),
    (9, 24017, "20170605", "20230605", "20230606", "20231206"),
    (10, 24018, "20171206", "20231206", "20231207", "20240607"),
    (11, 24019, "20180607", "20240607", "20240611", "20241210"),
    (12, 24020, "20181210", "20241210", "20241211", "20250611"),
    (13, 24021, "20190611", "20250611", "20250612", "20260105"),
]

#: Q5=A 冻结的有效块硬下限（唯一来源，禁止从汇总字段取值）
MIN_EFFECTIVE_BLOCKS = 60

EXIT_OK = 0
EXIT_INPUT_GATE_FAIL = 2
EXIT_CORE_GATE_FAIL = 3
EXIT_WRITE_SCOPE = 4
EXIT_HARD_ERROR = 5


def _prereg_core(precheck: Dict[str, Any]) -> Dict[str, Any]:
    """预登记头部、B0 候选定义与数据身份。"""
    cutoff = precheck["common_data_cutoff"]
    return {
        "schema": "p7_b0_composite_prereg/v2",
        "prereg_id": "p7-b0-composite-20261111",
        "evidence_id": precheck.get("evidence_id", GOVERNANCE_ID),
        "status": "draft_pending_review",
        "contract_date": CONTRACT_DATE,
        "alpha_accounting": {
            "candidate_count_P": 1,
            "alpha_total": 0.05,
            "sequential_reads": ["P-1", "T2", "T3"],
            "independent_research_lines": 1,
            "note": "P-1/T2/T3 是同一历史的顺序读取，不重置 α、不称独立验证；B5/B10 择优与事件消融不设",
        },
        "candidate_b0": {
            "sleeve_id": "B0",
            "description": "B 袖最小确定性候选（不是 B 袖全集，也不代表未来所有事件机制）",
            "event_types": ["forecast", "express", "fina_report"],
            "predicates": {
                "forecast": "forecast_type_score > 0（预增/略增/续盈/扭亏）",
                "express": "express_surprise > 0（yoy_net_profit − 同报告期最近一次预告中值，PIT ann_date≤公告日）",
                "fina_report": "or_yoy > 0 且 netprofit_yoy > 0（同日修正版按 update_flag=1 优先）",
            },
            "hold_days": 10,
            "max_concurrent_positions": 6,
            "initial_capital": 195000,
            "min_ticket": {
                "formula": "V = max(25000, min_commission / commission_rate)",
                "budget_reservation": "V + 100P + 预估买入费用（先预留再整手向下）",
            },
            "selection_order": [
                "knowledge_date desc",
                "event_source asc",
                "ts_code asc",
                "event_fingerprint asc",
            ],
            "active_position_rule": (
                "同股已有 B0 活跃持仓时不加仓、不重置买入日；后来事件只登记 "
                "suppressed_active_position，退出后不补入过期事件"
            ),
            "exit_rule": (
                "仅 10 个交易日事件到期与既有组合硬约束执行口径；不新增坏消息/个股分数退出；"
                "负面事件只用于边界与覆盖报告"
            ),
            "revision_policy": (
                "源内版本治理：同源同 (ts_code,end_date,公告日) 至多一个事件，修订全留、按当时可得版本评价；"
                "跨源不做报告期抢占"
            ),
        },
        "data_identity": {
            "common_data_cutoff": cutoff,
            "evaluation_start": precheck["evaluation_window"][0],
            "evaluation_window": precheck["evaluation_window"],
            "watermark_levels": {
                "forecast": "raw/_sync_watermark.json:20261029",
                "express": "raw/_sync_watermark.json:20261029",
                "fina_report": "raw 分区只读扫描 max(ann_date)=20261030（取 min 后 = 20261029）",
                "clean_daily": "20261029",
            },
            "fina_family": {
                "event_type": "fina_report",
                "event_date": "end_date",
                "knowledge_date": "ann_date",
                "revision_key": "(ts_code,end_date,ann_date,update_flag)",
                "revision_policy": "keep_all",
                "dedup_rule": "full_row",
                "same_day_tie_break": "update_flag=1 优先",
                "watermark_rule": "对全部 raw 分区只读扫描有效 ann_date 取最大值；不要求 _sync_watermark.json",
            },
            "common_builder_rule": (
                "T1/T2 必须调用同一只读纯事件构建器并绑定代码哈希；T2 只物化同一事件流，不另写 "
                "ann_date/knowledge_date 映射"
            ),
        },
    }


def _prereg_gates(precheck: Dict[str, Any], crash: Dict[str, Any]) -> Dict[str, Any]:
    """判据：可识别性、Q1 绝对夏普、Q2 利用率、相关、E4、Q5 复合门与 Q3 费用。"""
    core_cells = precheck["core_cells"]
    return {
        "gates": {
            "G1_identifiability": {
                "assignment": "每个匹配组按组 ID 唯一归类",
                "count_identity": (
                    "intent_total = resolved_filled_count + unfilled_group_count "
                    "+ unresolved_count"
                ),
                "zero_return_case": "资格/源链已确认且可证明六股均未建仓 ⇒ 整组收益与费用为真实 0，计入 unfilled_group_count",
                "unresolved_case": "资格、修订链或已持有头寸收益未知 ⇒ 计入 unresolved_count，禁止补 0",
                "freeze_rule": (
                    "任一核心格 unresolved_count>0 ⇒ not_estimable/inconclusive_frozen"
                    "，压力情景不得解除"
                ),
                "warning_only": "u=unresolved/intent 的 5% 仅作数据质量告警分级，不提供放行额度",
                "count_nonconservation": "计数不守恒 ⇒ 拒绝生成经济 verdict",
            },
            "Q1_absolute_sharpe": {
                "decision": "Q1=A",
                "rule": "L_B > Q_0.95(S^0_b)",
                "L_B": {
                    "definition": "B0 独立日收益流按 10 交易日循环块自举的 95% 独立夏普区间下界",
                    "resamples": 10000,
                    "seed": 42,
                    "quantile": 0.025,
                },
                "null_stream": {
                    "definition": (
                        "T2 定义的逐日匹配组内伪事件载荷转移 + 完整 B0 逐股执行组合流程；"
                        "sharpe_abs_null 只取伪事件载荷路径下 B0 组合的绝对日收益流"
                    ),
                    "forbidden": "T2 的 combo_delta_statistic 差额口径不得用于生成 S^0_b",
                    "rf_annual": 0.03,
                    "rf_per_day": "0.03/252",
                    "annualization": 252,
                    "guards": "零波动、非有限值、有效样本量不足 ⇒ 拒绝或冻结，不得记 pass",
                },
                "injection": {
                    "formula": "r'_t = r_t + delta_S * sigma_d / sqrt(252)，sigma_d 为日收益标准差",
                    "delta_S_grid": [0.25, 0.5, 1.0, 2.0],
                    "implied_sharpe_shift": "delta_S",
                    "mde_min_power": 0.8,
                    "grid_exhausted": "网格内未达 80% 功效 ⇒ inconclusive_frozen",
                },
            },
            "Q2_utilization": {
                "decision": "Q2=A",
                "monthly_utilization": "U_m = mean_{t in m}(MV_B,t / TV_B,t)",
                "median_min": 0.5,
                "months_in_denominator": "全部日历月，含零事件月",
                "report": "逐月分布 + 零事件月数",
                "capacity_gate": (
                    "下单预算按 V+100P+预估买入费用预留；实际新仓成交额低于 V、零手数或费用后资金不足"
                    "均记容量失败；理论 195000>=6V 不能替代实际成交检查"
                ),
            },
            "correlation": {
                "three_definitions": ["全样本", "10 日非重叠调仓频", "崩盘段条件相关"],
                "crash_segment_source": "crash_gate.windows（冻结窗口并集；运行期逐字段一致）",
                "upper_bound": 0.3,
                "rule": "三口径区间上界（97.5% 分位）均须 < 0.3",
                "interval_source": "crash_gate.interval.correlation（块长/重采样/种子/最小样本随规格冻结）",
                "insufficient_sample": "not_estimable ⇒ inconclusive_frozen（不补 0、不删窗口）",
            },
            "E4_vetoes": [
                {
                    "id": "E4-1",
                    "rule": crash["decision_rule"],
                    "rule_ref": "crash_gate.decision_rule（唯一来源，禁止在本处重复手写方向）",
                    "window_source": "crash_gate.windows",
                    "tolerance_pp": crash["tolerance_pp"],
                    "interval_source": "crash_gate.interval.e3_paired_delta_mdd",
                },
                {"id": "E4-2", "rule": "三口径相关上界 >= 0.3 即否决"},
                {"id": "E4-3", "rule": "实际成交容量门失败（不足额/零手数/费用后资金不足）即否决"},
                {"id": "E4-4", "rule": "月资金利用率中位数 < 50%（含零事件月）即否决"},
                {
                    "id": "E4-5",
                    "rule": (
                        "坏消息侧越界（滑向 R-002/R-005 个股退出排序）或 PIT 可得性不可归因即否决；"
                        "不得由其他指标兜底"
                    ),
                },
            ],
            "Q5_composite": {
                "decision": "Q5=A",
                "core_cells": [["forecast", 10], ["express", 10], ["fina_report", 10]],
                "sensitivity_cells": [["forecast", 5], ["express", 5], ["fina_report", 5]],
                "sensitivity_role": "仅完整报告敏感性；不得替换 10 日失败、不得择优",
                "precheck_effective_blocks": {
                    name: {
                        "core_10d_effective_blocks": cell["effective_blocks"],
                        "block_years": cell["block_years"],
                        "min_required": 60,
                        "pass": cell["gate_pass"],
                    }
                    for name, cell in core_cells.items()
                },
                "precheck_block_rule": precheck["block_rule"],
                "randomization": {
                    "inner_null": "组内事件标签随机化：每次从六股抽一股为伪事件、其余五股为伪对照，保留协变量/摩擦/价源/公司行为",
                    "block_resampling": (
                        "日期按 h 日 circular moving block 重采样；命名 randomization_block_h"
                    ),
                    "implementations_per_cell": 10000,
                    "seed": 42,
                    "component_quantile": {
                        "formula": "Q_{1-alpha/(3*lambda)}(D^0_{s,10,b})",
                        "nominal_alpha": "0.05/3（λ 冻结后按 0.05/(3λ)）",
                    },
                    "delta_R_grid": [0.5, 1.0, 2.0, 4.0],
                    "delta_R_note": "delta_R 为零假设均值位移（以 sigma(D^0) 为单位），与 Q1 的 delta_S 不同名不同量纲",
                },
                "three_streams": {
                    "calibration": "生成组件阈值；标 calibration_only=true",
                    "selection": {
                        "paths": 1000,
                        "independent_seed_stream": True,
                        "rule": (
                            "对完整顺序门只选一次 lambda ∈ {1,2,4,8}，取使选择流 Clopper-Pearson 单侧95%上界 "
                            "≤0.05 的最小值"
                        ),
                        "freeze": "选定后立即冻结，不得回改",
                    },
                    "verification": {
                        "paths": 1000,
                        "independent_seed_stream": True,
                        "rule": "阈值与 λ 冻结后只运行一次；CP 单侧95%上界 ≤0.05 且独立注入路径最终 MDE 功效 ≥80%",
                        "failure": "验证失败只能 inconclusive_frozen，不得复用验证流调参",
                    },
                    "path_definition": "每条路径运行完整时间轴与全部组件；N 是独立算法零假设路径数，不是历史块数",
                    "independence_caveat": "CP 只覆盖预登记随机化零假设，不声称各历史制度相互独立",
                },
                "composite_boolean": (
                    "FULL_GATE(path;lambda) = 三源核心格组件筛查 ∧ T2 组合级尺子 ∧ Q1 绝对夏普 ∧ 三口径相关 ∧ "
                    "实际容量与利用率 ∧ 五条 E4 否决线均未触发"
                ),
                "lambda_scaling": "所有随机型组件尾概率统一按 0.05/lambda 缩放；确定性门（相关/容量/利用率/否决线）不缩放",
            },
        },
    }


def _prereg_fee() -> Dict[str, Any]:
    """Q3=A：S5 合并卖出的实际费用分摊与有符号节省。"""
    return {
        "Q3_fee_allocation": {
            "decision": "Q3=A",
            "allocation": "f_i = F * a_i / A，A = Σa_i",
            "terminal_sleeve": "按 (a_i 降序, sleeve_id 升序) 首袖为终端，f_t = F − Σ_{i≠t} f_i",
            "constraints": "实际费用非负且 Σf_i = F；A=0 且 F=0 全零；A=0 且 F≠0 拒绝",
            "independent_fee": "按 (sleeve, 实际成交日, ts_code, 方向) 聚合全部实际经济片段后计算一次；跨日残差逐日计费",
            "signed_saving": "s_i = c_i − f_i 允许为负；F>Σc 时照常分摊并报告，不把负节省当负费用",
        }
    }


def _prereg_mechanical(precheck: Dict[str, Any]) -> Dict[str, Any]:
    """机械线（VV-H/VV-S）配置、授权公式、覆盖矩阵与 ΔŪ 语义。"""
    return {
        "pressure_scenarios": {
            "cases": ["事件股 −100% / 对照股 +100%", "反向情景"],
            "field": "stress_scenario_results",
            "diagnostic_only": True,
            "forbidden_uses": "不得作为识别上下界、不得参与 pass/fail、不得解除冻结、不得作为临界值选择依据",
        },
        "mechanical_line": {
            "arms": ["VV-H", "VV-S"],
            "sleeves": {"V20": 0.7, "V10": 0.3},
            "physical_initial_capital": 1000000,
            "virtual_cash": {"V20": 700000, "V10": 300000},
            "independent_nav_denominator": {"V20": 700000, "V10": 300000},
            "contribution_nav_denominator": 1000000,
            "hold_days": 20,
            "stagger_tranches": 2,
            "phase_offset_trading_days": 10,
            "credit_rate": 0,
            "quota_rho": {"V20": 0.7, "V10": 0.3},
            "netting_counters": {
                "reverse_netting": ["S1", "S2", "S3"],
                "same_direction": ["S4", "S5"],
                "separate_fields": True,
                "forbidden": "同向合并量与反向净额量混报、或用同向量解释 B1/B4 语义复核",
            },
            "authorization": {
                "cash_open": "日初已结算可用虚拟现金（含既有借款本金，绝不再加存量 credit）",
                "credit_outstanding": "既有未还本金，仅用于债务/NAV",
                "credit_granted": "本日新分配、尚未转入 cash_open 的借款 L",
                "VV_H": "A^H = max{a <= min(D,R): a + Fee(a) <= cash_open}",
                "VV_S": "A^S(L) = max{a <= min(D,R+L): a + Fee(a) <= cash_open+L}",
                "regular_quota": "R = max(0, rho*G - MV_hold)；原始需求 D = max(0, G - MV_hold)",
                "post_transfer": "cash = cash_open + L；credit = credit_outstanding + L（出借方为负）",
                "forbidden": (
                    "禁止 cash_open + credit_outstanding 或 cash + credit 双重背书；净卖/转账"
                    "预计回款不用于当日买入背书"
                ),
                "settlement_order": "预净额计划（零副作用）→ 授权 → 最终净额 → 仅净市场卖出 → 内部转账与净市场买入 → 一次结算",
            },
            "coverage_matrix": {
                "natural_required": [
                    "S1 等量抵消",
                    "S2/S3 不等量净残差",
                    "S4 同向合并",
                    "S5 合并卖出",
                    "S7 到期结算",
                    "有价抵消",
                    "净残差归属",
                    "集中度截断",
                    "B4 对手买入受阻而卖出完成",
                    "借用",
                    "召回",
                ],
                "natural_zero_observation": "G4=incomplete；不得换相位/换窗/调配额，不得用正常行冒充",
                "constructed_required": [
                    "无价顺延",
                    "全零/部分成交",
                    "尾单",
                    "F>Σ独立费用",
                    "负节省",
                    "部分归还",
                    "死局",
                    "发布前后恢复",
                ],
                "constructed_flag": "constructed_on_history",
                "registration_only": [
                    "自然死局次数",
                    "双向借款比例",
                    "自然负节省次数",
                    "右删失分布",
                ],
            },
            "delta_u": (
                "Delta U = |D|^-1 * Σ_t (U_VVS,t − U_VVH,t)，U = MV/物理总值；cap-b"
                "inding 构造配额路径，不代表独立策略/分散化/容量"
            ),
            "nonpositive_equity": (
                "任一袖 E<=0 ⇒ 禁新增买入与 credit、不被动卖出，聚合 I1/I3 继续审计；独立指标 invalid_no"
                "npositive_equity"
            ),
        },
    }


def _prereg_verdict(precheck: Dict[str, Any]) -> Dict[str, Any]:
    """三向 verdict 真值、H 报告 schema、kill criteria 与禁止项。"""
    return {
        "verdict_rules": {
            "b0_states": ["pass", "fail", "inconclusive_frozen"],
            "mechanical_states": ["pass", "fail", "incomplete"],
            "precedence": "先验可识别性：任一核心格 unresolved_count>0 ⇒ inconclusive_frozen，不由压力情景或其他格经济结果覆盖",
            "pass_requires": (
                "计数守恒且 unresolved_count=0 ∧ 真实逐组收益通过 ∧ 组件与完整复合门通过 ∧ E2~E4 全过 "
                "∧ 用户组织的独立存活评审通过"
            ),
            "fail_requires": "功效足够下的预登记经济门不通过",
            "frozen_requires": "空意图/样本不足/功效不足/独立验证不达标/其他不可判定",
            "frozen_handling": "属非通过：不进 T2/T3/生产/生效提交；未收口可执行实现删除，仅留规格与证据；新 prereg 才可重启",
            "pressure_and_subset": "压力情景与可解析子样本诊断不参与三向判定",
        },
        "report_schema": {
            "evidence_level": "H",
            "historical_virtual_clock": True,
            "not_forward_evidence": True,
            "not_real_clock_evidence": True,
            "not_production_authorization": True,
            "required_fields": [
                "run_id",
                "prereg_id",
                "prereg_sha256",
                "code_state",
                "data_state",
                "raw_clean_panel_watermarks",
                "event_builder_sha256",
                "model_versions_and_windows",
                "cost_calendar_randomization_config",
                "arm_config_diff",
                "enable_borrowing",
                "identifiability_counts",
                "gate_results",
                "coverage_matrix",
                "verdict",
                "written_files_sha256",
            ],
            "forbidden_phrases": [
                "前向验证通过",
                "影子期通过",
                "真实日历验证通过",
                "生产已验证",
                "独立策略分散化",
            ],
        },
        "kill_criteria": {
            "observation_points_months": [3, 6],
            "triggers": [
                "与 A 的三口径相关上界 >= 0.3",
                "崩盘段 ΔMaxDD 恶化（crash_gate 容差 0 + 区间下界规则）",
                "实际容量低于 Q2 口径",
                "PIT 可得性崩塌（新增事件延迟/前视审计越界）",
            ],
            "crash_gate_source": "crash_gate.spec_sha256（窗口/容差/算法随规格冻结，改规格即改身份）",
            "action": "触线即默认关并落退役档案；P7 本身不启用生产开关",
            "calendar_caveat": "本阶段为历史证据（H），不构成任何日历/前向验证结论",
        },
        "forbidden": [
            "B5/B10 择优与事件消融",
            "看结果后删源/改持期/改对照/调阈值",
            "复用 T1 共同槽位事件级临界值作 B0 组合级尺子",
            "把 cp 结论外推为各历史制度独立",
            "在持仓子集内重排 pct_* 或改用 cs_train 作分母",
            "调用 ±10bps 常量路径作为 B 域临界值",
            "在读取任何结果后改崩盘窗、容差或所引用规格",
        ],
    }


def _prereg_crash(crash: Dict[str, Any]) -> Dict[str, Any]:
    """崩盘门（R1-T05-03 处置）：窗口、容差与算法随规格 SHA 一并冻结。"""
    return {
        "crash_gate": {
            "spec_path": crash["spec_path"],
            "spec_sha256": crash["spec_sha256"],
            "basis": crash["basis"],
            "algorithm_module": crash["algorithm_module"],
            "algorithm_module_sha256": crash["algorithm_module_sha256"],
            "source_chain_nav_sha256": crash["source_sha256"],
            "windows": crash["windows"],
            "window_count": len(crash["windows"]),
            "min_windows": 2,
            "tolerance_pp": crash["tolerance_pp"],
            "decision_rule": crash["decision_rule"],
            "correlation_rule": "崩盘段条件相关在 windows 并集交易日上计算，窗口集冻结、禁止按实现重选",
            "coverage": crash.get("coverage") or {},
            "source": crash.get("source") or {},
            "interval": crash.get("interval") or {},
            "delta_sign": "ΔMaxDD = arm − base（signed 非正数；改善为正、pp）",
            "window_union_trading_days": crash.get("window_union_trading_days"),
            "identity_rule": "改窗口/容差/算法模块或引用规格 ⇒ crash_gate 与 prereg 身份同时改变",
            "reselect_forbidden": True,
        }
    }


def _prereg_data_gate(precheck: Dict[str, Any]) -> Dict[str, Any]:
    """输入验收门（R1-T05-02/04）：预检门与逐件附件 SHA 一并进入预登记身份。"""
    return {
        "data_gate": {
            "precheck_evidence_id": precheck.get("evidence_id", GOVERNANCE_ID),
            "precheck_schema": precheck.get("schema"),
            "input_gate_pass": precheck.get("input_gate_pass"),
            "input_gate_checks": precheck.get("input_gate", {}),
            "common_data_cutoff": precheck.get("common_data_cutoff"),
            "calendar_boundary": precheck.get("calendar_boundary"),
            "evaluation_window": precheck.get("evaluation_window"),
            "attachments_sha256": precheck.get("outputs_sha256"),
            "rule": "任一输入检查失败或缺件/字节不符 ⇒ 治理工件不得生成（fail-closed，零副作用）",
        }
    }


def build_prereg(precheck: Dict[str, Any], crash: Dict[str, Any]) -> Dict[str, Any]:
    """B0 单一候选的复合预登记（Q1~Q5 已按入口裁决全部取 A）。"""
    return {
        **_prereg_core(precheck),
        **_prereg_data_gate(precheck),
        **_prereg_crash(crash),
        **_prereg_gates(precheck, crash),
        **_prereg_fee(),
        **_prereg_mechanical(precheck),
        **_prereg_verdict(precheck),
    }


def build_authorization() -> Dict[str, Any]:
    """用户 R3 授权快照（来源、日期、原话摘要、上下限口径、非裁决名额）。"""
    return {
        "schema": "p7_t05_user_authorization/v1",
        "authorization_id": "p7-r3-20261111",
        "authorized_by": "user",
        "authorization_date": CONTRACT_DATE,
        "source": "2026-11-11 本轮 P7 设计任务的用户明确规则（非仓库既有契约、非 P6.5 先例）",
        "original_wording_status": (
            "摘要转述（依据 docs/plans/v2/p7_design_20261111.md §10.1 与 "
            "docs/plans/v2/p7_adjudication_package_20261111.md §3 的落档），非逐字引用；"
            "逐字原话由用户在生效确认时补登"
        ),
        "rule": {
            "coding_unit_range": [3, 5],
            "coding_unit_comparison_baseline": 5,
            "machine_time_reference_hours": 16,
            "machine_time_reference_derivation": "P-1 4h + 数据导入 4h + 四机械臂各 2h（设计参照，非架构固定总值）",
            "auto_pass_multiplier": 2,
            "auto_pass_lines": {"coding_units": 10, "machine_hours": 32},
        },
        "estimate": {
            "coding_units": 9.75,
            "machine_hours": 26.25,
            "coding_ratio_to_upper": 1.95,
            "coding_ratio_to_lower": 3.25,
            "machine_ratio": 1.64,
            "both_within_auto_pass": True,
        },
        "effect": {
            "occupies_adjudication_slot": False,
            "waives_actual_consumption_registration": False,
            "waives_per_task_ceiling": False,
            "waives_long_task_batch_authorization": False,
            "note": "自动通过只表示不新增预算裁决点；实耗按自包含改动会话登记，>20 分钟机器任务仍逐批申请",
        },
        "downgrade_option": {
            "id": "merge_t4_t5_session",
            "saving_coding_units": 0.25,
            "acceptance_items_preserved": True,
            "trigger": "任一 task 预计越线时先启用；仍越线交用户预算裁决，不得删验收项",
        },
    }


def build_budget_ledger() -> Dict[str, Any]:
    """逐 task 编码/机器上限与累计上限（实施计划 §11 数值即硬上限）。"""
    tasks = [
        ("T0.5", "共同治理、复合预登记与输入完整性预检", 1.00, 0.25),
        ("T1", "P-1 同域静态可执行性", 1.25, 4.00),
        ("T2", "事件物化、B0 与组合级尺子首测", 1.50, 4.50),
        ("T2.5", "A 逐日信号再生、相位目标与原始意图预检", 0.75, 3.00),
        ("T3", "A0/B0 单袖历史证据、E2~E4 与完整复合门", 1.25, 8.00),
        ("T4", "P7 专用生命周期、价源、双口径与状态", 1.50, 0.75),
        ("T5", "原始需求、配额、日内净额、借用会计与死局", 1.25, 0.75),
        ("T6", "VV-H/VV-S 两机械臂、综合证据与收口", 1.25, 5.00),
    ]
    return {
        "schema": "p7_t05_budget_ledger/v1",
        "contract_date": CONTRACT_DATE,
        "hard_ceiling_rule": "§11 数值为硬上限；任一 task 预计超限先停该 task 并重估，不得删验收项",
        "tasks": [
            {
                "task": name,
                "scope": scope,
                "coding_units_ceiling": coding,
                "machine_hours_ceiling": machine,
                "actual_coding_units": None,
                "actual_machine_hours": None,
                "session_log": [],
            }
            for name, scope, coding, machine in tasks
        ],
        "totals": {
            "coding_units_ceiling": 9.75,
            "machine_hours_ceiling": 26.25,
            "user_auto_pass_lines": {"coding_units": 10, "machine_hours": 32},
        },
        "t05_consumption_rule": (
            "编码实耗按自包含改动会话登记（起止、文件、测试、task 归属），禁止凭墙钟或行数换算；"
            "T0.5 实耗在生效提交时回填"
        ),
        "long_task_rule": "任一预计 >20 分钟的机器批次执行前单独申请，列单批/总时长与省时备选",
        "stop_rule": "预计越线即停并给降配方案；仍越线交用户裁决",
    }


def build_write_whitelist() -> Dict[str, Any]:
    """精确写白名单、禁改面与生产零变化约束。"""
    return {
        "schema": "p7_t05_write_whitelist/v1",
        "contract_date": CONTRACT_DATE,
        "work_root": f"data/paper_shadow/p7_testbed/{GOVERNANCE_ID}",
        "campaign_root_pattern": "data/paper_shadow/p7_testbed/<campaign_id>/**",
        "h_evidence_root_pattern": "data/frozen_reference/v2_p7/<run_id>/**",
        "events_root": "data/events/<event_type>/YYYY.parquet（仅由冻结事件构建器写）",
        "allowed_write": [
            "data/paper_shadow/p7_testbed/<campaign_id>/**",
            "data/frozen_reference/v2_p7/<run_id>/**（T1 起正式证据，原子发布、禁止覆盖）",
            "data/events/<event_type>/**（T2 事件物化）",
            "temp/**（临时产物）",
            "logs/**（临时日志）",
        ],
        "forbidden_write": [
            "data/paper/**",
            "data/runs/**",
            "data/models/**",
            "data/raw/**",
            "data/clean/**",
            "data/features/**",
            "data/frozen_reference/v2_p1/**",
            "data/frozen_reference/v2_p3/**",
            "data/frozen_reference/v2_p4/**",
            "data/frozen_reference/v2_p5b/**",
            "data/frozen_reference/v2_p65/**",
            "任何既有 run 目录",
        ],
        "frozen_surfaces": {
            "p5b": [
                "src/lazybull/v2/hosts/testbed/paths.py",
                "src/lazybull/v2/hosts/testbed/journal.py",
                "src/lazybull/v2/hosts/testbed/registration.py",
                "src/lazybull/v2/hosts/testbed/family_txn.py",
            ],
            "p65": [
                "src/lazybull/v2/core/decision/virtual_merger.py",
                "src/lazybull/v2/core/accounting/virtual_ledger.py",
                "src/lazybull/v2/core/accounting/reconciliation.py",
                "src/lazybull/v2/hosts/order_chain.py",
                "src/lazybull/v2/hosts/virtual_native_engine.py",
                "src/lazybull/v2/core/accounting/state_envelope.py",
            ],
            "production": [
                "scripts/paper_trade.py",
                "data/paper/config.yaml",
                "回测 runtime 默认装配",
                "P4 EngineState 键集",
            ],
        },
        "production_zero_change_evidence": [
            "既有 merger 调用计数保持预期",
            "生产入口静态 import 闭包不可达任何 P7 新模块",
            "冻结短窗成交逐笔与 NAV 对账一致",
            "旧证据根/raw/clean 前后路径 + SHA 不变",
        ],
        "p5b_role_note": "P7 不继承 calibration_only / not_real_candidate；新建 P7 包装，旧类行为与源文件哈希不变",
    }


def build_h_report_schema() -> Dict[str, Any]:
    """H 报告 schema 与 kill criteria 的机读字段。"""
    return {
        "schema": "p7_h_report_schema/v1",
        "contract_date": CONTRACT_DATE,
        "evidence_level": "H",
        "booleans": {
            "historical_virtual_clock": True,
            "not_forward_evidence": True,
            "not_real_clock_evidence": True,
            "not_production_authorization": True,
        },
        "required_manifest_fields": [
            "run_id",
            "prereg_id",
            "prereg_sha256",
            "code_state",
            "data_state",
            "raw_clean_panel_watermarks",
            "event_builder_sha256",
            "model_versions",
            "fold_windows",
            "cost_config",
            "calendar_config",
            "randomization_config",
            "arm_config_diff",
            "enable_borrowing",
            "files",
        ],
        "required_verdict_fields": [
            "b0_verdict",
            "b0_state_reason",
            "mechanical_verdict",
            "mechanical_state_reason",
            "gate_results",
            "coverage_matrix",
            "kill_criteria_status",
        ],
        "verdict_domain": {
            "b0": ["pass", "fail", "inconclusive_frozen"],
            "mechanical": ["pass", "fail", "incomplete"],
        },
        "forbidden_report_phrases": [
            "前向验证通过",
            "影子期通过",
            "真实日历验证通过",
            "生产已验证",
            "独立策略分散化收益",
            "容量提升证据",
        ],
        "allowed_phrasing": "历史证据下通过独立存活评审（证据等级 H）",
        "kill_criteria": {
            "observation_points_months": [3, 6],
            "machine_readable": True,
            "triggers": [
                "correlation_upper_bound>=0.3",
                "crash_segment_delta_mdd_worse",
                "actual_capacity_below_gate",
                "pit_availability_collapse",
            ],
            "action": "default_off_and_retire",
        },
    }


def build_snapshot(
    root: Path,
    precheck: Dict[str, Any],
    crash: Dict[str, Any],
    precheck_dir: Path,
    source_shas: Dict[str, str],
    effective_shas: Dict[str, str],
) -> Dict[str, Any]:
    """治理快照：各工件 SHA、契约草案与起点生效文本 SHA、禁改面摘要。"""
    files = {
        "prereg": "prereg/b0_composite_prereg_v1.json",
        "user_authorization": "governance/user_authorization.json",
        "budget_ledger": "governance/budget_ledger.json",
        "write_whitelist": "governance/write_whitelist.json",
        "h_report_schema": "governance/h_report_schema.json",
    }
    artifact_shas: Dict[str, Optional[str]] = {}
    for key, rel in files.items():
        payload_path = root / rel
        artifact_shas[key] = _sha256_file(payload_path) if payload_path.exists() else None
    artifact_shas["precheck_summary"] = _sha256_file(precheck_dir / PRECHECK_SUMMARY_NAME)
    artifact_shas["precheck_outputs"] = precheck["outputs_sha256"]
    artifact_shas["crash_spec"] = crash["spec_sha256"]

    return {
        "schema": "p7_t05_governance_snapshot/v2",
        "governance_id": GOVERNANCE_ID,
        "contract_date": CONTRACT_DATE,
        "status": "draft_pending_review",
        "effective_on": None,
        "effective_condition": "规定评审通过 + 用户确认后，T0.5 独立生效提交（F13）",
        "adjudication": {
            "record": ADJUDICATION_REL,
            "choices": {"Q1": "A", "Q2": "A", "Q3": "A", "Q4": "A", "Q5": "A"},
            "review_fixes": {
                "R1-T05-01": "日历截断至共同截止后重取（evidence_id 变更）",
                "R1-T05-02": "输入验收门 fail-closed，预检/治理双入口非零退出",
                "R1-T05-03": "崩盘窗规格冻结并绑定 SHA；容差 0 + 区间下界规则",
                "R1-T05-04": "六件预检附件逐件字节校验，缺件/不符即拒绝",
                "R1-T05-05": "两个 CLI 执行写域守卫，禁写面零副作用拒绝",
                "R1-T05-06": "相位第二档改为整体后移 10 日并独立集合校验；runoff 要求恰好 20 日",
            },
        },
        "input_gate": {
            "pass": bool(precheck.get("input_gate_pass")),
            "checks": precheck.get("input_gate", {}),
            "attachments_verified": sorted(PRECHECK_ATTACHMENTS),
            "attachment_sha256": precheck["outputs_sha256"],
        },
        "precheck_gate": {
            "core_gate_all_pass": precheck["core_gate_all_pass"],
            "min_effective_blocks": precheck["min_effective_blocks"],
            "core_cells": precheck["core_cells"],
            "fina_core_cell_pass": precheck["core_cells"]["fina_report"]["gate_pass"],
            "calendar_boundary": precheck.get("calendar_boundary"),
            "note": "Q5=A 的 fina 核心格不足 60 块须停止并回用户重裁；本次通过，且无静默改档",
        },
        "crash_gate": {
            "spec_sha256": crash["spec_sha256"],
            "basis": crash["basis"],
            "windows": crash["windows"],
            "tolerance_pp": crash["tolerance_pp"],
        },
        "artifact_sha256": artifact_shas,
        "source_sha256": source_shas,
        "effective_text_baseline_sha256": effective_shas,
        "next_effective_texts": {
            "architecture": "F26（下一版）",
            "netting": "F26（下一版）",
            "protocols": "F14（下一版）",
            "events": "F4（下一版）",
            "hypothesis_ledger": "F7（下一版）",
        },
        "forbidden_to_modify": {
            "p5b_sources": "见 write_whitelist.frozen_surfaces.p5b",
            "p65_sources": "见 write_whitelist.frozen_surfaces.p65",
            "production": "见 write_whitelist.frozen_surfaces.production",
        },
        "open_actions_at_effective_time": [
            "把契约草案文本并入五份契约并登记 F 行（架构 F26 / 净额化 F26 / 协议 F14 / events F4 / 台账 F7）",
            "按 DataStore 写入 B0 复合 prereg 台账条目并锁定 α=0.05（登记即锁定）",
            "回填 T0.5 实耗编码/机器时间到 budget_ledger",
            "刷新 governance_snapshot 的 effective_on 与实际生效文本 SHA",
        ],
    }


def _read_verified_attachments(precheck_dir: Path, registered: Dict[str, str]) -> Dict[str, Any]:
    """逐件读取附件字节、核对登记 SHA，并从**同一缓冲**解析。"""
    parsed: Dict[str, Any] = {}
    for name in PRECHECK_ATTACHMENTS:
        path = precheck_dir / name
        if not path.exists():
            raise FileNotFoundError(f"预检附件缺失：{path}")
        data = path.read_bytes()
        expected = registered.get(name)
        if expected is None:
            raise ValueError(f"预检汇总未登记附件 SHA：{name}")
        actual = sha256_bytes(data)
        if actual != expected:
            raise ValueError(f"预检附件字节不符：{name} 实际 {actual} != 登记 {expected}")
        parsed[name] = json.loads(data.decode("utf-8"))
    return parsed


REQUIRED_GATE_CHECKS = (
    "folds_match_frozen_table",
    "fold_count_is_14",
    "single_data_state",
    "model_versions_match_frozen",
    "test_domain_contiguous",
    "models_identity_match",
    "signal_days_complete",
    "phase_offset_matches",
    "runoff_complete",
    "fina_watermark_derivable",
    "calendar_truncated_at_cutoff",
    "calendar_covers_evaluation",
)

REQUIRED_SOURCES = ("forecast", "express", "fina_report")


def _require_true(value: Any, name: str) -> None:
    """严格布尔：字符串 "false"/1 之类的真值一律拒绝。"""
    if not isinstance(value, bool) or value is not True:
        raise ValueError(f"`{name}` 必须为布尔 true，实际 {value!r}")


def _require_gate_shape(summary: Dict[str, Any]) -> None:
    """门字段必须是严格布尔且键集精确，总 pass 与 failed_checks 必须自洽。"""
    gate = summary.get("input_gate")
    if not isinstance(gate, dict):
        raise ValueError("summary.input_gate 缺失或类型非法")
    missing = [name for name in REQUIRED_GATE_CHECKS if name not in gate]
    extra = [name for name in gate if name not in REQUIRED_GATE_CHECKS]
    if missing or extra:
        raise ValueError(f"summary.input_gate 键集不符：缺 {missing} / 多 {extra}")
    for name in REQUIRED_GATE_CHECKS:
        _require_true(gate[name], f"input_gate.{name}")
    _require_true(summary.get("input_gate_pass"), "input_gate_pass")
    if summary.get("input_gate_failed_checks") != []:
        failed = summary.get("input_gate_failed_checks")
        raise ValueError(f"input_gate_failed_checks 必须为空列表，实际 {failed!r}")


def _require_fold_facts(folds: Dict[str, Any]) -> None:
    """折表事实：冻结一致、14 折、模型序列与数据态唯一。"""
    _require_true(folds.get("frozen_folds_match"), "fold_check.frozen_folds_match")
    _require_true(folds.get("test_domain_contiguous"), "fold_check.test_domain_contiguous")
    row_count = len(folds.get("rows") or [])
    if int(folds.get("fold_count", -1)) != row_count or row_count != len(FROZEN_FOLDS):
        raise ValueError(f"fold_count/rows 不符：{folds.get('fold_count')} vs {row_count}")
    if list(folds.get("model_versions") or []) != [model for _, model, *_ in FROZEN_FOLDS]:
        raise ValueError("fold_check.model_versions 与冻结模型序列不一致")
    data_states = folds.get("data_state_ids") or []
    if len(data_states) != 1 or not data_states[0]:
        raise ValueError(f"data_state_ids 必须唯一非空，实际 {data_states!r}")


def _require_mirror_flags(summary: Dict[str, Any], parsed: Dict[str, Any]) -> None:
    """汇总的 6 个镜像布尔必须与附件事实同一对象相等（双向矛盾都拒绝）。"""
    pairs = [
        ("folds_match_frozen_table", parsed["fold_check.json"].get("frozen_folds_match")),
        ("test_domain_contiguous", parsed["fold_check.json"].get("test_domain_contiguous")),
        ("models_identity_match", parsed["model_identity.json"].get("identity_match")),
        ("signal_days_complete", parsed["panel_coverage.json"].get("signal_days_complete")),
        ("phase_offset_matches", parsed["phase_calendar.json"].get("phase_offset_matches")),
        (
            "fina_watermark_derivable",
            parsed["source_watermarks.json"].get("fina_watermark_derivable"),
        ),
    ]
    for name, actual in pairs:
        if summary.get(name) is not actual:
            raise ValueError(
                f"summary.{name} 与附件事实不一致：{summary.get(name)!r} vs {actual!r}"
            )


def _require_model_panel_phase(parsed: Dict[str, Any]) -> None:
    """模型/面板/相位/水位四类附件事实必须为真。"""
    models = parsed["model_identity.json"]
    panel = parsed["panel_coverage.json"]
    phase = parsed["phase_calendar.json"]
    watermarks = parsed["source_watermarks.json"]
    _require_true(models.get("identity_match"), "model_identity.identity_match")
    if (models.get("mismatch") or []) != []:
        raise ValueError(f"model_identity.mismatch 非空：{models.get('mismatch')}")
    _require_true(panel.get("signal_days_complete"), "panel_coverage.signal_days_complete")
    _require_true(phase.get("phase_offset_matches"), "phase_calendar.phase_offset_matches")
    _require_true(phase.get("runoff_complete"), "phase_calendar.runoff_complete")
    _require_true(phase.get("v10_shifted_matches_v20"), "phase_calendar.v10_shifted_matches_v20")
    _require_true(phase.get("v10_no_wraparound"), "phase_calendar.v10_no_wraparound")
    runoff_days = int(phase.get("runoff_trading_days", -1))
    if runoff_days != int(phase.get("runoff_required_trading_days", -2)):
        raise ValueError("runoff 天数与要求不一致")
    _require_true(
        watermarks.get("fina_watermark_derivable"), "source_watermarks.fina_watermark_derivable"
    )


def _require_calendar_facts(summary: Dict[str, Any], parsed: Dict[str, Any]) -> None:
    """共同截止三方一致，且日历截断与评价窗终点落在截止内。"""
    events = parsed["event_blocks.json"]
    cutoff = str(parsed["source_watermarks.json"].get("common_data_cutoff"))
    if cutoff != str(summary.get("common_data_cutoff")):
        raise ValueError("source_watermarks 与 summary 的共同截止不一致")
    boundary = summary.get("calendar_boundary") or {}
    _require_true(boundary.get("truncated_at_cutoff"), "calendar_boundary.truncated_at_cutoff")
    if str(boundary.get("bounded_last_date")) != cutoff:
        raise ValueError("日历截断末值与共同截止不一致")
    _require_true(
        events.get("calendar_bounded_by_cutoff"), "event_blocks.calendar_bounded_by_cutoff"
    )
    if str(events.get("common_data_cutoff")) != cutoff:
        raise ValueError("event_blocks 与 summary 的共同截止不一致")
    if str(events.get("evaluation_end")) > cutoff:
        raise ValueError("评价窗终点晚于共同截止")


def _require_frozen_minimum(summary: Dict[str, Any], events: Dict[str, Any]) -> None:
    """60 块硬下限取自冻结常量，汇总与每格的登记值都必须等于它（R2R-02）。"""
    declared = summary.get("min_effective_blocks")
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise ValueError(f"min_effective_blocks 必须为整数，实际 {declared!r}")
    if declared != MIN_EFFECTIVE_BLOCKS:
        raise ValueError(f"min_effective_blocks 被改为 {declared}，冻结值为 {MIN_EFFECTIVE_BLOCKS}")
    for name, cell in (events.get("cells") or {}).items():
        table = cell.get("core_10d") or {}
        cell_minimum = table.get("min_effective_blocks")
        if not isinstance(cell_minimum, int) or isinstance(cell_minimum, bool):
            raise ValueError(f"event_blocks.{name}.min_effective_blocks 必须为整数")
        if cell_minimum != MIN_EFFECTIVE_BLOCKS:
            raise ValueError(
                f"event_blocks.{name}.min_effective_blocks={cell_minimum} != {MIN_EFFECTIVE_BLOCKS}"
            )


def _require_core_cells(summary: Dict[str, Any], events: Dict[str, Any]) -> None:
    """核心三格：pass、块数与冻结下限必须与 event_blocks 重建值一致。"""
    _require_true(summary.get("core_gate_all_pass"), "core_gate_all_pass")
    _require_frozen_minimum(summary, events)
    cells = summary.get("core_cells") or {}
    if sorted(cells) != sorted(REQUIRED_SOURCES):
        raise ValueError(f"core_cells 必须恰为 {REQUIRED_SOURCES}，实际 {sorted(cells)}")
    event_cells = events.get("cells") or {}
    for name in REQUIRED_SOURCES:
        cell = cells[name]
        blocks = cell.get("effective_blocks")
        if not isinstance(blocks, int) or isinstance(blocks, bool):
            raise ValueError(f"core_cells.{name}.effective_blocks 必须为整数")
        if blocks < MIN_EFFECTIVE_BLOCKS:
            raise ValueError(f"{name} 有效块 {blocks} < 冻结下限 {MIN_EFFECTIVE_BLOCKS}")
        _require_true(cell.get("gate_pass"), f"core_cells.{name}.gate_pass")
        rebuilt = event_cells.get(name, {}).get("core_10d", {})
        if int(rebuilt.get("effective_blocks", -2)) != blocks:
            raise ValueError(f"{name} 汇总块数与 event_blocks 重建值不一致")
        expected_pass = blocks >= MIN_EFFECTIVE_BLOCKS
        if rebuilt.get("gate_pass") is not expected_pass:
            raise ValueError(f"event_blocks.{name}.gate_pass 与冻结下限判定不一致")
        _require_true(rebuilt.get("gate_pass"), f"event_blocks.{name}.gate_pass")


def _rebuild_input_gate(summary: Dict[str, Any], parsed: Dict[str, Any]) -> None:
    """从六件附件**重建**输入验收门，逐项必须为真；不信任汇总 pass 位（R1R-01）。"""
    _require_gate_shape(summary)
    _require_fold_facts(parsed["fold_check.json"])
    _require_mirror_flags(summary, parsed)
    _require_model_panel_phase(parsed)
    _require_calendar_facts(summary, parsed)
    _require_core_cells(summary, parsed["event_blocks.json"])


def load_precheck_inputs(precheck_dir: Path) -> Dict[str, Any]:
    """校验预检汇总与六件附件（缺件/字节不符/语义不一致/门不一致即拒绝）。

    Raises:
        FileNotFoundError / ValueError: 缺件、字节不符、schema 不符或语义不一致。
    """
    summary_path = precheck_dir / PRECHECK_SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少预检汇总：{summary_path}")
    summary = json.loads(summary_path.read_bytes().decode("utf-8"))
    if summary.get("schema") != "p7_t05_precheck/v2":
        raise ValueError(f"预检 schema 不受支持：{summary.get('schema')}（须 v2）")
    parsed = _read_verified_attachments(precheck_dir, summary.get("outputs_sha256") or {})
    _rebuild_input_gate(summary, parsed)
    summary["_precheck_dir"] = str(precheck_dir)
    return summary


def _require_sources() -> Dict[str, str]:
    """必需来源文档与生效文本必须存在（不得写成 null 或静默略去）。"""
    required = {
        "contract_revision_draft": CONTRACT_DRAFT_REL,
        "runbook": RUNBOOK_REL,
        "design": DESIGN_REL,
        "implementation_plan": PLAN_REL,
        "adjudication_record": ADJUDICATION_REL,
    }
    missing = [f"{name}:{rel}" for name, rel in required.items() if not (_REPO_ROOT / rel).exists()]
    missing += [rel for rel in EFFECTIVE_TEXTS if not (_REPO_ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(f"必需来源缺失，拒绝生成：{missing}")
    return {name: _sha256_file(_REPO_ROOT / rel) for name, rel in required.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P7 T0.5 治理工件生成器（确定性）")
    parser.add_argument("--root", default=DEFAULT_ROOT_REL, help="治理快照根目录")
    parser.add_argument(
        "--precheck-dir",
        default=None,
        help="预检证据目录（默认 <root>/precheck；不接受静默回退到其他根）",
    )
    parser.add_argument("--crash-spec", default=CRASH_SPEC_REL, help="崩盘窗规格（已冻结）")
    parser.add_argument(
        "--allow-root", action="append", default=[], help="追加允许写根（测试用，生产不得追加）"
    )
    args = parser.parse_args(argv)

    root = Path(args.root) if Path(args.root).is_absolute() else _REPO_ROOT / args.root
    roots = resolve_write_roots(args.allow_root)
    targets = {
        "prereg": root / "prereg" / "b0_composite_prereg_v1.json",
        "user_authorization": root / "governance" / "user_authorization.json",
        "budget_ledger": root / "governance" / "budget_ledger.json",
        "write_whitelist": root / "governance" / "write_whitelist.json",
        "h_report_schema": root / "governance" / "h_report_schema.json",
        "governance_snapshot": root / "governance" / "governance_snapshot.json",
    }
    try:
        resolved = assert_targets_within_write_roots(list(targets.values()), roots)
        targets = dict(zip(targets.keys(), resolved))
    except WriteScopeError as error:
        print(f"写域拒绝：{error}", file=sys.stderr)
        return EXIT_WRITE_SCOPE
    root = targets["prereg"].parent.parent
    precheck_dir = (Path(args.precheck_dir) if args.precheck_dir else root / "precheck").resolve()
    if args.precheck_dir and not precheck_dir.exists():
        print(f"预检目录不存在：{precheck_dir}", file=sys.stderr)
        return EXIT_HARD_ERROR

    try:
        precheck = load_precheck_inputs(precheck_dir)
        crash = load_spec(_REPO_ROOT / args.crash_spec)
        source_shas = _require_sources()
    except (FileNotFoundError, ValueError, CrashSpecError) as error:
        print(f"输入校验失败，未写出任何文件：{error}", file=sys.stderr)
        return EXIT_INPUT_GATE_FAIL
    effective_shas = {rel: _sha256_file(_REPO_ROOT / rel) for rel in EFFECTIVE_TEXTS}

    prereg = build_prereg(precheck, crash)
    assert_prereg_consistency(prereg)
    prereg_sha = _write_json(targets["prereg"], prereg)
    written = json.loads(targets["prereg"].read_bytes().decode("utf-8"))
    assert_prereg_consistency(written)
    auth_sha = _write_json(targets["user_authorization"], build_authorization())
    budget_sha = _write_json(targets["budget_ledger"], build_budget_ledger())
    white_sha = _write_json(targets["write_whitelist"], build_write_whitelist())
    h_sha = _write_json(targets["h_report_schema"], build_h_report_schema())

    snapshot = build_snapshot(root, precheck, crash, precheck_dir, source_shas, effective_shas)
    snapshot_sha = _write_json(targets["governance_snapshot"], snapshot)

    print(f"governance_root={root}")
    print(f"precheck_dir={precheck_dir}")
    print(f"prereg_sha256={prereg_sha}")
    print(f"user_authorization_sha256={auth_sha}")
    print(f"budget_ledger_sha256={budget_sha}")
    print(f"write_whitelist_sha256={white_sha}")
    print(f"h_report_schema_sha256={h_sha}")
    print(f"governance_snapshot_sha256={snapshot_sha}")
    print(f"status={snapshot['status']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
