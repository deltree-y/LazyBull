# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主核心（PaperHost）：T1 内核接入纸面场景的宿主。

模块边界：本模块持 ``PaperHost`` 单日原子驱动本体与 ``PaperDayResult``；装配工厂
``create_paper_host`` 在同包 ``factory.py``，半提交恢复职责族（完整代核对重建/幂等记录
补写）在同包 ``recovery.py``（``PaperHostRecoveryMixin``，经 mixin 继承，实例方法面不变）；
指令投影纯函数族在同包 ``instruction_projection.py``（经 state_store re-export）。
拆分均因行数硬上限（§7.4「新增功能对应新增文件」；行为不变，导入面经包
``__init__`` re-export 不变）。

形态总述（D1/D2）
=================
纸面一日 = **一次完整 ``run_step(date)``**（P1~P8 全相位，引擎语义即新契约）。
``run_t0`` 与 ``run_t1`` 是同一日驱动的两个对外视图（协议
``common/protocols/hosts.py::PaperHost`` 两方法）：**先被调用者驱动当日，后被调用者
幂等读产物**；``_run_daily`` 是唯一驱动入口。每日新建引擎实例、不跨日持有
（**单进程单引擎**——loguru sink 串扰面，§17.7 硬项⑧；两臂 = 独立进程）。

相位顺序对齐（§17.7 硬项②）
===========================
引擎 ``run_step`` 相位 = P1 延迟订单 → P2 止损 → P3 信号 → P4 执行（止损卖 → 条件卖 →
生成卖出 → 买入）→ P5 提前调仓 → P6 补齐 → P7 估值 → P8 收口（§17.3 注记）；宿主
**不重排**，run_t0/run_t1 共享同一原子 run_step。与旧链「先 T1 执行、后止损检查」的
已知差异面登记（成交/净值零影响，归影子 holding_path/execution_backfill 桶）；
**T5 门 5 须以旧纸面全链实证等价，不得只复述**（遗留义务，登记）。

run_t1 顺序（§7 测试矩阵「run_t1 顺序」行）
==========================================
退役政策（exposure，Q5）形态 = null ⇒ 空操作：``_evaluate_policy_phase`` 是 run_t1
前置相位的空操作实现，**顺序断言保留**（政策评估先于 T1 执行，经 ``_phase_trace``
留痕供测试断言）。

fills 形态（批量入账接缝，§17.7 硬项③；2026-10-28 用户裁决：引擎行情记账为权威）
====================================================================================
``run_t1(date, fills=None)``：fills=None ⇒ **引擎行情记账为权威**（真实行情记账语义，
与旧纸面链一致——旧链无人工回填入口，T1 按行情模拟成交）。fills 非空 ⇒ **提交前核对**
通道：与引擎当日撮合结果逐笔比对（恒等键 = ``(ts_code, action, shares, price)``，价格
float 严格相等）：一致 ⇒ 正常提交（引擎已入账，不重复入账，D2）；不一致 ⇒
**零副作用拒绝**（PaperHostError；核对在 commit 之前，当日未落盘，可修正 fills 后重试）。
同日已提交后的 fills 核对（幂等读路径）：不一致 ⇒ PaperHostError 告警（当日已在此前
提交，无法回滚，需人工核查）。人工回填**直接驱动入账**的形态不在纸面宿主（纸面 =
模拟盘，无真实成交来源）——登记为将来 live 实盘宿主阶段义务（落地规则成文：先
``export_state`` 持当日快照再批量 ``Ledger.apply_fills``，两阶段预检零副作用；状态前置
校验失败时前序笔已入账 ⇒ 以快照受控重放）。协议 ``protocols.md`` F11 §4「人工回填」
措辞与实际语义的偏差经 2026-10-28 用户裁决以本实现为准（契约勘误登记见规划 §18.10）。

同日幂等与半提交恢复（§17.7 硬项⑦；2026-10-28 T2 评审处置修订）
==================================================================
**单写入者整日写锁**：``_run_daily`` 全程持 ``store.write_lock()``（进入即取锁，锁内读取
权威末日并分派；冲突 fail-fast ``PaperLockError``——不等待）。``commit_day`` 附带权威代
比较交换守卫（``expected_base_date`` 与盘上权威末日不一致 ⇒ 拒绝过期提交）。
每次驱动/读取前先执行 ``_ensure_generation_complete(权威末日)``：当日全部投影都是权威
快照的确定性函数（account 族经 store ``check_consistency`` 双状态校验；trades/nav 当日行、
T+1 指令帧、pending 两队经宿主核对重建——快照经 ⑤ 定稿修订保留当日 trades 作重建原料），
与盘上内容一致 ⇒ 零写盘；缺失/不一致 ⇒ 以权威快照为准重建（告警留痕；trades/nav 当日行
先删后写，恰一次）。此后：
权威「最后处理日」（blob portfolio_values 末日）= 当日 ⇒ 当日已驱动：
- run 幂等记录齐 ⇒ **幂等返回既有指令投影**（读 instructions 投影重构 Order 序列
  + ``already_run`` 标记），**不重复执行**；
- 记录缺失 ⇒ **补写记录、不重跑**（投影完整已由上一步保证）。

就绪门与零未来行情
==================
``assemble_daily_data`` 就绪门 ``ok=False`` ⇒ 告警 + 缺口登记（``runs/gap_<date>.json``
+ log）⇒ **不生成信号、不落盘**（禁止静默用陈旧数据）。指令文件层零未来行情断言
（§16.6 T2 增项）：行情窗口含 ``trade_date > 当日`` 行 ⇒ fail-fast。

启动形态（首次 vs 恢复）
========================
- 无权威快照且无账户投影（``check_consistency`` = "fresh"）⇒ 全新引擎 +
  ``init_live_anchor``（**首次启动专用**；恢复启动**禁止重锚**——锚随快照恢复）；
- 有权威快照 ⇒ 先 ``check_consistency(auto_repair=True)``（滞后 ⇒ 受控重投影 + 告警；
  冲突 ⇒ PaperStateError 拒绝），再 ``restore_state``（fail-fast 上冒）；
- 无权威但有旧链投影（"legacy_projection"，T5 迁移前形态）⇒ PaperHostError 拒绝
  （禁止以全新账户覆盖旧链遗产布局；先执行 T5 迁移/回滚程序）。

live 排期异常兜底确认（§17.7 硬项⑥）
====================================
``apply_live_policy`` 的 fail-open（``except Exception`` ⇒ 当日放行 tranche0）覆盖第 3/4
分支（自锚重建 + K=1 实际成功日计数 / K>1 计划日窗口+漏批追赶判定）**整段任意异常**
——忠实复刻旧链兜底语义；「尚无成功」（分支 1）与「配置提案」（分支 2）的构表异常
**上冒**——宿主**不吞**：这两类异常上冒 = 当日失败告警（人介入），不静默重试。本模块
不对 ``apply_live_policy`` / ``run_step`` / ``settle_live_day`` 加任何 try/except。

投影决策登记（行为冻结：只搬运与投影，不重算决策数值）
=====================================================
- nav 跳空语义保持：nav_row 仅当日有成交才供给（旧链 ``_record_nav`` 触发口径）；
  nav 经账本层唯一入口 ``v2/core/accounting/nav.py::calculate_nav`` 派生（=
  ``total_value / initial_capital``，旧 ``runner/pricing.py:95`` 同式；T2 评审处置：
  投影层禁止自算（D3），原 ``if initial > 0 else 0.0`` 非正初始资金兜底随之移除——
  初始资金 >0 为宿主构造前置，差异登记）。
- trades 投影 ``commission/stamp_tax/slippage`` 分解：v2 成交记录只存 ``cost``
  总额，旧 schema 三列由同一 ``CostModel`` 按成交额确定性分解（``commission +
  stamp_tax + slippage`` 与 ``cost`` 逐笔校验 |Δ| ≤ 1e-6，不一致 ⇒ PaperHostError——
  分解是展示投影而非决策数值，成本模型配置漂移即检出）。
- trades 投影 ``reason``：卖出 ← ``sell_reason`` 或 ``sell_type`` 文本映射；买入 ←
  ``buy_reason``（加仓路径）或恒「信号生成」。差异登记：v2 成交记录不携带延迟/补位
  订单的原因文本，此类成交 reason 投影为「信号生成」（精确原因在指令文件与 pending
  队列投影中）。
- pending_buys/pending_sells 投影：v2 ``PendingOrder`` 无 ``target_weight`` /
  ``last_attempt_date`` 概念 ⇒ 恒 0.0 / ""（差异登记；旧读侧对缺省值兼容）。
- 指令帧 ``buy_price_type`` 恒 "close"（D7 指纹，config_adapter 已 fail-fast）；
  ``sell_price_type`` = ``sell_price``；``existing`` 合并 = 既有 T+1 文件优先
  （``store.load_instructions_frame`` 读回作 existing，复刻旧链合并语义）。
- T0 指令投影的 ``total_capital`` = 当日（T0）组合估值（T+1 实际买入预算以引擎执行
  时估值为准——指令股数为计划投影，真实成交由引擎记账）；``capital_fraction`` =
  分批资金比例（``stagger_tranches > 1`` 时经引擎 ``_get_tranche_capital_fraction``
  同口径，否则 1.0）。边界登记：满仓权重（Σw=1）下末笔买入受 T+1 顺序成交的现金
  约束缩量（``buy_execution`` 资金不足分支按可用现金降手），指令投影股数与真实
  成交可相差成本对应的手数（成本率 ~1.3‰ 下 ≦ 2 手/笔量级）——投影/执行语义边界，
  测试以「首笔逐位一致 + 缩量方向与上限」锁定。
- ``_run_total_days``（进度日志分母，纯展示字段）= 已处理日数 + 1（快照
  portfolio_values 长度承载；纸面逐日驱动无窗口概念，登记为展示口径）。

Order 投影（run_t0 返回面）
===========================
指令帧行 → ``v2.common.types.Order``（``source_sleeve="value"`` 单袖）：
``target_shares`` = 成交后期望持仓（买 = 现持仓 + 指令股数；卖 = 现持仓 − 指令股数，
下限 0）；``reason`` 由指令 reason 文本映射 OrderReason（止损/到期/调仓/补位，
兜底 = 买 REBALANCE / 卖 CONDITION_SELL）。登记：``OrderType`` 闭集仅
``MARKET_OPEN``，买入 close 口径无法在该枚举表达——价格口径由指令文件
``price_type`` 列承载，Order 层恒取 MARKET_OPEN（类型层缺口，登记）。

推送文本登记
============
同包 ``reporting.py`` 迁移件的指令格式化函数依赖旧 ``PaperTradingRunner`` /
``PaperStorage``（生产根读取），v2 宿主隔离根场景不可用——推送文本由
``_format_push_text`` 自指令帧本地格式化（措辞风格对齐旧「卖X笔 买Y笔」形态）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from loguru import logger

# 只读过渡依赖（E4 登记在册）：旧数据面 Storage/DataLoader（clean/daily 行情窗口、
# 引擎停牌日历数据源）。
from src.lazybull.data import DataLoader, Storage

# 只读永久沿用（§3.5 登记表）：选股域与 universe 不迁移。
from src.lazybull.universe import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.protocols.core import PortfolioMerger
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.common.types import (
    Fill,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TradeDate,
    TSCode,
)
from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    export_state,
    loads_blob,
    restore_state,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.execution.live_schedule import settle_live_day
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML
from src.lazybull.v2.hosts.paper.config_adapter import (
    ENGINE_V2,
    PaperHostError,
)
from src.lazybull.v2.hosts.paper.daily_assembly import (
    DailyAssembly,
    DerivedProducer,
    ReadinessReport,
    assemble_daily_data,
)
from src.lazybull.v2.hosts.paper.recovery import PaperHostRecoveryMixin
from src.lazybull.v2.hosts.paper.schedule import (
    configure_engine_schedule,
    initialize_anchor,
    prepare_daily_schedule,
    project_rebalance_state,
    validate_schedule,
    wire_day_inputs,
)
from src.lazybull.v2.hosts.paper.instruction_projection import (
    inject_instruction_shares,
    resolve_t0_status,
)
from src.lazybull.v2.hosts.paper.state_store import (
    GENERATION_INFO_KEY,
    PaperStateStore,
    build_account_projection,
    build_instruction_frame,
    build_ranked_candidates_projection,
    build_stop_loss_projection,
    build_strategy_state_projection,
)
from src.lazybull.v2.store.data_store import PanelDataStore

__all__ = [
    "PaperHost",
    "PaperHostError",
    "PaperDayResult",
    "SLEEVE_ID",
]

#: 单袖袖子 ID（D6；与 hosts/backtest/runtime.py 的 MERGER_SLEEVE_ID 同值同源语义）
SLEEVE_ID = "value"

#: 成本分解一致性容差（trades 投影 commission+stamp_tax+slippage vs cost）
_COST_SPLIT_TOLERANCE = 1e-6

#: 相位留痕上限（bot 长驻进程防爆内存；超上限裁前半段）
_PHASE_TRACE_MAX = 4096

#: 卖出 sell_type → 旧链指令/成交 reason 文本（entry 自带 sell_reason 时优先）
_SELL_TYPE_REASONS: Dict[str, str] = {
    "rebalance": "调仓卖出",
    "holding_period": "持有期到期",
    "stop_loss": "止损卖出",
}


@dataclass
class PaperDayResult:
    """纸面单日驱动产物（run_t0/run_t1 共享形态；经 ``PaperHost.last_day_result`` 暴露）。"""

    requested_date: str  # 调用方输入日期（原文）
    corrected_date: str  # 校正后交易日（YYYYMMDD）
    run_kind: str  # 驱动入口（"t0" / "t1"）
    driven: bool  # 本次调用是否真实驱动了当日（False = 幂等读产物 / 未就绪）
    t0_status: str  # success / no_targets / not_rebalance_day / already_run / not_ready
    fills: List[Dict[str, Any]]  # 当日成交（旧 trades.parquet schema 投影行）
    orders: List[Order]  # 次日计划投影（T+1 指令帧对应的 Order 序列）
    nav: Optional[float]  # 当日总资产（估值口径；未驱动/未就绪 = None）
    readiness: Optional[ReadinessReport] = None  # 就绪报告（未就绪时有值）
    push_text: str = ""  # 推送文本（展示投影）


class PaperHost(PaperHostRecoveryMixin):
    """纸面宿主（契约全量见模块 docstring）。

    Args:
        store: 纸面状态库（隔离根/生产根由调用方决定；宿主不写其它根）
        trading_config: v2 统一策略参数（model_version 必须已解析为具体版本）
        universe: 股票池（BasicUniverse；选股域永久沿用件）
        signal: 信号对象（MLSignal/EnsembleSignal；测试可注入桩信号）
        cost_model: 成本模型（引擎撮合与 trades 投影分解共用同一实例语义）
        stock_basic: 股票基本信息（行业约束用；无行业约束可 None）
        data_storage: 旧数据面 Storage（E4 只读过渡依赖；引擎停牌日历数据源）
        panel_store: v2 PanelDataStore（只读消费，构建语义禁动）
        trading_calendar: 完整交易日历（含未来日，pd.Timestamp 升序无重复）
        engine_kwargs: config→引擎构造参数映射（``to_engine_kwargs`` 产出）
        feature_columns: 模型特征列（就绪门第 3/4 条判定基准）
        train_params: 模型 train_params（派生族核对；freshness 缺省策略同 MLSignal）
        freshness_members: 集成各子模型的 (实际列集, train_params)；单模型传 None
        derived_families_provided: 宿主声明提供的运行时派生族（生产默认空）
        derived_producers: 完整列名 → 实际生产者；族声明无生产者或最终无有效产出均 fail-closed
        engine_mode: engine 切换键解析结果（Q9；观测/断言用，不参与驱动）
        verbose: 引擎/日志详细度
        data_loader: 可选 DataLoader 注入（缺省 ⇒ 由 ``data_storage`` 派生；
            注入子类/自定义实例语义保留，T2 评审处置）
        schedule_mode: 日级排期形态；缺省 ``live`` 保持持久化锚，``replay`` 使用窗口锚
        replay_calendar: 回放窗口交易日历（仅 ``schedule_mode='replay'`` 使用）
    """

    def __init__(
        self,
        *,
        store: PaperStateStore,
        trading_config: TradingConfig,
        universe: BasicUniverse,
        signal: Any,
        cost_model: CostModel,
        stock_basic: Optional[pd.DataFrame],
        data_storage: Storage,
        panel_store: PanelDataStore,
        trading_calendar: List[pd.Timestamp],
        engine_kwargs: Dict[str, Any],
        feature_columns: Sequence[str],
        train_params: Optional[Mapping[str, Any]] = None,
        freshness_members: Optional[Sequence[tuple[Sequence[str], Mapping[str, Any]]]] = None,
        derived_families_provided: Sequence[str] = (),
        derived_producers: Optional[Mapping[str, DerivedProducer]] = None,
        engine_mode: str = ENGINE_V2,
        verbose: bool = False,
        data_loader: Optional[DataLoader] = None,
        portfolio_merger: Optional[PortfolioMerger] = None,
        schedule_mode: str = "live",
        replay_calendar: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        if store.is_remote:
            raise PaperHostError(
                "远端只读 PaperStateStore 仅供展示读取，不支持驱动"
                "（写操作会被跳过，驱动将静默不落盘——构造期 fail-fast，T2 评审处置）"
            )
        calendar_strs = [day.strftime("%Y%m%d") for day in trading_calendar]
        if not calendar_strs:
            raise PaperHostError("交易日历为空（live 排期与 T+1 解析的前置）")
        if calendar_strs != sorted(calendar_strs) or len(set(calendar_strs)) != len(calendar_strs):
            raise PaperHostError("交易日历必须升序且无重复（宿主不自定日历，注入即契约）")
        if not feature_columns:
            raise PaperHostError("feature_columns 不能为空（就绪门第 3/4 条判定基准）")
        schedule_mode, replay_calendar = validate_schedule(
            schedule_mode, replay_calendar, calendar_strs
        )
        self.store = store
        self.trading_config = trading_config
        self.universe = universe
        self.signal = signal
        self.cost_model = cost_model
        self.stock_basic = stock_basic
        self.data_storage = data_storage
        self.panel_store = panel_store
        self.trading_calendar = list(trading_calendar)
        self.engine_kwargs = dict(engine_kwargs)
        self.feature_columns = list(feature_columns)
        self.train_params: Mapping[str, Any] = dict(train_params or {})
        self.freshness_members = freshness_members
        self.derived_families_provided = tuple(derived_families_provided)
        self.derived_producers = dict(derived_producers or {})
        self.engine_mode = engine_mode
        self.verbose = verbose
        self.schedule_mode = schedule_mode
        self.replay_calendar = replay_calendar
        self.portfolio_merger = portfolio_merger or SingleSleevePortfolioMerger()
        # DataLoader 由注入的 Storage 派生（E4：data_storage 是注入件，非隐藏全局创建）；
        # 显式注入 data_loader 时保留其实例语义（T2 评审处置）
        self.loader = (
            data_loader
            if data_loader is not None
            else DataLoader(storage=data_storage, verbose=verbose)
        )
        self._calendar_strs = calendar_strs
        self._calendar_str_set = frozenset(calendar_strs)
        self.last_day_result: Optional[PaperDayResult] = None
        # 相位留痕（run_t1 顺序断言面）：[(日期, 相位)]，相位 ∈ policy_eval/day_drive/day_read；
        # 有界（bot 长驻进程防爆内存，T2 评审处置）：超上限裁前半段
        self._phase_trace: List[Tuple[str, str]] = []

    def _trace(self, date: str, phase: str) -> None:
        """相位留痕追加点（有界：超 ``_PHASE_TRACE_MAX`` 裁掉前半段）。"""
        self._phase_trace.append((date, phase))
        if len(self._phase_trace) > _PHASE_TRACE_MAX:
            del self._phase_trace[: _PHASE_TRACE_MAX // 2]

    # ------------------------------------------------------------------
    # 协议入口（common/protocols/hosts.py::PaperHost）
    # ------------------------------------------------------------------

    def run_t0(self, date: Union[TradeDate, str, pd.Timestamp]) -> List[Order]:
        """T0 判定（收盘后生成次日指令）：驱动当日（若未驱动）→ 次日计划 Order 序列。"""
        corrected = self._correct_trade_date(date)
        result = self._run_daily(str(date).strip(), corrected, "t0")
        return result.orders

    def run_t1(
        self, date: Union[TradeDate, str, pd.Timestamp], fills: Optional[Sequence[Any]] = None
    ) -> None:
        """T+1 执行：前置政策评估相位（退役空操作，顺序断言保留）→ 驱动当日 → fills 提交前核对。

        fills=None ⇒ 引擎行情记账为权威；fills 非空 ⇒ **提交前核对**：与引擎当日撮合逐笔
        比对，一致 ⇒ 正常提交 / 不一致 ⇒ PaperHostError 零副作用拒绝（当日未落盘，可修正
        后重试）；同日已提交后再核对 ⇒ 不一致仅告警报错（无法回滚，需人工核查）。
        （模块 docstring 硬项③成文规则；2026-10-28 用户裁决：引擎行情记账为权威。）
        """
        corrected = self._correct_trade_date(date)
        self._evaluate_policy_phase(corrected)
        self._run_daily(str(date).strip(), corrected, "t1", fills=fills)
        return None

    # ------------------------------------------------------------------
    # 单日原子驱动（唯一驱动入口）
    # ------------------------------------------------------------------

    def _run_daily(
        self,
        requested: str,
        corrected: str,
        run_kind: str,
        *,
        fills: Optional[Sequence[Any]] = None,
    ) -> PaperDayResult:
        """唯一驱动入口：整日写锁 → 回退守卫 → 完整代核对重建 → 同日幂等分派。

        全程持 ``store.write_lock()``（单写入者临界区；冲突 fail-fast）；锁内读取权威
        末日；``_ensure_generation_complete`` 保证权威末代投影完整发布（半提交受控恢复）。
        """
        with self.store.write_lock():
            authority_last = self.store.get_authority_last_date()
            if authority_last is not None and corrected < authority_last:
                raise PaperHostError(
                    f"日期回退：输入日期 {corrected} 早于权威最后处理日 {authority_last}，"
                    "不允许回退执行（现役语义复刻，§7 测试矩阵「日期回退守卫」行）"
                )
            if authority_last is not None:
                self._ensure_generation_complete(authority_last)
            if authority_last == corrected:
                # 同日已驱动（硬项⑦）：幂等读产物，不重复执行；记录缺失 ⇒ 补写不重跑
                self._trace(corrected, "day_read")
                if not self.store.check_run_exists(run_kind, corrected):
                    self._rewrite_missing_run_record(run_kind, corrected)
                result = self._read_driven_day(requested, corrected, run_kind)
                if fills is not None:
                    self._reconcile_fills_readback(result, fills)
            else:
                self._trace(corrected, "day_drive")
                result = self._drive_day(
                    requested, corrected, run_kind, fills=fills, expected_base_date=authority_last
                )
            self.last_day_result = result
            return result

    def _drive_day(
        self,
        requested: str,
        corrected: str,
        run_kind: str,
        *,
        fills: Optional[Sequence[Any]] = None,
        expected_base_date: Optional[str] = None,
    ) -> PaperDayResult:
        """真实驱动当日：就绪门 → 零未来断言 → 装配/恢复 → 排期 → run_step → fills 提交前核对 → 提交。"""
        assembly = self._assemble(corrected)
        if not assembly.readiness.ok:
            self._write_gap_record(corrected, assembly.readiness)
            logger.warning(
                f"纸面宿主 {corrected} 数据未就绪，推迟生成（不落盘、不生成信号）: "
                f"{assembly.readiness.gaps}"
            )
            return PaperDayResult(
                requested_date=requested,
                corrected_date=corrected,
                run_kind=run_kind,
                driven=False,
                t0_status="not_ready",
                fills=[],
                orders=[],
                nav=None,
                readiness=assembly.readiness,
                push_text=f"({corrected}) 数据未就绪，推迟生成（缺口已登记 runs/gap_{corrected}.json）",
            )
        self._assert_no_future_price(assembly.price_data, corrected)
        corrected_ts = self._to_timestamp(corrected)
        engine = self._new_engine(assembly.features_by_date)
        self._restore_or_init_anchor(engine, corrected_ts)
        self._wire_day_inputs(engine, corrected_ts, assembly.price_data)
        day, was_signal_day, step_calendar = prepare_daily_schedule(self, engine, corrected_ts)
        trade_start = len(engine.trades)
        engine.run_step(corrected_ts, step_calendar, assembly.price_data)
        if day is not None:
            settle_live_day(engine, corrected_ts, day)
        if fills is not None:
            # 提交前核对（硬项③，零副作用拒绝：不一致 ⇒ 当日未落盘，可修正后重试）
            self._reconcile_fills_precommit(engine, trade_start, fills)
        return self._commit_day(
            engine,
            requested,
            corrected,
            corrected_ts,
            run_kind,
            trade_start,
            was_signal_day,
            expected_base_date=expected_base_date,
        )

    # ------------------------------------------------------------------
    # 前置守卫与数据组装
    # ------------------------------------------------------------------

    def _correct_trade_date(self, date: Union[TradeDate, str, pd.Timestamp]) -> str:
        """校正交易日期（复刻旧 ``paper/runner/calendar.py:62-102`` 语义）。

        在历 ⇒ 用之；不在 ⇒ 滚到其后首个开市日（warning）；异常 ⇒ 回原值（不阻断，
        后续就绪门/日历成员校验处 fail-fast）。
        """
        raw = self._normalize_date_text(date)
        try:
            TradeDate.from_str(raw)  # 形态/真实日期校验（非法 ⇒ ValueError）
            if raw in self._calendar_str_set:
                return raw
            for day in self._calendar_strs:
                if day > raw:
                    logger.warning(f"输入日期 {raw} 不是交易日，已自动校正到下一交易日: {day}")
                    return day
            logger.warning(f"未找到 {raw} 之后的交易日，使用原日期")
            return raw
        except (TypeError, ValueError) as exc:
            logger.error(f"校正交易日期失败: {exc}")
            return str(date).strip()

    @staticmethod
    def _normalize_date_text(date: Union[TradeDate, str, pd.Timestamp]) -> str:
        """入参日期 → YYYYMMDD 文本（TradeDate/Timestamp/str 三形态）。"""
        if isinstance(date, TradeDate):
            return str(date)
        if isinstance(date, pd.Timestamp):
            return date.strftime("%Y%m%d")
        return str(date).strip().replace("-", "").replace("/", "")

    def _to_timestamp(self, corrected: str) -> pd.Timestamp:
        """YYYYMMDD → 日历坐标系内的 pd.Timestamp（corrected 必须已校正为开市日）。"""
        if corrected not in self._calendar_str_set:
            raise PaperHostError(
                f"校正后日期 {corrected} 仍不在交易日历内（日历末日之后或非法输入），拒绝驱动"
            )
        return pd.Timestamp(TradeDate.from_str(corrected).value)

    def _next_trade_date(self, corrected: str) -> Optional[str]:
        """下一个交易日（日历内 corrected 之后的首个开市日；无 ⇒ None）。"""
        for day in self._calendar_strs:
            if day > corrected:
                return day
        return None

    def _assemble(self, corrected: str) -> DailyAssembly:
        """当日数据组装 + 就绪门（D4；日历/列集/派生族声明/执行价格形态均由宿主注入）。"""
        try:
            return assemble_daily_data(
                corrected,
                self.panel_store,
                self.loader,
                self._calendar_strs,
                self.feature_columns,
                train_params=self.train_params,
                freshness_members=self.freshness_members,
                derived_families_provided=self.derived_families_provided,
                derived_producers=self.derived_producers,
                sell_price_type=self.trading_config.sell_price,
            )
        except ValueError as exc:
            raise PaperHostError(f"纸面宿主数据组装输入非法（{corrected}）: {exc}") from exc

    @staticmethod
    def _assert_no_future_price(price_data: pd.DataFrame, corrected: str) -> None:
        """指令文件层零未来行情断言（§16.6 T2 增项）：含未来日行 ⇒ fail-fast。"""
        if price_data is None or price_data.empty or "trade_date" not in price_data.columns:
            return
        future_mask = price_data["trade_date"].astype(str) > corrected
        if bool(future_mask.any()):
            future_dates = sorted(price_data.loc[future_mask, "trade_date"].astype(str).unique())
            raise PaperHostError(
                f"行情窗口含未来交易日行（> {corrected}）: {future_dates[:5]}——"
                "零未来行情断言拒绝（指令文件层不得见未来行情）"
            )

    def _write_gap_record(self, corrected: str, readiness: ReadinessReport) -> None:
        """就绪缺口登记（``runs/gap_<date>.json``）。

        登记为宿主侧诊断产物（非账户状态投影，不经 commit_day——commit_day 语义 =
        同代提交账户状态，未就绪日本身不落任何账户代）；写盘失败不阻断推迟语义。
        """
        payload = {
            "kind": "readiness_gap",
            "date": corrected,
            "gaps": list(readiness.gaps),
            "row_count": int(readiness.row_count),
            "row_count_band": [str(v) for v in readiness.row_count_band],
            "uncovered_columns": list(readiness.uncovered_columns),
            "derived_family_note": readiness.derived_family_note,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        path = self.store.runs_path / f"gap_{corrected}.json"
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.error(f"就绪缺口登记写盘失败（不阻断推迟语义）: {path}（{exc}）")
            return
        logger.warning(f"就绪缺口已登记: {path}")

    # ------------------------------------------------------------------
    # 引擎装配（每日新建）与恢复
    # ------------------------------------------------------------------

    def _new_engine(self, features_by_date: Dict[str, pd.DataFrame]) -> BacktestEngineML:
        """每日新建引擎实例（不跨日持有；§17.7 硬项⑧单进程单引擎）。"""
        engine = NativeMergeBacktestEngineML(
            features_by_date=features_by_date,
            universe=self.universe,
            signal=self.signal,
            cost_model=self.cost_model,
            stock_basic=self.stock_basic,
            data_storage=self.data_storage,
            industry_reader=self.loader.load_shenwan_industry,
            verbose=self.verbose,
            portfolio_merger=self.portfolio_merger,
            sleeve_id=SLEEVE_ID,
            **self.engine_kwargs,
        )
        configure_engine_schedule(engine, self.schedule_mode)
        return engine

    def _restore_or_init_anchor(self, engine: BacktestEngineML, corrected_ts: pd.Timestamp) -> None:
        """恢复启动（快照 + 双状态一致性校验）或首次启动（init_live_anchor 专用）。"""
        blob = self.store.load_engine_state_blob()
        # 双状态一致性校验先于恢复：滞后 ⇒ 受控重投影 + 告警；冲突 ⇒ PaperStateError 拒绝
        status = self.store.check_consistency(auto_repair=True)
        if blob is None:
            if status == "legacy_projection":
                raise PaperHostError(
                    "检测到旧链遗产投影（account.json）但无权威引擎快照（T5 迁移前形态）："
                    "拒绝以全新账户启动 v2 宿主（防接回/防覆盖，Q9）——"
                    "请先执行 T5 迁移或 D11 回滚程序"
                )
            initialize_anchor(self, engine, corrected_ts)
            return
        restore_state(engine, state_from_blob(loads_blob(blob)))
        logger.info(f"纸面宿主恢复启动：权威快照已恢复（一致性校验 = {status}），禁止重锚")

    def _wire_day_inputs(
        self, engine: BacktestEngineML, corrected_ts: pd.Timestamp, price_data: pd.DataFrame
    ) -> None:
        """每日输入接线（wiring 字段每日重建，不入快照——字段归类注册表 wiring 类）。"""
        wire_day_inputs(self, engine, price_data)

    # ------------------------------------------------------------------
    # 导出与同代原子提交
    # ------------------------------------------------------------------

    def _commit_day(
        self,
        engine: BacktestEngineML,
        requested: str,
        corrected: str,
        corrected_ts: pd.Timestamp,
        run_kind: str,
        trade_start: int,
        was_signal_day: bool,
        *,
        expected_base_date: Optional[str] = None,
    ) -> PaperDayResult:
        """导出快照 + 组装全部投影 + commit_day 同代原子提交（写序 = 权威先行）。

        ``expected_base_date`` = 本次驱动计算所基于的权威「最后处理日」（锁内比较交换
        守卫：盘上权威已推进 ⇒ 拒绝过期提交）。
        """
        day_trades = list(engine.trades[trade_start:])
        # T1 失败后的残余槽位在当日收盘（T0 相位）具体化为下一交易日信号；
        # 必须先于指令投影/权威快照。
        if run_kind == "t1" and self.schedule_mode != "replay":
            engine.plan_residual_completion(corrected_ts)
        # T0 指令必须先于权威快照确定；把固化股数写回同一份 pending 信号，
        # 供 T1 与投影消费同一份计划（不能在 T1 再按权重重算）。
        frame, instructions_date = self._build_instructions(engine, corrected, corrected_ts)
        blob = state_to_blob(export_state(engine))
        if self.schedule_mode != "replay":
            inject_instruction_shares(blob, frame)
        # 提交相位随权威原子点持久化：崩溃在 runs 记录前时，跨日恢复仍能补齐原代标记。
        blob[GENERATION_INFO_KEY] = {"trade_date": corrected, "run_kind": run_kind}
        blob_bytes = dumps_blob(blob)
        orders = (
            self._orders_from_instruction_frame(frame, engine.positions)
            if frame is not None
            else []
        )
        if orders:
            netted = list(self.portfolio_merger.net_across_sleeves(orders))
            if netted != orders:
                raise PaperHostError(
                    "单袖 S6 返回序列发生变化，T3 只允许原序列直通；"
                    "跨袖净额化/改写归 P6.5，当日尚未提交"
                )
            orders = netted
        pending_buys, pending_sells = self._project_pending_queues(engine)
        trade_rows = [self._project_trade_row(record) for record in day_trades]
        nav_row = self._project_nav_row(engine, corrected, day_trades)
        t0_status = resolve_t0_status(engine, corrected_ts, was_signal_day)
        nav_value = (
            float(engine.portfolio_values[-1]["portfolio_value"])
            if engine.portfolio_values
            else None
        )
        run_record = {
            "trade_date": corrected,
            # engine 字段语义 = 配置解析结果（Q9 切换键），非「实际驱动引擎」注记——
            # v2 宿主驱动时当日恒由 v2 引擎执行（T2 评审处置登记，防影子记录读者误读）
            "engine": self.engine_mode,
            "run_kind": run_kind,
            "t0_status": t0_status,
            "buy_price_type": self.trading_config.buy_price,
            "sell_price_type": self.trading_config.sell_price,
            "instructions_count": len(orders),
            "orders_count": len(orders),
            "fills_count": len(trade_rows),
            "pending_buys_count": len(pending_buys),
            "pending_sells_count": len(pending_sells),
            "nav": nav_value,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.store.commit_day(
            date=corrected,
            engine_state_blob=blob_bytes,
            account=build_account_projection(engine.current_capital, engine.positions, corrected),
            strategy_state=build_strategy_state_projection(engine._last_rebalance_nav),
            ranked_candidates=self._project_ranked_candidates(engine),
            stop_loss_state=self._project_stop_loss_state(engine),
            rebalance_state=self._project_rebalance_state(engine),
            instructions=frame,
            instructions_date=instructions_date,
            pending_buys=pending_buys,
            pending_sells=pending_sells,
            nav_row=nav_row,
            trade_rows=trade_rows,
            run_kind=run_kind,
            run_record=run_record,
            expected_base_date=expected_base_date,
            trim_engine_state=self.schedule_mode != "replay",
        )
        push_text = self._format_push_text(
            corrected, instructions_date, frame, t0_status, trade_rows
        )
        return PaperDayResult(
            requested_date=requested,
            corrected_date=corrected,
            run_kind=run_kind,
            driven=True,
            t0_status=t0_status,
            fills=trade_rows,
            orders=orders,
            nav=nav_value,
            readiness=None,
            push_text=push_text,
        )

    def _build_instructions(
        self, engine: BacktestEngineML, corrected: str, corrected_ts: pd.Timestamp
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """引擎当日 T0 产出 → T+1 指令帧（经 state_store 纯函数；既有 T+1 文件合并优先）。"""
        instructions_date = self._next_trade_date(corrected)
        if instructions_date is None:
            logger.warning(f"{corrected} 无下一交易日，跳过生成次日指令投影")
            return None, None
        buy_plan = engine.pending_signals.get(corrected_ts)
        prices = self._close_prices_at(engine.price_data_cache, corrected)
        total_capital = (
            float(engine.portfolio_values[-1]["portfolio_value"])
            if engine.portfolio_values
            else float(engine.current_capital)
        )
        capital_fraction = 1.0
        if buy_plan and engine.stagger_tranches > 1:
            capital_fraction = engine._get_tranche_capital_fraction(
                int(buy_plan.get("tranche_idx", 0))
            )
        delayed_orders = (
            engine.pending_order_manager.get_all_orders()
            if engine.pending_order_manager is not None
            else []
        )
        existing = self.store.load_instructions_frame(instructions_date)
        frame = build_instruction_frame(
            signal_date=corrected,
            buy_plan=buy_plan,
            condition_sells=engine.pending_condition_sells,
            stop_loss_sells=engine.pending_stop_loss_sells,
            delayed_orders=delayed_orders,
            positions=engine.positions,
            prices=prices,
            total_capital=total_capital,
            capital_fraction=capital_fraction,
            buy_price_type=self.trading_config.buy_price,
            sell_price_type=self.trading_config.sell_price,
            existing=existing,
            trade_dates=self._calendar_strs,
        )
        return frame, instructions_date

    @staticmethod
    def _close_prices_at(price_data: pd.DataFrame, date: str) -> Dict[str, float]:
        """当日不复权收盘价字典（指令投影权重→股数口径）。"""
        if price_data is None or price_data.empty:
            return {}
        day = price_data[price_data["trade_date"].astype(str) == date]
        prices: Dict[str, float] = {}
        for row in day.itertuples(index=False):
            close = getattr(row, "close", None)
            if close is not None and pd.notna(close):
                prices[str(row.ts_code)] = float(close)
        return prices

    # ------------------------------------------------------------------
    # 投影（纯搬运：引擎态 → 旧 schema；不重算决策数值）
    # ------------------------------------------------------------------

    @staticmethod
    def _project_ranked_candidates(engine: BacktestEngineML) -> Optional[Dict[str, Any]]:
        """排序候选投影（无候选/无信号日 ⇒ None，commit_day 跳过）。"""
        if engine._last_signal_date is None or not engine._last_ranked_candidates:
            return None
        return build_ranked_candidates_projection(
            engine._last_signal_date, engine._last_ranked_candidates
        )

    @staticmethod
    def _project_stop_loss_state(engine: BacktestEngineML) -> Optional[Dict[str, Any]]:
        """止损状态投影（止损未启用 ⇒ None，与旧链「不启用不写文件」口径一致）。"""
        if engine.stop_loss_monitor is None:
            return None
        return build_stop_loss_projection(engine.stop_loss_monitor.consecutive_limit_down_days)

    def _project_rebalance_state(self, engine: BacktestEngineML) -> Dict[str, Any]:
        """调仓投影委托 schedule 模块（live 锚 / replay 窗口锚）。"""
        return project_rebalance_state(self, engine)

    @staticmethod
    def _build_nav_row(pv: Dict[str, Any], initial: float, corrected: str) -> Dict[str, Any]:
        """净值行构造（nav 经账本层唯一入口 ``calculate_nav`` 派生，D3；投影层不自算）。"""
        total_value = float(pv["portfolio_value"])
        return {
            "trade_date": corrected,
            "cash": float(pv["capital"]),
            "position_value": float(pv["market_value"]),
            "total_value": total_value,
            "nav": float(calculate_nav(total_value, initial)),
        }

    def _project_nav_row(
        self, engine: BacktestEngineML, corrected: str, day_trades: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """净值行投影（跳空语义保持：仅当日有成交才给行）。"""
        if not day_trades:
            return None
        return self._build_nav_row(
            engine.portfolio_values[-1], float(engine.initial_capital), corrected
        )

    def _project_nav_row_from_state(
        self, state: Any, corrected: str, day_trades: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """自权威快照重建当日净值行（半提交恢复用；跳空语义同 ``_project_nav_row``）。

        initial_capital 为 config 类字段（不在 EngineState 演化集内）⇒ 取宿主
        ``engine_kwargs``（与恢复引擎构造同一来源，口径一致）。
        """
        if not day_trades:
            return None
        initial = self.engine_kwargs.get("initial_capital")
        if initial is None:
            raise PaperHostError(
                "engine_kwargs 缺 initial_capital（半提交恢复重建当日 nav 行需要），拒绝继续"
            )
        return self._build_nav_row(state.portfolio_values[-1], float(initial), corrected)

    def _project_trade_row(
        self, record: Dict[str, Any], *, validate_cost: bool = True
    ) -> Dict[str, Any]:
        """v2 成交记录 → 旧 trades.parquet schema 行（成本三列按同一 CostModel 分解）。

        正常提交校验成本分解；半提交恢复传 ``validate_cost=False``，配置漂移只告警，
        由恢复层保证已发布完整行不改写、缺失行才按当前口径补建。
        """
        action = str(record["action"])
        amount = float(record["amount"])
        commission = self.cost_model.calculate_commission(amount)
        stamp_tax = self.cost_model.calculate_stamp_tax(amount) if action == "sell" else 0.0
        slippage = self.cost_model.calculate_slippage(amount)
        total_cost = float(record["cost"])
        cost_differs = abs(commission + stamp_tax + slippage - total_cost) > _COST_SPLIT_TOLERANCE
        if cost_differs and validate_cost:
            raise PaperHostError(
                f"成交成本分解与入账总额不一致（成本模型配置漂移检出）: "
                f"{record.get('stock')} {action} amount={amount} "
                f"分解={commission + stamp_tax + slippage} vs 入账 cost={total_cost}"
            )
        if cost_differs:
            logger.warning(
                f"半提交恢复检出成本模型配置漂移（不阻断、不改写已发布完整行）: "
                f"{record.get('stock')} {action} amount={amount} "
                f"当前分解={commission + stamp_tax + slippage} vs 入账 cost={total_cost}"
            )
        return {
            "trade_date": self._normalize_date_text(record["date"]),
            "ts_code": str(record["stock"]),
            "action": action,
            "shares": int(record["shares"]),
            "price": float(record["price"]),
            "amount": amount,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "slippage": slippage,
            "total_cost": total_cost,
            "reason": self._trade_reason(record, action),
        }

    @staticmethod
    def _trade_reason(record: Dict[str, Any], action: str) -> str:
        """成交 reason 投影（口径与差异登记见模块 docstring）。"""
        if action == "sell":
            if record.get("sell_reason"):
                return str(record["sell_reason"])
            sell_type = str(record.get("sell_type") or "")
            return _SELL_TYPE_REASONS.get(sell_type, sell_type or "卖出")
        if record.get("buy_reason"):
            return str(record["buy_reason"])
        return "信号生成"

    def _project_pending_queues(
        self, engine: BacktestEngineML
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """延迟订单管理器队列 → 旧 pending_buys/pending_sells 投影（确定性排序）。"""
        manager = engine.pending_order_manager
        orders = manager.get_all_orders() if manager is not None else []
        return self._project_pending_orders(orders, engine.positions)

    def _project_pending_orders(
        self, orders: Sequence[Any], positions: Mapping[str, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """PendingOrder 序列 → (pending_buys 投影, pending_sells 投影)。

        差异登记：v2 PendingOrder 无 target_weight/last_attempt_date ⇒ 恒 0.0/""；
        pending_sells 的 shares = 现持仓整手可卖股数（与指令投影同口径）。
        """
        buys: List[Dict[str, Any]] = []
        sells: List[Dict[str, Any]] = []
        ordered = sorted(
            orders,
            key=lambda order: (
                self._normalize_date_text(getattr(order, "create_date", "")),
                str(getattr(order, "stock", "")),
                str(getattr(order, "action", "")),
            ),
        )
        for order in ordered:
            stock = str(getattr(order, "stock", "") or "")
            action = str(getattr(order, "action", "") or "")
            if not stock or action not in ("buy", "sell"):
                continue
            row = {
                "ts_code": stock,
                "target_weight": 0.0,
                "reason": str(getattr(order, "last_reason", None) or "延迟订单"),
                "create_date": self._normalize_date_text(getattr(order, "create_date", "")),
                "attempts": int(getattr(order, "retry_count", 0) or 0),
                "last_attempt_date": "",
            }
            if action == "buy":
                row["original_signal_date"] = self._normalize_date_text(
                    getattr(order, "signal_date", "")
                )
                buys.append(row)
            else:
                info = positions.get(stock) or {}
                row["shares"] = int(info.get("shares", 0) or 0) // 100 * 100
                sells.append(row)
        return buys, sells

    # ------------------------------------------------------------------
    # Order 投影（run_t0 返回面）
    # ------------------------------------------------------------------

    def _orders_from_instruction_frame(
        self, frame: pd.DataFrame, positions: Mapping[str, Dict[str, Any]]
    ) -> List[Order]:
        """T+1 指令帧 → Order 序列（单袖 source_sleeve="value"；口径见模块 docstring）。"""
        orders: List[Order] = []
        for row in frame.to_dict("records"):
            ts_code = str(row["ts_code"])
            action = str(row["action"])
            shares = int(row["shares"])
            current = int((positions.get(ts_code) or {}).get("shares", 0) or 0)
            if action == "buy":
                side = OrderSide.BUY
                target = current + shares
            elif action == "sell":
                side = OrderSide.SELL
                target = max(current - shares, 0)
            else:
                logger.warning(f"指令帧含未知动作 {action!r}（{ts_code}），跳过 Order 投影")
                continue
            orders.append(
                Order(
                    ts_code=TSCode(ts_code),
                    side=side,
                    target_shares=Shares(target),
                    order_type=OrderType.MARKET_OPEN,
                    reason=self._map_order_reason(action, str(row["reason"])),
                    source_sleeve=SLEEVE_ID,
                )
            )
        return orders

    @staticmethod
    def _map_order_reason(action: str, reason: str) -> OrderReason:
        """指令 reason 文本 → OrderReason（关键词映射 + 动作兜底，登记口径）。"""
        if "止损" in reason:
            return OrderReason.STOP_LOSS
        if "到期" in reason or "持有期" in reason:
            return OrderReason.EXPIRY
        if "调仓" in reason or "信号" in reason:
            return OrderReason.REBALANCE
        if "补位" in reason or "延迟" in reason:
            return OrderReason.REFILL if action == "buy" else OrderReason.CONDITION_SELL
        return OrderReason.REBALANCE if action == "buy" else OrderReason.CONDITION_SELL

    # ------------------------------------------------------------------
    # 幂等读产物路径（同日已驱动）
    # ------------------------------------------------------------------

    def _read_driven_day(self, requested: str, corrected: str, run_kind: str) -> PaperDayResult:
        """幂等读产物（不重复执行）：指令投影重构 Order + 当日成交行自 trades 读回。"""
        state = self.store.load_engine_state()
        positions: Mapping[str, Dict[str, Any]] = state.positions if state is not None else {}
        instructions_date = self._next_trade_date(corrected)
        frame = (
            self.store.load_instructions_frame(instructions_date)
            if instructions_date is not None
            else None
        )
        orders = (
            self._orders_from_instruction_frame(frame, positions)
            if frame is not None and len(frame) > 0
            else []
        )
        fills = self._load_day_trade_rows(corrected)
        nav_value = (
            float(state.portfolio_values[-1]["portfolio_value"])
            if state is not None and state.portfolio_values
            else None
        )
        push_text = self._format_push_text(
            corrected, instructions_date, frame, "already_run", fills
        )
        return PaperDayResult(
            requested_date=requested,
            corrected_date=corrected,
            run_kind=run_kind,
            driven=False,
            t0_status="already_run",
            fills=fills,
            orders=orders,
            nav=nav_value,
            readiness=None,
            push_text=push_text,
        )

    def _load_day_trade_rows(self, corrected: str) -> List[Dict[str, Any]]:
        """当日成交行自 trades.parquet 读回（幂等路径；跳空语义下无成交日 = 空）。"""
        trades = self.store.load_all_trades()
        if trades is None or trades.empty:
            return []
        day = trades[trades["trade_date"].astype(str).str.replace("-", "") == corrected]
        return day.to_dict("records")

    # ------------------------------------------------------------------
    # 政策相位（退役空操作）与 fills 核对
    # ------------------------------------------------------------------

    def _evaluate_policy_phase(self, corrected: str) -> None:
        """run_t1 前置政策评估相位（退役形态 = null ⇒ 空操作，顺序断言保留）。

        暴露政策族已退役（Q5；v2 内核已摘除 exposure mixin），本相位为空操作；
        「政策评估先于 T1 执行」的顺序关系经 ``_phase_trace`` 留痕供测试断言
        （§7 测试矩阵「run_t1 顺序」行）。
        """
        self._trace(corrected, "policy_eval")
        logger.debug(f"run_t1 前置政策评估相位（退役空操作）: {corrected}")

    def _reconcile_fills_precommit(
        self, engine: BacktestEngineML, trade_start: int, fills: Sequence[Any]
    ) -> None:
        """fills 提交前核对（硬项③；不一致 ⇒ PaperHostError 零副作用拒绝——当日未落盘）。

        一致 ⇒ 正常提交（引擎已入账，不重复入账，D2）；不一致 ⇒ 拒绝（可修正 fills 后重试）。
        """
        engine_fills = sorted(
            self._fill_identity_from_record(record) for record in engine.trades[trade_start:]
        )
        external_fills = sorted(self._fill_identity(item) for item in fills)
        if engine_fills != external_fills:
            raise PaperHostError(
                f"外部 fills 与引擎当日撮合不一致——提交前核对拒绝（零副作用：当日未落盘，"
                f"可修正 fills 后重试；权威 = 引擎行情记账，2026-10-28 用户裁决）: "
                f"引擎侧 {len(engine_fills)} 笔 vs 外部 {len(external_fills)} 笔；"
                f"引擎侧={engine_fills}；外部={external_fills}"
            )
        logger.info(
            f"外部 fills 与引擎当日撮合逐笔一致（{len(external_fills)} 笔），"
            "提交前核对通过（引擎已入账，不重复入账）"
        )

    def _reconcile_fills_readback(self, result: PaperDayResult, fills: Sequence[Any]) -> None:
        """同日已提交后的 fills 核对（幂等读路径；不一致 ⇒ 告警错误——当日已提交，无法回滚）。"""
        committed_fills = sorted(self._fill_identity(row) for row in result.fills)
        external_fills = sorted(self._fill_identity(item) for item in fills)
        if committed_fills != external_fills:
            raise PaperHostError(
                f"外部 fills 与当日已提交成交不一致（当日已在此前提交，无法回滚，需人工核查）: "
                f"已提交 {len(committed_fills)} 笔 vs 外部 {len(external_fills)} 笔；"
                f"已提交={committed_fills}；外部={external_fills}"
            )
        logger.info(f"外部 fills 与当日已提交成交逐笔一致（{len(external_fills)} 笔），幂等接受")

    @staticmethod
    def _fill_identity_from_record(record: Mapping[str, Any]) -> Tuple[str, str, int, float]:
        """引擎当日成交记录 → 恒等键（与 ``_fill_identity`` 的投影行口径一致）。"""
        return (
            str(record["stock"]),
            str(record["action"]),
            int(record["shares"]),
            float(record["price"]),
        )

    @staticmethod
    def _fill_identity(item: Any) -> Tuple[str, str, int, float]:
        """fills 元素 → 恒等键 (ts_code, action, shares, price)（v2 Fill / dict 两形态）。"""
        if isinstance(item, Fill):
            return (
                str(item.order.ts_code.value),
                str(item.order.side.value),
                int(item.fill_shares.value),
                float(item.fill_price.value),
            )
        if isinstance(item, Mapping):
            missing = [key for key in ("ts_code", "shares", "price") if key not in item]
            if missing:
                raise PaperHostError(f"fills 元素缺必需键 {missing}: {dict(item)!r}")
            action = str(item.get("action") or item.get("side") or "")
            if not action:
                raise PaperHostError(f"fills 元素缺 action/side 键: {dict(item)!r}")
            try:
                return (
                    str(item["ts_code"]),
                    action,
                    int(item["shares"]),
                    float(item["price"]),
                )
            except (TypeError, ValueError) as exc:
                raise PaperHostError(
                    f"fills 元素 shares/price 形态非法: {dict(item)!r}（{exc}）"
                ) from exc
        raise PaperHostError(
            f"fills 元素形态非法（须为 v2 Fill 或 dict 投影行）: {type(item).__name__}"
        )

    # ------------------------------------------------------------------
    # 推送文本（展示投影；本地化决策见模块 docstring）
    # ------------------------------------------------------------------

    def _format_push_text(
        self,
        corrected: str,
        instructions_date: Optional[str],
        frame: Optional[pd.DataFrame],
        t0_status: str,
        fills: Sequence[Dict[str, Any]],
    ) -> str:
        """日驱动推送文本（措辞风格对齐旧「卖X笔 买Y笔」形态）。"""
        status_text = {
            "success": "调仓日：次日计划已生成",
            "no_targets": "调仓日但未生成目标",
            "not_rebalance_day": "非调仓日",
            "already_run": "今日已执行过（幂等读取产物）",
            "not_ready": "数据未就绪，已推迟",
        }.get(t0_status, t0_status)
        lines = [f"v2 纸面日驱动 ({corrected})", f"T0信号: {status_text}"]
        if fills:
            buy_count = sum(1 for row in fills if row.get("action") == "buy")
            lines.append(f"T1执行: 买{buy_count}笔 卖{len(fills) - buy_count}笔")
        lines.extend(self._format_instruction_lines(instructions_date, frame))
        return "\n".join(lines)

    @staticmethod
    def _format_instruction_lines(
        instructions_date: Optional[str], frame: Optional[pd.DataFrame]
    ) -> List[str]:
        """指令帧 → 推送文本行（先卖后买序；空帧/无次日 ⇒ 对应提示行）。"""
        if instructions_date is None:
            return ["下一交易日指令: 无后续交易日"]
        if frame is None or len(frame) == 0:
            return [f"下一交易日指令 ({instructions_date})", "无待执行买卖指令"]
        rows = frame.to_dict("records")
        sell_rows = [row for row in rows if row["action"] == "sell"]
        buy_rows = [row for row in rows if row["action"] == "buy"]
        lines = [
            f"下一交易日指令 ({instructions_date})",
            f"卖{len(sell_rows)}笔 买{len(buy_rows)}笔",
        ]
        if sell_rows:
            lines.append("卖出-")
            for index, row in enumerate(sell_rows, 1):
                lines.append(f"{index}. {row['ts_code']}")
                lines.append(f"   量{int(row['shares'])}, 因{row['reason']}")
        if buy_rows:
            lines.append("买入-")
            for index, row in enumerate(buy_rows, 1):
                lines.append(f"{index}. {row['ts_code']}")
                weight = float(row.get("target_weight", 0.0) or 0.0)
                weight_text = f", 权{weight:.2%}" if weight > 0 else ""
                lines.append(f"   量{int(row['shares'])}{weight_text}, 因{row['reason']}")
        return lines
