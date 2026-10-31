# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主半提交恢复 mixin（自 host.py 拆分，行为不变）。

拆分缘由（§7.4「新增功能对应新增文件」+ 复杂度门 1200 行文件硬上限，2026-10-28
T2 评审处置）：完整代核对重建/幂等记录补写是 ``PaperHost`` 的恢复职责族，与单日
驱动主路径正交。``PaperHost`` 经 mixin 继承（``class PaperHost(PaperHostRecoveryMixin)``），
实例方法面不变。

契约（宿主侧不变量）：状态投影以权威快照为准；已发布完整 trades/nav 行不因当前配置变化
而改写，仅缺失或身份不符时补建（配置漂移告警）。``runs/{run_kind}_{date}.json`` 存在 =
完整代发布标记（写序最后一步）；原 run_kind/date 随权威快照持久化，同日或次日首次恢复均须补齐。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.v2.core.accounting.engine_state import loads_blob, restore_state
from src.lazybull.v2.hosts.paper.config_adapter import PaperHostError
from src.lazybull.v2.hosts.paper.instruction_projection import (
    INSTRUCTION_COLUMNS,
    _new_instruction_row,
)
from src.lazybull.v2.hosts.paper.state_store import GENERATION_INFO_KEY


class PaperHostRecoveryMixin:
    """PaperHost 的半提交恢复职责族（mixin；self 即宿主实例，方法与属性经宿主解析）。"""

    def _rewrite_missing_run_record(self, run_kind: str, corrected: str) -> None:
        """幂等记录补写（崩溃在记录写入前的恢复形态）：只补 ⑤ 记录，不重跑。

        当日投影完整性由 ``_ensure_generation_complete`` 先行保证；同日重入与跨日推进前
        均可调用。此处仅补齐 ``runs/{run_kind}_{date}.json``（完整代发布标记）。
        """
        if self.store.load_engine_state_blob() is None:
            raise PaperHostError(f"权威快照缺失却要求补写记录（{corrected}），状态不一致")
        run_record = {
            "trade_date": corrected,
            "engine": self.engine_mode,
            "run_kind": run_kind,
            "note": "幂等补写：权威快照已含当日（崩溃在幂等记录写入前），未重跑；"
            "当日投影已经完整代核对重建",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.store.write_run_record(run_kind, corrected, run_record)
        logger.warning(f"幂等记录缺失（{run_kind}/{corrected}），已补写记录且未重跑当日")

    def _ensure_generation_published(self, corrected: str) -> List[str]:
        """推进新代前补齐旧代发布标记；原 run_kind 取自权威快照而非当前调用相位。"""
        if self.store.check_run_exists("t0", corrected) or self.store.check_run_exists(
            "t1", corrected
        ):
            return []
        raw = self.store.load_engine_state_blob()
        if raw is None:
            raise PaperHostError(f"权威快照缺失却要求补写记录（{corrected}），状态不一致")
        generation = loads_blob(raw).get(GENERATION_INFO_KEY)
        if not isinstance(generation, dict):
            raise PaperHostError(
                f"权威快照缺少 {GENERATION_INFO_KEY} 提交信息且 {corrected} 无任何发布标记，"
                "无法确定原 run_kind；请人工核查原调用相位后补写，禁止猜测"
            )
        run_kind = generation.get("run_kind")
        generation_date = generation.get("trade_date")
        if run_kind not in ("t0", "t1") or generation_date != corrected:
            raise PaperHostError(
                f"权威快照提交信息非法或代际不符（{corrected}）: {generation!r}，拒绝补写发布标记"
            )
        self._rewrite_missing_run_record(str(run_kind), corrected)
        return [f"runs/{run_kind}_{corrected}.json"]

    @staticmethod
    def _pending_orders_from_state(state: Any) -> List[Any]:
        """权威快照 → 延迟订单对象列表（pending_order_manager 子对象投影展开）。"""
        manager = state.pending_order_manager
        if not manager:
            return []
        return list(manager["pending_orders"].values())

    # ------------------------------------------------------------------
    # 半提交恢复（完整代核对重建；T2 评审处置）
    # ------------------------------------------------------------------

    def _ensure_generation_complete(self, authority_last: str) -> None:
        """完整代核对重建（半提交受控恢复；``_run_daily`` 分派前、整日写锁内执行）。

        已发布完整 trades/nav 行保持不动；缺失或身份不符才以权威快照补建，配置漂移仅告警。
        account 族双状态校验委托 store ``check_consistency``（滞后修复/冲突上冒）；权威末代
        发布标记在分派前一律补齐，因此次日首次重启也不会留下永久半提交形态。
        """
        state = self.store.load_engine_state()
        if state is None:
            raise PaperHostError(
                f"权威最后处理日 {authority_last} 但权威快照缺失，状态不一致（禁止继续）"
            )
        self.store.check_consistency(auto_repair=True)
        repaired: List[str] = []
        # 状态投影族确定性比对补齐（③ 中段崩溃形态当日收敛：account 已写而 strategy/
        # last_trade_date/ranked/rebalance/stop_loss 未写的缝隙覆盖；一致 ⇒ 零写盘）
        repaired.extend(self.store.sync_state_projections(state, authority_last))
        # 当日 trades 行（快照经 ⑤ 定稿修订保留当日成交 = 重建原料）
        day_trades = [
            record
            for record in state.trades
            if self._normalize_date_text(record.get("date")) == authority_last
        ]
        expected_trade_rows = [
            self._project_trade_row(record, validate_cost=False) for record in day_trades
        ]
        existing_trade_rows = self._load_day_trade_rows(authority_last)
        expected_identities = sorted(
            self._trade_projection_identity(row) for row in expected_trade_rows
        )
        existing_identities = sorted(
            self._trade_projection_identity(row) for row in existing_trade_rows
        )
        if existing_identities != expected_identities:
            # 缺失/身份不符才重建；成本模型变化只告警，不阻断恢复。
            self.store.rewrite_day_trades(authority_last, expected_trade_rows)
            repaired.append(f"trades({len(expected_trade_rows)}行)")
        elif existing_trade_rows != expected_trade_rows:
            logger.warning(
                f"成交投影非身份字段与当前配置重建值不同（{authority_last}），"
                "保留已发布完整行，不回溯改写（可能为成本模型配置漂移）"
            )
        repaired.extend(self._sync_day_nav(state, authority_last, day_trades))
        # T+1 指令帧（合并语义 = 既有优先；缺失行补齐，既有行原样保留）
        instructions_date = self._next_trade_date(authority_last)
        if instructions_date is not None:
            frame = self._rebuild_day_instructions(state, authority_last)
            existing = self.store.load_instructions_frame(instructions_date)
            if frame is not None and self._instruction_frame_differs(existing, frame):
                if self.store.write_instructions_frame(instructions_date, frame):
                    repaired.append(f"instructions({instructions_date})")
        # pending 两队（确定性全量投影；一致 ⇒ store 内跳过写盘）
        pending_buys, pending_sells = self._project_pending_orders(
            self._pending_orders_from_state(state), state.positions
        )
        repaired.extend(self.store.write_pending_queues(pending_buys, pending_sells))
        # 无论本次请求是旧日还是次日，都必须先让权威末代具备稳定发布标记。
        repaired.extend(self._ensure_generation_published(authority_last))
        if repaired:
            logger.warning(f"半提交代受控修复（{authority_last}）：以权威快照重建投影 {repaired}")

    def _sync_day_nav(
        self, state: Any, corrected: str, day_trades: List[Dict[str, Any]]
    ) -> List[str]:
        """恢复当日 NAV：已发布行保持不动，缺失才补建，配置漂移仅告警。"""
        existing_nav_row = self._load_day_nav_row(corrected)
        if day_trades and existing_nav_row is None:
            expected_nav_row = self._project_nav_row_from_state(state, corrected, day_trades)
            self.store.rewrite_day_nav(corrected, expected_nav_row)
            return ["nav"]
        if not day_trades and existing_nav_row is not None:
            self.store.rewrite_day_nav(corrected, None)
            return ["nav(删除无成交日异常行)"]
        if not day_trades or existing_nav_row is None:
            return []
        initial = self.engine_kwargs.get("initial_capital")
        if initial is None:
            return []
        current_nav_row = self._build_nav_row(state.portfolio_values[-1], float(initial), corrected)
        if current_nav_row != existing_nav_row:
            logger.warning(
                f"NAV 投影与当前 initial_capital 重建值不同（{corrected}），"
                "保留已发布历史行，不回溯改写"
            )
        return []

    def _trade_projection_identity(
        self, row: Dict[str, Any]
    ) -> tuple[str, str, str, int, float, float, str]:
        """成交发布行身份键（成本分解不参与身份判断，配置漂移不追溯改写）。"""
        return (
            self._normalize_date_text(row["trade_date"]),
            str(row["ts_code"]),
            str(row["action"]),
            int(row["shares"]),
            float(row["price"]),
            float(row["amount"]),
            str(row["reason"]),
        )

    def _load_day_nav_row(self, corrected: str) -> Optional[Dict[str, Any]]:
        """当日 nav 行自 nav.parquet 读回（无当日行 ⇒ None；跳空语义下 = 合法形态）。"""
        nav = self.store.load_all_nav()
        if nav is None or nav.empty:
            return None
        day = nav[nav["trade_date"].astype(str).str.replace("-", "") == corrected]
        if day.empty:
            return None
        row = day.iloc[-1]
        return {
            "trade_date": corrected,
            "cash": float(row["cash"]),
            "position_value": float(row["position_value"]),
            "total_value": float(row["total_value"]),
            "nav": float(row["nav"]),
        }

    def _rebuild_day_instructions(self, state: Any, corrected: str) -> Optional[pd.DataFrame]:
        """自权威快照重建 T+1 指令帧（半提交恢复用）。

        引擎实例 = 每日新建 + ``restore_state``（与正常驱动同口径）；价格输入 =
        clean/daily 当日帧（仅在需要价格时加载：买入计划/延迟买入存在时；缺失 ⇒
        fail-fast——无法确定指令股数，禁止静默产出不完整指令）。
        """
        engine = self._new_engine({})
        restore_state(engine, state)
        buy_plan = engine.pending_signals.get(self._to_timestamp(corrected))
        delayed_orders = (
            engine.pending_order_manager.get_all_orders()
            if engine.pending_order_manager is not None
            else []
        )
        rows = (buy_plan or {}).get("instruction_rows")
        if rows is not None:
            restored = pd.DataFrame(rows)
            missing = [col for col in INSTRUCTION_COLUMNS if col not in restored.columns]
            if missing:
                raise PaperHostError(f"权威指令行缺少列: {missing}")
            required = ("ts_code", "action", "shares", "price_type", "source_date")
            if restored[list(required)].isna().any().any():
                raise PaperHostError("权威指令行必要字段为空，拒绝恢复")
            return restored.reindex(columns=list(INSTRUCTION_COLUMNS))
        if buy_plan and buy_plan.get("instruction_shares"):
            raise PaperHostError(
                "权威计划缺少 instruction_rows，无法原样恢复指令；拒绝按当前行情/配置重算"
            )
        needs_prices = any(
            (order.get("action") if isinstance(order, dict) else getattr(order, "action", None))
            == "buy"
            for order in delayed_orders
        )
        if needs_prices:
            price_frame = self.loader.load_clean_daily_by_date(corrected)
            if price_frame is None or price_frame.empty:
                raise PaperHostError(
                    f"半提交恢复需重建 {corrected} 的 T+1 买入指令，但 clean/daily 当日行情缺失"
                    "（无法确定指令股数）——请先补齐数据再重试"
                )
            engine.price_data_cache = price_frame
        else:
            engine.price_data_cache = pd.DataFrame(columns=["trade_date", "ts_code", "close"])
        frame, _ = self._build_instructions(engine, corrected, self._to_timestamp(corrected))
        # 半提交恢复必须从权威计划恢复行集/股数/执行属性；重算只用于卖单等非计划行。
        shares_map = (buy_plan or {}).get("instruction_shares") or {}
        if shares_map:
            price_type = str(
                (buy_plan or {}).get("instruction_price_type")
                or self.trading_config.buy_price
            )
            if frame is None or frame.empty:
                frame = pd.DataFrame(columns=list(INSTRUCTION_COLUMNS))
            existing = set(
                frame.loc[frame["action"] == "buy", "ts_code"].astype(str)
            )
            reason = str((buy_plan or {}).get("instruction_reason") or "信号生成")
            meta_all = (buy_plan or {}).get("completion_meta") or {}
            signals = (buy_plan or {}).get("signals") or {}
            for code, shares in shares_map.items():
                code = str(code)
                if int(shares) <= 0:
                    continue
                if code in existing:
                    mask = (frame["action"] == "buy") & (frame["ts_code"].astype(str) == code)
                    frame.loc[mask, "shares"] = int(shares)
                    frame.loc[mask, "price_type"] = price_type
                    continue
                meta = meta_all.get(code) or {}
                row = _new_instruction_row(
                    code,
                    "buy",
                    int(shares),
                    price_type,
                    reason,
                    corrected,
                    target_weight=float(signals.get(code) or 0.0),
                    original_signal_date=corrected,
                    desired_position_count=int((buy_plan or {}).get("desired_position_count") or 0),
                    retry_attempt=int(meta.get("attempts") or 0),
                    replacement_slot_code=code,
                )
                frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        return frame

    @staticmethod
    def _instruction_frame_differs(existing: Optional[pd.DataFrame], frame: pd.DataFrame) -> bool:
        """指令帧一致性比对（行记录级；既有 None/空 vs 新帧空 ⇒ 一致）。"""
        if frame is None or len(frame) == 0:
            return existing is not None and len(existing) > 0
        if existing is None or len(existing) == 0:
            return True
        return existing.reset_index(drop=True).to_dict("records") != frame.reset_index(
            drop=True
        ).to_dict("records")
