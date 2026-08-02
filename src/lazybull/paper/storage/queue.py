# -*- coding: utf-8 -*-
"""PaperQueueMixin：src/lazybull/paper/storage.py 拆分出的 save_pending_sells, load_pending_sells, save_pending_buys, load_pending_buys, save_instructions, load_instructions, save_ranked_candidates, load_ranked_candidates, find_pending_instructions, find_latest_t0。"""

from ..models import PendingBuy
from ..models import PendingSell
from ..models import TradeInstruction
from loguru import logger
from typing import List
from typing import Optional
import json
import pandas as pd

class PaperQueueMixin:
    def save_pending_sells(self, pending_sells: List[PendingSell]) -> None:
        """保存延迟卖出队列
        
        Args:
            pending_sells: 延迟卖出订单列表
        """
        file_path = self.pending_sells_path / "pending_sells.json"
        
        # 转换为字典列表
        data = []
        for ps in pending_sells:
            data.append({
                'ts_code': ps.ts_code,
                'shares': ps.shares,
                'target_weight': ps.target_weight,
                'reason': ps.reason,
                'create_date': ps.create_date,
                'attempts': ps.attempts
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存延迟卖出队列: {file_path} ({len(pending_sells)} 条)")

    def load_pending_sells(self) -> List[PendingSell]:
        """读取延迟卖出队列
        
        Returns:
            延迟卖出订单列表，不存在返回空列表
        """
        file_path = self.pending_sells_path / "pending_sells.json"
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pending_sells = []
        for item in data:
            pending_sells.append(PendingSell(
                ts_code=item['ts_code'],
                shares=item['shares'],
                target_weight=item['target_weight'],
                reason=item['reason'],
                create_date=item['create_date'],
                attempts=item.get('attempts', 0),
                last_attempt_date=item.get('last_attempt_date', '')
            ))
        if self.verbose:
            logger.debug(f"读取延迟卖出队列: {file_path} ({len(pending_sells)} 条)")
        return pending_sells

    def save_pending_buys(self, pending_buys: List[PendingBuy]) -> None:
        """保存延迟买入队列（补位计划）
        
        Args:
            pending_buys: 延迟买入订单列表
        """
        file_path = self.pending_buys_path / "pending_buys.json"
        
        # 转换为字典列表
        data = []
        for pb in pending_buys:
            data.append({
                'ts_code': pb.ts_code,
                'target_weight': pb.target_weight,
                'reason': pb.reason,
                'create_date': pb.create_date,
                'attempts': pb.attempts,
                'last_attempt_date': pb.last_attempt_date,
                'original_signal_date': pb.original_signal_date
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存延迟买入队列: {file_path} ({len(pending_buys)} 条)")

    def load_pending_buys(self) -> List[PendingBuy]:
        """读取延迟买入队列（补位计划）
        
        Returns:
            延迟买入订单列表，不存在返回空列表
        """
        file_path = self.pending_buys_path / "pending_buys.json"
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pending_buys = []
        for item in data:
            pending_buys.append(PendingBuy(
                ts_code=item['ts_code'],
                target_weight=item['target_weight'],
                reason=item['reason'],
                create_date=item['create_date'],
                attempts=item.get('attempts', 0),
                last_attempt_date=item.get('last_attempt_date', ''),
                original_signal_date=item.get('original_signal_date', '')
            ))
        if self.verbose:
            logger.debug(f"读取延迟买入队列: {file_path} ({len(pending_buys)} 条)")
        return pending_buys

    def save_instructions(self, trade_date: str, instructions: List[TradeInstruction]) -> None:
        """保存交易指令列表
        
        Args:
            trade_date: 交易日期 YYYYMMDD（T1执行日期）
            instructions: 交易指令列表
        """
        file_path = self.instructions_path / f"{trade_date}.parquet"
        
        # 转换为DataFrame
        data = []
        for inst in instructions:
            data.append({
                'ts_code': inst.ts_code,
                'action': inst.action,
                'shares': inst.shares,
                'price_type': inst.price_type,
                'reason': inst.reason,
                'source_date': inst.source_date,
                'target_weight': inst.target_weight,
                'original_signal_date': inst.original_signal_date,
                'desired_position_count': inst.desired_position_count,
                'retry_attempt': inst.retry_attempt,
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
        logger.debug(f"保存交易指令: {file_path} ({len(instructions)} 条)")

    def load_instructions(self, trade_date: str) -> Optional[List[TradeInstruction]]:
        """读取交易指令列表
        
        Args:
            trade_date: 交易日期 YYYYMMDD（T1执行日期）
            
        Returns:
            交易指令列表，不存在返回None
        """
        file_path = self.instructions_path / f"{trade_date}.parquet"
        
        if not file_path.exists():
            logger.debug(f"交易指令文件不存在: {file_path}")
            return None
        
        df = pd.read_parquet(file_path)
        instructions = []
        for _, row in df.iterrows():
            instructions.append(TradeInstruction(
                ts_code=row['ts_code'],
                action=row['action'],
                shares=int(row['shares']),
                price_type=row['price_type'],
                reason=row['reason'],
                source_date=row['source_date'],
                target_weight=row.get('target_weight', 0.0),
                original_signal_date=row.get('original_signal_date', ''),
                desired_position_count=int(row.get('desired_position_count', 0) or 0),
                retry_attempt=int(row.get('retry_attempt', 0) or 0),
            ))
        
        logger.debug(f"读取交易指令: {file_path} ({len(instructions)} 条)")
        return instructions

    def save_ranked_candidates(self, ranked_candidates: List[tuple], signal_date: str) -> None:
        """保存 T0 生成的排序候选列表，供 T1 恢复使用
        
        Args:
            ranked_candidates: [(ts_code, ml_score), ...] 列表
            signal_date: 信号生成日期 YYYYMMDD
        """
        file_path = self.state_path / "ranked_candidates.json"
        data = {
            "signal_date": signal_date,
            "candidates": ranked_candidates
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"保存 ranked_candidates: signal_date={signal_date}, count={len(ranked_candidates)}")

    def load_ranked_candidates(self) -> Optional[tuple]:
        """加载上一个 T0 生成的排序候选列表
        
        Returns:
            (ranked_candidates, signal_date) 元组，不存在返回 None
            其中 ranked_candidates 是 [(ts_code, ml_score), ...] 列表
        """
        file_path = self.state_path / "ranked_candidates.json"
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            candidates = data.get("candidates", [])
            signal_date = data.get("signal_date", "")
            logger.debug(f"加载 ranked_candidates: signal_date={signal_date}, count={len(candidates)}")
            return (candidates, signal_date)
        except Exception as exc:
            logger.warning(f"加载 ranked_candidates 失败: {exc}")
            return None

    def find_pending_instructions(self, before_date: str) -> Optional[tuple]:
        """查找 <= before_date 且未执行的最新交易指令

        Args:
            before_date: 截止日期 YYYYMMDD（包含）

        Returns:
            (instruction_date, instructions) 元组，不存在返回 None
        """
        # 扫描 instructions/ 目录下所有 .parquet 文件
        instruction_files = sorted(self.instructions_path.glob("*.parquet"))

        if not instruction_files:
            return None

        # 从最新到最旧遍历，找第一个 <= before_date 且未执行的
        for f in reversed(instruction_files):
            inst_date = f.stem  # 文件名即日期 YYYYMMDD
            if inst_date > before_date:
                continue
            # 检查是否已执行（有对应的 t1 run record）
            if self.check_run_exists("t1", inst_date):
                continue
            # 找到未执行的指令
            instructions = self.load_instructions(inst_date)
            if instructions:
                return (inst_date, instructions)

        return None

    def find_latest_t0(self) -> Optional[str]:
        """查找最新的T0运行记录日期

        Returns:
            最新T0日期 YYYYMMDD，不存在返回None
        """
        t0_files = sorted(self.runs_path.glob("t0_*.json"))
        if not t0_files:
            return None
        # 文件名格式: t0_YYYYMMDD.json，取最后一个
        return t0_files[-1].stem.split('_')[1]
