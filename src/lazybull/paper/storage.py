"""纸面交易存储模块"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .models import AccountState, Fill, NAVRecord, PendingBuy, PendingSell, Position, TargetWeight, TradeInstruction


class PaperStorage:
    """纸面交易存储
    
    负责持久化和读取纸面交易的各类数据
    """
    
    def __init__(self, root_path: str = "./data/paper", verbose: bool = False):
        """初始化纸面交易存储
        
        Args:
            root_path: 数据根目录
            verbose: 是否输出详细日志
        """
        self.root_path = Path(root_path)
        self.state_path = self.root_path / "state"
        self.trades_path = self.root_path / "trades"
        self.nav_path = self.root_path / "nav"
        self.runs_path = self.root_path / "runs"
        self.pending_sells_path = self.root_path / "pending_sells"
        self.pending_buys_path = self.root_path / "pending_buys"
        self.instructions_path = self.root_path / "instructions"
        self.verbose = verbose
        
        # 确保目录存在
        for path in [self.state_path, self.trades_path, 
                     self.nav_path, self.runs_path, self.pending_sells_path, self.pending_buys_path,
                     self.instructions_path]:
            path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"纸面交易存储初始化完成，根目录: {self.root_path}")
    
    def save_account_state(self, state: AccountState) -> None:
        """保存账户状态
        
        Args:
            state: 账户状态
        """
        file_path = self.state_path / "account.json"
        
        # 转换为字典
        state_dict = {
            'cash': state.cash,
            'last_update': state.last_update,
            'positions': {}
        }
        
        for ts_code, pos in state.positions.items():
            state_dict['positions'][ts_code] = {
                'ts_code': pos.ts_code,
                'shares': pos.shares,
                'buy_price': pos.buy_price,
                'buy_cost': pos.buy_cost,
                'buy_date': pos.buy_date
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存账户状态: {file_path}")
    
    def load_account_state(self) -> Optional[AccountState]:
        """读取账户状态
        
        Returns:
            账户状态，不存在返回None
        """
        file_path = self.state_path / "account.json"
        
        if not file_path.exists():
            logger.warning(f"账户状态文件不存在: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # 重建持仓
        positions = {}
        for ts_code, pos_dict in state_dict.get('positions', {}).items():
            positions[ts_code] = Position(
                ts_code=pos_dict['ts_code'],
                shares=pos_dict['shares'],
                buy_price=pos_dict['buy_price'],
                buy_cost=pos_dict['buy_cost'],
                buy_date=pos_dict['buy_date']
            )
        
        state = AccountState(
            cash=state_dict['cash'],
            positions=positions,
            last_update=state_dict.get('last_update', '')
        )
        if self.verbose:
            logger.info(f"读取账户状态: {file_path}")
        return state
    
    def append_trade(self, fill: Fill) -> None:
        """追加成交记录
        
        Args:
            fill: 成交记录
        """
        file_path = self.trades_path / "trades.parquet"
        
        # 新记录
        new_data = pd.DataFrame([{
            'trade_date': fill.trade_date,
            'ts_code': fill.ts_code,
            'action': fill.action,
            'shares': fill.shares,
            'price': fill.price,
            'amount': fill.amount,
            'commission': fill.commission,
            'stamp_tax': fill.stamp_tax,
            'slippage': fill.slippage,
            'total_cost': fill.total_cost,
            'reason': fill.reason
        }])
        
        # 追加到现有文件
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            df = new_data
        
        df.to_parquet(file_path, index=False)
        logger.debug(f"追加成交记录: {file_path}")
    
    def load_all_trades(self) -> Optional[pd.DataFrame]:
        """读取所有成交记录
        
        Returns:
            成交记录DataFrame，不存在返回None
        """
        file_path = self.trades_path / "trades.parquet"
        
        if not file_path.exists():
            logger.warning(f"成交记录文件不存在: {file_path}")
            return None
        
        df = pd.read_parquet(file_path)
        logger.info(f"读取成交记录: {file_path} ({len(df)} 条)")
        return df
    
    def append_nav(self, nav_record: NAVRecord) -> None:
        """追加净值记录
        
        Args:
            nav_record: 净值记录
        """
        file_path = self.nav_path / "nav.parquet"
        
        # 新记录
        new_data = pd.DataFrame([{
            'trade_date': nav_record.trade_date,
            'cash': nav_record.cash,
            'position_value': nav_record.position_value,
            'total_value': nav_record.total_value,
            'nav': nav_record.nav
        }])
        
        # 追加到现有文件
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            df = new_data
        
        df.to_parquet(file_path, index=False)
        logger.debug(f"追加净值记录: {file_path}")
    
    def load_all_nav(self) -> Optional[pd.DataFrame]:
        """读取所有净值记录
        
        Returns:
            净值记录DataFrame，不存在返回None
        """
        file_path = self.nav_path / "nav.parquet"
        
        if not file_path.exists():
            logger.warning(f"净值记录文件不存在: {file_path}")
            return None
        
        df = pd.read_parquet(file_path)
        #logger.info(f"读取净值记录: {file_path} ({len(df)} 条)")
        return df
    
    def save_run_record(self, run_type: str, trade_date: str, record: dict) -> None:
        """保存执行记录（用于幂等性检查）
        
        Args:
            run_type: 运行类型 "t0" 或 "t1"
            trade_date: 交易日期 YYYYMMDD
            record: 记录字典（包含参数、时间戳、统计信息等）
        """
        file_path = self.runs_path / f"{run_type}_{trade_date}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存执行记录: {file_path}")
    
    def check_run_exists(self, run_type: str, trade_date: str) -> bool:
        """检查执行记录是否存在
        
        Args:
            run_type: 运行类型 "t0" 或 "t1"
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            True 如果记录存在
        """
        file_path = self.runs_path / f"{run_type}_{trade_date}.json"
        return file_path.exists()
    
    def save_rebalance_state(self, state: dict) -> None:
        """保存调仓状态（记录上次调仓日期）
        
        Args:
            state: 调仓状态字典 {"last_rebalance_date": "YYYYMMDD", ...}
        """
        file_path = self.runs_path / "rebalance_state.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"保存调仓状态: {file_path}")
    
    def load_rebalance_state(self) -> Optional[dict]:
        """读取调仓状态
        
        Returns:
            调仓状态字典，不存在返回None
        """
        file_path = self.runs_path / "rebalance_state.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        return state
    
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
            logger.info(f"读取延迟卖出队列: {file_path} ({len(pending_sells)} 条)")
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
            logger.info(f"读取延迟买入队列: {file_path} ({len(pending_buys)} 条)")
        return pending_buys
    
    def save_config(self, config: dict) -> None:
        """保存全局配置
        
        Args:
            config: 配置字典
        """
        file_path = self.root_path / "config.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存全局配置: {file_path}")
    
    def load_config(self) -> Optional[dict]:
        """读取全局配置
        
        Returns:
            配置字典，不存在返回None
        """
        file_path = self.root_path / "config.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        #logger.info(f"读取全局配置: {file_path}")
        return config
    
    def save_stop_loss_state(self, state: dict) -> None:
        """保存止损监控状态
        
        Args:
            state: 止损状态字典
        """
        file_path = self.state_path / "stop_loss_state.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"保存止损状态: {file_path}")
    
    def load_stop_loss_state(self) -> Optional[dict]:
        """读取止损监控状态
        
        Returns:
            止损状态字典，不存在返回None
        """
        file_path = self.state_path / "stop_loss_state.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        return state
    
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
                'original_signal_date': inst.original_signal_date
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
        logger.info(f"保存交易指令: {file_path} ({len(instructions)} 条)")
    
    def load_instructions(self, trade_date: str) -> Optional[List[TradeInstruction]]:
        """读取交易指令列表
        
        Args:
            trade_date: 交易日期 YYYYMMDD（T1执行日期）
            
        Returns:
            交易指令列表，不存在返回None
        """
        file_path = self.instructions_path / f"{trade_date}.parquet"
        
        if not file_path.exists():
            logger.info(f"交易指令文件不存在: {file_path}")
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
                original_signal_date=row.get('original_signal_date', '')
            ))
        
        logger.info(f"读取交易指令: {file_path} ({len(instructions)} 条)")
        return instructions
    
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

    def reset_t0(self, t0_date: Optional[str] = None) -> dict:
        """重置最新T0日及之后的所有数据，允许从该T0日重新执行

        若未指定 t0_date，则自动查找最新的T0记录。
        内部调用 truncate_since(t0_date) 清理T0及之后的所有运行记录、
        成交、净值、交易指令、延迟订单等，同时回滚账户 last_update 和调仓状态。

        Args:
            t0_date: T0日期 YYYYMMDD（可选，默认自动查找最新）

        Returns:
            操作结果统计字典
        """
        # 自动查找最新T0日期
        if t0_date is None:
            t0_date = self.find_latest_t0()

        stats = {
            't0_date': t0_date,
        }

        if t0_date is None:
            logger.warning("未找到任何T0运行记录")
            return stats

        # 回滚账户 last_update 到 T0 之前最近的运行日期
        account_state = self.load_account_state()
        if account_state and account_state.last_update >= t0_date:
            # 找到 t0_date 之前最近的 t1 运行记录日期
            t1_files = sorted(self.runs_path.glob("t1_*.json"))
            prev_date = ""
            for f in reversed(t1_files):
                file_date = f.stem.split('_')[1]
                if file_date < t0_date:
                    prev_date = file_date
                    break

            old_last_update = account_state.last_update
            account_state.last_update = prev_date
            self.save_account_state(account_state)
            logger.info(
                f"回滚账户 last_update: {old_last_update} -> "
                f"{prev_date if prev_date else '(空)'}"
            )

        # 使用 truncate_since 清理 T0 及之后的所有数据
        self.truncate_since(t0_date)

        return stats

    def truncate_since(self, cut_off_date: str) -> None:
        """截断/清理从指定日期开始的所有数据（包含该日期）
        
        用于手工修正账户后，清理 cut-off 日期及之后的所有记录，
        以便从该日期重新运行并保持一致性。
        
        清理范围：
        - trades.parquet: 删除 trade_date >= cut_off_date 的行
        - nav.parquet: 删除 trade_date >= cut_off_date 的行
        - runs/: 删除日期 >= cut_off_date 的 t0_*.json 和 t1_*.json 文件
        - instructions/: 删除日期 >= cut_off_date 的指令文件
        - pending_buys.json 和 pending_sells.json: 清空
        - rebalance_state.json: 按规则回滚
        
        Args:
            cut_off_date: 截断日期 YYYYMMDD（包含此日期）
        """
        logger.info("=" * 80)
        logger.info(f"开始清理数据：删除 >= {cut_off_date} 的所有记录")
        logger.info("=" * 80)
        
        # 1. 清理 trades.parquet
        trades_file = self.trades_path / "trades.parquet"
        if trades_file.exists():
            df = pd.read_parquet(trades_file)
            original_count = len(df)
            df = df[df['trade_date'] < cut_off_date]
            new_count = len(df)
            
            if new_count < original_count:
                df.to_parquet(trades_file, index=False)
                logger.info(f"清理成交记录: {original_count} -> {new_count} 条（删除 {original_count - new_count} 条）")
            else:
                logger.info(f"成交记录无需清理（无 >= {cut_off_date} 的记录）")
        else:
            logger.info("成交记录文件不存在，跳过")
        
        # 2. 清理 nav.parquet
        nav_file = self.nav_path / "nav.parquet"
        if nav_file.exists():
            df = pd.read_parquet(nav_file)
            original_count = len(df)
            df = df[df['trade_date'] < cut_off_date]
            new_count = len(df)
            
            if new_count < original_count:
                df.to_parquet(nav_file, index=False)
                logger.info(f"清理净值记录: {original_count} -> {new_count} 条（删除 {original_count - new_count} 条）")
            else:
                logger.info(f"净值记录无需清理（无 >= {cut_off_date} 的记录）")
        else:
            logger.info("净值记录文件不存在，跳过")
        
        # 3. 清理 runs/ 目录
        deleted_runs = 0
        for run_file in self.runs_path.glob("*.json"):
            if run_file.name == "rebalance_state.json":
                continue  # rebalance_state 单独处理
            
            # 提取日期：t0_YYYYMMDD.json 或 t1_YYYYMMDD.json
            parts = run_file.stem.split('_')
            if len(parts) == 2 and parts[0] in ['t0', 't1']:
                file_date = parts[1]
                if file_date >= cut_off_date:
                    run_file.unlink()
                    deleted_runs += 1
        
        if deleted_runs > 0:
            logger.info(f"清理运行记录: 删除 {deleted_runs} 个文件")
        else:
            logger.info("运行记录无需清理")
        
        # 4. 清理 instructions/ 目录
        deleted_instructions = 0
        for inst_file in self.instructions_path.glob("*.parquet"):
            # 提取日期：YYYYMMDD.parquet
            file_date = inst_file.stem
            if file_date >= cut_off_date:
                inst_file.unlink()
                deleted_instructions += 1
        
        if deleted_instructions > 0:
            logger.info(f"清理交易指令: 删除 {deleted_instructions} 个文件")
        else:
            logger.info("交易指令无需清理")
        
        # 5. 清空 pending_buys.json
        pending_buys_file = self.pending_buys_path / "pending_buys.json"
        if pending_buys_file.exists():
            with open(pending_buys_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info("清空延迟买入队列")
        else:
            logger.info("延迟买入队列文件不存在，跳过")
        
        # 6. 清空 pending_sells.json
        pending_sells_file = self.pending_sells_path / "pending_sells.json"
        if pending_sells_file.exists():
            with open(pending_sells_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info("清空延迟卖出队列")
        else:
            logger.info("延迟卖出队列文件不存在，跳过")
        
        # 7. 回滚 rebalance_state.json
        rebalance_state = self.load_rebalance_state()
        if rebalance_state and rebalance_state.get('last_rebalance_date', '') >= cut_off_date:
            # 需要回滚：找到 cut_off 之前最近的 t0 记录
            t0_files = sorted([f for f in self.runs_path.glob("t0_*.json")])
            rollback_date = None
            
            for t0_file in reversed(t0_files):
                file_date = t0_file.stem.split('_')[1]
                if file_date < cut_off_date:
                    rollback_date = file_date
                    break
            
            if rollback_date:
                rebalance_state['last_rebalance_date'] = rollback_date
                self.save_rebalance_state(rebalance_state)
                logger.info(f"回滚调仓状态: {rebalance_state.get('last_rebalance_date')} -> {rollback_date}")
            else:
                # cut_off 之前没有 t0 记录，删除 rebalance_state
                rebalance_file = self.runs_path / "rebalance_state.json"
                if rebalance_file.exists():
                    rebalance_file.unlink()
                logger.info("删除调仓状态（无有效的 t0 记录可回滚）")
        else:
            logger.info("调仓状态无需回滚")
        
        logger.info("=" * 80)
        logger.info("数据清理完成")
        logger.info("=" * 80)
