"""纸面交易存储模块"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .models import AccountState, Fill, NAVRecord, PendingSell, Position, TargetWeight


class PaperStorage:
    """纸面交易存储
    
    负责持久化和读取纸面交易的各类数据
    """
    
    def __init__(self, root_path: str = "./data/paper"):
        """初始化纸面交易存储
        
        Args:
            root_path: 数据根目录
        """
        self.root_path = Path(root_path)
        self.pending_path = self.root_path / "pending"
        self.state_path = self.root_path / "state"
        self.trades_path = self.root_path / "trades"
        self.nav_path = self.root_path / "nav"
        self.runs_path = self.root_path / "runs"
        self.pending_sells_path = self.root_path / "pending_sells"
        
        # 确保目录存在
        for path in [self.pending_path, self.state_path, self.trades_path, 
                     self.nav_path, self.runs_path, self.pending_sells_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"纸面交易存储初始化完成，根目录: {self.root_path}")
    
    def save_pending_weights(self, trade_date: str, targets: List[TargetWeight]) -> None:
        """保存待执行的目标权重
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            targets: 目标权重列表
        """
        file_path = self.pending_path / f"{trade_date}.parquet"
        
        # 转换为DataFrame
        data = []
        for target in targets:
            data.append({
                'ts_code': target.ts_code,
                'target_weight': target.target_weight,
                'reason': target.reason
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
        logger.info(f"保存待执行目标权重: {file_path} ({len(targets)} 条)")
    
    def load_pending_weights(self, trade_date: str) -> Optional[List[TargetWeight]]:
        """读取待执行的目标权重
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            目标权重列表，不存在返回None
        """
        file_path = self.pending_path / f"{trade_date}.parquet"
        
        if not file_path.exists():
            logger.info(f"待执行目标权重文件不存在: {file_path}")
            return None
        
        df = pd.read_parquet(file_path)
        targets = []
        for _, row in df.iterrows():
            targets.append(TargetWeight(
                ts_code=row['ts_code'],
                target_weight=row['target_weight'],
                reason=row.get('reason', '信号生成')
            ))
        
        logger.info(f"读取待执行目标权重: {file_path} ({len(targets)} 条)")
        return targets
    
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
        logger.info(f"读取净值记录: {file_path} ({len(df)} 条)")
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
                attempts=item.get('attempts', 0)
            ))
        
        logger.info(f"读取延迟卖出队列: {file_path} ({len(pending_sells)} 条)")
        return pending_sells
    
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
        
        logger.info(f"读取全局配置: {file_path}")
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
