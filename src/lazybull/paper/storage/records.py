# -*- coding: utf-8 -*-
"""PaperRecordMixin：src/lazybull/paper/storage.py 拆分出的 append_trade, load_all_trades, append_nav, load_all_nav, save_run_record, check_run_exists, save_rebalance_state, load_rebalance_state。"""

from ..models import Fill
from ..models import NAVRecord
from loguru import logger
from typing import Optional
import json
import pandas as pd

class PaperRecordMixin:
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
        if self._is_remote:
            return self._smb_reader.read_parquet("nav/nav.parquet") if self._smb_reader else None

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
        
        logger.debug(f"保存执行记录: {file_path}")

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
        if self._is_remote:
            if self._smb_reader is None:
                return None
            try:
                state = self._smb_reader.read_json("runs/rebalance_state.json")
                return state if state else None
            except Exception as exc:
                logger.warning(f"SMB 读取远端调仓状态失败: {exc}")
                return None

        file_path = self.runs_path / "rebalance_state.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        return state
