# -*- coding: utf-8 -*-
"""PaperStateMixin：src/lazybull/paper/storage.py 拆分出的 save_account_state, load_account_state, _load_account_state_remote, save_stop_loss_state, load_stop_loss_state, save_last_trade_date, load_last_trade_date, save_strategy_state, load_strategy_state。"""

from ..models import AccountState
from ..models import Position
from loguru import logger
from typing import Optional
import json

class PaperStateMixin:
    def save_account_state(self, state: AccountState) -> None:
        """保存账户状态
        
        Args:
            state: 账户状态
        """
        if self._is_remote:
            logger.warning("远端只读模式，跳过 save_account_state")
            return

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
                'buy_date': pos.buy_date,
                'buy_pnl_price': getattr(pos, 'buy_pnl_price', 0.0),
                'buy_atr_pct': getattr(pos, 'buy_atr_pct', 0.0),
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存账户状态: {file_path}")

    def load_account_state(self) -> Optional[AccountState]:
        """读取账户状态
        
        Returns:
            账户状态，不存在返回None
        """
        if self._is_remote:
            return self._load_account_state_remote()

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
                buy_date=pos_dict['buy_date'],
                buy_pnl_price=pos_dict.get('buy_pnl_price', 0.0),
                buy_atr_pct=pos_dict.get('buy_atr_pct', 0.0),
            )
        
        state = AccountState(
            cash=state_dict['cash'],
            positions=positions,
            last_update=state_dict.get('last_update', '')
        )
        if self.verbose:
            logger.info(f"读取账户状态: {file_path}")
        return state

    def _load_account_state_remote(self) -> Optional[AccountState]:
        """通过 SMB 远端读取账户状态。"""
        if self._smb_reader is None:
            return None
        try:
            state_dict = self._smb_reader.read_json("state/account.json")
            if not state_dict:
                logger.warning("SMB 远端账户状态文件为空")
                return None
        except FileNotFoundError as exc:
            logger.warning(f"SMB 远端账户状态文件不存在: {exc}")
            return None
        except ConnectionError as exc:
            logger.warning(f"SMB 连接远端失败，无法读取账户状态: {exc}")
            return None
        except ValueError as exc:
            logger.warning(f"SMB 远端账户状态文件格式错误: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"SMB 读取远端账户状态未知错误: {type(exc).__name__}: {exc}")
            return None

        positions = {}
        for ts_code, pos_dict in state_dict.get('positions', {}).items():
            positions[ts_code] = Position(
                ts_code=pos_dict['ts_code'],
                shares=pos_dict['shares'],
                buy_price=pos_dict['buy_price'],
                buy_cost=pos_dict['buy_cost'],
                buy_date=pos_dict['buy_date'],
                buy_pnl_price=pos_dict.get('buy_pnl_price', 0.0),
                buy_atr_pct=pos_dict.get('buy_atr_pct', 0.0),
            )

        state = AccountState(
            cash=state_dict['cash'],
            positions=positions,
            last_update=state_dict.get('last_update', '')
        )
        if self.verbose:
            logger.info("SMB 远端读取账户状态成功")
        return state
        return state

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

    def save_last_trade_date(self, trade_date: str) -> None:
        """保存最近执行交易的日期（供 trade next 等命令推算下一交易日）"""
        file_path = self.state_path / "last_trade_date.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'last_trade_date': trade_date}, f, ensure_ascii=False)

    def load_last_trade_date(self) -> Optional[str]:
        """读取最近执行交易的日期，不存在返回 None"""
        file_path = self.state_path / "last_trade_date.json"
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('last_trade_date')

    def save_strategy_state(self, state: dict) -> None:
        """保存纸面交易的策略运行状态。"""
        file_path = self.state_path / "strategy_state.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.debug(f"保存策略状态: {file_path}")

    def load_strategy_state(self) -> dict:
        """读取纸面交易的策略运行状态。"""
        file_path = self.state_path / "strategy_state.json"

        if not file_path.exists():
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
