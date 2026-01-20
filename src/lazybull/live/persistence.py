"""简单的JSON持久化模块

用于纸面交易的状态持久化，支持订单、持仓、账户和待执行信号的保存与加载。
适用于本地验证，生产环境应使用数据库替代。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger


class SimplePersistence:
    """基于JSON的简单持久化
    
    持久化内容包括：
    - orders: 订单历史
    - positions: 当前持仓
    - account: 账户状态（现金、总资产等）
    - pending_signals: 待执行信号（T日生成，T+1执行）
    
    Attributes:
        file_path: JSON文件路径
        state: 当前状态字典
    """
    
    def __init__(self, file_path: str = "data/trading_state.json"):
        """初始化持久化模块
        
        Args:
            file_path: JSON文件路径，默认为 data/trading_state.json
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_initialize()
        logger.info(f"持久化模块初始化完成，文件路径: {self.file_path}")
    
    def _load_or_initialize(self) -> Dict[str, Any]:
        """加载或初始化状态
        
        Returns:
            状态字典
        """
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"加载已有状态，订单数: {len(state.get('orders', []))}")
                return state
            except Exception as e:
                logger.warning(f"加载状态失败: {e}，使用空状态")
                return self._empty_state()
        else:
            logger.info("初次运行，创建空状态")
            return self._empty_state()
    
    def _empty_state(self) -> Dict[str, Any]:
        """创建空状态
        
        Returns:
            空状态字典
        """
        return {
            "orders": [],
            "positions": {},
            "account": {
                "cash": 0.0,
                "total_value": 0.0,
                "update_time": None
            },
            "pending_signals": []
        }
    
    def _save(self):
        """保存状态到文件"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            logger.debug(f"状态已保存到 {self.file_path}")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            raise
    
    def load_state(self) -> Dict[str, Any]:
        """加载完整状态
        
        Returns:
            当前状态字典
        """
        return self.state.copy()
    
    def save_order(self, order: Dict[str, Any]):
        """保存订单
        
        Args:
            order: 订单字典，包含 local_order_id, symbol, side, qty, price 等字段
        """
        self.state["orders"].append(order)
        self._save()
        logger.debug(f"保存订单: {order.get('local_order_id')}")
    
    def update_order_status(self, order_id: str, status: str, **kwargs):
        """更新订单状态
        
        Args:
            order_id: 订单ID
            status: 新状态
            **kwargs: 其他需要更新的字段
        """
        for order in self.state["orders"]:
            if order.get("local_order_id") == order_id:
                order["status"] = status
                order.update(kwargs)
                self._save()
                logger.debug(f"更新订单 {order_id} 状态为 {status}")
                return
        logger.warning(f"未找到订单 {order_id}")
    
    def save_positions(self, positions: Dict[str, Dict[str, Any]]):
        """保存持仓
        
        Args:
            positions: 持仓字典，key为股票代码，value为持仓信息（qty, cost_price等）
        """
        self.state["positions"] = positions
        self._save()
        logger.debug(f"保存持仓，当前持仓数: {len(positions)}")
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取当前持仓
        
        Returns:
            持仓字典
        """
        return self.state.get("positions", {}).copy()
    
    def save_account(self, cash: float, total_value: float):
        """保存账户状态
        
        Args:
            cash: 可用现金
            total_value: 账户总值
        """
        self.state["account"] = {
            "cash": cash,
            "total_value": total_value,
            "update_time": datetime.now().isoformat()
        }
        self._save()
        logger.debug(f"保存账户状态，现金: {cash:.2f}, 总值: {total_value:.2f}")
    
    def get_account(self) -> Dict[str, Any]:
        """获取账户状态
        
        Returns:
            账户状态字典
        """
        return self.state.get("account", {}).copy()
    
    def add_pending_signals(self, trade_date: str, signals: List[Dict[str, Any]]):
        """添加待执行信号
        
        Args:
            trade_date: 信号生成日期（YYYYMMDD格式）
            signals: 信号列表，每个信号包含 symbol, weight, signal_meta 等字段
        """
        pending = {
            "trade_date": trade_date,
            "signals": signals,
            "create_time": datetime.now().isoformat(),
            "executed": False
        }
        self.state["pending_signals"].append(pending)
        self._save()
        logger.info(f"添加待执行信号，日期: {trade_date}, 信号数: {len(signals)}")
    
    def pop_pending_signals(self, trade_date: str) -> Optional[List[Dict[str, Any]]]:
        """弹出并标记待执行信号
        
        Args:
            trade_date: 信号生成日期（YYYYMMDD格式）
        
        Returns:
            信号列表，如果未找到则返回None
        """
        for pending in self.state["pending_signals"]:
            if pending["trade_date"] == trade_date and not pending["executed"]:
                pending["executed"] = True
                pending["execute_time"] = datetime.now().isoformat()
                self._save()
                logger.info(f"弹出待执行信号，日期: {trade_date}, 信号数: {len(pending['signals'])}")
                return pending["signals"]
        logger.warning(f"未找到未执行的信号，日期: {trade_date}")
        return None
    
    def get_pending_signals(self, executed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取待执行信号列表
        
        Args:
            executed: 过滤条件，True=已执行, False=未执行, None=全部
        
        Returns:
            待执行信号列表
        """
        signals = self.state.get("pending_signals", [])
        if executed is None:
            return signals.copy()
        return [s for s in signals if s.get("executed") == executed]
    
    def get_orders(self, date_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取订单列表
        
        Args:
            date_filter: 日期过滤（YYYYMMDD格式），None表示全部
        
        Returns:
            订单列表
        """
        orders = self.state.get("orders", [])
        if date_filter is None:
            return orders.copy()
        return [o for o in orders if o.get("trade_date") == date_filter]
