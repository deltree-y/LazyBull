"""简单持久化模块（基于 JSON）

用于保存订单、持仓、账户与待执行的 T+1 信号。
生产环境请替换为数据库实现。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class SimplePersistence:
    """简单的 JSON 持久化实现
    
    用于 T+1 纸面交易的状态保存与恢复。
    所有数据存储在单个 JSON 文件中。
    
    数据结构：
    {
        "account": {
            "cash": float,
            "initial_cash": float,
            "last_update": str
        },
        "positions": {
            "股票代码": {
                "code": str,
                "shares": int,
                "avg_cost": float,
                "last_price": float,
                "update_time": str
            }
        },
        "orders": [
            {
                "order_id": str,
                "code": str,
                "direction": str,  # "buy" or "sell"
                "shares": int,
                "price": float,
                "amount": float,
                "cost": float,
                "status": str,     # "filled"
                "create_time": str,
                "fill_time": str
            }
        ],
        "pending_signals": [
            {
                "trade_date": str,      # 信号生成日期 YYYYMMDD
                "exec_date": str,       # 预期执行日期 YYYYMMDD
                "signals": {            # {股票代码: 权重}
                    "000001.SZ": 0.2
                },
                "top_n": int,
                "create_time": str,
                "executed": bool
            }
        ]
    }
    """
    
    def __init__(self, file_path: str = "data/trading_state.json"):
        """初始化持久化模块
        
        Args:
            file_path: JSON 文件路径
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化或加载状态
        self.state = self._load_or_init_state()
        logger.info(f"持久化模块初始化完成，文件: {self.file_path}")
    
    def _load_or_init_state(self) -> Dict[str, Any]:
        """加载或初始化状态"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info(f"从 {self.file_path} 加载状态成功")
                    return state
            except Exception as e:
                logger.warning(f"加载状态失败: {e}，使用新状态")
        
        # 初始状态
        return {
            "account": {
                "cash": 0.0,
                "initial_cash": 0.0,
                "last_update": self._now_str()
            },
            "positions": {},
            "orders": [],
            "pending_signals": []
        }
    
    def _save_state(self) -> None:
        """保存状态到文件"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            logger.debug(f"状态已保存到 {self.file_path}")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            raise
    
    def _now_str(self) -> str:
        """获取当前时间字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ===== 账户管理 =====
    
    def init_account(self, initial_cash: float) -> None:
        """初始化账户
        
        Args:
            initial_cash: 初始资金
        """
        self.state["account"] = {
            "cash": initial_cash,
            "initial_cash": initial_cash,
            "last_update": self._now_str()
        }
        self._save_state()
        logger.info(f"账户已初始化，初始资金: {initial_cash:.2f}")
    
    def get_account(self) -> Dict[str, Any]:
        """获取账户信息
        
        Returns:
            账户信息字典
        """
        return self.state["account"].copy()
    
    def update_cash(self, cash: float) -> None:
        """更新现金
        
        Args:
            cash: 新的现金数额
        """
        self.state["account"]["cash"] = cash
        self.state["account"]["last_update"] = self._now_str()
        self._save_state()
        logger.debug(f"现金已更新: {cash:.2f}")
    
    # ===== 持仓管理 =====
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有持仓
        
        Returns:
            持仓字典
        """
        return self.state["positions"].copy()
    
    def get_position(self, code: str) -> Optional[Dict[str, Any]]:
        """获取指定股票持仓
        
        Args:
            code: 股票代码
            
        Returns:
            持仓信息，不存在则返回 None
        """
        return self.state["positions"].get(code)
    
    def update_position(
        self,
        code: str,
        shares: int,
        avg_cost: float,
        last_price: float
    ) -> None:
        """更新持仓
        
        Args:
            code: 股票代码
            shares: 持仓股数
            avg_cost: 平均成本
            last_price: 最新价格
        """
        if shares <= 0:
            # 清空持仓
            if code in self.state["positions"]:
                del self.state["positions"][code]
                logger.debug(f"持仓已清空: {code}")
        else:
            self.state["positions"][code] = {
                "code": code,
                "shares": shares,
                "avg_cost": avg_cost,
                "last_price": last_price,
                "update_time": self._now_str()
            }
            logger.debug(f"持仓已更新: {code}, 股数={shares}, 成本={avg_cost:.2f}")
        
        self._save_state()
    
    def clear_positions(self) -> None:
        """清空所有持仓"""
        self.state["positions"] = {}
        self._save_state()
        logger.info("所有持仓已清空")
    
    # ===== 订单管理 =====
    
    def add_order(self, order: Dict[str, Any]) -> None:
        """添加订单记录
        
        Args:
            order: 订单信息字典
        """
        self.state["orders"].append(order)
        self._save_state()
        logger.debug(f"订单已保存: {order['order_id']}")
    
    def get_orders(
        self,
        code: Optional[str] = None,
        direction: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查询订单
        
        Args:
            code: 股票代码过滤
            direction: 方向过滤 ("buy" or "sell")
            limit: 返回最近 N 条
            
        Returns:
            订单列表
        """
        orders = self.state["orders"]
        
        # 过滤
        if code:
            orders = [o for o in orders if o["code"] == code]
        if direction:
            orders = [o for o in orders if o["direction"] == direction]
        
        # 限制数量（返回最新的）
        if limit:
            orders = orders[-limit:]
        
        return orders
    
    # ===== 待执行信号管理 =====
    
    def add_pending_signal(
        self,
        trade_date: str,
        exec_date: str,
        signals: Dict[str, float],
        top_n: int
    ) -> None:
        """添加待执行信号
        
        Args:
            trade_date: 信号生成日期 YYYYMMDD
            exec_date: 预期执行日期 YYYYMMDD
            signals: 信号字典 {股票代码: 权重}
            top_n: top N 数量
        """
        pending = {
            "trade_date": trade_date,
            "exec_date": exec_date,
            "signals": signals,
            "top_n": top_n,
            "create_time": self._now_str(),
            "executed": False
        }
        self.state["pending_signals"].append(pending)
        self._save_state()
        logger.info(f"待执行信号已保存: {trade_date} -> {exec_date}, {len(signals)} 只股票")
    
    def get_pending_signals(
        self,
        exec_date: Optional[str] = None,
        executed: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """获取待执行信号
        
        Args:
            exec_date: 执行日期过滤 YYYYMMDD
            executed: 是否已执行过滤
            
        Returns:
            待执行信号列表
        """
        signals = self.state["pending_signals"]
        
        # 过滤
        if exec_date is not None:
            signals = [s for s in signals if s["exec_date"] == exec_date]
        if executed is not None:
            signals = [s for s in signals if s["executed"] == executed]
        
        return signals
    
    def mark_signal_executed(self, trade_date: str) -> None:
        """标记信号为已执行
        
        Args:
            trade_date: 信号生成日期 YYYYMMDD
        """
        for signal in self.state["pending_signals"]:
            if signal["trade_date"] == trade_date and not signal["executed"]:
                signal["executed"] = True
                logger.info(f"信号已标记为执行: {trade_date}")
        
        self._save_state()
    
    def clear_executed_signals(self) -> None:
        """清除已执行的信号"""
        before = len(self.state["pending_signals"])
        self.state["pending_signals"] = [
            s for s in self.state["pending_signals"]
            if not s["executed"]
        ]
        after = len(self.state["pending_signals"])
        self._save_state()
        logger.info(f"已清除 {before - after} 条已执行信号")
    
    # ===== 状态管理 =====
    
    def get_state(self) -> Dict[str, Any]:
        """获取完整状态
        
        Returns:
            状态字典的深拷贝
        """
        return json.loads(json.dumps(self.state))
    
    def reset(self) -> None:
        """重置所有状态"""
        self.state = {
            "account": {
                "cash": 0.0,
                "initial_cash": 0.0,
                "last_update": self._now_str()
            },
            "positions": {},
            "orders": [],
            "pending_signals": []
        }
        self._save_state()
        logger.warning("所有状态已重置")
