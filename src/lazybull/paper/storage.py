"""纸面交易存储模块"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

from typing import TYPE_CHECKING

from ..common.config import get_paper_root
from ..common.trading_config import TradingConfig
from .models import AccountState, Fill, NAVRecord, PendingBuy, PendingSell, Position, TargetWeight, TradeInstruction

if TYPE_CHECKING:
    from ..common.smb_client import SMBFileReader


CONFIG_SECTION_LAYOUT = [
    (
        "model",
        "模型与集成配置",
        [
            "model_version 为主模型版本，null 表示读取最新注册模型。",
            "model_version_b 非 null 时启用双模型集成，ensemble_weight_a 表示模型 A 权重。",
        ],
        ["model_version", "model_version_b", "ensemble_weight_a"],
    ),
    (
        "signal_gate",
        "信号入口门控与动态 Top-N",
        [
            "统一管理旧版置信度门控、composite 门控、滚动质量监控和动态 Top-N。",
            "signal_gate_mode 可选 legacy / composite / disabled。",
        ],
        [
            "signal_confidence_gate_enabled",
            "signal_confidence_gate_top_k",
            "signal_confidence_gate_thresholds",
            "signal_confidence_gate_exposure_levels",
            "signal_gate_mode",
            "signal_gate_cost_multiplier",
            "signal_gate_round_trip_cost",
            "signal_gate_quality_enabled",
            "signal_gate_quality_window",
            "signal_gate_quality_threshold",
            "signal_gate_quality_halflife",
            "signal_gate_percentile_warmup",
            "signal_gate_dynamic_topn",
            "signal_gate_topn_high_multiplier",
            "signal_gate_topn_low_multiplier",
        ],
    ),
    (
        "portfolio",
        "组合约束与调仓节奏",
        [
            "top_n 为目标持仓数，rebalance_freq 为调仓频率（交易日）。",
            "max_per_industry / max_weight_per_stock 用于行业和个股约束。",
        ],
        [
            "top_n",
            "rebalance_freq",
            "stagger_tranches",
            "max_per_industry",
            "max_weight_per_stock",
            "enable_early_rebalance_on_empty",
            "exclude_st",
            "min_list_days",
        ],
    ),
    (
        "holding_management",
        "持仓保留奖励与盈亏动态持仓",
        [
            "holding_bonus_* 用于降低换手；early_exit_* / profit_extension_* 分别控制亏损提前换出和盈利延续。",
            "注意：early_exit_mode=disabled 表示原硬卖，不是关闭亏损提前换出。",
            "profit_extension_mode 可选 pnl / strength / disabled。",
        ],
        [
            "holding_bonus_enabled",
            "holding_bonus_sigma",
            "enable_profit_based_holding",
            "early_exit_loss_threshold",
            "early_exit_holding_ratio",
            "early_exit_mode",
            "early_exit_strength_protect_threshold",
            "early_exit_max_reprieves",
            "use_atr_for_early_exit",
            "atr_multiplier",
            "profit_extension_mode",
            "profit_extension_threshold",
            "profit_extension_days",
            "profit_extension_strength_threshold",
            "profit_extension_strength_weights",
            "take_profit_threshold",
            "take_profit_refill",
        ],
    ),
    (
        "stop_loss",
        "止损参数",
        [
            "stop_loss_enabled 为总开关，支持回撤止损、移动止损和连续跌停止损。",
        ],
        [
            "stop_loss_enabled",
            "stop_loss_drawdown_pct",
            "stop_loss_trailing_enabled",
            "stop_loss_trailing_pct",
            "stop_loss_consecutive_limit_down",
        ],
    ),
    (
        "weakness_exit",
        "表现弱势退出",
        [
            "纯价格表现评估，零模型依赖。每日用累计收益排名、连续下跌天数、回撤深度、回升乏力 4 个维度识别弱势股并提前换出。",
        ],
        [
            "weakness_exit_enabled",
            "weakness_exit_threshold",
            "weakness_exit_consecutive_days",
            "weakness_exit_min_holding_days",
            "weakness_exit_weights",
            "weakness_exit_industry_filter",
            "weakness_exit_industry_bottom_pct",
        ],
    ),
    (
        "equity_curve",
        "权益曲线交易（ECT）",
        [
            "drawdown_thresholds 和 exposure_levels 需要一一对应。",
        ],
        [
            "equity_curve_enabled",
            "equity_curve_drawdown_thresholds",
            "equity_curve_exposure_levels",
            "equity_curve_ma_short",
            "equity_curve_ma_long",
            "equity_curve_recovery_mode",
            "equity_curve_recovery_step",
            "equity_curve_recovery_delay_periods",
        ],
    ),
    (
        "market_regime",
        "市场择时仓位管理",
        [
            "market_regime_mode 可选 binary / vol_target / trend / combined。",
            "MA250 硬条件和 ATR 缩放也在本段统一配置。",
        ],
        [
            "market_regime_enabled",
            "market_regime_mode",
            "market_regime_bear_threshold",
            "market_regime_bear_exposure",
            "market_regime_vol_target",
            "market_regime_trend_threshold",
            "market_regime_min_exposure",
            "market_regime_combine_method",
            "market_regime_trend_guard",
            "market_regime_drawdown_guard",
            "market_regime_drawdown_threshold",
            "market_regime_ma250_hard_stop",
            "market_regime_ma250_threshold",
            "market_regime_ma250_exposure",
            "market_regime_ma250_atr_scaling",
        ],
    ),
    (
        "industry",
        "行业过滤与行业轮动加权",
        [
            "industry_momentum_filter 为硬过滤；industry_rotation_enhanced 为软加权。",
        ],
        [
            "industry_momentum_filter",
            "industry_momentum_bottom_pct",
            "industry_rotation_enhanced",
            "industry_rotation_alpha",
        ],
    ),
    (
        "position_management",
        "仓位管理模式",
        [
            "position_sizing 可选 equal / score / kelly / half_kelly。",
            "Kelly 模式使用 kelly_vol_window 和 kelly_max_leverage。",
        ],
        ["position_sizing", "kelly_vol_window", "kelly_max_leverage"],
    ),
    (
        "paper_trade",
        "纸面交易执行参数",
        [
            "buy_price / sell_price 控制 T0/T1 默认价格口径。",
            "min_buy_value_ratio 控制最小买入后持仓市值阈值（按平均仓位市值比例）。",
            "horizon 需要与模型标签周期保持一致。",
        ],
        [
            "buy_price",
            "sell_price",
            "initial_capital",
            "min_buy_value_ratio",
            "horizon",
            "universe",
        ],
    ),
]

CONFIG_SECTION_NAMES = {section_name for section_name, _, _, _ in CONFIG_SECTION_LAYOUT}
CONFIG_FIELD_NAMES = {
    field_name
    for _, _, _, field_names in CONFIG_SECTION_LAYOUT
    for field_name in field_names
}

CONFIG_SECTION_RENDER_GROUPS = {
    "model": [
        ("基础模型参数（始终生效）", ["model_version"]),
        (
            "以下参数仅在 model_version_b 非 null 时生效",
            ["model_version_b", "ensemble_weight_a"],
        ),
    ],
    "signal_gate": [
        ("门控模式总开关（disabled 表示整体关闭）", ["signal_gate_mode"]),
        (
            "以下参数在 signal_gate_mode=legacy / composite 时都会参与头部候选评估",
            ["signal_confidence_gate_top_k"],
        ),
        (
            "以下参数仅在 signal_gate_mode=legacy 时生效",
            [
                "signal_confidence_gate_enabled",
                "signal_confidence_gate_thresholds",
                "signal_confidence_gate_exposure_levels",
            ],
        ),
        (
            "以下参数仅在 signal_gate_mode=composite 时生效",
            [
                "signal_gate_cost_multiplier",
                "signal_gate_round_trip_cost",
                "signal_gate_percentile_warmup",
            ],
        ),
        (
            "滚动质量监控子开关（window / threshold / halflife 仅在 enabled=true 时生效）",
            [
                "signal_gate_quality_enabled",
                "signal_gate_quality_window",
                "signal_gate_quality_threshold",
                "signal_gate_quality_halflife",
            ],
        ),
        (
            "动态 Top-N 子开关（乘数仅在 enabled=true 时生效）",
            [
                "signal_gate_dynamic_topn",
                "signal_gate_topn_high_multiplier",
                "signal_gate_topn_low_multiplier",
            ],
        ),
    ],
    "portfolio": [("基础组合参数（始终生效）", CONFIG_SECTION_LAYOUT[2][3])],
    "holding_management": [
        (
            "持仓保留奖励子开关（sigma 仅在 enabled=true 时生效）",
            ["holding_bonus_enabled", "holding_bonus_sigma"],
        ),
        (
            "盈亏动态持仓总开关（关闭后以下盈利延续 / 亏损换出参数不生效）",
            ["enable_profit_based_holding"],
        ),
        (
            "亏损提前换出基础阈值（enable_profit_based_holding=true 时始终生效）",
            ["early_exit_loss_threshold", "early_exit_holding_ratio"],
        ),
        (
            "亏损提前换出二次确认子开关（disabled=原硬卖，strength_veto=启用二次确认）",
            ["early_exit_mode"],
        ),
        (
            "以下参数仅在 early_exit_mode=strength_veto 时生效",
            ["early_exit_strength_protect_threshold", "early_exit_max_reprieves"],
        ),
        (
            "ATR 动态阈值子开关（仅在 enable_profit_based_holding=true 且 use_atr_for_early_exit=true 时生效）",
            ["use_atr_for_early_exit", "atr_multiplier"],
        ),
        (
            "盈利延续子开关（disabled=关闭延续；仅在 enable_profit_based_holding=true 时生效）",
            ["profit_extension_mode"],
        ),
        (
            "以下参数仅在 profit_extension_mode=pnl 时生效",
            ["profit_extension_threshold", "profit_extension_days"],
        ),
        (
            "以下参数仅在 profit_extension_mode=strength 时生效",
            ["profit_extension_strength_threshold", "profit_extension_strength_weights"],
        ),
        (
            "整体止盈（独立于 enable_profit_based_holding）",
            ["take_profit_threshold", "take_profit_refill"],
        ),
    ],
    "stop_loss": [
        (
            "止损总开关（关闭后以下止损参数整体不生效）",
            [
                "stop_loss_enabled",
                "stop_loss_drawdown_pct",
                "stop_loss_consecutive_limit_down",
            ],
        ),
        (
            "移动止损子开关（trailing_pct 仅在 enabled=true 时生效）",
            ["stop_loss_trailing_enabled", "stop_loss_trailing_pct"],
        ),
    ],
    "weakness_exit": [
        (
            "表现弱势退出总开关（关闭后以下参数整体不生效）",
            [
                "weakness_exit_enabled",
                "weakness_exit_threshold",
                "weakness_exit_consecutive_days",
                "weakness_exit_min_holding_days",
                "weakness_exit_weights",
            ],
        ),
        (
            "弱势行业过滤子开关（仅在 enabled=true 且 industry_filter=true 时生效）",
            ["weakness_exit_industry_filter", "weakness_exit_industry_bottom_pct"],
        ),
    ],
    "equity_curve": [
        (
            "ECT 总开关（关闭后以下参数整体不生效）",
            [
                "equity_curve_enabled",
                "equity_curve_drawdown_thresholds",
                "equity_curve_exposure_levels",
                "equity_curve_ma_short",
                "equity_curve_ma_long",
                "equity_curve_recovery_mode",
                "equity_curve_recovery_step",
                "equity_curve_recovery_delay_periods",
            ],
        )
    ],
    "market_regime": [
        (
            "市场择时总开关（关闭后以下 binary / vol_target / trend / combined 参数不生效）",
            ["market_regime_enabled", "market_regime_mode"],
        ),
        (
            "以下参数在 market_regime_enabled=true 且 mode=binary 时生效",
            ["market_regime_bear_threshold", "market_regime_bear_exposure"],
        ),
        (
            "以下参数在 market_regime_enabled=true 且 mode=vol_target 时生效",
            ["market_regime_vol_target"],
        ),
        (
            "以下参数在 market_regime_enabled=true 且 mode=trend / combined 时生效",
            [
                "market_regime_trend_threshold",
                "market_regime_min_exposure",
                "market_regime_trend_guard",
            ],
        ),
        (
            "以下参数在 market_regime_enabled=true 且 mode=combined 时生效",
            ["market_regime_combine_method"],
        ),
        (
            "回撤保护子开关（drawdown_threshold 仅在 enabled=true 时生效）",
            ["market_regime_drawdown_guard", "market_regime_drawdown_threshold"],
        ),
        (
            "MA250 独立开关（可在 market_regime_enabled=false 时单独生效）",
            [
                "market_regime_ma250_hard_stop",
                "market_regime_ma250_threshold",
                "market_regime_ma250_exposure",
                "market_regime_ma250_atr_scaling",
            ],
        ),
    ],
    "industry": [
        (
            "行业动量过滤子开关（bottom_pct 仅在 enabled=true 时生效）",
            ["industry_momentum_filter", "industry_momentum_bottom_pct"],
        ),
        (
            "行业轮动加权子开关（alpha 仅在 enabled=true 时生效）",
            ["industry_rotation_enhanced", "industry_rotation_alpha"],
        ),
    ],
    "position_management": [
        ("仓位管理模式（始终生效）", ["position_sizing"]),
        (
            "以下参数仅在 position_sizing=kelly / half_kelly 时生效",
            ["kelly_vol_window", "kelly_max_leverage"],
        ),
    ],
    "paper_trade": [("基础执行参数（始终生效）", CONFIG_SECTION_LAYOUT[9][3])],
}


class PaperStorage:
    """纸面交易存储
    
    负责持久化和读取纸面交易的各类数据
    """
    
    def __init__(
        self,
        root_path: Optional[str] = None,
        verbose: bool = False,
        smb_reader: Optional["SMBFileReader"] = None,
    ):
        """初始化纸面交易存储
        
        Args:
            root_path: 数据根目录；未传时默认使用 data.root/paper
            verbose: 是否输出详细日志
            smb_reader: 远端 SMB 读取器；传入后读取走 SMB，写入自动跳过（只读模式）
        """
        self._smb_reader = smb_reader
        self._is_remote = smb_reader is not None
        self.root_path = Path(root_path or get_paper_root())
        self.state_path = self.root_path / "state"
        self.trades_path = self.root_path / "trades"
        self.nav_path = self.root_path / "nav"
        self.runs_path = self.root_path / "runs"
        self.pending_sells_path = self.root_path / "pending_sells"
        self.pending_buys_path = self.root_path / "pending_buys"
        self.instructions_path = self.root_path / "instructions"
        self.verbose = verbose
        
        # 远端只读模式不创建本地目录
        if not self._is_remote:
            for path in [self.state_path, self.trades_path,
                         self.nav_path, self.runs_path, self.pending_sells_path,
                         self.pending_buys_path, self.instructions_path]:
                path.mkdir(parents=True, exist_ok=True)
        if verbose:
            mode = "远端只读" if self._is_remote else "本地"
            logger.info(f"纸面交易存储初始化完成（{mode}），根目录: {self.root_path}")
    
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
    
    def save_config(self, config: dict) -> None:
        """保存全局配置
        
        Args:
            config: 配置字典
        """
        normalized_config = self._normalize_config(config)
        self._write_yaml_config(normalized_config)
        logger.info(f"保存全局配置: {self.root_path / 'config.yaml'}")

    def _render_config_yaml(self, config: dict) -> str:
        """渲染带中文注释的 YAML 配置模板。"""
        lines = [
            "# 纸面交易主配置（仅保留纸面交易实际可用参数）",
            "# 说明：优先编辑本文件；paper_trade.py config 命令会按相同的开关分组刷新本模板。",
            "# 说明：同一开关控制的参数会紧跟在该开关后面，便于判断当前是否生效。",
            "",
        ]

        for section_name, section_title, section_comments, field_names in CONFIG_SECTION_LAYOUT:
            section_config = {
                field_name: config[field_name]
                for field_name in field_names
                if field_name in config
            }
            if not section_config:
                continue

            lines.append("# =============================================================================")
            lines.append(f"# {section_title}")
            lines.append("# =============================================================================")
            for comment in section_comments:
                lines.append(f"# {comment}")
            lines.append(f"{section_name}:")

            rendered_fields = set()
            render_groups = CONFIG_SECTION_RENDER_GROUPS.get(
                section_name, [("基础参数（始终生效）", field_names)]
            )
            for group_index, (group_comment, group_field_names) in enumerate(render_groups):
                present_fields = [
                    field_name
                    for field_name in group_field_names
                    if field_name in section_config and field_name not in rendered_fields
                ]
                if not present_fields:
                    continue
                if group_index > 0:
                    lines.append("")
                if group_comment:
                    lines.append(f"  # {group_comment}")
                for field_name in present_fields:
                    dumped_field = yaml.safe_dump(
                        {field_name: section_config[field_name]},
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    ).rstrip()
                    for line in dumped_field.splitlines():
                        lines.append(f"  {line}")
                    rendered_fields.add(field_name)

            remaining_fields = [
                field_name for field_name in field_names if field_name in section_config and field_name not in rendered_fields
            ]
            if remaining_fields:
                if rendered_fields:
                    lines.append("")
                lines.append("  # 其他基础参数")
                for field_name in remaining_fields:
                    dumped_field = yaml.safe_dump(
                        {field_name: section_config[field_name]},
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    ).rstrip()
                    for line in dumped_field.splitlines():
                        lines.append(f"  {line}")
            lines.append("")

        extra_config = {
            key: value
            for key, value in config.items()
            if key not in CONFIG_FIELD_NAMES
        }
        if extra_config:
            lines.append("# =============================================================================")
            lines.append("# 兼容扩展字段")
            lines.append("# =============================================================================")
            lines.append("# 非 TradingConfig 标准字段会放在这里，避免手工新增字段被覆盖。")
            lines.append("extra:")
            dumped_extra = yaml.safe_dump(
                extra_config,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ).rstrip()
            for line in dumped_extra.splitlines():
                lines.append(f"  {line}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _write_yaml_config(self, config: dict) -> None:
        """写入带注释的 YAML 主配置文件。"""
        file_path = self.root_path / "config.yaml"
        file_path.write_text(self._render_config_yaml(config), encoding="utf-8")

    def _flatten_grouped_config(self, config: dict) -> dict:
        """将分段 YAML 配置展平为 TradingConfig 兼容的扁平字典。"""
        if not isinstance(config, dict):
            return {}

        flattened = {}
        for key, value in config.items():
            if key in CONFIG_SECTION_NAMES.union({"extra"}) and isinstance(value, dict):
                flattened.update(value)
            else:
                flattened[key] = value
        return flattened

    def _normalize_config(self, config: dict) -> dict:
        """将配置补齐为完整 TradingConfig 视图。"""
        normalized = self._flatten_grouped_config(config)
        if "position_sizing" not in normalized and "weight_method" in normalized:
            normalized["position_sizing"] = normalized["weight_method"]

        trading_config = TradingConfig.from_dict(normalized).to_dict()
        extra_keys = {
            key: value
            for key, value in normalized.items()
            if key not in trading_config and key != "weight_method"
        }
        return {**trading_config, **extra_keys}
    
    def load_config(self) -> Optional[dict]:
        """读取全局配置
        
        Returns:
            配置字典，不存在返回None
        """
        if self._is_remote:
            return self._load_config_remote()

        yaml_path = self.root_path / "config.yaml"
        if not yaml_path.exists():
            return None

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        return self._normalize_config(config)

    def _load_config_remote(self) -> Optional[dict]:
        """通过 SMB 远端读取配置。"""
        if self._smb_reader is None:
            return None
        try:
            config = self._smb_reader.read_yaml("config.yaml")
            if not config:
                return None
            return self._normalize_config(config)
        except Exception as exc:
            logger.warning(f"SMB 读取远端配置失败: {exc}")
            return None
    
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
    
    def save_weakness_exit_state(self, state: dict) -> None:
        """保存弱势退出监控状态

        Args:
            state: 弱势退出状态字典
        """
        file_path = self.state_path / "weakness_exit_state.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.debug(f"保存弱势退出状态: {file_path}")

    def load_weakness_exit_state(self) -> Optional[dict]:
        """读取弱势退出监控状态

        Returns:
            弱势退出状态字典，不存在返回None
        """
        file_path = self.state_path / "weakness_exit_state.json"

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

    def save_early_exit_state(self, state: dict) -> None:
        """保存亏损提前换出的缓刑状态

        Args:
            state: 缓刑状态字典，如 {"reprieve_counts": {"000001.SZ": 1}}
        """
        file_path = self.state_path / "early_exit_state.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.debug(f"保存亏损换出缓刑状态: {file_path}")

    def load_early_exit_state(self) -> dict:
        """读取亏损提前换出的缓刑状态

        Returns:
            缓刑状态字典，不存在返回空字典
        """
        file_path = self.state_path / "early_exit_state.json"

        if not file_path.exists():
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        return state

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
                original_signal_date=row.get('original_signal_date', ''),
                desired_position_count=int(row.get('desired_position_count', 0) or 0),
                retry_attempt=int(row.get('retry_attempt', 0) or 0),
            ))
        
        logger.info(f"读取交易指令: {file_path} ({len(instructions)} 条)")
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

    def reset_t0(self, t0_date: Optional[str] = None) -> dict:
        """重置纸面交易，清空所有交易数据恢复为新账户状态

        清空账户状态、成交记录、净值、运行记录、交易指令、延迟订单等，
        仅保留 config.yaml 配置文件。账户现金重置为 config 中的 initial_capital。

        Args:
            t0_date: 仅用于日志显示（可选，默认自动查找最新）

        Returns:
            操作结果统计字典
        """
        if t0_date is None:
            t0_date = self.find_latest_t0()

        stats = {'t0_date': t0_date}

        # 读取配置以获取初始资金
        config = self.load_config()
        initial_capital = config.get('initial_capital', 500000.0) if config else 500000.0

        # 清空各子目录下的所有文件
        dirs_to_clean = [
            self.state_path,
            self.trades_path,
            self.nav_path,
            self.runs_path,
            self.pending_sells_path,
            self.pending_buys_path,
            self.instructions_path,
        ]
        for dir_path in dirs_to_clean:
            for entry in dir_path.iterdir():
                if entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
            logger.info(f"已清空: {dir_path.name}/")

        # 重建空账户状态
        new_state = AccountState(
            cash=initial_capital,
            positions={},
            last_update="",
        )
        self.save_account_state(new_state)
        logger.info(f"已重建账户状态，初始资金: {initial_capital:,.2f}")

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

        # 6.5 清空策略状态
        strategy_state_file = self.state_path / "strategy_state.json"
        if strategy_state_file.exists():
            strategy_state_file.unlink()
            logger.info("清空策略状态")
        else:
            logger.info("策略状态文件不存在，跳过")
        
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

