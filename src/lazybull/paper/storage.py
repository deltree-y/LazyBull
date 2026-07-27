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
            "exclude_st",
            "min_list_days",
        ],
    ),
    (
        "holding_management",
        "持仓保留奖励与盈亏动态持仓",
        [
        ],
        [
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
    "portfolio": [],
    "holding_management": [],
    "stop_loss": [],
    "weakness_exit": [],
    "equity_curve": [],
    "market_regime": [],
    "industry": [],
    "position_management": [],
    "paper_trade": [],
}


def _ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


class PaperStorage:
    """纸面交易持久化存储。

    所有纸面交易状态（账户、指令、调仓记录等）以 JSON/Parquet 格式
    存储在 paper_root 目录下。
    """

    def __init__(self, paper_root: Optional[str] = None, verbose: bool = True):
        """初始化纸面交易存储。

        Args:
            paper_root: 纸面交易数据根目录，未传时使用项目配置 paper.root
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        self._paper_root = Path(self._resolve_paper_root(paper_root))
        _ensure_dir(self._paper_root)
        _ensure_dir(self._paper_root / "runs")
        _ensure_dir(self._paper_root / "instructions")

    @staticmethod
    def _resolve_paper_root(paper_root: Optional[str] = None) -> str:
        """解析纸面交易数据根目录。"""
        if paper_root:
            return paper_root
        return get_paper_root()

    # ── 配置文件 ─────────────────────────────────────────────

    def load_config(self) -> dict:
        """加载纸面交易配置文件（YAML）。"""
        config_path = self._paper_root / "config.yaml"
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}，返回空配置")
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save_config(self, config: dict) -> None:
        """保存纸面交易配置文件（YAML）。"""
        config_path = self._paper_root / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        if self.verbose:
            logger.info(f"配置文件已保存: {config_path}")

    # ── 账户状态 ─────────────────────────────────────────────

    def load_account_state(self) -> Optional[dict]:
        """加载账户状态。"""
        path = self._paper_root / "account.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_account_state(self, state: dict) -> None:
        """保存账户状态。"""
        path = self._paper_root / "account.json"
        # 备份旧文件
        if path.exists():
            backup = self._paper_root / "account.json.bak"
            shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    # ── 调仓状态 ─────────────────────────────────────────────

    def load_rebalance_state(self) -> Optional[dict]:
        """加载调仓状态。"""
        path = self._paper_root / "rebalance_state.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_rebalance_state(self, state: dict) -> None:
        """保存调仓状态。"""
        path = self._paper_root / "rebalance_state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    # ── 策略运行状态 ─────────────────────────────────────────

    def load_strategy_state(self) -> dict:
        """加载策略运行状态。"""
        path = self._paper_root / "strategy_state.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_strategy_state(self, state: dict) -> None:
        """保存策略运行状态。"""
        path = self._paper_root / "strategy_state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    # ── 交易指令 ─────────────────────────────────────────────

    def load_instructions(self, date: str) -> Optional[list]:
        """加载指定日期的交易指令。"""
        path = self._paper_root / "instructions" / f"{date}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_instructions(self, date: str, instructions: list) -> None:
        """保存指定日期的交易指令。"""
        _ensure_dir(self._paper_root / "instructions")
        path = self._paper_root / "instructions" / f"{date}.json"
        # 转换 TradeInstruction 对象为字典
        serializable = []
        for inst in instructions:
            if hasattr(inst, '__dict__'):
                d = {k: v for k, v in inst.__dict__.items() if not k.startswith('_')}
                serializable.append(d)
            else:
                serializable.append(inst)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

    # ── 运行记录（幂等性保障）─────────────────────────────────

    def check_run_exists(self, run_type: str, date: str) -> bool:
        """检查指定类型的运行是否已执行过。"""
        path = self._paper_root / "runs" / f"{run_type}_{date}.json"
        return path.exists()

    def save_run_record(self, run_type: str, date: str, record: dict) -> None:
        """保存运行记录。"""
        _ensure_dir(self._paper_root / "runs")
        path = self._paper_root / "runs" / f"{run_type}_{date}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    def load_last_trade_date(self) -> Optional[str]:
        """获取最近一次交易日期。"""
        path = self._paper_root / "last_trade_date.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip()

    def save_last_trade_date(self, date: str) -> None:
        """保存最近一次交易日期。"""
        path = self._paper_root / "last_trade_date.txt"
        path.write_text(str(date), encoding="utf-8")

    # ── 排序候选 ─────────────────────────────────────────────

    def save_ranked_candidates(self, candidates: list, date: str) -> None:
        """保存排序候选列表。"""
        path = self._paper_root / f"ranked_candidates_{date}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2, default=str)

    def load_ranked_candidates(self, date: Optional[str] = None) -> Optional[list]:
        """加载排序候选列表。"""
        if date:
            path = self._paper_root / f"ranked_candidates_{date}.json"
        else:
            # 加载最新的
            candidates_files = sorted(
                self._paper_root.glob("ranked_candidates_*.json"),
                reverse=True,
            )
            if not candidates_files:
                return None
            path = candidates_files[0]
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── 待执行买入 ───────────────────────────────────────────

    def load_pending_buys(self) -> Optional[list]:
        """加载待执行买入列表。"""
        path = self._paper_root / "pending_buys.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_pending_buys(self, buys: list) -> None:
        """保存待执行买入列表。"""
        path = self._paper_root / "pending_buys.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(buys, f, ensure_ascii=False, indent=2, default=str)

    # ── 净值记录 ─────────────────────────────────────────────

    def load_nav_history(self) -> Optional[pd.DataFrame]:
        """加载净值历史。"""
        path = self._paper_root / "nav.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save_nav_history(self, nav_df: pd.DataFrame) -> None:
        """保存净值历史。"""
        path = self._paper_root / "nav.parquet"
        nav_df.to_parquet(path, index=False)

    # ── 弱势退出状态 ─────────────────────────────────────────

    def load_weakness_exit_state(self) -> Optional[dict]:
        """加载弱势退出状态。"""
        path = self._paper_root / "weakness_exit_state.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_weakness_exit_state(self, state: dict) -> None:
        """保存弱势退出状态。"""
        path = self._paper_root / "weakness_exit_state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
