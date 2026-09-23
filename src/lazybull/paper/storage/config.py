# -*- coding: utf-8 -*-
"""PaperConfigMixin：src/lazybull/paper/storage.py 拆分出的 save_config, _render_config_yaml, _write_yaml_config, _flatten_grouped_config, _normalize_config, load_config, _load_config_remote。"""

from ...common.trading_config import TradingConfig
from loguru import logger
from typing import Optional
import yaml

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
            "enable_early_rebalance_on_empty",
            "exclude_st",
            "min_list_days",
        ],
    ),
    (
        "stop_loss",
        "止损参数",
        [
            "stop_loss_enabled 为总开关，支持回撤止损和连续跌停止损。",
        ],
        [
            "stop_loss_enabled",
            "stop_loss_drawdown_pct",
            "stop_loss_consecutive_limit_down",
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
    (
        "exposure",
        "暴露政策（terminal_loss；默认关）",
        [
            "exposure_policy 为策略字符串（如 arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5）；null = 关闭。",
            "启用时必须同时给出 policy_model_root（终损折模型根目录）与 policy_arm_suffix（如 _v6m_fscore）。",
            "exposure_replenish 为对称回补开关；exposure_trim_tolerance 为减仓/回补共用容差（组合总值比例，默认 0.03）。",
            "policy_coverage_start 为生效起点（此日之前只累积阈值历史、不动作）。",
        ],
        [
            "exposure_policy",
            "policy_model_root",
            "policy_arm_suffix",
            "policy_coverage_start",
            "exposure_replenish",
            "exposure_trim_tolerance",
        ],
    ),
    (
        "signal_penalty",
        "信号层下行风险惩罚（A5；默认关）",
        [
            "downside_penalty 为惩罚强度 λ（冻结网格 0 / 0.25 / 0.5；0 = 关闭且与基线逐位一致）。",
            "downside_penalty_column 为所用风险列（downside_vol_20 = 主臂；cvar_95_20 = 对照）。",
        ],
        [
            "downside_penalty",
            "downside_penalty_column",
        ],
    ),
]

#: 已下线功能的配置键（历史模板遗留）：保存/生成配置时一律剔除，避免 reset/刷新后复活。
#: 键名或前缀命中任一规则即视为已下线。
RETIRED_CONFIG_KEYS = {
    "holding_management",  # 盈亏动态持仓（实现已移除）
    "enable_profit_based_holding",
    "use_atr_for_early_exit",
    "atr_multiplier",
    "take_profit_threshold",
    "take_profit_refill",
    "weakness_exit",  # 表现弱势退出（实现已移除）
    "equity_curve",  # 权益曲线交易 ECT（已下线）
    "market_regime",  # 市场择时（未实现）
    "industry",  # 行业动量/轮动（未实现）
    "signal_gate_mode",  # 信号入口门控（未实现）
    "stop_loss_trailing_enabled",  # 移动止损（未实现）
    "stop_loss_trailing_pct",
}

RETIRED_CONFIG_PREFIXES = (
    "equity_curve_",
    "market_regime_",
    "weakness_exit_",
    "industry_momentum_",
    "industry_rotation_",
    "profit_extension_",
    "early_exit_",
    "holding_bonus_",
    "signal_gate_",
)


def is_retired_config_field(key: str) -> bool:
    """判断配置键是否属于已下线功能（单处定义，供保存/生成路径统一剔除）。"""
    key = str(key or "")
    if key in RETIRED_CONFIG_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in RETIRED_CONFIG_PREFIXES)

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
    "portfolio": [("基础组合参数（始终生效）", CONFIG_SECTION_LAYOUT[1][3])],
    "stop_loss": [
        (
            "止损总开关（关闭后以下止损参数整体不生效）",
            [
                "stop_loss_enabled",
                "stop_loss_drawdown_pct",
                "stop_loss_consecutive_limit_down",
            ],
        ),
    ],
    "position_management": [
        ("仓位管理模式（始终生效）", ["position_sizing"]),
        (
            "以下参数仅在 position_sizing=kelly / half_kelly 时生效",
            ["kelly_vol_window", "kelly_max_leverage"],
        ),
    ],
    "paper_trade": [("基础执行参数（始终生效）", CONFIG_SECTION_LAYOUT[4][3])],
    "exposure": [
        ("总开关（null = 关闭；关闭后以下参数不生效）", ["exposure_policy"]),
        (
            "以下参数仅在 exposure_policy 非 null 时生效",
            [
                "policy_model_root",
                "policy_arm_suffix",
                "policy_coverage_start",
                "exposure_replenish",
                "exposure_trim_tolerance",
            ],
        ),
    ],
    "signal_penalty": [
        ("始终生效（λ=0 时无行为变化）", ["downside_penalty", "downside_penalty_column"]),
    ],
}

class PaperConfigMixin:
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
        """将配置补齐为完整 TradingConfig 视图（并剔除已下线功能的遗留键）。"""
        normalized = self._flatten_grouped_config(config)
        if "position_sizing" not in normalized and "weight_method" in normalized:
            normalized["position_sizing"] = normalized["weight_method"]

        retired = sorted(key for key in normalized if is_retired_config_field(key))
        for key in retired:
            normalized.pop(key, None)
        if retired:
            logger.info(f"已剔除已下线功能的配置键（{len(retired)} 个）: {retired}")

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
