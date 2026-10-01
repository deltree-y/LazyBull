# -*- coding: utf-8 -*-
"""配置指纹键清单的唯一代码承载（runs 契约附录 A / baseline_freeze 附录 A）。

契约条款（`docs/contracts/runs_artifact_contract.md` 附录 A、
`docs/contracts/baseline_freeze.md` 附录 A，F3 起以本模块为唯一权威源）：
- 口径 = **全键入指纹 − 显式排除清单**（fail-safe：新增键默认进指纹，
  防「漏加进清单就逃逸校验」）；
- 排除清单只此一份，转换器 / 读入桥 / 对账测试一律从本模块导入，禁止重写；
- 127 键快照对账测试见 `tests/test_v2_fingerprint_keys.py`。

排除分两类：
- 运行标识与数据/代码态：batch_meta 另有专属字段（data_state / code_state）承载，
  入指纹会造成双重计入；
- 统计输出类：逐折回测指标（bt_* 9 个统计列）、训练产出（*_samples /
  best_iteration*）、信号层统计（key_*，契约 §3 的 KEY_* 映射列，随 Top-K 档位扩展）。
注意 `bt_*` 前缀中**仅统计列**排除；`bt_top_n` / `bt_rebalance_freq` /
`bt_initial_capital` 等配置键一律入指纹（baseline_freeze 附录 A 明文）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

#: 精确排除键（24 键）
EXCLUDE_EXACT = frozenset({
    # 运行标识类（4）
    "wf_run_id", "batch_run_id", "batch_period_label", "registered_at",
    # 数据/代码态（batch_meta 专属字段承载，6）
    "data_state_id", "git_commit", "git_dirty",
    "data_daily_latest", "data_cs_train_latest", "data_dividend_coverage",
    # 逐折回测统计输出（9）
    "bt_total_return", "bt_annual_return", "bt_max_drawdown", "bt_volatility",
    "bt_sharpe", "bt_calmar", "bt_trading_days", "bt_start", "bt_end",
    # 训练产出（5）
    "train_samples", "val_samples", "test_samples",
    "best_iteration", "best_iteration_floor_triggered",
})

#: 前缀排除：信号层统计输出（KEY_* → key_* 小写蛇形映射列，数量随档位扩展）
EXCLUDE_PREFIX = ("key_",)


def is_fingerprint_key(column: str) -> bool:
    """该 summary 列是否入配置指纹（fail-safe：不在排除清单即入）。"""
    return column not in EXCLUDE_EXACT and not any(
        column.startswith(p) for p in EXCLUDE_PREFIX
    )


def config_fingerprint_keys(all_cols: List[str]) -> List[str]:
    """从全部列中筛出入指纹键（保持传入顺序）。"""
    return [c for c in all_cols if is_fingerprint_key(c)]


def fingerprint(config: Dict[str, object]) -> str:
    """配置指纹：规范化序列化的 sha256 短指纹（16 位）。

    口径冻结（改动 = 全部既有指纹失效）：`json.dumps(sort_keys=True,
    default=str, ensure_ascii=False)` → sha256 hex 前 16 字符。
    """
    norm = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
