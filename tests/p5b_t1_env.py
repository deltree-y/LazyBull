# -*- coding: utf-8 -*-
"""P5b T1 测试环境构造（tests/ 全量豁免复杂度；不是测试用例模块）。

只造合成历史输入：``clean/daily`` 分区（占位字节即可，身份只做物理哈希）、
交易日历、模型三件套（metadata 含 ``train_end_date``）、扁平纸面配置。
**不读生产 data/paper、不联网**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 合成虚拟日（2026-07-17 / 07-20 / 07-21，连续三个交易日）
DAYS = ("20260717", "20260720", "20260721")
#: 合成交易日历（含窗口前后各一日，供连续性校验）
CALENDAR = ("20260716", "20260717", "20260720", "20260721", "20260722")
#: 模型训练截止（严格早于 DAYS[0]）
TRAIN_END_DATE = "20260710"


def write_partitions(data_root: Path, days: Sequence[str] = DAYS) -> None:
    """写合成 ``clean/daily`` 分区（内容固定，便于漂移测试改写）。"""
    for day in days:
        path = data_root / "clean" / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"daily-{day}".encode("utf-8"))


def write_optional_features(data_root: Path, days: Sequence[str] = DAYS) -> None:
    """写合成 ``features/cs_infer`` 分区（可选数据集，存在即纳入窗口身份）。"""
    for day in days:
        path = data_root / "features" / "cs_infer" / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"infer-{day}".encode("utf-8"))


def write_models(
    data_root: Path, version: int = 1, *, train_end_date: str = TRAIN_END_DATE
) -> Path:
    """写合成选股模型三件套（与 P4 门 1 身份同后缀与同载荷口径）。"""
    root = data_root / "models" / "stock_selection"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"v{version}_model.joblib").write_bytes(f"joblib-v{version}".encode("utf-8"))
    (root / f"v{version}_features.json").write_text(
        json.dumps(["feature_a", "feature_b"]), encoding="utf-8"
    )
    (root / f"v{version}_metadata.json").write_text(
        json.dumps(
            {
                "model_type": "xgboost",
                "train_start_date": "20200101",
                "train_end_date": train_end_date,
                "version": version,
            }
        ),
        encoding="utf-8",
    )
    return root


def base_config(version: int = 1, capital: float = 500000.0) -> Dict[str, Any]:
    """最小可用扁平纸面配置（v2 引擎 + 显式冻结模型版本）。"""
    return {
        "model_version": version,
        "top_n": 5,
        "rebalance_freq": 20,
        "stagger_tranches": 1,
        "position_sizing": "equal",
        "buy_price": "close",
        "sell_price": "open",
        "initial_capital": capital,
        "horizon": 20,
        "universe": "mainboard",
        "engine": "v2",
    }


def build_env(
    tmp_path: Path, *, days: Sequence[str] = DAYS, config: Optional[Mapping[str, Any]] = None
) -> SimpleNamespace:
    """构造隔离环境：data 根 + 模型根 + 试车场根 + 仓库根 + 日历。"""
    data_root = tmp_path / "data"
    write_partitions(data_root, days)
    models_dir = write_models(data_root)
    return SimpleNamespace(
        data_root=data_root,
        models_dir=models_dir,
        base_root=tmp_path / "testbed",
        repo_root=ROOT,
        days=tuple(days),
        calendar=CALENDAR,
        config=dict(config or base_config()),
    )


def register_base_campaign(
    env: SimpleNamespace, campaign_id: str = "t1_campaign"
) -> Dict[str, Any]:
    """登记一个标准 campaign（测试默认入口；多日窗口必带日历）。"""
    from src.lazybull.v2.hosts.testbed import register_campaign

    return register_campaign(
        campaign_id=campaign_id,
        d0_date=env.days[0],
        virtual_days=env.days,
        calendar=env.calendar,
        data_root=str(env.data_root),
        base_root=str(env.base_root),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )


def register_base_campaign_with_base(
    env: SimpleNamespace, base_root: Path, campaign_id: str
) -> Dict[str, Any]:
    """用指定 base_root 登记（不放开外部根；用于写域守卫反例）。"""
    from src.lazybull.v2.hosts.testbed import register_campaign

    return register_campaign(
        campaign_id=campaign_id,
        d0_date=env.days[0],
        virtual_days=env.days,
        calendar=env.calendar,
        data_root=str(env.data_root),
        base_root=str(base_root),
        repo_root=str(env.repo_root),
    )


def register_base_arm(
    env: SimpleNamespace,
    campaign_root: Path,
    arm_id: str,
    *,
    role: str = "aa_clone",
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """登记一个标准模拟臂（默认 A/A 克隆）。"""
    from src.lazybull.v2.hosts.testbed import register_arm

    return register_arm(
        campaign_root=campaign_root,
        arm_id=arm_id,
        arm_role=role,
        config_flat=dict(config or env.config),
        models_dir=str(env.models_dir),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )


def prepare_two_arms(
    env: SimpleNamespace, campaign_id: str = "t1_campaign"
) -> SimpleNamespace:
    """登记 campaign + 两个 A/A 克隆臂，返回 campaign 根与记录。"""
    campaign = register_base_campaign(env, campaign_id)
    root = Path(campaign["campaign_root"])
    arms = {
        arm_id: register_base_arm(env, root, arm_id) for arm_id in ("aa_1", "aa_2")
    }
    return SimpleNamespace(campaign=campaign, root=root, arms=arms)
