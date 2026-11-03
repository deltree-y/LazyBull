# -*- coding: utf-8 -*-
"""P5b T4 命名 seed 子流与 realization 身份（设计 v4 §5.3 / §5.5「路径数/种子」）。

口径
====
每个**验收单元**（阶段 × 场景 × 零假设 × 效应档）共享同一个种子根，用
``numpy.random.SeedSequence([seed, stage_code, scene_code, null_code, level_code])`` 派生子流：

- 子流与调用顺序无关（同参数必得同一路径），因此可复现、可审计；
- 阶段编码 ``train=1 / holdout=2 / effect=3`` —— 训练与 holdout 是**互斥子流**；
  ``holdout=2`` 对应 T0.5b PSD r3 的 ``validation`` 子流（同一 seed 根 ``20261103``），
  这是 T4 能逐值复现 r3 临界值的前提，已在 manifest 的 ``substreams`` 中登记；
- ``realization_id`` 记录「哪个阶段/场景/零假设/效应档的哪条子流」，与
  ``implementation_id`` 一起构成 §5.3 要求的「不同 seed + 不同实现编号 + 互斥数据块」。

禁止把训练与 holdout 放在同一子流（否则 holdout 不再是独立实现）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError

__all__ = [
    "IMPLEMENTATION_ID",
    "NULL_CODES",
    "SCENE_CODES",
    "STAGE_ALIASES",
    "STAGE_CODES",
    "STAGES",
    "Substream",
    "SubstreamError",
    "assert_distinct_streams",
    "implementation_ids",
    "stream_codes",
    "substream",
]

#: 收益裁判实现编号（写进 realization_id 与 manifest；T5 回撤裁判使用独立编号）
IMPLEMENTATION_ID = "p5b_t4_income_referee/v1"
#: 阶段编码（种子子流隔离的唯一来源；``holdout`` 沿用 T0.5b 的 ``validation`` 编码 2）
STAGE_CODES: Mapping[str, int] = {
    "train": 1,
    "holdout": 2,
    "effect": 3,
    # 独立验收子流（R2-P5B-04）：与 T0.5b PSD r3 已观察的 holdout 子流互斥
    "holdout_independent_v1": 4,
}
#: 阶段名与 T0.5b 旧命名的对照（只影响 realization_id 里出现的层级语义）
STAGE_ALIASES: Mapping[str, str] = {
    "train": "train",
    "holdout": "validation",
    "effect": "effect",
    "holdout_independent_v1": "validation_independent_v1",
}
#: 允许的阶段（训练 / holdout / 独立验收子流 / 功效效应网格）
STAGES: Tuple[str, ...] = ("train", "holdout", "holdout_independent_v1", "effect")
#: 场景编码
SCENE_CODES: Mapping[str, int] = {
    "ma20": 1,
    "ar1_06": 2,
    "regime_hetero": 3,
    "real_seed_block": 4,
}
#: 零假设编码（``mu=0`` / ``mu=-delta``；效应档不计零假设）
NULL_CODES: Mapping[str, int] = {"mu=0": 0, "mu=-delta": 1}
#: 效应档编码倍率（``level * 1000`` 取整）
_LEVEL_SCALE = 1000


class SubstreamError(TestbedError):
    """子流参数非法（未知阶段/场景/零假设、非单调编码）。"""

    __test__ = False


@dataclass(frozen=True)
class Substream:
    """一个命名子流：生成器 + 身份（实现编号 / 阶段 / realization_id / 编码）。"""

    generator: np.random.Generator
    implementation_id: str
    stage: str
    scene: str
    null: str
    level: float
    realization_id: str
    codes: Tuple[int, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "stage": self.stage,
            "scene": self.scene,
            "null": self.null,
            "level": float(self.level),
            "realization_id": self.realization_id,
            "codes": list(self.codes),
        }


def stream_codes(
    seed: int, stage: str, scene: str, null: str = "", level: float = 0.0
) -> Tuple[int, ...]:
    """子流编码向量（``SeedSequence`` 的唯一输入；与调用顺序无关）。

    Raises:
        SubstreamError: 阶段/场景/零假设未登记或 ``level`` 非有限。
    """
    if stage not in STAGE_CODES:
        raise SubstreamError(f"未登记阶段: {stage!r}（允许 {list(STAGE_CODES)}）")
    if scene not in SCENE_CODES:
        raise SubstreamError(f"未登记场景: {scene!r}（允许 {list(SCENE_CODES)}）")
    if null and null not in NULL_CODES:
        raise SubstreamError(f"未登记零假设: {null!r}（允许 {list(NULL_CODES)}）")
    if not np.isfinite(float(level)):
        raise SubstreamError(f"效应档必须有限: {level!r}")
    return (
        int(seed),
        int(STAGE_CODES[stage]),
        int(SCENE_CODES[scene]),
        int(NULL_CODES.get(null, 0)),
        int(round(float(level) * _LEVEL_SCALE)),
    )


def substream(
    seed: int,
    stage: str,
    scene: str,
    *,
    null: str = "",
    level: float = 0.0,
    prefix: str = "",
) -> Substream:
    """构造命名子流（同参数必得同一 Generator 与同一 realization_id）。"""
    codes = stream_codes(seed, stage, scene, null, level)
    generator = np.random.default_rng(np.random.SeedSequence(list(codes)))
    alias = STAGE_ALIASES[stage]
    label = null or "effect"
    realization_id = f"{prefix}:{alias}:{scene}:{label}:{float(level):g}".lstrip(":")
    return Substream(
        generator=generator,
        implementation_id=IMPLEMENTATION_ID,
        stage=stage,
        scene=scene,
        null=null,
        level=float(level),
        realization_id=realization_id,
        codes=codes,
    )


def implementation_ids() -> Dict[str, Any]:
    """manifest ``substreams`` 段（阶段编码 + 实现编号 + 子流方案）。"""
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "scheme": "numpy SeedSequence([seed, stage_code, scene_code, null_code, level_code])",
        "stages": dict(STAGE_CODES),
        "stage_aliases": dict(STAGE_ALIASES),
        "scene_codes": dict(SCENE_CODES),
        "null_codes": dict(NULL_CODES),
        "level_code_scale": _LEVEL_SCALE,
        "independence": {
            "train_holdout_disjoint": True,
            "reused_t05b_substream": "holdout=2 与 T0.5b PSD r3 的 validation 子流同编码同 seed 根",
        },
    }


def assert_distinct_streams(
    seed: int, pairs: Tuple[Tuple[str, str], ...], *, prefix: str = ""
) -> List[str]:
    """互斥性审计：给定 (stage, scene) 组合，任两条子流编码必须互不相同。

    由 ``tests/test_v2_p5b_t4_dgp_substreams.py`` 对 3 个阶段 × 4 个场景全组合调用，
    用于证明"训练与 holdout / 功效子流互斥"不依赖调用顺序。

    Returns:
        冲突列表（``[]`` 表无冲突）。
    """
    seen: Dict[Tuple[int, ...], str] = {}
    conflicts = []
    for stage, scene in pairs:
        item = substream(seed, stage, scene, prefix=prefix)
        if item.codes in seen:
            conflicts.append(f"{stage}|{scene} 与 {seen[item.codes]} 编码相同")
        else:
            seen[item.codes] = f"{stage}|{scene}"
    return conflicts
