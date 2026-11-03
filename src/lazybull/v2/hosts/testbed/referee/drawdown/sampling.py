# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判的命名 seed 子流（设计 v4 §5.3「命名种子」+ §5.5「路径数/种子」行）。

口径
====
每个验收单元共享同一 seed 根，用
``SeedSequence([seed, namespace_code, stage_code, scene_code, null_code, level_code])``
派生子流：

- ``namespace_code = 5`` 是**回撤裁判独立命名空间**：即便 seed 根、阶段、场景、零假设与收益
  裁判完全相同，编码向量也必然不同 ⇒ 两套仪器的路径互斥（收益证据不被回撤证据复用）；
- ``null_code`` 在本仪器里的语义是**方向/零假设槽**：``aa=0``（A/A 独立路径）、
  ``dd_i=1``（改善效应档）、``dd_w=2``（恶化效应档）；DD-W 与 DD-I **共用同一组 A/A 读数**
  （同 ``aa`` 子流），这是 §5.4.2 方向自检的前提；
- A/A 每个实现是**独立抽取**的 ``x = B0 + n``，减数是**冻结基准臂 B0**（见
  :mod:`realizations` / :mod:`thresholds`）：这不是"同一路径零注入"（后者会让 ``D_k ≡ 0``），
  也保证读数的噪声结构与效应单元一致（效应单元的读数同样以 B0 为减数）；
- 训练 / holdout / 效应是互斥子流，禁止把 holdout 放进训练子流。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee import sampling as income_sampling

__all__ = [
    "DIRECTION_CODES",
    "IMPLEMENTATION_ID",
    "NAMESPACE_CODE",
    "SCENE_CODES",
    "STAGE_ALIASES",
    "STAGE_CODES",
    "STAGES",
    "DrawdownSubstream",
    "SamplingError",
    "assert_independent_of_income",
    "implementation_ids",
    "stream_codes",
    "substream",
]

#: 回撤裁判实现编号（与收益裁判 ``p5b_t4_income_referee/v1`` 并列、互不复用）
IMPLEMENTATION_ID = "p5b_t5_drawdown_referee/v1"
#: 独立命名空间编码（插入到 SeedSequence 编码向量第 2 位）
NAMESPACE_CODE = 5
#: 阶段编码（与设计 §5.3 的 train/holdout/effect 三分一致）
STAGE_CODES: Mapping[str, int] = {"train": 1, "holdout": 2, "effect": 3}
#: 阶段别名（realization_id 里的层级语义）
STAGE_ALIASES: Mapping[str, str] = {"train": "train", "holdout": "validation", "effect": "effect"}
#: 允许阶段
STAGES: Tuple[str, ...] = ("train", "holdout", "effect")
#: 场景编码（沿用收益裁判的四类依赖场景）
SCENE_CODES: Mapping[str, int] = {
    "ma20": 1,
    "ar1_06": 2,
    "regime_hetero": 3,
    "real_seed_block": 4,
}
#: 方向/零假设槽编码（A/A 与两个注入方向分别独立）
DIRECTION_CODES: Mapping[str, int] = {"aa": 0, "dd_i": 1, "dd_w": 2}
#: 效应档编码倍率（level(pp) * 1000 取整）
_LEVEL_SCALE = 1000


class SamplingError(TestbedError):
    """子流参数非法（未知阶段/场景/方向、非有限档位、与收益裁判编码撞车）。"""

    __test__ = False


@dataclass(frozen=True)
class DrawdownSubstream:
    """一个命名子流：生成器 + 身份（实现编号 / 阶段 / 场景 / 方向槽 / realization_id）。"""

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
        SamplingError: 阶段/场景/方向未登记或 ``level`` 非有限。
    """
    if stage not in STAGE_CODES:
        raise SamplingError(f"未登记阶段: {stage!r}（允许 {list(STAGE_CODES)}）")
    if scene not in SCENE_CODES:
        raise SamplingError(f"未登记场景: {scene!r}（允许 {list(SCENE_CODES)}）")
    if null and null not in DIRECTION_CODES:
        raise SamplingError(f"未登记方向槽: {null!r}（允许 {list(DIRECTION_CODES)}）")
    if not np.isfinite(float(level)):
        raise SamplingError(f"效应档必须有限: {level!r}")
    return (
        int(seed),
        int(NAMESPACE_CODE),
        int(STAGE_CODES[stage]),
        int(SCENE_CODES[scene]),
        int(DIRECTION_CODES.get(null, 0)),
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
) -> DrawdownSubstream:
    """构造命名子流（同参数必得同一 Generator 与同一 realization_id）。"""
    codes = stream_codes(seed, stage, scene, null, level)
    generator = np.random.default_rng(np.random.SeedSequence(list(codes)))
    alias = STAGE_ALIASES[stage]
    label = null or "aa"
    realization_id = f"{prefix}:{alias}:{scene}:{label}:{float(level):g}".lstrip(":")
    return DrawdownSubstream(
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
    """manifest ``substreams`` 段（命名空间 + 编码 + 独立性声明）。"""
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "scheme": (
            "numpy SeedSequence([seed, namespace_code, stage_code, scene_code, "
            "direction_code, level_code])"
        ),
        "namespace_code": int(NAMESPACE_CODE),
        "stages": dict(STAGE_CODES),
        "stage_aliases": dict(STAGE_ALIASES),
        "scene_codes": dict(SCENE_CODES),
        "direction_codes": dict(DIRECTION_CODES),
        "level_code_scale": _LEVEL_SCALE,
        "independence": {
            "train_holdout_disjoint": True,
            "independent_of_income_referee": True,
            "aa_shared_across_directions": True,
            "aa_uses_independent_paths_not_zero_injection": True,
        },
    }


def assert_independent_of_income(seed: int, pairs: Tuple[Tuple[str, str], ...]) -> List[str]:
    """与收益裁判子流互斥的审计（命名空间不同 ⇒ 编码必然不同）。

    Args:
        seed: 命名种子根。
        pairs: ``(stage, scene)`` 组合。

    Returns:
        冲突列表（``[]`` = 与收益裁判无撞车）。
    """
    conflicts: List[str] = []
    for stage, scene in pairs:
        ours = stream_codes(seed, stage, scene)
        theirs = income_sampling.stream_codes(seed, stage, scene)
        if tuple(ours) == tuple(theirs):
            conflicts.append(f"{stage}|{scene} 与收益裁判子流编码相同")
    return conflicts
