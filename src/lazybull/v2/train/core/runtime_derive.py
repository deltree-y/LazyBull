# -*- coding: utf-8 -*-
"""运行时派生族派生能力注入容器（P3-T2，§16.4）。

依赖方向契约（`v2/train/__init__.py` docstring）：train 禁止 import factors。
运行时派生五族（holdertrade / repurchase / top10fh / top_inst / availability_markers）
的派生计算本体在 factors 包（§1.2 不改因子逻辑、§4.3 factors 沿用），由**调用方**
（薄入口 / 宿主）组装为 ``RuntimeDerivers`` 注入 ``prepare_training_data``；
族启用而未注入对应 deriver ⇒ 报错（与「启用而未传派生表必须报错」契约同级）。

验收臂五族全关（batch_walk_forward.ps1 生产口径实测），DI 路径不触发但代码与
测试闭合（测试从 factors 组装真实 derivers 注入，旧新对跑逐位一致）。
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class RuntimeFamilyDeriver:
    """单一运行时派生族的派生能力（factors 计算函数由调用方组装注入）。

    Attributes:
        version_col: 哨兵列名（取值 = factors 侧 ``*_VERSION_COL``，如
            ``holdertrade_schema_v1``）；
        schema_version: 哨兵期望版本（取值 = factors 侧 ``*_SCHEMA_VERSION``）；
        feature_columns_fn: 列集函数——**逐族调用约定**（reviewer1-P1-1 处置登记）：
            holdertrade / repurchase / top10fh 带 ``feature_set`` 单参调用；
            top_inst 无 feature_set 维度 ⇒ **无参调用**（对齐真实
            ``factors.top_inst.top_inst_feature_columns()`` 签名）；
        derive_fn: 派生函数，调用形态 ``derive_fn(df, source, wanted=..., log_prefix=...)``
            （source 为逐日查询表 dict 或报告期面板 DataFrame，随族而定）。
    """

    version_col: str
    schema_version: int
    feature_columns_fn: Callable[..., List[str]]
    derive_fn: Callable[..., List[str]]


@dataclass(frozen=True)
class RuntimeDerivers:
    """五族注入容器（默认全 None = 无派生能力；启用族而未注入 ⇒ 报错）。

    availability 为可用性标记派生函数（调用形态 ``deriver(df) -> List[str]``，
    返回新增标记列名；无哨兵/列集维度，与四族分开建模）。
    """

    holdertrade: Optional[RuntimeFamilyDeriver] = None
    repurchase: Optional[RuntimeFamilyDeriver] = None
    top10fh: Optional[RuntimeFamilyDeriver] = None
    top_inst: Optional[RuntimeFamilyDeriver] = None
    availability: Optional[Callable[[pd.DataFrame], List[str]]] = None
