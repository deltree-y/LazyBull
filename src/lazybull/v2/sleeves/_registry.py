# -*- coding: utf-8 -*-
"""袖子注册表（封闭，未知袖子 fail-fast；架构 §4.3 / 契约 protocols.md §6）。

**注册文件唯一**（R4 轻微 4）：注册条目字面量只允许出现在本文件；
全仓静态守卫测试锁定 ``SleeveSpec(`` 构造点（仅 ``sleeves/value/spec.py``
与 tests/）。现实先例 = ``universe/domains.py``（域注册表同形态）。

与契约 §6 ``SleeveRegistry`` 协议的关系（P3 规划 §18.1 登记备审）：契约签名
``register(sleeve_id, signal_class: type[Signal])`` 以「信号实现落 sleeves」终态为
前提；本期信号实现维持 ``v2/core/signal/`` 单处（§1.1-4），协议代码化与
「注册条目 = spec vs signal_class」形态对齐留待评审裁决（如需，走 F 版本号
契约修订流程）。
"""

from types import MappingProxyType

from src.lazybull.v2.sleeves.spec import SleeveSpec
from src.lazybull.v2.sleeves.value import VALUE_SLEEVE_SPEC

__all__ = ["get_sleeve_spec", "list_sleeve_ids"]

#: 已注册袖子（条目字面量唯一定义点；MappingProxyType 封闭可写面——语言层只读，
#: reviewer1-P3-1；值对象本身 frozen，双重不可变）
_REGISTRY: MappingProxyType[str, SleeveSpec] = MappingProxyType(
    {
        VALUE_SLEEVE_SPEC.sleeve_id: VALUE_SLEEVE_SPEC,
    }
)


def get_sleeve_spec(sleeve_id: str) -> SleeveSpec:
    """按 ID 获取袖子规格；未知取值直接报错（fail-fast，禁止静默回退）。

    查询侧严格化（reviewer1-P3-R4）：非 str / 前后空白同样拒绝——与注册键
    口径一致，不静默纠正（避免掩盖上游配置拼写错误）。

    Args:
        sleeve_id: 袖子唯一 ID（如 ``"value"``）

    Returns:
        对应 ``SleeveSpec``（不可变规格对象）

    Raises:
        ValueError: 袖子未登记或 ID 非法（非 str / 空 / 含前后空白）
    """
    if not isinstance(sleeve_id, str) or not sleeve_id or sleeve_id != sleeve_id.strip():
        raise ValueError(
            f"未知袖子: {sleeve_id!r}；已登记袖子: {sorted(_REGISTRY)}"
            "（袖子 ID 必须为非空无前后空白字符串）"
        )
    if sleeve_id not in _REGISTRY:
        raise ValueError(
            f"未知袖子: {sleeve_id!r}；已登记袖子: {sorted(_REGISTRY)}"
            "（新增袖子必须先以规格文件 + 本注册表条目登记）"
        )
    return _REGISTRY[sleeve_id]


def list_sleeve_ids() -> tuple[str, ...]:
    """列出全部已注册袖子 ID（注册表插入序）。"""
    return tuple(_REGISTRY)
