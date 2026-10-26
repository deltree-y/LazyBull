# -*- coding: utf-8 -*-
"""训练帧缓存（P3-T2 加载成本门，§16.5-①；评审处置 R-CODEX-01/02 + reviewer1-P3-1）。

背景（reviewer1-P1 / §15.7）：现役链路逐种子重复加载同一折训练帧
（3 种子 × 14 折 × 2 臂外推 ≈12.3h，击穿机器预算）。本缓存使**同一折
``TrainingFrame`` 只加载一次、3 种子共享**——缓存键 = 窗口 + 列集 + 标签列 +
数据源身份（见 ``_key``）。

契约登记：

- 命中返回**同一** ``TrainingFrame`` 对象（不复制）——消费方不得原地修改
  ``frame.df``；唯一例外 = 运行时派生族的幂等派生（``prepare`` 内 derive 路径，
  已登记；验收臂五族全关 ⇒ 无原地写）；
- **每次访问（含命中）先执行 ``validate_frame_request`` 轻量守卫**（配置/四族/
  模式必需对象/验收可达性；不含 IO，合法命中零额外读取）——缓存不得绕过适配器
  的强制拒绝规则（R-CODEX-02：同键非法请求必须照常报错）；
- 注入函数参数名 = ``frame_loader``（R-CODEX-01：避免与适配器 ``loader``
  （旧 DataLoader）参数冲突，``loader`` 经 ``**loader_kwargs`` 自然透传）；
- ``runtime_lookups`` 不进缓存键（大对象）：同键请求必须传语义相同的 lookup
  （帧携带首次加载时的 lookups；验收臂为 None）；
- 源身份取 ``store.root`` / ``storage.root_path``；cs_train 模式下 loader 与
  storage 同源为调用约定（T3 组装侧固定同一实例，reviewer1-P3-1 登记）；
- 编排侧按折 ``clear()`` 释放（一折一清，避免跨折驻留大帧）。
"""

from typing import Any, Callable, Mapping

from src.lazybull.v2.train.data_adapter import (
    AdapterConfig,
    TrainingFrame,
    load_training_frame,
    validate_frame_request,
)


def _source_identity(loader_kwargs: Mapping[str, Any]) -> str:
    """数据源身份（缓存键组成部分）：panel = store.root；cs_train = storage.root_path。"""
    store = loader_kwargs.get("store")
    if store is not None:
        return f"panel:{getattr(store, 'root', store)!s}"
    storage = loader_kwargs.get("storage")
    if storage is not None:
        return f"cs_train:{getattr(storage, 'root_path', storage)!s}"
    return "unknown"


def _key(config: AdapterConfig, loader_kwargs: Mapping[str, Any]) -> tuple:
    """缓存键 = 窗口 + 列集 + 标签列 + 模式 + 域 + 冻结上界 + 四族 + 源身份 + 验收用途。"""
    return (
        config.source_mode,
        config.start_date,
        config.end_date,
        tuple(config.feature_columns),
        config.label_column,
        tuple(config.universe) if config.universe is not None else None,
        config.frozen_through,
        tuple(sorted(config.runtime_families)),
        _source_identity(loader_kwargs),
        loader_kwargs.get("acceptance_purpose"),
    )


class TrainingFrameCache:
    """同键去重的训练帧缓存（3 种子共享同折一帧；``get_or_load`` 命中即返同对象）。"""

    def __init__(self) -> None:
        self._entries: dict[tuple, TrainingFrame] = {}

    def get_or_load(
        self,
        config: AdapterConfig,
        *,
        frame_loader: Callable[..., TrainingFrame] = load_training_frame,
        **loader_kwargs: Any,
    ) -> TrainingFrame:
        """按缓存键加载或复用训练帧；``frame_loader`` 可注入（测试 spy 计数用）。

        每次调用（含命中）先执行 ``validate_frame_request`` 守卫（R-CODEX-02）。
        """
        validate_frame_request(config, **loader_kwargs)
        key = _key(config, loader_kwargs)
        if key not in self._entries:
            self._entries[key] = frame_loader(config, **loader_kwargs)
        return self._entries[key]

    def clear(self) -> None:
        """清空缓存（编排侧按折调用）。"""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
