# -*- coding: utf-8 -*-
"""P5b 历史试车场产物根与写域守卫（T1 共用件）。

契约来源：
- 设计 `docs/plans/v2/p5b_testbed_design_20261101.md` v4 §7.1（产物根
  ``data/paper_shadow/p5b_testbed/<campaign_id>/``）；
- 架构 F22（`docs/contracts/v2_architecture_plan.md`）：补
  ``data/paper_shadow/p5b_testbed/``，**不写** ``data/runs/``；校准证据一律
  ``evidence_class=calibration_only``。

写域口径（评审处置 R1-T1-04 / R2-T1-02）：
- 正面允许 = ``{data_root}/paper_shadow/p5b_testbed/`` 与项目 ``temp/``，
  其中 ``data_root`` **包含调用方声明的隔离数据根**（不再只锚定配置根）；
- 反面禁止 = 项目根与配置根下的 ``paper`` / ``runs`` / ``models``、配置
  ``data.paper`` 实解析值、以及**声明 data_root** 下的同名三目录；
- ``allow_external_root`` 只放开正面允许清单（测试/临时脚本），**永不解除**反面禁止。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set, Tuple

from loguru import logger

__all__ = [
    "ARM_ROLES",
    "CAMPAIGN_ID_PATTERN",
    "D0_SCHEMA",
    "EVIDENCE_CLASS",
    "SCOPE",
    "TESTBED_SUBDIR",
    "TestbedError",
    "TestbedScopeError",
    "assert_write_scope",
    "forbidden_roots",
    "project_root",
    "resolve_campaign_root",
    "testbed_root",
    "validate_campaign_id",
]

#: 试车场历史范围标识（报告字段 ``scope``）
SCOPE = "historical_testbed"
#: 校准证据类别（禁止冒充前向/SLA 证据）
EVIDENCE_CLASS = "calibration_only"
#: 试车场产物子目录（相对 data 根）
TESTBED_SUBDIR = Path("paper_shadow") / "p5b_testbed"
#: 允许的模拟臂角色（设计 v4 §4.2；真实候选不在其中）
ARM_ROLES: Tuple[str, ...] = (
    "aa_clone",
    "mechanical_perturbation",
    "synthetic_effect",
    "fault_injection",
)
#: campaign/arm 标识形态（小写字母/数字/下划线/连字符，3~64 位）
CAMPAIGN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]{2,63}$")
#: D0 fresh 记录 schema 版本
D0_SCHEMA = "p5b_testbed_d0_fresh/v1"
#: 生产根目录名（任何 data 根下都禁止写入）
_PRODUCTION_SUBDIRS = ("paper", "runs", "models")


class TestbedError(RuntimeError):
    """试车场通用错误基类（注册/身份/D0 共用）。"""

    # pytest 会把以 Test 开头的类当测试类收集；异常类显式退出收集
    __test__ = False


class TestbedScopeError(TestbedError):
    """写域越界（试图写生产根或未授权根）。"""

    __test__ = False


#: 仓库根 = 本文件（``src/lazybull/v2/hosts/testbed/paths.py``）上溯 5 级祖先目录。
#: 不经旧顶层包 ``src.lazybull`` 取 ``PROJECT_ROOT``——v2 依赖方向约束（T7 G 组 AST 扫描：
#: ``v2/hosts`` 不得 import 旧顶层包），且数值与原 ``Path(PROJECT_ROOT)`` 逐字符一致。
_REPO_ROOT = Path(__file__).parents[5]


def project_root() -> Path:
    """仓库根目录（``src/lazybull/v2/hosts/testbed`` 的上溯 5 级；不依赖配置文件）。"""
    return _REPO_ROOT


def _configured_data_root() -> Optional[Path]:
    """配置解析的 data 根；配置不可用则告警并返回 None（禁止静默退化）。"""
    try:
        from src.lazybull.common.config import get_data_root

        return Path(get_data_root())
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        logger.warning(
            f"试车场写域守卫：配置 data 根解析失败，退化为项目静态根 {project_root() / 'data'}"
            f"（{type(exc).__name__}: {exc}）"
        )
        return None


def _configured_paper_root() -> Optional[Path]:
    """配置 ``data.paper`` 实解析值（远端 SMB 形态与非绝对路径跳过）。"""
    try:
        from src.lazybull.common.config import get_paper_root

        raw = str(get_paper_root())
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        logger.warning(
            f"试车场写域守卫：配置 data.paper 解析失败，仅以静态 paper 根为禁写面"
            f"（{type(exc).__name__}: {exc}）"
        )
        return None
    # 远端形态：POSIX 双斜杠 / Windows UNC（\\host\share）/ 相对路径，均不入本地禁写面
    if raw.startswith("//") or raw.startswith("\\\\") or not Path(raw).is_absolute():
        return None
    return Path(raw)


def testbed_root(data_root: Optional[str] = None) -> Path:
    """试车场根 ``{data_root}/paper_shadow/p5b_testbed``。"""
    if data_root:
        base = Path(data_root)
    else:
        base = Path(_configured_data_root() or project_root() / "data")
    return base / TESTBED_SUBDIR


def forbidden_roots(data_root: Optional[str] = None) -> Tuple[Path, ...]:
    """无条件禁止写入的生产根（项目根 ∪ 配置根 ∪ 声明 data_root ∪ 配置 paper 根）。"""
    roots: Set[Path] = set()
    declared = Path(data_root) if data_root else None
    bases = (project_root() / "data", _configured_data_root(), declared)
    for base in bases:
        if base is None:
            continue
        roots.update(base / name for name in _PRODUCTION_SUBDIRS)
    paper_root = _configured_paper_root()
    if paper_root is not None:
        roots.add(paper_root)
    return tuple(sorted(roots))


def validate_campaign_id(campaign_id: str, *, name: str = "campaign_id") -> str:
    """校验 campaign/arm 标识形态（fail-fast，含上下文）。"""
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID_PATTERN.match(campaign_id):
        raise TestbedError(f"{name} 非法（须匹配 {CAMPAIGN_ID_PATTERN.pattern}）: {campaign_id!r}")
    return campaign_id


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def assert_write_scope(
    path: Path,
    *,
    data_root: Optional[str] = None,
    allow_external_root: bool = False,
) -> Path:
    """写域守卫：生产根无条件拒绝；其余仅允许试车场根/``temp/``。

    Args:
        path: 待写入的 campaign 根（或臂根）目录。
        data_root: 该 campaign 声明的隔离历史数据根；其下 ``paper``/``runs``/
            ``models`` 同时进入禁写面，正面允许根也随之为
            ``{data_root}/paper_shadow/p5b_testbed``。
        allow_external_root: 仅测试与临时脚本可开的显式逃生门；**不解除**
            生产根禁令。

    Raises:
        TestbedScopeError: 命中生产根，或落在授权写域之外且未显式放开。
    """
    resolved = Path(path).resolve()
    for forbidden in forbidden_roots(data_root):
        if _is_within(resolved, forbidden.resolve()):
            raise TestbedScopeError(
                f"试车场产物禁止写入生产根: {resolved}（命中 {forbidden}）；"
                "P5b 只允许写 data/paper_shadow/p5b_testbed/ 或 temp/"
            )
    if allow_external_root:
        return resolved
    allowed = (testbed_root(data_root).resolve(), (project_root() / "temp").resolve())
    if not any(_is_within(resolved, root) for root in allowed):
        raise TestbedScopeError(
            f"试车场产物根越界: {resolved}；只允许 "
            f"{allowed[0]} 或 {allowed[1]}（测试须显式 allow_external_root=True）"
        )
    return resolved


def resolve_campaign_root(
    campaign_id: str,
    *,
    base_root: Optional[str] = None,
    data_root: Optional[str] = None,
    allow_external_root: bool = False,
) -> Path:
    """解析并守卫 campaign 根：``{base_root or testbed_root(data_root)}/{campaign_id}``。"""
    validate_campaign_id(campaign_id)
    base = Path(base_root) if base_root else testbed_root(data_root)
    return assert_write_scope(
        base / campaign_id, data_root=data_root, allow_external_root=allow_external_root
    )
