# -*- coding: utf-8 -*-
"""P6 T5 真实 P5b golden 独立核验（非 pytest、非 skipif；真实根缺失/不符即失败）。

设计口径（设计 §5.6 / §8.3、计划 §3-T5 第 4 条）：

- 三族 ``t4_seq10`` / ``t5_seq7`` / ``t6_seq21`` 的真实 campaign **逐族**核验：
  ① **精确路径**：campaign 目录（相对 campaigns 根）+ ``reports/`` + 文件集合**恰等于**冻结件
  （缺文件、多文件都失败）；
  ② **seal**：逐文件按上游原生读口 ``load_artifact(..., sealed=True)`` 校验 ``record_sha256``
  自封存与 schema（不自造字段、不改写 P5b 冻结字节）；
  ③ **链**：``hash_chain``（语义键）与 ``identity`` 回指 + ``manifest.files`` 上游引用，
  经 ``bundle.verify_p5b_bundle`` 判定 ``available``（任一断链即失败）；
  ④ **值**：冻结件登记的**点分路径 → 期望标量**逐个比对（campaign_id / run_id / schema /
  evidence_class / 关键判定标量），值不符即失败。
- 期望值来源 = ``docs/contracts/p6_t5_real_golden_freeze_<date>.json``（一次性冻结、之后
  append-only；重新冻结必须写新文件名并留下变更理由）。**本脚本不硬编码任何真实 campaign 名、
  路径或数值**——全部来自冻结件，避免“实现侧自证”。
- 真实根缺失/不符一律 ``verdict=fail``（退出码 3）；不使用 pytest、不使用 skipif/xfail。
- 写域（设计 §2.5）：只允许本次新 validation 目录；脚本内启用运行期 ``ReadonlyGuard``
  （禁网络/禁子进程），越界写入直接失败。

用法::

    # 核验（真实证据）
    python scripts/v2_p6/check_real_golden.py \\
        --root data/paper_shadow/p5b_testbed \\
        --out data/reports/p6/validations/research/<new_id>

    # 重新冻结（显式、必须写新文件名；拒绝覆盖）
    python scripts/v2_p6/check_real_golden.py \\
        --root data/paper_shadow/p5b_testbed \\
        --campaign t4_seq10=<rel_dir> --campaign t5_seq7=<rel_dir> --campaign t6_seq21=<rel_dir> \\
        --freeze-out docs/contracts/p6_t5_real_golden_freeze_<new_date>.json

退出码：``0`` 通过 / ``3`` 判据失败（真实根缺失、文件集合或 SHA/链/值不符）/ ``2`` 设置错误。
"""

# 复杂度豁免：阶段专用验收工具（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256

if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）
    sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed.storage import TestbedError  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    bundle as bundle_module,
    contracts,
    readonly_audit,
    validation,
)

__all__ = [
    "DEFAULT_FREEZE",
    "FREEZE_SCHEMA",
    "GoldenSetupError",
    "check_root",
    "freeze_root",
    "main",
]

#: 冻结件 schema（``p6_t5_real_golden_freeze/v1``）
FREEZE_SCHEMA = "p6_t5_real_golden_freeze/v1"
#: 默认冻结件（仓库相对 POSIX）
DEFAULT_FREEZE = "docs/contracts/p6_t5_real_golden_freeze_20261107.json"
#: 三族（顺序固定；与 ``bundle.BUNDLE_FAMILIES`` 同键）
FAMILIES: tuple[str, ...] = ("t4_seq10", "t5_seq7", "t6_seq21")
#: 各族“关键值”点分路径（**路径**可硬编码，**值**只能来自冻结件）
VALUE_PATHS: dict[str, dict[str, tuple[str, ...]]] = {
    "t4_seq10": {
        "verdict": (
            "campaign_id",
            "run_id",
            "schema",
            "implementation_id",
            "evidence_class",
            "not_forward_evidence",
            "income_referee.aa_passed",
            "income_referee.independent_acceptance",
            "income_referee.unit_count",
            "income_referee.wilson_upper_max",
            "income_referee.mde_grid_upper_level",
        ),
    },
    "t5_seq7": {
        "verdict": (
            "campaign_id",
            "run_id",
            "schema",
            "implementation_id",
            "evidence_class",
            "not_forward_evidence",
            "h5_conditions.all_conditions_met",
            "h5_conditions.holdout_gate_only",
            "drawdown_referee.referee_usable",
            "drawdown_referee.MDE_drawdown_improve.aa_gate_passed",
            "drawdown_referee.MDE_drawdown_worsen.aa_gate_passed",
        ),
    },
    "t6_seq21": {
        "verdict": (
            "campaign_id",
            "run_id",
            "schema",
            "evidence_class",
            "not_forward_evidence",
            "gate_owner",
            "h1_claim",
            "h6_claim",
            "h7_claim",
        ),
    },
}


class GoldenSetupError(RuntimeError):
    """参数/冻结件/输出目录不满足契约（不是判据失败，退出码 2）。"""


def _sha256_file(path: Path) -> str:
    """文件字节 SHA-256（分块读；与 ``record_sha256`` 口径分离）。"""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(document: Mapping[str, Any], dotted: str) -> Any:
    """点分路径取值；缺失返回 ``_MISSING``。"""
    current: Any = document
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING: Any = object()


def _family_spec(family: str) -> dict[str, Any]:
    """取上游原生族描述（文件名/schema/loader 全部来自上游，不手抄）。"""
    spec = bundle_module.BUNDLE_FAMILIES.get(family)
    if spec is None:  # 防御：族名与上游不同步即设置错误
        raise GoldenSetupError(f"未知 P5b bundle 族：{family!r}")
    return dict(spec)


def _reports_relative(family: str) -> str:
    """``reports`` 子目录名（从冻结件结构推断的同位常量；P5b 三类产物同布局）。"""
    return "reports"


def _values_of(
    family: str, documents: Mapping[str, Mapping[str, Any]], verdict_name: str
) -> dict[str, Any]:
    """按 ``VALUE_PATHS`` 抽取关键值（文档缺失 ⇒ 空映射，由链校验先失败）。"""
    document = documents.get(verdict_name)
    if document is None:
        return {}
    values: dict[str, Any] = {}
    for dotted in VALUE_PATHS[family]["verdict"]:
        value = _lookup(document, dotted)
        if value is _MISSING:
            continue
        values[dotted] = value
    return values


def _load_sealed(family: str, directory: Path) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """按上游原生读口加载各族文档（封存 + schema 一并校验）。"""
    spec = _family_spec(family)
    documents: dict[str, Mapping[str, Any]] = {}
    problems: list[str] = []
    names = [str(spec["manifest"]), str(spec["report"]), str(spec["verdict"])]
    names.extend(str(item) for item in dict(spec["aux"]).values())
    for name in names:
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: 文件缺失")
            continue
        try:
            documents[name] = spec["loader"](
                path,
                what=f"P5b 产物 {name}",
                sealed=True,
                schema=dict(spec["schemas"]).get(name),
            )
        except (TestbedError, OSError, json.JSONDecodeError) as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
    return documents, problems


def _file_set_problems(
    actual: Sequence[str], frozen: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    """文件集合精确比对：返回 ``(缺失, 多出)``（集合不等即失败，不按“至少包含”放行）。"""
    frozen_names = sorted(str(name) for name in frozen)
    actual_names = sorted(str(name) for name in actual)
    return (
        [name for name in frozen_names if name not in actual_names],
        [name for name in actual_names if name not in frozen_names],
    )


def _digest_problems(directory: Path, frozen_files: Mapping[str, str]) -> list[str]:
    """逐文件字节 SHA 比对（缺失/多出由 :func:`_file_set_problems` 承担）。"""
    problems: list[str] = []
    for name, expected in sorted(frozen_files.items()):
        path = directory / str(name)
        if not path.is_file():
            continue
        actual = _sha256_file(path)
        if actual != str(expected):
            problems.append(f"{name}: 字节 {actual[:12]} ≠ 冻结 {str(expected)[:12]}")
    return problems


def _value_problems(
    family: str,
    documents: Mapping[str, Mapping[str, Any]],
    frozen_values: Mapping[str, Any],
    verdict_name: str,
) -> list[str]:
    """冻结值逐个比对（缺失与不符分别登记）。"""
    document = documents.get(verdict_name)
    if document is None:
        return [f"{verdict_name}: 文档未加载，无法比对 {len(frozen_values)} 项冻结值"]
    problems: list[str] = []
    for dotted, expected in sorted(frozen_values.items()):
        actual = _lookup(document, dotted)
        if actual is _MISSING:
            problems.append(f"{verdict_name}: 冻结路径 {dotted} 在文档中缺失")
        elif actual != expected:
            problems.append(f"{verdict_name}: {dotted}={actual!r} ≠ 冻结 {expected!r}")
    return problems


def _family_check(family: str, root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """单族核验（路径 → 文件集合 → SHA → seal → 链 → 值）。"""
    campaign_dir = root / str(entry.get("campaign_dir") or "")
    reports = campaign_dir / _reports_relative(family)
    spec = _family_spec(family)
    verdict_name = str(spec["verdict"])
    frozen_files = dict(entry.get("files") or {})
    result: dict[str, Any] = {
        "family": family,
        "campaign_dir": str(entry.get("campaign_dir") or ""),
        "reports_dir": f"{entry.get('campaign_dir')}/{_reports_relative(family)}",
        "frozen_file_count": len(frozen_files),
        "checks": {},
        "failures": [],
    }
    if not reports.is_dir():
        result["failures"].append(f"campaign reports 目录不存在：{reports}")
        result["checks"]["reports_dir_present"] = False
        result["verdict"] = "fail"
        return result
    result["checks"]["reports_dir_present"] = True
    # 精确集合不能只看单层：子目录、符号链接、任何非冻结条目都必须计入“多出”（评审 A-08）
    actual_entries = sorted(path.relative_to(reports).as_posix() for path in reports.rglob("*"))
    actual_dirs = [
        path.relative_to(reports).as_posix() for path in reports.rglob("*") if path.is_dir()
    ]
    actual_symlinks = [
        path.relative_to(reports).as_posix() for path in reports.rglob("*") if path.is_symlink()
    ]
    actual_files = [name for name in actual_entries if name not in actual_dirs]
    missing, extra = _file_set_problems(actual_files, frozen_files)
    extra = sorted(set(extra) | set(actual_dirs) | set(actual_symlinks))
    result["checks"]["file_set_exact"] = not missing and not extra
    result["checks"]["actual_file_count"] = len(actual_files)
    result["checks"]["unexpected_dirs"] = actual_dirs
    result["checks"]["unexpected_symlinks"] = actual_symlinks
    result["failures"].extend(f"缺文件：{name}" for name in missing)
    result["failures"].extend(
        f"多出未冻结文件：{name}" for name in extra if name not in actual_dirs
    )
    result["failures"].extend(f"多出未冻结目录：{name}" for name in actual_dirs)
    if extra:
        result["checks"]["unexpected_files"] = extra
    digest = _digest_problems(reports, frozen_files)
    result["checks"]["sha_match"] = not digest
    result["failures"].extend(digest)
    documents, seal_problems = _load_sealed(family, reports)
    result["checks"]["sealed_load"] = not seal_problems
    result["failures"].extend(seal_problems)
    bundle_row = bundle_module.verify_p5b_bundle(family, reports, scope_base=root)
    chain_ok = bundle_row.get("report_status") == "available"
    result["checks"]["bundle_chain"] = chain_ok
    result["chain"] = {
        "report_status": bundle_row.get("report_status"),
        "reason_code": bundle_row.get("degradation_reason_code"),
        "campaign_identity": bundle_row.get("campaign_identity"),
        "failures": list(bundle_row.get("failures") or []),
        "external_refs": len(bundle_row.get("external_refs") or []),
    }
    if not chain_ok:
        result["failures"].extend(list(bundle_row.get("failures") or [])[:5])
    values = _values_of(family, documents, verdict_name)
    frozen_values = dict((entry.get("values") or {}).get("verdict") or {})
    value_problems = _value_problems(family, documents, frozen_values, verdict_name)
    result["checks"]["values_match"] = not value_problems
    result["failures"].extend(value_problems)
    result["values"] = values
    result["frozen_values"] = frozen_values
    seals = {
        name: document.get("record_sha256")
        for name, document in sorted(documents.items())
        if isinstance(document.get("record_sha256"), str)
    }
    result["seals"] = seals
    result["frozen_seals"] = dict(entry.get("seals") or {})
    seal_mismatch = [
        name
        for name, expected in sorted(dict(entry.get("seals") or {}).items())
        if seals.get(name) != expected
    ]
    result["checks"]["seal_match"] = not seal_mismatch
    result["failures"].extend(f"{name}: record_sha256 与冻结件不符" for name in seal_mismatch)
    result["verdict"] = "pass" if not result["failures"] else "fail"
    return result


def _families_of(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """冻结件中的家族映射（形态非法 ⇒ 设置错误）。"""
    families = freeze.get("families")
    if not isinstance(families, Mapping) or not families:
        raise GoldenSetupError("冻结件缺 families 映射（形态非法）")
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise GoldenSetupError(f"冻结件含未登记族：{unknown}（合法：{list(FAMILIES)}）")
    # 缺族即 fail-closed：三族是 T5 的真实 golden 面，冻结件只写 1 族不得被当成“全通过”。
    # 只有显式写 ``require_all_families: false``（单测用单族/双族夹具）才允许子集。
    require_all = freeze.get("require_all_families", True)
    if not isinstance(require_all, bool):
        raise GoldenSetupError(
            f"冻结件 require_all_families 必须是布尔：{require_all!r}（缺省 true）"
        )
    missing = [family for family in FAMILIES if family not in families]
    if require_all and missing:
        raise GoldenSetupError(f"冻结件缺族：{missing}（必须覆盖：{list(FAMILIES)}）")
    return dict(families)


def load_freeze(path: Path | str) -> dict[str, Any]:
    """读取并做形态校验的冻结件。"""
    target = Path(path)
    if not target.is_file():
        raise GoldenSetupError(f"冻结件不存在：{target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenSetupError(f"冻结件不可解析：{target}（{type(exc).__name__}: {exc}）") from exc
    if payload.get("schema") != FREEZE_SCHEMA:
        raise GoldenSetupError(f"冻结件 schema={payload.get('schema')!r} 不是 {FREEZE_SCHEMA}")
    _families_of(payload)
    return payload


def check_root(root: Path | str, freeze: Mapping[str, Any]) -> dict[str, Any]:
    """对 campaigns 根核验全部冻结族；**任一失败即 fail**（不 skip、不降级）。"""
    base = Path(root)
    families = _families_of(freeze)
    if not base.is_dir():
        return {
            "schema": "p6_t5_real_golden/v1",
            "root_present": False,
            "families": {
                family: {"family": family, "verdict": "fail"} for family in sorted(families)
            },
            "failed_families": sorted(families),
            "verdict": "fail",
            "detail": f"真实 campaign 根不存在：{base}（真实根缺失即失败，禁止 skipif）",
        }
    results: dict[str, Any] = {}
    for family in sorted(families):
        results[family] = _family_check(family, base, dict(families[family]))
    failed = sorted(name for name, row in results.items() if row.get("verdict") != "pass")
    return {
        "schema": "p6_t5_real_golden/v1",
        "root_present": True,
        "families": results,
        "failed_families": failed,
        "verdict": "pass" if not failed else "fail",
    }


def freeze_root(root: Path | str, campaigns: Mapping[str, str]) -> dict[str, Any]:
    """从真实字节**一次性冻结**期望（路径/文件集/SHA/seal/值）；拒绝覆盖既有文件。"""
    base = Path(root)
    if not base.is_dir():
        raise GoldenSetupError(f"真实 campaign 根不存在，无法冻结：{base}")
    missing = sorted(set(FAMILIES) - set(campaigns))
    if missing:
        raise GoldenSetupError(f"冻结必须给全三族 campaign 目录，缺 {missing}")
    families: dict[str, Any] = {}
    for family in FAMILIES:
        relative = str(campaigns[family]).replace("\\", "/").strip("/")
        reports = base / relative / _reports_relative(family)
        if not reports.is_dir():
            raise GoldenSetupError(f"{family} 的 reports 目录不存在：{reports}")
        spec = _family_spec(family)
        files = {
            path.name: _sha256_file(path)
            for path in sorted(reports.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        documents, problems = _load_sealed(family, reports)
        if problems:
            raise GoldenSetupError(f"{family} 冻结前封存校验失败：{problems}")
        seals = {
            name: document.get("record_sha256")
            for name, document in sorted(documents.items())
            if isinstance(document.get("record_sha256"), str)
        }
        families[family] = {
            "campaign_dir": relative,
            "files": files,
            "seals": seals,
            "values": {
                "verdict": _values_of(family, documents, str(spec["verdict"])),
            },
        }
    return {
        "schema": FREEZE_SCHEMA,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source_root": base.as_posix(),
        "families": families,
    }


def _assert_new_validation_out(out: Path) -> Path:
    """``--out`` 必须是尚不存在的 ``validations/<evidence_class>/<validation_id>``。"""
    target = out.resolve()
    validations_root = (ROOT / contracts.REPORTS_ROOT / "validations").resolve()
    try:
        relative = target.relative_to(validations_root)
    except ValueError as exc:
        raise GoldenSetupError(f"输出目录不在 validation 根下：{target}") from exc
    if len(relative.parts) != 2 or relative.parts[0] not in contracts.EVIDENCE_CLASSES:
        raise GoldenSetupError(f"输出目录形态非法：{relative}")
    validation.validate_validation_id(relative.parts[1])
    if target.exists():
        raise GoldenSetupError(f"输出目录已存在，拒绝覆盖：{target}")
    return target


def _guard(target: Path) -> Any:
    """运行期只读守卫：只允许写本次 validation 目录，禁网络/禁子进程。"""
    evidence_class, validation_id = target.parts[-2], target.parts[-1]
    return readonly_audit.ReadonlyGuard(
        allowed_patterns=(
            f"{contracts.REPORTS_ROOT}/validations/<evidence_class>/<validation_id>/**",
        ),
        worktree_root=ROOT,
        bindings={"evidence_class": evidence_class, "validation_id": validation_id},
        deny_network=True,
        deny_subprocess=True,
    )


def _write_check_output(
    out: Path, root: Path, freeze_path: Path, freeze: Mapping[str, Any], payload: Mapping[str, Any]
) -> Path:
    """写出 ``real_golden.json``（新文件、UTF-8/LF、拒绝覆盖）。"""
    document = dict(payload)
    document["root"] = root.as_posix()
    document["freeze"] = freeze_path.as_posix()
    document["freeze_schema"] = freeze.get("schema")
    document["frozen_at"] = freeze.get("frozen_at")
    document["generated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return validation.write_json_new(out / "real_golden.json", document)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口（核验 / 重新冻结两种模式；退出码见模块 docstring）。"""
    parser = argparse.ArgumentParser(description="P6 T5 真实 P5b golden 独立核验（非 pytest）")
    parser.add_argument(
        "--root", required=True, help="真实 campaigns 根（如 data/paper_shadow/p5b_testbed）"
    )
    parser.add_argument("--out", default=None, help="新的 validation 目录（核验模式必填）")
    parser.add_argument("--freeze", default=DEFAULT_FREEZE, help="冻结件路径（核验模式）")
    parser.add_argument("--freeze-out", default=None, help="重新冻结输出路径（必须不存在）")
    parser.add_argument(
        "--campaign",
        action="append",
        default=None,
        metavar="FAMILY=REL_DIR",
        help="冻结模式：族 → 相对 campaigns 根的 campaign 目录（可重复，三族必给全）",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        if args.freeze_out:
            campaigns: dict[str, str] = {}
            for item in args.campaign or ():
                family, _, relative = str(item).partition("=")
                if family not in FAMILIES or not relative:
                    raise GoldenSetupError(f"--campaign 形态须为 FAMILY=REL_DIR：{item!r}")
                campaigns[family] = relative
            freeze = freeze_root(root, campaigns)
            target = Path(args.freeze_out)
            if target.exists():
                raise GoldenSetupError(f"冻结件已存在，拒绝覆盖（重新冻结请换新文件名）：{target}")
            validation.write_json_new(target, freeze)
            print(
                json.dumps(
                    {"frozen": target.as_posix(), "families": sorted(freeze["families"])},
                    ensure_ascii=False,
                )
            )
            return 0
        if not args.out:
            raise GoldenSetupError("核验模式必须给出 --out（新的 validation 目录）")
        out = _assert_new_validation_out(Path(args.out))
        freeze_path = Path(args.freeze)
        freeze = load_freeze(freeze_path)
        with _guard(out):
            payload = check_root(root, freeze)
            validation.ensure_new_dir(out)
            _write_check_output(out, root, freeze_path, freeze, payload)
    except (GoldenSetupError, validation.ValidationError) as exc:
        print(f"[P6-T5] 设置错误：{exc}", file=sys.stderr)
        return 2
    except (OSError, readonly_audit.ReadonlyViolationError) as exc:
        print(f"[P6-T5] 执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "failed_families": payload.get("failed_families", []),
                "out": out.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["verdict"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
