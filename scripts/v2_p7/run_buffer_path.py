"""固定Top20续持的同态资金路径对照入口，只写新的研究证据根。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p7.buffer_path_account import run_account  # noqa: E402
from scripts.v2_p7.buffer_path_data import load_inputs  # noqa: E402
from scripts.v2_p7.buffer_path_metrics import compare_accounts  # noqa: E402

EXPECTED_SPEC = "208ff4dace85928672f381e6eb38f113d36d76c6402ec4804a957106208f9add"
CAPITALS = (1000000, 2000000)
TIME_LIMIT_SECONDS = 1800


def sha256(path: Path) -> str:
    """按实际文件字节登记身份。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_default(value: Any) -> Any:
    """只容纳numpy标量和路径；未知对象拒绝静默转字符串。"""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"证据含不支持对象：{type(value).__name__}")


def canonical(value: Any) -> str:
    """确定性规范JSON用于机械A/A逐值比较。"""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=json_default
    )


def write_json(path: Path, value: Any) -> None:
    """新件拒绝覆盖，保留精确数值。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=json_default
        )


def capture_dependencies(root: Path, cache: Path, spec: Path) -> dict[str, str]:
    """冻结研究方法、账户依赖、现金事件与共享缓存，价格/模型由缓存身份负责。"""
    files = {spec}
    files.update((root / "src/lazybull").rglob("*.py"))
    files.update((root / "scripts/v2_p7/t4").glob("*.py"))
    files.update((root / "scripts/v2_p7/t25").glob("*.py"))
    files.update((root / "scripts/v2_p7").glob("buffer_path_*.py"))
    files.add(Path(__file__).resolve())
    files.update((root / "tests").glob("test_p7_buffer_path_*.py"))
    files.update((root / "data/raw/dividend").glob("*.parquet"))
    files.update(p for p in cache.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}


def mechanical_pair(left: dict, right: dict) -> dict:
    """逐表完整比较；不把确定性检查解释为旧生产账户等价。"""
    result = {}
    for key in ["daily", "trades", "decisions", "holdings", "blocked", "corporate"]:
        a, b = canonical(left[key]), canonical(right[key])
        result[key] = {
            "left_rows": len(left[key]),
            "right_rows": len(right[key]),
            "left_sha256": hashlib.sha256(a.encode("utf-8")).hexdigest(),
            "right_sha256": hashlib.sha256(b.encode("utf-8")).hexdigest(),
            "equal": a == b,
        }
    result["all_equal"] = all(row["equal"] for row in result.values())
    result["production_equivalence_claimed"] = False
    return result


def budget_check(started: float, cache_seconds: float) -> None:
    """预算包含当次共享缓存构建，超限不发布完整结果。"""
    if time.perf_counter() - started + cache_seconds > TIME_LIMIT_SECONDS:
        raise TimeoutError("正式计算含共享缓存超过冻结30分钟上限")


def run(root: Path, cache: Path, output: Path, spec: Path) -> dict:
    """执行四经济账户、两份机械A/A及配对统计，无结果驱动调参。"""
    started = time.perf_counter()
    if sha256(spec) != EXPECTED_SPEC:
        raise ValueError("预登记规格SHA不符，拒绝用改后规格追认已读结果")
    if output.exists():
        raise FileExistsError(f"证据根已存在：{output}")
    output.mkdir(parents=True, exist_ok=False)
    methods = output / "methods"
    methods.mkdir()
    shutil.copyfile(spec, methods / spec.name)
    sources = sorted((root / "scripts/v2_p7").glob("buffer_path_*.py"))
    sources += [Path(__file__).resolve()]
    sources += sorted((root / "tests").glob("test_p7_buffer_path_*.py"))
    for path in sources:
        shutil.copyfile(path, methods / path.name)
    inputs = load_inputs(root, cache, spec_sha256=EXPECTED_SPEC)
    cache_validation = json.loads((cache / "validation.json").read_text(encoding="utf-8"))
    cache_seconds = float(cache_validation["preparation_seconds"])
    identities = capture_dependencies(root, cache, spec)
    write_json(output / "input_identity.json", identities)
    timing = {"cache_build_seconds": cache_seconds}
    all_accounts, paired, aa = {}, {}, {}
    for capital in CAPITALS:
        for enabled in (False, True):
            name = f"{'buffer' if enabled else 'base'}_{capital}"
            print(f"START {name}", flush=True)
            mark = time.perf_counter()
            result = run_account(inputs, capital, enabled, name)
            if [row["trade_date"] for row in result["daily"]] != list(inputs.calendar):
                raise ValueError(f"{name}账户日期与冻结完整日历不同")
            if not result["validation"]["valid"] or not result["validation"]["initial_nav_is_one"]:
                raise ValueError(f"{name}账户不变量或初始现金点无效")
            timing[name] = time.perf_counter() - mark
            folder = output / name
            folder.mkdir()
            for key, values in result.items():
                write_json(folder / f"{key}.json", values)
            all_accounts[name] = result
            print(f"DONE {name} {timing[name]:.3f}s", flush=True)
            budget_check(started, cache_seconds)
        name = f"base_{capital}"
        print(f"START aa_{capital}", flush=True)
        mark = time.perf_counter()
        repeated = run_account(inputs, capital, False, name)
        timing[f"aa_{capital}"] = time.perf_counter() - mark
        aa[str(capital)] = mechanical_pair(all_accounts[name], repeated)
        write_json(output / f"aa_{capital}.json", aa[str(capital)])
        if not aa[str(capital)]["all_equal"]:
            raise ValueError(f"{capital}档机械A/A不一致")
        mark = time.perf_counter()
        paired[str(capital)] = compare_accounts(
            all_accounts[name]["daily"], all_accounts[f"buffer_{capital}"]["daily"]
        )
        timing[f"metrics_{capital}"] = time.perf_counter() - mark
        print(f"DONE aa_and_metrics_{capital}", flush=True)
        budget_check(started, cache_seconds)
    # 重新核验缓存/行情/模型身份，而非只比较内存中的身份字典。
    load_inputs(root, cache, spec_sha256=EXPECTED_SPEC)
    if capture_dependencies(root, cache, spec) != identities:
        raise ValueError("运行期间规格、方法、缓存或分红输入发生变化")
    timing["path_run_total_seconds"] = time.perf_counter() - started
    timing["cache_and_path_seconds"] = cache_seconds + timing["path_run_total_seconds"]
    result = {
        "schema": "p7_buffer_path_result/v1",
        "spec_sha256": EXPECTED_SPEC,
        "data_purchase_yuan": 0,
        "paired": paired,
        "mechanical_aa": aa,
        "account_validation": {k: v["validation"] for k, v in all_accounts.items()},
        "timing": timing,
        "formal_admission": "not_evaluated",
        "production_changed": False,
        "limitations": [
            "资金可行研究基准与旧生产买入候补/补齐路径不同，不能声称逐笔等价。",
            "历史信号和数据沿既有cs_train_verify行域，不是新PIT或前向验证。",
            "重抽样频率不是未来成功概率，也没有新的误报/功效标定。",
            "现金分红及税费沿冻结研究口径，滑点是模型预算。",
        ],
    }
    budget_check(started, cache_seconds)
    write_json(output / "summary.json", result)
    outputs = {
        p.relative_to(output).as_posix(): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    write_json(
        output / "manifest.json",
        {
            "spec_sha256": EXPECTED_SPEC,
            "git_head": git_head,
            "working_tree_dirty": True,
            "input_identity": "input_identity.json",
            "outputs": outputs,
            "timing": timing,
            "complete": True,
        },
    )
    return result


def main() -> None:
    """显式选择冻结规格、共享缓存和新运行根。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(ROOT, args.input_cache.resolve(), args.output.resolve(), args.spec.resolve())
    print(canonical({"complete": True, "timing": result["timing"]}), flush=True)


if __name__ == "__main__":
    main()
