# -*- coding: utf-8 -*-
"""模型列集审计报告：markdown 摘要与产物落盘。"""

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from .columns import SourceColumns


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 20) -> str:
    """把 DataFrame 渲染为 markdown 表格（仅取指定列，超出 limit 截断）。"""
    subset = frame.loc[:, [c for c in columns if c in frame.columns]].head(limit)
    if subset.empty:
        return "（无）"
    header = "| " + " | ".join(str(c) for c in subset.columns) + " |"
    divider = "| " + " | ".join("---" for _ in subset.columns) + " |"
    body = [
        "| " + " | ".join(_fmt_cell(value) for value in row) + " |"
        for row in subset.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body])


def _fmt_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "/").replace("\n", " ")
    if len(text) > 120:
        text = text[:117] + "..."
    return text


def build_markdown(
    tables: Dict[str, pd.DataFrame],
    items: Sequence[SourceColumns],
    last: int,
) -> str:
    """生成 markdown 审计报告。"""
    lines = [
        "# 模型列集审计",
        "",
        "## 一、覆盖范围",
        "",
    ]
    scope = pd.DataFrame(
        [
            {
                "来源": item.label,
                "版本": item.name,
                "列数": len(item.columns),
                "清单文件": item.path.name,
            }
            for item in items
        ]
    )
    by_label = (
        scope.groupby("来源")
        .agg(版本数=("版本", "size"), 列数最小=("列数", "min"), 列数最大=("列数", "max"))
        .reset_index()
    )
    lines.append(f"- 扫描深度：每来源末尾 {last if last and last > 0 else '全部'} 个版本")
    lines.append("")
    lines.append(_markdown_table(by_label, ["来源", "版本数", "列数最小", "列数最大"]))
    lines.append("")

    config = tables.get("列集配置分组.csv")
    if config is not None and not config.empty:
        lines.extend(
            [
                "## 二、列集配置分组",
                "",
                '> 同一模型目录会累积多套实验配置（开关/特征集不同），逐版本 diff 会把"换配置"\n'
                '> 误读成"掉列"。先按列集完全一致分组，再判断真正的漂移。',
                "",
            ]
        )
        lines.append(
            _markdown_table(config, ["来源", "配置ID", "列数", "版本数", "版本列表"], limit=30)
        )
        lines.append("")

    config_diff = tables.get("配置差异对比.csv")
    if config_diff is not None and not config_diff.empty:
        lines.extend(
            [
                "### 与最新配置的差异",
                "",
            ]
        )
        lines.append(
            _markdown_table(
                config_diff,
                [
                    "配置ID",
                    "列数",
                    "版本数",
                    "相对最新配置缺失列数",
                    "相对最新配置多出列数",
                    "缺失列",
                ],
                limit=30,
            )
        )
        lines.append("")

    ledger = tables.get("列集漂移台账.csv")
    if ledger is not None and not ledger.empty:
        changed = ledger[(ledger["相对上一版本新增数"] > 0) | (ledger["相对上一版本移除数"] > 0)]
        lines.extend(
            [
                "## 三、列集漂移（相对同来源上一版本）",
                "",
                f"- 版本总数：{len(ledger)}；发生变化的版本数：{len(changed)}",
                "",
            ]
        )
        lines.append(
            _markdown_table(
                changed,
                ["来源", "版本", "列数", "相对上一版本新增数", "相对上一版本移除数", "移除列"],
                limit=30,
            )
        )
        lines.append("")

    diff = tables.get("跨来源列集差异.csv")
    if diff is not None and not diff.empty:
        lines.extend(
            [
                "## 四、跨来源差异（各自最新版本）",
                "",
                "> 部署模型与训练折目录的列集若不一致，说明线上推理与训练矩阵不是同一套特征；"
                "差异列必须逐列归因（缺失来源数据 / 高缺失门禁 / 常数列过滤 / 开关未对齐）。",
                "",
            ]
        )
        lines.append(
            _markdown_table(
                diff,
                ["来源A", "版本A", "来源B", "版本B", "A独有列数", "B独有列数", "共有列数"],
            )
        )
        lines.append("")

    family_summary = tables.get("漂移家族汇总.csv")
    if family_summary is not None and not family_summary.empty:
        lines.extend(["## 五、漂移家族汇总", "", ""])
        lines.append(_markdown_table(family_summary, ["来源", "家族", "状态", "列数"], limit=40))
        lines.append("")

    frequency = tables.get("列集出现频次.csv")
    if frequency is not None and not frequency.empty:
        unstable = frequency[frequency["状态"] != "常驻"]
        lines.extend(
            [
                "## 六、非恒定列（状态 != 常驻）",
                "",
                f"- 特征总数：{len(frequency)}；非恒定列：{len(unstable)}",
                "",
            ]
        )
        lines.append(
            _markdown_table(
                unstable.sort_values(["来源", "出现版本数"], ascending=[True, True]),
                ["来源", "特征", "家族", "出现版本数", "版本总数", "状态"],
                limit=40,
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 七、判读提示",
            "",
            "- 「已移除」列需要确认是**有意裁剪**（开关关闭 / 因子排除清单 / 高缺失门禁）还是**数据链路掉列**；",
            "- 「间断出现」列通常来自逐折缺失率波动（同一份数据不同窗口的缺失率不同），属可解释漂移，但仍需登记；",
            "- 本审计只读特征清单 JSON，不写回任何模型或特征产物。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(out_dir: Path, tables: Dict[str, pd.DataFrame], markdown: str) -> Dict[str, Path]:
    """落盘全部产物，返回产物路径字典。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path, index=(name == "列集存在矩阵.csv"), encoding="utf-8-sig")
        written[name] = path
    report_path = out_dir / "模型列集审计.md"
    report_path.write_text(markdown, encoding="utf-8")
    written["模型列集审计.md"] = report_path
    return written
