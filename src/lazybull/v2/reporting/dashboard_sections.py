# -*- coding: utf-8 -*-
"""驾驶舱六区与附属面板（账户 / 持仓 / 指令 / pending / 调仓 / 模型·质量·新鲜度）。

硬口径（P6 T2）：

- 只**读** T1 投影，不做任何新指标、不补值、不重算；汇总值一律取原值字段；
- 防御读取：字段缺席显示 ``（字段缺席）``，中性空列表显示 ``（未读取）``；
- 六区覆盖 正常 / 空（``empty_valid``）/ 降级（``degraded``、``not_available``、``invalid``）
  三态，状态词与原因码逐区可见；
- 降级清单区是 manifest ``degradations[]`` 的页面视图；``research`` 三端卡只陈列宁既有
  核对结果，实盘端固定不可用。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import status, svg
from src.lazybull.v2.reporting.view_primitives import (
    MAX_DEGRADATION_ROWS,
    MAX_INSTRUCTION_ROWS,
    MAX_PENDING_ROWS,
    MAX_POSITION_ROWS,
    MAX_QUALITY_ROWS,
    MAX_TRADE_ROWS,
    MISSING_VALUE,
    NOT_READ_VALUE,
    STATUS_LABELS,
    bool_of,
    cap_note,
    combine_status,
    empty_note,
    fact_view,
    join_text,
    number_of,
    percent_of,
    row_of,
    rows_of,
    section_view,
    status_view,
    table_view,
    text_of,
)

__all__ = [
    "REBALANCE_FIELDS",
    "account_section",
    "all_sections",
    "degradations_section",
    "instructions_section",
    "model_quality_section",
    "pending_section",
    "positions_section",
    "project_three_ends",
    "rebalance_section",
]

#: 调仓状态五字段（顺序固定，标签即中文列名）
REBALANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("last_rebalance_date", "上次调仓日"),
    ("last_scheduled_rebalance_date", "上次计划调仓日"),
    ("tranche_anchor_date", "分批锚点日"),
    ("rebalance_freq", "调仓频率"),
    ("stagger_tranches", "分批档数"),
)


def _generation_text(authority: Mapping[str, Any]) -> str:
    """权威代文本（``run_kind@trade_date``，取自 authority 状态行的 ``extra`` 值）。

    ``record`` 区是**状态行**（``authority.classify_run_record`` 的产出），不含 ``run_kind``；
    权威代只能从 ``authority.extra``（``cockpit.load_authority`` 写入）取——从 record 取会
    恒得“字段缺席”（T2-R1-01）。
    """
    kind = str(authority.get("run_kind") or "")
    day = str(authority.get("trade_date") or "")
    return f"{kind}@{day}" if kind and day else MISSING_VALUE


def _record_shape_text(record: Mapping[str, Any]) -> str:
    """记录形态文本（状态词 / 原因码；缺席显式说明）。"""
    word = text_of(record.get("report_status"), missing=NOT_READ_VALUE)
    reason = text_of(record.get("degradation_reason_code"), missing="—")
    return f"{word} / {reason}"


def _nav_series(nav: Mapping[str, Any]) -> list[dict[str, Any]]:
    """NAV 核对行 → 图元点（值取 ``nav_values``；未读取时返回空表）。"""
    values = nav.get("nav_values")
    value_map = dict(values) if isinstance(values, Mapping) else {}
    series: list[dict[str, Any]] = []
    for row in rows_of(nav.get("rows")):
        day = str(row.get("trade_date") or "")
        detail = row_of(value_map.get(day))
        series.append(
            {
                "date": day or MISSING_VALUE,
                "value": detail.get("total_value"),
                "matched": bool(row.get("matched")),
            }
        )
    return series


def _nav_count(nav: Mapping[str, Any], key: str) -> str:
    """NAV 列表类计数（键缺席显式标未读取，不写 0）。"""
    if key not in nav:
        return NOT_READ_VALUE
    values = nav.get(key)
    return str(len(values)) if isinstance(values, (list, tuple)) else NOT_READ_VALUE


def _nav_value_facts(nav: Mapping[str, Any]) -> list[dict[str, Any]]:
    """NAV 三原值（当前 NAV 日的 total_value / position_value / cash，原值展示）。"""
    nav_date = str(nav.get("nav_date") or "")
    values = nav.get("nav_values")
    detail = row_of(dict(values).get(nav_date)) if isinstance(values, Mapping) and nav_date else {}
    if not detail:
        return [
            fact_view("总资产（NAV 原值）", MISSING_VALUE),
            fact_view("持仓市值（NAV 原值）", MISSING_VALUE),
            fact_view("现金（NAV 原值）", MISSING_VALUE),
        ]
    return [
        fact_view("总资产（NAV 原值）", number_of(detail.get("total_value"), digits=4)),
        fact_view("持仓市值（NAV 原值）", number_of(detail.get("position_value"), digits=4)),
        fact_view("现金（NAV 原值）", number_of(detail.get("cash"), digits=4)),
    ]


def _nav_facts(
    nav: Mapping[str, Any], *, row: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """NAV 区事实（状态/日/已核对行/断链/仅 NAV 日/年化可用 + 三原值）。"""
    broken = nav.get("broken_links")
    nav_only = nav.get("nav_only_days")
    return (
        [
            fact_view("NAV 块状态（同区独立块）", status_view(nav)["detail"], row=row or nav),
            fact_view("NAV 日期", text_of(nav.get("nav_date"))),
            fact_view("NAV 日期状态", text_of(nav.get("nav_date_status"))),
            fact_view("已核对行", _nav_count(nav, "verified_rows")),
            fact_view(
                "断链日",
                join_text(list(broken)) if isinstance(broken, (list, tuple)) else NOT_READ_VALUE,
            ),
            fact_view(
                "仅 NAV 业务日",
                join_text(list(nav_only)) if isinstance(nav_only, (list, tuple))
                else NOT_READ_VALUE,
            ),
            fact_view("年化可用", bool_of(nav.get("annualization_allowed"))),
        ]
        + _nav_value_facts(nav)
    )


def _account_facts(
    record: Mapping[str, Any],
    authority: Mapping[str, Any],
    account: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """账户区事实（权威代/账户日/现金/持仓数/记录形态；不做任何汇总推算）。"""
    positions = account.get("positions")
    count = len(positions) if isinstance(positions, Mapping) else None
    return [
        fact_view("权威代", _generation_text(authority)),
        fact_view("账户日期", text_of(account.get("last_update"))),
        fact_view("账户现金", number_of(account.get("cash"))),
        fact_view("持仓数", f"{count} 只" if count is not None else NOT_READ_VALUE),
        fact_view("记录形态", _record_shape_text(record), row=record),
    ]


def _trade_table(trades: Mapping[str, Any]) -> dict[str, Any]:
    """当前日成交表（11 列投影的展示子集；成本分解不参与身份判断）。"""
    rows = rows_of(trades.get("trades"))
    shown = rows[:MAX_TRADE_ROWS]
    body = [
        [
            text_of(row.get("trade_date")),
            text_of(row.get("ts_code")),
            text_of(row.get("action")),
            text_of(row.get("shares")),
            number_of(row.get("price"), digits=4),
            number_of(row.get("amount"), digits=2),
            text_of(row.get("reason")),
        ]
        for row in shown
    ]
    return table_view(
        "当前日成交",
        ("日期", "代码", "方向", "股数", "价格", "金额", "原因"),
        body,
        total=len(rows),
        note=cap_note(len(rows), len(shown), "当前日成交"),
        row=trades,
    )


def account_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """账户区：账户核对 + NAV 曲线/年化 + 当前日成交（三块各自独立状态行）。

    区级状态取三块**最严重**者（契约③）：成交核对 `invalid` 而账户仍 `available` 时，
    区级不得显示“可用”（T2-R1-M1）。
    """
    account_row = row_of(projection.get("account"))
    account = row_of(account_row.get("account"))
    nav = row_of(projection.get("nav"))
    trades = row_of(projection.get("trades"))
    record = row_of(projection.get("record"))
    authority = row_of(projection.get("authority"))
    annualization = row_of(row_of(projection.get("config")).get("annualization"))
    if not annualization:
        # 配置区冲突分支（如 account_start_date_conflict）不写 annualization 子键：
        # 显示“字段缺席”，不写“—”（T2-R1-08）
        annual_value = MISSING_VALUE
    elif annualization.get("report_status") == "available":
        annual_value = percent_of(annualization.get("annualized_return"))
    else:
        annual_value = status_view(annualization)["detail"]
    facts = _account_facts(record, authority, account) + _nav_facts(nav, row=nav)
    facts.append(fact_view("纸面年化（既有实现，百分数）", annual_value, row=annualization))
    charts = [
        {
            "caption": "已核对 NAV（总资产）",
            "svg": svg.nav_chart(_nav_series(nav)),
            "note": "未核对点不参与连线；点不足两个不绘图（不插值、不补 0）。",
        }
    ]
    combined = combine_status(
        [account_row, nav, trades],
        detail=(
            f"账户 {status_view(account_row)['label']} / NAV {status_view(nav)['label']}"
            f" / 成交 {status_view(trades)['label']}"
        ),
    )
    return section_view(
        "sec-account",
        "账户 / NAV / 成交（区级状态取各块最严重）",
        combined,
        facts=facts,
        tables=[_trade_table(trades)],
        charts=charts,
        note=(
            "账户逐字段核对与 NAV 三原值核对均来自 T1 投影；本页不做修复或补值。"
            "NAV 中性空列表表示未读取，不代表零成交。"
        ),
    )


def _position_lines(account: Mapping[str, Any]) -> tuple[list[list[str]], int, bool]:
    """持仓行（9 键账户投影；返回 ``(行, 真实行数, 是否提供了持仓表)``）。"""
    positions = account.get("positions")
    if not isinstance(positions, Mapping):
        return [], 0, False
    ordered = sorted(positions.values(), key=lambda item: str(row_of(item).get("ts_code") or ""))
    lines: list[list[str]] = []
    for item in ordered[:MAX_POSITION_ROWS]:
        position = row_of(item)
        lines.append(
            [
                text_of(position.get("ts_code")),
                text_of(position.get("shares")),
                number_of(position.get("buy_price"), digits=4),
                number_of(position.get("buy_cost"), digits=2),
                text_of(position.get("buy_date")),
                number_of(position.get("buy_pnl_price"), digits=4),
                text_of(position.get("status")),
            ]
        )
    return lines, len(ordered), True


def positions_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """持仓区：账户投影里的持仓（费用额口径，不含本金）。

    **可用性前置**：空持仓只有在账户行**可用**（``available``/``empty_valid``）时才是
    “合法空”；非可用态（降级/不可用/无效）下必须回落到该状态对应的空表文案，不能把
    “没有持仓明细”读成“已核实为零”（T2-R1-M4 / T3 前遗留 R6-2）。
    """
    account_row = row_of(projection.get("account"))
    account = row_of(account_row.get("account"))
    lines, total, provided = _position_lines(account)
    if provided:
        # “合法空”只在账户核对**可用**时成立：非可用态 + 空持仓时必须回落到不可用文案，
        # 否则同一张表会出现“合法空”与本表状态“不可用/降级”互相打脸（T2-R1-M4）
        empty_text = (
            "合法空：账户投影持仓为空"
            if status.is_usable(account_row)
            else empty_note(account_row)
        )
        note = cap_note(total, len(lines), "持仓") or (empty_text if total == 0 else "")
    else:
        note = "账户区未提供持仓明细（不伪造空表）。"
    return section_view(
        "sec-positions",
        "持仓（区级状态 = 账户核对状态，唯一来源）",
        account_row,
        tables=[
            table_view(
                "当前持仓",
                ("代码", "股数", "买价", "费用额", "买日", "基准价", "状态"),
                lines,
                total=total,
                note=note,
                row=account_row,
            )
        ],
        note=(
            "持仓来自权威账户投影（本区无独立门禁，状态取账户逐字段核对结果）；"
            "“费用额”为 commission+slippage，不是本金。"
        ),
    )


def _instruction_table(instructions: Mapping[str, Any]) -> dict[str, Any]:
    """未来指令表（按 ``ts_code`` 稳定排序；计数与文件核对由 T1 完成）。

    必须带 ``row=`` 状态行：否则空表文案会指向不存在的“本表状态”，且
    ``empty_valid`` / ``not_available`` 在该表上无法区分（T2-R1-M2）。
    """
    rows = sorted(
        rows_of(instructions.get("instructions")), key=lambda row: str(row.get("ts_code"))
    )
    shown = rows[:MAX_INSTRUCTION_ROWS]
    body = [
        [
            text_of(row.get("ts_code")),
            text_of(row.get("action")),
            text_of(row.get("shares")),
            text_of(row.get("price_type")),
            number_of(row.get("target_weight"), digits=6),
            text_of(row.get("source_date")),
            text_of(row.get("reason")),
        ]
        for row in shown
    ]
    return table_view(
        "未来待执行指令",
        ("代码", "方向", "股数", "价格类型", "目标权重", "信号日", "原因"),
        body,
        total=len(rows),
        note=cap_note(len(rows), len(shown), "指令"),
        row=instructions,
    )


def instructions_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """指令区：只展示 ``E > D`` 且 ``source_date == D`` 的未来指令文件。"""
    row = row_of(projection.get("instructions"))
    future = row.get("future_files")
    facts = [
        fact_view("执行日", text_of(row.get("execution_date"))),
        fact_view(
            "未来文件",
            join_text(list(future)) if isinstance(future, (list, tuple)) else NOT_READ_VALUE,
        ),
        fact_view("历史文件数", text_of(row.get("history_files"))),
        fact_view("指令行数（计数核对后）", text_of(row.get("instruction_count"))),
    ]
    return section_view(
        "sec-instructions",
        "指令（正常 / 合法空 / 降级按能力展示）",
        row,
        facts=facts,
        tables=[_instruction_table(row)],
        note=(
            "v2 T1 代可生成下一交易日指令；切换前记录形态固定不可用，"
            "不以当日已执行指令推断未来。"
        ),
    )


def _pending_table(caption: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """pending 队列表（原值；不推断执行结果）。"""
    rows = rows_of(row.get("pending"))
    shown = rows[:MAX_PENDING_ROWS]
    body = [
        [
            text_of(item.get("ts_code")),
            text_of(item.get("action")),
            text_of(item.get("shares")),
            number_of(item.get("price"), digits=4),
            text_of(item.get("reason")),
        ]
        for item in shown
    ]
    return table_view(
        caption,
        ("代码", "方向", "股数", "价格", "原因"),
        body,
        total=len(rows),
        note=cap_note(len(rows), len(shown), caption),
        row=row,
    )


def pending_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """pending 区：买卖两路独立状态（合法空与缺文件严格区分）。"""
    pending = row_of(projection.get("pending"))
    buys = row_of(pending.get("buys"))
    sells = row_of(pending.get("sells"))
    combined = combine_status(
        [buys, sells],
        detail=f"买入队列 {status_view(buys)['label']} / 卖出队列 {status_view(sells)['label']}",
    )
    return section_view(
        "sec-pending",
        "pending 队列",
        combined,
        facts=[
            fact_view("买入队列长度", text_of(buys.get("pending_count"))),
            fact_view("卖出队列长度", text_of(sells.get("pending_count"))),
        ],
        tables=[_pending_table("待执行买入", buys), _pending_table("待执行卖出", sells)],
        note="恢复代缺计数不算 0；合法空只在文件为 list 且计数为 0 时成立。",
    )


def rebalance_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """调仓区：五字段原值（不重算，不以配置推断调仓进度）。"""
    row = row_of(projection.get("rebalance"))
    state = row_of(row.get("rebalance"))
    facts = [
        fact_view(label, text_of(state.get(key)) if state else NOT_READ_VALUE)
        for key, label in REBALANCE_FIELDS
    ]
    return section_view(
        "sec-rebalance",
        "调仓状态",
        row,
        facts=facts,
        note="五字段原值展示；缺失显示“字段缺席”，不以配置推断调仓进度。",
    )


def _model_facts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """模型区事实（A/B 版本与权重 + 实跑模型固定降级）。"""
    model = row_of(config.get("model"))
    actual = row_of(model.get("actual_model"))
    return [
        fact_view("模型 A", text_of(model.get("model_version")), row=model),
        fact_view("模型 B", text_of(model.get("model_version_b")), row=model),
        fact_view("权重 A", text_of(model.get("ensemble_weight_a")), row=model),
        fact_view("权重 B", text_of(model.get("ensemble_weight_b")), row=model),
        fact_view("跟随最新", bool_of(model.get("follows_latest")), row=model),
        fact_view("实跑模型", status_view(actual)["detail"], row=actual),
        fact_view("账户起始日", text_of(config.get("account_start_date")), row=config),
    ]


def _quality_status_cell(row: Mapping[str, Any], value_key: str, status_key: str) -> str:
    """质量单元格：无值时显示该列的**解析状态与原因码**（不写空串、不冒充缺值）。

    T1 把 ``partition_count``/``latest_partition``/``sync_watermark`` 的解析结果另存为
    ``*_status`` 嵌套状态；只读原值会把 ``invalid/multiple_quality_value`` 这类冲突降级成
    “字段缺席”（T2-R3-05）。
    """
    value = row.get(value_key)
    if value is not None:
        return str(value)
    status_row = row_of(row.get(status_key))
    word = str(status_row.get("report_status") or "")
    if not word or word == "available":
        return MISSING_VALUE
    reason = str(status_row.get("degradation_reason_code") or "")
    label = STATUS_LABELS.get(word, word)
    return f"（{label}：{reason}）" if reason else f"（{label}：{MISSING_VALUE}）"


def _quality_count_cell(row: Mapping[str, Any]) -> str:
    """``partition_count`` 单元格（无值 → 该组解析状态 + 原因码，不写 0）。"""
    return _quality_status_cell(row, "partition_count", "partition_count_status")


def _quality_table(quality: Mapping[str, Any]) -> dict[str, Any]:
    """质量数据集摘要表（按 layer/dataset 分组；错误/警告按指标行计数）。"""
    datasets = sorted(
        rows_of(quality.get("datasets")),
        key=lambda row: (str(row.get("layer")), str(row.get("dataset"))),
    )
    shown = datasets[:MAX_QUALITY_ROWS]
    body = [
        [
            text_of(row.get("layer")),
            text_of(row.get("dataset")),
            text_of(row.get("metric_count")),
            text_of(row.get("error_count")),
            text_of(row.get("warning_count")),
            _quality_count_cell(row),
            _quality_status_cell(row, "latest_partition", "latest_partition_status"),
            _quality_status_cell(row, "sync_watermark", "sync_watermark_status"),
        ]
        for row in shown
    ]
    return table_view(
        "质量数据集摘要",
        ("层", "数据集", "指标行", "错误行", "警告行", "分区数", "最新分区", "同步水位"),
        body,
        total=len(datasets),
        note=cap_note(len(datasets), len(shown), "数据集")
        or ("无质量数据集分组（来源缺失或未扫描；不伪造分布）" if not datasets else ""),
        row=quality,
    )


def _freshness_value(
    datasets_: Sequence[Mapping[str, Any]], value_key: str, status_key: str
) -> str:
    """新鲜度取值：值集合 + **并行保留**非可用状态（不因有一个正常值就吞掉别人的冲突）。

    只取“第一个正常值”会隐藏其他数据集的 ``invalid/multiple_quality_value``，而异常数据集
    完全可能落在表格可见上限之外（T2-R3-05-R1）。
    """
    values = sorted({str(row.get(value_key)) for row in datasets_ if row.get(value_key)})
    statuses: set[str] = set()
    for row in datasets_:
        status_row = row_of(row.get(status_key))
        word = str(status_row.get("report_status") or "")
        if not word or word == "available":
            continue
        reason = str(status_row.get("degradation_reason_code") or "")
        label = STATUS_LABELS.get(word, word)
        statuses.add(f"（{label}：{reason}）" if reason else f"（{label}：{MISSING_VALUE}）")
    summary = "；".join(sorted(statuses))
    if values:
        return f"{join_text(values)}；另有来源状态：{summary}" if summary else join_text(values)
    return summary or MISSING_VALUE


def _freshness_facts(
    projection: Mapping[str, Any], quality: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """新鲜度分栏：账户日 / NAV 日 / 质量最新分区与同步水位（不按当前日期外推）。"""
    authority = row_of(projection.get("authority"))
    nav = row_of(projection.get("nav"))
    datasets = rows_of(quality.get("datasets"))
    unavailable = not datasets
    return [
        fact_view("账户日", text_of(authority.get("trade_date"))),
        fact_view("NAV 日", text_of(nav.get("nav_date"))),
        fact_view(
            "质量最新分区",
            NOT_READ_VALUE
            if unavailable
            else _freshness_value(datasets, "latest_partition", "latest_partition_status"),
        ),
        fact_view(
            "质量同步水位",
            NOT_READ_VALUE
            if unavailable
            else _freshness_value(datasets, "sync_watermark", "sync_watermark_status"),
        ),
        fact_view("质量扫描时间", "暂无来源（不读取 mtime）", row=row_of(quality.get("scan_time"))),
    ]


def model_quality_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """模型·质量·新鲜度区（三块共用一区，各自独立状态行）。"""
    config = row_of(projection.get("config"))
    quality = row_of(projection.get("quality"))
    facts = (
        _model_facts(config)
        + [
            fact_view("质量区状态", status_view(quality)["detail"], row=quality),
            fact_view("质量指标行", text_of(quality.get("metric_count")), row=quality),
            fact_view("错误行", text_of(quality.get("error_count")), row=quality),
            fact_view("警告行", text_of(quality.get("warning_count")), row=quality),
        ]
        + _freshness_facts(projection, quality)
    )
    model_row = row_of(config.get("model"))
    combined = combine_status(
        [config, model_row, quality],
        detail=(
            f"配置 {status_view(config)['label']} / 模型 {status_view(model_row)['label']}"
            f" / 质量 {status_view(quality)['label']}"
        ),
    )
    return section_view(
        "sec-model-quality",
        "模型 · 质量 · 新鲜度（区级状态取各块最严重）",
        combined,
        facts=facts,
        tables=[_quality_table(quality)],
        charts=[
            {
                "caption": "质量指标行分布（错误/警告）",
                "svg": svg.quality_bars(rows_of(quality.get("datasets"))),
                "note": "仅画已有分组计数；无分组不绘制（不伪造分布）。",
            }
        ],
        note=(
            "实跑模型在当前 v2 代无持久来源，固定不可用；质量扫描时间/窗口/变化无来源，"
            "一律显示暂无来源，不用 mtime 冒充。"
        ),
    )


def degradations_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """降级清单区（manifest ``degradations[]`` 的页面视图，逐条带原因码）。"""
    rows = sorted(
        rows_of(projection.get("degradations")), key=lambda row: str(row.get("field_path"))
    )
    shown = rows[:MAX_DEGRADATION_ROWS]
    body = [
        [
            text_of(row.get("field_path")),
            text_of(row.get("report_status")),
            text_of(row.get("reason_code")),
            text_of(row.get("detail")),
        ]
        for row in shown
    ]
    empty = not rows
    # 区级状态词取**登记项中最严重者**（条目含 invalid 时不得只写 degraded，T2-R1-N2），
    # 原因码取同级的首条真实码（按 field_path 稳定排序），不借用 ``no_source``
    row = (
        status.make_status("available", detail="无降级登记")
        if empty
        else combine_status(
            [
                status.make_status(
                    str(entry.get("report_status") or "degraded"),
                    reason_code=(
                        str(entry.get("reason_code"))
                        if str(entry.get("reason_code") or "") in status.REASON_CODES
                        else "no_source"
                    ),
                    detail=str(entry.get("detail") or ""),
                )
                for entry in rows
            ],
            detail=f"{len(rows)} 条降级登记（逐条见下表）",
        )
    )
    return section_view(
        "sec-degradations",
        "降级清单",
        row,
        tables=[
            table_view(
                "降级登记",
                ("字段路径", "状态", "原因码", "明细"),
                body,
                total=len(rows),
                # “无降级登记”只留给**空表**分支：有登记时不得再说“无降级登记”（T2-R1-N7）
                note=cap_note(len(rows), len(shown), "降级项")
                or ("无降级登记" if empty else ""),
                row=row,
            )
        ],
        note="降级必须带原因码；不可用态不伪造明细与计数。",
    )


def _end_row(value: Any, *, detail: str) -> dict[str, Any]:
    """三端之一的状态行：缺省 ⇒ 固定不可用；显式给出则须是闭集状态词（fail-closed）。"""
    if value is None:
        return status.make_status("not_available", reason_code="no_source", detail=detail)
    material = row_of(value)
    word = str(material.get("report_status") or "")
    if word not in status.STATUS_WORDS:
        raise ValueError(f"三端卡状态词不在闭集内：{word!r}（输入须显式声明状态）")
    return material


def _identity_text(value: Any) -> str:
    """三端身份摘要文本（Mapping → ``k=v`` 排序拼接；标量原样；缺省写“—”）。"""
    if isinstance(value, Mapping):
        items = [f"{key}={item}" for key, item in sorted(dict(value).items())]
        return " / ".join(items) if items else "—"
    return text_of(value, missing="—")


def project_three_ends(projection: Mapping[str, Any]) -> dict[str, Any]:
    """research 三端身份卡（回测 / 纸面回放 / 纸面 + 实盘固定不可用）。

    本阶段**不接受**调用方注入三端身份（原 ``ends`` 参数已删除，T2-R3-03）：额外输入不进入
    canonical projection，封存包内无法复算；三端一律取自 projection（回测/回放端在 T3/T4
    接入真实输入前固定 ``not_available``）。
    """
    authority = row_of(projection.get("authority"))
    nav = row_of(projection.get("nav"))
    record = row_of(projection.get("record"))
    paper = dict(authority)
    paper["identity"] = {
        "run_kind": text_of(authority.get("run_kind")),
        "trade_date": text_of(authority.get("trade_date")),
        "record": _record_shape_text(record),
        "nav_verified_rows": _nav_count(nav, "verified_rows"),
    }
    rows = (
        (
            "回测端",
            "runs（规范化 / 旧 WF）",
            _end_row(None, detail="本面未接入回测 runs 输入（实验报告归 T3），不陈列推断值"),
        ),
        (
            "纸面回放端",
            "runs（paper_replay）",
            _end_row(
                None, detail="本面未接入纸面回放输入（归 research 实验报告），不陈列推断值"
            ),
        ),
        ("纸面端", "P4 纸面生产根（只读副本）", paper),
        (
            "实盘端",
            "P8（未建）",
            status.make_status(
                "not_available",
                reason_code="no_source",
                detail="P8 实盘端未建：不得以纸面/回放/回测读数冒充实盘",
            ),
        ),
    )
    return {
        "visible": True,
        "title": "三端身份卡（research）",
        "rows": [
            {
                "end": name,
                "source": source,
                "status": status_view(row),
                "identity": _identity_text(row.get("identity")),
            }
            for name, source, row in rows
        ],
        "note": "三端卡只陈列既有核对结果；不出现实盘读数、不产生裁决、不升格为生产摘要。",
    }


def all_sections(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """六区 + 降级清单区的固定顺序（模板按此顺序渲染，不依赖字典序）。"""
    return [
        account_section(projection),
        positions_section(projection),
        instructions_section(projection),
        pending_section(projection),
        rebalance_section(projection),
        model_quality_section(projection),
        degradations_section(projection),
    ]
