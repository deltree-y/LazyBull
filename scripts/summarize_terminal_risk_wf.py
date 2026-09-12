# -*- coding: utf-8 -*-
r"""期末异常亏损风险模型滚动 WF 汇总工具

扫描 batch_terminal_risk_wf.ps1 产出的各折目录（terminal_loss_report.json），
拼接 summary CSV，按超参消融后缀分组计算调参分，并与调参台账历史比较，
打印跨折门禁结论与历史最优超参。

核心指标（研究与选型层，详见方案 8.1）：
- lift = ES PR-AUC / ES 事件率：相对 null 基线的排序信息量（风险模型版的 RankIC）
- pred_bias = ES mean_pred - ES 事件率：校准漂移（C 段校准前的已知偏差方向，
  仅展示不进分数）
- 调参分 tuning_score = 0.5×组内lift几何均值 + 0.5×组内lift最小值：绝对量纲、
  跨 batch 可比的单一调参指标——几何均值代表平均排序能力，最小值代表最差折
  稳健性（与门禁阈值直接挂钩），用于 depth/lr 消融的组间选择与纵向对比。
- 分组：折目录名尾部 _d{depth} / _lr{lr} 消融后缀（batch ps1 数组多值时追加，
  可叠加）× 折 meta 超参签名——后缀只是目录展示名，超参身份以折 sidecar
  （terminal_loss_model.json 的 train_config/label_config/sampling）为权威；
  同后缀不同超参（如目录残留旧消融折）自动拆行，禁止混比。
- 历史最优：tuning_history.csv 台账（含当次）按超参签名聚合，score_best
  （历史最高单次调参分）为"效果最好"判定口径，score_median 供同签名多次
  运行参考；--no-history 时仅内存拼当次比较、不落盘。
- 跨折门禁：组内 lift 最小值 >= 阈值（默认 1.1）且跨折稳定，才建议进入第二阶段

产物：
- summary.csv：每折一行（当次覆盖）
- tuning_scores.csv：每超参组一行（当次覆盖）
- tuning_history.csv：每次汇总按组追加一行（跨 batch 调参台账；--no-history 跳过）

用法：
py .\scripts\summarize_terminal_risk_wf.py --wf-root data\walk_forward\terminal_risk_wf
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUMMARY_COLUMNS = [
    "fold",
    "train_start",
    "train_end",
    "es_start",
    "es_end",
    "n_train",
    "n_es",
    "train_event_rate",
    "es_event_rate",
    "best_iteration",
    "es_logloss",
    "es_brier",
    "es_pr_auc",
    "lift",
    "pred_bias",
    "depth",
    "learning_rate",
    "param_signature",
]

# 折目录名尾部消融后缀：_d{depth} / _lr{lr} / _w{years}y / _v{months}m /
# _em{metric} / _s{seed} 可叠加，均可省略（锚定结尾）。后缀仅供展示分组，
# 超参身份以折 sidecar 签名为权威。
_SUFFIX_PATTERN = re.compile(
    r"(?:_d(?P<depth>\d+))?"
    r"(?:_lr(?P<lr>\d+(?:\.\d+)?))?"
    r"(?:_w(?P<window>\d+)y)?"
    r"(?:_v(?P<valm>\d+)m)?"
    r"(?:_em(?P<emetric>[a-z_]+))?"
    r"(?:_s(?P<seed>\d+))?$"
)

# baseline 组展示名（分组键为空串，展示与键分离避免与真实后缀冲突）
BASELINE_LABEL = "(baseline)"

# 折 meta 缺 sidecar 配置段时的签名展示名（历史比较中独立成组，不与真签名合并）
UNREGISTERED_SIGNATURE = "(未登记签名)"

TUNING_COLUMNS = [
    "rank",
    "suffix",
    "param_signature",
    "depth",
    "learning_rate",
    "n_folds",
    "n_folds_valid_lift",
    "tuning_score",
    "lift_geo_mean",
    "lift_min",
    "lift_std",
    "pred_bias_mean",
    "gate_pass",
]

# 台账列 = 运行上下文 + 调参表（rank 依赖当次批内排序，纵向无意义，不入台账）
HISTORY_RUN_COLUMNS = ["timestamp", "wf_root", "lift_min_threshold"] + [
    col for col in TUNING_COLUMNS if col != "rank"
]

# ── 超参签名（历史比较的身份键；折 sidecar 为权威，目录后缀仅作展示）──────
# train_config：影响训练动态的消融位。random_state 是**多种子消融维度**，
# 必须入签名——否则多随机种子的折会被并进同一组，门禁会取跨种子最小值，
# 既污染分组语义又使门禁虚高（v0.108.5 起）。device 不入签名。
_SIGNATURE_TRAIN_KEYS = [
    "max_depth",
    "learning_rate",
    "n_estimators",
    "early_stopping_rounds",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "random_state",
    "eval_metric",
]
# label_config：改 k 即改任务，必须入签名
_SIGNATURE_LABEL_KEYS = ["loss_sigma_multiple", "h_max", "sigma_window"]
# sampling：预登记抽样参数，变更必须登记，入签名
_SIGNATURE_SAMPLING_KEYS = ["h_per_group", "every_n_days"]
_SIGNATURE_KEY_ALIASES = {
    "max_depth": "d",
    "learning_rate": "lr",
    "n_estimators": "nest",
    "early_stopping_rounds": "esr",
    "subsample": "sub",
    "colsample_bytree": "col",
    "reg_lambda": "lam",
    "random_state": "s",
    "eval_metric": "em",
    "loss_sigma_multiple": "k",
    "h_max": "hmax",
    "sigma_window": "sigw",
    "h_per_group": "hpg",
    "every_n_days": "end",
}

# 历史比较表列（台账 + 当次记录按超参签名聚合）
HISTORY_TABLE_COLUMNS = [
    "history_rank",
    "param_signature",
    "suffix",
    "depth",
    "learning_rate",
    "runs",
    "score_best",
    "score_median",
    "score_latest",
    "lift_min_best",
    "gate_pass_rate",
    "best_timestamp",
    "is_current_best",
]


def parse_experiment_suffix(fold_name: str) -> dict:
    """解析折目录名尾部的超参消融后缀。

    batch_terminal_risk_wf.ps1 在消融数组多值时给折目录追加 _d{depth} /
    _lr{lr} / _w{years}y / _em{metric} 后缀，单值时无后缀；早停段（Val）自
    v0.109.0 起恒追加 _v{months}m（标识"早停段与评估段分离"的新协议，旧产物
    无此后缀）。无法识别的尾缀按无后缀处理（同后缀目录仍会聚到同组），保证
    新后缀类型不中断汇总。注意：后缀仅是展示分组名，超参身份以折 sidecar
    签名为权威（后缀与 meta 可能不一致，如目录残留旧消融折未清理）。
    """
    match = _SUFFIX_PATTERN.search(fold_name)
    if match is None or not any(
        match.group(k) for k in ("depth", "lr", "window", "valm", "emetric", "seed")
    ):
        return {
            "suffix": "",
            "depth": None,
            "learning_rate": None,
            "train_window_years": None,
            "val_months": None,
            "eval_metric": None,
            "seed": None,
        }
    parts = []
    if match.group("depth"):
        parts.append(f"_d{match.group('depth')}")
    if match.group("lr"):
        parts.append(f"_lr{match.group('lr')}")
    if match.group("window"):
        parts.append(f"_w{match.group('window')}y")
    if match.group("valm"):
        parts.append(f"_v{match.group('valm')}m")
    if match.group("emetric"):
        parts.append(f"_em{match.group('emetric')}")
    if match.group("seed"):
        parts.append(f"_s{match.group('seed')}")
    return {
        "suffix": "".join(parts),
        "depth": int(match.group("depth")) if match.group("depth") else None,
        "learning_rate": float(match.group("lr")) if match.group("lr") else None,
        "train_window_years": (int(match.group("window")) if match.group("window") else None),
        "val_months": int(match.group("valm")) if match.group("valm") else None,
        "eval_metric": match.group("emetric") or None,
        "seed": int(match.group("seed")) if match.group("seed") else None,
    }


def _normalize_meta(meta: dict) -> dict:
    """统一两种元数据形状：折 sidecar 与注册表 metadata。

    - 折 sidecar（固定名别名 terminal_loss_model.json）：顶层 train_config /
      label_config，sampling 在 metadata 内；
    - 注册表 metadata（v{N}_metadata.json）：配置全在 train_params 内。

    归一后统一从 train_config / label_config / metadata 读取，避免两套键名
    在两处代码里分叉。
    """
    if "train_params" in meta:
        params = meta.get("train_params") or {}
        return {
            "train_config": params.get("train_config") or {},
            "label_config": params.get("label_config") or {},
            "metadata": params,
        }
    return meta


def _signature_window_years(meta: dict) -> Optional[int]:
    """训练窗口年数（由 ``stage_dates.train`` 起止日计算，跨折不变量）。

    训练窗口长度是**消融维度**（例 3 年 vs 5 年），必须入签名，否则不同窗口
    的折会被并进同一组。起止日缺失或非法时返回 None（调用方使整条签名返回
    None，独立成组，不与真签名合并）。
    """
    dates = ((meta.get("metadata") or {}).get("stage_dates") or {}).get("train")
    if not dates or len(dates) != 2 or not all(dates):
        return None
    start = pd.to_datetime(str(dates[0]), format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(str(dates[1]), format="%Y%m%d", errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return None
    return int(round(((end - start).days + 1) / 365.25))


def _signature_val_months(meta: dict) -> Optional[int]:
    """早停段（Val）月数（由 ``stage_dates.val`` 起止日计算，跨折不变量）。

    早停段长度决定门禁指标是否为"未参与早停的干净评估"（v0.109.0 协议），
    必须入签名。缺 ``stage_dates.val``（v0.109.0 之前的旧产物：早停即评估）
    返回 0，作为旧协议显式标签，绝不允许与新协议批次并组比较；起止日存在
    但非法时返回 None（调用方使整条签名返回 None，独立成组）。
    """
    dates = ((meta.get("metadata") or {}).get("stage_dates") or {}).get("val")
    if not dates:
        return 0
    if len(dates) != 2 or not all(dates):
        return None
    start = pd.to_datetime(str(dates[0]), format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(str(dates[1]), format="%Y%m%d", errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return None
    return int(round(((end - start).days + 1) / 30.44))


def build_param_signature(meta: dict) -> Optional[str]:
    """从折 sidecar 元数据构建超参签名（历史比较的身份键）。

    签名 = train_config 消融位（含 random_state 多种子维度）+ label_config
    任务定义 + sampling 预登记抽样 + 训练窗口年数 + 早停段月数，如
    "d=3|lr=0.03|nest=500|esr=30|sub=0.8|col=0.8|lam=1.0|s=42
    |k=1.0|hmax=20|sigw=20|hpg=2|end=3|wy=3|valm=6"。任一配置段缺键、或训练
    起止日无法解析时返回 None（调用方回退展示名，独立成组不与真签名合并）；
    device 与策略 A 不变量（min_child_weight/scale_pos_weight）不入签名。
    ``valm=0`` 表示 v0.109.0 之前"早停即评估"的旧协议产物。
    """
    sections = (
        (meta.get("train_config") or {}, _SIGNATURE_TRAIN_KEYS),
        (meta.get("label_config") or {}, _SIGNATURE_LABEL_KEYS),
        ((meta.get("metadata") or {}).get("sampling") or {}, _SIGNATURE_SAMPLING_KEYS),
    )
    parts: List[str] = []
    for cfg, keys in sections:
        for key in keys:
            if key not in cfg:
                return None
            parts.append(f"{_SIGNATURE_KEY_ALIASES.get(key, key)}={cfg[key]}")
    window_years = _signature_window_years(meta)
    if window_years is None:
        return None
    parts.append(f"wy={window_years}")
    val_months = _signature_val_months(meta)
    if val_months is None:
        return None
    parts.append(f"valm={val_months}")
    return "|".join(parts)


def _fmt(value: object, spec: str = ".3f") -> str:
    """数值格式化（None/NaN 统一显示 NA，用于日志与醒目打印）。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return format(value, spec)


# ── 控制台彩色输出（优胜者高亮）────────────────────────────────────────
# Windows 传统 conhost 默认不解析 ANSI 转义码，os.system("") 借 cmd 短暂运行
# 激活当前控制台 VT 模式（Win11 默认 Windows Terminal 原生支持，调用无副作用）；
# 重定向/管道（非 tty）时不加色码，保证落盘与管道文本干净。
_COLOR_ENABLED = sys.stdout.isatty()

_BOLD_CYAN = "1;36"  # 标题与分隔线（信息性）
_BOLD_GREEN = "1;92"  # 优胜者（历史最优签名/分数/表格首行）
_BOLD_YELLOW = "1;93"  # 刷新提示
_YELLOW = "93"  # 未达提示（弱于刷新）


def _enable_vt() -> None:
    """Windows 控制台启用 ANSI 转义处理（非 Windows 平台无操作）。"""
    if os.name == "nt":
        os.system("")


def _c(text: str, code: str) -> str:
    """按需包裹 ANSI 色码（非终端输出时保持纯文本）。"""
    if not _COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def _highlight_first_row(df: pd.DataFrame, code: str) -> str:
    """to_string 输出中高亮首个数据行（表已按分数降序，首行即优胜行）。"""
    rendered = df.to_string(index=False)
    lines = rendered.splitlines()
    if len(lines) < 2:
        return rendered
    return "\n".join([lines[0], _c(lines[1], code), *lines[2:]])


def _resolve_fold_artifacts(fold_dir: Path) -> tuple:
    """定位折的报告与 sidecar：固定名别名优先，否则回退最高版本化产物。

    训练脚本自 v0.108.0 起始终写注册制版本化产物（``v{N}_report.json`` /
    ``v{N}_metadata.json``），``--fixed-name`` 只是附加别名。别名缺失时
    必须能读版本化文件，否则仅注册过的折会被静默跳过。

    Returns:
        (report_path, meta_path)：不存在时对应项为 None
    """
    fixed_report = fold_dir / "terminal_loss_report.json"
    fixed_meta = fold_dir / "terminal_loss_model.json"
    if fixed_report.exists():
        return fixed_report, fixed_meta if fixed_meta.exists() else None
    versioned = sorted(
        fold_dir.glob("v*_report.json"),
        key=lambda p: int(p.name.split("_")[0].lstrip("v")),
    )
    if not versioned:
        return None, None
    latest = versioned[-1]
    version_str = latest.name.split("_")[0]
    meta_candidate = fold_dir / f"{version_str}_metadata.json"
    return latest, meta_candidate if meta_candidate.exists() else None


def collect_fold_rows(wf_root: Path) -> pd.DataFrame:
    """读取各折目录的 report 与模型元数据，拼 summary 行。

    depth/learning_rate/param_signature 优先取折 sidecar（权威），sidecar
    缺失时回退目录后缀解析值。
    """
    rows = []
    for fold_dir in sorted(p for p in wf_root.iterdir() if p.is_dir()):
        report_path, meta_path = _resolve_fold_artifacts(fold_dir)
        if report_path is None:
            logger.warning(f"跳过 {fold_dir.name}: 缺 terminal_loss_report.json / v*_report.json")
            continue
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        es = report["es"]
        stage_dates = {}
        best_iteration = None
        n_train = es_n = None
        train_cfg = {}
        param_signature = None
        if meta_path is not None:
            with open(meta_path, encoding="utf-8") as f:
                meta = _normalize_meta(json.load(f))
            metadata = meta.get("metadata", {})
            stage_dates = metadata.get("stage_dates", {})
            best_iteration = metadata.get("best_iteration")
            n_train = metadata.get("n_train")
            es_n = metadata.get("n_es")
            train_cfg = meta.get("train_config") or {}
            param_signature = build_param_signature(meta)
        suffix_info = parse_experiment_suffix(fold_dir.name)
        event_rate = es["event_rate"]
        rows.append(
            {
                "fold": fold_dir.name,
                "train_start": stage_dates.get("train", [None, None])[0],
                "train_end": stage_dates.get("train", [None, None])[1],
                "es_start": stage_dates.get("es", [None, None])[0],
                "es_end": stage_dates.get("es", [None, None])[1],
                "n_train": n_train,
                "n_es": es_n if es_n is not None else es.get("n"),
                "train_event_rate": report.get("train", {}).get("event_rate"),
                "es_event_rate": event_rate,
                "best_iteration": best_iteration,
                "es_logloss": es["logloss"],
                "es_brier": es["brier"],
                "es_pr_auc": es["pr_auc"],
                "lift": es["pr_auc"] / event_rate if event_rate > 0 else None,
                "pred_bias": es.get("mean_pred", float("nan")) - event_rate,
                "depth": train_cfg.get("max_depth", suffix_info["depth"]),
                "learning_rate": train_cfg.get("learning_rate", suffix_info["learning_rate"]),
                "param_signature": param_signature,
            }
        )
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def build_tuning_table(summary: pd.DataFrame, lift_min_threshold: float) -> pd.DataFrame:
    """按消融后缀 × 超参签名分组聚合调参表（每超参组一行），按调参分降序。

    后缀只是目录展示名；超参身份以 param_signature 为准——同后缀不同签名
    （目录残留旧消融折等）自动拆行并提示，禁止混比。lift 缺失或非正（如
    ES 事件率 0）的折不进入聚合，差额 = n_folds - n_folds_valid_lift；整组
    无有效 lift 时该组分数为空并排最后。
    """
    if summary.empty:
        return pd.DataFrame(columns=TUNING_COLUMNS)

    parsed = pd.DataFrame(
        summary["fold"].map(parse_experiment_suffix).tolist(), index=summary.index
    )
    # 仅取 suffix 列做展示分组键，避免与 summary 的 meta 权威超参列重名冲突
    # （concat 重名列会令取列返回 DataFrame）；窄 summary（测试/外部调用）缺
    # 超参列时回填后缀解析值，meta 权威列的空值同样以后缀解析值兜底；
    # param_signature 无后缀解析来源，缺列时按未登记处理
    work = pd.concat([parsed[["suffix"]], summary], axis=1)
    for col in ("depth", "learning_rate"):
        if col in summary.columns:
            work[col] = work[col].fillna(parsed[col])
        else:
            work[col] = parsed[col].to_numpy()
    if "param_signature" not in work.columns:
        work["param_signature"] = None
    work["signature"] = (
        work["param_signature"].astype(object).where(work["param_signature"].notna(), "")
    )

    # 同后缀出现多个签名 → 提示拆行（旧消融折残留或后缀与 meta 不一致）
    for suffix, sig_count in work.groupby("suffix")["signature"].nunique().items():
        if sig_count > 1:
            logger.info(
                f"后缀 {suffix or BASELINE_LABEL} 下检测到 {sig_count} 个不同超参签名，"
                f"已按签名拆分聚合（检查折目录是否残留旧消融产物）"
            )

    rows = []
    for (suffix, signature), group in work.groupby(
        ["suffix", "signature"], dropna=False, sort=False
    ):
        lifts = pd.to_numeric(group["lift"], errors="coerce")
        valid = lifts[lifts > 0]
        bias = pd.to_numeric(group["pred_bias"], errors="coerce").dropna()
        depth_vals = group["depth"].dropna()
        lr_vals = group["learning_rate"].dropna()
        lift_geo: Optional[float] = float(np.exp(np.log(valid).mean())) if len(valid) else None
        lift_min: Optional[float] = float(valid.min()) if len(valid) else None
        tuning_score: Optional[float] = (
            0.5 * (lift_geo + lift_min) if lift_geo is not None else None
        )
        rows.append(
            {
                "suffix": suffix if suffix else BASELINE_LABEL,
                # 未登记签名回退时带上后缀保持区分（与 legacy 回退对称），
                # 避免不同后缀的未登记组在历史聚合中被错误合并
                "param_signature": (
                    signature
                    if signature
                    else f"{UNREGISTERED_SIGNATURE}[{suffix if suffix else BASELINE_LABEL}]"
                ),
                "depth": int(depth_vals.iloc[0]) if len(depth_vals) else None,
                "learning_rate": float(lr_vals.iloc[0]) if len(lr_vals) else None,
                "n_folds": int(len(group)),
                "n_folds_valid_lift": int(len(valid)),
                "tuning_score": tuning_score,
                "lift_geo_mean": lift_geo,
                "lift_min": lift_min,
                "lift_std": float(valid.std()) if len(valid) > 1 else None,
                "pred_bias_mean": float(bias.mean()) if len(bias) else None,
                "gate_pass": bool(lift_min is not None and lift_min >= lift_min_threshold),
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values("tuning_score", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table[TUNING_COLUMNS]


def build_history_records(
    tuning_table: pd.DataFrame,
    wf_root: str,
    lift_min_threshold: float,
    timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """构建台账记录（每超参组一行，含运行上下文），追加与比较共用。"""
    stamp = timestamp or datetime.now().isoformat(timespec="seconds")
    records = []
    for _, row in tuning_table.iterrows():
        record = {
            "timestamp": stamp,
            "wf_root": str(wf_root),
            "lift_min_threshold": lift_min_threshold,
        }
        for col in HISTORY_RUN_COLUMNS[3:]:
            value = row.get(col)
            record[col] = "" if value is None or pd.isna(value) else value
        records.append(record)
    return pd.DataFrame(records, columns=HISTORY_RUN_COLUMNS)


def append_history_records(history_csv: Path, records: pd.DataFrame) -> int:
    """台账落盘追加（文件不存在时写表头），返回本次追加行数。

    首次创建用 utf-8-sig（Excel 友好），追加写 utf-8 无 BOM——pandas 的
    utf-8-sig 在追加模式下会重复写入 BOM，污染中间行。
    """
    if records.empty:
        return 0
    if history_csv.exists():
        records.to_csv(history_csv, mode="a", header=False, index=False, encoding="utf-8")
    else:
        records.to_csv(history_csv, index=False, encoding="utf-8-sig")
    return len(records)


def _load_history_csv(history_csv: Path) -> pd.DataFrame:
    """读回台账并补齐超参签名列（旧 schema 兼容）。

    旧台账（无 param_signature 列或存在空值）按 "(legacy) {suffix}" 回退
    身份，独立成组不与当前签名合并。
    """
    legacy = pd.read_csv(history_csv, encoding="utf-8-sig")
    if "suffix" not in legacy.columns:
        legacy["suffix"] = BASELINE_LABEL
    fallback = [f"(legacy) {s}" for s in legacy["suffix"].astype(str)]
    if "param_signature" not in legacy.columns:
        legacy["param_signature"] = fallback
    else:
        mask = legacy["param_signature"].isna() | (legacy["param_signature"].astype(str) == "")
        legacy.loc[mask, "param_signature"] = [f for f, m in zip(fallback, mask) if m]
    return legacy


def build_history_table(history_csv: Path, current_records: pd.DataFrame) -> pd.DataFrame:
    """台账（含当次记录）按超参签名聚合，输出历史最优比较表。

    "效果最好"判定口径 = score_best（历史最高单次调参分）；score_median
    供同签名多次运行参考，is_current_best 标记历史最优行是否来自当次汇总。
    """
    frames = []
    if history_csv.exists():
        frames.append(_load_history_csv(history_csv))
    frames.append(current_records)
    all_rows = pd.concat(frames, ignore_index=True)
    if all_rows.empty:
        return pd.DataFrame(columns=HISTORY_TABLE_COLUMNS)
    # 追加模式下台账已含当次行，与内存 records 重复 → 按当次时间戳+签名去重，
    # 避免同一次汇总被双计（两列均为稳定字符串，不受 CSV 往返 dtype 差异影响）
    all_rows = all_rows.drop_duplicates(subset=["timestamp", "param_signature"])

    for col in ("tuning_score", "lift_geo_mean", "lift_min"):
        all_rows[col] = pd.to_numeric(all_rows.get(col), errors="coerce")
    if "gate_pass" in all_rows.columns:
        all_rows["_gate"] = (
            all_rows["gate_pass"].astype(str).str.strip().str.lower().isin(["true", "1"])
        )
    else:
        all_rows["_gate"] = False
    current_stamps = set(current_records["timestamp"]) if not current_records.empty else set()

    rows = []
    for signature, group in all_rows.groupby("param_signature", dropna=False, sort=False):
        scores = group["tuning_score"].dropna()
        if scores.empty:
            continue
        best_row = group.loc[scores.idxmax()]
        latest_row = group.iloc[-1]
        rows.append(
            {
                "param_signature": signature,
                "suffix": str(latest_row.get("suffix", BASELINE_LABEL)),
                "depth": latest_row.get("depth"),
                "learning_rate": latest_row.get("learning_rate"),
                "runs": int(len(group)),
                "score_best": float(scores.max()),
                "score_median": float(scores.median()) if len(scores) > 1 else None,
                "score_latest": (
                    float(latest_row["tuning_score"])
                    if pd.notna(latest_row["tuning_score"])
                    else None
                ),
                "lift_min_best": float(group["lift_min"].max()),
                "gate_pass_rate": float(group["_gate"].mean()),
                "best_timestamp": str(best_row.get("timestamp", "")),
                "is_current_best": bool(best_row.get("timestamp") in current_stamps),
            }
        )
    if not rows:
        return pd.DataFrame(columns=HISTORY_TABLE_COLUMNS)
    table = pd.DataFrame(rows)
    table = table.sort_values("score_best", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    table.insert(0, "history_rank", range(1, len(table) + 1))
    return table[HISTORY_TABLE_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="terminal_loss 滚动 WF 汇总")
    parser.add_argument(
        "--wf-root", default="data/walk_forward/terminal_risk_wf", help="各折输出目录的父目录"
    )
    parser.add_argument(
        "--lift-min-threshold", type=float, default=1.1, help="组内 lift 最小值门禁（默认 1.1）"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="跳过 tuning_history.csv 台账追加（默认每次按组追加；"
        "历史比较仍基于既有台账 + 当次内存拼接）",
    )
    args = parser.parse_args()

    _enable_vt()
    wf_root = Path(args.wf_root)
    if not wf_root.exists():
        logger.error(f"目录不存在: {wf_root}")
        return 1
    summary = collect_fold_rows(wf_root)
    if summary.empty:
        logger.error("未找到任何折的 report")
        return 1

    out_csv = wf_root / "summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"summary 已写入 {out_csv}（{len(summary)} 折）")

    lifts = summary["lift"].dropna()
    if len(lifts) > 1:
        logger.info(
            f"全体折 lift 概览: mean={lifts.mean():.3f}, min={lifts.min():.3f}, "
            f"max={lifts.max():.3f}, std={lifts.std():.3f}（多组消融时仅供整体参考）"
        )
    bias = summary["pred_bias"].dropna()
    if len(bias):
        logger.info(
            f"pred_bias 跨折: mean={bias.mean():+.4f}（正值=概率高估，" f"C 段校准的输入信号）"
        )
    print(summary.to_string(index=False))

    # ── 调参分组指标（每超参组一行；单一调参分用于消融选择与跨 batch 对比）──
    tuning_table = build_tuning_table(summary, args.lift_min_threshold)
    out_scores = wf_root / "tuning_scores.csv"
    tuning_table.to_csv(out_scores, index=False, encoding="utf-8-sig")
    logger.info(f"tuning_scores 已写入 {out_scores}（{len(tuning_table)} 组）")
    print()
    print(_highlight_first_row(tuning_table, _BOLD_CYAN))

    for _, row in tuning_table.iterrows():
        conclusion = (
            "通过 → 建议进入第二阶段（C 段校准 + V 段阈值）"
            if row["gate_pass"]
            else "未通过 → 建议先做超参/特征消融再重跑"
        )
        logger.info(
            f"超参组 {row['suffix']}[{row['param_signature']}]: "
            f"lift_geo_mean={_fmt(row['lift_geo_mean'])}, "
            f"lift_min={_fmt(row['lift_min'])}, lift_std={_fmt(row['lift_std'])}, "
            f"lift 有效 {int(row['n_folds_valid_lift'])}/{int(row['n_folds'])} 折；"
            f"门禁（lift_min >= {args.lift_min_threshold}）{conclusion}"
        )

    best = tuning_table.iloc[0]
    print()
    print(_c("=" * 68, _BOLD_CYAN))
    print(
        _c(
            f"调参指标（本次最优）: 超参组 {best['suffix']}  调参分 = {_fmt(best['tuning_score'])}",
            _BOLD_CYAN,
        )
    )
    print(
        f"  tuning_score = 0.5×lift_geo_mean({_fmt(best['lift_geo_mean'])}) + "
        f"0.5×lift_min({_fmt(best['lift_min'])})；"
        f"{int(best['n_folds_valid_lift'])}/{int(best['n_folds'])} 折有效 lift，"
        f"门禁{'通过' if best['gate_pass'] else '未通过'}"
    )
    print(_c("=" * 68, _BOLD_CYAN))

    # ── 历史比较（超参签名为身份键；--no-history 时仅内存拼当次，不落盘）──
    history_records = build_history_records(tuning_table, str(wf_root), args.lift_min_threshold)
    history_csv = wf_root / "tuning_history.csv"
    if not args.no_history:
        appended = append_history_records(history_csv, history_records)
        logger.info(f"调参台账已追加 {appended} 行 → {history_csv}")

    history_table = build_history_table(history_csv, history_records)
    if history_table.empty:
        logger.warning("历史比较表为空（无有效调参分），跳过历史最优输出")
        return 0
    print()
    print(_highlight_first_row(history_table, _BOLD_GREEN))

    hist_best = history_table.iloc[0]
    cur_score = best["tuning_score"]
    cur_reaches_best = bool(
        cur_score is not None
        and pd.notna(cur_score)
        and float(cur_score) >= float(hist_best["score_best"])
    )
    print()
    print(_c("=" * 68, _BOLD_CYAN))
    print(
        _c(
            f"历史最优（含当次，共 {int(history_table['runs'].sum())} 次汇总、"
            f"{len(history_table)} 个超参组）",
            _BOLD_CYAN,
        )
    )
    print(f"  最优超参签名: {_c(str(hist_best['param_signature']), _BOLD_GREEN)}")
    print(
        f"  历史最好调参分 = {_c(_fmt(hist_best['score_best']), _BOLD_GREEN)}"
        f"（{int(hist_best['runs'])} 次运行，best @ {hist_best['best_timestamp']}，"
        f"lift_min_best={_fmt(hist_best['lift_min_best'])}，"
        f"门禁通过率 {float(hist_best['gate_pass_rate']):.0%}）"
    )
    if cur_reaches_best:
        print(
            _c(
                f"  ★ 当次超参组 {best['suffix']} 达到/刷新历史最优（调参分 {_fmt(cur_score)}）",
                _BOLD_YELLOW,
            )
        )
    else:
        gap = (
            float(cur_score) - float(hist_best["score_best"])
            if cur_score is not None and pd.notna(cur_score)
            else None
        )
        gap_str = f"{gap:+.3f}" if gap is not None else "NA"
        print(
            _c(
                f"  当次最优 {best['suffix']}: 调参分 {_fmt(cur_score)} —— "
                f"未达历史最优（差距 {gap_str}）",
                _YELLOW,
            )
        )
    print(_c("=" * 68, _BOLD_CYAN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
