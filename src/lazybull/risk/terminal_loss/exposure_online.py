"""政策层在线暴露系数（terminal_loss P2-5）：λ_t 随持仓**实时计算**。

目的：让回测/实盘**不再依赖预导出的系数表**——`λ_t` 由"折模型 + 当日持仓 + 当日截面"
每天现算。系数表退化为可选的缓存/导出/回放产物（其正确性绑定一份持仓路径，
换任何会改变持仓的配置都必须重算）。

**口径唯一（与离线链路共用同一套实现，禁止复制公式）**：

- 折模型映射：``policy_sidecar.load_fold_index`` + ``select_fold_for_date``
  （只用"该日期时 Train/Val 都已结束"的折模型；ES 段未参与训练，可用于评估）；
- 打分输入装配：``policy_sidecar._attach_inputs``（cs_train 直读 + σ 严格日历对齐 + 期限列）；
- 事后标签：``labels.build_terminal_loss_labels``（**成熟后**才入账，非 valid 剔除）；
- 日级聚合与判定：``exposure_gate.build_daily_frame`` + ``apply_rolling_gate`` /
  ``calibrate_gate`` + ``apply_gate``。

**语义（S1，引擎唯一口径）**：``λ_t`` 用**判定时点持仓**（当日执行前）计算。
因果链：判定日 T 收盘可得 → T+1 开盘执行；买入缩放按"**信号日** λ"（与
``backtest/exposure_override.py`` 契约一致）。实盘可实现的形态就是这一种。

**为何不能与冻结表逐位一致**：离线台账的 ``p_loss`` 均值用的是**当日收盘后**
（含当日成交）的持仓快照；引擎在盘中拿不到收盘后的持仓。二者差异只来自"当日持仓口径"，
可通过 ``replay_lambda_series``（收盘模式）自证实现同源，再单独度量 S1 的差异规模。

**同日单次评估**：引擎会在 T0 判定与 T+1（买入按**信号日**取 λ）各调一次；
``multiplier_for`` 对同一日期返回**首次评估结果**（缓存），不重复打分、不污染面板。
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .dataset import load_clean_daily_panels, load_trade_calendar
from .exposure_gate import (
    ARMS,
    ExposureGateConfig,
    RollingGateConfig,
    apply_gate,
    apply_rolling_gate,
    build_daily_frame,
    calibrate_gate,
)
from .labels import TerminalLossLabelConfig, build_terminal_loss_labels
from .policy_sidecar import (
    _attach_inputs,
    _default_model_loader,
    _read_snapshot,
    load_fold_index,
    select_fold_for_date,
)

#: 允许的策略参数键（单变量直读：未知键直接报错，禁止多层兼容回退）
_POLICY_KEYS = (
    "arm",
    "mode",
    "window",
    "min_window",
    "regime_q",
    "score_q",
    "lambda",
    "cost_bps",
    "min_layer",
    "calib_start",
    "calib_end",
)

#: 台账所需列（由 provider 自行组装：与离线台账内部键同名）
_LEDGER_COLUMNS = [
    "date",
    "fold",
    "ts_code",
    "weight",
    "remaining_intervals",
    "mkt_vol_20",
    "p_loss",
    "terminal_return",
    "loss_label",
    "label_status",
]


def to_date_str(value: Any) -> str:
    """统一为 YYYYMMDD 字符串（与各层日期契约一致）。"""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip().replace("-", "").replace("/", "")
    return text[:8]


@dataclass(frozen=True)
class OnlinePolicyConfig:
    """在线政策层配置（回测/实盘同源；只允许显式取值，无隐式默认链）。"""

    arm: str = "combined"
    mode: str = "rolling"
    window_days: int = 250
    min_window_days: int = 60
    regime_quantile: float = 2.0 / 3.0
    score_quantile: float = 0.5
    de_exposure_multiplier: float = 0.5
    cost_bps: float = 15.0
    min_layer_days: int = 20
    calibration_start: Optional[str] = None
    calibration_end: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"未知臂 {self.arm!r}；合法取值 {list(ARMS)}")
        if self.mode not in ("rolling", "fixed"):
            raise ValueError(f"未知阈值口径 {self.mode!r}；合法取值 ['rolling', 'fixed']")
        if self.window_days < 2:
            raise ValueError(f"window_days 至少为 2，当前 {self.window_days}")
        if not 2 <= self.min_window_days <= self.window_days:
            raise ValueError(
                f"min_window_days 必须落于 [2, window_days]，当前 {self.min_window_days}"
            )
        if not 0.0 < self.regime_quantile < 1.0:
            raise ValueError(f"regime_quantile 必须落于 (0, 1)，当前 {self.regime_quantile}")
        if self.arm != "regime" and not 0.0 < self.score_quantile < 1.0:
            raise ValueError(f"score_quantile 必须落于 (0, 1)，当前 {self.score_quantile}")
        if not 0.0 < self.de_exposure_multiplier <= 1.0:
            raise ValueError(f"lambda 必须落于 (0, 1]，当前 {self.de_exposure_multiplier}")
        if self.cost_bps < 0.0:
            raise ValueError(f"cost_bps 不能为负，当前 {self.cost_bps}")
        if self.mode == "fixed":
            if not self.calibration_start or not self.calibration_end:
                raise ValueError("mode=fixed 必须提供 calib_start 与 calib_end（校准段）")
            if self.calibration_end >= self.calibration_start:
                raise ValueError("calib_end 必须晚于 calib_start")

    @classmethod
    def parse(cls, spec: str) -> "OnlinePolicyConfig":
        """解析 ``k=v,k=v`` 形式的策略字符串（未知键直接报错）。"""
        values: Dict[str, str] = {}
        for item in str(spec or "").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"策略项需为 k=v 形式: {item!r}")
            key, value = item.split("=", 1)
            key = key.strip()
            if key not in _POLICY_KEYS:
                raise ValueError(f"未知策略键 {key!r}；合法键 {list(_POLICY_KEYS)}")
            values[key] = value.strip()
        if not values:
            raise ValueError("策略字符串为空：至少给出 arm=...")
        return cls(
            arm=values.get("arm", cls.arm),
            mode=values.get("mode", cls.mode),
            window_days=int(values.get("window", cls.window_days)),
            min_window_days=int(values.get("min_window", cls.min_window_days)),
            regime_quantile=float(values.get("regime_q", cls.regime_quantile)),
            score_quantile=float(values.get("score_q", cls.score_quantile)),
            de_exposure_multiplier=float(values.get("lambda", cls.de_exposure_multiplier)),
            cost_bps=float(values.get("cost_bps", cls.cost_bps)),
            min_layer_days=int(values.get("min_layer", cls.min_layer_days)),
            calibration_start=values.get("calib_start"),
            calibration_end=values.get("calib_end"),
            label=values.get("label", ""),
        )

    def describe(self) -> str:
        """规范字符串（回显、入 summary、指纹计算共用）。"""
        parts = [
            f"arm={self.arm}",
            f"mode={self.mode}",
            f"window={self.window_days}",
            f"min_window={self.min_window_days}",
            f"regime_q={self.regime_quantile:.6f}",
            f"score_q={self.score_quantile:.6f}",
            f"lambda={self.de_exposure_multiplier:.4f}",
            f"cost_bps={self.cost_bps:.2f}",
            f"min_layer={self.min_layer_days}",
        ]
        if self.mode == "fixed":
            parts.append(f"calib_start={self.calibration_start}")
            parts.append(f"calib_end={self.calibration_end}")
        return ",".join(parts)

    @property
    def arm_label(self) -> str:
        """用户可见臂标签（默认按口径生成；用于产物目录/日志）。"""
        if self.label:
            return self.label
        base = {"combined": "E2", "regime": "纯regime", "score": "纯分数"}[self.arm]
        if self.mode == "rolling":
            return f"{base}-滚动{self.window_days}日"
        return f"{base}-固定校准段"

    def rolling_config(self) -> RollingGateConfig:
        if self.mode != "rolling":
            raise ValueError("mode=rolling 才能构造 RollingGateConfig")
        return RollingGateConfig(
            arm=self.arm,
            window_days=self.window_days,
            regime_quantile=self.regime_quantile,
            score_quantile=self.score_quantile,
            de_exposure_multiplier=self.de_exposure_multiplier,
            cost_bps=self.cost_bps,
            min_window_days=self.min_window_days,
            label=self.arm_label,
        )

    def fixed_config(self) -> ExposureGateConfig:
        if self.mode != "fixed":
            raise ValueError("mode=fixed 才能构造 ExposureGateConfig")
        return ExposureGateConfig(
            arm=self.arm,
            regime_quantile=self.regime_quantile,
            score_quantile=self.score_quantile,
            de_exposure_multiplier=self.de_exposure_multiplier,
            cost_bps=self.cost_bps,
            min_layer_days=self.min_layer_days,
        )


class DailyExposureProvider:
    """逐日计算暴露系数 λ_t（引擎只调用 :meth:`multiplier_for`）。

    调用契约（每交易日一次，**判定时点**）：

    - ``holdings``: 当日持仓行，列 ``ts_code`` / ``weight`` / ``remaining_intervals``
      （与 ``common.sidecar_schema`` 的持仓快照同源；``remaining_intervals`` 为
      剩余持有交易日，必须在 [1, h_max] 内，超出直接报错）；
    - 返回：当日暴露系数（``<1`` = 触发降暴露；否则 1.0）。

    ``coverage_start`` 语义（**只抑制动作，不抑制历史**）：该日之前的交易日仍照常
    打分、入面板、累积滚动阈值历史（否则窗口冷启动，与离线导出“历史全量 + 表按窗口
    裁剪”的口径不一致、开头数十个交易日无法触发），但 λ 恒返回 1.0（不动作）；
    被抑制的触发日计入 ``pre_coverage_trigger_days``。
    """

    def __init__(
        self,
        config: OnlinePolicyConfig,
        *,
        risk_root: str,
        arm_suffix: str,
        data_root: str,
        feature_root: Optional[str] = None,
        coverage_start: Optional[str] = None,
        book: str = "pre_exec",
        verbose: bool = False,
        model_loader=None,
    ) -> None:
        if book not in ("pre_exec", "end_of_day"):
            raise ValueError(f"未知持仓口径 {book!r}；合法取值 ['pre_exec', 'end_of_day']")
        self.config = config
        self.risk_root = str(risk_root)
        self.arm_suffix = str(arm_suffix)
        self.data_root = str(data_root)
        self.feature_root = Path(feature_root or (Path(data_root) / "features" / "cs_train"))
        self.coverage_start = to_date_str(coverage_start) if coverage_start else None
        self.book = book
        self.verbose = verbose
        self._model_loader = model_loader or _default_model_loader
        self.folds = load_fold_index(self.risk_root, self.arm_suffix)
        self._models: Dict[str, Any] = {}
        self._sigma_long: Dict[str, pd.DataFrame] = {}
        self._panels: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, List[str]]] = {}
        # 已打分日行的**面板**（未成熟日带 NaN 标签；成熟后回填/剔除非法行）
        self._frames: Dict[str, pd.DataFrame] = {}
        self._pending: Dict[str, pd.DataFrame] = {}
        self._calendar: Optional[List[str]] = None
        self._calendar_pos: Dict[str, int] = {}
        self._fixed_calibration = None
        # 同一交易日只评估一次：引擎会在 T0 判定与 T+1（买入按**信号日**取 λ）各调一次，
        # 两次必须得到同一个 λ，且不得重置/重复污染面板。
        self._cache: Dict[str, float] = {}
        # 上一次判定结果（缺口日顺延；与离线系数表导出的 ffill 语义一致）
        self._last_multiplier: float = 1.0
        self.stats: Dict[str, Any] = {
            "judged_days": 0,
            "trigger_days": 0,
            "skipped_no_model": 0,
            "skipped_out_of_coverage": 0,
            "pre_coverage_judged_days": 0,
            "pre_coverage_trigger_days": 0,
            "skipped_no_holdings": 0,
            "skipped_window_short": 0,
            "scored_rows": 0,
            "matured_rows": 0,
            "dropped_invalid_rows": 0,
            "pending_rows": 0,
            "cached_days": 0,
            "carried_forward_days": 0,
            "dropped_days": [],
        }

    # ------------------------------------------------------------------ 对外
    @property
    def fingerprint(self) -> str:
        """政策指纹（策略定义 + 模型来源）：用于把导出的系数表绑定到口径。"""
        payload = "|".join(
            [
                self.config.describe(),
                str(Path(self.risk_root).resolve()),
                self.arm_suffix,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def multiplier_for(self, date: Any, holdings: pd.DataFrame) -> float:
        """返回给定交易日的暴露系数。

        - **同日重复调用**（引擎在 T0 判定与 T+1 缩放各取一次）返回**首次评估结果**；
        - **缺口日顺延前一状态**：无持仓 / 窗口不足 / 覆盖外时不重算，沿用上一次判定值
          （与离线系数表导出时的缺口补齐 `ffill` 语义一致；无历史时为 1.0）。
        """
        date_str = to_date_str(date)
        if date_str in self._cache:
            self.stats["cached_days"] += 1
            return self._cache[date_str]
        multiplier = self._evaluate(date_str, holdings)
        if multiplier is not None:
            self._last_multiplier = float(multiplier)
            result = float(multiplier)
        else:
            result = float(self._last_multiplier)
            self.stats["carried_forward_days"] += 1
        self._cache[date_str] = result
        return result

    def _evaluate(self, date_str: str, holdings: pd.DataFrame) -> Optional[float]:
        fold = select_fold_for_date(date_str, self.folds)
        if fold is None:
            self.stats["skipped_no_model"] += 1
            return None
        if not (fold.es_start <= date_str <= fold.es_end):
            self.stats["skipped_out_of_coverage"] += 1
            if self.verbose:
                logger.info(
                    f"暴露门控在线判定: {date_str} 不在折 {fold.fold} 的 ES 覆盖区间"
                    f"（{fold.es_start}~{fold.es_end}），不判定"
                )
            return None
        rows = self._normalize_holdings(date_str, fold.fold, holdings)
        if rows.empty:
            self.stats["skipped_no_holdings"] += 1
            return None
        scored = self._score_rows(fold, rows)
        scored["terminal_return"] = np.nan
        scored["loss_label"] = np.nan
        scored["label_status"] = "pending"
        self._frames[date_str] = scored
        self._pending[date_str] = scored
        self.stats["scored_rows"] += int(len(scored))
        self.stats["pending_rows"] = int(sum(len(item) for item in self._pending.values()))
        self._mature_pending(date_str)
        judged = self._judge_day(date_str)
        if judged is None:
            return None
        multiplier = float(judged["exposure_multiplier"].iloc[0])
        # 覆盖生效日之前：**面板/阈值历史照常累积**（否则滚动窗口冷启动、
        # 与离线导出“历史全量 + 表按窗口裁剪”的口径不一致），但**不产生动作**（λ 恒为 1.0）。
        if self.coverage_start and date_str < self.coverage_start:
            self.stats["pre_coverage_judged_days"] += 1
            if multiplier < 1.0:
                self.stats["pre_coverage_trigger_days"] += 1
            return 1.0
        self.stats["judged_days"] += 1
        if multiplier < 1.0:
            self.stats["trigger_days"] += 1
        if self.verbose and multiplier < 1.0:
            logger.info(
                f"暴露门控在线判定: {date_str} 触发降暴露 λ={multiplier:.2f}"
                f"（折 {fold.fold}，持仓 {len(rows)} 只）"
            )
        return multiplier

    def daily_series(self) -> pd.DataFrame:
        """已入账的日级面板（date/fold/mkt_vol_20/p_loss_mean/…），供审计。"""
        return build_daily_frame(self._ledger())

    def lambda_series(self) -> pd.DataFrame:
        """已评估的逐日暴露系数（`date` / `multiplier`，按日期升序）。

        包含引擎实际消费的全部日期（含缺口日顺延值），用于证据归档与回放对比；
        引擎不读该序列，判定仍由 `multiplier_for` 现算。
        """
        if not self._cache:
            return pd.DataFrame(columns=["date", "multiplier"])
        rows = sorted(self._cache.items())
        return pd.DataFrame(
            {
                "date": [day for day, _ in rows],
                "multiplier": [float(value) for _, value in rows],
            }
        )

    def stats_summary(self) -> Dict[str, Any]:
        """统计（含策略与模型来源），随 summary 落盘。"""
        payload = dict(self.stats)
        payload.update(
            {
                "policy_config": self.config.describe(),
                "policy_arm_label": self.config.arm_label,
                "policy_fingerprint": self.fingerprint,
                "policy_risk_root": self.risk_root,
                "policy_arm_suffix": self.arm_suffix,
                "policy_book": self.book,
                "policy_coverage_start": self.coverage_start or "",
            }
        )
        return payload

    # ------------------------------------------------------------- 内部实现
    def _calendar_list(self) -> List[str]:
        if self._calendar is None:
            self._calendar = [
                to_date_str(d) for d in load_trade_calendar(self.data_root, "19900101", "20991231")
            ]
            self._calendar_pos = {day: idx for idx, day in enumerate(self._calendar)}
        return self._calendar

    def _normalize_holdings(
        self, date_str: str, fold_name: str, holdings: pd.DataFrame
    ) -> pd.DataFrame:
        """规范化持仓行（与离线快照同口径；缺失/非法行明确报错）。"""
        if holdings is None or len(holdings) == 0:
            return pd.DataFrame(columns=_LEDGER_COLUMNS)
        frame = holdings.copy()
        missing = [c for c in ("ts_code", "weight", "remaining_intervals") if c not in frame]
        if missing:
            raise ValueError(f"持仓行缺少列 {missing}（需与 sidecar_schema 快照同源）")
        frame["remaining_intervals"] = pd.to_numeric(frame["remaining_intervals"], errors="coerce")
        frame = frame[frame["remaining_intervals"].notna()]
        frame = frame[frame["remaining_intervals"] >= 1].copy()
        if frame.empty:
            return pd.DataFrame(columns=_LEDGER_COLUMNS)
        frame["remaining_intervals"] = frame["remaining_intervals"].astype(int)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        frame = frame[frame["weight"].notna()].copy()
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["date"] = date_str
        frame["fold"] = fold_name
        return frame[["date", "fold", "ts_code", "weight", "remaining_intervals"]]

    def _model_for(self, fold) -> Any:
        if fold.fold not in self._models:
            self._models[fold.fold] = self._model_loader(str(fold.model_path))
        return self._models[fold.fold]

    def _panels_for(self, fold) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """折内数据面板（open/σ）+ 折内交易日轴；按折缓存，用完即释放。"""
        if fold.fold not in self._panels:
            calendar = self._calendar_list()
            # 折的 ES 边界可能不是交易日（如 20221231 为周六）⇒ 用二分定位到日历内位置
            start_idx = bisect_left(calendar, fold.es_start)
            end_idx = bisect_right(calendar, fold.es_end) - 1
            if end_idx < 0 or start_idx >= len(calendar):
                raise ValueError(
                    f"折 {fold.fold} 的 ES 区间 {fold.es_start}~{fold.es_end} "
                    f"落在交易日历之外，无法加载面板"
                )
            h_max = max(self._h_max_for(fold), 1)
            lo = max(0, start_idx - 30)
            hi = min(len(calendar), end_idx + h_max + 3)
            span = calendar[lo:hi]
            open_panel, close_panel, _, _ = load_clean_daily_panels(
                self.data_root, span[0], span[-1]
            )
            from ...factors.risk.volatility_factors import compute_sigma_daily_panel

            sigma_panel = compute_sigma_daily_panel(
                close_panel.stack().rename("close_adj").reset_index(),
                list(span),
                window=20,
            )
            self._panels[fold.fold] = (open_panel, sigma_panel, list(span))
            logger.info(
                f"暴露门控在线: 已加载折 {fold.fold} 面板"
                f"（{span[0]}~{span[-1]}，{len(span)} 个交易日）"
            )
        return self._panels[fold.fold]

    @staticmethod
    def _h_max_for(fold) -> int:
        return int(getattr(fold, "h_max", 20) or 20)

    def _score_rows(self, fold, rows: pd.DataFrame) -> pd.DataFrame:
        """按折模型给当日持仓打分（复用离线装配实现；缺列明确报错）。"""
        model = self._model_for(fold)
        feature_names = list(getattr(model, "feature_names", None) or fold.feature_names)
        open_panel, sigma_panel, span = self._panels_for(fold)
        if fold.fold not in self._sigma_long:
            stacked = sigma_panel.stack().rename("sigma_daily_20").reset_index()
            stacked.columns = ["date", "ts_code", "sigma_daily_20"]
            stacked["date"] = stacked["date"].astype(str)
            stacked["ts_code"] = stacked["ts_code"].astype(str)
            self._sigma_long[fold.fold] = stacked
        scored = _attach_inputs(
            rows.copy(),
            self.data_root,
            self.feature_root,
            feature_names,
            self._sigma_long[fold.fold],
            context_columns=("mkt_vol_20",),
        )
        missing = [c for c in feature_names if c not in scored.columns]
        if missing:
            raise ValueError(f"持仓打分缺少输入列 {sorted(missing)}（禁止静默降级）")
        scored["p_loss"] = np.asarray(model.predict_proba(scored[feature_names]), dtype=float)
        return scored[
            [
                "date",
                "fold",
                "ts_code",
                "weight",
                "remaining_intervals",
                "mkt_vol_20",
                "p_loss",
            ]
        ]

    def _mature_pending(self, date_str: str) -> None:
        """把已成熟的待定行补上事后标签；仅 `valid` 行保留（与离线一致）。

        未成熟日的行**仍留在面板里**（标签为 NaN）——离线台账是事后建的，面板里包含它们；
        若丢掉，滚动窗口会缺日、阈值不同。成熟后若发现非法，则从面板中剔除（影响后续阈值）。
        """
        calendar = self._calendar_list()
        today_pos = self._calendar_pos[date_str]
        for day in sorted(self._pending.keys()):
            if day == date_str:
                continue
            fold_name = str(self._pending[day]["fold"].iloc[0])
            fold = next((item for item in self.folds if item.fold == fold_name), None)
            if fold is None:
                raise ValueError(f"待定行引用了未知折 {fold_name!r}")
            h_max = self._h_max_for(fold)
            day_pos = self._calendar_pos[day]
            if today_pos < day_pos + h_max + 1:
                continue  # 仍不成熟
            open_panel, sigma_panel, span = self._panels_for(fold)
            tail = calendar[day_pos : min(len(calendar), day_pos + h_max + 2)]
            label_config = TerminalLossLabelConfig(
                h_min=1,
                h_max=h_max,
                loss_sigma_multiple=float(fold.loss_sigma_multiple),
            )
            # 只取该日待成熟股票的价格/σ 子集：标签按股票自身上下文计算，
            # 全市场展开会把 22 天 × 5,500 只 × 20 期限的网格算成百万行（实测秒级/日）
            codes = [
                c for c in dict.fromkeys(self._pending[day]["ts_code"]) if c in open_panel.columns
            ]
            rows_index = open_panel.index.intersection(tail)
            labels = build_terminal_loss_labels(
                open_panel.loc[rows_index, codes],
                sigma_panel.loc[sigma_panel.index.intersection(tail), codes],
                label_config,
            )
            if labels.empty:
                labels = pd.DataFrame(
                    columns=[
                        "trade_date",
                        "ts_code",
                        "h",
                        "terminal_return",
                        "loss_label",
                        "label_status",
                    ]
                )
            labels = labels.rename(columns={"trade_date": "date", "h": "remaining_intervals"})
            left = self._pending[day].drop(
                columns=["terminal_return", "loss_label", "label_status"], errors="ignore"
            )
            merged = left.merge(
                labels[
                    [
                        "date",
                        "ts_code",
                        "remaining_intervals",
                        "terminal_return",
                        "loss_label",
                        "label_status",
                    ]
                ],
                on=["date", "ts_code", "remaining_intervals"],
                how="left",
            )
            merged["terminal_return"] = pd.to_numeric(merged["terminal_return"], errors="coerce")
            invalid = merged["label_status"].fillna("missing") != "valid"
            if int(invalid.sum()):
                self.stats["dropped_invalid_rows"] += int(invalid.sum())
                self.stats["dropped_days"].append(day)
            valid = merged[~invalid].copy()
            if valid.empty:
                self._frames.pop(day, None)
            else:
                self._frames[day] = valid
                self.stats["matured_rows"] += int(len(valid))
            del self._pending[day]
            if not any(str(item["fold"].iloc[0]) == fold_name for item in self._pending.values()):
                self._panels.pop(fold_name, None)  # 释放该折面板
                self._sigma_long.pop(fold_name, None)

    def _ledger(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=_LEDGER_COLUMNS)
        return pd.concat(list(self._frames.values()), ignore_index=True)

    def _judge_day(self, date_str: str) -> Optional[pd.DataFrame]:
        """当日判定：阈值来自面板（含未成熟日的 NaN 标签行），当日行已在面板内。"""
        daily = build_daily_frame(self._ledger())
        if self.config.mode == "rolling":
            judged, skipped = apply_rolling_gate(
                daily, self.config.rolling_config(), date_str, date_str
            )
            if skipped:
                self.stats["skipped_window_short"] += 1
                return None
            return judged
        if self._fixed_calibration is None:
            calibration = calibrate_gate(
                daily,
                self.config.fixed_config(),
                str(self.config.calibration_start),
                str(self.config.calibration_end),
            )
            self._fixed_calibration = calibration
        judged = apply_gate(
            daily,
            self._fixed_calibration,
            date_str,
            date_str,
        )
        return judged


def replay_lambda_series(
    provider: DailyExposureProvider,
    snapshot_files: Sequence[Path],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """用**离线持仓快照**按日回放 provider，返回逐日 λ（自证：应与冻结表逐值一致）。

    Args:
        provider: 在线 provider（此时 ``holdings`` 来自快照 = 收盘后口径）
        snapshot_files: 中文表头持仓快照 CSV（``backtest/holdings_snapshot`` 产物）
        start/end: 可选日期裁剪（YYYYMMDD）

    Returns:
        DataFrame，列 ``date`` / ``multiplier`` / ``rows``
    """
    frames = []
    for path in snapshot_files:
        frame = _read_snapshot(Path(path))
        frames.append(frame)
    if not frames:
        raise ValueError("未提供任何持仓快照文件")
    snapshots = pd.concat(frames, ignore_index=True)
    snapshots["date"] = snapshots["date"].map(to_date_str)
    if start:
        snapshots = snapshots[snapshots["date"] >= to_date_str(start)]
    if end:
        snapshots = snapshots[snapshots["date"] <= to_date_str(end)]
    records: List[Dict[str, object]] = []
    for day, part in snapshots.groupby("date", sort=True):
        multiplier = provider.multiplier_for(day, part)
        records.append({"date": str(day), "multiplier": multiplier, "rows": int(len(part))})
    return pd.DataFrame(records)


def load_exported_table_meta(table_path: Path) -> Optional[Dict[str, Any]]:
    """读取系数表旁的 meta（``<table>.meta.json``）；缺失返回 None。"""
    meta_path = Path(table_path).with_suffix(Path(table_path).suffix + ".meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def log_exported_table_provenance(table_path: Path) -> Optional[Dict[str, Any]]:
    """打印系数表的来源（meta）；缺失时告警。

    系数表是**冻结产物**：它绑定一份持仓路径，改过任何影响持仓的配置（Top-N / 调仓周期 /
    仓位模式 / 分批 / 止损等）后**必须重新导出**，否则表与路径不一致。本函数只做"来源可见"，
    不做强制比对（表与在线政策互斥，加载侧无参照对象可比对指纹）。

    Returns:
        meta 字典（缺失时为 None）。
    """
    table_path = Path(table_path)
    meta = load_exported_table_meta(table_path)
    if meta:
        logger.info(
            "  系数表来源（meta）: 策略={}｜模型根={}（{}）｜指纹={}｜生效起点={}".format(
                meta.get("policy_config", "-"),
                meta.get("risk_root", "-"),
                meta.get("arm_suffix", "-"),
                meta.get("policy_fingerprint", "-"),
                meta.get("coverage_start") or "全区间",
            )
        )
        return meta
    logger.warning(
        "系数表缺少 meta（{}）⇒ 无法追溯其策略口径与模型来源；"
        "该表为冻结产物（绑定一份持仓路径），改过任何影响持仓的配置后必须重新导出，禁止复用".format(
            table_path.with_suffix(table_path.suffix + ".meta.json").name
        )
    )
    return None


def write_exported_table_meta(
    table_path: Path,
    *,
    policy_config: OnlinePolicyConfig,
    risk_root: str,
    arm_suffix: str,
    coverage_start: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """写出系数表 meta（绑定策略口径与模型来源，供加载侧校验）。"""
    table_path = Path(table_path)
    meta_path = table_path.with_suffix(table_path.suffix + ".meta.json")
    payload: Dict[str, Any] = {
        "policy_config": policy_config.describe(),
        "policy_fingerprint": hashlib.sha1(
            "|".join(
                [policy_config.describe(), str(Path(risk_root).resolve()), str(arm_suffix)]
            ).encode("utf-8")
        ).hexdigest()[:16],
        "risk_root": str(risk_root),
        "arm_suffix": str(arm_suffix),
        "coverage_start": to_date_str(coverage_start) if coverage_start else "",
    }
    if extra:
        payload.update(extra)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path
