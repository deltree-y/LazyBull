"""数据存储模块"""

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from ..common.config import get_data_path, get_data_root


def _partition_key_to_date(value: object) -> Optional[str]:
    """把分组键标准化为 YYYYMMDD 日期字符串（供单文件迁移分区用）。"""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d") if not pd.isna(value) else None
    text = str(value).strip().replace("-", "")
    if len(text) < 8 or not text[:8].isdigit():
        return None
    date_str = text[:8]
    # 校验真实日历日期（如 20230230 视为无效），避免保存分区时抛 ValueError 中断迁移
    from datetime import datetime as _dt

    try:
        _dt.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None
    return date_str


class Storage:
    """数据存储类

    负责数据的持久化存储，优先使用Parquet格式
    支持按日期分区存储raw和clean数据
    """

    def __init__(self, root_path: Optional[str] = None, verbose: bool = False):
        """初始化存储

        Args:
            root_path: 数据根目录；未传时使用项目配置中的 data.root / data.*
            verbose: 是否输出详细日志

        注意：
            - trade_cal和stock_basic使用单文件存储（不分区）
            - daily/daily_basic/adj_factor/suspend/stk_limit等使用按日期分区存储
            - clean层数据使用按日期分区存储
            - features层数据使用按日期分区存储
        """
        if root_path is None:
            self.root_path = Path(get_data_root())
            self.raw_path = Path(get_data_path("raw", str(self.root_path / "raw")))
            self.clean_path = Path(get_data_path("clean", str(self.root_path / "clean")))
            self.features_path = Path(get_data_path("features", str(self.root_path / "features")))
            self.reports_path = Path(get_data_path("reports", str(self.root_path / "reports")))
        else:
            self.root_path = Path(root_path)
            self.raw_path = self.root_path / "raw"
            self.clean_path = self.root_path / "clean"
            self.features_path = self.root_path / "features"
            self.reports_path = self.root_path / "reports"
        self.verbose = verbose

        # 确保目录存在
        for path in [self.raw_path, self.clean_path, self.features_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"数据存储初始化完成，根目录: {self.root_path}")

    def save_raw(
        self, df: pd.DataFrame, name: str, format: str = "parquet", is_force: bool = False
    ) -> None:
        """保存原始数据

        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.raw_path / name, format, is_force)

    def save_clean(
        self, df: pd.DataFrame, name: str, format: str = "parquet", is_force: bool = False
    ) -> None:
        """保存清洗后数据

        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.clean_path / name, format, is_force)

    def save_features(
        self, df: pd.DataFrame, name: str, format: str = "parquet", is_force: bool = False
    ) -> None:
        """保存特征数据

        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.features_path / name, format, is_force)

    def save_report(
        self, df: pd.DataFrame, name: str, format: str = "csv", is_force: bool = False
    ) -> None:
        """保存报告数据

        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，csv/parquet
        """
        self._save_data(df, self.reports_path / name, format, is_force)

    _WATERMARK_FILE = "_sync_watermark.json"

    def load_sync_watermark(self, name: str) -> Optional[str]:
        """读取数据集同步水位（已成功查询至的日期，YYYYMMDD）。

        水位记录该数据集已成功查询到目标日期的公告/事件数据（无公告日也算已同步），
        供增量补齐门控判断使用，避免空白日期被反复下载。
        """
        path = self.raw_path / name / self._WATERMARK_FILE
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("synced_to")
        except (OSError, ValueError) as exc:
            logger.warning(f"读取同步水位失败 {path}: {exc}")
            return None

    def save_sync_watermark(self, name: str, date_str: str) -> None:
        """保存数据集同步水位（YYYYMMDD）。

        通过同目录唯一临时文件 + os.replace 原子替换：崩溃时正式水位文件保持不变，
        只会触发安全重查；唯一临时文件名避免多进程并发同步同一数据集时互相覆盖。
        """
        path = self.raw_path / name / self._WATERMARK_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="_sync_watermark.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"synced_to": date_str}, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            # 失败时清理残留临时文件，不掩盖原始异常（正常路径已由 os.replace 移动）
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise

    def load_raw(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载原始数据

        Args:
            name: 文件名（不含扩展名）
            format: 文件格式

        Returns:
            数据DataFrame，不存在返回None
        """
        return self._load_data(self.raw_path / name, format)

    def load_clean(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载清洗后数据

        Args:
            name: 文件名（不含扩展名）
            format: 文件格式

        Returns:
            数据DataFrame，不存在返回None
        """
        return self._load_data(self.clean_path / name, format)

    def load_features(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载特征数据

        Args:
            name: 文件名（不含扩展名）
            format: 文件格式

        Returns:
            数据DataFrame，不存在返回None
        """
        return self._load_data(self.features_path / name, format)

    def save_raw_by_date(
        self, df: pd.DataFrame, name: str, trade_date: str, format: str = "parquet"
    ) -> None:
        """保存按日期分区的原始数据

        目录结构: data/raw/{name}/{YYYY-MM-DD}.parquet

        Args:
            df: 数据DataFrame
            name: 数据类型名称（如daily, daily_basic, suspend_d等）
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式，parquet/csv
        """
        # 转换日期格式为YYYY-MM-DD
        date_str = self._format_date(trade_date)

        # 创建分区目录
        partition_path = self.raw_path / name
        partition_path.mkdir(parents=True, exist_ok=True)

        # 保存数据
        self._save_data(df, partition_path / date_str, format)

    def load_raw_by_date(
        self,
        name: str,
        trade_date: str,
        format: str = "parquet",
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """加载按日期分区的原始数据

        Args:
            name: 数据类型名称
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式
            columns: 仅读取指定列（仅 parquet/csv 支持）

        Returns:
            数据DataFrame，不存在返回None
        """
        # 转换日期格式
        date_str = self._format_date(trade_date)

        # 尝试从分区目录加载
        partition_path = self.raw_path / name / date_str
        return self._load_data(partition_path, format, columns=columns)

    def load_raw_by_date_range(
        self,
        name: str,
        start_date: str,
        end_date: str,
        format: str = "parquet",
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """加载日期范围内的原始数据

        Args:
            name: 数据类型名称
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式
            columns: 仅读取指定列（仅 parquet/csv 支持）

        Returns:
            合并后的数据DataFrame，不存在返回None
        """
        partition_dir = self.raw_path / name

        if not partition_dir.exists():
            logger.warning(f"分区目录不存在: {partition_dir}")
            return None

        # 转换日期格式
        start_str = self._format_date(start_date)
        end_str = self._format_date(end_date)

        # 收集所有符合条件的文件 (过滤空 DataFrame 避免 pandas concat FutureWarning)
        dfs = []
        for file_path in sorted(partition_dir.glob(f"*.{format}")):
            date_part = file_path.stem  # 文件名（不含扩展名）
            if start_str <= date_part <= end_str:
                df = self._load_data(partition_dir / date_part, format, columns=columns)
                if df is not None and len(df) > 0:
                    dfs.append(df)

        if not dfs:
            logger.warning(f"没有找到符合日期范围的数据: {name} [{start_date}, {end_date}]")
            return None

        # 合并所有数据
        # 抑制 pandas 1.5+ 在 concat 含 all-NA 列时的 FutureWarning：
        # 占位/空数据日（如龙虎榜无榜单日）保存的是 0 行 DataFrame，
        # 此处虽已按 len(df) > 0 过滤零行，但部分有效分区中某些列整列为 NaN
        # （如龙虎榜的 reason 字段在某些日子全部缺失）仍会触发该警告。
        # 该警告对结果无影响，待 pandas 2.x 行为变更后再统一适配。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*DataFrame concatenation with empty or all-NA entries.*",
            )
            result = pd.concat(dfs, ignore_index=True)
        logger.debug(f"加载了 {len(dfs)} 个分区文件，共 {len(result)} 条记录")
        return result

    def save_clean_by_date(
        self, df: pd.DataFrame, name: str, trade_date: str, format: str = "parquet"
    ) -> None:
        """保存按日期分区的清洗数据

        目录结构: data/clean/{name}/{YYYY-MM-DD}.parquet

        Args:
            df: 数据DataFrame
            name: 数据类型名称（如daily, daily_basic等）
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式，parquet/csv
        """
        # 转换日期格式为YYYY-MM-DD
        date_str = self._format_date(trade_date)

        # 创建分区目录
        partition_path = self.clean_path / name
        partition_path.mkdir(parents=True, exist_ok=True)

        # 保存数据
        self._save_data(df, partition_path / date_str, format)

    def load_clean_by_date(
        self, name: str, trade_date: str, format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """加载按日期分区的清洗数据

        Args:
            name: 数据类型名称
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式

        Returns:
            数据DataFrame，不存在返回None
        """
        # 转换日期格式
        date_str = self._format_date(trade_date)

        # 尝试从分区目录加载
        partition_path = self.clean_path / name / date_str
        return self._load_data(partition_path, format)

    def load_clean_by_date_range(
        self, name: str, start_date: str, end_date: str, format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """加载日期范围内的清洗数据

        Args:
            name: 数据类型名称
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式

        Returns:
            合并后的数据DataFrame，不存在返回None
        """
        partition_dir = self.clean_path / name

        if not partition_dir.exists():
            logger.warning(f"分区目录不存在: {partition_dir}")
            return None

        # 转换日期格式
        start_str = self._format_date(start_date)
        end_str = self._format_date(end_date)

        # 收集所有符合条件的文件 (过滤空 DataFrame 避免 pandas concat FutureWarning)
        dfs = []
        for file_path in sorted(partition_dir.glob(f"*.{format}")):
            date_part = file_path.stem  # 文件名（不含扩展名）
            if start_str <= date_part <= end_str:
                df = self._load_data(partition_dir / date_part, format)
                if df is not None and len(df) > 0:
                    dfs.append(df)

        if not dfs:
            logger.warning(f"没有找到符合日期范围的数据: {name} [{start_date}, {end_date}]")
            return None

        # 合并所有数据
        # 抑制 pandas 1.5+ 在 concat 含 all-NA 列时的 FutureWarning：
        # 占位/空数据日（如龙虎榜无榜单日）保存的是 0 行 DataFrame，
        # 此处虽已按 len(df) > 0 过滤零行，但部分有效分区中某些列整列为 NaN
        # （如龙虎榜的 reason 字段在某些日子全部缺失）仍会触发该警告。
        # 该警告对结果无影响，待 pandas 2.x 行为变更后再统一适配。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*DataFrame concatenation with empty or all-NA entries.*",
            )
            result = pd.concat(dfs, ignore_index=True)
        logger.debug(f"加载了 {len(dfs)} 个分区文件，共 {len(result)} 条记录")
        return result

    def list_partitions(self, layer: str, name: str) -> List[str]:
        """列出某个数据类型的所有分区日期

        Args:
            layer: 数据层，'raw'或'clean'
            name: 数据类型名称

        Returns:
            日期列表（格式YYYY-MM-DD），按升序排序
        """
        if layer == "raw":
            partition_dir = self.raw_path / name
        elif layer == "clean":
            partition_dir = self.clean_path / name
        else:
            raise ValueError(f"不支持的数据层: {layer}")

        if not partition_dir.exists():
            return []

        # 收集所有.parquet文件的日期
        dates = []
        for file_path in partition_dir.glob("*.parquet"):
            date_str = file_path.stem
            # 验证日期格式（YYYY-MM-DD）
            if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                dates.append(date_str)

        # 同时检查.csv文件
        for file_path in partition_dir.glob("*.csv"):
            date_str = file_path.stem
            if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                if date_str not in dates:
                    dates.append(date_str)

        return sorted(dates)

    def migrate_raw_single_file_to_partitions(
        self,
        name: str,
        partition_date_col: str,
        dedup_cols: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """把旧单文件 raw/{name}.parquet 迁移为按日期分区存储 raw/{name}/{date}.parquet。

        express 等公告型数据历史为单文件存储，迁移后各下载/加载路径统一走分区。
        按 partition_date_col 分组写入分区，并与同日期已有分区合并去重
        （支持"部分分区 + 旧单文件"混合态，避免漏读旧数据；同主键冲突时已有分区优先，
        旧单文件不得覆盖新分区数据）；仅当全部记录成功迁移时才删除旧单文件，
        存在无效分区键的记录时保留旧文件待人工处理，防止静默丢数。
        空旧文件或缺少分区列的异常旧文件直接跳过并保留（空文件清理），
        返回 None 由调用方保留已有分区数据，避免异常旧文件遮蔽有效分区。

        Args:
            name: 数据类型名称
            partition_date_col: 分区依据列（如 end_date）
            dedup_cols: 分区内去重列（同 key 保留最后一条）

        Returns:
            迁移合并后的全量分区 DataFrame；旧单文件不存在/为空/缺少分区列时返回 None
        """
        legacy_df = self.load_raw(name)
        if legacy_df is None:
            return None
        if len(legacy_df) == 0:
            # 空旧文件为历史垃圾：清理后返回 None，不影响已有分区数据
            legacy = (self.raw_path / name).with_suffix(".parquet")
            if legacy.exists():
                try:
                    legacy.unlink()
                except OSError as exc:
                    logger.warning(f"[{name}] 删除空旧单文件失败 {legacy}: {exc}")
            return None
        if partition_date_col not in legacy_df.columns:
            logger.warning(f"[{name}] 旧单文件缺少分区列 {partition_date_col}，跳过自动迁移")
            return None

        skipped = 0
        for part_key, part in legacy_df.groupby(partition_date_col, sort=True, dropna=False):
            date_str = _partition_key_to_date(part_key)
            if date_str is None:
                skipped += len(part)
                logger.warning(f"[{name}] {len(part)} 条记录分区键无效 ({part_key!r})，不迁移")
                continue
            existing = self.load_raw_by_date(name, date_str)
            if existing is not None and len(existing) > 0:
                # 混合态合并：已有分区数据放在后面，keep="last" 保证同键冲突时新分区优先
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message=".*DataFrame concatenation with empty or all-NA entries.*",
                    )
                    part = pd.concat([part, existing], ignore_index=True)
            if dedup_cols:
                part = part.drop_duplicates(subset=dedup_cols, keep="last")
            self.save_raw_by_date(part, name, date_str)

        if skipped > 0:
            logger.warning(f"[{name}] {skipped} 条记录分区键无效未迁移，保留旧单文件待人工处理")
        else:
            # 分区全部写入成功后删除旧单文件，避免单文件与分区双份并存
            legacy = (self.raw_path / name).with_suffix(".parquet")
            if legacy.exists():
                try:
                    legacy.unlink()
                except OSError as exc:
                    logger.warning(f"[{name}] 删除旧单文件失败 {legacy}: {exc}")

        result = self._load_raw_all_partitions(name)
        migrated = len(result) if result is not None else 0
        logger.info(f"[{name}] 旧单文件迁移完成: 当前分区全量 {migrated} 条记录")
        return result

    def _load_raw_all_partitions(self, name: str) -> Optional[pd.DataFrame]:
        """读取 raw 层某数据集全部分区并合并（无分区返回 None）。"""
        dfs = []
        for partition in self.list_partitions("raw", name):
            df = self.load_raw_by_date(name, partition)
            if df is not None and len(df) > 0:
                dfs.append(df)
        if not dfs:
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*DataFrame concatenation with empty or all-NA entries.*",
            )
            return pd.concat(dfs, ignore_index=True)

    def _format_date(self, date_str: str) -> str:
        """统一日期格式为YYYY-MM-DD

        Args:
            date_str: 日期字符串，支持YYYYMMDD或YYYY-MM-DD

        Returns:
            格式化后的日期字符串YYYY-MM-DD

        Raises:
            ValueError: 如果日期格式无效
        """
        import re

        if len(date_str) == 8:  # YYYYMMDD
            # 验证格式
            if not re.match(r"^\d{8}$", date_str):
                raise ValueError(f"不支持的日期格式: {date_str}，YYYYMMDD格式应为8位数字")

            # 验证日期有效性（使用 datetime 校验，覆盖如2月30日等无效日期）
            try:
                from datetime import datetime as _dt

                _dt.strptime(date_str, "%Y%m%d")
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            except ValueError:
                raise ValueError(f"无效的日期: {date_str}")

        elif len(date_str) == 10:  # YYYY-MM-DD
            # 验证格式
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                raise ValueError(f"不支持的日期格式: {date_str}，YYYY-MM-DD格式应为YYYY-MM-DD")
            return date_str
        else:
            raise ValueError(f"不支持的日期格式: {date_str}，应为YYYYMMDD或YYYY-MM-DD")

    def save_cs_train_day(
        self,
        df: pd.DataFrame,
        trade_date: str,
        format: str = "parquet",
        has_label: bool = True,
        subdir: str = "cs_train",
    ) -> None:
        """保存单日截面数据

        Args:
            df: 数据DataFrame
            trade_date: 交易日期，格式YYYYMMDD
            format: 文件格式，parquet/csv
            has_label: 是否包含标签列数据(有标签才保存,否则会造成垃圾数据)，默认为True
            subdir: 子目录名称，训练用 "cs_train"（默认），推理用 "cs_infer"
        """
        if has_label:
            target_path = self.features_path / subdir
            target_path.mkdir(parents=True, exist_ok=True)
            self._save_data(df, target_path / trade_date, format)
        else:
            logger.info("保存截面训练数据时未包含标签列，跳过保存操作")

    def load_cs_train_day(
        self, trade_date: str, format: str = "parquet", subdir: str = "cs_train"
    ) -> Optional[pd.DataFrame]:
        """加载单日截面数据

        Args:
            trade_date: 交易日期，格式YYYYMMDD
            format: 文件格式
            subdir: 子目录名称，训练用 "cs_train"（默认），推理用 "cs_infer"

        Returns:
            数据DataFrame，不存在返回None
        """
        target_path = self.features_path / subdir
        return self._load_data(target_path / trade_date, format)

    def check_basic_data_freshness(self, name: str, required_end_date: str) -> bool:
        """检查基础数据（trade_cal或stock_basic）是否足够新

        Args:
            name: 数据名称，'trade_cal'或'stock_basic'
            required_end_date: 需要的结束日期，格式YYYYMMDD

        Returns:
            True表示数据足够新，False表示需要更新
        """
        df = self.load_raw(name)
        if df is None:
            logger.info(f"{name} 数据不存在，需要下载")
            return False

        # 获取数据中的最新日期
        if name == "trade_cal":
            if "cal_date" not in df.columns:
                logger.warning(f"{name} 缺少 cal_date 列")
                return False

            # 转换为字符串格式YYYYMMDD
            try:
                if pd.api.types.is_datetime64_any_dtype(df["cal_date"]):
                    latest_date = df["cal_date"].max().strftime("%Y%m%d")
                else:
                    latest_date = str(df["cal_date"].max()).replace("-", "")
            except Exception as e:
                logger.warning(f"无法解析 {name} 的日期: {e}")
                return False

            logger.info(f"{name} 最新日期: {latest_date}, 需要日期: {required_end_date}")
            return latest_date >= required_end_date

        elif name == "stock_basic":
            # stock_basic不基于日期判断，而是检查是否存在
            # 可以根据数据更新频率（如每季度）来判断是否需要更新
            # 这里简化为：如果文件存在就认为足够新
            logger.info(f"{name} 数据已存在，记录数: {len(df)}")
            return True

        return False

    def is_feature_exists(
        self, trade_date: str, format: str = "parquet", subdir: str = "cs_train"
    ) -> bool:
        """判断特征数据是否存在

        Args:
            trade_date: 交易日期，格式YYYYMMDD
            format: 文件格式
            subdir: 子目录名称，训练用 "cs_train"（默认），推理用 "cs_infer"

        Returns:
            True表示存在，False表示不存在
        """
        target_path = self.features_path / subdir
        path = target_path / trade_date

        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")

        return file_path.exists()

    def count_rows(
        self, layer: str, name: str, date: str, format: str = "parquet"
    ) -> Optional[int]:
        """快速统计分区文件行数（不加载全量数据）。

        用于覆盖度门控：文件存在但行数不足（如历史截断/中断落盘）时识别为未补齐。

        Args:
            layer: 数据层，'raw'或'clean'
            name: 数据类型名称
            date: 交易日期，格式YYYYMMDD
            format: 文件格式

        Returns:
            行数；文件不存在或读取失败返回 None
        """
        if layer == "raw":
            base_path = self.raw_path
        elif layer == "clean":
            base_path = self.clean_path
        else:
            raise ValueError(f"不支持的数据层: {layer}")
        path = base_path / name / self._format_date(date)

        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")

        if not file_path.exists():
            return None
        try:
            if format == "parquet":
                # 优先读取 pyarrow 元数据行数，避免加载全量数据
                import pyarrow.parquet as pq

                return pq.ParquetFile(str(file_path)).metadata.num_rows
            return len(pd.read_csv(file_path))
        except Exception as e:
            logger.warning(f"统计文件行数失败: {file_path}（{e}）")
            return None

    def is_data_exists(
        self,
        layer: str,
        name: str,
        date: str,
        format: str = "parquet",
        min_rows: Optional[int] = None,
    ) -> bool:
        """判断文件是否存在（可选覆盖度门控）

        Args:
            layer: 数据层，'raw'或'clean'
            name: 数据类型名称
            date: 交易日期，格式YYYYMMDD
            format: 文件格式
            min_rows: 若提供，则文件行数须 >= min_rows 才算已补齐
                （防止截断/中断落盘后缺口永久驻留）
        """
        if layer == "raw":
            base_path = self.raw_path
        elif layer == "clean":
            base_path = self.clean_path
        else:
            raise ValueError(f"不支持的数据层: {layer}")
        path = base_path / name / self._format_date(date)

        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")

        if not file_path.exists():
            return False
        if min_rows is not None:
            rows = self.count_rows(layer, name, date, format=format)
            if rows is None or rows < min_rows:
                return False
        return True

    def _save_data(self, df: pd.DataFrame, path: Path, format: str, is_force: bool = False) -> None:
        """保存数据

        Args:
            df: 数据DataFrame
            path: 文件路径（不含扩展名）
            format: 文件格式
        """
        if format == "parquet":
            file_path = path.with_suffix(".parquet")
            tmp_path = file_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(file_path)  # 原子替换，防止写入中断导致文件损坏
        elif format == "csv":
            file_path = path.with_suffix(".csv")
            tmp_path = file_path.with_suffix(".csv.tmp")
            df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            tmp_path.replace(file_path)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.debug(f"数据已保存: {file_path} ({len(df)} 条记录)")

    def _load_data(
        self,
        path: Path,
        format: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """加载数据

        Args:
            path: 文件路径（不含扩展名）
            format: 文件格式
            columns: 仅读取指定列（仅 parquet/csv 支持）

        Returns:
            数据DataFrame，不存在返回None
        """
        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")

        if not file_path.exists():
            logger.debug(f"文件不存在: {file_path}")
            return None

        try:
            if format == "parquet":
                df = pd.read_parquet(file_path, columns=columns)
            else:
                if columns is not None:
                    df = pd.read_csv(file_path, usecols=columns)
                else:
                    df = pd.read_csv(file_path)

            logger.debug(f"数据已加载: {file_path} ({len(df)} 条记录)")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {file_path}, 错误: {str(e)}")
            raise RuntimeError(f"读取数据文件失败: {file_path}") from e
