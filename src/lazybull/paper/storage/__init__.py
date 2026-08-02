"""纸面交易存储模块"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

from typing import TYPE_CHECKING

from ...common.config import get_paper_root
from ...common.trading_config import TradingConfig
from ..models import AccountState, Fill, NAVRecord, PendingBuy, PendingSell, Position, TargetWeight, TradeInstruction

if TYPE_CHECKING:
    from ...common.smb_client import SMBFileReader

from .state import PaperStateMixin
from .config import PaperConfigMixin
from .records import PaperRecordMixin
from .queue import PaperQueueMixin
from .maintenance import PaperMaintenanceMixin

class PaperStorage(
    PaperStateMixin,
    PaperConfigMixin,
    PaperRecordMixin,
    PaperQueueMixin,
    PaperMaintenanceMixin,
):
    """纸面交易存储

    负责持久化和读取纸面交易的各类数据
    """

    def __init__(
        self,
        root_path: Optional[str] = None,
        verbose: bool = False,
        smb_reader: Optional["SMBFileReader"] = None,
    ):
        """初始化纸面交易存储
        
        Args:
            root_path: 数据根目录；未传时默认使用 data.root/paper
            verbose: 是否输出详细日志
            smb_reader: 远端 SMB 读取器；传入后读取走 SMB，写入自动跳过（只读模式）
        """
        self._smb_reader = smb_reader
        self._is_remote = smb_reader is not None
        self.root_path = Path(root_path or get_paper_root())
        self.state_path = self.root_path / "state"
        self.trades_path = self.root_path / "trades"
        self.nav_path = self.root_path / "nav"
        self.runs_path = self.root_path / "runs"
        self.pending_sells_path = self.root_path / "pending_sells"
        self.pending_buys_path = self.root_path / "pending_buys"
        self.instructions_path = self.root_path / "instructions"
        self.verbose = verbose
        
        # 远端只读模式不创建本地目录
        if not self._is_remote:
            for path in [self.state_path, self.trades_path,
                         self.nav_path, self.runs_path, self.pending_sells_path,
                         self.pending_buys_path, self.instructions_path]:
                path.mkdir(parents=True, exist_ok=True)
        if verbose:
            mode = "远端只读" if self._is_remote else "本地"
            logger.info(f"纸面交易存储初始化完成（{mode}），根目录: {self.root_path}")
