# -*- coding: utf-8 -*-
"""PaperMaintenanceMixin：src/lazybull/paper/storage.py 拆分出的 reset_t0, truncate_since。"""

from ..models import AccountState
from loguru import logger
from typing import Optional
import json
import pandas as pd
import shutil

class PaperMaintenanceMixin:
    def reset_t0(self, t0_date: Optional[str] = None) -> dict:
        """重置纸面交易，清空所有交易数据恢复为新账户状态

        清空账户状态、成交记录、净值、运行记录、交易指令、延迟订单等，
        仅保留 config.yaml 配置文件。账户现金重置为 config 中的 initial_capital。

        Args:
            t0_date: 仅用于日志显示（可选，默认自动查找最新）

        Returns:
            操作结果统计字典
        """
        if t0_date is None:
            t0_date = self.find_latest_t0()

        stats = {'t0_date': t0_date}

        # 读取配置以获取初始资金
        config = self.load_config()
        initial_capital = config.get('initial_capital', 500000.0) if config else 500000.0

        # 清空各子目录下的所有文件
        dirs_to_clean = [
            self.state_path,
            self.trades_path,
            self.nav_path,
            self.runs_path,
            self.pending_sells_path,
            self.pending_buys_path,
            self.instructions_path,
        ]
        for dir_path in dirs_to_clean:
            for entry in dir_path.iterdir():
                if entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
            logger.info(f"已清空: {dir_path.name}/")

        # 重建空账户状态
        new_state = AccountState(
            cash=initial_capital,
            positions={},
            last_update="",
        )
        self.save_account_state(new_state)
        logger.info(f"已重建账户状态，初始资金: {initial_capital:,.2f}")

        return stats

    def truncate_since(self, cut_off_date: str) -> None:
        """截断/清理从指定日期开始的所有数据（包含该日期）
        
        用于手工修正账户后，清理 cut-off 日期及之后的所有记录，
        以便从该日期重新运行并保持一致性。
        
        清理范围：
        - trades.parquet: 删除 trade_date >= cut_off_date 的行
        - nav.parquet: 删除 trade_date >= cut_off_date 的行
        - runs/: 删除日期 >= cut_off_date 的 t0_*.json 和 t1_*.json 文件
        - instructions/: 删除日期 >= cut_off_date 的指令文件
        - pending_buys.json 和 pending_sells.json: 清空
        - rebalance_state.json: 按规则回滚
        
        Args:
            cut_off_date: 截断日期 YYYYMMDD（包含此日期）
        """
        logger.info("=" * 80)
        logger.info(f"开始清理数据：删除 >= {cut_off_date} 的所有记录")
        logger.info("=" * 80)
        
        # 1. 清理 trades.parquet
        trades_file = self.trades_path / "trades.parquet"
        if trades_file.exists():
            df = pd.read_parquet(trades_file)
            original_count = len(df)
            df = df[df['trade_date'] < cut_off_date]
            new_count = len(df)

            if new_count < original_count:
                df.to_parquet(trades_file, index=False)
                logger.info(f"清理成交记录: {original_count} -> {new_count} 条（删除 {original_count - new_count} 条）")
            else:
                logger.info(f"成交记录无需清理（无 >= {cut_off_date} 的记录）")
        else:
            logger.info("成交记录文件不存在，跳过")
        
        # 2. 清理 nav.parquet
        nav_file = self.nav_path / "nav.parquet"
        if nav_file.exists():
            df = pd.read_parquet(nav_file)
            original_count = len(df)
            df = df[df['trade_date'] < cut_off_date]
            new_count = len(df)
            
            if new_count < original_count:
                df.to_parquet(nav_file, index=False)
                logger.info(f"清理净值记录: {original_count} -> {new_count} 条（删除 {original_count - new_count} 条）")
            else:
                logger.info(f"净值记录无需清理（无 >= {cut_off_date} 的记录）")
        else:
            logger.info("净值记录文件不存在，跳过")
        
        # 3. 清理 runs/ 目录
        deleted_runs = 0
        for run_file in self.runs_path.glob("*.json"):
            if run_file.name == "rebalance_state.json":
                continue  # rebalance_state 单独处理
            
            # 提取日期：t0_YYYYMMDD.json 或 t1_YYYYMMDD.json
            parts = run_file.stem.split('_')
            if len(parts) == 2 and parts[0] in ['t0', 't1']:
                file_date = parts[1]
                if file_date >= cut_off_date:
                    run_file.unlink()
                    deleted_runs += 1
        
        if deleted_runs > 0:
            logger.info(f"清理运行记录: 删除 {deleted_runs} 个文件")
        else:
            logger.info("运行记录无需清理")
        
        # 4. 清理 instructions/ 目录
        deleted_instructions = 0
        for inst_file in self.instructions_path.glob("*.parquet"):
            # 提取日期：YYYYMMDD.parquet
            file_date = inst_file.stem
            if file_date >= cut_off_date:
                inst_file.unlink()
                deleted_instructions += 1
        
        if deleted_instructions > 0:
            logger.info(f"清理交易指令: 删除 {deleted_instructions} 个文件")
        else:
            logger.info("交易指令无需清理")
        
        # 5. 清空 pending_buys.json
        pending_buys_file = self.pending_buys_path / "pending_buys.json"
        if pending_buys_file.exists():
            with open(pending_buys_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info("清空延迟买入队列")
        else:
            logger.info("延迟买入队列文件不存在，跳过")
        
        # 6. 清空 pending_sells.json
        pending_sells_file = self.pending_sells_path / "pending_sells.json"
        if pending_sells_file.exists():
            with open(pending_sells_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info("清空延迟卖出队列")
        else:
            logger.info("延迟卖出队列文件不存在，跳过")

        # 6.5 清空策略状态
        strategy_state_file = self.state_path / "strategy_state.json"
        if strategy_state_file.exists():
            strategy_state_file.unlink()
            logger.info("清空策略状态")
        else:
            logger.info("策略状态文件不存在，跳过")
        
        # 7. 回滚 rebalance_state.json
        rebalance_state = self.load_rebalance_state()
        if rebalance_state and rebalance_state.get('last_rebalance_date', '') >= cut_off_date:
            # 需要回滚：找到 cut_off 之前最近的 t0 记录
            t0_files = sorted([f for f in self.runs_path.glob("t0_*.json")])
            rollback_date = None
            
            for t0_file in reversed(t0_files):
                file_date = t0_file.stem.split('_')[1]
                if file_date < cut_off_date:
                    rollback_date = file_date
                    break
            
            if rollback_date:
                rebalance_state['last_rebalance_date'] = rollback_date
                self.save_rebalance_state(rebalance_state)
                logger.info(f"回滚调仓状态: {rebalance_state.get('last_rebalance_date')} -> {rollback_date}")
            else:
                # cut_off 之前没有 t0 记录，删除 rebalance_state
                rebalance_file = self.runs_path / "rebalance_state.json"
                if rebalance_file.exists():
                    rebalance_file.unlink()
                logger.info("删除调仓状态（无有效的 t0 记录可回滚）")
        else:
            logger.info("调仓状态无需回滚")
        
        logger.info("=" * 80)
        logger.info("数据清理完成")
        logger.info("=" * 80)
