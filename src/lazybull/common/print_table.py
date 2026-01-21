# 名称: print_table.py
# 说明: 按显示宽度对齐打印表格（使用 wcwidth）

from wcwidth import wcswidth
import logging

logger = logging.getLogger(__name__)

def display_width(s: str) -> int:
    """返回字符串在终端的显示宽度（使用 wcwidth）。"""
    return wcswidth(s)

def pad(s: str, width: int, align: str = 'left') -> str:
    """按显示宽度填充字符串。
    align: 'left'|'right'|'center'
    """
    s = '' if s is None else str(s)
    w = display_width(s)
    if w >= width:
        return s
    pad_len = width - w
    if align == 'left':
        return s + ' ' * pad_len
    if align == 'right':
        return ' ' * pad_len + s
    # center
    left = pad_len // 2
    right = pad_len - left
    return ' ' * left + s + ' ' * right

def format_row(values, widths, aligns):
    """按列宽与对齐方式格式化一行并返回字符串。"""
    parts = [pad(v, w, a) for v, w, a in zip(values, widths, aligns)]
    return ' '.join(parts)
