#!/usr/bin/env python3
"""单独设置树莓派 LCD 背光亮度。"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_SYSFS_BRIGHTNESS_PATH = Path("/sys/class/backlight/soc:backlight/brightness")
DEFAULT_SYSFS_MAX_PATH = Path("/sys/class/backlight/soc:backlight/max_brightness")
DEFAULT_PWM_PIN = 18
DEFAULT_PWM_FREQUENCY = 1000


def _brightness_arg(value: str) -> int:
    """解析命令行亮度百分比。"""
    try:
        brightness = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("亮度必须是 0~100 的整数") from exc

    if brightness < 0 or brightness > 100:
        raise argparse.ArgumentTypeError("亮度必须是 0~100 的整数")
    return brightness


def _percent_to_sysfs_value(percent: int, max_brightness: int) -> int:
    """将百分比亮度换算为 sysfs 原始值。"""
    return int(max_brightness * percent / 100)


def _sysfs_paths_available(
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
) -> bool:
    """检查 sysfs 背光节点是否可用。"""
    return brightness_path.exists() and max_path.exists()


def _read_sysfs_backlight(
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
) -> dict:
    """读取当前 sysfs 背光值。"""
    max_brightness = int(max_path.read_text(encoding="utf-8").strip())
    if max_brightness <= 0:
        raise RuntimeError(f"max_brightness 非法: {max_brightness}")

    raw_value = int(brightness_path.read_text(encoding="utf-8").strip())
    percent = int(round(raw_value / max_brightness * 100))
    return {
        "method": "sysfs",
        "percent": percent,
        "raw_value": raw_value,
        "max_brightness": max_brightness,
        "brightness_path": str(brightness_path),
        "max_path": str(max_path),
    }


def _set_sysfs_backlight(
    percent: int,
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
) -> dict:
    """通过 sysfs 设置背光亮度。"""
    max_brightness = int(max_path.read_text(encoding="utf-8").strip())
    if max_brightness <= 0:
        raise RuntimeError(f"max_brightness 非法: {max_brightness}")

    raw_value = _percent_to_sysfs_value(percent, max_brightness)
    brightness_path.write_text(f"{raw_value}\n", encoding="utf-8")
    return {
        "method": "sysfs",
        "percent": percent,
        "raw_value": raw_value,
        "max_brightness": max_brightness,
        "brightness_path": str(brightness_path),
        "max_path": str(max_path),
    }


def _import_gpio_module():
    """延迟导入 RPi.GPIO，避免非树莓派环境启动失败。"""
    try:
        import RPi.GPIO as gpio_module  # type: ignore
    except ImportError as exc:
        raise RuntimeError("未找到 RPi.GPIO，无法使用 PWM 背光模式") from exc
    return gpio_module


def _set_pwm_backlight(
    percent: int,
    pin: int = DEFAULT_PWM_PIN,
    frequency: int = DEFAULT_PWM_FREQUENCY,
    gpio_module=None,
) -> dict:
    """通过 GPIO PWM 设置背光亮度。"""
    gpio = gpio_module or _import_gpio_module()
    gpio.setwarnings(False)
    gpio.setmode(gpio.BCM)
    gpio.setup(pin, gpio.OUT)
    pwm = gpio.PWM(pin, frequency)
    pwm.start(percent)
    return {
        "method": "pwm",
        "percent": percent,
        "pin": pin,
        "frequency": frequency,
        "gpio_module": gpio,
        "pwm": pwm,
    }


def set_backlight(
    percent: int,
    method: str = "auto",
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
    pin: int = DEFAULT_PWM_PIN,
    frequency: int = DEFAULT_PWM_FREQUENCY,
    gpio_module=None,
) -> dict:
    """根据指定方式设置背光亮度。"""
    resolved_method = method
    if method == "auto":
        resolved_method = "sysfs" if _sysfs_paths_available(brightness_path, max_path) else "pwm"

    if resolved_method == "sysfs":
        return _set_sysfs_backlight(percent, brightness_path=brightness_path, max_path=max_path)

    if resolved_method == "pwm":
        return _set_pwm_backlight(percent, pin=pin, frequency=frequency, gpio_module=gpio_module)

    raise ValueError(f"不支持的背光设置方式: {resolved_method}")


def _hold_pwm_session(result: dict) -> None:
    """保持 PWM 会话存活，便于现场试亮度。"""
    pwm = result["pwm"]
    gpio = result["gpio_module"]
    pin = result["pin"]

    print("PWM 模式需要保持进程运行，按 Ctrl+C 结束本次亮度测试")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("已结束 PWM 背光测试")
    finally:
        try:
            pwm.stop()
        finally:
            cleanup = getattr(gpio, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup(pin)
                except TypeError:
                    cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="单独调节树莓派 LCD 背光亮度。优先使用 sysfs，失败时可切到 GPIO PWM。",
    )
    parser.add_argument(
        "brightness",
        nargs="?",
        type=_brightness_arg,
        help="目标背光亮度百分比，范围 0~100",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="只读取当前 sysfs 背光值，不写入",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "sysfs", "pwm"],
        default="auto",
        help="背光控制方式，默认 auto",
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=DEFAULT_PWM_PIN,
        help=f"PWM 模式使用的 GPIO 引脚，默认 {DEFAULT_PWM_PIN}",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=DEFAULT_PWM_FREQUENCY,
        help=f"PWM 频率，默认 {DEFAULT_PWM_FREQUENCY}Hz",
    )
    parser.add_argument(
        "--sysfs-brightness-path",
        default=str(DEFAULT_SYSFS_BRIGHTNESS_PATH),
        help="sysfs brightness 节点路径",
    )
    parser.add_argument(
        "--sysfs-max-path",
        default=str(DEFAULT_SYSFS_MAX_PATH),
        help="sysfs max_brightness 节点路径",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    brightness_path = Path(args.sysfs_brightness_path)
    max_path = Path(args.sysfs_max_path)

    if args.read:
        if not _sysfs_paths_available(brightness_path, max_path):
            print("未找到 sysfs 背光节点，当前环境无法直接读取持久背光值", file=sys.stderr)
            return 1
        try:
            result = _read_sysfs_backlight(brightness_path=brightness_path, max_path=max_path)
        except Exception as exc:
            print(f"读取背光失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        print(
            f"当前背光: {result['percent']}% (raw {result['raw_value']}/{result['max_brightness']})"
        )
        print(f"brightness 节点: {result['brightness_path']}")
        print(f"max 节点: {result['max_path']}")
        return 0

    if args.brightness is None:
        parser.error("请提供亮度百分比，或使用 --read 查看当前背光")

    try:
        result = set_backlight(
            args.brightness,
            method=args.method,
            brightness_path=brightness_path,
            max_path=max_path,
            pin=args.pin,
            frequency=args.frequency,
        )
    except Exception as exc:
        print(f"设置背光失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if result["method"] == "sysfs":
        print(
            f"已通过 sysfs 设置背光为 {result['percent']}% "
            f"(raw {result['raw_value']}/{result['max_brightness']})"
        )
        print(f"写入节点: {result['brightness_path']}")
        return 0

    print(
        f"已通过 PWM 设置背光为 {result['percent']}% "
        f"(GPIO {result['pin']}, {result['frequency']}Hz)"
    )
    _hold_pwm_session(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())