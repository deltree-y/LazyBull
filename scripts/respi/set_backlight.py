#!/usr/bin/env python3
"""单独设置树莓派 LCD 背光亮度。"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_SYSFS_ROOT = Path("/sys/class/backlight")
DEFAULT_SYSFS_BRIGHTNESS_PATH = DEFAULT_SYSFS_ROOT / "soc:backlight" / "brightness"
DEFAULT_SYSFS_MAX_PATH = DEFAULT_SYSFS_ROOT / "soc:backlight" / "max_brightness"
DEFAULT_PWM_PIN = 18
DEFAULT_PWM_FREQUENCY = 1000
DEFAULT_LGPIO_CHIP = 0


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


def _discover_sysfs_backlights(backlight_root: Path = DEFAULT_SYSFS_ROOT) -> list[dict]:
    """扫描所有可用的 sysfs 背光节点。"""
    if not backlight_root.exists():
        return []

    devices = []
    for child in sorted(backlight_root.iterdir()):
        if not child.is_dir():
            continue

        brightness_path = child / "brightness"
        max_path = child / "max_brightness"
        if brightness_path.exists() and max_path.exists():
            devices.append(
                {
                    "name": child.name,
                    "brightness_path": brightness_path,
                    "max_path": max_path,
                }
            )
    return devices


def _resolve_sysfs_paths(
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
    backlight_root: Path = DEFAULT_SYSFS_ROOT,
    backlight_name: Optional[str] = None,
) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    """解析实际应使用的 sysfs 背光节点。"""
    if brightness_path.exists() and max_path.exists():
        resolved_name = brightness_path.parent.name if brightness_path.parent == max_path.parent else None
        return brightness_path, max_path, resolved_name

    devices = _discover_sysfs_backlights(backlight_root)
    if not devices:
        return None, None, None

    if backlight_name:
        for device in devices:
            if device["name"] == backlight_name:
                return device["brightness_path"], device["max_path"], device["name"]
        raise RuntimeError(f"未找到名为 {backlight_name} 的背光节点")

    device = devices[0]
    return device["brightness_path"], device["max_path"], device["name"]


def _sysfs_paths_available(
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
    backlight_root: Path = DEFAULT_SYSFS_ROOT,
    backlight_name: Optional[str] = None,
) -> bool:
    """检查 sysfs 背光节点是否可用。"""
    resolved_brightness_path, resolved_max_path, _ = _resolve_sysfs_paths(
        brightness_path=brightness_path,
        max_path=max_path,
        backlight_root=backlight_root,
        backlight_name=backlight_name,
    )
    return resolved_brightness_path is not None and resolved_max_path is not None


def _read_sysfs_backlight(
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
    backlight_root: Path = DEFAULT_SYSFS_ROOT,
    backlight_name: Optional[str] = None,
) -> dict:
    """读取当前 sysfs 背光值。"""
    resolved_brightness_path, resolved_max_path, resolved_name = _resolve_sysfs_paths(
        brightness_path=brightness_path,
        max_path=max_path,
        backlight_root=backlight_root,
        backlight_name=backlight_name,
    )
    if resolved_brightness_path is None or resolved_max_path is None:
        raise RuntimeError("未找到可用的 sysfs 背光节点")

    max_brightness = int(resolved_max_path.read_text(encoding="utf-8").strip())
    if max_brightness <= 0:
        raise RuntimeError(f"max_brightness 非法: {max_brightness}")

    raw_value = int(resolved_brightness_path.read_text(encoding="utf-8").strip())
    percent = int(round(raw_value / max_brightness * 100))
    return {
        "method": "sysfs",
        "backend": "sysfs",
        "backlight_name": resolved_name,
        "percent": percent,
        "raw_value": raw_value,
        "max_brightness": max_brightness,
        "brightness_path": str(resolved_brightness_path),
        "max_path": str(resolved_max_path),
    }


def _set_sysfs_backlight(
    percent: int,
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
    backlight_root: Path = DEFAULT_SYSFS_ROOT,
    backlight_name: Optional[str] = None,
) -> dict:
    """通过 sysfs 设置背光亮度。"""
    resolved_brightness_path, resolved_max_path, resolved_name = _resolve_sysfs_paths(
        brightness_path=brightness_path,
        max_path=max_path,
        backlight_root=backlight_root,
        backlight_name=backlight_name,
    )
    if resolved_brightness_path is None or resolved_max_path is None:
        raise RuntimeError("未找到可用的 sysfs 背光节点")

    max_brightness = int(resolved_max_path.read_text(encoding="utf-8").strip())
    if max_brightness <= 0:
        raise RuntimeError(f"max_brightness 非法: {max_brightness}")

    raw_value = _percent_to_sysfs_value(percent, max_brightness)
    resolved_brightness_path.write_text(f"{raw_value}\n", encoding="utf-8")
    return {
        "method": "sysfs",
        "backend": "sysfs",
        "backlight_name": resolved_name,
        "percent": percent,
        "raw_value": raw_value,
        "max_brightness": max_brightness,
        "brightness_path": str(resolved_brightness_path),
        "max_path": str(resolved_max_path),
    }


def _import_gpio_module():
    """延迟导入 RPi.GPIO，避免非树莓派环境启动失败。"""
    try:
        import RPi.GPIO as gpio_module  # type: ignore
    except ImportError as exc:
        raise RuntimeError("未找到 RPi.GPIO，无法使用 PWM 背光模式") from exc
    return gpio_module


def _import_lgpio_module():
    """延迟导入 lgpio，优先作为 PWM 后端。"""
    try:
        import lgpio  # type: ignore
    except ImportError as exc:
        raise RuntimeError("未找到 lgpio，无法使用 lgpio PWM 背光模式") from exc
    return lgpio


def _set_lgpio_backlight(
    percent: int,
    pin: int = DEFAULT_PWM_PIN,
    frequency: int = DEFAULT_PWM_FREQUENCY,
    gpiochip: int = DEFAULT_LGPIO_CHIP,
    lgpio_module=None,
) -> dict:
    """通过 lgpio 直接设置 GPIO PWM 背光。"""
    lgpio = lgpio_module or _import_lgpio_module()
    handle = lgpio.gpiochip_open(gpiochip)
    try:
        try:
            lgpio.gpio_claim_output(handle, pin, 0)
        except TypeError:
            lgpio.gpio_claim_output(handle, pin)
        lgpio.tx_pwm(handle, pin, frequency, float(percent))
    except Exception:
        try:
            lgpio.gpiochip_close(handle)
        except Exception:
            pass
        raise

    def _cleanup() -> None:
        try:
            try:
                lgpio.tx_pwm(handle, pin, 0, 0)
            except Exception:
                pass
            gpio_free = getattr(lgpio, "gpio_free", None)
            if callable(gpio_free):
                try:
                    gpio_free(handle, pin)
                except Exception:
                    pass
        finally:
            lgpio.gpiochip_close(handle)

    return {
        "method": "pwm",
        "backend": "lgpio",
        "percent": percent,
        "pin": pin,
        "frequency": frequency,
        "gpiochip": gpiochip,
        "handle": handle,
        "lgpio_module": lgpio,
        "cleanup": _cleanup,
    }


def _set_rpi_gpio_backlight(
    percent: int,
    pin: int = DEFAULT_PWM_PIN,
    frequency: int = DEFAULT_PWM_FREQUENCY,
    gpio_module=None,
) -> dict:
    """通过 RPi.GPIO 设置 PWM 背光。"""
    gpio = gpio_module or _import_gpio_module()
    gpio.setwarnings(False)
    gpio.setmode(gpio.BCM)
    gpio.setup(pin, gpio.OUT)
    pwm = gpio.PWM(pin, frequency)
    pwm.start(percent)

    def _cleanup() -> None:
        try:
            pwm.stop()
        finally:
            cleanup = getattr(gpio, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup(pin)
                except TypeError:
                    cleanup()

    return {
        "method": "pwm",
        "backend": "rpi-gpio",
        "percent": percent,
        "pin": pin,
        "frequency": frequency,
        "gpio_module": gpio,
        "pwm": pwm,
        "cleanup": _cleanup,
    }


def update_pwm_backlight_state(state: dict, percent: int) -> dict:
    """更新已初始化的 PWM 背光状态。"""
    backend = state.get("backend")

    if backend == "lgpio":
        lgpio = state["lgpio_module"]
        handle = state["handle"]
        pin = state["pin"]
        frequency = state["frequency"]
        lgpio.tx_pwm(handle, pin, frequency, float(percent))
        state["percent"] = percent
        return state

    if backend == "rpi-gpio":
        pwm = state["pwm"]
        pwm.ChangeDutyCycle(percent)
        state["percent"] = percent
        return state

    raise RuntimeError(f"不支持更新的 PWM 背光后端: {backend}")


def cleanup_backlight_state(state: Optional[dict]) -> None:
    """清理背光状态占用的资源。"""
    if not isinstance(state, dict):
        return

    cleanup = state.get("cleanup")
    if callable(cleanup):
        cleanup()


def _set_pwm_backlight(
    percent: int,
    pin: int = DEFAULT_PWM_PIN,
    frequency: int = DEFAULT_PWM_FREQUENCY,
    gpiochip: int = DEFAULT_LGPIO_CHIP,
    gpio_module=None,
    lgpio_module=None,
) -> dict:
    """按优先级尝试可用的 PWM 背光后端。"""
    errors = []

    try:
        return _set_lgpio_backlight(
            percent,
            pin=pin,
            frequency=frequency,
            gpiochip=gpiochip,
            lgpio_module=lgpio_module,
        )
    except Exception as exc:
        errors.append(f"lgpio: {type(exc).__name__}: {exc}")

    try:
        return _set_rpi_gpio_backlight(
            percent,
            pin=pin,
            frequency=frequency,
            gpio_module=gpio_module,
        )
    except Exception as exc:
        errors.append(f"RPi.GPIO: {type(exc).__name__}: {exc}")

    raise RuntimeError("所有 PWM 背光方式都失败: " + " | ".join(errors))


def set_backlight(
    percent: int,
    method: str = "auto",
    brightness_path: Path = DEFAULT_SYSFS_BRIGHTNESS_PATH,
    max_path: Path = DEFAULT_SYSFS_MAX_PATH,
    backlight_root: Path = DEFAULT_SYSFS_ROOT,
    backlight_name: Optional[str] = None,
    pin: int = DEFAULT_PWM_PIN,
    frequency: int = DEFAULT_PWM_FREQUENCY,
    gpiochip: int = DEFAULT_LGPIO_CHIP,
    gpio_module=None,
    lgpio_module=None,
) -> dict:
    """根据指定方式设置背光亮度。"""
    resolved_method = method
    if method == "auto":
        resolved_method = (
            "sysfs"
            if _sysfs_paths_available(
                brightness_path,
                max_path,
                backlight_root=backlight_root,
                backlight_name=backlight_name,
            )
            else "pwm"
        )

    if resolved_method == "sysfs":
        return _set_sysfs_backlight(
            percent,
            brightness_path=brightness_path,
            max_path=max_path,
            backlight_root=backlight_root,
            backlight_name=backlight_name,
        )

    if resolved_method == "pwm":
        return _set_pwm_backlight(
            percent,
            pin=pin,
            frequency=frequency,
            gpiochip=gpiochip,
            gpio_module=gpio_module,
            lgpio_module=lgpio_module,
        )

    if resolved_method == "lgpio":
        return _set_lgpio_backlight(
            percent,
            pin=pin,
            frequency=frequency,
            gpiochip=gpiochip,
            lgpio_module=lgpio_module,
        )

    if resolved_method == "rpi-gpio":
        return _set_rpi_gpio_backlight(
            percent,
            pin=pin,
            frequency=frequency,
            gpio_module=gpio_module,
        )

    raise ValueError(f"不支持的背光设置方式: {resolved_method}")


def _hold_pwm_session(result: dict) -> None:
    """保持 PWM 会话存活，便于现场试亮度。"""
    cleanup = result.get("cleanup")
    backend = result.get("backend", "pwm")

    print(f"PWM 模式正在通过 {backend} 保持亮度，按 Ctrl+C 结束本次亮度测试")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("已结束 PWM 背光测试")
    finally:
        if callable(cleanup):
            cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="单独调节树莓派 LCD 背光亮度。会自动扫描 sysfs 背光节点，失败时再尝试 PWM。",
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
        "--list",
        action="store_true",
        help="列出当前检测到的 sysfs 背光节点",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "sysfs", "pwm", "lgpio", "rpi-gpio"],
        default="auto",
        help="背光控制方式，默认 auto；pwm 表示自动尝试 lgpio 后回退 RPi.GPIO",
    )
    parser.add_argument(
        "--backlight-root",
        default=str(DEFAULT_SYSFS_ROOT),
        help="sysfs 背光根目录，默认 /sys/class/backlight",
    )
    parser.add_argument(
        "--backlight-name",
        default=None,
        help="指定要使用的背光节点名称，如 rpi_backlight",
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
        "--gpiochip",
        type=int,
        default=DEFAULT_LGPIO_CHIP,
        help=f"lgpio 使用的 gpiochip 编号，默认 {DEFAULT_LGPIO_CHIP}",
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


def _print_available_backlights(backlight_root: Path) -> None:
    """打印当前检测到的 sysfs 背光节点。"""
    devices = _discover_sysfs_backlights(backlight_root)
    if not devices:
        print(f"未在 {backlight_root} 下发现可用的 sysfs 背光节点")
        return

    print(f"在 {backlight_root} 下发现以下背光节点:")
    for device in devices:
        print(f"- {device['name']}")
        print(f"  brightness: {device['brightness_path']}")
        print(f"  max       : {device['max_path']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    brightness_path = Path(args.sysfs_brightness_path)
    max_path = Path(args.sysfs_max_path)
    backlight_root = Path(args.backlight_root)

    if args.list:
        _print_available_backlights(backlight_root)
        return 0

    if args.read:
        try:
            result = _read_sysfs_backlight(
                brightness_path=brightness_path,
                max_path=max_path,
                backlight_root=backlight_root,
                backlight_name=args.backlight_name,
            )
        except Exception as exc:
            print(f"读取背光失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            _print_available_backlights(backlight_root)
            return 1

        print(
            f"当前背光: {result['percent']}% (raw {result['raw_value']}/{result['max_brightness']})"
        )
        if result.get("backlight_name"):
            print(f"背光节点: {result['backlight_name']}")
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
            backlight_root=backlight_root,
            backlight_name=args.backlight_name,
            pin=args.pin,
            frequency=args.frequency,
            gpiochip=args.gpiochip,
        )
    except Exception as exc:
        print(f"设置背光失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.method in ("auto", "sysfs"):
            _print_available_backlights(backlight_root)
        return 1

    if result["method"] == "sysfs":
        print(
            f"已通过 sysfs 设置背光为 {result['percent']}% "
            f"(raw {result['raw_value']}/{result['max_brightness']})"
        )
        if result.get("backlight_name"):
            print(f"背光节点: {result['backlight_name']}")
        print(f"写入节点: {result['brightness_path']}")
        return 0

    print(
        f"已通过 PWM 设置背光为 {result['percent']}% "
        f"(backend {result['backend']}, GPIO {result['pin']}, {result['frequency']}Hz)"
    )
    _hold_pwm_session(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())