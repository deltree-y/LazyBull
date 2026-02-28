# -*- coding: utf-8 -*-
import time
import socket
import gpiod
import random
import threading
import psutil
from threading import Lock
from gpiod.line import Direction, Value

try:
    from fonts import F8X16, F6X8
except ImportError:
    exit(1)

CMD, DATA = 0, 1

class OledDevice:
    def __init__(self):
        self.PINS = {'dc': 20, 'din': 10, 'clk': 11, 'rst': 21, 'ce': 8}
        self._lock = Lock()
        self.buffer = bytearray(1024)
        try:
            self.chip = gpiod.Chip('/dev/gpiochip0')
            self.request = self.chip.request_lines(
                config={tuple(self.PINS.values()): gpiod.LineSettings(direction=Direction.OUTPUT)}
            )
        except: exit(1)
        self.oled_init()

    def set_pin(self, pin_name, value):
        self.request.set_value(self.PINS[pin_name], Value(1 if value else 0))

    def write_byte(self, data, is_cmd):
        self.set_pin('dc', 0 if is_cmd == CMD else 1)
        self.set_pin('ce', 0)
        for i in range(8):
            self.set_pin('clk', 0)
            self.set_pin('din', (data >> (7 - i)) & 1)
            self.set_pin('clk', 1)
        self.set_pin('ce', 1)

    def oled_init(self):
        self.set_pin('rst', 1); time.sleep(0.05)
        self.set_pin('rst', 0); time.sleep(0.1)
        self.set_pin('rst', 1); time.sleep(0.05)
        init_seq = [0xAE, 0x00, 0x10, 0x40, 0x81, 0xCF, 0xA1, 0xC8, 0xA6, 0xA8, 0x3F, 0xD3, 0x00, 0xD5, 0x80, 0xD9, 0xF1, 0xDA, 0x12, 0xDB, 0x40, 0x20, 0x02, 0x8D, 0x14, 0xA4, 0xA6, 0xAF]
        for cmd in init_seq: self.write_byte(cmd, CMD)
        self.clear_buffer(); self.refresh()

    def clear_buffer(self):
        for i in range(1024): self.buffer[i] = 0

    def refresh(self):
        with self._lock:
            for page in range(8):
                self.write_byte(0xb0 + page, CMD)
                self.write_byte(0x00, CMD); self.write_byte(0x10, CMD)
                for x in range(128): self.write_byte(self.buffer[page * 128 + x], DATA)

    def draw_8x16(self, x, y_page, text):
        for i, char in enumerate(text):
            offset = (ord(char) - ord(' ')) * 16
            if offset < 0: continue
            for col in range(8):
                tx = x + i * 8 + col
                if 0 <= tx < 128:
                    self.buffer[y_page * 128 + tx] = F8X16[offset + col]
                    self.buffer[(y_page + 1) * 128 + tx] = F8X16[offset + col + 8]

    def draw_6x8(self, x, y_page, text):
        for i, char in enumerate(text):
            code = ord(char) - ord(' ')
            if code < 0 or (code + 1) * 6 > len(F6X8):
                continue
            offset = code * 6
            for col in range(6):
                tx = x + i * 6 + col
                if 0 <= tx < 128:
                    self.buffer[y_page * 128 + tx] = F6X8[offset + col]

    def draw_16x32(self, x, y_page, text):
        for i, char in enumerate(text):
            offset = (ord(char) - ord(' ')) * 16
            raw = F8X16[offset : offset + 16]
            for col in range(8):
                def double_bits(b):
                    res = 0
                    for bit in range(4):
                        if b & (1 << bit): res |= (3 << (bit * 2))
                    return res
                p_bytes = [double_bits(raw[col]&0xF), double_bits((raw[col]>>4)&0xF), double_bits(raw[col+8]&0xF), double_bits((raw[col+8]>>4)&0xF)]
                for tc in range(2):
                    tx = x + i * 16 + col * 2 + tc
                    if 0 <= tx < 128:
                        for po in range(4):
                            if 0 <= y_page + po < 8: self.buffer[(y_page+po)*128 + tx] = p_bytes[po]

    def close(self):
        self.clear_buffer(); self.refresh(); self.write_byte(0xAE, CMD)

def get_sys_status():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000.0
        cpu = psutil.cpu_percent(interval=None)
        return f"{int(cpu)}%/{temp:.1f}C"
    except:
        return "0%/0.0C"

def time_worker(oled):
    last_min = -1
    by = 3
    psutil.cpu_percent(interval=None)
    is_screen_on = True

    while True:
        now = time.localtime()
        hour = now.tm_hour

        # --- 自动息屏逻辑：23点到早上6点 ---
        if 23 <= hour or hour < 6:
            if is_screen_on:
                oled.write_byte(0xAE, CMD) # 关闭显示指令
                is_screen_on = False
            time.sleep(10) # 息屏期间减少检测频率，节省资源
            continue
        else:
            if not is_screen_on:
                oled.write_byte(0xAF, CMD) # 开启显示指令
                is_screen_on = True

        # --- 正常运行逻辑 ---
        status_str = get_sys_status()

        if now.tm_min != last_min:
            by = random.choice([2, 3, 4])
            last_min = now.tm_min
        
        oled.clear_buffer()
        start_x = (128 - len(status_str) * 8) // 2
        oled.draw_8x16(start_x, 0, status_str)
        oled.draw_16x32(0, by, time.strftime("%H:%M:%S", now))
        
        oled.refresh()
        time.sleep(1)

if __name__ == "__main__":
    dev = OledDevice()
    t = threading.Thread(target=time_worker, args=(dev,), daemon=True)
    t.start()
    try:
        while True: time.sleep(1)
    except:
        dev.close()