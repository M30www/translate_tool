#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chat_text.py —— 聊天文本的归一化判据(probe_live.py 与 analyze_log.py 共用)
  目的:判断"这条消息是不是新的"时,要能穿透 OCR 抖动。
  抖动的主源在行首前缀(频道名+昵称),正文相对稳定,所以剥前缀后去空白标点再比。
  为什么单独成模块:两个脚本都要用,各写一份必然漂移(注释见 docs/dev-log.md D5)。
"""

from __future__ import annotations

import re

COLON = re.compile(r"^.{0,24}?[:：]")          # 行首到第一个冒号(含全角)视为前缀


def norm_key(text: str) -> str:
    """归一化去重键。`[r1s]:ㅋㅋㅋ` 与 `※[Ir1s]：ㅋㅋㅋ` 得到同一个键。

    只保留字母数字:中文/韩文/日文在 Python 里都算 alnum,所以一次就能剥掉
    空白、标点、※·▲[] 这类 OCR 抖动最爱乱塞的符号(手列符号表必然漏)。
    """
    match = COLON.match(text)
    body = text[match.end():] if match else text
    body = "".join(ch for ch in body.lower() if ch.isalnum())
    return _collapse_repeats(body)


def _collapse_repeats(body: str) -> str:
    """`5초만기다려5초만기다려5초만기다려` → `5초만기다려`。

    检测框偶尔会把同一条消息重复读几遍拼成一行(实测出现过),它们是同一条消息。
    """
    for size in range(1, len(body) // 2 + 1):
        if len(body) % size == 0:
            unit = body[:size]
            if unit * (len(body) // size) == body:
                return unit
    return body
