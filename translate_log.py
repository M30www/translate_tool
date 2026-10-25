#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
translate_log.py —— T0:离线把探针日志里的聊天批量翻成中文,用来评估翻译质量与成本
  用法:
    python translate_log.py                          # 取 logs/ 下最新的 .json,用 echo 跑通
    python translate_log.py logs\\live_xxx.json --engine llm --limit 20
    python translate_log.py --chat-color 00AEE9      # 手动指定频道色(自动检测猜错时)

  为什么先做离线:翻译质量是继 OCR 之后的第二个真实不确定项。用历史日志批量翻,
  既能立刻看清质量,又不用碰实时链路,还不受"漏消息"干扰。
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_text import norm_key                      # noqa: E402
from translate_text import build_translator, usage_report, write_pairs_md   # noqa: E402

for _stream in (sys.stdout,):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load(name: str, rel: str):
    """复用 ocr_probe 里已有的语种判定,不另写一份。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def newest_log() -> str:
    files = sorted(glob.glob(os.path.join("logs", "*.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit("[x] logs/ 下没有 .json,先跑一次 run_live.bat")
    return files[-1]


def hex_to_rgb(color: str) -> list:
    return [int(color[i:i + 2], 16) for i in (1, 3, 5)]


def dominant_color(events: list, tol: float = 40.0) -> str:
    """把颜色聚簇,取成员最多的那一类当"频道聊天色"。

    依据:E8 实测聊天 39 条青、系统提示 10 条黄橙 —— 聊天行数天然占多数。
    """
    groups: list = []
    for event in events:
        color = event.get("col") or ""
        if not color.startswith("#") or len(color) != 7:
            continue
        rgb = hex_to_rgb(color)
        for rep, members in groups:
            if sum(abs(a - b) for a, b in zip(rgb, rep)) / 3 <= tol:
                members.append(event)
                break
        else:
            groups.append((rgb, [event]))
    if not groups:
        return ""
    rep, members = max(groups, key=lambda item: len(item[1]))
    return "#%02X%02X%02X" % tuple(rep)


def select_events(events: list, chat_color: str, guess_lang) -> tuple[list, dict]:
    """挑出"值得翻译的聊天行",并统计每步砍掉了多少。"""
    stats = {"total": len(events), "off_color": 0, "no_text": 0, "chinese": 0,
             "untranslated": 0}
    picked = []
    target = hex_to_rgb(chat_color) if chat_color else None
    for event in events:
        color = event.get("col") or ""
        if target and color.startswith("#"):
            if sum(abs(a - b) for a, b in zip(hex_to_rgb(color), target)) / 3 > 40:
                stats["off_color"] += 1
                continue
        key = norm_key(event["text"])
        if len(key) < 2:
            stats["no_text"] += 1
            continue
        if guess_lang(event["text"]) == "zh":
            stats["chinese"] += 1
            continue
        picked.append(event)
    return picked, stats


def split_prefix(text: str, prefix_re) -> tuple[str, str]:
    """拆出 `[昵称]:` 前缀与正文。

    只把正文送去翻译:昵称是名字,翻了没意义,实测还会把全角冒号搞成 `[Ir1s] ：`。
    前缀原样拼回显示,读者仍然知道是谁在说。
    """
    match = prefix_re.match(text)
    if not match:
        return "", text
    end = match.end()
    # PREFIX 只认到 `]`,后面的冒号要一并吃掉,否则它会被送进翻译(实测译出 `[Ir1s]：`)
    colon = re.match(r"\s*[:：]\s*", text[end:])
    if colon:
        end += colon.end()
    return text[:end], text[end:]


def dedupe(events: list) -> list:
    """同一句话只翻一次;**残片若被完整版包含,只翻完整版**。

    依据:E8 里一条长句会先以 2 行版、再以 3 行版出现,两条都翻等于白花一次调用,
    而且报告里会出现两条几乎一样的译文。
    """
    seen, unique = set(), []
    for event in events:
        key = norm_key(event["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append((key, event))

    keys = [key for key, _event in unique]
    kept = []
    for key, event in unique:
        if any(key != other and len(key) >= 4 and key in other for other in keys):
            continue        # 自己只是别人的一部分 → 丢掉,保留完整版
        kept.append(event)
    return kept


def mark_untranslated(body: str, value: str) -> str:
    """译文与原句几乎一样 = 引擎没翻(常见于英文短句)。

    必须标出来:否则报告里看着"有译文",实际是原文照抄,会误导质量判断。
    """
    from difflib import SequenceMatcher
    if value.strip() and SequenceMatcher(None, norm_key(body), norm_key(value)).ratio() >= 0.9:
        return f"⟨未翻⟩{value}"
    return value


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="离线批量翻译探针日志里的聊天")
    parser.add_argument("path", nargs="?", help="日志 .json(默认取最新)")
    parser.add_argument("--engine", default="free",
                        choices=["echo", "free", "google", "mymemory", "llm"],
                        help="free=自动挑一个能用的免费引擎(默认);echo=回显不联网;llm=OpenAI 兼容")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="免费引擎逐条请求之间的间隔秒数(默认 0.2,防限流)")
    parser.add_argument("--limit", type=int, default=0, help="最多翻多少条(0=不限)")
    parser.add_argument("--batch-size", type=int, default=8, help="每次请求翻几条(默认 8)")
    parser.add_argument("--chat-color", help="频道色 #RRGGBB(默认按出现最多的颜色自动判)")
    parser.add_argument("--no-cache", action="store_true", help="不用缓存")
    parser.add_argument("--out", help="输出 .md 路径(默认 logs/translated_<时间>.md)")
    args = parser.parse_args(argv)

    path = args.path or newest_log()
    payload = json.load(open(path, encoding="utf-8"))
    events = payload.get("events", [])
    if not events:
        raise SystemExit(f"[x] {path} 里没有 events")

    probe = _load("tp_probe", os.path.join("ocr_probe.py"))
    ocr = _load("tp_ocr", os.path.join("explore", "ocr", "ocr_simple.py"))
    chat_color = args.chat_color or dominant_color(events)
    picked, stats = select_events(events, chat_color or "", probe.guess_lang)
    unique = dedupe(picked)
    if args.limit > 0:
        unique = unique[:args.limit]

    print("=" * 78)
    print(f"  离线翻译   {os.path.basename(path)}   引擎={args.engine}")
    print(f"  频道色={chat_color or '(未识别)'}   事件 {stats['total']} 条")
    print(f"  过滤:非频道色 -{stats['off_color']}, 无有效文字 -{stats['no_text']}, "
          f"中文 -{stats['chinese']}  → 候选 {len(picked)} 条 → 去重后 {len(unique)} 条")
    print("=" * 78)

    translator = build_translator(engine=args.engine,
                                  cache_path="" if args.no_cache else None,
                                  delay=args.delay, detect=probe.guess_lang)

    started = time.perf_counter()
    rows = []
    for start in range(0, len(unique), max(1, args.batch_size)):
        chunk = unique[start:start + max(1, args.batch_size)]
        parts = [split_prefix(item["text"], ocr.PREFIX) for item in chunk]
        try:
            translated = translator.translate_batch([body for _pfx, body in parts])
        except Exception as exc:                        # noqa: BLE001
            print(f"  [!] 第 {start // args.batch_size + 1} 批失败: "
                  f"{type(exc).__name__}: {exc}")
            translated = [f"⟨失败⟩{body}" for _pfx, body in parts]
        for event, (prefix, body), value in zip(chunk, parts, translated):
            # echo 是桩,输出必然"像是没翻",别拿它去标 ⟨未翻⟩
            marked = mark_untranslated(body, value) if args.engine != "echo" else value
            if marked.startswith("⟨未翻⟩"):
                stats["untranslated"] = stats.get("untranslated", 0) + 1
            rows.append((event["t"], event["text"], f"{prefix}{marked}".strip()))
        print(f"  … 已翻 {min(start + len(chunk), len(unique))}/{len(unique)}")
    elapsed = time.perf_counter() - started
    if hasattr(translator, "save"):
        translator.save()      # 收尾必须落盘:否则最后一小批(未满 save_every)会丢

    out_path = args.out or os.path.join(
        "logs", f"translated_{time.strftime('%Y%m%d_%H%M%S')}.md")
    write_pairs_md(out_path, f"翻译结果 {os.path.basename(path)}（引擎 {args.engine}）",
                  [(f"{stamp:.1f}s", source, value) for stamp, source, value in rows])

    print("\n" + "=" * 78)
    print(f"  完成 {len(rows)} 条,耗时 {elapsed:.1f}s "
          f"(平均 {elapsed / max(1, len(rows)) * 1000:.0f} ms/条)")
    inner_hits = getattr(translator, "hits", 0)
    print(f"  缓存命中 {inner_hits} 条  |  {usage_report(translator)}")
    if stats.get("untranslated"):
        print(f"  [!] 其中 {stats['untranslated']} 条引擎没翻(译=原文,已标 ⟨未翻⟩)"
              f"—— 免费引擎对英文短句/黑话常这样,不是我们管道的问题")
    print(f"  结果已写入: {os.path.abspath(out_path)}")
    return 0


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
