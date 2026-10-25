#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_log.py —— 读 probe_live.py 产出的 .json(或 .log),输出一份可粘贴的判读报告
  用法:python analyze_log.py logs\live_20261025_135021.json
  给出:性能摘要 / 事件时间线 / 假新行候选 / 低质量行 / 疑似漏行窗口 / 结论建议。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_text import norm_key   # noqa: E402  (与 probe_live.py 同一套判据)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SIMILAR = 0.80        # 两条文本相似度达到它,就怀疑是同一行的识别抖动
LOW_SCORE = 0.60      # 低于它算低质量行
GAP_FACTOR = 3.0      # 事件间隔超过"中位间隔的 N 倍"就算异常空窗
TIMELINE_LIMIT = 40   # 时间线最多打印多少条


def unify(events: List[dict]) -> List[List[dict]]:
    """按归一化键把同一消息的所有抖动变体收回一组。"""
    buckets: dict = {}
    for event in events:
        buckets.setdefault(norm_key(event["text"]), []).append(event)
    return list(buckets.values())


def load_events(path: str) -> Tuple[dict, List[dict]]:
    """返回 (summary, events)。.json 直接读;.log 用正则退化解析。"""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = dict(payload.get("summary", {}))
        summary["host"] = payload.get("host", "?")   # host/source 在 payload 顶层
        summary["source"] = payload.get("source", "")
        return summary, payload.get("events", [])

    events: List[dict] = []
    stamp = None
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            head = re.search(r"\[\s*([\d.]+)s\]\s*新行\s*\d+", line)
            if head:
                stamp = float(head.group(1))
                continue
            row = re.match(r"\s+(ko5|v6)\s+([\d.]+)\s+(.*)$", line.rstrip("\n"))
            if row and stamp is not None:
                events.append({"t": stamp, "src": row.group(1),
                               "score": float(row.group(2)), "text": row.group(3)})
    return {}, events


def is_low_quality(text: str, score: float) -> str:
    if score < LOW_SCORE:
        return f"置信度低({score:.2f})"
    if len(text.strip()) < 2:
        return "有效字符不足 2 个"
    if not any(ch.isalnum() for ch in text):
        return "纯符号"
    return ""


def group_similar(events: List[dict]) -> List[List[dict]]:
    """把疑似同一行的抖动变体聚成一簇(贪心,够用)。"""
    groups: List[List[dict]] = []
    for event in events:
        for group in groups:
            if SequenceMatcher(None, group[0]["text"], event["text"]).ratio() >= SIMILAR:
                group.append(event)
                break
        else:
            groups.append([event])
    return groups


def report(summary: dict, events: List[dict], path: str) -> None:
    print("=" * 78)
    print(f"  判读报告   {os.path.basename(path)}")
    print("=" * 78)
    if summary:
        print(f"  主机 {summary.get('host', '?')}   运行 {summary.get('wall_s')}s   "
              f"帧 {summary.get('frames')}   {summary.get('fps')} fps")
        print(f"  跳过率 {summary.get('skip_rate')}%   OCR {summary.get('ocr_runs')} 次   "
              f"单次 p50/p95 {summary.get('ocr_ms_p50')}/{summary.get('ocr_ms_p95')} ms")
        print(f"  端到端 p50/p95 {summary.get('e2e_ms_p50')}/{summary.get('e2e_ms_p95')} ms   "
              f"累计行 {summary.get('lines_total')}   新行 {summary.get('new_lines')}")

    if not events:
        print("\n  [!] 没有解析到任何『新行』事件 —— 要么全程静帧,要么阈值/模型不对。")
        return

    print(f"\n  ── 事件时间线(前 {min(len(events), TIMELINE_LIMIT)} 条 / 共 {len(events)} 条)──")
    last = 0.0
    for event in events[:TIMELINE_LIMIT]:
        gap = event["t"] - last
        mark = "  " if gap < 5 else " +"
        print(f"  {mark}[{event['t']:7.1f}s +{gap:5.1f}s] {event['src']:<4}"
              f" {event['score']:5.3f}  {event['text']}")
        last = event["t"]
    if len(events) > TIMELINE_LIMIT:
        print(f"  … 其余 {len(events) - TIMELINE_LIMIT} 条略")

    buckets = unify(events)
    print(f"\n  ── 按正文归一化后的真实消息 {len(buckets)} 条"
          f"(原始新行 {len(events)} 条,多余重复 {len(events) - len(buckets)} 条)──")
    for group in sorted(buckets, key=lambda g: -len(g))[:12]:
        best = max(group, key=lambda e: e["score"])   # 用置信度最高的那条代表整组
        label = best["text"] if norm_key(best["text"]) else "(无有效文字:纯符号/空行)"
        print(f"    ×{len(group):<3} {group[0]['t']:7.1f}~{group[-1]['t']:7.1f}s"
              f"  {best['src']:<4} {label}")
    if len(buckets) > 12:
        print(f"    … 其余 {len(buckets) - 12} 条略")

    runs = summary.get("ocr_runs")
    if runs:
        produced = len({e["t"] for e in events})
        print(f"\n  ── OCR 产出率 {produced}/{runs} 次触发生成了新行"
              f"({produced / runs * 100:.0f}% 有效,其余是画面在动但聊天没变)──")

    groups = [g for g in group_similar(events) if len(g) > 1]
    print(f"\n  ── 假新行候选(同一行被读成变体){len(groups)} 簇 ──")
    if not groups:
        print("   无：同一行没有出现多条近似变体，抖动控制良好。")
    for group in sorted(groups, key=lambda g: -len(g))[:6]:
        print(f"    ×{len(group)}  首次 {group[0]['t']:.1f}s")
        for event in group[:4]:
            print(f"        {event['src']:<4} {event['score']:5.3f}  {event['text']}")

    poor = [(e, why) for e in events if (why := is_low_quality(e["text"], e["score"]))]
    print(f"\n  ── 低质量行 {len(poor)} 条 ──")
    for event, why in poor[:10]:
        print(f"    {why:<14} [{event['t']:6.1f}s] {event['text']}")
    if not poor:
        print("   无。")

    gaps = [events[i]["t"] - events[i - 1]["t"] for i in range(1, len(events))]
    if gaps:
        ordered = sorted(gaps)
        median = ordered[len(ordered) // 2]
        limit = max(3.0, median * GAP_FACTOR)
        holes = [(events[i - 1]["t"], events[i]["t"], gaps[i - 1])
                 for i in range(1, len(events)) if gaps[i - 1] > limit]
        print(f"\n  ── 疑似漏行空窗(相邻新行间隔 > {limit:.1f}s){len(holes)} 处 ──")
        for start, end, gap in holes[:10]:
            print(f"    {start:6.1f}s → {end:6.1f}s   空 {gap:.1f}s")
        if not holes:
            print("   无。")

    print("\n  ── 建议怎么看这份数据 ──")
    print("    ① 新行数 vs 你实际发了几条：多了看「假新行候选」，少了看「疑似漏行空窗」。")
    print("    ② 抖动簇多且集中在同一行位 → 可以开 --dedupe-similar 0.85 压噪声。")
    print("    ③ 空窗集中在连发/滚动时段 → 说明串行阻塞漏帧，该做采集与 OCR 解耦了。")


def read_expected(path: str) -> List[str]:
    """从测试句文件里取标准答案:只收【小节】之间的句子,到"测试顺序"为止。

    文件里还有用法说明与测试步骤,它们不是待识别的内容,混进来会把召回率算错。
    """
    expected = []
    started = False
    with open(path, encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("【"):
                started = True
                continue
            if "测试顺序" in line:
                break
            if not started or not line or line[0] in "#=" or line[:6].find("：") >= 0:
                continue
            expected.append(line)
    return expected


def recall_check(expected_path: str, events: List[dict]) -> None:
    """拿测试句当标准答案,核对"该出现的句子是否被认出来了"——比猜空窗靠谱。"""
    try:
        expected = read_expected(expected_path)
    except OSError as exc:
        print(f"\n  [!] 读不到标准答案 {expected_path}: {exc}")
        return
    if not expected:
        return

    print(f"\n  ── 召回核对(标准答案 {len(expected)} 句)──")
    exact, partial = 0, []
    for line in expected:
        want = norm_key(line)
        if not want:
            continue
        best, best_ratio = None, 0.0
        for event in events:
            ratio = SequenceMatcher(None, want, norm_key(event["text"])).ratio()
            if ratio > best_ratio:
                best, best_ratio = event, ratio
        if best_ratio >= 0.85:
            exact += 1
            print(f"    ✓ {line}")
        elif best_ratio >= 0.55:
            partial.append(line)
            print(f"    ~ {line}   ← 部分识别 {best_ratio:.2f}: {best['text']}")
        else:
            partial.append(line)
            print(f"    ✗ {line}   ← 没出现,或读成了完全不同的字")
    print(f"    召回 {exact + len(partial)}/{len(expected)}"
          f"（完全正确 {exact} 句，部分/缺失 {len(partial)} 句）")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="读探针日志出判读报告")
    parser.add_argument("path", help="logs\\live_xxx.json 或 .log")
    parser.add_argument("--expect", metavar="FILE",
                        help="测试句文件(如 chat_samples.txt),用来核对召回")
    args = parser.parse_args(argv)

    summary, events = load_events(args.path)
    report(summary, events, args.path)
    if args.expect:
        recall_check(args.expect, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
