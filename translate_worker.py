#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
translate_worker.py —— 实时翻译线程:采集/OCR 线程只管投递,翻译在另一个线程里做
  为什么必须分线程:OCR 已占 300~1000ms,翻译是网络请求再加 300~1200ms。串进主循环会把
  2~3 fps 拖到 1 fps 以下,而且**采集会跟着停** —— 那就真开始漏消息了。
  队列满了丢最旧的一条:聊天场景宁可丢一条译文,也不能堵住采集。

  过滤策略(filter_mode):
    color —— 只翻"频道色"的行(已出现次数最多的颜色),系统提示不翻
    all   —— 除中文外的都翻(调试用)
"""

from __future__ import annotations

import queue
import threading
import time

from chat_text import norm_key

SKIP_REASONS = ("no_text", "chinese", "color")


class LiveTranslator:
    """投递即返回;翻译在后台线程做,结果通过 on_result 回调吐出。"""

    def __init__(self, translator, chunk: int = 4, window: float = 0.25,
                 prefix_re=None, filter_mode: str = "color", on_result=None,
                 queue_size: int = 64):
        self.translator = translator
        self.chunk = max(1, chunk)
        self.window = window
        self.prefix_re = prefix_re
        self.filter_mode = filter_mode
        self.on_result = on_result

        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stats = {"submitted": 0, "translated": 0, "dropped": 0, "failed": 0,
                      "no_text": 0, "chinese": 0, "color": 0}
        self.latencies: list[float] = []       # 每次"取批 → 出结果"的毫秒数
        self.color_counts: dict = {}
        self.results: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="translate")

    # ---------- 对外 ----------
    def start(self) -> None:
        self._thread.start()

    def submit(self, event: dict) -> None:
        """event: {t, text, col, score}。队列满时丢最旧的一条。"""
        self.stats["submitted"] += 1
        item = {"event": event, "enqueued": time.perf_counter()}
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()          # 丢最旧
                self.stats["dropped"] += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.stats["dropped"] += 1

    def stop(self, drain_timeout: float = 30.0) -> None:
        self._stop.set()
        self._thread.join(timeout=drain_timeout)

    # ---------- 过滤 ----------
    def _skip_reason(self, event: dict) -> str:
        """返回空串=要翻;否则是跳过原因。"""
        text = event.get("text", "")
        if len(norm_key(text)) < 2:
            return "no_text"
        body = self._split(text)[1]
        if _looks_chinese(body):
            return "chinese"
        if self.filter_mode != "color":
            return ""
        rep = _color_rep(event.get("col", ""), self.color_counts)
        if not rep:
            return ""
        self.color_counts[rep] = self.color_counts.get(rep, 0) + 1
        # 样本太少时不做颜色过滤,否则头几条可能因为"当时它自己最多"而全被放行/挡掉
        if self.stats["submitted"] <= 5:
            return ""
        dominant = max(self.color_counts, key=self.color_counts.get)
        return "" if rep == dominant else "color"

    def _split(self, text: str) -> tuple[str, str]:
        """拆出 `[昵称]:` 前缀与正文 —— 只送正文,前缀原样拼回。"""
        if not self.prefix_re:
            return "", text
        match = self.prefix_re.match(text)
        if not match:
            return "", text
        end = match.end()
        rest = text[end:]
        trimmed = rest.lstrip()
        if trimmed[:1] in (":", "："):
            end += len(rest) - len(trimmed) + 1
        return text[:end], text[end:]

    # ---------- 线程主体 ----------
    def _run(self) -> None:
        while True:
            if self._stop.is_set() and self.queue.empty():
                return
            try:
                first = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            items = [first]
            deadline = time.perf_counter() + self.window
            while len(items) < self.chunk:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            self._translate(items)

    def _translate(self, items: list) -> None:
        started = time.perf_counter()
        texts, keep = [], []
        for item in items:
            event = item["event"]
            reason = self._skip_reason(event)
            if reason:
                self.stats[reason] = self.stats.get(reason, 0) + 1
                continue
            prefix, body = self._split(event.get("text", ""))
            texts.append(body)
            keep.append((item, prefix, body))
        if not texts:
            return

        try:
            translated = self.translator.translate_batch(texts)
        except Exception as exc:                            # noqa: BLE001
            self.stats["failed"] += len(texts)
            for item, prefix, body in keep:
                self._emit({**item["event"], "translated": f"⟨失败⟩{body}",
                            "ms": 0.0, "queue_ms": 0.0, "error": str(exc)})
            return

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.latencies.append(elapsed_ms)
        per_item = elapsed_ms / max(1, len(keep))
        for (item, prefix, body), value in zip(keep, translated):
            self.stats["translated"] += 1
            self._emit({**item["event"], "translated": f"{prefix}{value}".strip(),
                        "ms": per_item,
                        "queue_ms": (time.perf_counter() - item["enqueued"]) * 1000})

    def _emit(self, record: dict) -> None:
        self.results.append(record)
        if self.on_result:
            self.on_result(record)


def _color_rep(color: str, known: dict, tol: float = 40.0) -> str:
    """把一个颜色归到已有的代表色上(距离 ≤ tol 视为同色)。

    必须按距离比较:同一频道色的每帧测量值会抖(实测青色散成 #00AFEB / #00A4DD / #00AFEA),
    用精确 hex 或过细的量化桶都会把同一频道判成不同颜色,导致大半聊天被当成系统提示丢掉(踩过)。
    """
    if not color.startswith("#") or len(color) != 7:
        return ""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    for rep in known:
        rep_rgb = [int(rep[i:i + 2], 16) for i in (1, 3, 5)]
        if sum(abs(a - b) for a, b in zip(rgb, rep_rgb)) / 3 <= tol:
            return rep
    return color


def _looks_chinese(text: str) -> bool:
    """含汉字且不含假名/谚文 = 中文。日文必带假名、韩文必带谚文,不会被误判。"""
    has_han = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    if not has_han:
        return False
    has_kana = any("\u3040" <= ch <= "\u30ff" for ch in text)
    has_hangul = any("\uac00" <= ch <= "\ud7a3" or "\u1100" <= ch <= "\u11ff"
                     or "\u3130" <= ch <= "\u318f" for ch in text)
    return not has_kana and not has_hangul


def describe(worker: LiveTranslator, wall_s: float) -> str:
    """给主流程打印用的摘要。"""
    s = worker.stats
    lat = sorted(worker.latencies)
    p50 = lat[len(lat) // 2] if lat else 0
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else 0
    return (f"投递 {s['submitted']} 条 → 已译 {s['translated']} "
            f"(跳过:中文 {s['chinese']},非频道色 {s['color']},无文字 {s['no_text']};"
            f" 失败 {s['failed']},队列丢弃 {s['dropped']})  "
            f"翻译耗时 p50 {p50:.0f}ms / p95 {p95:.0f}ms")
