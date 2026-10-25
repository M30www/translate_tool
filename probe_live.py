#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_live.py —— 半实时探针:定时截屏 → 帧差 → 画面变了才跑集成 OCR
  默认一直跑到 Ctrl+C;只输出新出现的行,状态靠 --heartbeat 心跳呈现。
  取图:--select 框选 / --bbox 坐标(真实屏幕);--replay 回放静态图(自测)。
复用:框选与 DPI 来自 ocr_probe,引擎与集成来自 explore/ocr/ocr_simple。
来由、遗留与实测数据见 docs/dev-log.md D2/D3、docs/experiments.md E3/E4。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_text import norm_key   # noqa: E402  (共用判据,见 chat_text.py)

# 中文 Windows 控制台是 GBK,直接 print 韩日文会 UnicodeEncodeError 崩掉。
# line_buffering:输出被重定向到文件时默认是块缓冲,日志会憋到最后才出现,长跑就看不见了。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))

# 帧差判据:缩到 160x90 灰度后比较,单像素亮度差 > DIFF_PIXEL 算"变了"。
# 聊天区文字很小,按原分辨率比会被抗锯齿噪声淹没;缩图后反而更稳。
THUMB = (160, 90)
DIFF_PIXEL = 16


def _load(name: str, rel: str):
    """把仓库里已有脚本当模块加载(它们都有 main guard,import 不会执行主流程)。"""
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记到 sys.modules:被加载文件里的 @dataclass 会按 __module__ 反查模块
    # 自己的命名空间,查不到就报 AttributeError: 'NoneType' object has no attribute '__dict__'。
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Stats:
    frames: int = 0
    ocr_runs: int = 0
    skipped: int = 0
    capture_ms: List[float] = field(default_factory=list)
    diff_ms: List[float] = field(default_factory=list)
    ocr_ms: List[float] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)
    lines: int = 0
    new_lines: int = 0
    updated: int = 0
    events: List[dict] = field(default_factory=list)   # 新行事件,供 analyze_log.py 判读


class Tee:
    """同时写控制台与日志文件;控制台看不全时,文件里有完整的。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def to_gray_thumb(img):
    from PIL import Image
    return _np().asarray(img.convert("L").resize(THUMB, Image.BILINEAR), dtype="int16")


def _np():
    import numpy
    return numpy


def to_bgr(img):
    np = _np()
    # RapidOCR 内部走 OpenCV 约定要 BGR;[::-1] 是负步长视图,必须 ascontiguousarray。
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


def diff_ratio(prev, cur) -> float:
    return float((_np().abs(prev - cur) > DIFF_PIXEL).mean())


def grab_screen(bbox: Tuple[int, int, int, int]):
    from PIL import ImageGrab
    return ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")


def make_replay_loader(path: str, sim_change_every: int):
    """静态图回放。sim_change_every > 0 时,每 N 帧贴一条漂移色块造出真实帧差。"""
    from PIL import Image, ImageDraw
    base = Image.open(path).convert("RGB")
    box = [int(base.width * 0.1), int(base.height * 0.5),
           int(base.width * 0.35), int(base.height * 0.5) + 12]
    counter = {"n": 0}

    def load():
        counter["n"] += 1
        frame = base.copy()
        if sim_change_every and counter["n"] % sim_change_every == 0:
            shift = (counter["n"] * 7) % max(1, base.width - (box[2] - box[0]))
            ImageDraw.Draw(frame).rectangle(
                [shift, box[1], shift + (box[2] - box[0]), box[3]], fill=(0, 0, 0))
        return frame

    return load


def resolve_source(probe, args):
    """返回 (取图函数, 描述文本)。"""
    if args.replay:
        extra = f"  造变化:每 {args.sim_change_every} 帧贴色块" if args.sim_change_every else ""
        return make_replay_loader(args.replay, args.sim_change_every), f"回放 {args.replay}{extra}"

    probe.enable_dpi_awareness()   # 必须在取坐标之前,否则高 DPI 下框选和截图错位
    if args.bbox:
        bbox = tuple(int(v) for v in args.bbox.replace(" ", "").split(","))
        if len(bbox) != 4:
            raise SystemExit("[x] --bbox 需要 4 个数字: x1,y1,x2,y2")
    else:
        print("请在屏幕上拖拽框选聊天区域(Esc 取消)…")
        bbox = probe.select_region_interactive()
    x1, y1, x2, y2 = bbox
    return (lambda: grab_screen(bbox)), f"屏幕区域 {x2 - x1} x {y2 - y1} @ ({x1},{y1})"


def emit(elapsed: float, ratio: float, capture_ms: float, diff_ms: float,
         ocr_ms: float, lines: list, seen: dict, reported: dict,
         similar_thr: float, events: list, visible_ttl: float,
         forget_after: float, history: list, prefix_re) -> Tuple[int, int, list]:
    """只报「刚出现在屏幕上、且没报过」的消息;没新行就一个字都不打。返回 (新行数, 补全数, 帧键)。

    判据全部建立在 norm_key 上(剥掉行首昵称前缀、去空白标点),因为 OCR 在这类小字上
    的抖动主要发生在前缀:正文一致的两帧会被原文比较判成两条不同消息(实测一条消息
    被报 18 次)。四条规则:
      · seen     : key → 最后一次在屏幕上看到的时刻。存活期内(visible_ttl)出现 = 不是新消息。
      · reported : key → 最后报出时刻。forget_after 内不重报,挡掉滚动/回看造成的重复。
      · 残片抑制 : 无昵称前缀 且 key 是某条已报消息的子串 → 那条消息已经展示过了,不重复报。
      · 补全更新 : 新的 key **包含**某条已报消息的 key → 同一条消息这次读得更全(头部滚回来了),
                   标成更新而不是新消息,让上层替换已显示的那条。这是"残片也要翻译"的前提:
                   先看到的残片照常翻,等完整版出现再更新文本。
    """
    now = time.perf_counter()
    fresh = []
    frame: dict = {}
    for top, left, src, score, text, rgb in lines:
        key = norm_key(text)
        if key:
            # 同帧重复只留第一条;行位(top/left)与颜色一并留存,便于判读与区分消息类型
            frame.setdefault(key, (top, left, src, score, text, rgb))

    for key, (top, left, src, score, text, rgb) in frame.items():
        was_on_screen = (now - seen.get(key, -1e9)) <= visible_ttl
        seen[key] = now
        if was_on_screen:
            continue
        if (now - reported.get(key, -1e9)) <= forget_after:
            continue

        supersedes = None
        if len(key) >= 4:
            for old_key, _old_text in reversed(history):
                if len(old_key) >= 4 and old_key != key and old_key in key:
                    supersedes = old_key
                    break
        if supersedes is None and not prefix_re.match(text) and len(key) >= 4:
            if any(key in old_key for old_key, _old_text in history):
                continue        # 残缺续行(头部被裁),那条完整消息已经报过
        if similar_thr > 0:
            from difflib import SequenceMatcher
            if any(SequenceMatcher(None, other, text).ratio() >= similar_thr
                   for other in [e["text"] for e in events[-6:]]):
                continue
        fresh.append((key, top, left, src, score, text, rgb,
                      supersedes, not prefix_re.match(text)))

    updates = 0
    if fresh:
        e2e = capture_ms + diff_ms + ocr_ms
        print(f"\n[{elapsed:7.1f}s] 新行 {len(fresh)}   变化 {ratio*100:5.2f}%"
              f"   OCR {ocr_ms:.0f} ms   端到端 {e2e:.0f} ms")
        for key, top, left, src, score, text, rgb, supersedes, partial in fresh:
            color = "#%02X%02X%02X" % tuple(rgb)
            if supersedes:
                updates += 1
                mark = "↻"
                history[:] = [item for item in history if item[0] != supersedes]
            else:
                mark = "…" if partial else " "
            print(f"  {mark} {src:<4} {score:5.3f} {color}  {text}")
            reported[key] = now
            history.append((key, text))
            del history[:-40]          # 只留最近 40 条,足够挡住刚滚出去的残片
            events.append({"t": round(elapsed, 3), "y": int(top), "x": int(left),
                           "col": color, "src": src, "score": round(float(score), 4),
                           "text": text, "partial": partial,
                           "updates": supersedes})
    return len(fresh) - updates, updates, list(frame)


def heartbeat(stats: Stats, elapsed: float) -> None:
    fps = stats.frames / elapsed if elapsed > 0 else 0
    skip = stats.skipped / stats.frames * 100 if stats.frames else 0
    print(f"[{elapsed:7.1f}s] 帧 {stats.frames:5d} {fps:4.1f}fps | 跳过 {skip:4.1f}% |"
          f" OCR {stats.ocr_runs:3d} 次 均 {pct(stats.ocr_ms, 50):4.0f}ms |"
          f" 新行 {stats.new_lines:3d} | 补全 {stats.updated:2d} | 累计行 {stats.lines:4d}")


def report(stats: Stats, args, wall_s: float) -> None:
    print("\n" + "=" * 78)
    print("  稳态汇总")
    print("=" * 78)
    fps = stats.frames / wall_s if wall_s > 0 else 0
    skip_rate = stats.skipped / stats.frames * 100 if stats.frames else 0
    print(f"  运行 {wall_s:.1f}s   帧数 {stats.frames}   实际 {fps:.1f} fps"
          f"   (目标间隔 {args.interval * 1000:.0f} ms)")
    print(f"  截屏     avg {pct(stats.capture_ms, 50):5.1f}  p95 {pct(stats.capture_ms, 95):5.1f} ms")
    print(f"  帧差     avg {pct(stats.diff_ms, 50):5.1f}  p95 {pct(stats.diff_ms, 95):5.1f} ms")
    print(f"  跳过率   {skip_rate:5.1f}%   (静帧被省掉的 OCR: {stats.skipped} 帧)")
    print(f"  OCR      avg {pct(stats.ocr_ms, 50):5.1f}  p95 {pct(stats.ocr_ms, 95):5.1f}"
          f"  max {max(stats.ocr_ms) if stats.ocr_ms else 0:5.1f} ms   共 {stats.ocr_runs} 次")
    print(f"  端到端   avg {pct(stats.e2e_ms, 50):5.1f}  p95 {pct(stats.e2e_ms, 95):5.1f} ms"
          "   (画面变化 → 拿到文本)")
    print(f"  文本     累计 {stats.lines} 行,其中新行 {stats.new_lines} 行"
          f",补全更新 {stats.updated} 次")
    budget = pct(stats.e2e_ms, 95)
    verdict = "够用(<1s)" if budget < 1000 else "偏慢(>1s),聊天场景会迟钝"
    print(f"\n  判断:端到端 p95 {budget:.0f} ms —— {verdict}")
    if stats.ocr_runs == 0:
        print("  [!] 一次 OCR 都没触发:要么画面完全静止,要么 --threshold 太高。")


def write_json(path: str, args, desc: str, stats: Stats, wall_s: float) -> None:
    """机器可读的结果:分析交给 analyze_log.py,人眼只看控制台那几行。"""
    payload = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "source": desc,
        "args": {k: v for k, v in vars(args).items() if k != "sim_change_every"},
        "summary": {
            "wall_s": round(wall_s, 2),
            "frames": stats.frames,
            "fps": round(stats.frames / wall_s, 2) if wall_s > 0 else 0,
            "skipped_frames": stats.skipped,
            "skip_rate": round(stats.skipped / stats.frames * 100, 1) if stats.frames else 0,
            "ocr_runs": stats.ocr_runs,
            "ocr_ms_p50": round(pct(stats.ocr_ms, 50), 1),
            "ocr_ms_p95": round(pct(stats.ocr_ms, 95), 1),
            "e2e_ms_p50": round(pct(stats.e2e_ms, 50), 1),
            "e2e_ms_p95": round(pct(stats.e2e_ms, 95), 1),
            "lines_total": stats.lines,
            "new_lines": stats.new_lines,
            "updated": stats.updated,
        },
        "events": stats.events,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_args(argv: Sequence[str]):
    p = argparse.ArgumentParser(
        description="半实时探针:定时截屏 → 帧差 → 变化才 OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python probe_live.py --select                        # 框选后一直跑到 Ctrl+C
  python probe_live.py --bbox 100,600,700,760 --duration 60
  python probe_live.py --replay shots\\a.png --sim-change-every 8   # 无界面自测
""",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--select", action="store_true", help="交互式框选屏幕区域")
    src.add_argument("--bbox", metavar="X1,Y1,X2,Y2", help="直接用屏幕绝对坐标")
    src.add_argument("--replay", metavar="PATH", help="回放一张静态图(自测用)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="运行秒数,0=一直跑到 Ctrl+C(默认 0)")
    p.add_argument("--interval", type=float, default=0.25, help="目标帧间隔秒数(默认 0.25)")
    p.add_argument("--threshold", type=float, default=0.002,
                   help="帧差比例超过它就 OCR(默认 0.002,即 0.2%% 像素变化)")
    p.add_argument("--min-score", type=float, default=0.5, help="低于此置信度的行丢弃")
    p.add_argument("--heartbeat", type=float, default=10.0,
                   help="每隔多少秒打一行状态心跳,0=不打(默认 10)")
    p.add_argument("--recent-ttl", type=float, default=120.0,
                   help="同一条消息报出后,这段时间内不再重复报(默认 120 秒)")
    p.add_argument("--visible-ttl", type=float, default=6.0,
                   help="消息在屏幕上存活的判定窗口,挡住漏读一帧导致的误报(默认 6 秒)")
    p.add_argument("--no-merge", action="store_true",
                   help="不做同一行碎片合并(默认合并:长句被检测框切碎时要靠它拼回去)")
    p.add_argument("--min-height-ratio", type=float, default=0.7,
                   help="框高低于中位数的这个比例 = 没翻译价值,直接丢弃(默认 0.7)")
    p.add_argument("--no-join-wrapped", action="store_true",
                   help="不把换行显示的续行拼回上一条(默认拼:靠颜色区分,不会误并系统提示)")
    p.add_argument("--dedupe-similar", type=float, default=0.0,
                   help="与最近几条相似度达该值就不报(0=关闭;归一化后一般不需要)")
    p.add_argument("--sim-change-every", type=int, default=0,
                   help="回放模式下每 N 帧造一次变化(默认 0=不造)")
    p.add_argument("--log-dir", default="logs", help="日志目录(默认 logs)")
    p.add_argument("--no-log", action="store_true", help="不落盘,只打控制台")
    return p.parse_args(argv)


def setup_log(args):
    """把 stdout 同时导到日志文件;返回 (日志路径, JSON 路径, 原 stdout)。"""
    console = sys.stdout
    if args.no_log:
        return None, None, console
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"live_{stamp}.log")
    # utf-8-sig:带 BOM,Windows 记事本打开韩日文不会乱码。
    handle = open(log_path, "w", encoding="utf-8-sig", buffering=1)
    sys.stdout = Tee(console, handle)
    return log_path, os.path.splitext(log_path)[0] + ".json", console


def main(argv: Sequence[str]) -> int:
    args = build_args(argv)
    log_path, json_path, console = setup_log(args)
    probe = _load("pp_probe", "ocr_probe.py")
    ocr = _load("pp_simple", os.path.join("explore", "ocr", "ocr_simple.py"))

    source, desc = resolve_source(probe, args)
    print("=" * 78)
    print(f"  半实时探针   取图={desc}   阈值={args.threshold*100:.1f}%")
    print(f"  启动 {time.strftime('%Y-%m-%d %H:%M:%S')}   {platform.node()}"
          f"   Python {platform.python_version()}")
    print("=" * 78)
    if log_path:
        print(f"  日志: {os.path.abspath(log_path)}")
        print(f"  判读: python analyze_log.py {os.path.abspath(json_path)}")
    print("加载模型 …")
    t0 = time.perf_counter()
    engines = {head: ocr.build(head) for head in ocr.HEADS}
    print(f"  就绪({time.perf_counter() - t0:.2f} s,一次性开销,不进入每帧延迟)\n")

    stats = Stats()
    seen: dict = {}               # 归一化键 → 最后一次在屏幕上看到的时刻
    reported: dict = {}           # 归一化键 → 最后报出时刻
    history: list = []            # 最近报过的 (key, text),用来识别"头部被裁掉的残片"
    prev_gray: Optional[object] = None
    started = time.perf_counter()
    deadline = started + args.duration if args.duration > 0 else None
    next_beat = started + args.heartbeat if args.heartbeat > 0 else float("inf")
    print("开始检测(没新行就不输出;Ctrl+C 结束并打印汇总)\n")

    # Ctrl+C 只结束循环,后面照常出汇总 —— 中断时最需要看的就是统计。
    try:
        while deadline is None or time.perf_counter() < deadline:
            frame_start = time.perf_counter()
            stats.frames += 1

            t = time.perf_counter()
            img = source()
            capture_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            gray = to_gray_thumb(img)
            ratio = 1.0 if prev_gray is None else diff_ratio(prev_gray, gray)
            prev_gray = gray
            diff_ms = (time.perf_counter() - t) * 1000
            stats.capture_ms.append(capture_ms)
            stats.diff_ms.append(diff_ms)

            if ratio >= args.threshold:
                t = time.perf_counter()
                lines = ocr.ensemble(to_bgr(img), engines, args.min_score,
                                     merge=not args.no_merge,
                                     join_lines=not args.no_join_wrapped,
                                     height_ratio=args.min_height_ratio)
                ocr_ms = (time.perf_counter() - t) * 1000
                stats.ocr_runs += 1
                stats.ocr_ms.append(ocr_ms)
                stats.e2e_ms.append(capture_ms + diff_ms + ocr_ms)
                stats.lines += len(lines)
                fresh, updated, _ = emit(time.perf_counter() - started, ratio,
                                         capture_ms, diff_ms, ocr_ms, lines,
                                         seen, reported, args.dedupe_similar, stats.events,
                                         args.visible_ttl, args.recent_ttl,
                                         history, ocr.PREFIX)
                stats.new_lines += fresh
                stats.updated += updated
            else:
                stats.skipped += 1

            now = time.perf_counter()
            if now >= next_beat:
                stale = [k for k, ts in reported.items() if now - ts > args.recent_ttl]
                for key in stale:
                    del reported[key]
                stale = [k for k, ts in seen.items() if now - ts > args.recent_ttl]
                for key in stale:
                    del seen[key]
                heartbeat(stats, now - started)
                next_beat = now + args.heartbeat

            remain = args.interval - (time.perf_counter() - frame_start)
            if remain > 0:
                time.sleep(remain)
    except KeyboardInterrupt:
        print("\n[!] 手动中断,以下是已采集的统计。")

    wall_s = time.perf_counter() - started
    report(stats, args, wall_s)

    if json_path:
        write_json(json_path, args, desc, stats, wall_s)
        print(f"\n  已写出机器可读结果: {os.path.abspath(json_path)}")
        print(f"  下一步: python analyze_log.py {os.path.abspath(json_path)}")
    if log_path:
        sys.stdout = console          # 还原,避免文件句柄被后续打印继续写
        print(f"  已写出日志: {os.path.abspath(log_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
