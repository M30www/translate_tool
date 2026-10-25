#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
color_probe.py —— 量出画面里每一行文字的「代表色」,用来给消息分类(聊天/系统/提示…)
  用法:python color_probe.py shots\\x.png            (整张图)
        python color_probe.py shots\\x.png --bbox 85,604,1019,1110

  为什么用颜色:OCR 的几何和文本每帧都在抖,而颜色是稳定的;游戏里不同类型的
  消息本来就用不同颜色(白=玩家、黄/橙=系统提示…),这是比"有没有昵称前缀"更硬的判据。

  做法:对每个检测框,取框内最亮的一批像素(= 字形笔画)求均值,即"文字色";
  再把所有行的颜色按 32 级量化聚簇,同一簇就是同一种文字。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_ocr():
    path = os.path.join("explore", "ocr", "ocr_simple.py")
    spec = importlib.util.spec_from_file_location("os_ocr", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["os_ocr"] = module
    spec.loader.exec_module(module)
    return module


def text_color(pixels, bright_ratio: float = 0.12):
    """取框内最亮的 12% 像素求均值 —— 那就是字形笔画的颜色,不含背景。"""
    import numpy as np
    flat = pixels.reshape(-1, 3).astype("int16")
    luma = flat.sum(axis=1)
    keep = max(3, int(len(flat) * bright_ratio))
    idx = np.argpartition(luma, -keep)[-keep:]
    return flat[idx].mean(axis=0)


def rgb_to_hsv(rgb):
    import colorsys
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s * 100, v * 100


def describe(rgb: list, hsv: tuple) -> str:
    """给个人话标签,方便和肉眼看到的对上。"""
    h, s, v = hsv
    if s < 18:
        return "白/灰" if v > 70 else "暗灰"
    if h < 20 or h >= 345:
        return "红/橙红"
    if h < 45:
        return "黄/橙"
    if h < 70:
        return "黄绿"
    if h < 160:
        return "绿"
    if h < 200:
        return "青"
    if h < 260:
        return "蓝"
    if h < 300:
        return "紫"
    return "粉/品红"


def main(argv):
    parser = argparse.ArgumentParser(description="量出每行文字的代表色")
    parser.add_argument("image")
    parser.add_argument("--bbox", metavar="X1,Y1,X2,Y2", help="只分析屏幕截图的这块区域")
    parser.add_argument("--min-score", type=float, default=0.5)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    ocr = load_ocr()
    image = Image.open(args.image).convert("RGB")
    offset = (0, 0)
    if args.bbox:
        box = tuple(int(v) for v in args.bbox.replace(" ", "").split(","))
        if len(box) != 4:
            raise SystemExit("[x] --bbox 需要 4 个数字:x1,y1,x2,y2")
        offset = (box[0], box[1])
        image = image.crop(box)
    pixels = np.array(image)

    engines = {head: ocr.build(head) for head in ocr.HEADS}
    boxes = ocr.read_boxes(engines["v6"], image)
    rows = []
    for _key, (text, score, box) in boxes.items():
        if score < args.min_score:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        left, right = int(max(0, min(xs))), int(min(image.width, max(xs)))
        top, bottom = int(max(0, min(ys))), int(min(image.height, max(ys)))
        if right - left < 3 or bottom - top < 3:
            continue
        rgb = text_color(pixels[top:bottom, left:right])
        hsv = rgb_to_hsv(rgb)
        rows.append({"y": int(top + offset[1]), "x": int(left + offset[0]),
                     "rgb": [int(round(v)) for v in rgb], "hsv": hsv,
                     "score": score, "text": text})

    rows.sort(key=lambda r: (r["y"], r["x"]))
    print(f"{'y':>5} {'x':>5}  {'hex':<8} {'RGB':<15} {'色相':>5} {'饱和':>5} {'亮度':>5}  判读")
    print("-" * 100)
    for row in rows:
        r, g, b = row["rgb"]
        h, s, v = row["hsv"]
        print(f"{row['y']:5d} {row['x']:5d}  #{r:02X}{g:02X}{b:02X} "
              f"({r:3d},{g:3d},{b:3d}) {h:5.0f} {s:5.0f} {v:5.0f}  {describe(row['rgb'], row['hsv'])}")
        print(f"{'':13}{row['text']}")

    palette: dict = {}
    for row in rows:
        key = tuple(int(c) // 32 for c in row["rgb"])
        palette.setdefault(key, []).append(row)
    print(f"\n=== 颜色聚簇(量化到 32 级):{len(palette)} 种 ===")
    for key, group in sorted(palette.items(), key=lambda kv: -len(kv[1])):
        avg = [sum(r["rgb"][i] for r in group) / len(group) for i in range(3)]
        hsv = rgb_to_hsv(avg)
        print(f"  ×{len(group):<2} #{int(avg[0]):02X}{int(avg[1]):02X}{int(avg[2]):02X}"
              f"  {describe([int(v) for v in avg], hsv):<8} 例:{group[0]['text'][:28]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
