"""临时:量被裁切行的几何特征(框高是否明显小于正常行)。"""
import importlib.util
import sys

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("os_ocr", "explore/ocr/ocr_simple.py")
ocr = importlib.util.module_from_spec(spec)
sys.modules["os_ocr"] = ocr
spec.loader.exec_module(ocr)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for name in ("chat_team_cut", "chat_team_system"):
    image = Image.open(f"shots/{name}.png").convert("RGB")
    img = np.ascontiguousarray(np.array(image)[:, :, ::-1])
    boxes = ocr.read_boxes(ocr.build("v6") if False else None, img) if False else None
    engines = {head: ocr.build(head) for head in ocr.HEADS}
    boxes = ocr.read_boxes(engines["v6"], img)
    rows = []
    for _key, (text, score, box) in boxes.items():
        ys = [float(p[1]) for p in box]
        xs = [float(p[0]) for p in box]
        rows.append((min(ys), max(ys), min(xs), text, score))
    rows.sort()
    heights = [b - a for a, b, _x, _t, _s in rows]
    median = sorted(heights)[len(heights) // 2]
    print(f"\n=== {name}  图高 {image.height}  框高中位数 {median:.0f}px ===")
    print(f"{'top':>6} {'bottom':>7} {'h':>5} {'h/中位':>7} {'贴边':>5}  文本")
    for top, bottom, left, text, score in rows:
        h = bottom - top
        near_edge = "顶" if top < image.height * 0.12 else ("底" if bottom > image.height * 0.88 else "")
        print(f"{top:6.0f} {bottom:7.0f} {h:5.0f} {h / median:7.2f} {near_edge:>5}  {text[:44]}")
