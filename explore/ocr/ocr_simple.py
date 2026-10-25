#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocr_simple.py —— 读图片 → 自动识别中/日/韩 → 在终端打印结果
================================================================================
用法(在项目根目录执行):

    .venv\\Scripts\\python.exe explore\\ocr\\ocr_simple.py shots\\
    .venv\\Scripts\\python.exe explore\\ocr\\ocr_simple.py shots\\Overwatch_xxx.png

它做什么:

    对每张图片,用两个识别模型各读一遍,再逐框择优。

      ko5 = korean_PP-OCRv5_rec_mobile.onnx  → 韩文
      v6  = PP-OCRv6_rec_small.onnx          → 中/繁中/日/英 等 52 种

    判据只有一条:
        某个框的韩文结果里含谚文(가-힣 / ㄱ-ㅎ / ㅏ-ㅣ)→ 采信韩文头
        否则                                              → 采信 v6

为什么用这条判据,而不是比置信度:

    · v6 的字典里**一个谚文字都没有**(实测 18707 条,谚文 0 个),
      所以它永远吐不出韩文;反过来说,只要韩文头吐出了谚文,
      这一行就一定是韩文 —— 这个判据不可能误判。
    · 而比置信度是行不通的:实测日文模型读简体中文会给出 0.972 的高分,
      但字形是错的(頻道 / 切換 / 臥伍)。那是"自信地错",
      分数再高也不能用,只有从文字体系判断才能发现。

为什么两个模型各跑一遍完整流程,而不是共享检测:

    共享检测能省大约 40ms,但要多写一套裁剪 + 调 RapidOCR 内部 API 的代码。
    实测两个模型各自跑完是 171ms / 197ms,合计约 370ms,对聊天场景够用了。
    这里优先选简单、不依赖内部接口的写法。

结果怎么对齐:

    两个模型用的是**同一个检测模型、同一张图**,所以检测框坐标完全一致。
    于是直接拿坐标当"行号"来配对,不需要按返回顺序硬对 —— 两个模型的
    置信度过滤阈值不同时,返回条数会不一样,按顺序硬对必然错位。

依赖:pillow、numpy、rapidocr-onnxruntime(setup_ocr.bat 已装好)
模型:项目根的 models\\ 目录,由 download_models.py 下载并校验
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# explore/ocr → 上跳两级才是项目根,models/ 和 .venv/ 都在那里
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(ROOT, "models")

# 两个识别头:名字 → (模型文件, 字典文件)
# ★ 模型和字典必须成对使用。字典定义了模型能输出哪些字符,
#   两者不匹配时会静默输出乱码,不报错 —— 这是 OCR 里最难自己发现的坑。
HEADS = {
    "ko5": ("korean_PP-OCRv5_rec_mobile.onnx", "ppocrv5_korean_dict.txt"),
    "v6": ("PP-OCRv6_rec_small.onnx", "ppocrv6_dict.txt"),
}

# 检测阶段短边上限。包内默认是 736,而实测对聊天区这种小图,
# 调低到 320 反而**又准又快** —— 默认值会把小图放大 3.7 倍再检测,框反而找不准。
DET_LIMIT = 320

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def has_hangul(text: str) -> bool:
    """
    是否含谚文。这是区分韩文和其他语言的唯一判据。

    依据:谚文是韩文独有的书写系统,别的语言不会用。三个区段都要查 ——
    尤其 0x3130-0x318F(兼容字母),聊天里的 ㅠㅠ 这类表情就在这一段。
    """
    for ch in text:
        if "\uac00" <= ch <= "\ud7a3":      # 谚文音节 가-힣
            return True
        if "\u1100" <= ch <= "\u11ff":      # 谚文字母
            return True
        if "\u3130" <= ch <= "\u318f":      # 兼容字母(ㅠ ㅋ ㅎ 这些)
            return True
    return False


def box_key(box) -> tuple:
    """
    给检测框算一个可比较的键,用来把两个模型的结果配对。

    依据:两个模型共用同一个检测模型、读同一张图,检测框坐标是**完全一致**的,
    所以坐标本身就可以当行号用。取整到 1 像素是为了挡掉浮点噪声。

    为什么不按返回顺序配对:两个模型的置信度过滤阈值不同,返回条数会不一样
    (实测同一张图,一个返回 7 行、另一个返回 8 行),按顺序硬对必然错位。
    """
    return tuple(round(float(v)) for point in box for v in point)


def build(head: str):
    """按名字建一个 OCR 引擎。检测/分类模型复用包内自带的,只有识别模型分语种。"""
    from rapidocr_onnxruntime import RapidOCR

    onnx_name, dict_name = HEADS[head]
    onnx_path = os.path.join(MODELS_DIR, onnx_name)
    dict_path = os.path.join(MODELS_DIR, dict_name)

    missing = [p for p in (onnx_path, dict_path) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "[x] 模型文件缺失:\n"
            + "".join(f"      {p}\n" for p in missing)
            + "    先在项目根目录运行:  .venv\\Scripts\\python.exe download_models.py"
        )

    return RapidOCR(
        rec_model_path=onnx_path,      # → Rec.model_path
        rec_keys_path=dict_path,       # → Rec.rec_keys_path(字典)
        det_limit_side_len=DET_LIMIT,  # → Det.limit_side_len
        use_cls=False,                 # 游戏聊天文字永远正向,关掉省一次推理
        # ★ text_score=0 是关键,不能省。RapidOCR 默认会按这个阈值在自己的
        #   结果里过滤,而两个模型的识别分数不同 → 过滤后返回的**框集合不一样**,
        #   于是"某个框只有一个模型返回"的情况就会出现,合并时只能糊涂兜底,
        #   结果把另一个模型的垃圾结果认了下来(实测会多出一行 `2`)。
        #   设为 0 让两边都返回全部检测框、集合完全相同,合并就成了纯规则判定。
        #   真正的分数过滤放在最后,针对"最终选中的那条文本"做,而不是各自做。
        text_score=0.0,
    )


def read_boxes(engine, img) -> dict:
    """跑一次识别,返回 {检测框坐标: (文本, 置信度, 框)}。"""
    result, _elapse = engine(img)
    out = {}
    for box, text, score in (result or []):
        out[box_key(box)] = (str(text), float(score), box)
    return out


def _same_visual_line(a: tuple, b: tuple, gap_ratio: float) -> bool:
    """两个框是否属于同一条视觉行。

    判据用「中心线高度差 <= 半个行高」而不是「垂直重叠 >= 50%」:检测框普遍带上下留白,
    相邻两条聊天行的框会重叠三成以上,用重叠比例判会把不同行合到一起(实测踩过)。
    """
    a_top, a_left, a_bottom, a_right = a
    b_top, b_left, b_bottom, b_right = b
    a_h = max(1.0, a_bottom - a_top)
    b_h = max(1.0, b_bottom - b_top)
    if abs((a_top + a_bottom) / 2 - (b_top + b_bottom) / 2) > 0.5 * min(a_h, b_h):
        return False
    gap = max(a_left, b_left) - min(a_right, b_right)     # 负值代表水平重叠,与左右顺序无关
    return -0.5 * max(a_h, b_h) <= gap <= gap_ratio * max(a_h, b_h)


def merge_fragments(entries: list, gap_ratio: float = 0.8) -> list:
    """把同一视觉行上相邻的检测框拼回一条消息。

    依据:检测模型会把一条长聊天行切成多段(实测一条长韩文句变 3 段),不合并就会
    把一句话拆成三条消息,翻译质量直接崩。判据只看几何,不看文本内容。

    必须收敛式合并(反复扫到不再变化):按 (top,left) 贪心一遍是不够的——
    第 3 段可能因为第 1 段先吸收了第 2 段才变得"相邻",单趟会漏合。

    entries: (top, left, bottom, right, src, score, text) → 返回同样形状的合并结果。
    """
    groups = [{"bbox": tuple(e[:4]), "items": [e]}
              for e in sorted(entries, key=lambda e: (e[0], e[1]))]

    changed = True
    while changed:
        changed = False
        merged: list = []
        for group in groups:
            for kept in merged:
                if _same_visual_line(kept["bbox"], group["bbox"], gap_ratio):
                    kt, kl, kb, kr = kept["bbox"]
                    gt, gl, gb, gr = group["bbox"]
                    kept["bbox"] = (min(kt, gt), min(kl, gl), max(kb, gb), max(kr, gr))
                    kept["items"].extend(group["items"])
                    changed = True
                    break
            else:
                merged.append(group)
        groups = merged

    result = []
    for group in groups:
        items = sorted(group["items"], key=lambda e: e[1])
        if len(items) > 1:
            # 单字符段基本是左侧头像/图标区的误检碎片(实测 `や`),拼进去会污染原文;
            # 但不合并时它可能是一条真实的一字消息,所以只在"同组还有别的段"时丢。
            items = [e for e in items if sum(ch.isalnum() for ch in e[6]) > 1] or items
        result.append((
            group["bbox"][0], group["bbox"][1],
            "+".join(sorted({e[4] for e in items})),
            max(e[5] for e in items),   # 取最高分:前后缀碎片分低但不影响正文可信度
            " ".join(e[6] for e in items),
        ))
    return result


def ensemble(img, engines: dict, min_score: float, merge: bool = True) -> list:
    """两个模型各读一遍,逐框择优,可选地把同一行的碎片拼起来,按屏幕顺序返回。"""
    ko = read_boxes(engines["ko5"], img)
    v6 = read_boxes(engines["v6"], img)

    entries = []
    # 取并集:某个框可能只被其中一个模型识别出来(被另一个的阈值过滤掉了),
    # 这种情况下直接采信有的那一边,不需要纠结。
    for key in set(ko) | set(v6):
        a, b = ko.get(key), v6.get(key)

        if a and has_hangul(a[0]):
            pick, src = a, "ko5"
        elif b:
            pick, src = b, "v6"
        else:
            pick, src = a, "ko5"

        text, score, box = pick
        if score < min_score:
            continue

        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        entries.append((min(ys), min(xs), max(ys), max(xs), src, score, text))

    lines = merge_fragments(entries) if merge else [
        (e[0], e[1], e[4], e[5], e[6]) for e in entries]

    # 依据:RapidOCR 返回的顺序不保证和视觉顺序一致。按「左上角」排序,
    # 输出才和屏幕上看到的聊天行顺序对得上;按 12px 分行位,容忍轻微抖动。
    lines.sort(key=lambda r: (round(r[0] / 12), r[1]))
    return lines


def collect_images(path: str) -> list:
    """路径可以是单张图片,也可以是装着图片的文件夹。"""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if os.path.splitext(name)[1].lower() in IMG_EXT
        ]
    raise SystemExit(f"[x] 路径不存在: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="读图片 → 中/日/韩识别 → 打印结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python ocr_simple.py shots\\\n"
            "  python ocr_simple.py shots\\a.png --min-score 0.3\n"
        ),
    )
    parser.add_argument("path", help="图片文件,或装着图片的文件夹")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="低于这个置信度的行不打印(默认 0.5,调低能看到更多候选)")
    parser.add_argument("--no-merge", action="store_true",
                        help="不做同一行碎片合并(用于对比合并前后的差异)")
    args = parser.parse_args()

    # 依据:中文 Windows 的控制台是 GBK,里面没有韩文和日文假名,
    # 直接 print 识别结果会 UnicodeEncodeError 崩掉。强制 UTF-8 输出。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    images = collect_images(args.path)
    if not images:
        raise SystemExit(f"[x] 这个路径下没找到图片: {args.path}")

    print(f"共 {len(images)} 张图片,正在加载模型 …")
    engines = {head: build(head) for head in HEADS}
    print("  ko5 = 韩文模型     v6 = 中/日/英等 52 种")
    print()

    import numpy as np
    from PIL import Image

    for index, path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {os.path.basename(path)}")

        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                # 依据:RapidOCR 内部走 OpenCV 约定,要 BGR;PIL 给的是 RGB,所以翻通道。
                # [::-1] 得到的是负步长视图而不是连续内存,必须 ascontiguousarray,
                # 否则推理后端可能直接报错。
                img = np.ascontiguousarray(np.array(rgb)[:, :, ::-1])
                size = f"{rgb.width} x {rgb.height}"
        except Exception as exc:                    # noqa: BLE001
            print(f"         [x] 打不开: {type(exc).__name__}: {exc}\n")
            continue

        started = time.perf_counter()
        lines = ensemble(img, engines, args.min_score, merge=not args.no_merge)
        cost_ms = (time.perf_counter() - started) * 1000

        if not lines:
            print(f"         {size}    (没有识别到文字)  {cost_ms:,.0f} ms\n")
            continue

        print(f"         {size}    {cost_ms:,.0f} ms")
        print(f"         {'#':>3}  {'来自':<4} {'置信度':>7}  文本")
        print(f"         {'-' * 62}")
        for number, (_top, _left, src, score, text) in enumerate(lines, 1):
            print(f"         {number:>3}  {src:<4} {score:>7.3f}  {text}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
