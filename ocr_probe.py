#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocr_probe.py —— 游戏聊天区 OCR 可行性探针
================================================================================
目的:在做正式工具之前,先用最小成本回答三个问题 ——

  1) OCR 到底认不认得出游戏里那种「小字号 + 白色描边 + 动态背景」的韩文/日文?
  2) 单次识别要多久?够不够跟上游戏聊天(通常 1 秒内出结果才不难受)?
  3) 换预处理方式(放大 / 灰度 / 二值化)对准确率有没有帮助?

三种取图方式,任选其一:
  A. 读图片文件        python ocr_probe.py --image shot.png --lang ko
  B. 交互式框选截屏    python ocr_probe.py --region        --lang ko
  C. 自检(合成图片)    python ocr_probe.py --selftest      --lang ko
                       --selftest 会用系统里的中/日/韩字体现场画一张
                       仿游戏聊天截图来跑,用来确认「整条链路是通的」,
                       和「你的安装没问题」,不需要你手上有截图。

为什么用 PaddleOCR 而不是 Windows 自带的 OCR:
  Windows.Media.Ocr 要装韩语/日语语言包才会出现对应引擎,而装语言包
  会给系统语言列表里多出输入法,影响用户切换键盘。PaddleOCR 的模型是
  自带的(首次运行时下载到用户目录),完全不碰系统语言设置。

注意:本脚本只是「测量工具」,不做任何屏幕常驻、不注入游戏、不模拟按键,
      与反作弊完全无交集。
================================================================================
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ==============================================================================
# 0. 控制台编码修正
# ------------------------------------------------------------------------------
# 依据:Windows 中文系统的控制台默认代码页是 936(GBK),GBK 里没有韩文/日文
# 假名。一旦直接 print 识别出来的韩文,Python 会抛 UnicodeEncodeError 直接崩掉,
# 探针就白跑了。所以这里强制把标准输出切成 UTF-8,并且用 errors="replace"
# 兜底 —— 宁可个别字显示成 ?,也不能让脚本崩。
# 配套:run_probe.bat 里会先执行 chcp 65001,让控制台本身也切到 UTF-8。
# ==============================================================================
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # 非交互式环境(如被重定向到文件)可能不支持,忽略即可


# ==============================================================================
# 1. 语种 / Unicode 区块工具
# ------------------------------------------------------------------------------
# 依据:游戏聊天极短(常常就一两个词),统计式语言识别(如 langdetect)在短文本上
# 非常不准。但日文和韩文有独一无二的 Unicode 区块特征,直接按码位判断是
# 免费、零延迟、且比任何 LID 库都可靠的方案。
# ==============================================================================
SCRIPTS: dict[str, List[Tuple[int, int]]] = {
    # 韩文音节(Hangul Syllables) + 韩文字母(Hangul Jamo)
    "hangul": [(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)],
    # 平假名 + 片假名
    "kana": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
    # CJK 统一汉字
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    # ASCII 字母
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A)],
}

# 用户请求的目标语种 → 期望出现的 Unicode 区块。
# 这个映射用来算「命中率」:如果你要的是韩文,而识别结果里一个韩文字符都没有,
# 那说明模型根本没认出来(而不是认错了)。
TARGET_SCRIPTS = {
    "ko": "hangul",
    "korean": "hangul",
    "ja": "kana",
    "japan": "kana",
    "japanese": "kana",
    "zh": "han",
    "ch": "han",
    "chinese": "han",
}


def classify_script(ch: str) -> Optional[str]:
    """判断单个字符属于哪个文字体系,不属于已知体系则返回 None。"""
    cp = ord(ch)
    for name, ranges in SCRIPTS.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def script_profile(text: str) -> Counter:
    """统计一段文本里各文字体系的字符数量(忽略空格、标点、数字)。"""
    profile: Counter = Counter()
    for ch in text:
        if ch.isspace():
            continue
        name = classify_script(ch)
        if name:
            profile[name] += 1
    return profile


def guess_lang(text: str) -> str:
    """
    按 Unicode 区块猜语种。
    顺序很重要:必须先查韩文和平假名/片假名,最后才查汉字。
    因为日文里混有大量汉字,如果先查汉字,日文会被误判成中文。
    """
    profile = script_profile(text)
    if profile.get("hangul"):
        return "ko"
    if profile.get("kana"):
        return "ja"
    if profile.get("han"):
        return "zh"
    if profile.get("latin"):
        return "en"
    return "?"


# ==============================================================================
# 2. 数据结构
# ==============================================================================
@dataclass
class OcrItem:
    """一行识别结果。box 是四点多边形(源图像素坐标),用于画框和排序。"""
    box: Optional[object]      # numpy.ndarray (4,2) 或 None
    text: str
    score: float

    @property
    def top(self) -> float:
        """取多边形最上沿的 y 值,用于把结果按屏幕上从上到下排序。"""
        try:
            return float(min(p[1] for p in self.box))
        except Exception:
            return 0.0

    @property
    def left(self) -> float:
        try:
            return float(min(p[0] for p in self.box))
        except Exception:
            return 0.0


@dataclass
class Timing:
    """分项计时。把各阶段拆开,才知道慢在哪、该优化谁。"""
    init: float = 0.0        # 引擎初始化(含首次运行下载模型)
    capture: float = 0.0     # 取图(截图 / 读文件)
    preprocess: float = 0.0  # 预处理
    infer: float = 0.0       # 纯 OCR 推理
    @property
    def total(self) -> float:
        return self.capture + self.preprocess + self.infer


# ==============================================================================
# 3. 取图:方式 A —— 从文件读取
# ==============================================================================
def load_from_file(path: str):
    """读图片文件。用 PIL 统一解码,顺便把 RGBA/调色板等格式规整成 RGB。"""
    from PIL import Image
    if not os.path.isfile(path):
        raise SystemExit(f"[x] 找不到图片文件: {path}")
    img = Image.open(path)
    # 依据:PNG 截图常带 alpha 通道。带 alpha 的图送去 OCR 可能出现
    # 「透明区域被当成黑色」的诡异结果,所以先铺一层深灰底再转 RGB。
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (18, 18, 22))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    return img


# ==============================================================================
# 4. 取图:方式 B —— 交互式框选截屏
# ==============================================================================
def get_virtual_screen() -> Tuple[int, int, int, int]:
    """
    取得「虚拟屏幕」的左上角坐标和尺寸,即所有显示器拼起来的整体范围。

    依据:只用 tkinter 的 winfo_screenwidth() 只能拿到主显示器的尺寸。
    如果你有第二块显示器在主屏左边,那它的坐标是负的,框选就会算错。
    所以这里直接问 Win32 拿真实的虚拟屏幕范围。
    GetSystemMetrics 的索引:76/77 = 虚拟屏幕原点 x/y,78/79 = 宽/高。
    """
    user32 = ctypes.windll.user32
    vx = user32.GetSystemMetrics(76)
    vy = user32.GetSystemMetrics(77)
    vw = user32.GetSystemMetrics(78)
    vh = user32.GetSystemMetrics(79)
    return vx, vy, vw, vh


def enable_dpi_awareness() -> None:
    """
    开启进程级 DPI 感知。

    依据(这是个很容易踩的坑):在高分屏 + 系统缩放到 125%/150% 时,如果进程
    不是 DPI 感知的,Windows 会把坐标「虚拟化」——tkinter 报告的是逻辑像素,
    而截图 API 用的是物理像素。结果就是框选的区域和实际截到的区域错位,
    而且缩放比例越大错得越多。必须在创建任何窗口之前设置。
    2 = PROCESS_PER_MONITOR_DPI_AWARE。
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def select_region_interactive() -> Tuple[int, int, int, int]:
    """
    弹一个半透明全屏遮罩,让用户拖拽框选一块区域,返回屏幕绝对坐标 (x1,y1,x2,y2)。

    为什么用 tkinter 而不是 mss/自定义窗口:
      tkinter 是 Python 自带模块,零额外依赖,而且做「拖拽画矩形」这件事足够用。
      为一个一次性的框选功能引入 GUI 框架不值得。
    """
    import tkinter as tk

    enable_dpi_awareness()          # 必须在 Tk() 之前调用,理由见函数注释
    vx, vy, vw, vh = get_virtual_screen()

    root = tk.Tk()
    root.overrideredirect(True)                  # 去掉标题栏和边框
    root.geometry(f"{vw}x{vh}+{vx}+{vy}")        # 覆盖整个虚拟屏幕
    root.attributes("-topmost", True)            # 压在最上层
    root.attributes("-alpha", 0.30)              # 半透明,方便看清底下的游戏画面
    root.configure(bg="black")

    canvas = tk.Canvas(root, cursor="cross", bg="black",
                       highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)

    # 提示文字。用中文和韩日字体都支持的字体,避免显示成方块。
    canvas.create_text(
        vw // 2, 80,
        text="拖拽框选聊天区域    ·    Esc 取消",
        fill="#00FF88", font=("Microsoft YaHei", 22, "bold"),
    )

    state = {"start": None, "rect": None, "bbox": None}

    def on_press(ev):
        # 用 x_root/y_root(屏幕绝对坐标)而不是 x/y(窗口内坐标),
        # 这样就不必手动加上窗口原点偏移,少一处出错的地方。
        state["start"] = (ev.x_root, ev.y_root)
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            ev.x_root - vx, ev.y_root - vy, ev.x_root - vx, ev.y_root - vy,
            outline="#00FF88", width=3,
        )

    def on_drag(ev):
        if not state["start"]:
            return
        x0, y0 = state["start"]
        canvas.coords(state["rect"],
                      x0 - vx, y0 - vy, ev.x_root - vx, ev.y_root - vy)

    def on_release(ev):
        if not state["start"]:
            return
        x0, y0 = state["start"]
        x1, y1 = ev.x_root, ev.y_root
        # 依据:用户可能从右下往左上拖,所以必须归一化,否则宽高会变成负数。
        state["bbox"] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        root.destroy()

    def on_escape(_ev):
        state["bbox"] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    # 依据:overrideredirect(无边框)窗口不保证自动获得键盘焦点。如果焦点没落到
    # 这个窗口,Escape 就收不到,用户会被永久困在全屏遮罩里 —— 必须避免。
    # 所以 root 和 canvas 两边都绑,并且强制抢一次焦点。
    root.bind("<Escape>", on_escape)
    canvas.bind("<Escape>", on_escape)
    root.focus_force()
    canvas.focus_set()

    root.mainloop()

    bbox = state["bbox"]
    if not bbox:
        raise SystemExit("[x] 已取消框选。")
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 5 or y2 - y1 < 5:
        raise SystemExit("[x] 框选区域太小了(至少 5x5 像素)。")
    return x1, y1, x2, y2


def grab_region(bbox: Tuple[int, int, int, int]):
    """
    按屏幕绝对坐标截图。

    依据(这里有个容易错的地方):Windows 上某些 API 的 bbox 是相对「主显示器」
    的,而虚拟屏幕原点在多显示器布局下可能不是 (0,0)(比如副屏放在主屏左边,
    原点就是负的)。为了不依赖这些细节,这里干脆先整屏抓下来,
    再用「相对虚拟屏幕原点」的坐标去裁 —— 逻辑唯一,不会错。
    代价是每次多抓一次全屏,一次性框选场景下完全可以接受。
    """
    from PIL import ImageGrab

    time.sleep(0.18)  # 等遮罩窗口真正从屏幕上消失,否则会把绿色选框也截进去

    vx, vy, vw, vh = get_virtual_screen()
    full = ImageGrab.grab(bbox=(vx, vy, vx + vw, vy + vh), all_screens=True)

    x1, y1, x2, y2 = bbox
    rel = (x1 - vx, y1 - vy, x2 - vx, y2 - vy)
    return full.crop(rel).convert("RGB")


# ==============================================================================
# 5. 取图:方式 C —— 自检用的合成图
# ==============================================================================
FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# 按优先级列出各语种可用的系统字体(取第一个存在的)。
FONT_CANDIDATES = {
    "ko": ["malgun.ttf", "gulim.ttc", "batang.ttc", "Dotum.ttf"],
    "ja": ["YuGothM.ttc", "meiryo.ttc", "msgothic.ttc", "msmincho.ttc", "YuMincho.ttc"],
    "zh": ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"],
}

# 仿游戏聊天的真实语料:短句、缩写、口语。
SAMPLE_LINES = {
    "ko": ["힐 좀 줘", "적 뒤에 있어", "궁 썼어", "잘했어!", "미안 ㅠㅠ", "리그 들어와"],
    "ja": ["ナイス!", "回復お願い", "後ろに敵いる", "うますぎ", "ありがとう", "押し込もう"],
    "zh": ["给我点治疗", "后面有人", "大招放了", "打得好!", "抱歉", "一起推"],
}

# 依据(踩过的坑):玩家名必须用「该语种字体真的含有的字形」。
# 最初我统一用韩文名字配日文字体渲染,而 Yu Gothic 里没有韩文字形,
# 渲染出来是缺字方块/fallback 字形,OCR 当然认不出来。
# 结果就是「识别失败」的假象 —— 那是测试图的问题,不是 OCR 的问题。
# 这类假阴性如果不排除,会直接导出错误结论,所以按语种分别给名字。
SAMPLE_NAMES = {
    "ko": ["김철수", "이영희", "박민수", "최지우", "Player"],
    "ja": ["佐藤", "田中", "鈴木", "高橋", "Player"],
    "zh": ["小明", "小红", "老王", "Player"],
}


def find_font(lang: str) -> Optional[str]:
    for name in FONT_CANDIDATES.get(lang, []):
        p = os.path.join(FONT_DIR, name)
        if os.path.isfile(p):
            return p
    return None


def make_selftest_image(lang: str):
    """
    现场合成一张「仿游戏聊天」截图。

    依据:探针真正要验证的是「在游戏那种条件下能不能认出来」,所以合成图要尽量
    复刻游戏的视觉特征:
      · 深色半透明底(游戏聊天区是叠在动态画面上的)
      · 白色文字 + 深色描边(游戏为了保证可读性都这么画)
      · 字号偏小(这才是 OCR 的真正难点)
      · 每行带一个 [玩家名]: 前缀,和守望先锋的聊天格式一致
    如果连这张「干净版」都认不出来,那就没必要去测真实截图了 —— 说明是
    模型或语言设置的问题,而不是背景干扰的问题。这个区分很有价值。
    """
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_font(lang)
    if not font_path:
        raise SystemExit(
            f"[x] 系统里找不到 {lang} 的字体,无法生成自检图。\n"
            f"    已尝试: {FONT_CANDIDATES.get(lang)}\n"
            f"    可以直接用 --region 框选真实截图来代替。"
        )

    lines = SAMPLE_LINES.get(lang, SAMPLE_LINES["zh"])
    names = SAMPLE_NAMES.get(lang, SAMPLE_NAMES["zh"])

    # 依据:字号 20px 大约是游戏里聊天文字的实际大小。故意用小字号,
    # 因为「小字号」是 OCR 的头号杀手,乐观的测试没有意义。
    size = 20
    font = ImageFont.truetype(font_path, size)

    pad, line_h = 10, size + 10
    width = 460
    height = pad * 2 + line_h * len(lines)

    img = Image.new("RGB", (width, height), (22, 24, 30))  # 深色底
    draw = ImageDraw.Draw(img)

    expected: List[str] = []
    for i, text in enumerate(lines):
        y = pad + i * line_h
        line = f"[{names[i % len(names)]}]: {text}"
        expected.append(line)   # 记下标准答案,后面用来算真实准确率
        # 先画一圈深色描边(往 8 个方向偏移各画一次),再画白色正字。
        # 这就是游戏 UI 保证文字在任何背景上都可读的做法,OCR 也受益于它。
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((pad + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((pad, y), line, font=font, fill=(255, 255, 255))

    # 返回标准答案 —— 自检模式的价值就在于「答案已知」,可以量化准确率,
    # 而不是靠肉眼看「好像还行」。
    return img, expected


# ==============================================================================
# 6. 预处理
# ------------------------------------------------------------------------------
# 依据:OCR 模型内部的识别网络会把每行文字缩放固定高度(通常 48px)。原图里
# 文字越小,缩放后的信息损失越大。所以「先放大再识别」往往能显著提升小字
# 的准确率 —— 这也是本探针要重点验证的一项。
# ==============================================================================
def otsu_threshold(gray_arr) -> int:
    """
    用大津法(Otsu)自动求二值化阈值,不需要 opencv。
    原理:遍历所有可能阈值,选「前景和背景两类之间方差最大」的那个。
    对「白字 + 动态背景」这种双峰分布特别合适。
    """
    import numpy as np
    hist = np.bincount(gray_arr.ravel(), minlength=256).astype(np.float64)
    total = gray_arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2   # 类间方差
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def preprocess(img, mode: str, scale: float):
    """
    返回预处理后的 PIL 图像。

    可选的 mode:
      none     —— 不做任何处理(最重要的对照组)
      upscale  —— 只放大
      gray     —— 放大 + 转灰度
      contrast —— 放大 + 灰度 + 自动对比度拉伸
      binary   —— 放大 + 灰度 + 大津法二值化(把动态背景压成纯黑白)
      auto     —— 放大 + 灰度 + 对比度拉伸(推荐的默认起手式)
    """
    from PIL import Image, ImageOps

    if mode == "none":
        return img

    out = img
    if scale and scale != 1.0:
        # LANCZOS 是重采样质量最高的滤波器。对小字放大,质量比速度重要。
        out = out.resize(
            (max(1, int(out.width * scale)), max(1, int(out.height * scale))),
            Image.LANCZOS,
        )

    if mode == "upscale":
        return out.convert("RGB")

    out = out.convert("L")  # 转灰度:颜色对文字识别没有信息量,去掉反而减少干扰

    if mode == "gray":
        return out.convert("RGB")

    if mode == "binary":
        import numpy as np
        arr = np.array(out)
        t = otsu_threshold(arr)
        # 依据:游戏聊天是「浅色字 + 深色描边」。阈值化之后保留亮的那一类,
        # 背景里的花哨画面就被压掉了,只留下文字形状。
        mask = (arr > t).astype(np.uint8) * 255
        return Image.fromarray(mask, mode="L").convert("RGB")

    # contrast / auto:自动对比度拉伸。cutoff=1 表示各裁掉 1% 的最暗和最亮像素,
    # 这样少量的纯黑背景或高光不会把拉伸范围压死。
    out = ImageOps.autocontrast(out, cutoff=1)
    return out.convert("RGB")


# ==============================================================================
# 7. OCR 引擎封装
# ==============================================================================
def _obj_get(obj, key, default=None):
    """兼容 dict 和对象两种访问方式(PaddleOCR 不同版本返回类型不一样)。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class PaddleEngine:
    """
    PaddleOCR 封装 —— **备选引擎**,需要额外安装(paddlepaddle + paddlex,约 1 GB)。

    实测在这台机器上 pip 解析它的依赖树会长时间空转卡死,而它和默认引擎
    跑的是**同一套 PP-OCR 模型**(只是模型格式不同:paddle 格式 vs ONNX),
    所以结论不会因为用哪个引擎而改变。除非你有特别理由,用默认的 rapidocr 即可。

    保留它的意义:如果哪天 ONNX 路线出问题,可以换这条官方实现的路验证一下。

    为什么要用 inspect 过滤参数:PaddleOCR 2.x 和 3.x 的构造参数改名了
    (比如 use_angle_cls → use_textline_orientation)。写死参数会在另一个
    大版本上直接 TypeError。用反射只传当前版本真正支持的参数,2.x / 3.x 都能跑。
    """

    name = "paddle"

    # --------------------------------------------------------------------
    # 【关键】PaddleOCR 的语种代码和我们命令行的简写不是一回事,必须映射。
    # 依据:传 'ko' 给 PaddleOCR 是无效值 —— 它既不报错,也不会给你韩文模型,
    # 而是静默退回默认的中英模型。于是你会看到一堆乱码/拉丁字母,却完全不知道
    # 问题出在哪(这正是 OCR 调试里最浪费时间的一类坑)。
    # 每个语种给多个候选,是为了兼容 PaddleOCR 不同版本对日文代码的写法差异。
    # --------------------------------------------------------------------
    LANG_CANDIDATES = {
        "ko": ["korean"],
        "korean": ["korean"],
        "ja": ["japan", "japanese"],
        "japan": ["japan", "japanese"],
        "japanese": ["japan", "japanese"],
        "zh": ["ch", "chinese"],
        "ch": ["ch", "chinese"],
        "chinese": ["ch", "chinese"],
        "en": ["en"],
    }

    def __init__(self, lang: str, **_ignored):
        import inspect
        import logging
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise SystemExit(
                "[x] 没装 paddleocr(备选引擎)。\n"
                "    它需要 paddlepaddle + paddlex,约 1 GB,而且实测在这台机器上\n"
                "    pip 解析它的依赖树会长时间卡住。除非有特别理由,直接用默认的\n"
                "    --engine rapidocr 即可,两者跑的是同一套 PP-OCR 模型。\n"
                "    确实要装:.venv\\Scripts\\python.exe -m pip install paddlepaddle paddleocr"
            )

        # 关掉 PaddleOCR 内部啰嗦的日志,否则输出里全是它自己的 INFO
        for lg in ("ppocr", "paddleocr", "paddlex"):
            logging.getLogger(lg).setLevel(logging.ERROR)

        sig = set(inspect.signature(PaddleOCR.__init__).parameters)
        base_kwargs: dict = {}

        # 依据:游戏截图是正向的纯文字块,既不需要做 90/270 度文档方向分类,
        # 也不需要做弯曲文档矫正。关掉这几个能省下两次模型下载和额外推理时间。
        for k in ("use_doc_orientation_classify", "use_doc_unwarping",
                  "use_textline_orientation"):
            if k in sig:
                base_kwargs[k] = False
        if "use_angle_cls" in sig:      # 2.x 的等价开关
            base_kwargs["use_angle_cls"] = False
        if "show_log" in sig:           # 2.x 的控制台日志开关
            base_kwargs["show_log"] = False
        if "use_gpu" in sig:
            base_kwargs["use_gpu"] = False   # 先跑 CPU,GPU 留到确定要优化时再说

        # 依次尝试候选语种代码,第一个能构造成功的就用它。
        candidates = self.LANG_CANDIDATES.get(lang, [lang])
        errors: list = []
        self._ocr = None
        for code in candidates:
            try:
                kwargs = dict(base_kwargs)
                if "lang" in sig:
                    kwargs["lang"] = code
                self._ocr = PaddleOCR(**kwargs)
                print(f"      PaddleOCR 语种代码 = '{code}'")
                break
            except Exception as e:      # noqa: BLE001 — 要收集并逐个尝试
                errors.append(f"{code}: {type(e).__name__}: {e}")

        if self._ocr is None:
            raise SystemExit(
                "[x] PaddleOCR 初始化失败,候选语种代码全部无效:\n    "
                + "\n    ".join(errors)
            )

        self._is_v3 = hasattr(self._ocr, "predict")

    def run(self, img) -> List[OcrItem]:
        import numpy as np
        # 依据:PIL 给的是 RGB,而 PaddleOCR 内部按 OpenCV 的 BGR 约定处理数组。
        # 文字识别对通道顺序不敏感,但保持约定一致可以避免以后接别的后处理时踩坑。
        # 另外必须 .copy():[::-1] 得到的是负步长的视图,不是连续内存,
        # 某些推理后端会直接报错。ascontiguousarray 保证内存连续。
        arr = np.ascontiguousarray(np.array(img)[:, :, ::-1])

        items: List[OcrItem] = []

        if self._is_v3:
            # ---- PaddleOCR 3.x ----
            try:
                raw = self._ocr.predict(arr)
            except TypeError:
                raw = self._ocr.predict(input=arr)
            for page in raw or []:
                texts = _obj_get(page, "rec_texts") or []
                scores = _obj_get(page, "rec_scores") or []
                polys = _obj_get(page, "rec_polys")
                if polys is None:
                    polys = _obj_get(page, "dt_polys") or []
                for i, t in enumerate(texts):
                    s = float(scores[i]) if i < len(scores) else 0.0
                    box = np.asarray(polys[i], dtype=float) if i < len(polys) else None
                    items.append(OcrItem(box=box, text=str(t), score=s))
        else:
            # ---- PaddleOCR 2.x ----
            raw = self._ocr.ocr(arr, cls=False)
            for page in raw or []:
                for line in page or []:
                    box, (text, score) = line[0], line[1]
                    items.append(OcrItem(
                        box=np.asarray(box, dtype=float),
                        text=str(text), score=float(score),
                    ))
        return items


class RapidOcrEngine:
    """
    RapidOCR(ONNX Runtime)引擎封装 —— 本项目的主力引擎。

    为什么选它:
      · 依赖轻(onnxruntime + opencv),不需要装 paddlepaddle 那一整套框架;
      · 它跑的是 **PP-OCR 的 ONNX 模型** —— 和 C# 版 RapidOcrNet 用的是同一套模型,
        所以这里量出来的准确率和耗时,对后面的 C# 实现基本可以照搬。

    语种是怎么处理的(这是全项目最关键的一个知识点):
      · 检测(det)和方向分类(cls)是**与语种无关**的 —— 文字长什么样和什么语言没关系,
        所以直接复用包内自带的模型;
      · 只有识别(rec)是**分语种**的,必须换成对应的模型;
      · ★ rec 的 .onnx 模型和 .txt 字典**必须成对使用**。字典定义了模型能输出
        哪些字符,两者不匹配时会输出一堆乱码 —— 而且不报错,极难自己发现。

    模型来源:由 download_models.py 从 ModelScope 下载(已校验 SHA256),
    放在项目根的 models\\ 目录下。
    """

    name = "rapidocr"

    # 语种 → (识别模型文件名, 字典文件名);None 表示用包内自带的中英模型
    REC_MODELS = {
        # ---- 按语种自动选(默认路径)----
        "ko": ("korean_PP-OCRv4_rec_mobile.onnx", "korean_dict.txt"),
        "korean": ("korean_PP-OCRv4_rec_mobile.onnx", "korean_dict.txt"),
        "ja": ("japan_PP-OCRv4_rec_mobile.onnx", "japan_dict.txt"),
        "japan": ("japan_PP-OCRv4_rec_mobile.onnx", "japan_dict.txt"),
        "japanese": ("japan_PP-OCRv4_rec_mobile.onnx", "japan_dict.txt"),
        "zh": None, "ch": None, "chinese": None, "en": None,

        # ---- 用 --rec-model 显式指定,专门用来做模型横向对比 ----
        # 这些键只是给模型起的短名字,和语种没有绑定关系。
        "ko4": ("korean_PP-OCRv4_rec_mobile.onnx", "korean_dict.txt"),
        "ko5": ("korean_PP-OCRv5_rec_mobile.onnx", "ppocrv5_korean_dict.txt"),
        "ja4": ("japan_PP-OCRv4_rec_mobile.onnx", "japan_dict.txt"),
        "v6": ("PP-OCRv6_rec_small.onnx", "ppocrv6_dict.txt"),
    }

    def __init__(self, lang: str, models_dir: Optional[str] = None,
                 det_limit: Optional[float] = None, use_cls: bool = True,
                 rec_model: Optional[str] = None):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise SystemExit(
                "[x] 没装 rapidocr-onnxruntime。\n"
                "    先跑一次 setup_ocr.bat,或执行:\n"
                "      .venv\\Scripts\\python.exe -m pip install pillow numpy rapidocr-onnxruntime"
            )

        here = os.path.dirname(os.path.abspath(__file__))
        models_dir = models_dir or os.path.join(here, "models")

        kwargs: dict = {}
        # 依据:--rec-model 用来绕开「语种 → 模型」的默认映射,直接指定测哪个识别模型。
        # 做模型横向对比时必须能这样,否则同一张图只能跑一个语种对应的模型,
        # 没法回答「换 v5 / 换 v6 之后到底变好还是变差」。
        key = rec_model or lang
        if rec_model and rec_model not in self.REC_MODELS:
            raise SystemExit(
                f"[x] 未知的 --rec-model: {rec_model}\n"
                "    可选: ko4(韩v4) / ko5(韩v5) / ja4(日v4) / v6(v6多语言)"
            )
        spec = self.REC_MODELS.get(key)
        if spec:
            rec_name, dict_name = spec
            rec_path = os.path.join(models_dir, rec_name)
            dict_path = os.path.join(models_dir, dict_name)
            missing = [p for p in (rec_path, dict_path) if not os.path.isfile(p)]
            if missing:
                raise SystemExit(
                    f"[x] 缺少 {key} 的识别模型文件:\n"
                    + "".join(f"      {p}\n" for p in missing)
                    + "    先运行:  .venv\\Scripts\\python.exe download_models.py"
                )
            # 依据:kwargs 里 rec_ 前缀会被 RapidOCR 分流到配置的 Rec 段,
            # rec_model_path → Rec.model_path,rec_keys_path → Rec.rec_keys_path。
            # 检测和分类模型不传,继续用包内自带的,它们与语种无关。
            kwargs["rec_model_path"] = rec_path
            kwargs["rec_keys_path"] = dict_path
            print(f"      识别模型: {rec_name}")
            print(f"      字典文件: {dict_name}")
        else:
            print(f"      识别模型: 包内自带中英模型(`{lang}` 不需要韩日模型)")

        if det_limit:
            # 依据(重要优化点):包内默认是 limit_type=min + limit_side_len=736,
            # 意思是「把图片短边放大到 736」。对一张 460x200 的聊天区截图,
            # 等于内部先放大 3.7 倍再做检测 —— 这是推理耗时的主要来源。
            # 聊天区文字本来就不算小,把这个值调低通常能大幅提速而几乎不掉准确率。
            # 本探针提供这个开关,就是为了把「速度/准确率」的取舍量出来。
            kwargs["det_limit_side_len"] = det_limit
            print(f"      检测短边上限: {det_limit:g}(包内默认 736,调低可提速)")
        if not use_cls:
            # 依据:游戏聊天文字永远是正向的,不会 180 度颠倒,
            # 关掉方向分类能省掉一次额外的模型推理。
            kwargs["use_cls"] = False
            print("      方向分类 cls: 已关闭(游戏文字不会倒置)")

        self._ocr = RapidOCR(**kwargs)

    def run(self, img) -> List[OcrItem]:
        import numpy as np
        # 依据:RapidOCR 内部用 OpenCV 约定,期望 BGR 的连续数组。
        # PIL 给的是 RGB,所以翻转通道;[::-1] 得到的是负步长视图而非连续内存,
        # 必须 ascontiguousarray 一下,否则推理后端可能直接报错。
        arr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
        result, _elapse = self._ocr(arr)
        items: List[OcrItem] = []
        for box, text, score in (result or []):
            items.append(OcrItem(box=np.asarray(box, dtype=float),
                                 text=str(text), score=float(score)))
        return items


ENGINES = {"paddle": PaddleEngine, "rapidocr": RapidOcrEngine}


def build_engine(name: str, lang: str, **opts):
    if name not in ENGINES:
        raise SystemExit(f"[x] 未知引擎 {name},可选: {list(ENGINES)}")
    return ENGINES[name](lang, **opts)


# ==============================================================================
# 8. 结果输出
# ==============================================================================
def print_results(items: List[OcrItem], lang: str) -> None:
    """把结果按屏幕上「从上到下、从左到右」的顺序打印出来。"""
    if not items:
        print("  (没有识别到任何文字)")
        return

    # 依据:OCR 返回的顺序不保证和视觉顺序一致。按多边形左上角排序,
    # 输出才和你在屏幕上看到的聊天行顺序对得上,方便逐行核对。
    ordered = sorted(items, key=lambda it: (round(it.top / 12), it.left))

    print(f"  {'#':>3}  {'置信度':>6}  {'猜测':>4}  识别文本")
    print("  " + "-" * 72)
    for i, it in enumerate(ordered, 1):
        txt = it.text
        print(f"  {i:>3}  {it.score:>6.3f}  {guess_lang(txt):>4}  {txt}")
    print("  " + "-" * 72)


def quality_report(items: List[OcrItem], lang: str) -> dict:
    """
    计算「目标语种命中率」—— 这是判断 OCR 到底有没有认出来的关键指标。

    依据:如果模型语言配错了(比如拿中英模型去认韩文),它不会报错,而是
    返回一堆看起来像文字、但完全不对的拉丁字母或符号。光看「识别到 6 行」
    会被骗。所以要看这 6 行里有多少行真的含有目标语种的字符。
    """
    target = TARGET_SCRIPTS.get(lang)
    if not items or not target:
        return {"target": target, "lines": len(items), "hits": 0, "ratio": 0.0}

    hits = 0
    for it in items:
        if script_profile(it.text).get(target):
            hits += 1

    scores = [it.score for it in items]
    return {
        "target": target,
        "lines": len(items),
        "hits": hits,
        "ratio": hits / len(items),
        "avg_score": sum(scores) / len(scores),
        "chars": sum(len(it.text) for it in items),
    }


def levenshtein(a: str, b: str) -> int:
    """编辑距离:把 a 改成 b 需要多少次增/删/改。手写实现,避免引入额外依赖。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,                            # 删掉 a[i]
                cur[j - 1] + 1,                         # 插入 b[j]
                prev[j - 1] + (0 if ca == cb else 1),   # 替换
            ))
        prev = cur
    return prev[-1]


def accuracy_report(items: List[OcrItem], expected: List[str]) -> dict:
    """
    自检模式专用:把识别结果和「合成图时画上去的标准答案」逐行对比,算真实准确率。

    依据:只统计「有没有认出目标语种字符」是不够的。例如
    `[김철수]: 미안 ㅠㅠ` 被识别成 `I[中ㄸ]`,里面照样混着韩文字符,
    按命中率算它「通过」,但它其实是错的 —— 命中率会把假阳性算成成功。
    自检模式下标准答案是已知的,所以可以直接算字符级准确率
    (1 - 编辑距离 / 答案长度),这才是能拿来做决策的硬指标。

    比较前去掉所有空白字符:OCR 本来就不输出空格,不去掉会白扣分。
    """

    def norm(s: str) -> str:
        return "".join(ch for ch in s if not ch.isspace())

    ordered = sorted(items, key=lambda it: (round(it.top / 12), it.left))
    pairs = list(zip(expected, ordered))

    per_line = []
    total_edit, total_len, exact = 0, 0, 0
    for exp, it in pairs:
        e, g = norm(exp), norm(it.text)
        d = levenshtein(e, g)
        if e == g:
            exact += 1
        total_edit += d
        total_len += len(e)
        per_line.append((exp, it.text, 1.0 - d / max(1, len(e))))

    return {
        "expected_lines": len(expected),
        "matched_lines": len(pairs),
        "exact": exact,
        "char_accuracy": 1.0 - total_edit / max(1, total_len),
        "per_line": per_line,
    }


def save_annotated(img, items: List[OcrItem], path: str) -> None:
    """
    把识别框画回图上再存盘。

    依据:数字(置信度 0.93)不能告诉你「框歪到哪去了」。看一眼标注图就能
    立刻区分三类问题:框位置对但认错字(模型问题)、框切断了文字(检测问题)、
    根本没框(没检测到)。这是调 OCR 时最省时间的工具。
    """
    from PIL import ImageDraw
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for it in items:
        if it.box is None:
            continue
        pts = [tuple(p) for p in it.box]
        if len(pts) >= 3:
            draw.polygon(pts, outline=(0, 255, 136))
    canvas.save(path)


# ==============================================================================
# 9. 主流程
# ==============================================================================
def build_args(argv: Sequence[str]):
    p = argparse.ArgumentParser(
        description="游戏聊天区 OCR 可行性探针(韩文/日文识别测试)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自检:用系统字体合成一张仿游戏聊天图来跑,确认整条链路是否通
  python ocr_probe.py --selftest --lang ko

  # 框选屏幕区域并识别(最贴近真实用法)
  python ocr_probe.py --region --lang ja

  # 识别已有截图,并跑 5 次测平均速度
  python ocr_probe.py --image shot.png --lang ko --repeat 5

  # 对比两种预处理对准确率的影响
  python ocr_probe.py --image shot.png --lang ko --preprocess none
  python ocr_probe.py --image shot.png --lang ko --preprocess binary
""",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", metavar="PATH", help="读一张已有图片")
    src.add_argument("--region", action="store_true", help="交互式框选屏幕区域并截图")
    src.add_argument("--selftest", action="store_true",
                     help="用系统字体合成仿游戏聊天图来自检")

    p.add_argument("--lang", default="ko",
                   choices=["ko", "ja", "zh", "en", "korean", "japanese", "chinese"],
                   help="要识别的语种(默认 ko)")
    p.add_argument("--engine", default="rapidocr", choices=list(ENGINES),
                   help="OCR 引擎(默认 rapidocr;paddle 是备选,需要额外安装且体积很大)")
    p.add_argument("--preprocess", default="none",
                   choices=["none", "upscale", "gray", "contrast", "binary", "auto"],
                   help="预处理方式(默认 none,即不做处理,作为对照)")
    p.add_argument("--scale", type=float, default=3.0,
                   help="预处理时的放大倍数(默认 3.0)")
    p.add_argument("--rec-model", default=None,
                   help="强制指定识别模型做对比:ko4 / ko5 / ja4 / v6(默认按 --lang 选)")
    p.add_argument("--det-limit", type=float, default=None,
                   help="检测阶段短边放大上限(包内默认 736;调低可显著提速)")
    p.add_argument("--no-cls", action="store_true",
                   help="关闭方向分类(游戏文字不会倒置,关掉可省一次推理)")
    p.add_argument("--repeat", type=int, default=1,
                   help="重复识别次数,用来测平均速度(默认 1)")
    p.add_argument("--save", metavar="PATH",
                   help="把取到的图存成 PNG(建议加上,方便反复调试同一张图)")
    p.add_argument("--annotated", metavar="PATH",
                   help="把识别框画到图上另存一份")
    p.add_argument("--json", dest="json_out", metavar="PATH",
                   help="把结果写成 JSON(避免控制台字体显示不了韩日文)")
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = build_args(argv)
    timing = Timing()

    print("=" * 78)
    print(f"  OCR 可行性探针    引擎={args.engine}  语种={args.lang}  预处理={args.preprocess}")
    print("=" * 78)

    # ---------- 步骤 1:取图 ----------
    # 自检模式下我们手里有标准答案,可以算真实的字符级准确率
    expected: Optional[List[str]] = None
    t0 = time.perf_counter()
    if args.image:
        print(f"\n[1/4] 读图片文件: {args.image}")
        img = load_from_file(args.image)
    elif args.region:
        print("\n[1/4] 请在屏幕上拖拽框选聊天区域(Esc 取消)…")
        bbox = select_region_interactive()
        print(f"      框选完成: {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} 像素")
        img = grab_region(bbox)
    else:
        print(f"\n[1/4] 自检模式:用系统字体合成 {args.lang} 仿游戏聊天图")
        img, expected = make_selftest_image(args.lang)
    timing.capture = time.perf_counter() - t0
    print(f"      图像尺寸: {img.width} x {img.height}   耗时 {timing.capture*1000:.0f} ms")

    if args.save:
        img.save(args.save)
        print(f"      已保存原图: {args.save}")

    # ---------- 步骤 2:初始化引擎 ----------
    print(f"\n[2/4] 初始化 {args.engine} 引擎…(首次运行会联网下载模型,可能要几分钟)")
    t0 = time.perf_counter()
    engine = build_engine(args.engine, args.lang, det_limit=args.det_limit,
                          use_cls=not args.no_cls, rec_model=args.rec_model)
    timing.init = time.perf_counter() - t0
    print(f"      初始化耗时 {timing.init:.2f} s"
          f"{'  ← 含首次模型下载' if timing.init > 20 else ''}")
    print("      注:初始化是一次性的,正式工具里只做一次,不进入每句话的延迟")

    # ---------- 步骤 3:预处理 + 识别 ----------
    print(f"\n[3/4] 预处理({args.preprocess},放大 {args.scale}x)+ 识别")
    t0 = time.perf_counter()
    proc = preprocess(img, args.preprocess, args.scale)
    timing.preprocess = time.perf_counter() - t0
    if proc.size != img.size:
        print(f"      预处理后尺寸: {proc.width} x {proc.height}   "
              f"耗时 {timing.preprocess*1000:.0f} ms")

    items: List[OcrItem] = []
    repeat = max(1, args.repeat)
    for i in range(repeat):
        t0 = time.perf_counter()
        items = engine.run(proc)
        dt = time.perf_counter() - t0
        timing.infer = dt   # 记最后一次
        if repeat > 1:
            print(f"      第 {i+1}/{repeat} 次推理: {dt*1000:.0f} ms")

    # ---------- 步骤 4:结果 ----------
    print(f"\n[4/4] 识别结果   推理耗时 {timing.infer*1000:.0f} ms")
    print()
    print_results(items, args.lang)
    print()

    rep = quality_report(items, args.lang)
    if rep.get("target"):
        target_name = {"hangul": "韩文", "kana": "日文假名", "han": "汉字"}.get(
            rep["target"], rep["target"])
        print(f"  目标语种命中率: {rep['hits']}/{rep['lines']} 行含{target_name}"
              f"({rep['ratio']*100:.0f}%)")
        print(f"  平均置信度: {rep.get('avg_score', 0):.3f}    识别字符数: {rep.get('chars', 0)}")
        if rep["ratio"] < 0.5:
            print("  [!] 命中率偏低。可能原因:①识别模型语种不对(检查 --lang);"
                  "②字号太小(试 --preprocess binary);③背景干扰太强")
        elif rep["ratio"] >= 0.8:
            print("  [√] 命中率良好,这个语种的 OCR 路线可行。")

    # ---- 自检模式:输出与标准答案的逐行对比(最硬的指标)----
    if expected:
        acc = accuracy_report(items, expected)
        print()
        print("  ── 与标准答案逐行对比(自检模式)──")
        for exp, got, a in acc["per_line"]:
            flag = "√" if a >= 0.95 else ("~" if a >= 0.7 else "×")
            print(f"   {flag} {a*100:5.1f}%  期望: {exp}")
            print(f"             实际: {got}")
        print(f"\n  字符级准确率: {acc['char_accuracy']*100:.1f}%"
              f"    整行完全正确: {acc['exact']}/{acc['expected_lines']}")
        print("  注:命中率只看『有没有认出目标语种字符』,会把假阳性算成成功;"
              "\n      字符级准确率才是能不能用的判据。")

    print("\n  ── 耗时汇总 ──")
    print(f"  取图     {timing.capture*1000:8.0f} ms")
    print(f"  预处理   {timing.preprocess*1000:8.0f} ms")
    print(f"  推理     {timing.infer*1000:8.0f} ms")
    print(f"  单次总计 {timing.total*1000:8.0f} ms   (不含一次性初始化 {timing.init:.2f} s)")

    if args.annotated:
        save_annotated(proc, items, args.annotated)
        print(f"\n  已保存标注图: {args.annotated}   ← 打开看一眼框准不准,这个最有用")

    if args.json_out:
        payload = {
            "engine": args.engine, "lang": args.lang,
            "preprocess": args.preprocess, "scale": args.scale,
            "timing_ms": {
                "capture": timing.capture * 1000,
                "preprocess": timing.preprocess * 1000,
                "infer": timing.infer * 1000,
                "init_s": timing.init,
            },
            "quality": rep,
            "items": [
                {"text": it.text, "score": it.score, "lang_guess": guess_lang(it.text)}
                for it in items
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  已保存 JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n已中断。")
        raise SystemExit(130)
