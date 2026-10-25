#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup_env.py —— 建虚拟环境 + 装依赖 + 下载模型
================================================================================
由 setup_ocr.bat 用系统 Python 调起,所有真正的逻辑都在这里。

为什么逻辑不写在 .bat 里(踩过的坑,值得记下来):
  最初把中文提示和 `chcp 65001` 都写在 .bat 里,结果一运行就报一堆
  「'xxx' 不是内部或外部命令」。原因不是编码乱码那么简单 ——
  **cmd.exe 是按字节偏移逐行读取批处理文件的**:执行到 `chcp` 切换代码页后,
  它下一次续读的偏移量会落在某个多字节字符的中间,从此行边界全部错位,
  `REM` 前缀被吃掉,注释内容就被当成命令去执行了。
  所以现在 .bat 保持**纯 ASCII 且只有十几行启动逻辑**,把一切内容放进 Python,
  从根上消除这一类问题。
================================================================================
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, ".venv")

# Windows 是 Scripts\python.exe,其他平台是 bin/python。这里都兼容一下,
# 免得以后换环境要改代码。
if os.name == "nt":
    VENV_PY = os.path.join(VENV_DIR, "Scripts", "python.exe")
else:
    VENV_PY = os.path.join(VENV_DIR, "bin", "python")

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEPS = ["pillow", "numpy", "rapidocr-onnxruntime"]

# 所有 pip 调用共用的参数。
# 依据:必须带 --retries/--timeout。国内网络下 pip 默认会长时间挂在一个
# 连不通的连接上,表现为「CPU 一直涨、网络却完全不动」,看起来像卡死。
# 注意 index-url 不放在这里,因为官方源回退时要用默认源。
PIP_INSTALL = [VENV_PY, "-m", "pip", "install",
               "--disable-pip-version-check", "--progress-bar", "off",
               "--retries", "2", "--timeout", "20"]

TOTAL_STEPS = 4


def step(n: int, title: str) -> None:
    print(f"\n[{n}/{TOTAL_STEPS}] {title}")


def run(cmd: list[str]) -> int:
    """
    执行外部命令,直接继承标准输入输出。

    依据:不用 capture_output,让 pip 的进度条原样打到控制台上 ——
    装包要下载一百多 MB,用户需要看到"它在动"。
    另外继承 stdio 也顺带避开了管道相关的权限问题。
    """
    print("      $ " + " ".join(cmd))
    # 依据:父进程的 stdout 在被重定向/管道时是「块缓冲」的,而子进程直接写
    # 同一个句柄。不先 flush 的话,子进程的输出会先冒出来,自己的提示反而
    # 排在后面 —— 看起来就像执行顺序乱了。每起一个子进程前先冲一次。
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.run(cmd, cwd=HERE).returncode


def deps_ready() -> bool:
    """
    探测虚拟环境里的依赖是否已经可用(能 import 就算就绪)。

    依据:即使所有包都已装好,pip 仍然要联网查索引确认版本。国内网络下这一步
    经常长时间卡住,而「重跑一次安装脚本」是非常常见的操作,不该每次都
    冒这个险。所以先 import 试一下,能用就彻底不碰 pip。
    用 DEVNULL 而不是 PIPE:避免走管道,也避免把探测的报错喷到屏幕上。
    """
    probe = "import PIL, numpy, onnxruntime, rapidocr_onnxruntime"
    return subprocess.run(
        [VENV_PY, "-c", probe], cwd=HERE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def main() -> int:
    # 依据:Python 在输出被重定向时会退回系统区域编码(中文 Windows 是 GBK),
    # 而 GBK 里没有韩文/日文假名,直接 print 识别结果会 UnicodeEncodeError 崩掉。
    # 强制 UTF-8 + errors=replace,宁可个别字变问号也不能让脚本崩。
    for stream in (sys.stdout, sys.stderr):
        try:
            # line_buffering=True:每行都立即刷出去,配合 run() 里的 flush,
            # 保证「父进程的提示」和「子进程的输出」顺序正确。
            stream.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        except Exception:
            pass

    print("=" * 66)
    print("  OCR 探针环境安装")
    print("=" * 66)

    # ---------------- 说明:为什么是这套方案 ----------------
    print("""
  为什么用 RapidOCR 而不是 PaddleOCR:
    · PaddleOCR 要装 paddlepaddle + paddlex,约 1 GB,而且实测在这台机器上
      pip 解析它的依赖树会长时间空转卡死(CPU 飙高但网络已经不动了);
    · RapidOCR 只有约 120 MB,而且跑的是**同一套 PP-OCR 模型**(格式为 ONNX),
      结论完全可比;
    · 更关键:这套 ONNX 模型和后面 C# 版 RapidOcrNet 用的是同一份,
      所以现在测出来的准确率和速度,可以直接搬到 C# 实现上。

  为什么不装 Windows 自带的韩/日 OCR 语言包:
    装语言包会给系统语言列表里多出输入法,影响切换键盘。
    本方案的模型自带,只落在项目目录里,不碰系统设置。""")

    # ---------------- 步骤 1:检查解释器 ----------------
    step(1, "检查 Python 解释器")
    print(f"      版本: {sys.version.split()[0]}")
    print(f"      路径: {sys.executable}")
    if sys.version_info < (3, 9) or sys.version_info >= (3, 13):
        print("\n  [!] 建议使用 Python 3.9 ~ 3.12。当前版本不一定能装到预编译包,")
        print("      如果后面依赖装不上,换个版本再试。")
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        print("  [!] 检测到当前已在某个虚拟环境里。建议在系统 Python 下运行本脚本,")
        print("      这样 .venv 才是干净独立的。")

    # ---------------- 步骤 2:建虚拟环境 ----------------
    step(2, "准备虚拟环境 .venv")
    if os.path.isfile(VENV_PY):
        print("      已存在,跳过创建")
    else:
        print("      创建中 …")
        if run([sys.executable, "-m", "venv", VENV_DIR]) != 0:
            print("\n  [x] 虚拟环境创建失败。")
            return 1
        if not os.path.isfile(VENV_PY):
            print(f"\n  [x] 建完了但找不到 {VENV_PY},环境可能不完整。")
            return 1
        print("      完成")

    # ---------------- 步骤 3:装依赖 ----------------
    step(3, "安装依赖(pillow / numpy / rapidocr-onnxruntime)")
    if deps_ready():
        print("      依赖已就绪,跳过 pip(重跑安装脚本时无需再次联网)")
        code = 0
    else:
        print("      约 120 MB,请耐心等,不要关窗口\n")

        # 依据:pip 自升级也必须走镜像。之前这里漏了 -i,导致它去连 pypi.org ——
        # 那个在国内是真的不通,于是整个安装就卡死在这一步。
        # 升级 pip 不是致命操作,失败也无所谓,所以给短超时并且不检查返回值。
        print("      -- 升级 pip(失败不影响后续)--")
        run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "-q",
             "--disable-pip-version-check", "-i", MIRROR,
             "--retries", "2", "--timeout", "20"])

        print("\n      -- 通过清华镜像安装依赖 --")
        code = run(PIP_INSTALL + ["-i", MIRROR] + DEPS)

        if code != 0:
            print("\n  [!] 镜像源失败,改用官方源重试 …")
            code = run(PIP_INSTALL + DEPS)

    if code != 0:
        print("\n  [x] 依赖安装失败。可以手动执行:")
        print(f"      {VENV_PY} -m pip install {' '.join(DEPS)}")
        return 1

    print("\n      验证依赖 …")
    verify = ("import PIL, numpy, onnxruntime, rapidocr_onnxruntime as r; "
              "print('      OK  onnxruntime', onnxruntime.__version__)")
    if run([VENV_PY, "-c", verify]) != 0:
        print("\n  [x] 依赖验证失败,请把上面的报错发给我。")
        return 1

    # ---------------- 步骤 4:下载识别模型 ----------------
    step(4, "下载韩文/日文识别模型")
    print("""
      依据:rapidocr 自带的识别模型只认中英文。检测和方向分类模型与语种无关,
      可以直接复用;只有「识别」这一步必须换成韩文/日文模型。
      ★ 模型(.onnx)和字典(.txt)必须成对使用,不配对会输出乱码。""")
    if run([VENV_PY, os.path.join(HERE, "download_models.py")]) != 0:
        print("\n  [!] 模型下载失败。不影响已有功能,但韩文/日文识别用不了。")
        print("      可以稍后重跑本脚本,或手动下载后放进 models\\ 目录。")
        return 1

    # ---------------- 完成 ----------------
    print("\n" + "=" * 66)
    print("  安装完成!")
    print("""
  第一步,自检(不需要游戏截图,脚本会自己画一张):
      run_probe.bat --selftest --lang ko --det-limit 320 --no-cls

  第二步,框选真实的游戏聊天区域:
      run_probe.bat --region --lang ko --save shots\\chat1.png

  注意:游戏请用「无边框窗口化」。独占全屏下截图会全黑 ——
        那时画面绕过了 Windows 的桌面合成,抓不到。""")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
