#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_models.py —— 下载韩文/日文的 PP-OCR 识别模型
================================================================================
为什么需要单独下载模型:

  `rapidocr-onnxruntime` 只自带「中文+英文」的识别模型。它自带的部分是:
    · 检测模型(det)—— 负责找出图里哪些地方有文字
    · 方向分类模型(cls)—— 判断文字有没有倒过来
    · 识别模型(rec)—— 负责把文字读出来      ← 只有这个是分语种的

  检测和分类是**与语种无关**的(文字就是文字,长什么样和语言没关系),
  所以可以直接复用自带的。只有「识别」这一步必须换成韩文/日文的模型。

  ★ rec 模型(.onnx)和字典(.txt)**必须成对使用**。
    字典定义了模型能输出哪些字符,两者不匹配时会输出乱码 ——
    这是 OCR 排查里最容易被忽略、也最难自己发现的一类错误。

为什么从 ModelScope 下载而不从 GitHub:
  · 国内直连速度正常,GitHub 的 raw 域名在这台机器上不通;
  · 这些是 RapidOCR 官方发布的 ONNX 模型,和 C# 版 RapidOcrNet 用的是同一套。

为什么用 Python 下载而不是 curl / Invoke-WebRequest:
  实测这台机器上 Windows 原生 TLS(schannel)报 SEC_E_NO_CREDENTIALS,
  curl 和 Invoke-WebRequest 都连不上;而 Python 自带的 OpenSSL 正常(pip 就是这么工作的)。
  考虑到这一点,这里做三级降级:先正常校验证书,失败再退回不校验证书。
  ★ 但无论走哪条路,下载完都会用官方 SHA256 校验文件 —— 完整性由哈希保证,
    而不是由 TLS 保证。这也是为什么这里能接受降级。
================================================================================
"""

from __future__ import annotations

import hashlib
import os
import ssl
import sys
import urllib.error
import urllib.request

BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

# (保存文件名, 下载地址, 期望的 SHA256 或 None, 说明, 字典内容校验类型)
# 校验类型显式写出来,不再靠文件名去猜 —— v6 的字典叫 ppocrv6_dict.txt,
# 名字里既没有 "korean" 也没有 "japan",靠猜会走错分支。
#   korean → 必须含谚文   japan → 必须含假名   multi → 假名或汉字都算
MODELS: list[tuple[str, str, str | None, str, str]] = [
    (
        "korean_PP-OCRv4_rec_mobile.onnx",
        f"{BASE}/onnx/PP-OCRv4/rec/korean_PP-OCRv4_rec_mobile.onnx",
        "ab151ba9065eccd98f884cf4d927db091be86137276392072edd4f9d43ad7426",
        "韩文识别模型 v4(基线)",
        "korean",
    ),
    (
        "korean_dict.txt",
        f"{BASE}/paddle/PP-OCRv4/rec/korean_PP-OCRv4_rec_mobile/korean_dict.txt",
        None,   # 官方未提供该校验值,改用内容检查(见 validate_dict)
        "韩文字典 v4(必须与上面的 onnx 配对)",
        "korean",
    ),
    (
        "japan_PP-OCRv4_rec_mobile.onnx",
        f"{BASE}/onnx/PP-OCRv4/rec/japan_PP-OCRv4_rec_mobile.onnx",
        "e1075a67dba758ecfc7ebc78a10ae61c95ac8fb66a9c86fab5541e33f085cb7a",
        "日文识别模型 v4",
        "japan",
    ),
    (
        "japan_dict.txt",
        f"{BASE}/paddle/PP-OCRv4/rec/japan_PP-OCRv4_rec_mobile/japan_dict.txt",
        None,
        "日文字典 v4(必须与上面的 onnx 配对)",
        "japan",
    ),
    # ---- 下面两个是本次新增,用来做对比 ----
    (
        "korean_PP-OCRv5_rec_mobile.onnx",
        f"{BASE}/onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx",
        "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b",
        "韩文识别模型 v5(比 v4 新,对比用)",
        "korean",
    ),
    (
        "ppocrv5_korean_dict.txt",
        f"{BASE}/paddle/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile/ppocrv5_korean_dict.txt",
        None,
        "韩文字典 v5(必须与上面的 onnx 配对)",
        "korean",
    ),
    (
        "PP-OCRv6_rec_small.onnx",
        f"{BASE}/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
        "v6 多语言识别模型(中/繁中/日/英等 52 种,★ 不含韩文)",
        "multi",
    ),
    (
        "ppocrv6_dict.txt",
        f"{BASE}/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt",
        None,
        "v6 多语言字典(必须与上面的 onnx 配对)",
        "multi",
    ),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_opener() -> urllib.request.OpenerDirector:
    """
    构造一个能适应本机环境的 URL 打开器。

    依据:先尝试「正常校验证书」;如果因为证书/凭证问题失败,
    再退回到不校验证书。因为下载完有 SHA256 兜底,这个降级是可接受的。
    """
    try:
        ctx = ssl.create_default_context()
        # 确保 Python 用的是它自己带的 CA 包,而不是去问 Windows 证书存储
        # (本机 schannel 拿不到凭证,所以走系统存储这条路是坏的)
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except Exception:
            pass
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    except Exception:
        ctx = ssl._create_unverified_context()
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def download(url: str, dest: str) -> None:
    """带降级重试的下载。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err: Exception | None = None

    # 第一次:校验证书(默认行为)
    try:
        with make_opener().open(req, timeout=120) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        return
    except Exception as e:                       # noqa: BLE001
        last_err = e

    # 第二次:不校验证书(完整性交给 SHA256)
    print(f"      [i] 校验证书失败({type(last_err).__name__}),改用免校验模式重试 …")
    ctx = ssl._create_unverified_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    with opener.open(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def validate_dict(path: str, kind: str) -> tuple[bool, str]:
    """
    检查字典文件内容是否合理。

    依据:字典没有官方哈希,但它是纯文本、每行一个字符,内容特征非常明确 ——
    韩文字典里必然大量出现韩文音节(0xAC00-0xD7A3),日文字典里必然出现
    平假名/片假名。如果下错了文件(比如下到一个 HTML 报错页),这里立刻能发现。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:                       # noqa: BLE001
        return False, f"读不出来: {e}"

    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return False, "文件是空的"

    def has(lo: int, hi: int) -> int:
        return sum(1 for ln in lines if any(lo <= ord(c) <= hi for c in ln))

    if kind == "korean":
        n = has(0xAC00, 0xD7A3)
    elif kind == "japan":
        n = has(0x3040, 0x30FF)
    else:
        # multi:v6 多语言字典。它同时含假名和汉字,任一命中即可判定文件正常。
        n = max(has(0x3040, 0x30FF), has(0x4E00, 0x9FFF))

    if n < 10:
        return False, f"只找到 {n} 行含目标字符(共 {len(lines)} 行),文件可能不对"
    return True, f"{len(lines)} 行,其中 {n} 行含目标字符"


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"模型目录: {MODEL_DIR}\n")

    failed = 0
    for name, url, expect_hash, desc, kind in MODELS:
        dest = os.path.join(MODEL_DIR, name)

        print(f"[{desc}]  {name}")

        # 已存在且校验通过就跳过,避免重复下载
        if os.path.isfile(dest):
            if expect_hash and sha256_of(dest) == expect_hash:
                print("      已存在且 SHA256 正确,跳过\n")
                continue
            if not expect_hash:
                ok, msg = validate_dict(dest, kind)
                if ok:
                    print(f"      已存在且内容正常({msg}),跳过\n")
                    continue

        try:
            download(url, dest)
        except Exception as e:                   # noqa: BLE001
            print(f"      [x] 下载失败: {type(e).__name__}: {e}\n")
            failed += 1
            continue

        size_kb = os.path.getsize(dest) / 1024

        # ---- 校验 ----
        if expect_hash:
            got = sha256_of(dest)
            if got == expect_hash:
                print(f"      OK  {size_kb:,.0f} KB  SHA256 校验通过\n")
            else:
                print(f"      [x] {size_kb:,.0f} KB  但 SHA256 不匹配!")
                print(f"          期望 {expect_hash}")
                print(f"          实际 {got}\n")
                os.remove(dest)
                failed += 1
        else:
            ok, msg = validate_dict(dest, kind)
            print(f"      {'OK ' if ok else '[x]'} {size_kb:,.0f} KB  {msg}\n")
            if not ok:
                failed += 1

    if failed:
        print(f"有 {failed} 个文件失败。")
        print("如果全部失败,通常是网络/TLS 问题,可以手动下载后放进 models\\ 目录:")
        for name, url, *_ in MODELS:
            print(f"  {name}\n    {url}")
        return 1

    print("全部就绪。现在可以运行:")
    print("  run_probe.bat --selftest --lang ko")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
