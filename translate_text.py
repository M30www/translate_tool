#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
translate_text.py —— 翻译后端抽象 + 缓存 + 术语表
  目的是让"翻译"可以随时换实现,而管道不用动。当前实现:
    EchoTranslator   不联网,只回显,用来验证管道/缓存/输出格式(零成本)
    GoogleTranslator 谷歌网页翻译的非官方接口:免 key、质量对游戏短句够用
    MyMemoryTranslator 公开免费 API:免 key,但需源语言,韩→中质量一般
    LlmTranslator    任何 OpenAI 兼容接口(默认 DeepSeek),一次请求翻一批
    CachedTranslator 用 chat_text.norm_key 做缓存键,命中即返回

  ★ 免费引擎只用于**探索**,不可用于正式分发:非官方接口随时可能失效/限流,
    且属于灰色用法。正式版换 LlmTranslator 或本地模型。
  零新增依赖:只用标准库 urllib —— 本机 .venv\\pip.exe 已失效(见 AGENTS.md),能不装包就不装。
  LLM 配置走环境变量:DS_LLM_API_KEY / DS_LLM_BASE_URL / DS_LLM_MODEL。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_text import norm_key   # noqa: E402  缓存键与去重键同源

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# 单价留空:不猜官网价格,只在统计里报 token 数。要算钱自己填(单位:元/百万 token)。
PRICE_INPUT_PER_M = 0.0
PRICE_OUTPUT_PER_M = 0.0

# 国服通用译名 + 玩家黑话。只是默认值,可用 --glossary 覆盖。
DEFAULT_GLOSSARY = {
    "genji": "源氏", "mercy": "天使", "tracer": "猎空", "reinhardt": "莱因哈特",
    "dva": "D.Va", "widow": "黑百合", "hanzo": "半藏", "lucio": "卢西奥",
    "moira": "莫伊拉", "zarya": "查莉娅", "winston": "温斯顿", "torb": "托比昂",
    "diff": "实力差距", "ns": "打得好", "gg": "打得好", "wp": "打得漂亮",
    "ult": "大招", "hitscan": "长枪位", "push": "推车", "throw": "故意送",
}

SYSTEM_PROMPT = """你是游戏内聊天的即时翻译器,把玩家聊天翻译成自然的简体中文。

规则:
1. 只输出译文,不解释、不加引号、不保留原文。
2. 用国服玩家的习惯说法,保留游戏语境与语气(粗口、玩笑、催促都不要变礼貌)。
3. 英雄名、地图名用国服通用译名;缩写按玩家习惯处理。
4. 输入可能是被截断的碎片,照常翻,不要脑补不存在的内容。
5. 已经是中文的,原样返回。

输出必须是严格的 JSON 数组,元素是字符串,数量与顺序和输入完全一致。"""


class Translator:
    """一次翻一批,顺序与输入一一对应。"""

    name = "base"

    def translate_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError


class EchoTranslator(Translator):
    """不联网的回显器:用来验证管道、缓存、输出格式,零成本。"""

    name = "echo"

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [f"〔译〕{text}" for text in texts]


def _fetch_json(url: str, timeout: float) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class GoogleTranslator(Translator):
    """谷歌网页翻译的非官方端点:免 key。

    画质对游戏短句够用,但**只用于探索**:非官方接口随时可能限流/失效,且属灰色用法。
    一次只能翻一条,所以逐条请求并加小延时(delay)避免被掐。
    """

    name = "google"
    URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, target: str = "zh-CN", timeout: float = 10.0, delay: float = 0.2):
        self.target = target
        self.timeout = timeout
        self.delay = delay

    def _one(self, text: str) -> str:
        query = urllib.parse.urlencode({"client": "gtx", "sl": "auto",
                                        "tl": self.target, "dt": "t", "q": text})
        data = _fetch_json(f"{self.URL}?{query}", self.timeout)
        # 返回形如 [[["译文","原文",...],...], null, "ko", ...]:把所有分段拼起来
        return "".join(segment[0] for segment in (data[0] or []) if segment and segment[0])

    def translate_batch(self, texts: list[str]) -> list[str]:
        out = []
        for index, text in enumerate(texts):
            if index:
                time.sleep(self.delay)
            out.append(self._one(text))
        return out


class MyMemoryTranslator(Translator):
    """MyMemory 公开免费 API:免 key,但要求给出源语言,韩→中质量一般。"""

    name = "mymemory"
    URL = "https://api.mymemory.translated.net/get"
    LANG = {"ko": "ko", "ja": "ja", "en": "en", "zh": "zh-CN"}

    def __init__(self, target: str = "zh-CN", timeout: float = 10.0, delay: float = 0.3,
                 detect=None):
        self.target = target
        self.timeout = timeout
        self.delay = delay
        self.detect = detect or (lambda _text: "en")

    def _one(self, text: str) -> str:
        source = self.LANG.get(self.detect(text), "en")
        query = urllib.parse.urlencode({"q": text, "langpair": f"{source}|{self.target}"})
        data = _fetch_json(f"{self.URL}?{query}", self.timeout)
        return str((data.get("responseData") or {}).get("translatedText") or text)

    def translate_batch(self, texts: list[str]) -> list[str]:
        out = []
        for index, text in enumerate(texts):
            if index:
                time.sleep(self.delay)
            out.append(self._one(text))
        return out


def probe_engines(names: list[str], timeout: float = 8.0) -> tuple[str, dict]:
    """逐个用一句韩文试翻,返回(第一个能用的引擎名, 各引擎探测结果)。

    国内直连 Google 经常不通,所以不能把引擎写死;先探再用,失败原因如实报出来。
    """
    results: dict[str, str] = {}
    for name in names:
        try:
            if name == "google":
                value = GoogleTranslator(timeout=timeout, delay=0)._one("안녕")
            elif name == "mymemory":
                value = MyMemoryTranslator(timeout=timeout, delay=0)._one("안녕")
            else:
                results[name] = "跳过(无需探测)"
                continue
            results[name] = f"可用 → {value}"
            return name, results
        except Exception as exc:                        # noqa: BLE001
            results[name] = f"不可用: {type(exc).__name__}: {exc}"
    return "", results


class LlmTranslator(Translator):
    """任何 OpenAI 兼容的 /chat/completions 接口(默认 DeepSeek)。"""

    name = "llm"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, glossary: dict | None = None,
                 timeout: float = 20.0):
        if not api_key:
            raise SystemExit(
                "[x] 没有 API key。设置环境变量 DS_LLM_API_KEY 后重试,"
                "或先用 --engine echo 跑通管道。")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.glossary = glossary or {}
        self.timeout = timeout
        self.prompt_tokens = 0        # 累计用量,用于统计成本
        self.completion_tokens = 0
        self.requests = 0

    def _system_prompt(self) -> str:
        if not self.glossary:
            return SYSTEM_PROMPT
        pairs = "; ".join(f"{k}={v}" for k, v in self.glossary.items())
        return SYSTEM_PROMPT + f"\n\n术语表(必须遵守):{pairs}"

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
            ],
        }
        result = self._post(payload)
        self.requests += 1

        usage = result.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

        content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_array(content)
        if parsed is None or len(parsed) != len(texts):
            # 条数对不上就整体降级:宁可全用原文,也不能错位(错位会把 A 的译文配给 B)
            return [f"⟨解析失败⟩{text}" for text in texts]
        return [str(item) for item in parsed]


def _parse_array(content: str) -> list | None:
    """从模型回复里抠出 JSON 数组,容忍 ```json 围栏与前后废话。"""
    text = content.strip()
    if "```" in text:
        text = text.split("```")[1] if text.count("```") >= 2 else text.replace("```", "")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


# 缓存键的版本前缀。**改了送翻内容的口径就要升版本**:否则旧值会被当成新结果命中。
# 踩过:实现"只翻正文"之后,缓存里旧值(带 `[Ir1s]:` 前缀)被原样返回,报告里出现重复前缀。
CACHE_VERSION = "v2"


class CachedTranslator(Translator):
    """按 norm_key 缓存译文。

    用 norm_key 当键是刻意的复用:它剥掉了昵称前缀与标点,所以
    `[r1s]: gg`、`※[Ir1s]: gg`、`[r1s]:gg` 命中同一条缓存 —— 游戏聊天重复率极高,这一步省掉大半调用。
    """

    def __init__(self, inner: Translator, path: str | None = "cache/translate_cache.json",
                 save_every: int = 20):
        self.inner = inner
        self.name = inner.name
        self.path = path
        self.save_every = save_every
        self.dirty = 0
        self.cache: dict[str, str] = {}
        self.hits = 0
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    self.cache = json.load(handle)
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def _key(self, text: str) -> str:
        return f"{CACHE_VERSION}|{norm_key(text) or text.strip()}"

    def translate_batch(self, texts: list[str]) -> list[str]:
        out: list[str | None] = [None] * len(texts)
        pending = []
        for index, text in enumerate(texts):
            key = self._key(text)
            if key in self.cache:
                out[index] = self.cache[key]
                self.hits += 1
            else:
                pending.append((index, key, text))

        if pending:
            translated = self.inner.translate_batch([item[2] for item in pending])
            for (index, key, _text), value in zip(pending, translated):
                out[index] = value
                self.cache[key] = value
                self.dirty += 1
            if self.path and self.dirty >= self.save_every:
                self.save()

        return [item if item is not None else "" for item in out]

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        temp = self.path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(self.cache, handle, ensure_ascii=False, indent=1)
        os.replace(temp, self.path)     # 原子替换:中途崩了不会留下半个缓存文件
        self.dirty = 0


def build_translator(engine: str = "llm", cache_path: str | None = None,
                     glossary: dict | None = None, api_key: str | None = None,
                     base_url: str | None = None, model: str | None = None,
                     delay: float = 0.2, detect=None) -> Translator:
    """按名字造一个(带缓存的)翻译器。

    缓存文件按引擎分开:否则先用 echo 跑通管道时写进去的假译文,
    会在切换成 llm 后被当真译文命中(踩过)。
    engine="free" 表示"自动挑一个能用的免费引擎"(国内直连 Google 经常不通)。
    """
    if engine == "free":
        picked, results = probe_engines(["google", "mymemory"])
        for name, outcome in results.items():
            print(f"  [探测] {name:<9} {outcome}")
        if not picked:
            raise SystemExit("[x] 两个免费引擎都不可用;可先用 --engine echo 验证管道")
        engine = picked
        print(f"  → 使用 {engine}")
    if cache_path is None:
        cache_path = f"cache/translate_{engine}.json"

    if engine == "echo":
        base: Translator = EchoTranslator()
    elif engine == "google":
        base = GoogleTranslator(delay=delay)
    elif engine == "mymemory":
        base = MyMemoryTranslator(delay=max(delay, 0.3), detect=detect)
    elif engine == "llm":
        base = LlmTranslator(
            api_key=api_key or os.environ.get("DS_LLM_API_KEY", ""),
            base_url=base_url or os.environ.get("DS_LLM_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.environ.get("DS_LLM_MODEL", DEFAULT_MODEL),
            glossary=glossary if glossary is not None else DEFAULT_GLOSSARY,
        )
    else:
        raise SystemExit(
            f"[x] 未知翻译引擎: {engine}(可选 echo / free / google / mymemory / llm)")
    return CachedTranslator(base, path=cache_path)


def write_pairs_md(path: str, title: str, rows: list) -> None:
    """rows: [(时间, 原文, 译文)] → 一张原文/译文对照表。translate_log 与实时通道共用。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n| # | 时间 | 原文 | 译文 |\n|---|---|---|---|\n")
        for index, (stamp, source, value) in enumerate(rows, 1):
            cell = lambda text: text.replace("|", "\\|").replace("\n", " ")
            handle.write(f"| {index} | {stamp} | {cell(source)} | {cell(value)} |\n")


def usage_report(translator: Translator) -> str:
    """把缓存层剥开,拿到底层 LLM 的用量。"""
    inner = getattr(translator, "inner", translator)
    if not isinstance(inner, LlmTranslator):
        return f"用量:不适用({inner.name} 引擎,免费接口无用量统计)"
    cost = (inner.prompt_tokens * PRICE_INPUT_PER_M
            + inner.completion_tokens * PRICE_OUTPUT_PER_M) / 1_000_000
    return (f"用量:请求 {inner.requests} 次,输入 {inner.prompt_tokens} / "
            f"输出 {inner.completion_tokens} tokens"
            + (f",约 ¥{cost:.4f}" if cost else "(单价未配置,不估算金额)"))
