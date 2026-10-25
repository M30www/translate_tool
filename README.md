# translate_tool

游戏内聊天实时翻译小工具：**截屏 → OCR → 翻译 → 显示**，把队友/对手的韩文、日文、英文聊天实时翻成中文。

> 原则：**零接触**。不注入、不 Hook、不读游戏内存、不改包、不自动操作游戏。
> 只做"看屏幕的眼睛"，所以反作弊风险最低。

## 现在能做什么

| 能力 | 状态 |
|---|---|
| 截屏 + 检测画面变化 | ✅ 可用（跳帧省掉 80% 的 OCR） |
| OCR 识别韩/日/中/英混排 | ✅ 可用（双模型集成，真机召回 25/25） |
| 消息去重、残缺行处理 | ✅ 可用（重复报 0 条） |
| 按颜色区分聊天与系统提示 | ✅ 可测量（尚未用于过滤） |
| 翻译 | 🟡 免费引擎可跑（探索用，质量中等）；LLM 接口已就绪待接 |
| 悬浮窗显示 | ⬜ 未开始 |

## 快速开始

```bash
# 1. 一次性准备：建虚拟环境 + 装依赖 + 下载模型（需联网）
setup_ocr.bat

# 2. 半实时探针：框选游戏聊天区 → 一直跑，Ctrl+C 结束
run_live.bat

# 3. 判读这次运行的效果（重复行、误读、漏行、颜色分布）
python analyze_log.py logs\live_xxx.json --expect chat_samples.txt

# 4. 离线翻译最新日志，输出原文/译文对照表（双击即可，无需 API key）
run_translate.bat
```

三个 bat 都可以**直接双击**，也可以带参数覆盖默认值，例如
`run_translate.bat --limit 36`、`run_live.bat --min-score 0.6`。

装包请用 `.venv\Scripts\python.exe -m pip install ...`（`pip.exe` 因环境搬迁已失效）。

## 目录结构

```
ocr_probe.py        单帧 OCR 探针（取图/预处理/模型对比/质量报告）
probe_live.py       半实时探针（帧差闸门 + 集成 OCR + 去重 + 日志落盘）
analyze_log.py      日志判读（抖动簇/低质量行/漏行空窗/召回核对）
translate_text.py   翻译后端（免费引擎 / LLM）+ 缓存 + 术语表
translate_log.py    离线批量翻译
chat_text.py        文本归一化判据（去重与缓存共用）
color_probe.py      测量每行文字的颜色（区分消息类型）
explore/ocr/        双模型集成与行重组的实验代码
models/             OCR 模型与字典（onnx 不入库，用 download_models.py 拉）
docs/               设计文档、变更记录、实验记录、知识库
```

## 文档

| 想知道什么 | 看哪里 |
|---|---|
| 为什么这样设计、风险在哪 | [docs/feasibility-and-design.md](docs/feasibility-and-design.md) |
| 系统现在怎么工作、参数怎么调 | [docs/knowledge-base.md](docs/knowledge-base.md) |
| 每次改动的原因与遗留 | [docs/dev-log.md](docs/dev-log.md) |
| 实测数据（条件+结论） | [docs/experiments.md](docs/experiments.md) |

## 免责

个人自用、探索性质。OCR 与翻译均通过公开接口实现，未触碰游戏进程；**请勿分发或用于商业化**，
免费翻译接口属第三方非官方服务，随时可能失效。
