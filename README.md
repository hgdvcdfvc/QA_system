# 中山大学人工智能学院知识问答助手 (RAG System)

本项目包含两套并行的知识库问答（RAG）实现方案。项目结构清晰，互不干扰，原有的 API 版本与新增的纯本地离线版本完整保留，可直接上传至 GitHub。

---

## 本地预训练模型版知识库问答

新增的 `local_model_demo.py`、`local_rag_engine.py` 和 `local_model.py` 是一套不依赖远程大模型 API 的并行实现：

1. `local_rag_engine.py` 使用本地向量模型生成知识库向量，并写入新的 Chroma 目录 `knowledge_base`。
2. `local_model_demo.py` 使用本地大语言模型读取检索片段，并在 Web 前端实现流式打字机效果的中文答案输出。
3. `local_model.py` 具备智能路径解析功能，会优先从工作区下的 `models` 或 `model` 文件夹自动匹配、解析模型路径。
4. 原来的 `api_model_demo.py` 和 `buildup.py` 没有被修改，云端 API 版本仍然并行保留。

## 云端 API 版知识库问答

1. `buildup.py` 负责调用云端大模型 API 的 Embedding 接口，将 `database.txt` 的内容切片上传，并持久化写入本地 Chroma 目录 `txt_knowledge_base`。
2. `api_model_demo.py` 基于统一的高性能云端模型提供问答中枢，并在输入框支持输入 `\quit` 触发后台安全无损关机。

---

## 推荐模型（本地版）

* **向量模型**：`models\bge-small-zh-v1.5` （或系统兼容格式 `bge-small-zh-v1___5`）
* **生成模型**：`models\Qwen2.5-1.5B-Instruct` （或系统兼容格式 `Qwen2___5-1___5B-Instruct`）

> 💡 如果本地机器显存或配置不足，可以把生成模型无缝替换为更轻量化的 `Qwen2.5-0.5B-Instruct`。

---

## 运行方式

### 方案 A：运行本地预训练模型版

```powershell
streamlit run local_model_demo.py

```

默认会优先从当前工作区的 `models` 文件夹加载模型。当前支持下面这种目录结构：

```text
models/
  bge-small-zh-v1.5/
  Qwen2.5-1.5B-Instruct/

```
如果你使用 `model` 文件夹，或想换成其他磁盘绝对路径，也可以通过 PowerShell 环境变量进行指定：

```powershell
$env:LOCAL_EMBEDDING_MODEL = ""
$env:LOCAL_LLM_MODEL = ""
streamlit run local_model_demo.py

```

### 方案 B：运行云端 API 版
配置apikey

```powershell
streamlit run api_model_demo.py

```

首次运行或本地原始文本 `database.txt` 更新时，请先在左侧边栏点击 **“更新数据库”** 按钮完成初始化。

---

## 依赖说明

当前实现完整复用项目中已有的核心依赖：

* `streamlit`
* `chromadb`
* `langchain_text_splitters`
* `transformers`
* `torch`
* `PyPDF2`

### 架构依赖分工

| 核心组件 | 本地预训练模型版 | 云端 API 版 |
| --- | --- | --- |
| **OpenAI / DashScope SDK** | ❌ 不需要 | 需要 (`pip install openai`) |
| **API Key 凭据** | ❌ 不需要，完全离线自治 | 需要 (在 `buildup.py` 中配置) |
| **Sentence-Transformers** | ❌ 不需要 | ❌ 不需要 |

* **本地版初始化说明**：第一次运行会加载向量模型并构建新的本地知识库。只有在 `models` / `model` 文件夹下找不到指定权重，且没有设置对应的环境变量时，系统才会自动回退到 Hugging Face 线上仓库名进行动态下载。

### 后续修改说明

可自由修改database.txt内的文本内容实现个性化。
