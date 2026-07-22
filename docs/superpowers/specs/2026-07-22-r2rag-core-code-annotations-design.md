# R2RAG 核心代码学习注释设计

## 目标

为下一次 R2RAG 学习任务涉及的四个核心文件增加中文分层注释，使学习者能够沿着“统一接口 → 动态路由 → 复杂度判断 → 单轮 RAG”追踪一次请求，同时不改变任何运行逻辑。

## 范围

- `R2RAG/src/systems/rag_interface.py`
- `R2RAG/src/systems/rag_router/rag_router_llm.py`
- `R2RAG/src/tools/classifiers/llm_query_complexity.py`
- `R2RAG/src/systems/vanilla_agent/vanilla_rag.py`

`vanilla_agent.py` 的多轮检索循环不在本次范围内，留作下一次独立精读。

## 注释结构

每个文件使用三层学习注释：

1. 文件级说明：指出文件在 R2RAG 调用链中的位置及其上下游；
2. 类/函数级说明：解释职责、输入、输出、调用关系和容易误解的设计；
3. 关键语句旁注：解释异步生成器、路由分支、模型延迟初始化、检索/重排/截断和流式响应状态。

普通 import、直接赋值和已有语义清晰的代码不逐行翻译，避免注释遮蔽程序结构。

## 不变量

- 不修改函数签名、参数默认值、导入关系和控制流；
- 不改变 prompt、日志内容、响应字段或异常处理；
- 不安装依赖，不调用 LLM，不下载数据；
- MedRAG/R2RAG 的语料、模型、日志和输出目录继续保持忽略状态。

## 验证

1. 使用 UTF-8 读取四个文件并检查中文学习标记存在；
2. 使用 `uv run python` 对四个文件执行无字节码 `compile()` 语法检查；若宿主机没有 `uv`，记录该环境限制并使用容器/可用 Python 做等价语法检查；
3. 检查 Git diff，确认变更集中于注释/docstring 和学习进度；
4. 使用 `git check-ignore` 确认 `MedRAG/corpus/`、`R2RAG/data/`、`R2RAG/models/`、`R2RAG/logs/` 不会进入提交。
