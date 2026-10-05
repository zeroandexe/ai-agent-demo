# AI Agent Demo

基于 **LangChain / LangGraph / Langfuse + SiliconFlow API** 的大模型应用开发示例仓库。

## 目录结构

```
ai-agent-demo/
├── README.md            # 本文件：仓库简介与导航
├── langChain-demo/      # LangChain LCEL 单链版 RAG Demo
│   ├── main.py
│   └── README.md        # 详细文档（架构、配置、运行说明）
└── LangGraph-demo/      # LangGraph 多 Agent 协作 + 人工介入版 RAG Demo
    ├── main.py
    └── README.md        # 详细文档（架构、配置、运行说明）
```

## Demo 一览

| Demo | 说明 |
|---|---|
| [langChain-demo](langChain-demo/README.md) | LangChain LCEL 单链版 RAG：查询改写（指代消解）→ 混合检索（向量 + BM25）→ RRF 融合 → Rerank 精排 → 阈值拒答 → 带出处生成 |
| [LangGraph-demo](LangGraph-demo/README.md) | LangGraph 多 Agent 版 RAG：在相同检索流水线之上，以状态图编排 路由 / 改写 / 检索 / 生成 / 审核 多个 Agent，并支持 interrupt 人工介入与断点续跑 |

两个 Demo 使用同一份内置知识库与同一套检索组件，可作为 **LCEL 单链编排 vs LangGraph 状态图编排** 的对照学习材料。

## 通用前置条件

- Python 依赖（各 Demo 的 README 中有完整列表）：
  ```bash
  pip install langchain-core langchain-community langchain-classic langchain-siliconflow langgraph langfuse jieba requests
  ```
- 在对应 `main.py` 顶部填入 `SILICONFLOW_API_KEY`
- 可选：配置 Langfuse（自托管或云端）的 `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`，用于全链路追踪

> 详细的架构说明、配置项、运行方式与演示问题列表，请见各子目录的 README。

## License

MIT
