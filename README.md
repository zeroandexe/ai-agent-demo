# AI Agent Demo

基于 **LangChain / LangGraph / Langfuse + SiliconFlow API** 的大模型应用开发示例仓库。

## 目录结构

```
ai-agent-demo/
├── README.md            # 本文件：仓库简介与导航
├── langChain-demo/      # LangChain LCEL 单链版 RAG Demo
│   ├── main.py
│   └── README.md        # 详细文档（架构、配置、运行说明）
├── LangGraph-demo/      # LangGraph 多 Agent 协作 + 人工介入版 RAG Demo
│   ├── main.py
│   └── README.md        # 详细文档（架构、配置、运行说明）
└── mcp-skill/           # MCP 改造版：检索能力抽离为 MCP Server，并引入低分自救 Skill
    ├── main.py
    ├── mcp_server.py
    ├── skills/
    │   └── rag-rescue/  # 低分自救 Skill（SKILL.md 手册 + 术语抽取脚本）
    └── README.md        # 详细文档（架构、配置、运行说明）
```

## Demo 一览

| Demo | 说明 |
|---|---|
| [langChain-demo](langChain-demo/README.md) | LangChain LCEL 单链版 RAG：查询改写（指代消解）→ 混合检索（向量 + BM25）→ RRF 融合 → Rerank 精排 → 阈值拒答 → 带出处生成 |
| [LangGraph-demo](LangGraph-demo/README.md) | LangGraph 多 Agent 版 RAG：在相同检索流水线之上，以状态图编排 路由 / 改写 / 检索 / 生成 / 审核 多个 Agent，并支持 interrupt 人工介入与断点续跑 |
| [mcp-skill](mcp-skill/README.md) | MCP 改造版 RAG：将检索能力（混合检索 / Rerank）抽离为 MCP Server（FastMCP，stdio），并新增低分自救 Skill（rag-rescue）：检索未达阈值时按 SKILL.md 手册自动重试，失败才升级人工介入 |

前两个 Demo 使用同一份内置知识库与同一套检索组件，可作为 **LCEL 单链编排 vs LangGraph 状态图编排** 的对照学习材料；mcp-skill Demo 则演示如何在 LangGraph 编排之上用 **MCP** 解耦检索服务、并用 **Skill** 将领域排查知识外置。

## 通用前置条件

- Python 依赖（各 Demo 的 README 中有完整列表）：
  ```bash
  pip install langchain-core langchain-community langchain-classic langchain-siliconflow langgraph langfuse fastmcp jieba requests
  ```
- 在对应 `main.py` 顶部填入 `SILICONFLOW_API_KEY`（mcp-skill Demo 的检索密钥填在 `mcp-skill/mcp_server.py` 中）
- 可选：配置 Langfuse（自托管或云端）的 `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`，用于全链路追踪

> 详细的架构说明、配置项、运行方式与演示问题列表，请见各子目录的 README。

## License

MIT
