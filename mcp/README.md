# MCP 改造版 RAG Demo（LangGraph Agent + MCP 检索 Server）

在 [LangGraph-demo](../LangGraph-demo/README.md) 的多 Agent 版 RAG 基础上，将**检索能力整体抽离为 MCP Server**，Agent 侧只保留"编排与推理"，演示如何用 **MCP（Model Context Protocol）** 在 Agent 与检索服务之间解耦。

## 架构流程

```
Agent 侧（main.py，LangGraph 编排）          MCP Server 侧（mcp_server.py，stdio 子进程）
─────────────────────────────              ───────────────────────────────────────────
原始 Query
   │
   ▼
路由 Agent ──────────────────────────────►  无需检索（chat 分流时 Server 零调用）
   │ (rag)
   ▼
改写 Agent（指代消解）
   │
   │  MCP call_tool("hybrid_search")
   ▼ ────────────────────────────────────►  向量检索 + BM25 两路召回
检索 Agent                                  → RRF 融合去重 → 返回候选块
   │  MCP call_tool("rerank")
   ▼ ────────────────────────────────────►  Cross-Encoder 精排 → 返回分数
人工介入门（Top1 分数 < 0.3 时 interrupt）
   ▼
生成 Agent（带出处生成）
   ▼
审核 Agent（PASS / FAIL，打回重试）
   ▼
最终答案
```

## 与原版 LangGraph Demo 的差异

| 维度 | LangGraph-demo | 本 Demo（MCP 版） |
|---|---|---|
| 进程模型 | 单进程，检索逻辑内联 | 双进程：Agent 编排 + MCP Server（stdio 子进程） |
| 检索能力归属 | `retrieve_node` 内直接调用 | 封装为 MCP 工具 `hybrid_search` / `rerank` / 资源 `kb://docs/{source}` |
| 凭证管理 | Embedding/Rerank/LLM 的 key 都在 main.py | `SILICONFLOW_API_KEY`（检索侧）只存在于 Server 进程，Agent 侧不可见 |
| 客户端 | 直接函数调用 | FastMCP `Client`：自动拉起 Server、建立会话、`list_tools` 能力发现 |
| Langfuse 追踪 | LangChain 回调 + `@observe` 补 Rerank span | LLM 走 LangChain 回调；两次 MCP 调用发生在 Server 进程，回调追踪不到，由 Client 侧 `@observe` 手动补 span |

## MCP Server 暴露的能力

定义在 `mcp_server.py`（FastMCP，`stdio` 传输）：

| 类型 | 名称 | 说明 |
|---|---|---|
| Tool | `hybrid_search(query)` | 向量 + BM25 两路召回，RRF 融合并按文本去重，返回 `{content, source, category}` 列表 |
| Tool | `rerank(query, documents, top_n)` | Cross-Encoder 精排，返回 `[{index, score}]`；服务不可用时降级为 `score=None`（交由调用方决定是否触发人工介入） |
| Resource | `kb://docs/{source}` | 按 source 编号（如 `rrf` / `bm25` / `rerank`）读取知识库原文 |

## 特性亮点

- **能力解耦**：检索逻辑（embedding、BM25、RRF、rerank）整体迁入 MCP Server，Agent 侧只通过工具调用使用，可独立替换检索实现而不改动编排代码。
- **凭证隔离**：检索服务的 API Key 只由 Server 进程持有，符合"敏感凭证不下发 Agent 侧"的部署惯例。
- **能力发现**：连接建立后执行 `list_tools`，运行时可打印 Server 实际暴露的工具列表。
- **降级设计**：Rerank 接口异常时不判分、按传入顺序截取，分数为 `None` 时由人工介入门兜底。
- **全链路追踪**：LLM 调用经 Langfuse 回调自动上报；MCP 调用跨进程无法被回调捕获，在 Client 侧用 `@observe(name="mcp-hybrid-search")` / `@observe(name="mcp-rerank")` 手动补 span 并写入 query、分数等 metadata。

## 环境依赖

```bash
pip install langchain-core langchain-community langchain-classic langchain-siliconflow langgraph langfuse fastmcp jieba requests
```

> 相比原版增加了 `fastmcp`；Demo 依赖 `langchain_siliconflow` 中封装的 SiliconFlow API（兼容 OpenAI 协议格式）。

## 配置说明

两个文件各自持有一部分配置：

**`mcp_server.py`（Server 侧，仅检索凭证与模型）**：

| 配置项 | 说明 |
|---|---|
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥（仅 Server 持有） |
| `EMBEDDING_MODEL` | 向量嵌入模型，默认 `Qwen/Qwen3-VL-Embedding-8B` |
| `RERANK_MODEL` | 重排序模型，默认 `Qwen/Qwen3-VL-Reranker-8B` |
| `TOP_K` | 每路召回数量，默认 `4` |

**`main.py`（Agent 侧，仅编排与推理配置）**：

| 配置项 | 说明 |
|---|---|
| `SILICONFLOW_API_KEY` | 推理模型密钥 |
| `LLM_MODEL` | 推理模型，默认 `zai-org/GLM-5.3` |
| `REFUSE_THRESHOLD` | 触发人工介入的 Top1 分数阈值，默认 `0.3` |
| `REFUSE_TEXT` | 拒答文案 |
| `MAX_GENERATE_RETRY` | 审核打回的最大重试次数，默认 `1` |

### Langfuse 追踪（可选）

`main.py` 顶部通过环境变量配置 Langfuse（默认指向自托管实例）：`LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`。

## 运行

```bash
python main.py
```

`main.py` 会通过 FastMCP `Client(Path("mcp_server.py"))` 以 stdio 子进程方式自动拉起 Server，无需手动启动。连接建立后会打印发现的工具列表，随后跑与 LangGraph-demo 相同的 5 个演示问题（含一次低分触发的人工介入）。

> 调试 Server 也可单独运行 `python mcp_server.py`（阻塞等待 stdio 输入）。

## 代码结构

| 组件 | 作用 |
|---|---|
| `mcp_server.py` | FastMCP Server：`hybrid_search` / `rerank` 工具与 `kb://docs/{source}` 资源，持有知识库与检索凭证 |
| `mcp_client` | FastMCP Client：以 stdio 子进程拉起 Server，建立会话 |
| `mcp_hybrid_search` / `mcp_rerank` | Client 侧封装：调用 MCP 工具 + `@observe` 补 Langfuse span |
| `retrieve_node` | 检索 Agent：连续两次 MCP 调用（混合检索 → Rerank） |
| 其余节点 | 与 LangGraph-demo 一致：router / rewrite / human_gate / refuse / generate / critic / direct_answer |

## 扩展建议

- **HTTP 传输**：FastMCP `Client` 同样支持连接远程 HTTP Server，可将检索服务部署为独立远程服务，多个 Agent 共享。
- **更多检索工具**：在 Server 侧新增 `@mcp.tool()`（如按 category 过滤、全文读取），Agent 侧无需改动编排即可通过 `list_tools` 发现。
- **Server 侧追踪**：如需观测 Server 内部耗时，可在 `mcp_server.py` 中引入 Langfuse/Python SDK 上报（当前 span 补在 Client 侧）。
