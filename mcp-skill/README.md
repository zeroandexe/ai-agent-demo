# MCP + Skill 改造版 RAG Demo（LangGraph Agent + MCP 检索 Server + 低分自救 Skill）

在 [LangGraph-demo](../LangGraph-demo/README.md) 的多 Agent 版 RAG 基础上，做两层改造：

1. **检索能力整体抽离为 MCP Server**（见 [mcp_server.py](mcp_server.py)，FastMCP，stdio），Agent 侧只保留"编排与推理"，演示如何用 **MCP（Model Context Protocol）** 在 Agent 与检索服务之间解耦。
2. **新增低分自救 Skill（rag-rescue）**：检索分数未达阈值时，不再立刻 interrupt 升级人工，而是先按 `skills/rag-rescue/SKILL.md` 手册自动排查修复（改写诊断 / 术语对齐 / HyDE / 子查询拆分），每轮用 MCP 检索工具重检验证，最多 3 轮，全部失败才走原人工介入——演示如何用 **Skill** 将领域排查知识外置到 Markdown，代码零改动即可调整策略。

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
低分自救 Skill（rag-rescue，图内 rescue 循环）
   │  未达标：读 SKILL.md 选手册策略 → 生成候选查询
   │  MCP call_tool("hybrid_search") + call_tool("rerank") 重检
   ├─ 达标 ──────────────► 直达生成
   └─ 3 轮全败 / GIVE_UP ─► 升级人工介入门
人工介入门（Top1 分数 < 0.3 且自救失败时 interrupt）
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
| 低分兜底 | Top1 < 0.3 直接 interrupt 人工介入 | 先进入 rescue 自救循环：按 SKILL.md 手册选策略（改写诊断/术语对齐/HyDE/子查询拆分）生成候选查询，用 MCP 工具重检，达标直达生成；3 轮全败或 GIVE_UP 才 interrupt |

## 低分自救 Skill（rag-rescue）

定义在 `skills/rag-rescue/SKILL.md`（frontmatter 声明 `name` / `trigger: top1_score < 0.3` / `max_attempts: 3`），并附带 `scripts/extract_terms.py`（jieba TF-IDF 抽取查询关键术语，供术语对齐步骤参考）。

**运行方式**：`main.py` 的 rescue 节点将 SKILL.md 全文加载进 system prompt，user 侧填入现场信息（原始问题 / 改写后查询 / 当前 top1 分数 / jieba 关键词 / 已尝试历史）。模型严格按手册输出约定格式（`策略: xxx` + 候选查询列表），调用方逐条用 MCP 工具重检并回报 top1 分数：

- 每轮只选**一个**策略、给出 1~2 个候选查询；
- 重检达标 → 图内直达生成，人工无感知；
- 未达标 → 把本轮"策略 / 查询 / 分数"追加进 `rescue_log`，回灌下一轮决策（已尝试策略不重复）；
- 连续 3 轮未达标或模型输出 `GIVE_UP` → 升级原人工介入 interrupt。

手册内置 4 个策略（改写诊断 → 术语对齐 → HyDE → 子查询拆分），详见 SKILL.md。**策略顺序与放弃条件只改 Markdown 即可生效，无需改动任何代码。**

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
- **知识外置（Skill）**：低分排查策略以 SKILL.md 手册形式存放在代码之外，rescue 节点只负责"读手册 → 选手册策略 → 重检验证"的通用循环，调优经验可随时增删改手册而不动代码。

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
| `MAX_RESCUE_ATTEMPTS` | 低分自救 Skill 的最大自救轮数，默认 `3` |
| `RESCUE_SKILL_PATH` | 自救手册路径，默认 `skills/rag-rescue/SKILL.md`（相对 main.py） |

### Langfuse 追踪（可选）

`main.py` 顶部通过环境变量配置 Langfuse（默认指向自托管实例）：`LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`。

## 运行

```bash
python main.py
```

`main.py` 会通过 FastMCP `Client(Path("mcp_server.py"))` 以 stdio 子进程方式自动拉起 Server，无需手动启动。连接建立后会打印发现的工具列表，随后跑 6 个演示问题：第 4 题触发低分自救 Skill（HyDE/术语对齐可救回，人工无感知）；第 5 题自救 3 轮全败，升级 interrupt 人工介入；第 6 题被路由 Agent 分流到 chat。

> 调试 Server 也可单独运行 `python mcp_server.py`（阻塞等待 stdio 输入）。

## 代码结构

| 组件 | 作用 |
|---|---|
| `mcp_server.py` | FastMCP Server：`hybrid_search` / `rerank` 工具与 `kb://docs/{source}` 资源，持有知识库与检索凭证 |
| `mcp_client` | FastMCP Client：以 stdio 子进程拉起 Server，建立会话 |
| `mcp_hybrid_search` / `mcp_rerank` | Client 侧封装：调用 MCP 工具 + `@observe` 补 Langfuse span |
| `retrieve_node` | 检索 Agent：连续两次 MCP 调用（混合检索 → Rerank） |
| `skills/rag-rescue/` | 低分自救 Skill：SKILL.md 手册（策略顺序/放弃条件，知识外置）+ `scripts/extract_terms.py`（jieba 术语抽取） |
| `rescue_node` | 自救 Skill 节点：加载 SKILL.md 选手册策略、生成候选查询、MCP 重检；达标直达生成，3 轮全败升级人工门 |
| 其余节点 | 与 LangGraph-demo 一致：router / rewrite / human_gate / refuse / generate / critic / direct_answer |

## 扩展建议

- **HTTP 传输**：FastMCP `Client` 同样支持连接远程 HTTP Server，可将检索服务部署为独立远程服务，多个 Agent 共享。
- **更多检索工具**：在 Server 侧新增 `@mcp.tool()`（如按 category 过滤、全文读取），Agent 侧无需改动编排即可通过 `list_tools` 发现。
- **Server 侧追踪**：如需观测 Server 内部耗时，可在 `mcp_server.py` 中引入 Langfuse/Python SDK 上报（当前 span 补在 Client 侧）。
- **更多自救策略**：直接在 `skills/rag-rescue/SKILL.md` 中追加策略（如按 category 过滤检索、多轮 HyDE），rescue 节点按手册通用循环执行，无需改动代码。
- **更多 Skill**：可仿照 rag-rescue 的模式新增其他 SKILL.md 手册（如生成质量自检、审核打回排查），由对应节点加载手册驱动，实现"经验知识外置、代码保持稳定"。
