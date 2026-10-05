# LangGraph 多 Agent 协作 + 人工介入版 RAG Demo

一个基于 **LangGraph + LangChain + Langfuse + SiliconFlow API** 的多轮对话 RAG 演示项目。与 [langChain-demo](../langChain-demo/README.md) 的 LCEL 单链版本使用同一份知识库与同一套检索组件，但改用 **LangGraph 状态图** 编排多个 Agent，并加入 **interrupt 人工介入** 与 **审核打回** 机制，可作为两种编排方式的对照学习材料。

## 架构流程

```
原始 Query
   │
   ▼
路由 Agent                     ← 判断走 RAG 链路（rag）还是闲聊直答（chat）
   │ (rag)
   ▼
改写 Agent（指代消解）          ← 结合对话历史，将"它是什么"改写为独立完整的问题
   │
   ▼
检索 Agent                     ← 混合检索（向量 + BM25）→ RRF 融合去重 → Rerank 精排
   │
   ▼
人工介入门                     ← Top1 分数 < 0.3 时 interrupt 挂起，由人决定：
   │                             refuse（拒答）/ force（强制生成）/ rewrite（换查询重检）
   ▼
生成 Agent                     ← 注入精选上下文 + 历史对话 + 原始 query（非改写后），带出处生成
   │
   ▼
审核 Agent                     ← 检查答案是否忠于上下文、标注来源、回答问题，
   │                             不合格打回生成 Agent（最多打回 1 次，防死循环）
   ▼
最终答案
```

## 与 LCEL 单链版的差异

| 维度 | langChain-demo（LCEL） | 本 Demo（LangGraph） |
|---|---|---|
| 编排方式 | `RunnableLambda` 单链 + `RunnableWithMessageHistory` | `StateGraph` 状态图 + 条件边 |
| 低分处理 | 代码层短路，直接拒答 | interrupt 挂起，**人工决策**（拒答/强制生成/换查询重检） |
| 闲聊分流 | 无 | 路由 Agent 分流到直答节点，不消耗检索资源 |
| 答案把关 | 无 | 审核 Agent 检查，不合格打回重新生成 |
| 会话历史 | `RunnableWithMessageHistory` 自动维护 | 外层循环手动维护，`checkpointer` 按 `thread_id` 隔离中断状态 |

## 特性亮点

- **路由分流**：路由 Agent 判断问题类型，闲聊/创作类请求直接由 LLM 回答，不经过检索链路。
- **人工介入（Human-in-the-loop）**：检索相关性不足时通过 `interrupt` 挂起整个图，人工可选择拒答（零生成成本）、强制生成、或提供新查询重新检索；借助 `InMemorySaver` checkpointer 从断点续跑。
- **生成-审核闭环**：审核 Agent 检查答案是否忠于上下文、是否标注来源，FAIL 时附带修改意见打回生成 Agent，`MAX_GENERATE_RETRY = 1` 防止两个 Agent 死循环。
- **检索流水线与单链版一致**：查询改写 → 混合检索（向量 + BM25，权重各 0.5）→ RRF 融合 → Rerank 精排，Rerank 异常时降级为不判分直接截取。
- **全链路追踪**：LangGraph 节点内的 LLM 调用经 Langfuse 回调自动上报；裸 `requests` 的 Rerank 用 `@observe` 手动补 span。

## 环境依赖

```bash
pip install langchain-core langchain-community langchain-classic langchain-siliconflow langgraph langfuse jieba requests
```

> Demo 依赖 `langchain_siliconflow` 中封装的 SiliconFlow API（兼容 OpenAI 协议格式）。

## 配置说明

在 `main.py` 顶部修改以下配置：

| 配置项 | 说明 |
|---|---|
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥 |
| `LLM_MODEL` | 推理模型，默认 `zai-org/GLM-5.3` |
| `EMBEDDING_MODEL` | 向量嵌入模型，默认 `Qwen/Qwen3-VL-Embedding-8B` |
| `RERANK_MODEL` | 重排序模型，默认 `Qwen/Qwen3-VL-Reranker-8B` |
| `REFUSE_THRESHOLD` | 触发人工介入的 Top1 分数阈值，默认 `0.3` |
| `REFUSE_TEXT` | 拒答文案 |
| `MAX_GENERATE_RETRY` | 审核打回的最大重试次数，默认 `1` |

### Langfuse 追踪（可选）

`main.py` 顶部通过环境变量配置 Langfuse（默认指向自托管实例）：

| 环境变量 | 说明 |
|---|---|
| `LANGFUSE_HOST` | Langfuse 地址，默认 `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse Project 的密钥对 |

## 运行

```bash
python main.py
```

Demo 内置 5 个演示问题（单会话 `thread_id = demo-001`）：

1. `什么是 RRF？` —— 首轮，直接检索
2. `它的公式是什么？` —— 演示改写 Agent 指代消解
3. `那向量检索和倒排索引有什么区别？` —— 专有名词利于 BM25 精确召回
4. `HNSW 索引的查询复杂度是多少？` —— 知识库中没有，低分触发 **interrupt 人工介入**（运行时需在终端输入决策）
5. `帮我写一首关于秋天的诗` —— 路由 Agent 分流到 chat，不消耗检索资源

> 注意：第 4 个问题触发人工介入时程序会等待终端输入，选择 `refuse` / `force` / `rewrite` 后图从断点继续执行。

## 代码结构

| 组件 | 作用 |
|---|---|
| `RagState` | LangGraph 状态定义：question / route / rewritten / retrieved / human_action / answer / critic 结果等 |
| `router_node` | 路由 Agent：rag / chat 分流 |
| `rewrite_node` | 改写 Agent：指代消解，生成独立查询 |
| `retrieve_node` | 检索 Agent：混合检索 → RRF 融合去重 → Rerank 精选 |
| `human_gate_node` | 人工介入门：达标放行，不达标 `interrupt` 挂起等待人工决策 |
| `refuse_node` / `direct_answer_node` | 拒答 / 闲聊直答 |
| `generate_node` | 生成 Agent：带出处生成，被打回时附带审核意见重新生成 |
| `critic_node` | 审核 Agent：输出 PASS / FAIL 及理由 |
| `ask_human_decision` | 终端读取人工决策，作为 `Command(resume=...)` 的恢复值 |
| `InMemorySaver` | checkpointer：保存 interrupt 断点状态，按 `thread_id` 隔离 |

## 扩展建议

- **持久化 checkpointer**：将 `InMemorySaver` 替换为 SQLite / Postgres 后端，人工介入决策可跨进程恢复。
- **异步 interrupt 处理**：将人工决策接入外部审批系统（如 IM 机器人、工单），而非终端输入。
- **审核 Agent 强化**：引入引用校验（答案中的来源编号是否真实存在于上下文）等程序化检查，减少纯 LLM 审核的误判。
