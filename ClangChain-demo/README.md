# Agent 编排 SDK 的 RAG 执行流程 Demo

一个基于 **LangChain + LangFuse + SiliconFlow API** 的多轮对话 RAG（检索增强生成）演示项目，完整展示了从查询改写、混合检索、RRF 融合、Rerank 精排到阈值拒答的工业级 RAG 流水线。

## 架构流程

```
原始 Query
   │
   ▼
查询改写（指代消解）          ← 结合对话历史，将"它是什么"改写为独立完整的问题
   │
   ▼
混合检索（向量 + BM25）       ← 语义召回 + 关键词精确匹配，优势互补
   │
   ▼
RRF 融合（倒数排名融合）      ← score(d) = Σ 1/(k + rank_i(d))，k 通常取 60
   │
   ▼
Rerank 精排                   ← Cross-Encoder 对候选块精排，输出 relevance_score
   │
   ▼
阈值拒答（代码层短路）        ← Top1 分数 < 0.3 直接拒答，零 LLM 成本、零延迟
   │
   ▼
带出处生成                    ← 注入精选上下文 + 历史对话 + 原始 query（非改写后）
```

## 特性亮点

- **查询改写 / 指代消解**：多轮对话中用户追问"它的公式是什么"，自动结合历史改写为"RRF 的公式是什么"。
- **混合检索**：向量检索（语义理解）+ BM25 倒排索引（专有名词、术语精确匹配），通过 `EnsembleRetriever` 以 RRF 融合，权重各 0.5。
- **Rerank 精排**：调用 SiliconFlow 的 Rerank 模型（Cross-Encoder）对融合后的候选块打分排序，比向量检索的 Bi-Encoder 更精细。
- **阈值拒答**：Rerank 后 Top1 分数低于 `REFUSE_THRESHOLD = 0.3` 时在代码层短路，直接返回拒答文案，不调用 LLM——零 token 消耗、零幻觉风险（如询问"写一首关于秋天的诗"这类知识库外问题）。
- **出处标注**：生成时注入的上下文中包含 `[来源: xxx](分数: xxx)` 元信息，LLM 被要求标注来源编号。
- **注意**：最终生成使用**原始 query** 而非改写后的 query，确保回答的是用户真正的问题。
- **降级容错**：Rerank 服务异常时自动降级为不判分直接截取前 N 个候选块。

## 环境依赖

```bash
pip install langchain-core langchain-community langchain-classic langchain-siliconflow jieba requests
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
| `REFUSE_THRESHOLD` | 拒答阈值，默认 `0.3` |
| `REFUSE_TEXT` | 拒答文案 |

## 运行

```bash
python main.py
```

Demo 内置 5 个演示问题（单会话 `demo-001`）：

1. `什么是 RRF？` —— 首轮，直接检索
2. `它的公式是什么？` —— 演示指代消解改写
3. `那向量检索和倒排索引有什么区别？` —— 专有名词利于 BM25 精确召回
4. `Rerank 又是做什么的？` —— 多轮上下文延续
5. `帮我写一首关于秋天的诗` —— 知识库无相关内容，触发阈值拒答

## 代码结构

| 组件 | 作用 |
|---|---|
| `MockVectorRetriever` | 模拟向量数据库检索器（embedding → 余弦相似度 → TopK） |
| `BM25Retriever` | 基于 jieba 分词的关键词检索器 |
| `EnsembleRetriever` | RRF 融合两个检索器的结果 |
| `dedup_by_text` | 按内容前缀去重，避免重复块干扰 |
| `siliconflow_rerank` | 调用 Rerank API 精排，异常时降级 |
| `condense_chain` | 查询改写链（消除指代） |
| `retrieve` | 完整召回流程，含阈值拒答短路 |
| `run_rag` | RAG 主流程：召回 → 构建消息 → LLM 生成 |
| `rag_chain` | `RunnableWithMessageHistory` 包装，支持多轮会话历史 |

## 内置知识库

`knowledge_base` 包含 6 个文档块，覆盖：RRF 算法、查询改写、向量检索、BM25、Rerank、LangChain，每个块带有 `source` / `category` 元数据。

## 扩展建议

- **查询改写前置小模型**：在原始 query → 检索之间接入 7B 级小模型，判断用户语义需要何种处理，改写更精准。
- **模型提供商抽象**：对 embedding / rerank / LLM 提供商做参数化或多态封装，切换模型商时代码不变。
- **真实向量数据库**：将 `MockVectorRetriever` 替换为 Milvus / FAISS / Chroma 等持久化向量库。
