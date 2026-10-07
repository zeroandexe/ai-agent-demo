# -*- coding: utf-8 -*-
# RAG 检索能力 MCP Server（FastMCP，stdio 传输）
#
# 封装原 main.py 中"检索Agent"依赖的全部检索能力：
#   - 向量检索（SiliconFlow Embedding + 余弦相似度）
#   - BM25 倒排检索（jieba 分词）
#   - RRF 混合融合 + 去重
#   - SiliconFlow Rerank 精排
#
# 与 Agent 侧（main_mcp.py，LangGraph）的分工约定：
#   - 凭证 SILICONFLOW_API_KEY 只存在于本进程，Agent 侧不再接触
#   - Langfuse 追踪留在 Agent 端，由 MCP client 调用处以 @observe 手动补 span
#   - 启动时一次性对知识库做 embedding，之后工具调用 = 本地计算 + rerank API
#
# 运行方式：由 main_mcp.py 通过 stdio 子进程拉起，无需手动启动；
# 也可单独调试：python mcp_server.py （阻塞等待 stdio 输入）

import math
from typing import List, Optional

import requests
import jieba

from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda

from fastmcp import FastMCP

# ---- 凭证与模型配置（仅 Server 侧持有，Agent 侧不再可见）----
SILICONFLOW_API_KEY = "sk-xxxxxx"  # 替换为真实 key
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-8B"

TOP_K = 4  # 每路召回的数量

embeddings = SiliconFlowEmbeddings(
    model=EMBEDDING_MODEL,
    siliconflow_api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
)

# ---- 知识库 corpus（原 main.py 的 knowledge_base，现归 Server 所有）----
knowledge_base = [
    Document(
        page_content="RRF（Reciprocal Rank Fusion，倒数排名融合）是一种将多个检索结果列表合并的算法。它对每个文档在所有结果列表中的排名取倒数后求和，公式为 score(d) = Σ 1/(k + rank_i(d))，其中 k 通常取 60。RRF 不依赖原始相似度分数，因此对不同检索器的分数尺度不敏感。",
        metadata={"source": "rrf", "category": "算法"},
    ),
    Document(
        page_content="查询改写（Query Rewriting）是多轮对话 RAG 中的关键步骤。它的核心作用是将用户带有指代（如'它'、'那个'）的追问，结合对话历史改写成一个独立、完整的查询，从而让检索器能够正确召回相关文档。",
        metadata={"source": "query_rewrite", "category": "技术"},
    ),
    Document(
        page_content="向量检索（Dense Retrieval）通过将查询和文档分别编码为高维向量，然后计算向量之间的余弦相似度来召回相关文档。它的优势是能够理解语义相似性，即使查询和文档没有字面重叠也能召回。",
        metadata={"source": "vector_search", "category": "技术"},
    ),
    Document(
        page_content="倒排索引检索（Inverted Index / BM25）是一种基于关键词匹配的检索方式。BM25 是其中最经典的排序函数，考虑了词频（TF）和逆文档频率（IDF）。优势是精确匹配能力强，适合处理专有名词和术语。",
        metadata={"source": "bm25", "category": "技术"},
    ),
    Document(
        page_content="Rerank（重排序）是在初步召回之后对文档进行精排的步骤。它通常使用 Cross-Encoder 模型，将查询和文档拼接后一起输入模型计算相关性分数。相比向量检索的 Bi-Encoder，Cross-Encoder 能更精细地捕捉查询和文档之间的交互信息，但计算成本更高，只适合对少量候选文档重排。",
        metadata={"source": "rerank", "category": "技术"},
    ),
    Document(
        page_content="LangChain 是一个用于构建大语言模型应用的开源框架。它提供了模块化的组件，包括模型封装、提示模板、输出解析器、检索器、记忆管理等。LCEL 允许用管道符 | 将组件组合成链。LangGraph 是生态中用于构建有状态 Agent 的库。",
        metadata={"source": "langchain", "category": "框架"},
    ),
]


# ---- 向量检索器（查询 -> embedding -> 余弦相似度 -> top-k）----
def _cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class MockVectorRetriever:
    """轻量向量检索器：启动时对 corpus 一次性编码，查询时余弦相似度取 top-k"""

    def __init__(self, documents, embedding_model, k=4):
        self.documents = documents
        self.embedding_model = embedding_model
        self.k = k
        self.doc_vectors = embedding_model.embed_documents(
            [d.page_content for d in documents]
        )

    def invoke(self, query: str) -> List[Document]:
        qv = self.embedding_model.embed_query(query)
        scored = [
            (_cosine_similarity(qv, dv), d)
            for d, dv in zip(self.documents, self.doc_vectors)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[: self.k]]


# 构建两路召回器 + RRF 融合（服务启动时完成，含一次性的全库 embedding）
vector_retriever = RunnableLambda(
    MockVectorRetriever(
        documents=knowledge_base, embedding_model=embeddings, k=TOP_K
    ).invoke
)
bm25_retriever = BM25Retriever.from_documents(
    knowledge_base,
    k=TOP_K,
    preprocess_func=lambda text: list(jieba.cut(text)),
)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)


def _dedup_by_text(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


mcp = FastMCP("rag-retrieval-server")


@mcp.tool()
def hybrid_search(query: str) -> List[dict]:
    """对知识库执行混合检索：向量检索与 BM25 两路召回，RRF 融合并去重。

    返回候选文档块列表，每个元素包含 content/source/category 三个字段，
    按 RRF 融合后的综合顺序排列。供下游 rerank 工具精排。
    """
    docs = _dedup_by_text(ensemble_retriever.invoke(query))
    return [
        {
            "content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "category": d.metadata.get("category", ""),
        }
        for d in docs
    ]


@mcp.tool()
def rerank(query: str, documents: List[str], top_n: int = 3) -> List[dict]:
    """对 hybrid_search 返回的候选文档块按与查询的相关性精排（Cross-Encoder）。

    返回 [{"index": 候选在原列表中的下标, "score": 相关性分数}]，按分数降序。
    服务不可用时 score 返回 None（由调用方决定是否触发人工介入）。
    """
    if not documents:
        return []
    url = f"{SILICONFLOW_BASE_URL}/rerank"
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException:
        # 降级：不判分，直接按传入顺序截取（对应原 demo 的退化逻辑）
        return [
            {"index": i, "score": None} for i in range(min(top_n, len(documents)))
        ]
    return [
        {"index": item["index"], "score": item.get("relevance_score")}
        for item in result.get("results", [])
    ][:top_n]


@mcp.resource("kb://docs/{source}")
def read_doc(source: str) -> str:
    """按 source 编号（如 rrf / bm25 / rerank）读取知识库中的原文"""
    for d in knowledge_base:
        if d.metadata.get("source") == source:
            return d.page_content
    return f"未找到 source={source} 对应的文档"


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输，等待 Client 拉起
