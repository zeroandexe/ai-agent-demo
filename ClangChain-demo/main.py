# -*- coding: utf-8 -*-
# Agent编排SDK对RAG的执行流程demo
#
# 架构: 查询改写(指代消解) -> 混合检索(向量+BM25) -> RRF融合 -> Rerank精排
#       -> 阈值拒答(代码层短路) -> 带出处生成(原始query+历史)
#

import math
import json
import os
import sys
import requests
import jieba

from typing import List, Optional, Tuple
from langchain_siliconflow import ChatSiliconFlow, SiliconFlowEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# demo采用轨迹流动的api进行验证，langchain已经封装了openai，并且轨迹流动与deepseek等模型提供商都支持openai协议格式。
SILICONFLOW_API_KEY = "sk-xxxxx"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# LLM推理模型
LLM_MODEL = "zai-org/GLM-5.3"
# 文本嵌入模型，用于将输入进行向量化，便与从向量数据库进行相似性检测，得到召回清单。
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
# 重排序模型，将检索召回块融合后的召回清单与问题进行和对精选，得到最终召回清单。
RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-8B"


# 重排序后的第一条召回块的评估阈值，低于该阈值则视为该问题无法回答，减少LLM推理的token消耗。
REFUSE_THRESHOLD = 0.3
# 默认提示语，向用户显示反馈当前的问题无法处理。
REFUSE_TEXT = "知识库中未找到足够相关的内容，建议咨询对应部门。"


# 配置推理模型对象
llm = ChatSiliconFlow(
    model=LLM_MODEL,
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    temperature=0,
)

# 配置文本嵌入模型对象
embeddings = SiliconFlowEmbeddings(
    model=EMBEDDING_MODEL,
    siliconflow_api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
)

# 模拟RAG当前数据库中的知识库内容
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

# 对知识库中的知识块进行向量匹配
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# 向量检索器（携带embedding模型，以及对数据库的向量检索方法，完成: 查询->embedding->向量检索->得到召回块的流程封装）
class MockVectorRetriever(BaseRetriever):
    documents: List[Document]
    embedding_model: object
    k: int = 4
    doc_vectors: List[List[float]] = []

    def __init__(self, documents, embedding_model, k=4, **kwargs):
        super().__init__(documents=documents, embedding_model=embedding_model, k=k, **kwargs)
        self.doc_vectors = embedding_model.embed_documents([d.page_content for d in documents])

    def _get_relevant_documents(self, query, *, run_manager):
        qv = self.embedding_model.embed_query(query)
        scored = [(cosine_similarity(qv, dv), d) for d, dv in zip(self.documents, self.doc_vectors)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[: self.k]]

vector_retriever = MockVectorRetriever(documents=knowledge_base, embedding_model=embeddings, k=4)

# 倒排索引方法BM25的检索器构造
bm25_retriever = BM25Retriever.from_documents(
    knowledge_base,
    k=4,
    preprocess_func=lambda text: list(jieba.cut(text)),
)

# 混合检索召回块的融合封装，并提供渠道权值的控制。
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

def dedup_by_text(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

# 使用轨迹流动服务来处理召回块的重新排序，并得到精确核对后的召回块
def siliconflow_rerank(query, documents, top_n=3) -> List[Tuple[Optional[float], Document]]:
    if not documents:
        return []
    # 走轨迹流动的rerank重排序服务路由
    url = f"{SILICONFLOW_BASE_URL}/rerank"
    # 将所有召回块发往模型提供商进行处理
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [d.page_content for d in documents],
        "top_n": top_n,
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Rerank ERROR] {e}，退化为不判分直接截取")
        return [(None, d) for d in documents[:top_n]]
    out = []
    for item in result.get("results", []):
        idx = item["index"]
        if idx < len(documents):
            out.append((item.get("relevance_score"), documents[idx]))
    return out[:top_n]

# query重写模板：消除原始query的语义问题
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个查询改写助手。根据对话历史，将用户的新问题改写为一个"
     "独立、完整、不依赖上下文的问题。只输出改写后的问题，不要添加任何解释。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
condense_chain = condense_prompt | llm | StrOutputParser()

# 构建最终生成LLM推理的上下文，包含精选召回块的临时注入，并包含历史对话，与用户的原始query（注意并非改写后的query，不然那就不是回答用户的问题了。）
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个知识助手。请仅基于以下上下文回答用户的问题，并在答案中标注来源编号。"
     "如果上下文不足以回答，请如实说根据现有资料无法回答。\n\n上下文：\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 召回的完整流程（query改写->检索->融合->重排序->精选召回块），得到精选召回块
def retrieve(question: str, chat_history: list):
    # 从第二轮开始对原始query进行语义消除（实际上可以在原始query->检索中间添加一个，小模型7B左右的小模型来检查用户语义需要进行如何处理，这样可以更好的处理）。
    if chat_history:
        rewritten = condense_chain.invoke({"question": question, "chat_history": chat_history})
        print(f"  [改写后查询] {rewritten}")
    else:
        rewritten = question

    # 对重写query进行混合检索，得到融合后的召回块
    docs = dedup_by_text(ensemble_retriever.invoke(rewritten))
    print(f"  [RRF融合+去重] {len(docs)} 个块")

    # 将召回块交给模型提供商的rerank模型进行精选（这里可以对模型提供商进行抽象化（参数化、多态化都可以），达到切换模型商代码不变的效果）
    scored = siliconflow_rerank(rewritten, docs, top_n=3)
    if scored:
        print(f"  [Rerank Top1分数] {scored[0][0]}")

    # 对第一个精选召回块进行分值判断，低于阈值则认为本次无法得到一个有效的回答，避免对LLM的token进行浪费。
    top1 = scored[0][0] if scored else None
    if not scored or (top1 is not None and top1 < REFUSE_THRESHOLD):
        return None
    return scored

def build_messages(question, chat_history, scored):
    context = "\n\n---\n\n".join([
        f"[来源: {doc.metadata.get('source', 'unknown')}]{'(分数:' + str(round(s, 3)) + ')' if s is not None else ''}\n{doc.page_content}"
        for s, doc in scored
    ])
    return answer_prompt.invoke({"context": context, "question": question, "chat_history": chat_history})

# 根据当前的上下文进行对用户问题查询的步骤封装，得到推理模型的推理结果
def run_rag(inputs: dict) -> str:
    question = inputs["question"]
    history = inputs.get("chat_history", [])
    scored = retrieve(question, history)
    if scored is None:
        return REFUSE_TEXT                      # 短路：不调 LLM，零成本零延迟
    messages = build_messages(question, history, scored)   # 生成端用原始 question
    return llm.invoke(messages).content

# 用户的多轮对话管理
store = {}

# 根据用户的会话id得到当前的历史信息
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建rag的执行对象
rag_chain = RunnableWithMessageHistory(
    RunnableLambda(run_rag),
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# demo程序入口，演示多轮对话对RAG的使用
if __name__ == "__main__":
    config = {"configurable": {"session_id": "demo-001"}}
    questions = [
        "什么是 RRF？",
        "它的公式是什么？",        # 消除指代处理，将转换为：“RRF的公式是什么”，从而精确用户的查询
        "那向量检索和倒排索引有什么区别？", # 这里  "向量检索"、"倒排索引"，两个专用名词会包含精确的信息，非常有利于BM25的检索，同时"区别"能够使LLM找到用户的推理结果期望。
        "Rerank 又是做什么的？",
        "帮我写一首关于秋天的诗",   # 知识库中并包含"秋天"、"诗"这些东西，倒排续检索不过，并且向量检索也不会得到信息，因此在召回率非常低，对于重排序也只能得到非常低的记过，让LLM去回答浪费token，并且也是幻觉。
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"用户: {q}")
        ans = rag_chain.invoke({"question": q}, config=config)
        print(f"助手: {ans}")