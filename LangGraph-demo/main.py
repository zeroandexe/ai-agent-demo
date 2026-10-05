# -*- coding: utf-8 -*-
# LangGraph 多Agent协作 + 人工介入版 RAG demo（与 main.py 的 LCEL 单链版本对照）
#
# 架构: 路由Agent(知识问答/闲聊分流) -> 改写Agent(指代消解) -> 检索Agent(混合检索+RRF融合+Rerank)
#       -> 人工介入门(低分interrupt，由人决定 拒答/强制生成/换查询重检)
#       -> 生成Agent(带出处生成) -> 审核Agent(检查答案是否基于上下文，不合格打回重新生成)
#
# 追踪: Langfuse自托管(http://localhost:3000)，LangGraph节点内的LLM调用经回调自动上报，
#       裸requests的rerank仍用@observe手动补span

import os

# ============ Langfuse 追踪配置 ============
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"   # 改成你实际访问的地址
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-xxxxxx"    # 替换为Project的Public Key
os.environ["LANGFUSE_SECRET_KEY"] = "sk-xxxxxx"    # 替换为Project的Secret Key

import math
import requests
import jieba

from typing import List, Optional, Tuple, TypedDict

from langchain_siliconflow import ChatSiliconFlow, SiliconFlowEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# LangGraph: 状态图编排 + interrupt人工介入 + checkpointer断点续跑
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

# Langfuse LangChain回调（自动追踪全链路）+ 手工追踪装饰器
from langfuse.langchain import CallbackHandler
from langfuse import get_client, observe

langfuse = get_client()
langfuse_handler = CallbackHandler()

# demo采用硅基流动的api进行验证，langchain已经封装了openai，并且硅基流动与deepseek等模型提供商都支持openai协议格式。
SILICONFLOW_API_KEY = "sk-xxxxx"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# LLM推理模型
LLM_MODEL = "zai-org/GLM-5.3"
# 文本嵌入模型，用于将输入进行向量化，便与从向量数据库进行相似性检测，得到召回清单。
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
# 重排序模型，将检索召回块融合后的召回清单与问题进行和对精选，得到最终召回清单。
RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-8B"

# 重排序后的第一条召回块的评估阈值，低于该阈值则触发人工介入，而不是像main.py那样直接代码短路拒答。
REFUSE_THRESHOLD = 0.3
# 默认提示语，向用户显示反馈当前的问题无法处理。
REFUSE_TEXT = "知识库中未找到足够相关的内容，建议咨询对应部门。"
# 审核Agent判定不合格时允许打回生成的最大次数，防止两个Agent死循环。
MAX_GENERATE_RETRY = 1


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


# 使用硅基流动服务来处理召回块的重新排序，并得到精确核对后的召回块
# @observe: rerank是裸requests调用，LangChain回调追踪不到，手动补span（含分数，方便在Langfuse里看拒答原因）
@observe(name="siliconflow-rerank")
def siliconflow_rerank(query, documents, top_n=3) -> List[Tuple[Optional[float], Document]]:
    if not documents:
        return []
    url = f"{SILICONFLOW_BASE_URL}/rerank"
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
    langfuse.update_current_span(metadata={"rerank_scores": [s for s, _ in out]})
    return out[:top_n]


# ============ 各Agent使用的提示语 ============

# 路由Agent：判断问题是否需要查知识库，决定走RAG链路还是直接闲聊回答
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是路由分发助手。判断用户问题属于哪一类：\n"
     "- rag: 需要查询技术知识库才能回答的问题（算法、框架、技术概念等）\n"
     "- chat: 闲聊、创作、或与知识库无关的通用请求\n"
     "只输出 rag 或 chat，不要输出任何其他内容。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 改写Agent：query重写模板，消除原始query的指代问题
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个查询改写助手。根据对话历史，将用户的新问题改写为一个"
     "独立、完整、不依赖上下文的问题。只输出改写后的问题，不要添加任何解释。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 生成Agent：基于精选召回块+历史对话回答用户的原始query（并非改写后的query）
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个知识助手。请仅基于以下上下文回答用户的问题，并在答案中标注来源编号。"
     "如果上下文不足以回答，请如实说根据现有资料无法回答。\n\n上下文：\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 审核Agent：检查答案是否忠于上下文、是否标注了来源，不合格则打回生成Agent
critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是答案审核员。检查下面的答案是否满足：1) 内容确实来自给定上下文，没有编造；"
     "2) 标注了来源；3) 回答了用户的问题。\n"
     "第一行只输出 PASS 或 FAIL，第二行起输出简要理由（FAIL时必须指出如何修改）。\n\n"
     "上下文：\n{context}"),
    ("human", "用户问题：{question}\n\n答案：{answer}"),
])


# ============ LangGraph 状态定义 ============

class RagState(TypedDict, total=False):
    question: str                                 # 用户原始问题
    chat_history: list                            # 多轮对话历史（由外层循环维护）
    route: str                                    # 路由Agent的分流结果: "rag" | "chat"
    rewritten: str                                # 改写后的独立查询
    retrieved: List[Tuple[Optional[float], Document]]  # rerank后的(分数, 召回块)
    top1_score: Optional[float]                   # Top1召回块的相关性分数
    human_action: str                             # 人工介入决策: pass / refuse / force / rewrite
    answer: str                                   # 最终答案
    critic_ok: bool                               # 审核Agent是否通过
    critic_feedback: str                          # 审核不通过的修改意见
    retry_count: int                              # 生成被打回的次数


# ============ 节点（每个节点即一个Agent的职责） ============

# 路由Agent节点
def router_node(state: RagState):
    route = (router_prompt | llm | StrOutputParser()).invoke({
        "question": state["question"],
        "chat_history": state.get("chat_history", []),
    }).strip().lower()
    # 根据回答进行用户问题的路由分配
    route = "chat" if "chat" in route else "rag"
    print(f"  [路由Agent] 分流到: {route}")
    return {"route": route}


# 改写Agent节点：从第二轮开始对原始query进行指代消解
def rewrite_node(state: RagState):
    history = state.get("chat_history", [])
    if history:
        # 存在历史聊天记录进行query改写（指代消除、hyde等等都可以在这一步进行处理来提高检索率）
        rewritten = (condense_prompt | llm | StrOutputParser()).invoke({
            "question": state["question"], "chat_history": history,
        })
        print(f"  [改写Agent] 改写后查询: {rewritten}")
    else:
        rewritten = state["question"]
    return {"rewritten": rewritten}


# 检索Agent节点：混合检索 -> RRF融合去重 -> Rerank精选
def retrieve_node(state: RagState):
    docs = dedup_by_text(ensemble_retriever.invoke(state["rewritten"]))
    print(f"  [检索Agent] RRF融合+去重: {len(docs)} 个块")
    scored = siliconflow_rerank(state["rewritten"], docs, top_n=3)
    top1 = scored[0][0] if scored else None
    if scored:
        print(f"  [检索Agent] Rerank Top1分数: {top1}")
    return {"retrieved": scored, "top1_score": top1}


# 人工介入门：分数达标直接放行；不达标则interrupt挂起整个图，
# 等待人工决策后从断点续跑（需要checkpointer支持）。
def human_gate_node(state: RagState):
    top1 = state.get("top1_score")
    if state.get("retrieved") and top1 is not None and top1 >= REFUSE_THRESHOLD:
        return {"human_action": "pass"}

    decision = interrupt({
        "reason": "检索相关性分数过低，直接生成有幻觉风险",
        "question": state["question"],
        "rewritten": state["rewritten"],
        "top1_score": top1,
        "options": {
            "refuse": "拒答，不消耗生成token",
            "force": "无视低分，强制生成答案",
            "rewrite": "由人工提供一个新查询，重新走检索",
        },
    })
    # interrupt恢复后，decision即人工传入的resume值: {"action": ..., "query": 可选}
    print(f"  [人工介入] 决策: {decision}")
    if decision["action"] == "rewrite":
        return {"human_action": "rewrite", "rewritten": decision.get("query", state["rewritten"])}
    return {"human_action": decision["action"]}


# 拒答节点：人工选择refuse时走这里，零生成成本
def refuse_node(state: RagState):
    return {"answer": REFUSE_TEXT}


# 生成Agent节点：带出处生成；若被审核Agent打回，则附带修改意见重新生成
def generate_node(state: RagState):
    context = "\n\n---\n\n".join([
        f"[来源: {doc.metadata.get('source', 'unknown')}]{'(分数:' + str(round(s, 3)) + ')' if s is not None else ''}\n{doc.page_content}"
        for s, doc in state.get("retrieved", [])
    ])
    question = state["question"]
    if state.get("retry_count"):
        question += f"\n\n（审核意见：{state['critic_feedback']}，请据此修正）"
    messages = answer_prompt.invoke({
        "context": context, "question": question,
        "chat_history": state.get("chat_history", []),
    })
    answer = llm.invoke(messages).content
    return {"answer": answer}


# 审核Agent节点：对生成结果把关，输出PASS/FAIL及理由
def critic_node(state: RagState):
    context = "\n\n".join(d.page_content for _, d in state.get("retrieved", []))
    verdict = (critic_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": state["question"], "answer": state["answer"],
    })
    first_line, _, reason = verdict.partition("\n")
    ok = first_line.strip().upper().startswith("PASS")
    print(f"  [审核Agent] {'通过' if ok else '打回: ' + reason.strip()}")
    return {
        "critic_ok": ok,
        "critic_feedback": "" if ok else reason.strip(),
        "retry_count": state.get("retry_count", 0) + (0 if ok else 1),
    }


# 闲聊直答节点：路由判定为chat时不走检索，直接由LLM回答
def direct_answer_node(state: RagState):
    answer = llm.invoke(state.get("chat_history", []) + [HumanMessage(state["question"])]).content
    return {"answer": answer}


# ============ 条件边 ============

def route_after_router(state: RagState):
    return "rewrite" if state["route"] == "rag" else "direct_answer"


def route_after_human_gate(state: RagState):
    action = state["human_action"]
    if action == "refuse":
        return "refuse"
    if action == "rewrite":
        return "retrieve"      # 用人工给的新查询重新检索
    return "generate"          # pass / force


def route_after_critic(state: RagState):
    if state["critic_ok"] or state.get("retry_count", 0) >= MAX_GENERATE_RETRY:
        return END
    return "generate"


# ============ 图的构建与编译 ============

builder = StateGraph(RagState)
# 注册状态机中的执行节点
builder.add_node("router", router_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("human_gate", human_gate_node)
builder.add_node("refuse", refuse_node)
builder.add_node("generate", generate_node)
builder.add_node("critic", critic_node)
builder.add_node("direct_answer", direct_answer_node)

# 设置直接节点之间的转换边界
builder.add_edge(START, "router")                            # 指定状态机的入口，进入路由节点
builder.add_conditional_edges("router", route_after_router,  # 路由节点执行完成之后的边界条件判断
                              {"rewrite": "rewrite", "direct_answer": "direct_answer"})   # 根据条件判断结果，决定下一个执行节点
builder.add_edge("rewrite", "retrieve")                      # query改写完走检索
builder.add_edge("retrieve", "human_gate")                   # 检索完走人工
builder.add_conditional_edges("human_gate", route_after_human_gate,
                              {"refuse": "refuse", "retrieve": "retrieve", "generate": "generate"})
builder.add_edge("refuse", END)                              # 问题被拒绝离开状态机
builder.add_edge("generate", "critic")                       # 根据检索生成完后，走审核
builder.add_conditional_edges("critic", route_after_critic,
                              {END: END, "generate": "generate"})
builder.add_edge("direct_answer", END)                       # 闲聊直接退出状态机

# interrupt需要checkpointer保存断点状态，resume时才能从挂起点继续执行
graph = builder.compile(checkpointer=InMemorySaver())


# 读取一次人工决策（interrupt恢复所需的resume值）
def ask_human_decision(payload: dict) -> dict:
    print(f"\n  >>> 人工介入请求: {payload['reason']}")
    print(f"      问题: {payload['question']} | 改写后: {payload['rewritten']} | Top1分数: {payload['top1_score']}")
    for k, v in payload["options"].items():
        print(f"      - {k}: {v}")
    try:
        action = input("      请选择 [refuse/force/rewrite]（直接回车=refuse）: ").strip() or "refuse"
    except (EOFError, KeyboardInterrupt):
        action = "refuse"
    if action == "rewrite":
        try:
            query = input("      请输入新的检索查询: ").strip()
        except (EOFError, KeyboardInterrupt):
            query = ""
        return {"action": "rewrite", "query": query or payload["rewritten"]}
    if action not in ("refuse", "force"):
        action = "refuse"
    return {"action": action}


# demo程序入口，演示多轮对话 + 多Agent协作 + 人工介入
if __name__ == "__main__":
    # thread_id即会话id，checkpointer按它隔离状态；Langfuse回调挂上后整条图链路自动上报
    config = {
        "configurable": {"thread_id": "demo-001"},
        "callbacks": [langfuse_handler],
        "recursion_limit": 30,
    }
    questions = [
        "什么是 RRF？",
        "它的公式是什么？",          # 改写Agent做指代消解："RRF的公式是什么"
        "那向量检索和倒排索引有什么区别？",
        "HNSW 索引的查询复杂度是多少？",  # 知识库中没有，低分触发人工介入interrupt
        "帮我写一首关于秋天的诗",       # 路由Agent分流到chat，不消耗检索资源
    ]

    history: list = []
    for q in questions:
        print(f"\n{'='*60}")
        print(f"用户: {q}")
        result = graph.invoke(
            {"question": q, "chat_history": list(history),
             "retry_count": 0, "critic_feedback": "", "human_action": ""},
            config=config,
        )
        # Agent工作流触发了中断，在图被interrupt挂起时会返回 __interrupt__，人工输入后以Command(resume=...)从断点续跑
        while result.get("__interrupt__"):
            decision = ask_human_decision(result["__interrupt__"][0].value)
            # 回到中断点所在的函数头继续跑
            result = graph.invoke(Command(resume=decision), config=config)

        answer = result.get("answer", REFUSE_TEXT)
        print(f"助手: {answer}")
        history += [HumanMessage(q), AIMessage(answer)]

    # 程序退出前flush，确保最后几条trace上报完成
    langfuse.flush()
