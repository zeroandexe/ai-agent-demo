# -*- coding: utf-8 -*-
# LangGraph 多Agent协作 + 人工介入版 RAG demo —— MCP 改造版（v2：FastMCP Client）
#
# 与原版 LangGraph 的差异：
#   - 检索能力整体移入 mcp_server.py（MCP Server，stdio 子进程）：
#       向量检索 / BM25 / RRF融合去重 / Rerank 精排，及 SILICONFLOW_API_KEY
#   - 本文件只保留"编排与推理"：路由/改写/生成/审核四个Agent + 人工介入门
#   - Agent 侧用 FastMCP 自带的 Client 连接 Server
#   - Langfuse 追踪在 Agent 端：LLM 走 LangChain 回调，两次 MCP 调用用 @observe 手动补 span

import os

# ============ Langfuse 追踪配置 ============
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"   # 改成你实际访问的地址
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-xxxxxx"     # 替换为Project的Public Key
os.environ["LANGFUSE_SECRET_KEY"] = "sk-xxxxxx""    # 替换为Project的Secret Key

import json
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

from langchain_siliconflow import ChatSiliconFlow
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# LangGraph: 状态图编排 + interrupt人工介入 + checkpointer断点续跑
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

# Langfuse LangChain回调（自动追踪全链路）+ 手工追踪装饰器
from langfuse.langchain import CallbackHandler
from langfuse import get_client, observe

# MCP client：FastMCP 自带的客户端，支持 stdio 子进程 / HTTP 等方式
from fastmcp import Client

langfuse = get_client()
langfuse_handler = CallbackHandler()

# demo采用硅基流动的api进行验证。注意：embedding/rerank 的 key 已移入 mcp_server.py
SILICONFLOW_API_KEY = "sk-xxxxxx""
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# LLM推理模型
LLM_MODEL = "zai-org/GLM-5.3"

# 重排序后的第一条召回块的评估阈值，低于该阈值触发人工介入
REFUSE_THRESHOLD = 0.3
# 默认提示语
REFUSE_TEXT = "知识库中未找到足够相关的内容，建议咨询对应部门。"
# 审核Agent判定不合格时允许打回生成的最大次数
MAX_GENERATE_RETRY = 1

# 配置推理模型对象（Agent 侧仅保留生成/判别类 LLM）
llm = ChatSiliconFlow(
    model=LLM_MODEL,
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    temperature=0,
)


# ============ MCP Client：连接检索能力 Server ============
MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")

# 传入脚本路径，FastMCP 自动以 stdio 子进程方式拉起 Server
mcp_client = Client(Path(MCP_SERVER_SCRIPT))

# 会话在 main() 的 async with 中建立；全局引用供节点函数使用
_mcp_session = None


def _extract_text(result) -> str:
    """从 MCP 调用结果中提取文本内容（兼容不同版本的返回结构）"""
    content = getattr(result, "content", None)
    if content:
        return "\n".join(getattr(c, "text", str(c)) for c in content)
    return str(result)


async def _mcp_call_tool(name: str, arguments: dict) -> str:
    result = await _mcp_session.call_tool(name, arguments)
    return _extract_text(result)


# ---- Langfuse 手动 span：MCP 调用发生在 Server 进程，LangChain 回调追踪不到，在此补 ----
@observe(name="mcp-hybrid-search")
async def mcp_hybrid_search(query: str) -> List[Document]:
    """经 MCP 调用混合检索（向量+BM25+RRF融合+去重），返回候选文档块"""
    text = await _mcp_call_tool("hybrid_search", {"query": query})
    items = json.loads(text)
    langfuse.update_current_span(metadata={"query": query, "candidates": len(items)})
    return [
        Document(
            page_content=it["content"],
            metadata={"source": it["source"], "category": it.get("category", "")},
        )
        for it in items
    ]


@observe(name="mcp-rerank")
async def mcp_rerank(query: str, documents: List[Document], top_n: int = 3) -> List[Tuple[Optional[float], Document]]:
    """经 MCP 调用 Rerank 精排，返回 (分数, 文档) 列表；分数可能为 None（服务降级）"""
    text = await _mcp_call_tool("rerank", {
        "query": query,
        "documents": [d.page_content for d in documents],
        "top_n": top_n,
    })
    results = json.loads(text)
    langfuse.update_current_span(metadata={"rerank_scores": [r["score"] for r in results]})
    out = []
    for r in results:
        idx = r["index"]
        if idx < len(documents):
            out.append((r["score"], documents[idx]))
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
    route = "chat" if "chat" in route else "rag"
    print(f"  [路由Agent] 分流到: {route}")
    return {"route": route}


# 改写Agent节点：从第二轮开始对原始query进行指代消解
def rewrite_node(state: RagState):
    history = state.get("chat_history", [])
    if history:
        rewritten = (condense_prompt | llm | StrOutputParser()).invoke({
            "question": state["question"], "chat_history": history,
        })
        print(f"  [改写Agent] 改写后查询: {rewritten}")
    else:
        rewritten = state["question"]
    return {"rewritten": rewritten}


# 检索Agent节点（异步）：两次 MCP 工具调用（混合检索 -> Rerank精排），本体逻辑已在 Server
async def retrieve_node(state: RagState):
    docs = await mcp_hybrid_search(state["rewritten"])
    print(f"  [检索Agent] 混合检索(MCP): {len(docs)} 个候选块")
    scored = await mcp_rerank(state["rewritten"], docs, top_n=3)
    top1 = scored[0][0] if scored else None
    if scored:
        print(f"  [检索Agent] Rerank(MCP) Top1分数: {top1}")
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
builder.add_node("router", router_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("human_gate", human_gate_node)
builder.add_node("refuse", refuse_node)
builder.add_node("generate", generate_node)
builder.add_node("critic", critic_node)
builder.add_node("direct_answer", direct_answer_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_after_router,
                              {"rewrite": "rewrite", "direct_answer": "direct_answer"})
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "human_gate")
builder.add_conditional_edges("human_gate", route_after_human_gate,
                              {"refuse": "refuse", "retrieve": "retrieve", "generate": "generate"})
builder.add_edge("refuse", END)
builder.add_edge("generate", "critic")
builder.add_conditional_edges("critic", route_after_critic,
                              {END: END, "generate": "generate"})
builder.add_edge("direct_answer", END)

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


# demo主流程：建立MCP会话后跑多轮对话（异步驱动整张图）
async def main():
    global _mcp_session
    config = {
        "configurable": {"thread_id": "demo-001"},
        "callbacks": [langfuse_handler],
        "recursion_limit": 30,
    }
    questions = [
        "什么是 RRF？",
        "它的公式是什么？",
        "那向量检索和倒排索引有什么区别？",
        "HNSW 索引的查询复杂度是多少？",  # 低分触发人工介入interrupt
        "帮我写一首关于秋天的诗",           # 路由Agent分流到chat
    ]

    async with mcp_client as session:
        _mcp_session = session
        # 连接建立后做能力发现（tools/list）
        discovered = await session.list_tools()
        tool_names = [t.name for t in getattr(discovered, "tools", discovered)]
        print(f"[MCP] 已连接检索 Server，发现工具: {tool_names}")

        history: list = []
        try:
            for q in questions:
                print(f"\n{'='*60}")
                print(f"用户: {q}")
                result = await graph.ainvoke(
                    {"question": q, "chat_history": list(history),
                     "retry_count": 0, "critic_feedback": "", "human_action": ""},
                    config=config,
                )
                while result.get("__interrupt__"):
                    decision = ask_human_decision(result["__interrupt__"][0].value)
                    result = await graph.ainvoke(Command(resume=decision), config=config)

                answer = result.get("answer", REFUSE_TEXT)
                print(f"助手: {answer}")
                history += [HumanMessage(q), AIMessage(answer)]
        finally:
            # 退出前flush，确保最后几条trace上报完成（会话由 async with 自动关闭）
            langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
