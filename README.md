# ClangChain

基于 **LangChain + LangFuse + SiliconFlow API** 的大模型应用开发示例仓库，包含完整的 RAG（检索增强生成）执行流程演示。

## 项目简介

本项目演示了工业级 RAG 系统的完整流水线：

**查询改写（指代消解）→ 混合检索（向量 + BM25）→ RRF 融合 → Rerank 精排 → 阈值拒答 → 带出处生成**

核心特性：

- 多轮对话中的查询改写，消除"它/那个"等指代歧义
- 向量检索（语义召回）与 BM25（关键词精确匹配）混合，RRF 融合
- Cross-Encoder Rerank 精排，Top1 分数低于阈值时代码层短路拒答，零 token 消耗、零幻觉
- 生成时标注来源编号，且使用原始 query 回答用户真实问题
- 模型提供商抽象设计，切换模型商时代码不变

## 目录结构

```
ClangChain/
├── README.md              # 本文件：项目总览
└── ClangChain-demo/       # RAG 执行流程 Demo
    ├── main.py            # 完整可运行的 RAG 流水线代码
    └── README.md          # Demo 详细文档（架构、配置、运行说明）
```

## 快速开始

```bash
cd ClangChain-demo
# 1. 安装依赖
pip install langchain-core langchain-community langchain-classic langchain-siliconflow jieba requests
# 2. 在 main.py 中填入 SILICONFLOW_API_KEY 并确认模型配置
# 3. 运行
python main.py
```

> 详细配置项说明与演示问题列表请见 [ClangChain-demo/README.md](ClangChain-demo/README.md)。

## 技术栈

| 组件 | 技术 |
|---|---|
| LLM 推理 | GLM-5.3（SiliconFlow，兼容 OpenAI 协议） |
| 向量嵌入 | Qwen3-VL-Embedding-8B |
| 重排序 | Qwen3-VL-Reranker-8B |
| 应用框架 | LangChain（LCEL / RunnableWithMessageHistory） |
| 关键词检索 | BM25 + jieba 中文分词 |

## License

MIT
