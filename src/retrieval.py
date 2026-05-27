"""
检索模块：BM25 + 向量混合检索，手动加权融合
"""

from langchain_community.retrievers import BM25Retriever
from src.config import DEFAULT_TOP_K, DEFAULT_BM25_WEIGHT


def hybrid_retrieve(query, vector_retriever, bm25_retriever, top_k=DEFAULT_TOP_K, bm25_weight=DEFAULT_BM25_WEIGHT):
    """
    混合检索：BM25关键词 + 向量语义，加权融合去重

    Args:
        query: 查询文本
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器
        top_k: 最终返回文档数
        bm25_weight: BM25权重（0-1），向量权重=1-bm25_weight
    """
    vector_weight = 1.0 - bm25_weight

    # 双路召回
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    # 按内容去重，记录两路排名
    seen = {}
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        if key not in seen:
            seen[key] = {"doc": doc, "bm25_rank": rank, "vector_rank": 999}
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        if key in seen:
            seen[key]["vector_rank"] = rank
        else:
            seen[key] = {"doc": doc, "bm25_rank": 999, "vector_rank": rank}

    # 按权重计算分数（排名越前分数越高）
    scored = []
    for item in seen.values():
        bm25_score = 1.0 / (1 + item["bm25_rank"])
        vector_score = 1.0 / (1 + item["vector_rank"])
        final_score = bm25_weight * bm25_score + vector_weight * vector_score
        scored.append((final_score, item["doc"]))

    # 按分数降序，取top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def create_hybrid_retriever(vectorstore, chunks, top_k=DEFAULT_TOP_K, bm25_weight=DEFAULT_BM25_WEIGHT):
    """
    创建混合检索器

    Returns:
        dict: {
            "retrieve": 检索函数,
            "vector_retriever": 向量检索器,
            "bm25_retriever": BM25检索器
        }
    """
    # 向量检索器（语义匹配）
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 3}
    )

    # BM25检索器（关键词匹配）
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k * 3

    def retrieve(query):
        return hybrid_retrieve(query, vector_retriever, bm25_retriever, top_k, bm25_weight)

    return {
        "retrieve": retrieve,
        "vector_retriever": vector_retriever,
        "bm25_retriever": bm25_retriever,
    }
