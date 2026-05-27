"""
向量存储模块：创建和管理Chroma向量数据库，分批写入
"""

from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings
from src.config import BATCH_SIZE


def create_vectorstore(chunks, persist_directory=None):
    """
    创建向量数据库，分批写入避免智谱API报错

    Args:
        chunks: 切分后的Document列表
        persist_directory: 持久化目录，None则内存存储

    Returns:
        vectorstore: Chroma实例
    """
    embeddings = get_embeddings()
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vectorstore.add_documents(batch)

    return vectorstore
