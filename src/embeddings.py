"""
Embedding配置模块
"""

from langchain_community.embeddings import ZhipuAIEmbeddings
from src.config import ZHIPU_API_KEY, ZHIPU_EMBEDDING_MODEL


def get_embeddings():
    """获取智谱Embedding实例"""
    return ZhipuAIEmbeddings(
        model=ZHIPU_EMBEDDING_MODEL,
        api_key=ZHIPU_API_KEY
    )
