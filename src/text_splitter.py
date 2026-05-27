"""
文本切分模块：将文档切分为文本块
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def split_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """将文档切分为文本块，过滤空内容"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # 过滤空文本块
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        raise ValueError("文档切分后没有有效内容，请检查文档是否为空")

    return chunks
