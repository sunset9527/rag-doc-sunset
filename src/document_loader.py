"""
文档加载模块：根据文件类型加载文档
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from src.config import SUPPORTED_EXTENSIONS


def load_document(file_path):
    """根据文件类型加载文档"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {ext}")

    if ext in ('.txt', '.md'):
        return TextLoader(file_path, encoding='utf-8').load()
    elif ext == '.pdf':
        return PyPDFLoader(file_path).load()
    elif ext == '.docx':
        return Docx2txtLoader(file_path).load()
