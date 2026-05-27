"""
全局配置：API Key、默认参数、Prompt模板
"""

import os

# ==================== API配置 ====================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
ZHIPU_EMBEDDING_MODEL = "embedding-2"

# ==================== 默认参数 ====================
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_BM25_WEIGHT = 0.3
BATCH_SIZE = 20

# ==================== 支持的文件格式 ====================
SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx']

# ==================== Prompt模板 ====================
SYSTEM_PROMPT = """你是一个专业的问答助手。请根据【参考资料】和【聊天记录】来回答用户的问题。

【参考资料】：
{context}

【要求】：
1. 只根据提供的资料回答问题，不要编造信息
2. 如果资料中没有相关信息，就说"抱歉，我在提供的资料中没找到相关信息"
3. 回答要简洁明了，条理清晰
4. 用中文回答
5. 在回答末尾标注引用来源编号，如[来源1][来源2]"""
