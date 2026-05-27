"""
🤖 RAG智能文档问答系统 - 主入口

运行方式：
  streamlit run app.py

项目结构：
  app.py          - 主入口
  src/            - 核心业务逻辑
    config.py         - API Key + 默认参数 + Prompt
    document_loader.py - 文档加载
    text_splitter.py   - 文本切分
    embeddings.py      - Embedding配置
    vectorstore.py     - 向量存储（分批写入）
    retrieval.py       - 混合检索（BM25+向量加权融合）
    chain.py           - RAG问答链 + 流式输出
  ui/             - Streamlit界面
    sidebar.py         - 侧边栏
    chat.py            - 对话界面
"""

import streamlit as st
from ui.sidebar import render_sidebar, process_documents, update_chain_if_needed
from ui.chat import render_chat
from src.config import DEFAULT_TOP_K, DEFAULT_BM25_WEIGHT

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide"
)

# ==================== Session State 初始化 ====================
for key, default in [
    ("chat_history", []),
    ("vectorstore", None),
    ("chain", None),
    ("retriever_dict", None),
    ("chunks", None),
    ("chunks_count", 0),
    ("file_names", []),
    ("last_top_k", DEFAULT_TOP_K),
    ("last_bm25_weight", DEFAULT_BM25_WEIGHT),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== 侧边栏 ====================
params = render_sidebar()

# ==================== 处理上传的文件 ====================
if params["uploaded_files"] and st.session_state.vectorstore is None:
    process_documents(
        params["uploaded_files"],
        params["chunk_size"],
        params["chunk_overlap"],
        params["top_k"],
        params["bm25_weight"]
    )

# ==================== 参数实时更新 ====================
update_chain_if_needed(params["top_k"], params["bm25_weight"])

# ==================== 主对话区域 ====================
st.title("🤖 RAG智能问答系统")
st.markdown("---")
render_chat()
