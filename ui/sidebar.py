"""
侧边栏UI：文档上传、参数调节、操作按钮、文档处理逻辑
"""

import streamlit as st
import os
import tempfile

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vectorstore import create_vectorstore
from src.retrieval import create_hybrid_retriever
from src.chain import create_rag_chain
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_BM25_WEIGHT


def render_sidebar():
    """渲染侧边栏，返回参数字典"""
    with st.sidebar:
        # ---- 文档上传 ----
        st.header("📁 文档上传")
        uploaded_files = st.file_uploader(
            "上传文档（支持 .txt, .md, .pdf, .docx）",
            type=['txt', 'md', 'pdf', 'docx'],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"已上传 {len(uploaded_files)} 个文件")

        # ---- 高级设置 ----
        with st.expander("⚙️ 高级设置"):
            chunk_size = st.slider(
                "文本块大小 (chunk_size)", 100, 1000, DEFAULT_CHUNK_SIZE, step=50,
                help="值越大，每个文本块包含的信息越多；值越小，检索越精确"
            )
            chunk_overlap = st.slider(
                "重叠大小 (chunk_overlap)", 0, 200, DEFAULT_CHUNK_OVERLAP, step=10,
                help="相邻文本块之间的重叠字符数，防止信息被截断"
            )
            top_k = st.slider(
                "检索数量 (top_k)", 1, 10, DEFAULT_TOP_K,
                help="每次检索返回的文本块数量"
            )
            bm25_weight = st.slider(
                "BM25权重", 0.0, 1.0, DEFAULT_BM25_WEIGHT, step=0.1,
                help="BM25关键词检索权重，向量检索权重=1-此值"
            )

        # ---- 操作按钮 ----
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重新处理", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.chain = None
                st.session_state.retriever_dict = None
                st.session_state.chunks = None
                st.session_state.chunks_count = 0
                st.session_state.file_names = []
                st.rerun()
        with col2:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # ---- 文档信息 ----
        if st.session_state.file_names:
            st.markdown("---")
            st.header("📄 已加载文档")
            for fname in st.session_state.file_names:
                st.markdown(f"- {fname}")
            st.caption(
                f"共 {st.session_state.chunks_count} 个文本块 | "
                f"top_k={st.session_state.last_top_k} | "
                f"BM25权重={bm25_weight}"
            )

        # ---- 对话历史摘要 ----
        if st.session_state.chat_history:
            st.markdown("---")
            st.header("📜 对话历史")
            with st.container(height=200):
                for msg in st.session_state.chat_history:
                    icon = "🧑" if msg["role"] == "user" else "🤖"
                    preview = msg["content"][:40] + "..." if len(msg["content"]) > 40 else msg["content"]
                    st.markdown(f"{icon} {preview}")

        # ---- 使用提示 ----
        st.markdown("---")
        with st.expander("💡 使用提示"):
            st.info("""
1. 上传一个或多个文档
2. 点击"重新处理"应用新参数
3. 在下方输入问题
4. 查看AI回答和引用来源
5. 支持多轮对话，可以追问
6. 混合检索：BM25关键词+向量语义双路召回
            """)

    return {
        "uploaded_files": uploaded_files,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "bm25_weight": bm25_weight,
    }


def process_documents(uploaded_files, chunk_size, chunk_overlap, top_k, bm25_weight):
    """处理上传的文档：加载→切分→向量化→创建检索器和问答链"""
    with st.spinner("正在处理文档..."):
        try:
            all_documents = []
            temp_files = []

            for i, uploaded_file in enumerate(uploaded_files):
                st.info(f"📄 正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)
                    docs = load_document(tmp_file.name)
                    all_documents.extend(docs)

            if not all_documents:
                st.error("❌ 未成功加载任何文档，请检查文件格式")
                return False

            # 切分
            chunks = split_documents(all_documents, chunk_size, chunk_overlap)

            # 向量化
            vectorstore = create_vectorstore(chunks, persist_directory="data/chroma_db")

            # 创建检索器
            retriever_dict = create_hybrid_retriever(vectorstore, chunks, top_k, bm25_weight)

            # 创建问答链
            chain = create_rag_chain(retriever_dict["retrieve"])

            # 保存到session
            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.retriever_dict = retriever_dict
            st.session_state.chain = chain
            st.session_state.chunks_count = len(chunks)
            st.session_state.file_names = [f.name for f in uploaded_files]
            st.session_state.last_top_k = top_k
            st.session_state.last_bm25_weight = bm25_weight

            st.success(f"✅ 文档处理完成！共 {len(all_documents)} 页/段，切分为 {len(chunks)} 个文本块")

            # 清理临时文件
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass

            return True

        except Exception as e:
            st.error(f"❌ 处理文档时出错: {str(e)}")
            st.caption("请检查文件格式是否正确，或尝试调整参数后重新处理")
            return False


def update_chain_if_needed(top_k, bm25_weight):
    """参数变化时重新创建链"""
    if st.session_state.vectorstore is None or st.session_state.chunks is None:
        return

    if (top_k != st.session_state.last_top_k or
        bm25_weight != st.session_state.get("last_bm25_weight", DEFAULT_BM25_WEIGHT)):

        retriever_dict = create_hybrid_retriever(
            st.session_state.vectorstore, st.session_state.chunks, top_k, bm25_weight
        )
        st.session_state.chain = create_rag_chain(retriever_dict["retrieve"])
        st.session_state.retriever_dict = retriever_dict
        st.session_state.last_top_k = top_k
        st.session_state.last_bm25_weight = bm25_weight
