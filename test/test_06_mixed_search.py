"""
  RAG智能问答系统 - Day6 混合检索+流式输出版

  运行方式：
  在终端执行: streamlit run app_day6.py

  需要安装：
  pip install rank-bm25

  功能特性：
  1. 混合检索（BM25关键词 + 向量语义，双路召回）
  2. 流式输出（打字机效果）
  3. 文件上传（支持 .txt, .md, .pdf, .docx）
  4. 多轮对话 + 引用来源展示
  5. 可配置检索参数（BM25/向量权重、top_k）
  6. 异常处理与边界case兜底
"""
from langchain_community.retrievers import BM25Retriever
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
import tempfile

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide"
)

# ==================== API配置 ====================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-60b5c65e6b224fe9aebd91c55576d08f")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "971473819fe2474b89375be212ec47ef.P4vkxVPfUp2KVcue")

os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY


# ==================== 工具函数 ====================
def load_document(file_path):
    """根据文件类型加载文档"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.txt', '.md'):
        return TextLoader(file_path, encoding='utf-8').load()
    elif ext == '.pdf':
        return PyPDFLoader(file_path).load()
    elif ext == '.docx':
        return Docx2txtLoader(file_path).load()
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def create_vectorstore(documents, chunk_size=500, chunk_overlap=50):
    """创建向量数据库"""
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

    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key=ZHIPU_API_KEY
    )

    # 分批写入，避免智谱API报错
    batch_size = 20
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings
            )
        else:
            vectorstore.add_documents(batch)

    return vectorstore, chunks


def create_chain(vectorstore, top_k=3, bm25_weight=0.3, chunks=None):
    """创建RAG问答链（混合检索 + 流式输出）"""
    # 1. 向量检索器（语义匹配）
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 3}
    )

    # 2. BM25检索器（关键词匹配）
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k * 3

    # 3. 手动混合检索：合并两路结果，按权重去重
    vector_weight = 1.0 - bm25_weight

    def hybrid_retrieve(query):
        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vector_retriever.invoke(query)

        # 用字典去重，key为内容
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

    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,
        request_timeout=60,
        streaming=True
    )

    system_prompt = """你是一个专业的问答助手。请根据【参考资料】和【聊天记录】来回答用户的问题。

【参考资料】：
{context}

【要求】：
1. 只根据提供的资料回答问题，不要编造信息
2. 如果资料中没有相关信息，就说"抱歉，我在提供的资料中没找到相关信息"
3. 回答要简洁明了，条理清晰
4. 用中文回答
5. 在回答末尾标注引用来源编号，如[来源1][来源2]"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[来源{i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )

    chain = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: format_docs(hybrid_retrieve(x["input"]))
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain, type('Retriever', (), {'invoke': lambda self, q: hybrid_retrieve(q)})()


def get_source_info(docs):
    """获取引用来源信息"""
    sources = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '未知来源')
        filename = os.path.basename(source) if isinstance(source, str) else source
        content = doc.page_content
        sources.append({
            "index": i + 1,
            "source": filename,
            "content_preview": content[:150] + "..." if len(content) > 150 else content,
            "full_content": content
        })
    return sources


def get_source_info(docs):
    """获取引用来源信息"""
    sources = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '未知来源')
        filename = os.path.basename(source) if isinstance(source, str) else source
        content = doc.page_content
        sources.append({
            "index": i + 1,
            "source": filename,
            "content_preview": content[:150] + "..." if len(content) > 150 else content,
            "full_content": content
        })
    return sources


# ==================== Session State 初始化 ====================
for key, default in [
    ("chat_history", []),
    ("vectorstore", None),
    ("chain", None),
    ("retriever", None),
    ("chunks", None),
    ("chunks_count", 0),
    ("file_names", []),
    ("last_top_k", 3),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📁 文档上传")
    uploaded_files = st.file_uploader(
        "上传文档（支持 .txt, .md, .pdf, .docx）",
        type=['txt', 'md', 'pdf', 'docx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"已上传 {len(uploaded_files)} 个文件")

    # 高级设置
    with st.expander("⚙️ 高级设置"):
        chunk_size = st.slider("文本块大小 (chunk_size)", 100, 1000, 500, step=50,
                               help="值越大，每个文本块包含的信息越多；值越小，检索越精确")
        chunk_overlap = st.slider("重叠大小 (chunk_overlap)", 0, 200, 50, step=10,
                                  help="相邻文本块之间的重叠字符数，防止信息被截断")
        top_k = st.slider("检索数量 (top_k)", 1, 10, 3,
                          help="每次检索返回的文本块数量")
        bm25_weight = st.slider("BM25权重", 0.0, 1.0, 0.3, step=0.1,
                                 help="BM25关键词检索权重，向量检索权重=1-此值")

    # 操作按钮
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重新处理", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.chain = None
            st.session_state.retriever = None
            st.session_state.chunks = None
            st.session_state.chunks_count = 0
            st.session_state.file_names = []
            st.rerun()
    with col2:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # 文档信息
    if st.session_state.file_names:
        st.markdown("---")
        st.header("📄 已加载文档")
        for fname in st.session_state.file_names:
            st.markdown(f"- {fname}")
        st.caption(f"共 {st.session_state.chunks_count} 个文本块 | top_k={st.session_state.last_top_k} | BM25权重={bm25_weight}")

    # 对话历史摘要
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("📜 对话历史")
        with st.container(height=200):
            for msg in st.session_state.chat_history:
                icon = "🧑" if msg["role"] == "user" else "🤖"
                preview = msg["content"][:40] + "..." if len(msg["content"]) > 40 else msg["content"]
                st.markdown(f"{icon} {preview}")

    # 使用提示
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

# ==================== 处理上传的文件 ====================
if uploaded_files and st.session_state.vectorstore is None:
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
            else:
                vectorstore, chunks = create_vectorstore(
                    all_documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.chunks = chunks
                st.session_state.chain, st.session_state.retriever = create_chain(
                    vectorstore, top_k=top_k, bm25_weight=bm25_weight, chunks=chunks
                )
                st.session_state.chunks_count = len(chunks)
                st.session_state.file_names = [f.name for f in uploaded_files]
                st.session_state.last_top_k = top_k

                st.success(f"✅ 文档处理完成！共 {len(all_documents)} 页/段，切分为 {len(chunks)} 个文本块")

            # 清理临时文件
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass

        except Exception as e:
            st.error(f"❌ 处理文档时出错: {str(e)}")
            st.caption("请检查文件格式是否正确，或尝试调整参数后重新处理")

# ==================== 参数实时更新 ====================
if st.session_state.vectorstore is not None and st.session_state.chunks is not None:
    need_update = False
    if top_k != st.session_state.last_top_k:
        need_update = True
    st.session_state.chain, st.session_state.retriever = create_chain(
        st.session_state.vectorstore, top_k=top_k, bm25_weight=bm25_weight, chunks=st.session_state.chunks
    )
    st.session_state.last_top_k = top_k

# ==================== 主对话区域 ====================
st.title("🤖 RAG智能问答系统")
st.markdown("---")

# 未上传文档提示
if st.session_state.chain is None:
    st.info("👈 请先在侧边栏上传文档，然后开始提问")
    st.stop()

# 显示聊天历史
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 查看引用来源"):
                    for source in message["sources"]:
                        st.markdown(f"**来源 #{source['index']}: {source['source']}**")
                        st.markdown(f"> {source['content_preview']}")
                        with st.popover("查看全文"):
                            st.text_area(
                                "完整内容",
                                value=source["full_content"],
                                height=200,
                                key=f"full_{source['index']}_{hash(source['full_content'][:50])}",
                                disabled=True
                            )

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # 添加到历史记录
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 显示用户消息
    with st.chat_message("user"):
        st.write(user_input)

    # 生成回答（流式输出）
    with st.chat_message("assistant"):
        try:
            # 构建chat_history
            chat_history = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user"
                else AIMessage(content=msg["content"])
                for msg in st.session_state.chat_history[:-1]
            ]

            # 获取相关文档
            relevant_docs = st.session_state.retriever.invoke(user_input)
            sources = get_source_info(relevant_docs)

            # 流式输出
            result = st.write_stream(
                st.session_state.chain.stream({
                    "input": user_input,
                    "chat_history": chat_history
                })
            )

            # 显示引用来源
            if sources:
                with st.expander("📚 查看引用来源"):
                    for source in sources:
                        st.markdown(f"**来源 #{source['index']}: {source['source']}**")
                        st.markdown(f"> {source['content_preview']}")
                        with st.popover("查看全文"):
                            st.text_area(
                                "完整内容",
                                value=source["full_content"],
                                height=200,
                                key=f"ans_full_{source['index']}_{hash(source['full_content'][:50])}",
                                disabled=True
                            )

            # 添加AI回复到历史记录
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result,
                "sources": sources
            })

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                st.error("❌ 请求超时，请稍后重试")
            elif "rate limit" in error_msg.lower():
                st.error("❌ API调用频率超限，请稍等片刻再试")
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                st.error("❌ API密钥无效，请检查配置")
            else:
                st.error(f"❌ 生成回答时出错: {error_msg}")
            st.session_state.chat_history.pop()
