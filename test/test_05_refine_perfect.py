"""
  RAG智能问答系统 - Day4 打磨优化版

  运行方式：
  在终端执行: streamlit run app_day4.py

  功能特性：
  1. 文件上传（支持 .txt, .md, .pdf, .docx）
  2. 实时问答界面 + 多轮对话
  3. 引用来源展示（预览+全文）
  4. 可配置的检索参数（实时生效）
  5. 异常处理与边界case兜底
  6. 重新处理文档 / 清空对话
  7. 对话历史侧边栏
"""

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
# 建议后续改用 .env 文件管理，这里先用环境变量 + 默认值
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
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        raise ValueError("文档切分后没有有效内容，请检查文档是否为空")

    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key=ZHIPU_API_KEY
    )

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


def create_chain(vectorstore, top_k=3):
    """创建RAG问答链"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,
        request_timeout=60,
        streaming=True,
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
            "context": lambda x: format_docs(retriever.invoke(x["input"]))
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain, retriever


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

    # 操作按钮
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重新处理", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.chain = None
            st.session_state.retriever = None
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
        st.caption(f"共 {st.session_state.chunks_count} 个文本块 | top_k={st.session_state.last_top_k}")

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
                st.session_state.chain, st.session_state.retriever = create_chain(
                    vectorstore, top_k=top_k
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

# ==================== top_k 实时更新 ====================
if st.session_state.vectorstore is not None and top_k != st.session_state.last_top_k:
    st.session_state.chain, st.session_state.retriever = create_chain(
        st.session_state.vectorstore, top_k=top_k
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
                        # 查看全文按钮
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

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 构建chat_history - 修复消息类型
                chat_history = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                    for msg in st.session_state.chat_history[:-1]
                ]

                # 调用RAG链
                result = st.write_stream(st.session_state.chain.stream({
                    "input": user_input,
                    "chat_history": chat_history
                }))

                # 获取相关文档用于显示来源
                relevant_docs = st.session_state.retriever.invoke(user_input)
                sources = get_source_info(relevant_docs)

                st.write(result)

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
                # 移除刚才添加的用户消息（因为没得到回复）
                st.session_state.chat_history.pop()
