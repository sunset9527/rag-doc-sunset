"""
  Streamlit界面 + 引用来源展示

  运行方式：
  在终端执行: streamlit run app.py

  功能特性：
  1. 文件上传（支持 .txt, .pdf, .docx）
  2. 实时问答界面
  3. 多轮对话支持
  4. 引用来源展示（带文件名和内容预览）
  5. 可配置的检索参数
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
# 设置页面配置‘
"""
    给网页起名字、选图标、设置布局
    浏览器标签会显示："RAG问答系统 🤖"
"""
st.set_page_config(
    page_title="RAG问答系统",
    page_icon="🤖",
    layout="wide"
)

# 初始化API密钥
os.environ["OPENAI_API_KEY"] = "sk-60b5c65e6b224fe9aebd91c55576d08f"


def load_document(file_path):
    """根据文件类型加载文档"""
    ext = os.path.splitext(file_path)[1].lower()  # 获取文件扩展名
    if ext == '.txt':
        return TextLoader(file_path, encoding='utf-8').load()
    elif ext == '.pdf':
        return PyPDFLoader(file_path).load()
    elif ext == '.docx':
        return Docx2txtLoader(file_path).load()
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def create_vectorstore(documents, chunk_size=200, chunk_overlap=30):
    """创建向量数据库"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key="971473819fe2474b89375be212ec47ef.P4vkxVPfUp2KVcue"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, chunks


def format_docs_with_source(docs):
    """格式化文档并保留来源信息"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '未知来源')
        content = doc.page_content
        formatted.append(f"[来源{i + 1}: {source}]\n{content}")
    return "\n\n".join(formatted), docs


def create_chain(vectorstore, top_k=3):
    """创建RAG问答链"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    model = ChatOpenAI(
        model="deepseek-v4-pro",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7
    )

    system_prompt = """
      你是一个专业的问答助手。请根据【参考资料】和【聊天记录】来回答用户的问题。

      【参考资料】：
      {context}

      【要求】：
      1. 只根据提供的资料回答问题，不要编造信息
      2. 如果资料中没有相关信息，就说"抱歉，我在资料中没找到相关信息"
      3. 回答要简洁明了，条理清晰
      4. 用中文回答
      5. 在回答末尾标注引用来源

      【用户问题】：
      {input}

      【你的回答】：
      """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
        # 提取文件名
        filename = os.path.basename(source) if isinstance(source, str) else source
        sources.append({
            "index": i + 1,
            "source": filename,
            "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
            "full_content": doc.page_content
        })
    return sources


# 标题
st.title("🤖 RAG智能问答系统")
st.markdown("---")

# 侧边栏 - 文件上传
with st.sidebar:
    st.header("📁 文档上传")
    uploaded_files = st.file_uploader(
        "上传文档（支持 .txt, .pdf, .docx）",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"已上传 {len(uploaded_files)} 个文件")

        # 高级设置
        with st.expander("⚙️ 高级设置"):
            chunk_size = st.slider("文本块大小", 100, 500, 200)
            chunk_overlap = st.slider("重叠大小", 0, 100, 30)
            top_k = st.slider("检索数量", 1, 10, 3)

    st.markdown("---")
    st.header("💡 使用提示")
    st.info("""
      1. 上传一个或多个文档
      2. 在下方输入问题
      3. 查看AI回答和引用来源
      4. 支持多轮对话，可以追问
      """)

# 主区域
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# 处理上传的文件
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("正在处理文档..."):
        try:
            all_documents = []
            temp_files = []

            for uploaded_file in uploaded_files:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)

                    # 加载文档
                    docs = load_document(tmp_file.name)
                    all_documents.extend(docs)

            # 创建向量数据库
            vectorstore, chunks = create_vectorstore(
                all_documents,
                chunk_size=chunk_size if 'chunk_size' in locals() else 200,
                chunk_overlap=chunk_overlap if 'chunk_overlap' in locals() else 30
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.chain, st.session_state.retriever = create_chain(
                vectorstore,
                top_k=top_k if 'top_k' in locals() else 3
            )

            st.success(f"✅ 文档处理完成！共切分为 {len(chunks)} 个文本块")

            # 清理临时文件
            for tmp_file in temp_files:
                os.unlink(tmp_file)

        except Exception as e:
            st.error(f"❌ 处理文档时出错: {str(e)}")

# 显示聊天历史
st.subheader("💬 对话区域")
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

                # 显示引用来源
                if "sources" in message:
                    with st.expander("📚 查看引用来源"):
                        for source in message["sources"]:
                            st.markdown(f"""
                              **来源 {source['index']}: {source['source']}**

                              内容预览: {source['content_preview']}
                              """)

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # 检查是否已加载文档
    if st.session_state.chain is None:
        st.warning("⚠️ 请先上传文档！")
    else:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(user_input)

        # 添加到历史记录
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    # 调用RAG链
                    result = st.session_state.chain.invoke({
                        "input": user_input,
                        "chat_history": [
                            HumanMessage(content=msg["content"]) if msg["role"] == "assistant" else msg["content"]
                            for msg in st.session_state.chat_history[:-1]
                        ]
                    })

                    # 获取相关文档用于显示来源
                    relevant_docs = st.session_state.retriever.invoke(user_input)
                    sources = get_source_info(relevant_docs)

                    st.write(result)

                    # 显示引用来源
                    if sources:
                        with st.expander("📚 查看引用来源"):
                            for source in sources:
                                st.markdown(f"""
                                  ---
                                  **来源 #{source['index']}: {source['source']}**

                                  📄 内容预览:
                                  > {source['content_preview']}
                                  """)

                    # 添加AI回复到历史记录
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"❌ 生成回答时出错: {str(e)}")

# 清空对话按钮
if st.session_state.chat_history:
    if st.button("🗑️ 清空对话"):
        st.session_state.chat_history = []
        st.rerun()