"""
                           Day 3 任务来了，今天把闭环串起来：
① RAG问答链
把昨天的 retriever 和 DeepSeek 接起来，用 create_retrieval_chain，能提问并返回答案。
② 对话记忆
加 ChatPromptTemplate + MessagesPlaceholder，支持追问，保留最近3-5轮上下文。
③ Streamlit界面
搭个简单前端：上传文档 + 提问 + 展示对话，一个 app.py 搞定。
④ 引用来源展示
答案下面标注来自哪个chunk，比如 [来源: doc_1.pdf, Chunk #3, 相似度: 0.87]。
"""
#导包
from langchain_community.document_loaders import TextLoader,PyPDFLoader,Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
import streamlit as st
import os
import tempfile
st.set_page_config(
    page_title="RAG问答系统",
    page_icon=":robot_face:",
    layout="wide",
)

os.environ["OPENAI_API_KEY"] = "sk-60b5c65e6b224fe9aebd91c55576d08f"

# RAG问答链
def test_03_retrieval_chain():
    """
    通俗解释：这是最基础的RAG问答功能
    工作流程：
        1. 加载文档（读取txt文件）
        2. 切分成小块（每块500字）
        3. 转成向量存到Chroma数据库
        4. 用户提问时，先检索相关片段
        5. 让AI根据检索到的内容回答问题

    """
    print("="*33)
    print("第一步：准备文档数据")
    print("="*33)
    file_path = "E:/python/LLM/RAG_project/test/retrieval_document"
    #加载文档
    loader = TextLoader(file_path,encoding="utf-8")
    document = loader.load()
    print(f"成功加载文件，文档数量为：{len( document)}")
    #切分,命名对象
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(document)
    print(f"成功切分，chunk数量为：{len(chunks)}")
    print(f"文本内容：{chunks[0].page_content}")
    #创建向量数据库对象
    chroma_path = "E:/python/LLM/RAG_project/test/chroma_db_deepseek"
    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key="971473819fe2474b89375be212ec47ef.P4vkxVPfUp2KVcue"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    print("存储了",len( chunks))
    #创建检索器
    retrieval = vectorstore.as_retriever(
        search_type ="similarity",
        search_kwargs={"k": 2}

    )
    #创建AI模型
    model = ChatOpenAI(
        model="deepseek-v4-pro",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7)

    prompt_template = """
    你是一个专业的高中教育问答助手。请根据下面的【参考资料】来回答用户的问题。

    【参考资料】：
    {context}
    
    【要求】：
    1. 只根据提供的资料回答问题，不要编造信息
    2. 如果资料中没有相关信息，就说"抱歉，我在资料中没找到相关信息"
    3. 回答要简洁明了，条理清晰
    4. 用中文回答
    
    【用户问题】：
    {input}
    
    【你的回答】：
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    print("完成")
    def format_docs(docs):
        formatted_docs = [f"{doc.metadata['source']} {doc.page_content}" for doc in docs]
        return "\n\n".join(formatted_docs)
    chain = (
        {
            "input": RunnablePassthrough(),
            "context": RunnablePassthrough().with_config(transformer=format_docs)
        }
        | prompt
        | model
        | StrOutputParser()
    )

    # 测试
    question = "高中时间安排？"
    print(f"\n❓ 问题：{question}")
    print("-" * 50)

    answer = chain.invoke(question)
    print(f"💡 回答：{answer}")
    print("-" * 50)

#② 对话记忆
def test_03_retrieval_memory():
    """
    对话记忆
    关键改进：
    1. 提示词模板增加 chat_history 占位符
    2. 手动维护一个列表保存对话历史
    3. 每次调用时传入历史记录
    """
    #1.加载文档
    file_path = "E:/python/LLM/RAG_project/test/r_m_document"
    loader = TextLoader(file_path,encoding="utf-8")
    document = loader.load()
    #2.切割文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(document)
    #3.创建向量数据库对象
    chroma_path = "E:/python/LLM/RAG_project/test/chroma_db_deepseek"
    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key="971473819fe2474b89375be212ec47ef.P4vkxVPfUp2KVcue"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    batch_size = 64
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        vectorstore.add_documents(documents=batch_chunks)
    retriever = vectorstore.as_retriever(
        search_type ="similarity",
        search_kwargs={"k": 3}
    )
    model = ChatOpenAI(
        model="deepseek-v4-pro",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7)
    print("="*33)
    system_prompt = """
    你是一个专业的问答助手。请根据【参考资料】和【聊天记录】来回答用户的问题。

    【参考资料】：
    {context}

    【聊天记录】：
    {chat_history}

    【要求】：
    1. 结合参考资料和聊天记录来回答问题
    2. 如果用户提到之前的内容，要根据聊天记录理解上下文
    3. 如果资料中没有相关信息，就说"抱歉，我在资料中没找到相关信息"
    4. 回答要简洁明了
    5. 用中文回答

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
    print("\n" + "=" * 50)
    print("开始多轮对话测试")
    print("=" * 50)

    chat_history = []

    # 第一轮对话
    question1 = "学业水平考试分为哪几类？"
    print(f"\n用户：{question1}")

    result1 = chain.invoke({
        "input": question1,
        "chat_history": chat_history
    })

    print(f"AI：{result1}")
    chat_history.append(HumanMessage(content=question1))
    chat_history.append(AIMessage(content=result1))

    # 第二轮对话
    question2 = "高中学习方法论有哪些？"
    print(f"\n用户：{question2}")

    result2 = chain.invoke({
        "input": question2,
        "chat_history": chat_history
    })

    print(f"AI：{result2}")
    chat_history.append(HumanMessage(content=question2))
    chat_history.append(AIMessage(content=result2))

    # 第三轮对话
    question3 = "数学的核心知识点是什么？"
    print(f"\n用户：{question3}")

    result3 = chain.invoke({
        "input": question3,
        "chat_history": chat_history
    })

    print(f"AI：{result3}")

    print("\n" + "=" * 50)
    print("多轮对话测试完成！")
    print("=" * 50)


#③ Streamlit界面
def test_03_retrieval_streamlit():
    # 1. 加载文档 (根据文件类型)
    def load_document(file_path):
        """根据文件类型加载文档"""
        ext = os.path.splitext(file_path)[1].lower()
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

    def create_chain(vectorstore):
        """创建RAG问答链"""
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
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
                st.session_state.chain, st.session_state.retriever = create_chain(vectorstore)

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
                                msg["content"] if msg["role"] == "user" else HumanMessage(content=msg["content"])
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


if __name__ == '__main__':
    # test_03_retrieval_chain()
    # test_03_retrieval_memory()
    test_03_retrieval_memory()