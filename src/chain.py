"""
RAG问答链模块：Prompt → LLM → 输出解析，支持流式输出
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, SYSTEM_PROMPT
)


def create_rag_chain(retriever_func):
    """
    创建RAG问答链（流式输出）

    Args:
        retriever_func: 检索函数，传入query返回Document列表

    Returns:
        chain: LCEL链
    """
    model = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_base=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        temperature=0.7,
        request_timeout=60,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[来源{i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )

    chain = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "context": lambda x: format_docs(retriever_func(x["input"]))
            }
            | prompt
            | model
            | StrOutputParser()
    )

    return chain


def build_chat_history(chat_history_list):
    """将聊天记录转为LangChain消息格式"""
    return [
        HumanMessage(content=msg["content"]) if msg["role"] == "user"
        else AIMessage(content=msg["content"])
        for msg in chat_history_list
    ]
