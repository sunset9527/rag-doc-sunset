"""
对话界面UI：消息展示、流式输出、引用来源
"""

import streamlit as st
import os

from src.chain import build_chat_history


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


def render_chat():
    """渲染主对话区域"""
    # 未上传文档提示
    if st.session_state.chain is None:
        st.info("👈 请先在侧边栏上传文档，然后开始提问")
        st.stop()

    # ---- 显示聊天历史 ----
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    _render_sources(message["sources"], prefix=f"hist_{idx}")

    # ---- 用户输入 ----
    user_input = st.chat_input("请输入你的问题...")

    if user_input:
        _handle_user_input(user_input)


def _handle_user_input(user_input):
    """处理用户输入，生成回答"""
    # 添加到历史
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            # 构建对话历史
            chat_history = build_chat_history(st.session_state.chat_history[:-1])

            # 获取检索结果
            retriever_dict = st.session_state.retriever_dict
            relevant_docs = retriever_dict["retrieve"](user_input)
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
                _render_sources(sources, prefix=f"ans_{len(st.session_state.chat_history)}")

            # 保存到历史
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


def _render_sources(sources, prefix=""):
    """渲染引用来源，key用prefix保证唯一"""
    with st.expander("📚 查看引用来源"):
        for source in sources:
            st.markdown(f"**来源 #{source['index']}: {source['source']}**")
            st.markdown(f"> {source['content_preview']}")
            with st.popover("查看全文"):
                st.text_area(
                    "完整内容",
                    value=source["full_content"],
                    height=200,
                    key=f"{prefix}_src_{source['index']}",
                    disabled=True
                )
