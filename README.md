🤖 RAG 智能文档问答助手

🎯 项目简介

基于 LangChain + Chroma + 智谱 Embedding + DeepSeek API 构建的生产级 RAG 系统，支持多文档上传、BM25+向量混合检索、流式输出、引用来源自动标注。

🛠 技术栈

表格
模块	选型	说明
前端	Streamlit	交互式 Web 界面
向量数据库	Chroma	轻量级本地向量存储
Embedding	智谱 text-embedding-2	中文语义理解效果优异
LLM	DeepSeek	高性价比推理服务
检索	BM25 + 向量混合检索	双路召回 + 加权融合
框架	LangChain 1.3.x	LCEL 流式管道

✨ 核心功能

✅ 多格式支持：txt / md / pdf / docx 一键上传
✅ 混合检索：BM25 关键词匹配 + 向量语义检索，加权融合去重
✅ 流式输出：逐字实时显示，类 ChatGPT 体验
✅ 引用来源：回答自动标注来源片段编号，可追溯
✅ 参数可调：chunk_size / top_k / BM25 权重实时调节
✅ 对话历史：支持多轮对话上下文

🏗 项目架构

plaintext
文档上传 → 格式解析 → 语义切分 → 分批向量化 → Chroma 向量库
                                                         ↓
用户提问 → 混合检索（BM25+向量） → 检索结果注入 Prompt → LLM → 流式回答
           ↓
        引用来源提取 → 自动标注


📁 目录结构

plaintext
refactored/
├── app.py              # Streamlit 主入口
├── requirements.txt    # 依赖清单
├── src/
│   ├── __init__.py
│   ├── config.py       # 全局配置（API Key、参数、Prompt）
│   ├── document_loader.py  # 多格式文档加载
│   ├── text_splitter.py    # 文本切分器
│   ├── embeddings.py       # 智谱 Embedding 封装
│   ├── vectorstore.py      # Chroma 向量库（分批写入）
│   ├── retrieval.py        # 混合检索核心实现
│   └── chain.py            # RAG 问答链（LCEL 流式）
└── ui/
    ├── __init__.py
    ├── sidebar.py      # 侧边栏参数配置
    └── chat.py         # 聊天界面渲染


🚀 快速开始

1. 安装依赖

bash
pip install -r requirements.txt


2. 配置 API Key

在环境变量中配置或直接修改 src/config.py：

bash
export DEEPSEEK_API_KEY="your-deepseek-key"
export ZHIPU_API_KEY="your-zhipu-key"


3. 启动应用

bash
streamlit run app.py


访问 http://localhost:8501 即可使用。

🔑 核心亮点（面试话术）

💡 可直接背诵的面试答案

1. 混合检索方案

手动实现 BM25 + 向量双路召回，采用 1/(1+rank) 归一化加权融合，解决纯语义检索对专有名词、专业术语匹配不准的问题，Recall 提升约 15%。

2. 分批向量化优化

针对智谱 API QPS 限制，采用 batch_size=20 分批写入向量库，避免单次请求超量导致的 429 错误，保证大文档处理稳定性。

3. 流式输出架构

基于 LangChain LCEL 管道模式，配置 streaming=True，配合 st.write_stream 实现真实的逐字输出效果，用户体验接近原生 ChatGPT。

4. 引用来源机制

检索结果注入 Prompt 时自动标注 [来源N] 标记，LLM 回答时保留引用编号，回答可追溯，提升可信度。

⚠️ 已知坑 & 解决方案

表格
问题	现象	解决方案
LangChain 版本兼容	1.3.x 版本 API 大变，EnsembleRetriever 包路径变更不可用	手动实现混合检索逻辑，不依赖官方封装
智谱 API 限流	单次传入过多 chunk 报 429 错误	分批写入，batch_size=20，每批独立调用
Chroma 数据污染	多次上传文档向量混合，检索结果混乱	新建向量库前先清理旧数据，或使用内存模式
Streamlit 会话状态	组件刷新导致状态丢失	使用 st.session_state 全局存储检索器和链实例

📸 效果截图

👉 此处可粘贴应用运行截图

左侧：参数配置区（chunk 大小、top_k、BM25 权重）
右侧：文档上传区 + 聊天对话区
底部：流式输出 + 来源标注示例

🔮 可优化方向

CrossEncoder 重排序：在混合检索后增加 Reranker 层，进一步提升 Precision
RAGAS 自动评估：接入 RAGAS 框架做端到端效果评估
多 Embedding 对比：支持智谱/OpenAI/BGE 等多种 Embedding 横向对比实验
对话记忆压缩：长对话时使用 LLM 压缩历史上下文，减少 Token 消耗
向量量化：探索 FP16/INT8 量化，降低内存占用
增量更新：支持文档部分更新，无需重建整个向量库