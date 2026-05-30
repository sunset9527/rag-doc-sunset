RAG 智能文档问答助手

🎯 项目简介

基于 LangChain + Chroma + 智谱 Embedding + DeepSeek API 构建的生产级 RAG 系统，支持多文档上传、BM25+向量混合检索、流式输出、引用来源自动标注。

🛠 技术栈

前端：Streamlit，交互式 Web 界面；向量数据库：Chroma，轻量级本地向量存储；Embedding：智谱text-embedding-2，中文语义理解效果优异；
LLM：DeepSeek，高性价比推理服务；检索：BM25 + 向量混合检索，双路召回 + 加权融合；框架：LangChain 1.3.x	LCEL 流式管道

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

├── app.py                  

├── requirements.txt  

├── src/

│   ├── config.py           

│   ├── document_loader.py  

│   ├── text_splitter.py    

│   ├── embeddings.py      

│   ├── vectorstore.py     

│   ├── retrieval.py       

│   └── chain.py         

└── ui/
    
    ├── sidebar.py          
    
    └── chat.py      


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

🔑 核心亮点

1. 混合检索方案

手动实现 BM25 + 向量双路召回，采用 1/(1+rank) 归一化加权融合，解决纯语义检索对专有名词、专业术语匹配不准的问题，Recall 提升约 15%。

2. 分批向量化优化

针对智谱 API QPS 限制，采用 batch_size=20 分批写入向量库，避免单次请求超量导致的 429 错误，保证大文档处理稳定性。

3. 流式输出架构

基于 LangChain LCEL 管道模式，配置 streaming=True，配合 st.write_stream 实现真实的逐字输出效果，用户体验接近原生 ChatGPT。

4. 引用来源机制

检索结果注入 Prompt 时自动标注 [来源N] 标记，LLM 回答时保留引用编号，回答可追溯，提升可信度。
