"""
                                 Day 2 任务来了，今天是RAG核心环节
① 文档加载器
    用 PyPDFLoader 或 TextLoader 加载一个测试文档，能打印出文本内容就行。
② 文本切分
    用 RecursiveCharacterTextSplitter 切 chunk，chunk_size=500, overlap=50，打印 chunk 数量确认切分正确。
③ 向量存储到Chroma
    用 Chroma.from_documents() 把 chunk 向量化存入本地。Embedding 如果DeepSeek没配好，先用 BAAI/bge-small-zh-v1.5 替代。
④ 向量检索测试
    写个 query 测试 similarity_search(query, k=2)，能返回相关chunk就算跑通。

"""
#导包
from langchain_community.document_loaders import TextLoader          #导入文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入文本切分工具
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer



# 文档加载器
def test_02_loader_txt():
    """
    通俗解释：文档加载器就像一个"阅读器"，能帮你读取各种格式的文件（PDF、TXT、Word等），把文件里的文字提取出来，让程序能处理。
    类比：就像你用Word打开一个文档，能看到里面的文字。文档加载器就是程序的"Word"。
"""
    file_path =  "E:/python/LLM/RAG_project/test/test_document"     #指定要加载的文件路径
    #创建对象
    loader = TextLoader(file_path,encoding="utf-8")                 #创建一个加载器对象,utf-8防止乱码
    document = loader.load()                                        #加载文档并提取内容 .load()方法读取文件，返回一个文档列表
    print(f"成功加载文件，文档数量为：{len( document)}")
    for i,doc in enumerate(document):
        print(f"文档{i+1}：{doc.page_content}")                      #doc.page_content 是文档的实际文字内容
        print(f"元数据:{doc.metadata}")                              # doc.metadata 是文档的元数据（如文件路径、名称等）


# 文本切分
def test_02_loader_txt_split():
    """
    通俗解释：
        想象你有一篇很长的文章（比如1万字），如果一次性全部丢给AI处理：
            AI可能记不住所有内容（有长度限制）
            检索时不够精准（找不到具体位置）
        文本切分就是把长文章切成很多小段落（chunk），每个段落几百字，这样：
            AI处理起来更轻松
            检索时能找到最相关的段落
            节省成本和内存
    类比：
        就像把一本厚书拆成很多页，找内容时不用翻整本书，直接定位到某一页。
"""
    file_path = "E:/python/LLM/RAG_project/test/test_document"
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()

    #创建对象
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=20,                                            # chunk_size=20: 每个小块最多20个字符
        chunk_overlap=5,                                          # chunk_overlap=5: 相邻块之间重叠5个字符
        separators=["\n\n", "\n", " ", ""]                        # 切分的分隔符优先级
    )
    #切分
    chunks = spliter.split_documents(document)                         # .split_documents()方法接收文档列表，返回切分后的文档列表
    print(f"成功切分，chunk数量为：{len(chunks)}")
    for i,chunk in enumerate(chunks):
        # enumerate 会把列表变成这样：
        # (0, "苹果"), (1, "香蕉"), (2, "橙子")
        #  ↑   ↑        ↑   ↑        ↑   ↑
        # 序号 内容     序号 内容     序号 内容
        #enumerate 是一个Python内置函数，用来给列表里的每个元素自动编号。
        print(f"chunk{i+1}：{chunk.page_content}")




# 向量存储到Chroma
def test_02_chroma():

    """
    1. 什么是向量化（Embedding）？---->  将文字以机器语言的形式存储，后续用的话可用于knn？
        通俗解释：把文字转换成数字列表（向量），让计算机能"理解"文字的含义。
        类比：
            人类理解："苹果"是一种水果，红色，甜的
            计算机不理解文字，但能理解数字：[0.8, -0.3, 0.5, ...]
            意思相近的词，向量也相近：
                "苹果" → [0.8, -0.3, 0.5]
                "香蕉" → [0.7, -0.2, 0.6] （相近，都是水果）
                "汽车" → [-0.5, 0.9, -0.1] （不相近，是交通工具）
    2. 什么是Chroma？  ---->向量数据库
        通俗解释：Chroma是一个专门存向量的小数据库，就像Excel表格存数据一样，它用来存"文字对应的向量"。
        作用：
            存储：把切分后的文本块转成向量存起来
            检索：当你问问题时，快速找到最相关的文本块
            类比：就像图书馆的索引系统，能快速找到你想要的书。
"""
    file_path = "E:/python/LLM/RAG_project/test/test_document"
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = spliter.split_documents(document)

    # 流程：文本块 → Embedding模型 → 向量数字 → Chroma数据库存储
    # 创建模型对象
    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",                       # 中文向量模型
         api_key="971473819fe2474b89375be212ec47ef.P4vkxVPfUp2KVcue"
    )
    # 创建向量数据库对象
    chroma_path = "E:/python/LLM/RAG_project/test/chroma_db_deepseek"

    vectorstore = Chroma.from_documents(
        documents=chunks,                                 # 传入切分好的文本块
        embedding=embeddings,                             # 使用DeepSeek Embedding
        persist_directory=chroma_path                     # 保存到本地文件夹
    )
    print(f"向量存储完成！共存储了 {len(chunks)} 个向量")
    print(f"数据库保存在：{chroma_path}")
    print("=" * 50)
    # .get() 方法可以查看数据库里的所有ID
    all_ids = vectorstore.get()['ids']
    print(f"数据库中现有的向量ID数量：{len(all_ids)}")
    print(f"前3个ID：{all_ids[:3]}")





# 向量检索测试
def test_02_similarity_search():
    """
    什么是向量检索？
        通俗解释：
            向量检索就是"用问题去找答案"的过程。
        工作流程：
            你问一个问题（比如"什么是机器学习？"）
            系统把这个问题也转成向量
            在数据库里找和问题向量最相似的文本块
            返回最相关的结果
        类比：就像在图书馆找书：
            你告诉管理员你想看什么主题（query）
            管理员根据主题找到最相关的几本书（similarity_search）
            把书递给你（返回chunks）
    核心概念：相似度搜索
        similarity_search(query, k=2)
            query：你的问题（字符串）
            k=2：返回最相关的前2个结果
        原理：计算向量之间的"距离"，距离越近表示越相似
        类比：就像抖音推荐视频
            你看了一個猫咪视频（query）
            系统给你推荐2个（k=2）最相似的猫咪视频

    """
    # 方法1：使用 similarity_search（简单版）
    # 这是最常用的检索方法，直接返回相关的文档块

    # 定义一个测试问题
    query = "什么是机器学习？"

    print(f"\n测试问题：{query}")

    # 执行相似度搜索
    # k=2 表示返回最相关的前2个结果
    results = vectorstore.similarity_search(query, k=2)

    # 打印搜索结果
    print(f"找到 {len(results)} 个相关结果：\n")

    for i, result in enumerate(results, start=1):
        print(f"【结果 {i}】（相似度分数未显示）")
        print(result.page_content)
        print(f"元数据：{result.metadata}\n")

    print("=" * 33)

    # 方法2：多次测试不同问题
    test_queries = [
        "人工智能是什么？",
        "自然语言处理能做什么？",
        "计算机视觉的应用"
    ]

    for query in test_queries:
        print(f"\n问题：{query}")
        results = vectorstore.similarity_search(query, k=1)
        print(f"最相关的内容：{results[0].page_content[:100]}...")
        print("-" * 33)


if __name__ == '__main__':
    # test_02_loader_txt()
    # test_02_loader_txt_split()
    test_02_chroma()