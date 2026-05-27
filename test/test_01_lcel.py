import os #是Python自带的一个工具包，用来和操作系统打交道/设置环境变量
from openai import OpenAI
from langchain_openai import ChatOpenAI#专门用来和AI聊天的工具类，可以连接任何兼容OpenAI格式的AI服务
from langchain_core.messages import HumanMessage#用来包装"人类说的话"，让AI能听懂
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser


# 创建OpenAI客户端
# client = OpenAI(                               #创建一个"通信器"，用来和DeepSeek服务器对话;保存手机联系人，存好号码和称呼
#     api_key="sk-60b5c65e6b224fe9aebd91c55576d08f",
#     base_url="https://api.deepseek.com"
# )
#
# response = client.chat.completions.create(
#     model="deepseek-v4-pro",
#     messages=[
#         {"role": "user", "content": "你好，请简单介绍一下你自己"}    #{"role": "user", "content": "..."}：用户说的话
#                                                                 #role可以是"user"（用户）、"assistant"（AI）、"system"（系统设定）
#     ],
#     temperature=0.7
# )
#
# print(response.choices[0].message.content)

os.environ["OPENAI_API_KEY"] = "sk-60b5c65e6b224fe9aebd91c55576d08f"  #全局设置本"，可以记录一些全局配置,
def test_01_api():
    #langchain中api的调用配置
    model= ChatOpenAI(
        model="deepseek-v4-pro",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,   #温度参数，控制AI回答的"创造性"数值越低（如0.1）：AI回答更保守、更准确、更一致;数值越高（如0.9）：AI回答更有创意、更多样化，但也可能更不准确
    )

    # 构建消息
    messages = [HumanMessage(content="你好，请简单介绍一下你自己")] #把你想要问AI的问题打包好，放进一个列表里
    # 调用API并获取响应
    response = model.invoke(messages)    #invoke(...)"执行"或"调用"
    # 输出结果
    print(response.content) #.content：从回答中提取出真正的文字内容;因为response包含很多信息（时间、模型名称等），我们只需要文字部分



"""
        LCEL = LangChain Expression Language（LangChain 表达式语言）
        简单来说，LCEL 是一种用管道符号 | 把多个组件串联起来的方式，让数据像流水线一样从一个组件流向下一个组件。
        用户输入 → Prompt模板 → AI模型 → 输出解析器 → 最终结果
        优势：
           简洁：用 | 符号连接，代码更易读
           灵活：可以随时插入新组件
           自动批处理，支持批量调用，提高效率
           流式输出，可以实时看到AI生成的文字（像打字机效果）

        输出方式 .invoke()、.batch()、.stream() 方法
        
            # 1. invoke - 单次调用
            result = chain.invoke({"input": "你好"})
            
            # 2. batch - 批量调用（并行处理多个输入）
            results = chain.batch([
                {"input": "介绍Python"},
                {"input": "介绍Java"},
                {"input": "介绍C++"}
            ])
            
            # 3. stream - 流式输出（实时显示）
            for chunk in chain.stream({"input": "写一首诗"}):
                print(chunk.content, end="", flush=True)
"""
def test_01_lcel_one():
    prompt = ChatPromptTemplate.from_template(
        "请简单介绍一下自己{name}"
    )
    model = ChatOpenAI(
        model="deepseek-v4-pro",
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,
    )
    output = StrOutputParser()
    chain = prompt | model | output
    result = chain.invoke({"name": "张三"})
    print(result)
def test_01_lcel_two():
    # 创建组件
    prompt = ChatPromptTemplate.from_template(
        "请用JSON格式回答：{{'名称': '{topic}', '简介': '一句话介绍'}}"
    )
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        api_key="sk-60b5c65e6b224fe9aebd91c55576d08f"
    )
    parser = JsonOutputParser()  # JSON解析器

    # 连接成链
    chain = prompt | model | parser

    # 调用（直接得到字典）
    result = chain.invoke({"topic": "机器学习"})
    print(result)  # {'名称': '机器学习', '简介': '...'}
    print(result['简介'])  # 可以直接按键访问



if __name__ == '__main__':
    # test_01_lcel_one()
    pass



