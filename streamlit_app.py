from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st
st.set_page_config(page_title="C137 Test", page_icon=":shark:")
st.header("Welcome to C137 Test! It's adventure time!")



import getpass
import os
from dotenv import main
from pathlib import Path

dotenv_path = Path('.env')
# dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
main.load_dotenv(dotenv_path=dotenv_path)

# main.load_dotenv()

os.environ["MISTRAL_API_KEY"]=os.getenv("MISTRAL_API_KEY") 
api_key = os.getenv("MISTRAL_API_KEY") 

from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )

# Setting which embedding model you want to use.
embeddings_model = MistralAIEmbeddings(
    model="mistral-embed",
)

chat_file = open("./data/ian_chat_history.txt", "rt", encoding='utf-8-sig') # open lorem.txt for reading text
raw_documents = chat_file.read()         # read the entire file to string
chat_file.close()                   # close the file


documents = text_splitter.split_text(raw_documents)    # .split_document when using pdf file
db = FAISS.from_texts(documents, embeddings_model)   # .from_documents when using pdf file
memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False  # if want to see the memory message, set to True 
    )



llm = init_chat_model("mistral-small-latest", model_provider="mistralai")



from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent # This is for OpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain_mistralai.chat_models import ChatMistralAI



# 初始化搜尋工具

wrapper = DuckDuckGoSearchAPIWrapper(region="tw-tzh", max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=wrapper)  #, source="news")  # You can also fix the source from news

# 定義工具
tools = [
    # Tool.from_function(
    #     func=search.invoke,
    #     name="duch_search",
    #     description="useful for when you need to answer questions about current events"
    # ),
    search,
    create_retriever_tool(
        db.as_retriever(),  
        "chat_history",  # name
        "Searches and returns dialogue said by ian yu"  # description
    )
]



# test_query = "有免費旅遊諮詢電話嗎？"

from langchain_core.messages import HumanMessage

from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''
         You are Ian001, and you are a chatbot that imitates Ian Yu, whose MBTI personality is INTJ.
         
         Please analyze Ian Yu's chat history and use it to chat with users.
         You should behave like him and you are also allow to use the emoji that Ian Yu would use.
         You should answer questions in a way that is consistent with Ian Yu's personality and style.         
         When you respond to the user, you should use the same tone and style as ian yu.

         If you cannot answer the user's question, instead of making up an answer, you should guide the user to ask 我本人.
                  
         Please use Chinese Traditional (zh-tw) to answer questions. 
         
         '''),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)



# Create the agent with the LLM, tools, and prompt

agent = create_tool_calling_agent(llm, tools, prompt)
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)



if __name__ == "__main__":

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="你好，我是C137測試機器人(Ian001)。")]
    if "memory" not in st.session_state:
        st.session_state['memory'] = memory
 
    
    for msg in st.session_state.messages:
        try:
            st.chat_message(msg.role).write(msg.content)    
        except:
            st.chat_message(msg["role"]).write(msg["content"])

    
    
    if user_query := st.chat_input("Say something to Ian001"):
         
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query) 
        
        



        with st.chat_message("assistant"):


            st_cb = StreamlitCallbackHandler(st.container())
            # response = agent(user_query, callbacks=[st_cb])
            
            response = agent_executor.invoke({"input":[HumanMessage(content=user_query)], #"chat_history": [memory.load_memory_variables({})],
        })

            # response = agent_executor.invoke(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            
            st.write(response['output'])
            
            

    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = []






