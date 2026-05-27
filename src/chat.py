from search import search_prompt, PROMPT_TEMPLATE
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda 
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore') 

load_dotenv()

session_histories: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

def main():
    

    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{pergunta}")
    ])
    
    chain = chat_prompt | llm
    
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="pergunta",
        history_messages_key="history"
    )
    
    config = {"configurable": {"session_id": "demo_session"}}
    
    while True:
        
        print("Faça sua pergunta:\n")
        
        comando = input("PERGUNTA: ").strip().lower()

        if comando == 'sair':
            print("Saindo do chat. Até mais!")
            break

        contexto = search_prompt(comando)
        
        resposta = conversational_chain.invoke(
            {
                "contexto": contexto,
                "pergunta": comando
            },
            config=config
        )
        
        print(f"\nRESPOSTA: {resposta.content}\n")
        
if __name__ == "__main__":
    main()