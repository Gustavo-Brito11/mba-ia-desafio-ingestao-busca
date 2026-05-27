from search import search_prompt, PROMPT_TEMPLATE
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

def main():
    

    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE
    )
    
    chain = prompt | llm
    
    while True:
        
        print("Faça sua pergunta:\n")
        
        comando = input("PERGUNTA: ").strip().lower()

        if comando == 'sair':
            print("Saindo do chat. Até mais!")
            break

        contexto = search_prompt(comando)
        
        resposta = chain.invoke(
            {
                "contexto": contexto,
                "pergunta": comando
            }
        )
        
        print(f"\nRESPOSTA: {resposta.content}\n")
        
if __name__ == "__main__":
    main()