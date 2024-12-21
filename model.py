from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ui import init_api_keys
from langchain_openai import ChatOpenAI

def init_retriever():
    vectorstore = Chroma(
        persist_directory="chroma_vector_db",
        embedding_function=SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
    )
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

def init_llm():
    openai_api_key = init_api_keys()
    return ChatOpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key, max_tokens=1500)