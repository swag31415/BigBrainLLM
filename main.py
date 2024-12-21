from src.model import init_retriever, init_llm
from src.app import app
from src.ui import start_ui

if __name__ == "__main__":
    retriever = init_retriever()
    llm = init_llm()
    _app = app(retriever, llm)
    start_ui(_app)