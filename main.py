from src.model import init_retriever, init_llm
from src.app import app
from src.ui import start_ui

def get_app_instance():
    retriever = init_retriever()
    llm = init_llm()
    return app(retriever, llm)

def main():
    start_ui(get_app_instance())

if __name__ == "__main__":
    main()