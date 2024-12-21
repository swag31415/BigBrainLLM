from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

from src.templates import detailed_template, concise_template

class State(TypedDict):
    messages: Annotated[list, "The messages in the conversation"]
    context: Annotated[str, "Retrieved context"]
    question: Annotated[str, "User question"]

def classify_question(question: str) -> str:
    # Basic keyword-based classification
    detailed_keywords = ["explain", "how", "why", "describe"]
    if any(kw in question.lower() for kw in detailed_keywords):
        return "detailed"
    return "concise"

def format_docs(docs):
    return "\n\n".join(f"Chapter {doc.metadata.get('chapter', 'N/A')}, Verse {doc.metadata.get('verse', 'N/A')}: {doc.page_content}" for doc in docs)

def app(retriever, llm):
    # Define the nodes
    def retrieve_context(state: State) -> State:
        question = state["messages"][-1]["content"]
        docs = retriever.get_relevant_documents(question)
        new_context = format_docs(docs)
        combined_context = f"{state.get('context', '')}\n\n{new_context}".strip()
        return {
            "messages": state["messages"],
            "context": combined_context,
            "question": question
        }
    def generate_response(state: State) -> State:
        question = state["question"]
        question_type = classify_question(question)
        # Choose the template dynamically
        template = concise_template if question_type == "concise" else detailed_template
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][:-1]])
        response = llm.invoke(
            template.format(
                context=state["context"],
                chat_history=chat_history,
                question=question
            )
        )
        updated_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
        return {
            "messages": updated_messages,
            "context": state["context"],
            "question": question
        }
    # Set up the graph
    graph = StateGraph(State)
    graph.add_node("retriever", retrieve_context)
    graph.add_node("generator", generate_response)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    # Compile the graph
    return graph.compile()
