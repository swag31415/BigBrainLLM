import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
openai_api_key = os.getenv("OPEN_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key not found in the environment variables.")

# Initialize ChromaDB and retriever
embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
vectorstore = Chroma(
    persist_directory="chroma_vector_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

# Define prompt template
template = """
As Srila Prabhupada, answer this question based on the Bhagavad Gita teachings and lectures:
Context from Bhagavad Gita and lectures: {context}
Previous conversation: {chat_history}
Devotee's Question: {question}

These are some examples of answers you can provide to given questions:

Question 1: 
Why is the Bhagavad-Gita the perfect theistic science? (1.1)

Answer 1: 
It is the perfect theistic science because it is directly spoken by the Supreme Personality of Godhead, Lord Śrī Kṛṣṇa.

Question 2: 
Name four powerful fighters each on the side of the Kauravas and the Pandavas. (1.4-1.6, 1.8)

Answer 2: 
Kaurava's Side:
-Drona
-Bhisma
-Karna
-Kripacarya

Pandava's Side:
-Arjuna
-Bhima
-Virata
-Drupada
-Abhimanyu

Question 3:
Describe the significance of the blowing of conchshells on both the sides. (1.14, 1.19)

Answer 3:
The sounding of the transcendental conchshells of Krsna and Arjuna indicated that there was no hope of victory for the other side because Kṛṣṇa was on the side of the Pāṇḍavas.
While there was no fear in the Pandava's hearts when the Kaurava's blew their conch shells, the hearts of the sons of Dhṛtarāṣṭra were shattered by the sounds vibrated by the Pāṇḍavas' party. 
This is due to the Pāṇḍavas and their confidence in Lord Kṛṣṇa. One who takes shelter of the Supreme Lord has nothing to fear, even in the midst of the greatest calamity.
...
My dear devotee, let me explain this point according to the Bhagavad Gita's teachings:
"""
rag_prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

# Define the state
class State(TypedDict):
    messages: Annotated[list, "The messages in the conversation"]
    context: Annotated[str, "Retrieved context"]
    question: Annotated[str, "User question"]

# LLM instance
llm = ChatOpenAI(temperature=0, streaming=False, openai_api_key=openai_api_key, max_tokens=1500)

# Define the graph
graph = StateGraph(State)
def retrieve_context(state: State):
    question = state["messages"][-1]["content"]
    docs = retriever.get_relevant_documents(question)
    formatted_context = "\n\n".join(doc.page_content for doc in docs)
    return {
        "messages": state["messages"],
        "context": formatted_context,
        "question": question
    }

def generate_response(state: State):
    chat_history = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"][:-1])
    response = llm.invoke(
        rag_prompt.format(
            context=state["context"],
            chat_history=chat_history,
            question=state["question"]
        )
    )
    updated_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
    return {
        "messages": updated_messages,
        "context": state["context"],
        "question": state["question"]
    }

# Add nodes to the graph
graph.add_node("retriever", retrieve_context)
graph.add_node("generator", generate_response)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)

# Compile graph into app instance
app = graph.compile()

# Chatbot instance
class ChatbotApp:
    def __init__(self, app):
        self.app = app

    def query(self, question):
        state = {
            "messages": [{"role": "user", "content": question}],
            "context": "",
            "question": question
        }
        return self.app.invoke(state)

# Create a chatbot instance
chatbot_instance = ChatbotApp(app)

# print('Created Chatbot Instance')

# # Query the chatbot 
# response = chatbot_instance.query("What is the purpose of life?")

# print(response['messages'][-1]["content"])