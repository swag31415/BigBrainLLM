import os
import time
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langgraph import Graph, Node
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the OpenAI API key securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    exit()

# Initialize OpenAI LLM with the API key
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1500)

# Streamlit UI improvements
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background-color: #f7f7f7;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .user-msg, .bot-msg {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            max-width: 80%;
        }
        .user-msg {
            background-color: #DCF8C6;
            margin-left: auto;
        }
        .bot-msg {
            background-color: #E5E5E5;
        }
        .stTextInput input {
            border-radius: 15px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            border-radius: 15px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Check if session_state.messages exists, and initialize it if not
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Step 1: Define LangGraph Nodes ---

def input_node():
    """Input node to handle user queries."""
    query = st.text_input("Enter your message:", key="input_text", placeholder="Type something...")
    return query

def memory_node(history, new_message):
    """Memory node to manage conversation history."""
    if new_message:
        history.append(f"User: {new_message}")
    return history

def llm_node(history, query):
    """LLM node to generate responses based on the query."""
    conversation_history = "\n".join(history)
    response = llm.predict(f"{conversation_history}\nUser: {query}\nBot:")
    return response

def output_node(messages):
    """Output node to display the chat."""
    for message in messages:
        if "User:" in message:
            st.markdown(f'<div class="user-msg">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{message}</div>', unsafe_allow_html=True)

# --- Step 2: Construct LangGraph ---

# Create LangGraph nodes
input_node_obj = Node(name="Input", function=input_node)
memory_node_obj = Node(name="Memory", function=memory_node)
llm_node_obj = Node(name="LLM", function=llm_node)
output_node_obj = Node(name="Output", function=output_node)

# Create LangGraph and add nodes
chat_graph = Graph()
chat_graph.add_node(input_node_obj)
chat_graph.add_node(memory_node_obj)
chat_graph.add_node(llm_node_obj)
chat_graph.add_node(output_node_obj)

# Define the workflow edges
chat_graph.add_edge(input_node_obj, memory_node_obj)
chat_graph.add_edge(memory_node_obj, llm_node_obj)
chat_graph.add_edge(llm_node_obj, output_node_obj)

# --- Step 3: Execute the Graph ---

# Container for the chatbot UI
with st.container():
    st.title("LangGraph-powered Chatbot 🤖")
    st.write("""
        Start a conversation, and I will remember what we talked about!
    """)

    # Display the chat history
    output_node_obj.run(messages=st.session_state.messages)

    # Get user input
    user_query = input_node_obj.run()

    if st.button("Send") and user_query:
        # Update memory and get bot response
        st.session_state.messages = memory_node_obj.run(history=st.session_state.messages, new_message=user_query)
        bot_response = llm_node_obj.run(history=st.session_state.messages, query=user_query)
        st.session_state.messages.append(f"Bot: {bot_response}")

        # Rerun the output node to display the updated chat
        output_node_obj.run(messages=st.session_state.messages)
