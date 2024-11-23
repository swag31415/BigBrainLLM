import os
import time
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# Load environment variables from the .env file
load_dotenv()

# Fetch the OpenAI API key securely from environment variables
openai_api_key = os.getenv("OPEN_API_KEY")

if not openai_api_key:
    st.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    exit()

# Initialize embedding model with LangChain's SentenceTransformerEmbeddings wrapper
embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')

# Load the existing vector database from the specified directory
vector_db_directory = "chroma_vector_db"
vectordb = None

try:
    if os.path.exists(vector_db_directory):
        print(f"Vector database loaded from {vector_db_directory}.")
        vectordb = Chroma(persist_directory=vector_db_directory, embedding_function=embedding_model)
    else:
        print(f"Vector database does not exist in {vector_db_directory}.")
        exit()
except Exception as e:
    st.error(f"Error occurred while loading vector database: {e}")
    exit()

# Initialize the retriever from the loaded vector database
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize OpenAI LLM with the API key
try:
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key,max_tokens=1500)
except Exception as e:
    st.error(f"Error initializing OpenAI LLM: {e}")
    exit()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # Initialize OpenAI LLM with the API key
# llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1500)

# Initialize memory to store conversation history, explicitly setting the memory key to 'history'
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Create the ConversationChain
conversation = ConversationChain(llm=llm, memory=memory)

# Streamlit UI improvements
st.set_page_config(page_title="Memory-enabled Chatbot", page_icon="🤖", layout='centered')

# Custom CSS to enhance the UI
# st.markdown("""
#     <style>
#         .chat-container {
#             max-width: 800px;
#             margin: auto;
#             padding: 20px;
#             background-color: #f7f7f7;
#             border-radius: 10px;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         }
#         .user-msg, .bot-msg {
#             padding: 10px;
#             margin: 5px 0;
#             border-radius: 10px;
#             max-width: 80%;
#         }
#         .user-msg {
#             background-color: #DCF8C6;
#             margin-left: auto;
#         }
#         .bot-msg {
#             background-color: #E5E5E5;
#         }
#         .stTextInput input {
#             border-radius: 15px;
#             padding: 10px;
#             font-size: 16px;
#         }
#         .stButton button {
#             border-radius: 15px;
#             background-color: #4CAF50;
#             color: white;
#             font-weight: bold;
#             padding: 10px;
#         }
#         .stButton button:hover {
#             background-color: #45a049;
#         }
#     </style>
# """, unsafe_allow_html=True)

# Check if the messages history exists in session_state, and initialize it if not
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Chat UI container
with st.container():
    st.title("Prabhupad Bhagavad Gita Chat Bot 🤖")
    st.write("""
        Start a conversation, and I will remember what we talked about!
    """)

    # Display the chat history in a scrollable container
    for message in st.session_state.messages:
        if "User:" in message:
            st.markdown(f'<div class="user-msg">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{message}</div>', unsafe_allow_html=True)

    # Input box for the user message
    query = st.text_input("Enter your message:", key="input_text", placeholder="Type something...")

    # Define the function to handle message submission
    # def handle_message_submission():
    #     query = st.session_state.input_text
    #     if query:
    #         # Display the user's message
    #         st.session_state.messages.append(f"User: {query}")

    #         # Simulate a slight delay to make it feel more conversational
    #         with st.spinner("Thinking..."):
    #             time.sleep(1)  # Add delay before response

    #         # Get the response from the chatbot
    #         response = conversation.predict(input=query)
    #         st.session_state.messages.append(f"Bot: {response}")

    #         # Display the bot's response
    #         st.markdown(f'<div class="bot-msg">{response}</div>', unsafe_allow_html=True)

    #         # Scroll to the bottom after each new message
    #         st.rerun()

    #     else:
    #         st.warning("Please enter a message.")

    # Define the function to handle message submission
    def handle_message_submission():
        query = st.session_state.input_text  # Get the current input
        if query:
            # Add the user's query to the conversation history
            st.session_state.messages.append(f"User: {query}")

            # Simulate a slight delay to make it feel more conversational
            with st.spinner("Thinking..."):
                time.sleep(1)  # Add delay before response

            # Retrieve context-relevant information
            try:
                response = qa_chain.run({"query": query, "history": memory.load_memory_variables({})})
            except Exception as e:
                response = f"An error occurred while fetching the response: {e}"

            # Append the bot's response to messages and memory
            st.session_state.messages.append(f"Bot: {response}")
            memory.save_context({"input": query}, {"output": response})

            # Clear the text input indirectly by resetting a temporary key
            st.session_state.input_text_temp = ""

            # Display the bot's response
            st.markdown(f'<div class="bot-msg">{response}</div>', unsafe_allow_html=True)

            # Refresh the UI
            st.rerun()
        else:
            st.warning("Please enter a message.")


    # Button to submit the message and get the answer
    if st.button('Send'):
        handle_message_submission()

    # # Handle Enter key press by triggering the same function
    # if query:
    #     handle_message_submission()
