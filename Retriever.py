import os
import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# Set OpenAI API key if not already set in environment
os.environ["OPENAI_API_KEY"] = "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"

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
    print(f"Error occurred while loading vector database: {e}")
    exit()

# Initialize the retriever from the loaded vector database
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize OpenAI LLM with the API key
llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("Bhagavad Gita Question Answering")

st.write("""
    Ask a question related to the Bhagavad Gita, and get an insightful response based on the teachings.
""")

# Input for the user question
query = st.text_input("Enter your question:")

# Button to submit the question and get the answer
if st.button('Get Answer'):
    if query:
        # Generate a response using the RetrievalQA chain
        response = qa_chain.run(query)
        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a question.")

