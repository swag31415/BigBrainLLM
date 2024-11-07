import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set OpenAI API key if not already set in environment
#os.environ["OPENAI_API_KEY"] = ""  # Set this securely in the environment or directly

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

# Initialize OpenAI LLM with the API key (either passed directly or from environment)
llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Example query to generate a response
query = "How to be happy in life as mentioned by Bhagavat Gita?"

# Generate a response using the RetrievalQA chain
response = qa_chain.run(query)

# Print the response
print("Response:", response)
