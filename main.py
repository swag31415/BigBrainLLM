from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.interactable.chatgpt import open_api_instance, app  # Import chatbot class
# from RAG_Retrieval import app  # Import the RAG state graph application

# Initialize the FastAPI app
api_app = FastAPI()

# Initialize the chatbot using the RAG application graph
chatbot = open_api_instance

# Define the request model for the API
class QueryRequest(BaseModel):
    question: str  # User's question

# Define the response model for the API
class QueryResponse(BaseModel):
    answer: str  # Chatbot's answer

# Define the API endpoint to query the chatbot
@api_app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    """
    Handles a query to the chatbot.

    Args:
        request (QueryRequest): The user's question.

    Returns:
        QueryResponse: The chatbot's answer.
    """
    print('This is working!')
    try:
        # Use the chatbot instance to process the user's question
        result = chatbot.query(request.question)
        
        # Extract the assistant's final response from the result
        assistant_message = result["messages"][-1]["content"]
        
        return QueryResponse(answer=assistant_message)
    except Exception as e:
        # Handle any exceptions gracefully and return a proper error
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")
