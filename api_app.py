from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import get_app_instance

# Initialize the FastAPI app
api_app = FastAPI()

# Initialize the chatbot instance
app = get_app_instance()
def query_chatbot(question: str):
    state = {
        "messages": [{"role": "user", "content": question}],
        "context": "",
        "question": question
    }
    result = app.invoke(state)
    return result["messages"][-1]["content"]

# Define the request and response model for the API
class QueryRequest(BaseModel):
    question: str
class QueryResponse(BaseModel):
    answer: str

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
    try:
        return QueryResponse(answer=query_chatbot(request.question))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

@api_app.get("/")
def read_root():
    """
    Returns a simple message indicating that the API is running.
    """
    return {"message": "API is running"}