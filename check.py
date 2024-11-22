import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the OpenAI API key
openai_api_key = os.getenv("OPEN_API_KEY")

openai_api_key = os.getenv("OPENAI_API_KEY")

# Debugging
print("OpenAI API Key:", openai_api_key)