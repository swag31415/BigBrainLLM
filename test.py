from chatbot import chatbot_instance
from models.instances.mistral import mistral_chat_instance

print('Created Chatbot Instance')

# Query the chatbot
response = mistral_chat_instance.query("What is the purpose of life?")
print("Response: ", response["messages"][-1]["content"])