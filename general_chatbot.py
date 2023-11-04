"""
This module implements a basic chatbot using OpenAI's language model.
The chatbot interacts with the user in a conversational manner and can handle a variety of topics.
"""
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Placing your actual OpenAI API key here
OPENAI_API_KEY = "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"

# Initializing the chat model with the OpenAI API key and preferred model settings
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)

def chatbot():
    """
    Initiates and manages a conversation with the user. The chatbot continues interacting
    until the user types 'exit'.
    """
    print("Hello! I am a Chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Creating the message history context for the AI, including the latest user message
        messages = [HumanMessage(content=user_input)]

        # Generating the AI's response considering the current conversation context
        chatbot_response = chat(messages)

        print(f"Chatbot: {chatbot_response.content}")

if __name__ == "__main__":
    chatbot()
