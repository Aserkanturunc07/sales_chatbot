"""
This module creates a chatbot simulating seller responses to buyer comments.
Utilizing a trained AI model, the chatbot generates persuasive responses aimed at enhancing
sales dialogues. Seller responses are retrieved based on similarity from a pre-defined dataset
of buyer-seller conversations.
"""

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Replace the placeholder with your actual OpenAI API key
OPENAI_API_KEY = "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"


class ResponseRetriever:
    """
    A class for retrieving seller responses to buyer inquiries.

    Attributes:
        vectorizer (TfidfVectorizer): Transforms text to vector form.
        data (DataFrame): A dataset containing 'Buyer' and 'Seller' conversation pairs.
    """

    def __init__(self, data):
        """
        Initializes ResponseRetriever with the provided dataset.

        Parameters:
            data (DataFrame): Dataset of sales conversation data.
        """
        self.vectorizer = TfidfVectorizer()
        self.data = data
        # Vectorizing the 'Buyer' statements for later comparison
        self.vectorizer.fit(self.data['Buyer'])

    def get_response(self, input_question):
        """
        Retrieves the seller response most similar to the input buyer question.

        Parameters:
            input_question (str): The inquiry or statement from the buyer.

        Returns:
            str: The corresponding seller response, matching in context.
        """
        # Transforming the input question into vector form
        question_vector = self.vectorizer.transform([input_question])
        # Transforming all buyer statements for comparison
        data_vectors = self.vectorizer.transform(self.data['Buyer'])
        # Calculating similarity scores
        similarities = cosine_similarity(question_vector, data_vectors)
        # Identifying the index of the closest-matching buyer statement
        most_similar_index = similarities.argmax()
        # Returning the associated seller response
        return self.data.iloc[most_similar_index]['Seller']


# Initialize the chat model with predefined parameters.
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY, max_tokens=200)
response_retriever = ResponseRetriever(pd.read_csv("sales_conversation_data.csv"))


def chatbot():
    """
    Initiates a chatbot conversation, simulating a seller's interaction.
    The chat continues until the user opts to exit by typing 'exit'.
    """
    print("Hello! I am a Chatbot trained to converse like I am a seller. Start with a prompt like 'Why should I buy this product' or 'This product looks expensive'  Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Retrieving the closest matching buyer response from the dataset
        buyer_response = response_retriever.get_response(user_input)

        # Creating a message history context for the AI
        messages = [
            SystemMessage(
                content="This chatbot is generating persuasive responses to buyers' comments from a seller's perspective. Keep responses under 100 words, aiming to sell."),
            HumanMessage(content=user_input),
            AIMessage(content=buyer_response)  # Simulating a response based on the retrieved data
        ]

        # Generating the chatbot's response given the conversation context
        chatbot_response = chat(messages)

        print(f"Chatbot: {chatbot_response.content}")


if __name__ == "__main__":
    chatbot()
