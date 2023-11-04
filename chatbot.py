import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = "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"

# Reading in the csv as a DataFrame
df = pd.read_csv("sales_training_data.csv")

class ResponseRetriever:
    def __init__(self, data):
        self.vectorizer = TfidfVectorizer()
        self.data = data
        self.vectorizer.fit(self.data['Seller Question'])  # Vectorize the questions for later comparison

    def get_response(self, input_question):
        # Vectorize the input question
        question_vector = self.vectorizer.transform([input_question])
        # Vectorize all questions from the dataset for comparison
        data_vectors = self.vectorizer.transform(self.data['Seller Question'])

        # Calculate similarities
        similarities = cosine_similarity(question_vector, data_vectors)

        # Find the index of the most similar question
        most_similar_index = similarities.argmax()

        # Return the most similar response
        return self.data.iloc[most_similar_index]['Buyer Response']


# Assuming you've set your OPENAI_API_KEY
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY,max_tokens=100)
response_retriever = ResponseRetriever(df)


def chatbot():
    print("Hello! I am a Chatbot trained to respond like a buyer. Type 'exit' to end the conversation.")
    print(1)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Retrieve the most similar buyer response from the dataset
        buyer_response = response_retriever.get_response(user_input)

        # Create the message history context for the AI
        messages = [SystemMessage(content="This chatbot generates responses to seller"
                                          " questions from a buyer perspective"),
                    HumanMessage(content=user_input),
                    AIMessage(content=buyer_response)]  # Simulated AI response based on retrieved data

        # Get the AI's generated response considering the context
        chatbot_response = chat(messages)

        print(f"Chatbot: {chatbot_response.content}")

if __name__ == "__main__":
    chatbot()
