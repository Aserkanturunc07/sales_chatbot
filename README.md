Sales Chatbot
This project features a chatbot that simulates buyer responses in a sales conversation. Utilizing a vector space model for understanding the context and providing relevant answers, it leverages a dataset of seller and buyer interactions to mimic a human buyer.

Features
Vector Space Model: Leverages TfidfVectorizer to understand the context of the conversation and find the most appropriate response.
Cosine Similarity: Uses cosine similarity to match user input with the most similar question in the dataset and retrieves a corresponding buyer response.
OpenAI Integration: Employs OpenAI's GPT-3.5 Turbo to generate responses that are conditioned on previously retrieved buyer responses.
Prerequisites
Python 3.6+
pandas
scikit-learn
OpenAI GPT-3.5 Turbo (API access)
langchain library
Ensure you have the above prerequisites installed and an API key from OpenAI before proceeding.

Installation
Clone or download the repository to your local machine:
Navigate to the cloned directory:
Run the general_chatbot or sales_chatbot script!
Follow the prompt and start interacting with your chatbot!

To exit the conversation, type exit.
