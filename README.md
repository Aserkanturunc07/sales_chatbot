# Sales Chatbot

## Overview
This repository contains two chatbot models designed for enhancing the sales process. The first is a straightforward chatbot utilizing OpenAI's GPT-3.5 Turbo for general conversation. The second is a specialized sales chatbot trained to simulate a seller in a conversation with a buyer using a dataset of sales interactions.

### Simple Chatbot
The simple chatbot leverages OpenAI's GPT-3.5 Turbo model to engage in a broad range of discussions. It can be used to answer general queries and is not specifically tailored to sales conversations.

### Sales Chatbot
The sales chatbot, on the other hand, is specifically designed to assist sellers in dealing with customers who are hesistant to buy a product. It uses a combination of `TfidfVectorizer` from the `scikit-learn` library to analyze conversation context and `cosine_similarity` to find the most appropriate sales-oriented response. Its recommended to use this chatbot to train sellers to respond effectively to a buyer who has doubts about buying a product. When using this chatbot imagine you are in the middle of a sales conversation.

## Prerequisites
To run the chatbots, you'll need to have the following prerequisites installed:

- Python 3.6 or higher
- `pandas`
- `scikit-learn`
- `langchain`
- OpenAI's GPT-3.5 Turbo (with API access)

OPENAI_API_KEY: Your OpenAI API key for the model.

## Installation

Clone or download the repository to your local machine:
Install the required Python packages
Navigate to the cloned directory
For the sales chatbot, ensure you have a dataset named sales_conversation_data.csv in the project directory with 'Buyer' and 'Seller' columns.
Run the general_chatbot or sales_chatbot script:

Interact with the chatbot as prompted.

To exit the conversation, type exit.



