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
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/your-username/sales-chatbot.git
Navigate to the cloned directory:
bash
Copy code
cd sales-chatbot
Install the required packages:
Copy code
pip install -r requirements.txt
Usage
Place your sales interaction dataset named sales_training_data.csv in the project directory. Ensure it has columns labeled 'Seller Question' and 'Buyer Response'.

Run the chatbot script:

Copy code
python chatbot.py
Follow the prompt and start interacting with your chatbot!

To exit the conversation, type exit.

Configuration
To configure the chatbot settings, edit the following parameters in chatbot.py:

OPENAI_API_KEY: Set this to your OpenAI API key.
max_tokens: Adjust this to limit the length of the chatbot's responses.
Contribution
Contributions to this project are welcome. Please fork the repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Remember to replace https://github.com/your-username/sales-chatbot.git with the actual URL of your GitHub repository and add a LICENSE.md if you choose to include licensing. Additionally, ensure any other configuration or usage details specific to your project are updated accordingly in the README.
