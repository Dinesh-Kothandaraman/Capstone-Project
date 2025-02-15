# import streamlit as st
# import pandas as pd
# import requests
# from langchain.schema import Document
# from docGPT import DocGPT
# import sqlite3

# # API Configuration
# API_KEY = "D9AKJQG6VC99GUBR"
# BASE_URL = "https://www.alphavantage.co/query"
# PARAMS = {
#     "function": "TIME_SERIES_MONTHLY",
#     "symbol": "IBM",
#     "apikey": API_KEY
# }

# DATABASE = "question_history.db"

# def fetch_api_data():
#     """Fetches stock data from API and returns a list of LangChain Document objects."""
#     response = requests.get(BASE_URL, params=PARAMS)

#     if response.status_code == 200:
#         stock_data = response.json()
#         time_series = stock_data.get("Monthly Time Series", {})
        
#         if not time_series:
#             st.error("Invalid API response. Check API key and parameters.")
#             return []
        
#         # Convert time-series data into LangChain Documents
#         documents = [
#             Document(page_content=f"Date: {date}, Data: {data}")
#             for date, data in time_series.items()
#         ]
#         return documents
#     else:
#         st.error(f"API request failed: {response.status_code} - {response.text}")
#         return []

# def main():
#     st.title("Stock Data QA System")

#     # Initialize database and load history
#     initialize_database()
#     question_history = load_history()

#     # Fetch stock data from API
#     st.write("Fetching stock data from API...")
#     docs = fetch_api_data()

#     if not docs:
#         return

#     # Train the model
#     st.write("Training the model on API data...")
#     doc_gpt = DocGPT(docs)
#     doc_gpt.create_qa_chain()
#     st.success("Training complete!")

#     # Accept user queries
#     query = st.text_input("Ask a question about the stock data:")
#     if query:
#         response = doc_gpt.run(query)
#         save_to_history(query, response)
#         question_history.append((query, response))
#         if "Question:" in response:
#             response = response.split("Question:")[-1]  # Remove everything before "Question:"
#         if "Retrieved Context:" in response:
#             response = response.split("Retrieved Context:")[-1]  # Remove retrieved context

#     # Keep only the final answer
#         answer_only = response.split("\n")[0].strip()  
#         st.write(answer_only)  # Remove "Response:" label
#     # Display question history
#     if question_history:
#         st.write("### Question History")
#         for i, question in enumerate(question_history, 1):
#             st.write(f"{i}. {question}")

# def initialize_database():
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS question_history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             question TEXT NOT NULL,
#             answer TEXT NOT NULL
#         )
#     """)
#     conn.commit()
#     conn.close()


# def load_history():
#     """Load the question history from the SQLite database."""
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()
#     cursor.execute("SELECT question FROM question_history")
#     history = [row[0] for row in cursor.fetchall()]
#     conn.close()
#     return history

# def save_to_history(question, answer):
#     """Save a new question and answer to the SQLite database."""
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO question_history (question, answer) VALUES (?, ?)", (question, answer))
#     conn.commit()
#     conn.close()


# if __name__ == "__main__":
#     main()


import streamlit as st
import requests
import sqlite3
from langchain.schema import Document
from docGPT import DocGPT

# API Configuration
STOCK_API_KEY = "D9AKJQG6VC99GUBR"
STOCK_BASE_URL = "https://www.alphavantage.co/query"
NEWS_API_KEY = "D9AKJQG6VC99GUBR"  # Replace with actual News API Key
NEWS_API_URL = "https://www.alphavantage.co/query"

DATABASE = "question_history.db"

# Fetch stock data
def fetch_stock_data():
    """Fetches stock data from Alpha Vantage API."""
    params = {
        "function": "TIME_SERIES_MONTHLY",
        "symbol": "IBM",
        "apikey": STOCK_API_KEY
    }
    response = requests.get(STOCK_BASE_URL, params=params)
    
    if response.status_code == 200:
        stock_data = response.json()
        time_series = stock_data.get("Monthly Time Series", {})
        
        if not time_series:
            st.error("Invalid stock API response. Check API key and parameters.")
            return []
        
        return [
            Document(page_content=f"Date: {date}, Data: {data}")
            for date, data in time_series.items()
        ]
    else:
        st.error(f"Stock API request failed: {response.status_code} - {response.text}")
        return []

# Fetch stock-related news
def fetch_stock_news():
    """Fetches stock-related news from NewsAPI."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("feed", [])
        
        if not articles:
            st.warning("No relevant news found.")
            return []
        
        return [
            Document(page_content=f"Headline: {article['title']}, Summary: {article['description']}")
            for article in articles
        ]
    else:
        st.error(f"News API request failed: {response.status_code} - {response.text}")
        return []

# Main Streamlit App
def main():
    st.title("📈 Stock Data & News QA System")

    # Initialize database and load history
    initialize_database()
    question_history = load_history()

    # Fetch stock data and news
    st.write("Fetching stock data and news from API...")
    stock_docs = fetch_stock_data()
    news_docs = fetch_stock_news()

    # Merge both sources into one dataset
    docs = stock_docs + news_docs

    if not docs:
        return

    # Train the model
    st.write("Training the model on API data...")
    doc_gpt = DocGPT(docs)
    doc_gpt.create_qa_chain()
    st.success("✅ Training complete!")

    # Accept user queries
    query = st.text_input("Ask a question about the stock data or news:")

    if query:
        response = doc_gpt.run(query)
        save_to_history(query, response)
        question_history.append((query, response))

        # # Process response to remove unnecessary details
        # if "Question:" in response:
        #     response = response.split("Question:")[-1]
        # if "Retrieved Context:" in response:
        #     response = response.split("Retrieved Context:")[-1]

        # # Keep only the final answer
        # answer_only = response.split("\n")[0].strip()  
        st.write("**Answer:**", response)

    # Display question history
    if question_history:
        st.write("### 📜 Question History")
        for i, question in enumerate(question_history, 1):
            st.write(f"{i}. {question}")

# Database functions
def initialize_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def load_history():
    """Load the question history from the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT question FROM question_history")
    history = [row[0] for row in cursor.fetchall()]
    conn.close()
    return history

def save_to_history(question, answer):
    """Save a new question and answer to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO question_history (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
