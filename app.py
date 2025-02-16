import streamlit as st
import requests
import sqlite3
import json
from langchain.schema import Document
from docGPT import DocGPT

# API Configuration
STOCK_API_KEY = "D9AKJQG6VC99GUBR"  # Replace with your real API Key
STOCK_BASE_URL = "https://www.alphavantage.co/query"

NEWS_API_KEY = "D9AKJQG6VC99GUBR"  # Replace with your real News API Key
NEWS_API_URL = "https://www.alphavantage.co/query"

QA_PAIRS_PATH = r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\Data\news_qa_pairs.json"  # Pre-generated QA file
DATABASE = "question_history.db"

# Fetch Stock Data from API
def fetch_stock_data():
    """Fetches real-time stock data from Alpha Vantage API."""
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
            st.warning("No stock data found.")
            return []

        return [
            Document(page_content=f"Stock Date: {date}\nStock Data: {json.dumps(data)}")
            for date, data in time_series.items()
        ]
    else:
        st.error(f"Stock API request failed: {response.status_code}")
        return []

# Fetch Stock-Related News from API
def fetch_stock_news():
    """Fetches real-time stock-related news from Alpha Vantage API."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "apikey": NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("feed", [])

        if not articles:
            st.warning("No relevant news found.")
            return []

        return [
            Document(page_content=f"News Title: {article['title']}\nSummary: {article['summary']}\nSentiment: {article['overall_sentiment_label']}")
            for article in articles
        ]
    else:
        st.error(f"News API request failed: {response.status_code}")
        return []

# Load Pre-generated QA Pairs
def load_qa_pairs():
    """Loads pre-generated QA pairs from the JSON file."""
    try:
        with open(QA_PAIRS_PATH, "r") as f:
            qa_data = json.load(f)
        
        return [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in qa_data]
    except Exception as e:
        st.error(f"Error loading QA pairs: {e}")
        return []

# Initialize Database
def initialize_database():
    """Creates a database table if it doesn't exist."""
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

# Load Question History from Database
def load_history():
    """Retrieves past user queries from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM question_history")
    history = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return history


# Save User Query and Response to Database
def save_to_history(question, answer):
    """Stores the user's query and the model's response."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO question_history (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# Main Streamlit App
def main():
    st.title("📊 Stock & News QA System (Two APIs + QA Pairs)")

    # Initialize database and load history
    initialize_database()
    question_history = load_history()

    # Fetch stock data and news data dynamically
    st.write("🔄 Fetching stock data, news, and QA pairs...")
    stock_docs = fetch_stock_data()
    news_docs = fetch_stock_news()
    qa_docs = load_qa_pairs()  # Load QA dataset

    # Combine API data + QA pairs
    docs = stock_docs + news_docs + qa_docs

    if not docs:
        st.error("No data available for training.")
        return

    # Train the DocGPT model with API stock data, news articles, and QA pairs
    st.write("🧠 Training model with API data + QA pairs...")
    doc_gpt = DocGPT(docs)
    doc_gpt.create_qa_chain()
    st.success("✅ Training complete!")

    # Accept user queries
    query = st.text_input("🔍 Ask a question about stock data, news, or QA pairs:")

    if query:
        response = doc_gpt.run(query)
        save_to_history(query, response)
        question_history.append((query, response))
        st.write("**Answer:**", response)

    # Display question history
    if question_history:
        st.write("### 📜 Question History")
        for i, (q, a) in enumerate(question_history, 1):
            st.write(f"{i}. **Q:** {q}  \n   **A:** {a}")


if __name__ == "__main__":
    main()
