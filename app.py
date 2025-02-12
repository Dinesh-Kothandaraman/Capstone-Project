import streamlit as st
import pandas as pd
import requests
from langchain.schema import Document
from docGPT import DocGPT
import sqlite3

# API Configuration
API_KEY = "D9AKJQG6VC99GUBR"
BASE_URL = "https://www.alphavantage.co/query"
PARAMS = {
    "function": "TIME_SERIES_MONTHLY",
    "symbol": "IBM",
    "apikey": API_KEY
}

DATABASE = "question_history.db"

def fetch_api_data():
    """Fetches stock data from API and returns a list of LangChain Document objects."""
    response = requests.get(BASE_URL, params=PARAMS)

    if response.status_code == 200:
        stock_data = response.json()
        time_series = stock_data.get("Monthly Time Series", {})
        
        if not time_series:
            st.error("Invalid API response. Check API key and parameters.")
            return []
        
        # Convert time-series data into LangChain Documents
        documents = [
            Document(page_content=f"Date: {date}, Data: {data}")
            for date, data in time_series.items()
        ]
        return documents
    else:
        st.error(f"API request failed: {response.status_code} - {response.text}")
        return []

def main():
    st.title("Stock Data QA System")

    # Initialize database and load history
    initialize_database()
    question_history = load_history()

    # Fetch stock data from API
    st.write("Fetching stock data from API...")
    docs = fetch_api_data()

    if not docs:
        return

    # Train the model
    st.write("Training the model on API data...")
    doc_gpt = DocGPT(docs)
    doc_gpt.create_qa_chain()
    st.success("Training complete!")

    # Accept user queries
    query = st.text_input("Ask a question about the stock data:")
    # if query:
    #     # Save the query to the database
    #     save_to_history(query)
    #     question_history.append(query)

    #     response = doc_gpt.run(query)
    #     st.write("Response:", response)
    if query:
        response = doc_gpt.run(query)
        save_to_history(query, response)  # Ensure you pass both question and answer
        question_history.append((query, response))
        st.write("Response:", response)


    # Display question history
    if question_history:
        st.write("### Question History")
        for i, question in enumerate(question_history, 1):
            st.write(f"{i}. {question}")

# def initialize_database():
#     """Initialize the SQLite database."""
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS question_history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             question TEXT NOT NULL
#         )
#     """)
#     conn.commit()
#     conn.close()
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

# def save_to_history(question):
#     """Save a new question to the SQLite database."""
#     conn = sqlite3.connect(DATABASE)
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO question_history (question) VALUES (?)", (question,))
#     conn.commit()
#     conn.close()
def save_to_history(question, answer):
    """Save a new question and answer to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO question_history (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
