import streamlit as st
import requests
import sqlite3
import json
import PyPDF2  # For PDF processing
from langchain.schema import Document
from docGPT import DocGPT

# API Configuration
STOCK_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real API Key
STOCK_BASE_URL = "https://www.alphavantage.co/query"
NEWS_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real News API Key
NEWS_API_URL = "https://www.alphavantage.co/query"
QA_PAIRS_PATH = r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\Data\news_qa_pairs.json"
DATABASE = "question_history.db"

# Initialize Database
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

# Main Streamlit App
def main():
    st.title("📊 Stock & News QA System")

    # Initialize database and load history
    initialize_database()

    # User query input
    user_query = st.text_input("🔍 Ask a question about trends in stock data or news:")

    # Visualize trends based on user query
    if "trend" in user_query.lower():
        # Example visualization (you need to replace this with actual data retrieval and processing logic)
        data = {'Dates': ['2021-01-01', '2021-02-01', '2021-03-01'],
                'Values': [100, 200, 300]}
        df = pd.DataFrame(data)
        st.line_chart(df.set_index('Dates'))

        response = "Displayed trends for your query."
        # Save to database (you might want to adjust this based on actual data processing)
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO question_history (question, answer) VALUES (?, ?)", (user_query, response))
        conn.commit()
        conn.close()

        st.write("**Response:**", response)

    # Display question history
    st.write("### 📜 Question History")
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM question_history")
    history = cursor.fetchall()
    for i, (q, a) in enumerate(history, 1):
        st.write(f"{i}. **Q:** {q}  \n   **A:** {a}")

if __name__ == "__main__":
    main()
