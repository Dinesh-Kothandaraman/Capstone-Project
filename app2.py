import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sqlite3
import json
import PyPDF2

# API Configuration
STOCK_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real API Key
STOCK_BASE_URL = "https://www.alphavantage.co/query"
NEWS_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real News API Key
NEWS_API_URL = "https://www.alphavantage.co/query"
QA_PAIRS_PATH = r"C:\Users\sriha\OneDrive\Documents\GitHub\Capstone-Project\Data\news_qa_pairs.json"
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

# Function to read uploaded document
def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "txt":
            return uploaded_file.read().decode("utf-8")
        elif file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        else:
            st.error("Unsupported file format. Please upload a .txt or .pdf file.")
            return None
    return None

# Fetch Stock Data from API
def fetch_stock_data(symbol="IBM"):
    params = {
        "function": "TIME_SERIES_MONTHLY",
        "symbol": symbol,
        "apikey": STOCK_API_KEY
    }
    response = requests.get(STOCK_BASE_URL, params=params)
    if response.status_code == 200:
        stock_data = response.json().get("Monthly Time Series", {})
        if not stock_data:
            st.warning("No stock data found.")
            return pd.DataFrame()
        df = pd.DataFrame([
            {"date": date, "close": float(data["4. close"])}
            for date, data in stock_data.items()
        ]).sort_values("date")
        return df
    else:
        st.error(f"Stock API request failed: {response.status_code}")
        return pd.DataFrame()

# Visualization of stock data
def plot_stock_data(df, symbol):
    if not df.empty:
        fig = px.line(df, x='date', y='close', title=f'Monthly Closing Prices for {symbol}')
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data to display.")

# Main Streamlit App
def main():
    st.title("📊 Enhanced Stock & News Dashboard")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Configuration")
        stock_symbol = st.selectbox("Select a stock symbol", ["IBM", "AAPL", "GOOGL", "MSFT"], index=0)
        document_uploaded = st.file_uploader("Upload a document (TXT or PDF)", type=["txt", "pdf"])

    # Database initialization and history loading
    initialize_database()

    # Data fetching and plotting
    if st.button("Fetch and Visualize Stock Data"):
        stock_data = fetch_stock_data(stock_symbol)
        plot_stock_data(stock_data, stock_symbol)

    # Handling uploaded documents
    if document_uploaded:
        extracted_text = extract_text_from_file(document_uploaded)
        if extracted_text:
            st.text_area("Extracted Text", extracted_text, height=300)

    # Handling user queries and displaying history from database (Example placeholder)
    user_query = st.text_input("Enter your question about the stock market:")
    if user_query:
        st.write("Response will be generated here...")  # Placeholder for actual response logic

if __name__ == "__main__":
    main()

