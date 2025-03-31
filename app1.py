import streamlit as st
import requests
import sqlite3
import logging
from vector_database.faiss_indexing import load_faiss_index
from docGPT import DocGPT

# Logging setup
logging.basicConfig(filename="logs/app.log", level=logging.INFO)

# Load FAISS Index & DocGPT
db = load_faiss_index()
docs = []  # Load dataset here
doc_gpt = DocGPT(docs)
doc_gpt.create_qa_chain()

# SQLite Connection for Storing Queries
DATABASE = "question_history.db"
conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, query TEXT, response TEXT)")
conn.commit()

def save_to_history(question, answer):
    cursor.execute("INSERT INTO queries (query, response) VALUES (?, ?)", (question, answer))
    conn.commit()

st.title("📊 Stock & News QA System")

query = st.text_input("🔍 Ask a question:")

if st.button("Get Answer"):
    if query:
        response = doc_gpt.run(query)
        st.write("**Answer:**", response)

        # Log query
        save_to_history(query, response)
        logging.info(f"Query: {query} | Response: {response}")

    else:
        st.warning("⚠️ Please enter a query.")

# Display previous queries
st.subheader("📜 Query History")
history = cursor.execute("SELECT query, response FROM queries ORDER BY id DESC LIMIT 5").fetchall()
for q, a in history:
    st.write(f"**Q:** {q}  \n **A:** {a}")

conn.close()
