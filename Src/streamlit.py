import streamlit as st
from src.inference import run_inference
from src.data_ingestion import fetch_stock_data

def main():
    st.title("Stock QA System")
    query = st.text_input("Ask a question:")
    if query:
        response = run_inference(query, st.session_state.doc_gpt)
        st.write("Answer:", response)

if __name__ == "__main__":
    main()