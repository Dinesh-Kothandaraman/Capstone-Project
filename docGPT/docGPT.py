from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
import torch
import matplotlib.pyplot as plt
import os
import io
import base64
import json
from io import BytesIO
import chat2plot as c2p

# saved_model = r"C:\Users\dines\Downloads\Capstone_model"
class DocGPT:
    def __init__(self, docs, embedding_model="BAAI/bge-large-en",model_name=r"D:\New folder"):
    # def __init__(self, docs, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    
        """
        Initializes DocGPT with a given dataset.
        Supports multiple document types: stock data, news articles, QA pairs.
        """
        self.docs = docs
        self.qa_chain = None
        self.embedding_model = embedding_model
        self._llm = pipeline("text-generation", model=model_name, max_new_tokens=256)
        self._db = None  # Store FAISS DB to avoid recomputation

    def _preprocess_docs(self):
        """Optimized document chunking to improve retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunked_docs = [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in self.docs
            for chunk in text_splitter.split_text(doc.page_content)
        ]
        return chunked_docs

    def _embeddings(self):
        """Creates embeddings and a FAISS vector database."""
        if self._db:
            return self._db  # Avoid redundant FAISS creation
        
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        chunked_docs = self._preprocess_docs()
        if not chunked_docs:
            raise ValueError("No documents available for embedding. Check data sources.")
        self._db = FAISS.from_documents(chunked_docs, embedding=embeddings)
        return self._db

    def create_qa_chain(self, retriever_k=5, model_path=r"C:\Users\dines\Downloads\Capstone_model"):
        """Sets up the RAG pipeline using the fine-tuned model."""
        db = self._embeddings()
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})

        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
        # model.config.max_length = 1024 

        # Create the pipeline for inference
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,max_new_tokens=512, temperature=0.7, do_sample=True)

        local_llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
        )

    # def run(self, query: str) -> str:
    #     """Processes user query and returns a generated response efficiently."""
    #     if not self.qa_chain:
    #         return "Error: QA chain not initialized. Please create the QA chain first."

    #     try:
    #         print(f"Received Query: {query}")  # Debug line
    #         response = self.qa_chain(query)
    #         print(f"Response from Model: {response}")  # Debug line
    #         if isinstance(response, dict):
    #             return response.get("result", "No answer generated.")
    #         return response
    #     except Exception as e:
    #         print(f"Error in run(): {e}")  # Debug error
    #         return f"Error: {e}"

    def generate_visualization_code(self, query):
        """Generates Python code for a visualization based on the query."""
        prompt = f"""
        You are an expert in Python data visualization. Given the following query:
        "{query}"
        Generate a Python script using Matplotlib and Pandas that:
        - Loads or creates sample data if necessary.
        - Creates an appropriate visualization.
        - Labels the axes and adds a title.
        - Uses clear and professional formatting.
        Only return the Python code without explanation.
        """
        st.write("**Generating visualization code...**")
        response = self._llm(prompt)  # Directly pass the prompt string
        return response[0]["generated_text"] 

    def run(self, query):
        """Processes the query and determines if it's a visualization request or a QA request."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please create the QA chain first."

        try:
            if "visualize" in query.lower() or "chart" in query.lower():
                code = self.generate_visualization_code(query)
                return {"code": code}

            response = self.qa_chain(query)
            return response.get("result", "No answer generated.") if isinstance(response, dict) else response

        except Exception as e:
            return f"Error: {e}"
