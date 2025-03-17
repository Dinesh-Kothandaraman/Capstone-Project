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
import ast
import re

class DocGPT:
    def __init__(self, docs, embedding_model="BAAI/bge-large-en"):
    # def __init__(self, docs, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    
        """
        Initializes DocGPT with a given dataset.
        Supports multiple document types: stock data, news articles, QA pairs.
        """
        self.docs = docs
        self.qa_chain = None
        self.embedding_model = embedding_model
        # self._llm = pipeline("text2text-generation", model=model_name, max_new_tokens=256)
        self._llm = pipeline("text2text-generation", model="EleutherAI/gpt-neo-2.7B", max_new_tokens=256,temperature=0.7, do_sample=True)
        # self._llm = pipeline("text-generation", model="Salesforce/codegen2-1B", max_new_tokens=256)

        # self._llm =None
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

    def extract_python_code(self,response):
        """Extracts Python code from the model's response."""
        if isinstance(response, list) and len(response) > 0:
            full_text = response[0].get("generated_text", "").strip()
        elif isinstance(response, dict):
            full_text = response.get("generated_text", "").strip()
        else:
            return {"error": "Invalid response format."}

        # Try extracting code with regex
        code_match = re.search(r"```python\n(.*?)\n```", full_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Fallback: Extract lines that look like Python code
            lines = full_text.split("\n")
            code_lines = [
                line for line in lines
                if line.strip().startswith(("import", "from", "plt.", "df.", "fig", "ax", "pd."))
            ]
            code = "\n".join(code_lines).strip()

        if not code:
            return {"error": "No valid Python code found in response."}

        # Validate syntax
        try:
            ast.parse(code)  # Check for syntax errors
            return {"code": code}
        except SyntaxError as e:
            return {"error": f"Syntax Error in generated code: {e}"}

    # def extract_python_code(self, response):
    #     """Extracts and validates Python code from the model's response."""
    #     if isinstance(response, list) and response:
    #         full_text = response[0].get("generated_text", "").strip()
    #     elif isinstance(response, dict):
    #         full_text = response.get("generated_text", "").strip()
    #     else:
    #         return {"error": "Invalid response format."}

    #     # Extract Python code using regex
    #     code_match = re.search(r"```python\s*(.*?)\s*```", full_text, re.DOTALL)
    #     if code_match:
    #         code = code_match.group(1).strip()
    #     else:
    #         return {"error": "No valid Python code found in response."}

    #     # Validate extracted code
    #     try:
    #         ast.parse(code)  # Syntax check
    #         return {"code": code}
    #     except SyntaxError as e:
    #         return {"error": f"Syntax Error in generated code: {e}"}

    def execute_code(self,code_str):
        """Safely executes the extracted Python code."""
        if not isinstance(code_str, str) or not code_str.strip():
            return {"error": "No valid code provided for execution."}

        try:
            exec(code_str, globals())
            return {"success": "Code executed successfully."}
        except Exception as e:
            return {"error": f"Error executing code: {e}"}

    def generate_visualization_code(self, query):
        """Generates Python code for a visualization based on the query."""
        response = self.qa_chain(query)
        st.write("response1", response)

        # Extract relevant stock data from response
        if isinstance(response, dict) and "result" in response:
            stock_data = response["result"]
        else:
            return {"error": "Invalid response format. Expected a dictionary with a 'result' key."}

        prompt = (
            f"You are an expert in Python data visualization. "
            f"Given the following stock data:\n"
            f"'{stock_data}'\n"
            f"Generate a Python script that does the following:\n"
            f"1. Loads the data using pandas (or creates a small sample dataframe if necessary).\n"
            f"2. Uses Matplotlib to visualize the stock's open, high, low, and close prices.\n"
            f"3. Labels the axes and adds a title.\n"
            f"4. Formats the chart professionally.\n\n"
            f"**Output only the Python code** without any extra explanation. "
            f"Do not include markdown formatting or any text outside the script."
        )        # st.write("**Generating visualization code...**")

        # prompt = (
        #     f"You are an expert in Python data visualization. "
        #     f"Given the following stock data:\n"
        #     f"'{stock_data}'\n"
        #     f"Generate a Python script that:\n"
        #     f"1. Loads the data using pandas.\n"
        #     f"2. Uses Matplotlib to visualize open, high, low, and close prices.\n"
        #     f"3. Labels axes and adds a title.\n"
        #     f"4. Formats the chart professionally.\n\n"
        #     f"Strictly output only Python code inside ```python ... ``` block, with no explanations, markdown, or additional text."
        # )


        st.write("**Generating visualization code...**")

        try:
            response = self._llm(prompt)
            st.write("response", response)

            # Extract clean Python code
            return self.extract_python_code(response)

        except Exception as e:
            return {"error": f"Code generation failed: {e}"}



    # def run(self, query):
    #     """Processes the query and determines if it's a visualization request or a QA request."""
    #     if not self.qa_chain:
    #         return "Error: QA chain not initialized. Please create the QA chain first."

    #     try:
    #         if "visualize" in query.lower() or "chart" in query.lower():
    #             code = self.generate_visualization_code(query)
    #             return {"code": code}

    #         response = self.qa_chain(query)
    #         return response.get("result", "No answer generated.") if isinstance(response, dict) else response

    #     except Exception as e:
    #         return f"Error: {e}"

    def run(self, query):
        """Processes the query and determines if it's a visualization request or a QA request."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please create the QA chain first."

        try:
            if "visualize" in query.lower() or "chart" in query.lower():
                code_result = self.generate_visualization_code(query)
                
                if "error" in code_result:
                    return code_result  # Return extraction errors directly
                
                execution_result = self.execute_code(code_result["code"])
                return execution_result

            response = self.qa_chain(query)
            return response.get("result", "No answer generated.") if isinstance(response, dict) else response

        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
