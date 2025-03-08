from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain.docstore.document import Document
import torch
import matplotlib.pyplot as plt
import io
import base64
import json

# class DocGPT:
#     def __init__(self, docs, embedding_model="BAAI/bge-large-en"):
#     # def __init__(self, docs, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    
#         """
#         Initializes DocGPT with a given dataset.
#         Supports multiple document types: stock data, news articles, QA pairs.
#         """
#         self.docs = docs
#         self.qa_chain = None
#         self.embedding_model = embedding_model
#         self._llm = None
#         self._db = None  # Store FAISS DB to avoid recomputation

#     def _preprocess_docs(self):
#         """Optimized document chunking to improve retrieval."""
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#         chunked_docs = [
#             Document(page_content=chunk, metadata=doc.metadata)
#             for doc in self.docs
#             for chunk in text_splitter.split_text(doc.page_content)
#         ]
#         return chunked_docs

#     def _embeddings(self):
#         """Creates embeddings and a FAISS vector database."""
#         if self._db:
#             return self._db  # Avoid redundant FAISS creation
        
#         embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
#         chunked_docs = self._preprocess_docs()
#         if not chunked_docs:
#             raise ValueError("No documents available for embedding. Check data sources.")
#         self._db = FAISS.from_documents(chunked_docs, embedding=embeddings)
#         return self._db

#     def create_qa_chain(self, retriever_k=5, model_path="google/flan-t5-base"):
#     # def create_qa_chain(self, retriever_k=5, model_path="facebook/bart-base"):
#         """Sets up the RAG pipeline with an efficient retriever."""
#         db = self._embeddings()
#         retriever = db.as_retriever(search_kwargs={"k": retriever_k})

#         # Load Tokenizer and Model Efficiently
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16) #, device_map="auto"

#         # Create Optimized LLM Pipeline
#         # pipe = pipeline(
#         #     "text-generation", model=model, tokenizer=tokenizer,
#         #     max_new_tokens=150, temperature=0.7, do_sample=True
#         # )
#         pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

#         local_llm = HuggingFacePipeline(pipeline=pipe)

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
#         )

#     # def create_qa_chain(self, retriever_k=5, model_path="facebook/bart-large"):
#     #     """Sets up the RAG pipeline with an efficient retriever using a Facebook model."""
#     #     db = self._embeddings()
#     #     retriever = db.as_retriever(search_kwargs={"k": retriever_k})

#     #     # Load Tokenizer and Model Efficiently
#     #     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     #     # Use correct model type based on Facebook's architecture
#     #     if "bart" in model_path:
#     #         model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
#     #         pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
#     #     elif "opt" in model_path:
#     #         model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
#     #         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
#     #     else:
#     #         raise ValueError("Unsupported Facebook model. Use a BART or OPT model.")

#     #     local_llm = HuggingFacePipeline(pipeline=pipe)

#     #     self.qa_chain = RetrievalQA.from_chain_type(
#     #         llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
#     #     )

        
#     def run(self, query: str) -> str:
#         """Processes user query and returns a generated response efficiently."""
#         if not self.qa_chain:
#             return "Error: QA chain not initialized. Please create the QA chain first."

#         try:
#             print(f"Received Query: {query}")  # Debug line
#             response = self.qa_chain(query)
#             print(f"Response from Model: {response}")  # Debug line

#             return response.get("result", "No answer generated.")
#         except Exception as e:
#             print(f"Error in run(): {e}")  # Debug error
#             return f"Error: {e}"
    




from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st

class DocGPT:
    def __init__(self, docs, model_path="google/flan-t5-base", embedding_model="BAAI/bge-large-en", epochs=5, batch_size=4):
        st.write("Initializing DocGPT...")
        self.docs = docs  # Dataset containing stock data, news, QA pairs, and uploaded files
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.qa_chain = None
        self._db = None
        self.train_model()
        self.create_qa_chain()

    def _preprocess_docs(self):
        st.write("Preprocessing documents...")
        """Optimized document chunking for retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunked_docs = [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in self.docs
            for chunk in text_splitter.split_text(doc.page_content)
        ]
        return chunked_docs

    def _embeddings(self):
        st.write
        """Creates embeddings and FAISS vector database."""
        if self._db:
            return self._db
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        chunked_docs = self._preprocess_docs()
        if not chunked_docs:
            raise ValueError("No documents available for embedding.")
        self._db = FAISS.from_documents(chunked_docs, embedding=embeddings)
        return self._db

    def create_qa_chain(self, retriever_k=5):
        st.write("Creating QA chain...")
        """Sets up the RAG pipeline with an efficient retriever."""
        db = self._embeddings()
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, torch_dtype=torch.float16)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        local_llm = HuggingFacePipeline(pipeline=pipe)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
        )

    def run(self, query: str) -> str:
        st.write("Processing user query...")
        """Processes user query and returns a generated response."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please create the QA chain first."
        try:
            response = self.qa_chain(query)
            return response.get("result", "No answer generated.")
        except Exception as e:
            return f"Error: {e}"

    # def train_model(self):
    #     st.write("Training the model...")
    #     """Train the model and capture training & validation loss."""
    #     dataset = []
    #     # st.write("docs", self.docs)
    #     for doc in self.docs:
    #         content = doc.page_content.strip()

    #         # st.write("content", content)
    #         try:
    #             parsed_content = json.loads(content)
    #             st.write("parsed_content", parsed_content)
    #             # Check if the parsed content contains "question" and "answer" keys
    #             if "question" in parsed_content and "answer" in parsed_content:
    #                 # dataset.append((parsed_content["question"], parsed_content["answer"]))
    #                 question = parsed_content.get("question", None)
    #                 answer = parsed_content.get("answer", None)
    #                 if question and answer:
    #                     dataset.append((question, answer))
    #                 else:
    #                     dataset.append((content, content)) # Use content as both input and target
    #             else:
    #                 dataset.append((content, content))  # Use content as both input and target
    #         except json.JSONDecodeError:
    #             st.write("content is not JSON, treating as raw text")
    #             dataset.append((content, content))  # Handle raw text files properly
        
    #     if not dataset:
    #         raise ValueError("No valid data found for training!")
    #     st.write("dataset", dataset)
        
    #     train_size = int(0.8 * len(dataset))
    #     val_size = len(dataset) - train_size
    #     st.write("train_size", train_size)
    #     st.write("val_size", val_size)  
    #     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #     st.write("train_dataset", train_dataset)
    #     st.write("val_dataset", val_dataset)    
    #     optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
    #     training_losses, validation_losses = [], []
    #     training_accuracies, validation_accuracies = [], []

    #     for epoch in range(self.epochs):
    #         self.model.train()
    #         total_train_loss = 0
    #         correct_train = 0
    #         total_train = 0

    #         for input_text, target_text in train_dataset:
    #             encoding = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    #             target_encoding = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    #             if encoding["input_ids"].shape[1] == 0:
    #                 continue  # Skip empty tokenized inputs
    #             input_ids = encoding["input_ids"].squeeze().to(self.device)
    #             st.write("input_ids", input_ids)
    #             attention_mask = encoding["attention_mask"].squeeze().to(self.device)
    #             labels = target_encoding["input_ids"].squeeze().to(self.device)
    #             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             st.write("outputs", outputs)
    #             loss = outputs.loss
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             total_train_loss += loss.item()
            
    #         avg_train_loss = total_train_loss / len(train_dataset)
    #         training_losses.append(avg_train_loss)
    #         print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}")
        
    #     self._plot_metrics(training_losses)

    def train_model(self):
        st.write("Training the model...")
        
        dataset = []

        for doc in self.docs:
            content = doc.page_content.strip()
            # st.write("Training Data Sample:", content[:300])  # Show first 300 chars

            # Extract input-target pairs
            if content.startswith("Q:") and " A: " in content:
                question, answer = content.split(" A: ", 1)  # Split into (Q, A)
                # st.write("question", question)
                # st.write("answer", answer)
                dataset.append((question, answer))

            # Handle stock data separately
            elif "Stock Date" in content and "Stock Data" in content:
                date = content.split(" ")[2]  # Extract stock date
                question = f"What was the stock data on {date}?"
                # st.write("question", question)
                # st.write("content", content)
                # st.write("date", date)
                dataset.append((question, content))

            # Handle news data separately
            elif "News Title:" in content:
                question = "Summarize this news article."
                # st.write("question", question)
                # st.write("content", content)
                dataset.append((question, content))  # Teach model to generate full news

            # Default case (self-supervised learning)
            else:
                dataset.append((content, content))  # Use the text itself as both input and target
                # st.write("dataset", dataset)
                # st.write("dataset", dataset)


        if not dataset:
            raise ValueError("No valid data found for training!")
        
        st.write("dataset2", dataset)

        st.write(f"Total training samples: {len(dataset)}")
        
        # Split into training and validation sets
        st.write("dataset shape",dataset[0])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        training_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for data in train_dataset:
                if len(data) != 2:  # Ensure all data is in (input, target) format
                    st.write("Skipping invalid data entry:", data)
                    continue
                
                input_text, target_text = data  # Safe unpacking

                encoding = self.tokenizer(
                    input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
                )
                target_encoding = self.tokenizer(
                    target_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
                )

                # Move tensors to the correct device
                input_ids = encoding["input_ids"].squeeze().to(self.device)
                attention_mask = encoding["attention_mask"].squeeze().to(self.device)
                labels = target_encoding["input_ids"].squeeze().to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataset)
            training_losses.append(avg_train_loss)
            st.write(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}")

        self._plot_metrics(training_losses)


    def _plot_metrics(self, training_losses):
        """Plot Training Loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(training_losses, label="Training Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

