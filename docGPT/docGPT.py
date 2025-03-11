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


# saved_model = r"C:\Users\dines\Downloads\Capstone_model"
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
        self._llm = None
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

    # def create_qa_chain(self, retriever_k=5, model_path="google/flan-t5-base"):
    # # def create_qa_chain(self, retriever_k=5, model_path="facebook/bart-base"):
    #     """Sets up the RAG pipeline with an efficient retriever."""
    #     db = self._embeddings()
    #     retriever = db.as_retriever(search_kwargs={"k": retriever_k})

    #     # Load Tokenizer and Model Efficiently
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16) #, device_map="auto"

    #     # Create Optimized LLM Pipeline
    #     # pipe = pipeline(
    #     #     "text-generation", model=model, tokenizer=tokenizer,
    #     #     max_new_tokens=150, temperature=0.7, do_sample=True
    #     # )
    #     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    #     local_llm = HuggingFacePipeline(pipeline=pipe)

    #     self.qa_chain = RetrievalQA.from_chain_type(
    #         llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
    #     )

    def create_qa_chain(self, retriever_k=5, model_path=r"C:\Users\dines\Downloads\Capstone_model"):
        """Sets up the RAG pipeline using the fine-tuned model."""
        db = self._embeddings()
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})

        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
        model.config.max_length = 1024 

        # Create the pipeline for inference
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,max_new_tokens=512, temperature=0.7, do_sample=True)

        local_llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
        )

    def run(self, query: str) -> str:
        """Processes user query and returns a generated response efficiently."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please create the QA chain first."

        try:
            print(f"Received Query: {query}")  # Debug line
            response = self.qa_chain(query)
            print(f"Response from Model: {response}")  # Debug line
            if isinstance(response, dict):
                return response.get("result", "No answer generated.")
            return response
        except Exception as e:
            print(f"Error in run(): {e}")  # Debug error
            return f"Error: {e}"
    




# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import json
# from langchain.docstore.document import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline
# import streamlit as st

# class TextDataset(Dataset):
#     def __init__(self, tokenizer, dataset_x, dataset_y, max_length=512):
#         self.tokenizer = tokenizer
#         self.dataset_x = dataset_x
#         self.dataset_y = dataset_y
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.dataset_x)

#     def __getitem__(self, idx):
#         input_text = self.dataset_x[idx]
#         target_text = self.dataset_y[idx]

#         encoding = self.tokenizer(
#             input_text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         target_encoding = self.tokenizer(
#             target_text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "labels": target_encoding["input_ids"].squeeze(0),
#         }


# class DocGPT:
#     def __init__(self, docs, model_path="google/flan-t5-base", embedding_model="BAAI/bge-large-en", epochs=5, batch_size=4):
#         st.write("Initializing DocGPT...")
#         self.docs = docs  # Dataset containing stock data, news, QA pairs, and uploaded files
#         self.model_path = model_path
#         self.embedding_model = embedding_model
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)
#         self.qa_chain = None
#         self._db = None
#         self.train_model()
#         self.create_qa_chain()

#     def _preprocess_docs(self):
#         st.write("Preprocessing documents...")
#         """Optimized document chunking for retrieval."""
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#         chunked_docs = [
#             Document(page_content=chunk, metadata=doc.metadata)
#             for doc in self.docs
#             for chunk in text_splitter.split_text(doc.page_content)
#         ]
#         return chunked_docs

#     def _embeddings(self):
#         st.write
#         """Creates embeddings and FAISS vector database."""
#         if self._db:
#             return self._db
#         embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
#         chunked_docs = self._preprocess_docs()
#         if not chunked_docs:
#             raise ValueError("No documents available for embedding.")
#         self._db = FAISS.from_documents(chunked_docs, embedding=embeddings)
#         return self._db

#     def create_qa_chain(self, retriever_k=5):
#         st.write("Creating QA chain...")
#         """Sets up the RAG pipeline with an efficient retriever."""
#         db = self._embeddings()
#         retriever = db.as_retriever(search_kwargs={"k": retriever_k})
#         tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#         model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, torch_dtype=torch.float16)
#         pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
#         local_llm = HuggingFacePipeline(pipeline=pipe)
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
#         )

#     def run(self, query: str) -> str:
#         st.write("Processing user query...")
#         """Processes user query and returns a generated response."""
#         if not self.qa_chain:
#             return "Error: QA chain not initialized. Please create the QA chain first."
#         try:
#             response = self.qa_chain(query)
#             return response.get("result", "No answer generated.")
#         except Exception as e:
#             return f"Error: {e}"

    

#     def train_model(self):
#             st.write("Training the model...")

#             dataset_x = []
#             dataset_y = []

#             # Extract training pairs from the provided documents
#             for doc in self.docs:
#                 content = doc.page_content.strip()

#                 if content.startswith("Q:") and " A: " in content:
#                     question, answer = content.split(" A: ", 1)
#                     dataset_x.append(question)
#                     dataset_y.append(answer)

#                 elif "Stock Date" in content and "Stock Data" in content:
#                     date = content.split(" ")[2]  # Extract stock date
#                     question = f"What was the stock data on {date}?"
#                     dataset_x.append(question)
#                     dataset_y.append(content)

#                 elif "News Title:" in content:
#                     question = "Summarize this news article."
#                     dataset_x.append(question)
#                     dataset_y.append(content)

#                 else:
#                     dataset_x.append(content)
#                     dataset_y.append(content)

#             if not dataset_x or not dataset_y:
#                 raise ValueError("No valid data found for training!")

#             # Create dataset using the tokenizer
#             dataset = TextDataset(self.tokenizer, dataset_x, dataset_y)

#             # Split dataset into training and validation sets
#             train_size = int(0.8 * len(dataset))
#             val_size = len(dataset) - train_size
#             train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#             # Create data loaders
#             train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

#             optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
#             training_losses = []

#             # Training loop
#             for epoch in range(self.epochs):
#                 st.write(f"Epoch {epoch+1}/{self.epochs}")
#                 self.model.train()
#                 total_train_loss = 0

#                 for batch in train_loader:
#                     input_ids = batch["input_ids"].to(self.device)
#                     attention_mask = batch["attention_mask"].to(self.device)
#                     labels = batch["labels"].to(self.device)

#                     optimizer.zero_grad()
#                     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#                     loss = outputs.loss
#                     loss.backward()
#                     optimizer.step()

#                     total_train_loss += loss.item()

#                 avg_train_loss = total_train_loss / len(train_loader)
#                 training_losses.append(avg_train_loss)
#                 st.write(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}")

#             self._plot_metrics(training_losses)

#     def _plot_metrics(self, training_losses):
#         import matplotlib.pyplot as plt

#         plt.figure(figsize=(6, 4))
#         plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o", linestyle="-", label="Training Loss")
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         plt.title("Training Loss Over Time")
#         plt.legend()
#         plt.grid()
#         st.pyplot(plt)
