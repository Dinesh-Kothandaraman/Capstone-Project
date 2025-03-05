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

class DocGPT:
    def __init__(self, docs, model_path="google/flan-t5-base", epochs=5, batch_size=4):
        self.docs = docs  # Dataset containing stock data, news, QA pairs, and uploaded files
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _extract_data(self):
        """Extract stock data, news, QA pairs, and uploaded file content."""
        qa_pairs = []
        stock_news = []
        stock_data = []
        uploaded_files = []
        for doc in self.docs:
            content = doc.page_content
            try:
                parsed_content = json.loads(content)
                if "question" in parsed_content and "answer" in parsed_content:
                    qa_pairs.append((parsed_content["question"], parsed_content["answer"]))
                elif "News Title" in parsed_content:
                    stock_news.append(parsed_content["News Title"] + " " + parsed_content["Summary"])
                elif "Stock Date" in parsed_content:
                    stock_data.append(parsed_content["Stock Date"] + " " + parsed_content["Stock Data"])
                else:
                    uploaded_files.append(content)
            except json.JSONDecodeError:
                uploaded_files.append(content)  # Treat as raw text if not JSON
        return qa_pairs, stock_news, stock_data, uploaded_files

    class CustomDataset(Dataset):
        """Dataset class for tokenizing and preparing data."""
        def __init__(self, qa_pairs, stock_news, stock_data, uploaded_files, tokenizer, max_length=512):
            self.data = qa_pairs + stock_news + stock_data + uploaded_files
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx] if isinstance(self.data[idx], str) else self.data[idx][0]
            target = "" if isinstance(self.data[idx], str) else self.data[idx][1]
            
            encoding = self.tokenizer(text, padding="max_length", truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
            target_encoding = self.tokenizer(target, padding="max_length", truncation=True,
                                             max_length=self.max_length, return_tensors="pt")
            return {**encoding, "labels": target_encoding["input_ids"].squeeze()}

    def train_model(self):
        """Train the model and capture training & validation loss."""
        qa_pairs, stock_news, stock_data, uploaded_files = self._extract_data()
        if not qa_pairs and not stock_news and not stock_data and not uploaded_files:
            raise ValueError("No relevant data found in the dataset!")

        dataset = self.CustomDataset(qa_pairs, stock_news, stock_data, uploaded_files, self.tokenizer)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        training_losses, validation_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].squeeze().to(self.device)
                attention_mask = batch["attention_mask"].squeeze().to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            # Validation Phase
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].squeeze().to(self.device)
                    attention_mask = batch["attention_mask"].squeeze().to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        self._plot_loss(training_losses, validation_losses)

    def _plot_loss(self, training_losses, validation_losses):
        """Plot Training and Validation Loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss", marker="o")
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()
