import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, tokenizer, dataset_x, dataset_y, max_length=256):
        self.tokenizer = tokenizer
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.dataset_x[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            self.dataset_y[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }

# DocGPT Training Class
class DocGPT:
    def __init__(self, docs, model_path="google/flan-t5-base", embedding_model="BAAI/bge-large-en", epochs=5, batch_size=2):
        self.docs = docs
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device)
        self.train_model()

    def train_model(self):
        dataset_x, dataset_y = [], []
        for doc in self.docs:
            content = doc.page_content.strip()
            if content.startswith("Q:") and " A: " in content:
                question, answer = content.split(" A: ", 1)
                dataset_x.append(question)
                dataset_y.append(answer)
            else:
                dataset_x.append(content)
                dataset_y.append(content)
        if not dataset_x:
            raise ValueError("No valid data found for training!")
        dataset = TextDataset(self.tokenizer, dataset_x, dataset_y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["labels"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_train_loss/len(train_loader):.4f}")

# Load QA Data
with open("news_qa_pairs.json", "r") as f:
    qa_data = json.load(f)
docs = [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in qa_data]

# Train Model
docgpt = DocGPT(docs)
