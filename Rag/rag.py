# import json
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments
# from datasets import load_dataset

# # Step 1: Load and Process Q&A Dataset
# def load_qa_dataset(json_file):
#     with open(json_file, "r") as f:
#         data = json.load(f)
#     return data

# dataset = load_qa_dataset(r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\qa_dataset.json")

# # Step 2: Generate Embeddings for Retrieval
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# contexts = [item["context"] for item in dataset]
# embeddings = embed_model.encode(contexts, convert_to_numpy=True)

# # Store in FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)
# faiss.write_index(index, "faiss_index.bin")
# print("Embeddings stored successfully!")

# # Step 3: Fine-Tune RAG Model
# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", use_dummy_dataset=True)
# retriever.index.deserialize_from("faiss_index.bin")
# model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# # Convert dataset to Hugging Face format
# def convert_to_hf_format(data):
#     return {"question": [item["question"] for item in data], "answer": [item["answer"] for item in data]}

# dataset_hf = convert_to_hf_format(dataset)

# dataset_hf = load_dataset("json", data_files=r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\qa_dataset.json")

# training_args = TrainingArguments(
#     output_dir="./rag_finetuned",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     save_steps=500,
#     evaluation_strategy="epoch",
#     logging_dir="./logs"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_hf["train"],
#     tokenizer=tokenizer
# )

# trainer.train()
# print("RAG model fine-tuning complete!")


# import json
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sentence_transformers import SentenceTransformer
# from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments

# # Step 1: Load and Process Q&A Dataset
# def load_qa_dataset(json_file):
#     with open(json_file, "r") as f:
#         data = json.load(f)
#     return data

# dataset = load_qa_dataset(r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\qa_dataset.json")

# # Step 2: Create a Custom Dataset Class for PyTorch
# class QADataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         question = item["question"]
#         answer = item["answer"]

#         # Tokenize the inputs and outputs
#         inputs = self.tokenizer(question, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
#         labels = self.tokenizer(answer, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

#         # Return the input IDs and labels
#         return {
#             "input_ids": inputs["input_ids"].squeeze(),
#             "attention_mask": inputs["attention_mask"].squeeze(),
#             "labels": labels["input_ids"].squeeze()
#         }

# # Step 3: Fine-Tune RAG Model with Custom Dataset
# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", use_dummy_dataset=True)

# # Step 4: Generate Embeddings for FAISS Index
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# contexts = [item["context"] for item in dataset]
# embeddings = embed_model.encode(contexts, convert_to_numpy=True)

# # Store in FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)
# faiss.write_index(index, "faiss_index.bin")
# print("Embeddings stored successfully!")

# # Initialize model
# model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# # Step 5: Convert dataset to PyTorch format
# train_dataset = QADataset(dataset, tokenizer)

# # Create DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# # Set up training arguments
# training_args = TrainingArguments(
#     output_dir="./rag_finetuned",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     save_steps=500,
#     evaluation_strategy="epoch",
#     logging_dir="./logs"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer
# )

# trainer.train()
# print("RAG model fine-tuning complete!")


import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import RagTokenizer, RagSequenceForGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import default_data_collator

# Step 1: Load and Process Q&A Dataset
def load_qa_dataset(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

dataset = load_qa_dataset(r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\qa_dataset.json")

# Step 2: Generate Embeddings for Retrieval
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
contexts = [item["context"] for item in dataset]
embeddings = embed_model.encode(contexts, convert_to_numpy=True)

# Store in FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")
print("Embeddings stored successfully!")

# Step 3: Fine-Tune RAG Model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Manually handle retrieval with FAISS
class CustomRetriever:
    def __init__(self, index, embed_model):
        self.index = index
        self.embed_model = embed_model

    def retrieve(self, question, top_k=1):
        query_embedding = self.embed_model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, top_k)  # D is distances, I is indices
        return [contexts[i] for i in I[0]]

retriever = CustomRetriever(index, embed_model)

# Step 4: Prepare Dataset for Training
class QADataset(Dataset):
    def __init__(self, dataset, tokenizer, retriever):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.retriever = retriever

    def __getitem__(self, idx):
        question = self.dataset[idx]["question"]
        answer = self.dataset[idx]["answer"]
        context = self.retriever.retrieve(question)[0]  # Get context for the question

        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        labels = self.tokenizer(answer, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids
        
        return {
            "input_ids": inputs.input_ids.flatten(),
            "attention_mask": inputs.attention_mask.flatten(),
            "labels": labels.flatten(),
        }

    def __len__(self):
        return len(self.dataset)

# Use default collator to automatically pad sequences
train_dataset = QADataset(dataset, tokenizer, retriever)

train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=default_data_collator)

# Convert dataset into PyTorch DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=2)

# Training configuration
training_args = TrainingArguments(
    output_dir="./rag_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    evaluation_strategy="no",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
print("RAG model fine-tuning complete!")
