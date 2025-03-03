import json
from docGPT import DocGPT
from langchain.schema import Document  # Import Document class


# Path to your evaluation dataset (Update this path)
EVAL_DATASET_PATH = r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\Data\news_qa_pairs.json"

def load_eval_dataset():
    """Loads evaluation dataset containing questions and expected answers."""
    try:
        with open(EVAL_DATASET_PATH, "r") as f:
            eval_data = json.load(f)
        return eval_data
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return []

def evaluate_model(doc_gpt, eval_data):
    """Evaluates the model on the evaluation dataset."""
    if not eval_data:
        print("No evaluation data found. Exiting...")
        return

    correct = 0
    total = len(eval_data)
    
    print("\n=== Model Evaluation ===\n")
    for idx, item in enumerate(eval_data, start=1):
        question = item.get("question", "").strip()
        expected_answer = item.get("answer", "").strip()

        if not question or not expected_answer:
            print(f"Skipping invalid entry at index {idx}")
            continue
        
        generated_answer = doc_gpt.run(question).strip()

        # print(f"Q{idx}: {question}")
        # print(f"Expected: {expected_answer}")
        # print(f"Generated: {generated_answer}\n")

        if generated_answer.lower() == expected_answer.lower():
            correct += 1

    accuracy = (correct / total) * 100
    print(f"\n=== Evaluation Complete ===")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Model Accuracy: {accuracy:.2f}%")

# Step 1: Load evaluation dataset
eval_data = load_eval_dataset()
docs = [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in eval_data]

if not docs:
    print("Error: No documents available for embedding. Ensure your evaluation dataset is not empty.")
    exit(1)  # Stop execution

# Step 2: Initialize DocGPT model (Ensure you have trained it)
doc_gpt = DocGPT(docs)
doc_gpt.create_qa_chain()


# Step 3: Run Evaluation
evaluate_model(doc_gpt, eval_data)
