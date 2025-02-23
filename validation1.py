import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from docGPT import DocGPT
from langchain.schema import Document

# Download necessary NLTK data
nltk.download("punkt")

# Path to your evaluation dataset
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

def calculate_bleu(reference, candidate):
    """Computes BLEU score."""
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    return sentence_bleu(reference_tokens, candidate_tokens)

def calculate_rouge(reference, candidate):
    """Computes ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores["rougeL"].fmeasure

def calculate_f1(reference, candidate):
    """Computes F1-score at the word level."""
    reference_tokens = set(nltk.word_tokenize(reference.lower()))
    candidate_tokens = set(nltk.word_tokenize(candidate.lower()))

    common_tokens = reference_tokens.intersection(candidate_tokens)
    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(candidate_tokens)
    recall = len(common_tokens) / len(reference_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(doc_gpt, eval_data):
    """Evaluates the model on the dataset using BLEU, ROUGE, and F1-score."""
    if not eval_data:
        print("No evaluation data found. Exiting...")
        return

    bleu_scores, rouge_scores, f1_scores = [], [], []
    
    print("\n=== Model Evaluation ===\n")
    for idx, item in enumerate(eval_data, start=1):
        question = item.get("question", "").strip()
        expected_answer = item.get("answer", "").strip()

        if not question or not expected_answer:
            print(f"Skipping invalid entry at index {idx}")
            continue
        
        generated_answer = doc_gpt.run(question).strip()

        print(f"Q{idx}: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Generated: {generated_answer}\n")

        # Compute Metrics
        bleu = calculate_bleu(expected_answer, generated_answer)
        rouge = calculate_rouge(expected_answer, generated_answer)
        f1 = calculate_f1(expected_answer, generated_answer)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        f1_scores.append(f1)

    # Compute average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    print(f"\n=== Evaluation Complete ===")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-L Score: {avg_rouge:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

# Step 1: Load evaluation dataset
eval_data = load_eval_dataset()

# Step 2: Assign eval_data to docs correctly
docs = [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in eval_data]

if not docs:
    print("Error: No documents available for embedding. Ensure your evaluation dataset is not empty.")
    exit(1)  # Stop execution

# Step 3: Initialize and train the model
doc_gpt = DocGPT(docs)
doc_gpt.create_qa_chain()

# Step 4: Run Evaluation
evaluate_model(doc_gpt, eval_data)
