import os
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ========================
# Step 1: Read and clean text
def read_and_clean_text(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    clean_text = re.sub(r'<[^>]+>', ' ', content)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    print("\n Văn bản đã được chia câu:")
    return sentences

# Step 3: Sử dụng Sentence-BERT thay vì TF-IDF
def compute_sbert_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    print("\n🔍 Đang tải mô hình SBERT và tính embedding...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    print("Đã tạo xong embedding cho từng câu.")
    return embeddings

# Step 4: Compute cosine similarity
def compute_similarity(embeddings):
    similarity = cosine_similarity(embeddings)
    print("\n🔗 Cosine similarity matrix from SBERT:")
    print(similarity)
    return similarity

# Step 5: Build adjacency matrix
def build_graph(similarity_matrix, threshold=0.25, output_path='adjacency_matrix.txt'):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i, j] >= threshold:
                adjacency_matrix[i, j] = 1

    print(f"\n🕸️ Adjacency matrix (threshold = {threshold}):")
    print(adjacency_matrix)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Adjacency matrix (threshold = {threshold}):\n")
        for row in adjacency_matrix:
            f.write(' '.join(map(str, row)) + '\n')

    print(f"\n Đã lưu ma trận kề vào file: {output_path}")
    return adjacency_matrix

# Step 6: PageRank algorithm
def pagerank(similarity_matrix, damping=0.85, max_iter=100, tol=1e-6):
    n = similarity_matrix.shape[0]
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    stochastic_matrix = similarity_matrix / np.where(row_sums != 0, row_sums, 1)

    scores = np.ones(n) / n

    for _ in range(max_iter):
        prev_scores = np.copy(scores)
        scores = (1 - damping) / n + damping * stochastic_matrix.T @ prev_scores
        if np.linalg.norm(scores - prev_scores, ord=1) < tol:
            break

    print("\n Điểm PageRank của từng câu:")
    for i, score in enumerate(scores):
        print(f"Câu {i+1}: {score:.4f}")

    return scores

# Step 7: Extract summary
def summarize(sentences, scores, ratio=0.1):
    total_sentences = len(scores)
    top_n = max(1, int(total_sentences * ratio))
    top_indices = sorted(range(total_sentences), key=lambda i: scores[i], reverse=True)[:top_n]
    top_indices.sort()
    print(f"\n Summary (Top {top_n} sentences ~ {ratio:.0%}):")
    for i in top_indices:
        print(f"- {sentences[i]}")
    return [sentences[i] for i in top_indices]

# Step 8: Combine summary sentences into a paragraph
def combine_summary_sentences(summary_sentences):
    combined = ' '.join(summary_sentences)
    combined = re.sub(r'\s+([.,!?])', r'\1', combined)
    return combined

# Step 9: Evaluate summary by sentence
def evaluate_summary(system_summary, reference_path):
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_text = f.read()
    reference_text = re.sub(r'<[^>]+>', ' ', reference_text)
    reference_text = re.sub(r'\s+', ' ', reference_text).strip()

    system_sentences = sent_tokenize(system_summary.strip())
    reference_sentences = sent_tokenize(reference_text.strip())

    def normalize(sentences):
        return set(re.sub(r'\s+', ' ', s.lower().strip()) for s in sentences)

    system_set = normalize(system_sentences)
    reference_set = normalize(reference_sentences)

    true_positives = len(system_set & reference_set)
    precision = true_positives / len(system_set) if system_set else 0.0
    recall = true_positives / len(reference_set) if reference_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print("\n Summary evaluation (by sentence):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    return precision, recall, f1

# ========================
# RUN FULL PIPELINE
# ========================
file_name = input("Enter the file name (without extension): ").strip()
if not file_name:
    raise ValueError("File name cannot be empty.")
file_path = os.path.abspath(f"DUC_TEXT/train/{file_name}")
reference_path = os.path.abspath(f"DUC_SUM/{file_name}")

sentences = read_and_clean_text(file_path)
embeddings = compute_sbert_embeddings(sentences)
similarity = compute_similarity(embeddings)
graph = build_graph(similarity, threshold=0.25)
scores = pagerank(graph)
summary = summarize(sentences, scores, ratio=0.1)
combined_summary = combine_summary_sentences(summary)

print("\n Complete summary:")
print(combined_summary)

# Evaluate summary
evaluate_summary(combined_summary, reference_path)