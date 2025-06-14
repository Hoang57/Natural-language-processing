import os
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

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
    print("📄 Văn bản đã được chia câu:")
    return sentences

# Step 2: Preprocess sentences
def preprocess_sentences(sentences):
    processed = []
    for sentence in sentences:
        words = sentence.split()
        filtered = [word for word in words if word.lower() not in stop_words]
        processed.append(' '.join(filtered))
    print("\n🧹 After stopword removal:")
    for i, s in enumerate(processed, 1):
        print(f"Processed {i}: {s}")
    return processed

# Step 3: Compute TF-IDF vectors
def compute_tfidf_vectors(processed_sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    tf_raw = tfidf_matrix.copy()
    tf_raw.data = np.ones_like(tf_raw.data)
    tf = tf_raw.multiply(1.0 / tf_raw.sum(axis=1))

    idf = vectorizer.idf_
    idf_diag = np.diag(idf)

    tfidf_combined = tf @ idf_diag

    print("\n📌 Term Frequency (TF):")
    print(tf.toarray())
    print("\n📌 Inverse Document Frequency (IDF):")
    print(idf)
    print("\n📊 TF-IDF (kết quả TF x IDF):")
    print(tfidf_combined)

    return tfidf_combined

# Step 4: Compute cosine similarity
def compute_similarity(tfidf_matrix):
    similarity = cosine_similarity(tfidf_matrix)
    print("\n🔗 Cosine similarity matrix:")
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

    print(f"\n✅ Đã lưu ma trận kề vào file: {output_path}")
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

    print("\n🏁 Điểm PageRank của từng câu (hiển thị 5 cột):")
    for i in range(0, n, 5):
        row = ''
        for j in range(i, min(i + 5, n)):
            row += f"Câu {j+1}: {scores[j]:.4f}    "
        print(row.strip())

    return scores

# Step 7: Extract summary
def summarize(sentences, scores, ratio=0.1):
    total_sentences = len(scores)
    top_n = max(1, int(total_sentences * ratio))
    top_indices = sorted(range(total_sentences), key=lambda i: scores[i], reverse=True)[:top_n]
    top_indices.sort()

    print(f"\n📝 Summary (Top {top_n} sentences ~ {ratio:.0%}):")
    for i in top_indices:
        print(f"- {sentences[i]}")

    return [sentences[i] for i in top_indices]

# Step 8: Combine summary sentences into a paragraph
def combine_summary_sentences(summary_sentences):
    combined = ' '.join(summary_sentences)
    combined = re.sub(r'\s+([.,!?])', r'\1', combined)
    return combined

# ✅ Step 9: Evaluate summary using cosine similarity
def evaluate_summary(system_summary, reference_path, similarity_threshold=0.5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_text = f.read()

    reference_text = re.sub(r'<[^>]+>', ' ', reference_text)
    reference_text = re.sub(r'\s+', ' ', reference_text).strip()

    system_sentences = sent_tokenize(system_summary.strip())
    reference_sentences = sent_tokenize(reference_text.strip())

    if not system_sentences or not reference_sentences:
        print("⚠️ Không có câu nào trong summary hoặc reference để đánh giá.")
        return 0.0, 0.0, 0.0

    all_sentences = system_sentences + reference_sentences
    vectorizer = TfidfVectorizer().fit(all_sentences)

    sys_vectors = vectorizer.transform(system_sentences)
    ref_vectors = vectorizer.transform(reference_sentences)

    matched_ref_idx = set()
    true_positives = 0
    for i, sys_vec in enumerate(sys_vectors):
        similarities = cosine_similarity(sys_vec, ref_vectors)[0]
        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] >= similarity_threshold and best_match_idx not in matched_ref_idx:
            true_positives += 1
            matched_ref_idx.add(best_match_idx)

    precision = true_positives / len(system_sentences)
    recall = true_positives / len(reference_sentences)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"\n📊 Summary Evaluation (Cosine Similarity ≥ {similarity_threshold:.2f}):")
    print(f"🔹 Precision: {precision:.2%}")
    print(f"🔹 Recall:    {recall:.2%}")
    print(f"🔹 F1-score:  {f1:.2%}")
    
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
processed = preprocess_sentences(sentences)
tfidf = compute_tfidf_vectors(processed)
similarity = compute_similarity(tfidf)
graph = build_graph(similarity, threshold=0.25)
scores = pagerank(graph)
summary = summarize(sentences, scores, ratio=0.1)
combined_summary = combine_summary_sentences(summary)

print("\n🖋️ Complete summary:")
print(combined_summary)

# Evaluate (theo câu với cosine similarity)
evaluate_summary(combined_summary, reference_path)
