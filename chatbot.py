import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = "faq.tsv"
THRESHOLD = 0.35
TOPK = 3

def normalize_text(text):
    return str(text).lower().strip()

def load_faq(path=FAQ_PATH):
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df

def build_index(df):
    questions = df["question"].tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    faq_matrix = vectorizer.fit_transform(questions)
    return vectorizer, faq_matrix

def topk_indices(sims, k):
    k = min(k, sims.shape[0])
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return idx

def get_response(user_question, df, vectorizer, faq_matrix):
    user_question = normalize_text(user_question)

    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    user_vec = vectorizer.transform([user_question])
    sims = cosine_similarity(user_vec, faq_matrix).flatten()

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    raw_top = topk_indices(sims, TOPK + 1)
    top_idx = [i for i in raw_top if i != best_idx][:TOPK]
    suggestions = [str(df.iloc[i]["question"]) for i in top_idx]

    if best_score < THRESHOLD:
        return "Mình chưa chắc bạn đang hỏi ý nào.", suggestions

    answer = str(df.iloc[best_idx]["answer"])
    return answer, suggestions
