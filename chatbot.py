from pypdf import PdfReader
import sys
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FAQ_PATH = "faq.tsv"
PDF_PATH = "tuyensinh_ctu.pdf"

THRESHOLD = 0.35
TOPK = 3


# =========================
# ĐỌC FILE PDF
# =========================
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    return text


pdf_text = load_pdf(PDF_PATH)
pdf_lines = [line.strip() for line in pdf_text.split("\n") if line.strip()]


# =========================
# LOAD FAQ
# =========================
def load_faq(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df


# =========================
# BUILD INDEX FAQ
# =========================
def build_index(df):

    questions = df["question"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    faq_matrix = vectorizer.fit_transform(questions)

    return vectorizer, faq_matrix


# =========================
# TÌM TRONG PDF
# =========================
def search_pdf(question):
    q = question.lower()
    words = [w for w in q.split() if len(w) > 2]

    best_line = None
    best_score = 0

    for line in pdf_lines:
        l = line.lower()

        # bỏ dòng tiêu đề quá chung
        if len(l) < 20:
            continue

        score = 0
        for w in words:
            if w in l:
                score += 1

        # ưu tiên dòng có nhiều từ khóa hơn
        if score > best_score:
            best_score = score
            best_line = line

    if best_score >= 2:
        return best_line

    return None


# =========================
# TOP K
# =========================
def topk_indices(sims, k):

    k = min(k, sims.shape[0])

    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]

    return idx


# =========================
# GET RESPONSE
# =========================
def get_response(user_question, df, vectorizer, faq_matrix):

    user_question = user_question.strip().lower()

    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    # =====================
    # ƯU TIÊN TÌM TRONG PDF
    # =====================
    pdf_answer = search_pdf(user_question)

    if pdf_answer:
        return pdf_answer, []

    # =====================
    # NẾU KHÔNG CÓ → TÌM FAQ
    # =====================
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