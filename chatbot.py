from pypdf import PdfReader
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PDF_PATH = "tuyensinh_ctu.pdf"
FAQ_PATH = "faq.tsv"

THRESHOLD = 0.35
TOPK = 3


# =====================
# ĐỌC PDF
# =====================
def load_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        t = page.extract_text()

        if t:
            text += t + "\n"

    return text


pdf_text = load_pdf(PDF_PATH)
pdf_lines = [line.strip() for line in pdf_text.split("\n") if line.strip()]


# =====================
# LOAD FAQ
# =====================
def load_faq(path):

    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)

    return df


# =====================
# BUILD INDEX
# =====================
def build_index(df):

    questions = df["question"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        lowercase=True
    )

    faq_matrix = vectorizer.fit_transform(questions)

    return vectorizer, faq_matrix


# =====================
# TÌM TRONG PDF
# =====================
def search_pdf(question):

    words = question.lower().split()

    best_line = None
    best_score = 0

    for line in pdf_lines:

        score = 0

        for w in words:
            if w in line.lower():
                score += 1

        if score > best_score:
            best_score = score
            best_line = line

    if best_score >= 2:
        return best_line

    return None


# =====================
# TOPK
# =====================
def topk_indices(sims, k):

    k = min(k, sims.shape[0])

    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]

    return idx


# =====================
# GET RESPONSE
# =====================
def get_response(user_question, df, vectorizer, faq_matrix):

    user_question = user_question.strip().lower()

    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    # --------
    # tìm PDF trước
    # --------
    pdf_answer = search_pdf(user_question)

    if pdf_answer:

    parts = pdf_answer.split()

    if len(parts) >= 6:

        ma_nganh = parts[1]
        ten_nganh = " ".join(parts[2:-2])
        chi_tieu = parts[-2]
        to_hop = parts[-1]

        answer = f"""
Ngành: {ten_nganh}
Mã ngành: {ma_nganh}
Chỉ tiêu tuyển sinh: {chi_tieu}
Tổ hợp xét tuyển: {to_hop}
"""

        return answer, []

    return pdf_answer, []