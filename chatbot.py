from pypdf import PdfReader
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PDF_PATH = "tuyensinh_ctu.pdf"
FAQ_PATH = "faq.tsv"

THRESHOLD = 0.35
TOPK = 3
SYNONYMS = {
    "ctu": "đại học cần thơ",
    "đhct": "đại học cần thơ",
    "đh cần thơ": "đại học cần thơ",

    "bao nhiêu tiền": "học phí",
    "đóng tiền": "học phí",
    "đóng học phí": "học phí",

    "xin chậm đóng": "trễ hạn học phí",
    "gia hạn học phí": "trễ hạn học phí",

    "lấy bao nhiêu điểm": "điểm chuẩn",
    "điểm trúng tuyển": "điểm chuẩn"
}

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
import re

def search_pdf(question):

    words = question.lower().split()

    best_line = None
    best_score = 0

    for line in pdf_lines:

        
        if not re.search(r"\b\d{7}\b", line):
            continue

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
def normalize_text(text):

    t = text.lower()

    for k, v in SYNONYMS.items():
        t = t.replace(k, v)

    return t
def get_response(user_question, df, vectorizer, faq_matrix):

    user_question = normalize_text(user_question.strip())

    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    # ===== tìm trong PDF trước =====
    pdf_answer = search_pdf(user_question)

    if pdf_answer:

        parts = pdf_answer.split()

        if len(parts) >= 6:

            ma_nganh = parts[0]
            chi_tieu = parts[-5]

            ten_nganh = " ".join(parts[1:-5])

            # 4 tổ hợp xét tuyển
            to_hop = parts[-4:]

            to_hop_text = "\n".join([f"- {t}" for t in to_hop])

            answer = f"""
Ngành: {ten_nganh}
Mã ngành: {ma_nganh}
Chỉ tiêu tuyển sinh: {chi_tieu}

Tổ hợp xét tuyển:
{to_hop_text}
"""

            return answer, []


    # ===== nếu PDF không có thì tìm FAQ =====
    user_vec = vectorizer.transform([user_question])
    sims = cosine_similarity(user_vec, faq_matrix).flatten()

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    idx = topk_indices(sims, TOPK)
    suggestions = [str(df.iloc[i]["question"]) for i in idx]

    if best_score < THRESHOLD:
        return "Mình chưa chắc bạn đang hỏi ý nào.", suggestions

    answer = str(df.iloc[best_idx]["answer"])

    return answer, suggestions