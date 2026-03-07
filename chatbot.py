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
def get_response(user_question: str, df: pd.DataFrame, vectorizer, faq_matrix):
    user_question = user_question.strip()
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
        major = guess_major(user_question)

        fallback = (
            "Mình chưa chắc bạn đang hỏi ý nào. "
            "Bạn chọn 1 trong 3 hướng dưới đây:"
        )

        s3 = (
            f"Môn đại cương của ngành {major} gồm những gì?"
            if major
            else "Môn đại cương của ngành [bạn ghi tên ngành] gồm những gì?"
        )

        suggestions = [
            "Học phí trường Đại học Cần Thơ bao nhiêu?",
            "Điểm chuẩn CTU các năm gần đây",
            s3,
        ]
        return fallback, suggestions

    answer = str(df.iloc[best_idx]["answer"])
    return answer, suggestions

Ngành: {ten_nganh}
Mã ngành: {ma_nganh}
Chỉ tiêu tuyển sinh: {chi_tieu}
Tổ hợp xét tuyển: {to_hop}
"""


            return answer, []
