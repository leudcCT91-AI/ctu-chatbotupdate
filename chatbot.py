from pypdf import PdfReader
import pandas as pd
import numpy as np
import re
def normalize_text(text):
    return text.lower().strip()
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
        ngram_range=(1, 2),
        lowercase=True
    )

    faq_matrix = vectorizer.fit_transform(questions)

    return vectorizer, faq_matrix


# =====================
# TÌM NGÀNH TRONG PDF
# =====================
def search_pdf(question):

    q = question.lower()
    words = q.split()

    best_line = None
    best_score = 0

    for line in pdf_lines:

        line_lower = line.lower()

        # chỉ lấy dòng có mã ngành
        if not re.search(r"\b\d{7}\b", line_lower):
            continue

        score = 0

        for w in words:

            if w in line_lower:

                if len(w) >= 4:
                    score += 2
                else:
                    score += 1

        if score > best_score:
            best_score = score
            best_line = line

    if best_score >= 4:
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
# FORMAT PDF RESULT
# =====================
def format_pdf_answer(line):

    parts = line.split()

    stt = parts[0]
    ma_nganh = parts[1]

    chi_tieu = parts[-5]

    to_hop = parts[-4:]

    ten_nganh = " ".join(parts[2:-5])

    to_hop_text = "\n".join([f"• {t}" for t in to_hop])

    answer = f"""
**Ngành:** {ten_nganh}  
**Mã ngành:** {ma_nganh}  
**STT:** {stt}  
**Chỉ tiêu tuyển sinh:** {chi_tieu}  

**Tổ hợp xét tuyển:**

{to_hop_text}
"""

    return answer


# =====================
# GET RESPONSE
# =====================
def get_response(user_question, df, vectorizer, faq_matrix):

    user_question = normalize_text(user_question)

    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    # ===== TÌM TRONG PDF TRƯỚC =====
    pdf_line = search_pdf(user_question)

    if pdf_line:

        answer = format_pdf_answer(pdf_line)

        return answer, []

    # ===== KHÔNG CÓ TRONG PDF → TÌM FAQ =====
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

    # ===== KHÔNG CÓ TRONG PDF → TÌM FAQ =====
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