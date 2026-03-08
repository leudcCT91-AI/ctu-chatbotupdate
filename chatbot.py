from pypdf import PdfReader
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PDF_PATH = "tuyensinh_ctu-3.pdf"
FAQ_PATH = "faq-2.tsv"

THRESHOLD = 0.35
TOPK = 3
MAJOR_THRESHOLD = 0.45


def normalize_text(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def extract_major_rows(pdf_text):
    lines = [re.sub(r"\s+", " ", x.strip()) for x in pdf_text.splitlines() if x.strip()]
    rows = []
    i = 0

    while i < len(lines):
        line = lines[i]

        m = re.match(r"^(\d+)\s+([0-9A-Z]{7,8})\s+(.+)$", line)
        if not m:
            i += 1
            continue

        stt = m.group(1)
        ma = m.group(2)
        rest = m.group(3)

        block = [rest]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if re.match(r"^\d+\s+[0-9A-Z]{7,8}\s+", nxt):
                break
            if nxt.startswith(("A00", "A01", "A02", "B00", "B03", "B08", "C00", "C01", "C02", "C04", "C14", "C19", "C20", "D01", "D03", "D07", "D09", "D14", "D15", "D24", "D29", "D44", "D64", "D66", "M01", "M05", "M06", "M11", "T00", "T01", "T06", "T10", "V00", "V01", "V02", "V03", "TH1", "TH2", "TH3", "TH4", "TH5", "TH7")):
                block.append(nxt)
                j += 1
                break
            block.append(nxt)
            j += 1

        full = " ".join(block)
        full = re.sub(r"\(\*\*\)", "", full).strip()

        mh = re.search(r"(.+?)\s+(\d+)\s+((?:[A-Z]{1,2}\d{2}(?:,\s*)?)+.*)$", full)
        if mh:
            ten_nganh = mh.group(1).strip(" -")
            chi_tieu = mh.group(2).strip()
            to_hop = mh.group(3).strip()
            rows.append({
                "stt": stt,
                "ma_nganh": ma,
                "ten_nganh": ten_nganh,
                "chi_tieu": chi_tieu,
                "to_hop": to_hop,
                "ten_nganh_norm": normalize_text(ten_nganh)
            })

        i = j

    return pd.DataFrame(rows)


def load_faq(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["question_norm"] = df["question"].map(normalize_text)
    return df


def build_faq_index(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False)
    faq_matrix = vectorizer.fit_transform(df["question_norm"])
    return vectorizer, faq_matrix


def build_major_index(major_df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False)
    major_matrix = vectorizer.fit_transform(major_df["ten_nganh_norm"])
    return vectorizer, major_matrix


def topk_indices(sims, k):
    k = min(k, sims.shape[0])
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return idx


def is_major_query(q):
    qn = normalize_text(q)
    keywords = [
        "ma nganh", "ma xet tuyen", "to hop", "to hop xet tuyen",
        "chi tieu", "nganh", "co nganh", "xet tuyen"
    ]
    return any(k in qn for k in keywords)


def answer_major_query(user_question, major_df, major_vectorizer, major_matrix):
    qn = normalize_text(user_question)
    qv = major_vectorizer.transform([qn])
    sims = cosine_similarity(qv, major_matrix).flatten()

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < MAJOR_THRESHOLD:
        return None

    row = major_df.iloc[best_idx]
    return (
        f"Ngành {row['ten_nganh']} có mã tuyển sinh {row['ma_nganh']}, "
        f"chỉ tiêu dự kiến {row['chi_tieu']}, "
        f"tổ hợp xét tuyển gồm {row['to_hop']}."
    )


def get_response(user_question, faq_df, faq_vectorizer, faq_matrix, major_df, major_vectorizer, major_matrix):
    user_question = normalize_text(user_question)

    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    if "diem chuan" in user_question:
        return "Mình chưa nên trả lời điểm chuẩn từ file này vì đây là tài liệu thông tin tuyển sinh, không phải bảng điểm chuẩn. Bạn cần nạp thêm file điểm chuẩn 2025 riêng để mình trả chính xác.", []

    if is_major_query(user_question):
        major_answer = answer_major_query(user_question, major_df, major_vectorizer, major_matrix)
        if major_answer:
            return major_answer, []

    user_vec = faq_vectorizer.transform([user_question])
    sims = cosine_similarity(user_vec, faq_matrix).flatten()

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    raw_top = topk_indices(sims, TOPK + 1)
    top_idx = [i for i in raw_top if i != best_idx][:TOPK]
    suggestions = [str(faq_df.iloc[i]["question"]) for i in top_idx]

    if best_score < THRESHOLD:
        return "Mình chưa chắc bạn đang hỏi ý nào.", suggestions

    return str(faq_df.iloc[best_idx]["answer"]), suggestions
