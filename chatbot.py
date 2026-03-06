from pypdf import PdfReader
import sys
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = "faq.tsv"
THRESHOLD = 0.30
TOPK = 3


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    return text


pdf_text = load_pdf("tuyensinh_ctu.pdf")


def topk_indices(sims: np.ndarray, k: int):
    k = min(k, sims.shape[0])
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return idx


def load_faq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df


def build_index(df: pd.DataFrame):
    questions = df["question"].to_list()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    faq_matrix = vectorizer.fit_transform(questions)
    return vectorizer, faq_matrix


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

    # Nếu FAQ không đủ giống
    if best_score < THRESHOLD:

        # thử tìm trong PDF
        if user_question.lower() in pdf_text.lower():
            return "Thông tin có trong tài liệu tuyển sinh CTU.", []

        return "Mình chưa chắc bạn đang hỏi ý nào.", suggestions

    answer = str(df.iloc[best_idx]["answer"])
    return answer, suggestions