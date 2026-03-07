import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = "faq.tsv"
THRESHOLD = 0.30
TOPK = 3


def topk_indices(sims: np.ndarray, k: int):
    k = min(k, sims.shape[0])
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return idx


def load_faq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    required = {"question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"FAQ thiếu cột: {sorted(missing)}. Cần tối thiểu: question, answer"
        )
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df


def build_index(df: pd.DataFrame):
    questions = df["question"].to_list()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    faq_matrix = vectorizer.fit_transform(questions)
    return vectorizer, faq_matrix


def guess_major(text: str):
    t = text.lower()
    majors = [
        "công nghệ thông tin", "cntt",
        "kỹ thuật phần mềm",
        "quản trị kinh doanh", "qtkd",
        "tài chính ngân hàng", "tài chính - ngân hàng",
        "kế toán",
        "ngôn ngữ anh",
        "thú y",
        "công nghệ thực phẩm",
    ]
    for m in majors:
        if m in t:
            return m
    return None


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
