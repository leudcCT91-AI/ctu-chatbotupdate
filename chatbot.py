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
pdf_text = load_pdf("tuyensinh_ctu.pdf")

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
        raise SystemExit(f"FAQ thiếu cột: {sorted(missing)}. Cần tối thiểu: question, answer")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df


def build_index(df: pd.DataFrame):
    questions = df["question"].to_list()
    vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b"
)
    faq_matrix = vectorizer.fit_transform(questions)
    return vectorizer, faq_matrix
SYNONYMS = {
    "cntt": "công nghệ thông tin",
    "it": "công nghệ thông tin",
    "qtkd": "quản trị kinh doanh",
    "ktpm": "kỹ thuật phần mềm",
    "kt": "kế toán",
    "tcnh": "tài chính ngân hàng",
    "nn anh": "ngôn ngữ anh",
}
SYNONYMS = {
    "ctu": "đại học cần thơ",
    "đh cần thơ": "đại học cần thơ",
    "đhct": "đại học cần thơ",
    "bao nhiêu điểm": "điểm chuẩn",
    "lấy bao nhiêu điểm": "điểm chuẩn",
    "điểm trúng tuyển": "điểm chuẩn",
}
def normalize_text(text):
    t = text.lower()
    for k, v in SYNONYMS.items():
        t = t.replace(k, v)
    return t

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
    user_question = normalize_text(user_question.strip())
    if not user_question:
        return "Bạn nhập câu hỏi giúp mình nhé.", []

    user_vec = vectorizer.transform([user_question])
    sims = cosine_similarity(user_vec, faq_matrix).flatten()

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    raw_top = topk_indices(sims, TOPK + 1)
    top_idx = [i for i in raw_top if i != best_idx][:TOPK]
    suggestions = [str(df.iloc[i]["question"]) for i in top_idx]
    if answer is None or similarity < 0.4:

    if message.lower() in pdf_text.lower():

        return "Thông tin trong tài liệu tuyển sinh:\n" + message, []
    if best_score < THRESHOLD:
        major = guess_major(user_question)
        s3 = (
            f"Môn đại cương của ngành {major} gồm những gì?"
            if major
            else "Môn đại cương của ngành [bạn ghi tên ngành] gồm những gì?"
        )
        fallback = (
            "Mình chưa chắc bạn đang hỏi ý nào. "
            "Bạn chọn 1 trong 3 hướng dưới đây (copy đúng câu để hỏi lại):"
        )
        suggestions = [
            "Học phí trường Đại học Cần Thơ bao nhiêu?",
            "Điểm chuẩn CTU năm 2025 bao nhiêu?",
            s3,
        ]
        return fallback, suggestions

    answer = str(df.iloc[best_idx]["answer"])
    return answer, suggestions


def main():
    df = load_faq(faq.tsv)
    vectorizer, faq_matrix = build_index(df)

    print("=" * 65)
    print("CTU FAQ Chatbot (TF-IDF + Cosine Similarity)")
    print("Gõ 'thoat' để kết thúc")
    print("=" * 65)

    while True:
        q = input("Bạn: ").strip()
        if q.lower() in ("thoat", "exit", "quit"):
            print("Chatbot: Tạm biệt!")
            break
        if not q:
            continue

        answer, suggestions = get_response(q, df, vectorizer, faq_matrix)
        print("Chatbot:", answer)
        if suggestions:
            print("Gợi ý:")
            for s in suggestions:
                print("-", s)
        print()


if __name__ == "__main__":

    main()
    def load_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    return text
