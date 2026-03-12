from pypdf import PdfReader
import pandas as pd
import numpy as np
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

PDF_PATH = "tuyensinh_ctu.pdf"
FAQ_PATH = "faq.tsv"

THRESHOLD = 0.22
FUZZY_THRESHOLD = 0.62
TOPK = 3


def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()

    replacements = {
        "hay không": "khong",
        "được không": "khong",
        "đc không": "khong",
        "được ko": "khong",
        "duoc khong": "khong",
        "ko": "khong",
        "k ": " khong ",
        "hông": "khong",
        "hok": "khong",
        "sv": "sinh vien",
        "ctu": "can tho",
        "đhct": "can tho",
        "đại học cần thơ": "can tho",
        "hoc bong": "học bổng",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_no_accent(text):
    return remove_accents(normalize_text(text))


def token_set(text):
    stopwords = {
        "em", "mình", "minh", "cho", "với", "voi", "ạ", "a", "ah", "à",
        "la", "là", "thi", "thì", "co", "có", "duoc", "được", "khong", "không",
        "hay", "nao", "nào", "gi", "gì", "ve", "về", "toi", "tôi", "ban", "bạn"
    }
    toks = normalize_no_accent(text).split()
    return [t for t in toks if t not in stopwords and len(t) > 1]


def overlap_score(q1, q2):
    s1 = set(token_set(q1))
    s2 = set(token_set(q2))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / max(1, len(s1))


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, normalize_no_accent(a), normalize_no_accent(b)).ratio()


def normalize_for_vector(text):
    base = normalize_text(text)
    no_accent = remove_accents(base)
    return f"{base} {no_accent}"


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
    df["group"] = df["group"].astype(str)
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)

    df["question_norm"] = df["question"].apply(normalize_text)
    df["question_norm_no_accent"] = df["question"].apply(normalize_no_accent)

    df["search_text"] = (
        df["group"].fillna("") + " " +
        df["question"].fillna("") + " " +
        df["answer"].fillna("")
    ).apply(normalize_for_vector)

    return df


# =====================
# BUILD INDEX
# =====================
def build_index(df):
    texts = df["search_text"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        lowercase=False,
        min_df=1,
        sublinear_tf=True
    )

    faq_matrix = vectorizer.fit_transform(texts)
    return vectorizer, faq_matrix


# =====================
# TÌM NGÀNH TRONG PDF
# =====================
def search_pdf(question):
    q = normalize_no_accent(question)
    words = [w for w in q.split() if len(w) >= 2]

    best_line = None
    best_score = 0

    for line in pdf_lines:
        line_lower = normalize_no_accent(line)

        if not re.search(r"\b\d{7}\b", line_lower):
            continue

        score = 0
        for w in words:
            if w in line_lower:
                score += 2 if len(w) >= 4 else 1

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
    ma_nganh = parts[1]
    chi_tieu = parts[-5]
    to_hop = parts[-4:]
    ten_nganh = " ".join(parts[2:-5])

    to_hop_text = "\n".join([f"• {t}" for t in to_hop])

    answer = f"""**Ngành:** {ten_nganh}\n
**Mã ngành:** {ma_nganh}\n
**Chỉ tiêu tuyển sinh:** {chi_tieu}\n
**Tổ hợp xét tuyển:**\n
{to_hop_text}"""

    return answer


# =====================
# RERANK FAQ
# =====================
def rerank_scores(user_question, df, sims):
    scores = []

    for i, base_sim in enumerate(sims):
        faq_q = df.iloc[i]["question"]

        fuzzy = fuzzy_ratio(user_question, faq_q)
        overlap = overlap_score(user_question, faq_q)

        final_score = 0.55 * float(base_sim) + 0.30 * fuzzy + 0.15 * overlap
        scores.append(final_score)

    return np.array(scores)


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

    # ===== TÌM FAQ =====
    user_vec = vectorizer.transform([normalize_for_vector(user_question)])
    sims = cosine_similarity(user_vec, faq_matrix).flatten()

    final_scores = rerank_scores(user_question, df, sims)

    best_idx = int(np.argmax(final_scores))
    best_score = float(final_scores[best_idx])

    raw_top = topk_indices(final_scores, TOPK + 1)
    top_idx = [i for i in raw_top if i != best_idx][:TOPK]

    suggestions = [str(df.iloc[i]["question"]) for i in top_idx]

    best_fuzzy = fuzzy_ratio(user_question, df.iloc[best_idx]["question"])

    if best_score < THRESHOLD and best_fuzzy < FUZZY_THRESHOLD:
        return "Mình chưa chắc bạn đang hỏi ý nào.", suggestions

    answer = str(df.iloc[best_idx]["answer"])
    return answer, suggestions

