import streamlit as st
from chatbot import load_faq, build_index, get_response

FAQPATH = "faq.tsv"

# load dữ liệu
df = load_faq(FAQPATH)
vectorizer, faqmatrix = build_index(df)

st.title("🎓 CTU Chatbot")

question = st.text_input("Nhập câu hỏi")

if question:

    answer, suggestions = get_response(
        question, df, vectorizer, faqmatrix
    )

    st.write("Bot:", answer)

    if suggestions:
        st.write("Gợi ý:")
        for s in suggestions:
            st.write("-", s)