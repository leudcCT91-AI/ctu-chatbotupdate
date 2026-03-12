import streamlit as st
from chatbot import load_faq, build_index, get_response

st.set_page_config(page_title="CTU Chatbot Assistant")
st.title("🎓 CTU Chatbot Assistant")

df = load_faq("faq.tsv")
vectorizer, faq_matrix = build_index(df)

question = st.text_input("Bạn cần hỗ trợ vấn đề gì?")

if question:
    answer, suggestions = get_response(question, df, vectorizer, faq_matrix)
    st.write(answer)

    if suggestions:
        st.subheader("Gợi ý")
        for s in suggestions:

            st.write("- " + s)
