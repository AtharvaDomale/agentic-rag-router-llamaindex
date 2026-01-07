import streamlit as st
from rag_engine import build_query_engine

st.set_page_config(page_title="RAG with LlamaIndex", layout="wide")

st.title("📄 RAG Chat with PDF")

@st.cache_resource
def load_engine():
    return build_query_engine("test.pdf")

query_engine = load_engine()

question = st.text_input("Ask a question about the document")

if question:
    with st.spinner("Thinking..."):
        response = query_engine.query(question)
        st.markdown("### Answer")
        st.write(str(response))
