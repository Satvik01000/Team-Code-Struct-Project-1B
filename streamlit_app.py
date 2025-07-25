import streamlit as st
from app.processing import get_pdf_text, get_text_chunks, get_vector_store

def run_app():
    st.set_page_config(page_title="Chat with multiple PDF's", page_icon=":books:")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.header("Chat with Multiple PDFs")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(user_question)
        st.subheader("Relevant Sections Found:")
        for doc in docs:
            st.write(doc.page_content)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and then click process", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("Processing Complete!")

if __name__ == '__main__':
    run_app()