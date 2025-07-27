import streamlit as st
from app.processing import process_and_chunk_pdfs, get_vector_store

def run_app():
    st.set_page_config(page_title="Document Intelligence", page_icon=":books:")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.header("Ask Questions About Your Documents")
    user_question = st.text_input("Enter your query:")

    if user_question and st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(user_question, k=4)
        st.subheader("Relevant Information Found:")
        for doc in docs:
            # We display the source and the content of the relevant chunk
            st.write(f"**Source: {doc.metadata['source_document']}**")
            st.info(doc.page_content) # Use st.info for better formatting
            st.write("---")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and then click process", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Call the new, correct pipeline
                    intelligent_chunks = process_and_chunk_pdfs(pdf_docs)
                    
                    st.session_state.vector_store = get_vector_store(intelligent_chunks)
                    
                    st.success("Processing Complete!")

if __name__ == '__main__':
    run_app()