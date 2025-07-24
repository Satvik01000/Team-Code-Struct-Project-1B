import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import fitz


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vector_store(text_chunks):
    model_path = "./models/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def main():
    load_dotenv()
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
    main()