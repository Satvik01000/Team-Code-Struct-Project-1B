import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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