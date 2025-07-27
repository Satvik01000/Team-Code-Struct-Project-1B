import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def process_and_chunk_pdfs(pdf_docs):
    """
    Extracts text from PDFs and chunks it intelligently using a recursive splitter.
    This is a general-purpose approach.
    """
    all_chunks = []
    
    # This splitter tries to split text on paragraphs (\\n\\n), then sentences (.), then words.
    # This respects the natural structure of the document.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The maximum size of a chunk
        chunk_overlap=200, # Helps maintain context between chunks
        length_function=len
    )

    for pdf_file in pdf_docs:
        # 1. Extract all text from the PDF
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        # 2. Split the text into meaningful chunks
        chunks = text_splitter.split_text(full_text)

        # 3. Add source metadata to each chunk
        for chunk in chunks:
            all_chunks.append({
                "source_document": pdf_file.name,
                "text_content": chunk
            })
            
    return all_chunks

def get_vector_store(intelligent_chunks):
    """Creates a FAISS vector store from the chunks."""
    if not intelligent_chunks:
        return None
        
    texts = [chunk["text_content"] for chunk in intelligent_chunks]
    metadatas = intelligent_chunks  # Pass the whole dict as metadata
    
    # NOTE: You mentioned using e5-base-v2. If you use it, you need a custom class.
    # For simplicity and compliance, I am reverting to the standard HuggingFaceEmbeddings
    # with the compliant 'all-MiniLM-L6-v2' model.
    model_path = "./models/all-MiniLM-L6-v2" 
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vector_store