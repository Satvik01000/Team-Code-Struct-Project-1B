import fitz
import os

def parse_pdfs_to_chunks(pdf_directory: str) -> list[dict]:
    all_text_chunks = []
    print(f"Scanning for PDF files in: {pdf_directory}")

    if not os.path.isdir(pdf_directory):
        print(f"Error: Directory not found at {pdf_directory}")
        return []

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the directory.")
        return []

    print(f"Found {len(pdf_files)} PDF(s): {', '.join(pdf_files)}")

    for filename in pdf_files:
        doc_path = os.path.join(pdf_directory, filename)
        
        try:
            doc = fitz.open(doc_path)
            print(f"Processing {filename}...")

            for page_num, page in enumerate(doc):
                # Using get_text("dict") is a robust way to get structured text data
                page_blocks = page.get_text("dict").get("blocks", [])
                
                for block in page_blocks:
                    # We only care about text blocks (type 0)
                    if block.get("type") == 0 and "lines" in block:
                        
                        # Reconstruct the full text of the paragraph/block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                        
                        block_text = block_text.strip()

                        # A simple filter to avoid noise (e.g., page numbers, single words)
                        # We consider a chunk meaningful if it has at least a few words.
                        if len(block_text.split()) > 5:
                            chunk = {
                                "source_document": filename,
                                "page_number": page_num + 1,
                                "text_content": block_text
                            }
                            all_text_chunks.append(chunk)
            
            doc.close()
            print(f"Finished processing {filename}. Found {len(all_text_chunks)} chunks so far.")

        except Exception as e:
            print(f"Could not process file {filename}. Reason: {e}")

    print(f"\nTotal text chunks extracted from all documents: {len(all_text_chunks)}")
    return all_text_chunks