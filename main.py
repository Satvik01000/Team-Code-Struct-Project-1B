# main.py

import os
import json
from app.persona_engine import PersonaIntelligenceEngine

# These directories are fixed according to the hackathon's Docker environment
INPUT_DIRECTORY = "input"
OUTPUT_DIRECTORY = "output"

def main():
    """
    Main execution function for Challenge 1B.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # [cite_start]The official input file is always named challenge1b_input.json [cite: 1]
    input_file_path = os.path.join(INPUT_DIRECTORY, "challenge1b_input.json")
    
    if not os.path.exists(input_file_path):
        print(f"❌ Error: Input file not found at '{input_file_path}'")
        return

    # [cite_start]Load the input specification [cite: 1]
    with open(input_file_path, 'r') as f:
        spec = json.load(f)
    
    # Correctly parse the nested JSON structure
    persona = spec.get("persona", {}).get("role", "")
    job = spec.get("job_to_be_done", {}).get("task", "")
    
    # [cite_start]Extract just the filenames from the list of document objects [cite: 1]
    pdf_filenames = [doc.get("filename") for doc in spec.get("documents", []) if doc.get("filename")]
    pdf_files = [os.path.join(INPUT_DIRECTORY, fname) for fname in pdf_filenames]

    # Validate that the specified PDF files exist
    valid_pdfs = [pdf for pdf in pdf_files if os.path.exists(pdf)]
    if len(valid_pdfs) != len(pdf_files):
        missing = set(pdf_files) - set(valid_pdfs)
        print(f"⚠️ Warning: The following PDF documents were not found: {missing}")

    print(f"🚀 Starting Challenge 1B Analysis for Persona: '{persona}'")

    # Initialize and run the intelligence engine
    engine = PersonaIntelligenceEngine()
    result = engine.analyze_document_collection(valid_pdfs, persona, job)

    # The output file must be named challenge1b_output.json
    output_path = os.path.join(OUTPUT_DIRECTORY, "challenge1b_output.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"\n✅ Analysis complete. Output saved to '{output_path}'")

if __name__ == "__main__":
    main()