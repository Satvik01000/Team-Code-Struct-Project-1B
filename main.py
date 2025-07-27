# main_1b.py

import os
import json
from app.persona_engine import PersonaIntelligenceEngine

INPUT_DIRECTORY = "input_1b" # Use a separate input directory for 1B
OUTPUT_DIRECTORY = "output"

def run():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # For this challenge, we read a single persona file
    persona_file_path = os.path.join(INPUT_DIRECTORY, "persona.json")
    if not os.path.exists(persona_file_path):
        print(f"❌ Error: Input file not found at {persona_file_path}")
        return

    with open(persona_file_path, 'r') as f:
        spec = json.load(f)
    
    persona = spec.get("persona")
    job = spec.get("job_to_be_done")
    pdf_files = [os.path.join(INPUT_DIRECTORY, doc) for doc in spec.get("documents", [])]

    print(f"🚀 Starting Challenge 1B Analysis for Persona: '{persona}'")

    # Initialize and run the engine
    engine = PersonaIntelligenceEngine()
    result = engine.analyze_document_collection(pdf_files, persona, job)

    # Save the output
    output_path = os.path.join(OUTPUT_DIRECTORY, "challenge1b_output.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"\n✅ Analysis complete. Output saved to {output_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run()