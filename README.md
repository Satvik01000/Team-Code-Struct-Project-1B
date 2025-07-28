# Persona-Driven Document Intelligence Solution
## Overview

This project is a solution for Challenge 1B of the Adobe India Hackathon. It's an intelligent system that analyzes a collection of PDF documents to extract and rank the most relevant sections based on a given user persona and a specific "job-to-be-done."

The solution leverages a combination of document structure analysis and modern natural language processing (NLP) techniques to understand the content semantically. It operates entirely offline within a Docker container.

## How It Works

The system processes the document collection through a sophisticated pipeline:

1.  **Structure & Segment**: The process begins by analyzing every PDF in the collection to identify its structure. Using the same heading-detection logic from Challenge 1A, it intelligently segments each document into distinct sections (e.g., the content between "Chapter 1" and "Chapter 2").

2.  **Semantic Understanding**: A pre-trained NLP model from the `sentence-transformers` library (`all-MiniLM-L6-v2`) is used to convert text into numerical representations called **embeddings**. It creates embeddings for:
    * The user's query (a combination of the persona and job description).
    * Every document section extracted in the previous step.

3.  **Relevance Scoring**: The system calculates the **cosine similarity** between the user's query embedding and each section's embedding. This produces a relevance score that measures how semantically related a section is to the user's goal. Sections are then ranked by this score.

4.  **Rule-Based Refinement**: To improve accuracy, a layer of rule-based logic is applied on top of the semantic ranking:
    * **Filtering**: It filters out sections that contradict the query (e.g., removing recipes with meat for a "vegetarian" persona).
    * **Re-ranking**: It boosts the scores of sections containing keywords that are highly relevant to the query, refining the final order.

5.  **Output Generation**: The final output is a single JSON file containing metadata about the request, a ranked list of the most important sections from across all documents, and a brief text analysis of the top-ranked sections.

### Code Structure

-   **`main.py`**: The entry point. It reads the `challenge1b_input.json` file, parses the persona, job, and document list, and orchestrates the analysis process.
-   **`app/persona_engine.py`**: The core of the 1B solution. It manages the entire analysis pipeline, from sectioning documents to semantic scoring, filtering, and generating the final ranked list.
-   **`app/structure_analyzer.py`**: A reusable component from Challenge 1A that provides the foundational heading detection and document structuring capabilities.
-   **`app/document_processor.py`**: The low-level PDF reader that extracts text and style information using `PyMuPDF`.
-   **`models/`**: This directory contains the pre-trained `all-MiniLM-L6-v2` sentence-transformer model, allowing the container to run without internet access.

## Key Libraries & Models

-   **`sentence-transformers` & `torch`**: The core of the NLP engine, used for creating text embeddings and calculating semantic similarity.
-   **Model**: `all-MiniLM-L6-v2`, a fast and effective model for generating sentence and paragraph embeddings.
-   **`PyMuPDF`**: Used for the initial PDF parsing.

## How to Build and Run

The solution is containerized using a multi-stage `Dockerfile` for a smaller final image size. The `.dockerignore` file ensures that local data and build artifacts are not included in the container.

### 1. Build the Docker Image

From the project's root directory, run the build command:

```bash
docker build --platform linux/amd64 -t persona-analyzer .
````

### 2\. Run the Container

This solution expects an `input` directory containing all the necessary PDF files and a single `challenge1b_input.json` file that specifies the persona, job, and documents to analyze.

```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none persona-analyzer
```

The container will read the specification from `/app/input/challenge1b_input.json`, process the listed PDFs, and write a single `challenge1b_output.json` file to the `/app/output` directory.