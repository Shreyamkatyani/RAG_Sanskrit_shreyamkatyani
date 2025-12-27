# Sanskrit RAG System

A simple, CPU-based Retrieval-Augmented Generation (RAG) system for Sanskrit documents.

## Directory Structure
- `code/`: Python implementation.
- `data/`: Place your Sanskrit PDF/TXT files here.
- `report/`: Technical documentation.

## Prerequisites
- Python 3.9+
- Internet connection (for initial model download)

## Setup

1.  Navigate to the `code` directory:
    ```bash
    cd code
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the required models (Embedding & LLM):
    ```bash
    python download_models.py
    ```

## Usage

1.  Ensure your data (PDF/TXT) is in the `../data` folder.
2.  Run the pipeline:
    ```bash
    python rag_pipeline.py
    ```
3.  The system will:
    -   Ingest and index the documents.
    -   Start an interactive query loop.
4.  Enter your query when prompted. Type `exit` to quit.

## Example Prompts
You can try the following queries:

**General/English:**
*   "Summarize the document."
*   "What are the key concepts discussed?"

**Sanskrit (Transliterated):**
*   "Kim asti Yoga?" (What is Yoga?)
*   "Satyameva Jayate" (Truth alone triumphs)

**Sanskrit (Devanagari):**
*   "योगः कर्मसु कौशलम्"
*   "कः अर्थः धर्मस्य?"

## Components
-   **LLM**: GPT4All (Orca Mini 3B)
-   **Vector DB**: ChromaDB
-   **Embeddings**: Sentence-Transformers (Multilingual MiniLM)
