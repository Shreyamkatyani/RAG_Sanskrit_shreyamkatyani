# Sanskrit RAG System

A straightforward, CPU-based Retrieval-Augmented Generation (RAG) system that handles Sanskrit texts.

## Directory Structure
- `code/`: Implementation in Python.
- `data/`: Here, you are to place your Sanskrit PDF/TXT files.
- `report/`: Documentation of the techonolgy.

## Prerequisites
- Python 3.9+
- Whitedirt connection (initial model download)

## Setup

1.  Move to the `code` directory:
    ```bash
    cd code
    ```

2.  Install all the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  The models needed (Embedding & LLM) will be downloaded:
    ```bash
    python download_models.py
    ```

## Usage

1.  Check that your data (PDF/TXT) is stored in the `../data` folder.
2.  Execute the pipeline:
    ```bash
    python rag_pipeline.py
    ```
3.  The system will:
    -   Index and ingest the files.
    -   Bring up a loop for queries interactively.
4.  When the system asks for it, input your question. To get out, type `exit`.

## Example Prompts
The following are some examples of queries that you can try:

**General/English:**
*   "Please summarize the document."
*   "Can you tell me the main ideas?"

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