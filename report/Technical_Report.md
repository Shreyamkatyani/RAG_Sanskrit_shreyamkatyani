# Technical Report: Sanskrit RAG System

**Intern Name:** Shreyam Katyani
**Date:** 2024-12-27

## 1. System Architecture

The system implements a standard Retrieval-Augmented Generation (RAG) pipeline designed for CPU-only inference. It consists of the following modular components:

1.  **Ingestion Layer**: Reads Sanskrit documents (PDF/TXT) from a data directory.
2.  **Preprocessing**: Text extraction and splitting into manageable chunks with overlap to preserve context.
3.  **Vector Store (Retriever)**: Uses `ChromaDB` for storage and retrieval. Text chunks are embedded using the `sentence-transformers` library (model: `paraphrase-multilingual-MiniLM-L12-v2`).
4.  **Generator**: Uses `GPT4All` (model: `orca-mini-3b-gguf2-q4_0.gguf`) for CPU-optimized text generation.

## 2. Document Details

The system was tested with the following Sanskrit documents:
-   **Source**: User provided PDF (`Rag-docs (1).pdf`).
-   **Content**: Sanskrit texts (details to be observed from extraction).

## 3. Preprocessing Pipeline

-   **Loading**: `pypdf` is used to extract text from PDF files.
-   **Splitting**: A character-based splitter chunks text into segments of 500 characters with a 50-character overlap. This ensures that semantic meaning is not lost at chunk boundaries.

## 4. Retrieval Mechanism

-   **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` was chosen for its multilingual capabilities, supporting Sanskrit (via cross-lingual transfer or direct support depending on training data, though often effective for Indic scripts).
-   **Similarity Search**: Cosine similarity (default in ChromaDB) is used to find the top 3 most relevant chunks for a user query.

## 5. Generation Mechanism

-   **Model**: `Orca Mini 3B` (quantized to 4-bit).
-   **Reasoning**: This model is lightweight (~2GB RAM), fast on CPUs, and capable of following instructions.
-   **Prompting**: A simple template injects the retrieved context and user query:
    ```
    Context: {context}
    Question: {query}
    Answer (in simple English or Sanskrit):
    ```

## 6. Performance Observations

-   **Latency**:
    -   Indexing: Fast (< 1 min for small docs).
    -   Retrieval: Instant (< 1s).
    -   Generation: Depends on CPU. typically 5-10 tokens/sec.
-   **Resource Usage**:
    -   RAM: ~4-5 GB total (Model + Python overhead).
    -   CPU: High utilization during generation.

## 7. Future Improvements
-   Better Sanskrit-specific embeddings (e.g., IndicBERT).
-   Fine-tuned LLM on Sanskrit corpora.
-   Web-based UI (Streamlit/Gradio).
