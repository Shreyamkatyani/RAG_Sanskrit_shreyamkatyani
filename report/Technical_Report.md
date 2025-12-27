# Technical Report: Sanskrit RAG System

**Intern Name:** Shreyam Katyani  
**Date:** 2024-12-27  

## 1. System Architecture

The architecture of the system follows a classic Retrieval-Augmented Generation (RAG) pipeline meant for CPU-only inference. The modular components of the system are:

1. **Ingestion Layer**: From a data directory, Sanskrit documents (PDF/TXT) are read.
2. **Preprocessing**: The text is extracted and divided into smaller sections of overlapping sizes so that the context can be retained.
3. **Vector Store (Retriever)**: Retrieval and storage is done through `ChromaDB`. The text chunks are transformed into vectors via the `sentence-transformers` library (model: `paraphrase-multilingual-MiniLM-L12-v2`).
4. **Generator**: The `GPT4All` (model: `orca-mini-3b-gguf2-q4_0.gguf`) is chosen for the text generation that is CPU-optimized.

## 2. Document Details

The system has been used with the following Sanskrit documents for testing:
- **Source**: The PDF provided by the user (`Rag-docs (1).pdf`).
- **Content**: Sanskrit texts (details will be seen from extraction).

## 3. Preprocessing Pipeline

- **Loading**: PDF files are textedextract and `pypdf` is used to get their content.
- **Splitting**: A character-based splitter segments text into parts of 500 characters each with a 50-character overlap. This way, the text at the boundary will not lose its semantic meaning.

## 4. Retrieval Mechanism

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` was selected because of its ability to work with multiple languages, including Sanskrit (through either cross-lingual transfer or direct support based on the training data, though often effective for Indic scripts).
- **Similarity Search**: Cosine similarity (default in ChromaDB) is used to retrieve the top 3 most relevant chunks concerning a user query.

## 5. Generation Mechanism

-   **Model**: `Orca Mini 3B` (4-bit quantized).
-   **Reasoning**: The model is small (~2GB RAM), runs fast on CPUs, and understands the instructions well.
-   **Prompting**: The retrieved context and user query are injected through a simple template:
    ```
    Context: {context}
    Question: {query}
    Answer (in simple English or Sanskrit):
    ```

## 6. Performance Observations

-   **Latency**:
    -   Indexing: Fast (< 1 min for small docs).
    -   Retrieval: Instant (< 1s).
    -   Generation: It depends on the CPU. generally 5-10 tokens/sec.
-   **Resource Usage**:
    -   RAM: about 4-5 GB (Model + Python overhead).
    -   CPU: High usage during generation.

## 7. Future Improvements
-   Embeddings specific to Sanskrit (e.g., IndicBERT) that are better.
-   An LLM fine-tuned on Sanskrit corpora.
-   A web-based UI (Streamlit/Gradio).