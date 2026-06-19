# RAG-Based PDF Chatbot

**A Production-Ready Retrieval-Augmented Generation (RAG) Application**

---

## Overview

This project implements a complete **RAG (Retrieval-Augmented Generation)** chatbot that enables intelligent, context-aware conversations based on the content of a provided PDF document. The application combines document ingestion, vector-based retrieval, and conversational AI within a single, easy-to-run Streamlit interface.

Built with a focus on simplicity, performance, and maintainability, the entire backend (document processing, embedding, vector storage, and retrieval) and frontend (chat interface) are consolidated into one Python file — `demo.py`.

---

## Features

- **PDF Document Ingestion**: Mannual loads, splits, and processes PDF files.
- **Advanced Embeddings**: Utilizes Hugging Face sentence-transformers for high-quality document embeddings.
- **Vector Store**: In-memory vector database (Chroma) for fast similarity search and retrieval.
- **LLM Integration**: Powered by **OpenRouter** for flexible access to state-of-the-art large language models.
- **Conversational Interface**: Clean, interactive chatbot UI built with **Streamlit**.
- **Single-File Architecture**: All logic (backend + frontend) contained in `demo.py` for rapid development and deployment.


---

## Technology Stack

| Component          | Technology                          | Purpose |
|--------------------|-------------------------------------|---------|
| **UI Framework**   | Streamlit                           | Interactive chatbot interface |
| **Document Loader**| PyPDF or LangChain PDF loader       | Extract text from PDFs |
| **Text Splitter**  | RecursiveCharacterTextSplitter      | Chunk documents optimally |
| **Embedding Model**| Hugging Face (`sentence-transformers`) | Generate dense vector representations |
| **Vector Store**   | Chroma (persistent/in-memory)       | Store and retrieve embeddings |
| **LLM Provider**   | OpenRouter API                      | Access to various LLMs |
| **Orchestration**  | LangChain                           | Retrieval and generation pipeline |
| **Language**       | Python 3.9+                         | Core development |

---


