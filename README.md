# Multimodal RAG with Qdrant & VLM for QCMed

A Python-based Retrieval-Augmented Generation (RAG) system for querying PDF medical courses. This enhanced version uses **Docling** for structured parsing, **Ollama VLM** (e.g., LLaVA) for image description, **Hierarchical Chunking** for better context, and **Qdrant** for Hybrid Search (Dense + Sparse).

## Features

- **Multimodal PDF Processing**: Extract text, tables, and images using Docling.
- **VLM Image Description**: Automatically generates textual descriptions for images using local Vision Language Models (Ollama).
- **Hierarchical Chunking**: Smart splitting by Markdown headers with recursive fallback and parent context injection.
- **Hybrid Search**: Combines semantic search (Dense) with keyword search (Sparse/BM25) using Qdrant.
- **Local Privacy**: Fully local execution with Ollama and local vector store.

## Architecture

```
PDF → Docling → Images → VLM (Ollama) → Descriptions
         ↓                                    ↓
      Markdown <------------------------------+
         ↓
Hierarchical Chunking → Hybrid Embeddings (Dense + Sparse) → Qdrant
                                                                ↓
                                                      Query → Retrieval → LLM → Answer
```

## Prerequisites

1. **Python 3.10+** (Python 3.13 supported)
2. **Ollama** installed and running:
   ```bash
   # Install Ollama: https://ollama.ai
   
   # Pull LLM for reasoning
   ollama pull llama3:latest
   
   # Pull VLM for image description
   ollama pull llava
   ```

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd multimodal-rag-QCMed_demo

# Create virtual environment
python -m venv env
source env/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Process a PDF (with VLM)

Convert PDF to Markdown and generate image descriptions:

```bash
python scripts/process_pdf.py data/pdfs/cours_1.pdf \
  --output data/processed/cours_1.md \
  --vlm-model llava
```

*Arguments:*
- `--no-ocr`: Disable OCR (faster but less accurate for scanned docs).
- `--no-images`: Skip image extraction.
- `--vlm-model`: Specify VLM model (default: `llava`).

### 2. Index the Document (Qdrant)

Index the processed Markdown into Qdrant with Hierarchical Chunking:

```bash
python scripts/index.py data/processed/ \
  --collection-name multimodal_rag_qcmed \
  --chunk-size 512 \
  --chunk-overlap 50 \
  --reset
```

*Arguments:*
- `--reset`: Deletes existing collection before indexing.
- `--chunk-size`: Max tokens per chunk (default: 512).

### 3. Query the System

Run a Hybrid Search query:

```bash
# Single query with context display
python scripts/query.py "Quels sont les critères majeurs de Duke ?" \
  --collection-name multimodal_rag_qcmed \
  --top-k 5 \
  --show-context

# Interactive mode
python scripts/query.py
```

*Arguments:*
- `--show-context`: Print retrieved chunks and their sources.
- `--stream`: Stream the answer token-by-token.

## Project Structure

```
multimodal-rag-QCMed_demo/
├── data/
│   ├── pdfs/              # Input PDF files
│   └── processed/         # Converted markdown files
├── src/
│   ├── pdf_processor.py   # Docling conversion
│   ├── vlm_processor.py   # Image description (Ollama)
│   ├── chunker.py         # Hierarchical chunking
│   ├── embedder.py        # Dense embedding generation
│   ├── vector_store.py    # Qdrant interface (Hybrid)
│   └── rag_engine.py      # RAG logic & System Prompt
├── scripts/
│   ├── process_pdf.py     # CLI: Process PDFs
│   ├── index.py           # CLI: Index documents
│   └── query.py           # CLI: Query system
├── qdrant_db/             # Qdrant local storage (auto-created)
├── requirements.txt
└── README.md
```

## Configuration

Key parameters can be adjusted in the scripts:

- **Chunking**: Default 512 tokens, 50 overlap.
- **Embeddings**: `nomic-ai/nomic-embed-text-v1.5` (Dense), `Qdrant/bm25` (Sparse).
- **LLM**: Defaults to `llama3:latest`.
- **VLM**: Defaults to `llava`.

## Troubleshooting

### Ollama Connection
Ensure Ollama is running:
```bash
ollama serve
```

### Dependency Issues
If you face issues with `scipy` or `pandas` on macOS, ensure you are using the pinned versions in `requirements.txt` or try Python 3.11/3.12.
