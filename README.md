# Multimodal RAG with ChromaDB for QCMed (Demo- still in progress)

A Python-based Retrieval-Augmented Generation (RAG) system for querying PDF medical courses. Uses Docling for structured PDF parsing, ChromaDB for vector storage, and Ollama for local LLM inference.

## Features

- **Multimodal PDF Processing**: Extract text, tables, images, and formulas using Docling
- **Image Summarization**: Replace embedded images with text summaries
- **Token-based Chunking**: Smart text chunking with overlap for better context
- **ChromaDB Vector Store**: Fast, local vector database
- **Local LLM**: Privacy-friendly querying with Ollama
- **CLI Scripts**: Process PDFs, index documents, and query from command line
- **Jupyter Notebook**: Interactive testing and experimentation

## Architecture

```
PDF → Docling → Markdown → Chunking → Embeddings → ChromaDB
                                                        ↓
                                              Query → Retrieval → LLM → Answer
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running:
   ```bash
   # Install Ollama: https://ollama.ai
   ollama pull llama3b:latest
   ```
3. **Tesseract OCR** (for OCR functionality): (not used for now)
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   ```

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd multimodal-rag-QCMed_demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Process a PDF

```bash
python scripts/process_pdf.py data/pdfs/cours_1.pdf \
  --output data/processed/cours_1.md \
  --use-summaries
```

### 2. Index the Document

```bash
python scripts/index.py data/processed/cours_1.md \
  --collection-name multimodal_rag_qcmed \
  --chunk-size 1024 \
  --chunk-overlap 100
```

### 3. Query the System

```bash
# Single query
python scripts/query.py "Quels sont les critères majeurs de Duke pour l’endocardite infectieuse ?" \
  --collection-name multimodal_rag_qcmed \
  --top-k 3

# Interactive mode
python scripts/query.py
```

## Project Structure

```
multimodal-rag-chromadb/
├── data/
│   ├── pdfs/              # Input PDF files
│   └── processed/         # Converted markdown files
├── src/
│   ├── pdf_processor.py   # PDF → Markdown conversion
│   ├── image_summarizer.py # Image summary management
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # ChromaDB interface
│   └── rag_engine.py      # RAG query engine
├── scripts/
│   ├── process_pdf.py     # CLI: Process PDFs
│   ├── index.py           # CLI: Index documents
│   └── query.py           # CLI: Query system
├── notebooks/
│   └── test_notebook.ipynb  # Interactive testing
├── chroma_db/             # ChromaDB persistence (auto-created)
├── requirements.txt
└── README.md
```

## Configuration

Key parameters can be adjusted in the scripts or notebook:

- **Chunk Size**: Default 1024 tokens
- **Chunk Overlap**: Default 100 tokens
- **Embedding Model**: nomic-ai/nomic-embed-text-v1.5
- **LLM Model**: llama3b:latest (or any Ollama model)
- **Top-K Retrieval**: Default 3 documents

## Advanced Usage

### Custom Image Summaries

Edit `src/image_summarizer.py` to add custom summaries:

```python
from src.image_summarizer import add_image_summary

add_image_summary(
    "my_diagram.png",
    "A flowchart showing the data pipeline..."
)
```

### Batch Processing

Process multiple PDFs:

```bash
for pdf in data/pdfs/*.pdf; do
    python scripts/process_pdf.py "$pdf" --use-summaries
done

python scripts/index.py data/processed/ --reset
```

### Different Embedding Models

Change in `src/embedder.py`:

```python
embedder = Embedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64
)
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve
```

### ChromaDB Persistence

ChromaDB automatically persists to `./chroma_db`. To reset:

```bash
rm -rf chroma_db
```

### Memory Issues

For large documents, reduce batch size:

```bash
python scripts/index.py document.md --batch-size 8
```


## References

- [Docling](https://github.com/docling-project/docling) - Document understanding
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) - Embedding model


## What's next : 
- Fixing Docling pdf to markdown conversion 
- Trying Docling's hierarchical RAG & using pre-computed descriptions for figures and tables (using OCR)
- Trying better models (Adding ability to use API-based models and longer-context models (gemini...) wela with vision capabilities)
- Analyse des résultats


