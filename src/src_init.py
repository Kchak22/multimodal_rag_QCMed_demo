"""
Multimodal RAG with ChromaDB

A modular RAG system for querying multimodal PDF documents.
"""

__version__ = "1.0.0"

from .pdf_processor import PDFProcessor
from .chunker import TextChunker
from .embedder import Embedder
from .vector_store import ChromaVectorStore
from .rag_engine import RAGEngine
from .image_summarizer import get_image_summaries, add_image_summary

__all__ = [
    "PDFProcessor",
    "TextChunker",
    "Embedder",
    "ChromaVectorStore",
    "RAGEngine",
    "get_image_summaries",
    "add_image_summary",
]
