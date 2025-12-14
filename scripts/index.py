"""
CLI script to index documents into ChromaDB
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.chunker import HierarchicalChunker
from src.embedder import Embedder
from src.vector_store import QdrantVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Index markdown documents into Qdrant"
    )
    parser.add_argument(
        "markdown_path",
        type=str,
        help="Path to markdown file or directory"
    )
    parser.add_argument(
        "--collection-name",
        "-c",
        type=str,
        default="multimodal_rag",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset collection before indexing"
    )
    
    args = parser.parse_args()
    
    # Load markdown files
    markdown_path = Path(args.markdown_path)
    if markdown_path.is_file():
        print(f"Loading single file: {markdown_path}")
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        markdown_texts = [markdown_text]
    elif markdown_path.is_dir():
        print(f"Loading all .md files from: {markdown_path}")
        markdown_files = list(markdown_path.glob("*.md"))
        markdown_texts = []
        for md_file in markdown_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_texts.append(f.read())
        print(f"Loaded {len(markdown_texts)} files")
    else:
        print(f"Error: {markdown_path} not found")
        return
    
    # Initialize components
    print("\n=== Initializing Components ===")
    chunker = HierarchicalChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    embedder = Embedder(batch_size=args.batch_size)
    
    vector_store = QdrantVectorStore(
        collection_name=args.collection_name,
        reset_collection=args.reset
    )
    
    # Process documents
    print("\n=== Chunking Documents ===")
    all_chunks_data = []
    
    # Get file names to track source document
    if markdown_path.is_file():
        file_names = [markdown_path.stem]  # Get filename without extension
    else:
        file_names = [f.stem for f in markdown_path.glob("*.md")]
    
    for i, text in enumerate(markdown_texts):
        source_doc = file_names[i] if i < len(file_names) else "Unknown"
        chunks = chunker.chunk_markdown(text)
        
        # Add source document name to each chunk's metadata
        for chunk in chunks:
            chunk["metadata"]["source_document"] = source_doc
        
        all_chunks_data.extend(chunks)
    
    texts = [c["text"] for c in all_chunks_data]
    metadatas = [c["metadata"] for c in all_chunks_data]
    
    print(f"Total chunks: {len(texts)}")
    
    print("\n=== Generating Embeddings ===")
    embeddings = embedder.embed_texts(texts)
    
    print("\n=== Indexing into Qdrant ===")
    vector_store.add_documents(
        texts=texts,
        dense_embeddings=embeddings,
        metadatas=metadatas
    )
    
    stats = vector_store.get_stats()
    print("\n=== Indexing Complete ===")
    print(f"Collection: {stats['collection_name']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Persist directory: {stats['persist_directory']}")


if __name__ == "__main__":
    main()
