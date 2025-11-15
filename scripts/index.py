"""
CLI script to index documents into ChromaDB
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.chunker import TextChunker
from src.embedder import Embedder
from src.vector_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Index markdown documents into ChromaDB"
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
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
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
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    embedder = Embedder(batch_size=args.batch_size)
    
    vector_store = ChromaVectorStore(
        collection_name=args.collection_name,
        reset_collection=args.reset
    )
    
    # Process documents
    print("\n=== Chunking Documents ===")
    all_chunks = chunker.chunk_texts(markdown_texts)
    
    print("\n=== Generating Embeddings ===")
    embeddings = embedder.embed_texts(all_chunks)
    
    print("\n=== Indexing into ChromaDB ===")
    vector_store.add_documents(
        texts=all_chunks,
        embeddings=embeddings
    )
    
    stats = vector_store.get_stats()
    print("\n=== Indexing Complete ===")
    print(f"Collection: {stats['collection_name']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Persist directory: {stats['persist_directory']}")


if __name__ == "__main__":
    main()
