"""
CLI script to query the RAG system
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.embedder import Embedder
from src.vector_store import ChromaVectorStore
from src.rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG system"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query text"
    )
    parser.add_argument(
        "--collection-name",
        "-c",
        type=str,
        default="multimodal_rag",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--llm-model",
        "-m",
        type=str,
        default="llama3:latest", # Local model for now
        help="Ollama model name"
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print("=== Initializing RAG System ===")
    embedder = Embedder()
    
    vector_store = ChromaVectorStore(
        collection_name=args.collection_name,
        reset_collection=False
    )
    
    rag = RAGEngine(
        vector_store=vector_store,
        embedder=embedder,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    # Execute query
    print(f"\n=== Query: {args.query} ===\n")
    
    if args.stream:
        print("Answer (streaming):")
        for token in rag.stream_query(args.query):
            print(token, end='', flush=True)
        print("\n")
    else:
        result = rag.query(args.query, return_context=args.show_context)
        
        print("Answer:")
        print(result["answer"])
        
        if args.show_context:
            print("\n=== Retrieved Context ===")
            for i, doc in enumerate(result["retrieved_docs"], 1):
                print(f"\n--- Document {i} (distance: {doc['distance']:.4f}) ---")
                print(doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"])


def interactive_mode():
    """Interactive query mode"""
    print("=== Interactive Query Mode ===")
    print("Type 'exit' to quit\n")
    
    # Initialize once
    embedder = Embedder()
    vector_store = ChromaVectorStore(collection_name="multimodal_rag")
    rag = RAGEngine(vector_store=vector_store, embedder=embedder)
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        print("\nAnswer:")
        result = rag.query(query)
        print(result["answer"])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()
