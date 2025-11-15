"""
Complete RAG Pipeline Example
Demonstrates the full workflow from PDF to querying
"""

from src.pdf_processor import PDFProcessor
from src.image_summarizer import get_image_summaries
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vector_store import ChromaVectorStore
from src.rag_engine import RAGEngine


def main():
    # Configuration
    pdf_path = "data/pdfs/cours_1.pdf"
    markdown_path = "data/processed/cours_1.md"
    collection_name = "multimodal_rag"
    
    print("=" * 80)
    print("MULTIMODAL RAG PIPELINE QCMed DEMO")
    print("=" * 80)
    
    # Step 1: Process PDF
    print("\n[1/5] Processing PDF to Markdown...")
    processor = PDFProcessor()
    image_summaries = get_image_summaries()
    
    markdown_text = processor.convert_to_markdown(
        pdf_path=pdf_path,
        output_path=markdown_path,
        image_summaries=image_summaries
    )
    print(f"✓ Processed {len(markdown_text)} characters")
    
    # Step 2: Chunk the text
    print("\n[2/5] Chunking text...")
    chunker = TextChunker(chunk_size=1024, chunk_overlap=100)
    chunks = chunker.chunk_text(markdown_text)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    print("\n[3/5] Generating embeddings...")
    embedder = Embedder(batch_size=32)
    embeddings = embedder.embed_texts(chunks, show_progress=True)
    print(f"✓ Generated {len(embeddings)} embeddings ({embedder.embedding_dim}D)")
    
    # Step 4: Index into ChromaDB
    print("\n[4/5] Indexing into ChromaDB...")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        reset_collection=True
    )
    vector_store.add_documents(texts=chunks, embeddings=embeddings)
    stats = vector_store.get_stats()
    print(f"✓ Indexed {stats['total_documents']} documents")
    
    # Step 5: Initialize RAG and test queries
    print("\n[5/5] Initializing RAG engine...")
    rag = RAGEngine(
        vector_store=vector_store,
        embedder=embedder,
        llm_model="llama3:latest", # Local model for now
        top_k=3
    )
    print("✓ RAG engine ready")
    
    # Test queries
    print("\n" + "=" * 80)
    print("TESTING QUERIES")
    print("=" * 80)
    
    test_queries = [
        "Pourquoi les toxicomanes IV sont-ils à risque d’endocardite ?",
        "Qu’est-ce qu’une endocardite infectieuse en termes simples ?",
        "Quelle est la posologie de la vancomycine dans ce document ?",
        "Quels sont les critères majeurs de Duke pour l’endocardite infectieuse ?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {query}")
        
        result = rag.query(query, return_context=False)
        print(f"A: {result['answer']}\n")
    
    print("=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    
    # Interactive mode
    print("\n\nEnter interactive mode? (y/n): ", end='')
    choice = input().strip().lower()
    
    if choice == 'y':
        print("\nInteractive Query Mode (type 'exit' to quit)")
        while True:
            query = input("\nYour query: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            if not query:
                continue
            
            result = rag.query(query)
            print(f"\nAnswer: {result['answer']}")
    
    print("\nEnd of Chat!")


if __name__ == "__main__":
    main()
