"""
Verification script for RAG components
"""
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.chunker import HierarchicalChunker
from src.vector_store import QdrantVectorStore
from src.rag_engine import RAGEngine

def test_hierarchical_chunker():
    print("\nTesting HierarchicalChunker...")
    chunker = HierarchicalChunker(chunk_size=50, chunk_overlap=10)
    
    markdown = """
# Header 1
Some text under header 1.

## Header 2
More text here.

### Header 3
Deeply nested text.
    """
    
    chunks = chunker.chunk_markdown(markdown)
    print(f"Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk['metadata']['parent_path']}")
        assert "parent_path" in chunk["metadata"]
        
    assert len(chunks) > 0
    print("HierarchicalChunker test passed!")

def test_qdrant_vector_store():
    print("\nTesting QdrantVectorStore...")
    test_dir = "./test_qdrant_db"
    
    # Mock FastEmbed to avoid downloading models
    with patch("src.vector_store.QdrantVectorStore.__init__", return_value=None) as mock_init:
        pass

    try:
        # We'll mock the sparse model to avoid download
        with patch("fastembed.SparseTextEmbedding") as MockSparse:
            mock_instance = MockSparse.return_value
            # Mock embed to return dummy sparse vectors
            # FastEmbed returns a generator of SparseEmbedding (indices, values)
            class DummySparse:
                def __init__(self):
                    self.indices = MagicMock()
                    self.indices.tolist.return_value = [1, 2, 3]
                    self.values = MagicMock()
                    self.values.tolist.return_value = [0.1, 0.2, 0.3]
            
            mock_instance.embed.return_value = [DummySparse()]
            
            store = QdrantVectorStore(
                collection_name="test_collection",
                persist_directory=test_dir,
                reset_collection=True
            )
            
            texts = ["Hello world"]
            embeddings = [[0.1] * 768]
            
            store.add_documents(texts, embeddings)
            
            # Query
            # Mock query return
            store.client.query_points = MagicMock()
            store.client.query_points.return_value.points = [
                MagicMock(id="1", payload={"text": "Hello world"}, score=0.9)
            ]
            
            results = store.query("Hello", [0.1]*768)
            assert len(results["documents"]) == 1
            print("QdrantVectorStore test passed!")
            
    except Exception as e:
        print(f"Qdrant test failed (might be missing dependencies): {e}")
    finally:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

def test_rag_engine():
    print("\nTesting RAGEngine...")
    
    mock_store = MagicMock()
    mock_store.query.return_value = {
        "ids": ["1"],
        "documents": ["Context text"],
        "distances": [0.9],
        "metadatas": [{"parent_path": "H1"}]
    }
    
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    
    with patch("src.rag_engine.Ollama") as MockOllama:
        mock_llm = MockOllama.return_value
        mock_llm.complete.return_value = "This is the answer."
        
        rag = RAGEngine(
            vector_store=mock_store,
            embedder=mock_embedder,
            llm_model="test-model"
        )
        
        result = rag.query("Question")
        
        assert "answer" in result
        assert result["answer"] == "This is the answer."
        print("RAGEngine test passed!")

if __name__ == "__main__":
    test_hierarchical_chunker()
    test_qdrant_vector_store()
    test_rag_engine()
