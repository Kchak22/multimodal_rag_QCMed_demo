"""
ChromaDB vector store implementation
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path


class ChromaVectorStore:
    """Manages document embeddings in ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        persist_directory: str = "./chroma_db",
        reset_collection: bool = False
    ):
        """
        Initialize ChromaDB client and collection
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            reset_collection: If True, delete and recreate collection
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        if reset_collection:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
        
        print(f"Collection '{collection_name}' ready with {self.collection.count()} documents")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the collection
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: Optional metadata for each chunk
            ids: Optional custom IDs (auto-generated if not provided)
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        
        # ChromaDB expects documents as the text content
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(texts)} documents to collection")
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query the collection for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dict with ids, documents, distances, and metadatas
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Flatten single query results
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0]
        }
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_directory": str(self.persist_directory)
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
