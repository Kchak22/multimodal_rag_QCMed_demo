"""
Qdrant vector store implementation with Hybrid Search (Dense + Sparse)
"""
from typing import List, Dict, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

class QdrantVectorStore:
    """Manages document embeddings in Qdrant with Hybrid Search"""
    
    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        persist_directory: str = "./qdrant_db",
        reset_collection: bool = False,
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        """
        Initialize Qdrant client and collection
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client
        self.client = QdrantClient(path=str(self.persist_directory))

        try:
            from fastembed import SparseTextEmbedding
            self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        except ImportError:
            print("FastEmbed not installed or model not found. Sparse search might fail.")
            self.sparse_model = None

        if reset_collection:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
            
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=768,  # Adjust based on your dense model (e.g. Nomic v1.5 is 768)
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection '{collection_name}' ready")

    def add_documents(
        self,
        texts: List[str],
        dense_embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the collection with both dense and sparse vectors
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        else:
            # Ensure text is in metadata for retrieval
            for i, meta in enumerate(metadatas):
                if "text" not in meta:
                    meta["text"] = texts[i]

        # Generate sparse embeddings
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        points = []
        for i in range(len(texts)):
            # Convert sparse embedding to Qdrant format
            sparse_vector = models.SparseVector(
                indices=sparse_embeddings[i].indices.tolist(),
                values=sparse_embeddings[i].values.tolist()
            )
            
            points.append(models.PointStruct(
                id=ids[i],
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": sparse_vector
                },
                payload=metadatas[i]
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(texts)} documents to Qdrant")

    def query(
        self,
        query_text: str,
        query_dense_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> Dict:
        """
        Hybrid query: Dense + Sparse with RRF or simple fusion
        """
        # Generate sparse query vector
        sparse_query = list(self.sparse_model.embed([query_text]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_query.indices.tolist(),
            values=sparse_query.values.tolist()
        )
        
        # 1. Search using Sparse vector to get top N (e.g. 50)      
        prefetch = [
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=50,
            )
        ]
        # 2. Rescore these N using Dense vector
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=query_dense_embedding,
            using="dense",
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Format results
        ids = []
        documents = []
        distances = []
        metadatas = []
        
        for hit in results.points:
            ids.append(hit.id)
            documents.append(hit.payload.get("text", ""))
            distances.append(hit.score)
            metadatas.append(hit.payload)
            
        return {
            "ids": ids,
            "documents": documents,
            "distances": distances,
            "metadatas": metadatas
        }
    
    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
    
    def get_stats(self):
        """Get collection statistics"""
        collection_info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "total_documents": collection_info.points_count,
            "persist_directory": str(self.persist_directory)
        }

