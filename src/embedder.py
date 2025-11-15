"""
Text embedding generation using HuggingFace models
"""
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm


class Embedder:
    """Handles text embedding generation"""
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: int = 32,
        cache_folder: str = "./hf_cache"
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for embedding generation
            cache_folder: Cache directory for model files
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"Loading embedding model: {model_name}")
        self.model = HuggingFaceEmbedding(
            model_name=model_name,
            trust_remote_code=True,
            cache_folder=cache_folder
        )
        print("Embedding model loaded successfully")
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        iterator = self._batch_iterate(texts, self.batch_size)
        if show_progress:
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            iterator = tqdm(iterator, total=total_batches, desc="Generating embeddings")
        
        for batch in iterator:
            batch_embeddings = self.model.get_text_embedding_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.model.get_query_embedding(query)
    
    @staticmethod
    def _batch_iterate(lst: List, batch_size: int):
        """Yield batches from a list"""
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension"""
        # Generate a test embedding to get dimension
        test_embedding = self.embed_query("test")
        return len(test_embedding)
