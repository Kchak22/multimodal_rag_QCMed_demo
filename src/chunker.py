"""
Text chunking with token-based sliding window
"""
from typing import List
from transformers import AutoTokenizer


class TextChunker:
    """Handles text chunking with overlapping windows"""
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        chunk_size: int = 1024,
        chunk_overlap: int = 100
    ):
        """
        Initialize chunker with tokenizer
        
        Args:
            model_name: Model name for tokenizer
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer loaded. Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token count
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Tokenize the entire text
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        
        # Create overlapping chunks
        for i in range(0, len(input_ids), stride):
            chunk_ids = input_ids[i:i + self.chunk_size]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Stop if we've processed all tokens
            if i + self.chunk_size >= len(input_ids):
                break
        
        print(f"Created {len(chunks)} chunks from {len(input_ids)} tokens")
        return chunks
    
    def chunk_texts(self, texts: List[str]) -> List[str]:
        """
        Chunk multiple texts
        
        Args:
            texts: List of texts to chunk
            
        Returns:
            Flattened list of all chunks
        """
        all_chunks = []
        for text in texts:
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)
        
        print(f"Total chunks from {len(texts)} texts: {len(all_chunks)}")
        return all_chunks
    
    def get_stats(self, text: str) -> dict:
        """Get chunking statistics for a text"""
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = self.chunk_text(text)
        
        return {
            "total_tokens": len(input_ids),
            "num_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "avg_tokens_per_chunk": len(input_ids) / len(chunks) if chunks else 0
        }
