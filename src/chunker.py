"""
Text chunking with token-based sliding window
"""
from typing import List, Dict
import re
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


class HierarchicalChunker:
    """Chunks text based on Markdown structure with recursive fallback"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def chunk_markdown(self, markdown_text: str) -> List[Dict]:
        """
        Split markdown by headers and then recursively if needed.
        Returns list of dicts with text and metadata.
        """
        chunks = []
        lines = markdown_text.split('\n')
        
        current_headers = {} # Level -> Header Text
        current_section_lines = []
        
        def flush_section():
            if not current_section_lines:
                return
                
            text = "\n".join(current_section_lines).strip()
            if not text:
                return
                
            # Construct path string
            # Sort keys to ensure order H1 -> H2 -> H3
            sorted_levels = sorted(current_headers.keys())
            path_parts = [current_headers[l] for l in sorted_levels]
            parent_path = " > ".join(path_parts)
            
            # Check size
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= self.chunk_size:
                # Small enough
                chunks.append({
                    "text": text,
                    "metadata": {
                        "parent_path": parent_path,
                        "type": "text", 
                        "token_count": len(tokens)
                    }
                })
            else:               
                # Use a sliding window on tokens for the sub-chunks
                stride = self.chunk_size - self.chunk_overlap
                for i in range(0, len(tokens), stride):
                    chunk_ids = tokens[i:i + self.chunk_size]
                    chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "parent_path": parent_path,
                            "type": "text",
                            "token_count": len(chunk_ids),
                            "is_subchunk": True
                        }
                    })
            
            current_section_lines.clear()

        for line in lines:
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Flush previous section
                flush_section()
                
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update headers context
                # Clear deeper levels
                keys_to_remove = [k for k in current_headers if k >= level]
                for k in keys_to_remove:
                    del current_headers[k]
                
                current_headers[level] = title
            else:
                current_section_lines.append(line)
                
        # Flush last section
        flush_section()
        
        return chunks

