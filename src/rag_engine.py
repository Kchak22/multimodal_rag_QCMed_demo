"""
RAG engine combining retrieval and generation
"""
from typing import List, Dict
from llama_index.llms.ollama import Ollama


class RAGEngine:
    """Retrieval-Augmented Generation engine"""
    
    def __init__(
        self,
        vector_store,
        embedder,
        llm_model: str = "llama3.2",
        top_k: int = 3,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_store: ChromaVectorStore instance
            embedder: Embedder instance
            llm_model: Ollama model name
            top_k: Number of documents to retrieve
            ollama_base_url: Ollama server URL
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        
        print(f"Initializing LLM: {llm_model}")
        self.llm = Ollama(
            model=llm_model,
            base_url=ollama_base_url,
            request_timeout=120.0
        )
        
        self.prompt_template = """
        Tu disposes d’un contexte ci-dessous. Utilise-le en priorité et fais une réponse correcte, concise et factuelle.

        MÉCANISME DE CONTRÔLE :
        1. Self-check interne (non affiché) :
        - Vérifie que les éléments spécifiques de ta réponse sont bien présents dans le contexte.
        - Si la réponse est d'ordre général et ne nécessite pas d'informations précises du contexte,
            tu peux utiliser tes connaissances générales.
        - Ne rejette pas automatiquement si le contexte est vide : analyse la nature de la question.

        2. Self-consistency :
        - Génère mentalement plusieurs formulations possibles.
        - Choisis la réponse cohérente entre ces formulations.
        - Si plusieurs versions internes divergent, considère que tu n'es pas certain.

        3. Règle de certitude :
        - Si tu peux répondre de manière fiable (via le contexte OU via connaissances générales non spécifiques),
            réponds normalement.
        - Si la question requiert des faits précis non présents dans le contexte,
            ou si tu n'es pas suffisamment confiant, répond exactement :
            "Je suis incapable de vous aider avec cela. Puis-je vous assister avec une autre question ?"

        ---------------------
        Contexte :
        {context}
        ---------------------

        Requête : {query}

        Réponse : la réponse est 
        """
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            
        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results["documents"])):
            retrieved_docs.append({
                "text": results["documents"][i],
                "distance": results["distances"][i],
                "metadata": results["metadatas"][i]
            })
        
        return retrieved_docs
    
    def generate_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Combine retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Combined context string
        """
        contexts = [doc["text"] for doc in retrieved_docs]
        return "\n\n---\n\n".join(contexts)
    
    def query(self, query: str, return_context: bool = False) -> Dict:
        """
        Execute full RAG query: retrieve + generate
        
        Args:
            query: User query
            return_context: If True, include retrieved context in response
            
        Returns:
            Dict with answer and optionally retrieved context
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        # Generate context
        context = self.generate_context(retrieved_docs)
        
        # Create prompt
        prompt = self.prompt_template.format(context=context, query=query)
        
        # Generate answer
        response = self.llm.complete(prompt)
        answer = str(response)
        
        result = {"answer": answer}
        
        if return_context:
            result["retrieved_docs"] = retrieved_docs
            result["context"] = context
        
        return result
    
    def stream_query(self, query: str):
        """
        Stream the response token by token
        
        Args:
            query: User query
            
        Yields:
            Response tokens
        """
        retrieved_docs = self.retrieve(query)
        context = self.generate_context(retrieved_docs)
        prompt = self.prompt_template.format(context=context, query=query)
        
        response_gen = self.llm.stream_complete(prompt)
        for token in response_gen:
            yield str(token)
