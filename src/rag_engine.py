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
        llm_model: str = "llama3",
        top_k: int = 3,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_store: QdrantVectorStore instance
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
Vous êtes un assistant utile. Utilisez uniquement le contexte fourni pour répondre aux questions. Si la réponse n’est pas dans le contexte, indiquez que vous ne savez pas.

IMPORTANT - Instructions de citation :
- Chaque extrait de contexte est étiqueté avec le nom du document et le chemin de la section
- Format : "Document : [nom du cours] | Section : [chemin de la section]"
- Lorsque vous citez, incluez à la fois le nom du document et le chemin de la section
- Exemple de citation : [12-Cancer-du-cavum-2025-QCMed, Section 2.1.3 Les parois latérales]
- Citez toujours exactement le document et le chemin de la section fournis

---------------------
Contexte :
{context}
---------------------

Question : {query}

Réponse (avec citations du document et du chemin de section) :

"""
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query using Hybrid Search
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store (Qdrant Hybrid)
        results = self.vector_store.query(
            query_text=query,
            query_dense_embedding=query_embedding,
            top_k=self.top_k
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results["documents"])):
            retrieved_docs.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "distance": results["distances"][i],
                "metadata": results["metadatas"][i]
            })
        
        return retrieved_docs
    
    def generate_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Combine retrieved documents into context string with clear section references
        Includes source document (course title) and section path
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            path = meta.get("parent_path", "Unknown Section")
            source_doc = meta.get("source_document", "Unknown Document")
            
            # Create a clear, citable reference with document title and section path
            context_str = f"--- Document: {source_doc} | Section: {path} ---\n{text}"
            contexts.append(context_str)
            
        return "\n\n".join(contexts)
    
    def query(self, query: str, return_context: bool = False) -> Dict:
        """
        Execute full RAG query: retrieve + generate
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
        """
        retrieved_docs = self.retrieve(query)
        context = self.generate_context(retrieved_docs)
        prompt = self.prompt_template.format(context=context, query=query)
        
        response_gen = self.llm.stream_complete(prompt)
        for token in response_gen:
            yield str(token)
