"""
VLM Processor for image description using Ollama
"""
import base64
import io
from typing import Optional
from PIL import Image

class VLMProcessor:
    """Handles image description generation using a Vision Language Model"""
    
    def __init__(
        self,
        model_name: str = "llava",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize VLM processor
        
        Args:
            model_name: Name of the VLM model (e.g., llava, moondream)
            base_url: Ollama base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        
        # Try to import MultiModal Ollama
        try:
            from llama_index.multi_modal_llms.ollama import OllamaMultiModal
            self.mm_llm = OllamaMultiModal(model=model_name, base_url=base_url, request_timeout=120.0)
            self.use_llama_index = True
            print(f"Initialized VLM with LlamaIndex MultiModal: {model_name}")
        except ImportError:
            # Fallback to direct Ollama API call
            self.mm_llm = None
            self.use_llama_index = False
            print(f"LlamaIndex MultiModal not available. Using direct Ollama API for: {model_name}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def process_image(self, image: Image.Image) -> str:
        """
        Generate a description for an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Text description of the image
        """
        prompt = (
            "Analyze this image/chart. Provide a detailed textual description of "
            "the data, content, or visual elements shown. Output text only."
        )
        
        try:
            if self.use_llama_index and self.mm_llm:
                # Use LlamaIndex MultiModal
                from llama_index.core.schema import ImageDocument
                
                # Convert to bytes
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                
                image_doc = ImageDocument(image=img_bytes)
                response = self.mm_llm.complete(
                    prompt=prompt,
                    image_documents=[image_doc]
                )
                return str(response)
            else:
                # Fallback: Direct Ollama API call
                import requests
                
                img_b64 = self._image_to_base64(image)
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "[No response from VLM]")
                else:
                    return f"[VLM Error: {response.status_code}]"
                    
        except Exception as e:
            print(f"Error processing image with VLM: {e}")
            return "[Image description failed]"
