"""
PDF to Markdown conversion using Docling
"""
import re
from pathlib import Path
from typing import Dict, Optional
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TesseractOcrOptions, 
    AcceleratorDevice, 
    AcceleratorOptions
)


class PDFProcessor:
    """Handles PDF to markdown conversion with image summary replacement"""
    
    def __init__(
        self,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        generate_images: bool = True,
        images_scale: int = 2,
        num_threads: int = 4
    ):
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            generate_picture_images=generate_images,
            generate_page_images=generate_images,
            do_formula_enrichment=True,
            images_scale=images_scale,
            table_structure_options={"do_cell_matching": True},
            ocr_options=TesseractOcrOptions(),
            accelerator_options=AcceleratorOptions(
                num_threads=num_threads, 
                device=AcceleratorDevice.CPU
            ),
        )
        
        self.format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
        }
        self.converter = DocumentConverter(format_options=self.format_options)
    
    def convert_to_markdown(
        self, 
        pdf_path: str, 
        output_path: Optional[str] = None,
        image_summaries: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Convert PDF to markdown with optional image summary replacement
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save markdown
            image_summaries: Dict mapping image filenames to text summaries
            
        Returns:
            Markdown text
        """
        result = self.converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown(image_mode="embedded")
        
        # Replace base64 images with summaries if provided
        if image_summaries:
            markdown_text = self._replace_images_with_summaries(
                markdown_text, 
                image_summaries
            )
        
        # Save if output path specified
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            print(f"Saved markdown to: {output_path}")
        
        return markdown_text
    
    def _replace_images_with_summaries(
        self, 
        md_text: str, 
        summary_dict: Dict[str, str]
    ) -> str:
        """Replace base64 embedded images with text summaries"""
        # This pattern matches the Docling embedded image format
        pattern = r'!\[.*?\]\(data:image\/.*?;base64,[A-Za-z0-9+/=\n]+\)'

        # We need to replace them in order. 
        # Docling usually exports images in order.
        # If summary_dict is keyed by filename, we might have a mismatch if we don't know which image is which.
        # However, the previous implementation assumed an iterator.
        
        # Better approach: The caller should provide a list of summaries corresponding to the images in the doc.
        # Or we rely on the fact that we processed them in order.
        
        summary_iter = iter(summary_dict.values())
        
        def replacement(match):
            try:
                summary = next(summary_iter)
                return f"\n\n**Image Description:**\n{summary}\n\n"
            except StopIteration:
                return "\n\n[Image removed - no summary available]\n\n"
        
        return re.sub(pattern, replacement, md_text)

    def process_pdf_with_vlm(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        vlm_processor = None
    ) -> str:
        """
        Convert PDF to markdown and generate image descriptions using VLM
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save markdown
            vlm_processor: Instance of VLMProcessor
            
        Returns:
            Markdown text with image descriptions
        """
        print(f"Converting PDF: {pdf_path}")
        result = self.converter.convert(pdf_path)
        
        # Extract images and generate summaries
        image_summaries = {}
        
        if vlm_processor:
            print("Extracting and processing images with VLM...")
            
            counter = 0
            for element, _ in result.document.iterate_items():
                # Check if element has an image reference
                if hasattr(element, "image") and element.image:
                    print(f"Processing image {counter+1}...")
                    
                    try:
                        # Docling ImageRef has a 'pil_image' property or we need to get it differently
                        # Try different ways to get the PIL image
                        pil_image = None
                        
                        if hasattr(element.image, 'pil_image'):
                            pil_image = element.image.pil_image
                        elif hasattr(element.image, 'get_pil_image'):
                            pil_image = element.image.get_pil_image()
                        elif hasattr(element.image, 'image'):
                            pil_image = element.image.image
                        elif hasattr(element.image, 'as_pil'):
                            pil_image = element.image.as_pil()
                        else:
                            # Try to access via the document's image store
                            # Docling stores images in result.document.pictures
                            print(f"  ImageRef type: {type(element.image)}, attrs: {dir(element.image)}")
                        
                        if pil_image:
                            description = vlm_processor.process_image(pil_image)
                            image_summaries[f"image_{counter}"] = description
                        else:
                            image_summaries[f"image_{counter}"] = "[Image could not be extracted]"
                            
                    except Exception as e:
                        print(f"  Error extracting image: {e}")
                        image_summaries[f"image_{counter}"] = "[Image extraction failed]"
                    
                    counter += 1
        
        # Export to markdown
        markdown_text = result.document.export_to_markdown(image_mode="embedded")
        
        # Replace images
        if image_summaries:
            markdown_text = self._replace_images_with_summaries(
                markdown_text, 
                image_summaries
            )
            
        # Save if output path specified
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            print(f"Saved markdown to: {output_path}")
            
        return markdown_text
