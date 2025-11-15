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
        pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

        summary_iter = iter(summary_dict.values())
        
        def replacement(match):
            try:
                summary = next(summary_iter)
                return f"\n\n{summary}\n\n"
            except StopIteration:
                return "\n\n[Image removed - no summary available]\n\n"
        
        return re.sub(pattern, replacement, md_text)
