"""
CLI script to process a PDF file into markdown
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.vlm_processor import VLMProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Process PDF to markdown with Docling and VLM"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output markdown file path (default: data/processed/<pdf_name>.md)"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        default=False, # Default to OCR enabled
        help="Disable OCR"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't extract images"
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="llava",
        help="VLM model name for image description (default: llava)"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        pdf_name = Path(args.pdf_path).stem
        output_path = f"data/processed/{pdf_name}.md"
    else:
        output_path = args.output
    
    processor = PDFProcessor(
        do_ocr=not args.no_ocr,
        generate_images=not args.no_images
    )
    
    # Initialize VLM if images are enabled
    vlm_processor = None
    if not args.no_images:
        try:
            vlm_processor = VLMProcessor(model_name=args.vlm_model)
        except Exception as e:
            print(f"Warning: Could not initialize VLM: {e}. Image descriptions will be skipped.")
    
    # Convert PDF
    print(f"Processing PDF: {args.pdf_path}")
    if vlm_processor:
        markdown = processor.process_pdf_with_vlm(
            pdf_path=args.pdf_path,
            output_path=output_path,
            vlm_processor=vlm_processor
        )
    else:
        markdown = processor.convert_to_markdown(
            pdf_path=args.pdf_path,
            output_path=output_path
        )
    
    print(f"\nMarkdown saved to: {output_path}")
    print(f"Length: {len(markdown)} characters")


if __name__ == "__main__":
    main()
