"""
CLI script to process a PDF file into markdown
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.image_summarizer import get_image_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Process PDF to markdown with Docling"
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
        default=True,
        help="Disable OCR"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't extract images"
    )
    parser.add_argument(
        "--use-summaries",
        action="store_true",
        help="Replace images with text summaries"
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
    
    # Get image summaries if requested
    image_summaries = None
    if args.use_summaries:
        image_summaries = get_image_summaries()
    
    # Convert PDF
    print(f"Processing PDF: {args.pdf_path}")
    markdown = processor.convert_to_markdown(
        pdf_path=args.pdf_path,
        output_path=output_path,
        image_summaries=image_summaries
    )
    
    print(f"\nMarkdown saved to: {output_path}")
    print(f"Length: {len(markdown)} characters")


if __name__ == "__main__":
    main()
