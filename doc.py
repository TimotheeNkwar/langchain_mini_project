#type: ignore
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
def convert_doc_to_txt(input_path, output_path):
    """Convert a document (PDF, DOCX, etc.) to plain text."""
    try:
        result = converter.convert(input_path)
        text_content = result.document.export_to_markdown()
        with open(output_path, "w") as f:
            f.write(text_content)
        print(f"✓ Document converted to {output_path}")
    except Exception as e:
        print(f"Error converting document: {e}")

convert_doc_to_txt("data/data.pdf", "data/data.txt")