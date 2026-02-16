"""Extract text from PDF files with page boundary markers."""
import fitz  # PyMuPDF


def extract_text_with_pages(pdf_path: str) -> str:
    """
    Extract full text from a PDF with [PAGE X] markers between pages.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Full text with page markers inserted.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(f"[PAGE {page_num + 1}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract full text from PDF bytes with [PAGE X] markers.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file.
        
    Returns:
        Full text with page markers inserted.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(f"[PAGE {page_num + 1}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages)


def get_document_stats(text: str) -> dict:
    """Return basic stats about the extracted document text."""
    page_count = text.count("[PAGE ")
    char_count = len(text)
    word_count = len(text.split())
    estimated_tokens = word_count * 1.3  # rough estimate
    return {
        "page_count": page_count,
        "char_count": char_count,
        "word_count": word_count,
        "estimated_tokens": int(estimated_tokens),
    }