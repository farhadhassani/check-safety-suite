"""
OCR package for check text extraction using docTR and TrOCR
CPU-only implementation for production deployment
"""

from .base_ocr import run_raw_ocr, correct_orientation
from .field_parser import parse_cheque_fields

def extract_cheque_text(image):
    """
    Extract structured text fields from a cheque image.
    
    Args:
        image: np.ndarray, BGR or RGB cheque image
    
    Returns:
        Dict containing parsed fields (payee, date, amount_numeric, amount_words, micr, etc.)
    """
    # Correct orientation (deskew/rotate 90 degrees if needed)
    image = correct_orientation(image)

    # Run OCR to get all words with bboxes
    words = run_raw_ocr(image)
    
    # Parse into structured fields (pass image for optional TrOCR refinement)
    fields = parse_cheque_fields(words, image.shape, image)
    
    return fields

__all__ = ['extract_cheque_text', 'run_raw_ocr', 'parse_cheque_fields']
