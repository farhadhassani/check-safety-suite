import cv2
import numpy as np
import pytesseract

def extract_micr(image_cv):
    """
    Extracts MICR line from a check image.
    Args:
        image_cv: OpenCV image (BGR)
    Returns:
        dict: Extracted MICR data (raw text, parsed fields)
    """
    # 1. Preprocessing
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Locate MICR band (bottom 20% of the check usually)
    h, w = gray.shape
    micr_roi = gray[int(h*0.8):h, 0:w]
    
    # 3. Thresholding / Morphological operations to isolate characters
    # Otsu's thresholding
    _, thresh = cv2.threshold(micr_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 4. OCR with Tesseract
    # Configure Tesseract for MICR (E-13B font)
    # Note: Standard Tesseract might struggle without specific training, 
    # but we'll try with standard configuration or specific whitelist if possible.
    # E-13B characters: 0-9 and 4 special symbols (A, B, C, D or similar mappings)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCD'
    
    micr_text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # 5. Parse MICR text
    # Simple regex-based parsing (Improvement: use specific MICR E-13B patterns)
    import re
    
    # Remove non-alphanumeric (keep special chars if mapped)
    # Assuming Tesseract returns A,B,C,D for special symbols or just numbers
    clean_text = re.sub(r'[^0-9A-D]', '', micr_text)
    
    # Heuristic parsing:
    # Routing is usually 9 digits between symbols
    # Account is variable length
    # Check number is often at the end or beginning
    
    routing = None
    account = None
    check_number = None
    
    # Try to find 9-digit routing number
    routing_match = re.search(r'\d{9}', clean_text)
    if routing_match:
        routing = routing_match.group(0)
        
    return {
        "raw_text": micr_text.strip(),
        "cleaned_text": clean_text,
        "parsed": {
            "routing": routing,
            "account": account, # TODO: better parsing
            "check_number": check_number # TODO: better parsing
        }
    }
