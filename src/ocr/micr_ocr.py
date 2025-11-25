import cv2
import numpy as np
import re

# Try to import pytesseract, but make it optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def read_micr_region(image_cv):
    """
    Extracts MICR line from a check image.
    Args:
        image_cv: OpenCV image (BGR)
    Returns:
        routing (str), account (str), check_number (str), aba_ok (bool)
    """
    # Initialize return values
    routing = "Not Detected"
    account = "Not Detected"
    check_number = "Not Detected"
    aba_ok = False
    
    # Return "Not Detected" values if Tesseract is not available
    # (MICR requires OCR for accurate extraction)
    if not TESSERACT_AVAILABLE:
        return routing, account, check_number, aba_ok
    
    try:
        # 1. Preprocessing
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # 2. Locate MICR band (bottom 20% of the check usually)
        h, w = gray.shape
        micr_roi = gray[int(h*0.8):h, 0:w]
        
        # 3. Threshold for better OCR
        _, thresh = cv2.threshold(micr_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 4. OCR with MICR-optimized config
        custom_config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ACD '
        micr_text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # 5. Parse MICR text
        # Clean text - remove non-MICR characters
        clean_text = re.sub(r'[^0-9A-D]', '', micr_text)
        
        # Try to find 9-digit routing number
        routing_match = re.search(r'\d{9}', clean_text)
        if routing_match:
            routing = routing_match.group(0)
            # Validate ABA
            aba_ok = validate_aba_routing(routing)
            
        # Placeholder for account/check parsing
        # In a real implementation, we'd use the delimiters (A,C,D) to split fields
        
        return routing, account, check_number, aba_ok
    except Exception:
        # Return "Not Detected" on any error instead of placeholder values
        return routing, account, check_number, aba_ok


def validate_aba_routing(routing_number):
    """Validate ABA routing number checksum"""
    if not routing_number or len(routing_number) != 9 or not routing_number.isdigit():
        return False
    digits = [int(d) for d in routing_number]
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    checksum = sum(d * w for d, w in zip(digits[:8], weights))
    check_digit = (10 - (checksum % 10)) % 10
    return check_digit == digits[8]
