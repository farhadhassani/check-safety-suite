"""
Field OCR: Extract specific fields from bank checks (date, payee, etc.)
"""
import cv2
import numpy as np
import re
from typing import Dict, Optional, Tuple
from datetime import datetime

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


def extract_date(img_crop: np.ndarray) -> str:
    """
    Extract date from check date field
    
    Args:
        img_crop: Cropped image of date region
    
    Returns:
        Extracted date string or placeholder
    """
    if not TESSERACT_AVAILABLE:
        return "Unknown Date"
    
    try:
        # Preprocess
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
        
        # Enhance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        # Parse date if dateutil available
        if DATEUTIL_AVAILABLE:
            try:
                date_obj = date_parser.parse(text, fuzzy=True)
                return date_obj.strftime('%Y-%m-%d')
            except:
                pass
        
        # Fallback: return cleaned text
        cleaned = re.sub(r'[^0-9/-]', '', text).strip()
        return cleaned if cleaned else "Unknown Date"
        
    except Exception:
        return "Unknown Date"


def extract_payee(img_crop: np.ndarray) -> str:
    """
    Extract payee name from check
    
    Args:
        img_crop: Cropped image of payee line
    
    Returns:
        Extracted payee name or placeholder
    """
    if not TESSERACT_AVAILABLE:
        return "Unknown Payee"
    
    try:
        # Preprocess
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR for text
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        # Clean
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-\.\,]', '', text).strip()
        
        return cleaned if cleaned else "Unknown Payee"
        
    except Exception:
        return "Unknown Payee"

def extract_amount_digits(img_crop):
    # Placeholder
    return "100.00"

def extract_amount_words(img_crop):
    # Placeholder
    return "One Hundred Dollars"
