"""
Field parser for cheque text extraction
Uses bbox positions, regex, and heuristics to parse structured fields
"""
import numpy as np
import cv2
import re
from typing import List, Dict, Any, Tuple
import logging

from .trocr_handwritten import recognize_handwritten_line, TROCR_AVAILABLE

logger = logging.getLogger(__name__)


def normalize_amount(text: str) -> float:
    """Extract numeric amount from text"""
    # Remove currency symbols and commas
    clean = re.sub(r'[$,]', '', text)
    # Find first number
    match = re.search(r'\d+\.?\d*', clean)
    if match:
        try:
            return float(match.group())
        except:
            pass
    return 0.0


def normalize_date(text: str) -> str:
    """Extract and normalize date"""
    # Common date patterns
    patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
        r'(\d{1,2})\s+(\w+)\s+(\d{2,4})',  # 23 Nov 2025
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    
    return text


def normalize_micr(text: str) -> Dict[str, str]:
    """
    Parse MICR line to extract routing, account, check number
    """
    # Remove special MICR symbols
    clean = re.sub(r'[⑆⑈⑉⑊]', ' ', text)
    
    # Try to find routing (9 digits), account, check number
    numbers = re.findall(r'\d+', clean)
    
    result = {
        'routing': numbers[0] if len(numbers) > 0 and len(numbers[0]) == 9 else '',
        'account': numbers[1] if len(numbers) > 1 else '',
        'check_number': numbers[2] if len(numbers) > 2 else '',
        'raw': text
    }
    
    return result


def get_region_words(words: List[Dict], y_min: float, y_max: float, x_min: float = 0.0, x_max: float = 1.0) -> List[Dict]:
    """Get words in a specific region"""
    region_words = []
    for word in words:
        bbox = word['bbox']
        word_center_y = (bbox[1] + bbox[3]) / 2
        word_center_x = (bbox[0] + bbox[2]) / 2
        
        if y_min <= word_center_y <= y_max and x_min <= word_center_x <= x_max:
            region_words.append(word)
    
    return region_words


def crop_region_from_image(image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop image region based on normalized bbox"""
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    x1 = int(x_min * w)
    y1 = int(y_min * h)
    x2 = int(x_max * w)
    y2 = int(y_max * h)
    
    # Add some padding
    padding = 5
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def parse_cheque_fields(words: List[Dict], image_shape: Tuple, image: np.ndarray = None) -> Dict[str, Any]:
    """
    Parse OCR words into structured cheque fields
    
    Args:
        words: List of word dicts from run_raw_ocr
        image_shape: (height, width, channels) of original image
        image: Optional original image for TrOCR refinement
    
    Returns:
        Dict with fields: payee, date, amount_numeric, amount_words, micr, bank_name, all_words
    """
    if not words:
        return {
            'payee': {'text': '', 'confidence': 0.0, 'bbox': None},
            'date': {'text': '', 'normalized': '', 'confidence': 0.0, 'bbox': None},
            'amount_numeric': {'text': '', 'normalized': 0.0, 'confidence': 0.0, 'bbox': None},
            'amount_words': {'text': '', 'normalized': 0.0, 'confidence': 0.0, 'bbox': None},
            'micr': {'routing': '', 'account': '', 'check_number': '', 'confidence': 0.0, 'bbox': None},
            'bank_name': {'text': '', 'confidence': 0.0, 'bbox': None},
            'all_words': []
        }
    
    height = image_shape[0]
    
    # Sort words by position
    sorted_words = sorted(words, key=lambda w: (w['bbox'][1], w['bbox'][0]))
    
    # Initialize result
    result = {
        'payee': {'text': '', 'confidence': 0.0, 'bbox': None},
        'date': {'text': '', 'normalized': '', 'confidence': 0.0, 'bbox': None},
        'amount_numeric': {'text': '', 'normalized': 0.0, 'confidence': 0.0, 'bbox': None},
        'amount_words': {'text': '', 'normalized': 0.0, 'confidence': 0.0, 'bbox': None},
        'micr': {'routing': '', 'account': '', 'check_number': '', 'confidence': 0.0, 'bbox': None, 'raw': ''},
        'bank_name': {'text': '', 'confidence': 0.0, 'bbox': None},
        'all_words': words
    }
    
    # Heuristic regions (approximate)
    # Bank name: top 15%
    bank_words = get_region_words(words, 0.0, 0.15)
    if bank_words:
        result['bank_name'] = {
            'text': ' '.join([w['text'] for w in bank_words[:3]]),  # First 3 words
            'confidence': np.mean([w['confidence'] for w in bank_words[:3]]),
            'bbox': bank_words[0]['bbox']
        }
    
    # Date: typically top-right (y: 0-0.25, x: 0.6-1.0)
    date_words = get_region_words(words, 0.0, 0.25, 0.6, 1.0)
    if date_words:
        date_text = ' '.join([w['text'] for w in date_words])
        result['date'] = {
            'text': date_text,
            'normalized': normalize_date(date_text),
            'confidence': np.mean([w['confidence'] for w in date_words]),
            'bbox': date_words[0]['bbox']
        }
    
    # Payee: middle area (y: 0.2-0.4, x: 0.1-0.8)
    payee_words = get_region_words(words, 0.2, 0.4, 0.1, 0.8)
    if payee_words:
        result['payee'] = {
            'text': ' '.join([w['text'] for w in payee_words]),
            'confidence': np.mean([w['confidence'] for w in payee_words]),
            'bbox': payee_words[0]['bbox']
        }
    
    # Amount numeric: right side (y: 0.3-0.5, x: 0.7-1.0)
    amount_num_words = get_region_words(words, 0.3, 0.5, 0.7, 1.0)
    if amount_num_words:
        # Find word with $ or numbers
        for w in amount_num_words:
            if '$' in w['text'] or re.search(r'\d', w['text']):
                result['amount_numeric'] = {
                    'text': w['text'],
                    'normalized': normalize_amount(w['text']),
                    'confidence': w['confidence'],
                    'bbox': w['bbox']
                }
                break
    
    # Amount in words: middle-left (y: 0.4-0.6, x: 0.0-0.7)
    amount_word_region = get_region_words(words, 0.4, 0.6, 0.0, 0.7)
    if amount_word_region:
        amount_text = ' '.join([w['text'] for w in amount_word_region])
        avg_conf = np.mean([w['confidence'] for w in amount_word_region])
        
        # If confidence is low and TrOCR is available, try refinement
        if avg_conf < 0.6 and image is not None and TROCR_AVAILABLE:
            try:
                # Get bbox for entire amount words region
                bboxes = [w['bbox'] for w in amount_word_region]
                x_min = min(b[0] for b in bboxes)
                y_min = min(b[1] for b in bboxes)
                x_max = max(b[2] for b in bboxes)
                y_max = max(b[3] for b in bboxes)
                
                # Crop and recognize with TrOCR
                cropped = crop_region_from_image(image, (x_min, y_min, x_max, y_max))
                trocr_text, trocr_conf = recognize_handwritten_line(cropped)
                
                if trocr_conf > avg_conf:
                    amount_text = trocr_text
                    avg_conf = trocr_conf
                    logger.info(f"TrOCR improved amount_words: {trocr_text} (conf: {trocr_conf:.2f})")
            except Exception as e:
                logger.warning(f"TrOCR refinement failed: {e}")
        
        result['amount_words'] = {
            'text': amount_text,
            'normalized': normalize_amount(amount_text),
            'confidence': avg_conf,
            'bbox': amount_word_region[0]['bbox'] if amount_word_region else None
        }
    
    # MICR: bottom 15%
    micr_words = get_region_words(words, 0.85, 1.0)
    if micr_words:
        micr_text = ' '.join([w['text'] for w in micr_words])
        micr_parsed = normalize_micr(micr_text)
        result['micr'] = {
            **micr_parsed,
            'confidence': np.mean([w['confidence'] for w in micr_words]),
            'bbox': micr_words[0]['bbox']
        }
    
    logger.info(f"Parsed {len(words)} words into cheque fields")
    return result
