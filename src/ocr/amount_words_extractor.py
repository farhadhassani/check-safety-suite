"""
amount_words_extractor.py
Extract and parse written check amounts ("One Hundred Twenty-Five Dollars")
"""
import cv2
import numpy as np
import re
from typing import Dict, Optional, Tuple

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from word2number import w2n
    W2N_AVAILABLE = True
except ImportError:
    W2N_AVAILABLE = False


class AmountWordsExtractor:
    """Extract and parse written check amounts"""
    
    def __init__(self):
        """Initialize extractor with number word mappings"""
        self.number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
            'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000, 'million': 1000000
        }
        
        # Common variations/misspellings
        self.word_corrections = {
            'fourty': 'forty',
            'fourthy': 'forty',
            'hunderd': 'hundred',
            'hundered': 'hundred'
        }
    
    def extract_amount_words(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Extract written amount from check
        
        Args:
            image: Check image (BGR format)
            roi: Region of interest (x, y, w, h) for amount line
        
        Returns:
            {
                'text': str,  # Cleaned text
                'amount': float or None,
                'confidence': float,
                'raw_text': str
            }
        """
        # Crop to amount line region
        if roi:
            x, y, w, h = roi
            words_img = image[y:y+h, x:x+w]
        else:
            words_img = self._detect_amount_line(image)
        
        # Preprocess
        processed = self._preprocess_amount_line(words_img)
        
        # OCR with text mode
        if not TESSERACT_AVAILABLE:
            return {
                'text': '',
                'amount': None,
                'confidence': 0.0,
                'raw_text': 'Tesseract not available'
            }
        
        try:
            text = pytesseract.image_to_string(
                processed,
                config='--psm 6'  # Assume uniform block of text
            )
        except Exception as e:
            return {
                'text': '',
                'amount': None,
                'confidence': 0.0,
                'raw_text': f'OCR error: {str(e)}'
            }
        
        # Clean and parse
        cleaned = self._clean_amount_text(text)
        
        # Convert to number
        amount, confidence = self._words_to_number(cleaned)
        
        return {
            'text': cleaned,
            'amount': amount,
            'confidence': confidence,
            'raw_text': text.strip()
        }
    
    def _detect_amount_line(self, image: np.ndarray) -> np.ndarray:
        """Detect the written amount line (middle of check typically)"""
        h, w = image.shape[:2]
        # Typical amount line: left 85% of width, middle height
        x_start = int(w * 0.05)
        y_start = int(h * 0.55)
        x_end = int(w * 0.90)
        y_end = int(h * 0.73)
        
        return image[y_start:y_end, x_start:x_end]
    
    def _preprocess_amount_line(self, img: np.ndarray) -> np.ndarray:
        """Preprocess for text OCR"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Threshold
        _, thresh = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return thresh
    
    def _clean_amount_text(self, text: str) -> str:
        """Clean OCR text for parsing"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^a-z0-9\s/-]', '', text)
        
        # Apply corrections
        words = text.split()
        corrected = [self.word_corrections.get(w, w) for w in words]
        text = ' '.join(corrected)
        
        # Remove filler words
        text = text.replace('and', ' ')
        text = text.replace('dollars', '')
        text = text.replace('cents', '/100')
        text = text.replace('only', '')
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _words_to_number(self, text: str) -> Tuple[Optional[float], float]:
        """
        Convert written amount to number
        
        Returns:
            (amount, confidence)
        """
        if not text:
            return None, 0.0
        
        # Try word2number library first (if available)
        if W2N_AVAILABLE:
            try:
                # Handle cents notation (e.g., "50/100")
                if '/100' in text:
                    parts = text.split('/100')
                    dollars = w2n.word_to_num(parts[0].strip())
                    cents = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0
                    return float(dollars) + (cents / 100.0), 0.90
                else:
                    amount = w2n.word_to_num(text)
                    return float(amount), 0.90
            except Exception:
                pass
        
        # Fallback: manual parsing
        amount, confidence = self._manual_parse(text)
        return amount, confidence
    
    def _manual_parse(self, text: str) -> Tuple[Optional[float], float]:
        """Manual parsing for complex amounts"""
        # Handle cents notation first
        dollars = 0.0
        cents = 0.0
        confidence = 0.7
        
        if '/100' in text:
            parts = text.split('/100')
            # Parse dollars part (before /100)
            dollars_text = parts[0].strip()
            dollars = self._parse_number_words(dollars_text)
            
            # Parse cents if present (after /100)
            if len(parts) > 1 and parts[1].strip():
                cents_text = parts[1].strip().split()[0] if parts[1].strip() else ''
                if cents_text.isdigit():
                    cents = int(cents_text)
                else:
                    cents = self._parse_number_words(cents_text)
            
            total = dollars + (cents / 100.0)
            return total if total > 0 else None, confidence
        
        # Parse as full amount
        amount = self._parse_number_words(text)
        return amount if amount > 0 else None, confidence
    
    def _parse_number_words(self, text: str) -> float:
        """
        Parse number words to value
        
        Example: "one hundred twenty five" -> 125.0
        """
        words = text.split()
        total = 0.0
        current = 0.0
        
        for word in words:
            word = word.strip()
            
            if word in self.number_words:
                value = self.number_words[word]
                
                if value >= 100:
                    # Multiplier (hundred, thousand, million)
                    if current == 0:
                        current = 1  # Handle "hundred" without preceding number
                    current *= value
                    
                    if value >= 1000:
                        total += current
                        current = 0
                else:
                    # Regular number
                    current += value
        
        return total + current
    
    def _calculate_confidence(self, raw_text: str, parsed_amount: Optional[float]) -> float:
        """Calculate confidence based on text quality and parsing success"""
        if parsed_amount is None:
            return 0.0
        
        # Check text length (reasonable amount strings are 10-100 chars)
        text_len = len(raw_text.strip())
        if text_len < 5 or text_len > 150:
            return 0.5
        
        # Count number words found
        words = raw_text.lower().split()
        number_word_count = sum(1 for w in words if w in self.number_words or w in self.word_corrections)
        
        # More number words = higher confidence
        if number_word_count >= 3:
            return 0.85
        elif number_word_count >= 2:
            return 0.75
        elif number_word_count >= 1:
            return 0.65
        else:
            return 0.50


# Convenience function for backward compatibility
def extract_amount_words(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    Simple wrapper for backward compatibility
    
    Returns:
        Written amount text or placeholder
    """
    extractor = AmountWordsExtractor()
    result = extractor.extract_amount_words(image, roi)
    
    if result['text']:
        return result['text']
    
    return "Unknown Amount"
