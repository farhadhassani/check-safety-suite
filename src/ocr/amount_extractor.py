"""
amount_extractor.py
Enhanced numerical amount extraction from bank checks using multi-OCR approach
"""
import cv2
import numpy as np
import re
from typing import Dict, Optional, Tuple
from pathlib import Path

# Try multiple OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class AmountExtractor:
    """Extract numerical check amounts using multi-OCR approach"""
    
    def __init__(self, use_ensemble: bool = True):
        """
        Initialize amount extractor
        
        Args:
            use_ensemble: Use multiple OCR engines for consensus
        """
        self.use_ensemble = use_ensemble and EASYOCR_AVAILABLE
        
        # Initialize EasyOCR if available and requested
        self.easyocr_reader = None
        if self.use_ensemble:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception:
                self.use_ensemble = False
    
    def extract_amount_digits(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Extract numerical amount from check
        
        Args:
            image: Check image (BGR format)
            roi: Region of interest (x, y, w, h) for amount box
        
        Returns:
            {
                'amount': float or None,
                'raw_text': str,
                'confidence': float,
                'currency': str,
                'method': str
            }
        """
        # Crop to amount region if ROI provided
        if roi:
            x, y, w, h = roi
            amount_img = image[y:y+h, x:x+w]
        else:
            # Use top-right region (typical amount box location)
            amount_img = self._detect_amount_box(image)
        
        # Preprocess for better OCR
        processed = self._preprocess_amount_box(amount_img)
        
        # Extract using available methods
        results = []
        
        # Method 1: Tesseract with digits config
        if TESSERACT_AVAILABLE:
            tess_result = self._tesseract_extract_amount(processed)
            results.append(tess_result)
        
        # Method 2: EasyOCR (better for varying fonts)
        if self.use_ensemble and self.easyocr_reader:
            easy_result = self._easyocr_extract_amount(processed)
            results.append(easy_result)
        
        # Resolve consensus or select best result
        final_result = self._resolve_consensus(results)
        
        return final_result
    
    def _detect_amount_box(self, image: np.ndarray) -> np.ndarray:
        """Auto-detect amount box region (top-right typically)"""
        h, w = image.shape[:2]
        # Typical amount box: top-right 40% width, top 20% height
        x_start = int(w * 0.55)
        y_start = int(h * 0.42)
        x_end = int(w * 0.95)
        y_end = int(h * 0.60)
        
        return image[y_start:y_end, x_start:x_end]
    
    def _preprocess_amount_box(self, img: np.ndarray) -> np.ndarray:
        """Optimize image for amount OCR"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 10
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _tesseract_extract_amount(self, img: np.ndarray) -> Dict:
        """Tesseract with amount-specific config"""
        if not TESSERACT_AVAILABLE:
            return {'amount': None, 'confidence': 0.0, 'source': 'tesseract', 'raw_text': ''}
        
        try:
            # Config for digits, currency symbols, decimal points
            config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789$,.'
            
            text = pytesseract.image_to_string(img, config=config)
            amount, confidence = self._parse_amount_text(text)
            
            return {
                'amount': amount,
                'confidence': confidence,
                'source': 'tesseract',
                'raw_text': text.strip()
            }
        except Exception as e:
            return {'amount': None, 'confidence': 0.0, 'source': 'tesseract', 'raw_text': '', 'error': str(e)}
    
    def _easyocr_extract_amount(self, img: np.ndarray) -> Dict:
        """EasyOCR extraction"""
        if not self.easyocr_reader:
            return {'amount': None, 'confidence': 0.0, 'source': 'easyocr', 'raw_text': ''}
        
        try:
            results = self.easyocr_reader.readtext(img)
            
            # Combine all detected text
            text = ' '.join([result[1] for result in results])
            
            # Use average confidence
            avg_conf = np.mean([result[2] for result in results]) if results else 0.0
            
            amount, _ = self._parse_amount_text(text)
            
            return {
                'amount': amount,
                'confidence': avg_conf if amount else 0.0,
                'source': 'easyocr',
                'raw_text': text
            }
        except Exception as e:
            return {'amount': None, 'confidence': 0.0, 'source': 'easyocr', 'raw_text': '', 'error': str(e)}
    
    def _parse_amount_text(self, text: str) -> Tuple[Optional[float], float]:
        """
        Parse amount from OCR text
        
        Returns:
            (amount, confidence)
        """
        if not text:
            return None, 0.0
        
        # Clean text
        text = text.strip().replace('$', '').replace(',', '').replace(' ', '')
        
        # Pattern 1: Standard decimal amount (e.g., "125.50")
        pattern1 = r'(\d+\.\d{2})'
        match1 = re.search(pattern1, text)
        
        if match1:
            try:
                amount = float(match1.group(1))
                return amount, 0.95
            except ValueError:
                pass
        
        # Pattern 2: Amount without cents (e.g., "125")
        pattern2 = r'(\d+)'
        match2 = re.search(pattern2, text)
        
        if match2:
            try:
                amount = float(match2.group(1))
                return amount, 0.75  # Lower confidence without decimal
            except ValueError:
                pass
        
        # Pattern 3: With asterisks/check protection (e.g., "***125.50")
        pattern3 = r'\*+(\d+\.?\d*)'
        match3 = re.search(pattern3, text)
        
        if match3:
            try:
                amount = float(match3.group(1))
                return amount, 0.85
            except ValueError:
                pass
        
        return None, 0.0
    
    def _resolve_consensus(self, results: list) -> Dict:
        """
        Resolve consensus from multiple OCR results
        
        Args:
            results: List of extraction results from different engines
        
        Returns:
            Best result based on confidence and consensus
        """
        if not results:
            return {
                'amount': None,
                'raw_text': '',
                'confidence': 0.0,
                'currency': 'USD',
                'method': 'none'
            }
        
        # Filter out failed extractions
        valid_results = [r for r in results if r.get('amount') is not None]
        
        if not valid_results:
            # Return the result with highest confidence even if no amount extracted
            best = max(results, key=lambda x: x.get('confidence', 0.0))
            return {
                'amount': None,
                'raw_text': best.get('raw_text', ''),
                'confidence': 0.0,
                'currency': 'USD',
                'method': best.get('source', 'unknown')
            }
        
        # If only one valid result, return it
        if len(valid_results) == 1:
            result = valid_results[0]
            return {
                'amount': result['amount'],
                'raw_text': result.get('raw_text', ''),
                'confidence': result['confidence'],
                'currency': 'USD',
                'method': result['source']
            }
        
        # Check for consensus (amounts agree within $0.01)
        amounts = [r['amount'] for r in valid_results]
        if len(set(round(a, 2) for a in amounts)) == 1:
            # Perfect consensus - boost confidence
            result = max(valid_results, key=lambda x: x['confidence'])
            return {
                'amount': result['amount'],
                'raw_text': result.get('raw_text', ''),
                'confidence': min(result['confidence'] + 0.1, 1.0),  # Boost for consensus
                'currency': 'USD',
                'method': 'consensus'
            }
        
        # No consensus - return highest confidence result
        best = max(valid_results, key=lambda x: x['confidence'])
        return {
            'amount': best['amount'],
            'raw_text': best.get('raw_text', ''),
            'confidence': best['confidence'] * 0.8,  # Reduce confidence due to disagreement
            'currency': 'USD',
            'method': best['source']
        }


# Convenience function for backward compatibility
def extract_amount_digits(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    Simple wrapper for backward compatibility
    
    Returns:
        Formatted amount string (e.g., "125.50") or placeholder
    """
    extractor = AmountExtractor(use_ensemble=False)
    result = extractor.extract_amount_digits(image, roi)
    
    if result['amount'] is not None:
        return f"{result['amount']:.2f}"
    
    return "UNKNOWN"
