"""
Base OCR using docTR for CPU-only text detection and recognition
"""
import numpy as np
import cv2
from typing import List, Dict
import logging

# CPU-only docTR import
try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    logging.warning("docTR not available. Install with: pip install python-doctr[cpu]")

logger = logging.getLogger(__name__)


class DocTROCR:
    """CPU-only OCR using docTR with lightweight models"""
    
    def __init__(self):
        if not DOCTR_AVAILABLE:
            raise ImportError("docTR is required. Install with: pip install python-doctr[cpu]")
        
        # Initialize with light models for CPU
        logger.info("Initializing docTR with CPU-only light models...")
        self.predictor = ocr_predictor(
            det_arch='db_mobilenet_v3_large',    # Light detection model
            reco_arch='crnn_mobilenet_v3_large',  # Light recognition model
            pretrained=True
        )
        logger.info("docTR initialized successfully")
    
    def extract_words(self, image: np.ndarray) -> List[Dict]:
        """
        Extract words from image using docTR
        
        Args:
            image: np.ndarray, BGR or RGB image
        
        Returns:
            List of dicts with keys: text, confidence, bbox (normalized)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if likely BGR (OpenCV default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Run OCR
        result = self.predictor([image_rgb])
        
        # Extract words with bboxes
        words = []
        
        for page in result.pages:
            h, w = page.dimensions
           
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Get normalized bbox
                        bbox = word.geometry  # ((x_min, y_min), (x_max, y_max))
                        
                        words.append({
                            'text': word.value,
                            'confidence': float(word.confidence),
                            'bbox': (
                                float(bbox[0][0]),  # x_min
                                float(bbox[0][1]),  # y_min
                                float(bbox[1][0]),  # x_max
                                float(bbox[1][1])   # y_max
                            )
                        })
        
        logger.info(f"Extracted {len(words)} words from image")
        return words


# Global instance (lazy loading)
_ocr_instance = None

def get_ocr_instance():
    """Get or create global OCR instance"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = DocTROCR()
    return _ocr_instance


def run_raw_ocr(image: np.ndarray) -> List[Dict]:
    """
    Run OCR on image and return raw word tokens
    
    Args:
        image: np.ndarray, BGR or RGB cheque image
    
    Returns:
        List of dicts: {"text": str, "confidence": float, "bbox": (x_min, y_min, x_max, y_max)}
        where bbox is normalized to [0, 1]
    """
    if not DOCTR_AVAILABLE:
        logger.warning("docTR not available, returning empty list")
        return []
    
    try:
        ocr = get_ocr_instance()
        return ocr.extract_words(image)
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return []


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct image orientation (90, 180, 270 degrees)
    Uses a simple projection profile heuristic or docTR's orientation if available.
    For now, we implement a robust projection-based method for 90-degree rotations.
    """
    def get_score(img):
        # Calculate variance of projection profile
        # Text lines create high variance in horizontal projection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        h_proj = np.sum(thresh, axis=1)
        h_score = np.var(h_proj)
        
        # Vertical projection
        v_proj = np.sum(thresh, axis=0)
        v_score = np.var(v_proj)
        
        return h_score, v_score

    # Check 0 and 90 degrees
    h0, v0 = get_score(image)
    
    # Rotate 90
    img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    h90, v90 = get_score(img_90)
    
    # If horizontal variance of 90 deg is significantly higher than 0 deg, 
    # it means lines are horizontal in the 90 deg version.
    # Checks usually have strong horizontal lines (text lines, lines).
    
    logger.info(f"Orientation scores - 0 deg: {h0:.2f}, 90 deg: {h90:.2f}")
    
    if h90 > h0 * 1.2: # Significant difference
        logger.info("Detected vertical orientation, rotating 90 degrees clockwise")
        return img_90
        
    # Check 270 (which is 90 counter-clockwise)
    img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h270, v270 = get_score(img_270)
    
    if h270 > h0 * 1.2:
        logger.info("Detected vertical orientation, rotating 90 degrees counter-clockwise")
        return img_270

    return image

