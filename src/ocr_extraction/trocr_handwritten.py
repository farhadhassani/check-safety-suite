"""
TrOCR for handwritten text recognition (CPU-only)
"""
import numpy as np
import cv2
from typing import Tuple
import logging
from PIL import Image

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.warning("Transformers/torch not available for TrOCR")

logger = logging.getLogger(__name__)


class TrOCRHandwritten:
    """TrOCR for handwritten text recognition on CPU"""
    
    def __init__(self):
        if not TROCR_AVAILABLE:
            raise ImportError("transformers and torch required for TrOCR")
        
        logger.info("Loading TrOCR handwritten model (CPU)...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Force CPU
        self.model = self.model.to('cpu')
        self.model.eval()
        
        logger.info("TrOCR loaded successfully")
    
    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize handwritten text from a line image
        
        Args:
            image: np.ndarray, cropped line image (BGR or RGB)
        
        Returns:
            (text, confidence) tuple
        """
        try:
            # Convert to PIL RGB
            if len(image.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(image).convert('RGB')
            elif image.shape[2] == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image)
            
            # Process input
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to('cpu')
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Confidence is approximate (not directly available from generation)
            # Use sequence length as proxy
            confidence = min(0.7 + len(generated_text) * 0.01, 0.95)
            
            return generated_text, confidence
            
        except Exception as e:
            logger.error(f"TrOCR recognition failed: {e}")
            return "", 0.0


# Global instance
_trocr_instance = None

def get_trocr_instance():
    """Get or create global TrOCR instance"""
    global _trocr_instance
    if _trocr_instance is None:
        _trocr_instance = TrOCRHandwritten()
    return _trocr_instance


def recognize_handwritten_line(image: np.ndarray) -> Tuple[str, float]:
    """
    Recognize handwritten text from a cropped line image
    
    Args:
        image: np.ndarray, cropped line (e.g., amount in words)
    
    Returns:
        (text, confidence) tuple
    """
    if not TROCR_AVAILABLE:
        logger.warning("TrOCR not available, returning empty")
        return "", 0.0
    
    try:
        trocr = get_trocr_instance()
        return trocr.recognize(image)
    except Exception as e:
        logger.error(f"Handwritten recognition failed: {e}")
        return "", 0.0
