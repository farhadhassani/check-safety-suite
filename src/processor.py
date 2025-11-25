import cv2
import numpy as np
from PIL import Image

from src.micr import extract_micr
from src.validation import validate_aba_routing, validate_check_date, validate_amount_consistency

def process_check(image_pil, use_donut=False):
    """
    Main entry point for check processing.
    Args:
        image_pil: PIL Image object
        use_donut: Boolean to enable Donut fallback
    Returns:
        dict: Structured JSON result
        image: Annotated image (PIL or numpy)
    """
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 1. Detect Check Region (Crop if necessary)
    # For this demo, we assume the image is mostly the check.
    # TODO: Add contour detection to crop if background is present.
    
    # 2. Extract MICR
    micr_result = extract_micr(img_cv)
    
    # 3. Extract Other Fields (Date, Amount, Payee)
    # Placeholder for OCR/Donut
    fields = {
        "date": "2023-10-27", # Mock
        "amount_numeric": "100.00", # Mock
        "amount_words": "One Hundred Dollars", # Mock
        "payee": "John Doe" # Mock
    }
    
    # 4. Validate
    validation_results = {
        "routing_checksum": False,
        "date_valid": True,
        "amount_match": True,
        "signature_detected": False # Placeholder
    }
    
    if micr_result["parsed"]["routing"]:
        validation_results["routing_checksum"] = validate_aba_routing(micr_result["parsed"]["routing"])
        
    # 5. Triage
    triage_verdict = "review"
    if validation_results["routing_checksum"] and validation_results["date_valid"]:
        triage_verdict = "approve"
    
    return {
        "status": "processed",
        "micr": micr_result,
        "fields": fields,
        "validation": validation_results,
        "triage": triage_verdict
    }, img_cv
