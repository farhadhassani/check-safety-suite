"""
demo_ocr_extraction.py
Demonstration of OCR field extraction from bank checks
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ocr.amount_extractor import AmountExtractor
from src.ocr.amount_words_extractor import AmountWordsExtractor
from src.ocr.field_ocr import extract_date, extract_payee
from src.ocr.micr_ocr import read_micr_region
from src.validation import AmountValidator


def demo_field_extraction(image_path: str):
    """
    Demonstrate complete field extraction from a check
    
    Args:
        image_path: Path to check image
    """
    print("=" * 70)
    print("CHECK OCR EXTRACTION DEMO")
    print("=" * 70)
    
    # Load image
    print(f"\n📄 Loading check image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h} pixels")
    
    # Define ROIs (regions of interest) - typical check layout
    rois = {
        'date': (int(0.55*w), 0, int(0.40*w), int(0.20*h)),
        'payee': (int(0.05*w), int(0.23*h), int(0.80*w), int(0.18*h)),
        'amount_digits': (int(0.55*w), int(0.42*h), int(0.40*w), int(0.18*h)),
        'amount_words': (int(0.05*w), int(0.55*h), int(0.85*w), int(0.18*h)),
    }
    
    print("\n" + "=" * 70)
    print("EXTRACTING FIELDS")
    print("=" * 70)
    
    # 1. Extract MICR (routing/account/check number)
    print("\n1️⃣  MICR LINE EXTRACTION")
    print("-" * 70)
    routing, account, check_num, aba_valid = read_micr_region(image)
    print(f"   Routing Number:  {routing}")
    print(f"   Account Number:  {account}")
    print(f"   Check Number:    {check_num}")
    print(f"   ABA Checksum:    {'✓ Valid' if aba_valid else '✗ Invalid'}")
    
    # 2. Extract Date
    print("\n2️⃣  DATE EXTRACTION")
    print("-" * 70)
    x, y, w_roi, h_roi = rois['date']
    date_roi = image[y:y+h_roi, x:x+w_roi]
    date = extract_date(date_roi)
    print(f"   Extracted Date:  {date}")
    
    # 3. Extract Payee
    print("\n3️⃣  PAYEE EXTRACTION")
    print("-" * 70)
    x, y, w_roi, h_roi = rois['payee']
    payee_roi = image[y:y+h_roi, x:x+w_roi]
    payee = extract_payee(payee_roi)
    print(f"   Payee Name:      {payee}")
    
    # 4. Extract Amount (Numerical)
    print("\n4️⃣  AMOUNT (DIGITS) EXTRACTION")
    print("-" * 70)
    amount_extractor = AmountExtractor(use_ensemble=False)  # Single OCR for demo
    x, y, w_roi, h_roi = rois['amount_digits']
    amount_roi = image[y:y+h_roi, x:x+w_roi]
    
    amount_result = amount_extractor.extract_amount_digits(amount_roi)
    print(f"   Amount:          ${amount_result['amount']:.2f}" if amount_result['amount'] else "   Amount:          Not detected")
    print(f"   Raw Text:        '{amount_result['raw_text']}'")
    print(f"   Confidence:      {amount_result['confidence']:.1%}")
    print(f"   Method:          {amount_result['method']}")
    
    # 5. Extract Amount (Words)
    print("\n5️⃣  AMOUNT (WORDS) EXTRACTION")
    print("-" * 70)
    words_extractor = AmountWordsExtractor()
    x, y, w_roi, h_roi = rois['amount_words']
    words_roi = image[y:y+h_roi, x:x+w_roi]
    
    words_result = words_extractor.extract_amount_words(words_roi)
    print(f"   Written Amount:  {words_result['text']}")
    print(f"   Parsed Value:    ${words_result['amount']:.2f}" if words_result['amount'] else "   Parsed Value:    Not detected")
    print(f"   Confidence:      {words_result['confidence']:.1%}")
    print(f"   Raw OCR:         '{words_result['raw_text'][:50]}...'")
    
    # 6. Validate Amount Consistency
    print("\n6️⃣  AMOUNT CONSISTENCY VALIDATION")
    print("-" * 70)
    validator = AmountValidator()
    validation = validator.validate_amount_consistency(
        amount_result['amount'],
        words_result['amount']
    )
    
    print(f"   Numerical:       ${validation['digits']:.2f}" if validation['digits'] else "   Numerical:       N/A")
    print(f"   Written:         ${validation['words']:.2f}" if validation['words'] else "   Written:         N/A")
    print(f"   Consistent:      {'✓ Yes' if validation['consistent'] else '✗ No'}")
    print(f"   Difference:      ${validation['difference']:.2f}" if validation['difference'] is not None else "   Difference:      N/A")
    print(f"   Fraud Risk:      {validation['fraud_risk'].upper()}")
    print(f"   Message:         {validation['message']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    summary = {
        'Check Number': check_num,
        'Date': date,
        'Payee': payee,
        'Amount': f"${amount_result['amount']:.2f}" if amount_result['amount'] else "N/A",
        'Routing': routing,
        'Valid': '✓' if aba_valid and validation.get('consistent', False) else '✗'
    }
    
    for key, value in summary.items():
        print(f"   {key:15} {value}")
    
    print("\n" + "=" * 70)
    
    return {
        'micr': {'routing': routing, 'account': account, 'check_number': check_num, 'aba_valid': aba_valid},
        'date': date,
        'payee': payee,
        'amount_digits': amount_result,
        'amount_words': words_result,
        'validation': validation
    }


def create_synthetic_test_case():
    """
    Create a synthetic check image for testing when no real check is available
    """
    print("\n🎨 Creating synthetic check for demonstration...")
    
    # Create blank check-like image
    img = np.ones((600, 1200, 3), dtype=np.uint8) * 255
    
    # Add some text regions for demonstration
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Date region
    cv2.putText(img, "11/23/2025", (700, 80), font, 1, (0, 0, 0), 2)
    
    # Payee region
    cv2.putText(img, "John Doe", (80, 200), font, 1, (0, 0, 0), 2)
    
    # Amount box
    cv2.putText(img, "$125.50", (700, 300), font, 1.2, (0, 0, 0), 2)
    
    # Amount words
    cv2.putText(img, "One Hundred Twenty-Five and 50/100", (80, 400), font, 0.8, (0, 0, 0), 2)
    
    # MICR line (bottom)
    cv2.putText(img, "A123456789A 987654321A 1001A", (100, 550), font, 0.7, (0, 0, 0), 2)
    
    # Save synthetic check
    output_path = "outputs/demo/synthetic_check.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    
    print(f"✓ Synthetic check created: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo OCR extraction from bank checks")
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to check image (will create synthetic if not provided)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Force creation of synthetic check'
    )
    
    args = parser.parse_args()
    
    # Determine image to use
    if args.synthetic or args.image is None:
        # Create synthetic check
        image_path = create_synthetic_test_case()
    else:
        image_path = args.image
        
        # Check if file exists
        if not Path(image_path).exists():
            print(f"\n⚠️  Warning: Image not found: {image_path}")
            print("Creating synthetic check instead...")
            image_path = create_synthetic_test_case()
    
    # Run extraction demo
    try:
        result = demo_field_extraction(image_path)
        
        print("\n✅ Extraction complete!")
        print("\nTo test with your own check image:")
        print(f"  python scripts/demo_ocr_extraction.py --image path/to/your/check.jpg")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
