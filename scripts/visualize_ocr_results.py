"""
visualize_ocr_results.py
Create side-by-side visualization of check image and extracted OCR fields
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ocr.amount_extractor import AmountExtractor
from src.ocr.amount_words_extractor import AmountWordsExtractor
from src.ocr.field_ocr import extract_date, extract_payee
from src.ocr.micr_ocr import read_micr_region
from src.validation import AmountValidator


def visualize_ocr_extraction(image_path: str, output_path: str = "outputs/demo/ocr_visualization.png"):
    """
    Create visualization showing check image with OCR results side-by-side
    
    Args:
        image_path: Path to check image
        output_path: Where to save visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Define ROIs
    rois = {
        'date': (int(0.55*w), 0, int(0.40*w), int(0.20*h)),
        'payee': (int(0.05*w), int(0.23*h), int(0.80*w), int(0.18*h)),
        'amount_digits': (int(0.55*w), int(0.42*h), int(0.40*w), int(0.18*h)),
        'amount_words': (int(0.05*w), int(0.55*h), int(0.85*w), int(0.18*h)),
    }
    
    # Extract fields
    print("Extracting fields...")
    
    # MICR
    routing, account, check_num, aba_valid = read_micr_region(image)
    
    # Date
    x, y, w_roi, h_roi = rois['date']
    date = extract_date(image[y:y+h_roi, x:x+w_roi])
    
    # Payee
    x, y, w_roi, h_roi = rois['payee']
    payee = extract_payee(image[y:y+h_roi, x:x+w_roi])
    
    # Amount (digits)
    amount_extractor = AmountExtractor(use_ensemble=False)
    x, y, w_roi, h_roi = rois['amount_digits']
    amount_result = amount_extractor.extract_amount_digits(image[y:y+h_roi, x:x+w_roi])
    
    # Amount (words)
    words_extractor = AmountWordsExtractor()
    x, y, w_roi, h_roi = rois['amount_words']
    words_result = words_extractor.extract_amount_words(image[y:y+h_roi, x:x+w_roi])
    
    # Validation
    validator = AmountValidator()
    validation = validator.validate_amount_consistency(
        amount_result.get('amount'),
        words_result.get('amount')
    )
    
    # Create figure
    fig = plt.figure(figsize=(18, 10), dpi=100)
    
    # Left side: Original image with ROI boxes
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image_rgb)
    ax1.set_title('Check Image with Detected Regions', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Draw ROI boxes
    colors = {
        'date': '#3498db',
        'payee': '#2ecc71',
        'amount_digits': '#e74c3c',
        'amount_words': '#f39c12'
    }
    
    for roi_name, (x, y, w_roi, h_roi) in rois.items():
        rect = Rectangle((x, y), w_roi, h_roi, 
                        linewidth=3, edgecolor=colors[roi_name],
                        facecolor='none', linestyle='--')
        ax1.add_patch(rect)
        
        # Add label
        ax1.text(x, y-5, roi_name.replace('_', ' ').title(),
                fontsize=10, fontweight='bold',
                color=colors[roi_name],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Right side: Extracted information
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Title
    ax2.text(0.5, 0.95, 'Extracted Information', 
            ha='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='black', linewidth=2))
    
    # MICR Section
    y_pos = 0.85
    ax2.text(0.05, y_pos, '🔢 MICR Line', fontsize=14, fontweight='bold', color='#34495e')
    y_pos -= 0.06
    
    micr_info = [
        f"Routing:  {routing}",
        f"Account:  {account}",
        f"Check #:  {check_num}",
        f"ABA:      {'✓ Valid' if aba_valid else '✗ Invalid'}"
    ]
    
    for info in micr_info:
        ax2.text(0.08, y_pos, info, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.5))
        y_pos -= 0.045
    
    # Date Section
    y_pos -= 0.03
    ax2.text(0.05, y_pos, '📅 Date', fontsize=14, fontweight='bold', color='#3498db')
    y_pos -= 0.06
    ax2.text(0.08, y_pos, date, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', alpha=0.8))
    
    # Payee Section
    y_pos -= 0.08
    ax2.text(0.05, y_pos, '👤 Payee', fontsize=14, fontweight='bold', color='#2ecc71')
    y_pos -= 0.06
    payee_display = payee if len(payee) < 30 else payee[:27] + "..."
    ax2.text(0.08, y_pos, payee_display, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f4e6', alpha=0.8))
    
    # Amount Section
    y_pos -= 0.08
    ax2.text(0.05, y_pos, '💰 Amount', fontsize=14, fontweight='bold', color='#e74c3c')
    y_pos -= 0.06
    
    amount_val = amount_result.get('amount')
    if amount_val:
        ax2.text(0.08, y_pos, f"Numerical: ${amount_val:.2f}", fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))
        ax2.text(0.55, y_pos, f"Conf: {amount_result['confidence']:.0%}", 
                fontsize=10, color='#c0392b')
    else:
        ax2.text(0.08, y_pos, "Numerical: Not detected", fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))
    
    y_pos -= 0.05
    words_val = words_result.get('amount')
    words_text = words_result.get('text', '')
    if words_val:
        ax2.text(0.08, y_pos, f"Written: ${words_val:.2f}", fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))
        ax2.text(0.55, y_pos, f"Conf: {words_result['confidence']:.0%}",
                fontsize=10, color='#c0392b')
        y_pos -= 0.04
        if words_text and len(words_text) > 0:
            text_display = words_text if len(words_text) < 35 else words_text[:32] + "..."
            ax2.text(0.08, y_pos, f'"{text_display}"', fontsize=9, style='italic', color='#7f8c8d')
    else:
        ax2.text(0.08, y_pos, "Written: Not detected", fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))
    
    # Validation Section
    y_pos -= 0.08
    ax2.text(0.05, y_pos, '✓ Validation', fontsize=14, fontweight='bold', color='#9b59b6')
    y_pos -= 0.06
    
    # Fraud risk color coding
    risk_colors = {
        'low': '#27ae60',
        'medium': '#f39c12',
        'high': '#e67e22',
        'critical': '#c0392b'
    }
    
    risk = validation.get('fraud_risk', 'unknown')
    risk_color = risk_colors.get(risk, '#95a5a6')
    
    ax2.text(0.08, y_pos, f"Consistent: {'Yes' if validation['consistent'] else 'No'}", 
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8daef', alpha=0.8))
    
    y_pos -= 0.05
    ax2.text(0.08, y_pos, f"Risk Level: {risk.upper()}", 
            fontsize=12, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=risk_color, alpha=0.9))
    
    y_pos -= 0.05
    if validation.get('difference') is not None:
        ax2.text(0.08, y_pos, f"Difference: ${validation['difference']:.2f}", 
                fontsize=11, color='#7f8c8d')
    
    # Status at bottom
    y_pos = 0.05
    status = "✅ CHECK VALID" if aba_valid and validation['consistent'] else "⚠️ REVIEW REQUIRED"
    status_color = '#27ae60' if aba_valid and validation['consistent'] else '#e67e22'
    
    ax2.text(0.5, y_pos, status, 
            ha='center', fontsize=16, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=status_color, edgecolor='black', linewidth=2))
    
    # Overall title
    fig.suptitle('Check Safety Suite: OCR Field Extraction Demo', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.close()
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize OCR extraction results")
    parser.add_argument('--image', type=str, required=True, help='Path to check image')
    parser.add_argument('--output', type=str, default='outputs/demo/ocr_visualization.png',
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    visualize_ocr_extraction(args.image, args.output)
    
    print("\n✅ Visualization complete!")
    print(f"View the result at: {args.output}")
