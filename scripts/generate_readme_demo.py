import cv2
import sys
import shutil
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.check_pipeline import run_check

def generate_demo_image():
    # Input image (the one that needs rotation)
    input_path = "data/BCSD/TestSet/X/X_013.jpeg"
    
    # Load image
    print(f"Loading {input_path}...")
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not load image")
        return

    # Run pipeline (this will now auto-rotate thanks to our fix)
    print("Running pipeline...")
    # We pass a dummy mask just to satisfy the function if needed, or None
    # run_check handles seg_mask=None gracefully
    out, out_img_path, out_json_path = run_check(
        img, 
        seg_mask=None, 
        out_dir="outputs/demo_gen", 
        fname="readme_demo", 
        include_text=True
    )
    
    print(f"Pipeline finished. Output saved to {out_img_path}")
    
    # Move/Rename to README location
    target_path = "outputs/demo/ocr_extraction_demo.png"
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(out_img_path, target_path)
    print(f"Updated README image at: {target_path}")

if __name__ == "__main__":
    generate_demo_image()
