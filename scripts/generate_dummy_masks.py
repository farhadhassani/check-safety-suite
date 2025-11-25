"""
generate_dummy_masks.py
Creates a simple binary mask (all zeros) for each image in the dataset.
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np


def create_dummy_mask(image_path: Path, mask_path: Path):
    # Create a zero mask with same size as image
    with Image.open(image_path) as img:
        size = img.size
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)  # H, W
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_path)


def main():
    data_root = Path(os.getenv('DATA_ROOT', 'd:/Antigraph/check-safety-suite/data/IDRBT Cheque Image Dataset/300'))
    images_dir = data_root / 'images'
    masks_dir = data_root / 'masks'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Assuming images already exist in images_dir; if not, copy from source
    # For this example, we just generate masks for existing images
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in {'.tif', '.png', '.jpg', '.jpeg'}:
            mask_file = masks_dir / img_file.name
            if not mask_file.exists():
                create_dummy_mask(img_file, mask_file)
                print(f"Created mask for {img_file.name}")

if __name__ == "__main__":
    main()
