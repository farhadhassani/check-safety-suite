import os, json, argparse, random
import cv2
import numpy as np
from src.pipeline.check_pipeline import rois, deskew

def _shift_lab_inplace(bgr, box, dL=5, da=12, db=-5):
    x,y,w,h = box
    crop = bgr[y:y+h, x:x+w]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] = np.clip(lab[:,:,0] + (dL*255/100.0), 0, 255)
    lab[:,:,1] = np.clip(lab[:,:,1] + da, 0, 255)
    lab[:,:,2] = np.clip(lab[:,:,2] + db, 0, 255)
    bgr[y:y+h, x:x+w] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def _paste_digits(bgr, box, text=None):
    if text is None:
        text = f"{random.randint(1,9)}{random.randint(0,9)}{random.randint(0,9)}.{random.randint(0,9)}{random.randint(0,9)}"
    x,y,w,h = box
    overlay = bgr.copy()
    org = (x+int(0.1*w), y+int(0.65*h))
    cv2.putText(overlay, f"${text}", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20,20,20), 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, bgr, 0.3, 0, bgr)

def _jpeg_artifacts_inplace(bgr, q=30):
    enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    bgr[:,:,:] = dec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="Folder of clean IDRBT images")
    ap.add_argument("--out_dir", required=True, help="Output folder for tampered images")
    ap.add_argument("--n", type=int, default=100, help="How many tampered images to create")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imgs = [os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir)
            if f.lower().endswith((".png",".jpg",".jpeg",".tif"))]
    random.shuffle(imgs)
    labels = []

    for i, p in enumerate(imgs[:args.n]):
        img = cv2.imread(p)
        if img is None: continue
        img = deskew(img)
        boxes = rois(img)
        tampered = img.copy()

        # Strategy: shift ink color in amount-in-words; occasionally paste digits & add compression artifacts
        if random.random() < 0.9:
            _shift_lab_inplace(tampered, boxes["amt_w"], dL=random.uniform(-4,6),
                               da=random.uniform(8,16), db=random.uniform(-10,4))
        if random.random() < 0.4:
            _paste_digits(tampered, boxes["amt$"])
        if random.random() < 0.5:
            _jpeg_artifacts_inplace(tampered, q=random.randint(20,60))

        outp = os.path.join(args.out_dir, f"tamper_{i:04d}.jpg")
        cv2.imwrite(outp, tampered)
        labels.append({"path": outp, "label": 1, "source": p})
        # also keep a matching clean sample
        labels.append({"path": p, "label": 0, "source": p})

    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {len(labels)} labeled entries to labels.json")

if __name__ == "__main__":
    main()
