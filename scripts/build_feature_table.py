import os, json, argparse, cv2, torch
import pandas as pd
from tqdm import tqdm
from src.models.unet import UNet
from src.pipeline.check_pipeline import run_check
import numpy as np

def _infer_mask(model, img_bgr, img_size=1024):
    h,w=img_bgr.shape[:2]
    x=cv2.resize(img_bgr,(img_size,img_size))[:,:,::-1]/255.0
    x=torch.from_numpy(x.transpose(2,0,1)).float()[None]
    # Use cpu if cuda not available
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        logits=model(x)
        prob=torch.sigmoid(logits)[0,0].cpu().numpy()
    return cv2.resize((prob>0.5).astype("float32"),(w,h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_json", required=False, help="JSON with entries [{path,label},...]")
    ap.add_argument("--seg_weights", required=False) # Made optional for demo
    ap.add_argument("--out_csv", default="outputs/features/features.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(3, 1).to(device) # Changed input channels to 3 for RGB/BGR
    
    if args.seg_weights and os.path.exists(args.seg_weights):
        model.load_state_dict(torch.load(args.seg_weights, map_location=device))
    else:
        print("Warning: No weights loaded for segmentation model. Using random weights.")
        
    model.eval()

    label_map = {}
    if args.labels_json and os.path.exists(args.labels_json):
        lbls = json.load(open(args.labels_json))
        for e in lbls:
            label_map[os.path.abspath(e["path"])] = int(e["label"])

    rows=[]
    paths=[os.path.join(args.images_dir,p) for p in os.listdir(args.images_dir)
           if p.lower().endswith((".png",".jpg",".jpeg",".tif"))]
    for p in tqdm(paths, desc="features"):
        img=cv2.imread(p)
        if img is None: continue
        
        seg=_infer_mask(model, img)
        out,_,_ = run_check(img, seg_mask=seg, out_dir="outputs/_ft", fname=os.path.splitext(os.path.basename(p))[0])

        # transform features so that higher ~ more risk
        aba_invalid = 0 if out["aba_valid"] else 1
        amount_inconsistent = 0 if out["amount_consistent"] else 1
        sig_gap = max(0.0, 0.05 - (out["signature_coverage_pct"] or 0.0))  # expect ≥5% coverage
        de00 = out["ink_deltaE00_amount_vs_body"] if out["ink_deltaE00_amount_vs_body"] is not None else np.nan

        rows.append({
            "path": os.path.abspath(p),
            "label": label_map.get(os.path.abspath(p), None),
            "aba_invalid": aba_invalid,
            "amount_inconsistent": amount_inconsistent,
            "signature_gap": float(sig_gap),
            "deltaE00": float(de00) if de00==de00 else np.nan,
            "deltaE76": float(out["ink_deltaE76_amount_vs_body"] or np.nan)
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Saved", args.out_csv)

if __name__=="__main__":
    main()
