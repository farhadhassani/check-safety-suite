import cv2, json, uuid, numpy as np
from pathlib import Path
from ..ocr.micr_ocr import read_micr_region
from ..ocr.field_ocr import extract_date, extract_payee, extract_amount_digits, extract_amount_words
from ..ink.deltae import region_deltaE_metrics

def deskew(img):
    g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(g,50,150)
    lines=cv2.HoughLines(edges,1,np.pi/180,200)
    ang=0.0
    if lines is not None:
        angs=[]
        for rho,theta in lines[:,0]:
            deg=theta*180/np.pi
            if 30<deg<150: angs.append(deg-90)
        if angs: ang = float(np.median(angs))
    h,w=g.shape
    M=cv2.getRotationMatrix2D((w//2,h//2), ang,1.0)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def rois(img):
    h,w=img.shape[:2]
    return {
        "date":   (int(0.55*w), 0,             int(0.45*w), int(0.20*h)),
        "payee":  (int(0.05*w), int(0.23*h),   int(0.80*w), int(0.18*h)),
        "amt$":   (int(0.55*w), int(0.42*h),   int(0.40*w), int(0.18*h)),
        "amt_w":  (int(0.05*w), int(0.55*h),   int(0.85*w), int(0.18*h)),
        "body":   (int(0.05*w), int(0.25*h),   int(0.80*w), int(0.45*h))
    }

def crop(img, box): x,y,w,h=box; return img[y:y+h, x:x+w].copy()

def run_check(img_bgr, seg_mask=None, out_dir="outputs/demo", fname="check", include_text=False):
    """
    Run check analysis pipeline.
    
    Args:
        img_bgr: BGR image
        seg_mask: Segmentation mask
        out_dir: Output directory
        fname: Filename prefix
        include_text: If True, include OCR text extraction
    
    Returns:
        (output_dict, annotated_image_path, json_path)
    """
    Path(out_dir,"annotated").mkdir(parents=True, exist_ok=True)
    Path(out_dir,"json").mkdir(parents=True, exist_ok=True)
    
    # Correct orientation (90 degree rotation)
    try:
        from ..ocr_extraction.base_ocr import correct_orientation
        img_bgr = correct_orientation(img_bgr)
    except ImportError:
        pass

    # Deskew (small angle correction)
    img = deskew(img_bgr)
    
    # Use corrected image for visualization
    vis = img.copy()

    routing, account, chk, aba_ok = read_micr_region(img)
    r = rois(img)
    date = extract_date(crop(img, r["date"]))
    payee= extract_payee(crop(img, r["payee"]))
    amt_d= extract_amount_digits(crop(img, r["amt$"]))
    amt_w= extract_amount_words(crop(img, r["amt_w"]))
    amt_consistent = bool(amt_d and amt_w and (amt_d.replace(",","")==amt_w.replace(",","")))

    de76, de00 = region_deltaE_metrics(img, r["amt_w"], r["body"])
    flags=[]
    if not aba_ok: flags.append("invalid_routing")
    if amt_d and amt_w and not amt_consistent: flags.append("amount_mismatch")
    # ΔE00 threshold tuned via synthetic ROC; initial sensible start ~12.0
    if de00 is not None and de00>12.0: flags.append("amount_ink_mismatch")

    for name,box in r.items():
        cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)
        cv2.putText(vis,name,(box[0],max(15,box[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    if seg_mask is not None:
        # Resize mask to match corrected image if needed (rotation changes dims)
        if seg_mask.shape[:2] != vis.shape[:2]:
            seg_mask = cv2.resize(seg_mask, (vis.shape[1], vis.shape[0]))
            
        m = (seg_mask>0.5).astype(np.uint8)*255
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 1.0, m, 0.4, 0)

    uid = uuid.uuid4().hex[:8]
    out_img = str(Path(out_dir,"annotated", f"{fname}_{uid}.png"))
    out_js  = str(Path(out_dir,"json", f"{fname}_{uid}.json"))
    cv2.imwrite(out_img, vis)
    
    out = {
        "routing_number": routing, "aba_valid": aba_ok,
        "account_number": account, "check_number": chk,
        "date": date, "payee": payee,
        "amount_digits": amt_d, "amount_words": amt_w,
        "amount_consistent": amt_consistent,
        "signature_coverage_pct": float(seg_mask.mean()) if seg_mask is not None else None,
        "ink_deltaE76_amount_vs_body": de76,
        "ink_deltaE00_amount_vs_body": de00,
        "flags": flags
    }
    
    # Add OCR text extraction if requested
    if include_text:
        try:
            from ..ocr_extraction import extract_cheque_text
            # Pass the already corrected image
            extracted_fields = extract_cheque_text(img)
            out["extracted_fields"] = extracted_fields
        except Exception as e:
            import logging
            logging.warning(f"Text extraction failed: {e}")
            out["extracted_fields"] = None
    
    with open(out_js, "w") as f: json.dump(out, f, indent=2)
    return out, out_img, out_js
