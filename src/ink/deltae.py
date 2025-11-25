import cv2
import numpy as np
# Patch colormath for numpy compatibility
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

from sklearn.cluster import KMeans
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def _writing_mask(gray):
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 12)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), 1)
    return thr

def _dominant_lab_center(bgr, bin_mask, k=2):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pts = lab[bin_mask>0].reshape(-1,3)
    if pts.shape[0] < 50:
        return None
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(pts)
    centers = km.cluster_centers_
    # Choose the most populous cluster as dominant
    labs = centers.astype(np.float32)
    # OpenCV’s L in [0,255], a/b in [0,255] with 128 center; convert to colormath scale
    L = labs[:,0] * (100.0/255.0)
    a = labs[:,1] - 128.0
    b = labs[:,2] - 128.0
    return np.array([L.mean(), a.mean(), b.mean()], dtype=np.float32)

def deltaE76(c1, c2):
    return float(np.linalg.norm(c1 - c2))

def deltaE00_from_centers(c1, c2):
    c1 = LabColor(lab_l=float(c1[0]), lab_a=float(c1[1]), lab_b=float(c1[2]))
    c2 = LabColor(lab_l=float(c2[0]), lab_a=float(c2[1]), lab_b=float(c2[2]))
    return float(delta_e_cie2000(c1, c2))

def region_deltaE_metrics(bgr, roi1, roi2, k=2):
    """Return (ΔE76, ΔE00) for writing strokes in two ROIs; None if insufficient pixels."""
    def roi(b): x,y,w,h=b; return bgr[y:y+h, x:x+w]
    r1 = roi(roi1); r2 = roi(roi2)
    g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY); m1 = _writing_mask(g1)
    g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY); m2 = _writing_mask(g2)
    c1 = _dominant_lab_center(r1, m1, k=k); c2 = _dominant_lab_center(r2, m2, k=k)
    if c1 is None or c2 is None:
        return None, None
    return deltaE76(c1, c2), deltaE00_from_centers(c1, c2)
