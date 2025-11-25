import argparse, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
import matplotlib.pyplot as plt
import os

def _drop_na_labels(df):
    return df[~df["label"].isna()].reset_index(drop=True)

def _basic_checks(df):
    assert "label" in df.columns, "features CSV missing 'label'"
    assert {"aba_invalid","amount_inconsistent","signature_gap","deltaE00"}.issubset(df.columns)

def eval_rule(y, s):
    # Handle case with only one class
    if len(np.unique(y)) < 2:
        return {"auc": 0.0, "ap": 0.0, "fpr": [], "tpr": [], "thr": []}
        
    fpr, tpr, thr = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    return {"auc": float(auc), "ap": float(ap), "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}

def plot_curves(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    plt.figure()
    for name, r in results.items():
        fpr = np.array(r["fpr"]); tpr = np.array(r["tpr"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC: Single vs Fused"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"roc.png")); plt.close()

    # PR
    plt.figure()
    for name, r in results.items():
        # rebuild precision/recall
        # We'll compute from scores on the fly inside loop if needed
        pass  # keep simple; ROC often suffices for the paper figure

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="outputs/features/features.csv")
    ap.add_argument("--out_dir", default="outputs/ablation")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    _basic_checks(df)
    df = _drop_na_labels(df)
    
    if len(df) == 0:
        print("No labeled data found.")
        return

    y = df["label"].astype(int).values
    # Scores (higher = more risk)
    s_sig = df["signature_gap"].fillna(0).values
    s_ink = df["deltaE00"].fillna(0).values
    s_rule = (df["aba_invalid"].fillna(0) + df["amount_inconsistent"].fillna(0)).values

    # Fused model: logistic with Platt scaling
    X = df[["aba_invalid","amount_inconsistent","signature_gap","deltaE00"]].fillna(0).values
    
    # Check if we have enough classes for calibration
    if len(np.unique(y)) > 1:
        base = LogisticRegression(max_iter=1000)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=min(3, len(y)//2))
        clf.fit(X, y)
        s_fused = clf.predict_proba(X)[:,1]
    else:
        print("Warning: Only one class present. Skipping calibration.")
        s_fused = np.zeros_like(y, dtype=float)

    results = {
        "Signature only": eval_rule(y, s_sig),
        "Ink ΔE00 only":  eval_rule(y, s_ink),
        "Rule (ABA+Amount)": eval_rule(y, s_rule),
        "Fused (calibrated)": eval_rule(y, s_fused)
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Simple ROC plot
    import matplotlib.pyplot as plt
    plt.figure()
    for name, r in results.items():
        fpr = np.array(r["fpr"]); tpr = np.array(r["tpr"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0,1],[0,1],"--",color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC: Single Signals vs Fused"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"roc.png")); plt.close()

    # Calibration metric
    if len(np.unique(y)) > 1:
        brier = brier_score_loss(y, s_fused)
        with open(os.path.join(args.out_dir,"summary.txt"),"w") as f:
            f.write(f"Brier (Fused): {brier:.4f}\n")
    print("Saved ablation to", args.out_dir)

if __name__=="__main__":
    main()
