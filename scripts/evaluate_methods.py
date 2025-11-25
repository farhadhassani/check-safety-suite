"""
Comprehensive Evaluation Script
Compare deep learning model against traditional baselines
"""
import argparse
import json
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.baselines.traditional_methods import (
    ErrorLevelAnalysis,
    NoiseAnalysis,
    SIFTBasedDetection,
    LocalBinaryPatternAnalysis,
    TraditionalEnsemble
)


def load_model(checkpoint_path, device):
    """Load trained UNet model"""
    model = UNet(n_channels=3, n_classes=1).to(device)
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def infer_unet(model, img_bgr, device, img_size=512):
    """Run UNet inference"""
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (img_size, img_size))[:, :, ::-1] / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
    
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    return cv2.resize(prob, (w, h))


def evaluate_method(predictions, labels, method_name):
    """Calculate evaluation metrics"""
    # Convert to binary arrays
    preds = np.array(predictions)
    labs = np.array(labels)
    
    # Calculate metrics
    auc = roc_auc_score(labs, preds)
    ap = average_precision_score(labs, preds)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labs, preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions at optimal threshold
    binary_preds = (preds >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labs, binary_preds).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'method': method_name,
        'auc': auc,
        'ap': ap,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate detection methods')
    parser.add_argument('--labels_json', required=True, help='Path to labels.json')
    parser.add_argument('--model_checkpoint', default='outputs/training/best_model.pth')
    parser.add_argument('--output_dir', default='outputs/evaluation')
    parser.add_argument('--max_images', type=int, default=None, help='Limit number of images')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    unet_model = load_model(args.model_checkpoint, device)
    ela = ErrorLevelAnalysis()
    noise = NoiseAnalysis()
    sift = SIFTBasedDetection()
    lbp = LocalBinaryPatternAnalysis()
    traditional = TraditionalEnsemble()
    
    # Load dataset
    print(f"Loading dataset from {args.labels_json}")
    with open(args.labels_json, 'r') as f:
        labels_data = json.load(f)
    
    if args.max_images:
        labels_data = labels_data[:args.max_images]
    
    # Get project root
    labels_abs = Path(args.labels_json).absolute()
    project_root = labels_abs.parent.parent.parent
    
    # Storage for predictions
    results = {
        'unet': {'preds': [], 'labels': []},
        'ela': {'preds': [], 'labels': []},
        'noise': {'preds': [], 'labels': []},
        'sift': {'preds': [], 'labels': []},
        'lbp': {'preds': [], 'labels': []},
        'traditional_ensemble': {'preds': [], 'labels': []}
    }
    
    # Process images
    print(f"Processing {len(labels_data)} images...")
    for entry in tqdm(labels_data):
        # Load image
        img_path = entry['path']
        if not Path(img_path).is_absolute():
            img_path = project_root / img_path
        
        if not Path(img_path).exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read: {img_path}")
            continue
        
        label = entry['label']
        
        # UNet prediction
        try:
            unet_prob = infer_unet(unet_model, img, device)
            unet_score = float(unet_prob.max())
            results['unet']['preds'].append(unet_score)
            results['unet']['labels'].append(label)
        except Exception as e:
            print(f"UNet failed on {img_path}: {e}")
        
        # Traditional methods
        try:
            # ELA
            ela_map = ela.detect(img)
            results['ela']['preds'].append(float(ela_map.max()))
            results['ela']['labels'].append(label)
            
            # Noise
            noise_map = noise.detect(img)
            results['noise']['preds'].append(float(noise_map.max()))
            results['noise']['labels'].append(label)
            
            # SIFT
            sift_mask, sift_info = sift.detect(img)
            sift_score = 1.0 if sift_info['suspicious'] else sift_mask.max() / 255.0
            results['sift']['preds'].append(float(sift_score))
            results['sift']['labels'].append(label)
            
            # LBP
            lbp_map = lbp.detect(img)
            results['lbp']['preds'].append(float(lbp_map.max()))
            results['lbp']['labels'].append(label)
            
            # Traditional ensemble
            combined_map, _ = traditional.detect(img)
            results['traditional_ensemble']['preds'].append(float(combined_map.max()))
            results['traditional_ensemble']['labels'].append(label)
            
        except Exception as e:
            print(f"Traditional methods failed on {img_path}: {e}")
    
    # Evaluate all methods
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics_list = []
    for method_name, data in results.items():
        if len(data['preds']) > 0:
            metrics = evaluate_method(data['preds'], data['labels'], method_name)
            metrics_list.append(metrics)
            
            print(f"\n{method_name.upper()}")
            print(f"  AUC-ROC: {metrics['auc']:.4f}")
            print(f"  AP: {metrics['ap']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values('auc', ascending=False)
    metrics_csv_path = output_dir / 'method_comparison.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nMetrics saved to: {metrics_csv_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. AUC comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = metrics_df['method'].values
    aucs = metrics_df['auc'].values
    colors = ['#667eea' if m == 'unet' else '#f093fb' for m in methods]
    
    ax.barh(methods, aucs, color=colors)
    ax.set_xlabel('AUC-ROC Score', fontsize=12)
    ax.set_title('Method Comparison: AUC-ROC', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (method, auc) in enumerate(zip(methods, aucs)):
        ax.text(auc + 0.01, i, f'{auc:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'auc_comparison.png'}")
    
    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method_name, data in results.items():
        if len(data['preds']) > 0:
            fpr, tpr, _ = roc_curve(data['labels'], data['preds'])
            auc = roc_auc_score(data['labels'], data['preds'])
            ax.plot(fpr, tpr, label=f'{method_name} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Method Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'roc_curves.png'}")
    
    # 3. Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method_name, data in results.items():
        if len(data['preds']) > 0:
            precision, recall, _ = precision_recall_curve(data['labels'], data['preds'])
            ap = average_precision_score(data['labels'], data['preds'])
            ax.plot(recall, precision, label=f'{method_name} (AP={ap:.3f})', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pr_curves.png'}")
    
    # 4. Metrics heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = metrics_df[['method', 'auc', 'ap', 'f1', 'precision', 'recall']].set_index('method')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
    ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'metrics_heatmap.png'}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - method_comparison.csv")
    print(f"  - auc_comparison.png")
    print(f"  - roc_curves.png")
    print(f"  - pr_curves.png")
    print(f"  - metrics_heatmap.png")


if __name__ == "__main__":
    main()
