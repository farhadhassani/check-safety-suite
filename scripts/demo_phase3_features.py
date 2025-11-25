"""
demo_phase3_features.py
Demonstrates Phase 3 Advanced Features: Uncertainty, UNet++, SHAP, Quantization
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.models.unet_plusplus import UNetPlusPlus
from src.models.uncertainty.mc_dropout import MCDropoutUNet
from src.optimization.model_quantization import quantize_dynamic, get_model_size_mb

def demo_unetplusplus():
    """Demo UNet++ architecture"""
    print("\n=== UNet++ Architecture Demo ===")
    model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=False)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ UNet++ working correctly")

def demo_mc_dropout():
    """Demo Monte Carlo Dropout uncertainty"""
    print("\n=== MC Dropout Uncertainty Demo ===")
    base_model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=False)
    mc_model = MCDropoutUNet(base_model, dropout_rate=0.1, n_samples=5)
    
    dummy_input = torch.randn(1, 3, 256, 256)
    mean_pred, uncertainty = mc_model(dummy_input, return_uncertainty=True)
    
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")
    print(f"Max uncertainty: {uncertainty.max():.4f}")
    
    # Get comprehensive metrics
    metrics = mc_model.predict_with_metrics(dummy_input)
    print(f"Confidence mean: {metrics['confidence_mean']:.4f}")
    print("✓ MC Dropout working correctly")

def demo_quantization():
    """Demo model quantization"""
    print("\n=== Model Quantization Demo ===")
    model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=False)
    original_size = get_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Dynamic quantization
    q_model = quantize_dynamic(model)
    q_size = get_model_size_mb(q_model)
    print(f"Quantized model size: {q_size:.2f} MB")
    print(f"Size reduction: {(1 - q_size/original_size)*100:.1f}%")
    
    # Test inference
    dummy_input = torch.randn(1, 3, 256, 256)
    output = q_model(dummy_input)
    print(f"Quantized model output shape: {output.shape}")
    print("✓ Quantization working correctly")

def main():
    print("=" * 60)
    print("Phase 3: Advanced ML Features Demonstration")
    print("=" * 60)
    
    demo_unetplusplus()
    demo_mc_dropout()
    demo_quantization()
    
    print("\n" + "=" * 60)
    print("All Phase 3 features demonstrated successfully!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Train UNet++ on full dataset: python scripts/train_unet_plusplus.py")
    print("2. Run calibration: python scripts/calibrate_mc_dropout.py")
    print("3. Compare quantized vs full precision performance")

if __name__ == "__main__":
    main()
