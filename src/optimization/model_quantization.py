"""
Model Quantization for Performance Optimization
This module provides utilities to quantize PyTorch models for faster inference and reduced memory usage.
Supports dynamic quantization and static quantization (post-training).
"""

import torch
import torch.nn as nn
import torch.quantization
import logging
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_dynamic(
    model: nn.Module,
    layers_to_quantize: Optional[set] = None,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic quantization to the model.
    
    Dynamic quantization quantizes weights ahead of time but activations are
    quantized dynamically at runtime. This is good for LSTM/RNNs or BERT,
    but less effective for CNNs than static quantization.
    
    Args:
        model: PyTorch model to quantize
        layers_to_quantize: Set of layer types to quantize (default: {nn.Linear, nn.LSTM, nn.GRU, nn.RNNCell})
        dtype: Target data type (default: torch.qint8)
        
    Returns:
        Quantized model
    """
    if layers_to_quantize is None:
        layers_to_quantize = {nn.Linear, nn.LSTM, nn.GRU, nn.RNNCell}
        
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=dtype
        )
        logger.info("Dynamic quantization applied successfully")
        return quantized_model
    except Exception as e:
        logger.error(f"Dynamic quantization failed: {e}")
        raise

def quantize_static(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Apply static quantization (Post-Training Quantization).
    
    Static quantization quantizes both weights and activations. It requires
    calibration with a representative dataset to determine activation ranges.
    Best for CNNs.
    
    Args:
        model: PyTorch model to quantize
        calibration_loader: DataLoader with calibration data
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        Quantized model
    """
    try:
        model.eval()
        
        # 1. Set qconfig
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # 2. Fuse modules (optional but recommended)
        # We try to fuse common patterns like Conv+BN+ReLU
        # This requires knowing the model structure, so we'll do a generic attempt
        # or skip if too complex to automate safely without model knowledge
        # For now, we skip automatic fusion to avoid errors on unknown architectures
        
        # 3. Prepare
        model_prepared = torch.quantization.prepare(model)
        
        # 4. Calibrate
        logger.info("Starting calibration...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model_prepared(inputs)
                if i >= 10: # Limit calibration batches
                    break
                    
        # 5. Convert
        quantized_model = torch.quantization.convert(model_prepared)
        logger.info("Static quantization applied successfully")
        
        return quantized_model
        
    except Exception as e:
        logger.error(f"Static quantization failed: {e}")
        raise

def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

if __name__ == "__main__":
    # Example usage
    print("Model Quantization - Example Usage")
    
    # Create dummy model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.fc = nn.Linear(16 * 2 * 2, 10)
            
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
            
    model = SimpleModel()
    print(f"Original size: {get_model_size_mb(model):.4f} MB")
    
    # Dynamic Quantization
    q_model = quantize_dynamic(model)
    print(f"Quantized size (dynamic): {get_model_size_mb(q_model):.4f} MB")
