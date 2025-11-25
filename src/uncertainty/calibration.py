"""
Model Calibration for Reliable Confidence Scores
Includes reliability diagrams, ECE, and temperature scaling
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import cv2


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for model calibration
    
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """
        Scale logits by temperature
        
        Args:
            logits: Model logits (before sigmoid)
        
        Returns:
            Scaled logits
        """
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Fit temperature parameter on validation set
        
        Args:
            logits: Model logits (N, 1, H, W)
            labels: Ground truth labels (N, 1, H, W)
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.BCEWithLogitsLoss()(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()


class CalibrationAnalyzer:
    """
    Analyze and visualize model calibration
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for reliability diagram
        """
        self.n_bins = n_bins
    
    def compute_ece(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = None
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            predictions: Model predictions [0, 1]
            labels: Ground truth labels {0, 1}
            n_bins: Number of bins
        
        Returns:
            ece: Expected Calibration Error
        """
        if n_bins is None:
            n_bins = self.n_bins
        
        # Flatten arrays
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = labels[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = predictions[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def compute_mce(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = None
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE)
        
        Args:
            predictions: Model predictions [0, 1]
            labels: Ground truth labels {0, 1}
            n_bins: Number of bins
        
        Returns:
            mce: Maximum Calibration Error
        """
        if n_bins is None:
            n_bins = self.n_bins
        
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def reliability_diagram(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = None,
        title: str = "Reliability Diagram"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create reliability diagram
        
        Args:
            predictions: Model predictions [0, 1]
            labels: Ground truth labels {0, 1}
            save_path: Path to save figure
            title: Plot title
        
        Returns:
            bin_centers: Bin centers
            accuracies: Accuracy in each bin
        """
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        
        accuracies = []
        confidences = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            count = in_bin.sum()
            counts.append(count)
            
            if count > 0:
                accuracy = labels[in_bin].mean()
                confidence = predictions[in_bin].mean()
            else:
                accuracy = 0
                confidence = (bin_lower + bin_upper) / 2
            
            accuracies.append(accuracy)
            confidences.append(confidence)
        
        accuracies = np.array(accuracies)
        confidences = np.array(confidences)
        counts = np.array(counts)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.bar(bin_centers, accuracies, width=1/self.n_bins, 
                alpha=0.7, edgecolor='black', label='Accuracy')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Add ECE and MCE
        ece = self.compute_ece(predictions.reshape(-1, 1), labels.reshape(-1, 1))
        mce = self.compute_mce(predictions.reshape(-1, 1), labels.reshape(-1, 1))
        ax1.text(0.05, 0.95, f'ECE: {ece:.4f}\nMCE: {mce:.4f}',
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Histogram of predictions
        ax2.hist(predictions, bins=self.n_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Reliability diagram saved to {save_path}")
        
        plt.close()
        
        return bin_centers, accuracies
    
    def calibration_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Compute all calibration metrics
        
        Args:
            predictions: Model predictions [0, 1]
            labels: Ground truth labels {0, 1}
        
        Returns:
            metrics: Dictionary of calibration metrics
        """
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # ECE and MCE
        ece = self.compute_ece(predictions, labels)
        mce = self.compute_mce(predictions, labels)
        
        # Brier score
        brier = brier_score_loss(labels_flat, predictions_flat)
        
        # Negative Log Likelihood
        epsilon = 1e-8
        nll = -np.mean(
            labels_flat * np.log(predictions_flat + epsilon) +
            (1 - labels_flat) * np.log(1 - predictions_flat + epsilon)
        )
        
        # Confidence statistics
        conf_mean = predictions_flat.mean()
        conf_std = predictions_flat.std()
        
        # Accuracy
        accuracy = (predictions_flat.round() == labels_flat).mean()
        
        metrics = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'nll': nll,
            'confidence_mean': conf_mean,
            'confidence_std': conf_std,
            'accuracy': accuracy
        }
        
        return metrics


def calibrate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Tuple[TemperatureScaling, float]:
    """
    Calibrate model using temperature scaling on validation set
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
    
    Returns:
        temp_scaler: Fitted temperature scaling module
        temperature: Optimal temperature value
    """
    model.eval()
    temp_scaler = TemperatureScaling().to(device)
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            
            all_logits.append(logits)
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Fit temperature
    temperature = temp_scaler.fit(all_logits, all_labels)
    
    print(f"✅ Optimal temperature: {temperature:.4f}")
    
    return temp_scaler, temperature


if __name__ == "__main__":
    # Example usage
    print("🔬 Calibration Analysis Demo\n")
    
    # Simulate predictions and labels
    np.random.seed(42)
    n_samples = 10000
    
    # Uncalibrated predictions (overconfident)
    predictions = np.random.beta(2, 2, n_samples)
    predictions = np.clip(predictions * 1.2, 0, 1)  # Make overconfident
    
    # Generate labels based on predictions with noise
    labels = (predictions + np.random.normal(0, 0.1, n_samples) > 0.5).astype(float)
    
    # Analyze calibration
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    print("📊 Calibration Metrics (Uncalibrated):")
    metrics = analyzer.calibration_metrics(predictions, labels)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create reliability diagram
    analyzer.reliability_diagram(
        predictions,
        labels,
        save_path="outputs/reliability_diagram_uncalibrated.png",
        title="Reliability Diagram (Uncalibrated)"
    )
    
    # Apply simple calibration (Platt scaling simulation)
    predictions_calibrated = 1 / (1 + np.exp(-2 * (predictions - 0.5)))
    
    print("\n📊 Calibration Metrics (Calibrated):")
    metrics_cal = analyzer.calibration_metrics(predictions_calibrated, labels)
    for key, value in metrics_cal.items():
        print(f"  {key}: {value:.4f}")
    
    analyzer.reliability_diagram(
        predictions_calibrated,
        labels,
        save_path="outputs/reliability_diagram_calibrated.png",
        title="Reliability Diagram (Calibrated)"
    )
    
    print(f"\n✅ Improvement:")
    print(f"  ECE: {metrics['ece']:.4f} → {metrics_cal['ece']:.4f} ({(1-metrics_cal['ece']/metrics['ece'])*100:.1f}% better)")
    print(f"  Brier: {metrics['brier_score']:.4f} → {metrics_cal['brier_score']:.4f}")
