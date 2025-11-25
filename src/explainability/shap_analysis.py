"""
SHAP Analysis for Feature Importance
This module provides utilities to explain model predictions using SHAP (SHapley Additive exPlanations).
It supports both tabular data (for fusion models) and image data (for deep learning models).
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP explainer for model interpretability.
    
    Supports:
    - TreeExplainer for tree-based models (XGBoost, LightGBM, RandomForest)
    - DeepExplainer/GradientExplainer for PyTorch models
    - KernelExplainer for generic models
    """
    
    def __init__(
        self,
        model: Any,
        background_data: Optional[Union[np.ndarray, pd.DataFrame, torch.Tensor]] = None,
        model_type: str = 'tree'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            background_data: Background dataset for baseline (required for Deep/Kernel explainers)
            model_type: Type of model ('tree', 'deep', 'kernel', 'gradient')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        
        self._init_explainer(background_data)
        
    def _init_explainer(self, background_data):
        """Initialize the appropriate SHAP explainer"""
        try:
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'deep':
                if background_data is None:
                    raise ValueError("Background data required for DeepExplainer")
                self.explainer = shap.DeepExplainer(self.model, background_data)
            elif self.model_type == 'gradient':
                if background_data is None:
                    raise ValueError("Background data required for GradientExplainer")
                self.explainer = shap.GradientExplainer(self.model, background_data)
            elif self.model_type == 'kernel':
                if background_data is None:
                    raise ValueError("Background data required for KernelExplainer")
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
                
            logger.info(f"Initialized SHAP {self.model_type} explainer")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise

    def explain_prediction(
        self,
        features: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP values for a specific prediction.
        
        Args:
            features: Input features for the prediction
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary containing SHAP values and metadata
        """
        try:
            if self.model_type == 'tree':
                shap_values = self.explainer(features)
            elif self.model_type in ['deep', 'gradient']:
                shap_values = self.explainer.shap_values(features)
            else:
                shap_values = self.explainer.shap_values(features)
                
            # Handle different return types from shap
            if isinstance(shap_values, list):
                # For classification, it might return a list of arrays (one per class)
                # We usually care about the positive class (index 1)
                if len(shap_values) == 2:
                    values = shap_values[1]
                else:
                    values = shap_values[0]
            else:
                values = shap_values
                
            # If it's an Explanation object (newer SHAP versions)
            base_values = None
            if hasattr(shap_values, 'base_values'):
                base_values = shap_values.base_values
                values = shap_values.values
            elif hasattr(self.explainer, 'expected_value'):
                base_values = self.explainer.expected_value
                
            return {
                'shap_values': values,
                'base_value': base_values,
                'feature_names': feature_names if feature_names else getattr(features, 'columns', None),
                'data': features
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise

    def plot_waterfall(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        index: int = 0,
        max_display: int = 10,
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Generate a waterfall plot for a single prediction.
        
        Args:
            features: Input features
            index: Index of the sample to explain
            max_display: Maximum number of features to show
            show: Whether to display the plot
            save_path: Path to save the plot
        """
        try:
            # Ensure we have an Explanation object for the plot
            if self.model_type == 'tree':
                shap_values = self.explainer(features)
                explanation = shap_values[index]
            else:
                # Construct explanation object manually if needed
                # This is complex for non-tree models, might need simplification
                logger.warning("Waterfall plot is best supported for Tree models currently")
                return

            plt.figure()
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved waterfall plot to {save_path}")
                
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting waterfall: {e}")

    def plot_summary(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        plot_type: str = 'dot',
        max_display: int = 20,
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Generate a summary plot for multiple predictions.
        
        Args:
            features: Input features
            plot_type: 'dot' or 'bar'
            max_display: Maximum number of features to show
            show: Whether to display the plot
            save_path: Path to save the plot
        """
        try:
            if self.model_type == 'tree':
                shap_values = self.explainer(features)
            else:
                shap_values = self.explainer.shap_values(features)
            
            plt.figure()
            shap.summary_plot(shap_values, features, plot_type=plot_type, max_display=max_display, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Saved summary plot to {save_path}")
                
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting summary: {e}")

if __name__ == "__main__":
    # Example usage
    print("SHAP Analysis - Example Usage")
    
    # Create dummy data
    X, y = shap.datasets.adult()
    X = X.iloc[:100]  # Small subset
    y = y[:100]
    
    # Train simple model
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=10)
    model.fit(X, y)
    
    # Initialize explainer
    explainer = SHAPExplainer(model, model_type='tree')
    
    # Explain prediction
    result = explainer.explain_prediction(X.iloc[0:1])
    print("SHAP values shape:", result['shap_values'].shape)
    
    # Plot
    print("Generating plots...")
    # explainer.plot_waterfall(X, index=0, show=False)
    # explainer.plot_summary(X, show=False)
    print("Done!")
