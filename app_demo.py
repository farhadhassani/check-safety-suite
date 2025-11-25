"""
Check Safety Suite - Interactive Demo
A professional Streamlit application for fraud detection in bank cheques.
"""
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.unet import UNet
from src.pipeline.check_pipeline import run_check
from src.models.uncertainty.mc_dropout import MCDropoutWrapper
from src.explainability.gradcam_plusplus import GradCAMPlusPlus

# Page config
st.set_page_config(
    page_title="Check Safety Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained UNet model with MC Dropout and GradCAM++"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    checkpoint_path = Path("outputs/training/best_model.pth")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    else:
        st.warning("⚠️ Trained model not found. Using random weights for demo.")
        model.eval()
    
    # Initialize MC Dropout wrapper
    mc_model = MCDropoutWrapper(model, dropout_rate=0.1, n_samples=10)
    
    # Initialize GradCAM++
    try:
        # Find last conv layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is not None:
            gradcam = GradCAMPlusPlus(model, target_layer=target_layer)
        else:
            gradcam = None
    except Exception as e:
        st.warning(f"GradCAM++ initialization failed: {e}")
        gradcam = None
    
    return model, mc_model, gradcam, device

def infer_mask(model, img_bgr, device, img_size=512):
    """Run segmentation inference"""
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (img_size, img_size))[:, :, ::-1] / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
    
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    return cv2.resize(prob, (w, h))

def create_heatmap_overlay(img, heatmap, alpha=0.5):
    """Create a heatmap overlay on the original image"""
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def infer_with_uncertainty(mc_model, img_bgr, device, img_size=512):
    """Run inference with uncertainty quantification"""
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (img_size, img_size))[:, :, ::-1] / 255.0
    x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
    
    # MC Dropout inference
    mean_pred, uncertainty = mc_model(x_tensor, return_uncertainty=True)
    prob_map = torch.sigmoid(mean_pred)[0, 0].cpu().numpy()
    uncertainty_map = uncertainty[0, 0].cpu().numpy()
    
    # Resize to original size
    prob_map = cv2.resize(prob_map, (w, h))
    uncertainty_map = cv2.resize(uncertainty_map, (w, h))
    
    return prob_map, uncertainty_map

def generate_gradcam(gradcam, img_bgr, device, img_size=512):
    """Generate GradCAM++ visualization"""
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (img_size, img_size))[:, :, ::-1] / 255.0
    x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
    
    cam = gradcam.generate_cam(x_tensor)
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = gradcam.overlay_on_image(cam_resized, img_rgb, alpha=0.5)
    
    return cam_resized, overlay

def calculate_risk_score(features, tamper_prob):
    """Calculate overall risk score (0-10)"""
    risk = 0.0
    
    # Segmentation-based risk
    risk += min(tamper_prob * 10, 5.0)
    
    # Feature-based risk
    if features.get('aba_invalid'):
        risk += 2.0
    if features.get('amount_inconsistent'):
        risk += 2.0
    if features.get('signature_gap', 0) > 0.03:
        risk += 1.0
    
    return min(risk, 10.0)

def create_gauge_chart(value, title):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#a8edea'},
                {'range': [30, 70], 'color': '#fcb69f'},
                {'range': [70, 100], 'color': '#f5576c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🔍 Check Safety Suite</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Financial Document Fraud Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/bank-card-back-side.png", width=80)
        st.title("Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence to flag as tampered"
        )
        
        show_heatmap = st.checkbox("Show Heatmap Overlay", value=True)
        show_features = st.checkbox("Show Feature Analysis", value=True)
        
        st.markdown("### 🔬 Advanced Features")
        show_uncertainty = st.checkbox(
            "Enable Uncertainty Quantification",
            value=False,
            help="Use MC Dropout to estimate prediction uncertainty (slower)"
        )
        
        show_gradcam = st.checkbox(
            "Enable GradCAM++ Explainability",
            value=False,
            help="Show which regions influenced the model's decision"
        )
        
        show_ocr = st.checkbox(
            "Enable OCR Text Extraction",
            value=False,
            help="Extract text fields (Payee, Amount, Date, MICR) using docTR + TrOCR"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This system uses deep learning to detect tampering in bank cheques. "
            "Upload a cheque image to analyze it for potential fraud."
        )
        
        st.markdown("### Model Info")
        st.success(f"✅ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.success("✅ Model: UNet (ResNet34 encoder)")
    
    # Load model
    with st.spinner("Loading model..."):
        model, mc_model, gradcam, device = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a cheque image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, TIF, TIFF"
    )
    
    # Sample images
    st.markdown("### Or try a sample:")
    col1, col2, col3 = st.columns(3)
    
    sample_dir = Path("data/idrbt_tamper")
    if sample_dir.exists():
        with open("data/idrbt_tamper/labels.json", "r") as f:
            labels = json.load(f)
        
        tampered_samples = [x for x in labels if x['label'] == 1][:3]
        
        with col_left:
            st.subheader("📄 Original Image")
            st.image(img_rgb, use_container_width=True)
        
        # Run inference
        with st.spinner("🔍 Analyzing image..."):
            # Choose inference method
            if show_uncertainty and mc_model is not None:
                tamper_prob_map, uncertainty_map = infer_with_uncertainty(mc_model, img_bgr, device)
            else:
                tamper_prob_map = infer_mask(model, img_bgr, device)
                uncertainty_map = None
            
            max_prob = tamper_prob_map.max()
            mean_prob = tamper_prob_map.mean()
            
            # GradCAM++ generation
            if show_gradcam and gradcam is not None:
                try:
                    gradcam_map, gradcam_overlay = generate_gradcam(gradcam, img_bgr, device)
                except Exception as e:
                    st.warning(f"GradCAM++ generation failed: {e}")
                    gradcam_map, gradcam_overlay = None, None
            else:
                gradcam_map, gradcam_overlay = None, None
            
            # Feature extraction
            try:
                features, _, _ = run_check(
                    img_bgr,
                    seg_mask=(tamper_prob_map > confidence_threshold).astype(float),
                    out_dir="outputs/_demo",
                    fname="temp",
                    include_text=show_ocr
                )
            except Exception as e:
                st.warning(f"Feature extraction failed: {e}")
                features = {}
        
        with col_right:
            st.subheader("🎯 Tamper Probability Map")
            
            if show_heatmap:
                overlay = create_heatmap_overlay(img_rgb, tamper_prob_map, alpha=0.6)
                st.image(overlay, use_container_width=True)
            else:
                # Show as grayscale heatmap
                fig = px.imshow(tamper_prob_map, color_continuous_scale='RdYlGn_r')
                fig.update_layout(coloraxis_showscale=True, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Uncertainty and GradCAM++ visualizations
        if show_uncertainty or show_gradcam:
            st.markdown("---")
            st.subheader("🔬 Advanced Analysis")
            
            viz_cols = st.columns(2 if (show_uncertainty and show_gradcam) else 1)
            
            if show_uncertainty and uncertainty_map is not None:
                with viz_cols[0] if show_gradcam else viz_cols[0]:
                    st.markdown("#### Uncertainty Map")
                    st.caption("Red = High uncertainty, Blue = Low uncertainty")
                    
                    uncertainty_overlay = create_heatmap_overlay(
                        img_rgb,
                        uncertainty_map,
                        alpha=0.6
                    )
                    st.image(uncertainty_overlay, use_container_width=True)
                    
                    # Uncertainty metrics
                    st.metric(
                        "Mean Uncertainty",
                        f"{uncertainty_map.mean():.3f}",
                        help="Average uncertainty across the image"
                    )
                    st.metric(
                        "Max Uncertainty",
                        f"{uncertainty_map.max():.3f}",
                        help="Maximum uncertainty value"
                    )
            
            if show_gradcam and gradcam_overlay is not None:
                with viz_cols[1] if show_uncertainty else viz_cols[0]:
                    st.markdown("#### GradCAM++ Heatmap")
                    st.caption("Red = High importance, Blue = Low importance")
                    st.image(gradcam_overlay, use_container_width=True)
                    
                    st.info(
                        "💡 This visualization shows which regions of the check "
                        "influenced the model's decision the most."
                    )
        
        # OCR Results Display
        if show_ocr and features.get('extracted_fields'):
            st.markdown("---")
            st.subheader("📝 Extracted Text (OCR)")
            
            ocr_data = features['extracted_fields']
            
            ocr_col1, ocr_col2 = st.columns(2)
            
            with ocr_col1:
                st.markdown("#### 🏦 Payee & Bank")
                st.text_input("Payee Name", value=ocr_data['payee']['text'], disabled=True)
                st.text_input("Bank Name", value=ocr_data['bank_name']['text'], disabled=True)
                
                st.markdown("#### 📅 Date")
                st.text_input("Date", value=ocr_data['date']['text'], disabled=True)
            
            with ocr_col2:
                st.markdown("#### 💵 Amount")
                st.text_input("Numeric Amount", value=ocr_data['amount_numeric']['text'], disabled=True)
                st.text_input("Amount in Words", value=ocr_data['amount_words']['text'], disabled=True)
                
                st.markdown("#### 🏧 MICR Data")
                micr = ocr_data['micr']
                st.code(f"Routing: {micr['routing']}\nAccount: {micr['account']}\nCheck #: {micr['check_number']}")
            
            with st.expander("View Raw JSON"):
                st.json(ocr_data)

        # Results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Calculate risk
        risk_score = calculate_risk_score(features, max_prob)
        is_tampered = max_prob > confidence_threshold
        
        # Metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="Verdict",
                value="⚠️ TAMPERED" if is_tampered else "✅ AUTHENTIC",
                delta="High Risk" if is_tampered else "Low Risk"
            )
        
        with metric_col2:
            st.metric(
                label="Max Confidence",
                value=f"{max_prob:.1%}",
                delta=f"{(max_prob - confidence_threshold):.1%}"
            )
        
        with metric_col3:
            st.metric(
                label="Risk Score",
                value=f"{risk_score:.1f}/10",
                delta="Critical" if risk_score > 7 else "Moderate" if risk_score > 4 else "Low"
            )
        
        with metric_col4:
            st.metric(
                label="Affected Area",
                value=f"{(tamper_prob_map > confidence_threshold).mean():.1%}",
                delta=f"{(tamper_prob_map > confidence_threshold).sum()} pixels"
            )
        
        # Gauge charts
        st.markdown("### Confidence Metrics")
        gauge_col1, gauge_col2 = st.columns(2)
        
        with gauge_col1:
            st.plotly_chart(
                create_gauge_chart(max_prob, "Maximum Tamper Probability"),
                use_container_width=True
            )
        
        with gauge_col2:
            st.plotly_chart(
                create_gauge_chart(risk_score / 10, "Overall Risk Score"),
                use_container_width=True
            )
        
        # Feature analysis
        if show_features and features:
            st.markdown("---")
            st.subheader("🔬 Feature Analysis")
            
            feat_col1, feat_col2 = st.columns(2)
            
            with feat_col1:
                st.markdown("#### Document Validation")
                
                aba_status = "✅ Valid" if features.get('aba_valid', False) else "❌ Invalid"
                amount_status = "✅ Consistent" if features.get('amount_consistent', False) else "❌ Inconsistent"
                
                st.markdown(f"**ABA Routing Number:** {aba_status}")
                st.markdown(f"**Amount Consistency:** {amount_status}")
                
                sig_coverage = features.get('signature_coverage_pct', 0)
                st.markdown(f"**Signature Coverage:** {sig_coverage:.2%}")
                
                if sig_coverage < 0.05:
                    st.warning("⚠️ Signature coverage is unusually low")
            
            with feat_col2:
                st.markdown("#### Ink Analysis")
                
                delta_e00 = features.get('ink_deltaE00_amount_vs_body')
                delta_e76 = features.get('ink_deltaE76_amount_vs_body')
                
                if delta_e00 is not None:
                    st.markdown(f"**ΔE00 (Amount vs Body):** {delta_e00:.2f}")
                delta_e76 = features.get('ink_deltaE76_amount_vs_body')
                
                if delta_e00 is not None:
                    st.markdown(f"**ΔE00 (Amount vs Body):** {delta_e00:.2f}")
                    if delta_e00 > 10:
                        st.warning("⚠️ Significant color difference detected")
                
                if delta_e76 is not None:
                    st.markdown(f"**ΔE76 (Amount vs Body):** {delta_e76:.2f}")
        
        # Alert
        if is_tampered:
            st.error(
                "🚨 **FRAUD ALERT**: This cheque shows signs of tampering. "
                "Manual review is recommended before processing."
            )
        else:
            st.success(
                "✅ **AUTHENTIC**: No significant signs of tampering detected. "
                "However, always verify critical details manually."
            )
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Save heatmap
            heatmap_colored = cv2.applyColorMap(
                (tamper_prob_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            _, buffer = cv2.imencode('.png', heatmap_colored)
            
            st.download_button(
                label="📥 Download Heatmap",
                data=buffer.tobytes(),
                file_name="tamper_heatmap.png",
                mime="image/png"
            )
        
        with export_col2:
            # Save report
            report = {
                "verdict": "TAMPERED" if is_tampered else "AUTHENTIC",
                "max_confidence": float(max_prob),
                "risk_score": float(risk_score),
                "affected_area_pct": float((tamper_prob_map > confidence_threshold).mean()),
                "features": {k: str(v) for k, v in features.items()}
            }
            
            st.download_button(
                label="📥 Download Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="fraud_analysis_report.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
