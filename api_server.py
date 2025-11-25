"""
FastAPI Inference Server for Check Safety Suite
Production-ready API with monitoring, validation, and comprehensive responses
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import torch
import cv2
import numpy as np
from pathlib import Path
import io
import base64
import time
import logging
from datetime import datetime
import json

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.pipeline.check_pipeline import run_check
from src.models.uncertainty.mc_dropout import MCDropoutWrapper
from src.explainability.gradcam_plusplus import GradCAMPlusPlus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions')
TAMPERED_DETECTED = Counter('tampered_detected', 'Number of tampered cheques detected')
INFERENCE_DURATION = Histogram('inference_duration_seconds', 'Inference time in seconds')
MODEL_CONFIDENCE = Histogram('model_confidence', 'Model confidence scores', buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

# FastAPI app
app = FastAPI(
    title="Check Safety Suite API",
    description="AI-powered fraud detection for financial documents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TamperRegion(BaseModel):
    bbox: List[int] = Field(..., description="Bounding box [x, y, w, h]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    area_pixels: int = Field(..., description="Area in pixels")

class UncertaintyMetrics(BaseModel):
    uncertainty_mean: float = Field(..., description="Mean uncertainty across image")
    uncertainty_max: float = Field(..., description="Maximum uncertainty value")
    uncertainty_std: float = Field(..., description="Standard deviation of uncertainty")
    high_uncertainty_regions: List[TamperRegion] = Field(default_factory=list, description="Regions with high uncertainty")
    confidence_mean: float = Field(..., description="Mean confidence (1 - uncertainty)")

class FeatureAnalysis(BaseModel):
    aba_valid: Optional[bool] = Field(None, description="ABA routing number validity")
    amount_consistent: Optional[bool] = Field(None, description="Amount consistency check")
    signature_coverage_pct: Optional[float] = Field(None, description="Signature coverage percentage")
    ink_deltaE00: Optional[float] = Field(None, description="Color difference (CIEDE2000)")
    ink_deltaE76: Optional[float] = Field(None, description="Color difference (CIE76)")

class PredictionResponse(BaseModel):
    verdict: str = Field(..., description="TAMPERED or AUTHENTIC")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    risk_score: float = Field(..., ge=0.0, le=10.0, description="Risk score (0-10)")
    tamper_probability_map: str = Field(..., description="Base64 encoded heatmap")
    tamper_regions: List[TamperRegion] = Field(default_factory=list, description="Detected tamper regions")
    features: FeatureAnalysis = Field(..., description="Feature analysis results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(default="2.0.0", description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")
    # NEW: Uncertainty quantification fields
    uncertainty: Optional[UncertaintyMetrics] = Field(None, description="Uncertainty metrics (if enabled)")
    gradcam_heatmap: Optional[str] = Field(None, description="GradCAM++ heatmap (if enabled)")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str
    uptime_seconds: float

# Global model
MODEL = None
MC_MODEL = None  # MC Dropout wrapper
GRADCAM = None  # GradCAM++ instance
DEVICE = None
START_TIME = time.time()

def load_model():
    """Load the trained model"""
    global MODEL, MC_MODEL, GRADCAM, DEVICE
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = UNet(n_channels=3, n_classes=1).to(DEVICE)
    
    checkpoint_path = Path("outputs/training/best_model.pth")
    if checkpoint_path.exists():
        MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        logger.info(f"Model loaded from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found, using random weights")
    
    MODEL.eval()
    
    # Initialize MC Dropout wrapper
    MC_MODEL = MCDropoutWrapper(MODEL, dropout_rate=0.1, n_samples=10)
    logger.info("MC Dropout wrapper initialized")
    
    # Initialize GradCAM++
    try:
        # Find the last convolutional layer
        target_layer = None
        for name, module in MODEL.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is not None:
            GRADCAM = GradCAMPlusPlus(MODEL, target_layer=target_layer)
            logger.info("GradCAM++ initialized")
        else:
            logger.warning("Could not find target layer for GradCAM++")
    except Exception as e:
        logger.warning(f"GradCAM++ initialization failed: {e}")
    
    logger.info(f"Model ready on device: {DEVICE}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Check Safety Suite API...")
    load_model()
    logger.info("API ready to serve requests")

def infer_mask(img_bgr: np.ndarray, img_size: int = 512) -> np.ndarray:
    """Run segmentation inference"""
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (img_size, img_size))[:, :, ::-1] / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(DEVICE)
    
    with torch.no_grad():
        logits = MODEL(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    return cv2.resize(prob, (w, h))

def extract_tamper_regions(prob_map: np.ndarray, threshold: float = 0.5) -> List[TamperRegion]:
    """Extract bounding boxes for tampered regions"""
    binary_mask = (prob_map > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area > 100:  # Filter small regions
            # Calculate average confidence in this region
            region_mask = np.zeros_like(prob_map)
            cv2.drawContours(region_mask, [contour], -1, 1, -1)
            avg_confidence = (prob_map * region_mask).sum() / region_mask.sum()
            
            regions.append(TamperRegion(
                bbox=[int(x), int(y), int(w), int(h)],
                confidence=float(avg_confidence),
                area_pixels=int(area)
            ))
    
    return sorted(regions, key=lambda r: r.confidence, reverse=True)

def calculate_risk_score(features: Dict, max_prob: float) -> float:
    """Calculate risk score (0-10)"""
    risk = min(max_prob * 5, 5.0)
    
    if not features.get('aba_valid', True):
        risk += 2.0
    if not features.get('amount_consistent', True):
        risk += 2.0
    if features.get('signature_coverage_pct', 1.0) < 0.05:
        risk += 1.0
    
    return min(risk, 10.0)

def encode_image_base64(img: np.ndarray) -> str:
    """Encode image as base64 string"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    return_heatmap: bool = True,
    return_uncertainty: bool = False,
    return_gradcam: bool = False,
    include_text: bool = False
):
    """
    Predict tampering in a cheque image
    
    - **file**: Image file (PNG, JPG, JPEG, TIF)
    - **confidence_threshold**: Threshold for tamper detection (0.0-1.0)
    - **return_heatmap**: Whether to return the probability heatmap
    - **return_uncertainty**: Whether to return uncertainty metrics (slower)
    - **return_gradcam**: Whether to return GradCAM++ visualization
    - **include_text**: Whether to include OCR text extraction
    """
    start_time = time.time()
    
    PREDICTIONS_TOTAL.inc()
    ACTIVE_REQUESTS.inc()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Correct orientation (90 degree rotation)
        try:
            from src.ocr_extraction.base_ocr import correct_orientation
            img_bgr = correct_orientation(img_bgr)
        except ImportError:
            pass
        
        # Run inference
        uncertainty_map = None
        uncertainty_metrics = None
        
        if return_uncertainty and MC_MODEL is not None:
            # MC Dropout inference
            with INFERENCE_DURATION.time():
                h, w = img_bgr.shape[:2]
                x = cv2.resize(img_bgr, (512, 512))[:, :, ::-1] / 255.0
                x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(DEVICE)
                
                mean_pred, uncertainty = MC_MODEL(x_tensor, return_uncertainty=True)
                prob_map = torch.sigmoid(mean_pred)[0, 0].cpu().numpy()
                uncertainty_map = uncertainty[0, 0].cpu().numpy()
                
                # Resize to original size
                prob_map = cv2.resize(prob_map, (w, h))
                uncertainty_map = cv2.resize(uncertainty_map, (w, h))
                
                # Extract high uncertainty regions
                high_unc_regions = extract_tamper_regions(uncertainty_map, threshold=0.3)
                
                uncertainty_metrics = UncertaintyMetrics(
                    uncertainty_mean=float(uncertainty_map.mean()),
                    uncertainty_max=float(uncertainty_map.max()),
                    uncertainty_std=float(uncertainty_map.std()),
                    high_uncertainty_regions=high_unc_regions[:5],  # Top 5
                    confidence_mean=float(1.0 - uncertainty_map.mean())
                )
        else:
            # Standard inference
            with INFERENCE_DURATION.time():
                prob_map = infer_mask(img_bgr)
        
        max_prob = float(prob_map.max())
        mean_prob = float(prob_map.mean())
        
        MODEL_CONFIDENCE.observe(max_prob)
        
        # Extract features
        try:
            features_dict, _, _ = run_check(
                img_bgr,
                seg_mask=(prob_map > confidence_threshold).astype(float),
                out_dir="outputs/_api",
                fname=f"api_{int(time.time())}",
                include_text=include_text
            )
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features_dict = {}
        
        # Extract tamper regions
        regions = extract_tamper_regions(prob_map, confidence_threshold)
        
        # Calculate risk
        risk_score = calculate_risk_score(features_dict, max_prob)
        is_tampered = max_prob > confidence_threshold
        
        if is_tampered:
            TAMPERED_DETECTED.inc()
        
        # Encode heatmap
        if return_heatmap:
            heatmap_colored = cv2.applyColorMap(
                (prob_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_b64 = encode_image_base64(heatmap_colored)
        else:
            heatmap_b64 = ""
        
        # Generate GradCAM++ if requested
        gradcam_b64 = None
        if return_gradcam and GRADCAM is not None:
            try:
                h, w = img_bgr.shape[:2]
                x = cv2.resize(img_bgr, (512, 512))[:, :, ::-1] / 255.0
                x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(DEVICE)
                
                cam = GRADCAM.generate_cam(x_tensor)
                cam_resized = cv2.resize(cam, (w, h))
                overlay = GRADCAM.overlay_on_image(cam_resized, img_bgr, alpha=0.5)
                gradcam_b64 = encode_image_base64(overlay)
            except Exception as e:
                logger.warning(f"GradCAM++ generation failed: {e}")
        
        # Build response
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            verdict="TAMPERED" if is_tampered else "AUTHENTIC",
            confidence=max_prob,
            risk_score=risk_score,
            tamper_probability_map=heatmap_b64,
            tamper_regions=regions,
            features=FeatureAnalysis(**{
                k: v for k, v in features_dict.items()
                if k in FeatureAnalysis.__fields__
            }),
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            uncertainty=uncertainty_metrics,
            gradcam_heatmap=gradcam_b64
        )
        
        logger.info(
            f"Prediction: {response.verdict}, "
            f"Confidence: {max_prob:.3f}, "
            f"Risk: {risk_score:.1f}, "
            f"Time: {processing_time:.1f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        device=DEVICE if DEVICE else "unknown",
        model_version="2.0.0",
        uptime_seconds=time.time() - START_TIME
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Check Safety Suite API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
