# Limitations & Assumptions

**Purpose**: Honest scientific disclosure of system limitations  
**Author**: Farhad Hassani, Ph.D., EIT  
**Last Updated**: November 24, 2024

---

## Philosophical Note

> "All models are wrong, but some are useful." - George Box

This document acknowledges the boundaries of our tamper detection system. Transparent limitation disclosure is crucial for:
1. **Scientific integrity**: Preventing overclaiming
2. **Deployment safety**: Understanding failure modes  
3. **Future research**: Guiding improvement priorities

---

## 1. Dataset Limitations

### 1.1 Sample Size

**Current**: 158 total samples (126 train, 32 val)

**Industry Standard**: 10,000+ for production vision systems

**Implications**:
- ❌ **Statistical Power**: Insufficient for robust performance estimation
- ❌ **Generalization**: High risk of overfitting to BCSD characteristics
- ❌ **Rare Events**: Likely missing uncommon tamper types
- ⚠️ **Con fidence Intervals**: Wide (unreliable precision estimates)

**Mitigation Strategies**:
- Short-term: Cross-validation to maximize data usage
- Long-term: Collect 1000+ annotated samples

### 1.2 Annotation Quality

**Assumption**: Human labels are ground truth

**Reality Checks**:
- Inter-annotator agreement NOT measured
- Annotation guidelines NOT standardized
- Quality control process UNKNOWN
- Edge precision potentially inconsistent

**Risk**: Model may learn annotation biases, not true tamper patterns.

**Recommended Action**: Conduct inter-rater reliability study (Cohen's κ)

### 1.3 Domain Coverage

**Dataset Source**: BCSD (appears to be specific institution/region)

**Uncovered Scenarios**:
- Different check designs/layouts
- International checks (non-US)
- Aged/degraded documents
- Novel tamper techniques (deepfakes, AI-generated)
- High-resolution scans vs phone photos

**Covariate Shift Risk**: Performance degradation on out-of-distribution inputs.

### 1.4 Class Imbalance

**Current**: ~95% authentic, ~5% tampered pixels

**Training Adjustments**: Focal Loss to handle imbalance

**Remaining Concerns**:
- Validation metrics may be misleading (high accuracy from predicting "all authentic")
- Need class-specific metrics (precision/recall per class)
- Rare tamper subtypes under-represented

---

## 2. Model Limitations

### 2.1 Architecture Constraints

**UNet++ Design Choices**:
- ✅ Good: Multi-scale feature fusion
- ❌ Limited: No attention to inter-region relationships (e.g., amount-words consistency)
- ❌ Fixed: Cannot handle variable-resolution inputs efficiently

**ImprovedUNet Issues**:
- Implemented but NOT production-tested
- Dimension mismatches at different resolutions
- Requires further debugging

### 2.2 Computational Requirements

**Current Model**: 9.2M parameters

**Inference Time** (256×256, CPU):
- Estimated: ~500ms per image (not benchmarked)
- Target: <100ms for real-time processing

**Memory Footprint**:
- Model: ~35 MB
- Activation memory: ~200 MB (batch=1)

**Deployment Constraint**: May be too slow for high-throughput scenarios without GPU.

### 2.3 Resolution Trade-offs

**Training Resolution**: 256×256 (downsampled from ~2000×800)

**Information Loss**:
- Small text may become illegible
- Fine ink alterations (<5px) lost
- MICR line details degraded

**Solution**: Retrain at 512×512 or implement multi-scale inference.

### 2.4 Lack of Uncertainty Quantification

**Current**: Deterministic predictions (no confidence scores)

**Missing**:
- Out-of-distribution detection
- Per-pixel uncertainty estimates
- Ensemble disagreement metrics

**Risk**: Model may confidently predict on images it has never seen (e.g., receipts, invoices).

**Recommended**: Implement Monte Carlo Dropout or Deep Ensembles.

---

## 3. Training Limitations

### 3.1 Hybrid Loss Weights

**Selected**: α=0.5, β=0.3, γ=0.2

**Selection Method**: Grid search on SMALL validation set (32 samples)

**Issues**:
- Weights NOT optimized - could be suboptimal
- No theoretical justification for specific values
- Sensitive to dataset characteristics

**Future Work**: Learn weights via meta-optimization or AutoML.

### 3.2 No Data Augmentation

**Current**: Zero augmentation (to preserve boundary quality)

**Missing Invariances**:
- Rotation (±10°)
- Scale (zoom)
- Photometric (brightness, contrast)
- Perspective distortion

**Risk**: Model may fail on slightly rotated/distorted checks.

**Mitigation**: Add augmentation in future training.

### 3.3 Hyperparameter Tuning

**Current**: Manual selection

**Untuned Parameters**:
- Learning rate schedule
- Batch size
- Optimizer choice (AdamW vs SGD vs RMSprop)
- Weight decay strength

**Potential Improvement**: Automated hyperparameter search could yield +2-5% performance.

### 3.4 Single-Run Training

**Experiments**: 1 training run only

**Missing**:
- Multiple random seed runs (assess variance)
- Statistical significance testing
- Robustness to initialization

**Implication**: Reported performance may be lucky/unlucky outlier.

**Standard Practice**: Train ≥5 runs, report mean ± std dev.

---

## 4. Evaluation Limitations

### 4.1 Validation Set Size

**Current**: 32 samples

**Statistical Power**: Insufficient for reliable performance estimation

**Confidence Intervals**: ±10% on most metrics (wide!)

**Recommended**: 100+ validation samples for ±3% confidence.

### 4.2 Metrics Limitations

**IoU/Dice Limitations**:
- Sensitive to small shifts (high penalty for near-miss)
- Ignore boundary quality (shape matching)
- No distinction between "slightly wrong" vs "completely wrong"

**Missing Metrics**:
- Boundary F1-score (edge precision)
- Hausdorff distance (worst-case error)
- Structural similarity (SSIM on masks)

### 4.3 No Cross-Dataset Evaluation

**Tested On**: BCSD only

**Untested**:  
- External check datasets
- Different countries/institutions
- Synthetic vs real-world distribution shift

**Critical Risk**: Model may be "overfitting to BCSD" rather than learning general tamper detection.

### 4.4 No Adversarial Testing

**Assumption**: Adversaries use "natural" tamper methods

**Reality**: Attackers may craft adversarial examples to fool model

**Missing**:
- Adversarial attack robustness (FGSM, PGD)
- Certified defenses
- Worst-case performance bounds

---

## 5. Deployment Constraints

### 5.1 Real-Time Requirements

**Target**: <100ms latency

**Current**: ~500ms (estimated, unverified)

**Gap**: 5x too slow for real-time processing

**Solutions**:
- Model quantization (INT8)
- GPU deployment
- Model distillation (smaller student)

### 5.2 Integration Constraints

**Current**: Python-only implementation

**Enterprise Needs**:
- Java/C++ bindings
- Mobile deployment (iOS/Android)
- Edge devices (limited compute)

**Compatibility**: Requires ONNX export + deployment engineering.

### 5.3 Failure Modes

**Known Failure Cases** (anecdotal, not systematically tested):
- Very low contrast images
- Extreme rotations (>15°)
- Partial checks (cropped)
- Non-check documents (receipts, forms)

**Graceful Degradation**: NOT implemented - model may fail silently.

**Recommended**: Add input validation + uncertainty thresholds.

###  5.4 Legal/Compliance

**Assumption**: Model predictions used as decision support, NOT sole arbiter

**Regulatory Concerns**:
- Fair lending (bias against certain demographics?)
- Explainability req uirements (can we explain each decision?)
- Audit trails (model versioning, logging)

**Current Status**: NOT compliance-ready.

---

## 6. Scientific Rigor Limitations

### 6.1 No Statistical Significance Testing

**Current**: Point estimates only (e.g., "Val Loss = 0.3034")

**Missing**:
- Confidence intervals
- p-values for improvement claims
- Effect sizes (Cohen's d)

**Implication**: Cannot definitively claim improvements are real vs noise.

### 6.2 Cherry-Picked Results?

**Concern**: Are reported results from single "lucky" run?

**Transparency**:
- Only ONE training run executed
- No report of failed experiments
- No negative results disclosed

**Standard**: Report all experiments, including failures.

### 6.3 Theoretical Guarantees

**What We CANNOT Prove**:
- Convergence to global optimum
- Generalization bounds (PAC-learning)
- Adversarial robustness certificates
- Fairness guarantees

**What We CAN Say**:
- Empirically converges on BCSD
- Follows best-practice architectures
- Uses principled loss functions

---

## 7. Ethical & Social Limitations

### 7.1 Bias Concerns

**Potential Biases**:
- Check design bias (favor certain bank formats?)
- Demographic bias (if tampers correlate with protected attributes)
- Annotation bias (human labelers' systematic errors)

**Mitigation**: NOT TESTED - require fairness audit.

### 7.2 Misuse Potential

**Dual-Use Risk**: Same technology could be used to:
- Detect forgeries (intended use)
- CREATE better forgeries (adversarial use)

**Model Security**: Weights should be protected to prevent adversarial training.

### 7.3 Over-Reliance Risk

**Danger**: Users may blindly trust model predictions

**Reality**: Model is NOT perfect (see all above limitations)

**Recommendation**: Always require human review for high-stakes decisions.

---

## 8. Future Work Priorities (by Impact)

### High Priority
1. **Collect more data** (target: 1000+ samples)
2. **Cross-dataset evaluation** (test generalization)
3. **Quantify uncertainty** (flag out-of-distribution)
4. **Inference speed** optimization (GPU/quantization)

### Medium Priority
5. **Ablation studies** (validate architecture choices)
6. **Data augmentation** (improve robustness)
7. **Higher resolution** training (512×512)
8. **Statistical testing** (confidence intervals)

### Low Priority (Nice-to-Have)
9. **Adversarial robustness**
10. **Fairness audit**
11. **Explainability** enhancements
12. **Mobile deployment**

---

## 9. Responsible Disclosure

### What We Know Works
✅ Model learns on BCSD dataset  
✅ Hybrid Loss converges smoothly  
✅ No obvious overfitting  
✅ Architecture follows best practices

### What We Don't Know
❓ Real-world generalization  
❓ Cross-dataset performance  
❓ Adversarial robustness  
❓ Long-tail failure modes  
❓ Computational efficiency  
❓ Bias/fairness properties

### Deployment Recommendation

**Status**: **Proof-of-Concept** ✅  
**Production-Ready**: **No** ❌

**Path to Production**:
1. Expand dataset to 1000+ samples
2. Cross-validate on external data
3. Implement uncertainty quantification
4. Optimize for real-time inference
5. Conduct security/fairness audit  
6. A/B test in low-risk environment

---

**Document Philosophy**: *"It is better to know the limits of our knowledge than to pretend omniscience."*

This document will be updated as limitations are discovered or mitigated.

---

**Status**: Living document  
**Version**: 1.0  
**Last Updated**: November 24, 2024
