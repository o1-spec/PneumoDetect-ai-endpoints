# PneumoDetect AI - Transfer Learning Pipeline

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL SYSTEM PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ├── scripts/preprocess.py          → RSNA dataset preprocessing
   └── scripts/prepare_chestxray14.py → ChestX-ray14 preprocessing

2. MODEL TRAINING (TRANSFER LEARNING)
   ├── pretrain_chestxray14.py        → Stage 1: Domain pretraining
   └── fine_tune_rsna.py              → Stage 2: Task-specific fine-tuning

3. INFERENCE & EVALUATION
   ├── evaluate.py                    → Model evaluation on test set
   └── app.py                         → Flask REST API with Grad-CAM

4. DEPLOYMENT
   └── NestJS backend integration
```

## Three-Stage Transfer Learning

### Stage 1: ImageNet (Pre-trained Weights)
- MobileNetV2 / DenseNet121 / ResNet50
- Trained on 1M+ images
- General vision features

### Stage 2: ChestX-ray14 Pretraining
- **Purpose**: Learn chest radiograph domain
- **Dataset**: 61,765 total images (60,412 NORMAL + 1,353 PNEUMONIA)
  - Training: 48,330 NORMAL + 1,082 PNEUMONIA
  - Validation: 12,082 NORMAL + 271 PNEUMONIA
- **Script**: `pretrain_chestxray14.py`
- **Output**: Pretrained weights tuned to medical imaging
- **Duration**: ~2-3 hours (10 epochs)
- **Key Features**:
  - Balanced class weighting (handles 45:1 imbalance)
  - Advanced architecture (256→128 dense layers)
  - Adaptive learning rate (ReduceLROnPlateau)
  - Early stopping based on AUC
  - TensorBoard logging

### Stage 3: RSNA Fine-tuning
- **Purpose**: Adapt to specific RSNA pneumonia detection task
- **Dataset**: RSNA dataset (smaller, task-specific)
- **Script**: `fine_tune_rsna.py`
- **Key Features**:
  - Loads pretrained weights from Stage 2
  - Unfreezes top layers for fine-tuning
  - Lower learning rate (1e-5)
  - Lighter data augmentation
  - Domain-adapted model

## Why This Matters

### For Your Supervisor
```
✓ Proper transfer learning pipeline
✓ Domain-specific pretraining (not just ImageNet)
✓ Quantified metrics (Precision, Recall, AUC)
✓ Reproducible: can switch architectures
✓ Academic rigor: class balancing, proper train/val splits
```

### For Medical AI
```
✓ Handles extreme class imbalance (45:1)
✓ Smaller, faster models (MobileNetV2)
✓ Explainability (Grad-CAM)
✓ Proper validation methodology
```

## How to Run

### 1. Prepare Data
```bash
# RSNA dataset
python scripts/preprocess.py

# ChestX-ray14 dataset
python scripts/prepare_chestxray14.py
```

### 2. Pretrain on ChestX-ray14
```bash
python pretrain_chestxray14.py
# Output: models/pretrained/mobilenetv2_chestxray14_weights.h5
# Time: ~2-3 hours
```

### 3. Fine-tune on RSNA
```bash
python fine_tune_rsna.py
# Output: models/finetuned/mobilenetv2_rsna_final.keras
# Time: ~1-2 hours
```

### 4. Evaluate
```bash
python evaluate.py
```

### 5. Deploy API
```bash
python app.py
# API available at: http://localhost:5000
```

## Model Configuration

### Pretraining (ChestX-ray14)
- **Architecture**: MobileNetV2 (can switch to DenseNet121, ResNet50)
- **Image Size**: 224×224
- **Batch Size**: 32
- **Epochs**: 10 (lightweight pretraining)
- **Learning Rate**: 0.0001
- **Data Augmentation**: Flip, Rotate(±10°), Zoom(±15%), Translate(±10%)

### Fine-tuning (RSNA)
- **Learning Rate**: 0.00001 (10x lower)
- **Epochs**: 20
- **Unfreezes**: Layers after index 100
- **Data Augmentation**: Lighter (Flip, Rotate(±5°), Zoom(±10%))

## Output Artifacts

### From Pretraining
```
models/pretrained/
├── mobilenetv2_chestxray14_weights.h5     # Weights for transfer
├── mobilenetv2_chestxray14_final.keras    # Full model
├── mobilenetv2_chestxray14_base_weights.h5
└── mobilenetv2_chestxray14_metadata.json  # Training metadata
```

### From Fine-tuning
```
models/finetuned/
├── mobilenetv2_rsna_final.keras           # Production model
├── mobilenetv2_rsna_weights.h5
└── mobilenetv2_rsna_metadata.json
```

## Key Improvements Over Initial train.py

| Feature | train.py | pretrain_chestxray14.py + fine_tune_rsna.py |
|---------|----------|---------------------------------------------|
| Transfer Learning | ImageNet only | ImageNet → ChestX-ray14 → RSNA (3-stage) |
| Class Imbalance | Manual calculation | sklearn.utils.class_weight (correct) |
| Architecture | Fixed Dense layers | Rich head (256→128 with BatchNorm) |
| Metrics | Accuracy only | Accuracy + Precision + Recall + AUC |
| Learning Rate | Fixed | Adaptive (ReduceLROnPlateau) |
| Model Flexibility | Hard-coded | Configurable (MobileNetV2/DenseNet/ResNet) |
| Logging | None | TensorBoard |
| Weight Checkpointing | Basic | Best model based on AUC |

## Why MobileNetV2 (Default)

1. **Already integrated into Flask app**
2. **Grad-CAM already works with it**
3. **Lighter/faster (important for deployment)**
4. **Good accuracy-efficiency tradeoff**
5. **Easy to export (ONNX, TFLite)**

Later: Compare with DenseNet121, ResNet50

## Next Steps

1. ✅ Run pretraining
   ```bash
   python pretrain_chestxray14.py
   ```

2. ✅ Run fine-tuning
   ```bash
   python fine_tune_rsna.py
   ```

3. 🔄 Evaluate on final test set
   ```bash
   python evaluate.py
   ```

4. 🚀 Deploy
   - Update `app.py` to use finetuned model path
   - Run Flask API
   - Integrate with NestJS backend

## For Your Presentation

**Talk Points:**
- "We implemented a 3-stage transfer learning pipeline"
- "ImageNet → Medical domain (ChestX-ray14) → Task (RSNA)"
- "Handles 45:1 class imbalance with weighted class balancing"
- "Achieves X% accuracy on RSNA with Y% precision/recall"
- "Interpretable predictions with Grad-CAM"
- "Deployable via Flask REST API"
- "Scalable: can swap architectures (MobileNetV2/DenseNet/ResNet)"

## Research References

This architecture follows:
- **Transfer Learning**: Yosinski et al. (2014) - "How transferable are features in DNNs?"
- **Class Weighting**: King & Zeng (2001) - "Logistic regression in rare events data"
- **Medical Imaging**: CheXNet (Rajkomar et al., 2017)
- **Model Interpretability**: Selvaraju et al. (2017) - Grad-CAM

---

**Status**: 🟢 Production-ready for undergraduate capstone
**Last Updated**: May 20, 2026
