---
title: PneumoDetect AI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PneumoDetect AI 🫁

An advanced deep learning system for pneumonia detection from chest X-ray images, utilizing a **3-Stage Transfer Learning Pipeline** and a **Dual-Backbone Ensemble** (ResNet50 + DenseNet121) with integrated Grad-CAM visual explainability.

## 🚀 System Architecture

PneumoDetect AI employs a multi-stage training pipeline to adapt general vision filters to specific clinical features:

1. **Stage 1: ImageNet Pretraining**
   - Base backbones (ResNet50, DenseNet121, and MobileNetV2) initialized with weights pre-trained on 1M+ general natural images.
2. **Stage 2: Domain-Specific Pretraining (ChestX-ray14)**
   - Pretrained on a selected subset of **61,765 images** (60,412 Normal + 1,353 Pneumonia) to transition feature extractors from natural objects to chest radiographs.
   - Handled extreme class imbalance (45:1) using dynamic class weighting.
3. **Stage 3: Task-Specific Fine-tuning (RSNA Dataset)**
   - Fine-tuned on the entire RSNA Pneumonia Detection dataset (**26,684 images**).
   - Unfroze top convolution blocks for fine-grained adaptation.
4. **Dual-Model Ensemble Inference**
   - Blends predictions at runtime: $\text{Final Score} = 0.52 \cdot \text{ResNet50} + 0.48 \cdot \text{DenseNet121}$.
   - Generates fused Grad-CAM explainability heatmaps ($0.48 \cdot \text{ResNet50} + 0.52 \cdot \text{DenseNet121}$) to highlight infected regions.

---

## ⚙️ Setup

Initialize the environment and install dependencies:

```bash
conda create -n pneumodetect python=3.11
conda activate pneumodetect
pip install -r requirements.txt
```

---

## 🛠️ Training & Evaluation

1. **Preprocess datasets:**
   ```bash
   python scripts/preprocess.py
   python scripts/prepare_chestxray14.py
   ```
2. **Pretrain on ChestX-ray14:**
   ```bash
   python pretrain_chestxray14.py
   ```
3. **Fine-tune on RSNA:**
   ```bash
   python fine_tune_rsna.py
   ```
4. **Evaluate performance:**
   ```bash
   python evaluate.py
   ```

---

## 📊 Model Performance (Validation Set)

### Primary Ensemble Components (RSNA Fine-tuned):
* **DenseNet121**: 
  - Accuracy: **79.91%** | AUC: **85.22%** | Recall: **72.21%** | Precision: **54.05%**
* **ResNet50**: 
  - Accuracy: **79.85%** | AUC: **85.18%** | Recall: **70.13%** | Precision: **54.07%**

### Baseline Model:
* **MobileNetV2**: 
  - Accuracy: **82.68%** | AUC: **86.01%** | Recall: **62.48%** | Precision: **61.36%**

---

## 🔌 API Server

Run the local API endpoint (Flask):

```bash
python app.py
```
*Server starts on http://localhost:7860*

### Prediction Endpoint:
* **Route:** `POST /predict`
* **Payload:** Form data containing a single file (key: `file`)
* **Response Format:**
  ```json
  {
    "result": "PNEUMONIA",
    "confidence": 0.9333,
    "raw_predictions": {
      "resnet50": 0.9412,
      "densenet121": 0.9248,
      "ensemble_average": 0.9333
    },
    "heatmap": "data:image/png;base64,iVBORw0KGgoAAAANS..."
  }
  ```

---

## 📄 License

MIT