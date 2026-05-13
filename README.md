# PneumoDetect AI 🫁

A deep learning model for pneumonia detection from chest X-ray images using transfer learning with MobileNetV2.

## Features
- Binary classification: NORMAL vs PNEUMONIA
- 79% overall accuracy
- Flask REST API for predictions
- Transfer learning with pre-trained MobileNetV2

## Setup

```bash
conda create -n pneumodetect python=3.11
conda activate pneumodetect
pip install -r requirements.txt
```

## Training

```bash
python scripts/preprocess.py
python train.py
```

## Evaluation

```bash
python evaluate.py
```

## API Server

```bash
python app.py
```

Then test predictions:

```bash
curl -X POST -F "file=@path/to/xray.png" http://localhost:5000/predict
```

## Model Performance

- **Overall Accuracy**: 79%
- **NORMAL Precision**: 90%
- **PNEUMONIA Recall**: 69%

## Dataset

Uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

## License

MIT