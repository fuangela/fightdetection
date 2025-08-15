# Human Pose Based Fight Detection System

A comprehensive computer vision system for real-time fight detection using pose estimation and machine learning techniques.

## Overview

2 approaches to fight detection:
- **Projectv4**: LSTM-based pose detection
- **Projectv3**: Experimental VLM (Visual Language Model) approach using LLaVA

Note: **FightDetectionPoseLSTM** is a pose-based classification system and contains dataset tools and preprocessing scripts taken from https://github.com/jpowellgz/FightDetectionPoseLSTM.git

## Projectv4
- Real-time pose estimation using MediaPipe
- LSTM temporal modeling to classify action
- Multi-person tracking using bounding box
- Pose feature extraction
- Real-time camera streaming
- Diagnostics included for troubleshooting
- 
## Repository Structure

```
├── Projectv4/                    # Main production pipeline
│   ├── camera.py                 # Real-time detection
│   ├── cameratesting.py          # Testing utilities
│   ├── config.py                 # Configuration management
│   ├── diagnostics.py            # Model evaluation
│   ├── featureextract.py         # Pose feature engineering
│   ├── posedetect.py             # Pose estimation core
│   ├── processdata.py            # Data preprocessing
│   ├── trainmodel.py             # Model training
│   └── run_*.py                  # Execution scripts
├── Projectv3/                    # VLM experimental approach
│   ├── vlmtry.py                 # LLaVA-based detection
│   └── LLaVA/                    # VLM implementation
├── FightDetectionPoseLSTM/       # Structured pose analysis
│   ├── 1_MultipleVideoFrameExtraction.py
│   ├── 2_AngleVectorsCalculation.py
│   ├── 3_AngleVectorsAdjustement.py
│   ├── 4_1_LSTMClassification.py
│   └── 4_2_MLClassification.py
└── README.md                     # This file
```

## Quick Start

### Prerequisites
```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn
```

### Running the System
```bash
cd Projectv4
python camera.py  # Start real-time detection
```

### Training Custom Models
```bash
cd Projectv4
python run_processdata.py  # Prepare training data
python run_trainmodel.py   # Train LSTM model
```

## Configuration

Edit `Projectv4/config.py` to customize:
- Detection confidence thresholds
- Sequence length for temporal analysis
- Camera settings and resolution
- Model parameters


## Others:
## Supported Input

- Live webcam feed
- Video files (MP4, AVI, etc.)
- Image sequences
- Multiple camera streams

## Dataset

The system was trained on the fight-detection-surv-dataset containing:
- Fight scenarios: Punching, kicking, wrestling
- No-fight scenarios: Normal activities, conversations
- Various environments and lighting conditions and durations 

---

**Note**: Large model files and datasets are excluded from this repository. Please refer to the documentation for downloading and setting up required models and datasets.
