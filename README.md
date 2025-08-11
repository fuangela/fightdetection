# Human Pose Based Fight Detection System

A comprehensive computer vision system for real-time fight detection using pose estimation and machine learning techniques.

## 🎯 Project Overview

This repository contains multiple approaches to fight detection:
- **Projectv4**: Production-ready LSTM-based pose detection (Recommended)
- **Projectv3**: Experimental VLM (Visual Language Model) approach using LLaVA
- **FightDetectionPoseLSTM**: Structured pose-based classification system
- Dataset tools and preprocessing utilities

## 🏆 Best Pipeline: Projectv4

Projectv4 offers the most robust and efficient fight detection pipeline with:
- Real-time pose estimation using MediaPipe
- LSTM temporal modeling for action classification
- Advanced person tracking and ID assignment
- Production-ready camera integration
- Comprehensive diagnostics and configuration

### Key Features:
- ✅ Real-time webcam processing
- ✅ Multi-person tracking
- ✅ Temporal sequence analysis (60 frames)
- ✅ Configurable confidence thresholds
- ✅ Efficient pose feature extraction (21D vectors)

## 📁 Repository Structure

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

## 🚀 Quick Start

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

## 🔧 Configuration

Edit `Projectv4/config.py` to customize:
- Detection confidence thresholds
- Sequence length for temporal analysis
- Camera settings and resolution
- Model parameters

## 📊 Performance

**Projectv4 Benchmarks:**
- **Accuracy**: ~85-90% on test dataset
- **Speed**: 15-30 FPS real-time processing
- **Latency**: <100ms per frame
- **Resource**: CPU-optimized (no GPU required)

## 🔬 Alternative Approaches

### VLM Approach (Projectv3)
- Uses LLaVA vision-language model
- High accuracy but computationally expensive
- Requires GPU and model server setup
- Better for offline analysis

### Traditional ML (FightDetectionPoseLSTM)
- Structured pipeline with angle-based features
- Good baseline implementation
- Less optimized for real-time use

## 📈 Model Architecture

**Projectv4 Pipeline:**
1. **Pose Detection**: MediaPipe Holistic
2. **Feature Extraction**: 21D vectors (4 angles + 17 velocities)
3. **Temporal Modeling**: LSTM with 60-frame sequences
4. **Classification**: Binary fight/no-fight prediction
5. **Post-processing**: Confidence filtering and smoothing

## 🎥 Supported Input

- Live webcam feed
- Video files (MP4, AVI, etc.)
- Image sequences
- Multiple camera streams

## 📝 Dataset

The system was trained on the fight-detection-surv-dataset containing:
- Fight scenarios: Punching, kicking, wrestling
- No-fight scenarios: Normal activities, conversations
- Various environments and lighting conditions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation
- Fight detection surveillance dataset contributors
- LLaVA team for vision-language model research

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**Note**: Large model files and datasets are excluded from this repository. Please refer to the documentation for downloading and setting up required models and datasets.
