# Automated Mitosis Detection in Histopathology Images

**A two-stage deep learning pipeline for mitotic figure detection and localization.**

## 📋 Overview

This project implements a two-stage system for automated detection of mitotic figures in histopathology images using the TUPAC16 dataset.

### Key Results

| Configuration | F1 Score | Status |
|---|---|---|
| Stage 1 (Binary) | 0.8894 | Screening |
| **Stage 2 (Detection)** | **0.7742** | ✅ **OPTIMAL** |
| Stage 1+2 Pipeline | 0.5806 | Not recommended |

### ⚠️ Critical Finding

**63% F1 degradation on unseen scanner/stain** reveals severe domain shift requiring multi-center validation.

## 🚀 Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train Stage 2 (Recommended: Direct Detection)
python src/stage2_detector.py --mode train --data_dir data/processed/stage2 --epochs 100

# Evaluate
python src/evaluate.py --checkpoint checkpoints/stage2_best.pth --data_dir data/processed/stage2

# Interactive Demo
python gradio_demo.py  # http://localhost:7860
```

## 📂 Project Structure

```
.
├── models/                      # Model architectures
│   ├── stage1_classifier.py     # Binary classifier + Focal Loss
│   └── stage2_detector.py       # Faster R-CNN with custom anchors
├── configs/                     # Configuration classes
├── src/                         # Training & evaluation scripts
├── checkpoints/                 # Pre-trained weights
├── outputs/                     # Results & visualizations
│   ├── false_positive_gallery/  # 3 FP crops
│   ├── false_negative_gallery/  # 4 FN crops
│   ├── wsi_overlay/             # 6 slide visualizations
│   └── cross_center_results.csv
├── REPORT.tex                   # Full technical report (LaTeX)
├── gradio_demo.py               # Interactive web UI
├── requirements.txt
└── README.md
```

## 🔍 Key Findings

1. **Direct Detection Outperforms Cascade**
   - Stage 2 alone: F1=0.7742 ✅
   - Two-stage pipeline: F1=0.5806 ❌
   - Reason: Filtering removes difficult mitosis cases

2. **Domain Shift is Severe**
   - Slides 65, 66, 63: F1 = 1.0000 (perfect)
   - Slide 34: F1 = 0.2857 ❌ (different scanner/stain)
   - Gap: 63% F1 degradation across centers

3. **Error Analysis**
   - False Positives: 3 crops (20% rate)
   - False Negatives: 4 crops (25% rate)

## 📊 Architecture Details

### Stage 1: Binary Classifier
- **Model:** EfficientNet-B3
- **Input:** 64×64 RGB patch
- **Loss:** Focal Loss
- **Performance:** F1=0.8894 (Epoch 8)

### Stage 2: Object Detector
- **Model:** Faster R-CNN ResNet50-FPN
- **Input:** 512×512 RGB patch
- **Anchors:** Custom [8, 16, 32, 64, 128]px
- **Performance:** mAP@0.5=0.7598 (Epoch 65)

## 🏥 Deployment Status

- ✅ **Research-ready:** Competitive F1, modular design
- ❌ **NOT clinical-ready:** 63% cross-center gap, needs multi-center validation

### Deployment Roadmap

| Phase | Task | Timeline |
|---|---|---|
| 1 | External validation (new scanner) | Immediate |
| 2 | Stain normalization | 1 week |
| 3 | Multi-center training | 2-4 weeks |
| 4 | Domain adaptation (if needed) | 4-8 weeks |

## 📚 References

Key papers:

```bibtex
@inproceedings{Ren2015,
  title={Faster R-CNN: Towards Real-time Object Detection with Region Proposal Networks},
  author={Ren, S. and He, K. and Girshick, R. and Sun, J.},
  booktitle={ICCV},
  year={2015}
}

@inproceedings{Lin2017,
  title={Focal Loss for Dense Object Detection},
  author={Lin, T.-Y. and Goyal, P. and Girshick, R. and He, K. and Dollár, P.},
  booktitle={ICCV},
  year={2017}
}
```

## 📄 License

MIT License

## Citation

```bibtex
@misc{mitosis_detection_2026,
  title={Automated Mitosis Detection in Histopathology Using Two-Stage Deep Learning},
  author={Your Name},
  year={2026},
  publisher={GitHub}
}
```

---

**Status:** Course Project Complete ✅  
**Last Updated:** March 21, 2026  
**Full Report:** See REPORT.tex
