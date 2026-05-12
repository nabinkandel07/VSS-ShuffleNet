# VSS-ShuffleNet: A Lightweight YOLOv8-Based Framework for Real-Time Steel Surface Defect Detection

[![Paper](https://img.shields.io/badge/Paper-EAAI-blue)](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blue)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Note:** This paper is currently under review. The code will be fully released upon paper acceptance.

---

## 📋 Abstract

Accurate and real-time detection of steel surface defects is critical for ensuring product quality and operational safety in modern manufacturing systems. Traditional manual inspection and conventional machine vision approaches are often inefficient, subjective, and unable to meet the high-speed demands of industrial production lines. Although recent deep learning-based detectors have significantly improved detection performance, many suffer from high computational complexity and limited capability to capture subtle defects embedded in complex background textures.

To address these challenges, this paper proposes **VSS-ShuffleNet**, a lightweight and efficient steel surface defect detection framework built upon the YOLOv8 architecture. The framework integrates three key innovations:

- **ShuffleNet V2 Backbone** - Reduces computational overhead while preserving effective multi-scale feature extraction
- **Visual State Space (VSS) Block** - Captures long-range spatial dependencies with linear computational complexity
- **Adaptive Gaussian Preprocessing** - Enhances defect boundaries under real-world industrial conditions

---

## 🏆 Results

### NEU-DET Dataset (6 classes)

| Configuration | mAP@0.5 | mAP@0.5:0.95 | Params (M) | FLOPs (G) | FPS |
|---------------|---------|--------------|------------|-----------|-----|
| YOLOv8n (Baseline) | 79.0% | 45.4% | 3.0 | 8.1 | 101 |
| + Preprocessing | 88.7% | 48.6% | 3.0 | 8.1 | 178.6 |
| + ShuffleNet V2 | 91.4% | 52.1% | 3.8 | 7.5 | 200.0 |
| + VSS Block | 92.7% | 53.8% | 3.5 | 7.6 | 230.3 |
| **VSS-ShuffleNet (Full)** | **94.6%** | **67.8%** | **9.7** | **8.5** | **147** |

### GC10-DET Dataset (10 classes)

| Model | mAP@0.5 |
|-------|---------|
| YOLOv8n (Baseline) | 68.7% |
| **VSS-ShuffleNet (Ours)** | **76.1%** |

### Key Improvements

- **+5.9% mAP** over YOLOv8 on NEU-DET
- **+7.4% mAP** over YOLOv8 on GC10-DET
- **46.7% parameter reduction** compared to YOLOv8m
- **Real-time inference** at 147 FPS on NVIDIA RTX 4090D

---

## 📁 Repository Structure

```
VSS-ShuffleNet/
├── vss_shufflenet/              # VSS-ShuffleNet model modules
│   ├── __init__.py
│   ├── vss_block.py             # Visual State Space Block
│   ├── shufflenet_v2.py         # ShuffleNet V2 backbone
│   ├── preprocessing.py         # Gaussian preprocessing pipeline
│   └── model.py                 # Complete VSS-ShuffleNet model
├── ultralytics/                 # YOLOv8 framework
├── improve_models/              # Custom model improvements
├── train/                       # Training images
├── valid/                       # Validation images
├── runs/                        # Training runs and weights
├── train_vss.py                 # Training script
├── test_vss.py                  # Testing script
├── heatmap.py                   # Grad-CAM visualization
├── xml_to_txt.py                # Dataset conversion utility
├── yolo_slice_sort.py           # Data preprocessing utility
├── data.yaml                    # Dataset configuration
└── requirements.txt             # Dependencies
```

---

## 🔧 Requirements

```bash
Python 3.8.10
PyTorch 2.0.0
CUDA 11.8
Ultralytics YOLOv8
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.5
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.2
pandas>=1.5.3
scikit-learn>=1.2.2
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.12.0
ultralytics>=8.0.0
thop>=0.1.1
pillow>=9.4.0
albumentations>=1.3.0
einops>=0.8.1
mamba
```

## How to install Mamba

```
You can see the Mamba.md files
```

---

## 📥 Dataset Preparation

### NEU-DET Dataset
1. Download from [NEU-DET](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
2. Place images in `train/images/` and `valid/images/`
3. Place annotations in `train/labels/` and `valid/labels/`

### GC10-DET Dataset
1. Download from [GC10-DET](https://github.com/lvxiaoming2019/GC10-DET)
2.  Place images in `train/images/` and `valid/images/`
3.  Place annotations in `train/labels/` and `valid/labels/`
4.  Convert XML annotations to YOLO format using `xml_to_txt.py.`
   

### Dataset Configuration for NEU-DET (`data.yaml`)

```yaml
path: ./
train: train/images
val: valid/images

nc: 6  # number of classes (NEU-DET)
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
```

### Dataset Configuration for GC10-DET (`data.yaml`)
train: train/images
val: valid/images

# Number of classes (corrected to match your dataset)
nc: 10

# Class names (replace with your own class names)
names: ['punching', 'weld line', 'crescent gap', 'water spot', 'oil spot', 'silk spot', 'inclusion', 'rolled pit', 'crease', 'waist folding' ]

```
---
## 🚀 Training

### Train VSS-ShuffleNet on NEU-DET

```bash
python train_vss.py
```

### Training Parameters (as per paper)

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Batch Size | 32 |
| Image Size | 640×640 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Scheduler | Cosine Annealing |

### Training with Custom Settings

```bash
python train_vss.py --epochs 300 --batch 16 --lr 0.005
```

---

## 🧪 Testing and Evaluation

### Evaluate Model

```bash
python test_vss.py
```

### Run Inference on Single Image

```python
from ultralytics import YOLO

model = YOLO('runs/train/vss_shufflenet/weights/best.pt')
results = model.predict('path/to/image.jpg', save=True)
```

### Compute FLOPs and Parameters

```bash
python -c "from vss_shufflenet.model import VSSShuffleNet; model = VSSShuffleNet(); print(f'Params: {sum(p.numel() for p in model.parameters()):,}')"
```

---

## 📊 Visualization

### Generate Heatmaps (Grad-CAM)

```bash
python heatmap.py --weights runs/train/vss_shufflenet/weights/best.pt --source valid/images
```

### Plot Training Curves

Training metrics are automatically saved in `runs/train/vss_shufflenet/`, including:
- Loss curves
- Precision-Recall curves
- Confusion matrix
- F1-score curves

---

## 📦 Pretrained Weights

| Model | Dataset | mAP@0.5 | Download |
|-------|---------|---------|----------|
| VSS-ShuffleNet | NEU-DET | 94.6% | [Link]() |
| VSS-ShuffleNet | GC10-DET | 76.1% | [Link]() |

> **Note:** Pretrained weights will be released upon paper acceptance.

---

## 🎯 Model Architecture

```
VSS-ShuffleNet
│
├── Input (640×640×3)
│
├── Preprocessing (Adaptive Gaussian)
│   ├── Blur Assessment (Variance of Laplacian)
│   ├── Conditional Smoothing
│   ├── Unsharp Masking
│   └── Noise Suppression
│
├── Backbone: ShuffleNet V2
│   ├── Stage 1: 48 channels
│   ├── Stage 2: 96 channels (VSS Block)
│   ├── Stage 3: 192 channels (VSS Block)
│   └── Stage 4: 384 channels (VSS Block)
│
├── Neck: Parallel Multi-Branch
│   ├── Upsampling + Concatenation
│   └── VSS Block × 3
│
└── Detection Head (YOLOv8)
    ├── Bounding Box Regression
    └── Classification (6/10 classes)

Total Parameters: 9.7M
FLOPs: 8.5G
FPS: 147 (RTX 4090D)
```

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `vss_shufflenet/vss_block.py` | VSS Block with SS2D mechanism |
| `vss_shufflenet/shufflenet_v2.py` | ShuffleNet V2 backbone implementation |
| `vss_shufflenet/preprocessing.py` | Adaptive Gaussian preprocessing pipeline |
| `vss_shufflenet/model.py` | Complete VSS-ShuffleNet model |
| `train_vss.py` | Training script |
| `test_vss.py` | Testing and evaluation script |
| `heatmap.py` | Grad-CAM visualization |
| `xml_to_txt.py` | Convert XML to YOLO format |
| `yolo_slice_sort.py` | Dataset splitting utility |

---

## 🔬 Ablation Studies

### Component Contribution (NEU-DET)

| Configuration | mAP@0.5 | Δ |
|---------------|---------|---|
| Baseline (YOLOv8n) | 79.0% | - |
| + Preprocessing | 88.7% | +9.7% |
| + ShuffleNet V2 | 91.4% | +2.7% |
| + VSS Block | 92.7% | +1.3% |
| **Full Model** | **94.6%** | **+1.9%** |

### VSS Block Placement Analysis

| Placement | mAP@0.5 |
|-----------|---------|
| No VSS Block | 79.0% |
| Backbone Stage 2 | 81.5% |
| Backbone Stage 4 | 83.2% |
| Neck Entry | 84.5% |
| Before Detection Head | 84.1% |
| **Neck Middle (Ours)** | **85.2%** |

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026vssshufflenet,
  title={VSS-ShuffleNet: A Lightweight YOLOv8-Based Framework for Real-Time Steel Surface Defect Detection},
  author={Wu, Ping and Kandel, Nabin},
  journal={Engineering Applications of Artificial Intelligence},
  note={Under review},
  year={2026}
}
```

---

## 📧 Contact

| Author | Email |
|--------|-------|
| Ping Wu (Corresponding Author) | pingwu@zstu.edu.cn |
| Nabin Kandel | nabinkandel60@gmail.com |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work was supported by:
- "Pioneer" and "Leading Goose" R&D Program of Zhejiang (Grant 2026C02A3004)
- National Natural Science Foundation of China (Grant 62573387)
- Natural Science Foundation of Zhejiang Province (Grant LY24F030004)
- Fundamental Research Funds of Zhejiang Sci-Tech University (25222139-Y)

---

## ⭐ Star History

If you find this repository useful, please consider giving it a star ⭐

---

**Note:** This repository will be made fully public upon paper acceptance. For review purposes, please contact the authors for access.
```
