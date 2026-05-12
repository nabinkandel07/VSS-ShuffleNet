# VSS-ShuffleNet: Code for EAAI Review

**Paper Title:** VSS-ShuffleNet: A Lightweight YOLOv8-Based Framework for Real-Time Steel Surface Defect Detection  
**Journal:** Engineering Applications of Artificial Intelligence  
**Status:** Under Review

## ⚠️ Important Note

This code is provided for **review purposes only**. The code will be publicly released upon paper acceptance.

## 📊 Key Results (From Paper)

| Configuration | mAP@0.5 | Params (M) | FPS |
|---------------|---------|------------|-----|
| YOLOv8n (baseline) | 79.0% | 3.0 | 101 |
| + Preprocessing | 88.7% | 3.0 | 178.6 |
| + ShuffleNet V2 | 91.4% | 3.8 | 200.0 |
| + VSS Block | 92.7% | 3.5 | 230.3 |
| **Full VSS-ShuffleNet** | **94.6%** | **9.7** | **147** |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
