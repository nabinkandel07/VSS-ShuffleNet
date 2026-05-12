# 🚀 Complete Mamba Installation & Integration Guide (for YOLO)
This guide covers the full installation of **Mamba-SSM**, its dependencies, and how to integrate it into your YOLO (Ultralytics) project, based on your steps.

---

## 📋 1. Prerequisites (Environment Requirements)
You must have the following setup before proceeding:

| Component | Requirement |
|-----------|------------|
| OS        | Linux (recommended; Windows may have compatibility issues) |
| GPU       | NVIDIA GPU with CUDA support |
| PyTorch   | ≥ 1.12 (your environment uses `2.0.0+cu118`) |
| CUDA      | ≥ 11.6 (your environment uses `11.8`) |

> **Note**: PyTorch and CUDA versions must match exactly. Most installation errors come from version mismatches.

```bash
# Check your versions
pip list | grep torch
nvcc --version
```

<img width="531" height="85" alt="image" src="https://github.com/user-attachments/assets/38bf4fcf-9294-4bd0-91a2-85446419196a" />


---

## 🛠️ 2. Install Required Dependencies
### 2.1 Install `mmcv`
If the basic install fails, use `mim`:
```bash
# Option 1: Basic install
pip install mmcv

# Option 2: If install fails
pip install -U openmim
mim install mmcv
```

### 2.2 Install `causal-conv1d`
This is required for Mamba kernels:
```bash
pip install causal-conv1d
```
> If the build hangs, this is usually a network issue. Try installing a pre-built wheel or using a mirror.

### 2.3 Install `mamba-ssm`
```bash
pip install mamba-ssm
```
> Again, if building wheels takes too long, check your network connection or use a local mirror.

---

## ⚙️ 3. Compile Custom Mamba Modules
You need to compile two custom modules: `mamba` and `selective_scan` into your Ultralytics project.

### 3.1 Prepare Project Structure
1.  Download the custom `mamba` and `selective_scan` modules from https://github.com/nabinkandel07/VSS-ShuffleNet
2.  Unzip them and place them in your project:
    ```
    ultralytics/
    └── nn/
        └── AddModules/
            ├── mamba/
            └── selective_scan/
    ```
<img width="261" height="377" alt="image" src="https://github.com/user-attachments/assets/bb3555cf-caa3-4f21-99a3-34ee87d46bc6" />


### 3.2 Compile `mamba` Module
```bash
# Navigate to the directory
cd ultralytics/nn/AddModules/mamba

# Compile and install
python setup.py install
```

If you get a "CUDA version mismatch" error, clean the build first:
```bash
python setup.py clean --all
python setup.py install
```

### 3.3 Compile `selective_scan` Module
```bash
# Navigate to the directory
cd ultralytics/nn/AddModules/selective_scan
<img width="254" height="375" alt="image" src="https://github.com/user-attachments/assets/c70eef44-83d3-4760-a5d3-028c9cef469d" />

# Compile and install
python setup.py install
```
<img width="672" height="343" alt="image" src="https://github.com/user-attachments/assets/d76b4fe7-2e68-485d-aaa5-7f8838468e33" />

---

## 🔧 4. Configure Your Codebase
Now, add the Mamba-YOLO modules to your Ultralytics codebase.

### 4.1 Create `mamba_yolo.py`
1.  Create the file `ultralytics/nn/AddModules/mamba_yolo.py`.
2.  Paste your Mamba-YOLO model code into this file.


### 4.2 Update `__init__.py`
Add the new module to `ultralytics/nn/AddModules/__init__.py`:
```python
# Add this line
from .mamba_yolo import *
```
<img width="498" height="387" alt="image" src="https://github.com/user-attachments/assets/5a576b6b-240a-4e2b-bae1-f4f3676fe1af" />



### 4.3 Modify `tasks.py`
Edit `ultralytics/nn/modules/tasks.py` to register your new modules.

1.  **Import the module** at the top:
    ```python
    from .AddModules import *
    ```
<img width="498" height="387" alt="image" src="https://github.com/user-attachments/assets/b14300dc-cbf4-46f3-891f-f97177296785" />

2.  **Register the modules** in the `parse_model` function (add them to the list):
    ```python
    if m in {
        # ... (other modules)
        SimpleStem, VisionClueMerge, VSSBlock_YOLO, XSSBlock
    }:
    ```
<img width="744" height="235" alt="image" src="https://github.com/user-attachments/assets/756ca168-9b17-448d-b82d-6cc385d2a596" />

<img width="1429" height="430" alt="image" src="https://github.com/user-attachments/assets/30109c46-e38b-41bf-9832-fcbaaa0465b7" />


3.  **Fix the `DetectionModel` class** (if you get a stride error):
    Find this line:
    ```python
    m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
    ```
    Replace it with the following try/except block to handle both CPU and CUDA:
    ```python
    try:
        m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
    except RuntimeError:
        try:
            self.model.to(torch.device('cuda'))
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(
                torch.zeros(1, ch, s, s).to(torch.device('cuda')))])
        except RuntimeError as error:
            raise error
    ```
    *(Or comment out the original line entirely if you just want to bypass it)*

---

## ✅ 5. Final Verification
To verify everything works, run a simple test in Python:
```python
import torch
from ultralytics import YOLO

# Load your custom Mamba-YOLO config
model = YOLO("your_custom_mamba_yolo.yaml")

# Test with a dummy input
dummy_input = torch.randn(1, 3, 640, 640)
output = model(dummy_input)
print("✅ Mamba-YOLO installed and verified successfully!")
```
