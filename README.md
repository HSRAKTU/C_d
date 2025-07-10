# 🚀 Project Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environments and dependency resolution.

## 📦 Install `uv`

### On **Windows** (PowerShell)
```powershell
curl -sSf https://astral.sh/uv/install.ps1 | iex
```

### On Linux / macOS (bash/zsh):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
ℹ️ After installation, make sure uv is on your PATH. You may need to restart your terminal or manually add ~/.cargo/bin to your shell config.

## 🔧 Environment Setup
Create and activate a virtual environment, then install dependencies with CUDA-enabled PyTorch:

### On **Windows** (PowerShell)
```powershell
uv venv .venv
.venv/Scripts/activate
uv sync
```

### On Linux / macOS (bash/zsh):
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## 📌 Notes

- This project uses **PyTorch 2.3.0 + CUDA 12.1**, installed from the official PyTorch index:  
  `https://download.pytorch.org/whl/cu121`


# Get Started

1. Download the data set.
2. Slice (optionally visualize)
```
python -m src.main slice
```
3. Prepare Dataset
```
# For prepaing without padding:
python -m src.main prep

# For preparing with padding:
python -m src.main prep --target-points 6500
```
4. Train
5. Evaluate
6. Predict

