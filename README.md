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
.venv/bin/activate
uv pip install . --index-strategy unsafe-best-match
```

### On Linux / macOS (bash/zsh):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install . --index-strategy unsafe-best-match
```

## 📌 Notes

- This project uses **PyTorch 2.3.0 + CUDA 12.1**, installed from the official PyTorch index:  
  `https://download.pytorch.org/whl/cu121`

- The flag `--index-strategy unsafe-best-match` is **required** because `uv` prioritizes the first index where a package is found.  
  Since `torch` exists on PyPI (without the `+cu121` variant), `uv` needs explicit permission to install the GPU version from the secondary index.


# About Project 

First Download the dataset and extract it .
Then Create Slices
Then Create Pad_masked_slices
Then Train model

