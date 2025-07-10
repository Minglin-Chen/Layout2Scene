# Layout2Scene

## Installation

Clone the repo via:
```bash
git clone --recurse-submodules git@github.com:Minglin-Chen/Layout2Scene.git
```

### Environment setup

```bash
conda create -n layout2scene python==3.11
conda activate layout2scene

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xformers --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Run

```bash
# (Optional)
export HF_ROOT=/path/to/local/huggingface


```