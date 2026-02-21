#!/bin/bash
# ============================================================
# LuLing-OCR 一键环境部署脚本
# 用法: bash setup.sh
# 在租借的 GPU 机器上执行，自动安装依赖并验证环境
# ============================================================

set -e  # 出错即停

echo "=========================================="
echo "  LuLing-OCR 环境部署"
echo "=========================================="

# ---------- 1. 系统信息 ----------
echo ""
echo "[1/5] 系统信息"
echo "  OS:   $(uname -s -r)"
echo "  CPU:  $(nproc) cores"
echo "  RAM:  $(free -h | awk '/^Mem:/{print $2}')"

# ---------- 2. CUDA / GPU 检查 ----------
echo ""
echo "[2/5] GPU / CUDA 检查"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "  CUDA available via nvidia-smi: YES"
else
    echo "  WARNING: nvidia-smi not found, may not have GPU"
fi

# ---------- 3. Python 环境 ----------
echo ""
echo "[3/5] Python 环境"
PYTHON=${PYTHON:-python3}
$PYTHON --version

# 安装依赖
echo "  Installing dependencies..."
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet \
    torch torchvision \
    Pillow \
    opencv-python-headless \
    numpy \
    pandas \
    openpyxl \
    onnx \
    onnxruntime \
    tensorboard

echo "  Dependencies installed."

# ---------- 4. PyTorch + CUDA 验证 ----------
echo ""
echo "[4/5] PyTorch CUDA 验证"
$PYTHON -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU count:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
    # 快速运算测试
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'  GPU compute test: PASS')
else:
    print('  WARNING: CUDA not available, will train on CPU (very slow)')
"

# ---------- 5. 项目模块验证 ----------
echo ""
echo "[5/5] 项目模块验证"
cd "$(dirname "$0")"
$PYTHON -c "
from src.config import Config
from src.vocab import Vocabulary
from src.model import CRNN
from src.dataset import SyntheticOCRDataset, collate_fn
from src.augmentation import OCRAugmentation
import torch

config = Config()
vocab = Vocabulary(config.CHARS_L2_PATH)
model = CRNN(num_classes=vocab.num_classes,
             lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
             lstm_num_layers=config.LSTM_NUM_LAYERS)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
x = torch.randn(2, 1, 32, 200, device=device)
with torch.no_grad():
    out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f'  Vocab:  {vocab.num_classes} classes')
print(f'  Model:  {total_params:,} params ({total_params*4/1024/1024:.1f} MB)')
print(f'  Device: {device}')
print(f'  Forward pass on {device}: {x.shape} -> {out.shape}  PASS')
"

echo ""
echo "=========================================="
echo "  环境部署完成！"
echo ""
echo "  下一步："
echo "    1. 预热验证: python warmup.py"
echo "    2. 正式训练: python main.py train"
echo "=========================================="
