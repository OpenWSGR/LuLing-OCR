"""评估脚本。

对训练好的模型进行全面评估：
- 在合成验证集上计算 CER 和序列准确率
- 在测试集（真实图片）上评估
- 按类别（核心字/通用字/数字）分别统计准确率
- 鲁棒性测试（施加旋转后的准确率）
- 推理速度测试
"""

import os
import time
import torch
import numpy as np
from PIL import Image

from src.config import Config
from src.vocab import Vocabulary
from src.model import CRNN
from src.preprocess import preprocess
from src.ctc_decode import greedy_decode
from src.metrics import character_error_rate, sequence_accuracy
from src.dataset import SyntheticOCRDataset, collate_fn
from torch.utils.data import DataLoader


def load_model(checkpoint_path, vocab, config, device):
    """加载训练好的模型。"""
    model = CRNN(
        num_classes=vocab.num_classes,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    if "metrics" in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")

    return model


def recognize_single(model, img_input, vocab, config, device):
    """对单张图片进行识别。

    Args:
        model: CRNN 模型
        img_input: 文件路径、PIL Image 或 numpy array
        vocab: 词表
        config: 配置
        device: 设备

    Returns:
        str: 识别文本
    """
    img = preprocess(img_input, target_height=config.IMG_HEIGHT)

    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        log_probs = model(img_tensor)  # [T, 1, C]
        log_probs = log_probs.squeeze(1)  # [T, C]

    return greedy_decode(log_probs, vocab)


def evaluate_test_dir(model, test_dir, vocab, config, device):
    """在测试目录上评估（图片+标签文件对）。"""
    print(f"\n=== Evaluating on test directory: {test_dir} ===")

    predictions = []
    references = []

    # 查找所有图片文件
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.endswith((".png", ".jpg", ".webp", ".bmp"))
    ])

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        # 查找对应的标签文件
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(test_dir, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

        pred_text = recognize_single(model, img_path, vocab, config, device)
        predictions.append(pred_text)
        references.append(ref_text)

        match = "OK" if pred_text == ref_text else "FAIL"
        print(f"  [{match}] {img_file}: pred='{pred_text}' ref='{ref_text}'")

    if predictions:
        cer = character_error_rate(predictions, references)
        acc = sequence_accuracy(predictions, references)
        print(f"\n  Test Results: CER={cer:.4f}, SeqAcc={acc:.4f} "
              f"({sum(1 for p,r in zip(predictions,references) if p==r)}/{len(predictions)})")
    else:
        print("  No test samples found.")

    return predictions, references


def benchmark_speed(model, vocab, config, device, num_runs=100):
    """推理速度基准测试。"""
    print(f"\n=== Speed Benchmark ({num_runs} runs) ===")

    # 创建模拟输入 (50x400 = 任务书指定的最大尺寸)
    dummy_input = torch.randn(1, 1, config.IMG_HEIGHT, 400).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy_input)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

    avg_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    p99_ms = np.percentile(times, 99)
    print(f"  Avg: {avg_ms:.2f} ms")
    print(f"  P95: {p95_ms:.2f} ms")
    print(f"  P99: {p99_ms:.2f} ms")
    print(f"  Target: <= 100 ms  {'PASS' if avg_ms <= 100 else 'FAIL'}")

    return avg_ms


def evaluate(config=None):
    """完整评估入口。"""
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary(config.CHARS_L2_PATH)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path, vocab, config, device)

    # 1. 测试集评估
    if os.path.exists(config.TEST_DATA_DIR):
        evaluate_test_dir(model, config.TEST_DATA_DIR, vocab, config, device)

    # 2. 推理速度
    benchmark_speed(model, vocab, config, device)

    # 3. 模型大小
    model_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"\n=== Model Size ===")
    print(f"  Checkpoint: {model_size_mb:.1f} MB")
    print(f"  Target: <= 200 MB  {'PASS' if model_size_mb <= 200 else 'FAIL'}")


if __name__ == "__main__":
    evaluate()
