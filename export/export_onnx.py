"""ONNX 导出脚本。

将训练好的 PyTorch 模型导出为 ONNX 格式，支持 ONNX Runtime 推理。
动态轴允许可变宽度输入。
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.vocab import Vocabulary
from src.model import CRNN


def export_to_onnx(checkpoint_path=None, output_path=None, config=None):
    """导出 ONNX 模型。

    Args:
        checkpoint_path: PyTorch 检查点路径
        output_path: ONNX 输出路径
        config: Config 实例
    """
    if config is None:
        config = Config()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if output_path is None:
        output_path = config.ONNX_EXPORT_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载词表和模型
    vocab = Vocabulary(config.CHARS_L2_PATH)
    model = CRNN(
        num_classes=vocab.num_classes,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 虚拟输入 [B, C, H, W]
    dummy_input = torch.randn(1, 1, config.IMG_HEIGHT, 400)

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 3: "width"},
            "output": {0: "time", 1: "batch"},
        },
    )

    # 验证
    onnx_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX model exported to: {output_path}")
    print(f"ONNX model size: {onnx_size_mb:.1f} MB")

    # 可选：验证 ONNX 输出与 PyTorch 一致
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

        with torch.no_grad():
            torch_output = model(dummy_input).numpy()

        diff = np.abs(onnx_output - torch_output).max()
        print(f"Max difference (PyTorch vs ONNX): {diff:.6e}")
        print(f"Verification: {'PASS' if diff < 1e-4 else 'WARN (large diff)'}")
    except ImportError:
        print("onnxruntime not installed, skipping verification.")


if __name__ == "__main__":
    export_to_onnx()
