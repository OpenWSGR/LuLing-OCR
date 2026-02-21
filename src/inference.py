"""推理引擎。

提供 OCREngine 类，封装完整的预处理 → 模型推理 → CTC 解码流程。
支持 PyTorch 和 ONNX Runtime 两种推理后端。
"""

import os
import torch
import numpy as np
from PIL import Image

from src.config import Config
from src.vocab import Vocabulary
from src.model import CRNN
from src.preprocess import preprocess
from src.ctc_decode import greedy_decode, beam_search_decode


class OCREngine:
    """OCR 推理引擎。"""

    def __init__(self, model_path=None, vocab_path=None, config=None,
                 device="cpu", use_onnx=False):
        """
        Args:
            model_path: 模型文件路径（.pth 或 .onnx）
            vocab_path: 词表文件路径
            config: Config 实例
            device: 推理设备
            use_onnx: 是否使用 ONNX Runtime
        """
        if config is None:
            config = Config()
        self.config = config
        self.device = device

        # 加载词表
        if vocab_path is None:
            vocab_path = config.CHARS_L2_PATH
        self.vocab = Vocabulary(vocab_path)

        # 加载模型
        if model_path is None:
            model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

        self.use_onnx = use_onnx
        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
        else:
            self.model = CRNN(
                num_classes=self.vocab.num_classes,
                lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
                lstm_num_layers=config.LSTM_NUM_LAYERS,
            )
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(device)
            self.model.eval()

    def recognize(self, img_input, use_beam_search=False, beam_width=10):
        """识别单张图片中的文字。

        Args:
            img_input: 文件路径(str)、PIL Image 或 numpy array
            use_beam_search: 是否使用束搜索解码（更慢但更准确）
            beam_width: 束搜索宽度

        Returns:
            str: 识别出的文本
        """
        # 预处理
        img = preprocess(img_input, target_height=self.config.IMG_HEIGHT)

        # 转张量 [1, 1, H, W]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        if self.use_onnx:
            outputs = self.session.run(
                None, {"input": img_tensor.numpy()}
            )
            log_probs = torch.from_numpy(outputs[0])  # [T, 1, C]
        else:
            with torch.no_grad():
                log_probs = self.model(img_tensor.to(self.device))  # [T, 1, C]

        log_probs = log_probs.squeeze(1)  # [T, C]

        if use_beam_search:
            return beam_search_decode(log_probs, self.vocab, beam_width)
        else:
            return greedy_decode(log_probs, self.vocab)

    def recognize_batch(self, img_inputs):
        """批量识别。

        Args:
            img_inputs: 文件路径或图像列表

        Returns:
            list[str]: 识别结果列表
        """
        return [self.recognize(img) for img in img_inputs]
