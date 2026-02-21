"""CRNN 模型定义。

轻量级 5 层 CNN + 2 层双向 LSTM + CTC 投影层。
输入: [B, 1, 32, W]  (单通道灰度图，高度 32，宽度可变)
输出: [T, B, num_classes]  (log_softmax 概率，T = W/4)
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """轻量级 CRNN OCR 模型。

    CNN 部分：5 层卷积，宽度方向总缩减 4 倍（两次 MaxPool 2×2），
    高度方向缩减到 4（两次 2×2 + 一次 2×1），然后做 height-mean-pooling。
    LSTM 部分：2 层双向 LSTM。
    输出：CTC 格式的 log_softmax。
    """

    def __init__(self, num_classes, lstm_hidden_size=256, lstm_num_layers=2):
        """
        Args:
            num_classes: 总类别数（含 CTC blank）
            lstm_hidden_size: LSTM 隐层大小
            lstm_num_layers: LSTM 层数
        """
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1: 1 → 64, MaxPool(2,2) → H/2, W/2
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 → 128, MaxPool(2,2) → H/4, W/4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128 → 256 (无池化)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: 256 → 256, MaxPool(2,1) → H/8, W/4 (仅高度减半)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 5: 256 → 512 (无池化)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=False,
        )

        # CTC 投影层
        lstm_output_size = lstm_hidden_size * 2  # 双向
        self.projection = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        """前向传播。

        Args:
            x: [B, 1, H, W] 输入图像张量

        Returns:
            log_probs: [T, B, num_classes] CTC 格式的 log 概率
        """
        # CNN 特征提取
        features = self.cnn(x)  # [B, 512, H', W/4]

        # 高度维度平均池化 → [B, 512, W/4]
        features = features.mean(dim=2)

        # 转为 LSTM 输入格式 [T, B, C]，T = W/4
        features = features.permute(2, 0, 1)

        # 序列建模
        sequence, _ = self.lstm(features)  # [T, B, 512]

        # 投影 + log_softmax
        logits = self.projection(sequence)  # [T, B, num_classes]
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        return log_probs
