import torch
import torch.nn as nn
from torchvision.models import resnet18

class OCRNet(nn.Module):
    '''
    OCRNet模型，用于OCR任务
    特征提取器使用ResNet18，LSTM用于序列建模
    目前使用ResNet18作为特征提取器，可以根据需要替换为其他模型
    
    '''
    def __init__(self, num_classes=10, lstm_hidden_size=256, lstm_num_layers=2):
        super(OCRNet, self).__init__()
        
        # 不使用预训练权重，并且移除resnet的最后两层，仅仅作为特征提取器使用
        # 由于输入图像为单通道，所以修改第一层卷积层的输入通道数
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        
        # 添加LSTM层，用于对特征进行序列建模
        feature_channels = 512
        self.lstm = nn.LSTM(
            input_size=feature_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
        )
        
        self.prediction_layer = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x):
        """前向传播

        Args:
            x: 输入图像张量, 形状为[batch, channels, height, width]

        Returns:
            输出预测，形状为[batch, sequence_length, num_classes]
        """
        features = self.feature_extractor(x)
        b, c, h, w = features.shape
        
        # 高度维度上进行平均池化，将高度压缩为1
        # [batch, channels, h, w] -> [batch, channels, 1, w]
        features = features.mean(dim=2, keepdim=True)
        
        # 移除高度维度，得到[batch, channels, w]
        features = features.squeeze(2)
        
        # 转置为[batch, w, channels]以适应LSTM输入
        features = features.permute(0, 2, 1)
        
        # 序列建模
        sequence, _ = self.lstm(features)
        
        logits = self.prediction_layer(sequence)
        
        return logits
        
