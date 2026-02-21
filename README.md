# LuLing-OCR

轻量级单行印刷体 OCR 引擎，为 [AutoWSGR](https://github.com/huan-yp/Auto-WSGR)（战舰少女R 自动化框架）提供游戏 UI 文字识别能力。

## 特性

- **CRNN 架构**：5 层 CNN + 2 层双向 LSTM + CTC 解码
- **轻量高效**：8.9M 参数，模型 ~102 MB，CPU 推理 < 100ms
- **大字库**：支持 7000 常用汉字 + 116 符号/字母（共 7117 类）
- **在线合成训练**：基于字体渲染的合成数据，无需收集标注真实数据
- **预处理管道**：自动颜色通道选择、对比度拉伸、极性检测

## 快速开始

### 安装依赖

```bash
pip install torch torchvision pillow opencv-python-headless numpy
```

### 推理

从 [Releases](https://github.com/OpenWSGR/LuLing-OCR/releases) 下载 `best_model.pth` 放到 `checkpoints/` 目录，然后：

```bash
python main.py infer <图片路径>
```

或在代码中调用：

```python
from src.inference import OCREngine

engine = OCREngine()
text = engine.recognize("screenshot.png")
```

### 训练

需要 Source Han Sans SC 字体文件，放到 `data/fonts/` 目录：

```
data/fonts/SourceHanSansSC-Bold.otf      # 必需
data/fonts/SourceHanSansSC-Regular.otf   # 可选
data/fonts/SourceHanSansSC-Light.otf     # 可选
```

GPU 训练（推荐）：

```bash
bash setup.sh          # 环境检查 + 依赖安装
python main.py train   # 100 epoch 分阶段训练
```

微调已有模型：

```bash
python main.py finetune [epochs] [lr]
```

### 评估

```bash
python main.py evaluate
```

使用 `data/test/` 下的 20 组标注测试数据计算序列准确率和 CER。

## 架构

```
输入图片 → 预处理 → CRNN → CTC 解码 → 文本
```

**预处理管道**（`src/preprocess.py`）：
1. 最佳通道灰度化（自动选择 B/G/R/Gray 中对比度最高的通道）
2. 百分位对比度拉伸
3. 极性归一化（确保白字黑底）
4. 高度归一化到 32px

**模型**（`src/model.py`）：
```
Conv2d(1→64) + BN + ReLU + MaxPool(2×2)
Conv2d(64→128) + BN + ReLU + MaxPool(2×2)
Conv2d(128→256) + BN + ReLU
Conv2d(256→256) + BN + ReLU + MaxPool(2×1)
Conv2d(256→512) + BN + ReLU
→ Height Mean Pooling → [B, 512, T]
→ BiLSTM(512, hidden=256, layers=2) → [T, B, 512]
→ Linear(512 → 7117) + LogSoftmax
```

**训练策略**（`src/train.py`）：
- Stage 1 (epoch 1-50)：3500 常用字，lr=1e-3
- Stage 2 (epoch 51-80)：7000 全字库，lr=3e-4
- Stage 3 (epoch 81-100)：难例强化，lr=1e-4
- Warmup 5 epoch + 文本长度课程学习

## 项目结构

```
├── src/
│   ├── config.py          # 配置中心
│   ├── vocab.py           # 词表管理（encode/decode）
│   ├── preprocess.py      # 预处理管道
│   ├── augmentation.py    # 数据增强（旋转/噪声/模糊/亮度）
│   ├── dataset.py         # 在线合成数据集 + glyph 缓存
│   ├── model.py           # CRNN 模型定义
│   ├── ctc_decode.py      # 贪心/束搜索解码
│   ├── metrics.py         # CER、序列准确率
│   ├── train.py           # 训练 + 微调
│   ├── evaluate.py        # 评估脚本
│   └── inference.py       # 推理引擎（OCREngine）
├── export/
│   └── export_onnx.py     # ONNX 导出
├── data/
│   ├── charsets/           # 字符集定义
│   ├── fonts/              # 字体文件（需自行下载）
│   └── test/               # 标注测试数据
├── main.py                 # CLI 入口
├── setup.sh                # GPU 环境一键部署
└── requirements.txt
```

## 性能指标

| 指标 | 结果 |
|------|------|
| 合成测试集准确率 | 20/20 (100%) |
| 字符错误率 (CER) | 0.0000 |
| 模型大小 | 102.4 MB |
| 推理速度 (CPU) | < 100 ms |
| 支持字符数 | 7117 |

## License

MIT
