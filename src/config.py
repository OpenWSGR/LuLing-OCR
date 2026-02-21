import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # ==================== 路径 ====================
    FONT_PATH = os.path.join(BASE_DIR, "data", "fonts", "SourceHanSansSC-Bold.otf")
    FONT_DIR = os.path.join(BASE_DIR, "data", "fonts")

    # 多字体支持：自动扫描 data/fonts/ 下所有 .otf/.ttf 文件
    @classmethod
    def get_font_paths(cls):
        """获取所有可用字体路径，至少包含主字体。"""
        paths = []
        if os.path.isdir(cls.FONT_DIR):
            for f in sorted(os.listdir(cls.FONT_DIR)):
                if f.lower().endswith(('.otf', '.ttf')):
                    paths.append(os.path.join(cls.FONT_DIR, f))
        if not paths:
            paths = [cls.FONT_PATH]
        return paths
    CHARS_L1_PATH = os.path.join(BASE_DIR, "data", "charsets", "chars_l1.txt")
    CHARS_L2_PATH = os.path.join(BASE_DIR, "data", "charsets", "chars_l2.txt")
    TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    ONNX_EXPORT_PATH = os.path.join(BASE_DIR, "export", "luling_ocr.onnx")

    # ==================== 图像 ====================
    IMG_HEIGHT = 32
    IMG_MAX_WIDTH = 400
    IMG_CHANNELS = 1

    # ==================== 模型 ====================
    CNN_OUTPUT_CHANNELS = 512
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # ==================== 训练 ====================
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 5
    GRAD_CLIP_MAX_NORM = 5.0
    NUM_WORKERS = 4         # GPU 训练用多 worker 喂数据

    # Stage 分界
    STAGE1_EPOCHS = 50      # 仅常用字
    STAGE2_EPOCHS = 80      # 全字库
    STAGE2_LR = 3e-4
    STAGE3_LR = 1e-4

    # 每 epoch 训练样本数（在线生成，glyph 缓存后速度快）
    TRAIN_SAMPLES_PER_EPOCH = 200000
    VAL_SAMPLES = 10000

    # ==================== 数据生成 ====================
    FONT_SIZES = [16, 20, 24, 28, 32]
    MIN_TEXT_LEN = 1
    MAX_TEXT_LEN = 20
    TEXT_COLOR = (255, 255, 255)    # 白色
    BG_COLOR = (0, 0, 0)           # 黑色

    # ==================== 数据增强 ====================
    ROTATION_RANGE = 0.02   # 弧度 (≈1.15°)
    NOISE_PROB = 0.3
    NOISE_SIGMA_RANGE = (3, 15)
    BLUR_PROB = 0.2
    BLUR_RADIUS_RANGE = (0.5, 1.5)
    BRIGHTNESS_PROB = 0.3
    BRIGHTNESS_RANGE = (0.7, 1.3)
    SCALE_JITTER_RANGE = (0.9, 1.1)
    MORPH_PROB = 0.2
    DOWNSAMPLE_PROB = 0.15  # 降采样再升采样模拟模糊

    # ==================== 词表 ====================
    VOCAB_LEVEL = 2         # 1=3500+符号, 2=7000+符号
    CTC_BLANK_IDX = 0
