"""在线合成 OCR 数据集。

核心优化：预缓存所有单字符 glyph 图像（numpy array），
运行时只做 numpy 水平拼接 + 增强，避免每次调用 PIL 渲染。
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import cv2

from src.config import Config
from src.vocab import Vocabulary
from src.augmentation import OCRAugmentation


# 全局 glyph 缓存（单例），避免每 epoch 重建
_GLOBAL_GLYPH_CACHE = {}  # key: (font_path, frozenset(chars)) → {font_size: {char: ndarray}}


def _build_glyph_cache(chars, font_path, font_size):
    """预渲染所有单字符 glyph 为固定高度的 numpy array。

    Returns:
        dict[str, np.ndarray]: 字符 → 灰度图 (H, W), uint8
    """
    font = ImageFont.truetype(font_path, font_size)
    cache = {}

    # 用单个 draw 对象测量所有字符
    temp_img = Image.new("L", (1, 1), 0)
    temp_draw = ImageDraw.Draw(temp_img)

    for ch in chars:
        bbox = temp_draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1] + int(font_size * 0.2)
        if w <= 0 or h <= 0:
            w = max(w, font_size)
            h = max(h, font_size)

        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), ch, font=font, fill=255)
        cache[ch] = np.array(img, dtype=np.uint8)

    return cache


def _get_or_build_glyph_cache(chars, font_path, font_sizes):
    """获取或构建全局 glyph 缓存（单例模式，只在首次调用时构建）。"""
    global _GLOBAL_GLYPH_CACHE
    cache_key = (font_path, len(chars))  # 用字符数量作为简单 key

    if cache_key not in _GLOBAL_GLYPH_CACHE:
        font_name = os.path.basename(font_path)
        print(f"  Building glyph cache: {font_name} × {len(chars)} chars × {len(font_sizes)} sizes...")
        cache = {}
        for size in font_sizes:
            cache[size] = _build_glyph_cache(chars, font_path, size)
        _GLOBAL_GLYPH_CACHE[cache_key] = cache
        print(f"  Glyph cache ready: {font_name}")
    else:
        font_name = os.path.basename(font_path)
        print(f"  Reusing cached glyphs: {font_name} ({len(chars)} chars)")

    return _GLOBAL_GLYPH_CACHE[cache_key]


class SyntheticOCRDataset(Dataset):
    """在线合成 OCR 训练数据集（glyph 缓存加速版）。"""

    # 纯数字字符
    DIGITS = list("0123456789")

    def __init__(self, vocab, config, augmentation=None, num_samples=50000,
                 char_subset=None, max_text_len=None,
                 digit_ratio=0.20, repeat_ratio=0.15):
        """
        Args:
            vocab: Vocabulary 实例
            config: Config 实例
            augmentation: OCRAugmentation 实例（可选）
            num_samples: 每个 epoch 的虚拟样本数
            char_subset: 字符子集列表，None 表示使用全词表
            max_text_len: 最大文本长度（课程学习用，None 使用 config 默认值）
            digit_ratio: 纯数字样本比例（默认 0.20）
            repeat_ratio: 带重复字符样本比例（默认 0.15）
        """
        self.vocab = vocab
        self.config = config
        self.augmentation = augmentation
        self.num_samples = num_samples
        self.img_height = config.IMG_HEIGHT
        self.max_text_len = max_text_len or config.MAX_TEXT_LEN
        self.digit_ratio = digit_ratio
        self.repeat_ratio = repeat_ratio

        # 可用字符列表
        if char_subset is not None:
            self.chars = [ch for ch in char_subset if ch in vocab]
        else:
            self.chars = [vocab.idx_to_char[i] for i in range(1, vocab.num_classes)]

        # 从可用字符中筛选出数字
        self.digit_chars = [ch for ch in self.DIGITS if ch in vocab]

        # 多字体支持：为每个字体构建 glyph 缓存
        self.font_paths = config.get_font_paths()
        self.glyph_caches = {}  # {font_path: {font_size: {char: ndarray}}}
        for fp in self.font_paths:
            self.glyph_caches[fp] = _get_or_build_glyph_cache(
                self.chars, fp, config.FONT_SIZES
            )
        # 兼容旧代码：默认使用第一个字体
        self.glyph_cache = self.glyph_caches[self.font_paths[0]]

    def __len__(self):
        return self.num_samples

    def _random_text(self):
        """生成随机文本（混合策略），比例由 digit_ratio 和 repeat_ratio 控制。"""
        r = random.random()
        if r < self.digit_ratio and self.digit_chars:
            return self._random_digit_text()
        elif r < self.digit_ratio + self.repeat_ratio:
            return self._random_text_with_repeats()
        else:
            return self._random_plain_text()

    def _random_plain_text(self):
        """原始随机文本生成。"""
        length = random.randint(self.config.MIN_TEXT_LEN, self.max_text_len)
        return "".join(random.choices(self.chars, k=length))

    def _random_digit_text(self):
        """纯数字文本，高概率出现连续重复数字。"""
        length = random.randint(self.config.MIN_TEXT_LEN, self.max_text_len)
        text = []
        while len(text) < length:
            digit = random.choice(self.digit_chars)
            remaining = length - len(text)
            # 50% 概率连续重复 2-8 次
            if random.random() < 0.5 and remaining >= 2:
                repeat = random.randint(2, min(8, remaining))
                text.extend([digit] * repeat)
            else:
                text.append(digit)
        return "".join(text[:length])

    def _random_text_with_repeats(self):
        """带连续重复字符的普通文本。"""
        length = random.randint(self.config.MIN_TEXT_LEN, self.max_text_len)
        text = []
        while len(text) < length:
            ch = random.choice(self.chars)
            remaining = length - len(text)
            # 30% 概率连续重复 2-4 次
            if random.random() < 0.3 and remaining >= 2:
                repeat = random.randint(2, min(4, remaining))
                text.extend([ch] * repeat)
            else:
                text.append(ch)
        return "".join(text[:length])

    def _compose_text_image(self, text, font_size, font_path=None):
        """用预缓存的 glyph 拼接文本图像（纯 numpy 操作，极快）。

        Returns:
            np.ndarray (H, W), uint8
        """
        if font_path is None:
            font_path = self.font_paths[0]
        cache = self.glyph_caches[font_path][font_size]
        glyphs = [cache[ch] for ch in text if ch in cache]
        if not glyphs:
            return np.zeros((font_size, font_size), dtype=np.uint8)

        # 统一高度（取最大高度）
        max_h = max(g.shape[0] for g in glyphs)
        padded = []
        for g in glyphs:
            if g.shape[0] < max_h:
                pad = np.zeros((max_h - g.shape[0], g.shape[1]), dtype=np.uint8)
                g = np.vstack([g, pad])
            padded.append(g)

        # 水平拼接
        return np.hstack(padded)

    def __getitem__(self, idx):
        """生成一个训练样本。

        Returns:
            (image_tensor, target_indices, target_length)
            - image_tensor: [1, H, W] float32, 归一化到 [0, 1]
            - target_indices: list[int], 字符索引
            - target_length: int
        """
        text = self._random_text()
        font_path = random.choice(self.font_paths)
        font_size = random.choice(self.config.FONT_SIZES)

        # 用缓存拼接（纯 numpy，极快）
        img_array = self._compose_text_image(text, font_size, font_path)

        # 数据增强（需要 PIL Image）
        if self.augmentation:
            img = Image.fromarray(img_array, mode="L")
            img = self.augmentation(img)
            img_array = np.array(img, dtype=np.uint8)

        # 高度归一化（用 cv2，比 PIL 快）
        h, w = img_array.shape
        if h > 0 and w > 0:
            new_w = max(1, int(w * self.img_height / h))
            new_w = min(new_w, self.config.IMG_MAX_WIDTH)
            img_array = cv2.resize(img_array, (new_w, self.img_height),
                                   interpolation=cv2.INTER_LINEAR)

        # 转张量 [1, H, W]
        img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0).unsqueeze(0)

        # 编码标签
        target = self.vocab.encode(text)
        target_length = len(target)

        return img_tensor, target, target_length


def collate_fn(batch):
    """自定义 collate 函数，处理变宽图片和变长标签。

    图片按 batch 内最大宽度做右侧零填充。
    标签拼接为一维张量（CTC 格式）。

    Returns:
        images: [B, 1, H, max_W] float32
        targets: [sum(target_lengths)] int32
        target_lengths: [B] int32
        input_lengths: [B] int32, 每个样本的模型输出序列长度
    """
    images, targets, target_lengths = zip(*batch)

    # 图片右侧填充到最大宽度
    max_w = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w), value=0)
        padded.append(img)

    # 计算每个样本经过 CNN 后的实际序列长度（必须用原始宽度，不是 padding 后的！）
    # CNN 宽度缩减因子 = 4（两次 MaxPool(2×2) 的宽度方向）
    input_lengths = torch.IntTensor([img.shape[2] // 4 for img in images])

    images = torch.stack(padded, dim=0)  # [B, 1, H, max_W]

    # 拼接所有标签
    all_targets = []
    for t in targets:
        all_targets.extend(t)
    targets = torch.IntTensor(all_targets)
    target_lengths = torch.IntTensor(list(target_lengths))

    return images, targets, target_lengths, input_lengths
