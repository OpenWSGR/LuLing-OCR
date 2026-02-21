"""数据增强模块。

模拟真实游戏 UI 截图的各种变化：
- 微旋转（±0.02 rad ≈ ±1.15°）
- 高斯噪声
- 高斯模糊
- 亮度/对比度微调
- 腐蚀/膨胀（模拟字体粗细变化）
- 缩放抖动
- 降采样再升采样（模拟多尺度变换）
"""

import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2


class OCRAugmentation:
    """OCR 数据增强变换。作用于 PIL Image（灰度模式 'L'）。"""

    def __init__(self, config):
        self.rotation_range = config.ROTATION_RANGE
        self.noise_prob = config.NOISE_PROB
        self.noise_sigma_range = config.NOISE_SIGMA_RANGE
        self.blur_prob = config.BLUR_PROB
        self.blur_radius_range = config.BLUR_RADIUS_RANGE
        self.brightness_prob = config.BRIGHTNESS_PROB
        self.brightness_range = config.BRIGHTNESS_RANGE
        self.scale_jitter_range = config.SCALE_JITTER_RANGE
        self.morph_prob = config.MORPH_PROB
        self.downsample_prob = config.DOWNSAMPLE_PROB

    def __call__(self, img):
        """对 PIL Image 施加随机增强。

        Args:
            img: PIL Image，灰度模式 'L'

        Returns:
            增强后的 PIL Image
        """
        # 1. 微旋转 (±0.02 rad)
        if random.random() < 0.5:
            angle_deg = random.uniform(
                -self.rotation_range * 180 / math.pi,
                self.rotation_range * 180 / math.pi,
            )
            img = img.rotate(angle_deg, fillcolor=0, expand=False)

        # 2. 缩放抖动
        if random.random() < 0.3:
            scale = random.uniform(*self.scale_jitter_range)
            w, h = img.size
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)

        # 3. 高斯噪声
        if random.random() < self.noise_prob:
            arr = np.array(img, dtype=np.float32)
            sigma = random.uniform(*self.noise_sigma_range)
            noise = np.random.normal(0, sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")

        # 4. 高斯模糊
        if random.random() < self.blur_prob:
            radius = random.uniform(*self.blur_radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # 5. 亮度/对比度微调
        if random.random() < self.brightness_prob:
            factor = random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # 6. 腐蚀/膨胀（模拟字体粗细变化）
        if random.random() < self.morph_prob:
            arr = np.array(img)
            kernel = np.ones((2, 2), np.uint8)
            if random.random() < 0.5:
                arr = cv2.erode(arr, kernel, iterations=1)
            else:
                arr = cv2.dilate(arr, kernel, iterations=1)
            img = Image.fromarray(arr, mode="L")

        # 7. 降采样再升采样（模拟多尺度模糊，任务书要求）
        if random.random() < self.downsample_prob:
            w, h = img.size
            scale = random.uniform(0.4, 0.7)
            small_w = max(1, int(w * scale))
            small_h = max(1, int(h * scale))
            img = img.resize((small_w, small_h), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)

        return img
