"""预处理管道：特征工程前置。

遵循任务书"不要过分依赖端到端黑盒模型"的核心思想，
用传统图像处理先把图像清洗干净，降低神经网络负担。

管道：灰度化 → 二值化(Otsu) → 倾斜校正 → 边缘裁剪 → 高度归一化
"""

import cv2
import numpy as np
from PIL import Image


def to_grayscale(img):
    """转为灰度图。

    Args:
        img: numpy array (H, W) 或 (H, W, 3)

    Returns:
        numpy array (H, W), uint8
    """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize(img):
    """Otsu 自适应二值化。

    Args:
        img: 灰度图 numpy array (H, W), uint8

    Returns:
        二值图 numpy array (H, W), uint8, 值为 0 或 255
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def correct_tilt(img, max_angle_deg=3.0, angle_step=0.1):
    """基于水平投影方差最大化的倾斜校正。

    原理：正水平的文本行具有最大的水平投影方差（行有字处投影值高，行间隙投影值低）。
    搜索 [-max_angle, +max_angle] 范围内使水平投影方差最大的角度进行校正。

    Args:
        img: 二值图 numpy array (H, W)
        max_angle_deg: 最大搜索角度（度）
        angle_step: 搜索步长（度）

    Returns:
        校正后的图像 numpy array (H, W)
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    best_angle = 0.0
    best_score = -1.0

    angle = -max_angle_deg
    while angle <= max_angle_deg:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (w, h), borderValue=0)
        projection = np.sum(rotated, axis=1, dtype=np.float64)
        score = np.var(projection)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += angle_step

    if abs(best_angle) > angle_step * 0.5:
        rot_mat = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, (w, h), borderValue=0)

    return img


def crop_edges(img, threshold=0):
    """裁剪多余黑边，紧贴文字区域。

    Args:
        img: numpy array (H, W)
        threshold: 像素值阈值，大于此值视为有内容

    Returns:
        裁剪后的图像；如果图像全黑则返回原图
    """
    coords = np.argwhere(img > threshold)
    if coords.size == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return img[y_min:y_max + 1, x_min:x_max + 1]


def normalize_height(img, target_height=32):
    """等比缩放到固定高度。

    Args:
        img: numpy array (H, W)
        target_height: 目标高度

    Returns:
        numpy array (target_height, new_W)
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, 1), dtype=np.uint8)

    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def _best_channel_grayscale(img_bgr):
    """从彩色图像中选择对比度最高的通道作为灰度图。

    对于纯色文字+纯色背景，某个颜色通道可能比标准灰度转换
    有更高的对比度（例如红字绿底在 R 通道对比度远高于灰度图）。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best = gray
    best_contrast = float(np.percentile(gray, 95)) - float(np.percentile(gray, 5))

    for ch in cv2.split(img_bgr):  # B, G, R
        contrast = float(np.percentile(ch, 95)) - float(np.percentile(ch, 5))
        if contrast > best_contrast:
            best = ch
            best_contrast = contrast

    return best


def load_image(img_input):
    """加载图像并转为高对比度灰度图。

    对于彩色图像，自动选择对比度最高的通道（而非简单灰度化），
    以处理各种颜色组合的纯色文字。

    Args:
        img_input: 文件路径(str)、PIL Image、或 numpy array

    Returns:
        灰度 numpy array (H, W), uint8
    """
    if isinstance(img_input, str):
        img_color = cv2.imread(img_input, cv2.IMREAD_COLOR)
        if img_color is None:
            raise FileNotFoundError(f"无法读取图像: {img_input}")
        if img_color.ndim == 3 and img_color.shape[2] == 3:
            img = _best_channel_grayscale(img_color)
        else:
            img = img_color
    elif isinstance(img_input, Image.Image):
        img_input = img_input.convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
        img = _best_channel_grayscale(img_bgr)
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 3 and img_input.shape[2] == 3:
            img = _best_channel_grayscale(img_input)
        else:
            img = to_grayscale(img_input)
    else:
        raise TypeError(f"不支持的输入类型: {type(img_input)}")
    return img


def normalize_contrast(img):
    """百分位对比度拉伸，将纯色文字图像拉到 0-255 全范围。

    使用 2%/98% 百分位避免噪声像素干扰，保留抗锯齿渐变。
    """
    low = float(np.percentile(img, 2))
    high = float(np.percentile(img, 98))

    if high - low < 10:
        # 对比度极低，无法拉伸
        return img

    stretched = (img.astype(np.float32) - low) / (high - low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def ensure_white_on_black(img):
    """确保图像为白字黑底（与训练数据一致）。

    通过比较边缘像素均值和图像均值判断极性：
    如果背景（边缘）亮度高于前景，则反转。
    """
    h, w = img.shape
    # 取边缘 2 像素作为背景采样
    border = np.concatenate([
        img[0, :], img[-1, :], img[:, 0], img[:, -1]
    ])
    bg_mean = border.mean()

    # 背景亮 → 黑字白底 → 需要反转
    if bg_mean > 127:
        return 255 - img
    return img


def normalize_stroke_width(img, target_ratio=0.08):
    """笔画粗细归一化：将细笔画（Regular/Light）膨胀到接近 Bold。

    通过距离变换估算笔画半宽占图像高度的比例。
    Bold 字体约 8-12%，Regular 约 5-7%，Light 约 3-5%。
    如果检测到比目标更细的笔画，用形态学膨胀加粗。

    Args:
        img: 白字黑底灰度图 (H, W), uint8
        target_ratio: 目标笔画半宽/图像高度比（默认 0.08）

    Returns:
        归一化后的灰度图
    """
    h, w = img.shape
    if h == 0 or w == 0:
        return img

    # 二值化分析笔画
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_count = np.sum(binary > 0)
    if white_count < 10:
        return img

    # 距离变换：每个白色像素到最近黑色像素的距离
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_values = dist[binary > 0]
    median_half_width = float(np.median(dist_values))

    # 当前笔画占比
    current_ratio = median_half_width / h

    if current_ratio >= target_ratio:
        return img  # Bold 或更粗，无需处理

    # 计算需要膨胀的像素数
    dilation_px = max(1, int(round((target_ratio - current_ratio) * h)))
    dilation_px = min(dilation_px, 3)  # 最多膨胀 3 像素，避免过度

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    return cv2.dilate(img, kernel, iterations=1)


def preprocess(img_input, target_height=32, full_pipeline=False):
    """预处理管道。

    默认只做灰度化 + 极性归一化 + 高度归一化（与训练数据一致）。
    自动检测黑字白底并反转为白字黑底。
    full_pipeline=True 时额外启用二值化 + 倾斜校正 + 裁剪。

    Args:
        img_input: 文件路径(str)、PIL Image、或 numpy array
        target_height: 目标高度
        full_pipeline: 是否使用完整预处理（二值化+校正+裁剪）

    Returns:
        预处理后的灰度 numpy array，固定高度
    """
    img = load_image(img_input)

    # 对比度拉伸：纯色文字+纯色背景 → 全范围灰度
    img = normalize_contrast(img)

    # 极性归一化：确保白字黑底
    img = ensure_white_on_black(img)

    if full_pipeline:
        img = binarize(img)
        img = correct_tilt(img)
        img = crop_edges(img)

    # 高度归一化
    img = normalize_height(img, target_height)

    return img
