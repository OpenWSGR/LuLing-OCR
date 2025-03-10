from PIL import Image, ImageDraw, ImageFont
from .gen_words import write_file, gen_word_filepath

import time
import os

FONT_PATH = "SourceHanSansSC-Bold.otf"
TEXT_COLOR = (255, 255, 255)  # 白色
BG_COLOR = (0, 0, 0)  # 黑色
FONT_SIZE = 24
WORDS_PREFIX = "data/words-"
WORDS_SUFFIX = ".txt"

fonts = {}

def text_to_image(texts, font_size, output_path):
    # 加载字体
    prefix = str(time.time())
    if font_size not in fonts:
        fonts[font_size] = ImageFont.truetype(FONT_PATH, font_size)
    font = fonts[font_size]
    if not isinstance(texts, list):
        texts = [texts]
    total = len(texts)
    
    for i, text in enumerate(texts):
        if i < 5 or i % 100 == 0:
            print(f"正在处理第 {i + 1}/{total} 个请求: {text}")
            
        # 创建临时图像以计算文字大小
        temp_image = Image.new("RGB", (1, 1), BG_COLOR)
        temp_draw = ImageDraw.Draw(temp_image)
        left, top, right, bottom = temp_draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top + int(font_size * 0.5)
        
        # 创建足够大的图像
        image = Image.new("RGB", (text_width, text_height), BG_COLOR)
        draw = ImageDraw.Draw(image)

        # 绘制文字
        draw.text((0, 0), text, font=font, fill=TEXT_COLOR, )

        # 保存为 WebP 格式        
        image.save(os.path.join(output_path, f"{prefix}.{i}.webp"), format="WEBP")
        write_file(os.path.join(output_path, f"{prefix}.{i}.txt"), text)
        

def gen_common_words(level) -> str:
    with open(gen_word_filepath(level), "r", encoding="utf-8") as f:
        return f.read()

def gen_universal_7000():
    pass

# 示例用法
if __name__ == "__main__":
    print(len(gen_common_words(2)))
    pass
