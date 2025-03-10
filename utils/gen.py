from PIL import Image, ImageDraw, ImageFont
from utils.gen_words import write_file, gen_word_filepath

import time
import os

FONT_PATH = "data/font/SourceHanSansSC-Bold.otf"
TEXT_COLOR = (255, 255, 255)  # 白色
BG_COLOR = (0, 0, 0)  # 黑色
FONT_SIZE = 24
WORDS_PREFIX = "data/words-"
WORDS_SUFFIX = ".txt"

fonts = {}

def write_image(img: Image.Image, output_path, text=""):
    img.save(output_path, format="png")
    # write_file(output_path, text)

def text_to_images(texts, font_size) -> list[Image.Image]:
    # 加载字体
    if font_size not in fonts:
        fonts[font_size] = ImageFont.truetype(FONT_PATH, font_size)
    font = fonts[font_size]
    if not isinstance(texts, list):
        texts = [texts]
    total = len(texts)
    images = []
    
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
        images.append(image)
        
    return images

def merge_image_h(images, *args):
    # 纵向拼接合并图片
    images:list[Image.Image] = (images if isinstance(images, list) else [images]) + list(args)
    max_width = max([image.width for image in images])
    total_height = sum([image.height for image in images])
    new_image = Image.new("RGB", (max_width, total_height), BG_COLOR)
    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.height
    return new_image

def gen_common_words(level) -> str:
    with open(gen_word_filepath(level), "r", encoding="utf-8") as f:
        return f.read()

def gen_universal_7000():
    pass

def gen_test_data():
    data = [
        ["春风拂面", "柳绿花红", "燕舞蝶飞"],
        ["青山绿水", "白云悠悠", "碧波荡漾"],
        ["繁星点点", "皓月当空", "夜色温柔"],
        ["书山有路", "学海无涯", "志在四方"],
        ["鸟语花香", "桃红柳绿", "春色满园"],
        ["金风送爽", "秋菊傲霜", "丹桂飘香"],
        ["寒梅傲雪", "冰天雪地", "瑞雪兆丰年"],
        ["海阔天空", "心旷神怡", "自由自在"],
        ["晨曦初照", "朝霞满天", "新的一天"],
        ["繁花似锦", "绿草如茵", "生机勃勃"],
        ["古道西风", "瘦马夕阳", "断肠人在天涯"],
        ["山清水秀", "钟灵毓秀", "人杰地灵"],
        ["红日初升", "其道大光", "前途似锦"],
        ["岁月静好", "现世安稳", "时光温柔"],
        ["风华正茂", "书生意气", "挥斥方遒"],
        ["落英缤纷", "芳草鲜美", "桃花源里"] + ["玉树琼枝", "银装素裹", "北国风光"],
        ["星汉灿烂", "若出其里", "宇宙浩瀚"],
        ["松柏常青", "岁寒知松", "坚韧不拔"],
        ["海誓山盟", "白头偕老", "情比金坚"],
        ["云卷云舒", "花开花落", "岁月悠悠"],
        ["繁花似锦", "绿草如茵", "春意盎然", "万象更新"],
        ["金风玉露", "一相逢", "便胜却人间无数"]
    ]
    for img in data:
        print("P:", img)
        image = merge_image_h(text_to_images(img, FONT_SIZE))
        write_image(image, f"train/test-{str(time.time())}.png")
        
# 示例用法
if __name__ == "__main__":
    print(len(gen_common_words(2)))
    pass
