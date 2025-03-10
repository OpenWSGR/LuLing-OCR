import pandas
import time
import os

EXTRA_NOTES = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~·—‘’“”…、。《》【】！（），：；？￥"
LEVEL1_COMMON_WORD = "data/words/3500常用汉字.xlsx"
LEVEL2_COMMON_WORD = "data/words/7000通用汉字.xlsx"
WORDS_PREFIX = "data/words/words-"
WORDS_SUFFIX = ".txt"

def gen_word_filepath(level):
    return f"{WORDS_PREFIX}-{level}.{WORDS_SUFFIX}"

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def get_common_characters_from_csv(level):
    def read_excel(path):
        sheet = pandas.ExcelFile(path).sheet_names[0]
        df = pandas.read_excel(path, sheet_name=sheet)
        return [val[1]['hz'] for val in df.iterrows()]

    if level == 1:
        return read_excel(LEVEL1_COMMON_WORD) + list(EXTRA_NOTES)
    if level == 2:
        return read_excel(LEVEL2_COMMON_WORD) + list(EXTRA_NOTES)
    
# 示例用法
if __name__ == "__main__":
    write_file(gen_word_filepath(1), "".join(get_common_characters_from_csv(1)))
    write_file(gen_word_filepath(2), "".join(get_common_characters_from_csv(2)))
