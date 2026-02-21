"""词表管理模块。

从字符列表文件构建确定性的 字符↔索引 映射。
index 0 保留给 CTC blank token。
"""


class Vocabulary:
    def __init__(self, chars_file_path):
        """从字符列表文件构建词表。

        Args:
            chars_file_path: 字符列表文件路径 (chars_l1.txt 或 chars_l2.txt)，
                             文件内容为一行连续字符。
        """
        with open(chars_file_path, "r", encoding="utf-8") as f:
            chars = f.read().strip()

        self.blank_label = 0
        self.char_to_idx = {}
        self.idx_to_char = {0: ""}  # blank 解码为空字符串

        for i, ch in enumerate(chars):
            idx = i + 1  # 0 留给 blank
            self.char_to_idx[ch] = idx
            self.idx_to_char[idx] = ch

    @property
    def num_classes(self):
        """总类别数（含 blank）。"""
        return len(self.idx_to_char)

    def encode(self, text):
        """将文本字符串编码为索引列表。

        Args:
            text: 输入文本

        Returns:
            list[int]: 索引列表，跳过词表中不存在的字符
        """
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, indices):
        """将索引列表解码为文本字符串（跳过 blank）。

        Args:
            indices: 索引列表

        Returns:
            str: 解码后的文本
        """
        return "".join(
            self.idx_to_char[idx] for idx in indices
            if idx in self.idx_to_char and idx != self.blank_label
        )

    def __len__(self):
        return self.num_classes

    def __contains__(self, char):
        return char in self.char_to_idx
