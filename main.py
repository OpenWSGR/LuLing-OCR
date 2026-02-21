"""LuLing-OCR 命令行入口。

Usage:
    python main.py train              训练模型
    python main.py finetune           微调模型（强化数字+重复字符）
    python main.py evaluate           评估模型
    python main.py infer <image>      识别单张图片
    python main.py export             导出 ONNX 模型
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    if command == "train":
        from src.train import train
        train()

    elif command == "finetune":
        from src.train import finetune
        # 支持可选参数：python main.py finetune [epochs] [lr]
        kwargs = {}
        if len(sys.argv) >= 3:
            kwargs["epochs"] = int(sys.argv[2])
        if len(sys.argv) >= 4:
            kwargs["lr"] = float(sys.argv[3])
        finetune(**kwargs)

    elif command == "evaluate":
        from src.evaluate import evaluate
        evaluate()

    elif command == "infer":
        if len(sys.argv) < 3:
            print("Usage: python main.py infer <image_path>")
            return

        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return

        from src.inference import OCREngine
        engine = OCREngine()
        result = engine.recognize(image_path)
        print(f"Result: {result}")

    elif command == "export":
        from export.export_onnx import export_to_onnx
        export_to_onnx()

    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
