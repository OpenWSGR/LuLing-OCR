"""训练主循环。

分阶段训练策略：
- Stage 1 (epoch 1-50): 仅 3500 常用字 + 符号，快速收敛
- Stage 2 (epoch 51-80): 扩展到 7000 全字库
- Stage 3 (epoch 81-100): 难例挖掘，降低 lr

日志记录：
- CSV 日志 (logs/train_log.csv)：每个 epoch 一行，便于后续分析
- TensorBoard (logs/tensorboard/)：可视化 loss/CER/accuracy 曲线
- 样本预览：每 5 个 epoch 打印预测 vs 参考对比
"""

import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.vocab import Vocabulary
from src.model import CRNN
from src.dataset import SyntheticOCRDataset, collate_fn
from src.augmentation import OCRAugmentation
from src.ctc_decode import greedy_decode_batch
from src.metrics import character_error_rate, sequence_accuracy


def get_lr_with_warmup(optimizer, epoch, warmup_epochs, base_lr):
    """线性 warmup 学习率调整。"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    return optimizer.param_groups[0]["lr"]


def get_char_subset(vocab, level):
    """获取指定级别的字符子集。"""
    if level == 1:
        chars_l1_path = Config.CHARS_L1_PATH
        with open(chars_l1_path, "r", encoding="utf-8") as f:
            return list(f.read().strip())
    else:
        return None  # 全字库


def validate(model, val_loader, vocab, device, max_batches=50):
    """验证循环。

    Returns:
        (metrics_dict, all_preds, all_refs)
    """
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_label, reduction="mean",
                             zero_infinity=True)

    all_preds = []
    all_refs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, input_lengths) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            log_probs = model(images)

            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1

            preds = greedy_decode_batch(log_probs, input_lengths, vocab)
            all_preds.extend(preds)

            offset = 0
            for tl in target_lengths:
                tl = tl.item()
                ref_indices = targets[offset:offset + tl].tolist()
                all_refs.append(vocab.decode(ref_indices))
                offset += tl

    cer = character_error_rate(all_preds, all_refs)
    seq_acc = sequence_accuracy(all_preds, all_refs)
    avg_loss = total_loss / max(num_batches, 1)

    model.train()
    return {"cer": cer, "seq_acc": seq_acc, "avg_loss": avg_loss}, all_preds, all_refs


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """保存检查点。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"  Checkpoint saved: {path}")


class TrainLogger:
    """训练日志记录器：CSV + TensorBoard。"""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)

        # CSV 日志
        self.csv_path = os.path.join(log_dir, "train_log.csv")
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "epoch", "stage", "train_loss", "val_loss",
            "cer", "seq_acc", "lr", "time_sec",
        ])

        # TensorBoard（可选，导入失败不影响训练）
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(tb_dir)
            print(f"  TensorBoard logs: {tb_dir}")
            print(f"  View: tensorboard --logdir {tb_dir}")
        except ImportError:
            print("  TensorBoard not available, using CSV only")

    def log(self, epoch, stage, train_loss, val_loss, cer, seq_acc, lr, time_sec):
        self.csv_writer.writerow([
            epoch, stage, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{cer:.6f}", f"{seq_acc:.6f}", f"{lr:.8f}", f"{time_sec:.1f}",
        ])
        self.csv_file.flush()

        if self.tb_writer:
            self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
            self.tb_writer.add_scalar("Metrics/CER", cer, epoch)
            self.tb_writer.add_scalar("Metrics/SeqAcc", seq_acc, epoch)
            self.tb_writer.add_scalar("LR", lr, epoch)

    def close(self):
        self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()


def train(config=None):
    """训练入口。"""
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 词表
    vocab = Vocabulary(config.CHARS_L2_PATH)
    print(f"Vocabulary: {vocab.num_classes} classes (含 blank)")

    # 模型
    model = CRNN(
        num_classes=vocab.num_classes,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params ({total_params * 4 / 1024 / 1024:.1f} MB)")

    # 损失函数
    ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_label, reduction="mean",
                             zero_infinity=True)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # 数据增强
    augmentation = OCRAugmentation(config)

    # 日志
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    logger = TrainLogger(log_dir)
    print(f"CSV log: {logger.csv_path}")

    # 最佳指标
    best_seq_acc = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Epochs: {config.NUM_EPOCHS}  (Stage1: 1-{config.STAGE1_EPOCHS}, "
          f"Stage2: {config.STAGE1_EPOCHS+1}-{config.STAGE2_EPOCHS}, "
          f"Stage3: {config.STAGE2_EPOCHS+1}-{config.NUM_EPOCHS})")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Samples/epoch: {config.TRAIN_SAMPLES_PER_EPOCH}")
    print(f"  LR: {config.LEARNING_RATE} -> {config.STAGE2_LR} -> {config.STAGE3_LR}")
    print()

    train_start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ====== 分阶段策略 ======
        if epoch <= config.STAGE1_EPOCHS:
            stage = 1
            char_subset = get_char_subset(vocab, level=1)
            base_lr = config.LEARNING_RATE
        elif epoch <= config.STAGE2_EPOCHS:
            stage = 2
            char_subset = None
            base_lr = config.STAGE2_LR
            if epoch == config.STAGE1_EPOCHS + 1:
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr
                print(f"\n{'='*60}")
                print(f"  Stage 2: Full charset (7000+), lr={base_lr}")
                print(f"{'='*60}\n")
        else:
            stage = 3
            char_subset = None
            base_lr = config.STAGE3_LR
            if epoch == config.STAGE2_EPOCHS + 1:
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr
                print(f"\n{'='*60}")
                print(f"  Stage 3: Hard examples, lr={base_lr}")
                print(f"{'='*60}\n")

        # Warmup（仅 Stage 1 前几个 epoch）
        if stage == 1:
            lr = get_lr_with_warmup(optimizer, epoch - 1, config.WARMUP_EPOCHS, base_lr)
        else:
            lr = optimizer.param_groups[0]["lr"]

        # 文本长度课程学习：前 10 个 epoch 逐渐增加文本长度
        # epoch 1: max_len=5, epoch 5: max_len=11, epoch 10+: max_len=20
        if epoch <= 10:
            curriculum_max_len = min(config.MAX_TEXT_LEN,
                                     max(3, 3 + int((config.MAX_TEXT_LEN - 3) * epoch / 10)))
        else:
            curriculum_max_len = config.MAX_TEXT_LEN

        if epoch <= 10 or epoch == 11:
            print(f"  Text length curriculum: max_len={curriculum_max_len}")

        # 创建本 epoch 的数据集
        train_dataset = SyntheticOCRDataset(
            vocab=vocab, config=config, augmentation=augmentation,
            num_samples=config.TRAIN_SAMPLES_PER_EPOCH, char_subset=char_subset,
            max_text_len=curriculum_max_len,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        val_dataset = SyntheticOCRDataset(
            vocab=vocab, config=config, augmentation=None,
            num_samples=config.VAL_SAMPLES, char_subset=char_subset,
            max_text_len=curriculum_max_len,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        # ====== 训练循环 ======
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets, target_lengths, input_lengths) in enumerate(train_loader):
            images = images.to(device)

            log_probs = model(images)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / num_batches
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr={lr:.6f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # ====== 验证 ======
        val_metrics, all_preds, all_refs = validate(model, val_loader, vocab, device)
        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - train_start

        # 打印 epoch 摘要
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} (Stage {stage}) "
              f"train_loss={avg_train_loss:.4f} "
              f"val_loss={val_metrics['avg_loss']:.4f} "
              f"CER={val_metrics['cer']:.4f} "
              f"SeqAcc={val_metrics['seq_acc']:.4f} "
              f"lr={lr:.6f} "
              f"time={elapsed:.0f}s "
              f"total={total_elapsed/60:.0f}min")

        # 每 5 个 epoch 打印样本预览
        if epoch % 5 == 0 or epoch == 1:
            print("  Sample predictions:")
            for i in range(min(5, len(all_preds))):
                match = "OK" if all_preds[i] == all_refs[i] else "!!"
                print(f"    [{match}] pred='{all_preds[i]}'  ref='{all_refs[i]}'")

        # 记录日志
        logger.log(epoch, stage, avg_train_loss, val_metrics["avg_loss"],
                   val_metrics["cer"], val_metrics["seq_acc"], lr, elapsed)

        # 保存最佳模型
        if val_metrics["seq_acc"] > best_seq_acc:
            best_seq_acc = val_metrics["seq_acc"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
            )

        # 每 10 个 epoch 保存一次
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch}.pth"),
            )

    total_time = time.time() - train_start
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Best SeqAcc: {best_seq_acc:.4f}")
    print(f"  Best model: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"  CSV log: {logger.csv_path}")
    print(f"{'='*60}")


def finetune(config=None, epochs=80, lr=3e-4, checkpoint_path=None,
             warmup_epochs=5, digit_ratio=0.35, repeat_ratio=0.15):
    """多字体微调：用较高学习率 + warmup + cosine 衰减重新适应新字体。

    与原始 finetune 不同，这里用接近从头训练的 lr（3e-4）来让 CNN
    特征提取器充分适应新字体的笔画粗细差异，同时通过 warmup 避免
    初始阶段跳出好的权重空间。

    Args:
        epochs: 微调轮数（默认 80，需要足够多的 epoch 学习新字体）
        lr: 峰值学习率（默认 3e-4，接近原始训练的 1e-3 但更保守）
        warmup_epochs: warmup 轮数（默认 5）
        digit_ratio: 纯数字样本比例
        repeat_ratio: 重复字符样本比例
    """
    import math

    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 词表
    vocab = Vocabulary(config.CHARS_L2_PATH)
    print(f"Vocabulary: {vocab.num_classes} classes")

    # 模型
    model = CRNN(
        num_classes=vocab.num_classes,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
    ).to(device)

    # 加载已有权重
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")
    if "metrics" in checkpoint:
        print(f"  Previous metrics: {checkpoint['metrics']}")

    # 损失函数
    ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_label, reduction="mean",
                             zero_infinity=True)

    # 优化器（新的 optimizer，不加载旧 optimizer state）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 显示字体信息
    font_paths = config.get_font_paths()
    font_names = [os.path.basename(fp) for fp in font_paths]
    print(f"\nFine-tuning config:")
    print(f"  Epochs: {epochs}, Peak LR: {lr}")
    print(f"  Warmup: {warmup_epochs} epochs → cosine decay")
    print(f"  Fonts: {', '.join(font_names)}")
    print(f"  Samples/epoch: {config.TRAIN_SAMPLES_PER_EPOCH}")
    print(f"  Data strategy: {digit_ratio:.0%} digits + {repeat_ratio:.0%} repeats + {1-digit_ratio-repeat_ratio:.0%} normal")
    print()

    # 数据增强
    augmentation = OCRAugmentation(config)

    # 日志
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    logger = TrainLogger(log_dir)

    best_seq_acc = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 备份旧模型
    import shutil
    backup_path = os.path.join(config.CHECKPOINT_DIR, "best_model_before_finetune.pth")
    if not os.path.exists(backup_path):
        shutil.copy2(checkpoint_path, backup_path)
        print(f"  Backed up model to {backup_path}")

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ====== 学习率调度：warmup + cosine decay ======
        if epoch <= warmup_epochs:
            # 线性 warmup: 从 lr/10 → lr
            current_lr = lr * (0.1 + 0.9 * epoch / warmup_epochs)
        else:
            # cosine annealing: lr → lr/100
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            current_lr = lr * 0.01 + 0.5 * (lr - lr * 0.01) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # 文本长度课程：前 5 个 epoch 从短文本开始
        if epoch <= 5:
            curriculum_max_len = min(config.MAX_TEXT_LEN,
                                     max(3, 3 + int((config.MAX_TEXT_LEN - 3) * epoch / 5)))
        else:
            curriculum_max_len = config.MAX_TEXT_LEN

        # 每个 epoch 使用全字库 + 高比例数字/重复策略
        train_dataset = SyntheticOCRDataset(
            vocab=vocab, config=config, augmentation=augmentation,
            num_samples=config.TRAIN_SAMPLES_PER_EPOCH, char_subset=None,
            max_text_len=curriculum_max_len,
            digit_ratio=digit_ratio, repeat_ratio=repeat_ratio,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        val_dataset = SyntheticOCRDataset(
            vocab=vocab, config=config, augmentation=None,
            num_samples=config.VAL_SAMPLES, char_subset=None,
            digit_ratio=digit_ratio, repeat_ratio=repeat_ratio,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        )

        # 训练循环
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets, target_lengths, input_lengths) in enumerate(train_loader):
            images = images.to(device)
            log_probs = model(images)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / num_batches
                print(f"  FT Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr={current_lr:.6f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # 验证
        val_metrics, all_preds, all_refs = validate(model, val_loader, vocab, device)
        elapsed = time.time() - epoch_start

        print(f"FT Epoch {epoch}/{epochs} "
              f"train_loss={avg_train_loss:.4f} "
              f"val_loss={val_metrics['avg_loss']:.4f} "
              f"CER={val_metrics['cer']:.4f} "
              f"SeqAcc={val_metrics['seq_acc']:.4f} "
              f"lr={current_lr:.6f} "
              f"time={elapsed:.0f}s")

        # 每 5 个 epoch 打印样本预览
        if epoch % 5 == 0 or epoch == 1:
            print("  Sample predictions:")
            for i in range(min(5, len(all_preds))):
                match = "OK" if all_preds[i] == all_refs[i] else "!!"
                print(f"    [{match}] pred='{all_preds[i]}'  ref='{all_refs[i]}'")

        # 日志
        logger.log(epoch, "FT", avg_train_loss, val_metrics["avg_loss"],
                   val_metrics["cer"], val_metrics["seq_acc"], current_lr, elapsed)

        # 保存最佳
        if val_metrics["seq_acc"] >= best_seq_acc:
            best_seq_acc = val_metrics["seq_acc"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
            )

        # 每 20 个 epoch 保存一次
        if epoch % 20 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.CHECKPOINT_DIR, f"finetune_epoch_{epoch}.pth"),
            )

    total_time = time.time() - train_start
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best SeqAcc: {best_seq_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "finetune":
        finetune()
    else:
        train()
