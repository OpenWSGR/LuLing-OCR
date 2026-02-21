"""CTC 解码模块。

提供贪心解码（推理用，速度快）和束搜索解码（评估用，准确率更高）。
"""

import torch
import numpy as np
from collections import defaultdict


def greedy_decode(log_probs, vocab):
    """贪心 CTC 解码。

    Args:
        log_probs: [T, num_classes] 张量（单样本）或 numpy array
        vocab: Vocabulary 实例

    Returns:
        str: 解码后的文本
    """
    if isinstance(log_probs, torch.Tensor):
        indices = log_probs.argmax(dim=1).cpu().tolist()
    else:
        indices = np.argmax(log_probs, axis=1).tolist()

    # 合并重复 → 去除 blank
    collapsed = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != vocab.blank_label:
                collapsed.append(idx)
            prev = idx

    return vocab.decode(collapsed)


def greedy_decode_batch(log_probs, input_lengths, vocab):
    """批量贪心 CTC 解码。

    Args:
        log_probs: [T, B, num_classes] 张量
        input_lengths: [B] 每个样本的实际序列长度
        vocab: Vocabulary 实例

    Returns:
        list[str]: 解码后的文本列表
    """
    batch_size = log_probs.shape[1]
    results = []

    for b in range(batch_size):
        length = input_lengths[b].item()
        sample_probs = log_probs[:length, b, :]
        text = greedy_decode(sample_probs, vocab)
        results.append(text)

    return results


def beam_search_decode(log_probs, vocab, beam_width=10):
    """束搜索 CTC 解码。

    Args:
        log_probs: [T, num_classes] 张量或 numpy array（单样本）
        vocab: Vocabulary 实例
        beam_width: 束宽度

    Returns:
        str: 解码后的文本
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()

    T, num_classes = log_probs.shape
    blank = vocab.blank_label

    # beam: (prefix_tuple, log_prob)
    # 每个 prefix 维护两个概率：以 blank 结尾和以非 blank 结尾
    beams = {(): (0.0, float("-inf"))}  # {prefix: (p_blank, p_non_blank)}

    for t in range(T):
        new_beams = defaultdict(lambda: (float("-inf"), float("-inf")))

        for prefix, (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)

            for c in range(num_classes):
                p_c = log_probs[t, c]

                if c == blank:
                    # blank 延续当前 prefix
                    old_b, old_nb = new_beams[prefix]
                    new_beams[prefix] = (
                        np.logaddexp(old_b, p_total + p_c),
                        old_nb,
                    )
                else:
                    # 非 blank
                    new_prefix = prefix + (c,)

                    if len(prefix) > 0 and c == prefix[-1]:
                        # 重复字符：只有经过 blank 才能添加
                        old_b, old_nb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b,
                            np.logaddexp(old_nb, p_b + p_c),
                        )
                        # 不经过 blank 则延续当前 prefix
                        old_b2, old_nb2 = new_beams[prefix]
                        new_beams[prefix] = (
                            old_b2,
                            np.logaddexp(old_nb2, p_nb + p_c),
                        )
                    else:
                        old_b, old_nb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b,
                            np.logaddexp(old_nb, p_total + p_c),
                        )

        # 剪枝：保留 top-k
        scored = []
        for prefix, (p_b, p_nb) in new_beams.items():
            scored.append((prefix, np.logaddexp(p_b, p_nb), p_b, p_nb))
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {s[0]: (s[2], s[3]) for s in scored[:beam_width]}

    # 选择最优
    best_prefix = max(beams, key=lambda p: np.logaddexp(beams[p][0], beams[p][1]))
    return vocab.decode(list(best_prefix))
