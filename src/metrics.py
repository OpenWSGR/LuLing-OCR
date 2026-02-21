"""评估指标模块。

- 字符错误率 (CER)
- 序列准确率 (Sequence Accuracy)
- Levenshtein 编辑距离
"""


def levenshtein_distance(s1, s2):
    """计算两个字符串的 Levenshtein 编辑距离。

    Args:
        s1, s2: 字符串

    Returns:
        int: 编辑距离
    """
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # 使用一维 DP 优化空间
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,       # 插入
                prev[j] + 1,            # 删除
                prev[j - 1] + cost,     # 替换
            )
        prev, curr = curr, prev

    return prev[len2]


def character_error_rate(predictions, references):
    """计算字符错误率 (CER)。

    CER = sum(edit_distance) / sum(reference_length)

    Args:
        predictions: list[str] 预测文本列表
        references: list[str] 参考文本列表

    Returns:
        float: CER (0.0 = 完美)
    """
    total_dist = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        total_dist += levenshtein_distance(pred, ref)
        total_len += len(ref)
    return total_dist / max(total_len, 1)


def sequence_accuracy(predictions, references):
    """计算序列准确率（完全匹配比例）。

    Args:
        predictions: list[str]
        references: list[str]

    Returns:
        float: 准确率 (0.0 ~ 1.0)
    """
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / max(len(predictions), 1)


def category_accuracy(predictions, references, char_categories):
    """按字符类别统计准确率。

    Args:
        predictions: list[str]
        references: list[str]
        char_categories: dict[str, set], 类别名称 → 字符集合
            例如 {"digits": set("0123456789"), "common": set(...)}

    Returns:
        dict[str, dict]: {类别: {"correct": int, "total": int, "accuracy": float}}
    """
    results = {}
    for cat_name, char_set in char_categories.items():
        correct = 0
        total = 0
        for pred, ref in zip(predictions, references):
            for i, ch in enumerate(ref):
                if ch in char_set:
                    total += 1
                    if i < len(pred) and pred[i] == ch:
                        correct += 1
        accuracy = correct / max(total, 1)
        results[cat_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
    return results
