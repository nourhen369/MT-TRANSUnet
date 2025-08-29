import random
from collections import defaultdict
from typing import List, Tuple


def undersample_majority_class(
    x: List[str],
    y1: List,
    y2: List,
    y3: List,
    labels: List[float]
) -> Tuple[List[str], List, List, List, List[float]]:
    """
    Undersamples the majority class to balance binary classification dataset.
    
    Args:
        x (List[str]): List of image paths.
        y1, y2, y3 (List): Corresponding mask/label lists.
        labels (List[float]): Binary class labels (0 or 1).

    Returns:
        Tuple[List[str], List, List, List, List[float]]: Balanced dataset lists.
    """
    indices_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_class[label].append(idx)

    if 0 not in indices_by_class or 1 not in indices_by_class:
        print("[WARN] One class is missing. Skipping undersampling.")
        return x, y1, y2, y3, labels

    # Determine minority class size
    minority_class_size = min(len(indices_by_class[0]), len(indices_by_class[1]))

    # Keep all minority + random sample of majority
    undersampled_indices = indices_by_class[0] + random.sample(indices_by_class[1], minority_class_size)
    random.shuffle(undersampled_indices)

    undersampled_x = [x[i] for i in undersampled_indices]
    undersampled_y1 = [y1[i] for i in undersampled_indices]
    undersampled_y2 = [y2[i] for i in undersampled_indices]
    undersampled_y3 = [y3[i] for i in undersampled_indices]
    undersampled_labels = [labels[i] for i in undersampled_indices]

    print(f"[INFO] Undersampled dataset -> {len(undersampled_labels)} samples "
          f"(pos={sum(undersampled_labels)}, neg={len(undersampled_labels)-sum(undersampled_labels)})")
    return undersampled_x, undersampled_y1, undersampled_y2, undersampled_y3, undersampled_labels