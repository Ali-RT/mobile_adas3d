from pathlib import Path
from typing import List
import random


def read_split_file(split_path: str | Path) -> List[str]:
    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with split_path.open("r") as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]

    return sample_ids


def write_split_file(sample_ids: List[str], split_path: str | Path) -> None:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    with split_path.open("w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")


def create_train_val_test_split(
    sample_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, List[str]]:
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}"
        )

    sample_ids = sorted(sample_ids)

    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    n = len(sample_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:n_train + n_val]
    test_ids = sample_ids[n_train + n_val:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }