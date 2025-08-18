from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data")
SPLITS = ["train", "val", "test"]

def count_images(root: Path):
    """Count number of images per class for each split."""
    stats = defaultdict(Counter)
    for split in SPLITS:
        split_dir = root / split
        for cls in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            n = sum(1 for _ in cls.glob("*.*"))
            stats[split][cls.name] = n
    return stats

def main():
    stats = count_images(ROOT)
    classes = sorted(set().union(*[set(s.keys()) for s in stats.values()]))

    print("Per-split class counts:")
    for split in SPLITS:
        print(f"\n[{split.upper()}]")
        total = 0
        for c in classes:
            n = stats[split].get(c, 0)
            total += n
            print(f"{c:12s} : {n}")
        print(f"TOTAL: {total}")

    # Highlight classes missing in any split
    print("\nMissing classes per split (should be > 0 everywhere):")
    for c in classes:
        missing = [sp for sp in SPLITS if stats[sp].get(c, 0) == 0]
        if missing:
            print(f"- {c}: missing in {missing}")

if __name__ == "__main__":
    main()
