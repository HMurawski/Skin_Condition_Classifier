
from pathlib import Path
import random
import shutil

random.seed(42)

SRC = Path("data_pool")  # pooled images per class (no 'healthy')
DST = Path("data")       # will be overwritten safely into a new structure
SPLITS = {"train": 0.75, "val": 0.15, "test": 0.10}

def main():
    # Create split dirs
    for sp in SPLITS:
        (DST / sp).mkdir(parents=True, exist_ok=True)

    classes = [d for d in SRC.iterdir() if d.is_dir()]
    for cls_dir in sorted(classes):
        files = [p for p in cls_dir.iterdir() if p.is_file()]
        random.shuffle(files)
        n = len(files)
        if n == 0:
            print(f"[WARN] No files for class: {cls_dir.name}")
            continue

        # Compute split sizes
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])
        n_test  = n - n_train - n_val

        # Ensure each split gets at least 1 if possible (n >= 3)
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val   == 0:
                n_val   = 1
            n_test = n - n_train - n_val
            if n_test == 0:
                # borrow one from the largest bucket
                if n_train >= n_val:
                    n_train -= 1
                else:
                    n_val -= 1
                n_test = 1

        buckets = {
            "train": files[:n_train],
            "val":   files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:]
        }

        for sp, flist in buckets.items():
            out_dir = DST / sp / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in flist:
                shutil.copy2(str(f), out_dir / f.name)

        print(f"[OK] {cls_dir.name}: train={len(buckets['train'])}, val={len(buckets['val'])}, test={len(buckets['test'])}")

    print("[DONE] New stratified split created under 'data/'.")

if __name__ == "__main__":
    main()
