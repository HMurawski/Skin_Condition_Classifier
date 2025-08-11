# scripts/consolidate_pool.py
from pathlib import Path
import shutil
import uuid

SRC = Path("data")
DST = Path("data_pool")
EXCLUDE = {"healthy"}  # exclude this class entirely

def unique_name(dst_dir: Path, name: str) -> Path:
    """Avoid filename collisions by appending a short uuid if needed."""
    out = dst_dir / name
    if not out.exists():
        return out
    stem = out.stem
    suf = out.suffix
    return dst_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suf}"

def main():
    DST.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_dir = SRC / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            if cls in EXCLUDE:
                continue
            out_cls = DST / cls
            out_cls.mkdir(parents=True, exist_ok=True)
            for f in cls_dir.iterdir():
                if f.is_file():
                    dst_path = unique_name(out_cls, f.name)
                    shutil.copy2(str(f), dst_path)
    print("[OK] Consolidated all splits into data_pool/ (excluding 'healthy').")

if __name__ == "__main__":
    main()
