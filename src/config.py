from pathlib import Path
import os

# Optionally load variables from a local .env (ignored in git)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# ---- helpers ----
def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default

def _default_workers() -> int:
    # Safer default on Windows; on Linux/mac leave some headroom
    if os.name == "nt":
        return 0
    n = os.cpu_count() or 1
    return max(2, min(8, n - 2))

# ---- paths ----
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

MODEL_OUT = Path("artifacts")
MODEL_OUT.mkdir(exist_ok=True, parents=True)
BEST_WEIGHTS = MODEL_OUT / "best_resnet18.pt"
CLASS_INDEX_PATH = MODEL_OUT / "classes.txt"

# ---- hyperparams (ENV overrides) ----
IMG_SIZE      = _get_int("IMG_SIZE", 224)
BATCH_SIZE    = _get_int("BATCH_SIZE", 32)
NUM_WORKERS   = _get_int("NUM_WORKERS", _default_workers())
EPOCHS        = _get_int("EPOCHS", 10)
LR            = _get_float("LR", 1e-4)
WEIGHT_DECAY  = _get_float("WEIGHT_DECAY", 1e-4)
SEED          = _get_int("SEED", 42)

# DEVICE: "auto" (prefer GPU if available), or "cuda"/"cpu"
DEVICE = os.getenv("DEVICE", "auto").lower()

# inference threshold
PRED_THRESHOLD = _get_float("PRED_THRESHOLD", 0.75)
