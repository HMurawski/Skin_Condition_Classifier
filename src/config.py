from pathlib import Path
import os

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR/"train"
VAL_DIR   = DATA_DIR/"val"
TEST_DIR  = DATA_DIR/"test"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0 #try changing to 2 on mac
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = os.getenv("DEVICE", "CPU")  # "auto"/"cuda"/"cpu"
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.75"))

MODEL_OUT = Path("artifacts")
MODEL_OUT.mkdir(exist_ok=True, parents=True)
BEST_WEIGHTS = MODEL_OUT/"best_resnet18.pt"
CLASS_INDEX_PATH = MODEL_OUT/"classes.txt"
