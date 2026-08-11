from pathlib import Path

# -------------------------
# Dataset
# -------------------------

DATASET_DIR = Path(__file__).parent / "datasets"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# -------------------------
# Image
# -------------------------

IMAGE_SIZE = 224

# -------------------------
# Training
# -------------------------

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# -------------------------
# Model
# -------------------------

MODEL_NAME = "mobilenet_v3_small"
PRETRAINED = True

# -------------------------
# Save Directory
# -------------------------

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

BEST_MODEL = WEIGHTS_DIR / "best_model.pth"