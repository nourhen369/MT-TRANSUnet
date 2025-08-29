import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "G:/My_Drive")

TRAIN_IMAGES = os.path.join(DATA_DIR, "train/images")
TRAIN_TE_MASKS = os.path.join(DATA_DIR, "train/GT_TE")
TRAIN_ZP_MASKS = os.path.join(DATA_DIR, "train/GT_ZP")
TRAIN_ICM_MASKS = os.path.join(DATA_DIR, "train/GT_ICM")

VAL_IMAGES = os.path.join(DATA_DIR, "valid/images")
VAL_TE_MASKS = os.path.join(DATA_DIR, "valid/GT_TE")
VAL_ZP_MASKS = os.path.join(DATA_DIR, "valid/GT_ZP")
VAL_ICM_MASKS = os.path.join(DATA_DIR, "valid/GT_ICM")

TEST_IMAGES = os.path.join(DATA_DIR, "test/images")
TEST_TE_MASKS = os.path.join(DATA_DIR, "test/GT_TE")
TEST_ZP_MASKS = os.path.join(DATA_DIR, "test/GT_ZP")
TEST_ICM_MASKS = os.path.join(DATA_DIR, "test/GT_ICM")

MASTERLIST_PATH = os.path.join(DATA_DIR, "MasterlistAug30-2017.xlsx")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "outputs")