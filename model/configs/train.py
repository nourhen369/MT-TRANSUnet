# training hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
SEG_LOSS_FN = "dice"  
CLS_LOSS_FN = "cross_entropy"
OPTIMIZER = "adam"
SCHEDULER = True
EARLY_STOPPING_PATIENCE = 10


# model-specific constants
SIZE = 256
INPUT_SHAPE = (SIZE, SIZE, 3)
N_CLASSES = 3
BACKBONE = "unet"
USE_ATTENTION = False
WEIGHT_DECAY =1e-8

# segmentation constants
SMOOTH = 1e-15