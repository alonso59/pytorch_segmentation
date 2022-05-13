import os
""" Dataset directories """
DATASET_DIR = 'dataset/'

TRAIN_IMAGES = DATASET_DIR + 'new_128_3C/train_images/'
TRAIN_MASKS = DATASET_DIR + 'new_128_3C/train_masks/'
VAL_IMAGES = DATASET_DIR + 'new_128_3C/val_images/'
VAL_MASKS = DATASET_DIR + 'new_128_3C/val_masks/'

""" HYPER-PARAMETERS """
LOSS_FN = 'dice loss'
EPOCHS = 800
BATCH_SIZE = 128
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.95
WEIGHT_DECAY = 0.0001
CLASS_WEIGHTS = [1, 1, 1, 1]
SCHEDULER = 'step' #step, cosine
GAMMA = 0.7
STEP_SIZE = EPOCHS * 0.1
GPUS_ID = [0]
PRETRAIN = True

""" GERNERAL SETTINGS """
IMAGE_SIZE = 128
CLASSES = 4
NUM_WORKERS = os.cpu_count()

embed_dim = 24
depths = [2, 2, 2, 2]
num_heads = [3, 6, 12, 24]
window_size = 7
dropout = 0.1