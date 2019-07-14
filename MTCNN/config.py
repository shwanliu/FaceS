import os

MODEL_STORE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/model"
DATASETS_STORE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/datasets"
LOG_DIR =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/log"

USE_CUDA = False
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 0.1
END_EPOCH = 20


