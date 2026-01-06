D_MODEL = 64
SEQ_LEN = 128
NUM_HEADS = 8
MAX_SEQ_LEN = 2048
BASE_FREQUENCY = 10000.0
NORMALIZE = False

VISUALIZATION_FIGSIZE = (12, 8)
SAVE_VISUALIZATIONS = False
OUTPUT_DIR = "outputs"

ENCODING_TYPES = [
    'sinusoidal',
    'rotary',
    'alibi',
    'binary',
    'index'
]
