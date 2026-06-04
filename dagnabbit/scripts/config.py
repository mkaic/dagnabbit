# --- model ---
NODE_EMBEDDING_DIM = 256
TRUNK_NODE_TYPE_IN_DEGREES = 2
NUM_TRUNK_NODE_TYPES = 1
CONDENSER_NODE_TYPE_IN_DEGREE = 2
NUM_ROOT_NODES = 4
NUM_OUTPUT_NODES = 4
# Number of hidden layers in every MLP (encoders, decoders, type predictor).
MLP_DEPTH = 2
# Hidden-layer width as a multiplicative expansion factor of each MLP's input dim.
MLP_EXPANSION_FACTOR = 2.0

# --- DAG sampling ---
NUM_TRUNK_NODES = 64

# --- training ---
NUM_STEPS = 10_000
LEARNING_RATE = 1e-3
GRADIENT_ACCUMULATION_STEPS = 64
# Max L2 norm of gradients across all parameters before each optimizer step.
# Set to None to disable clipping.
GRADIENT_CLIP_MAX_NORM = 1.0
LOG_EVERY = 64
DEVICE = "cpu"
SEED = 1
TENSORBOARD_LOG_DIR = "runs"

# --- loss weights ---
W_CONDENSER_DECODED_CLASSIFICATION = 1.0
W_CONDENSER_DECODED_SIMILARITY = 0.01

W_PRIMARY_DECODED_CLASSIFICATION = 1.0
W_PRIMARY_DECODED_SIMILARITY = 0.01
