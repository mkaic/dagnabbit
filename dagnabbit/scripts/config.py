import torch

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
NUM_STEPS = 1_000_000
LEARNING_RATE = 1e-3
GRADIENT_ACCUMULATION_STEPS = 8 
# Max L2 norm of gradients across all parameters before each optimizer step.
# Set to None to disable clipping.
GRADIENT_CLIP_MAX_NORM = 1
LOG_EVERY = 8
CHECK_BEST_EVERY = 1000
# DEVICE="cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1
TENSORBOARD_LOG_DIR = "runs"

# --- loss weights ---
# Reconstruction = cosine distance between each node's decode-side combined
# prediction and its encode-side embedding (direct embedding-reconstruction /
# teacher-forcing target).
W_CONDENSER_DECODED_CLASSIFICATION = 1.0
W_CONDENSER_RECONSTRUCTION = 1.0

W_PRIMARY_DECODED_CLASSIFICATION = 1.0
W_PRIMARY_RECONSTRUCTION = 1.0
