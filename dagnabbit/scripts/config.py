import torch
import math

# --- model ---
NODE_EMBEDDING_DIM = 256
TRUNK_NODE_TYPE_IN_DEGREES = 2
NUM_TRUNK_NODE_TYPES = 2
CONDENSER_NODE_TYPE_IN_DEGREE = 2
NUM_ROOT_NODES = 4
NUM_OUTPUT_NODES = 4
# Number of hidden layers in every MLP (encoders, decoders, type predictor).
MLP_DEPTH = 2
# Hidden-layer width as a multiplicative expansion factor of each MLP's input dim.
MLP_EXPANSION_FACTOR = 4.0

# --- DAG sampling ---
NUM_TRUNK_NODES = 64

# --- training ---
NUM_STEPS = 1_000_000
GRADIENT_ACCUMULATION_STEPS = 8 
LEARNING_RATE = 1e-3

# Max L2 norm of gradients across all parameters before each optimizer step.
# Set to None to disable clipping.
GRADIENT_CLIP_MAX_NORM = 1.0

LOG_EVERY = GRADIENT_ACCUMULATION_STEPS
CHECK_BEST_EVERY = 1000
# DEVICE="cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1
TENSORBOARD_LOG_DIR = "runs"

# --- loss weights ---
# Uniform scale applied to the weighted sum of all loss terms before backward.
GLOBAL_LOSS_MULTIPLIER = 1.0

# Reconstruction = cosine distance between each node's decode-side combined
# prediction and its encode-side embedding (direct embedding-reconstruction /
# teacher-forcing target).
#
# Each training step runs two decode passes over one shared encode pass:
#   * autoregressive: the genuine model; predictions compound down the DAG.
#   * teacher-forced (TF): every node decodes its true encode embedding, so the
#     autoregressive chain is severed and the decoders are scored on recovering
#     parent identity from clean inputs.
# The two are weighted independently; set the TF weights to 0.0 to disable the
# teacher-forced pass' contribution to the loss (it is still computed for
# logging, so zero them *and* skip logging if you want it fully gone).
W_CONDENSER_DECODED_CLASSIFICATION = 1.0
W_CONDENSER_RECONSTRUCTION = 0.0

W_PRIMARY_DECODED_CLASSIFICATION = 1.0
W_PRIMARY_RECONSTRUCTION = 0.0

# --- teacher-forced decode pass loss weights ---
W_TF_CONDENSER_DECODED_CLASSIFICATION = 0.0
W_TF_CONDENSER_RECONSTRUCTION = 0.0

W_TF_PRIMARY_DECODED_CLASSIFICATION = 0.0
W_TF_PRIMARY_RECONSTRUCTION = 0.0
