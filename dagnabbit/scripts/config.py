import torch

# --- model ---
NODE_EMBEDDING_DIM = 256
TRUNK_NODE_TYPE_IN_DEGREES = 2
NUM_TRUNK_NODE_TYPES = 2
NUM_ROOT_NODES = 16
NUM_OUTPUT_NODES = 8

# Hidden-layer width as a multiplicative expansion factor of each transformer
# feed-forward input dim.
MLP_EXPANSION_FACTOR = 4.0
# Shared residual-free transformer settings for the encoder and decoder.
TRANSFORMER_NUM_LAYERS = 2
# Number of expanded hidden layers inside each transformer feed-forward MLP.
TRANSFORMER_MLP_DEPTH = 1
TRANSFORMER_NUM_REGISTER_TOKENS = 2
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_DROPOUT = 0.0

# Compile the repeated encoder/decoder tensor kernels during CUDA training.
# This intentionally does not compile the whole graph-shaped training step,
# whose Python DAG traversal changes every iteration.
TORCH_COMPILE = torch.cuda.is_available()
TORCH_COMPILE_MODE = "reduce-overhead"
TORCH_COMPILE_DYNAMIC = True

# --- DAG sampling ---
NUM_TRUNK_NODES = 128

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

# Each training step runs two decode passes over one shared encode pass:
#   * autoregressive: the genuine model; predictions compound down the DAG.
#   * teacher-forced (TF): every node decodes its true encode embedding, so the
#     autoregressive chain is severed and the decoders are scored on recovering
#     parent identity from clean inputs.
# The two are weighted independently; set the TF weight to 0.0 to disable the
# teacher-forced pass' contribution to the loss (it is still computed for
# logging, so zero it *and* skip logging if you want it fully gone).
W_PRIMARY_DECODED_CLASSIFICATION = 1.0

# --- teacher-forced decode pass loss weights ---
W_TF_PRIMARY_DECODED_CLASSIFICATION = 1.0

# parent-reconstruction (per-edge: predicted parent vs true encode embedding)
W_PRIMARY_PARENT_RECONSTRUCTION = 1.0
W_TF_PRIMARY_PARENT_RECONSTRUCTION = 1.0
# parent-consistency (per-parent agreement among child predictions); opt-in
W_PRIMARY_PARENT_CONSISTENCY = 1.0
W_TF_PRIMARY_PARENT_CONSISTENCY = 1.0
# detach encoder_buffer when used as the reconstruction target (train decoder only)
RECONSTRUCTION_DETACH_TARGET = True
