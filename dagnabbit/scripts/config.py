import torch

# --- model ---
NODE_EMBEDDING_DIM = 512
TRUNK_NODE_TYPE_IN_DEGREES = 2
NUM_TRUNK_NODE_TYPES = 2
NUM_ROOT_NODES = 16
NUM_OUTPUT_NODES = 8
NUM_TRUNK_NODES = 128

# Hidden-layer width as a multiplicative expansion factor of each transformer
# feed-forward input dim.
MLP_EXPANSION_FACTOR = 4.0
# Layer counts for the dense sequence-space transformers that replace the old
# recursive decode: the Compressor squeezes the canonical node sequence into
# the output-position latent; the Decoder reconstructs the full sequence from
# that latent plus mask tokens.
COMPRESSOR_NUM_LAYERS = 4
DECODER_NUM_LAYERS = 4
# Shared residual transformer settings for the encoder and decoder.
TRANSFORMER_NUM_LAYERS = 2
# Number of expanded hidden layers inside each transformer feed-forward MLP.
TRANSFORMER_MLP_DEPTH = 1
TRANSFORMER_NUM_REGISTER_TOKENS = 2
TRANSFORMER_NUM_HEADS = 16
TRANSFORMER_DROPOUT = 0.0

# Compile the repeated encoder/decoder tensor kernels during CUDA training.
# This intentionally does not compile the whole graph-shaped training step,
# whose Python DAG traversal changes every iteration.
TORCH_COMPILE = False
TORCH_COMPILE_MODE = "reduce-overhead"
TORCH_COMPILE_DYNAMIC = True
# The training step invokes compiled encoder/decoder kernels many times before
# one backward pass. CUDA graph replay is fragile for that pattern, so keep
# Inductor's CUDA graph fast path disabled unless explicitly testing it.
TORCH_COMPILE_CUDAGRAPHS = False


# --- training ---
NUM_STEPS = 10_000_000
GRAPH_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4

OPTIMIZER_CLASS = torch.optim.Adam
OPTIMIZER_KWARGS = {
    "lr": LEARNING_RATE,
    "betas": (0.9, 0.999),
}
# Number of optimizer updates used to linearly ramp from 1/warmup to full LR.
LR_WARMUP_OPTIMIZER_STEPS = 100

# Max L2 norm of gradients across all parameters before each optimizer step.
# Set to None to disable clipping.
GRADIENT_CLIP_MAX_NORM = 4.0

LOG_EVERY = GRADIENT_ACCUMULATION_STEPS
CHECK_BEST_EVERY = 1000
# Save an immutable training snapshot after this many completed graphs. The
# interval must land on both a graph-batch and optimizer-update boundary so a
# checkpoint represents a complete training state. Set to None to disable.
CHECKPOINT_EVERY_GRAPHS = None
# DEVICE="cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1
TENSORBOARD_LOG_DIR = "runs"

# --- loss weights ---
# Uniform scale applied to the weighted sum of all loss terms before backward.
GLOBAL_LOSS_MULTIPLIER = 1.0

# Node-type classification cross-entropy over the reconstructed sequence, at
# every canonical position. Logged as loss/primary_decoded_classification so
# new runs overlay the old scheme's primary decode curve in TensorBoard.
W_TYPE_CLASSIFICATION = 1.0

# Parent-pointer cross-entropy, averaged over valid input slots: each slot's
# query must pick its true parent's canonical position from the
# strictly-earlier non-output positions. Logged as loss/parent_pointer.
W_PARENT_POINTER = 1.0

# Balance the classification cross-entropy between two node groups so they
# contribute equally to each graph's loss: (a) roots + the single output class,
# and (b) trunk classes. Weights are normalized to average 1 across each graph's
# nodes, so this is a pure reweighting that preserves the overall loss magnitude
# (it does not shrink the classification term relative to the other losses).
# Set False for plain per-node cross-entropy.
CLASS_BALANCED_CLASSIFICATION_LOSSES = True
