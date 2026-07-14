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


# --- decode passes ---
# The training step runs up to three decode streams over one shared encode pass
# (autoregressive-with-aggregation, teacher-forced, single-sample). Set this to
# False to skip the autoregressive-with-aggregation stream entirely -- it is then
# neither computed, scored, nor logged, so no compute is spent on it. Use this to
# run with only the single-sample ("random child") and teacher-forced streams.
# When False, W_PRIMARY_DECODED_CLASSIFICATION / W_PRIMARY_PARENT_RECONSTRUCTION /
# W_PRIMARY_PARENT_CONSISTENCY are ignored.
COMPUTE_AGGREGATE_DECODE_PASS = False


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

# Each training step runs two decode passes over one shared encode pass:
#   * autoregressive: the genuine model; predictions compound down the DAG.
#   * teacher-forced (TF): every node decodes its true encode embedding, so the
#     autoregressive chain is severed and the decoders are scored on recovering
#     parent identity from clean inputs.
# The two are weighted independently; set the TF weight to 0.0 to disable the
# teacher-forced pass' contribution to the loss (it is still computed for
# logging, so zero it *and* skip logging if you want it fully gone).
W_PRIMARY_DECODED_CLASSIFICATION = 1.0

# Single-sample autoregressive decode pass: each node is fed exactly one
# uniformly-sampled child prediction (no aggregation), passed forward down the
# DAG, and classified. Directly trains the blind-decode regime (single compounded
# embedding) rather than only the denoised aggregate. The aggregate pass above is
# kept as the gradient highway / fast learner; this is the auxiliary that closes
# the train/blind-decode gap. Lower it if early training is unstable.
W_PRIMARY_SINGLE_SAMPLE_CLASSIFICATION = 1.0

# Balance the classification cross-entropy between two node groups so they
# contribute equally to each graph's summed loss: (a) roots + the single output
# class, and (b) trunk classes. Each node is scaled by 1 / (nodes in its group)
# within its graph. Applies to all classification streams (autoregressive,
# teacher-forced, single-sample). Set False for plain per-node cross-entropy.
CLASS_BALANCED_CLASSIFICATION_LOSSES = True

# --- teacher-forced decode pass loss weights ---
W_TF_PRIMARY_DECODED_CLASSIFICATION = 1.0

# parent-reconstruction (per-edge: predicted parent vs true encode embedding).
# Keep this disabled while both reconstruction weights are zero to avoid spending
# each training step on a loss that cannot affect gradients.
COMPUTE_RECONSTRUCTION_LOSS = True
W_PRIMARY_PARENT_RECONSTRUCTION = 0.0
W_TF_PRIMARY_PARENT_RECONSTRUCTION = 0.1
# parent-consistency (per-parent agreement among child predictions); opt-in
W_PRIMARY_PARENT_CONSISTENCY = 0.0
W_TF_PRIMARY_PARENT_CONSISTENCY = 0.0
# detach encoder_buffer when used as the reconstruction target (train decoder only)
RECONSTRUCTION_DETACH_TARGET = True
