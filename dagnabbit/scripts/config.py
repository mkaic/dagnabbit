import torch

from dagnabbit.optimizers import AutoMuon

# --- model ---
NODE_EMBEDDING_DIM = 256
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
# Set COMPRESSOR_NUM_LAYERS to None (or 0) to drop the Compressor entirely and
# feed the encoder's own output-node embeddings straight into the Decoder as
# the latent.
COMPRESSOR_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6
# Layer count for the recursive structural encoder's shared per-node
# transformer. The remaining block geometry (64-wide attention heads, MLP
# depth, register tokens, dropout) is fixed in dagnabbit/dag/autoencoder.py.
ENCODER_NUM_LAYERS = 2

# Compile the repeated encoder/decoder tensor kernels during CUDA training.
# This intentionally does not compile the whole graph-shaped training step,
# whose Python DAG traversal changes every iteration.
TORCH_COMPILE = True
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

# AutoMuon runs torch.optim.Muon on the transformer blocks' attention/MLP
# weight matrices and torch.optim.AdamW on everything Muon is not meant for
# (biases, LayerNorm gains, embedding tables, learned position/register/mask
# tokens). "match_rms_adamw" rescales Muon's orthogonal update to AdamW's RMS,
# which is what lets both rules share one tuned LEARNING_RATE; the alternative
# is adjust_lr_fn=None ("original"), where Muon wants its own much larger LR
# (~0.02). LR warmup scales both groups.
#
# ``adam_module_names`` names the output heads: module type alone cannot tell a
# classifier apart from a hidden layer, and the type predictor's output axis
# indexes classes, not a hidden space. The pointer projections are ordinary
# square hidden-space maps and stay on Muon.
OPTIMIZER_CLASS = AutoMuon
OPTIMIZER_KWARGS = {
    "muon_lr": LEARNING_RATE,
    "adam_lr": LEARNING_RATE,
    "adjust_lr_fn": "match_rms_adamw",
    "momentum": 0.95,
    "adam_betas": (0.9, 0.999),
    # No weight decay, matching the plain-Adam setup this replaced.
    "muon_weight_decay": 0.0,
    "adam_weight_decay": 0.0,
    "adam_module_names": ("node_type_predictor",),
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
# Trunk-type classification cross-entropy over the reconstructed sequence.
# Only trunk positions are scored: the canonical layout fixes the first
# positions as the ordered roots and the last positions as the ordered
# outputs, so their identities are known by construction. Logged as
# loss/primary_decoded_classification for TensorBoard continuity.
W_TYPE_CLASSIFICATION = 1.0

# Parent-pointer cross-entropy, averaged over valid input slots: each slot's
# query must pick its true parent's canonical position from the
# strictly-earlier non-output positions. Logged as loss/parent_pointer.
W_PARENT_POINTER = 1.0
