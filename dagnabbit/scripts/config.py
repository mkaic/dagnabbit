"""Phase 0 configuration: training the graph -> truth-table simulator."""

import torch

from dagnabbit.dag.canonical import Geometry
from dagnabbit.dag.model import SimulatorConfig
from dagnabbit.optimizers import AutoMuon

# --- graph geometry ---
# 16 roots + 128 NAND/NOR gates + 8 outputs = a 152-token sequence, and a
# 2^16-row truth table.
GEOMETRY = Geometry(
    num_root_nodes=16,
    num_trunk_nodes=128,
    num_output_nodes=8,
    num_trunk_node_types=2,
    trunk_node_in_degrees=(2, 2),
)

# --- model ---
# num_simulator_layers is the one knob the depth analysis actually pins down.
# Under pure value propagation a layer buys one hop, and output nodes in this
# sampling distribution sit at median depth 11 (p95 15, max 23) -- so 8 layers
# would leave three quarters of outputs out of reach and 16 covers p95. If
# accuracy stratified by output rank falls off a cliff at the layer count, that
# bound is binding and this is the number to raise; if it degrades smoothly
# past it, the model found something cheaper than message passing.
MODEL = SimulatorConfig(
    embedding_dim=128,
    attention_head_dim=32,
    mlp_expansion_factor=4.0,
    num_simulator_layers=16,
    num_decoder_layers=2,
    num_patches=256,
    dropout=0.0,
)

# --- training ---
NUM_STEPS = 1_000_000
GRAPH_BATCH_SIZE = 256
# Patches scored per step, out of MODEL.num_patches. The full table is 524288
# bits per graph; at 32 patches a step sees 1/8 of it, which at batch 256 is
# ~17M logits. Raise for a lower-variance gradient, lower if VRAM is tight --
# it trades directly against GRAPH_BATCH_SIZE.
PATCHES_PER_STEP = 32

LEARNING_RATE = 1e-3
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CLIP_MAX_NORM = 1.0
LR_WARMUP_OPTIMIZER_STEPS = 200

# AutoMuon runs torch.optim.Muon on the transformer blocks' attention/MLP weight
# matrices and AdamW on everything Muon is not meant for (biases, LayerNorm
# gains, embedding tables, learned position/query tokens). "match_rms_adamw"
# rescales Muon's orthogonal update to AdamW's RMS, which is what lets both
# rules share one tuned LEARNING_RATE.
#
# ``adam_module_names`` names the output head: its output axis indexes truth
# table bits, not a hidden space, and module type alone cannot tell that apart
# from a hidden layer.
OPTIMIZER_CLASS = AutoMuon
OPTIMIZER_KWARGS = {
    "muon_lr": LEARNING_RATE,
    "adam_lr": LEARNING_RATE,
    "adjust_lr_fn": "match_rms_adamw",
    "momentum": 0.95,
    "adam_betas": (0.9, 0.999),
    "muon_weight_decay": 0.0,
    "adam_weight_decay": 0.0,
    "adam_module_names": ("decoder.head",),
}

# --- mixed precision ---
# bfloat16 only: it shares fp32's exponent range, so no loss scaling is needed
# and no GradScaler is wired up.
AMP_ENABLED = True
AMP_DTYPE = torch.bfloat16

# The model is a fixed-shape dense stack now -- no per-graph Python traversal --
# so the whole training step is compilable.
TORCH_COMPILE = True
TORCH_COMPILE_MODE = "default"

# --- logging ---
LOG_EVERY = 10
# Steps between the depth-stratified evaluation: held-out random graphs scored
# with bit accuracy bucketed by each output node's longest-path rank. This is
# the Phase 0 gate -- average accuracy can look fine while deep outputs sit at
# chance, and that gap is what says whether the simulator is simulating.
EVAL_EVERY = 500
EVAL_BATCH_SIZE = 256
# Patches per eval graph. Higher than training since there is no backward pass.
EVAL_PATCHES = 64
CHECKPOINT_EVERY = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1
LOG_DIR = "runs"
