import contextlib

import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable, Iterator
from dataclasses import dataclass
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_condenser_graph_description,
)
import torch.nn.functional as F


def _class_balance_weights(node_types: list[int]) -> list[float]:
    """Return per-node weights of ``1 / count_of_that_type``.

    Scaling each node's classification loss by this weight makes every type
    present in ``node_types`` contribute equally to the summed loss, regardless
    of how many nodes of that type appear in the graph.
    """
    counts: dict[int, int] = {}
    for t in node_types:
        counts[t] = counts.get(t, 0) + 1
    return [1.0 / counts[t] for t in node_types]


# Variance-preserving gain for GELU under standard-normal inputs: the reciprocal
# of the standard deviation of ``GELU(z)`` for ``z ~ N(0, 1)`` (Brock et al.,
# 2021, "Characterizing Signal Propagation to Close the Performance Gap in
# Unnormalized ResNets"). Multiplying GELU outputs by this restores unit
# variance, so a stack of Scaled-WS linears + GELUs is variance-preserving.
GELU_VARIANCE_PRESERVING_GAIN = 1.7015043497085571


class GammaScaledGELU(nn.Module):
    """GELU scaled by its variance-preserving gain (Brock et al., 2021).

    Computes ``GELU(x) * gamma``. The gain is applied *after* GELU on purpose:
    it is calibrated as ``1 / std(GELU(z))`` for ``z ~ N(0, 1)``, so it only
    restores unit variance when GELU is fed a unit-variance input and its
    (variance-reduced) output is rescaled afterwards. Scaling before GELU would
    break the calibration, since GELU is nonlinear.
    """

    def __init__(self, gain: float = GELU_VARIANCE_PRESERVING_GAIN):
        super().__init__()
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x) * self.gain


class StandardizedLinear(nn.Module):
    """Linear layer with Scaled Weight Standardization (Brock et al., 2021).

    On every forward pass each output neuron's fan-in weights are standardized
    to zero mean and, after fan-in scaling by ``1 / sqrt(fan_in)``, unit
    sum-of-squares, then multiplied by ``gain``::

        W_hat[i, j] = gain * (W[i, j] - mean_i) / sqrt(var_i * fan_in + eps)

    For zero-mean, unit-variance inputs this makes ``Var(output) = gain**2``
    (i.e. variance-preserving when ``gain == 1``) and forces each row's weight
    mean to exactly zero, which removes the mean-shift that a positive-mean
    activation (e.g. GELU) would otherwise accumulate across depth. The
    standardization is a differentiable reparameterization, so gradients flow
    into the underlying ``weight``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gain = gain
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Per-step memoization of the standardized weight. Populated by
        # `prime_weight_cache` (see `DagnabbitAutoEncoder.cached_standardized_weights`)
        # so the standardization is computed once per step instead of once per
        # node visit. A plain attribute, not a Parameter/buffer, so it never
        # enters the state_dict.
        self._weight_cache: Tensor | None = None

    def _standardized_weight(self) -> Tensor:
        mean = self.weight.mean(dim=1, keepdim=True)
        var = self.weight.var(dim=1, unbiased=False, keepdim=True)
        denom = torch.sqrt(var * self.in_features + self.eps)
        return self.gain * (self.weight - mean) / denom

    def prime_weight_cache(self, dtype: torch.dtype | None = None) -> None:
        """Compute the standardized weight once and memoize it for reuse across
        every call in the current forward/backward.

        This is a differentiable reparameterization just like calling
        ``_standardized_weight`` per forward: the cached tensor is part of the
        autograd graph, so gradients still flow back into ``self.weight`` through
        the standardization (Adam keeps optimizing the raw latent weight exactly
        as before). It is merely computed once instead of once per node visit.

        ``dtype`` casts the cached weight a single time (e.g. to the autocast
        compute dtype). This matters because autocast only caches its weight
        casts for *leaf* tensors; the standardized weight is non-leaf, so without
        this each ``F.linear`` would re-cast (and save for backward) a fresh
        low-precision copy on every call.
        """
        weight = self._standardized_weight()
        if dtype is not None:
            weight = weight.to(dtype)
        self._weight_cache = weight

    def clear_weight_cache(self) -> None:
        self._weight_cache = None

    def forward(self, x: Tensor) -> Tensor:
        weight = self._weight_cache
        if weight is None:
            weight = self._standardized_weight()
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(
        self,
        vector_dims: Iterable[int],
        activation: nn.Module = GammaScaledGELU(),
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                StandardizedLinear(vector_dims[i], vector_dims[i + 1])
                for i in range(len(vector_dims) - 1)
            ]
        )
        self.activation = activation

    # ``dynamic=True``: grouped (rank-batched) evaluation calls every MLP with a
    # leading batch axis ``G`` that varies per topological rank and per random
    # graph. Compiling statically would recompile (and blow past dynamo's
    # cache_size_limit) once per distinct ``G``; marking the batch dim dynamic
    # compiles a single shape-polymorphic kernel instead. Only the leading dim
    # varies -- each MLP instance has fixed layer widths -- so this is safe.
    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.activation(x)

        return x


def _mlp_dims(
    in_dim: int,
    out_dim: int,
    mlp_depth: int,
    mlp_expansion_factor: float,
) -> list[int]:
    """Build an MLP layer-size list with ``mlp_depth`` hidden layers between
    ``in_dim`` and ``out_dim``. Each hidden layer's width is
    ``round(in_dim * mlp_expansion_factor)`` (and at least 1)."""
    assert mlp_depth >= 0
    assert mlp_expansion_factor > 0
    hidden_width = max(1, round(in_dim * mlp_expansion_factor))
    return [in_dim] + [hidden_width] * mlp_depth + [out_dim]


class NodeEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        in_degree: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.encoder = MLP(
            _mlp_dims(
                node_embedding_dim * in_degree,
                node_embedding_dim,
                mlp_depth,
                mlp_expansion_factor,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten()
        x = self.encoder(x)

        return x

    def forward_batch(self, x: Tensor) -> Tensor:
        """Batched encode over a leading group axis.

        ``x`` is ``[G, in_degree, D]``; the ``in_degree`` parent embeddings of
        each of the ``G`` nodes are flattened into a single ``in_degree * D``
        vector and fed through the encoder in one MLP call, yielding ``[G, D]``.
        Numerically identical to calling :meth:`forward` on each node's
        ``[in_degree, D]`` slice (same flatten order, same weights).
        """
        x = x.reshape(x.shape[0], self.in_degree * self.node_embedding_dim)
        return self.encoder(x)


class NodeDecoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        in_degree: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.decoder = MLP(
            _mlp_dims(
                node_embedding_dim,
                node_embedding_dim * in_degree,
                mlp_depth,
                mlp_expansion_factor,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        x = x.view(self.in_degree, self.node_embedding_dim)
        x = torch.unbind(x, dim=0)

        return x

    def forward_batch(self, x: Tensor) -> Tensor:
        """Batched decode over a leading group axis.

        ``x`` is ``[G, D]``; each node embedding is decoded into its
        ``in_degree`` predicted parent embeddings in one MLP call, returned as
        ``[G, in_degree, D]`` (the parent axis is kept rather than unbound, so
        callers can scatter into a parent buffer). Numerically identical to
        calling :meth:`forward` per node.
        """
        x = self.decoder(x)
        return x.view(x.shape[0], self.in_degree, self.node_embedding_dim)


class NodeAutoEncoder(nn.Module):
    """Uniform node autoencoder interface.

    Both ``encode`` and ``decode`` receive the full ``node_type`` index from
    the graph. Subclasses are free to use or ignore it: the trunk / condenser
    autoencoder ignores it, while ``OutputNodeAutoEncoder`` uses it to look up
    the per-output-slot embedding.
    """

    def encode(self, input_node_embeddings: Tensor, node_type: int) -> Tensor:
        raise NotImplementedError

    def decode(self, node_embedding: Tensor, node_type: int) -> Tensor:
        raise NotImplementedError

    def encode_batch(self, parent_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        """Batched :meth:`encode` over a group of same-supertype nodes.

        ``parent_embeddings`` is ``[G, in_degree, D]`` and ``subtypes`` is the
        per-node ``node_type`` index ``[G]`` (used by subclasses that need it,
        e.g. output-slot embedding lookup). Returns ``[G, D]``.
        """
        raise NotImplementedError

    def decode_batch(self, node_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        """Batched :meth:`decode` over a group of same-supertype nodes.

        ``node_embeddings`` is ``[G, D]`` and ``subtypes`` is ``[G]``. Returns
        the predicted parent embeddings as ``[G, in_degree, D]``.
        """
        raise NotImplementedError


class FixedInDegreeNodeAutoEncoder(NodeAutoEncoder):
    def __init__(
        self,
        node_embedding_dim: int,
        in_degree,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()

        self.node_embedding_dim = node_embedding_dim

        self.encoder = NodeEncoder(
            node_embedding_dim, in_degree, mlp_depth, mlp_expansion_factor
        )
        self.decoder = NodeDecoder(
            node_embedding_dim, in_degree, mlp_depth, mlp_expansion_factor
        )

    def encode(
        self,
        input_node_embeddings: Tensor,
        node_type: int,
    ) -> Tensor:
        del node_type
        return self.encoder(input_node_embeddings)

    def decode(self, node_embedding: Tensor, node_type: int) -> Tensor:
        del node_type
        return self.decoder(node_embedding)

    def encode_batch(self, parent_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        del subtypes
        return self.encoder.forward_batch(parent_embeddings)

    def decode_batch(self, node_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        del subtypes
        return self.decoder.forward_batch(node_embeddings)


class OutputNodeAutoEncoder(NodeAutoEncoder):
    def __init__(
        self,
        node_embedding_dim: int,
        num_output_nodes: int,
        output_node_types_start: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.output_node_types_start = output_node_types_start
        self.encoder = NodeEncoder(
            node_embedding_dim,
            in_degree=2,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        self.decoder = NodeDecoder(
            node_embedding_dim,
            in_degree=1,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        # Frozen for the same reason as the root embeddings (see
        # DagnabbitAutoEncoder.__init__): output slots follow an adjacent
        # taxonomy to roots, so pinning them to fixed, near-orthogonal targets
        # forces the encode/decode path to preserve each output slot's identity
        # rather than letting the learnable slot embeddings collapse together.
        self.output_node_embeddings = nn.Embedding(num_output_nodes, node_embedding_dim)
        self.output_node_embeddings.weight.requires_grad_(False)

    def encode(self, input_node_embeddings: Tensor, node_type: int) -> Tensor:
        output_slot_idx = node_type - self.output_node_types_start
        # `input_node_embeddings` is [1, D] (an output node has exactly one
        # graph-parent). Slice instead of index-scalar so the slot embedding
        # keeps the same rank ([1, D]) and the concat gives [2, D].
        output_slot_embedding = self.output_node_embeddings.weight[
            output_slot_idx : output_slot_idx + 1
        ]
        x = torch.cat([input_node_embeddings, output_slot_embedding], dim=0)

        return self.encoder(x)

    def decode(self, node_embedding: Tensor, node_type: int) -> Tensor:
        del node_type
        return self.decoder(node_embedding)

    def encode_batch(self, parent_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        # `parent_embeddings` is [G, 1, D] (each output node has exactly one
        # graph-parent). Gather each node's learnable per-output-slot embedding
        # and concatenate it as a second "parent", giving [G, 2, D] to match the
        # in_degree=2 encoder (batched form of the single-node `encode`).
        output_slot_idx = subtypes - self.output_node_types_start
        output_slot_embeddings = self.output_node_embeddings.weight[output_slot_idx]
        x = torch.cat([parent_embeddings, output_slot_embeddings.unsqueeze(1)], dim=1)
        return self.encoder.forward_batch(x)

    def decode_batch(self, node_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        del subtypes
        return self.decoder.forward_batch(node_embeddings)


@dataclass
class TrainingStepLossReturnType:
    # 1-D tensors over the contributing nodes. The condenser excludes its roots
    # (they are not decoded); the primary includes every node.
    condenser_node_classification_losses: Tensor
    condenser_node_reconstruction_losses: Tensor
    primary_node_classification_losses: Tensor
    primary_node_reconstruction_losses: Tensor
    # Raw per-node logits ``[N, num_types]`` and true type labels (1-D
    # ``LongTensor``) for downstream diagnostics (e.g. per-node-type accuracy per
    # class). ``*_predicted_type_logits[i]`` corresponds to ``*_true_types[i]``.
    condenser_node_predicted_type_logits: Tensor
    condenser_node_true_types: Tensor
    primary_node_predicted_type_logits: Tensor
    primary_node_true_types: Tensor


class DagnabbitAutoEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        trunk_node_type_in_degrees: int | list[int],
        num_trunk_node_types: int,
        condenser_node_type_in_degree: int,
        num_root_nodes: int,
        num_output_nodes: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()

        if isinstance(trunk_node_type_in_degrees, int):
            trunk_node_type_in_degrees = [
                trunk_node_type_in_degrees
            ] * num_trunk_node_types
        assert len(trunk_node_type_in_degrees) == num_trunk_node_types

        self.node_embedding_dim = node_embedding_dim
        self.num_trunk_node_types = num_trunk_node_types
        self.num_root_nodes = num_root_nodes
        self.num_output_nodes = num_output_nodes
        self.trunk_node_in_degrees = trunk_node_type_in_degrees
        self.condenser_node_in_degree = condenser_node_type_in_degree
        self.num_node_types = num_trunk_node_types + num_root_nodes + num_output_nodes
        self.mlp_depth = mlp_depth
        self.mlp_expansion_factor = mlp_expansion_factor

        # ---- Trunk node autoencoders (one per node type) ----
        self.trunk_node_autoencoders = nn.ModuleList(
            [
                FixedInDegreeNodeAutoEncoder(
                    node_embedding_dim,
                    in_degree,
                    mlp_depth=mlp_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for in_degree in self.trunk_node_in_degrees
            ]
        )

        # ---- Output autoencoder (shared across all output slots) ----
        # Each output node has exactly one graph-parent; OutputNodeAutoEncoder
        # concatenates that parent embedding with a learnable per-output-slot
        # embedding (looked up from the node_type it is called with), so its
        # internal encoder has in_degree=2.
        output_node_types_start = num_trunk_node_types + num_root_nodes
        self.output_autoencoder = OutputNodeAutoEncoder(
            node_embedding_dim,
            num_output_nodes=num_output_nodes,
            output_node_types_start=output_node_types_start,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        # ---- Condenser node autoencoder (shared across all condenser nodes) ----
        self.condenser_node_autoencoder = FixedInDegreeNodeAutoEncoder(
            node_embedding_dim,
            in_degree=condenser_node_type_in_degree,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        # ---- Root node embeddings ----
        # Frozen on purpose. When these are learnable they tend to collapse
        # together (root identity is only weakly supervised through the shared
        # type head), which leaves root classification stuck at chance. Freezing
        # a random init pins the four roots to fixed, near-orthogonal targets
        # (random high-dim vectors are ~orthogonal), so the encode/decode path is
        # forced to preserve and recover each root's identity instead.
        self.root_node_embeddings = nn.Embedding(
            self.num_root_nodes, self.node_embedding_dim
        )
        self.root_node_embeddings.weight.requires_grad_(False)

        # ---- Unified node-autoencoder lookup by node type ----
        # Trunk types live at [0, num_trunk_node_types); output types live at
        # [output_node_types_start, num_node_types). All output types share the
        # same OutputNodeAutoEncoder instance, which reads the node_type passed
        # to .encode / .decode to pick the right per-slot embedding.
        self.node_autoencoders: dict[int, NodeAutoEncoder] = {}
        for node_type_idx, ae in enumerate(self.trunk_node_autoencoders):
            self.node_autoencoders[node_type_idx] = ae
        for output_slot_idx in range(num_output_nodes):
            type_idx = output_node_types_start + output_slot_idx
            self.node_autoencoders[type_idx] = self.output_autoencoder

        # ---- Auxiliary node-type prediction head ----
        self.node_type_predictor = MLP(
            _mlp_dims(
                in_dim=self.node_embedding_dim,
                out_dim=self.num_node_types,
                mlp_depth=mlp_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
        )

        self._raise_dynamo_cache_limit_for_mlp_shapes()

    def _raise_dynamo_cache_limit_for_mlp_shapes(self) -> None:
        """Size dynamo's compile cache to this model's distinct MLP shapes.

        Grouped (rank-batched) evaluation funnels every MLP through the single
        decorated ``MLP.forward``; dynamo keys its compile cache on that one
        shared code object, accumulating one specialization per distinct
        (layer-shape x grad-mode x size-1-vs-dynamic-batch) combination. The
        layer-shapes are fixed by this model's config (trunk in-degrees, output
        / condenser in-degrees, embedding dim, type-head width), and
        ``dynamic=True`` keeps the batch axis from multiplying the count, so the
        total is bounded. We compute the number of distinct MLP layer-shapes
        actually instantiated and give the cache room for each one's grad/no-grad
        and batch-warmup variants (x4) plus headroom -- otherwise this bounded
        set would thrash against dynamo's default ``cache_size_limit`` of 8.
        """
        distinct_mlp_shapes = {
            tuple((layer.in_features, layer.out_features) for layer in mlp.layers)
            for mlp in self.modules()
            if isinstance(mlp, MLP)
        }
        needed = len(distinct_mlp_shapes) * 4 + 8
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, needed
        )

    @contextlib.contextmanager
    def cached_standardized_weights(
        self, dtype: torch.dtype | None = None
    ) -> Iterator[None]:
        """Standardize every ``StandardizedLinear``'s weight once, reuse it for
        every node visit in this step, then drop the cache on exit.

        Numerically identical to recomputing the standardization on every call
        (so the optimizer still trains the raw latent weights through it), but
        for this model it collapses ~1.7k standardizations/step down to the ~21
        distinct linear layers. Wrap a single step's forward+backward in this;
        the cache is rebuilt each step because the weights change after
        ``optimizer.step()``.
        """
        linears = [m for m in self.modules() if isinstance(m, StandardizedLinear)]
        for m in linears:
            m.prime_weight_cache(dtype)
        try:
            yield
        finally:
            for m in linears:
                m.clear_weight_cache()

    @staticmethod
    def evaluate_graph(
        graph: FixedInDegreeDAGDescription,
        root_node_embeddings: Tensor,
        node_autoencoders: dict[int, NodeAutoEncoder],
        node_embedding_dim: int,
    ) -> Tensor:
        assert root_node_embeddings.shape[0] == graph.num_root_nodes

        device = root_node_embeddings.device
        embeddings_buffer = torch.empty(
            (graph.num_nodes, node_embedding_dim),
            dtype=root_node_embeddings.dtype,
            device=device,
        )
        embeddings_buffer[: graph.num_root_nodes] = root_node_embeddings

        # Walk topological ranks ascending. Rank 0 is exactly the roots (any node
        # with parents has rank >= 1), already seeded above, so skip it. Within a
        # rank every node's parents live at strictly lower ranks and are therefore
        # already written, so all groups at the rank can be evaluated in one
        # batched MLP call each.
        for rank, groups in enumerate(graph.rank_groups):
            if rank == 0:
                continue
            for group in groups:
                node_buffer_indices = group.node_buffer_indices.to(device)
                parent_gather_indices = group.parent_buffer_gather_indices.to(device)
                subtypes = group.subtypes.to(device)

                # [G, in_degree, D]: each node's ordered parent embeddings.
                parent_embeddings = embeddings_buffer[parent_gather_indices]

                # Every node in a group shares one module (trunk groups are split
                # by subtype; outputs/condenser share a single module), so the
                # subtype of any member resolves it.
                node_autoencoder = node_autoencoders[int(group.subtypes[0])]
                embeddings_buffer[node_buffer_indices] = node_autoencoder.encode_batch(
                    parent_embeddings, subtypes
                ).to(embeddings_buffer.dtype)

        return embeddings_buffer

    def encode_graph_with_condenser(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        return_buffers: bool = False,
    ) -> tuple[FixedInDegreeDAGDescription, Tensor]:
        primary_buffer = self.evaluate_graph(
            graph=primary_graph,
            root_node_embeddings=self.root_node_embeddings.weight,
            node_autoencoders=self.node_autoencoders,
            node_embedding_dim=self.node_embedding_dim,
        )

        condenser_graph = make_condenser_graph_description(primary_graph)
        leaf_embeddings = primary_buffer[primary_graph.leaf_node_indices]
        condenser_buffer = self.evaluate_graph(
            graph=condenser_graph,
            root_node_embeddings=leaf_embeddings,
            node_autoencoders={0: self.condenser_node_autoencoder},
            node_embedding_dim=self.node_embedding_dim,
        )
        # The condenser reduces to exactly one leaf by construction (see
        # `make_condenser_graph_description`); index with a scalar so the
        # graph embedding is shape [D] to match the per-node entries the
        # decode buffer accumulates elsewhere.
        assert len(condenser_graph.leaf_node_indices) == 1
        graph_embedding = condenser_buffer[condenser_graph.leaf_node_indices[0]]
        if return_buffers:
            return condenser_graph, graph_embedding, primary_buffer, condenser_buffer
        else:
            return condenser_graph, graph_embedding

    def training_forward(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        return_buffers: bool = False,
    ) -> (
        TrainingStepLossReturnType
        | tuple[
            TrainingStepLossReturnType,
            Tensor,
            Tensor,
        ]
    ):
        """Returns losses for the training step.

        When ``return_buffers`` is True, additionally returns the primary-graph
        encode buffer (per-node embeddings from the forward pass) and the primary
        decode buffer: a node-indexed ``[num_nodes, D]`` tensor of each node's
        decode-side combined prediction, for diagnostics such as signal
        propagation analysis.
        """

        # Encode
        condenser_graph, graph_embedding, primary_buffer, condenser_buffer = (
            self.encode_graph_with_condenser(
                primary_graph,
                return_buffers=True,
            )
        )

        device = primary_buffer.device

        # Guided Autoregressive Decode.
        #
        # The decode propagates predicted parent embeddings up each DAG. We hold
        # two dense buffers per graph instead of per-node prediction lists:
        #   child_sum[n]   = running sum of embeddings predicted for node n by
        #                    its (already-decoded) children, and
        #   child_count[n] = how many such predictions have landed on n.
        # A node's combined prediction is then ``child_sum / sqrt(child_count)``
        # (the same variance-preserving combine as before), and a child pushes
        # into its parents with ``index_add_`` (an order-independent sum, so the
        # result matches the old per-node list-sum up to float reduction order).

        # ---- Condenser decode ----
        condenser_child_sum = torch.zeros(
            condenser_graph.num_nodes,
            self.node_embedding_dim,
            dtype=condenser_buffer.dtype,
            device=device,
        )
        condenser_child_count = torch.zeros(
            condenser_graph.num_nodes, dtype=condenser_buffer.dtype, device=device
        )
        # Seed the condenser's single top node (last index) with the graph
        # embedding: one prediction, so its combine is the graph embedding itself.
        condenser_child_sum[-1] = graph_embedding
        condenser_child_count[-1] = 1.0

        condenser_labels = torch.as_tensor(
            list(condenser_graph.node_types), dtype=torch.long, device=device
        )
        condenser_weights = torch.as_tensor(
            _class_balance_weights(list(condenser_graph.node_types)),
            dtype=condenser_buffer.dtype,
            device=device,
        )

        (
            _condenser_combined,
            condenser_logits,
            condenser_class_losses,
            condenser_recon_losses,
        ) = self._decode_graph(
            graph=condenser_graph,
            encoder_buffer=condenser_buffer,
            child_sum=condenser_child_sum,
            child_count=condenser_child_count,
            node_autoencoders={0: self.condenser_node_autoencoder},
            labels=condenser_labels,
            class_weights=condenser_weights,
            process_roots=False,
            device=device,
        )

        # ---- Transplant condenser roots -> primary leaves ----
        # Condenser root ``i`` is the same semantic node as primary node
        # ``primary_graph.leaf_node_indices[i]`` (see `encode_graph_with_condenser`),
        # so the predictions accumulated on condenser-root entries seed the
        # corresponding primary leaves' child buffers.
        primary_child_sum = torch.zeros(
            primary_graph.num_nodes,
            self.node_embedding_dim,
            dtype=primary_buffer.dtype,
            device=device,
        )
        primary_child_count = torch.zeros(
            primary_graph.num_nodes, dtype=primary_buffer.dtype, device=device
        )
        leaf_indices = torch.as_tensor(
            primary_graph.leaf_node_indices, dtype=torch.long, device=device
        )
        condenser_root_indices = torch.arange(
            condenser_graph.num_root_nodes, device=device
        )
        primary_child_sum.index_add_(
            0, leaf_indices, condenser_child_sum[condenser_root_indices]
        )
        primary_child_count.index_add_(
            0, leaf_indices, condenser_child_count[condenser_root_indices]
        )

        # ---- Primary decode ----
        primary_labels = torch.as_tensor(
            list(primary_graph.node_types), dtype=torch.long, device=device
        )
        primary_weights = torch.as_tensor(
            _class_balance_weights(list(primary_graph.node_types)),
            dtype=primary_buffer.dtype,
            device=device,
        )

        (
            primary_combined,
            primary_logits,
            primary_class_losses,
            primary_recon_losses,
        ) = self._decode_graph(
            graph=primary_graph,
            encoder_buffer=primary_buffer,
            child_sum=primary_child_sum,
            child_count=primary_child_count,
            node_autoencoders=self.node_autoencoders,
            labels=primary_labels,
            class_weights=primary_weights,
            process_roots=True,
            device=device,
        )

        # ---- Assemble the return contract ----
        # Return the dense per-node tensors directly (no per-node relisting). The
        # condenser exposes only its trunk nodes (roots are not decoded), exactly
        # as before; the primary exposes every node. All four loss tensors and the
        # logits/true-type pairs stay node-aligned, which is all consumers require.
        num_condenser_roots = condenser_graph.num_root_nodes
        losses = TrainingStepLossReturnType(
            condenser_node_classification_losses=condenser_class_losses[
                num_condenser_roots:
            ],
            condenser_node_reconstruction_losses=condenser_recon_losses[
                num_condenser_roots:
            ],
            primary_node_classification_losses=primary_class_losses,
            primary_node_reconstruction_losses=primary_recon_losses,
            condenser_node_predicted_type_logits=condenser_logits[
                num_condenser_roots:
            ],
            condenser_node_true_types=condenser_labels[num_condenser_roots:],
            primary_node_predicted_type_logits=primary_logits,
            primary_node_true_types=primary_labels,
        )

        if return_buffers:
            # `primary_combined` is the node-indexed [num_nodes, D] decode-side
            # combined-prediction buffer, consumed directly by diagnostics.
            return losses, primary_buffer, primary_combined
        return losses

    def _decode_graph(
        self,
        graph: FixedInDegreeDAGDescription,
        encoder_buffer: Tensor,
        child_sum: Tensor,
        child_count: Tensor,
        node_autoencoders: dict[int, NodeAutoEncoder],
        labels: Tensor,
        class_weights: Tensor,
        process_roots: bool,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the batched guided-autoregressive decode over one graph.

        Walks topological ranks **descending** (every edge strictly increases
        rank, so all of a node's children -- which live at higher ranks -- have
        already pushed their predictions into ``child_sum`` / ``child_count`` by
        the time the node is processed, and no two same-rank nodes are
        parent/child). At each rank it batches:

        1. the variance-preserving combine ``child_sum / sqrt(child_count)``,
        2. classification (one ``node_type_predictor`` call) with class-balanced
           cross-entropy, and
        3. reconstruction (cosine distance to ``encoder_buffer``);

        then, per supertype group with parents, one ``decode_batch`` call whose
        predicted parent embeddings are ``index_add_``-scattered into the
        parents' ``child_sum`` (and ``child_count`` incremented).

        ``process_roots`` controls whether rank-0 (root) nodes are decoded:
        primary roots are classified/reconstructed (they have no parents, so
        nothing is scattered), whereas condenser roots are skipped entirely --
        they only hold accumulated predictions for the transplant.

        Returns dense per-node tensors ``(combined, logits, classification_loss,
        reconstruction_loss)``; entries for nodes that were not processed stay
        zero.
        """
        num_nodes = graph.num_nodes
        combined_buffer = torch.zeros(
            num_nodes, self.node_embedding_dim, dtype=child_sum.dtype, device=device
        )
        logits_buffer = torch.zeros(
            num_nodes, self.num_node_types, dtype=encoder_buffer.dtype, device=device
        )
        classification_losses = torch.zeros(
            num_nodes, dtype=encoder_buffer.dtype, device=device
        )
        reconstruction_losses = torch.zeros(
            num_nodes, dtype=encoder_buffer.dtype, device=device
        )

        for rank in reversed(range(len(graph.rank_groups))):
            if rank == 0 and not process_roots:
                continue
            groups = graph.rank_groups[rank]

            # All nodes at this rank, across supertype groups: combine, classify
            # and reconstruct uniformly (these use the shared type head and the
            # encoder buffer regardless of supertype).
            rank_node_indices = torch.cat(
                [g.node_buffer_indices for g in groups]
            ).to(device)

            counts = child_count[rank_node_indices]
            combined = child_sum[rank_node_indices] / counts.sqrt().unsqueeze(-1)
            combined_buffer[rank_node_indices] = combined

            logits = self.node_type_predictor(combined)
            logits_buffer[rank_node_indices] = logits.to(logits_buffer.dtype)
            cross_entropy = F.cross_entropy(
                logits, labels[rank_node_indices], reduction="none"
            )
            classification_losses[rank_node_indices] = (
                cross_entropy * class_weights[rank_node_indices]
            ).to(classification_losses.dtype)

            reconstruction_losses[rank_node_indices] = (
                1.0
                - F.cosine_similarity(
                    combined, encoder_buffer[rank_node_indices], dim=-1
                )
            ).to(reconstruction_losses.dtype)

            # Per group with parents: decode this rank's combined predictions
            # into predicted parent embeddings and scatter them up the DAG.
            for group in groups:
                in_degree = group.parent_buffer_gather_indices.shape[1]
                if in_degree == 0:
                    continue

                node_buffer_indices = group.node_buffer_indices.to(device)
                parent_gather_indices = group.parent_buffer_gather_indices.to(device)
                subtypes = group.subtypes.to(device)

                node_autoencoder = node_autoencoders[int(group.subtypes[0])]
                # [G, in_degree, D]: predicted embedding of each ordered parent.
                predicted_parent_embeddings = node_autoencoder.decode_batch(
                    combined_buffer[node_buffer_indices], subtypes
                )

                flat_parent_indices = parent_gather_indices.reshape(-1)
                flat_predictions = predicted_parent_embeddings.reshape(
                    -1, self.node_embedding_dim
                ).to(child_sum.dtype)
                child_sum.index_add_(0, flat_parent_indices, flat_predictions)
                child_count.index_add_(
                    0,
                    flat_parent_indices,
                    torch.ones_like(flat_parent_indices, dtype=child_count.dtype),
                )

        return (
            combined_buffer,
            logits_buffer,
            classification_losses,
            reconstruction_losses,
        )

    def inference_decode_blind_autoregressive(
        self,
        graph_embedding: Tensor,
    ):
        pass
