import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable
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


class MLP(nn.Module):
    """Pre-norm MLP: ``LayerNorm -> Linear`` per layer, GELU between layers.

    Replaces the previous Scaled-Weight-Standardization scheme
    (``StandardizedLinear`` + variance-preserving ``GammaScaledGELU``). A
    LayerNorm on the input of every linear keeps activation scale and mean
    controlled as these MLPs are composed recursively across DAG depth, so the
    network stays well-conditioned without normalizing the weights. The
    normalization now lives in the forward activations rather than in a weight
    reparameterization, so there is nothing to memoize per step.
    """

    def __init__(
        self,
        vector_dims: Iterable[int],
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(vector_dims[i], vector_dims[i + 1])
                for i in range(len(vector_dims) - 1)
            ]
        )
        # Pre-norm: normalize the input fed to each linear. This does the
        # signal-propagation work that weight standardization used to do
        # (re-centering removes GELU mean-shift; re-scaling keeps variance ~1).
        self.norms = nn.ModuleList(
            [nn.LayerNorm(vector_dims[i]) for i in range(len(vector_dims) - 1)]
        )
        self.activation = activation if activation is not None else nn.GELU()

    # ``dynamic=True``: grouped (rank-batched) evaluation calls every MLP with a
    # leading batch axis ``G`` that varies per topological rank and per random
    # graph. Compiling statically would recompile (and blow past dynamo's
    # cache_size_limit) once per distinct ``G``; marking the batch dim dynamic
    # compiles a single shape-polymorphic kernel instead. Only the leading dim
    # varies -- each MLP instance has fixed layer widths -- so this is safe.
    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.norms[i](x)
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
        self.output_node_embeddings = nn.Embedding(num_output_nodes, node_embedding_dim)

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
    # Teacher-forced counterparts of the four loss tensors and the two logit
    # tensors above (same node alignment, same true labels). These come from a
    # second decode pass that feeds each node its true encode embedding instead
    # of its own prediction; see ``DagnabbitAutoEncoder._decode_pipeline``.
    teacher_forced_condenser_node_classification_losses: Tensor
    teacher_forced_condenser_node_reconstruction_losses: Tensor
    teacher_forced_primary_node_classification_losses: Tensor
    teacher_forced_primary_node_reconstruction_losses: Tensor
    teacher_forced_condenser_node_predicted_type_logits: Tensor
    teacher_forced_primary_node_predicted_type_logits: Tensor


@dataclass
class _DecodePipelineResult:
    """Dense per-node outputs of a single decode pass (condenser + primary).

    ``*_class_losses`` / ``*_recon_losses`` / ``*_logits`` are node-indexed; the
    condenser tensors still include their (skipped) root slots at the front, so
    callers slice ``[num_condenser_roots:]`` exactly as before.
    """

    condenser_logits: Tensor
    condenser_class_losses: Tensor
    condenser_recon_losses: Tensor
    primary_combined: Tensor
    primary_logits: Tensor
    primary_class_losses: Tensor
    primary_recon_losses: Tensor


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
        self.root_node_embeddings = nn.Embedding(
            self.num_root_nodes, self.node_embedding_dim
        )

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

        # ---- Per-graph labels and class-balance weights ----
        # Shared by both decode passes (same graph, same node types).
        condenser_labels = torch.as_tensor(
            list(condenser_graph.node_types), dtype=torch.long, device=device
        )
        condenser_weights = torch.as_tensor(
            _class_balance_weights(list(condenser_graph.node_types)),
            dtype=condenser_buffer.dtype,
            device=device,
        )
        primary_labels = torch.as_tensor(
            list(primary_graph.node_types), dtype=torch.long, device=device
        )
        primary_weights = torch.as_tensor(
            _class_balance_weights(list(primary_graph.node_types)),
            dtype=primary_buffer.dtype,
            device=device,
        )

        # Two decode passes over the *same* (shared) encode pass. The
        # autoregressive pass is the genuine model -- predictions compound down
        # each DAG. The teacher-forced pass instead feeds every node its true
        # encode embedding, severing the autoregressive chain so the decoders are
        # asked to recover each parent's identity from a clean input. The combine
        # / classify / reconstruct targets are identical between the two; only
        # what the decoders are fed differs (see ``_decode_graph``).
        #
        # The two passes are independent of each other at every rank (each only
        # depends on its own higher-rank predictions), so they are run in
        # lockstep and batched together: at each rank their inputs are
        # concatenated along the batch axis for a single ``node_type_predictor``
        # / ``decode_batch`` (and loss) launch, then split back apart. Every op
        # involved is row-wise, so this is numerically identical to two separate
        # pipelines but roughly halves the kernel-launch count.
        autoregressive, teacher_forced = self._decode_pipeline(
            primary_graph=primary_graph,
            condenser_graph=condenser_graph,
            primary_buffer=primary_buffer,
            condenser_buffer=condenser_buffer,
            graph_embedding=graph_embedding,
            primary_labels=primary_labels,
            primary_weights=primary_weights,
            condenser_labels=condenser_labels,
            condenser_weights=condenser_weights,
            device=device,
        )

        # ---- Assemble the return contract ----
        # The condenser exposes only its trunk nodes (roots are not decoded); the
        # primary exposes every node. Autoregressive and teacher-forced loss /
        # logit tensors stay node-aligned and share the same true labels.
        num_condenser_roots = condenser_graph.num_root_nodes
        losses = TrainingStepLossReturnType(
            condenser_node_classification_losses=autoregressive.condenser_class_losses[
                num_condenser_roots:
            ],
            condenser_node_reconstruction_losses=autoregressive.condenser_recon_losses[
                num_condenser_roots:
            ],
            primary_node_classification_losses=autoregressive.primary_class_losses,
            primary_node_reconstruction_losses=autoregressive.primary_recon_losses,
            condenser_node_predicted_type_logits=autoregressive.condenser_logits[
                num_condenser_roots:
            ],
            condenser_node_true_types=condenser_labels[num_condenser_roots:],
            primary_node_predicted_type_logits=autoregressive.primary_logits,
            primary_node_true_types=primary_labels,
            teacher_forced_condenser_node_classification_losses=(
                teacher_forced.condenser_class_losses[num_condenser_roots:]
            ),
            teacher_forced_condenser_node_reconstruction_losses=(
                teacher_forced.condenser_recon_losses[num_condenser_roots:]
            ),
            teacher_forced_primary_node_classification_losses=(
                teacher_forced.primary_class_losses
            ),
            teacher_forced_primary_node_reconstruction_losses=(
                teacher_forced.primary_recon_losses
            ),
            teacher_forced_condenser_node_predicted_type_logits=(
                teacher_forced.condenser_logits[num_condenser_roots:]
            ),
            teacher_forced_primary_node_predicted_type_logits=(
                teacher_forced.primary_logits
            ),
        )

        if return_buffers:
            # `autoregressive.primary_combined` is the node-indexed [num_nodes, D]
            # decode-side combined-prediction buffer from the genuine
            # (autoregressive) pass, consumed directly by diagnostics.
            return losses, primary_buffer, autoregressive.primary_combined
        return losses

    def _decode_pipeline(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        condenser_graph: FixedInDegreeDAGDescription,
        primary_buffer: Tensor,
        condenser_buffer: Tensor,
        graph_embedding: Tensor,
        primary_labels: Tensor,
        primary_weights: Tensor,
        condenser_labels: Tensor,
        condenser_weights: Tensor,
        device: torch.device,
    ) -> tuple["_DecodePipelineResult", "_DecodePipelineResult"]:
        """Run the full decode (condenser -> transplant -> primary) for both the
        autoregressive and teacher-forced passes at once.

        The decode propagates predicted parent embeddings up each DAG. We hold
        two dense buffers per graph *per pass* instead of per-node prediction
        lists:
          child_sum[n]   = running sum of embeddings predicted for node n by its
                           (already-decoded) children, and
          child_count[n] = how many such predictions have landed on n.
        A node's combined prediction is then ``child_sum / sqrt(child_count)``,
        and a child pushes into its parents with ``index_add_`` (an
        order-independent sum). The autoregressive and teacher-forced passes get
        their own fresh buffers, so they never share decode state -- but they are
        decoded in lockstep inside :meth:`_decode_graph`, which concatenates the
        two passes for each MLP/loss launch.

        Returns ``(autoregressive_result, teacher_forced_result)``.
        """

        def _zeros(graph: FixedInDegreeDAGDescription, dtype: torch.dtype):
            child_sum = torch.zeros(
                graph.num_nodes,
                self.node_embedding_dim,
                dtype=dtype,
                device=device,
            )
            child_count = torch.zeros(graph.num_nodes, dtype=dtype, device=device)
            return child_sum, child_count

        # ---- Condenser decode (both passes) ----
        ar_condenser_sum, ar_condenser_count = _zeros(
            condenser_graph, condenser_buffer.dtype
        )
        tf_condenser_sum, tf_condenser_count = _zeros(
            condenser_graph, condenser_buffer.dtype
        )
        # Seed the condenser's single top node (last index) with the graph
        # embedding: one prediction, so its combine is the graph embedding
        # itself. Both passes start from the same shared graph embedding.
        ar_condenser_sum[-1] = graph_embedding
        ar_condenser_count[-1] = 1.0
        tf_condenser_sum[-1] = graph_embedding
        tf_condenser_count[-1] = 1.0

        (
            (_, ar_condenser_logits, ar_condenser_class, ar_condenser_recon),
            (_, tf_condenser_logits, tf_condenser_class, tf_condenser_recon),
        ) = self._decode_graph(
            graph=condenser_graph,
            encoder_buffer=condenser_buffer,
            autoregressive_child_sum=ar_condenser_sum,
            autoregressive_child_count=ar_condenser_count,
            teacher_forced_child_sum=tf_condenser_sum,
            teacher_forced_child_count=tf_condenser_count,
            node_autoencoders={0: self.condenser_node_autoencoder},
            labels=condenser_labels,
            class_weights=condenser_weights,
            process_roots=False,
            device=device,
        )

        # ---- Transplant condenser roots -> primary leaves (per pass) ----
        # Condenser root ``i`` is the same semantic node as primary node
        # ``primary_graph.leaf_node_indices[i]`` (see `encode_graph_with_condenser`),
        # so the predictions accumulated on condenser-root entries seed the
        # corresponding primary leaves' child buffers.
        ar_primary_sum, ar_primary_count = _zeros(primary_graph, primary_buffer.dtype)
        tf_primary_sum, tf_primary_count = _zeros(primary_graph, primary_buffer.dtype)
        leaf_indices = torch.as_tensor(
            primary_graph.leaf_node_indices, dtype=torch.long, device=device
        )
        condenser_root_indices = torch.arange(
            condenser_graph.num_root_nodes, device=device
        )
        ar_primary_sum.index_add_(
            0, leaf_indices, ar_condenser_sum[condenser_root_indices]
        )
        ar_primary_count.index_add_(
            0, leaf_indices, ar_condenser_count[condenser_root_indices]
        )
        tf_primary_sum.index_add_(
            0, leaf_indices, tf_condenser_sum[condenser_root_indices]
        )
        tf_primary_count.index_add_(
            0, leaf_indices, tf_condenser_count[condenser_root_indices]
        )

        # ---- Primary decode (both passes) ----
        (
            (ar_primary_combined, ar_primary_logits, ar_primary_class, ar_primary_recon),
            (tf_primary_combined, tf_primary_logits, tf_primary_class, tf_primary_recon),
        ) = self._decode_graph(
            graph=primary_graph,
            encoder_buffer=primary_buffer,
            autoregressive_child_sum=ar_primary_sum,
            autoregressive_child_count=ar_primary_count,
            teacher_forced_child_sum=tf_primary_sum,
            teacher_forced_child_count=tf_primary_count,
            node_autoencoders=self.node_autoencoders,
            labels=primary_labels,
            class_weights=primary_weights,
            process_roots=True,
            device=device,
        )

        autoregressive = _DecodePipelineResult(
            condenser_logits=ar_condenser_logits,
            condenser_class_losses=ar_condenser_class,
            condenser_recon_losses=ar_condenser_recon,
            primary_combined=ar_primary_combined,
            primary_logits=ar_primary_logits,
            primary_class_losses=ar_primary_class,
            primary_recon_losses=ar_primary_recon,
        )
        teacher_forced = _DecodePipelineResult(
            condenser_logits=tf_condenser_logits,
            condenser_class_losses=tf_condenser_class,
            condenser_recon_losses=tf_condenser_recon,
            primary_combined=tf_primary_combined,
            primary_logits=tf_primary_logits,
            primary_class_losses=tf_primary_class,
            primary_recon_losses=tf_primary_recon,
        )
        return autoregressive, teacher_forced

    def _decode_graph(
        self,
        graph: FixedInDegreeDAGDescription,
        encoder_buffer: Tensor,
        autoregressive_child_sum: Tensor,
        autoregressive_child_count: Tensor,
        teacher_forced_child_sum: Tensor,
        teacher_forced_child_count: Tensor,
        node_autoencoders: dict[int, NodeAutoEncoder],
        labels: Tensor,
        class_weights: Tensor,
        process_roots: bool,
        device: torch.device,
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        """Run the batched guided-autoregressive decode over one graph for the
        autoregressive and teacher-forced passes **simultaneously**.

        Walks topological ranks **descending** (every edge strictly increases
        rank, so all of a node's children -- which live at higher ranks -- have
        already pushed their predictions into the ``child_sum`` / ``child_count``
        buffers by the time the node is processed, and no two same-rank nodes are
        parent/child). At each rank it batches:

        1. the variance-preserving combine ``child_sum / sqrt(child_count)``,
        2. classification (``node_type_predictor``) with class-balanced
           cross-entropy, and
        3. reconstruction (cosine distance to ``encoder_buffer``);

        then, per supertype group with parents, one ``decode_batch`` whose
        predicted parent embeddings are ``index_add_``-scattered into the
        parents' ``child_sum`` (and ``child_count`` incremented).

        The two passes differ only in *what each node's decoder is fed*. The
        autoregressive pass decodes each node's own predicted ``combined``
        embedding, so prediction error compounds down the DAG. The teacher-forced
        pass instead decodes each node's *true* encode-side embedding
        (``encoder_buffer``), so each parent prediction is produced from ground
        truth -- this severs the autoregressive chain and tests whether the
        decoders can recover a parent's identity (e.g. which root) from a clean
        input. Both passes' combine / classify / reconstruct still use their own
        *accumulated* predictions, so loss targets match between the two; only
        the decoder inputs differ.

        Because the two passes are independent of each other at every rank, their
        inputs are concatenated along the batch axis (autoregressive rows first,
        teacher-forced rows second) for a single ``node_type_predictor`` /
        ``decode_batch`` (and cross-entropy / cosine) launch, then split back
        apart. Every op is row-wise, so this is numerically identical to running
        the two passes separately while roughly halving the kernel-launch count.

        ``process_roots`` controls whether rank-0 (root) nodes are decoded:
        primary roots are classified/reconstructed (they have no parents, so
        nothing is scattered), whereas condenser roots are skipped entirely --
        they only hold accumulated predictions for the transplant.

        Returns ``(autoregressive_tensors, teacher_forced_tensors)`` where each
        is a dense per-node tuple ``(combined, logits, classification_loss,
        reconstruction_loss)``; entries for nodes that were not processed stay
        zero.
        """
        num_nodes = graph.num_nodes

        def _alloc_pass_buffers():
            combined = torch.zeros(
                num_nodes,
                self.node_embedding_dim,
                dtype=autoregressive_child_sum.dtype,
                device=device,
            )
            logits = torch.zeros(
                num_nodes,
                self.num_node_types,
                dtype=encoder_buffer.dtype,
                device=device,
            )
            class_losses = torch.zeros(
                num_nodes, dtype=encoder_buffer.dtype, device=device
            )
            recon_losses = torch.zeros(
                num_nodes, dtype=encoder_buffer.dtype, device=device
            )
            return combined, logits, class_losses, recon_losses

        ar_combined, ar_logits, ar_class, ar_recon = _alloc_pass_buffers()
        tf_combined, tf_logits, tf_class, tf_recon = _alloc_pass_buffers()

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
            num_rank_nodes = rank_node_indices.shape[0]

            # Per-pass combine (each pass accumulates into its own buffers).
            ar_counts = autoregressive_child_count[rank_node_indices]
            tf_counts = teacher_forced_child_count[rank_node_indices]
            ar_combined_rank = (
                autoregressive_child_sum[rank_node_indices]
                / ar_counts.sqrt().unsqueeze(-1)
            )
            tf_combined_rank = (
                teacher_forced_child_sum[rank_node_indices]
                / tf_counts.sqrt().unsqueeze(-1)
            )
            ar_combined[rank_node_indices] = ar_combined_rank
            tf_combined[rank_node_indices] = tf_combined_rank

            # Classify + reconstruct both passes in one launch each. Rows
            # ``[:num_rank_nodes]`` are autoregressive, ``[num_rank_nodes:]`` are
            # teacher-forced; the type head and cosine are row-wise, so this is
            # identical to two separate calls.
            combined_both = torch.cat([ar_combined_rank, tf_combined_rank], dim=0)
            logits_both = self.node_type_predictor(combined_both)
            ar_logits[rank_node_indices] = logits_both[:num_rank_nodes].to(
                ar_logits.dtype
            )
            tf_logits[rank_node_indices] = logits_both[num_rank_nodes:].to(
                tf_logits.dtype
            )

            rank_labels = labels[rank_node_indices]
            rank_weights = class_weights[rank_node_indices]
            cross_entropy_both = F.cross_entropy(
                logits_both,
                torch.cat([rank_labels, rank_labels]),
                reduction="none",
            ) * torch.cat([rank_weights, rank_weights])
            ar_class[rank_node_indices] = cross_entropy_both[:num_rank_nodes].to(
                ar_class.dtype
            )
            tf_class[rank_node_indices] = cross_entropy_both[num_rank_nodes:].to(
                tf_class.dtype
            )

            rank_encoder = encoder_buffer[rank_node_indices]
            recon_both = 1.0 - F.cosine_similarity(
                combined_both, torch.cat([rank_encoder, rank_encoder]), dim=-1
            )
            ar_recon[rank_node_indices] = recon_both[:num_rank_nodes].to(
                ar_recon.dtype
            )
            tf_recon[rank_node_indices] = recon_both[num_rank_nodes:].to(
                tf_recon.dtype
            )

            # Per group with parents: decode this rank's combined predictions
            # into predicted parent embeddings and scatter them up the DAG.
            for group in groups:
                in_degree = group.parent_buffer_gather_indices.shape[1]
                if in_degree == 0:
                    continue

                node_buffer_indices = group.node_buffer_indices.to(device)
                parent_gather_indices = group.parent_buffer_gather_indices.to(device)
                subtypes = group.subtypes.to(device)
                num_group_nodes = node_buffer_indices.shape[0]

                node_autoencoder = node_autoencoders[int(group.subtypes[0])]
                # Concatenate the autoregressive decoder inputs (each node's own
                # predicted ``combined``) with the teacher-forced inputs (each
                # node's true encode embedding) so both go through this group's
                # decoder in a single kernel launch, then split the predictions
                # back apart for per-pass scatter.
                decode_input_both = torch.cat(
                    [
                        ar_combined[node_buffer_indices],
                        encoder_buffer[node_buffer_indices],
                    ],
                    dim=0,
                )
                subtypes_both = torch.cat([subtypes, subtypes])
                # [2G, in_degree, D]: predicted ordered-parent embeddings.
                predicted_both = node_autoencoder.decode_batch(
                    decode_input_both, subtypes_both
                )
                ar_predicted = predicted_both[:num_group_nodes]
                tf_predicted = predicted_both[num_group_nodes:]

                flat_parent_indices = parent_gather_indices.reshape(-1)
                ones = torch.ones_like(
                    flat_parent_indices, dtype=autoregressive_child_count.dtype
                )

                autoregressive_child_sum.index_add_(
                    0,
                    flat_parent_indices,
                    ar_predicted.reshape(-1, self.node_embedding_dim).to(
                        autoregressive_child_sum.dtype
                    ),
                )
                autoregressive_child_count.index_add_(0, flat_parent_indices, ones)
                teacher_forced_child_sum.index_add_(
                    0,
                    flat_parent_indices,
                    tf_predicted.reshape(-1, self.node_embedding_dim).to(
                        teacher_forced_child_sum.dtype
                    ),
                )
                teacher_forced_child_count.index_add_(0, flat_parent_indices, ones)

        return (
            (ar_combined, ar_logits, ar_class, ar_recon),
            (tf_combined, tf_logits, tf_class, tf_recon),
        )

    def inference_decode_blind_autoregressive(
        self,
        graph_embedding: Tensor,
    ):
        pass
