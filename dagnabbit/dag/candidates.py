"""Differentiable candidate circuits: what Phase 1 hill-climbs.

A candidate is a set of categorical logits -- per trunk position one
distribution over (gate type or ``<MASK>``), per parent slot one distribution
over legal producers -- and :meth:`CandidatePopulation.sample` turns them into
a *real* :class:`~dagnabbit.dag.graphs.GraphBatch` plus the straight-through
tokens a frozen simulator can differentiate. Forward, the tokens are exactly
the encoding of the hard graph (one-hots select single table rows); backward,
the one-hots carry Gumbel-softmax gradients into the logits. Token
construction is *linear* in every one-hot -- an embedding lookup is a one-hot
times a table -- which is what makes the straight-through forward exact rather
than approximate.

How ``<MASK>`` stays legal under gradient flow
----------------------------------------------
The invariant is that masked positions form a trailing block that nothing
references. Types are sampled *before* parents (the same order the compiled
sampler uses), then any position that drew ``<MASK>`` is stably sorted to the
back. Stability preserves the relative order of live positions, so the static
"parents point strictly earlier" triangle survives compaction unchanged, and
the only dynamic constraint parents need is "the producer must be live" --
applied by masking those logits to ``-inf`` before sampling. The permutation
itself is discrete and carries no gradient; the type choice still learns
through its embedding mixture at whatever position the slot lands on.

One deliberate difference from the training sampler: candidates may leave a
live gate childless, which the sampler's coverage pass never does. That is
mildly off-distribution for the simulator, and it is the surrogate-refresh
fine-tuning's job to absorb it.

Slot space versus node-index space
----------------------------------
Logits live in *slot* space: producer ``j`` means root ``j`` for ``j < R`` and
trunk slot ``j - R`` otherwise, independent of where compaction puts things.
Each sample builds the slot -> node-index map once and translates hard parent
choices and position-embedding tables through it.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dagnabbit.dag.graphs import Geometry, GraphBatch
from dagnabbit.dag.model import NodeTokens


@dataclass
class CandidateSample:
    """One Gumbel draw from every candidate.

    ``graphs`` is the hard batch -- exact-evaluable, detached. ``tokens`` is
    ``[P, N, D]``: forward-equal to ``NodeTokens`` on ``graphs``, with
    gradients flowing to the population's logits.
    """

    graphs: GraphBatch
    tokens: Tensor


class CandidatePopulation(nn.Module):
    """``num_candidates`` independent circuit hypotheses over one geometry."""

    def __init__(self, num_candidates: int, geometry: Geometry):
        super().__init__()
        if len(set(geometry.trunk_node_in_degrees)) != 1:
            raise ValueError(
                "candidate sampling assumes one uniform in-degree; got "
                f"{geometry.trunk_node_in_degrees}"
            )
        self.geometry = geometry
        self.num_candidates = num_candidates
        roots = geometry.num_root_nodes
        trunk = geometry.num_trunk_nodes
        slots = geometry.maximum_indegree
        num_types = geometry.num_trunk_node_types

        # Final column is <MASK>; the rest are the gate types in id order.
        self.type_logits = nn.Parameter(
            torch.zeros(num_candidates, trunk, num_types + 1)
        )
        self.trunk_parent_logits = nn.Parameter(
            torch.zeros(num_candidates, trunk, slots, roots + trunk)
        )
        self.output_parent_logits = nn.Parameter(
            torch.zeros(num_candidates, geometry.num_output_nodes, roots + trunk)
        )

        # Static legality: trunk slot t may read roots and strictly earlier
        # slots. Liveness is per-sample and is intersected in at sample time.
        producer = torch.arange(roots + trunk)
        self.register_buffer(
            "trunk_static_legal",
            producer[None, :] < roots + torch.arange(trunk)[:, None],
            persistent=False,
        )
        self.register_buffer(
            "type_choice_ids",
            torch.cat(
                [torch.arange(num_types), torch.tensor([geometry.mask_type])]
            ),
            persistent=False,
        )

    @torch.no_grad()
    def initialize_from(self, graphs: GraphBatch, concentration: float) -> None:
        """Concentrate each candidate's logits on one seed graph.

        Zero everywhere except ``+concentration`` on the seed's own choices, so
        every candidate starts as a soft neighbourhood of a distinct graph
        rather than the whole population sharing one maximum-entropy prior.
        Seeds come straight from the sampler, where position and slot coincide,
        so no translation is needed.
        """
        if len(graphs) != self.num_candidates:
            raise ValueError(
                f"{len(graphs)} seed graphs for {self.num_candidates} candidates"
            )
        geometry = self.geometry
        roots = geometry.num_root_nodes

        self.type_logits.zero_()
        self.trunk_parent_logits.zero_()
        self.output_parent_logits.zero_()

        masked = graphs.trunk_is_masked
        type_choice = torch.where(
            masked,
            torch.full_like(graphs.trunk_types, geometry.num_trunk_node_types),
            graphs.trunk_types,
        )
        self.type_logits.scatter_(2, type_choice.unsqueeze(-1), concentration)

        # A producer's slot-space index equals its node index (roots first,
        # then trunk), so parent indices can be scattered as-is.
        trunk_parents = graphs.parent_indices[:, roots : geometry.output_start]
        self.trunk_parent_logits.scatter_(
            3, trunk_parents.unsqueeze(-1), concentration
        )
        # Masked seed positions hold parent index 0 as padding, not a choice;
        # leave those rows uniform rather than biased toward root 0.
        self.trunk_parent_logits[masked] = 0.0

        output_parents = graphs.parent_indices[:, geometry.output_start :, 0]
        self.output_parent_logits.scatter_(
            2, output_parents.unsqueeze(-1), concentration
        )

    def mean_entropy(self) -> dict[str, float]:
        """Average per-choice entropy in nats -- the collapse diagnostic.

        Parent entropies are taken under the *static* legality mask only;
        liveness varies per sample and is ignored here, so treat these as an
        upper bound.
        """
        def entropy(logits: Tensor) -> float:
            log_probs = F.log_softmax(logits, dim=-1)
            return float(-(log_probs.exp() * log_probs).nan_to_num().sum(-1).mean())

        trunk = self.trunk_parent_logits.masked_fill(
            ~self.trunk_static_legal[None, :, None, :], float("-inf")
        )
        return {
            "types": entropy(self.type_logits),
            "trunk_parents": entropy(trunk),
            "output_parents": entropy(self.output_parent_logits),
        }

    def sample(
        self, node_tokens: NodeTokens, temperature: float = 1.0
    ) -> CandidateSample:
        """One straight-through Gumbel draw per candidate."""
        geometry = self.geometry
        roots = geometry.num_root_nodes
        trunk = geometry.num_trunk_nodes
        slots = geometry.maximum_indegree
        num_nodes = geometry.num_nodes
        output_start = geometry.output_start
        population = self.num_candidates
        device = self.type_logits.device

        # --- types first, exactly like the sampler ---
        type_y = F.gumbel_softmax(self.type_logits, tau=temperature, hard=True)
        live = type_y.detach()[..., -1] == 0  # [P, T]

        # --- stable compaction: live slots keep their order, masks trail ---
        order = torch.argsort((~live).long(), dim=1, stable=True)  # position -> slot
        position_of_slot = torch.empty_like(order)
        position_of_slot.scatter_(
            1, order, torch.arange(trunk, device=device).expand(population, trunk)
        )
        # Producer slot j -> node index, per candidate.
        producer_index = torch.cat(
            [
                torch.arange(roots, device=device).expand(population, roots),
                roots + position_of_slot,
            ],
            dim=1,
        )  # [P, R + T]

        # --- parents second, restricted to live producers ---
        live_producer = torch.cat(
            [torch.ones(population, roots, dtype=torch.bool, device=device), live],
            dim=1,
        )
        trunk_legal = self.trunk_static_legal[None] & live_producer[:, None, :]
        parent_y = F.gumbel_softmax(
            self.trunk_parent_logits.masked_fill(
                ~trunk_legal[:, :, None, :], float("-inf")
            ),
            tau=temperature,
            hard=True,
        )  # [P, T, S, R + T]
        output_y = F.gumbel_softmax(
            self.output_parent_logits.masked_fill(
                ~live_producer[:, None, :], float("-inf")
            ),
            tau=temperature,
            hard=True,
        )  # [P, O, R + T]

        # --- everything below is in position (node index) space ---
        def to_position(x: Tensor) -> Tensor:
            index = order.reshape(
                population, trunk, *([1] * (x.ndim - 2))
            ).expand_as(x)
            return x.gather(1, index)

        type_y_pos = to_position(type_y)
        parent_y_pos = to_position(parent_y)
        live_pos = live.gather(1, order)

        # --- hard graph tensors ---
        node_types = torch.empty(
            population, num_nodes, dtype=torch.long, device=device
        )
        node_types[:, :roots] = geometry.root_type_start + torch.arange(
            roots, device=device
        )
        node_types[:, roots:output_start] = self.type_choice_ids[
            type_y_pos.detach().argmax(-1)
        ]
        node_types[:, output_start:] = geometry.output_type

        trunk_parent_slot = parent_y_pos.detach().argmax(-1)  # [P, T, S]
        trunk_parents = producer_index.gather(
            1, trunk_parent_slot.reshape(population, -1)
        ).reshape(population, trunk, slots)
        output_parents = producer_index.gather(1, output_y.detach().argmax(-1))

        parent_indices = torch.zeros(
            population, num_nodes, slots, dtype=torch.long, device=device
        )
        parent_slot_mask = torch.zeros(
            population, num_nodes, slots, dtype=torch.bool, device=device
        )
        live_slots = live_pos.unsqueeze(-1).expand(population, trunk, slots)
        parent_indices[:, roots:output_start] = torch.where(
            live_slots, trunk_parents, 0
        )
        parent_slot_mask[:, roots:output_start] = live_slots
        parent_indices[:, output_start:, 0] = output_parents
        parent_slot_mask[:, output_start:, 0] = True

        graphs = GraphBatch(
            node_types=node_types,
            parent_indices=parent_indices,
            parent_slot_mask=parent_slot_mask,
            ranks=_ranks(parent_indices, parent_slot_mask, roots),
            geometry=geometry,
        )
        tokens = self._tokens(
            node_tokens, type_y_pos, parent_y_pos, output_y, live_pos, producer_index
        )
        return CandidateSample(graphs=graphs, tokens=tokens)

    def _tokens(
        self,
        node_tokens: NodeTokens,
        type_y_pos: Tensor,
        parent_y_pos: Tensor,
        output_y: Tensor,
        live_pos: Tensor,
        producer_index: Tensor,
    ) -> Tensor:
        """``[P, N, D]`` straight-through tokens from the sampled one-hots.

        Mirrors :meth:`NodeTokens.forward` term by term, with every lookup
        replaced by (one-hot @ table). Kept adjacent to a test asserting exact
        agreement with the module on the hard tensors, because silent drift
        between the two paths is this file's biggest hazard.
        """
        geometry = self.geometry
        roots = geometry.num_root_nodes
        population = self.num_candidates
        device = type_y_pos.device

        table = node_tokens.type_embeddings.weight
        positions = node_tokens.position_embeddings.to(dtype=table.dtype)
        self_terms = node_tokens.self_projection(positions)  # [N, D]
        # Position embeddings of producer slot j, per candidate: [P, R+T, D].
        producer_positions = positions[producer_index]

        root_ids = geometry.root_type_start + torch.arange(roots, device=device)
        root_tokens = (table[root_ids] + self_terms[:roots]).expand(
            population, -1, -1
        )
        trunk_tokens = (
            type_y_pos @ table[self.type_choice_ids]
            + self_terms[roots : geometry.output_start]
        )
        output_tokens = (
            table[geometry.output_type] + self_terms[geometry.output_start :]
        ).expand(population, -1, -1)

        for slot, projection in enumerate(node_tokens.parent_projections):
            null = node_tokens.null_parent[slot]
            root_tokens = root_tokens + null
            expectation = torch.einsum(
                "ptj,pjd->ptd", parent_y_pos[:, :, slot], producer_positions
            )
            trunk_tokens = trunk_tokens + torch.where(
                live_pos.unsqueeze(-1), projection(expectation), null
            )
            if slot == 0:
                output_tokens = output_tokens + projection(
                    torch.einsum("poj,pjd->pod", output_y, producer_positions)
                )
            else:
                output_tokens = output_tokens + null

        tokens = torch.cat([root_tokens, trunk_tokens, output_tokens], dim=1)
        return node_tokens.output_norm(tokens)


def _ranks(
    parent_indices: Tensor, parent_slot_mask: Tensor, num_root_nodes: int
) -> Tensor:
    """Longest-path depth per node, one batched sweep in index order."""
    population, num_nodes, _ = parent_indices.shape
    ranks = torch.zeros(
        population, num_nodes, dtype=torch.long, device=parent_indices.device
    )
    for node in range(num_root_nodes, num_nodes):
        filled = parent_slot_mask[:, node]
        parent_ranks = ranks.gather(1, parent_indices[:, node])
        deepest = torch.where(filled, parent_ranks, -1).max(dim=1).values
        ranks[:, node] = torch.where(
            filled.any(dim=1), deepest + 1, torch.zeros_like(deepest)
        )
    return ranks
