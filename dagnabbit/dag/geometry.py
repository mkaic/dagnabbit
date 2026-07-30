"""The shape of the graphs to sample, and the batch-sampling entry point.

Both training stages consume the same thing -- a list of freshly sampled
:class:`~dagnabbit.dag.description.FixedInDegreeDAGDescription`. Stage one feeds
the batch to ``training_forward_batch``; stage two evaluates it and encodes it.
Nothing here is stage-specific.

Why the data can never be cached instead
----------------------------------------
Because unlimited fresh uncorrelated data is the premise of the whole setup: no
example is ever seen twice, so the training loss *is* the generalization loss.
Storing a corpus would trade that away. See the module docstring of
:mod:`dagnabbit.scripts.train_flow_proposer`.

Why there are no background workers
-----------------------------------
There were, and they were removed. Generation used to be 70-90% of a stage-one
step, which is what motivated hiding it in another process, and the whole
strategy reduced to one inequality: unpickling a batch has to be cheaper than
generating it. That never held comfortably -- shipping a description means
shipping its tensors, so the best measured result was 1.19-1.26x, and only after
fixing a pickle blowup that had made workers a 0.7x *regression*.

Making generation itself 2.3x faster removed the margin the workers were living
on: there is much less left to hide and the IPC cost did not shrink with it.
Synchronous generation is now both faster in practice and a great deal simpler --
no spawn, no per-worker seeding, no bounded queue, no drain-before-join, no
liveness polling, and no silent identically-seeded-worker failure mode. Further
gains belong in the generator itself, not in more processes.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.dag.generate import sample_graph_batch

if TYPE_CHECKING:
    from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder


@dataclass(frozen=True)
class GraphGeometry:
    """The shape of the graphs to sample. Plain data.

    Deliberately carries no model, task, device or tensor: it describes a
    geometry, it is not a handle on anything live.
    """

    num_root_nodes: int
    num_trunk_nodes: int
    num_output_nodes: int
    num_trunk_node_types: int
    # Tuple rather than list so the dataclass stays frozen and hashable.
    trunk_node_in_degrees: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg) -> "GraphGeometry":
        """Read the geometry stage-one trains at out of ``config.py``."""
        in_degrees = cfg.TRUNK_NODE_TYPE_IN_DEGREES
        if isinstance(in_degrees, int):
            in_degrees = [in_degrees] * cfg.NUM_TRUNK_NODE_TYPES
        return cls(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            trunk_node_in_degrees=tuple(in_degrees),
        )

    @classmethod
    def from_model(cls, model: "DagnabbitAutoEncoder") -> "GraphGeometry":
        """Read the geometry a *frozen checkpoint* was trained at.

        Stage two must use this rather than ``from_config``: the config tracks
        whichever run is current and drifts away from older checkpoints, and a
        geometry mismatch here produces graphs the loaded decoder cannot
        describe.
        """
        return cls(
            num_root_nodes=model.num_root_nodes,
            num_trunk_nodes=model.num_trunk_nodes,
            num_output_nodes=model.num_output_nodes,
            num_trunk_node_types=model.num_trunk_node_types,
            trunk_node_in_degrees=tuple(model.trunk_node_in_degrees),
        )

    def sample_batch(self, count: int) -> list[FixedInDegreeDAGDescription]:
        """``count`` freshly sampled graphs. The unit of work everywhere.

        One call into the compiled generator for the whole batch; see
        :mod:`dagnabbit.dag.generate` for why the boundary is crossed per batch
        rather than per graph.
        """
        return sample_graph_batch(
            count,
            num_root_nodes=self.num_root_nodes,
            num_trunk_nodes=self.num_trunk_nodes,
            num_output_nodes=self.num_output_nodes,
            num_trunk_node_types=self.num_trunk_node_types,
            trunk_node_in_degrees=list(self.trunk_node_in_degrees),
        )
