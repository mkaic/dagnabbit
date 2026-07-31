"""Draw a :class:`~dagnabbit.dag.graphs.GraphBatch` with Graphviz.

Left to right, roots pinned to the first rank and outputs to the last. Trunk
nodes are coloured by gate type, and two shapes carry the things worth spotting
by eye:

* a **doublecircle** is an inverter: a NAND that drew the same producer into
  both slots, since ``NAND(x, x) = NOT x``. These exist only because the sampler
  allows duplicate parents; without them the reference adder's buffers would be
  unreachable from the sampling distribution.
* a **triangle** is a gate that drew the same producer twice but is *not* a
  NAND, and so is a constant rather than an inverter -- ``XOR(x, x) = 0`` and
  ``XNOR(x, x) = 1``. Worth telling apart: with three gate types only about a
  third of duplicate-parent gates actually invert.
* a **diamond** is a node nothing reads. The coverage pass is supposed to make
  these impossible, so any diamond outside the output block is a bug.

Needs the Graphviz system package for the ``dot`` binary (``brew install
graphviz``); the Python wrapper alone is not enough.
"""

import graphviz

from dagnabbit.dag.graphs import GraphBatch
from dagnabbit.tasks.logic_gates.operators import GATE_NAMES

# Feeding a gate the same producer twice only yields NOT x for NAND. For any
# other gate in the set it collapses to a constant, so the two cases are drawn
# and counted apart.
INVERTING_GATE = 0

# One colour per trunk gate type, indexed the same way GATE_OPERATORS is.
GATE_PALETTE = ["#59a14f", "#f28e2b", "#b07aa1", "#4e79a7", "#edc948", "#9c755f"]
ROOT_COLOR = "#d3d3d3"
ROOT_FONT_COLOR = "#333333"
DEAD_COLOR = "#2ecc71"
OUTPUT_COLOR = "#e74c3c"
EDGE_COLOR = "#888888"
BG_COLOR = "#ffffff"


def render_dag(
    graphs: GraphBatch,
    output_path: str = "dag_render",
    fmt: str = "png",
    index: int = 0,
) -> str:
    """Render one graph of a batch to an image file. Returns the written path."""
    geometry = graphs.geometry
    node_types = graphs.node_types[index].tolist()
    parents = graphs.parent_indices[index]
    mask = graphs.parent_slot_mask[index]

    edges: list[tuple[int, int]] = []
    referenced: set[int] = set()
    inverters: set[int] = set()
    constants: set[int] = set()
    for node in range(geometry.num_root_nodes, geometry.num_nodes):
        slots = [
            int(parents[node, slot])
            for slot in range(parents.shape[1])
            if mask[node, slot]
        ]
        if len(slots) > 1 and len(set(slots)) == 1 and node < geometry.output_start:
            target = inverters if node_types[node] == INVERTING_GATE else constants
            target.add(node)
        for parent in dict.fromkeys(slots):
            edges.append((parent, node))
            referenced.add(parent)

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "LR",
            "bgcolor": BG_COLOR,
            "splines": "polyline",
            "dpi": "150",
            "pad": "0.5",
            "nodesep": "0.25",
            "ranksep": "2.0",
            "sep": "+8",
        },
        node_attr={
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "8",
            "width": "0.3",
            "height": "0.3",
            "fixedsize": "true",
        },
        edge_attr={
            "color": EDGE_COLOR,
            "arrowsize": "0.4",
            "tailport": "e",
            "headport": "w",
        },
    )

    with dot.subgraph() as roots:
        roots.attr(rank="min")
        for i in range(geometry.num_root_nodes):
            roots.node(
                str(i),
                label=f"R{i}",
                shape="square",
                fillcolor=ROOT_COLOR,
                fontcolor=ROOT_FONT_COLOR,
            )

    for i in range(geometry.num_trunk_nodes):
        node = geometry.num_root_nodes + i
        if node not in referenced:
            # The coverage pass forbids this; drawn loudly rather than hidden.
            dot.node(
                str(node),
                label=str(node),
                shape="diamond",
                fillcolor=DEAD_COLOR,
                fontcolor="#ffffff",
            )
            continue
        if node in inverters:
            shape = "doublecircle"
        elif node in constants:
            shape = "triangle"
        else:
            shape = "square"
        dot.node(
            str(node),
            label=str(node),
            shape=shape,
            fillcolor=GATE_PALETTE[node_types[node] % len(GATE_PALETTE)],
            fontcolor="white",
        )

    with dot.subgraph() as outputs:
        outputs.attr(rank="max")
        for i in range(geometry.num_output_nodes):
            dot.node(
                str(geometry.output_start + i),
                label=f"O{i}",
                shape="doublecircle",
                fillcolor=OUTPUT_COLOR,
                fontcolor="#ffffff",
            )

    for source, target in edges:
        dot.edge(str(source), str(target))

    return dot.render(output_path, format=fmt, cleanup=True)


def describe(graphs: GraphBatch, index: int = 0) -> str:
    """One line of what the picture should show, for cross-checking by eye."""
    geometry = graphs.geometry
    parents = graphs.parent_indices[index]
    mask = graphs.parent_slot_mask[index]
    trunk = slice(geometry.num_root_nodes, geometry.output_start)
    types = graphs.node_types[index][trunk]
    repeated = (
        (parents[trunk, 0] == parents[trunk, 1]) & mask[trunk, 0] & mask[trunk, 1]
    )
    inverters = int((repeated & (types == INVERTING_GATE)).sum())
    constants = int((repeated & (types != INVERTING_GATE)).sum())
    referenced = set(parents[mask].tolist())
    dead = geometry.output_start - len(referenced)
    gates = types.tolist()
    histogram = "  ".join(
        f"{name} {gates.count(gate_type)}" for gate_type, name in enumerate(GATE_NAMES)
    )
    return (
        f"ranks 0..{int(graphs.ranks[index].max())}  {histogram}  "
        f"inverters {inverters}  repeat-constants {constants}  dead {dead}"
    )
