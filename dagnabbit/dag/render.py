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
``<MASK>`` trunk positions -- sequence slots the graph did not spend a gate on
-- are collapsed into a single dashed summary box rather than drawn one by one.
They have no edges and sit on no path, so an individual box would be placed at
a rank it does not have, and at a low trunk draw ninety of them bury the
circuit. Nothing reads them, so they are never counted as dead either.

Needs the Graphviz system package for the ``dot`` binary (``brew install
graphviz``); the Python wrapper alone is not enough.
"""

from collections.abc import Sequence

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
MASK_COLOR = "#f0f0f0"
MASK_FONT_COLOR = "#999999"
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

    masked = [
        geometry.num_root_nodes + i
        for i in range(geometry.num_trunk_nodes)
        if node_types[geometry.num_root_nodes + i] == geometry.mask_type
    ]
    if masked:
        # One summary box, not one box per position. Masked positions have no
        # edges, so drawing them individually lets the layout engine strew them
        # through the picture at ranks they do not have -- they sit on no path
        # at all -- and at a low trunk draw they swamp the circuit that is
        # actually the point. They are also interchangeable, so the count and
        # the span say everything an individual box would.
        dot.node(
            "masked_block",
            label=f"{len(masked)} x <MASK>\\npositions {masked[0]}-{masked[-1]}",
            shape="box",
            style="filled,dashed",
            fillcolor=MASK_COLOR,
            fontcolor=MASK_FONT_COLOR,
            fixedsize="false",
            width="1.4",
            height="0.5",
        )

    for i in range(geometry.num_trunk_nodes):
        node = geometry.num_root_nodes + i
        if node_types[node] == geometry.mask_type:
            continue
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


def describe(
    graphs: GraphBatch,
    index: int = 0,
    gate_names: Sequence[str] = GATE_NAMES,
) -> str:
    """One line of what the picture should show, for cross-checking by eye.

    ``gate_names`` labels the type histogram; pass ``config.GATES.names`` when
    the configured set is not the library default, or the counts come out
    labelled with the wrong gates.
    """
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
    masked = int((types == geometry.mask_type).sum())
    referenced = set(parents[mask].tolist())
    # Masked positions are producers only in the indexing sense; nothing may
    # read them, so leaving them out is what keeps "dead" meaning a bug.
    dead = geometry.output_start - masked - len(referenced)
    gates = types.tolist()
    histogram = "  ".join(
        f"{name} {gates.count(gate_type)}" for gate_type, name in enumerate(gate_names)
    )
    return (
        f"ranks 0..{int(graphs.ranks[index].max())}  {histogram}  "
        f"masked {masked}  inverters {inverters}  repeat-constants {constants}  "
        f"dead {dead}"
    )
