"""Draw the 68-NAND ripple-carry adder.

The circuit lives inside a 152-node description, 60 of whose trunk nodes are
dead filler needed only to meet the fixed node count. Rendering it with the
generic :func:`~dagnabbit.dag.render.render_dag` would therefore draw mostly
noise, so this script uses the builder's own role annotations to keep only the
live gates and to group them into the eight full-adder stages they form.

Two views:

* ``cell`` -- one full-adder stage, all nine NANDs labelled with what they
  compute. This is the repeating unit.
* ``full`` -- the whole circuit, one cluster per bit position, with the carry
  chain highlighted as it ripples from the LSB to the MSB.

Usage::

    python -m dagnabbit.scripts.render_adder
    python -m dagnabbit.scripts.render_adder --mode full --format svg
"""

import argparse
from pathlib import Path

import graphviz

from dagnabbit.tasks.logic_gates.reference_circuits import (
    AdderAnnotations,
    build_nand_ripple_carry_adder,
)

ROOT_FILL = "#d9d9d9"
HELPER_FILL = "#4e79a7"
XOR_FILL = "#76b7b2"
SUM_FILL = "#59a14f"
CARRY_FILL = "#f28e2b"
OUTPUT_FILL = "#e15759"
CARRY_EDGE = "#e8770d"
BUFFER_FILL = "#adb5bd"
EDGE = "#9099a3"


def role_fill(role: str) -> str:
    if role.startswith("buffer"):
        return BUFFER_FILL
    if role.startswith("sum"):
        return SUM_FILL
    if role.startswith("carry"):
        return CARRY_FILL
    if role == "a⊕b":
        return XOR_FILL
    return HELPER_FILL


def add_legend(dot: graphviz.Digraph) -> None:
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="legend", fontsize="11", color="#cccccc", style="rounded")
        for name, fill, text in (
            ("lg_root", ROOT_FILL, "input bit"),
            ("lg_helper", HELPER_FILL, "NAND"),
            ("lg_xor", XOR_FILL, "a XOR b"),
            ("lg_sum", SUM_FILL, "sum bit"),
            ("lg_carry", CARRY_FILL, "carry out"),
            ("lg_buffer", BUFFER_FILL, "identity buffer"),
            ("lg_output", OUTPUT_FILL, "output"),
        ):
            legend.node(
                name,
                label=text,
                shape="box",
                style="filled,rounded",
                fillcolor=fill,
                fontcolor="#ffffff" if fill != ROOT_FILL else "#333333",
                fontsize="10",
                width="1.1",
                height="0.3",
            )
        # Chain them invisibly so the legend stacks rather than spreading.
        names = [
            "lg_root",
            "lg_helper",
            "lg_xor",
            "lg_sum",
            "lg_carry",
            "lg_buffer",
            "lg_output",
        ]
        for first, second in zip(names, names[1:]):
            legend.edge(first, second, style="invis")


def render_cell(
    graph,
    annotations: AdderAnnotations,
    bit: int,
    output_path: Path,
    fmt: str,
) -> str:
    """One full-adder stage with every gate labelled by what it computes."""
    nodes = sorted(
        node
        for node, b in annotations.bit_of_node.items()
        if b == bit and node in annotations.core_nodes
    )
    node_set = set(nodes)

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "white",
            "splines": "spline",
            "dpi": "150",
            "pad": "0.4",
            "nodesep": "0.35",
            "ranksep": "1.1",
            "label": (
                f"\nFull-adder stage for bit {bit} "
                f"({len(nodes)} NAND gates)\\n"
                "sum = a XOR b XOR cin,   cout = (a AND b) OR ((a XOR b) AND cin)"
            ),
            "labelloc": "t",
            "fontsize": "15",
            "fontname": "Helvetica",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "Helvetica",
            "fontsize": "10",
            "fontcolor": "white",
            "margin": "0.12,0.06",
        },
        edge_attr={"color": EDGE, "arrowsize": "0.6", "penwidth": "1.2"},
    )

    for label, key in (("a", ("a", bit)), ("b", ("b", bit))):
        root = annotations.root_of_input[key]
        dot.node(
            f"n{root}",
            label=f"{label}{bit}\\n(root {root})",
            fillcolor=ROOT_FILL,
            fontcolor="#333333",
            shape="box",
        )

    if bit > 0:
        carry_in = annotations.carry_node_of_bit[bit - 1]
        dot.node(
            f"n{carry_in}",
            label=f"carry in\\n(from bit {bit - 1})",
            fillcolor=CARRY_FILL,
            shape="box",
        )

    for node in nodes:
        role = annotations.role_of_node[node]
        dot.node(f"n{node}", label=f"{role}\\ngate {node}", fillcolor=role_fill(role))

    output_node = annotations.output_of_bit[bit]
    dot.node(
        f"n{output_node}",
        label=f"sum bit {bit}\\n(output)",
        fillcolor=OUTPUT_FILL,
        shape="box",
    )

    drawn = (
        node_set
        | {annotations.root_of_input[("a", bit)]}
        | {annotations.root_of_input[("b", bit)]}
    )
    if bit > 0:
        drawn.add(annotations.carry_node_of_bit[bit - 1])

    for target in nodes + [output_node]:
        for source in graph.node_inputs_indices[target]:
            if source in drawn:
                is_carry = bit > 0 and source == annotations.carry_node_of_bit[bit - 1]
                dot.edge(
                    f"n{source}",
                    f"n{target}",
                    color=CARRY_EDGE if is_carry else EDGE,
                    penwidth="2.0" if is_carry else "1.2",
                )

    return dot.render(str(output_path), format=fmt, cleanup=True)


def render_full(
    graph,
    annotations: AdderAnnotations,
    output_path: Path,
    fmt: str,
) -> str:
    """The whole circuit, clustered per bit, carry chain highlighted."""
    width = len(annotations.sum_node_of_bit)

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
            "splines": "spline",
            "dpi": "150",
            "pad": "0.4",
            "nodesep": "0.22",
            "ranksep": "0.6",
            "compound": "true",
            "label": (
                f"\n8-bit ripple-carry adder: {annotations.core_gates} logic gates "
                f"+ {annotations.buffer_gates} identity buffers = "
                f"{annotations.gates_used} (no dead nodes)\\n"
                "orange = wires crossing between stages, the carry ripple; "
                "grey nodes are transparent buffers padding to the trunk budget"
            ),
            "labelloc": "t",
            "fontsize": "20",
            "fontname": "Helvetica",
        },
        node_attr={
            "shape": "circle",
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": "white",
            "width": "0.34",
            "height": "0.34",
            "fixedsize": "true",
        },
        edge_attr={"color": EDGE, "arrowsize": "0.45", "penwidth": "0.9"},
    )

    for bit in range(width):
        stage = sorted(node for node, b in annotations.bit_of_node.items() if b == bit)
        core = sum(node in annotations.core_nodes for node in stage)
        with dot.subgraph(name=f"cluster_bit{bit}") as cell:
            cell.attr(
                label=(
                    f"  bit {bit}  (weight {2**bit})  "
                    f"{core} logic + {len(stage) - core} buffer  "
                ),
                fontsize="13",
                fontname="Helvetica",
                color="#c9d1d9",
                style="rounded",
                bgcolor="#fafbfc",
            )
            for label, key in (("a", ("a", bit)), ("b", ("b", bit))):
                root = annotations.root_of_input[key]
                cell.node(
                    f"n{root}",
                    label=f"{label}{bit}",
                    fillcolor=ROOT_FILL,
                    fontcolor="#333333",
                    shape="square",
                )
            for node in stage:
                role = annotations.role_of_node[node]
                cell.node(f"n{node}", label=str(node), fillcolor=role_fill(role))
            output_node = annotations.output_of_bit[bit]
            cell.node(
                f"n{output_node}",
                label=f"s{bit}",
                fillcolor=OUTPUT_FILL,
                shape="doublecircle",
            )

    live = set(annotations.bit_of_node)
    roots = set(annotations.root_of_input.values())
    outputs = set(annotations.output_of_bit.values())
    drawn = live | roots | outputs

    for target in sorted(drawn):
        for source in graph.node_inputs_indices[target]:
            if source not in drawn:
                continue
            # A carry edge is one leaving a carry gate for the next stage.
            crosses_stage = (
                source in annotations.bit_of_node
                and target in annotations.bit_of_node
                and annotations.bit_of_node[source] != annotations.bit_of_node[target]
            )
            dot.edge(
                f"n{source}",
                f"n{target}",
                color=CARRY_EDGE if crosses_stage else EDGE,
                penwidth="2.6" if crosses_stage else "0.9",
                arrowsize="0.7" if crosses_stage else "0.45",
            )

    add_legend(dot)
    return dot.render(str(output_path), format=fmt, cleanup=True)


SURVIVED_EDGE = "#2f9e44"
BROKEN_EDGE = "#e03131"


def predicted_parent_positions(model, graph):
    """Encode -> decode a graph and read back the decoder's parent pointers.

    Returns ``[num_nodes, max_indegree]`` of canonical *positions*, which is the
    space the decoder predicts in, alongside the graph's own canonical truth.
    """
    import torch

    with torch.no_grad():
        latent = model.encode_to_latent([graph]).float()
        reconstructed = model.decode_latent(latent)
        predicted = model.parent_pointer_logits(reconstructed).argmax(dim=-1)[0]
    return predicted


def live_cone(graph, seeds: list[int]) -> set[int]:
    """Every node that can reach one of ``seeds`` (its backward reachable set)."""
    reached: set[int] = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in reached:
            continue
        reached.add(node)
        stack.extend(graph.node_inputs_indices[node])
    return reached


def render_roundtrip(
    graph,
    annotations: AdderAnnotations,
    model,
    output_path: Path,
    fmt: str,
) -> str:
    """The adder's own layout, with each wire coloured by whether it survived.

    The decoder predicts in canonical *position* space, so each original edge
    is checked by mapping both endpoints through ``canonical_positions`` and
    asking whether the pointer head picked the same source for that slot.
    """
    predicted = predicted_parent_positions(model, graph)
    positions = graph.canonical_positions
    width = len(annotations.sum_node_of_bit)

    survived_count = 0
    total_count = 0

    def edge_survived(target: int, slot: int, source: int) -> bool:
        return int(predicted[positions[target]][slot]) == positions[source]

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
            "splines": "spline",
            "dpi": "150",
            "pad": "0.4",
            "nodesep": "0.22",
            "ranksep": "0.6",
            "labelloc": "t",
            "fontsize": "20",
            "fontname": "Helvetica",
        },
        node_attr={
            "shape": "circle",
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": "white",
            "width": "0.34",
            "height": "0.34",
            "fixedsize": "true",
        },
        edge_attr={"arrowsize": "0.45", "penwidth": "0.9"},
    )

    for bit in range(width):
        stage = sorted(node for node, b in annotations.bit_of_node.items() if b == bit)
        with dot.subgraph(name=f"cluster_bit{bit}") as cell:
            cell.attr(
                label=f"  bit {bit}  ",
                fontsize="13",
                fontname="Helvetica",
                color="#c9d1d9",
                style="rounded",
                bgcolor="#fafbfc",
            )
            for label, key in (("a", ("a", bit)), ("b", ("b", bit))):
                root = annotations.root_of_input[key]
                cell.node(
                    f"n{root}",
                    label=f"{label}{bit}",
                    fillcolor=ROOT_FILL,
                    fontcolor="#333333",
                    shape="square",
                )
            for node in stage:
                cell.node(
                    f"n{node}",
                    label=str(node),
                    fillcolor=role_fill(annotations.role_of_node[node]),
                )
            output_node = annotations.output_of_bit[bit]
            cell.node(
                f"n{output_node}",
                label=f"s{bit}",
                fillcolor=OUTPUT_FILL,
                shape="doublecircle",
            )

    drawn = (
        set(annotations.bit_of_node)
        | set(annotations.root_of_input.values())
        | set(annotations.output_of_bit.values())
    )
    for target in sorted(drawn):
        for slot, source in enumerate(graph.node_inputs_indices[target]):
            if source not in drawn:
                continue
            survived = edge_survived(target, slot, source)
            total_count += 1
            survived_count += survived
            dot.edge(
                f"n{source}",
                f"n{target}",
                color=SURVIVED_EDGE if survived else BROKEN_EDGE,
                penwidth="1.0" if survived else "2.2",
                style="solid" if survived else "dashed",
            )

    dot.attr(
        label=(
            "\nThe adder after one encode -> decode round trip\\n"
            f"green = wire reproduced, red dashed = wire changed   "
            f"({survived_count}/{total_count} survived, "
            f"{survived_count / total_count:.1%})\\n"
            "fitness falls from 1.000 to 0.503 -- chance"
        )
    )
    return dot.render(str(output_path), format=fmt, cleanup=True)


def render_decoded(
    graph,
    model,
    output_path: Path,
    fmt: str,
) -> str:
    """What the decoder actually built: the live cone of the decoded graph."""
    with __import__("torch").no_grad():
        latent = model.encode_to_latent([graph]).float()
        decoded = model.generate(latent)[0]

    outputs = list(
        range(decoded.num_root_nodes + decoded.num_trunk_nodes, decoded.num_nodes)
    )
    cone = live_cone(decoded, outputs)
    gates = sorted(
        node
        for node in cone
        if decoded.num_root_nodes
        <= node
        < decoded.num_root_nodes + decoded.num_trunk_nodes
    )

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
            "splines": "spline",
            "dpi": "150",
            "pad": "0.4",
            "nodesep": "0.2",
            "ranksep": "0.55",
            "labelloc": "t",
            "fontsize": "18",
            "fontname": "Helvetica",
            "label": (
                "\nWhat the decoder actually built\\n"
                f"{len(gates)} live gates feeding the 8 outputs "
                f"(the adder needs 68, arranged in 8 repeating stages)"
            ),
        },
        node_attr={
            "shape": "circle",
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "8",
            "fontcolor": "white",
            "width": "0.3",
            "height": "0.3",
            "fixedsize": "true",
        },
        edge_attr={"color": EDGE, "arrowsize": "0.4", "penwidth": "0.8"},
    )

    for node in sorted(cone):
        if node < decoded.num_root_nodes:
            dot.node(
                f"d{node}",
                label=f"r{node}",
                fillcolor=ROOT_FILL,
                fontcolor="#333333",
                shape="square",
            )
        elif node in outputs:
            slot = node - (decoded.num_root_nodes + decoded.num_trunk_nodes)
            dot.node(
                f"d{node}",
                label=f"s{7 - slot}",
                fillcolor=OUTPUT_FILL,
                shape="doublecircle",
            )
        else:
            dot.node(f"d{node}", label=str(node), fillcolor=HELPER_FILL)

    for target in sorted(cone):
        for source in decoded.node_inputs_indices[target]:
            if source in cone:
                dot.edge(f"d{source}", f"d{target}")

    return dot.render(str(output_path), format=fmt, cleanup=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["cell", "full", "roundtrip", "both", "all"],
        default="both",
        help=(
            "'roundtrip' needs --checkpoint and draws the circuit after an "
            "encode/decode pass, plus what the decoder actually built."
        ),
    )
    parser.add_argument(
        "--bit", type=int, default=1, help="Which stage --mode cell draws."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Autoencoder checkpoint or run directory, for --mode roundtrip.",
    )
    parser.add_argument("--format", default="png")
    parser.add_argument(
        "--out-dir", default=".", help="Directory for the rendered files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph, annotations = build_nand_ripple_carry_adder()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("cell", "both", "all"):
        path = render_cell(
            graph, annotations, args.bit, out_dir / "adder_cell", args.format
        )
        print(f"wrote {path}")
    if args.mode in ("full", "both", "all"):
        path = render_full(graph, annotations, out_dir / "adder_full", args.format)
        print(f"wrote {path}")
    if args.mode in ("roundtrip", "all"):
        if args.checkpoint is None:
            raise SystemExit("--mode roundtrip needs --checkpoint")
        import torch

        from dagnabbit.dag.checkpoint import load_model

        model, _ = load_model(args.checkpoint, torch.device("cpu"))
        path = render_roundtrip(
            graph, annotations, model, out_dir / "adder_roundtrip", args.format
        )
        print(f"wrote {path}")
        path = render_decoded(graph, model, out_dir / "adder_decoded", args.format)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
