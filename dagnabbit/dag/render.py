import graphviz

from dagnabbit.dag.description import FixedInDegreeDAGDescription

PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#86bcb6",
    "#8cd17d",
    "#b6992d",
    "#499894",
    "#d37295",
    "#a0cbe8",
]

ROOT_COLOR = "#d3d3d3"
ROOT_FONT_COLOR = "#333333"
EDGE_COLOR = "#888888"
BG_COLOR = "#ffffff"


def render_dag(
    dag: FixedInDegreeDAGDescription,
    output_path: str = "dag_render",
    fmt: str = "png",
) -> str:
    """Render a FixedInDegreeDAGDescription to an image file using Graphviz.

    Args:
        dag: The DAG description to render.
        output_path: Output file path (without extension).
        fmt: Image format (png, pdf, svg, etc.).

    Returns:
        The path to the rendered image file.
    """
    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "LR",
            "bgcolor": BG_COLOR,
            "splines": "ortho",
            "dpi": "150",
            "pad": "0.5",
            "nodesep": "0.3",
            "ranksep": "0.6",
        },
        node_attr={
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "10",
            "width": "0.4",
            "height": "0.4",
            "fixedsize": "true",
        },
        edge_attr={
            "color": EDGE_COLOR,
            "arrowsize": "0.5",
        },
    )

    with dot.subgraph() as roots:
        roots.attr(rank="min")
        for i in range(dag.num_root_nodes):
            roots.node(
                str(i),
                label=f"R{i}",
                shape="square",
                fillcolor=ROOT_COLOR,
                fontcolor=ROOT_FONT_COLOR,
            )

    for i in range(dag.num_trunk_nodes):
        global_idx = i + dag.num_root_nodes
        node_type = dag.trunk_node_types[i].item()
        color = PALETTE[node_type % len(PALETTE)]
        dot.node(
            str(global_idx),
            label=str(global_idx),
            shape="circle",
            fillcolor=color,
            fontcolor="white",
        )

    for i in range(dag.num_trunk_nodes):
        target = i + dag.num_root_nodes
        for source in dag.trunk_node_inputs_indices[i]:
            dot.edge(str(source), str(target))

    rendered_path = dot.render(output_path, format=fmt, cleanup=True)
    return rendered_path
