from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.dag.render import render_dag


def main():
    primary = make_random_graph_description(
        num_root_nodes=8,
        num_trunk_nodes=128,
        num_output_nodes=4,
        trunk_node_in_degrees=2,
        num_trunk_node_types=2,
    )

    output = render_dag(primary, output_path="dag_primary", fmt="png")
    print(f"Rendered primary graph to {output}")


if __name__ == "__main__":
    main()
