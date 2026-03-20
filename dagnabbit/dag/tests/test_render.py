from dagnabbit.dag.description import (
    make_random_graph_description,
    make_condenser_graph_description,
    graft_condenser_graph_onto_primary_graph,
)
from dagnabbit.dag.render import render_dag


def main():
    primary = make_random_graph_description(
        num_root_nodes=8,
        num_trunk_nodes=128,
        trunk_node_in_degrees=2,
        num_node_types=2,
    )

    output = render_dag(primary, output_path="dag_primary", fmt="png")
    print(f"Rendered primary graph to {output}")

    condenser = make_condenser_graph_description(primary)
    if condenser is not None:
        output = render_dag(condenser, output_path="dag_condenser", fmt="png")
        print(f"Rendered condenser graph to {output}")

        dag = graft_condenser_graph_onto_primary_graph(primary, condenser)
    else:
        dag = primary

    output = render_dag(dag, output_path="dag_render", fmt="png")
    print(f"Rendered combined DAG to {output}")


if __name__ == "__main__":
    main()
