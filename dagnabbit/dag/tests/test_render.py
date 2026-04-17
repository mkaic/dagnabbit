from dagnabbit.dag.description import (
    make_random_graph_description,
    make_condenser_graph_description,
    FixedInDegreeDAGDescription,
)
from dagnabbit.dag.render import render_dag


def graft_condenser_graph_onto_primary_graph(
    primary_graph: FixedInDegreeDAGDescription,
    condenser_graph: FixedInDegreeDAGDescription,
) -> FixedInDegreeDAGDescription:
    """
    Combine a primary graph and a matching condenser graph into a single DAG
    suitable for rendering. This description is NOT intended to be compatible
    with the neural autoencoder -- it just produces a structurally valid
    FixedInDegreeDAGDescription so that render_dag can visualize the combined
    computation.

    Grafted global node layout:
        [0, n_root)                                         -> primary roots
        [n_root, n_root + n_ptrunk)                         -> primary trunks
        [n_root + n_ptrunk, ... + n_poutput)                -> primary outputs, promoted to trunks
        [..., ... + n_ctrunk)                               -> condenser trunks
        [last]                                              -> a single output node pointing at the final condenser trunk

    Grafted trunk type layout:
        [0, N)     primary trunk types (unchanged numerically)
        N          "promoted primary output" trunk type (in_degree 1)
        N + 1      "condenser" trunk type (in_degree 2)
    where N == primary_graph.num_trunk_node_types. Grafted roots and the
    grafted single output get their own unique per-slot type indices
    immediately after the trunk types, per the new node-types convention.
    """
    assert condenser_graph.num_root_nodes == len(primary_graph.leaf_node_indices), (
        "condenser roots must line up with primary leaves"
    )
    assert condenser_graph.num_output_nodes == 0, (
        "condenser graph is expected to have no output nodes of its own"
    )

    n_root = primary_graph.num_root_nodes
    n_ptrunk = primary_graph.num_trunk_nodes
    n_poutput = primary_graph.num_output_nodes
    n_ctrunk = condenser_graph.num_trunk_nodes

    primary_trunks_start = n_root
    primary_outputs_start = n_root + n_ptrunk
    condenser_trunks_start = n_root + n_ptrunk + n_poutput
    final_output_idx = condenser_trunks_start + n_ctrunk

    num_primary_trunk_types = primary_graph.num_trunk_node_types
    promoted_output_trunk_type = num_primary_trunk_types
    condenser_trunk_type = num_primary_trunk_types + 1
    grafted_num_trunk_node_types = num_primary_trunk_types + 2

    grafted_trunk_in_degrees: list[int] = list(primary_graph.trunk_node_in_degrees) + [
        1,
        2,
    ]

    root_types_start = grafted_num_trunk_node_types
    output_types_start = grafted_num_trunk_node_types + n_root

    node_inputs_indices: list[list[int]] = []
    node_types: list[int] = []

    for i in range(n_root):
        node_inputs_indices.append([])
        node_types.append(root_types_start + i)

    for i in range(n_ptrunk):
        primary_idx = primary_trunks_start + i
        node_inputs_indices.append(list(primary_graph.node_inputs_indices[primary_idx]))
        node_types.append(primary_graph.node_types[primary_idx])

    for i in range(n_poutput):
        primary_idx = primary_outputs_start + i
        # Inputs reference primary roots/trunks, whose global indices are unchanged.
        node_inputs_indices.append(list(primary_graph.node_inputs_indices[primary_idx]))
        node_types.append(promoted_output_trunk_type)

    for i in range(n_ctrunk):
        condenser_idx = condenser_graph.num_root_nodes + i
        remapped: list[int] = []
        for src in condenser_graph.node_inputs_indices[condenser_idx]:
            if src < condenser_graph.num_root_nodes:
                remapped.append(primary_graph.leaf_node_indices[src])
            else:
                condenser_local_trunk_i = src - condenser_graph.num_root_nodes
                remapped.append(condenser_trunks_start + condenser_local_trunk_i)
        node_inputs_indices.append(remapped)
        node_types.append(condenser_trunk_type)

    # Single output node referencing the final condenser trunk.
    last_condenser_trunk_idx = final_output_idx - 1
    node_inputs_indices.append([last_condenser_trunk_idx])
    node_types.append(output_types_start + 0)

    return FixedInDegreeDAGDescription(
        num_root_nodes=n_root,
        num_trunk_nodes=n_ptrunk + n_poutput + n_ctrunk,
        num_output_nodes=1,
        num_trunk_node_types=grafted_num_trunk_node_types,
        trunk_node_in_degrees=grafted_trunk_in_degrees,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )


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

    condenser = make_condenser_graph_description(primary)
    output = render_dag(condenser, output_path="dag_condenser", fmt="png")
    print(f"Rendered condenser graph to {output}")

    dag = graft_condenser_graph_onto_primary_graph(primary, condenser)
    output = render_dag(dag, output_path="dag_render", fmt="png")
    print(f"Rendered combined DAG to {output}")


if __name__ == "__main__":
    main()
