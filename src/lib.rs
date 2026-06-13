use std::collections::VecDeque;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Root,
    Trunk,
    Leaf,
}

const DEFAULT_MCMC_PASSES_PER_SPLIT_ROUND: usize = 8;

struct EdgeSplitGraph {
    parents: Vec<Vec<usize>>,
    children: Vec<Vec<usize>>,
    roles: Vec<Role>,
    trunk_types: Vec<Option<usize>>,
}

fn randrange(rng: &mut StdRng, stop: usize) -> usize {
    debug_assert!(stop > 0);
    rng.random_range(0..stop)
}

fn shuffle<T>(values: &mut [T], rng: &mut StdRng) {
    for i in (1..values.len()).rev() {
        let j = rng.random_range(0..=i);
        values.swap(i, j);
    }
}

fn remove_first(values: &mut Vec<usize>, needle: usize) -> Result<(), String> {
    let index = values
        .iter()
        .position(|value| *value == needle)
        .ok_or_else(|| format!("missing edge target {needle}"))?;
    values.remove(index);
    Ok(())
}

fn replace_first(values: &mut [usize], needle: usize, replacement: usize) -> Result<(), String> {
    let index = values
        .iter()
        .position(|value| *value == needle)
        .ok_or_else(|| format!("missing edge parent {needle}"))?;
    values[index] = replacement;
    Ok(())
}

struct ReachScratch {
    seen_forward: Vec<usize>,
    seen_backward: Vec<usize>,
    forward_stack: Vec<usize>,
    backward_stack: Vec<usize>,
    token: usize,
}

impl ReachScratch {
    fn new(num_nodes: usize) -> Self {
        Self {
            seen_forward: vec![0; num_nodes],
            seen_backward: vec![0; num_nodes],
            forward_stack: Vec::new(),
            backward_stack: Vec::new(),
            token: 0,
        }
    }

    fn can_reach(
        &mut self,
        children: &[Vec<usize>],
        parents: &[Vec<usize>],
        src: usize,
        dst: usize,
    ) -> bool {
        if src == dst {
            return true;
        }

        self.token += 1;
        let token = self.token;
        self.forward_stack.clear();
        self.backward_stack.clear();
        self.seen_forward[src] = token;
        self.seen_backward[dst] = token;
        self.forward_stack.push(src);
        self.backward_stack.push(dst);

        while !self.forward_stack.is_empty() && !self.backward_stack.is_empty() {
            if self.forward_stack.len() <= self.backward_stack.len() {
                let node = self
                    .forward_stack
                    .pop()
                    .expect("forward stack is non-empty");
                for child in &children[node] {
                    if self.seen_backward[*child] == token {
                        return true;
                    }
                    if self.seen_forward[*child] != token {
                        self.seen_forward[*child] = token;
                        self.forward_stack.push(*child);
                    }
                }
            } else {
                let node = self
                    .backward_stack
                    .pop()
                    .expect("backward stack is non-empty");
                for parent in &parents[node] {
                    if self.seen_forward[*parent] == token {
                        return true;
                    }
                    if self.seen_backward[*parent] != token {
                        self.seen_backward[*parent] = token;
                        self.backward_stack.push(*parent);
                    }
                }
            }
        }

        false
    }
}

fn initialize_edge_split_graph(
    num_root_nodes: usize,
    num_trunk_nodes: usize,
    num_output_nodes: usize,
    rng: &mut StdRng,
) -> EdgeSplitGraph {
    let num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes;
    let mut parents = vec![Vec::new(); num_nodes];
    let mut children = vec![Vec::new(); num_nodes];
    let mut roles = vec![Role::Trunk; num_nodes];
    let trunk_types = vec![None; num_nodes];

    for role in roles.iter_mut().take(num_root_nodes) {
        *role = Role::Root;
    }

    let leaf_start = num_root_nodes;
    let trunk_start = num_root_nodes + num_output_nodes;
    for leaf in leaf_start..trunk_start {
        roles[leaf] = Role::Leaf;
        let root = randrange(rng, num_root_nodes);
        parents[leaf].push(root);
        children[root].push(leaf);
    }

    EdgeSplitGraph {
        parents,
        children,
        roles,
        trunk_types,
    }
}

fn active_nodes(
    num_root_nodes: usize,
    num_output_nodes: usize,
    inserted_trunks: usize,
) -> Vec<usize> {
    let trunk_start = num_root_nodes + num_output_nodes;
    let mut nodes = Vec::with_capacity(num_root_nodes + num_output_nodes + inserted_trunks);
    nodes.extend(0..num_root_nodes);
    nodes.extend(num_root_nodes..trunk_start);
    nodes.extend(trunk_start..trunk_start + inserted_trunks);
    nodes
}

fn active_nonleaf_nodes(
    num_root_nodes: usize,
    num_output_nodes: usize,
    inserted_trunks: usize,
) -> Vec<usize> {
    let trunk_start = num_root_nodes + num_output_nodes;
    let mut nodes = Vec::with_capacity(num_root_nodes + inserted_trunks);
    nodes.extend(0..num_root_nodes);
    nodes.extend(trunk_start..trunk_start + inserted_trunks);
    nodes
}

fn active_edges(graph: &EdgeSplitGraph, active_nodes: &[usize]) -> Vec<(usize, usize)> {
    let mut is_active = vec![false; graph.parents.len()];
    for node in active_nodes {
        is_active[*node] = true;
    }

    let mut edges = Vec::new();
    for parent in active_nodes {
        for child in &graph.children[*parent] {
            if is_active[*child] {
                edges.push((*parent, *child));
            }
        }
    }
    edges
}

fn sample_valid_parent_by_reachability(
    eligible: &[usize],
    graph: &EdgeSplitGraph,
    forbidden_node: usize,
    reach: &mut ReachScratch,
    rng: &mut StdRng,
) -> usize {
    for _ in 0..usize::max(8, eligible.len() * 2) {
        let candidate = eligible[randrange(rng, eligible.len())];
        if !reach.can_reach(
            &graph.children,
            &graph.parents,
            forbidden_node,
            candidate,
        ) {
            return candidate;
        }
    }

    let mut valid = Vec::new();
    for candidate in eligible {
        if !reach.can_reach(
            &graph.children,
            &graph.parents,
            forbidden_node,
            *candidate,
        ) {
            valid.push(*candidate);
        }
    }
    debug_assert!(!valid.is_empty());
    valid[randrange(rng, valid.len())]
}

#[allow(clippy::too_many_arguments)]
fn split_active_edge(
    graph: &mut EdgeSplitGraph,
    edge: (usize, usize),
    inserted_trunks: usize,
    num_root_nodes: usize,
    num_output_nodes: usize,
    num_trunk_node_types: usize,
    trunk_node_in_degrees: &[usize],
    reach: &mut ReachScratch,
    rng: &mut StdRng,
) -> Result<(), String> {
    let (u, v) = edge;
    let trunk_start = num_root_nodes + num_output_nodes;
    let w = trunk_start + inserted_trunks;
    let trunk_type = randrange(rng, num_trunk_node_types);
    graph.trunk_types[w] = Some(trunk_type);
    let in_degree = trunk_node_in_degrees[trunk_type];
    let eligible = active_nonleaf_nodes(num_root_nodes, num_output_nodes, inserted_trunks);

    let extras: Vec<usize> = (0..in_degree.saturating_sub(1))
        .map(|_| sample_valid_parent_by_reachability(&eligible, graph, v, reach, rng))
        .collect();

    remove_first(&mut graph.children[u], v)?;
    graph.children[u].push(w);
    replace_first(&mut graph.parents[v], u, w)?;
    graph.parents[w].push(u);
    graph.parents[w].extend(extras.iter().copied());
    graph.children[w].push(v);
    for extra in extras {
        graph.children[extra].push(w);
    }

    Ok(())
}

fn build_interleaved_edge_split_graph(
    num_root_nodes: usize,
    num_trunk_nodes: usize,
    num_output_nodes: usize,
    num_trunk_node_types: usize,
    trunk_node_in_degrees: &[usize],
    mcmc_passes_per_split_round: usize,
    rng: &mut StdRng,
) -> Result<EdgeSplitGraph, String> {
    let mut graph =
        initialize_edge_split_graph(num_root_nodes, num_trunk_nodes, num_output_nodes, rng);
    let mut reach = ReachScratch::new(graph.parents.len());
    let mut inserted_trunks = 0;

    while inserted_trunks < num_trunk_nodes {
        let active = active_nodes(num_root_nodes, num_output_nodes, inserted_trunks);
        let mut edges = active_edges(&graph, &active);
        debug_assert!(!edges.is_empty());
        shuffle(&mut edges, rng);

        for edge in edges {
            if inserted_trunks >= num_trunk_nodes {
                break;
            }
            split_active_edge(
                &mut graph,
                edge,
                inserted_trunks,
                num_root_nodes,
                num_output_nodes,
                num_trunk_node_types,
                trunk_node_in_degrees,
                &mut reach,
                rng,
            )?;
            inserted_trunks += 1;
        }

        let active = active_nodes(num_root_nodes, num_output_nodes, inserted_trunks);
        mcmc_rewire_active(
            &mut graph,
            &active,
            mcmc_passes_per_split_round * active.len(),
            rng,
        )?;
    }

    if num_trunk_nodes == 0 {
        let active = active_nodes(num_root_nodes, num_output_nodes, 0);
        mcmc_rewire_active(
            &mut graph,
            &active,
            mcmc_passes_per_split_round * active.len(),
            rng,
        )?;
    }

    Ok(graph)
}

fn mcmc_rewire_active(
    graph: &mut EdgeSplitGraph,
    active_nodes: &[usize],
    num_steps: usize,
    rng: &mut StdRng,
) -> Result<(), String> {
    if num_steps == 0 {
        return Ok(());
    }

    let nonleaf: Vec<usize> = active_nodes
        .iter()
        .filter_map(|node| (graph.roles[*node] != Role::Leaf).then_some(*node))
        .collect();
    if nonleaf.is_empty() {
        return Ok(());
    }

    let mut reach = ReachScratch::new(graph.children.len());
    let mut node_order: Vec<usize> = active_nodes.to_vec();
    shuffle(&mut node_order, rng);
    let mut node_order_idx = 0;

    for _ in 0..num_steps {
        if node_order_idx == node_order.len() {
            shuffle(&mut node_order, rng);
            node_order_idx = 0;
        }
        let node = node_order[node_order_idx];
        node_order_idx += 1;

        let parent_count = graph.parents[node].len();
        if parent_count == 0 {
            continue;
        }

        let parent_slot = randrange(rng, parent_count);
        let parent = graph.parents[node][parent_slot];
        let new_parent = nonleaf[randrange(rng, nonleaf.len())];

        if new_parent == node || new_parent == parent {
            continue;
        }
        if graph.children[parent].len() == 1 {
            continue;
        }

        if reach.can_reach(&graph.children, &graph.parents, node, new_parent) {
            continue;
        }

        graph.parents[node][parent_slot] = new_parent;
        remove_first(&mut graph.children[parent], node)?;
        graph.children[new_parent].push(node);
    }

    Ok(())
}

fn assemble_graph_arrays(
    graph: &EdgeSplitGraph,
    num_root_nodes: usize,
    num_trunk_nodes: usize,
    num_output_nodes: usize,
    num_trunk_node_types: usize,
    rng: &mut StdRng,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), String> {
    let mut in_degrees: Vec<usize> = graph.parents.iter().map(Vec::len).collect();
    let mut ready: VecDeque<usize> = in_degrees
        .iter()
        .enumerate()
        .filter_map(|(node, in_degree)| (*in_degree == 0).then_some(node))
        .collect();
    let mut topo_order = Vec::with_capacity(graph.parents.len());

    while let Some(node) = ready.pop_front() {
        topo_order.push(node);
        for child in &graph.children[node] {
            in_degrees[*child] -= 1;
            if in_degrees[*child] == 0 {
                ready.push_back(*child);
            }
        }
    }

    if topo_order.len() != graph.parents.len() {
        return Err("generated graph contains a cycle".to_string());
    }

    let mut roots = Vec::with_capacity(num_root_nodes);
    let mut trunks = Vec::with_capacity(num_trunk_nodes);
    let mut leaves = Vec::with_capacity(num_output_nodes);
    for node in topo_order {
        match graph.roles[node] {
            Role::Root => roots.push(node),
            Role::Trunk => trunks.push(node),
            Role::Leaf => leaves.push(node),
        }
    }

    let mut final_order = roots;
    final_order.extend(trunks);
    final_order.extend(leaves);

    let mut old_to_new = vec![usize::MAX; graph.parents.len()];
    for (new_node, old_node) in final_order.iter().enumerate() {
        old_to_new[*old_node] = new_node;
    }

    let root_node_types_start = num_trunk_node_types;
    let output_node_types_start = num_trunk_node_types + num_root_nodes;
    let mut root_slot = 0;
    let mut output_slot = 0;
    let mut parent_offsets = Vec::with_capacity(graph.parents.len() + 1);
    let mut parent_indices = Vec::new();
    let mut node_types = vec![0; graph.parents.len()];

    parent_offsets.push(0);
    for (new_node, old_node) in final_order.iter().enumerate() {
        let role = graph.roles[*old_node];
        let mut parent_list = graph.parents[*old_node].clone();
        if role == Role::Trunk {
            shuffle(&mut parent_list, rng);
        }
        parent_indices.extend(parent_list.iter().map(|parent| old_to_new[*parent]));
        parent_offsets.push(parent_indices.len());

        node_types[new_node] = match role {
            Role::Root => {
                let node_type = root_node_types_start + root_slot;
                root_slot += 1;
                node_type
            }
            Role::Trunk => graph.trunk_types[*old_node]
                .ok_or_else(|| format!("trunk node {old_node} has no type"))?,
            Role::Leaf => {
                let node_type = output_node_types_start + output_slot;
                output_slot += 1;
                node_type
            }
        };
    }

    if root_slot != num_root_nodes {
        return Err(format!(
            "expected {num_root_nodes} roots, found {root_slot}"
        ));
    }
    if output_slot != num_output_nodes {
        return Err(format!(
            "expected {num_output_nodes} outputs, found {output_slot}"
        ));
    }

    Ok((parent_offsets, parent_indices, node_types))
}

#[pyfunction]
#[pyo3(signature = (
    num_root_nodes,
    num_trunk_nodes,
    num_output_nodes,
    trunk_node_in_degrees,
    num_trunk_node_types,
    seed,
    mcmc_passes_per_split_round=DEFAULT_MCMC_PASSES_PER_SPLIT_ROUND
))]
fn generate_random_graph_arrays(
    num_root_nodes: usize,
    num_trunk_nodes: usize,
    num_output_nodes: usize,
    trunk_node_in_degrees: Vec<usize>,
    num_trunk_node_types: usize,
    seed: u64,
    mcmc_passes_per_split_round: usize,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    if trunk_node_in_degrees.len() != num_trunk_node_types {
        return Err(PyValueError::new_err(
            "trunk_node_in_degrees length must match num_trunk_node_types",
        ));
    }
    if trunk_node_in_degrees
        .iter()
        .any(|in_degree| *in_degree == 0)
    {
        return Err(PyValueError::new_err("all trunk in-degrees must be >= 1"));
    }
    if num_root_nodes == 0 {
        return Err(PyValueError::new_err("num_root_nodes must be >= 1"));
    }
    if num_output_nodes == 0 {
        return Err(PyValueError::new_err("num_output_nodes must be >= 1"));
    }
    if num_trunk_node_types == 0 {
        return Err(PyValueError::new_err("num_trunk_node_types must be >= 1"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let graph = build_interleaved_edge_split_graph(
        num_root_nodes,
        num_trunk_nodes,
        num_output_nodes,
        num_trunk_node_types,
        &trunk_node_in_degrees,
        mcmc_passes_per_split_round,
        &mut rng,
    )
    .map_err(PyValueError::new_err)?;

    assemble_graph_arrays(
        &graph,
        num_root_nodes,
        num_trunk_nodes,
        num_output_nodes,
        num_trunk_node_types,
        &mut rng,
    )
    .map_err(PyValueError::new_err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_random_graph_arrays, m)?)?;
    Ok(())
}
