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

struct DescendantBitsets {
    words: usize,
    bits: Vec<u64>,
    scratch_bits: Vec<u64>,
    stack: Vec<usize>,
}

impl DescendantBitsets {
    fn new(num_nodes: usize) -> Self {
        let words = num_nodes.div_ceil(u64::BITS as usize);
        Self {
            words,
            bits: vec![0; num_nodes * words],
            scratch_bits: vec![0; words],
            stack: Vec::new(),
        }
    }

    fn row_start(&self, node: usize) -> usize {
        node * self.words
    }

    fn contains(&self, node: usize, descendant: usize) -> bool {
        let word = descendant / u64::BITS as usize;
        let bit = descendant % u64::BITS as usize;
        (self.bits[self.row_start(node) + word] & (1_u64 << bit)) != 0
    }

    fn set(&mut self, node: usize, descendant: usize) {
        let word = descendant / u64::BITS as usize;
        let bit = descendant % u64::BITS as usize;
        let index = self.row_start(node) + word;
        self.bits[index] |= 1_u64 << bit;
    }

    fn clear_row(&mut self, node: usize) {
        let start = self.row_start(node);
        self.bits[start..start + self.words].fill(0);
    }

    fn copy_descendants_with_node_to_scratch(&mut self, node: usize) {
        let start = self.row_start(node);
        self.scratch_bits
            .copy_from_slice(&self.bits[start..start + self.words]);
        let word = node / u64::BITS as usize;
        let bit = node % u64::BITS as usize;
        self.scratch_bits[word] |= 1_u64 << bit;
    }

    fn copy_scratch_to_row(&mut self, node: usize) {
        let start = self.row_start(node);
        self.bits[start..start + self.words].copy_from_slice(&self.scratch_bits);
    }

    fn set_only_scratch(&mut self, node: usize) {
        self.scratch_bits.fill(0);
        let word = node / u64::BITS as usize;
        let bit = node % u64::BITS as usize;
        self.scratch_bits[word] = 1_u64 << bit;
    }

    fn propagate_scratch_to_ancestors(&mut self, parents: &[Vec<usize>], start: usize) {
        self.stack.clear();
        self.stack.push(start);
        while let Some(node) = self.stack.pop() {
            let row_start = self.row_start(node);
            let mut added = false;
            for word in 0..self.words {
                let index = row_start + word;
                let combined = self.bits[index] | self.scratch_bits[word];
                if combined != self.bits[index] {
                    self.bits[index] = combined;
                    added = true;
                }
            }
            if added {
                self.stack.extend(parents[node].iter().copied());
            }
        }
    }
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

fn sample_valid_parent(
    eligible: &[usize],
    descendants: &DescendantBitsets,
    forbidden_node: usize,
    rng: &mut StdRng,
) -> usize {
    for _ in 0..usize::max(8, eligible.len() * 2) {
        let candidate = eligible[randrange(rng, eligible.len())];
        if candidate != forbidden_node && !descendants.contains(forbidden_node, candidate) {
            return candidate;
        }
    }

    let valid_count = eligible
        .iter()
        .filter(|candidate| {
            **candidate != forbidden_node && !descendants.contains(forbidden_node, **candidate)
        })
        .count();
    debug_assert!(valid_count > 0);

    let selected = randrange(rng, valid_count);
    let mut seen_valid = 0;
    for candidate in eligible {
        if *candidate == forbidden_node || descendants.contains(forbidden_node, *candidate) {
            continue;
        }
        if seen_valid == selected {
            return *candidate;
        }
        seen_valid += 1;
    }

    unreachable!("valid parent count was nonzero");
}

fn build_edge_split_graph(
    num_root_nodes: usize,
    num_trunk_nodes: usize,
    num_output_nodes: usize,
    num_trunk_node_types: usize,
    trunk_node_in_degrees: &[usize],
    rng: &mut StdRng,
) -> Result<EdgeSplitGraph, String> {
    let num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes;
    let mut parents = vec![Vec::new(); num_nodes];
    let mut children = vec![Vec::new(); num_nodes];
    let mut roles = vec![Role::Trunk; num_nodes];
    let mut trunk_types = vec![None; num_nodes];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut eligible: Vec<usize> = (0..num_root_nodes).collect();
    let mut descendants = DescendantBitsets::new(num_nodes);

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
        edges.push((root, leaf));
        descendants.set(root, leaf);
    }

    let mut split_idx = 0;
    while split_idx < num_trunk_nodes {
        let mut round_edge_indices: Vec<usize> = (0..edges.len()).collect();
        shuffle(&mut round_edge_indices, rng);

        for edge_idx in round_edge_indices {
            if split_idx >= num_trunk_nodes {
                break;
            }

            let (u, v) = edges[edge_idx];
            let w = trunk_start + split_idx;
            let trunk_type = randrange(rng, num_trunk_node_types);
            trunk_types[w] = Some(trunk_type);
            let in_degree = trunk_node_in_degrees[trunk_type];

            let extras: Vec<usize> = (0..in_degree.saturating_sub(1))
                .map(|_| sample_valid_parent(&eligible, &descendants, v, rng))
                .collect();

            roles[w] = Role::Trunk;
            remove_first(&mut children[u], v)?;
            children[u].push(w);
            replace_first(&mut parents[v], u, w)?;
            parents[w].push(u);
            parents[w].extend(extras.iter().copied());
            children[w].push(v);
            for extra in &extras {
                children[*extra].push(w);
            }

            edges[edge_idx] = (u, w);
            edges.push((w, v));
            edges.extend(extras.iter().map(|extra| (*extra, w)));

            descendants.clear_row(w);
            descendants.copy_descendants_with_node_to_scratch(v);
            descendants.copy_scratch_to_row(w);
            descendants.set_only_scratch(w);
            descendants.propagate_scratch_to_ancestors(&parents, u);
            descendants.copy_descendants_with_node_to_scratch(w);
            for extra in &extras {
                descendants.propagate_scratch_to_ancestors(&parents, *extra);
            }

            eligible.push(w);
            split_idx += 1;
        }
    }

    Ok(EdgeSplitGraph {
        parents,
        children,
        roles,
        trunk_types,
    })
}

fn mcmc_rewire(
    graph: &mut EdgeSplitGraph,
    num_steps: usize,
    rng: &mut StdRng,
) -> Result<(), String> {
    let nonleaf: Vec<usize> = graph
        .roles
        .iter()
        .enumerate()
        .filter_map(|(node, role)| (*role != Role::Leaf).then_some(node))
        .collect();
    let mut edges: Vec<(usize, usize, usize)> = graph
        .parents
        .iter()
        .enumerate()
        .flat_map(|(node, parent_list)| {
            parent_list
                .iter()
                .enumerate()
                .map(move |(parent_slot, parent)| (*parent, node, parent_slot))
        })
        .collect();

    let edge_count = edges.len();
    let mut reach = ReachScratch::new(graph.children.len());

    for _ in 0..num_steps {
        if edge_count >= 2 && randrange(rng, 2) == 0 {
            let first_idx = randrange(rng, edge_count);
            let mut second_idx = randrange(rng, edge_count - 1);
            if second_idx >= first_idx {
                second_idx += 1;
            }

            let (first_parent, first_child, first_parent_slot) = edges[first_idx];
            let (second_parent, second_child, second_parent_slot) = edges[second_idx];

            if first_child == second_child || first_parent == second_parent {
                continue;
            }
            if first_parent == second_child || second_parent == first_child {
                continue;
            }

            if reach.can_reach(&graph.children, &graph.parents, second_child, first_parent)
                || reach.can_reach(&graph.children, &graph.parents, first_child, second_parent)
            {
                continue;
            }

            remove_first(&mut graph.children[first_parent], first_child)?;
            remove_first(&mut graph.children[second_parent], second_child)?;
            edges[first_idx] = (first_parent, second_child, second_parent_slot);
            edges[second_idx] = (second_parent, first_child, first_parent_slot);
            graph.parents[first_child][first_parent_slot] = second_parent;
            graph.parents[second_child][second_parent_slot] = first_parent;
            graph.children[first_parent].push(second_child);
            graph.children[second_parent].push(first_child);
            continue;
        }

        let edge_idx = randrange(rng, edge_count);
        let (parent, node, parent_slot) = edges[edge_idx];
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

        edges[edge_idx] = (new_parent, node, parent_slot);
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
    num_mixing_steps=None
))]
fn generate_random_graph_arrays(
    num_root_nodes: usize,
    num_trunk_nodes: usize,
    num_output_nodes: usize,
    trunk_node_in_degrees: Vec<usize>,
    num_trunk_node_types: usize,
    seed: u64,
    num_mixing_steps: Option<usize>,
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
    let mut graph = build_edge_split_graph(
        num_root_nodes,
        num_trunk_nodes,
        num_output_nodes,
        num_trunk_node_types,
        &trunk_node_in_degrees,
        &mut rng,
    )
    .map_err(PyValueError::new_err)?;

    let steps = num_mixing_steps.unwrap_or(num_trunk_nodes * 8);
    mcmc_rewire(&mut graph, steps, &mut rng).map_err(PyValueError::new_err)?;
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
