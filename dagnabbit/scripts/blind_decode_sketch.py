"""
This file contains non-functional PSEUDOCODE for a blind decode strategy.
"""

import torch
from torch import Tensor
import torch.nn as nn
from dataclasses import dataclass
from dagnabbit.dag.description import FixedInDegreeDAGDescription


# A cluster is a list of 1 or more embeddings
# When a cluster has to act like a single embedding,
# it represents itself as the centroid of its constituent embeddings
class Cluster(list[Tensor]):
    def __init__(self, embeddings: list[Tensor]):
        self.embeddings: list[Tensor] = embeddings

    def centroid(self) -> Tensor:
        return torch.mean(self.embeddings, dim=0)

    def merge(self, other: "Cluster") -> "Cluster":
        return Cluster(self.embeddings + other.embeddings)

class Rank:
    clusters: list[Cluster]

class Buffer:
    ranks: list[Rank]
    roots: Rank

# GIVEN

# — The fixed-size set of embeddings of the output nodes of a given DAG
leaf_embeddings: Tensor

# — The requisite decoder modules
decoders: dict[int, nn.Module]

# — The classifier module
classifier: nn.Module

# HERE IS A PARTIALLY UNROLLED VIEW OF AN ALGORITHM TO RECONSTRUCT THE GRAPH.


def pairwise_cosine_similarity(a: Rank, b: Rank) -> Tensor:
    """
    compute the cosine similarity between all pairs of vectors in a and b
    """
    pass


def suppress_non_maximum_similarities(similarities_matrix: Tensor) -> Tensor:
    """
    Mask out all similarities that are not the maximum similarity for each downstream cluster.
    """
    pass


def discover_upward_merges(
    downstream_rank: Rank,
    upstream_rank: Rank,
    similarity_threshold: float = 0.99,
) -> list[tuple[int, int]]:
    """
    Returns a list of tuples (downstream_index, upstream_index)
    for all pairs of clusters in downstream_rank and upstream_rank
    that are similar enough to each other.
    """
    # Get the pairwise
    similarities_matrix = pairwise_cosine_similarity(
        downstream_rank.clusters, upstream_rank.clusters
    )

    # Find all pairs of vectors with similarity greater than the threshold
    # If a downstream cluster is similar enough to *more* than one upstream cluster,
    # it should be merged with the upstream cluster that it is most similar to.
    # So we first mask out the non-maximum similarites, and then threshold the remaining similarities.

    similarities_matrix = suppress_non_maximum_similarities(similarities_matrix)
    matches = (similarities_matrix > similarity_threshold).nonzero()

    return matches.tolist()


def commit_upward_merges(
    downstream_clusters: Rank,
    upstream_clusters: Rank,
    merges: list[tuple[int, int]],
) -> tuple[Rank, Rank]:

    """
    for each merge identified, we remove the downstream cluster from the
    list of downstream cluster and merge it with the upstream cluster we determined
    it should merge with.
    """

    pass


def commit_coplanar_merges(clusters: Rank, merges: list[tuple[int, int]]) -> Rank:
    """
    Handles the case where we want to merge clusters *within* the same list, 
    instead of move-and-merging clusters from a downstream list to an upstream list.
    """
    pass

def termination_condition_met(buffer: Buffer) -> bool:
    """
    Returns True if a termination condition is met, False otherwise.
    Marks the buffer with a termination reason (max nodes, natural_termination, etc) when it is terminated.
    """
    pass

def convert_terminated_buffer(buffer: Buffer) -> FixedInDegreeDAGDescription:
    """
    Converts the terminated buffer into a FixedInDegreeDAGDescription.
    """
    pass


seed_rank: Rank = decoders["output"].decode(leaf_embeddings)

root_merges = discover_upward_merges(seed_rank, seed_rank.roots)
seed_rank = commit_upward_merges(seed_rank, seed_rank.roots, root_merges)
merges = discover_upward_merges(seed_rank, seed_rank)
seed_rank = commit_coplanar_merges(seed_rank, merges)


buffer = Buffer(ranks=[seed_rank])

# DO WHILE (a) NOT ALL ANCESTRIES END IN ROOTS, or (b) THE MAXIMUM NODE COUNT HAS BEEN REACHED.
while not termination_condition_met(buffer):

    # Predict the next rank:
    next_rank: Rank = decoders["output"].decode(buffer.ranks[0])

    # If the next rank has merges with the roots, we handle them
    root_merges = discover_upward_merges(next_rank, buffer.roots)
    next_rank = commit_upward_merges(next_rank, buffer.roots, root_merges)

    # If the next rank has merges within itself, we handle them
    self_merges = discover_upward_merges(next_rank, buffer)
    next_rank = commit_coplanar_merges(next_rank, self_merges)

    buffer.ranks = [next_rank] + buffer.ranks

    # Then we iterate over the ranks downstream from new rank, and we do upward_merges with the new rank
    for rank in buffer.ranks[1:]:
        merges = discover_upward_merges(rank, next_rank)
        rank = commit_upward_merges(rank, next_rank, merges)

