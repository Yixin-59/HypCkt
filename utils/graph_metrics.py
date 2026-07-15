from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm


def is_same_dag(graph_a, graph_b):
    """Compare two topologically ordered DAGs by node types and incoming edges."""
    if graph_a.vcount() != graph_b.vcount():
        return False
    for index in range(graph_a.vcount()):
        if graph_a.vs[index]["type"] != graph_b.vs[index]["type"]:
            return False
        incoming_a = set(graph_a.neighbors(index, "in"))
        incoming_b = set(graph_b.neighbors(index, "in"))
        if incoming_a != incoming_b:
            return False
    return True


def ratio_same_dag(reference_graphs, candidate_graphs):
    """Return the fraction of candidate graphs found in the reference set."""
    if not candidate_graphs:
        return 0.0
    matches = 0
    for candidate in tqdm(candidate_graphs, desc="novelty"):
        if any(is_same_dag(candidate, reference) for reference in reference_graphs):
            matches += 1
    return matches / len(candidate_graphs)


def is_valid_dag(graph, subgraph_level=True):
    """Check acyclicity and the presence of one input and one output node."""
    start_type, end_type = (0, 1) if subgraph_level else (8, 9)
    if not graph.is_dag():
        return False

    n_start = 0
    n_end = 0
    for vertex in graph.vs:
        if vertex["type"] == start_type:
            n_start += 1
        elif vertex["type"] == end_type:
            n_end += 1
        if vertex.outdegree() == 0 and vertex["type"] != end_type:
            return False
    return n_start == 1 and n_end == 1


def is_valid_circuit(graph, subgraph_level=True):
    """Apply the circuit validity rules used by the benchmark."""
    if subgraph_level:
        if not is_valid_dag(graph, subgraph_level=True):
            return False
        for vertex in graph.vs:
            if vertex["pos"] in (2, 3, 4) and vertex["type"] in (8, 9):
                return False
        return True

    if not is_valid_dag(graph, subgraph_level=False):
        return False
    diameter_path = graph.get_diameter(directed=True)
    if len(diameter_path) < 3:
        return False
    for path_index, vertex_index in enumerate(diameter_path):
        vertex_type = graph.vs[vertex_index]["type"]
        if path_index == 0 and vertex_type != 8:
            return False
        if path_index == len(diameter_path) - 1 and vertex_type != 9:
            return False
        if 0 < path_index < len(diameter_path) - 1 and vertex_type in (4, 5):
            return False
    return True


def extract_latent_z(data, model, infer_batch_size=64):
    """Encode graphs and return their tangent-space latent means."""
    model.eval()
    batches = []
    graph_batch = []
    with torch.no_grad():
        for index, graph in enumerate(tqdm(data, desc="encoding")):
            graph_batch.append(deepcopy(graph))
            if len(graph_batch) == infer_batch_size or index == len(data) - 1:
                collated = model._collate_fn(graph_batch)
                mean, _ = model.encode(collated)
                batches.append(mean.detach().cpu().numpy())
                graph_batch = []
    return np.concatenate(batches, axis=0)


# Backward-compatible aliases used by the original evaluation code.
is_same_DAG = is_same_dag
ratio_same_DAG = ratio_same_dag
is_valid_DAG = is_valid_dag
is_valid_Circuit = is_valid_circuit
