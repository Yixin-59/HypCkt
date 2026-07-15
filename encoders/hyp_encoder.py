"""Hyperbolic encoder for subgraph-level circuit DAGs."""

import torch
import torch.nn as nn

from encoders.topology_aware_hyp_message_passing import (
    TopologyAwareHypMessagePassing,
)


class HypEncoder(nn.Module):
    """Encode circuit DAGs as wrapped-normal posterior parameters."""

    def __init__(
        self,
        manifold,
        max_n,
        nvt,
        max_pos=8,
        emb_dim=16,
        feat_emb_dim=8,
        hs=301,
        nz=66,
        pos=True,
        debug_nan=False,
    ):
        super().__init__()
        self.max_pos = max_pos + 1
        self.max_n = max_n
        self.nvt = nvt
        self.hs = hs
        self.nz = nz
        self.feat_emb_dim = feat_emb_dim
        self.pos = pos

        self.message_passing = TopologyAwareHypMessagePassing(
            manifold=manifold,
            nvt=nvt,
            max_pos=self.max_pos,
            hs=hs,
            max_n=max_n,
            pos=pos,
            debug_nan=debug_nan,
        )
        self.df_enc = nn.Sequential(
            nn.Linear(self.max_pos * 3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, feat_emb_dim),
        )
        self.fc1 = nn.Linear(hs + feat_emb_dim, nz)
        self.fc2 = nn.Linear(hs + feat_emb_dim, nz)

    def _device(self):
        return next(self.parameters()).device

    def _collate_fn(self, graphs):
        return [graph.copy() for graph in graphs]

    @staticmethod
    def _get_graph_state_hyp(graphs):
        sink_states = [
            graph.vs[graph.vcount() - 1]["H_forward"] for graph in graphs
        ]
        return torch.cat(sink_states, dim=0)

    def encode(self, graphs):
        """Return posterior mean and log variance in the origin tangent space."""
        if not isinstance(graphs, list):
            graphs = [graphs]

        initial_state = torch.zeros(
            len(graphs), self.hs, dtype=torch.float64, device=self._device()
        )
        self.message_passing.propagate_from(
            graphs, vertex=0, initial_state=initial_state
        )

        graph_state = self._get_graph_state_hyp(graphs)
        tangent_state = self.message_passing.safe_logmap0(graph_state).float()

        device_features = []
        for graph in graphs:
            features = [0.0] * (3 * self.max_pos)
            for vertex in graph.vs:
                position = vertex["pos"]
                features[position * 3] = vertex["r"]
                features[position * 3 + 1] = vertex["c"]
                features[position * 3 + 2] = vertex["gm"]
            device_features.append(features)

        feature_tensor = torch.tensor(
            device_features, dtype=torch.float32, device=self._device()
        )
        encoded_features = self.df_enc(feature_tensor)
        fused_state = torch.cat([tangent_state, encoded_features], dim=1)
        return self.fc1(fused_state), self.fc2(fused_state)

    def forward(self, graphs):
        return self.encode(graphs)
