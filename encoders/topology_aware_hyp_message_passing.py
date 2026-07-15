"""Topology-aware hyperbolic message passing for circuit DAGs.

Predecessor states are aggregated in the origin tangent space and fused with
node features by a GRU cell before being mapped back to the Poincare ball.
"""

import torch
import torch.nn as nn

MAX_NORM = 15.0


class TopologyAwareHypMessagePassing(nn.Module):

    def __init__(self, manifold, nvt, max_pos, hs, max_n,
                 pos=True, dropout=0.0, debug_nan=False):
        super().__init__()
        self.manifold = manifold
        self.nvt = nvt
        self.max_pos = max_pos
        self.hs = hs
        self.max_n = max_n
        self.pos = pos
        self.debug_nan = debug_nan

        self.vs = hs + max_pos if pos else hs
        self.input_size = nvt + max_pos

        self.gate = nn.Sequential(
            nn.Linear(self.vs, hs),
            nn.Sigmoid(),
        )
        self.mapper = nn.Sequential(
            nn.Linear(self.vs, hs, bias=False),
        )
        self.gru = nn.GRUCell(self.input_size, hs)

        self.double()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _device(self):
        return next(self.parameters()).device

    def _get_zeros(self, n, length):
        return torch.zeros(n, length, dtype=torch.float64, device=self._device())

    def _one_hot(self, idx, length):
        if isinstance(idx, (list, range)):
            if len(idx) == 0:
                return None
            idx_t = torch.LongTensor(idx).unsqueeze(1)
            x = torch.zeros(len(idx), length, dtype=torch.float64)
            x.scatter_(1, idx_t, 1.0)
        else:
            idx_t = torch.LongTensor([[idx]])
            x = torch.zeros(1, length, dtype=torch.float64)
            x.scatter_(1, idx_t, 1.0)
        return x.to(self._device())

    # ------------------------------------------------------------------
    # Safe manifold operations
    # ------------------------------------------------------------------

    def safe_logmap0(self, x):
        x = self.manifold.proj(x)
        t = self.manifold.logmap0(x)
        if self.debug_nan:
            assert not torch.isnan(t).any(), "NaN after logmap0"
        return t

    def safe_expmap0(self, u):
        norms = u.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        u = torch.where(norms > MAX_NORM, u * MAX_NORM / norms, u)
        h = self.manifold.expmap0(u)
        if self.debug_nan:
            assert not torch.isnan(h).any(), "NaN after expmap0"
        return h

    # ------------------------------------------------------------------
    # Gated aggregation in tangent space
    # ------------------------------------------------------------------

    def _hyp_gated(self, H_pred_tan):
        """Gated aggregation in tangent space.

        Args:
            H_pred_tan: (B, max_n_pred, vs) tangent vectors with pos concat.
        Returns:
            (B, hs) point on Poincare ball.
        """
        gated = self.gate(H_pred_tan) * self.mapper(H_pred_tan)
        H_agg_tan = gated.sum(dim=1)
        return self.safe_expmap0(H_agg_tan)

    # ------------------------------------------------------------------
    # Core propagation
    # ------------------------------------------------------------------

    def _hyp_propagate_to(self, G, v, H=None, reverse=False):
        """Propagate messages to vertex v for all graphs in G.

        Args:
            G: list of igraph.Graph
            v: vertex index
            H: (B, hs) initial hidden state on ball, or None
            reverse: use successors instead of predecessors
        Returns:
            Hv_hyp: (B, hs) on Poincare ball, or None if no graphs
        """
        idx = [i for i, g in enumerate(G) if g.vcount() > v]
        G = [G[i] for i in idx]
        if len(G) == 0:
            return None
        if H is not None:
            H = H[idx]

        # Step 2: node features (Euclidean)
        v_types = [g.vs[v]['type'] for g in G]
        pos_feats = [g.vs[v]['pos'] for g in G]
        X = torch.cat([self._one_hot(v_types, self.nvt),
                        self._one_hot(pos_feats, self.max_pos)], dim=1)

        # Step 3: predecessor hidden states
        H_name = 'H_backward' if reverse else 'H_forward'
        if reverse:
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
        else:
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]

        # Step 4: compute H_agg_hyp
        if H is None:
            max_n_pred = max(len(x) for x in H_pred)
            if max_n_pred == 0:
                H_agg_hyp = self.manifold.proj(self._get_zeros(len(G), self.hs))
            else:
                H_pred_tan = []
                for i, g in enumerate(G):
                    preds = g.successors(v) if reverse else g.predecessors(v)
                    tangents = []
                    for j, h in enumerate(H_pred[i]):
                        t = self.safe_logmap0(h)
                        if self.pos:
                            p = self._one_hot([g.vs[preds[j]]['pos']], self.max_pos)
                            t = torch.cat([t, p], dim=1)
                        tangents.append(t)
                    n_pad = max_n_pred - len(tangents)
                    if n_pad > 0:
                        tangents.append(self._get_zeros(n_pad, self.vs))
                    H_pred_tan.append(torch.cat(tangents, dim=0).unsqueeze(0))
                H_pred_tan = torch.cat(H_pred_tan, dim=0)
                H_agg_hyp = self._hyp_gated(H_pred_tan)
        else:
            H_agg_hyp = H

        # Move to the tangent space for the GRU update, then map back.
        H_agg_tan = self.safe_logmap0(H_agg_hyp)
        Hv_tan = self.gru(X, H_agg_tan)
        Hv_hyp = self.safe_expmap0(Hv_tan)

        # Step 6: store on graph vertices
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv_hyp[i:i+1]

        return Hv_hyp

    def propagate_from(self, G, vertex, initial_state=None, reverse=False):
        """Propagate along topological order starting from v.

        Args:
            G: list of igraph.Graph
            vertex: starting vertex index
            initial_state: (B, hs) initial hidden state on ball
            reverse: propagation direction
        Returns:
            Hv: hidden state of starting vertex
        """
        prop_order = (
            range(vertex, -1, -1)
            if reverse
            else range(vertex, self.max_n)
        )
        Hv = self._hyp_propagate_to(
            G, vertex, initial_state, reverse=reverse
        )
        for v_ in list(prop_order)[1:]:
            self._hyp_propagate_to(G, v_, reverse=reverse)
        return Hv
