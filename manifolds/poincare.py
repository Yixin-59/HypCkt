import torch
import torch.nn as nn
from geoopt.manifolds.stereographic import math as gmath
from .base import Manifold


class PoincareBall(nn.Module, Manifold):
    """Poincare ball manifold with optional learnable curvature.

    All operations delegate to geoopt.manifolds.stereographic.math,
    with k = -c (geoopt uses signed curvature: negative = hyperbolic).

    Args:
        c: curvature parameter (positive float). Default 1.0.
        learnable: if True, c is an nn.Parameter; otherwise a buffer.
    """

    C_MIN = 0.1
    C_MAX = 2.0

    def __init__(self, c=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.c = nn.Parameter(torch.tensor([float(c)]))
        else:
            self.register_buffer('c', torch.tensor([float(c)]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_c(self):
        """Get clamped curvature value."""
        return torch.clamp(self.c, min=self.C_MIN, max=self.C_MAX)

    def _get_k(self):
        """Get geoopt-compatible signed curvature: k = -c (negative = hyperbolic)."""
        return -self._get_c()

    def _eps(self, dtype):
        """Dtype-adaptive epsilon for avoiding division by zero."""
        return {torch.float32: 1e-7, torch.float64: 1e-15}.get(dtype, 1e-15)

    def _boundary_eps(self, dtype):
        """Dtype-adaptive boundary margin for projection."""
        return {torch.float32: 1e-5, torch.float64: 1e-7}.get(dtype, 1e-5)

    # ------------------------------------------------------------------
    # Manifold interface
    # ------------------------------------------------------------------

    def proj(self, x):
        """Project point x onto the Poincare ball (ensures ||x|| < 1/sqrt(c))."""
        k = self._get_k()
        return gmath.project(x, k=k)

    def proj_tan(self, u, x):
        """Project vector u onto tangent space at x.

        For the Poincare ball, all vectors are valid tangent vectors.
        We ensure shape/contiguity consistency to prevent silent bugs.
        """
        return u.expand_as(x).contiguous() if u.shape != x.shape else u

    def expmap(self, u, x):
        """Exponential map at x: T_x M -> M."""
        k = self._get_k()
        res = gmath.expmap(x, u, k=k)
        return gmath.project(res, k=k)

    def logmap(self, y, x):
        """Logarithmic map at x: M -> T_x M."""
        k = self._get_k()
        return gmath.logmap(x, y, k=k)

    def expmap0(self, u):
        """Exponential map at the origin."""
        k = self._get_k()
        res = gmath.expmap0(u, k=k)
        return gmath.project(res, k=k)

    def logmap0(self, x):
        """Logarithmic map at the origin."""
        k = self._get_k()
        return gmath.logmap0(x, k=k)

    def ptransp(self, x, y, u):
        """Parallel transport of u from T_x M to T_y M."""
        k = self._get_k()
        return gmath.parallel_transport(x, y, u, k=k)

    def ptransp0(self, y, u):
        """Parallel transport from origin to y."""
        k = self._get_k()
        return gmath.parallel_transport0(y, u, k=k)

    def dist(self, x, y):
        """Geodesic distance between x and y."""
        k = self._get_k()
        return gmath.dist(x, y, k=k)

    def mobius_add(self, x, y):
        """Mobius addition x (+) y."""
        k = self._get_k()
        res = gmath.mobius_add(x, y, k=k)
        return gmath.project(res, k=k)

    def mobius_matvec(self, m, x):
        """Mobius matrix-vector multiplication."""
        k = self._get_k()
        res = gmath.mobius_matvec(m, x, k=k)
        return gmath.project(res, k=k)

    @property
    def name(self):
        return 'PoincareBall'

    def __repr__(self):
        return 'PoincareBall(c={:.4f}, learnable={})'.format(
            self.c.item(), isinstance(self.c, nn.Parameter)
        )
