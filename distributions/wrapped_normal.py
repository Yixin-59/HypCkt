"""Wrapped Normal distribution on the Poincare ball (Nagano 2019, Mathieu 2019).

Sampling (Nagano 2019, three steps):
    v ~ N(0, I) * scale     in T_0 M
    u = ptransp(0 -> loc, v) in T_loc M
    z = expmap(loc, u)      on the ball

Log-density (change of variables):
    log p(z) = log N(v; 0, scale^2) - log|det J_expmap|
    log|det J_expmap| = (d-1) * log( sinh(sqrt(c) * d_H) / (sqrt(c) * d_H) )
    where d_H = manifold.dist(loc, z) is the curvature-c geodesic distance.

The sqrt(c) factor inside sinh comes from the generalized hyperbolic sine
sn_{-c}(r) = sinh(sqrt(c) * r) / sqrt(c) used in the polar volume element of
H^n with sectional curvature -c. pvae's c=1 implementation omits this factor;
that is correct only at unit curvature. Our c is learnable in [0.1, 2.0], so
the factor must appear explicitly.

References:
    - Nagano et al. 2019, https://arxiv.org/abs/1902.02992
    - Mathieu et al. 2019, https://arxiv.org/abs/1901.06033
    - emilemathieu/pvae (reference implementation, c=1 only)
"""

import math

import torch
from torch.distributions import Distribution, constraints

from utils.dtype import to_hyp_dtype


class WrappedNormal(Distribution):
    """Wrapped Normal on the Poincare ball with curvature -c.

    Args:
        loc: (..., d) tensor containing the mean on the ball. Values are
            projected with ``manifold.proj``.
        scale: (..., d) tensor containing tangent-space standard deviations.
            Values are clamped to ``[exp(-5), exp(2)]``.
        manifold: PoincareBall instance; curvature read from manifold._get_c().
        validate_args: passed to torch.distributions.Distribution.

    All manifold computations use float64.
    """

    arg_constraints = {'scale': constraints.positive}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, scale, manifold, validate_args=None):
        self.manifold = manifold

        loc = to_hyp_dtype(loc)
        loc = manifold.proj(loc)
        scale = to_hyp_dtype(scale).clamp(
            min=math.exp(-5.0), max=math.exp(2.0)
        )
        loc, scale = torch.broadcast_tensors(loc, scale)

        self.loc = loc
        self.scale = scale

        super().__init__(
            batch_shape=loc.shape[:-1],
            event_shape=loc.shape[-1:],
            validate_args=validate_args,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.loc.shape
        loc_exp = self.loc.expand(shape)
        scale_exp = self.scale.expand(shape)

        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        v = eps * scale_exp                              # T_0 M
        u = self.manifold.ptransp0(loc_exp, v)           # T_loc M
        z = self.manifold.expmap(u, loc_exp)             # on ball
        return self.manifold.proj(z)

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    def log_prob(self, z):
        z = to_hyp_dtype(z)
        loc_exp = self.loc.expand_as(z)
        scale_exp = self.scale.expand_as(z)
        d = z.shape[-1]

        # Recover u, v
        u = self.manifold.logmap(z, loc_exp)         # tangent vector at loc
        zero = torch.zeros_like(loc_exp)
        v = self.manifold.ptransp(loc_exp, zero, u)

        log_gauss = (
            -0.5 * (v / scale_exp).pow(2).sum(-1)
            - scale_exp.log().sum(-1)
            - 0.5 * d * math.log(2.0 * math.pi)
        )

        # The volume correction uses the Riemannian norm at the location.
        c = to_hyp_dtype(self.manifold._get_c())
        sqrt_c = c.sqrt()

        # Riemannian norm: ||u||_g = lambda_loc * ||u||_2
        loc_norm_sq = loc_exp.pow(2).sum(-1, keepdim=True).clamp(max=(1.0/c - 1e-7))
        lambda_loc = 2.0 / (1.0 - c * loc_norm_sq).clamp(min=1e-15)   # conformal factor
        u_norm_eucl = u.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        r_riemannian = (lambda_loc * u_norm_eucl).squeeze(-1)         # ||u||_g

        sc_r = sqrt_c * r_riemannian
        log_det_J = (d - 1) * self._log_sinh_over_x(sc_r)

        return log_gauss - log_det_J

    # ------------------------------------------------------------------
    # Numerically stable helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_sinh_over_x(x):
        """Stable log(sinh(x)/x).

        Taylor branch (|x| < 1e-6):  x^2/6 - x^4/180
        Safe branch  (|x| >= 1e-6):  log(sinh(x)) - log(x); float64 sinh is
                                     accurate up to x ~ 700.
        """
        small = x.abs() < 1e-6
        safe_x = x.clamp(min=1e-15)
        safe_val = (safe_x.sinh() / safe_x).clamp(min=1e-15).log()
        taylor_val = x.pow(2) / 6.0 - x.pow(4) / 180.0
        return torch.where(small, taylor_val, safe_val)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def mc_kl(p, q, n_samples=1):
    """Monte Carlo estimate of KL(p || q) using reparameterized samples from p.

    Args:
        p, q: WrappedNormal instances with broadcastable batch shapes.
        n_samples: number of MC samples (default 1, matches HVAE convention).

    Returns:
        Tensor of broadcasted batch_shape, float64.
    """
    z = p.rsample(torch.Size([n_samples]))
    log_p = p.log_prob(z)
    log_q = q.log_prob(z)
    return (log_p - log_q).mean(dim=0)


def standard_wrapped_normal(batch_shape, dim, manifold, device, dtype=torch.float64):
    """Return an origin-centered, unit-variance wrapped-normal prior."""
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    batch_shape = tuple(batch_shape)
    loc = torch.zeros(*batch_shape, dim, device=device, dtype=dtype)
    loc = manifold.proj(loc)
    scale = torch.ones(*batch_shape, dim, device=device, dtype=dtype)
    return WrappedNormal(loc, scale, manifold)
