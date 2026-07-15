"""Utilities for the wrapped-normal variational latent space."""

import torch

from distributions.wrapped_normal import (
    WrappedNormal,
    mc_kl,
    standard_wrapped_normal,
)
from utils.dtype import to_hyp_dtype, to_out_dtype


def make_posterior(manifold, mean_tangent, log_variance, eps_scale=1.0):
    """Construct a wrapped-normal posterior from tangent-space parameters."""
    mean_ball = manifold.expmap0(to_hyp_dtype(mean_tangent))
    scale = (to_hyp_dtype(log_variance) / 2.0).exp() * eps_scale
    return WrappedNormal(mean_ball, scale, manifold)


def posterior_sample(posterior):
    """Return a sample on the ball and its origin-tangent representation."""
    sample_ball = posterior.rsample()
    sample_tangent = to_out_dtype(posterior.manifold.logmap0(sample_ball))
    return sample_ball, sample_tangent


def posterior_kl(posterior, latent_dim, n_samples=1):
    """Estimate ``KL(q || p)`` with Monte Carlo samples and sum the batch."""
    prior = standard_wrapped_normal(
        batch_shape=posterior.batch_shape,
        dim=latent_dim,
        manifold=posterior.manifold,
        device=posterior.loc.device,
        dtype=torch.float64,
    )
    return mc_kl(posterior, prior, n_samples).sum()
