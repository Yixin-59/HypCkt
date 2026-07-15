"""HypCkt model with a hyperbolic encoder and wrapped-normal latent space."""

import torch

from distributions import (
    make_posterior,
    posterior_kl,
    posterior_sample,
    standard_wrapped_normal,
)
from encoders.hyp_encoder import HypEncoder
from manifolds import PoincareBall
from models.GRU_decoder import GRUDecoder


class HypCkt(GRUDecoder):
    """Hyperbolic circuit encoder with a Euclidean autoregressive decoder."""

    def __init__(
        self,
        max_n,
        nvt,
        subn_nvt,
        START_TYPE,
        END_TYPE,
        max_pos=8,
        emb_dim=16,
        feat_emb_dim=8,
        hs=301,
        nz=66,
        bidirectional=False,
        pos=True,
        scale=True,
        scale_factor=102,
        topo_feat_scale=0.01,
        c=1.0,
        learnable_curvature=True,
        kl_n_samples=1,
        eps_scale=1.0,
    ):
        if bidirectional:
            raise ValueError("HypCkt currently supports forward encoding only")

        super().__init__(
            max_n=max_n,
            nvt=nvt,
            subn_nvt=subn_nvt,
            START_TYPE=START_TYPE,
            END_TYPE=END_TYPE,
            max_pos=max_pos,
            emb_dim=emb_dim,
            feat_emb_dim=feat_emb_dim,
            hs=hs,
            nz=nz,
            bidirectional=bidirectional,
            pos=pos,
            scale=scale,
            scale_factor=scale_factor,
            topo_feat_scale=topo_feat_scale,
        )

        # Replace the inherited encoder while retaining the Euclidean decoder.
        del self.df_enc
        del self.fc1
        del self.fc2
        del self.grue_forward
        del self.grue_backward
        del self.gate_backward
        del self.mapper_backward

        self.manifold = PoincareBall(c=c, learnable=learnable_curvature)
        self.hyp_encoder = HypEncoder(
            manifold=self.manifold,
            max_n=max_n,
            nvt=nvt,
            max_pos=max_pos,
            emb_dim=emb_dim,
            feat_emb_dim=feat_emb_dim,
            hs=hs,
            nz=nz,
            pos=pos,
        )
        self.kl_n_samples = kl_n_samples
        self.eps_scale = eps_scale

    @classmethod
    def from_config(cls, config):
        dataset = config["dataset"]
        encoder = config["encoder"]
        vae = config["vae"]
        manifold = config["manifold"]
        decoder = config.get("decoder", {})
        return cls(
            max_n=dataset["max_n"],
            nvt=dataset["nvt"],
            subn_nvt=dataset["subn_nvt"],
            START_TYPE=dataset["START_TYPE"],
            END_TYPE=dataset["END_TYPE"],
            max_pos=dataset["max_pos"],
            emb_dim=encoder["emb_dim"],
            feat_emb_dim=encoder["feat_emb_dim"],
            hs=encoder["hs"],
            nz=vae["latent_dim"],
            bidirectional=encoder.get("bidirectional", False),
            pos=encoder.get("pos", True),
            scale=decoder.get("scale", True),
            scale_factor=decoder.get("scale_factor", 102),
            topo_feat_scale=encoder.get("topo_feat_scale", 0.01),
            c=manifold["c_init"],
            learnable_curvature=manifold.get("learnable_curvature", True),
            kl_n_samples=vae.get("kl_n_samples", 1),
            eps_scale=vae.get("eps_scale", 1.0),
        )

    def encode(self, graphs):
        """Return tangent-space posterior parameters with shape ``(B, nz)``."""
        return self.hyp_encoder.encode(graphs)

    def reparameterize(self, mean, log_variance, eps_scale=None):
        """Sample on the Poincare ball and return the origin-tangent vector."""
        if eps_scale is None:
            eps_scale = self.eps_scale
        posterior = make_posterior(
            self.manifold, mean, log_variance, eps_scale=eps_scale
        )
        _, tangent_sample = posterior_sample(posterior)
        return tangent_sample

    def loss(
        self,
        mean,
        log_variance,
        true_graphs,
        beta=0.005,
        reg_scale=0.05,
        pos_scale=0.5,
    ):
        """Compute reconstruction loss and a Monte Carlo manifold KL term."""
        total_without_kl, recon, _, type_loss, pos_loss, feature_loss = super().loss(
            mean,
            log_variance,
            true_graphs,
            beta=0.0,
            reg_scale=reg_scale,
            pos_scale=pos_scale,
        )
        posterior = make_posterior(
            self.manifold, mean, log_variance, eps_scale=self.eps_scale
        )
        kl = posterior_kl(
            posterior, self.nz, n_samples=self.kl_n_samples
        ).float()
        total = total_without_kl + beta * kl
        return total, recon, kl, type_loss, pos_loss, feature_loss

    def generate_sample(self, n_samples):
        """Generate graphs from the standard wrapped-normal prior."""
        prior = standard_wrapped_normal(
            batch_shape=(n_samples,),
            dim=self.nz,
            manifold=self.manifold,
            device=self.get_device(),
            dtype=torch.float64,
        )
        latent_ball = prior.rsample()
        latent_tangent = self.manifold.logmap0(latent_ball).float()
        return self.decode(latent_tangent)
