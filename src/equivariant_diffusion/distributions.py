import flax.linen as nn
from jax import random

from equivariant_diffusion.utils import (
    center_gravity_zero_gaussian_log_likelihood_with_mask,
    standard_gaussian_log_likelihood_with_mask,
    center_gravity_zero_gaussian_log_likelihood,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_center_gravity_zero_gaussian,
    sample_gaussian_with_mask,
)


class PositionFeaturePrior(nn.Module):
    n_dim: int
    in_node_nf: int

    def __call__(self, z_x, z_h, node_mask=None):
        assert z_x.ndim == 3
        assert node_mask.ndim == 3
        assert node_mask.shape[:2] == z_x.shape[:2]

        assert (z_x * (1 - node_mask)).sum() < 1e-8 and (
            z_h * (1 - node_mask)
        ).sum() < 1e-8, "These variables should be properly masked."

        log_pz_x = center_gravity_zero_gaussian_log_likelihood_with_mask(z_x, node_mask)

        log_pz_h = standard_gaussian_log_likelihood_with_mask(z_h, node_mask)

        log_pz = log_pz_x + log_pz_h
        return log_pz

    def sample(self, key, n_samples, n_nodes, node_mask):
        z_x_key, z_h_key = random.split(key)

        z_x = sample_center_gravity_zero_gaussian_with_mask(
            key=z_x_key,
            size=(n_samples, n_nodes, self.n_dim),
            node_mask=node_mask,
        )

        z_h = sample_gaussian_with_mask(
            key=z_h_key,
            size=(n_samples, n_nodes, self.in_node_nf),
            node_mask=node_mask,
        )

        return z_x, z_h


class PositionPrior(nn.Module):
    def __call__(self, x):
        return center_gravity_zero_gaussian_log_likelihood(x)

    def sample(self, key, size, device):
        samples = sample_center_gravity_zero_gaussian(key, size, device)
        return samples
