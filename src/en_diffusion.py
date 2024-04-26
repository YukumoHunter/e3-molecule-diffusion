import jax
import jax.numpy as jnp
import numpy as np

# def expm1(x: torch.Tensor) -> torch.Tensor:
#     return torch.expm1(x)
# [batch, x1,x2,x3,x4,...] -> [batch, x1*x2*x3]


def expm1(x):
    return jax.lax.expm1(x)

def softplus(x):
    return jax.nn.softplus(x)

def sum_except_batch(x):
    return jnp.sum(x.reshape((x.shape[0], -1)), axis = -1)

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = jnp.concatenate([jnp.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = jnp.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = jnp.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = jnp.linspace(0, steps, steps)
    alphas2 = (1 - jnp.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = jnp.linspace(0, steps, steps)
    alphas_cumprod = jnp.cos(((x / steps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = jnp.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = jnp.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * jnp.log(2 * jnp.pi * sigma**2) + 0.5
    )

def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                jnp.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask)

def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)

    if len(q_sigma.shape) != 1 or len(p_sigma.shape) != 1:
        raise ValueError("Dimensions of q_sigma and p_sigma must be 1")

    return d * jnp.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d

#Not sure
class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias=True, weight_init_offset=-2):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with uniform distribution
        weights_shape = (self.out_features, self.in_features)
        self.weight = jnp.zeros(weights_shape)

        # Initialize biases if necessary
        if bias:
            self.bias = jnp.zeros(self.out_features)
        else:
            self.bias = None
            
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Kaiming uniform initialization
        rng_key = random.PRNGKey(0)
        bound = jnp.sqrt(5)
        fan_in = self.in_features
        fan_out = self.out_features
        shape = (fan_out, fan_in)
        self.weight = random.uniform(rng_key, shape, minval=-bound, maxval=bound)

        # Add weight initialization offset
        self.weight += self.weight_init_offset

        # Initialize biases if necessary
        if self.bias is not None:
            self.bias = random.uniform(rng_key, (fan_out,), minval=-bound, maxval=bound)

    def forward(self, input):
        # Apply softplus to ensure positive weights
        positive_weight = lax.logaddexp(0.0, self.weight)
        
        # Compute linear transformation
        output = jnp.dot(input, positive_weight.T)

        # Add bias if necessary
        if self.bias is not None:
            output = output + self.bias

        return output