#%%
import jax
import jax.numpy as jnp
import numpy as np

# def expm1(x: torch.Tensor) -> torch.Tensor:
#     return torch.expm1(x)
# [batch, x1,x2,x3,x4,...] -> [batch, x1*x2*x3]
#%%

#%%

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


# %%
