import numpy as np
import getpass
import os
import jax.numpy as jnp
import jax
from jax import random
import optax
from typing import Any, NamedTuple


# Folders
def create_folders(args):
    try:
        os.makedirs("outputs")
    except OSError:
        pass

    try:
        os.makedirs("outputs/" + args.exp_name)
    except OSError:
        pass


# Model checkpoints


def save_model(model, path):
    jnp.savez(path, **model)


def load_model(path):
    return jnp.load(path)


# Gradient clipping
class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

class GradientClippingState(NamedTuple):
    gradnorm_queue: Any

def gradient_clipping(gradnorm_queue: Queue, max_len=50):
    def init_fn(params):
        return GradientClippingState(gradnorm_queue=gradnorm_queue)

    def update_fn(updates, state, params=None):        
        # Update the queue with the new grad norm
        mean = state.gradnorm_queue.mean()
        std = state.gradnorm_queue.std()
        max_grad_norm = 1.5 * mean + 2 * std
        
        grad_norm = optax.global_norm(updates)
        state.gradnorm_queue.add(min(float(grad_norm), float(max_grad_norm)))

        # Clip gradients
        clipped_updates = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm), updates
        )

        return clipped_updates, state

    return optax.GradientTransformation(init_fn, update_fn)

# Use the custom gradient clipping in AdamW_with_amsgrad
def AdamW_with_amsgrad(
    learning_rate: float = 0.001,
    b1: float = 0.9,
    b2: float = 0.999,
    weight_decay: float = 0.0001,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype = None,
    mask = None,
    gradnorm_queue = None,
):
    if gradnorm_queue:
        queue = Queue(max_len=gradnorm_queue)

        return optax.chain(
            optax.scale_by_amsgrad(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
            optax.add_decayed_weights(weight_decay, mask),
            optax.scale_by_learning_rate(learning_rate),
            gradient_clipping(queue)
        )
    else:
        return optax.chain(
            optax.scale_by_amsgrad(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
            optax.add_decayed_weights(weight_decay, mask),
            optax.scale_by_learning_rate(learning_rate)
        )




# Rotation data augmntation
def random_rotation(x, key):
    bs, n_nodes, n_dims = x.shape
    angle_range = jnp.pi * 2
    if n_dims == 2:
        theta = random.uniform(key, (bs, 1, 1)) * angle_range - jnp.pi
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        R_row0 = jnp.concatenate([cos_theta, -sin_theta], axis=2)
        R_row1 = jnp.concatenate([sin_theta, cos_theta], axis=2)
        R = jnp.concatenate([R_row0, R_row1], axis=1)

        x = jnp.swapaxes(x, 1, 2)
        x = jnp.matmul(R, x)
        x = jnp.swapaxes(x, 1, 2)

    elif n_dims == 3:
        # Build Rx
        Rx = jnp.eye(3).reshape(1, 3, 3).repeat(bs, axis=0)
        Ry = jnp.eye(3).reshape(1, 3, 3).repeat(bs, axis=0)
        Rz = jnp.eye(3).reshape(1, 3, 3).repeat(bs, axis=0)

        theta_x = random.uniform(key, (bs, 1, 1)) * angle_range - jnp.pi
        theta_y = random.uniform(key, (bs, 1, 1)) * angle_range - jnp.pi
        theta_z = random.uniform(key, (bs, 1, 1)) * angle_range - jnp.pi

        Rx = jax.ops.index_update(
            Rx[:, 1:2, 1:2], jax.ops.index[:, :], jnp.cos(theta_x)
        )
        Rx = jax.ops.index_update(
            Rx[:, 1:2, 2:3], jax.ops.index[:, :], jnp.sin(theta_x)
        )
        Rx = jax.ops.index_update(
            Rx[:, 2:3, 1:2], jax.ops.index[:, :], -jnp.sin(theta_x)
        )
        Rx = jax.ops.index_update(
            Rx[:, 2:3, 2:3], jax.ops.index[:, :], jnp.cos(theta_x)
        )

        Ry = jax.ops.index_update(
            Ry[:, 0:1, 0:1], jax.ops.index[:, :], jnp.cos(theta_y)
        )
        Ry = jax.ops.index_update(
            Ry[:, 0:1, 2:3], jax.ops.index[:, :], -jnp.sin(theta_y)
        )
        Ry = jax.ops.index_update(
            Ry[:, 2:3, 0:1], jax.ops.index[:, :], jnp.sin(theta_y)
        )
        Ry = jax.ops.index_update(
            Ry[:, 2:3, 2:3], jax.ops.index[:, :], jnp.cos(theta_y)
        )

        Rz = jax.ops.index_update(
            Rz[:, 0:1, 0:1], jax.ops.index[:, :], jnp.cos(theta_z)
        )
        Rz = jax.ops.index_update(
            Rz[:, 0:1, 1:2], jax.ops.index[:, :], jnp.sin(theta_z)
        )
        Rz = jax.ops.index_update(
            Rz[:, 1:2, 0:1], jax.ops.index[:, :], -jnp.sin(theta_z)
        )
        Rz = jax.ops.index_update(
            Rz[:, 1:2, 1:2], jax.ops.index[:, :], jnp.cos(theta_z)
        )

        x = jnp.swapaxes(x, 1, 2)
        x = jnp.matmul(Rx, x)
        x = jnp.matmul(Ry, x)
        x = jnp.matmul(Rz, x)
        x = jnp.swapaxes(x, 1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()


# Other utilities
def get_wandb_username(username):
    if username == "cvignac":
        return "cvignac"
    current_user = getpass.getuser()
    if current_user == "victor" or current_user == "garciasa":
        return "vgsatorras"
    else:
        return username


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)  # Initialize a random key for reproducibility

    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = jax.random.normal(key, (bs, n_nodes, n_dims))  # Generate random input tensor
    print("Original x:")
    print(x)

    # Apply random rotation
    x_rotated = random_rotation(x, key)
    print("Rotated x:")
    print(x_rotated)
