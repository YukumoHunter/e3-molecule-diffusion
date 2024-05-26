import numpy as np
import getpass
import os
import jax.numpy as jnp
import jax
from jax import random


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


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0
    )

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(
            f"Clipped gradient with value {grad_norm:.1f} "
            f"while allowed {max_grad_norm:.1f}"
        )
    return grad_norm


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
